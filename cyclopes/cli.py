"""Training, calibration, evaluation, export, and parity commands."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .metrics import THRESHOLD, best_balanced_threshold, binary_metrics


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_optional(path: str | Path | None) -> str | None:
    return _sha256(path) if path is not None else None


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return value


def _emit(report: dict[str, Any], path: str | Path | None = None) -> dict[str, Any]:
    payload = json.dumps(_jsonable(report), sort_keys=True, indent=2, default=_json_default, allow_nan=False)
    if path is not None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return report


def _torch():
    import torch

    return torch


def _build_model(
    *,
    architecture: str = "mobilenet_v3_large_scalepair",
    pretrained: bool = False,
    model_repo: str | None = None,
    model_revision: str | None = None,
    layers: tuple[int, ...] = (4, 8, 12),
):
    if architecture == "mobilenet_v3_large_scalepair":
        from .modeling import ScalePairMobileNet

        return ScalePairMobileNet(pretrained=pretrained)
    if architecture == "vit_multilayer_scalepair":
        from .vit_modeling import MODEL_REPO, MODEL_REVISION, MultiLayerScalePairViT

        loader = MultiLayerScalePairViT.from_pretrained if pretrained else MultiLayerScalePairViT.from_config
        return loader(model_repo or MODEL_REPO, model_revision or MODEL_REVISION, layers=layers)
    raise ValueError(f"unknown architecture: {architecture}")


def _data_helpers():
    from .data import ManifestDataset, load_manifest

    return ManifestDataset, load_manifest


def _load_checkpoint(path: str | Path, *, device: str = "cpu"):
    torch = _torch()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    config = checkpoint.get("config", {})
    model = _build_model(
        architecture=config.get("architecture", "mobilenet_v3_large_scalepair"),
        pretrained=False,
        model_repo=config.get("model_repo"),
        model_revision=config.get("model_revision"),
        layers=tuple(config.get("layers", (4, 8, 12))),
    )
    model.load_state_dict(state)
    model.to(device).eval()
    return model, checkpoint


def _dataset(
    manifest: str | Path,
    split: str,
    *,
    training: bool = False,
    seed: int = 323,
    browser_view: bool = False,
    model=None,
):
    ManifestDataset, load_manifest = _data_helpers()
    samples = load_manifest(manifest)
    return samples, ManifestDataset(
        samples,
        split,
        training=training,
        seed=seed,
        browser_view=browser_view,
        image_size=getattr(model, "image_size", 224),
        mean=getattr(model, "mean", (0.485, 0.456, 0.406)),
        std=getattr(model, "std", (0.229, 0.224, 0.225)),
    )


def _checkpoint_config(args: argparse.Namespace, model) -> dict[str, Any]:
    config: dict[str, Any] = {"architecture": args.architecture, "seed": args.seed}
    if args.architecture == "vit_multilayer_scalepair":
        config.update(
            model_repo=args.model_repo,
            model_revision=args.model_revision,
            layers=list(model.layers),
        )
    return config


def _labels(dataset) -> np.ndarray:
    return np.asarray([sample.label for sample in dataset.samples], dtype=np.int8)


def _raw_outputs(model, dataset, *, batch_size: int, device: str, workers: int = 0) -> tuple[np.ndarray, np.ndarray]:
    torch = _torch()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                         persistent_workers=workers > 0)
    fused: list[np.ndarray] = []
    current: list[np.ndarray] = []
    with torch.inference_mode():
        for images, _labels_batch, _indices in loader:
            outputs = model.components(images.to(device))
            fused.append(outputs.fused_logit.detach().cpu().numpy())
            current.append(outputs.current_logit.detach().cpu().numpy())
    empty = np.empty(0, dtype=np.float64)
    return (np.concatenate(fused) if fused else empty, np.concatenate(current) if current else empty)


def _paired_raw_outputs(model, dataset, *, batch_size: int, device: str, workers: int = 0) -> dict[str, np.ndarray]:
    torch = _torch()
    random.seed(dataset.seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(dataset.seed),
    )
    values: dict[str, list[np.ndarray]] = defaultdict(list)
    with torch.inference_mode():
        for clean, web, _labels_batch, _families, _moderate, _indices in loader:
            for name, images in (("clean", clean), ("web", web)):
                outputs = model.components(images.to(device))
                values[f"{name}_fused"].append(outputs.fused_logit.detach().cpu().numpy())
                values[f"{name}_current"].append(outputs.current_logit.detach().cpu().numpy())
    return {key: np.concatenate(items) for key, items in values.items()}


def _probabilities(logits: np.ndarray, temperature: float = 1.0, bias: float = 0.0) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64) / temperature + bias
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _fit_calibration(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """Fit NLL calibration, then align its best BA boundary to the UI threshold."""
    from scipy.optimize import minimize

    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.size == 0 or x.size != y.size:
        raise ValueError("calibration data must be non-empty and equally sized")

    def objective(parameters: np.ndarray) -> float:
        temperature = math.exp(float(parameters[0]))
        values = x / temperature + float(parameters[1])
        return float(np.mean(np.logaddexp(0.0, values) - y * values))

    fitted = minimize(objective, np.zeros(2, dtype=np.float64), method="L-BFGS-B")
    temperature = math.exp(float(fitted.x[0]))
    nll_bias = float(fitted.x[1])
    calibrated = _probabilities(x, temperature, nll_bias)
    probability_threshold, _ = best_balanced_threshold(y.astype(np.int8), calibrated)
    probability_threshold = float(np.clip(probability_threshold, 1e-6, 1 - 1e-6))
    calibrated_logit = math.log(probability_threshold / (1 - probability_threshold))
    raw_operating_logit = (calibrated_logit - nll_bias) * temperature
    from .modeling import threshold_alignment

    final_bias = threshold_alignment(raw_operating_logit, temperature, THRESHOLD)
    return temperature, float(final_bias), float(1 / (1 + math.exp(-raw_operating_logit)))


def _load_calibration(path: str | Path | None) -> tuple[float, float, float]:
    if path is None:
        return 1.0, 0.0, 1.0
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "calibration" in payload:
        payload = payload["calibration"]
    temperature = float(payload["temperature"])
    bias = float(payload["bias"])
    blend_weight = float(payload.get("blend_weight", 1.0))
    if temperature <= 0 or not 0 <= blend_weight <= 1:
        raise ValueError("invalid calibration")
    return temperature, bias, blend_weight


def _blend(outputs: tuple[np.ndarray, np.ndarray], weight: float) -> np.ndarray:
    fused, current = outputs
    return weight * fused + (1 - weight) * current


def _metrics_report(samples: Iterable[Any], scores: np.ndarray) -> dict[str, Any]:
    samples = list(samples)
    labels = np.asarray([sample.label for sample in samples], dtype=np.int8)
    metrics = binary_metrics(labels, scores, threshold=THRESHOLD).to_dict()

    def grouped(attribute: str) -> dict[str, Any]:
        groups: defaultdict[str, list[int]] = defaultdict(list)
        for index, sample in enumerate(samples):
            groups[getattr(sample, attribute) or "unknown"].append(index)
        return {
            name: binary_metrics(labels[indices], scores[indices], threshold=THRESHOLD).to_dict()
            for name, indices in sorted(groups.items())
        }

    return {
        "threshold": THRESHOLD,
        "count": len(samples),
        "metrics": metrics,
        **metrics,
        "per_source": grouped("source"),
        "per_generator": grouped("generator"),
        "per_generator_family": grouped("generator_family"),
        "per_content_domain": grouped("content_domain"),
    }


def _ema_update(ema, model, decay: float) -> None:
    torch = _torch()
    with torch.no_grad():
        for ema_value, value in zip(ema.parameters(), model.parameters(), strict=True):
            ema_value.mul_(decay).add_(value, alpha=1 - decay)
        for ema_value, value in zip(ema.buffers(), model.buffers(), strict=True):
            ema_value.copy_(value)


def _selection_metrics(model, dataset, labels: np.ndarray, *, batch_size: int, device: str, workers: int) -> dict[str, float]:
    random.seed(dataset.seed)
    outputs = _paired_raw_outputs(model, dataset, batch_size=batch_size, device=device, workers=workers)
    result: dict[str, float] = {}
    for view in ("clean", "web"):
        scores = _probabilities(outputs[f"{view}_fused"])
        threshold, metrics = best_balanced_threshold(labels, scores)
        result[f"{view}_balanced_accuracy"] = metrics.balanced_accuracy
        result[f"{view}_threshold"] = threshold
    result["selection_score"] = min(result["clean_balanced_accuracy"], result["web_balanced_accuracy"])
    return result


def train(args: argparse.Namespace) -> int:
    torch = _torch()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.initial_checkpoint:
        model, initial = _load_checkpoint(args.initial_checkpoint, device=args.device)
        initial_config = initial.get("config", {})
        args.architecture = initial_config.get("architecture", args.architecture)
        args.model_repo = initial_config.get("model_repo", args.model_repo)
        args.model_revision = initial_config.get("model_revision", args.model_revision)
    else:
        model = _build_model(
            architecture=args.architecture,
            pretrained=args.pretrained,
            model_repo=args.model_repo,
            model_revision=args.model_revision,
        ).to(args.device)
    _samples, dataset = _dataset(args.manifest, args.split, training=True, seed=args.seed, model=model)
    _validation_samples, validation = _dataset(
        args.manifest, args.validation_split, training=True, seed=args.seed + 1, model=model
    )
    ema = copy.deepcopy(model).eval()

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=dataset.balanced_sampler(args.seed),
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.workers > 0,
    )
    backbone_module = getattr(model, "backbone", getattr(model, "features", None))
    if backbone_module is None:
        raise TypeError("model must expose backbone or features")
    backbone = list(backbone_module.parameters())
    backbone_ids = {id(parameter) for parameter in backbone}
    head = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    optimizer = torch.optim.AdamW(
        (({"params": backbone, "lr": args.backbone_lr}), ({"params": head, "lr": args.head_lr})),
        weight_decay=args.weight_decay,
    )
    total_steps = min(args.max_steps, args.epochs * len(loader))
    warmup_steps = max(1, round(total_steps * args.warmup_fraction))

    def lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return max(1e-3, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    if hasattr(model, "freeze_backbone"):
        model.freeze_backbone()
    else:
        for parameter in backbone_module.parameters():
            parameter.requires_grad_(False)

    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    global_step = 0
    best_score = -1.0
    epoch_reports: list[dict[str, Any]] = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_losses: list[float] = []
        consistency_weight = 0.0 if epoch == 0 else 0.02 if epoch == 1 else args.consistency_weight
        for clean, web, labels, families, moderate, _indices in loader:
            if global_step == args.freeze_steps:
                if hasattr(model, "unfreeze_last_blocks"):
                    model.unfreeze_last_blocks(args.unfreeze_last_blocks)
                else:
                    for parameter in backbone_module.parameters():
                        parameter.requires_grad_(True)
            clean = clean.to(args.device, non_blocking=True)
            web = web.to(args.device, non_blocking=True)
            loss_device = "cpu" if args.device == "mps" else args.device
            labels = labels.float().to(loss_device, non_blocking=True)
            families = families.to(loss_device, non_blocking=True)
            moderate = moderate.to(loss_device, non_blocking=True)
            targets = labels * 0.98 + 0.01
            optimizer.zero_grad(set_to_none=True)
            device_type = "cuda" if args.device.startswith("cuda") else "cpu"
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
                clean_outputs = model.components(clean)
                web_outputs = model.components(web)
                clean_fused = bce(clean_outputs.fused_logit.to(loss_device), targets)
                web_fused = bce(web_outputs.fused_logit.to(loss_device), targets)
                fused_loss = 0.5 * (clean_fused.mean() + web_fused.mean())
                auxiliary = torch.stack(
                    (
                        bce(clean_outputs.current_logit.to(loss_device), targets).mean(),
                        bce(clean_outputs.probe_logit.to(loss_device), targets).mean(),
                        bce(web_outputs.current_logit.to(loss_device), targets).mean(),
                        bce(web_outputs.probe_logit.to(loss_device), targets).mean(),
                    )
                ).mean()
                family_mask = families >= 0
                if family_mask.any():
                    family_loss = torch.nn.functional.cross_entropy(clean_outputs.family_logits.to(loss_device)[family_mask], families[family_mask])
                else:
                    family_loss = fused_loss * 0
                moderate_loss = torch.maximum(clean_fused, web_fused)[moderate].mean() if moderate.any() else fused_loss * 0
                consistency = 1 - torch.nn.functional.cosine_similarity(
                    clean_outputs.embedding.to(loss_device), web_outputs.embedding.to(loss_device)
                ).mean()
                loss = (
                    0.65 * fused_loss
                    + 0.15 * auxiliary
                    + 0.10 * family_loss
                    + 0.10 * moderate_loss
                    + consistency_weight * consistency
                )
            if not torch.isfinite(loss) or loss < -1e-5:
                raise FloatingPointError(f"non-finite training loss at step {global_step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_losses.append(float(loss.detach().cpu()))
            if global_step == args.ema_start:
                ema.load_state_dict(model.state_dict())
            elif global_step > args.ema_start:
                _ema_update(ema, model, args.ema_decay)
            if global_step % args.checkpoint_every == 0:
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "ema_state": ema.state_dict(),
                        "config": _checkpoint_config(args, model),
                        "step": global_step,
                        "epoch": epoch + 1,
                    },
                    checkpoint_dir / f"step-{global_step:05d}.pt",
                )
            if global_step >= total_steps:
                break

        labels_array = _labels(validation)
        raw_metrics = _selection_metrics(model.eval(), validation, labels_array, batch_size=args.batch_size,
                                         device=args.device, workers=args.workers)
        ema_metrics = _selection_metrics(ema.eval(), validation, labels_array, batch_size=args.batch_size,
                                         device=args.device, workers=args.workers)
        use_ema = global_step >= args.ema_start and ema_metrics["selection_score"] >= raw_metrics["selection_score"]
        selected = ema if use_ema else model
        selected_metrics = ema_metrics if use_ema else raw_metrics
        epoch_report = {
            "epoch": epoch + 1,
            "step": global_step,
            "loss": float(np.mean(epoch_losses)) if epoch_losses else None,
            "raw": raw_metrics,
            "ema": ema_metrics,
            "selected": "ema" if use_ema else "raw",
        }
        epoch_reports.append(epoch_report)
        print(json.dumps(epoch_report, sort_keys=True), flush=True)
        if selected_metrics["selection_score"] > best_score:
            best_score = selected_metrics["selection_score"]
            torch.save(
                {
                    "model_state": selected.state_dict(),
                    "raw_state": model.state_dict(),
                    "ema_state": ema.state_dict(),
                    "config": _checkpoint_config(args, model),
                    "epoch": epoch + 1,
                    "step": global_step,
                    "selection_metrics": selected_metrics,
                },
                output,
            )
        if global_step >= total_steps:
            break

    report = {
        "command": "train",
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "checkpoint": str(output.resolve()),
        "checkpoint_sha256": _sha256(output),
        "initial_checkpoint_sha256": _sha256_optional(args.initial_checkpoint),
        "samples": len(dataset),
        "steps": global_step,
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else args.device,
        },
        "best_selection_score": best_score,
        "epoch_metrics": epoch_reports,
        "threshold": THRESHOLD,
    }
    _emit(report, args.report)
    return 0


def calibrate(args: argparse.Namespace) -> int:
    _torch().manual_seed(args.seed)
    model, _checkpoint = _load_checkpoint(args.checkpoint, device=args.device)
    _samples, dataset = _dataset(args.manifest, args.split, training=True, seed=args.seed, model=model)
    random.seed(args.seed)
    raw = _paired_raw_outputs(model, dataset, batch_size=args.batch_size, device=args.device, workers=args.workers)
    labels = _labels(dataset)
    candidates: list[dict[str, float]] = []
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        clean = weight * raw["clean_fused"] + (1 - weight) * raw["clean_current"]
        web = weight * raw["web_fused"] + (1 - weight) * raw["web_current"]
        temperature, bias, raw_threshold = _fit_calibration(clean, labels)
        clean_metrics = binary_metrics(labels, _probabilities(clean, temperature, bias), THRESHOLD)
        web_metrics = binary_metrics(labels, _probabilities(web, temperature, bias), THRESHOLD)
        candidates.append(
            {
                "blend_weight": weight,
                "temperature": temperature,
                "bias": bias,
                "raw_threshold": raw_threshold,
                "clean_balanced_accuracy": clean_metrics.balanced_accuracy,
                "web_balanced_accuracy": web_metrics.balanced_accuracy,
                "selection_score": min(clean_metrics.balanced_accuracy, web_metrics.balanced_accuracy),
            }
        )
    selected = max(candidates, key=lambda item: (item["selection_score"], item["clean_balanced_accuracy"]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {**selected, "threshold": THRESHOLD}
    output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    _emit(
        {
            "command": "calibrate",
            "manifest_sha256": _sha256(args.manifest),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "calibration_sha256": _sha256(output),
            "samples": len(dataset),
            "calibration": payload,
            "candidates": candidates,
        },
        args.report,
    )
    return 0


def evaluate(args: argparse.Namespace) -> int:
    model, _checkpoint = _load_checkpoint(args.checkpoint, device=args.device)
    temperature, bias, weight = _load_calibration(args.calibration)
    if args.paired_views:
        random.seed(args.seed)
        _samples, dataset = _dataset(
            args.manifest, args.split, training=True, seed=args.seed, model=model
        )
        paired = _paired_raw_outputs(model, dataset, batch_size=args.batch_size, device=args.device, workers=args.workers)
        view_scores = {
            view: _probabilities(
                weight * paired[f"{view}_fused"] + (1 - weight) * paired[f"{view}_current"],
                temperature,
                bias,
            )
            for view in ("clean", "web")
        }
        view_reports = {view: _metrics_report(dataset.samples, scores) for view, scores in view_scores.items()}
        _emit(
            {
                "command": "evaluate",
                "manifest_sha256": _sha256(args.manifest),
                "checkpoint_sha256": _sha256(args.checkpoint),
                "calibration_sha256": _sha256_optional(args.calibration),
                "split": args.split,
                "paired_views": True,
                "seed": args.seed,
                "calibration": {"temperature": temperature, "bias": bias, "blend_weight": weight},
                "selection_score": min(
                    view_reports["clean"]["balanced_accuracy"], view_reports["web"]["balanced_accuracy"]
                ),
                "views": view_reports,
            },
            args.report,
        )
        return 0

    _samples, dataset = _dataset(args.manifest, args.split, browser_view=args.browser_view, model=model)
    raw = _raw_outputs(model, dataset, batch_size=args.batch_size, device=args.device, workers=args.workers)
    scores = _probabilities(_blend(raw, weight), temperature, bias)
    predictions_sha256 = None
    if args.predictions:
        _emit(
            {
                "threshold": THRESHOLD,
                "items": [
                    {
                        "path": str(sample.path),
                        "expected": sample.label,
                        "score": float(score),
                        "predicted": int(score >= THRESHOLD),
                        "group": sample.group,
                    }
                    for sample, score in zip(dataset.samples, scores, strict=True)
                ],
            },
            args.predictions,
        )
        predictions_sha256 = _sha256(args.predictions)
    _emit(
        {
            "command": "evaluate",
            "manifest_sha256": _sha256(args.manifest),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "calibration_sha256": _sha256_optional(args.calibration),
            "split": args.split,
            "browser_view": args.browser_view,
            "calibration": {"temperature": temperature, "bias": bias, "blend_weight": weight},
            "predictions_sha256": predictions_sha256,
            **_metrics_report(dataset.samples, scores),
        },
        args.report,
    )
    return 0


def export_model(args: argparse.Namespace) -> int:
    torch = _torch()
    from .modeling import ExportedScalePair

    model, _checkpoint = _load_checkpoint(args.checkpoint, device=args.device)
    temperature, bias, weight = _load_calibration(args.calibration)
    wrapper = ExportedScalePair(model, weight, temperature, bias).to(args.device).eval()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image_size = getattr(model, "image_size", 224)
    dummy = torch.zeros((1, 3, image_size, image_size), dtype=torch.float32, device=args.device)
    torch.onnx.export(
        wrapper,
        dummy,
        output,
        input_names=["image"],
        output_names=["logit"],
        opset_version=args.opset,
        dynamo=False,
    )
    _emit(
        {
            "command": "export",
            "checkpoint_sha256": _sha256(args.checkpoint),
            "calibration_sha256": _sha256_optional(args.calibration),
            "onnx": str(output.resolve()),
            "onnx_sha256": _sha256(output),
            "input_size": image_size,
            "threshold": THRESHOLD,
            "calibration": {"temperature": temperature, "bias": bias, "blend_weight": weight},
        },
        args.report,
    )
    return 0


def parity(args: argparse.Namespace) -> int:
    model, _checkpoint = _load_checkpoint(args.checkpoint, device=args.device)
    _samples, dataset = _dataset(args.manifest, args.split, browser_view=args.browser_view, model=model)
    raw = _raw_outputs(model, dataset, batch_size=args.batch_size, device=args.device, workers=args.workers)
    temperature, bias, weight = _load_calibration(args.calibration)
    python_scores = _probabilities(_blend(raw, weight), temperature, bias)
    import onnxruntime as ort

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    values: list[float] = []
    for index in range(len(dataset)):
        image = dataset[index][0].numpy()[None, ...]
        logit = float(np.asarray(session.run(None, {session.get_inputs()[0].name: image})[0]).reshape(-1)[0])
        values.append(float(_probabilities(np.asarray([logit]))[0]))
    onnx_scores = np.asarray(values)
    differences = np.abs(python_scores - onnx_scores)
    report = {
        "command": "onnx-parity",
        "manifest_sha256": _sha256(args.manifest),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "calibration_sha256": _sha256_optional(args.calibration),
        "onnx_sha256": _sha256(args.onnx),
        "threshold": THRESHOLD,
        "browser_view": args.browser_view,
        "count": len(differences),
        "max_abs_diff": float(differences.max()) if differences.size else 0.0,
        "mean_abs_diff": float(differences.mean()) if differences.size else 0.0,
        "within_tolerance": bool(np.all(differences <= args.tolerance)),
        "per_image": [
            {
                "path": str(sample.path),
                "python": float(python_scores[index]),
                "onnx": float(onnx_scores[index]),
                "abs_diff": float(differences[index]),
            }
            for index, sample in enumerate(dataset.samples)
        ],
    }
    _emit(report, args.report)
    return 0 if report["within_tolerance"] else 1


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--calibration")
    parser.add_argument("--report")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cyclopes")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--manifest", required=True)
    train_parser.add_argument("--split", default="train")
    train_parser.add_argument("--validation-split", default="validation")
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--report")
    train_parser.add_argument("--epochs", type=int, default=6)
    train_parser.add_argument("--max-steps", type=int, default=4000)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--backbone-lr", type=float, default=6e-5)
    train_parser.add_argument("--head-lr", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.02)
    train_parser.add_argument("--warmup-fraction", type=float, default=0.05)
    train_parser.add_argument("--consistency-weight", type=float, default=0.05)
    train_parser.add_argument("--gradient-clip", type=float, default=1.0)
    train_parser.add_argument("--freeze-steps", type=int, default=200)
    train_parser.add_argument("--ema-start", type=int, default=500)
    train_parser.add_argument("--ema-decay", type=float, default=0.9995)
    train_parser.add_argument("--checkpoint-every", type=int, default=250)
    train_parser.add_argument("--seed", type=int, default=323)
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--workers", type=int, default=8)
    train_parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument(
        "--architecture",
        choices=("mobilenet_v3_large_scalepair", "vit_multilayer_scalepair"),
        default="mobilenet_v3_large_scalepair",
    )
    train_parser.add_argument(
        "--model-repo",
        default="buildborderless/CommunityForensics-DeepfakeDet-ViT",
    )
    train_parser.add_argument(
        "--model-revision",
        default="ac6ee457bea904a373065754107451793b56db00",
    )
    train_parser.add_argument("--unfreeze-last-blocks", type=int, default=0)
    train_parser.add_argument("--initial-checkpoint")
    train_parser.set_defaults(handler=train)

    calibrate_parser = commands.add_parser("calibrate")
    _common(calibrate_parser)
    calibrate_parser.add_argument("--output", required=True)
    calibrate_parser.add_argument("--seed", type=int, default=323)
    calibrate_parser.set_defaults(handler=calibrate)

    evaluate_parser = commands.add_parser("evaluate")
    _common(evaluate_parser)
    evaluate_parser.add_argument("--predictions")
    evaluate_parser.add_argument("--browser-view", action="store_true")
    evaluate_parser.add_argument("--paired-views", action="store_true")
    evaluate_parser.add_argument("--seed", type=int, default=323)
    evaluate_parser.set_defaults(handler=evaluate)

    export_parser = commands.add_parser("export")
    export_parser.add_argument("--checkpoint", required=True)
    export_parser.add_argument("--device", default="cpu")
    export_parser.add_argument("--calibration")
    export_parser.add_argument("--report")
    export_parser.add_argument("--output", required=True)
    export_parser.add_argument("--opset", type=int, default=17)
    export_parser.set_defaults(handler=export_model)

    parity_parser = commands.add_parser("onnx-parity", aliases=["parity", "onnx_parity"])
    _common(parity_parser)
    parity_parser.add_argument("--onnx", required=True)
    parity_parser.add_argument("--tolerance", type=float, default=0.01)
    parity_parser.add_argument("--browser-view", action="store_true")
    parity_parser.set_defaults(handler=parity)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
