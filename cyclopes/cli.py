"""Command line tools for training, calibration, evaluation, and export."""

from __future__ import annotations

import argparse
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


def _load_calibration(path: str | Path | None) -> tuple[float, float]:
    if path is None:
        return 1.0, 0.0
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "calibration" in payload:
        payload = payload["calibration"]
    temperature = float(payload["temperature"])
    bias = float(payload["bias"])
    if temperature <= 0:
        raise ValueError("calibration temperature must be positive")
    return temperature, bias


def _build_model(*, pretrained: bool = False):
    from .modeling import ForensicMobileNet

    return ForensicMobileNet(pretrained=pretrained)


def _data_helpers():
    from .data import ManifestDataset, load_manifest

    return ManifestDataset, load_manifest


def _load_checkpoint(path: str | Path, *, device: str = "cpu", pretrained: bool = False):
    torch = _torch()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    model = _build_model(pretrained=pretrained)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, checkpoint


def _raw_scores(model, dataset: ManifestDataset, *, batch_size: int, device: str) -> np.ndarray:
    torch = _torch()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for images, _labels, _indices in loader:
            logits = model(images.to(device))
            values.append(logits.detach().cpu().numpy().reshape(-1))
    return np.concatenate(values) if values else np.empty(0, dtype=np.float64)


def _probabilities(logits: np.ndarray, temperature: float = 1.0, bias: float = 0.0) -> np.ndarray:
    values = logits / temperature + bias
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _labels(dataset: ManifestDataset) -> np.ndarray:
    return np.asarray([sample.label for sample in dataset.samples], dtype=np.int8)


def _fit_calibration(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """Align the best calibration operating point with the frozen UI threshold."""
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.size == 0 or x.size != y.size:
        raise ValueError("calibration data must be non-empty and equally sized")
    temperature = 1.0
    raw_threshold, _ = best_balanced_threshold(y.astype(np.int8), _probabilities(x))
    raw_threshold = float(np.clip(raw_threshold, 1e-6, 1 - 1e-6))
    raw_logit = math.log(raw_threshold / (1.0 - raw_threshold))
    # Align the selected calibration operating point with the frozen 0.65 UI threshold.
    from .modeling import threshold_alignment

    bias = threshold_alignment(raw_logit, temperature, THRESHOLD)
    return temperature, float(bias), raw_threshold


def _metrics_report(samples: Iterable[Any], scores: np.ndarray) -> dict[str, Any]:
    samples = list(samples)
    labels = np.asarray([sample.label for sample in samples], dtype=np.int8)
    metrics = binary_metrics(labels, scores, threshold=THRESHOLD).to_dict()

    def grouped(attribute: str) -> dict[str, Any]:
        groups: defaultdict[str, list[int]] = defaultdict(list)
        for index, sample in enumerate(samples):
            groups[getattr(sample, attribute) or "unknown"].append(index)
        result: dict[str, Any] = {}
        for name, indices in sorted(groups.items()):
            result[name] = binary_metrics(labels[indices], scores[indices], threshold=THRESHOLD).to_dict()
        return result

    return {
        "threshold": THRESHOLD,
        "count": len(samples),
        "metrics": metrics,
        **metrics,
        "per_source": grouped("source"),
        "per_generator": grouped("generator"),
    }


def _dataset(manifest: str | Path, split: str, *, training: bool = False):
    ManifestDataset, load_manifest = _data_helpers()
    samples = load_manifest(manifest)
    return samples, ManifestDataset(samples, split, training=training)


def train(args: argparse.Namespace) -> int:
    torch = _torch()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    samples, dataset = _dataset(args.manifest, args.split, training=True)
    _validation_samples, validation = _dataset(args.manifest, args.validation_split)
    if args.initial_checkpoint:
        model, _ = _load_checkpoint(args.initial_checkpoint, device=args.device)
    else:
        model = _build_model(pretrained=args.pretrained).to(args.device)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=dataset.balanced_sampler(args.seed),
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.workers > 0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    losses: list[float] = []
    epochs: list[dict[str, float]] = []
    best_accuracy = -1.0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        for images, labels, _indices in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                loss = criterion(model(images.to(args.device, non_blocking=True)), labels.float().to(args.device, non_blocking=True))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        validation_logits = _raw_scores(model, validation, batch_size=args.batch_size, device=args.device)
        validation_scores = _probabilities(validation_logits)
        validation_threshold, validation_metrics = best_balanced_threshold(_labels(validation), validation_scores)
        validation_accuracy = validation_metrics.balanced_accuracy
        epochs.append({"epoch": epoch + 1, "loss": losses[-1], "validation_balanced_accuracy": validation_accuracy, "validation_threshold": validation_threshold})
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {"seed": args.seed, "split": args.split, "epochs": epoch + 1},
                },
                output,
            )
        print(json.dumps(epochs[-1], sort_keys=True), flush=True)
    report = {
        "command": "train",
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "checkpoint": str(output.resolve()),
        "checkpoint_sha256": _sha256(output),
        "initial_checkpoint_sha256": _sha256_optional(args.initial_checkpoint),
        "split": args.split,
        "samples": len(dataset),
        "epochs": args.epochs,
        "best_validation_balanced_accuracy": best_accuracy,
        "epoch_metrics": epochs,
        "loss": losses[-1] if losses else None,
        "threshold": THRESHOLD,
    }
    _emit(report, args.report)
    return 0


def calibrate(args: argparse.Namespace) -> int:
    model, _checkpoint = _load_checkpoint(args.checkpoint, device=args.device)
    samples, dataset = _dataset(args.manifest, args.split)
    logits = _raw_scores(model, dataset, batch_size=args.batch_size, device=args.device)
    labels = _labels(dataset)
    temperature, bias, raw_threshold = _fit_calibration(logits, labels)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "temperature": temperature,
        "bias": bias,
        "raw_threshold": raw_threshold,
        "threshold": THRESHOLD,
    }
    output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    report = {
        "command": "calibrate",
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "calibration_sha256": _sha256(output),
        "split": args.split,
        "samples": len(dataset),
        "calibration": payload,
    }
    _emit(report, args.report)
    return 0


def evaluate(args: argparse.Namespace) -> int:
    model, _checkpoint = _load_checkpoint(args.checkpoint, device=args.device)
    samples, dataset = _dataset(args.manifest, args.split)
    logits = _raw_scores(model, dataset, batch_size=args.batch_size, device=args.device)
    temperature, bias = _load_calibration(args.calibration)
    scores = _probabilities(logits, temperature, bias)
    report = {
        "command": "evaluate",
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "calibration_sha256": _sha256_optional(args.calibration),
        "split": args.split,
        "calibration": {"temperature": temperature, "bias": bias},
        **_metrics_report(dataset.samples, scores),
    }
    _emit(report, args.report)
    return 0


def export_model(args: argparse.Namespace) -> int:
    torch = _torch()
    model, _checkpoint = _load_checkpoint(args.checkpoint, device=args.device)
    temperature, bias = _load_calibration(args.calibration)
    if args.calibration:
        class CalibratedLogit(torch.nn.Module):
            def __init__(self, detector, fitted_temperature: float, fitted_bias: float) -> None:
                super().__init__()
                self.detector = detector
                self.temperature = fitted_temperature
                self.bias = fitted_bias

            def forward(self, image):
                return self.detector(image) / self.temperature + self.bias

        model = CalibratedLogit(model, temperature, bias).to(args.device).eval()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, 3, 256, 256), dtype=torch.float32, device=args.device)
    torch.onnx.export(
        model,
        dummy,
        output,
        input_names=["image"],
        output_names=["score"],
        dynamic_axes={"image": {0: "batch"}, "score": {0: "batch"}},
        opset_version=args.opset,
        dynamo=False,
    )
    report = {
        "command": "export",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "calibration_sha256": _sha256_optional(args.calibration),
        "onnx": str(output.resolve()),
        "onnx_sha256": _sha256(output),
        "threshold": THRESHOLD,
        "calibration": {"temperature": temperature, "bias": bias},
    }
    _emit(report, args.report)
    return 0


def parity(args: argparse.Namespace) -> int:
    model, _checkpoint = _load_checkpoint(args.checkpoint, device=args.device)
    samples, dataset = _dataset(args.manifest, args.split)
    logits = _raw_scores(model, dataset, batch_size=args.batch_size, device=args.device)
    temperature, bias = _load_calibration(args.calibration)
    python_scores = _probabilities(logits, temperature, bias)
    import onnxruntime as ort

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    values: list[np.ndarray] = []
    torch = _torch()
    for start in range(0, len(dataset), args.batch_size):
        images = torch.stack([dataset[index][0] for index in range(start, min(start + args.batch_size, len(dataset)))])
        output = np.asarray(session.run(None, {input_name: images.numpy()})[0]).reshape(-1)
        output = _probabilities(output)
        values.append(output)
    onnx_scores = np.concatenate(values) if values else np.empty(0, dtype=np.float64)
    differences = np.abs(python_scores - onnx_scores)
    report: dict[str, Any] = {
        "command": "onnx-parity",
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "calibration_sha256": _sha256_optional(args.calibration),
        "onnx": str(Path(args.onnx).resolve()),
        "onnx_sha256": _sha256(args.onnx),
        "split": args.split,
        "threshold": THRESHOLD,
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
    if args.report:
        _emit(report, args.report)
    else:
        _emit(report)
    return 0


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
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
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--workers", type=int, default=4)
    train_parser.add_argument("--pretrained", action="store_true")
    train_parser.add_argument("--initial-checkpoint")
    train_parser.set_defaults(handler=train)

    for name, handler in (("calibrate", calibrate), ("evaluate", evaluate)):
        command = commands.add_parser(name)
        _common(command)
        command.add_argument("--output", required=(name == "calibrate"))
        command.set_defaults(handler=handler)

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
    parity_parser.set_defaults(handler=parity)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
