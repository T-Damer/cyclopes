#!/usr/bin/env python3
"""Score a manifest with the Apache-2.0 Sentry ConvNeXt teacher."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from aidetector.sentry_adapter import SentryConvNeXtDetector, preprocess_sentry_image

from cyclopes.metrics import binary_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    root = args.manifest.resolve().parent
    for row in rows:
        path = Path(row["path"])
        row["path"] = str(path if path.is_absolute() else root / path)
    detector = SentryConvNeXtDetector(device="cuda")

    def load(row: dict[str, str]):
        with Image.open(row["path"]) as image:
            return preprocess_sentry_image(image)

    scores: list[float] = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for start in range(0, len(rows), 512):
            batch = rows[start : start + 512]
            images = detector._torch.stack(list(pool.map(load, batch))).to(detector.device)
            with detector._torch.inference_mode(), detector._torch.autocast("cuda", dtype=detector._torch.bfloat16):
                probabilities = detector._torch.softmax(detector.model(images), dim=-1)[:, 0].float().cpu()
            scores.extend(probabilities.tolist())
            print(f"{len(scores)}/{len(rows)}", flush=True)

    labels = np.asarray([int(row["label"]) for row in rows])
    values = np.asarray(scores)
    order = np.argsort(-values, kind="mergesort")
    positives = labels.sum()
    negatives = len(labels) - positives
    true_positive = np.cumsum(labels[order])
    false_positive = np.cumsum(1 - labels[order])
    index = int(np.argmax((true_positive / positives + (negatives - false_positive) / negatives) / 2))
    threshold = float(values[order[index]])
    report = {
        "teacher": "InfImagine/Sentry_image_models:convnext_small_4xb256_fake5m-lr4e-4/epoch_15.pth",
        "license": "Apache-2.0",
        "fixed_0_5": binary_metrics(labels, scores, 0.5).to_dict(),
        "best_threshold": threshold,
        "best": binary_metrics(labels, scores, threshold).to_dict(),
        "predictions": [
            {"path": row["path"], "label": int(row["label"]), "score": float(score)}
            for row, score in zip(rows, scores, strict=True)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
