#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from datasets import load_dataset
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("output")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--config")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--license", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        revision=args.revision,
        streaming=True,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HF_READ_ONLY_TOKEN"),
    )
    count = 0
    for row_index, row in enumerate(dataset):
        value = row.get(args.image_column)
        if value is None:
            continue
        try:
            image = value if isinstance(value, Image.Image) else Image.open(value)
            image = image.convert("RGB")
            digest = hashlib.sha256(f"{args.dataset}:{args.revision}:{args.split}:{row_index}".encode()).hexdigest()
            image.save(output / f"{digest}.jpg", "JPEG", quality=95, subsampling=0)
        except Exception:
            continue
        count += 1
        if count >= args.limit:
            break
    if count < args.limit:
        raise RuntimeError(f"materialized {count} images, expected {args.limit}")
    provenance = {
        "dataset": args.dataset,
        "revision": args.revision,
        "config": args.config,
        "split": args.split,
        "image_column": args.image_column,
        "count": count,
        "license": args.license,
        "codec": "JPEG quality=95 subsampling=0",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
