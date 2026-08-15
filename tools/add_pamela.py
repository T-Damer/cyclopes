#!/usr/bin/env python3
"""Add unique PAMELA Flux 2 / Nano Banana images to a training manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image

FIELDS = ("path", "label", "source_dataset", "generator_model", "content_group", "split", "family", "domain", "license", "sha256")


def split_for(group: str) -> str:
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "calibration" if bucket < 88 else "validation" if bucket < 94 else "test"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("base_manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    with args.base_manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["path"] = str((args.base_manifest.parent / row["path"]).resolve())

    unique: dict[str, dict] = {}
    for annotation in sorted((args.root / "annotations").glob("*.json")):
        for item in json.loads(annotation.read_text()):
            unique[item["image_path"]] = item
    for relative, item in sorted(unique.items()):
        source = (args.root / relative.removeprefix("./")).resolve()
        if not source.is_file() or source.stat().st_size < 10_000:
            continue
        path = args.output.parent / "images" / "pamela" / f'{item["image_id"]}.jpg'
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(source) as image:
                image.convert("RGB").save(path, "JPEG", quality=92, subsampling=0)
        except OSError:
            continue
        group = f'pamela:{item["image_id"]}'
        rows.append({
            "path": path,
            "label": 1,
            "source_dataset": "pamela-dataset/pamela",
            "generator_model": "pamela-modern",
            "content_group": group,
            "split": split_for(group),
            "family": "modern",
            "domain": "mixed",
            "license": "CC-BY-4.0",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["path"] = Path(row["path"]).resolve()
            writer.writerow(row)
    print(f"pamela={len(unique)} rows={len(rows)} sha256={hashlib.sha256(args.output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
