#!/usr/bin/env python3
"""Add unique PAMELA Flux 2 / Nano Banana images to a training manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

FIELDS = ("path", "label", "source", "generator", "group", "split")


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
        path = (args.root / relative.removeprefix("./")).resolve()
        if not path.is_file() or path.stat().st_size < 10_000:
            continue
        group = f'pamela:{item["image_id"]}'
        rows.append({
            "path": path,
            "label": 1,
            "source": "pamela-dataset/pamela",
            "generator": "nano-banana-pro" if "NanoBananaPro" in path.name else "flux-2",
            "group": group,
            "split": split_for(group),
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
