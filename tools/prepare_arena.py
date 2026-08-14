#!/usr/bin/env python3
"""Convert the frozen AI Detector Arena v0.1 archive into evaluation manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from PIL import Image

FIELDS = ("path", "label", "source", "generator", "group", "split")


def write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--skip-stress", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    rows: list[dict[str, object]] = []
    stress: list[dict[str, object]] = []
    missing = 0
    seen: set[str] = set()
    stress_seen: set[str] = set()
    with (root / "metadata/images_metadata.csv").open(newline="", encoding="utf-8") as handle:
        for item in csv.DictReader(handle):
            source = root / item["filename"]
            if not source.is_file():
                missing += 1
                continue
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            label = int(item["is_ai"].lower() == "true")
            generator = item["generator"] or "camera"
            base = {"label": label, "source": "ai-detector-arena-v0.1", "generator": generator, "group": item["id"], "split": "test"}
            rows.append({"path": source.relative_to(root), **base})
            if args.skip_stress:
                continue
            with Image.open(source) as opened:
                image = opened.convert("RGB")
                width, height = image.size
                variants = {
                    "jpeg70": image,
                    "resize50": image.resize((max(32, width // 2), max(32, height // 2)), Image.Resampling.BILINEAR).resize(image.size, Image.Resampling.BILINEAR),
                    "crop80": image.crop((width // 10, height // 10, width - width // 10, height - height // 10)).resize(image.size, Image.Resampling.BILINEAR),
                }
                for name, variant in variants.items():
                    target = root / "stress" / name / f'{item["id"]}.jpg'
                    target.parent.mkdir(parents=True, exist_ok=True)
                    variant.save(target, "JPEG", quality=70 if name == "jpeg70" else 92, subsampling=2)
                    digest = hashlib.sha256(target.read_bytes()).hexdigest()
                    if digest in stress_seen:
                        target.unlink()
                        continue
                    stress_seen.add(digest)
                    stress.append({"path": target.relative_to(root), **base})
    write(root / "manifest.csv", rows)
    write(root / "stress-manifest.csv", stress)
    print(f"original={len(rows)} stress={len(stress)} missing={missing}")


if __name__ == "__main__":
    main()
