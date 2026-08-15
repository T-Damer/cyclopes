#!/usr/bin/env python3
"""Materialize pinned AI/real pixel-art pairs for expert training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

from PIL import Image


AI_DATASET = "jainr3/diffusiondb-pixelart"
AI_REVISION = "05bae04ee9a090d2935f0f17a978b94858f39083"
REAL_DATASET = "bghira/free-to-use-pixelart"
REAL_REVISION = "53caca03739b797f0a9924a7879babe98dc943bd"
FIELDS = ("path", "label", "source_dataset", "generator_model", "content_group", "split", "family", "domain", "license", "sha256")


def split_for(group: str) -> str:
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "calibration" if bucket < 88 else "validation" if bucket < 94 else "test"


def save(image: Image.Image, target: Path) -> str:
    image.convert("RGB").save(target, "JPEG", quality=92, subsampling=0)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=2_000)
    args = parser.parse_args()

    import pandas as pd
    from huggingface_hub import hf_hub_download

    images = args.output / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows = []
    seen = set()
    for part in ("images/part-000001.zip", "images/part-000002.zip"):
        archive = hf_hub_download(AI_DATASET, part, repo_type="dataset", revision=AI_REVISION)
        with ZipFile(archive) as handle:
            for name in handle.namelist():
                if len(rows) >= args.limit:
                    break
                try:
                    with Image.open(BytesIO(handle.read(name))) as opened:
                        target = images / f"ai-{len(rows):05d}.jpg"
                        digest = save(opened, target)
                except (OSError, ValueError):
                    continue
                if digest in seen:
                    target.unlink()
                    continue
                seen.add(digest)
                group = f"diffusiondb-pixelart:{Path(name).stem}"
                rows.append({"path": target.relative_to(args.output), "label": 1, "source_dataset": AI_DATASET,
                             "generator_model": "stable-diffusion-pixelart", "content_group": group,
                             "split": split_for(group), "family": "diffusion", "domain": "pixel-art",
                             "license": "CC0-1.0", "sha256": digest})

    parquet = hf_hub_download(REAL_DATASET, "pixilart.parquet", repo_type="dataset", revision=REAL_REVISION)
    records = pd.read_parquet(parquet).head(args.limit).to_dict("records")

    def fetch(record: dict) -> bytes | None:
        try:
            request = Request(record["image_url"], headers={"User-Agent": "cyclopes-dataset/1.0"})
            with urlopen(request, timeout=20) as response:
                return response.read()
        except OSError:
            return None

    with ThreadPoolExecutor(max_workers=16) as pool:
        for record, payload in zip(records, pool.map(fetch, records), strict=True):
            if not payload:
                continue
            try:
                with Image.open(BytesIO(payload)) as opened:
                    target = images / f"real-{len(rows):05d}.jpg"
                    digest = save(opened, target)
            except (OSError, ValueError):
                continue
            if digest in seen:
                target.unlink()
                continue
            seen.add(digest)
            group = f"pixilart:{record['image_hash']}"
            rows.append({"path": target.relative_to(args.output), "label": 0, "source_dataset": REAL_DATASET,
                         "generator_model": "human-created", "content_group": group, "split": split_for(group),
                         "family": "real", "domain": "pixel-art", "license": "MIT", "sha256": digest})

    manifest = args.output / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report = {"rows": len(rows), "ai_rows": sum(row["label"] for row in rows),
              "real_rows": sum(not row["label"] for row in rows),
              "sources": {AI_DATASET: {"revision": AI_REVISION, "license": "CC0-1.0"},
                          REAL_DATASET: {"revision": REAL_REVISION, "license": "MIT"}},
              "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    (args.output / "provenance.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
