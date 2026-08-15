#!/usr/bin/env python3
"""Add CC0 real artwork from a Met Open Access parquet shard."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

FIELDS = ("path", "label", "source_dataset", "generator_model", "content_group", "split", "family", "domain", "license", "sha256")


def split_for(group: str) -> str:
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "calibration" if bucket < 88 else "validation" if bucket < 94 else "test"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", type=Path)
    parser.add_argument("base_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=10_000)
    args = parser.parse_args()
    output = args.output.resolve()
    image_root = output.parent / "images" / "met"
    with args.base_manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["path"] = str((args.base_manifest.parent / row["path"]).resolve())

    table = pq.read_table(args.parquet, columns=["isPublicDomain", "objectID", "image"])
    added = 0
    seen: set[str] = set()
    columns = table.to_pydict()
    for public_domain, object_id, encoded in zip(
        columns["isPublicDomain"], columns["objectID"], columns["image"], strict=True
    ):
        if not public_domain or not encoded or added >= args.limit:
            continue
        target = image_root / f"{object_id}.jpg"
        try:
            payload = encoded["bytes"] if isinstance(encoded, dict) else encoded
            with Image.open(io.BytesIO(payload)) as image:
                target.parent.mkdir(parents=True, exist_ok=True)
                image.convert("RGB").save(target, "JPEG", quality=92, subsampling=0)
        except Exception as error:
            print(f"skip {object_id}: {error}")
            continue
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest in seen:
            target.unlink()
            continue
        seen.add(digest)
        group = f"met:{object_id}"
        rows.append({
            "path": target,
            "label": 0,
            "source_dataset": "metmuseum/openaccess",
            "generator_model": "human-art",
            "content_group": group,
            "split": split_for(group),
            "family": "real",
            "domain": "traditional-art",
            "license": "CC0-1.0",
            "sha256": digest,
        })
        added += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"met={added} rows={len(rows)} sha256={hashlib.sha256(output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
