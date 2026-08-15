#!/usr/bin/env python3
"""Materialize the pinned Apache-2.0 Memotion mirror as real meme negatives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


DATASET = "Leonardo6/memotion"
REVISION = "17fd08d1e0ef4acb01a2e10a2801ab3b6d869a12"
FIELDS = ("path", "label", "source_dataset", "generator_model", "content_group", "split", "family", "domain", "license", "sha256")


def split_for(group: str) -> str:
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "calibration" if bucket < 88 else "validation" if bucket < 94 else "test"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=6_000)
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset(DATASET, revision=REVISION, split="train")
    images = args.output / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows = []
    seen = set()
    for index, item in enumerate(dataset):
        if len(rows) >= args.limit:
            break
        source_images = item.get("images") or []
        if not source_images:
            continue
        target = images / f"{index:05d}.jpg"
        source_images[0].convert("RGB").save(target, "JPEG", quality=92, subsampling=0)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest in seen:
            target.unlink()
            continue
        seen.add(digest)
        group = f"memotion:{index}"
        rows.append({"path": target.relative_to(args.output), "label": 0, "source_dataset": DATASET,
                     "generator_model": "human-created", "content_group": group, "split": split_for(group),
                     "family": "real", "domain": "meme", "license": "Apache-2.0", "sha256": digest})

    manifest = args.output / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report = {"dataset": DATASET, "revision": REVISION, "license": "Apache-2.0", "rows": len(rows),
              "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    (args.output / "provenance.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
