#!/usr/bin/env python3
"""Merge normalized manifests while preserving absolute image paths."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

FIELDS = ("path", "label", "source_dataset", "generator_model", "content_group", "split", "family", "domain", "license", "sha256")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("manifests", nargs="+", type=Path)
    args = parser.parse_args()
    rows = []
    paths = set()
    digests = set()
    group_splits: dict[str, str] = {}
    for manifest in args.manifests:
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                path = (manifest.parent / row["path"]).resolve() if not Path(row["path"]).is_absolute() else Path(row["path"])
                digest = row.get("sha256") or hashlib.sha256(path.read_bytes()).hexdigest()
                group = row.get("content_group") or row.get("group", "")
                split = row["split"]
                if path in paths or digest in digests:
                    continue
                if group in group_splits and group_splits[group] != split:
                    raise ValueError(f"group crosses splits: {group}")
                paths.add(path)
                digests.add(digest)
                group_splits[group] = split
                rows.append({
                    "path": path, "label": row["label"], "source_dataset": row.get("source_dataset") or row.get("source", ""),
                    "generator_model": row.get("generator_model") or row.get("generator", ""), "content_group": group,
                    "split": split, "family": row.get("family", "real" if row["label"] == "0" else "modern"),
                    "domain": row.get("domain", "mixed"), "license": row.get("license", ""), "sha256": digest,
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows={len(rows)} sha256={hashlib.sha256(args.output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
