#!/usr/bin/env python3
"""Build the frozen Synthbuster + RAISE-1k evaluation manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import urllib.request
from pathlib import Path

INDEX_URL = "https://raw.githubusercontent.com/grip-unina/ClipBased-SyntheticImageDetection/3247caed4078825d756ee9d8497c935c149a2911/data/commercial_tools.csv"
FIELDS = ("path", "label", "source", "generator", "group", "split")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--index-url", default=INDEX_URL)
    args = parser.parse_args()
    with urllib.request.urlopen(args.index_url) as response:
        index = response.read()
    rows = []
    missing = []
    seen: set[str] = set()
    for item in csv.DictReader(io.StringIO(index.decode("utf-8"))):
        path = (args.root / item["filename"]).resolve()
        if not path.is_file():
            missing.append(str(path))
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        kind = item["typ"]
        rows.append({
            "path": path,
            "label": int(kind != "real"),
            "source": "synthbuster+raise-1k",
            "generator": "camera" if kind == "real" else kind,
            "group": f"synthbuster:{path.stem}",
            "split": "test",
        })
    if missing:
        raise SystemExit(f"missing {len(missing)} indexed images; first: {missing[0]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows={len(rows)} index_sha256={hashlib.sha256(index).hexdigest()} manifest_sha256={hashlib.sha256(args.output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
