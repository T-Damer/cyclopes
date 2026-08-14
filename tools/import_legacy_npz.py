#!/usr/bin/env python3
"""Convert an official generator sample archive to the Cyclopes manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from generate_legacy import FIELDS, save, split_for


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--generator", default="adm")
    args = parser.parse_args()
    root = args.output.resolve()
    images = np.load(args.archive)["arr_0"]
    rows = []
    for index, array in enumerate(images):
        path = root / "images" / f"{index:06d}.jpg"
        digest = save(Image.fromarray(array), path)
        group = f"{args.generator}:{index:06d}"
        rows.append({"path": path.relative_to(root), "label": 1, "source_dataset": f"self-generated/{args.generator}",
                     "generator_model": args.generator, "content_group": group, "split": split_for(group),
                     "family": "legacy-diffusion", "domain": "mixed", "license": "MIT", "sha256": digest})
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows={len(rows)} sha256={hashlib.sha256(manifest.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
