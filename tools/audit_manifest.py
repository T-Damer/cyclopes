#!/usr/bin/env python3
"""Perceptually deduplicate a manifest and emit the paid-run preflight report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imagehash
from PIL import Image, ImageStat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cyclopes.data import load_manifest
try:
    from tools.merge_manifests import FIELDS
except ModuleNotFoundError:  # Direct script execution.
    from merge_manifests import FIELDS


def perceptual_signature(path: Path) -> tuple[int, int, float]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        contrast = sum(ImageStat.Stat(rgb.resize((64, 64))).stddev) / 3
        return int(str(imagehash.phash(rgb)), 16), int(str(imagehash.dhash(rgb)), 16), contrast


def near_duplicate(first: tuple[int, int], second: tuple[int, int], phash_distance: int, dhash_distance: int) -> bool:
    return (first[0] ^ second[0]).bit_count() <= phash_distance and (first[1] ^ second[1]).bit_count() <= dhash_distance


def low_information(signature: tuple[int, int, float], min_contrast: float) -> bool:
    return signature[2] < min_contrast


def normalized_group_split(sample) -> tuple[str, str]:
    if sample.source == "google/docci":
        prefix, number = sample.path.stem.rsplit("_", 1)
        group = f"docci:{prefix}:{int(number) // 5:05d}"
        bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
        return group, "train" if bucket < 80 else "calibration" if bucket < 88 else "validation" if bucket < 94 else "test"
    return sample.group, sample.split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--dhash-distance", type=int, default=8)
    parser.add_argument("--min-contrast", type=float, default=3.0)
    parser.add_argument("--min-class-count", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    samples = load_manifest(args.manifest)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        hashes = list(pool.map(perceptual_signature, (sample.path for sample in samples)))
    bands: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    kept = []
    kept_hashes = []
    dropped = []
    dropped_low_information = []
    for sample, value in zip(samples, hashes, strict=True):
        if low_information(value, args.min_contrast):
            dropped_low_information.append({"path": str(sample.path), "contrast": value[2]})
            continue
        candidates = set()
        for band in range(8):
            candidates.update(bands[(band, (value[0] >> (band * 8)) & 0xFF)])
        duplicate = next((index for index in sorted(candidates)
                          if near_duplicate((value[0], value[1]), (kept_hashes[index][0], kept_hashes[index][1]),
                                            args.distance, args.dhash_distance)), None)
        if duplicate is not None:
            previous = kept[duplicate]
            if previous.label != sample.label:
                raise ValueError(f"near-duplicate label conflict: {previous.path} and {sample.path}")
            dropped.append({"path": str(sample.path), "duplicate_of": str(previous.path),
                            "phash_distance": (value[0] ^ kept_hashes[duplicate][0]).bit_count(),
                            "dhash_distance": (value[1] ^ kept_hashes[duplicate][1]).bit_count()})
            continue
        index = len(kept)
        kept.append(sample)
        kept_hashes.append(value)
        for band in range(8):
            bands[(band, (value[0] >> (band * 8)) & 0xFF)].append(index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    normalized = []
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for sample in kept:
            group, split = normalized_group_split(sample)
            normalized.append((sample, group, split))
            writer.writerow({"path": sample.path, "label": sample.label, "source_dataset": sample.source,
                             "generator_model": sample.generator, "content_group": group, "split": split,
                             "family": sample.generator_family, "domain": sample.content_domain, "license": sample.license,
                             "sha256": sample.sha256 or hashlib.sha256(sample.path.read_bytes()).hexdigest()})
    class_counts = Counter(sample.label for sample in kept)
    if min(class_counts.values(), default=0) < args.min_class_count:
        raise ValueError(f"insufficient class count after deduplication: {dict(class_counts)}")
    report = {
        "input_rows": len(samples), "kept_rows": len(kept), "dropped_near_duplicates": len(dropped),
        "dropped_low_information": len(dropped_low_information),
        "class_counts": dict(class_counts), "split_counts": dict(Counter(split for _sample, _group, split in normalized)),
        "source_counts": dict(Counter(sample.source for sample in kept)),
        "family_counts": dict(Counter(sample.generator_family for sample in kept)),
        "domain_counts": dict(Counter(sample.content_domain for sample in kept)),
        "license_counts": dict(Counter(sample.license for sample in kept)),
        "manifest_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(), "examples_dropped": dropped[:25],
        "examples_low_information": dropped_low_information[:25],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
