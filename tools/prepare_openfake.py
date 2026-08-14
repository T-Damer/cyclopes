#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from PIL import Image


DATASET_ID = "ComplexDataLab/OpenFake"
REVISION = "3fd1109dc3258874243fa31c5bda9ee24260163b"
DATASET_LICENSE = "cc-by-nc-4.0"
FIELDS = ("path", "label", "source", "generator", "group", "split")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a balanced, metadata-free OpenFake subset")
    parser.add_argument("--output", type=Path, default=Path("data/openfake"))
    parser.add_argument("--train-per-generator", type=int, default=600)
    parser.add_argument("--validation-per-generator", type=int, default=100)
    parser.add_argument("--test-per-generator", type=int, default=200)
    parser.add_argument("--real-per-source", type=int, default=20_000)
    parser.add_argument("--max-edge", type=int, default=768)
    parser.add_argument("--seed", type=int, default=323)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--include-reddit", action="store_true")
    return parser.parse_args()


def encoded_image(image: Image.Image, max_edge: int) -> bytes:
    image = image.convert("RGB")
    image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, "WEBP", quality=95, method=6)
    return output.getvalue()


def stable_group(config: str, upstream_split: str, index: int, model: str) -> str:
    value = f"{config}:{upstream_split}:{model}:{index}".encode()
    return hashlib.sha256(value).hexdigest()[:24]


def target_split(upstream: str, group: str) -> str:
    if upstream == "train":
        return "train"
    if upstream == "test":
        return "test"
    bucket = int(group[:8], 16) % 2
    return "calibration" if bucket == 0 else "validation"


def materialize(
    config: str,
    upstream_split: str,
    limit_fake: int,
    limit_real: int,
    args: argparse.Namespace,
    writer: csv.DictWriter,
    counts: Counter,
    seen_digests: dict[str, tuple[str, int]] | None = None,
) -> None:
    if seen_digests is None:
        seen_digests = {}
    dataset = load_dataset(
        DATASET_ID,
        config,
        split=upstream_split,
        revision=args.revision,
        streaming=True,
    ).shuffle(seed=args.seed, buffer_size=10_000)

    for index, row in enumerate(dataset):
        label = 1 if row["label"] == "fake" else 0
        model = str(row["model"])
        quota = limit_fake if label else limit_real
        key = (config, upstream_split, label, model)
        if counts[key] >= quota:
            continue

        group = stable_group(config, upstream_split, index, model)
        split = target_split(upstream_split, group)
        image_bytes = encoded_image(row["image"], args.max_edge)
        digest = hashlib.sha256(image_bytes).hexdigest()
        previous = seen_digests.get(digest)
        if previous is not None:
            if previous != (split, label):
                raise ValueError(
                    f"duplicate image content crosses split or label: {digest} "
                    f"({previous[0]}, {previous[1]}) vs ({split}, {label})"
                )
            continue
        seen_digests[digest] = (split, label)
        relative = Path("images") / split / f"{digest}.webp"
        destination = args.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(image_bytes)
        writer.writerow(
            {
                "path": relative.as_posix(),
                "label": label,
                "source": f"openfake_{config}_{model if label == 0 else 'synthetic'}",
                "generator": model if label else "",
                "group": group,
                "split": split,
            }
        )
        counts[key] += 1


def main() -> None:
    args = parse_args()
    if min(args.train_per_generator, args.validation_per_generator, args.test_per_generator, args.real_per_source) < 1:
        raise SystemExit("all sample limits must be positive")
    random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.csv"
    counts: Counter = Counter()
    seen_digests: dict[str, tuple[str, int]] = {}

    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        materialize("core", "train", args.train_per_generator, args.real_per_source, args, writer, counts, seen_digests)
        materialize("core", "validation", args.validation_per_generator, args.validation_per_generator, args, writer, counts, seen_digests)
        materialize("core", "test", args.test_per_generator, args.test_per_generator, args, writer, counts, seen_digests)
        if args.include_reddit:
            materialize("reddit", "test", args.test_per_generator, args.test_per_generator, args, writer, counts, seen_digests)

    provenance = {
        "dataset": DATASET_ID,
        "revision": args.revision,
        "license": DATASET_LICENSE,
        "seed": args.seed,
        "max_edge": args.max_edge,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "counts": {"|".join(map(str, key)): value for key, value in sorted(counts.items())},
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
