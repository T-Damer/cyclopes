#!/usr/bin/env python3
"""Build a codec-normalized, commercially usable training manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

SOURCES = (
    ("bitmind/nano-banana", "9ea8da32a5be03f4946e6cb10c2d2f8e90f0a0a4", "gemini-2.5-flash-image", 300, "MIT"),
    ("ash12321/nano-banana-pro-generated-1k", "01f43edd35eb2f47f5ea77cfb223d173083d61ad", "nano-banana-pro", 200, "MIT"),
)
DOCCI_REVISION = "a0a43eaf34676ffd008fb6565dd8c2ba00d09100"


def split_for(group: str) -> str:
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 88:
        return "calibration"
    if bucket < 94:
        return "validation"
    return "test"


def save(image: Image.Image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, "JPEG", quality=92, subsampling=0)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docci-root", type=Path, required=True)
    parser.add_argument("--cocoxgen-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    images = output / "images"
    rows: list[dict[str, object]] = []
    seen: set[str] = set()

    descriptions = args.docci_root / "docci_descriptions.jsonlines"
    docci = [json.loads(line) for line in descriptions.read_text().splitlines()]

    def process_docci(item: dict[str, str]):
        source = args.docci_root / "images" / item["image_file"]
        target = images / "real" / f'{item["example_id"]}.jpg'
        with Image.open(source) as image:
            digest = save(image, target)
        return item, target, digest

    with ThreadPoolExecutor(max_workers=16) as pool:
        for item, target, digest in pool.map(process_docci, docci):
            if digest in seen:
                target.unlink()
                continue
            seen.add(digest)
            group = f'docci:{item["example_id"]}'
            rows.append({"path": target, "label": 0, "source": "google/docci", "generator": "camera", "group": group, "split": split_for(group)})

    from datasets import load_dataset

    provenance = [{"dataset": "google/docci", "revision": DOCCI_REVISION, "license": "CC-BY-4.0", "rows": len(rows)}]
    for dataset_id, revision, generator, limit, license_name in SOURCES:
        count = 0
        dataset = load_dataset(dataset_id, revision=revision, streaming=True, split="train")
        for index, item in enumerate(dataset):
            if count >= limit:
                break
            target = images / "ai" / generator / f"{index:06d}.jpg"
            try:
                digest = save(item["image"], target)
            except Exception as error:
                print(f"skip {dataset_id}:{index}: {error}")
                continue
            if digest in seen:
                target.unlink()
                continue
            seen.add(digest)
            group = f"{dataset_id}:{item.get('id', index)}"
            rows.append({"path": target, "label": 1, "source": dataset_id, "generator": generator, "group": group, "split": split_for(group)})
            count += 1
        provenance.append({"dataset": dataset_id, "revision": revision, "license": license_name, "rows": count})

    cocoxgen_sources = sorted(path for directory in ("fooocus", "sdxl") for path in (args.cocoxgen_root / directory).glob("*.*"))

    def process_cocoxgen(source: Path):
        generator = source.parent.name
        target = images / "ai" / "cocoxgen" / f"{generator}-{source.stem}.jpg"
        with Image.open(source) as image:
            digest = save(image, target)
        return source, target, digest, generator

    count = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        for source, target, digest, generator in pool.map(process_cocoxgen, cocoxgen_sources):
            if digest in seen:
                target.unlink()
                continue
            seen.add(digest)
            group = f"cocoxgen:{source.stem.split('-')[0]}"
            rows.append({"path": target, "label": 1, "source": "heikeadel/cocoxgen", "generator": generator, "group": group, "split": split_for(group)})
            count += 1
    provenance.append({"dataset": "heikeadel/cocoxgen", "revision": "c336ad187c2ab298ce825df65088bdacbae104f6", "license": "CC-BY-4.0", "rows": count})

    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "label", "source", "generator", "group", "split"))
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["path"] = Path(row["path"]).relative_to(output)
            writer.writerow(row)
    payload = {
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "rows": len(rows),
        "sources": provenance,
        "codec": "JPEG quality=92 subsampling=0",
    }
    (output / "provenance.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
