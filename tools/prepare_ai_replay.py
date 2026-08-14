#!/usr/bin/env python3
"""Download a compact modern-AI replay set without retaining source archives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import hf_hub_url, list_repo_files
from PIL import Image, ImageFile

try:
    from tools.prepare_hard_negatives import FIELDS, split_for
except ModuleNotFoundError:  # direct script execution
    from prepare_hard_negatives import FIELDS, split_for

ImageFile.LOAD_TRUNCATED_IMAGES = True

SOURCES = {
    "cocoxgen": ("heikeadel/cocoxgen", "c336ad187c2ab298ce825df65088bdacbae104f6", "CC-BY-4.0"),
    "pamela": ("pamela-dataset/pamela", "14ebd68d2a2c34367d41020b62ee60b7504725fb", "CC-BY-4.0"),
}


def cocoxgen_group(filename: str) -> str:
    stem = Path(filename).stem
    return stem.replace("-long-fooocus", "").replace("-short-fooocus", "").replace("-long-sdxl", "").replace("-short-sdxl", "")


def fetch(url: str) -> bytes:
    headers = {"User-Agent": "Cyclopes dataset builder/0.2"}
    token = os.environ.get("HF_READ_ONLY_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                payload = response.read(20 * 1024 * 1024 + 1)
            if len(payload) > 20 * 1024 * 1024:
                raise ValueError("image exceeds 20 MiB")
            return payload
        except (OSError, urllib.error.HTTPError):
            if attempt == 4:
                raise
            time.sleep(min(8, 2 ** attempt))
    raise RuntimeError("unreachable")


def download(item: tuple[str, str], root: Path) -> dict | None:
    source, filename = item
    dataset, revision, license_name = SOURCES[source]
    target = root / "images" / source / f"{hashlib.sha256(filename.encode()).hexdigest()[:20]}.jpg"
    try:
        if not target.is_file():
            with Image.open(io.BytesIO(fetch(hf_hub_url(dataset, filename, repo_type="dataset", revision=revision)))) as opened:
                image = opened.convert("RGB")
            if min(image.size) < 128:
                return None
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target, "JPEG", quality=90, subsampling=0)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
    except (OSError, ValueError):
        return None
    identity = cocoxgen_group(filename) if source == "cocoxgen" else Path(filename).stem
    group = f"{source}:{identity}"
    generator = Path(filename).parent.name if source == "cocoxgen" else "pamela-modern"
    return {
        "path": target.relative_to(root), "label": 1, "source_dataset": dataset,
        "generator_model": generator, "content_group": group, "split": split_for(group),
        "family": "modern", "domain": "mixed", "license": license_name, "sha256": digest,
    }


def choose(source: str, count: int) -> list[tuple[str, str]]:
    dataset, revision, _license = SOURCES[source]
    files = sorted(path for path in list_repo_files(dataset, repo_type="dataset", revision=revision)
                   if path.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
    if source == "cocoxgen":
        fooocus = [path for path in files if path.startswith("fooocus/")]
        sdxl = [path for path in files if path.startswith("sdxl/")]
        files = fooocus[: count // 2] + sdxl[: count - count // 2]
    return [(source, path) for path in files[:count]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--cocoxgen", type=int, default=2_750)
    parser.add_argument("--pamela", type=int, default=2_750)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.output.resolve()
    items = choose("cocoxgen", args.cocoxgen) + choose("pamela", args.pamela)
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(lambda item: download(item, root), items):
            if result is not None:
                rows.append(result)
                if len(rows) % 100 == 0:
                    print(f"ai-replay: {len(rows)}/{len(items)}", flush=True)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "rows": len(rows), "requested": len(items), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "sources": [{"dataset": dataset, "revision": revision, "license": license_name}
                    for dataset, revision, license_name in SOURCES.values()],
    }
    (root / "provenance.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
