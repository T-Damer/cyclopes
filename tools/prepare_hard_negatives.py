#!/usr/bin/env python3
"""Build a small, licensed hard-negative set from browser-sized previews."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import time
import urllib.parse
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageFile
import pyarrow.parquet as pq

ImageFile.LOAD_TRUNCATED_IMAGES = True

OGA_ID = "nyuuzyou/OpenGameArt-CC0"
OGA_REVISION = "9c26738cbece58950c8208debb0d3f257fa7c975"
COMMON_ID = "common-canvas/commoncatalog-cc-by"
COMMON_REVISION = "80f50fe4a1ca937f37a11be3f8eee5199d776ff3"
ROWS_API = "https://datasets-server.huggingface.co/rows?"
OGA_PARQUET = "https://huggingface.co/datasets/nyuuzyou/OpenGameArt-CC0/resolve/refs%2Fconvert%2Fparquet/default/{split}/0000.parquet"
FIELDS = (
    "path", "label", "source_dataset", "generator_model", "content_group",
    "split", "family", "domain", "license", "sha256",
)
BANNED = ("midjourney", "stable diffusion", "ai generated", "ai-generated", "generative ai")


def split_for(group: str) -> str:
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "calibration" if bucket < 88 else "validation" if bucket < 94 else "test"


def year(value: object) -> int | None:
    match = re.search(r"\b(19|20)\d{2}\b", str(value or ""))
    return int(match.group()) if match else None


def eligible_oga(row: dict) -> bool:
    text = " ".join((str(row.get("title", "")), str(row.get("description", "")), " ".join(row.get("tags") or []))).lower()
    return (
        row.get("licenses") == ["CC0"]
        and bool(row.get("preview_images"))
        and (year(row.get("post_date")) or 9999) <= 2021
        and not any(term in text for term in BANNED)
        and not any(str(tag).lower() in {"font", "music", "sound", "sound effect"} for tag in (row.get("tags") or []))
    )


def eligible_photo(row: dict) -> bool:
    taken = year(row.get("datetaken"))
    text = " ".join(str(row.get(key, "")) for key in ("title", "caption", "blip2_caption", "description")).lower()
    return (
        bool(row.get("jpg", {}).get("src"))
        and bool(row.get("capturedevice"))
        and taken is not None
        and taken <= 2021
        and min(int(row.get("width") or 0), int(row.get("height") or 0)) >= 256
        and not any(term in text for term in BANNED)
    )


def api_rows(dataset: str, split: str, offset: int, length: int = 100) -> list[dict]:
    query = urllib.parse.urlencode({"dataset": dataset, "config": "default", "split": split, "offset": offset, "length": length})
    request = urllib.request.Request(ROWS_API + query, headers={"User-Agent": "Cyclopes dataset builder/0.2"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                rows = [item["row"] for item in json.load(response)["rows"]]
            time.sleep(0.2)
            return rows
        except (urllib.error.HTTPError, urllib.error.URLError) as error:
            if (isinstance(error, urllib.error.HTTPError) and error.code not in {429, 500, 502, 503, 504}) or attempt == 5:
                raise
            time.sleep(min(8, 2 ** attempt))
    raise RuntimeError("unreachable")


def candidates(dataset: str, split: str, count: int, predicate, group_key, source_url) -> list[dict]:
    selected: list[dict] = []
    authors: Counter[str] = Counter()
    offset = 0
    while len(selected) < count * 2:
        if dataset == OGA_ID:
            request = urllib.request.Request(OGA_PARQUET.format(split=split), headers={"User-Agent": "Cyclopes dataset builder/0.2"})
            with urllib.request.urlopen(request, timeout=60) as response:
                rows = pq.read_table(io.BytesIO(response.read())).to_pylist()
        else:
            rows = api_rows(dataset, split, offset)
        if not rows:
            break
        offset += len(rows)
        for row in rows:
            group = group_key(row)
            if predicate(row) and group and authors[group] < 5:
                authors[group] += 1
                selected.append({"row": row, "group": group, "url": source_url(row)})
                if len(selected) >= count * 2:
                    break
        if dataset == OGA_ID:
            break
    return selected


def download(item: dict, root: Path, source: str, domain: str) -> dict | None:
    identity = str(item["row"].get("url") or item["row"].get("pageurl") or item["url"])
    stem = hashlib.sha256(identity.encode()).hexdigest()[:20]
    target = root / "images" / source / f"{stem}.jpg"
    try:
        if not target.is_file():
            request = urllib.request.Request(item["url"].replace(" ", "%20"), headers={"User-Agent": "Cyclopes dataset builder/0.2"})
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = response.read(20 * 1024 * 1024 + 1)
            if len(payload) > 20 * 1024 * 1024:
                return None
            with Image.open(io.BytesIO(payload)) as opened:
                image = opened.convert("RGB")
            if min(image.size) < 128:
                return None
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target, "JPEG", quality=90, subsampling=0)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
    except (OSError, ValueError):
        return None
    group = f"{source}:{item['group']}"
    license_name = "CC0-1.0" if source == "opengameart" else str(item["row"].get("licenseurl") or "CC-BY")
    return {
        "path": target.relative_to(root), "label": 0, "source_dataset": source,
        "generator_model": "human-created", "content_group": group, "split": split_for(group),
        "family": "real", "domain": domain, "license": license_name, "sha256": digest,
        "source_url": identity, "author": item["row"].get("author") or item["row"].get("unickname"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--2d", type=int, default=3_000, dest="art_2d")
    parser.add_argument("--3d", type=int, default=1_500, dest="art_3d")
    parser.add_argument("--photos", type=int, default=1_000)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    root = args.output.resolve()

    specs = (
        ("2d_art", args.art_2d, eligible_oga, lambda row: str(row.get("author_url") or row.get("author") or ""),
         lambda row: row["preview_images"][0], "opengameart", "digital-art"),
        ("3d_art", args.art_3d, eligible_oga, lambda row: str(row.get("author_url") or row.get("author") or ""),
         lambda row: row["preview_images"][0], "opengameart", "cgi"),
        ("train", args.photos, eligible_photo, lambda row: str(row.get("uid") or ""),
         lambda row: row["jpg"]["src"], "commoncatalog", "photo"),
    )
    rows: list[dict] = []
    provenance: list[dict] = []
    for split, wanted, predicate, group_key, source_url, source, domain in specs:
        dataset = OGA_ID if source == "opengameart" else COMMON_ID
        pool = candidates(dataset, split, wanted, predicate, group_key, source_url)
        kept = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for start in range(0, len(pool), args.workers):
                for result in executor.map(lambda item: download(item, root, source, domain), pool[start:start + args.workers]):
                    if result is not None and kept < wanted:
                        rows.append(result)
                        kept += 1
                        if kept % 100 == 0:
                            print(f"{domain}: {kept}/{wanted}", flush=True)
                if kept >= wanted:
                    break
        provenance.append({"dataset": dataset, "revision": OGA_REVISION if source == "opengameart" else COMMON_REVISION,
                           "domain": domain, "requested": wanted, "kept": kept})

    manifest = root / "manifest.csv"
    root.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({key: row[key] for key in FIELDS} for row in rows)
    payload = {
        "rows": len(rows), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "sources": provenance,
        "items": [{"path": str(row["path"]), "source_url": row["source_url"], "author": row["author"], "license": row["license"]} for row in rows],
    }
    (root / "provenance.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "items"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
