#!/usr/bin/env python3
"""Add a balanced, codec-normalized FakeClue/GenImage slice to a manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
FIELDS = ("path", "label", "source", "generator", "group", "split")


def split_for(group: str) -> str:
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "calibration" if bucket < 88 else "validation" if bucket < 94 else "test"


def generator(filename: str, label: int) -> str:
    if not label:
        return "camera"
    lowered = filename.lower()
    for name in ("midjourney", "biggan", "glide", "vqdm", "sdv4", "adm"):
        if name in lowered:
            return name
    return "genimage"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("base_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-archive", type=int, default=2000)
    args = parser.parse_args()
    output = args.output.resolve()
    image_root = output.parent / "images" / "fakeclue"
    with args.base_manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["path"] = str((args.base_manifest.parent / row["path"]).resolve())

    archives = sorted(args.root.glob("data/*/genimage/*.zip"))

    def convert(archive: Path) -> list[dict[str, object]]:
        label = int("/fake/" in archive.as_posix())
        converted = []
        with zipfile.ZipFile(archive) as zipped:
            members = [item for item in zipped.infolist() if not item.is_dir()][: args.per_archive]
            for index, member in enumerate(members):
                try:
                    with Image.open(io.BytesIO(zipped.read(member))) as image:
                        target = image_root / ("ai" if label else "real") / f"{archive.stem}-{index:05d}.jpg"
                        target.parent.mkdir(parents=True, exist_ok=True)
                        image.convert("RGB").save(target, "JPEG", quality=92, subsampling=0)
                except Exception as error:
                    print(f"skip {archive.name}:{member.filename}: {error}")
                    continue
                group = f"fakeclue:{label}:{archive.stem}:{Path(member.filename).stem}"
                converted.append({
                    "path": target,
                    "label": label,
                    "source": "bitmind/FakeClue-genimage",
                    "generator": generator(member.filename, label),
                    "group": group,
                    "split": split_for(group),
                })
        return converted

    with ThreadPoolExecutor(max_workers=len(archives)) as pool:
        for converted in pool.map(convert, archives):
            rows.extend(converted)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["path"] = Path(row["path"]).relative_to(output.parent)
            writer.writerow(row)
    print(f"rows={len(rows)} sha256={hashlib.sha256(output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
