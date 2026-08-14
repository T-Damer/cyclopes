#!/usr/bin/env python3
"""Download licensed Blender Open Movies and extract group-disjoint CGI frames."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path

from PIL import Image

API = "https://video.blender.org/api/v1/videos/"
FIELDS = (
    "path", "label", "source_dataset", "generator_model", "content_group",
    "split", "family", "domain", "license", "sha256",
)
FILMS = {
    "spring": ("3d95fb3d-c866-42c8-9db1-fe82f48ccb95", "train", "CC-BY-SA"),
    "coffee-run": ("ff8fe61b-026f-4f07-b66b-2a790d6f6ab1", "train", "CC-BY"),
    "charge": ("04da454b-9893-4184-98f3-248d00625efe", "train", "CC-BY"),
    "sintel": ("0eb052d0-fd51-43e6-aa33-ecdbf77a5d40", "train", "CC-BY-3.0"),
    "cosmos-laundromat": ("f507dfdc-e73e-45a4-9778-d758cbe1ce96", "train", "CC-BY"),
    "big-buck-bunny": ("6402b77c-b61f-4a06-96ca-c8420a2becf4", "calibration", "CC-BY"),
    "sprite-fright": ("a69d68a5-a0e0-4a80-9d66-49f093c97aaf", "validation", "CC-BY"),
    "wing-it": ("bd0084a5-1d26-4816-ab5e-1bad9e2fb990", "test", "CC-BY"),
}


def video_info(uuid: str) -> dict:
    request = urllib.request.Request(API + uuid, headers={"User-Agent": "Cyclopes dataset builder/0.2"})
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def download_url(info: dict) -> str:
    files = info.get("files", [])
    candidates = [item for item in files if item.get("fileDownloadUrl")]
    if not candidates:
        raise RuntimeError(f'no downloadable file for {info.get("name", "video")}')
    under_1080 = [item for item in candidates if (item.get("resolution", {}).get("id") or 10_000) <= 1080]
    chosen = max(under_1080 or candidates, key=lambda item: item.get("resolution", {}).get("id") or 0)
    return chosen["fileDownloadUrl"]


def fetch(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Cyclopes dataset builder/0.2"})
            with urllib.request.urlopen(request, timeout=60) as response:
                expected = int(response.headers.get("Content-Length", "0"))
                if expected and target.is_file() and target.stat().st_size == expected:
                    return
                with part.open("wb") as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
            if expected and part.stat().st_size != expected:
                raise OSError(f"incomplete video: {part.stat().st_size} != {expected}")
            part.replace(target)
            return
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=float, default=1.25)
    args = parser.parse_args()
    root = args.output.resolve()
    rows = []
    for slug, (uuid, split, license_name) in FILMS.items():
        info = video_info(uuid)
        movie = root / "movies" / f"{slug}.mp4"
        fetch(download_url(info), movie)
        frames = root / "images" / slug
        frames.mkdir(parents=True, exist_ok=True)
        duration = max(1, int(info["duration"]) - 25)
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", "8", "-to", str(duration),
             "-i", str(movie), "-vf", f"fps={args.fps},scale='min(1280,iw)':-2", "-q:v", "3", str(frames / "%06d.jpg")],
            check=True,
        )
        for path in sorted(frames.glob("*.jpg")):
            with Image.open(path) as opened:
                normalized = opened.convert("RGB")
            normalized.save(path, "JPEG", quality=92, subsampling=0)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append({
                "path": path.relative_to(root), "label": 0, "source_dataset": "blender-open-movies",
                "generator_model": "human-cgi", "content_group": f"blender:{slug}", "split": split,
                "family": "real", "domain": "cgi", "license": license_name, "sha256": digest,
            })
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
