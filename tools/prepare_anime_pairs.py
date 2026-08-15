#!/usr/bin/env python3
"""Build independent human/AI anime pairs for domain fine-tuning."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import zipfile
from contextlib import ExitStack
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.merge_manifests import FIELDS
from tools.prepare_training import save, split_for

CSIP = ("deepghs/csip_v1", "9b052ac0bcf1e3c08a659e99595d075778b442e9", "csip_v1_p8.zip")
DIFFUSIONDB = ("poloclub/diffusiondb", "fb620fbe49fa4420e0734bd9c0df11f51176b61f")
DIFFUSION_PARTS = tuple(f"diffusiondb-large-part-1/part-{index:06d}.zip" for index in range(1, 9))
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
ART_TERMS = (
    "anime", "manga", "illustration", "digital art", "concept art", "comic",
    "cartoon", "character design", "cel shading", "game art", "fantasy art",
)


def art_prompt(prompt: str) -> bool:
    prompt = prompt.casefold()
    return any(term in prompt for term in ART_TERMS)


def author_group(member: str, parent_counts: dict[str, int]) -> str:
    parent = Path(member).parent.name
    if parent and len(parent_counts) >= 20:
        return f"csip:{parent}"
    return f"csip:{hashlib.sha256(member.encode()).hexdigest()[:16]}"


def download(dataset: str, revision: str, filename: str, cache_dir: Path | None) -> Path:
    return Path(hf_hub_download(
        dataset, filename, repo_type="dataset", revision=revision,
        token=os.environ.get("HF_READ_ONLY_TOKEN"), cache_dir=cache_dir,
    ))


def write_image(encoded: bytes, target: Path) -> str:
    with Image.open(io.BytesIO(encoded)) as image:
        return save(image, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--real-limit", type=int, default=4_000)
    parser.add_argument("--ai-limit", type=int, default=4_000)
    args = parser.parse_args()
    output = args.output.resolve()
    rows: list[dict[str, object]] = []
    seen: set[str] = set()

    dataset, revision, filename = CSIP
    csip_zip = download(dataset, revision, filename, args.cache_dir)
    real_count = 0
    with zipfile.ZipFile(csip_zip) as archive:
        members = [name for name in archive.namelist() if Path(name).suffix.casefold() in IMAGE_SUFFIXES]
        parent_counts = {Path(name).parent.name: 1 for name in members}
        members.sort(key=lambda name: hashlib.sha256(name.encode()).digest())
        for member in members:
            if real_count >= args.real_limit:
                break
            target = output / "images" / "real" / f"{hashlib.sha256(member.encode()).hexdigest()[:20]}.jpg"
            try:
                digest = write_image(archive.read(member), target)
            except (KeyError, OSError, ValueError):
                continue
            if digest in seen:
                target.unlink(missing_ok=True)
                continue
            seen.add(digest)
            group = author_group(member, parent_counts)
            rows.append({"path": target, "label": 0, "source_dataset": dataset,
                         "generator_model": "human-anime-art", "content_group": group,
                         "split": split_for(group), "family": "real", "domain": "illustration",
                         "license": "CC-BY-4.0", "sha256": digest})
            real_count += 1

    ai_candidates: list[tuple[int, str, Path, str]] = []
    diffusion_files = []
    dataset, revision = DIFFUSIONDB
    for filename in DIFFUSION_PARTS:
        archive_path = download(dataset, revision, filename, args.cache_dir)
        diffusion_files.append(filename)
        with zipfile.ZipFile(archive_path) as archive:
            metadata_name = next(name for name in archive.namelist() if name.endswith(".json"))
            metadata = json.loads(archive.read(metadata_name))
            for member in archive.namelist():
                if Path(member).suffix.casefold() not in IMAGE_SUFFIXES:
                    continue
                item = metadata.get(Path(member).name, {})
                prompt = str(item.get("p", ""))
                rank = 0 if art_prompt(prompt) else 1
                ai_candidates.append((rank, hashlib.sha256((member + prompt).encode()).hexdigest(), archive_path, member))
    ai_candidates.sort(key=lambda item: (item[0], item[1]))
    ai_count = 0
    with ExitStack() as stack:
        archives = {path: stack.enter_context(zipfile.ZipFile(path)) for path in {item[2] for item in ai_candidates}}
        for _rank, key, archive_path, member in ai_candidates:
            if ai_count >= args.ai_limit:
                break
            target = output / "images" / "ai" / f"{key[:20]}.jpg"
            try:
                archive = archives[archive_path]
                digest = write_image(archive.read(member), target)
            except (KeyError, OSError, ValueError):
                continue
            if digest in seen:
                target.unlink(missing_ok=True)
                continue
            seen.add(digest)
            group = f"diffusiondb:{key[:16]}"
            rows.append({"path": target, "label": 1, "source_dataset": dataset,
                         "generator_model": "stable-diffusion", "content_group": group,
                         "split": split_for(group), "family": "diffusion", "domain": "illustration",
                         "license": "CC0-1.0", "sha256": digest})
            ai_count += 1

    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["path"] = Path(row["path"]).relative_to(output)
            writer.writerow(row)
    counts = {str(label): sum(int(row["label"]) == label for row in rows) for label in (0, 1)}
    if min(counts.values()) < min(args.real_limit, args.ai_limit) * 0.9:
        raise ValueError(f"insufficient anime pairs: {counts}")
    provenance = {
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "counts": counts,
        "sources": [
            {"dataset": CSIP[0], "revision": CSIP[1], "files": [CSIP[2]], "license": "CC-BY-4.0"},
            {"dataset": DIFFUSIONDB[0], "revision": DIFFUSIONDB[1], "files": diffusion_files, "license": "CC0-1.0"},
        ],
        "codec": "JPEG quality=92 subsampling=0",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps(provenance, sort_keys=True))


if __name__ == "__main__":
    main()
