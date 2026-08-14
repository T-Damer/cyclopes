#!/usr/bin/env python3
"""Generate licensed modern anime positives matched to human anime negatives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.merge_manifests import FIELDS
from tools.prepare_training import split_for

MODEL_ID = "cagliostrolab/animagine-xl-4.0"
REVISION = "2b7c1b397761bf5bd3cc42e5b39ec99314a75a96"
SUBJECTS = (
    "1girl", "1boy", "2girls", "1girl and 1boy", "anthropomorphic hedgehog",
    "anthropomorphic fox", "kemonomimi girl", "robot girl", "fantasy warrior",
    "magical girl", "school student", "vampire", "elf", "cyborg", "superhero", "musician",
)
SCENES = (
    "portrait, looking at viewer", "dynamic action pose", "sitting indoors", "running outdoors",
    "city at night", "forest background", "bedroom interior", "beach sunset",
)
STYLES = ("anime screencap", "digital illustration", "cel shading", "detailed game art")


def prompts(limit: int) -> list[str]:
    values = [
        f"{subject}, {scene}, {style}, masterpiece, high score, great score, absurdres"
        for style in STYLES for scene in SCENES for subject in SUBJECTS
    ]
    return values[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()

    import torch
    from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

    output = args.output.resolve()
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with args.real_manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["label"]) != 0:
                continue
            row = dict(row)
            row["path"] = str((args.real_manifest.parent / row["path"]).resolve())
            rows.append(row)

    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, revision=REVISION, torch_dtype=torch.bfloat16)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    values = prompts(args.limit)
    negative = "lowres, bad anatomy, bad hands, text, error, worst quality, jpeg artifacts, watermark"
    for start in range(0, len(values), args.batch_size):
        batch = values[start:start + args.batch_size]
        portrait = (start // args.batch_size) % 2 == 1
        generated = pipe(
            batch, negative_prompt=[negative] * len(batch),
            width=640 if portrait else 768, height=896 if portrait else 768,
            num_inference_steps=args.steps, guidance_scale=5.0,
            generator=[torch.Generator("cuda").manual_seed(323_000 + index)
                       for index in range(start, start + len(batch))],
        ).images
        for index, (prompt, image) in enumerate(zip(batch, generated, strict=True), start=start):
            target = images_dir / f"{index:05d}.jpg"
            image.convert("RGB").save(target, "JPEG", quality=92, subsampling=0)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            group = f"animagine:{hashlib.sha256(prompt.encode()).hexdigest()[:16]}"
            rows.append({"path": target, "label": 1, "source_dataset": MODEL_ID,
                         "generator_model": "animagine-xl-4.0", "content_group": group,
                         "split": split_for(group), "family": "modern", "domain": "illustration",
                         "license": "CreativeML-OpenRAIL++-M", "sha256": digest})

    manifest = output / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "model": MODEL_ID, "revision": REVISION, "license": "CreativeML Open RAIL++-M",
        "generated": len(values), "human_real": len(rows) - len(values),
        "steps": args.steps, "seed_start": 323_000,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps(provenance, sort_keys=True))


if __name__ == "__main__":
    main()
