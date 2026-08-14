#!/usr/bin/env python3
"""Generate reproducible legacy BigGAN or VQ-Diffusion positives."""

from __future__ import annotations

import argparse
import csv
import hashlib
from itertools import product
from pathlib import Path

import torch
from PIL import Image

FIELDS = ("path", "label", "source_dataset", "generator_model", "content_group", "split", "family", "domain", "license", "sha256")
ADJECTIVES = ("red", "blue", "old", "new", "small", "large", "bright", "dark", "wooden", "metal")
SUBJECTS = ("dog", "cat", "bird", "car", "train", "house", "tree", "flower", "boat", "chair",
            "clock", "camera", "bicycle", "airplane", "bridge", "castle", "robot", "fruit", "shoe", "lamp")
SETTINGS = ("in a forest", "on a street", "in a studio", "near the ocean", "on a table",
            "under dramatic light", "at sunset", "in winter", "in a garden", "against a plain background")


def split_for(group: str) -> str:
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "calibration" if bucket < 88 else "validation" if bucket < 94 else "test"


def save(image: Image.Image, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(target, "JPEG", quality=92, subsampling=0)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def biggan(count: int, batch_size: int):
    from pytorch_pretrained_biggan import BigGAN, one_hot_from_int, truncated_noise_sample

    model = BigGAN.from_pretrained("biggan-deep-256").to("cuda").eval()
    for start in range(0, count, batch_size):
        size = min(batch_size, count - start)
        truncation = (0.4, 0.7, 1.0)[(start // batch_size) % 3]
        classes = [(start + index) % 1000 for index in range(size)]
        noise = torch.from_numpy(truncated_noise_sample(truncation=truncation, batch_size=size, seed=323 + start)).cuda()
        labels = torch.from_numpy(one_hot_from_int(classes, batch_size=size)).cuda()
        with torch.inference_mode():
            images = model(noise, labels, truncation).add(1).div(2).clamp(0, 1).cpu()
        for offset, tensor in enumerate(images):
            yield start + offset, Image.fromarray(tensor.mul(255).byte().permute(1, 2, 0).numpy())


def diffusion(count: int, batch_size: int, model_id: str):
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    prompts = [f"a {adjective} {subject} {setting}" for adjective, subject, setting in product(ADJECTIVES, SUBJECTS, SETTINGS)]
    for start in range(0, count, batch_size):
        batch = prompts[start:start + min(batch_size, count - start)]
        generator = torch.Generator(device="cuda").manual_seed(323 + start)
        options = {"truncation_rate": 1.0} if "vq-diffusion" in model_id else {"num_inference_steps": 25}
        images = pipe(batch, generator=generator, **options).images
        for offset, image in enumerate(images):
            yield start + offset, image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("generator", choices=("biggan", "vqdiffusion", "ldm"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.generator == "biggan":
        source = biggan(args.count, args.batch_size)
    else:
        model_id = "microsoft/vq-diffusion-ithq" if args.generator == "vqdiffusion" else "CompVis/ldm-text2im-large-256"
        source = diffusion(args.count, args.batch_size, model_id)
    family = "legacy-gan" if args.generator == "biggan" else "legacy-diffusion"
    license_name = "MIT" if args.generator in {"biggan", "vqdiffusion"} else "CreativeML-OpenRAIL-M"
    rows = []
    for index, image in source:
        path = root / "images" / f"{index:06d}.jpg"
        digest = save(image, path)
        group = f"{args.generator}:{index:06d}"
        rows.append({"path": path.relative_to(root), "label": 1, "source_dataset": f"self-generated/{args.generator}",
                     "generator_model": args.generator, "content_group": group, "split": split_for(group),
                     "family": family, "domain": "mixed", "license": license_name, "sha256": digest})
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows={len(rows)} sha256={hashlib.sha256(manifest.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
