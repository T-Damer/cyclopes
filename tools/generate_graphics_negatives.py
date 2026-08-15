#!/usr/bin/env python3
"""Create deterministic non-AI logos, posters, flags, and diagram hard negatives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from generate_legacy import FIELDS, save, split_for

WORDS = ("OPEN", "VOTE", "NEWS", "GREEN", "CITY", "CLUB", "STUDIO", "2026", "PEACE", "FUTURE", "NORTH", "ART")
PALETTES = (
    ("#ffffff", "#111827", "#ef4444", "#f59e0b"), ("#f8fafc", "#166534", "#84cc16", "#facc15"),
    ("#0f172a", "#f8fafc", "#38bdf8", "#a78bfa"), ("#fff7ed", "#7c2d12", "#fb923c", "#fde047"),
)


def font(size: int):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render(seed: int) -> Image.Image:
    rng = random.Random(seed)
    width, height = rng.choice(((512, 512), (768, 512), (512, 768), (640, 480), (480, 640)))
    background, foreground, accent, second = rng.choice(PALETTES)
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    style = seed % 4
    if style == 0:
        draw.rectangle((0, 0, width, height // 3), fill=accent)
        draw.ellipse((width // 4, height // 4, 3 * width // 4, 3 * height // 4), fill=second, outline=foreground, width=max(3, width // 100))
    elif style == 1:
        for index in range(rng.randint(4, 9)):
            x = rng.randint(-width // 4, width)
            draw.polygon(((x, 0), (x + width // 3, 0), (x - width // 3, height), (x - 2 * width // 3, height)), fill=(accent, second)[index % 2])
    elif style == 2:
        for index in range(rng.randint(5, 12)):
            radius = rng.randint(width // 20, width // 5)
            x, y = rng.randint(0, width), rng.randint(0, height)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(accent, second)[index % 2], width=max(4, width // 80))
    else:
        columns = rng.randint(3, 6)
        rows = rng.randint(3, 6)
        for row in range(rows):
            for column in range(columns):
                x0, y0 = column * width // columns, row * height // rows
                draw.rectangle((x0, y0, (column + 1) * width // columns, (row + 1) * height // rows),
                               fill=(background, accent, second)[(row + column) % 3])
    message = " ".join(rng.sample(WORDS, rng.randint(1, 3)))
    text_font = font(max(24, width // (len(message) + 3)))
    box = draw.textbbox((0, 0), message, font=text_font, stroke_width=1)
    x = (width - (box[2] - box[0])) // 2
    y = rng.randint(height // 8, 7 * height // 8 - (box[3] - box[1]))
    draw.text((x, y), message, font=text_font, fill=foreground, stroke_width=max(1, width // 300), stroke_fill=background)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=2_000)
    args = parser.parse_args()
    root = args.output.resolve()
    rows = []
    for index in range(args.count):
        path = root / "images" / f"{index:06d}.jpg"
        digest = save(render(323 + index), path)
        group = f"procedural-graphics:{index:06d}"
        rows.append({"path": path.relative_to(root), "label": 0, "source_dataset": "cyclopes-procedural-graphics",
                     "generator_model": "pillow", "content_group": group, "split": split_for(group), "family": "real",
                     "domain": "logo-poster-vector", "license": "CC0-1.0", "sha256": digest})
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows={len(rows)} sha256={hashlib.sha256(manifest.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
