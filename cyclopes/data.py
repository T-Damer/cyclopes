from __future__ import annotations

import csv
import hashlib
import io
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms import functional as TF


CORE_FIELDS = ("path", "label", "source", "generator", "group", "split")
IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
GENERATOR_FAMILIES = (
    "legacy-gan",
    "legacy-diffusion",
    "modern",
    "diffusion",
    "recent-commercial",
    "other",
)
FAMILY_TO_INDEX = {name: index for index, name in enumerate(GENERATOR_FAMILIES)}
CONTENT_CLASSES = ("general", "photo", "art-game", "composite-ui")


def content_index(domain: str) -> int:
    value = domain.lower()
    if value in {"mixed", "unknown"}:
        return 0
    if "photo" in value:
        return 1
    if any(token in value for token in ("art", "illustration", "cgi", "game", "anime", "cartoon")):
        return 2
    if any(token in value for token in ("meme", "composite", "ui", "screenshot", "logo", "poster", "text", "graphic")):
        return 3
    return 0


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    source: str
    generator: str
    group: str
    split: str
    generator_family: str = "unknown"
    content_domain: str = "unknown"
    license: str = "unknown"
    sha256: str = ""

    @property
    def family_index(self) -> int:
        return FAMILY_TO_INDEX.get(self.generator_family, -1) if self.label else -1

    @property
    def content_index(self) -> int:
        return content_index(self.content_domain)


def _field(row: dict[str, str], primary: str, fallback: str = "") -> str:
    return (row.get(primary) or row.get(fallback) or "").strip()


def load_manifest(path: str | Path) -> list[Sample]:
    manifest_path = Path(path).resolve()
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        required = {"path", "label", "split"}
        if (not required.issubset(fields) or not ({"source", "source_dataset"} & fields)
                or not ({"group", "content_group"} & fields)):
            raise ValueError("manifest must contain path, label, source, group, and split fields")
        samples: list[Sample] = []
        for line, row in enumerate(reader, 2):
            try:
                label = int(row["label"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"line {line}: label must be 0 or 1") from error
            if label not in (0, 1):
                raise ValueError(f"line {line}: label must be 0 or 1")
            image_path = Path(row["path"])
            if not image_path.is_absolute():
                image_path = manifest_path.parent / image_path
            samples.append(
                Sample(
                    path=image_path.resolve(),
                    label=label,
                    source=_field(row, "source_dataset", "source"),
                    generator=_field(row, "generator_model", "generator"),
                    group=_field(row, "content_group", "group"),
                    split=_field(row, "split"),
                    generator_family=_field(row, "generator_family", "family") or "unknown",
                    content_domain=_field(row, "content_domain", "domain") or "unknown",
                    license=_field(row, "license") or "unknown",
                    sha256=_field(row, "sha256"),
                )
            )

    if not samples:
        raise ValueError("manifest is empty")
    _validate_groups(samples)
    return samples


def _validate_groups(samples: list[Sample]) -> None:
    splits_by_group: dict[str, set[str]] = {}
    paths_seen: set[Path] = set()
    contents_seen: dict[str, Path] = {}
    for sample in samples:
        if not sample.source or not sample.group or not sample.split:
            raise ValueError("source, group, and split must not be empty")
        if sample.path in paths_seen:
            raise ValueError(f"duplicate image path: {sample.path}")
        paths_seen.add(sample.path)
        if sample.path.is_file():
            digest = hashlib.sha256(sample.path.read_bytes()).hexdigest()
            if sample.sha256 and digest != sample.sha256:
                raise ValueError(f"sha256 mismatch: {sample.path}")
            previous = contents_seen.get(digest)
            if previous is not None:
                raise ValueError(f"duplicate image content: {previous} and {sample.path}")
            contents_seen[digest] = sample.path
        splits_by_group.setdefault(sample.group, set()).add(sample.split)
    leaked = sorted(group for group, splits in splits_by_group.items() if len(splits) > 1)
    if leaked:
        raise ValueError(f"groups cross splits: {', '.join(leaked[:5])}")


def decode_rgb(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    background.alpha_composite(rgba)
    return background.convert("RGB")


def aspect_pad(image: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    if image.mode != "RGB":
        image = decode_rgb(image)
    scale = min(size / image.width, size / image.height)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    resized = image.resize((width, height), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    canvas.paste(resized, ((size - width) // 2, (size - height) // 2))
    return canvas


def tensor_for_browser(
    image: Image.Image,
    *,
    size: int = IMAGE_SIZE,
    mean: tuple[float, float, float] = MEAN,
    std: tuple[float, float, float] = STD,
) -> torch.Tensor:
    return TF.normalize(TF.to_tensor(aspect_pad(image, size)), mean, std)


def _reencode(image: Image.Image, rng: random.Random, quality: int, codec: str) -> Image.Image:
    buffer = io.BytesIO()
    if codec == "JPEG":
        image.save(buffer, "JPEG", quality=quality, subsampling=rng.choice((0, 2)))
    else:
        image.save(buffer, "WEBP", quality=quality, method=4)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def web_variant(image: Image.Image, rng: random.Random, *, moderate: bool | None = None) -> Image.Image:
    if image.mode != "RGB":
        image = decode_rgb(image)
    moderate = rng.random() < 0.60 if moderate is None else moderate
    if moderate:
        short_edge = rng.choice((384, 448, 512, 640, 768))
        quality = rng.randint(70, 95)
        codec = rng.choice(("JPEG", "JPEG", "WEBP"))
    else:
        short_edge = rng.choice((96, 128, 160, 192, 224, 256, 320))
        quality = rng.randint(30, 70)
        codec = rng.choice(("JPEG", "WEBP"))
        if rng.random() < 0.25:
            crop = rng.uniform(0.0, 0.08)
            left = round(image.width * crop * rng.random())
            top = round(image.height * crop * rng.random())
            right = image.width - round(image.width * crop * rng.random())
            bottom = image.height - round(image.height * crop * rng.random())
            if right > left and bottom > top:
                image = image.crop((left, top, right, bottom))

    current_short = min(image.size)
    scale = min(1.0, short_edge / current_short)
    if scale < 1:
        target = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(target, rng.choice((Image.Resampling.BILINEAR, Image.Resampling.BICUBIC, Image.Resampling.LANCZOS)))
    image = _reencode(image, rng, quality, codec)
    if rng.random() < (0.20 if moderate else 0.35):
        image = _reencode(image, rng, max(30, quality - rng.randint(5, 20)), rng.choice(("JPEG", "WEBP")))
    if rng.random() < 0.12:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, 0.8 if moderate else 1.2)))
    elif rng.random() < 0.08:
        image = ImageEnhance.Sharpness(image).enhance(rng.uniform(1.05, 1.35))
    return image


def composite_variant(image: Image.Image, rng: random.Random) -> Image.Image:
    """Add common meme/collage structure without changing the image label."""
    image = image.convert("RGB")
    if rng.random() < 0.5:
        band = max(24, image.height // 5)
        output = ImageOps.expand(image, border=(0, band, 0, 0), fill="white")
        draw = ImageDraw.Draw(output)
        draw.text((band // 3, band // 3), "WHEN THE CAPTION CHANGES THE CONTEXT", fill="black")
        return output
    panel = image.resize((image.width, max(1, image.height // 2)), Image.Resampling.BICUBIC)
    output = Image.new("RGB", (image.width, panel.height * 2 + 8), "white")
    output.paste(panel, (0, 0))
    output.paste(panel.transpose(Image.Transpose.FLIP_LEFT_RIGHT), (0, panel.height + 8))
    return output


class ManifestDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        split: str,
        training: bool = False,
        seed: int = 323,
        browser_view: bool = False,
        image_size: int = IMAGE_SIZE,
        mean: tuple[float, float, float] = MEAN,
        std: tuple[float, float, float] = STD,
        composite_probability: float = 0.0,
    ) -> None:
        self.samples = [sample for sample in samples if sample.split == split]
        if not self.samples:
            raise ValueError(f"manifest has no samples in split {split!r}")
        self.training = training
        self.seed = seed
        self.browser_view = browser_view
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.composite_probability = composite_probability

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as opened:
            image = opened.copy() if opened.mode == "RGB" else decode_rgb(opened)
        if not self.training:
            tensor = (
                browser_crops(image, size=self.image_size, mean=self.mean, std=self.std)[0]
                if self.browser_view
                else tensor_for_browser(image, size=self.image_size, mean=self.mean, std=self.std)
            )
            return tensor, sample.label, index

        rng = random.Random(random.getrandbits(64) ^ self.seed ^ index)
        content = sample.content_index
        if rng.random() < self.composite_probability:
            image = composite_variant(image, rng)
            content = 3
        if rng.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        clean = tensor_for_browser(image, size=self.image_size, mean=self.mean, std=self.std)
        moderate = rng.random() < 0.60
        web = tensor_for_browser(
            web_variant(image, rng, moderate=moderate),
            size=self.image_size,
            mean=self.mean,
            std=self.std,
        )
        return clean, web, sample.label, sample.family_index, content, moderate, index

    def balanced_sampler(self, seed: int) -> WeightedRandomSampler:
        counts = Counter((sample.label, sample.source) for sample in self.samples)
        sources_per_label = Counter()
        for label in (0, 1):
            sources_per_label[label] = len({sample.source for sample in self.samples if sample.label == label})
        weights = [
            1.0 / (2 * sources_per_label[sample.label] * counts[(sample.label, sample.source)])
            for sample in self.samples
        ]
        generator = torch.Generator().manual_seed(seed)
        return WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)


def browser_crops(
    image: Image.Image,
    *,
    size: int = IMAGE_SIZE,
    mean: tuple[float, float, float] = MEAN,
    std: tuple[float, float, float] = STD,
) -> list[torch.Tensor]:
    width, height = image.size
    if max(width, height) / min(width, height) >= 1.25:
        crop_size = min(width, height)
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        image = image.crop((left, top, left + crop_size, top + crop_size))
    return [tensor_for_browser(image, size=size, mean=mean, std=std)]
