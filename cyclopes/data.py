from __future__ import annotations

import csv
import hashlib
import io
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFilter
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import functional as F


FIELDS = ("path", "label", "source", "generator", "group", "split")
IMAGE_SIZE = 256
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    source: str
    generator: str
    group: str
    split: str


def load_manifest(path: str | Path) -> list[Sample]:
    manifest_path = Path(path).resolve()
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(field not in reader.fieldnames for field in FIELDS):
            raise ValueError(f"manifest must contain columns: {', '.join(FIELDS)}")
        samples = []
        for line, row in enumerate(reader, 2):
            try:
                label = int(row["label"])
            except ValueError as error:
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
                    source=row["source"].strip(),
                    generator=row["generator"].strip(),
                    group=row["group"].strip(),
                    split=row["split"].strip(),
                )
            )

    if not samples:
        raise ValueError("manifest is empty")
    _validate_groups(samples)
    return samples


def _validate_groups(samples: list[Sample]) -> None:
    splits_by_group: dict[str, set[str]] = {}
    paths_seen: dict[Path, int] = {}
    contents_seen: dict[str, Path] = {}
    for sample in samples:
        if not sample.source or not sample.group or not sample.split:
            raise ValueError("source, group, and split must not be empty")
        if sample.path in paths_seen:
            raise ValueError(f"duplicate image path: {sample.path}")
        paths_seen[sample.path] = 1
        if sample.path.is_file():
            digest = hashlib.sha256(sample.path.read_bytes()).hexdigest()
            previous = contents_seen.get(digest)
            if previous is not None:
                raise ValueError(f"duplicate image content: {previous} and {sample.path}")
            contents_seen[digest] = sample.path
        splits_by_group.setdefault(sample.group, set()).add(sample.split)
    leaked = sorted(group for group, splits in splits_by_group.items() if len(splits) > 1)
    if leaked:
        preview = ", ".join(leaked[:5])
        raise ValueError(f"groups cross splits: {preview}")


class SymmetricWebAugment:
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < 0.45:
            quality = random.randint(45, 95)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality, subsampling=random.choice((0, 2)))
            buffer.seek(0)
            image = Image.open(buffer).convert("RGB")
        if random.random() < 0.25:
            scale = random.uniform(0.45, 0.9)
            smaller = (max(32, round(image.width * scale)), max(32, round(image.height * scale)))
            image = image.resize(smaller, Image.Resampling.BILINEAR).resize(image.size, Image.Resampling.BICUBIC)
        if random.random() < 0.15:
            image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 1.2)))
        return image


def build_transform(training: bool) -> transforms.Compose:
    if training:
        return transforms.Compose(
            [
                transforms.Resize(320, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
                SymmetricWebAugment(),
                transforms.RandomResizedCrop(
                    IMAGE_SIZE,
                    scale=(0.30, 1.0),
                    ratio=(0.75, 1.3333333333),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.01),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


class ManifestDataset(Dataset):
    def __init__(self, samples: list[Sample], split: str, training: bool = False) -> None:
        self.samples = [sample for sample in samples if sample.split == split]
        if not self.samples:
            raise ValueError(f"manifest has no samples in split {split!r}")
        self.transform = build_transform(training)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, sample.label, index

    def balanced_sampler(self, seed: int) -> WeightedRandomSampler:
        counts = Counter((sample.label, sample.source) for sample in self.samples)
        weights = [1.0 / counts[(sample.label, sample.source)] for sample in self.samples]
        import torch

        generator = torch.Generator().manual_seed(seed)
        return WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)


def browser_crops(image: Image.Image) -> list:
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    return [F.normalize(F.to_tensor(image), MEAN, STD)]
