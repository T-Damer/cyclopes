import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from cyclopes.data import aspect_pad, browser_crops, composite_variant, load_manifest, tensor_for_browser, web_variant
from cyclopes.metrics import best_balanced_threshold, binary_metrics
from cyclopes.modeling import ScalePairMobileNet, threshold_alignment


def test_balanced_metrics_use_fixed_threshold() -> None:
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.7, 0.65, 0.9])
    assert metrics.ai_recall == 1.0
    assert metrics.real_specificity == 0.5
    assert metrics.balanced_accuracy == 0.75
    assert metrics.roc_auc == 0.75


def test_best_threshold() -> None:
    threshold, metrics = best_balanced_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert threshold == 0.8
    assert metrics.balanced_accuracy == 1.0


def test_alignment_maps_raw_threshold_to_point_65() -> None:
    raw_threshold = 0.4
    temperature = 1.7
    bias = threshold_alignment(raw_threshold, temperature)
    probability = 1 / (1 + np.exp(-(raw_threshold / temperature + bias)))
    assert probability == pytest.approx(0.65)


def test_manifest_rejects_group_leakage(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "label", "source", "generator", "group", "split"))
        writer.writeheader()
        writer.writerow({"path": "a.png", "label": 0, "source": "s", "generator": "", "group": "pair", "split": "train"})
        writer.writerow({"path": "b.png", "label": 1, "source": "s", "generator": "g", "group": "pair", "split": "test"})
    with pytest.raises(ValueError, match="groups cross splits"):
        load_manifest(manifest)


def test_extended_manifest_fields_drive_family_auxiliary_label(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(image)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "label", "source_dataset", "generator_model", "content_group", "split", "family", "domain"))
        writer.writeheader()
        writer.writerow({"path": image.name, "label": 1, "source_dataset": "generated", "generator_model": "biggan",
                         "content_group": "seed-1", "split": "train", "family": "legacy-gan", "domain": "mixed"})
    sample = load_manifest(manifest)[0]
    assert sample.family_index == 0
    assert sample.content_domain == "mixed"
    assert sample.content_index == 0


def test_aspect_padding_preserves_the_complete_image() -> None:
    source = Image.new("RGB", (400, 200), (255, 0, 0))
    padded = aspect_pad(source)
    assert padded.size == (224, 224)
    assert padded.getpixel((112, 0)) == (128, 128, 128)
    assert padded.getpixel((112, 112)) == (255, 0, 0)
    assert tensor_for_browser(source).shape == (3, 224, 224)


def test_browser_view_crops_only_strong_aspect_ratios() -> None:
    portrait = Image.new("RGB", (100, 200), (255, 0, 0))
    landscape = Image.new("RGB", (120, 100), (0, 255, 0))
    assert torch.equal(browser_crops(portrait)[0], tensor_for_browser(portrait.crop((0, 50, 100, 150))))
    assert torch.equal(browser_crops(landscape)[0], tensor_for_browser(landscape))


def test_web_variant_is_deterministic_and_keeps_a_valid_image() -> None:
    source = Image.new("RGB", (900, 600), (20, 40, 60))
    first = web_variant(source, __import__("random").Random(323), moderate=True)
    second = web_variant(source, __import__("random").Random(323), moderate=True)
    assert first.size == second.size
    assert first.tobytes() == second.tobytes()
    assert min(first.size) <= 600


def test_composite_variant_adds_a_meme_layout() -> None:
    source = Image.new("RGB", (400, 300), (20, 40, 60))
    first = composite_variant(source, __import__("random").Random(323))
    second = composite_variant(source, __import__("random").Random(323))
    assert first.size == second.size
    assert first.tobytes() == second.tobytes()
    assert first.size != source.size


def test_scalepair_uses_one_shared_backbone_for_both_views() -> None:
    model = ScalePairMobileNet(pretrained=False).eval()
    image = torch.zeros((1, 3, 224, 224))
    with torch.inference_mode():
        outputs = model.components(image)
    assert outputs.fused_logit.shape == (1,)
    assert outputs.current_logit.shape == (1,)
    assert outputs.probe_logit.shape == (1,)
    assert outputs.family_logits.shape == (1, 6)
    assert outputs.embedding.shape == (1, 128)
