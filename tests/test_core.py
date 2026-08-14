import csv
from pathlib import Path

import numpy as np
import pytest

from cyclopes.data import load_manifest
from cyclopes.metrics import best_balanced_threshold, binary_metrics
from cyclopes.modeling import threshold_alignment


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
