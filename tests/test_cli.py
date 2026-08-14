from pathlib import Path

import numpy as np

from cyclopes.cli import _fit_calibration, _metrics_report, build_parser
from cyclopes.data import Sample


def test_parser_exposes_all_pipeline_commands() -> None:
    parser = build_parser()
    train = parser.parse_args(["train", "--manifest", "m", "--output", "c", "--initial-checkpoint", "old"])
    assert train.command == "train"
    assert train.initial_checkpoint == "old"
    assert parser.parse_args(["calibrate", "--manifest", "m", "--checkpoint", "c", "--output", "j"]).command == "calibrate"
    assert parser.parse_args(["evaluate", "--manifest", "m", "--checkpoint", "c"]).command == "evaluate"
    assert parser.parse_args(["export", "--checkpoint", "c", "--output", "o"]).command == "export"
    assert parser.parse_args(["onnx-parity", "--manifest", "m", "--checkpoint", "c", "--onnx", "o"]).command == "onnx-parity"


def test_metrics_report_is_frozen_at_point_65() -> None:
    samples = [
        Sample(Path("real.png"), 0, "photos", "", "r", "test"),
        Sample(Path("fake.png"), 1, "generated", "diffusion", "f", "test"),
    ]
    report = _metrics_report(samples, np.array([0.64, 0.65]))
    assert report["threshold"] == 0.65
    assert report["metrics"]["true_positive"] == 1
    assert report["metrics"]["true_negative"] == 1


def test_calibration_aligns_selected_raw_threshold() -> None:
    temperature, bias, raw_threshold = _fit_calibration(
        np.array([-2.0, -1.0, 1.0, 2.0]), np.array([0, 0, 1, 1])
    )
    probability = 1 / (1 + np.exp(-(np.log(raw_threshold / (1 - raw_threshold)) / temperature + bias)))
    assert temperature > 0
    np.testing.assert_allclose(probability, 0.65, atol=1e-7)
