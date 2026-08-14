#!/usr/bin/env python3
"""Export the pinned Sentry ConvNeXt teacher as a browser-sized FP16 ONNX graph."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch
from aidetector.sentry_adapter import SentryConvNeXtDetector
from onnxconverter_common import float16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    temporary = args.output.with_suffix(".fp32.onnx")
    detector = SentryConvNeXtDetector(device="cpu")
    torch.onnx.export(
        detector.model.eval(),
        torch.zeros(1, 3, 224, 224),
        temporary,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    model = float16.convert_float_to_float16(onnx.load(temporary), keep_io_types=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    temporary.unlink()


if __name__ == "__main__":
    main()
