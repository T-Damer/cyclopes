from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ForensicMobileNet(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        import timm

        kernels = torch.tensor(
            [
                [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1)
        self.register_buffer("forensic_kernels", kernels)
        self.backbone = timm.create_model(
            "fastvit_t8.apple_in1k",
            pretrained=pretrained,
            in_chans=6,
            num_classes=1,
            drop_rate=0.2,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        gray = image[:, 0:1] * 0.2989 + image[:, 1:2] * 0.5870 + image[:, 2:3] * 0.1140
        residuals = torch.tanh(F.conv2d(gray, self.forensic_kernels, padding=1) * 0.25)
        return self.backbone(torch.cat((image, residuals), dim=1)).flatten()


class CalibratedModel(nn.Module):
    def __init__(self, detector: ForensicMobileNet, temperature: float, bias: float) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.detector = detector
        self.register_buffer("temperature", torch.tensor(float(temperature)))
        self.register_buffer("bias", torch.tensor(float(bias)))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.detector(image) / self.temperature + self.bias)


def threshold_alignment(raw_threshold: float, temperature: float, target: float = 0.65) -> float:
    if not 0 < target < 1:
        raise ValueError("target must be in (0, 1)")
    return math.log(target / (1 - target)) - raw_threshold / temperature
