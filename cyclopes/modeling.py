from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ScalePairOutputs:
    fused_logit: torch.Tensor
    current_logit: torch.Tensor
    probe_logit: torch.Tensor
    family_logits: torch.Tensor
    embedding: torch.Tensor
    content_logits: torch.Tensor | None = None


class ScalePairMobileNet(nn.Module):
    """One shared CNN comparing a current image with its synthetic thumbnail."""

    def __init__(self, pretrained: bool = True, family_classes: int = 6) -> None:
        super().__init__()
        from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = mobilenet_v3_large(weights=weights)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.early_index = 6
        self.view_projection = nn.Sequential(nn.Linear(960, 256), nn.LayerNorm(256), nn.SiLU())
        self.texture_projection = nn.Sequential(nn.Linear(40 * 3, 64), nn.LayerNorm(64), nn.SiLU())
        self.fusion = nn.Sequential(
            nn.Linear(256 * 4 + 64 * 3, 384),
            nn.LayerNorm(384),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(384, 128),
            nn.SiLU(),
        )
        self.fused_head = nn.Linear(128, 1)
        self.current_head = nn.Linear(320, 1)
        self.probe_head = nn.Linear(320, 1)
        self.family_head = nn.Linear(128, family_classes)

    @staticmethod
    def thumbnail_probe(image: torch.Tensor) -> torch.Tensor:
        probe = F.interpolate(image, size=(112, 112), mode="bilinear", align_corners=False)
        return F.interpolate(probe, size=(224, 224), mode="bilinear", align_corners=False)

    def _encode_views(self, views: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        early = None
        value = views
        for index, block in enumerate(self.features):
            value = block(value)
            if index == self.early_index:
                early = value
        if early is None:
            raise RuntimeError("MobileNet early feature was not captured")

        pooled = self.avgpool(value).flatten(1)
        view_embedding = self.view_projection(pooled)
        mean = early.mean(dim=(2, 3))
        variance = (early - mean[:, :, None, None]).square().mean(dim=(2, 3))
        deviation = variance.clamp_min(1e-8).sqrt()
        local = F.avg_pool2d(early, kernel_size=3, stride=1, padding=1)
        residual = (early - local).abs().mean(dim=(2, 3))
        texture = self.texture_projection(torch.cat((mean, deviation, residual), dim=1))
        return view_embedding, texture

    def components(self, image: torch.Tensor) -> ScalePairOutputs:
        batch = image.shape[0]
        probe = self.thumbnail_probe(image)
        embeddings, textures = self._encode_views(torch.cat((image, probe), dim=0))
        current, thumbnail = embeddings[:batch], embeddings[batch:]
        current_texture, thumbnail_texture = textures[:batch], textures[batch:]
        fused_input = torch.cat(
            (
                current,
                thumbnail,
                (current - thumbnail).abs(),
                current * thumbnail,
                current_texture,
                thumbnail_texture,
                (current_texture - thumbnail_texture).abs(),
            ),
            dim=1,
        )
        fused_embedding = self.fusion(fused_input)
        return ScalePairOutputs(
            fused_logit=self.fused_head(fused_embedding).flatten(),
            current_logit=self.current_head(torch.cat((current, current_texture), dim=1)).flatten(),
            probe_logit=self.probe_head(torch.cat((thumbnail, thumbnail_texture), dim=1)).flatten(),
            family_logits=self.family_head(fused_embedding),
            embedding=fused_embedding,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.components(image).fused_logit


class ExportedScalePair(nn.Module):
    """Frozen blend and calibration included in the browser ONNX graph."""

    def __init__(
        self,
        detector: ScalePairMobileNet,
        blend_weight: float,
        temperature: float,
        bias: float,
    ) -> None:
        super().__init__()
        if not 0 <= blend_weight <= 1:
            raise ValueError("blend_weight must be in [0, 1]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.detector = detector
        self.register_buffer("blend_weight", torch.tensor(float(blend_weight)))
        self.register_buffer("temperature", torch.tensor(float(temperature)))
        self.register_buffer("bias", torch.tensor(float(bias)))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        outputs = self.detector.components(image)
        raw = self.blend_weight * outputs.fused_logit + (1 - self.blend_weight) * outputs.current_logit
        return raw / self.temperature + self.bias


def threshold_alignment(raw_threshold: float, temperature: float, target: float = 0.65) -> float:
    if not 0 < target < 1:
        raise ValueError("target must be in (0, 1)")
    return math.log(target / (1 - target)) - raw_threshold / temperature
