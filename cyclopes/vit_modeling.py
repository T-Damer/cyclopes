from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .modeling import ScalePairOutputs


MODEL_REPO = "buildborderless/CommunityForensics-DeepfakeDet-ViT"
MODEL_REVISION = "ac6ee457bea904a373065754107451793b56db00"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class MultiLayerScalePairViT(nn.Module):
    """A frozen detector prior plus trainable layer/scale residual experts."""

    image_size = 384
    mean = CLIP_MEAN
    std = CLIP_STD

    def __init__(self, source: nn.Module, layers: Sequence[int] = (4, 8, 12), family_classes: int = 6) -> None:
        super().__init__()
        self.backbone = source.vit
        self.base_head = source.classifier
        self.image_size = int(source.config.image_size)
        self.layers = tuple(int(layer) for layer in layers)
        hidden_size = int(source.config.hidden_size)
        self.projections = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 128), nn.GELU())
            for _ in self.layers
        )
        embedding_size = 128 * len(self.layers)
        self.residual_head = nn.Sequential(
            nn.LayerNorm(embedding_size * 2),
            nn.Linear(embedding_size * 2, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
        self.family_head = nn.Linear(embedding_size, family_classes)
        self.content_router = nn.Sequential(nn.LayerNorm(embedding_size), nn.Linear(embedding_size, 5))
        self.expert_heads = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(embedding_size * 2), nn.Linear(embedding_size * 2, 64), nn.GELU(), nn.Linear(64, 1))
            for _ in range(5)
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        for expert in self.expert_heads:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)
        self.freeze_backbone()

    @classmethod
    def from_pretrained(
        cls,
        repo: str = MODEL_REPO,
        revision: str = MODEL_REVISION,
        layers: Sequence[int] = (4, 8, 12),
    ) -> MultiLayerScalePairViT:
        from transformers import AutoModelForImageClassification

        source = AutoModelForImageClassification.from_pretrained(repo, revision=revision)
        return cls(source, layers=layers)

    @classmethod
    def from_config(
        cls,
        repo: str = MODEL_REPO,
        revision: str = MODEL_REVISION,
        layers: Sequence[int] = (4, 8, 12),
    ) -> MultiLayerScalePairViT:
        from transformers import AutoConfig, AutoModelForImageClassification

        config = AutoConfig.from_pretrained(repo, revision=revision)
        return cls(AutoModelForImageClassification.from_config(config), layers=layers)

    def freeze_backbone(self) -> None:
        for parameter in (*self.backbone.parameters(), *self.base_head.parameters()):
            parameter.requires_grad_(False)

    def unfreeze_last_blocks(self, count: int) -> None:
        if count < 0 or count > len(self.backbone.encoder.layer):
            raise ValueError("unfreeze_last_blocks is out of range")
        self.freeze_backbone()
        for block in self.backbone.encoder.layer[-count:] if count else ():
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        if count:
            for parameter in self.backbone.layernorm.parameters():
                parameter.requires_grad_(True)

    def freeze_prior(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in (*self.content_router.parameters(), *self.expert_heads[4].parameters()):
            parameter.requires_grad_(True)

    def thumbnail_probe(self, image: torch.Tensor) -> torch.Tensor:
        probe_size = max(32, round(self.image_size * 5 / 12))
        probe = F.interpolate(image, size=(probe_size, probe_size), mode="bilinear", align_corners=False)
        return F.interpolate(probe, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

    def _encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.backbone(images, output_hidden_states=True, return_dict=True)
        projected = [projection(outputs.hidden_states[layer][:, 0]) for layer, projection in zip(self.layers, self.projections, strict=True)]
        embedding = torch.cat(projected, dim=1)
        base_logit = self.base_head(outputs.last_hidden_state[:, 0]).flatten()
        return embedding, base_logit

    def components(self, image: torch.Tensor, content_targets: torch.Tensor | None = None) -> ScalePairOutputs:
        batch = image.shape[0]
        embeddings, base_logits = self._encode(torch.cat((image, self.thumbnail_probe(image)), dim=0))
        current, probe = embeddings[:batch], embeddings[batch:]
        current_logit, probe_logit = base_logits[:batch], base_logits[batch:]
        expert_input = torch.cat((current, (current - probe).abs()), dim=1)
        residual = self.residual_head(expert_input).flatten()
        content_logits = self.content_router(current)
        expert_logits = torch.stack([head(expert_input).flatten() for head in self.expert_heads], dim=1)
        routes = F.one_hot(
            content_targets if content_targets is not None else content_logits.argmax(dim=1),
            num_classes=len(self.expert_heads),
        ).to(expert_logits.dtype)
        expert_residual = (routes * expert_logits).sum(dim=1)
        return ScalePairOutputs(
            fused_logit=current_logit + residual + expert_residual,
            current_logit=current_logit,
            probe_logit=probe_logit,
            family_logits=self.family_head(current),
            embedding=current,
            content_logits=content_logits,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.components(image).fused_logit
