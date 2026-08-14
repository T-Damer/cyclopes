import torch
from transformers import ViTConfig, ViTForImageClassification

from cyclopes.vit_modeling import MultiLayerScalePairViT


def test_multilayer_vit_starts_as_the_source_detector() -> None:
    source = ViTForImageClassification(
        ViTConfig(
            image_size=32,
            patch_size=16,
            hidden_size=32,
            num_hidden_layers=3,
            num_attention_heads=4,
            intermediate_size=64,
            num_labels=1,
        )
    ).eval()
    model = MultiLayerScalePairViT(source, layers=(1, 2, 3)).eval()
    image = torch.rand(2, 3, 32, 32)

    with torch.inference_mode():
        expected = source(image).logits.flatten()
        outputs = model.components(image)

    torch.testing.assert_close(outputs.fused_logit, expected)
    assert outputs.embedding.shape == (2, 384)
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert any(parameter.requires_grad for parameter in model.residual_head.parameters())
