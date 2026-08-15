import torch
from transformers import ViTConfig, ViTForImageClassification

from cyclopes.vit_modeling import MultiLayerScalePairViT


def _tiny_source() -> ViTForImageClassification:
    return ViTForImageClassification(
        ViTConfig(
            image_size=32,
            patch_size=16,
            hidden_size=32,
            num_hidden_layers=3,
            num_attention_heads=4,
            intermediate_size=64,
            num_labels=1,
        )
    )


def test_multilayer_vit_starts_as_the_source_detector() -> None:
    source = _tiny_source().eval()
    model = MultiLayerScalePairViT(source, layers=(1, 2, 3)).eval()
    image = torch.rand(2, 3, 32, 32)

    with torch.inference_mode():
        source_logits = source(image).logits.flatten()
        outputs = model.components(image)

    torch.testing.assert_close(outputs.fused_logit, source_logits)
    assert outputs.embedding.shape == (2, 384)
    assert outputs.content_logits.shape == (2, 5)
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert any(parameter.requires_grad for parameter in model.residual_head.parameters())

    model.freeze_prior()
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    expected = {
        name
        for prefix, module in (
            ("content_router.", model.content_router),
            ("expert_heads.0.", model.expert_heads[0]),
            ("expert_heads.1.", model.expert_heads[1]),
            ("expert_heads.2.", model.expert_heads[2]),
            ("expert_heads.3.", model.expert_heads[3]),
            ("expert_heads.4.", model.expert_heads[4]),
        )
        for name in [f"{prefix}{item}" for item, _ in module.named_parameters()]
    }
    assert trainable == expected

    with torch.inference_mode():
        routed = model.components(image, torch.tensor([1, 4]))
    torch.testing.assert_close(routed.fused_logit, source_logits)


def test_expert_training_mode_updates_only_selected_experts() -> None:
    model = MultiLayerScalePairViT(_tiny_source(), layers=(1, 2, 3))
    model.set_expert_training_mode()

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert any(name.startswith("content_router.") for name in trainable)
    for index in range(5):
        assert any(name.startswith(f"expert_heads.{index}.") for name in trainable)

    images = torch.rand(5, 3, 32, 32)
    route = torch.arange(5, dtype=torch.long)
    outputs = model.components(images, route)
    loss = outputs.fused_logit.mean() + torch.nn.functional.cross_entropy(outputs.content_logits, route)
    loss.backward()

    for name, parameter in model.named_parameters():
        if name.startswith("content_router."):
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0
        elif name.startswith("expert_heads."):
            assert parameter.grad is not None
            if name.endswith("3.weight") or name.endswith("3.bias"):
                assert parameter.grad.abs().sum() > 0
        else:
            assert parameter.grad is None
