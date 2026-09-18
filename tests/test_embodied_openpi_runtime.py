from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from art_embodied.policies.openpi_runtime import install_openpi_vision_embedding


class _Embeddings(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embedding = torch.nn.Linear(3, 4, dtype=torch.float32)

    def forward(self, image):
        return self.patch_embedding(image)


class _Encoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(4, 4, dtype=torch.bfloat16)
        self.input_dtype = None

    def forward(self, *, inputs_embeds):
        self.input_dtype = inputs_embeds.dtype
        return SimpleNamespace(last_hidden_state=self.projection(inputs_embeds))


class _Owner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        config = SimpleNamespace(_attn_implementation="sdpa")
        encoder = _Encoder()
        encoder.layers = [
            SimpleNamespace(
                self_attn=SimpleNamespace(
                    config=SimpleNamespace(_attn_implementation="sdpa")
                )
            )
        ]
        vision = SimpleNamespace(
            config=config,
            embeddings=_Embeddings(),
            encoder=encoder,
            post_layernorm=torch.nn.LayerNorm(4, dtype=torch.bfloat16),
        )
        paligemma = SimpleNamespace(
            model=SimpleNamespace(
                vision_tower=SimpleNamespace(vision_model=vision),
                multi_modal_projector=torch.nn.Linear(
                    4, 5, dtype=torch.bfloat16
                ),
            )
        )
        self.paligemma = paligemma

    def embed_image(self, image):
        raise AssertionError("native image path should be replaced")


def test_openpi_vision_embedding_preserves_mixed_precision_boundary() -> None:
    owner = _Owner()
    model = SimpleNamespace(paligemma_with_expert=owner)
    install_openpi_vision_embedding(model)

    output = owner.embed_image(torch.ones(2, 3, dtype=torch.float32))

    vision = owner.paligemma.model.vision_tower.vision_model
    assert vision.encoder.input_dtype == torch.bfloat16
    assert vision.config._attn_implementation == "eager"
    assert vision.encoder.layers[0].self_attn.config._attn_implementation == "eager"
    assert output.dtype == torch.bfloat16
    assert output.shape == (2, 5)
