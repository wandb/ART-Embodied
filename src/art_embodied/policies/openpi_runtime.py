"""Runtime adapters for imported OpenPI mixed-precision checkpoints."""

from __future__ import annotations

from types import MethodType
from typing import Any


def install_openpi_vision_embedding(model: Any) -> None:
    """Install OpenPI's FP32-embedding to BF16-encoder precision boundary.

    OpenPI deliberately keeps the SigLIP patch and position embeddings in
    FP32 while its encoder and multimodal projector are BF16. LeRobot 0.6
    instead forces the complete vision path to FP32. Restoring checkpoint
    dtypes therefore also requires the explicit activation cast performed by
    OpenPI's patched Transformers runtime.
    """

    owner = model.paligemma_with_expert
    vision = owner.paligemma.model.vision_tower.vision_model
    # OpenPI's patched Transformers runtime executes SigLIP attention through
    # its eager implementation. Newer Transformers may otherwise select SDPA
    # for the vision config even though LeRobot already pins only the language
    # model to eager attention.
    vision.config._attn_implementation = "eager"
    for layer in vision.encoder.layers:
        layer.self_attn.config._attn_implementation = "eager"
    owner.embed_image = MethodType(_openpi_embed_image, owner)


def _openpi_embed_image(owner: Any, image: Any) -> Any:
    paligemma = owner.paligemma.model
    vision = paligemma.vision_tower.vision_model
    patch_dtype = vision.embeddings.patch_embedding.weight.dtype
    hidden_states = vision.embeddings(image.to(dtype=patch_dtype))

    encoder_dtype = next(vision.encoder.parameters()).dtype
    hidden_states = hidden_states.to(dtype=encoder_dtype)
    encoder_outputs = vision.encoder(inputs_embeds=hidden_states)
    last_hidden_state = vision.post_layernorm(encoder_outputs.last_hidden_state)

    projector = paligemma.multi_modal_projector
    projector_dtype = next(projector.parameters()).dtype
    return projector(last_hidden_state.to(dtype=projector_dtype))
