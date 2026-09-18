"""ART-native OpenVLA-OFT rollout contract.

The Hugging Face OpenVLA-OFT checkpoints contain the vision-language model and
action-token helpers.  This module supplies only the small runtime contract that
is not serialized in those checkpoints: action geometry, multi-camera setup,
and the placeholder-action embedding used for rollout and rescoring.
"""

from __future__ import annotations

from types import MethodType
from typing import Any


def configure_native_openvla_oft_model(
    model: Any,
    processor: Any,
    *,
    action_dim: int,
    num_action_chunks: int,
    max_prompt_length: int,
    num_images_in_input: int,
    unnorm_key: str | None,
) -> dict[str, Any]:
    """Configure a remote-code OpenVLA-OFT model for ART-native RL rollout."""

    required = (
        "_prepare_input_for_action_prediction",
        "_process_vision_features",
        "_build_multimodal_attention",
        "get_input_embeddings",
        "language_model",
        "vision_backbone",
    )
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise TypeError(
            "ART's native OpenVLA-OFT loader requires a compatible action-token "
            f"checkpoint; missing model attributes: {missing}"
        )

    action_dim = int(action_dim)
    num_action_chunks = int(num_action_chunks)
    max_prompt_length = int(max_prompt_length)
    num_images_in_input = int(num_images_in_input)
    if min(action_dim, num_action_chunks, max_prompt_length, num_images_in_input) <= 0:
        raise ValueError("OpenVLA-OFT native runtime dimensions must be positive")

    model.action_dim = action_dim
    model.num_action_chunks = num_action_chunks
    model.max_prompt_length = max_prompt_length
    model.input_processor = processor
    model.processor = processor
    model.unnorm_key = _resolve_native_unnorm_key(model, unnorm_key)
    model._art_action_stop_token_id = _resolve_action_stop_token_id(model, processor)

    # `from_pretrained(torch_dtype=...)` does not necessarily cast non-persistent
    # buffers on recent Transformers releases. In particular, leaving Llama RoPE
    # frequencies in FP32 while the policy is BF16 changes every action logit.
    # RLinf's validated worker casts the full model after loading, so make that
    # dtype contract explicit in the independent ART runtime too.
    try:
        parameter_dtype = next(model.parameters()).dtype
    except StopIteration:
        parameter_dtype = None
    if parameter_dtype is not None:
        model.to(dtype=parameter_dtype)

    set_num_images = getattr(model.vision_backbone, "set_num_images_in_input", None)
    if not callable(set_num_images):
        raise TypeError(
            "ART's native OpenVLA-OFT loader requires "
            "vision_backbone.set_num_images_in_input()"
        )
    set_num_images(num_images_in_input)

    # The checkpoint's generic forward path orders padded prompt tokens for its
    # VERL helper.  Rollout GRPO instead needs fixed prompt positions followed by
    # zero-valued action placeholders. Bind that contract to this model instance
    # without importing or subclassing another RL framework.
    model._build_embedding = MethodType(_build_action_placeholder_embedding, model)
    model._art_native_openvla_oft = True

    return {
        "runtime": "art_native_openvla_oft",
        "action_dim": action_dim,
        "num_action_chunks": num_action_chunks,
        "action_tokens_per_chunk": action_dim,
        "action_tokens_per_policy_call": action_dim * num_action_chunks,
        "max_prompt_length": max_prompt_length,
        "num_images_in_input": num_images_in_input,
        "unnorm_key": model.unnorm_key,
        "stop_token_id": model._art_action_stop_token_id,
        "model_dtype": str(parameter_dtype) if parameter_dtype is not None else None,
    }


def _build_action_placeholder_embedding(
    self: Any,
    input_ids: Any,
    attention_mask: Any,
    pixel_values: Any,
) -> tuple[Any, Any]:
    """Build the exact fixed-position action-token embedding used by ART RL."""

    import torch

    stop_index = int(self._art_action_stop_token_id)
    if not bool(torch.all(input_ids[:, -1] == stop_index)):
        raise ValueError("OpenVLA-OFT action input must end with the stop token")
    if input_ids.shape != attention_mask.shape:
        raise ValueError(
            "OpenVLA-OFT input_ids and attention_mask must have equal shapes"
        )

    # The stop token is required while constructing the sequence but is not fed
    # into the language model. The preceding action slots are embedding zeros.
    input_ids = input_ids[:, :-1]
    attention_mask = attention_mask[:, :-1]
    action_token_count = int(self.action_dim) * int(self.num_action_chunks)
    if int(input_ids.shape[1]) < action_token_count:
        raise ValueError(
            "OpenVLA-OFT input is shorter than its configured action placeholders: "
            f"sequence={input_ids.shape[1]}, action_tokens={action_token_count}"
        )
    action_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    action_mask[:, -action_token_count:] = True
    input_embeddings = self.get_input_embeddings()(input_ids)
    input_embeddings = input_embeddings * (~action_mask.unsqueeze(-1))

    projected_patches = self._process_vision_features(
        pixel_values,
        None,
        use_film=False,
    )
    expected_patches = (
        self.vision_backbone.get_num_patches()
        * self.vision_backbone.get_num_images_in_input()
    )
    if int(projected_patches.shape[1]) != int(expected_patches):
        raise ValueError(
            "OpenVLA-OFT vision token count mismatch: "
            f"expected={expected_patches}, actual={projected_patches.shape[1]}"
        )
    projected_patches = projected_patches.reshape(
        input_embeddings.shape[0],
        -1,
        *projected_patches.shape[2:],
    )
    multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
        input_embeddings,
        projected_patches,
        attention_mask,
    )
    return multimodal_embeddings, multimodal_attention_mask


def _resolve_action_stop_token_id(model: Any, processor: Any) -> int:
    candidates = (
        getattr(model, "stop_index", None),
        getattr(getattr(model, "language_model", None), "config", None),
        getattr(getattr(processor, "tokenizer", None), "eos_token_id", None),
    )
    for candidate in candidates:
        if candidate is not None and not hasattr(candidate, "eos_token_id"):
            value = int(candidate)
            if value >= 0:
                return value
        eos_token_id = getattr(candidate, "eos_token_id", None)
        if eos_token_id is not None and int(eos_token_id) >= 0:
            return int(eos_token_id)
    raise ValueError(
        "ART-native OpenVLA-OFT could not resolve the action stop token from "
        "model.stop_index, language_model.config.eos_token_id, or the tokenizer"
    )


def _resolve_native_unnorm_key(model: Any, configured: str | None) -> str | None:
    if configured is None:
        configured = getattr(getattr(model, "config", None), "unnorm_key", None)
    if configured is None:
        return None
    norm_stats = getattr(model, "norm_stats", {}) or {}
    if configured in norm_stats:
        return str(configured)
    no_noops = f"{configured}_no_noops"
    if no_noops in norm_stats:
        return no_noops
    available = sorted(str(key) for key in norm_stats)
    raise KeyError(
        f"OpenVLA-OFT unnorm_key {configured!r} is unavailable; available keys: {available}"
    )
