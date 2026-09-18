"""Narrow runtime compatibility fixes for native LeRobot policies."""

from __future__ import annotations

from functools import wraps
import inspect
from typing import Any


def ensure_art_mask_patch_compatibility() -> bool:
    """Adapt ART's legacy mask patch to the installed Transformers contract.

    ART's packed-text wrapper and the installed Transformers helper can use
    different signatures: 4.57 accepts ``cache_position``, while 5.5 accepts
    ``encoder_hidden_states``. Match the native helper in either direction,
    retaining ART's 3-D ``position_ids`` normalization.

    Returns ``True`` when an incompatible ART wrapper was replaced.
    """

    try:
        import art.transformers.patches as art_patches
        from transformers import masking_utils
    except (ImportError, AttributeError):
        return False

    current = masking_utils._preprocess_mask_arguments
    art_wrapper = getattr(art_patches, "_patched_preprocess_mask_arguments", None)
    original = getattr(art_patches, "_preprocess_mask_arguments", None)
    if current is not art_wrapper or original is None:
        return False

    parameters = tuple(inspect.signature(original).parameters)
    if tuple(inspect.signature(current).parameters) == parameters:
        return False

    legacy_parameters = (
        "config",
        "input_embeds",
        "attention_mask",
        "cache_position",
        "past_key_values",
        "position_ids",
        "layer_idx",
    )
    modern_parameters = (
        "config",
        "inputs_embeds",
        "attention_mask",
        "past_key_values",
        "position_ids",
        "layer_idx",
        "encoder_hidden_states",
    )
    if parameters == legacy_parameters:

        @wraps(original)
        def _compatible_legacy_preprocess_mask_arguments(
            config: Any,
            input_embeds: Any,
            attention_mask: Any,
            cache_position: Any,
            past_key_values: Any,
            position_ids: Any,
            layer_idx: Any,
        ) -> Any:
            if position_ids is not None and position_ids.ndim == 3:
                position_ids = position_ids[0]
            return original(
                config,
                input_embeds,
                attention_mask,
                cache_position,
                past_key_values,
                position_ids,
                layer_idx,
            )

        masking_utils._preprocess_mask_arguments = (
            _compatible_legacy_preprocess_mask_arguments
        )
        return True

    if parameters != modern_parameters:
        return False

    @wraps(original)
    def _compatible_preprocess_mask_arguments(
        config: Any,
        inputs_embeds: Any,
        attention_mask: Any,
        past_key_values: Any,
        position_ids: Any,
        layer_idx: Any,
        encoder_hidden_states: Any = None,
    ) -> Any:
        if position_ids is not None and position_ids.ndim == 3:
            position_ids = position_ids[0]
        return original(
            config,
            inputs_embeds,
            attention_mask,
            past_key_values,
            position_ids,
            layer_idx,
            encoder_hidden_states,
        )

    masking_utils._preprocess_mask_arguments = _compatible_preprocess_mask_arguments
    return True
