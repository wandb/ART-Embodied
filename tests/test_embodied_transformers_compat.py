from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

from art_embodied.policies.transformers_compat import (  # noqa: E402
    ensure_art_mask_patch_compatibility,
)


def test_art_mask_patch_matches_installed_transformers() -> None:
    import art.transformers.patches as art_patches
    from transformers import masking_utils

    original_current = masking_utils._preprocess_mask_arguments
    try:
        masking_utils._preprocess_mask_arguments = (
            art_patches._patched_preprocess_mask_arguments
        )

        changed = ensure_art_mask_patch_compatibility()

        expected_change = tuple(
            inspect.signature(art_patches._patched_preprocess_mask_arguments).parameters
        ) != tuple(inspect.signature(art_patches._preprocess_mask_arguments).parameters)
        assert changed is expected_change
        assert tuple(
            inspect.signature(masking_utils._preprocess_mask_arguments).parameters
        ) == tuple(inspect.signature(art_patches._preprocess_mask_arguments).parameters)
    finally:
        masking_utils._preprocess_mask_arguments = original_current


def test_art_mask_compatibility_does_not_replace_unrelated_function(
    monkeypatch,
) -> None:
    from transformers import masking_utils

    sentinel = lambda *args, **kwargs: (args, kwargs)  # noqa: E731
    monkeypatch.setattr(masking_utils, "_preprocess_mask_arguments", sentinel)

    assert ensure_art_mask_patch_compatibility() is False
    assert masking_utils._preprocess_mask_arguments is sentinel


def _legacy_native(
    config,
    input_embeds,
    attention_mask,
    cache_position,
    past_key_values,
    position_ids,
    layer_idx,
):
    return (
        config,
        input_embeds,
        attention_mask,
        cache_position,
        past_key_values,
        position_ids,
        layer_idx,
    )


def _modern_native(
    config,
    inputs_embeds,
    attention_mask,
    past_key_values,
    position_ids,
    layer_idx,
    encoder_hidden_states=None,
):
    return (
        config,
        inputs_embeds,
        attention_mask,
        past_key_values,
        position_ids,
        layer_idx,
        encoder_hidden_states,
    )


@pytest.mark.parametrize("native", [_legacy_native, _modern_native])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("keywords", [False, True])
def test_art_mask_signature_bridge_preserves_arguments(
    monkeypatch,
    native,
    packed,
    keywords,
) -> None:
    import art.transformers.patches as art_patches
    from transformers import masking_utils

    incompatible = _modern_native if native is _legacy_native else _legacy_native
    monkeypatch.setattr(art_patches, "_preprocess_mask_arguments", native)
    monkeypatch.setattr(art_patches, "_patched_preprocess_mask_arguments", incompatible)
    monkeypatch.setattr(masking_utils, "_preprocess_mask_arguments", incompatible)
    assert ensure_art_mask_patch_compatibility() is True
    repaired = masking_utils._preprocess_mask_arguments
    assert inspect.signature(repaired) == inspect.signature(native)
    positions = torch.arange(4).reshape(1, 4)
    if packed:
        positions = positions.unsqueeze(0)
    arguments = {key: object() for key in inspect.signature(native).parameters}
    arguments["position_ids"] = positions
    actual = repaired(**arguments) if keywords else repaired(*arguments.values())
    expected_arguments = dict(arguments)
    expected_arguments["position_ids"] = positions[0] if packed else positions
    expected = native(**expected_arguments)
    for key, value, reference in zip(arguments, actual, expected, strict=True):
        if key == "position_ids":
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
        else:
            assert value is reference
    assert ensure_art_mask_patch_compatibility() is False
    assert masking_utils._preprocess_mask_arguments is repaired


def test_art_mask_bridge_preserves_native_forward_and_gradients(monkeypatch) -> None:
    import art.transformers.patches as art_patches
    from transformers import LlamaConfig, LlamaModel, masking_utils

    torch.manual_seed(123)
    model = LlamaModel(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    ).eval()
    ids = torch.tensor([[1, 2, 3]])
    monkeypatch.setattr(
        masking_utils,
        "_preprocess_mask_arguments",
        art_patches._preprocess_mask_arguments,
    )
    reference = model(ids).last_hidden_state
    reference.square().mean().backward()
    gradients = {
        name: parameter.grad.clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)
    monkeypatch.setattr(
        masking_utils,
        "_preprocess_mask_arguments",
        art_patches._patched_preprocess_mask_arguments,
    )
    ensure_art_mask_patch_compatibility()
    actual = model(ids).last_hidden_state
    actual.square().mean().backward()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for name, parameter in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)
        else:
            assert parameter.grad is None
