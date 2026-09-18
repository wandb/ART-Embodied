from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from art_embodied.policies.pi import _apply_checkpoint_dtypes


class _MixedPrecisionModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(3, 2, dtype=torch.float32)
        self.register_buffer("counter", torch.zeros(1, dtype=torch.int64))


def test_apply_checkpoint_dtypes_restores_parameters_and_buffers() -> None:
    model = _MixedPrecisionModule()

    _apply_checkpoint_dtypes(
        model,
        {
            "projection.weight": "BF16",
            "projection.bias": "F32",
            "counter": "I32",
        },
    )

    assert model.projection.weight.dtype == torch.bfloat16
    assert model.projection.bias.dtype == torch.float32
    assert model.counter.dtype == torch.int32


def test_apply_checkpoint_dtypes_rejects_unknown_serialized_dtype() -> None:
    with pytest.raises(ValueError, match="Unsupported OpenPI safetensors dtype"):
        _apply_checkpoint_dtypes(
            _MixedPrecisionModule(),
            {"projection.weight": "FP8_FUTURE"},
        )
