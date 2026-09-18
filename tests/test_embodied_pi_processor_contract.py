from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from art_embodied.policies.pi import _set_tokenizer_padding_side


def test_openpi_processor_sets_padding_on_tokenizer_instance() -> None:
    tokenizer = SimpleNamespace(padding_side="left")
    processor = SimpleNamespace(
        steps=[SimpleNamespace(), SimpleNamespace(input_tokenizer=tokenizer)]
    )

    _set_tokenizer_padding_side(processor, "right")

    assert tokenizer.padding_side == "right"


def test_openpi_processor_rejects_missing_or_ambiguous_tokenizers() -> None:
    with pytest.raises(RuntimeError, match="found 0"):
        _set_tokenizer_padding_side(SimpleNamespace(steps=[]), "right")

    tokenizers = [
        SimpleNamespace(input_tokenizer=SimpleNamespace()),
        SimpleNamespace(input_tokenizer=SimpleNamespace()),
    ]
    with pytest.raises(RuntimeError, match="found 2"):
        _set_tokenizer_padding_side(SimpleNamespace(steps=tokenizers), "right")
