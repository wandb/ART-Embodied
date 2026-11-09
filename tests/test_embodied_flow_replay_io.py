from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from art_embodied.policies.flow_policy import FlowModelInputs, FlowSDERollout
from art_embodied.policies.flow_replay_io import (
    compact_masked_images,
    load_flow_replay,
    save_flow_replay,
)
from art_embodied.policies.flow_sde import FlowSDETransitionRecord


def record():
    return FlowSDERollout(
        actions=torch.randn(4, 15, 8),
        transition=FlowSDETransitionRecord(
            torch.randn(4, 15, 32),
            torch.randn(4, 15, 32),
            torch.arange(4),
            torch.randn(4, 15, 32),
        ),
        inputs=FlowModelInputs(
            images=(torch.randn(4, 3, 8, 8), torch.full((4, 3, 8, 8), -1.0)),
            image_masks=(
                torch.ones(4, dtype=torch.bool),
                torch.zeros(4, dtype=torch.bool),
            ),
            language_tokens=torch.ones(4, 5, dtype=torch.long),
            language_masks=torch.ones(4, 5, dtype=torch.bool),
            state=torch.randn(4, 32),
        ),
    )


def test_replay_roundtrip_uses_weights_only_and_preserves_compressed_storage(tmp_path):
    before = record()
    path = tmp_path / "chunk.pt"
    save_flow_replay(path, before, valid_count=3)
    assert isinstance(torch.load(path, weights_only=True)["inputs"], dict)
    after, valid = load_flow_replay(path)
    assert valid == 3
    torch.testing.assert_close(after.actions, before.actions, rtol=0, atol=0)
    torch.testing.assert_close(
        after.transition.old_logprobs, before.transition.old_logprobs, rtol=0, atol=0
    )
    for a, b in zip(after.inputs.images, before.inputs.images):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert after.inputs.images[1].untyped_storage().nbytes() == 4
    with pytest.raises(FileExistsError):
        save_flow_replay(path, before, valid_count=3)


def test_compaction_does_not_change_nonconstant_or_unmasked_images():
    original = record()
    varied = torch.randn_like(original.inputs.images[1])
    value = replace(
        original,
        inputs=replace(original.inputs, images=(original.inputs.images[0], varied)),
    )
    result = compact_masked_images(value)
    assert result.inputs.images[0] is value.inputs.images[0]
    assert result.inputs.images[1] is varied
    active = replace(
        original,
        inputs=replace(
            original.inputs,
            image_masks=(
                original.inputs.image_masks[0],
                torch.ones(4, dtype=torch.bool),
            ),
        ),
    )
    assert compact_masked_images(active).inputs.images[1] is active.inputs.images[1]


@pytest.mark.parametrize("valid", [0, 5, True, 1.5])
def test_replay_valid_count_is_checked(tmp_path, valid):
    with pytest.raises(ValueError, match="valid_count"):
        save_flow_replay(tmp_path / "bad.pt", record(), valid_count=valid)


def test_loader_rejects_nonfinite_and_malformed_batches(tmp_path):
    original = record()
    original.actions[0, 0, 0] = float("nan")
    path = tmp_path / "bad.pt"
    save_flow_replay(path, original, valid_count=4)
    with pytest.raises(ValueError, match="Nonfinite"):
        load_flow_replay(path)
    payload = torch.load(path, weights_only=True)
    payload["actions"] = torch.zeros(3, 15, 8)
    torch.save(payload, path)
    with pytest.raises(ValueError, match="dimensions"):
        load_flow_replay(path)
