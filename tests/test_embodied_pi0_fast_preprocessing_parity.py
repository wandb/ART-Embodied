from copy import deepcopy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from examples.embodied.pi0_fast_preprocessing_parity import (
    compare_observations,
    environment_oracle,
    rollout_input,
)


def test_repackage_does_not_rotate_normalize_or_mutate():
    image = torch.arange(18, dtype=torch.uint8).reshape(3, 2, 3)
    sample = {"observation.images.image": image, "observation.state": torch.ones(8)}
    result = rollout_input(
        sample, {"image": "observation.images.image", "state": "observation.state"}
    )
    np.testing.assert_array_equal(result["image"], image.permute(1, 2, 0).numpy())
    assert result["image"].flags.c_contiguous
    assert result["image"].dtype == np.uint8
    assert torch.equal(sample["observation.images.image"], image)
    sample["observation.images.image"] = image.float()
    with pytest.raises(ValueError, match="uint8"):
        rollout_input(sample, {"image": "observation.images.image"})


def test_exact_observations_ignore_sft_only_action_fields():
    teacher = {
        "observation.images.image": torch.arange(12).float().reshape(1, 3, 2, 2),
        "observation.language.tokens": torch.tensor([[1, 2]]),
        "observation.language.attention_mask": torch.tensor([[True, False]]),
        "action": torch.ones(1, 10, 7),
    }
    rollout = {k: v.clone() for k, v in teacher.items() if k != "action"}
    assert all(v["exact"] for v in compare_observations(teacher, rollout).values())
    for key in rollout:
        changed = deepcopy(rollout)
        changed[key].flatten()[0] = 0
        if key.endswith("image"):
            changed[key] = changed[key].flip(-1)
        assert not compare_observations(teacher, changed)[key]["exact"]


def test_invalid_comparisons_fail_closed():
    key = "observation.state"
    with pytest.raises(ValueError, match="empty"):
        compare_observations({}, {})
    with pytest.raises(ValueError, match="key sets"):
        compare_observations({key: torch.ones(8)}, {})
    with pytest.raises(ValueError, match="shape/dtype"):
        compare_observations({key: torch.ones(8)}, {key: torch.ones(1, 8)})
    with pytest.raises(ValueError, match="Nonfinite"):
        compare_observations({key: torch.tensor([float("nan")])}, {key: torch.ones(1)})


def test_environment_oracle_counts_changed_token_rows():
    pytest.importorskip("lerobot.processor.env_processor")
    calls = []

    def processor(batch):
        calls.append(batch)
        tokens = torch.zeros(len(batch["task"]), 4, dtype=torch.int64)
        if len(calls) == 2:
            tokens[0, 2] = 7
        return {"observation.language.tokens": tokens}

    result = environment_oracle(processor, "bowl", random_states=16)
    assert result["samples"] == 21
    assert result["language_token_changed_rows"] == 1
    assert result["language_token_changed_indices"] == [0]
    assert all(result["camera_rotation_exact"].values())
    assert len(calls) == 2
