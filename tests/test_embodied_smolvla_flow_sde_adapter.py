from __future__ import annotations

import re

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from art_embodied.backends.flow_sde import TRANSIENT_FLOW_SDE_ROLLOUT_KEY
from art_embodied.integrations.smolvla_flow_sde import (
    SmolVLAFlowSDEPolicyAdapter,
    _map_smolvla_observation,
)
from art_embodied.policies.flow_policy import FlowModelInputs, FlowSDERollout
from art_embodied.policies.flow_sde import FlowSDESchedule, FlowSDETransitionRecord
from art_embodied.vla_trainable import _full_action_expert_lora_targets


class _Processor:
    def __init__(self, *, scale: float = 1.0) -> None:
        self.scale = scale
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def __call__(self, value):
        if isinstance(value, torch.Tensor):
            return value * self.scale
        return {
            "observation.state": value["observation.state"],
            "task": value["task"],
        }


class _Policy:
    family = "smolvla"
    device = "cpu"
    execution_horizon = 1
    action_dim = 2
    schedule = FlowSDESchedule(num_steps=3, noise_level=0.2)

    def __init__(self) -> None:
        self.policy = _Processor()
        self.preprocessor = _Processor()
        self.postprocessor = _Processor(scale=2.0)
        self.selected_indices = None
        self.processed = None
        self.native_predictions = 0

    def reset(self) -> None:
        for component in (self.policy, self.preprocessor, self.postprocessor):
            component.reset()

    def sample_flow_sde(self, processed, *, selected_index, **kwargs):
        del kwargs
        self.processed = processed
        self.selected_indices = selected_index.clone()
        batch = processed["observation.state"].shape[0]
        inputs = FlowModelInputs(
            images=(torch.zeros(batch, 3, 2, 2),),
            image_masks=(torch.ones(batch, dtype=torch.bool),),
            language_tokens=torch.ones(batch, 2, dtype=torch.long),
            language_masks=torch.ones(batch, 2, dtype=torch.bool),
            state=processed["observation.state"],
        )
        return FlowSDERollout(
            actions=torch.tensor([[[1.0, 2.0], [9.0, 9.0]]]).repeat(batch, 1, 1),
            transition=FlowSDETransitionRecord(
                previous_states=torch.zeros(batch, 2, 2),
                next_states=torch.ones(batch, 2, 2),
                selected_indices=selected_index,
                old_logprobs=torch.tensor([[[-0.2, -0.3], [-0.4, -0.5]]]).repeat(
                    batch, 1, 1
                ),
            ),
            inputs=inputs,
        )

    def predict_native_action_chunk(self, processed):
        self.native_predictions += 1
        batch = processed["observation.state"].shape[0]
        return torch.tensor([[[3.0, 4.0], [9.0, 9.0]]]).repeat(batch, 1, 1)


def _prepare(observation, device, task, robot_type):
    assert device == torch.device("cpu")
    assert robot_type == "panda"
    return {
        "observation.state": torch.from_numpy(observation["observation.state"])[None],
        "task": task,
    }


def test_smolvla_adapter_batches_same_reset_siblings_with_one_denoise_step(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "art_embodied.integrations.smolvla_flow_sde.random.randint",
        lambda lower, upper: 2,
    )
    policy = _Policy()
    adapter = SmolVLAFlowSDEPolicyAdapter(
        policy=policy,
        robot_type="panda",
        prepare_observation_fn=_prepare,
    )

    predictions = adapter.predict_batch(
        [
            {"observation.state": np.array([0.5, 1.0], dtype=np.float32)},
            {"observation.state": np.array([1.5, 2.0], dtype=np.float32)},
        ],
        tasks=["move block", "move block"],
        step=4,
    )

    torch.testing.assert_close(policy.selected_indices, torch.tensor([2, 2]))
    torch.testing.assert_close(
        policy.processed["observation.state"],
        torch.tensor([[0.5, 1.0], [1.5, 2.0]]),
    )
    assert len(predictions) == 2
    for prediction in predictions:
        torch.testing.assert_close(
            prediction.native_action,
            torch.tensor([[2.0, 4.0]]),
        )
        torch.testing.assert_close(
            prediction.predicted_action_chunk,
            torch.tensor([[2.0, 4.0], [18.0, 18.0]]),
        )
        assert prediction.execution_horizon == 1
        assert prediction.action.logprobs["chunk_logprob"] == pytest.approx(-0.5)
        assert prediction.action.metadata["model_batch_size"] == 2
        assert prediction.action.metadata["selected_denoise_index"] == 2
        retained = prediction.action.metadata[TRANSIENT_FLOW_SDE_ROLLOUT_KEY]
        assert retained.inputs.batch_size == 1


def test_smolvla_adapter_uses_native_ode_for_evaluation() -> None:
    policy = _Policy()
    adapter = SmolVLAFlowSDEPolicyAdapter(
        policy=policy,
        robot_type="panda",
        prepare_observation_fn=_prepare,
        sampling_mode="eval",
    )

    prediction = adapter.predict(
        {"observation.state": np.array([0.5, 1.0], dtype=np.float32)},
        task="move block",
        step=0,
    )

    assert policy.native_predictions == 1
    torch.testing.assert_close(
        prediction.native_action,
        torch.tensor([[6.0, 8.0]]),
    )
    torch.testing.assert_close(
        prediction.predicted_action_chunk,
        torch.tensor([[6.0, 8.0], [18.0, 18.0]]),
    )
    assert prediction.action.logprobs is None
    assert prediction.action.metadata["probability_model"] == "native_flow_ode"


def test_smolvla_adapter_resets_native_state_and_processors() -> None:
    seeds = []
    policy = _Policy()
    adapter = SmolVLAFlowSDEPolicyAdapter(
        policy=policy,
        seed_fn=seeds.append,
        prepare_observation_fn=_prepare,
    )

    adapter.reset(seed=19)

    assert seeds == [19]
    assert policy.policy.reset_count == 1
    assert policy.preprocessor.reset_count == 1
    assert policy.postprocessor.reset_count == 1


def test_smolvla_observation_mapping_is_explicit_and_contiguous() -> None:
    rotated = np.zeros((4, 4, 3), dtype=np.uint8)[::-1, ::-1]

    mapped = _map_smolvla_observation(
        {
            "image": rotated,
            "wrist_image": rotated,
            "proprio_state": np.zeros(8, dtype=np.float32),
        },
        observation_key_map={
            "image": "observation.images.image",
            "wrist_image": "observation.images.image2",
            "proprio_state": "observation.state",
        },
    )

    assert set(mapped) == {
        "observation.images.image",
        "observation.images.image2",
        "observation.state",
    }
    assert all(value.flags.c_contiguous for value in mapped.values())


def test_smolvla_full_action_expert_lora_excludes_vlm_layers() -> None:
    pattern = _full_action_expert_lora_targets("smolvla_action_expert_lora", "smolvla")

    assert re.fullmatch(
        pattern,
        "base_model.model.vlm_with_expert.lm_expert.layers.0.mlp.up_proj",
    )
    assert re.fullmatch(pattern, "base_model.model.action_out_proj")
    assert not re.fullmatch(
        pattern,
        "base_model.model.vlm_with_expert.vlm.language_model.layers.0.mlp.up_proj",
    )
