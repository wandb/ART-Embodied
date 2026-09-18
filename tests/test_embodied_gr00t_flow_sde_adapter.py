from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from art_embodied.backends.flow_sde import (  # noqa: E402
    TRANSIENT_FLOW_SDE_ROLLOUT_KEY,
)
from art_embodied.integrations.gr00t_flow_sde import (  # noqa: E402
    GR00TN15FlowSDEPolicyAdapter,
    _libero_observation_batch,
)
from art_embodied.policies.flow_policy import (  # noqa: E402
    FlowSDERollout,
    GR00TFlowModelInputs,
)
from art_embodied.policies.flow_sde import (  # noqa: E402
    FlowSDESchedule,
    FlowSDETransitionRecord,
)
from art_embodied.policies.gr00t import GR00TN15FlowPolicy  # noqa: E402


class _Policy:
    family = "gr00t_n1d5"
    device = "cpu"
    execution_horizon = 2
    action_dim = 7
    model_action_horizon = 4
    schedule = FlowSDESchedule(
        num_steps=3,
        noise_level=0.3,
        noise_time="zero",
    )

    def __init__(self) -> None:
        self.native_policy = SimpleNamespace(model=torch.nn.Linear(1, 1))
        self.selected_indices = None
        self.normalized = None
        self.native_predictions = 0

    def apply_observation_transforms(self, observations):
        self.normalized = observations
        return observations

    def prepare_flow_inputs(self, normalized):
        batch_size = normalized["state.x"].shape[0]
        return GR00TFlowModelInputs(
            vision_language_features=torch.zeros(batch_size, 2, 3),
            vision_language_attention_mask=torch.ones(batch_size, 2, dtype=torch.bool),
            state=torch.zeros(batch_size, 1, 64),
            embodiment_id=torch.zeros(batch_size, dtype=torch.long),
        )

    def sample_flow_sde(self, inputs, *, selected_index, **kwargs):
        del kwargs
        self.selected_indices = selected_index.clone()
        batch_size = inputs.batch_size
        actions = torch.arange(4 * 7, dtype=torch.float32).reshape(1, 4, 7)
        return FlowSDERollout(
            actions=actions.repeat(batch_size, 1, 1),
            transition=FlowSDETransitionRecord(
                previous_states=torch.zeros(batch_size, 2, 7),
                next_states=torch.ones(batch_size, 2, 7),
                selected_indices=selected_index,
                old_logprobs=torch.full((batch_size, 2, 7), -0.25),
            ),
            inputs=inputs,
        )

    def unapply_action_transforms(self, actions):
        return _action_components(actions)

    def predict_native_action_chunk(self, observations):
        self.native_predictions += 1
        batch_size = observations["state.x"].shape[0]
        actions = torch.arange(4 * 7, dtype=torch.float32).reshape(1, 4, 7)
        return _action_components(actions.repeat(batch_size, 1, 1))


def _action_components(actions):
    keys = (
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    )
    array = actions.detach().cpu().numpy()
    return {key: array[..., index : index + 1] for index, key in enumerate(keys)}


def _observation(offset: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "image": np.zeros((8, 8, 3), dtype=np.uint8),
        "wrist_image": np.ones((8, 8, 3), dtype=np.uint8),
        "proprio_state": np.arange(8, dtype=np.float32) + offset,
    }


def test_gr00t_libero_mapping_preserves_official_n1d5_keys() -> None:
    mapped = _libero_observation_batch(
        [_observation(), _observation(1.0)],
        ["first", "second"],
    )

    assert mapped["video.image"].shape == (2, 1, 8, 8, 3)
    assert mapped["video.wrist_image"].shape == (2, 1, 8, 8, 3)
    assert mapped["state.x"].shape == (2, 1, 1)
    assert mapped["state.gripper"].shape == (2, 1, 2)
    assert mapped["annotation.human.action.task_description"].tolist() == [
        "first",
        "second",
    ]


def test_gr00t_adapter_records_only_executed_prefix_and_retains_lookahead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "art_embodied.integrations.gr00t_flow_sde.random.randint",
        lambda lower, upper: 1,
    )
    policy = _Policy()
    adapter = GR00TN15FlowSDEPolicyAdapter(policy=policy)

    predictions = adapter.predict_batch(
        [_observation(), _observation(1.0)],
        tasks=["move block", "move block"],
        step=7,
    )

    torch.testing.assert_close(policy.selected_indices, torch.tensor([1, 1]))
    assert len(predictions) == 2
    for prediction in predictions:
        assert prediction.native_action.shape == (2, 7)
        assert prediction.predicted_action_chunk.shape == (4, 7)
        assert np.asarray(prediction.action.decoded).shape == (2, 7)
        assert prediction.action.logprobs["chunk_logprob"] == pytest.approx(-3.5)
        assert prediction.action.metadata["selected_denoise_index"] == 1
        retained = prediction.action.metadata[TRANSIENT_FLOW_SDE_ROLLOUT_KEY]
        assert retained.inputs.batch_size == 1
    np.testing.assert_array_equal(
        predictions[0].predicted_action_chunk[:, -1],
        np.sign(1.0 - 2.0 * np.array([6.0, 13.0, 20.0, 27.0])),
    )


def test_gr00t_adapter_uses_native_ode_for_eval() -> None:
    policy = _Policy()
    adapter = GR00TN15FlowSDEPolicyAdapter(policy=policy, sampling_mode="eval")

    prediction = adapter.predict(_observation(), task="move block", step=0)

    assert policy.native_predictions == 1
    assert prediction.native_action.shape == (2, 7)
    assert prediction.predicted_action_chunk.shape == (4, 7)
    assert prediction.action.logprobs is None
    assert prediction.action.metadata["probability_model"] == "native_flow_ode"


def test_gr00t_adapter_rejects_missing_task_or_invalid_state() -> None:
    adapter = GR00TN15FlowSDEPolicyAdapter(policy=_Policy())

    with pytest.raises(ValueError, match="task descriptions"):
        adapter.predict(_observation(), task=None, step=0)
    invalid = _observation()
    invalid["proprio_state"] = np.zeros(7, dtype=np.float32)
    with pytest.raises(ValueError, match="8 state values"):
        adapter.predict(invalid, task="move block", step=0)


def test_gr00t_inverse_transform_receives_fp32_actions() -> None:
    received: dict[str, torch.Tensor] = {}

    class _NativePolicy:
        def unapply_transforms(self, values):
            received.update(values)
            return values

    policy = object.__new__(GR00TN15FlowPolicy)
    policy.native_policy = _NativePolicy()

    result = policy.unapply_action_transforms(
        torch.zeros(1, 4, 7, dtype=torch.bfloat16)
    )

    assert received["action"].dtype == torch.float32
    assert received["action"].device.type == "cpu"
    assert result["action"] is received["action"]
