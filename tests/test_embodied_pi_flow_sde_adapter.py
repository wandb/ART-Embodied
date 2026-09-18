from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from art_embodied.backends.flow_sde import (  # noqa: E402
    TRANSIENT_FLOW_SDE_ROLLOUT_KEY,
)
from art_embodied.integrations.pi_flow_sde import (  # noqa: E402
    PIFlowSDEPolicyAdapter,
    _postprocess_pi_actions,
    _prepare_pi_observation,
)
from art_embodied.policies.flow_sde import (  # noqa: E402
    FlowSDESchedule,
    FlowSDETransitionRecord,
)
from art_embodied.policies.pi_flow_sde import (  # noqa: E402
    PIFlowModelInputs,
    PIFlowSDERollout,
)


class _Resettable:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


class _Preprocessor(_Resettable):
    def __call__(self, batch):
        assert batch["task"] == "move block"
        return {
            "observation.state": batch["observation.state"],
            "action": None,
            "next.reward": 0.0,
            "robot_type": "panda",
        }


class _Postprocessor(_Resettable):
    def __init__(self) -> None:
        super().__init__()
        self.received_horizons = []

    def __call__(self, actions):
        self.received_horizons.append(int(actions.shape[1]))
        return actions * 2.0


class _Policy:
    family = "pi05"
    device = "cpu"
    execution_horizon = 2
    action_dim = 1
    schedule = FlowSDESchedule(num_steps=3, noise_level=0.3)

    def __init__(self) -> None:
        self.policy = _Resettable()
        self.preprocessor = _Preprocessor()
        self.postprocessor = _Postprocessor()
        self.selected_indices = None
        self.processed = None
        self.native_predictions = 0
        self.observation_key_map = {}

    def reset(self) -> None:
        for component in (self.policy, self.preprocessor, self.postprocessor):
            component.reset()

    def sample_flow_sde(self, processed, *, selected_index):
        self.processed = processed
        self.selected_indices = selected_index.clone()
        batch_size = int(processed["observation.state"].shape[0])
        inputs = PIFlowModelInputs(
            images=(torch.zeros(batch_size, 3, 2, 2),),
            image_masks=(torch.ones(batch_size, dtype=torch.bool),),
            language_tokens=torch.ones(batch_size, 2, dtype=torch.long),
            language_masks=torch.ones(batch_size, 2, dtype=torch.bool),
            state=processed["observation.state"],
        )
        return PIFlowSDERollout(
            actions=torch.tensor([[[1.0], [2.0], [9.0]]]).repeat(
                batch_size, 1, 1
            ),
            transition=FlowSDETransitionRecord(
                previous_states=torch.zeros(batch_size, 2, 1),
                next_states=torch.ones(batch_size, 2, 1),
                selected_indices=selected_index,
                old_logprobs=torch.tensor([[[-0.2], [-0.3]]]).repeat(
                    batch_size, 1, 1
                ),
            ),
            inputs=inputs,
        )

    def predict_native_action_chunk(self, processed):
        self.native_predictions += 1
        batch_size = int(processed["observation.state"].shape[0])
        return torch.tensor([[[3.0], [4.0], [9.0]]]).repeat(batch_size, 1, 1)


def test_pi_batch_postprocessor_restores_each_rows_reference_state() -> None:
    class RelativeStep:
        _last_state = None

    class AbsoluteStep:
        enabled = True
        relative_step = RelativeStep()

    class Postprocessor:
        steps = [AbsoluteStep()]

        def __call__(self, actions):
            return actions + self.steps[0].relative_step._last_state[:, None, :]

    policy = type(
        "RelativePolicy",
        (),
        {
            "config": type("Config", (), {"use_relative_actions": True})(),
            "postprocessor": Postprocessor(),
        },
    )()
    actions = torch.zeros(2, 3, 2)

    result = _postprocess_pi_actions(
        policy,
        actions,
        prepared_rows=[
            {"observation.state": torch.tensor([[1.0, 2.0]])},
            {"observation.state": torch.tensor([[3.0, 4.0]])},
        ],
    )

    torch.testing.assert_close(
        result,
        torch.tensor(
            [
                [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],
                [[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]],
            ]
        ),
    )


def _prepare(observation, device, task, robot_type):
    assert device == torch.device("cpu")
    assert robot_type == "panda"
    return {
        "observation.state": torch.from_numpy(observation["observation.state"])[None],
        "task": task,
    }


def test_pi_flow_sde_adapter_preserves_processor_and_probability_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeds = []
    monkeypatch.setattr(
        "art_embodied.integrations.pi_flow_sde.random.randint",
        lambda lower, upper: 2,
    )
    policy = _Policy()
    adapter = PIFlowSDEPolicyAdapter(
        policy=policy,
        robot_type="panda",
        seed_fn=seeds.append,
        prepare_observation_fn=_prepare,
    )

    adapter.reset(seed=17)
    prediction = adapter.predict(
        {"observation.state": np.array([0.5], dtype=np.float32)},
        task="move block",
        step=4,
        seed=999,
    )

    assert seeds == [17]
    assert policy.policy.reset_count == 1
    torch.testing.assert_close(
        prediction.native_action,
        torch.tensor([[2.0], [4.0]]),
    )
    assert policy.postprocessor.received_horizons == [3]
    assert prediction.action.kind == "continuous"
    assert prediction.action.step == 4
    assert prediction.action.logprobs["chunk_logprob"] == pytest.approx(-0.5)
    assert (
        prediction.action.metadata[TRANSIENT_FLOW_SDE_ROLLOUT_KEY].inputs.batch_size
        == 1
    )
    assert 0 <= prediction.action.metadata["selected_denoise_index"] < 3
    assert prediction.action.metadata["selected_denoise_index"] == 2
    torch.testing.assert_close(policy.selected_indices, torch.tensor([2]))
    assert prediction.action.metadata["task"] == "move block"


def test_pi_flow_sde_adapter_rejects_unloaded_processor_pipeline() -> None:
    policy = _Policy()
    policy.preprocessor = None

    with pytest.raises(RuntimeError, match="processor pipelines"):
        PIFlowSDEPolicyAdapter(policy=policy)


def test_pi_flow_sde_adapter_uses_native_ode_for_evaluation() -> None:
    policy = _Policy()
    adapter = PIFlowSDEPolicyAdapter(
        policy=policy,
        robot_type="panda",
        prepare_observation_fn=_prepare,
        sampling_mode="eval",
    )

    prediction = adapter.predict(
        {"observation.state": np.array([0.5], dtype=np.float32)},
        task="move block",
        step=2,
    )

    assert policy.native_predictions == 1
    assert policy.selected_indices is None
    torch.testing.assert_close(
        prediction.native_action,
        torch.tensor([[6.0], [8.0]]),
    )
    assert policy.postprocessor.received_horizons == [3]
    assert prediction.action.logprobs is None
    assert prediction.action.metadata["probability_model"] == "native_flow_ode"
    assert TRANSIENT_FLOW_SDE_ROLLOUT_KEY not in prediction.action.metadata


def test_pi_flow_sde_adapter_batches_counterfactual_siblings_with_shared_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "art_embodied.integrations.pi_flow_sde.random.randint",
        lambda lower, upper: 2,
    )
    policy = _Policy()
    adapter = PIFlowSDEPolicyAdapter(
        policy=policy,
        robot_type="panda",
        prepare_observation_fn=_prepare,
    )

    predictions = adapter.predict_batch(
        [
            {"observation.state": np.array([0.5], dtype=np.float32)},
            {"observation.state": np.array([1.5], dtype=np.float32)},
        ],
        tasks=["move block", "move block"],
        step=3,
    )

    assert len(predictions) == 2
    torch.testing.assert_close(policy.selected_indices, torch.tensor([2, 2]))
    torch.testing.assert_close(policy.processed["next.reward"], torch.zeros(2))
    assert policy.processed["action"] is None
    assert policy.processed["robot_type"] == "panda"
    for index, prediction in enumerate(predictions):
        torch.testing.assert_close(
            prediction.native_action,
            torch.tensor([[2.0], [4.0]]),
        )
        assert prediction.predicted_action_chunk.shape[0] == 3
        assert prediction.execution_horizon == 2
        assert prediction.action.metadata["model_batch_size"] == 2
        assert prediction.action.metadata["selected_denoise_index"] == 2
        retained = prediction.action.metadata[TRANSIENT_FLOW_SDE_ROLLOUT_KEY]
        assert retained.inputs.batch_size == 1
        torch.testing.assert_close(
            retained.inputs.state,
            torch.tensor([[0.5 + index]], dtype=torch.float32),
        )
    assert policy.postprocessor.received_horizons == [3]


def test_pi_flow_sde_adapter_batches_relative_actions_against_each_row_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RelativeStep:
        _last_state = None

    class AbsoluteStep:
        enabled = True
        relative_step = RelativeStep()

    class StatefulPostprocessor(_Resettable):
        steps = [AbsoluteStep()]

        def __call__(self, actions):
            return actions + self.steps[0].relative_step._last_state[:, None, :]

    class StatefulPreprocessor(_Preprocessor):
        def __call__(self, batch):
            StatefulPostprocessor.steps[0].relative_step._last_state = batch[
                "observation.state"
            ]
            return super().__call__(batch)

    monkeypatch.setattr(
        "art_embodied.integrations.pi_flow_sde.random.randint",
        lambda lower, upper: 2,
    )
    policy = _Policy()
    policy.family = "pi0"
    policy.config = type("Config", (), {"use_relative_actions": True})()
    policy.preprocessor = StatefulPreprocessor()
    policy.postprocessor = StatefulPostprocessor()
    adapter = PIFlowSDEPolicyAdapter(
        policy=policy,
        robot_type="panda",
        prepare_observation_fn=_prepare,
    )

    predictions = adapter.predict_batch(
        [
            {"observation.state": np.array([0.5], dtype=np.float32)},
            {"observation.state": np.array([1.5], dtype=np.float32)},
        ],
        tasks=["move block", "move block"],
        step=3,
    )

    torch.testing.assert_close(
        predictions[0].native_action,
        torch.tensor([[1.5], [2.5]]),
    )
    torch.testing.assert_close(
        predictions[1].native_action,
        torch.tensor([[2.5], [3.5]]),
    )


def test_pi_flow_sde_adapter_accepts_explicit_per_sample_steps() -> None:
    policy = _Policy()
    adapter = PIFlowSDEPolicyAdapter(
        policy=policy,
        robot_type="panda",
        prepare_observation_fn=_prepare,
    )

    predictions = adapter.predict_batch(
        [
            {"observation.state": np.array([0.5], dtype=np.float32)},
            {"observation.state": np.array([1.5], dtype=np.float32)},
        ],
        tasks=["move block", "move block"],
        step=3,
        selected_indices=[0, 2],
    )

    torch.testing.assert_close(policy.selected_indices, torch.tensor([0, 2]))
    assert [
        prediction.action.metadata["selected_denoise_index"]
        for prediction in predictions
    ] == [0, 2]


def test_pi_observation_mapping_pads_state_before_lerobot_processor(
    monkeypatch,
) -> None:
    captured = {}

    def prepare(observation, device, task, robot_type):
        captured.update(
            observation=observation,
            device=device,
            task=task,
            robot_type=robot_type,
        )
        return observation

    monkeypatch.setattr(
        "art_embodied.integrations.pi_flow_sde._load_prepare_observation",
        lambda: prepare,
    )
    policy = _Policy()
    policy.config = type("Config", (), {"max_state_dim": 4})()
    policy.observation_key_map = {
        "image": "observation.images.image",
        "proprio_state": "observation.state",
    }

    result = _prepare_pi_observation(
        {
            "image": np.zeros((2, 2, 3), dtype=np.uint8),
            "proprio_state": np.array([1.0, 2.0], dtype=np.float32),
        },
        policy=policy,
        task="pick",
        robot_type="panda",
    )

    np.testing.assert_array_equal(
        result["observation.state"],
        np.array([1.0, 2.0, 0.0, 0.0], dtype=np.float32),
    )
    assert "observation.images.image" in result
    assert captured["task"] == "pick"


def test_pi_observation_mapping_contiguously_preserves_rotated_image(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "art_embodied.integrations.pi_flow_sde._load_prepare_observation",
        lambda: (lambda observation, device, task, robot_type: observation),
    )
    policy = _Policy()
    policy.config = type("Config", (), {"max_state_dim": 4})()
    policy.observation_key_map = {"image": "observation.images.image"}
    source = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    rotated = source[::-1, ::-1]
    assert not rotated.flags.c_contiguous

    result = _prepare_pi_observation(
        {"image": rotated},
        policy=policy,
        task="pick",
        robot_type="panda",
    )

    mapped = result["observation.images.image"]
    assert mapped.flags.c_contiguous
    np.testing.assert_array_equal(mapped, rotated)


def test_pi_observation_mapping_uses_openpi_resize_for_imported_checkpoint(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "art_embodied.integrations.pi_flow_sde._load_prepare_observation",
        lambda: (lambda observation, device, task, robot_type: observation),
    )
    policy = _Policy()
    policy.model_format = "rlinf_openpi_safetensors"
    policy.config = type(
        "Config",
        (),
        {
            "max_state_dim": 4,
            "image_resolution": (4, 4),
            "image_features": {"observation.images.image": object()},
        },
    )()
    policy.observation_key_map = {"image": "observation.images.image"}

    result = _prepare_pi_observation(
        {"image": np.full((2, 4, 3), 255, dtype=np.uint8)},
        policy=policy,
        task="pick",
        robot_type="panda",
    )

    mapped = result["observation.images.image"]
    assert mapped.shape == (4, 4, 3)
    np.testing.assert_array_equal(mapped[0], np.zeros((4, 3), dtype=np.uint8))
    np.testing.assert_array_equal(
        mapped[1:3], np.full((2, 4, 3), 255, dtype=np.uint8)
    )
