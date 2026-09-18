from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from art_embodied.backends.flow_sde import (  # noqa: E402
    FLOW_SDE_ACTION_DIMENSION_MASK_KEY,
    TRANSIENT_FLOW_SDE_ROLLOUT_KEY,
)
from art_embodied.config import EmbodiedExperimentConfig  # noqa: E402
from art_embodied.integrations.gr00t_flow_sde import (  # noqa: E402
    GR00TN17FlowSDEPolicyAdapter,
    _droid_action_batch,
    _droid_n1d7_observation_batch,
    _libero_n1d7_observation_batch,
    _robocasa_gr1_action_batch,
    _robocasa_gr1_n1d7_observation_batch,
)
from art_embodied.policies.factory import (  # noqa: E402
    make_gr00t_n1d7_flow_policy,
    policy_capabilities,
)
from art_embodied.policies.flow_policy import (  # noqa: E402
    FlowSDERollout,
    GR00TFlowModelInputs,
)
from art_embodied.policies.flow_sde import (  # noqa: E402
    FlowSDESchedule,
    FlowSDETransitionRecord,
)
from art_embodied.policies.gr00t_n1d7 import (  # noqa: E402
    GR00TN17FlowPolicy,
    _resolve_checkpoint_path,
)
from art_embodied.policies.gr00t_n1d7_flow_sde import (  # noqa: E402
    GR00TN17FlowSDEBridge,
)


class _IdentityStateEncoder(torch.nn.Module):
    def forward(self, state, embodiment_id):
        del embodiment_id
        return state[..., :7]


def test_n1d7_load_checks_mask_compatibility_before_loading_model(monkeypatch) -> None:
    from art_embodied.policies import transformers_compat

    class MaskCheckReached(Exception):
        pass

    def check_mask():
        raise MaskCheckReached

    monkeypatch.setattr(
        transformers_compat, "ensure_art_mask_patch_compatibility", check_mask
    )
    policy = _partitioned_checkpoint_policy()
    policy.native_policy = None
    with pytest.raises(MaskCheckReached):
        policy.load()
    # A loaded policy does not re-run process-global compatibility patches.
    policy.native_policy = object()
    policy.load()


class _ActionEncoder(torch.nn.Module):
    def forward(self, actions, timestep, embodiment_id):
        del timestep, embodiment_id
        return actions


class _DiT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, *, hidden_states, encoder_hidden_states, timestep, **kwargs):
        del encoder_hidden_states, timestep
        assert "image_mask" in kwargs
        assert "backbone_attention_mask" in kwargs
        return hidden_states * self.scale


class _Decoder(torch.nn.Module):
    def forward(self, hidden_states, embodiment_id):
        del embodiment_id
        return hidden_states


class _N17Head(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            action_horizon=16,
            max_action_dim=7,
            add_pos_embed=False,
            use_alternate_vl_dit=True,
        )
        self.action_dim = self.config.max_action_dim
        self.num_timestep_buckets = 1000
        self.state_encoder = _IdentityStateEncoder()
        self.action_encoder = _ActionEncoder()
        self.model = _DiT()
        self.action_decoder = _Decoder()


def _flow_inputs(batch_size: int = 2) -> GR00TFlowModelInputs:
    return GR00TFlowModelInputs(
        vision_language_features=torch.zeros(batch_size, 3, 7),
        vision_language_attention_mask=torch.ones(batch_size, 3, dtype=torch.bool),
        state=torch.zeros(batch_size, 1, 64),
        embodiment_id=torch.zeros(batch_size, dtype=torch.long),
        image_mask=torch.ones(batch_size, 3, dtype=torch.bool),
        model_family="gr00t_n1d7",
    )


def test_n1d7_bridge_scores_only_the_executed_prefix() -> None:
    bridge = GR00TN17FlowSDEBridge(
        _N17Head(),
        schedule=FlowSDESchedule(
            num_steps=4,
            noise_level=0.3,
            noise_time="zero",
        ),
        execution_horizon=8,
        action_dim=7,
    )
    rollout = bridge.sample(
        _flow_inputs(),
        selected_index=torch.tensor([1, 2]),
        generator=torch.Generator().manual_seed(7),
    )
    rescored = bridge.rescore(rollout)

    assert rollout.actions.shape == (2, 16, 7)
    assert rollout.batch_signature()[0][0] == "gr00t_n1d7_backbone"
    assert rescored.shape == (2, 8, 7)
    torch.testing.assert_close(
        rescored,
        rollout.transition.old_logprobs[:, :8, :7],
        rtol=0,
        atol=0,
    )
    (-rescored.mean()).backward()
    assert bridge.action_head.model.scale.grad is not None


class _AdapterPolicy:
    family = "gr00t_n1d7"
    device = "cpu"
    execution_horizon = 8
    processor_action_horizon = 16
    model_action_horizon = 40
    action_dim = 7
    schedule = FlowSDESchedule(num_steps=4, noise_level=0.3, noise_time="zero")

    def __init__(self) -> None:
        self.native_policy = SimpleNamespace(model=torch.nn.Linear(1, 1))
        self.model = self.native_policy.model

    def prepare_flow_inputs(self, native_input):
        return _flow_inputs(len(native_input["video"]["image"]))

    def sample_flow_sde(self, inputs, *, selected_index, **kwargs):
        del kwargs
        batch = inputs.batch_size
        return FlowSDERollout(
            actions=torch.zeros(batch, 40, 7),
            transition=FlowSDETransitionRecord(
                previous_states=torch.zeros(batch, 40, 7),
                next_states=torch.ones(batch, 40, 7),
                selected_indices=selected_index,
                old_logprobs=torch.full((batch, 40, 7), -0.25),
            ),
            inputs=inputs,
        )

    def decode_action_transforms(self, actions, native_input):
        del native_input
        array = actions.detach().cpu().numpy()[:, :16]
        keys = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
        return {key: array[..., i : i + 1] for i, key in enumerate(keys)}

    def predict_native_action_chunk(self, native_input):
        batch = len(native_input["video"]["image"])
        array = np.zeros((batch, 16, 7), dtype=np.float32)
        keys = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
        return {key: array[..., i : i + 1] for i, key in enumerate(keys)}


def _observation() -> dict[str, np.ndarray]:
    return {
        "image": np.zeros((8, 8, 3), dtype=np.uint8),
        "wrist_image": np.ones((8, 8, 3), dtype=np.uint8),
        "proprio_state": np.arange(8, dtype=np.float32),
    }


def test_n1d7_adapter_preserves_three_distinct_horizons() -> None:
    policy = _AdapterPolicy()
    prediction = GR00TN17FlowSDEPolicyAdapter(policy=policy).predict(
        _observation(), task="move block", step=3, seed=5
    )

    assert prediction.native_action.shape == (8, 7)
    assert prediction.predicted_action_chunk.shape == (16, 7)
    assert np.asarray(prediction.action.decoded).shape == (8, 7)
    assert prediction.action.logprobs["chunk_logprob"] == pytest.approx(-14.0)
    assert prediction.action.metadata["model_action_horizon"] == 40
    assert prediction.action.metadata["processor_action_horizon"] == 16
    retained = prediction.action.metadata[TRANSIENT_FLOW_SDE_ROLLOUT_KEY]
    assert retained.transition.old_logprobs.shape == (1, 40, 7)


def test_n1d7_stochastic_evaluation_drops_training_payload() -> None:
    prediction = GR00TN17FlowSDEPolicyAdapter(
        policy=_AdapterPolicy(), sampling_mode="stochastic_eval"
    ).predict(_observation(), task="move block", step=3, seed=5)

    assert prediction.action.metadata["probability_model"] == "gaussian_flow_sde"
    assert "selected_denoise_index" in prediction.action.metadata
    assert TRANSIENT_FLOW_SDE_ROLLOUT_KEY not in prediction.action.metadata
    assert prediction.action.logprobs is None


def test_n1d7_libero_mapping_uses_native_nested_processor_schema() -> None:
    mapped = _libero_n1d7_observation_batch(
        [_observation(), _observation()], ["first", "second"]
    )

    assert mapped["video"]["image"].shape == (2, 1, 8, 8, 3)
    assert mapped["state"]["gripper"].shape == (2, 1, 2)
    assert mapped["state"]["x"].dtype == np.float32
    assert mapped["language"]["annotation.human.action.task_description"] == [
        ["first"],
        ["second"],
    ]


def _droid_observation() -> dict[str, np.ndarray]:
    return {
        "external_image": np.zeros((180, 320, 3), dtype=np.uint8),
        "wrist_image": np.ones((180, 320, 3), dtype=np.uint8),
        "eef_9d": np.arange(9, dtype=np.float32),
        "gripper_position": np.asarray([0.25], dtype=np.float32),
        "joint_position": np.arange(7, dtype=np.float32),
    }


def test_n1d7_droid_mapping_matches_official_processor_schema() -> None:
    mapped = _droid_n1d7_observation_batch(
        [_droid_observation(), _droid_observation()], ["first", "second"]
    )

    assert mapped["video"]["exterior_image_1_left"].shape == (2, 1, 180, 320, 3)
    assert mapped["state"]["eef_9d"].shape == (2, 1, 9)
    assert mapped["state"]["gripper_position"].shape == (2, 1, 1)
    assert mapped["state"]["joint_position"].shape == (2, 1, 7)
    assert mapped["language"]["annotation.language.language_instruction"] == [
        ["first"],
        ["second"],
    ]


def _robocasa_observation() -> dict[str, object]:
    return {
        "video.ego_view_pad_res256_freq20": np.zeros((256, 256, 3), dtype=np.uint8),
        "video.ego_view_bg_crop_pad_res256_freq20": np.ones(
            (256, 256, 3), dtype=np.uint8
        ),
        "state.left_arm": np.arange(7, dtype=np.float32),
        "state.left_hand": np.arange(6, dtype=np.float32),
        "state.right_arm": np.arange(7, dtype=np.float32),
        "state.right_hand": np.arange(6, dtype=np.float32),
        "state.waist": np.arange(3, dtype=np.float32),
        "annotation.human.coarse_action": "unlocked_waist: move the bottle",
    }


def test_n1d7_robocasa_mapping_matches_official_gym_wrapper_schema() -> None:
    mapped = _robocasa_gr1_n1d7_observation_batch(
        [_robocasa_observation(), _robocasa_observation()]
    )

    assert mapped["video"]["ego_view_bg_crop_pad_res256_freq20"].shape == (
        2,
        1,
        256,
        256,
        3,
    )
    assert mapped["state"]["left_arm"].shape == (2, 1, 7)
    assert mapped["state"]["waist"].shape == (2, 1, 3)
    assert set(mapped["state"]) == {
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
        "waist",
    }
    assert mapped["language"]["task"] == [
        ["unlocked_waist: move the bottle"],
        ["unlocked_waist: move the bottle"],
    ]


def test_n1d7_robocasa_action_layout_executes_and_scores_all_29_dimensions() -> None:
    sizes = (7, 7, 6, 6, 3)
    names = (
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
        "waist",
    )
    components = {
        f"action.{name}": np.full((2, 8, size), index, dtype=np.float32)
        for index, (name, size) in enumerate(zip(names, sizes, strict=True))
    }
    layout = tuple(
        (name, size, True, index)
        for index, (name, size) in enumerate(zip(names, sizes, strict=True))
    )

    chunks, dimension_mask = _robocasa_gr1_action_batch(
        components,
        action_components_layout=layout,
    )

    assert chunks.shape == (2, 8, 29)
    assert dimension_mask == (True,) * 29
    np.testing.assert_array_equal(chunks[0, 0, :7], np.zeros(7))
    np.testing.assert_array_equal(chunks[0, 0, -3:], np.full(3, 4))


def test_n1d7_droid_action_layout_masks_eef_and_executes_joint_then_gripper() -> None:
    components = {
        "action.eef_9d": np.zeros((2, 40, 9), dtype=np.float32),
        "action.gripper_position": np.full((2, 40, 1), 0.75, dtype=np.float32),
        "action.joint_position": np.broadcast_to(
            np.arange(7, dtype=np.float32), (2, 40, 7)
        ),
    }
    chunks, dimension_mask = _droid_action_batch(
        components,
        action_components_layout=(
            ("eef_9d", 9, False, None),
            ("gripper_position", 1, True, 1),
            ("joint_position", 7, True, 0),
        ),
    )

    assert chunks.shape == (2, 40, 8)
    np.testing.assert_array_equal(chunks[0, 0, :7], np.arange(7))
    assert chunks[0, 0, 7] == 1.0
    assert dimension_mask == (False,) * 9 + (True,) * 8


class _DroidAdapterPolicy(_AdapterPolicy):
    processor_action_horizon = 40
    action_dim = 17
    action_components = (
        ("eef_9d", 9, False, None),
        ("gripper_position", 1, True, 1),
        ("joint_position", 7, True, 0),
    )

    def prepare_flow_inputs(self, native_input):
        return _flow_inputs(len(native_input["video"]["exterior_image_1_left"]))

    def sample_flow_sde(self, inputs, *, selected_index, **kwargs):
        del kwargs
        batch = inputs.batch_size
        return FlowSDERollout(
            actions=torch.zeros(batch, 40, 17),
            transition=FlowSDETransitionRecord(
                previous_states=torch.zeros(batch, 40, 17),
                next_states=torch.ones(batch, 40, 17),
                selected_indices=selected_index,
                old_logprobs=torch.full((batch, 40, 17), -0.25),
            ),
            inputs=inputs,
        )

    def decode_action_transforms(self, actions, native_input):
        del native_input
        array = actions.detach().cpu().numpy()
        return {
            "eef_9d": array[..., :9],
            "gripper_position": np.full((*array.shape[:2], 1), 0.75, np.float32),
            "joint_position": array[..., 10:17],
        }


def test_n1d7_droid_adapter_credits_only_executed_action_dimensions() -> None:
    prediction = GR00TN17FlowSDEPolicyAdapter(
        policy=_DroidAdapterPolicy(), runtime_profile="droid"
    ).predict(_droid_observation(), task="cube then banana", step=0, seed=5)

    assert prediction.native_action.shape == (8, 8)
    assert prediction.predicted_action_chunk.shape == (40, 8)
    assert prediction.action.logprobs["chunk_logprob"] == pytest.approx(-16.0)
    assert (
        prediction.action.metadata[FLOW_SDE_ACTION_DIMENSION_MASK_KEY]
        == [False] * 9 + [True] * 8
    )
    retained = prediction.action.metadata[TRANSIENT_FLOW_SDE_ROLLOUT_KEY]
    assert retained.transition.old_logprobs.shape == (1, 40, 17)


def test_n1d7_positive_control_declares_distinct_horizons() -> None:
    config = EmbodiedExperimentConfig.from_yaml(
        "examples/embodied/"
        "gr00t_n1d7_libero_spatial_flow_sde_grpo_positive_control.yaml"
    )
    policy = make_gr00t_n1d7_flow_policy(config, load=False)
    capabilities = policy_capabilities(config)

    assert policy.model_action_horizon == 40
    assert policy.processor_action_horizon == 16
    assert policy.execution_horizon == 8
    assert capabilities.action_shape == (8, 7)


def test_n1d7_contract_rejects_execution_beyond_processor_horizon() -> None:
    config = EmbodiedExperimentConfig.from_yaml(
        "examples/embodied/"
        "gr00t_n1d7_libero_spatial_flow_sde_grpo_positive_control.yaml"
    )
    raw = config.model_dump(mode="python")
    raw["policy"]["load_kwargs"]["execution_horizon"] = 17
    raw["environment"]["kwargs"]["action_chunk_size"] = 17
    raw["training"]["schedule"]["action_chunk_size"] = 17

    with pytest.raises(ValueError, match="processor_action_horizon"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_n1d7_root_checkpoint_downloads_the_complete_repository(tmp_path) -> None:
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    resolved = _resolve_checkpoint_path(
        model_id="nvidia/GR00T-N1.7-DROID",
        revision="exact-sha",
        checkpoint_subfolder=".",
        snapshot_download=download,
    )

    assert resolved == tmp_path
    assert calls == [
        {
            "repo_id": "nvidia/GR00T-N1.7-DROID",
            "revision": "exact-sha",
            "repo_type": "model",
        }
    ]


class _FakePartitionedPeftModel:
    def __init__(self) -> None:
        self.peft_config = {"task_000": object(), "task_001": object()}
        self.base_model = SimpleNamespace(set_adapter=self._set_adapter)
        self.active_adapters: list[str] = []

    def _set_adapter(self, adapters: list[str]) -> None:
        self.active_adapters = list(adapters)

    def save_pretrained(self, destination: Path) -> None:
        for adapter_name in self.peft_config:
            (destination / adapter_name).mkdir(parents=True)


def _partitioned_checkpoint_policy() -> GR00TN17FlowPolicy:
    policy = GR00TN17FlowPolicy(
        model_id="test/gr00t-n1.7",
        revision="exact-revision",
        checkpoint_subfolder=".",
        device="cpu",
        embodiment_tag="ROBOCASA_GR1_TABLETOP",
        execution_horizon=8,
        processor_action_horizon=8,
        action_dim=29,
        action_components=(("actions", 29, True, 0),),
        model_action_horizon=40,
        schedule=FlowSDESchedule(num_steps=4, noise_level=0.5),
        disable_dropout=True,
    )
    policy.native_policy = SimpleNamespace(model=_FakePartitionedPeftModel())
    return policy


def test_n1d7_partitioned_adapter_checkpoint_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _partitioned_checkpoint_policy()
    policy.save_checkpoint(str(tmp_path))

    metadata = json.loads(
        (tmp_path / "art_embodied_gr00t_n1d7_snapshot.json").read_text()
    )
    assert metadata["adapter_names"] == ["task_000", "task_001"]

    loaded_paths: list[str] = []
    loaded_adapters: list[str] = []
    save_and_load = ModuleType("peft.utils.save_and_load")

    def load_peft_weights(path: str, *, device: str):
        assert device == "cpu"
        loaded_paths.append(path)
        return {"path": path}

    def set_peft_model_state_dict(model, state, *, adapter_name: str):
        assert model is policy.model
        assert state["path"].endswith(adapter_name)
        loaded_adapters.append(adapter_name)
        return SimpleNamespace(unexpected_keys=[])

    save_and_load.load_peft_weights = load_peft_weights
    save_and_load.set_peft_model_state_dict = set_peft_model_state_dict
    peft = ModuleType("peft")
    peft_utils = ModuleType("peft.utils")
    peft.utils = peft_utils
    peft_utils.save_and_load = save_and_load
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "peft.utils", peft_utils)
    monkeypatch.setitem(sys.modules, "peft.utils.save_and_load", save_and_load)

    policy.load_checkpoint(tmp_path)

    assert loaded_paths == [
        str(tmp_path / "task_000"),
        str(tmp_path / "task_001"),
    ]
    assert loaded_adapters == ["task_000", "task_001"]
    assert policy.model.active_adapters == ["task_000", "task_001"]


def test_n1d7_partitioned_adapter_checkpoint_rejects_geometry_mismatch(
    tmp_path: Path,
) -> None:
    policy = _partitioned_checkpoint_policy()
    policy.save_checkpoint(str(tmp_path))
    metadata_path = tmp_path / "art_embodied_gr00t_n1d7_snapshot.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["adapter_names"] = ["task_001", "task_000"]
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(RuntimeError, match="snapshot geometry mismatch"):
        policy.load_checkpoint(tmp_path)


def test_n1d7_reference_rescore_disables_adapters_and_detaches() -> None:
    class ReferenceModel:
        adapter_enabled = True

        @contextmanager
        def disable_adapter(self):
            assert self.adapter_enabled
            self.adapter_enabled = False
            try:
                yield
            finally:
                self.adapter_enabled = True

    class ReferenceBridge:
        def __init__(self, model: ReferenceModel) -> None:
            self.model = model

        def rescore(self, rollout) -> torch.Tensor:
            del rollout
            value = 1.0 if self.model.adapter_enabled else -1.0
            return torch.tensor([[[value]]], requires_grad=True)

    policy = _partitioned_checkpoint_policy()
    model = ReferenceModel()
    policy.native_policy = SimpleNamespace(model=model)
    policy.bridge = ReferenceBridge(model)

    score = policy.flow_sde_reference_logprobs(object())

    torch.testing.assert_close(score, torch.tensor([[[-1.0]]]))
    assert score.requires_grad is False
    assert model.adapter_enabled is True


def test_n1d7_reference_rescore_requires_peft_disable_adapter() -> None:
    policy = _partitioned_checkpoint_policy()
    policy.bridge = SimpleNamespace(rescore=lambda rollout: torch.zeros(1, 1, 1))

    with pytest.raises(RuntimeError, match="disable_adapter"):
        policy.flow_sde_reference_logprobs(object())
