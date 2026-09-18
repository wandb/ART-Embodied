from __future__ import annotations

from pathlib import Path
import random

import pytest

from art_embodied.config import EmbodiedExperimentConfig
from art_embodied.policies import factory as policy_factories
from art_embodied.policies.factory import (
    make_gr00t_n1d5_flow_policy,
    make_openvla_oft_policy,
    make_pi0_fast_policy,
    make_pi_flow_policy,
    make_policy,
    make_smolvla_flow_policy,
    policy_capabilities,
    registered_policy_types,
)
from art_embodied.vla_trainable import _full_language_model_lora_targets

PI05_CONFIG = (
    Path(__file__).parents[1]
    / "examples/embodied/pi05_libero_object_flow_sde_grpo_rlinf_positive_control.yaml"
)
SMOLVLA_CONFIG = (
    Path(__file__).parents[1]
    / "examples/embodied/smolvla_libero_object_flow_sde_grpo_smoke_h100.yaml"
)
PI0_FAST_CONFIG = (
    Path(__file__).parents[1]
    / "examples/embodied/pi0_fast_libero_spatial_grpo_development_h100.yaml"
)


def _config() -> EmbodiedExperimentConfig:
    return EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml"
    )


def _pi05_config() -> EmbodiedExperimentConfig:
    raw = EmbodiedExperimentConfig.from_yaml(PI05_CONFIG).model_dump(mode="python")
    raw["policy"]["path"] = "lerobot/pi05_libero_finetuned_v044"
    raw["policy"]["revision"] = "test-revision"
    raw["policy"]["dtype"] = "bfloat16"
    raw["policy"]["load_kwargs"] = {
        "runtime_contract": "lerobot_v060",
        "model_format": "lerobot",
        "execution_horizon": 5,
        "action_dim": 7,
        "processor_path": None,
        "processor_revision": None,
        "model_chunk_size": None,
        "normalization_stats_file": None,
        "discrete_state_input": None,
        "extra_delta_transform": None,
        "observation_key_map": {},
        "strict_weights": True,
        "compile_model": False,
        "gradient_checkpointing": False,
        "train_expert_only": True,
    }
    raw["policy"]["trainable_parameter_strategy"] = "pi05_action_expert_lora"
    raw["policy"]["lora"]["target_modules"] = ["auto"]
    raw["rollout"]["action_payload"]["kind"] = "continuous"
    raw["algorithm"]["flow_sde"] = {
        "noise_level": 0.3,
        "num_denoise_steps": 5,
        "stochastic_transitions_per_sample": 1,
        "selected_step_sampling": "uniform",
        "joint_logprob": False,
    }
    return EmbodiedExperimentConfig.model_validate(raw)


def _smolvla_config() -> EmbodiedExperimentConfig:
    return EmbodiedExperimentConfig.from_yaml(SMOLVLA_CONFIG)


def _pi0_fast_config() -> EmbodiedExperimentConfig:
    return EmbodiedExperimentConfig.from_yaml(PI0_FAST_CONFIG)


def test_pi0_fast_full_language_model_lora_target_is_narrowly_scoped() -> None:
    target = _full_language_model_lora_targets("pi0_fast_lora", "pi0_fast")

    assert "language_model" in target
    assert "q_proj|k_proj|v_proj|o_proj" in target
    assert "gate_proj|up_proj|down_proj" in target
    assert "vision_tower" not in target


def test_full_language_model_lora_target_rejects_other_policy_families() -> None:
    with pytest.raises(ValueError, match="only supported for the pi0-FAST"):
        _full_language_model_lora_targets("openvla_oft_lora", "openvla_oft")


def _gr00t_n1d5_config() -> EmbodiedExperimentConfig:
    raw = _pi05_config().model_dump(mode="python")
    raw["policy"]["type"] = "gr00t_n1d5"
    raw["policy"]["path"] = "RLinf/RLinf-GR00T-SFT-Spatial"
    raw["policy"]["revision"] = "test-revision"
    raw["policy"]["load_kwargs"] = {
        "runtime_contract": "nvidia_n1d5",
        "model_format": "nvidia_n1d5",
        "embodiment_tag": "libero_franka",
        "data_config": "art_embodied.integrations.gr00t_n1d5:LiberoFrankaDataConfig",
        "execution_horizon": 5,
        "action_dim": 7,
        "model_action_horizon": 16,
        "language_padding_length": 570,
        "disable_dropout": True,
        "compile_model": False,
        "gradient_checkpointing": False,
        "train_expert_only": True,
    }
    raw["policy"]["trainable_parameter_strategy"] = "gr00t_n1d5_action_head_lora"
    return EmbodiedExperimentConfig.model_validate(raw)


def test_openvla_factory_maps_every_runtime_condition_from_yaml() -> None:
    policy = make_openvla_oft_policy(_config(), load=False)

    assert policy.model_id == "Haozhan72/Openvla-oft-SFT-libero-object-traj1"
    assert policy.revision == "62e5a8daba3f619c993f23b248ae04b0d9677bb5"
    assert policy.model_loader == "native"
    assert policy.robot_platform == "libero"
    assert policy.action_output_kind == "token"
    assert policy.capture_action_tokens is True
    assert policy.do_sample is True
    assert policy.temperature == 1.6
    assert policy.logprob_batch_size == 32
    assert policy.strict_batched_logprobs is True
    assert policy.max_prompt_length == 128
    assert policy.num_images_in_input == 1
    assert policy.use_proprio is False
    assert policy.action_dim == 7
    assert policy.num_action_chunks == 8


def test_make_policy_seeds_adapter_initialization_from_experiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _pi05_config()
    monkeypatch.setitem(
        policy_factories._POLICY_FACTORIES,
        "pi05",
        lambda _config, *, load: (load, random.random()),
    )

    first = make_policy(config)
    random.seed(config.experiment.seed + 1)
    second = make_policy(config)

    assert first == second


def test_openvla_declares_probability_and_action_capabilities() -> None:
    capabilities = policy_capabilities(_config())

    assert capabilities.action_kind == "token"
    assert capabilities.probability_model == "categorical_tokens"
    assert capabilities.action_shape == (8, 7)
    assert capabilities.chunk_horizon == 8
    assert capabilities.exact_logprobs is True
    assert capabilities.teacher_forced_rescore is True
    assert capabilities.rng_replay is True


def test_pi05_factory_maps_flow_sde_contract_without_loading_weights() -> None:
    pytest.importorskip("torch")

    policy = make_pi_flow_policy(_pi05_config(), load=False)

    assert policy.family == "pi05"
    assert policy.model_id == "lerobot/pi05_libero_finetuned_v044"
    assert policy.execution_horizon == 5
    assert policy.model_format == "lerobot"
    assert policy.action_dim == 7
    assert policy.schedule.num_steps == 5
    assert policy.schedule.noise_level == 0.3


def test_gr00t_n1d5_factory_declares_ascending_flow_contract() -> None:
    pytest.importorskip("torch")

    policy = make_gr00t_n1d5_flow_policy(_gr00t_n1d5_config(), load=False)
    capabilities = policy_capabilities(_gr00t_n1d5_config())

    assert policy.family == "gr00t_n1d5"
    assert policy.execution_horizon == 5
    assert policy.model_action_horizon == 16
    assert policy.schedule.num_steps == 5
    assert policy.schedule.noise_time == "zero"
    assert capabilities.action_shape == (5, 7)
    assert capabilities.probability_model == "gaussian_flow_sde"


def test_gr00t_n1d5_contract_rejects_action_horizon_drift() -> None:
    raw = _gr00t_n1d5_config().model_dump(mode="python")
    raw["policy"]["load_kwargs"]["execution_horizon"] = 17

    with pytest.raises(ValueError, match="model_action_horizon"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_pi05_factory_requires_lerobot_reference_for_rlinf_raw_weights() -> None:
    raw = _pi05_config().model_dump(mode="python")
    raw["policy"]["load_kwargs"]["model_format"] = "rlinf_openpi_safetensors"
    raw["policy"]["load_kwargs"]["processor_path"] = None

    with pytest.raises(ValueError, match="processor_path"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_pi05_factory_maps_rlinf_raw_weight_contract() -> None:
    pytest.importorskip("torch")

    raw = _pi05_config().model_dump(mode="python")
    raw["policy"]["path"] = "RLinf/RLinf-Pi05-SFT"
    raw["policy"]["load_kwargs"]["model_format"] = "rlinf_openpi_safetensors"
    raw["policy"]["load_kwargs"]["processor_path"] = (
        "lerobot/pi05_libero_finetuned_v044"
    )
    raw["policy"]["load_kwargs"]["processor_revision"] = "processor-revision"
    raw["policy"]["load_kwargs"]["model_chunk_size"] = 10
    raw["policy"]["load_kwargs"]["normalization_stats_file"] = (
        "physical-intelligence/libero/norm_stats.json"
    )
    raw["policy"]["load_kwargs"]["discrete_state_input"] = False
    raw["policy"]["load_kwargs"]["extra_delta_transform"] = False
    raw["policy"]["load_kwargs"]["observation_key_map"] = {
        "image": "observation.images.image",
        "wrist_image": "observation.images.image2",
        "proprio_state": "observation.state",
    }
    config = EmbodiedExperimentConfig.model_validate(raw)

    policy = make_pi_flow_policy(config, load=False)

    assert policy.model_id == "RLinf/RLinf-Pi05-SFT"
    assert policy.model_format == "rlinf_openpi_safetensors"
    assert policy.processor_path == "lerobot/pi05_libero_finetuned_v044"
    assert policy.processor_revision == "processor-revision"
    assert policy.model_chunk_size == 10
    assert policy.discrete_state_input is False
    assert policy.extra_delta_transform is False
    assert policy.observation_key_map["proprio_state"] == "observation.state"


def test_pi05_raw_weights_require_explicit_state_prompt_contract() -> None:
    raw = _pi05_config().model_dump(mode="python")
    raw["policy"]["load_kwargs"].update(
        {
            "model_format": "rlinf_openpi_safetensors",
            "processor_path": "lerobot/pi05_libero_finetuned_v044",
            "model_chunk_size": 10,
            "normalization_stats_file": "physical-intelligence/libero/norm_stats.json",
        }
    )

    with pytest.raises(ValueError, match="discrete_state_input"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_pi05_raw_weights_require_explicit_action_coordinate_contract() -> None:
    raw = _pi05_config().model_dump(mode="python")
    raw["policy"]["load_kwargs"].update(
        {
            "model_format": "rlinf_openpi_safetensors",
            "processor_path": "lerobot/pi05_libero_finetuned_v044",
            "model_chunk_size": 10,
            "normalization_stats_file": "physical-intelligence/libero/norm_stats.json",
            "discrete_state_input": False,
        }
    )

    with pytest.raises(ValueError, match="extra_delta_transform"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_pi05_serialized_processor_rejects_action_coordinate_override() -> None:
    raw = _pi05_config().model_dump(mode="python")
    raw["policy"]["load_kwargs"]["extra_delta_transform"] = False

    with pytest.raises(ValueError, match="serialized LeRobot processor"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_pi05_declares_sampler_aligned_continuous_capabilities() -> None:
    capabilities = policy_capabilities(_pi05_config())

    assert capabilities.action_kind == "continuous"
    assert capabilities.probability_model == "gaussian_flow_sde"
    assert capabilities.action_shape == (5, 7)
    assert capabilities.exact_logprobs is True
    assert capabilities.teacher_forced_rescore is True
    assert capabilities.rng_replay is True


def test_smolvla_factory_maps_serialized_checkpoint_contract() -> None:
    pytest.importorskip("torch")

    policy = make_smolvla_flow_policy(_smolvla_config(), load=False)

    assert policy.family == "smolvla"
    assert policy.model_id == "HuggingFaceVLA/smolvla_libero"
    assert policy.revision == "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
    assert policy.execution_horizon == 1
    assert policy.action_dim == 7
    assert policy.observation_key_map == {
        "image": "observation.images.image",
        "wrist_image": "observation.images.image2",
        "proprio_state": "observation.state",
    }
    assert policy.schedule.num_steps == 10
    assert policy.schedule.noise_level == 0.1
    assert policy.schedule.deterministic_sampler == "native_euler"


def test_smolvla_declares_sampler_aligned_continuous_capabilities() -> None:
    capabilities = policy_capabilities(_smolvla_config())

    assert capabilities.action_kind == "continuous"
    assert capabilities.probability_model == "gaussian_flow_sde"
    assert capabilities.action_shape == (1, 7)
    assert capabilities.exact_logprobs is True
    assert capabilities.teacher_forced_rescore is True


def test_pi0_fast_factory_maps_revision_pinned_token_contract() -> None:
    policy = make_pi0_fast_policy(_pi0_fast_config(), load=False)
    capabilities = policy_capabilities(_pi0_fast_config())

    assert policy.family == "pi0_fast"
    assert policy.model_id == "lerobot/pi0fast-libero"
    assert policy.revision == "840f4b503f4c09110421c33c810a85b6684fd658"
    assert policy.execution_horizon == 10
    assert policy.action_dim == 7
    assert policy.max_decoding_steps == 256
    assert policy.use_kv_cache is True
    assert policy.model_compute_dtype == "checkpoint"
    assert capabilities.action_kind == "token"
    assert capabilities.probability_model == "categorical_tokens"
    assert capabilities.action_shape == (10, 7)
    assert capabilities.exact_logprobs is True
    assert capabilities.teacher_forced_rescore is True


def test_pi0_fast_factory_maps_float32_model_compute_dtype() -> None:
    raw = _pi0_fast_config().model_dump(mode="python")
    raw["policy"]["load_kwargs"]["model_compute_dtype"] = "float32"
    config = EmbodiedExperimentConfig.model_validate(raw)

    policy = make_pi0_fast_policy(config, load=False)

    assert policy.model_compute_dtype == "float32"
    assert policy.training_logprob_mode == "kv"


@pytest.mark.parametrize("mode", ["kv", "full_sequence"])
@pytest.mark.parametrize("compute_dtype", ["checkpoint", "float32"])
def test_pi0_fast_training_scorer_is_explicit_and_differentiable(
    mode, compute_dtype, monkeypatch
):
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")

    from art_embodied.policies import pi0_fast

    raw = _pi0_fast_config().model_dump(mode="python")
    raw["policy"]["load_kwargs"].update(
        model_compute_dtype=compute_dtype, training_logprob_mode=mode
    )
    policy = make_pi0_fast_policy(
        EmbodiedExperimentConfig.model_validate(raw), load=False
    )
    policy.policy = SimpleNamespace(model=object())
    calls = []
    parameter = torch.nn.Parameter(torch.tensor(2.0))

    def scorer(name):
        def score(**kwargs):
            calls.append((name, kwargs))
            return [parameter * 3]

        return score

    monkeypatch.setattr(pi0_fast, "_pi0_fast_token_logprobs", scorer("full_sequence"))
    monkeypatch.setattr(pi0_fast, "_pi0_fast_kv_token_logprobs", scorer("kv"))
    rows = policy.processed_action_token_logprobs({}, [[1, 2]], temperature=0.2)
    rows[0].backward()
    assert parameter.grad.item() == 3
    assert calls[0][0] == mode
    assert calls[0][1]["token_rows"] == [[1, 2]]
    assert calls[0][1]["temperature"] == 0.2
    assert policy.use_kv_cache is True
    # The explicit reference remains KV even with the full-sequence option.
    policy.processed_action_token_logprobs_kv({}, [[1, 2]], temperature=0.2)
    assert calls[-1][0] == "kv"
    with pytest.raises(ValueError, match="temperature"):
        policy.processed_action_token_logprobs({}, [[1]], temperature=0)


def test_pi0_fast_full_sequence_preserves_checkpoint_precision():
    raw = _pi0_fast_config().model_dump(mode="python")
    raw["policy"]["load_kwargs"].update(
        model_compute_dtype="checkpoint", training_logprob_mode="full_sequence"
    )
    policy = make_pi0_fast_policy(
        EmbodiedExperimentConfig.model_validate(raw), load=False
    )
    assert policy.model_compute_dtype == "checkpoint"
    assert policy.training_logprob_mode == "full_sequence"


def test_openvla_factory_maps_non_libero_robot_platform() -> None:
    raw = _config().model_dump(mode="python")
    raw["policy"]["load_kwargs"]["robot_platform"] = "aloha"
    raw["policy"]["load_kwargs"]["action_dim"] = 14
    raw["policy"]["load_kwargs"]["num_action_chunks"] = 25
    raw["policy"]["unnorm_key"] = "aloha"
    config = EmbodiedExperimentConfig.model_validate(raw)

    policy = make_openvla_oft_policy(config, load=False)

    assert policy.robot_platform == "aloha"
    assert policy.action_dim == 14
    assert policy.num_action_chunks == 25


def test_openvla_factory_rejects_unknown_robot_platform() -> None:
    raw = _config().model_dump(mode="python")
    raw["policy"]["load_kwargs"]["robot_platform"] = "libero_object"
    config = EmbodiedExperimentConfig.model_validate(raw)

    with pytest.raises(ValueError, match="robot_platform"):
        make_openvla_oft_policy(config, load=False)


def test_openvla_factory_wires_revision_to_native_loader() -> None:
    raw = _config().model_dump(mode="python")
    raw["policy"]["revision"] = "0123456789abcdef"
    config = EmbodiedExperimentConfig.model_validate(raw)

    policy = make_openvla_oft_policy(config, load=False)

    assert policy.revision == "0123456789abcdef"


def test_openvla_factory_rejects_remote_revision_with_rlinf_loader() -> None:
    raw = _config().model_dump(mode="python")
    raw["policy"]["revision"] = "0123456789abcdef"
    raw["policy"]["load_kwargs"]["model_loader"] = "rlinf"
    config = EmbodiedExperimentConfig.model_validate(raw)

    with pytest.raises(ValueError, match="revision-pinned local snapshot"):
        make_openvla_oft_policy(config, load=False)


def test_openvla_factory_attaches_declared_lora_surface(
    monkeypatch,
) -> None:
    captured = {}

    def load(self):
        self.model = object()

    def configure(policy, *, algorithm_cfg, policy_type):
        captured["config"] = algorithm_cfg
        captured["policy_type"] = policy_type
        return policy, {"ok": True, "trainable_parameters": 123}

    monkeypatch.setattr("art_embodied.policies.factory.OpenVLAPolicy.load", load)
    monkeypatch.setattr(
        "art_embodied.policies.factory.raise_for_openvla_oft_v01_runtime",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "art_embodied.policies.factory.configure_vla_trainable_parameters",
        configure,
    )

    policy = make_openvla_oft_policy(_config())

    assert policy.trainable_report == {"ok": True, "trainable_parameters": 123}
    assert captured["policy_type"] == "openvla_oft"
    peft = captured["config"]["peft"]
    assert peft["enabled"] is True
    assert peft["r"] == 32
    assert peft["lora_alpha"] == 32
    assert peft["lora_dropout"] == 0.0
    assert peft["init_lora_weights"] == "gaussian"
    assert peft["apply_layer_selection"] is False
    assert "q_proj" in peft["target_modules"]


def test_openvla_factory_fails_before_load_on_runtime_drift(monkeypatch) -> None:
    loaded = False

    def load(_self):
        nonlocal loaded
        loaded = True

    monkeypatch.setattr("art_embodied.policies.factory.OpenVLAPolicy.load", load)
    monkeypatch.setattr(
        "art_embodied.policies.factory.raise_for_openvla_oft_v01_runtime",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("runtime drift")),
    )

    with pytest.raises(RuntimeError, match="runtime drift"):
        make_openvla_oft_policy(_config())

    assert loaded is False


def test_openvla_factory_rejects_unknown_loader_options() -> None:
    config = _config()
    raw = config.model_dump(mode="python")
    raw["policy"]["load_kwargs"]["silent_typo"] = True
    config = EmbodiedExperimentConfig.model_validate(raw)

    with pytest.raises(ValueError, match="silent_typo"):
        make_openvla_oft_policy(config, load=False)


@pytest.mark.parametrize("attention", ["eager", "flash_attention_2"])
def test_openvla_factory_rejects_attention_override_for_v01_contract(
    attention: str,
) -> None:
    raw = _config().model_dump(mode="python")
    raw["policy"]["load_kwargs"]["attn_implementation"] = attention
    config = EmbodiedExperimentConfig.model_validate(raw)

    with pytest.raises(ValueError, match="bidirectional action-token logits"):
        make_openvla_oft_policy(config, load=False)


def test_openvla_factory_allows_attention_override_only_when_unchecked() -> None:
    raw = _config().model_dump(mode="python")
    raw["policy"]["load_kwargs"]["runtime_contract"] = "unchecked"
    raw["policy"]["load_kwargs"]["attn_implementation"] = "flash_attention_2"
    config = EmbodiedExperimentConfig.model_validate(raw)

    policy = make_openvla_oft_policy(config, load=False)

    assert policy.attn_implementation == "flash_attention_2"


def test_openvla_factory_owns_its_top_p_capability_check() -> None:
    raw = _config().model_dump(mode="python")
    raw["policy"]["rollout_generation"]["top_p"] = 0.9
    raw["policy"]["train_generation"]["top_p"] = 0.9
    config = EmbodiedExperimentConfig.model_validate(raw)

    with pytest.raises(ValueError, match="top_p must be 1.0"):
        make_openvla_oft_policy(config, load=False)


def test_registered_custom_policy_factory_receives_generic_yaml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _config().model_dump(mode="python")
    raw["policy"]["type"] = "test_custom_lerobot_policy"
    raw["policy"]["load_kwargs"] = {"native_option": 7}
    raw["policy"]["rollout_generation"]["top_p"] = 0.9
    raw["policy"]["train_generation"]["top_p"] = 0.9
    config = EmbodiedExperimentConfig.model_validate(raw)
    captured = {}

    def factory(config, *, load):
        captured["config"] = config
        captured["load"] = load
        return "custom-policy"

    monkeypatch.setitem(
        policy_factories._POLICY_FACTORIES,
        "test_custom_lerobot_policy",
        factory,
    )

    assert make_policy(config, load=False) == "custom-policy"
    assert captured == {"config": config, "load": False}
    assert "openvla_oft" in registered_policy_types()
    assert "test_custom_lerobot_policy" in registered_policy_types()


def test_make_policy_rejects_unregistered_type() -> None:
    raw = _config().model_dump(mode="python")
    raw["policy"]["type"] = "missing_policy_plugin"
    config = EmbodiedExperimentConfig.model_validate(raw)

    with pytest.raises(ValueError, match="missing_policy_plugin"):
        make_policy(config, load=False)


def test_custom_policy_without_capability_declaration_fails_handshake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _config().model_dump(mode="python")
    raw["policy"]["type"] = "undeclared_policy"
    config = EmbodiedExperimentConfig.model_validate(raw)
    monkeypatch.setitem(
        policy_factories._POLICY_FACTORIES,
        "undeclared_policy",
        lambda _config, *, load: load,
    )

    with pytest.raises(ValueError, match="no probability capability declaration"):
        policy_capabilities(config)
