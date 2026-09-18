from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from art_embodied.backends import factory as backend_factories
from art_embodied.backends.action_token import ActionTokenGRPOBackend
from art_embodied.backends.factory import (
    make_action_token_backend,
    make_embodied_backend,
    make_flow_sde_backend,
    registered_embodied_backends,
)
from art_embodied.backends.flow_sde import FlowSDEGRPOBackend
from art_embodied.backends.flow_sde_local_process import LocalProcessFlowSDEBackend
from art_embodied.backends.local_process import LocalProcessActionTokenBackend
from art_embodied.config import EmbodiedExperimentConfig
from art_embodied.conformance.rlinf import RlinfScheduledActionTokenBackend


class _Policy:
    def action_token_logprobs(self, observation, prompt, tokens, **kwargs):
        return [0.0] * len(tokens)


class _FlowPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))


class _ReferenceFlowPolicy(_FlowPolicy):
    def flow_sde_reference_logprobs(self, rollout):
        return rollout.transition.old_logprobs


def _config() -> EmbodiedExperimentConfig:
    path = (
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml"
    )
    return EmbodiedExperimentConfig.from_yaml(path)


def test_factory_maps_positive_control_without_hidden_backend_defaults() -> None:
    distributed = make_action_token_backend(_config(), policy=_Policy())
    backend = distributed.backend

    assert isinstance(distributed, LocalProcessActionTokenBackend)
    assert isinstance(backend, ActionTokenGRPOBackend)
    assert backend.lr == 5e-5
    assert backend.optimizer_weight_decay == 0.01
    assert backend.optimizer_adam_eps == 1e-5
    assert backend.clip_epsilon_low == 0.2
    assert backend.clip_epsilon_high == 0.28
    assert backend.action_advantage_mode == "rlinf_action_level_cumulative"
    assert backend.rlinf_action_level_score_source == "chunk_rewards"
    assert backend.loss_aggregation == "rlinf_masked_mean_ratio"
    assert backend.advantage_std_unbiased is True
    assert backend.advantage_epsilon == 1e-6
    assert backend.rlinf_action_level_extra_global_normalization is False
    assert backend.logprob_eval_mode is False
    assert backend.logprob_microbatch_size == 32
    assert backend.train_logprob_microbatch_size == 8
    assert backend.skip_optimizer_step_without_policy_gradient_signal is False
    assert backend.progress_path == (
        distributed.config.storage.output_dir
        / "diagnostics/action_token_progress.jsonl"
    )
    assert backend.progress_every_microbatches == 128
    assert backend.progress_max_bytes == 64 * 1024 * 1024
    assert backend.checkpoint_config_fingerprint == distributed.config.fingerprint
    assert (
        backend.checkpoint_resume_contract_fingerprint
        == distributed.config.resume_contract_fingerprint
    )
    assert distributed.config.training.optimizer_steps_per_update == 4
    assert distributed.config.training.schedule.global_batch_size == 16384


def test_factory_maps_pi0_fast_to_distributed_action_token_backend() -> None:
    config = EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1]
        / "examples/embodied/pi0_fast_libero_spatial_grpo_development_h100.yaml"
    )

    distributed = make_action_token_backend(config, policy=_Policy())

    assert isinstance(distributed, LocalProcessActionTokenBackend)
    assert isinstance(distributed.backend, ActionTokenGRPOBackend)
    assert distributed.config.policy.type == "pi0_fast"


def test_factory_rejects_policy_without_action_token_logprobs() -> None:
    with pytest.raises(TypeError, match="action_token_logprobs"):
        make_action_token_backend(_config(), policy=object())


def test_builtin_factory_preserves_single_device_scheduled_training() -> None:
    raw = _config().model_dump(mode="python")
    raw["runtime"]["distributed_training"] = False
    raw["runtime"]["training_devices"] = ["cuda:0"]
    config = EmbodiedExperimentConfig.model_validate(raw)

    scheduled = make_action_token_backend(config, policy=_Policy())

    assert isinstance(scheduled, RlinfScheduledActionTokenBackend)
    assert isinstance(scheduled.backend, ActionTokenGRPOBackend)


def test_flow_factory_wraps_native_backend_for_distributed_training() -> None:
    path = (
        Path(__file__).parents[1]
        / "examples/embodied/pi05_libero_object_flow_sde_grpo_smoke_h100.yaml"
    )
    raw = EmbodiedExperimentConfig.from_yaml(path).model_dump(mode="python")
    raw["runtime"]["distributed_training"] = True
    raw["runtime"]["training_devices"] = ["cuda:0", "cuda:1"]
    config = EmbodiedExperimentConfig.model_validate(raw)

    distributed = make_flow_sde_backend(config, policy=_FlowPolicy())

    assert isinstance(distributed, LocalProcessFlowSDEBackend)
    assert isinstance(distributed.backend, FlowSDEGRPOBackend)
    assert distributed.backend.pre_update_logprob_kl_tolerance == (
        config.algorithm.pre_update_logprob_kl_tolerance
    )
    assert distributed.backend.pre_update_ratio_tolerance == (
        config.algorithm.pre_update_ratio_tolerance
    )


def test_flow_factory_rejects_policy_without_reference_kl_capability() -> None:
    path = (
        Path(__file__).parents[1]
        / "examples/embodied/pi05_libero_object_flow_sde_grpo_smoke_h100.yaml"
    )
    raw = EmbodiedExperimentConfig.from_yaml(path).model_dump(mode="python")
    raw["algorithm"]["kl_coefficient"] = 0.01
    config = EmbodiedExperimentConfig.model_validate(raw)

    with pytest.raises(TypeError, match="flow_sde_reference_logprobs"):
        make_flow_sde_backend(config, policy=_FlowPolicy())


def test_flow_factory_wires_reference_kl_coefficient() -> None:
    path = (
        Path(__file__).parents[1]
        / "examples/embodied/pi05_libero_object_flow_sde_grpo_smoke_h100.yaml"
    )
    raw = EmbodiedExperimentConfig.from_yaml(path).model_dump(mode="python")
    raw["algorithm"]["kl_coefficient"] = 0.01
    raw["runtime"]["distributed_training"] = False
    raw["runtime"]["training_devices"] = ["cuda:0"]
    config = EmbodiedExperimentConfig.model_validate(raw)

    backend = make_flow_sde_backend(config, policy=_ReferenceFlowPolicy())

    assert isinstance(backend, FlowSDEGRPOBackend)
    assert backend.reference_kl_coefficient == 0.01


def test_generic_backend_registry_dispatches_without_openvla_conditionals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _config().model_dump(mode="python")
    raw["rollout"]["action_payload"]["kind"] = "continuous"
    raw["training"]["schedule"] = {"type": "full_update"}
    raw["training"]["optimizer_steps_per_update"] = 1
    raw["algorithm"]["pad_fixed_horizon_examples"] = False
    config = EmbodiedExperimentConfig.model_validate(raw)
    sentinel = object()

    def factory(config, *, policy, optimizer):
        assert policy == "custom-policy"
        assert optimizer == "custom-optimizer"
        return sentinel

    monkeypatch.setitem(
        backend_factories._BACKEND_FACTORIES,
        ("continuous", "grpo"),
        factory,
    )
    monkeypatch.delitem(
        backend_factories._BACKEND_REQUIREMENTS,
        ("continuous", "grpo"),
        raising=False,
    )

    result = make_embodied_backend(
        config,
        policy="custom-policy",
        optimizer="custom-optimizer",
    )

    assert result is sentinel
    assert ("token", "grpo") in registered_embodied_backends()
    assert ("continuous", "grpo") in registered_embodied_backends()


def test_generic_backend_registry_rejects_missing_capability() -> None:
    raw = _config().model_dump(mode="python")
    raw["rollout"]["action_payload"]["kind"] = "continuous"
    raw["training"]["schedule"] = {"type": "full_update"}
    raw["training"]["optimizer_steps_per_update"] = 1
    raw["algorithm"]["pad_fixed_horizon_examples"] = False
    config = EmbodiedExperimentConfig.model_validate(raw)

    with pytest.raises(ValueError, match="continuous"):
        make_embodied_backend(config, policy=object())
