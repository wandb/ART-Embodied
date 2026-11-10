from __future__ import annotations

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from art_embodied.policies.flow_policy import FlowModelInputs, FlowSDERollout
from art_embodied.policies.flow_sde import FlowSDESchedule, FlowSDETransitionRecord
from art_embodied.policies.openpi import OpenPIFlowPolicy, validate_openpi_config


class Native(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(action_horizon=3, action_dim=4)
        self.pi05 = True
        self.action_in_proj = torch.nn.Linear(4, 4).to(torch.bfloat16)
        self.action_out_proj = torch.nn.Linear(4, 4).to(torch.bfloat16)
        self.paligemma = torch.nn.Linear(4, 4).to(torch.bfloat16)

    def denoise_step(self):
        pass


def policy():
    return OpenPIFlowPolicy(
        SimpleNamespace(_model=Native()),
        base_identity="fixed-checkpoint-digest",
        config_name="pi05_droid_jointpos_polaris",
        schedule=FlowSDESchedule(4, 0.5, deterministic_sampler="native_euler"),
        batch_size=4,
        execution_horizon=2,
        action_dim=2,
        lora_config={
            "enabled": True,
            "r": 2,
            "lora_alpha": 2,
            "lora_dropout": 0.0,
            "target_modules": ["auto"],
            "init_lora_weights": "gaussian",
        },
    )


def test_only_lora_is_trainable_and_native_precision_is_retained():
    value = policy()
    assert value.model.get_base_model().native is value.bridge.model
    for name, parameter in value.named_parameters():
        if parameter.requires_grad:
            assert "lora_" in name and "paligemma." not in name
            assert parameter.dtype == torch.float32
        else:
            assert parameter.dtype == torch.bfloat16


def test_safe_checkpoint_restores_adapter_and_all_cpu_rng(tmp_path):
    value = policy()
    torch.manual_seed(234)
    np.random.seed(456)
    random.seed(789)
    before = {
        name: parameter.detach().clone() for name, parameter in value.named_parameters()
    }
    destination = tmp_path / "policy"
    destination.mkdir()  # Distributed coordinator creates an empty directory.
    value.save_checkpoint(destination)
    expected_rng = (torch.rand(5), np.random.rand(5), random.random())
    assert (destination / "rng.safetensors").exists()
    assert not (destination / "rng.pt").exists()
    with torch.no_grad():
        for parameter in value.parameters():
            if parameter.requires_grad:
                parameter.add_(3)
    value.load_checkpoint(destination)
    for name, parameter in value.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    torch.testing.assert_close(torch.rand(5), expected_rng[0], rtol=0, atol=0)
    np.testing.assert_array_equal(np.random.rand(5), expected_rng[1])
    assert random.random() == expected_rng[2]
    with pytest.raises(FileExistsError):
        value.save_checkpoint(destination)


def test_checkpoint_contract_rejected_before_weight_change(tmp_path):
    value = policy()
    value.save_checkpoint(tmp_path / "checkpoint")
    path = tmp_path / "checkpoint/runtime.json"
    contract = json.loads(path.read_text())
    contract["batch_size"] = 8
    path.write_text(json.dumps(contract))
    before = {
        name: parameter.detach().clone() for name, parameter in value.named_parameters()
    }
    with pytest.raises(ValueError, match="runtime contract"):
        value.load_checkpoint(path.parent)
    for name, parameter in value.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_replay_padding_is_discarded_without_discarding_real_rows():
    value = policy()
    inputs = FlowModelInputs(
        images=(torch.zeros(2, 2, 2, 3),),
        image_masks=(torch.ones(2, dtype=torch.bool),),
        language_tokens=torch.ones(2, 2, dtype=torch.long),
        language_masks=torch.ones(2, 2, dtype=torch.bool),
        state=torch.arange(8).reshape(2, 4).float(),
    )
    states = torch.zeros(2, 3, 4)
    rollout = FlowSDERollout(
        torch.zeros(2, 3, 2),
        FlowSDETransitionRecord(
            states, states.clone(), torch.ones(2, dtype=torch.long), states.clone()
        ),
        inputs,
    )
    seen = []

    def rescore(padded):
        seen.append(padded)
        return padded.inputs.state[:, :2, None].expand(-1, -1, 2)

    value.bridge.rescore = rescore
    scores = value.flow_sde_logprobs(rollout)
    assert scores.shape == (2, 2, 2)
    assert seen[0].inputs.batch_size == 4
    torch.testing.assert_close(seen[0].inputs.state[2:], inputs.state[1:].expand(2, -1))


def config():
    return SimpleNamespace(
        policy=SimpleNamespace(
            load_kwargs={
                "checkpoint_sha256": "a" * 64,
                "normalization_sha256": "b" * 64,
            },
            dtype="bfloat16",
            force_trainable_float32=True,
            trainable_parameter_strategy="pi05_action_expert_lora",
            lora=SimpleNamespace(enabled=True, dropout=0.0, rank_partition=None),
        ),
        algorithm=SimpleNamespace(
            type="grpo",
            training_unit="action",
            action_advantage_mode="example",
            score_source="trajectory_reward",
            loss_aggregation="rlinf_chunk_mean",
            rollout_logprob_source="rollout_action",
            advantage=SimpleNamespace(
                normalize=True, scope="group", extra_global_normalization=False
            ),
            flow_sde=SimpleNamespace(
                joint_logprob=False,
                stochastic_transitions_per_sample=1,
                selected_step_sampling="uniform",
            ),
        ),
        rollout=SimpleNamespace(
            action_payload=SimpleNamespace(
                kind="continuous",
                require_old_logprobs=True,
                trainable_action_selection="all",
            ),
            max_policy_steps=70,
            max_episode_steps=1050,
        ),
        environment=SimpleNamespace(kwargs={"action_chunk_size": 15}),
        training=SimpleNamespace(
            schedule=SimpleNamespace(type="full_update"), optimizer_steps_per_update=1
        ),
    )


def test_runtime_contract_rejects_unimplemented_settings():
    value = config()
    assert validate_openpi_config(value).execution_horizon == 15
    value.policy.load_kwargs["extra_delta_transform"] = True
    with pytest.raises(ValueError):
        validate_openpi_config(value)
    value = config()
    value.policy.dtype = "float32"
    with pytest.raises(ValueError, match="BF16"):
        validate_openpi_config(value)
    value = config()
    value.algorithm.flow_sde.joint_logprob = True
    with pytest.raises(ValueError, match="one uniformly"):
        validate_openpi_config(value)


def test_worker_index_is_used_by_checkpoint_load():
    value = policy()
    calls = []
    value.restore_checkpoint = lambda path, **kwargs: calls.append((path, kwargs))
    value.set_worker_index(3)
    value.load_checkpoint("snapshot")
    assert calls == [("snapshot", {"cuda_rng_source_index": 3})]
    with pytest.raises(ValueError, match="worker index"):
        value.set_worker_index(-1)


def test_openpi_rejects_ignored_geometry_and_objective_flags():
    value = config()
    value.environment.kwargs["action_chunk_size"] = 10
    with pytest.raises(ValueError, match="geometry"):
        validate_openpi_config(value)
    value = config()
    value.algorithm.advantage.normalize = False
    with pytest.raises(ValueError, match="objective"):
        validate_openpi_config(value)
    value = config()
    value.training.optimizer_steps_per_update = 2
    with pytest.raises(ValueError, match="multiple"):
        validate_openpi_config(value)


@pytest.mark.parametrize("has_hook", [False, True])
def test_common_worker_configures_optional_replica_hook_before_load(
    monkeypatch, has_hook
):
    from art_embodied.backends import flow_sde_worker

    events = []
    instance = SimpleNamespace(
        load_checkpoint=lambda path: events.append(("load", path))
    )
    if has_hook:
        instance.set_worker_index = lambda index: events.append(("index", index))
    sentinel = object()
    monkeypatch.setattr(flow_sde_worker, "_worker_config", lambda spec: sentinel)
    monkeypatch.setattr(flow_sde_worker, "make_policy", lambda config: instance)
    actual, actual_config = flow_sde_worker._build_policy(
        {"policy_snapshot": "snapshot", "worker_index": 3}
    )
    assert actual is instance and actual_config is sentinel
    assert events == ([("index", 3)] if has_hook else []) + [("load", "snapshot")]
