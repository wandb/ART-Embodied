from collections import Counter
from dataclasses import asdict

import numpy as np
import pytest

from examples.embodied.pi0_fast_long_shared_sft import (
    BalancedBatches,
    match_tasks,
    sft_dataset_root,
)
from examples.embodied.pi0_fast_spatial_teacher_control import verify_wandb_identity


def test_diagnostic_account_must_match_operator_configuration(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.delenv("ART_EMBODIED_EXPECTED_WANDB_EMAIL", raising=False)
    with pytest.raises(ValueError, match="Set ART_EMBODIED_EXPECTED_WANDB_EMAIL"):
        verify_wandb_identity({"email": "operator@example.com"})
    monkeypatch.setenv("ART_EMBODIED_EXPECTED_WANDB_EMAIL", "operator@example.com")
    verify_wandb_identity({"email": "operator@example.com"})
    verify_wandb_identity(SimpleNamespace(email="operator@example.com"))
    with pytest.raises(ValueError, match="Unexpected W&B account"):
        verify_wandb_identity({"email": "another@example.com"})


def test_sft_dataset_location_is_explicit(monkeypatch, tmp_path):
    monkeypatch.delenv("ART_EMBODIED_SFT_DATASET_ROOT", raising=False)
    with pytest.raises(ValueError, match="ART_EMBODIED_SFT_DATASET_ROOT"):
        sft_dataset_root()
    monkeypatch.setenv("ART_EMBODIED_SFT_DATASET_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="meta/tasks.parquet"):
        sft_dataset_root()
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/tasks.parquet").touch()
    assert sft_dataset_root() == tmp_path.resolve()


def test_match_by_language_not_numeric_task_id():
    mapping = match_tasks(
        {0: "pick up bowl", 1: "close drawer"},
        [("close drawer", 32), ("pick_up_bowl", 81)],
    )
    assert mapping == {0: 81, 1: 32}
    with pytest.raises(ValueError, match="one exact"):
        match_tasks({0: "pick up bowl"}, [("close drawer", 0)])
    with pytest.raises(ValueError, match="one exact"):
        match_tasks({0: "pick up bowl"}, [("pick up bowl", 0), ("pick_up_bowl", 1)])


def test_balanced_distributed_batches():
    rows = {task: np.arange(task * 100, (task + 1) * 100) for task in range(10)}
    counts = Counter()
    streams = []
    for rank in range(8):
        batches = list(BalancedBatches(rows, steps=2, rank=rank, seed=7))
        assert len(batches) == 10
        assert Counter(i // 100 for b in batches[:5] for i in b) == Counter(range(10))
        counts.update(i // 100 for b in batches[:5] for i in b)
        streams.append(batches)
    assert counts == Counter({i: 8 for i in range(10)})
    assert streams[0] != streams[1]
    assert streams[0] == list(BalancedBatches(rows, steps=2, rank=0, seed=7))


def test_materialized_plan(tmp_path, monkeypatch):
    from art_embodied.config import EmbodiedExperimentConfig
    from examples.embodied import pi0_fast_long_restart as restart
    from examples.embodied.libero.state_manifest import (
        load_state_manifest,
        state_sha256,
        write_state_manifest,
    )

    # Synthetic parser fixtures, not simulator states or sealed policy outcomes.
    dev = load_state_manifest(
        "examples/embodied/libero/state_manifests/pi05_long_dev_v1/manifest.json"
    )
    states, entries = {}, []
    for index, entry in enumerate(dev.entries):
        state = np.array([1000.0 + index, 0.0])
        states[entry.state_key] = state
        entries.append(asdict(entry) | {
            "state_sha256": state_sha256(state), "state_length": len(state),
        })
    sealed = write_state_manifest(
        tmp_path / "synthetic-sealed", suite_name=dev.suite_name,
        simulator_compatibility=dev.simulator_compatibility, states=states,
        entries=entries, generator={"unit_test_only": True},
    )

    monkeypatch.setattr(restart, "check_runtime", lambda: {"mujoco": "3.3.0"})
    from examples.embodied.libero import records

    monkeypatch.setattr(
        records, "_task_languages", lambda _: {i: f"fixture task {i}" for i in range(10)},
    )
    restart.plan(tmp_path, sealed_manifest=sealed)
    config = EmbodiedExperimentConfig.from_yaml(tmp_path / "grpo.yaml")
    assert config.policy.lora.rank_partition is None
    assert config.policy.lora.rank == 32
    assert config.rollout.groups_per_update * config.algorithm.group_size == 960
    assert (
        config.training.optimizer_steps_per_update
        * config.training.schedule.minibatch_trajectories
        == 960
    )
    assert config.training.updates == 100
    assert config.evaluation.every_updates == 5
    assert not config.evaluation.evaluate_after_first_update
    assert config.environment.kwargs["wait_steps_after_reset"] == 15
    assert config.policy.load_kwargs["action_decoder"] == "native"
    # Preserve established fast paths when materializing a new recipe. These
    # checks complement, but cannot replace, a complete-update timing gate.
    assert config.policy.load_kwargs["use_kv_cache"] is True
    assert config.policy.load_kwargs["training_logprob_mode"] == "full_sequence"
    assert config.runtime.rollout_execution.group_batching is True
    assert config.runtime.rollout_execution.inference_max_batch_size >= 4
    assert config.runtime.distributed_training is True
    assert config.runtime.rollout_devices == [f"cuda:{i}" for i in range(8)]
    assert config.runtime.training_devices == [f"cuda:{i}" for i in range(8)]
    assert config.rollout.action_payload.trainable_action_selection == "all"
    assert config.rollout.action_payload.max_trainable_actions_per_trajectory is None
    assert config.algorithm.loss_aggregation == "seq_mean_token_sum"
    assert config.policy.load_kwargs["sft_anchor"] is None
    assert config.observability.wandb.project == "art-embodied-pi0-fast-long"
    assert config.observability.weave.project == config.observability.wandb.project
    for arm in ("baseline", "candidate"):
        sealed = EmbodiedExperimentConfig.from_yaml(tmp_path / f"sealed-{arm}.yaml")
        assert sealed.evaluation.data_role == "sealed_test"
        assert sealed.evaluation.kwargs["require_policy_checkpoint"]
    from art_embodied.evaluation import validate_paired_evaluation_configs

    validate_paired_evaluation_configs(
        EmbodiedExperimentConfig.from_yaml(tmp_path / "sealed-baseline.yaml"),
        EmbodiedExperimentConfig.from_yaml(tmp_path / "sealed-candidate.yaml"),
    )


def test_global_token_weighted_gradient_matches_unsharded_batch():
    torch = pytest.importorskip("torch")

    x = torch.arange(1, 18, dtype=torch.float64)
    target = x.sin()
    reference = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    ((reference * x - target).square().mean()).backward()
    shards = [slice(0, 2), slice(2, 7), slice(7, 17)]
    gradients = []
    for shard in shards:
        weight = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        local_mean = (weight * x[shard] - target[shard]).square().mean()
        (local_mean * len(x[shard]) / len(x)).backward()
        gradients.append(weight.grad)
    torch.testing.assert_close(sum(gradients), reference.grad)


def test_isolated_lifecycle_check_validates_and_restores(monkeypatch):
    import art_embodied.art_compat as lifecycle
    from examples.embodied import pi0_fast_long_restart as restart

    calls = []

    def original(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(lifecycle, "require_compatible_runtime", original)
    checked = []
    monkeypatch.setattr(restart, "check_runtime", lambda: checked.append(True))
    with pytest.raises(RuntimeError, match="sentinel"):
        with restart.isolated_art_runtime():
            lifecycle.require_compatible_runtime(profile="pi0_fast")
            lifecycle.require_compatible_runtime(profile="pi")
            raise RuntimeError("sentinel")
    assert lifecycle.require_compatible_runtime is original
    assert checked == [True]
    assert calls == [
        {"profile": "control"},
        {"require_lerobot": False, "profile": "pi"},
    ]
