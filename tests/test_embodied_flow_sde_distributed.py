from __future__ import annotations

import io
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from art_embodied.backends.action_token_gradients import (  # noqa: E402
    load_action_token_gradient_payload,
)
from art_embodied.backends.flow_sde import FlowSDEExample  # noqa: E402
from art_embodied.backends.flow_sde_local_process import (  # noqa: E402
    LocalProcessFlowSDEBackend,
    _aggregate_flow_subupdates,
    _aggregate_flow_worker_metrics,
    _cleanup_flow_subupdate_artifacts,
    _flow_payload_cache_plan,
    _FlowGradientWorkerPool,
    _load_flow_worker_audit,
    _partition_flow_examples,
    _precalculate_distributed_flow_sde_logprobs,
    _weighted_worker_metric,
)
from art_embodied.backends.flow_sde_worker import (  # noqa: E402
    _compute_gradient_job,
    _worker_config,
)
from art_embodied.checkpointing import CheckpointManager  # noqa: E402
from art_embodied.config import EmbodiedExperimentConfig  # noqa: E402
from art_embodied.policies.flow_sde import (  # noqa: E402
    FlowSDETransitionRecord,
)
from art_embodied.policies.pi_flow_sde import (  # noqa: E402
    PIFlowModelInputs,
    PIFlowSDERollout,
)


class _Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.rescore_batch_sizes: list[int] = []

    def flow_sde_logprobs(self, rollout: PIFlowSDERollout) -> torch.Tensor:
        self.rescore_batch_sizes.append(rollout.inputs.batch_size)
        feature = rollout.inputs.language_tokens[:, :1].float().view(-1, 1, 1)
        return rollout.transition.old_logprobs + self.weight * feature


class _ReferencePolicy(_Policy):
    def flow_sde_reference_logprobs(
        self,
        rollout: PIFlowSDERollout,
    ) -> torch.Tensor:
        return torch.zeros_like(rollout.transition.old_logprobs)


def _example(index: int, *, task_key: str = "__unassigned__") -> FlowSDEExample:
    feature = -1 if index == 0 else 1
    rollout = PIFlowSDERollout(
        actions=torch.zeros(1, 2, 1),
        transition=FlowSDETransitionRecord(
            previous_states=torch.zeros(1, 2, 1),
            next_states=torch.zeros(1, 2, 1),
            selected_indices=torch.zeros(1, dtype=torch.long),
            old_logprobs=torch.zeros(1, 2, 1),
        ),
        inputs=PIFlowModelInputs(
            images=(torch.zeros(1, 3, 2, 2),),
            image_masks=(torch.ones(1, dtype=torch.bool),),
            language_tokens=torch.tensor([[feature]]),
            language_masks=torch.ones(1, 1, dtype=torch.bool),
            state=None,
        ),
    )
    return FlowSDEExample(
        rollout=rollout,
        group_index=0,
        trajectory_index=index,
        action_index=0,
        reward=float(index),
        loss_mask=True,
        trajectory_primitive_steps=2,
        task_key=task_key,
    )


class _PartitionedPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = torch.nn.ModuleDict(
            {
                "task_000": torch.nn.Linear(1, 1, bias=False),
                "task_001": torch.nn.Linear(1, 1, bias=False),
            }
        )
        for parameter in self.parameters():
            torch.nn.init.zeros_(parameter)

    def flow_sde_logprobs(self, rollout: PIFlowSDERollout) -> torch.Tensor:
        feature = rollout.inputs.language_tokens[:, :1].float()
        delta = sum(layer(feature) for layer in self.lora_A.values())
        return rollout.transition.old_logprobs + delta.view(-1, 1, 1)


def test_partition_flow_examples_preserves_exact_pairs() -> None:
    examples = [_example(index % 2) for index in range(7)]
    advantages = [float(index) for index in range(7)]

    partitions = _partition_flow_examples(examples, advantages, workers=3)

    flattened_advantages = [
        advantage for _, values in partitions for advantage in values
    ]
    assert sorted(flattened_advantages) == advantages
    assert [len(values) for values, _ in partitions] == [3, 2, 2]


def test_partition_flow_examples_can_assign_one_task_per_worker() -> None:
    examples = [
        _example(0, task_key="task-b"),
        _example(1, task_key="task-a"),
        _example(0, task_key="task-b"),
    ]
    advantages = [1.0, 2.0, 3.0]

    partitions = _partition_flow_examples(
        examples,
        advantages,
        workers=2,
        task_keys=["task-a", "task-b"],
    )

    assert [[row.task_key for row in rows] for rows, _ in partitions] == [
        ["task-a"],
        ["task-b", "task-b"],
    ]
    assert [values for _, values in partitions] == [[2.0], [1.0, 3.0]]


def test_partition_flow_examples_rejects_unknown_task() -> None:
    with pytest.raises(ValueError, match="no configured task_pcgrad worker"):
        _partition_flow_examples(
            [_example(0, task_key="task-c")],
            [1.0],
            workers=2,
            task_keys=["task-a", "task-b"],
        )


def test_distributed_precalculate_matches_worker_shard_geometry() -> None:
    class BatchGeometryPolicy(_Policy):
        def flow_sde_logprobs(self, rollout: PIFlowSDERollout) -> torch.Tensor:
            self.rescore_batch_sizes.append(rollout.inputs.batch_size)
            return (
                torch.full_like(
                    rollout.transition.old_logprobs,
                    float(rollout.inputs.batch_size),
                )
                + self.weight
            )

    examples = [_example(index) for index in range(4)]
    advantages = [-1.0, 1.0, -1.0, 1.0]
    subupdates = [(examples, advantages, 4, False)]
    policy = BatchGeometryPolicy()

    refreshed, refreshed_subupdates, report = (
        _precalculate_distributed_flow_sde_logprobs(
            policy,
            examples,
            subupdates,
            workers=2,
            microbatch_size=4,
            device="cpu",
        )
    )

    assert policy.rescore_batch_sizes == [2, 2]
    assert report["old_logprob_rescore_rows"] == 4.0
    assert report["old_logprob_rescore_microbatches"] == 2.0
    assert refreshed_subupdates[0][0] == refreshed
    for partition, _partition_advantages in _partition_flow_examples(
        refreshed,
        advantages,
        workers=2,
    ):
        rollout = PIFlowSDERollout.concatenate(
            [example.rollout for example in partition]
        )
        torch.testing.assert_close(
            rollout.transition.old_logprobs,
            policy.flow_sde_logprobs(rollout),
        )


def test_flow_payload_cache_reuses_identical_rlinf_epoch_slots() -> None:
    plans = [
        _flow_payload_cache_plan(
            subupdate_index=index,
            subupdate_count=96,
            update_epochs=4,
        )
        for index in range(96)
    ]

    assert plans[0] == ("slot-0000", True)
    assert plans[23] == ("slot-0023", True)
    assert plans[24] == ("slot-0000", False)
    assert plans[95] == ("slot-0023", False)
    assert _flow_payload_cache_plan(
        subupdate_index=0,
        subupdate_count=1,
        update_epochs=1,
    ) == (None, True)


def test_flow_metrics_distinguish_pre_update_alignment_from_policy_drift() -> None:
    metrics = _aggregate_flow_subupdates(
        [
            {
                "loss": 0.0,
                "ratio_mean": 1.0,
                "approximate_kl": 0.0,
                "approximate_kl_per_primitive": 0.0,
                "clip_fraction": 0.0,
                "previous_abs_delta_mean": 0.01,
                "previous_abs_delta_max": 0.04,
                "previous_ratio_mean": 1.001,
                "active_previous_abs_delta_mean": 0.002,
                "active_previous_abs_delta_max": 0.008,
                "active_previous_ratio_mean": 1.0002,
                "previous_abs_delta_per_primitive_mean": 0.001,
                "previous_abs_delta_per_primitive_max": 0.004,
                "active_previous_abs_delta_per_primitive_mean": 0.0002,
                "active_previous_abs_delta_per_primitive_max": 0.0008,
                "worker_seconds_max": 1.0,
                "payload_cache_hit_fraction": 0.0,
                "embodied_flow_sde_grpo/worker_gradient_pairwise_cosine_mean": 0.4,
                "embodied_flow_sde_grpo/worker_gradient_resultant_ratio": 0.6,
            },
            {
                "loss": 0.1,
                "ratio_mean": 0.9,
                "approximate_kl": 0.2,
                "approximate_kl_per_primitive": 0.1,
                "clip_fraction": 0.3,
                "previous_abs_delta_mean": 0.19,
                "previous_abs_delta_max": 0.4,
                "previous_ratio_mean": 0.98,
                "active_previous_abs_delta_mean": 0.12,
                "active_previous_abs_delta_max": 0.3,
                "active_previous_ratio_mean": 0.99,
                "previous_abs_delta_per_primitive_mean": 0.019,
                "previous_abs_delta_per_primitive_max": 0.04,
                "active_previous_abs_delta_per_primitive_mean": 0.012,
                "active_previous_abs_delta_per_primitive_max": 0.03,
                "worker_seconds_max": 1.0,
                "payload_cache_hit_fraction": 1.0,
                "embodied_flow_sde_grpo/worker_gradient_pairwise_cosine_mean": 0.2,
                "embodied_flow_sde_grpo/worker_gradient_resultant_ratio": 0.3,
            },
        ]
    )

    assert metrics[
        "embodied_flow_sde_grpo/pre_update_alignment_abs_delta_mean"
    ] == pytest.approx(0.01)
    assert metrics[
        "embodied_flow_sde_grpo/pre_update_alignment_abs_delta_max"
    ] == pytest.approx(0.04)
    assert metrics[
        "embodied_flow_sde_grpo/pre_update_alignment_ratio_mean"
    ] == pytest.approx(1.001)
    assert metrics[
        "embodied_flow_sde_grpo/pre_update_active_alignment_abs_delta_mean"
    ] == pytest.approx(0.002)
    assert metrics[
        "embodied_flow_sde_grpo/pre_update_active_alignment_abs_delta_max"
    ] == pytest.approx(0.008)
    assert metrics[
        "embodied_flow_sde_grpo/pre_update_active_alignment_ratio_mean"
    ] == pytest.approx(1.0002)
    assert metrics[
        "embodied_flow_sde_grpo/"
        "pre_update_active_alignment_abs_delta_per_primitive_mean"
    ] == pytest.approx(0.0002)
    assert metrics[
        "embodied_flow_sde_grpo/optimization_old_policy_abs_delta_mean"
    ] == pytest.approx(0.1)
    # Keep the legacy key for existing dashboards while making its meaning
    # explicit through the new alias.
    assert metrics["embodied_flow_sde_grpo/previous_abs_delta_mean"] == pytest.approx(
        0.1
    )
    assert metrics[
        "embodied_flow_sde_grpo/pre_update_worker_gradient_pairwise_cosine_mean"
    ] == pytest.approx(0.4)
    assert metrics[
        "embodied_flow_sde_grpo/worker_gradient_pairwise_cosine_mean_across_subupdates"
    ] == pytest.approx(0.3)
    assert metrics[
        "embodied_flow_sde_grpo/worker_gradient_pairwise_cosine_mean"
    ] == pytest.approx(0.2)
    assert metrics[
        "embodied_flow_sde_grpo/pre_update_worker_gradient_resultant_ratio"
    ] == pytest.approx(0.6)


def test_flow_worker_metrics_use_identity_ratio_when_no_rows_are_active() -> None:
    result = {
        "elapsed_seconds": 1.0,
        "payload_cache_hit": False,
        "metrics": {
            "loss": 0.0,
            "ratio_mean": 1.0,
            "approximate_kl": 0.0,
            "approximate_kl_per_primitive": 0.0,
            "clip_fraction": 0.0,
            "rows": 8.0,
            "valid_rows": 0.0,
            "previous_abs_delta_mean": 0.0,
            "previous_abs_delta_max": 0.0,
            "previous_ratio_mean": 1.0,
            "active_previous_abs_delta_mean": 0.0,
            "active_previous_abs_delta_max": 0.0,
            "active_previous_ratio_mean": 1.0,
            "previous_abs_delta_per_primitive_mean": 0.0,
            "previous_abs_delta_per_primitive_max": 0.0,
            "active_previous_abs_delta_per_primitive_mean": 0.0,
            "active_previous_abs_delta_per_primitive_max": 0.0,
        },
    }

    metrics = _aggregate_flow_worker_metrics(
        [result],
        apply_metrics={},
        subupdate_index=0,
    )

    assert metrics["ratio_mean"] == 1.0
    assert metrics["active_previous_abs_delta_mean"] == 0.0
    assert metrics["active_previous_ratio_mean"] == 1.0
    assert (
        _weighted_worker_metric(
            [result],
            "active_previous_ratio_mean",
            weight="valid_rows",
            empty_value=1.0,
        )
        == 1.0
    )


def test_flow_worker_audit_is_resolved_from_each_result_directory(
    tmp_path: Path,
) -> None:
    result_paths = []
    for index in range(2):
        directory = tmp_path / f"worker-{index}"
        directory.mkdir()
        result_path = directory / "result.json"
        torch.save({"worker_index": torch.tensor(index)}, directory / "audit.pt")
        result_paths.append(result_path)

    loaded = [_load_flow_worker_audit(path, max_mb=1) for path in result_paths]

    assert [int(item["worker_index"]) for item in loaded] == [0, 1]


def test_consumed_flow_subupdate_handoffs_are_removed_incrementally(
    tmp_path: Path,
) -> None:
    initial_snapshot = tmp_path / "snapshot-initial"
    current_snapshot = tmp_path / "snapshot-current"
    initial_snapshot.mkdir()
    current_snapshot.mkdir()
    result_paths = []
    for index in range(2):
        directory = tmp_path / f"worker-{index}" / "subupdate-0001"
        directory.mkdir(parents=True)
        result_path = directory / "result.json"
        result_path.write_text("{}", encoding="utf-8")
        (directory / "examples.pkl").write_bytes(b"consumed")
        result_paths.append(result_path)

    _cleanup_flow_subupdate_artifacts(
        result_paths=result_paths,
        snapshot=current_snapshot,
        initial_snapshot=initial_snapshot,
    )

    assert initial_snapshot.is_dir()
    assert not current_snapshot.exists()
    assert all(not path.parent.exists() for path in result_paths)


def test_flow_worker_returns_globally_normalized_gradient(tmp_path: Path) -> None:
    examples = [_example(0), _example(1)]
    payload_path = tmp_path / "examples.pkl"
    with payload_path.open("wb") as handle:
        pickle.dump(
            {"examples": examples, "advantages": [-1.0, 1.0]},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    gradient_path = tmp_path / "gradients.pt"
    audit_path = tmp_path / "audit.pt"

    policy = _Policy()
    result = _compute_gradient_job(
        {
            "worker_index": 0,
            "examples_path": str(payload_path),
            "gradient_path": str(gradient_path),
            "microbatch_size": 2,
            "loss_denominator": 2,
            "length_normalized": True,
            "max_episode_steps": 2,
            "clip_epsilon_low": 0.2,
            "clip_epsilon_high": 0.2,
            "clip_ratio_c": 3.0,
            "audit_path": str(audit_path),
        },
        policy=policy,
    )

    gradient = load_action_token_gradient_payload(gradient_path)
    assert result["ok"] is True
    assert result["payload_cache_hit"] is False
    assert result["metrics"]["rows"] == 2.0
    assert result["metrics"]["previous_abs_delta_max"] == pytest.approx(0.0)
    assert policy.rescore_batch_sizes == [2]
    assert gradient["gradients"]["weight"].item() < 0.0
    audit = torch.load(audit_path, map_location="cpu", weights_only=True)
    assert set(audit) == {
        "action_mask",
        "advantages",
        "current_chunk_logprobs",
        "loss_denominator",
        "loss_mask",
        "max_episode_steps",
        "microbatch_losses",
        "old_chunk_logprobs",
        "row_weights",
        "trajectory_primitive_steps",
    }
    torch.testing.assert_close(audit["advantages"], torch.tensor([-1.0, 1.0]))
    torch.testing.assert_close(
        audit["current_chunk_logprobs"], audit["old_chunk_logprobs"]
    )
    assert audit["loss_mask"].tolist() == [True, True]
    torch.testing.assert_close(audit["row_weights"], torch.ones(2))
    assert float(audit["microbatch_losses"].sum()) == pytest.approx(
        result["metrics"]["loss"]
    )


def test_flow_worker_applies_reference_anchor(tmp_path: Path) -> None:
    examples = [_example(1), _example(1)]
    payload_path = tmp_path / "examples.pkl"
    with payload_path.open("wb") as handle:
        pickle.dump(
            {"examples": examples, "advantages": [0.0, 0.0]},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    gradient_path = tmp_path / "gradients.pt"
    policy = _ReferencePolicy()
    policy.weight.data.fill_(0.5)

    result = _compute_gradient_job(
        {
            "worker_index": 0,
            "examples_path": str(payload_path),
            "gradient_path": str(gradient_path),
            "microbatch_size": 2,
            "loss_denominator": 2,
            "length_normalized": False,
            "max_episode_steps": 2,
            "clip_epsilon_low": 0.2,
            "clip_epsilon_high": 0.2,
            "clip_ratio_c": 3.0,
            "reference_kl_coefficient": 1.0,
        },
        policy=policy,
    )

    gradient = load_action_token_gradient_payload(gradient_path)
    assert gradient["gradients"]["weight"].item() > 0.0
    assert result["metrics"]["policy_loss"] == pytest.approx(0.0)
    assert result["metrics"]["reference_kl"] > 0.0
    assert result["metrics"]["reference_kl_coefficient"] == 1.0


def test_flow_worker_reuses_disk_staged_payload_across_update_epochs(
    tmp_path: Path,
) -> None:
    payload_path = tmp_path / "examples.pkl"
    with payload_path.open("wb") as handle:
        pickle.dump(
            {"examples": [_example(0), _example(1)], "advantages": [-1.0, 1.0]},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    base_command = {
        "worker_index": 0,
        "payload_cache_key": "slot-0000",
        "microbatch_size": 2,
        "loss_denominator": 2,
        "length_normalized": True,
        "max_episode_steps": 2,
        "clip_epsilon_low": 0.2,
        "clip_epsilon_high": 0.2,
        "clip_ratio_c": 3.0,
    }
    first = _compute_gradient_job(
        {
            **base_command,
            "examples_path": str(payload_path),
            "gradient_path": str(tmp_path / "first.pt"),
        },
        policy=_Policy(),
    )
    second = _compute_gradient_job(
        {
            **base_command,
            "examples_path": str(payload_path),
            "payload_cache_hit": True,
            "gradient_path": str(tmp_path / "second.pt"),
        },
        policy=_Policy(),
    )

    assert first["payload_cache_hit"] is False
    assert second["payload_cache_hit"] is True
    assert second["metrics"] == first["metrics"]
    first_gradient = load_action_token_gradient_payload(tmp_path / "first.pt")
    second_gradient = load_action_token_gradient_payload(tmp_path / "second.pt")
    torch.testing.assert_close(
        first_gradient["gradients"]["weight"],
        second_gradient["gradients"]["weight"],
    )


def test_flow_worker_shard_gradients_sum_to_single_worker_gradient(
    tmp_path: Path,
) -> None:
    examples = [_example(0), _example(1)]

    def run(name: str, selected, advantages):
        directory = tmp_path / name
        directory.mkdir()
        payload_path = directory / "examples.pkl"
        with payload_path.open("wb") as handle:
            pickle.dump(
                {"examples": selected, "advantages": advantages},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        gradient_path = directory / "gradients.pt"
        _compute_gradient_job(
            {
                "worker_index": 0,
                "examples_path": str(payload_path),
                "gradient_path": str(gradient_path),
                "microbatch_size": 1,
                "loss_denominator": 2,
                "length_normalized": True,
                "max_episode_steps": 2,
                "clip_epsilon_low": 0.2,
                "clip_epsilon_high": 0.2,
                "clip_ratio_c": 3.0,
            },
            policy=_Policy(),
        )
        return load_action_token_gradient_payload(gradient_path)["gradients"]["weight"]

    full = run("full", examples, [-1.0, 1.0])
    left = run("left", examples[:1], [-1.0])
    right = run("right", examples[1:], [1.0])

    torch.testing.assert_close(left + right, full)


def test_flow_worker_routes_each_task_gradient_to_its_rank_block(
    tmp_path: Path,
) -> None:
    config = SimpleNamespace(
        policy=SimpleNamespace(
            lora=SimpleNamespace(
                rank=2,
                rank_partition=SimpleNamespace(
                    mode="task_all_active",
                    task_keys=["task-a", "task-b"],
                ),
            )
        )
    )

    def run(name: str, examples, advantages):
        directory = tmp_path / name
        directory.mkdir()
        examples_path = directory / "examples.pkl"
        with examples_path.open("wb") as handle:
            pickle.dump(
                {"examples": examples, "advantages": advantages},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        gradient_path = directory / "gradients.pt"
        _compute_gradient_job(
            {
                "worker_index": 0,
                "examples_path": str(examples_path),
                "gradient_path": str(gradient_path),
                "microbatch_size": 2,
                "loss_denominator": 2,
                "length_normalized": True,
                "max_episode_steps": 2,
                "clip_epsilon_low": 0.2,
                "clip_epsilon_high": 0.2,
                "clip_ratio_c": 3.0,
            },
            policy=_PartitionedPolicy(),
            config=config,
        )
        return load_action_token_gradient_payload(gradient_path)

    task_a = _example(0, task_key="task-a")
    task_b = _example(1, task_key="task-b")
    combined = run("combined", [task_a, task_b], [-1.0, 1.0])
    only_a = run("only-a", [task_a], [-1.0])
    only_b = run("only-b", [task_b], [1.0])
    name_a = next(name for name in combined["gradients"] if ".task_000." in name)
    name_b = next(name for name in combined["gradients"] if ".task_001." in name)

    assert combined["active_adapters"] == ["task_000", "task_001"]
    torch.testing.assert_close(
        combined["gradients"][name_a], only_a["gradients"][name_a]
    )
    torch.testing.assert_close(
        combined["gradients"][name_b], only_b["gradients"][name_b]
    )
    torch.testing.assert_close(
        only_a["gradients"][name_b], torch.zeros_like(only_a["gradients"][name_b])
    )
    torch.testing.assert_close(
        only_b["gradients"][name_a], torch.zeros_like(only_b["gradients"][name_a])
    )


def test_task_partitioned_worker_config_preserves_internal_distributed_role() -> None:
    config_path = (
        Path(__file__).parents[1] / "examples/embodied/"
        "smolvla_libero_10_task_partitioned_flow_sde_grpo_r128_h100.yaml"
    )
    parent = EmbodiedExperimentConfig.from_yaml(config_path)

    worker = _worker_config({"config": parent.model_dump(mode="json")})

    assert parent.runtime.distributed_training is True
    assert worker.runtime.distributed_training is False
    assert worker.runtime.training_devices == ["cuda:0"]
    assert worker.policy.lora.rank_partition == parent.policy.lora.rank_partition


def test_flow_worker_pool_reuses_offloaded_workers_across_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = (
        Path(__file__).parents[1]
        / "examples/embodied/pi05_libero_object_flow_sde_grpo_smoke_h100.yaml"
    )
    raw = EmbodiedExperimentConfig.from_yaml(config_path).model_dump(mode="python")
    raw["runtime"]["worker_handoff_dir"] = tmp_path / "handoffs"
    raw["runtime"]["training_devices"] = ["cuda:0", "cuda:1"]
    raw["runtime"]["distributed_training"] = True
    raw["runtime"]["training_worker_lifecycle"] = "cpu_offload"
    config = EmbodiedExperimentConfig.model_validate(raw)

    pool = _FlowGradientWorkerPool(config=config, policy=object(), update_step=0)
    pool.run_dir = tmp_path / "handoffs" / "flow-training-test"
    worker_dir = pool.run_dir / "worker-00"
    worker_dir.mkdir(parents=True)
    stdin = io.StringIO()
    pool.workers = [SimpleNamespace(index=0, directory=worker_dir, stdin=stdin)]
    pool.offloaded = True
    monkeypatch.setattr(
        "art_embodied.backends.flow_sde_local_process._save_policy_snapshot",
        lambda policy, path: path.mkdir(parents=True),
    )
    monkeypatch.setattr(
        "art_embodied.backends.flow_sde_local_process._move_policy_model",
        lambda policy, device: None,
    )

    def complete_jobs(jobs) -> None:
        for _worker, result_path, _payload_path in jobs:
            result_path.write_text('{"ok": true}\n', encoding="utf-8")

    monkeypatch.setattr(pool, "_wait_jobs", complete_jobs)

    pool.begin_update(update_step=1)

    restore = json.loads(stdin.getvalue().splitlines()[0])
    assert restore["op"] == "restore"
    assert restore["policy_snapshot"].endswith("update-0001/initial")
    assert pool.reused_workers is True
    assert pool.offloaded is False

    pool.finish_update()

    offload = json.loads(stdin.getvalue().splitlines()[1])
    assert offload["op"] == "offload"
    assert pool.offloaded is True


def test_distributed_flow_backend_restores_policy_optimizer_and_cursor(
    tmp_path: Path,
) -> None:
    config_path = (
        Path(__file__).parents[1]
        / "examples/embodied/pi05_libero_object_flow_sde_grpo_smoke_h100.yaml"
    )
    raw = EmbodiedExperimentConfig.from_yaml(config_path).model_dump(mode="python")
    raw["runtime"]["distributed_training"] = True
    raw["runtime"]["training_devices"] = ["cpu", "cpu"]
    raw["training"]["updates"] = 20
    raw["storage"]["output_dir"] = tmp_path / "run"
    raw["storage"]["resume_from_checkpoint"] = None
    config = EmbodiedExperimentConfig.model_validate(raw)

    source_parameter = torch.nn.Parameter(torch.tensor(1.0))
    source_optimizer = torch.optim.AdamW([source_parameter], lr=1.0e-4)
    source_parameter.grad = torch.tensor(0.5)
    source_optimizer.step()
    checkpoint = tmp_path / "checkpoint" / "step-000010"

    def write(staging: Path) -> None:
        policy_dir = staging / "policy"
        policy_dir.mkdir()
        (policy_dir / "art_embodied_pi_snapshot.json").write_text("{}\n")
        torch.save(
            {"step": 10, "optimizer": source_optimizer.state_dict()},
            staging / "art_embodied_training_state.pt",
        )
        (staging / "art_embodied_training_state.json").write_text('{"step": 10}\n')

    CheckpointManager().publish(
        checkpoint,
        writer=write,
        config_fingerprint=config.fingerprint,
        resume_contract_fingerprint=config.resume_contract_fingerprint,
    )
    raw = config.model_dump(mode="python")
    raw["storage"]["resume_from_checkpoint"] = checkpoint
    raw["evaluation"]["evaluate_before_training"] = False
    resumed_config = EmbodiedExperimentConfig.model_validate(raw)

    class Policy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))
            self.loaded: Path | None = None

        def load_checkpoint(self, path) -> None:
            self.loaded = Path(path)

        def to(self, device):
            super().to(device)
            return self

    policy = Policy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=9.0e-4)
    backend = SimpleNamespace(optimizer=optimizer, step=0)

    distributed = LocalProcessFlowSDEBackend(
        config=resumed_config,
        policy=policy,
        backend=backend,
    )

    assert policy.loaded == checkpoint / "policy"
    assert distributed.update_step == 10
    assert backend.step == 10
    assert backend.optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-4)
    assert backend.optimizer.state_dict()["state"]
