from __future__ import annotations

import math
from pathlib import Path
import sys
import types

import pytest

from art_embodied import Action, EmbodiedTrajectory, EmbodiedTrajectoryGroup
from art_embodied.backends.action_token import (
    ActionTokenExample,
    ActionTokenGRPOBackend,
    ActionTokenGSPOBackend,
)
from art_embodied.backends.action_token_worker import (
    _offload_runtime,
    _refresh_policy_snapshot,
)
from art_embodied.backends.local_process import (
    LocalProcessActionTokenBackend,
    _aggregate_batch_metrics,
    _expose_coordinator_gspo_metrics,
    _partition_examples,
    _PersistentGradientWorkerPool,
    _prepare_gradient_batches,
    _raise_for_worker_alignment_rejection,
    _worker_environment,
    _WorkerResult,
)
from art_embodied.config import EmbodiedExperimentConfig
from art_embodied.policies.openvla import refresh_openvla_peft_adapter


def test_full_update_enforces_pre_update_alignment_guard(tmp_path: Path) -> None:
    config = EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_action_token_distributed_smoke.yaml"
    )
    raw = config.model_dump(mode="python")
    raw["algorithm"]["pad_fixed_horizon_examples"] = False
    raw["training"]["schedule"] = {"type": "full_update"}
    raw["training"]["optimizer_steps_per_update"] = 1
    raw["storage"]["output_dir"] = str(tmp_path / "run")
    raw["runtime"]["worker_handoff_dir"] = str(tmp_path / "handoff")
    config = EmbodiedExperimentConfig.model_validate(raw)
    pool = object.__new__(_PersistentGradientWorkerPool)
    pool.config = config
    pool.job_index = 0

    assert pool._guard_alignment_for_job() is True
    pool.job_index = 1
    assert pool._guard_alignment_for_job() is False


def test_full_update_epochs_reuse_complete_prepared_batch(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    import yaml

    source = (
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_action_token_distributed_smoke.yaml"
    )
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["algorithm"]["pad_fixed_horizon_examples"] = False
    raw["training"]["schedule"] = {"type": "full_update", "update_epochs": 2}
    raw["training"]["optimizer_steps_per_update"] = 2
    raw["storage"]["output_dir"] = str(tmp_path / "run")
    raw["runtime"]["worker_handoff_dir"] = str(tmp_path / "handoff")
    config = EmbodiedExperimentConfig.model_validate(raw)
    backend = ActionTokenGRPOBackend(
        policy=object(),
        normalize_advantages=True,
        advantage_normalization_scope="group",
        advantage_std_unbiased=False,
        require_prompts=True,
        loss_aggregation="trajectory_mean",
    )
    trajectories = []
    for index, reward in enumerate((1.0, 0.0)):
        trajectory = EmbodiedTrajectory(task="test", reward=reward)
        trajectory.actions.append(
            Action(
                step=0,
                kind="token",
                raw={"tokens": [index], "prompt": "test"},
                logprobs={"token_logprobs": [math.log(0.5)]},
            )
        )
        trajectories.append(trajectory)

    batches, _ = _prepare_gradient_batches(
        [EmbodiedTrajectoryGroup(trajectories)],
        config=config,
        backend=backend,
    )

    assert [batch.index for batch in batches] == [0, 1]
    assert batches[0].examples == batches[1].examples
    assert batches[0].denominator_examples == batches[1].denominator_examples == 2


def _example(index: int, tokens: int) -> ActionTokenExample:
    return ActionTokenExample(
        task="test",
        tokens=list(range(tokens)),
        logprobs=[0.0] * tokens,
        reward=float(index % 2),
        trajectory_index=index,
        action_index=0,
        step=0,
        metadata={},
    )


def test_partition_examples_balances_token_work_without_duplication() -> None:
    examples = [_example(index, tokens) for index, tokens in enumerate((9, 8, 7, 6, 5))]

    partitions = _partition_examples(examples, workers=3)

    flattened = [
        example.trajectory_index for partition in partitions for example in partition
    ]
    loads = [
        sum(len(example.tokens) for example in partition) for partition in partitions
    ]
    assert sorted(flattened) == list(range(5))
    assert max(loads) - min(loads) <= 5


def test_partition_examples_routes_one_homogeneous_partition_per_task() -> None:
    examples = [
        _example(index, 3).model_copy(update={"task": task})
        for index, task in enumerate(("task-b", "task-a", "task-c", "task-a"))
    ]

    partitions = _partition_examples(
        examples,
        workers=2,
        task_keys=("task-a", "task-b", "task-c"),
    )

    assert len(partitions) == 3
    assert [{example.task for example in rows} for rows in partitions] == [
        {"task-a"},
        {"task-b"},
        {"task-c"},
    ]


def test_partition_examples_requires_every_task_in_each_update() -> None:
    with pytest.raises(ValueError, match="must contain every task"):
        _partition_examples(
            [_example(0, 3).model_copy(update={"task": "task-a"})],
            workers=2,
            task_keys=("task-a", "task-b"),
        )


def test_worker_environment_maps_physical_device_to_worker_cuda_zero() -> None:
    env = _worker_environment("cuda:7")

    assert env["CUDA_VISIBLE_DEVICES"] == "7"
    assert env["TOKENIZERS_PARALLELISM"] == "false"


def test_worker_environment_rejects_ambiguous_cuda_device() -> None:
    with pytest.raises(ValueError, match="cuda:N"):
        _worker_environment("cuda")


def test_persistent_worker_lifecycle_metrics_sum_across_subupdates() -> None:
    metrics = _aggregate_batch_metrics(
        [
            {
                "embodied_action_token_grpo/distributed_policy_loads": 2.0,
                "embodied_action_token_grpo/distributed_adapter_refreshes": 0.0,
                "embodied_action_token_grpo/distributed_worker_seconds_mean": 4.0,
                "embodied_action_token_grpo/clip_fraction": 0.1,
                "embodied_action_token_grpo/loss": 0.0,
                "embodied_action_token_grpo/ratio_min": 1.0,
                "embodied_action_token_grpo/ratio_max": 1.0,
                "embodied_action_token_grpo/ratio_mean": 1.0,
                "embodied_action_token_grpo/optimizer_state_entries_before": 0.0,
                "embodied_action_token_grpo/optimizer_state_step_min_before": 0.0,
                "embodied_action_token_grpo/optimizer_state_step_max_before": 0.0,
                "embodied_action_token_grpo/optimizer_state_entries_after": 878.0,
                "embodied_action_token_grpo/optimizer_state_step_min_after": 1.0,
                "embodied_action_token_grpo/optimizer_state_step_max_after": 1.0,
                "embodied_action_token_grpo/optimizer_state_step_mean_after": 1.0,
            },
            {
                "embodied_action_token_grpo/distributed_policy_loads": 0.0,
                "embodied_action_token_grpo/distributed_adapter_refreshes": 2.0,
                "embodied_action_token_grpo/distributed_worker_seconds_mean": 6.0,
                "embodied_action_token_grpo/clip_fraction": 0.7,
                "embodied_action_token_grpo/loss": 0.01,
                "embodied_action_token_grpo/ratio_min": 0.2,
                "embodied_action_token_grpo/ratio_max": 2.5,
                "embodied_action_token_grpo/ratio_mean": 0.8,
                "embodied_action_token_grpo/optimizer_state_entries_before": 878.0,
                "embodied_action_token_grpo/optimizer_state_step_min_before": 1.0,
                "embodied_action_token_grpo/optimizer_state_step_max_before": 1.0,
                "embodied_action_token_grpo/optimizer_state_entries_after": 878.0,
                "embodied_action_token_grpo/optimizer_state_step_min_after": 2.0,
                "embodied_action_token_grpo/optimizer_state_step_max_after": 2.0,
                "embodied_action_token_grpo/optimizer_state_step_mean_after": 2.0,
            },
        ]
    )

    assert metrics["embodied_action_token_grpo/distributed_policy_loads"] == 2.0
    assert metrics["embodied_action_token_grpo/distributed_adapter_refreshes"] == 2.0
    assert metrics["embodied_action_token_grpo/distributed_worker_seconds_mean"] == 5.0
    assert metrics["embodied_action_token_grpo/clip_fraction"] == pytest.approx(0.4)
    assert metrics["embodied_action_token_grpo/clip_fraction_first_subupdate"] == 0.1
    assert metrics["embodied_action_token_grpo/clip_fraction_last_subupdate"] == 0.7
    assert metrics["embodied_action_token_grpo/loss_first_subupdate"] == 0.0
    assert metrics["embodied_action_token_grpo/loss_last_subupdate"] == 0.01
    assert metrics["embodied_action_token_grpo/ratio_min_first_subupdate"] == 1.0
    assert metrics["embodied_action_token_grpo/ratio_min_last_subupdate"] == 0.2
    assert metrics["embodied_action_token_grpo/ratio_max_first_subupdate"] == 1.0
    assert metrics["embodied_action_token_grpo/ratio_max_last_subupdate"] == 2.5
    assert metrics["embodied_action_token_grpo/ratio_mean"] == pytest.approx(0.9)
    assert metrics["embodied_action_token_grpo/ratio_mean_first_subupdate"] == 1.0
    assert metrics["embodied_action_token_grpo/ratio_mean_last_subupdate"] == 0.8
    assert metrics["embodied_action_token_grpo/optimizer_state_entries_before"] == 0.0
    assert metrics["embodied_action_token_grpo/optimizer_state_step_min_before"] == 0.0
    assert metrics["embodied_action_token_grpo/optimizer_state_step_max_before"] == 1.0
    assert metrics["embodied_action_token_grpo/optimizer_state_entries_after"] == 878.0
    assert metrics["embodied_action_token_grpo/optimizer_state_step_min_after"] == 2.0
    assert metrics["embodied_action_token_grpo/optimizer_state_step_max_after"] == 2.0
    assert metrics["embodied_action_token_grpo/optimizer_state_step_mean_after"] == 2.0


def test_coordinator_exposes_only_final_gspo_metrics() -> None:
    metrics = {
        "embodied_action_token_grpo/optimizer_step_completed": 1.0,
        "embodied_action_token_grpo/policy_parameters_updated": 1.0,
        "embodied_action_token_gspo/optimizer_step_completed": 0.0,
    }

    _expose_coordinator_gspo_metrics(metrics)

    assert metrics["embodied_action_token_gspo/optimizer_step_completed"] == 1.0
    assert metrics["embodied_action_token_gspo/policy_parameters_updated"] == 1.0
    assert not any(key.startswith("embodied_action_token_grpo/") for key in metrics)


def test_distributed_alignment_rejection_fails_closed() -> None:
    worker = _WorkerResult(
        index=3,
        device="cuda:3",
        metrics={
            "embodied_action_token_grpo/optimizer_step_skipped_logprob_misalignment": 1.0,
            "embodied_action_token_grpo/alignment_approx_kl_abs_mean": 0.25,
            "embodied_action_token_grpo/alignment_ratio_mean": 1.25,
        },
        gradient_payload={},
        elapsed_seconds=0.1,
        policy_loads=1,
        adapter_refreshes=0,
    )

    with pytest.raises(RuntimeError, match=r"alignment_ratio_mean.*1\.25"):
        _raise_for_worker_alignment_rejection([worker], batch_index=2)


def test_local_backend_publishes_policy_through_snapshot_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import art_embodied.backends.local_process as local_process_module

    backend = object.__new__(LocalProcessActionTokenBackend)
    backend.policy = object()
    backend.update_step = 3
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        local_process_module,
        "_save_policy_snapshot",
        lambda policy, path: calls.append(("save", (policy, path))),
    )
    monkeypatch.setattr(
        local_process_module,
        "_move_policy_model",
        lambda policy, device: calls.append(("move", (policy, device))),
    )

    snapshot = tmp_path / "snapshot"
    backend.save_snapshot(snapshot, update=3)
    backend.offload()
    backend.restore("cuda:2")

    assert calls == [
        ("save", (backend.policy, snapshot)),
        ("move", (backend.policy, "cpu")),
        ("move", (backend.policy, "cuda:2")),
    ]
    with pytest.raises(ValueError, match="requested=2, backend=3"):
        backend.save_snapshot(snapshot, update=2)


def test_persistent_worker_refreshes_peft_parameters_in_place(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import art_embodied.vla_trainable as trainable_module

    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    calls: dict[str, object] = {}

    class FakePeftModel:
        peft_config = {"default": object()}
        active_adapter = "default"

        def to(self, device: str):
            calls["device"] = device
            return self

    model = FakePeftModel()
    policy = types.SimpleNamespace(
        model=model,
        device="cuda:0",
        peft_adapter_path=None,
    )

    fake_peft = types.ModuleType("peft")

    def fake_set_state(target, state, *, adapter_name):
        calls["target"] = target
        calls["state"] = state
        calls["adapter_name"] = adapter_name
        return types.SimpleNamespace(missing_keys=[], unexpected_keys=[])

    fake_peft.set_peft_model_state_dict = fake_set_state
    fake_utils = types.ModuleType("peft.utils")
    fake_save_and_load = types.ModuleType("peft.utils.save_and_load")
    fake_save_and_load.load_peft_weights = lambda path, device: {
        "path": path,
        "device": device,
    }
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setitem(sys.modules, "peft.utils", fake_utils)
    monkeypatch.setitem(
        sys.modules,
        "peft.utils.save_and_load",
        fake_save_and_load,
    )
    monkeypatch.setattr(
        trainable_module,
        "_shield_broken_transformer_engine_for_peft",
        lambda: None,
    )
    monkeypatch.setattr(
        trainable_module,
        "_disable_incompatible_optional_peft_dispatches",
        lambda: None,
    )

    report = refresh_openvla_peft_adapter(policy, str(adapter_path))

    assert policy.model is model
    assert policy.peft_adapter_path == str(adapter_path)
    assert calls == {
        "target": model,
        "state": {"path": str(adapter_path), "device": "cpu"},
        "adapter_name": "default",
        "device": "cuda:0",
    }
    assert report["base_model_reloaded"] is False
    assert report["missing_keys"] == 0
    assert report["unexpected_keys"] == 0


def test_worker_cpu_offload_releases_model_and_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    class FakeModel:
        def to(self, device: str):
            calls.append(("model", device))
            return self

    class FakeOptimizer:
        def zero_grad(self, *, set_to_none: bool) -> None:
            calls.append(("zero_grad", set_to_none))

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(empty_cache=lambda: calls.append("empty_cache"))
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    policy = types.SimpleNamespace(model=FakeModel())
    backend = types.SimpleNamespace(optimizer=FakeOptimizer())

    report = _offload_runtime(policy, backend)

    assert report["ok"] is True
    assert report["offloaded"] is True
    assert calls == [("zero_grad", True), ("model", "cpu"), "empty_cache"]


def test_worker_refreshes_model_native_action_token_snapshot(tmp_path: Path) -> None:
    loaded: list[object] = []
    model = types.SimpleNamespace(
        to=lambda device: loaded.append(("to", device)),
        eval=lambda: loaded.append("eval"),
    )
    policy = types.SimpleNamespace(
        family="pi0_fast",
        device="cuda:0",
        model=model,
        load_checkpoint=lambda checkpoint: loaded.append(checkpoint),
    )
    snapshot = tmp_path / "adapter"

    report = _refresh_policy_snapshot(policy, str(snapshot))

    assert loaded == [
        {"path": str(snapshot)},
        ("to", "cuda:0"),
        "eval",
    ]
    assert report["adapter_path"] == str(snapshot)
    assert report["base_model_reloaded"] is False


def test_full_update_preparation_counts_advantage_signs() -> None:
    pytest.importorskip("torch")
    config = EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_action_token_smoke.yaml"
    )
    backend = ActionTokenGRPOBackend(
        policy=object(),
        normalize_advantages=True,
        advantage_normalization_scope="group",
        require_prompts=True,
        loss_aggregation="token_mean",
    )
    trajectories = []
    for index, reward in enumerate((1.0, 1.0, 0.0, 0.0)):
        trajectory = EmbodiedTrajectory(task="test", reward=reward)
        trajectory.actions.append(
            Action(
                step=0,
                kind="token",
                raw={"tokens": [index % 2], "prompt": "test"},
                logprobs={"token_logprobs": [math.log(0.5)]},
            )
        )
        trajectories.append(trajectory)

    batches, _ = _prepare_gradient_batches(
        [EmbodiedTrajectoryGroup(trajectories)],
        config=config,
        backend=backend,
    )

    assert len(batches) == 1
    assert batches[0].positive_tokens == 2
    assert batches[0].negative_tokens == 2


def test_full_update_trajectory_mean_denominator_counts_trajectories() -> None:
    pytest.importorskip("torch")
    config = EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_action_token_smoke.yaml"
    )
    backend = ActionTokenGRPOBackend(
        policy=object(),
        normalize_advantages=True,
        advantage_normalization_scope="group",
        advantage_std_unbiased=False,
        require_prompts=True,
        training_unit="action",
        rlinf_action_level_score_source="trajectory_reward",
        loss_aggregation="trajectory_mean",
    )
    trajectories = []
    for reward, action_count in ((1.0, 1), (0.0, 3)):
        trajectory = EmbodiedTrajectory(task="test", reward=reward)
        for step in range(action_count):
            trajectory.actions.append(
                Action(
                    step=step,
                    kind="token",
                    raw={"tokens": [int(reward)], "prompt": "test"},
                    logprobs={"token_logprobs": [math.log(0.5)]},
                )
            )
        trajectories.append(trajectory)

    batches, _ = _prepare_gradient_batches(
        [EmbodiedTrajectoryGroup(trajectories)],
        config=config,
        backend=backend,
    )

    assert len(batches) == 1
    assert len(batches[0].examples) == 4
    assert batches[0].denominator_examples == 2


def test_gspo_trajectory_minibatches_shuffle_once_and_preserve_scalar_advantages(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    import yaml

    source = (
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_libero_object_gspo_lora_smoke_h100.yaml"
    )
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["training"]["optimizer_steps_per_update"] = 2
    raw["training"]["schedule"] = {
        "type": "trajectory_minibatch",
        "minibatch_trajectories": 8,
        "shuffle_seed": 41,
    }
    raw["storage"]["output_dir"] = str(tmp_path / "run")
    raw["runtime"]["worker_handoff_dir"] = str(tmp_path / "handoff")
    config = EmbodiedExperimentConfig.model_validate(raw)
    backend = ActionTokenGSPOBackend(
        policy=object(),
        normalize_advantages=True,
        advantage_normalization_scope="group",
        advantage_std_unbiased=False,
        require_prompts=True,
        loss_aggregation="trajectory_mean",
    )
    trajectories = []
    for index in range(16):
        trajectory = EmbodiedTrajectory(
            task="test",
            reward=float(index < 8),
        )
        trajectory.actions.append(
            Action(
                step=0,
                kind="token",
                raw={"tokens": [index % 2], "prompt": "test"},
                logprobs={"token_logprobs": [math.log(0.5)]},
            )
        )
        trajectories.append(trajectory)
    groups = [EmbodiedTrajectoryGroup(trajectories)]

    batches, _ = _prepare_gradient_batches(
        groups,
        config=config,
        backend=backend,
        update_step=3,
    )
    repeated, _ = _prepare_gradient_batches(
        groups,
        config=config,
        backend=backend,
        update_step=3,
    )

    assert len(batches) == 2
    assert [len(batch.examples) for batch in batches] == [8, 8]
    assert [batch.denominator_examples for batch in batches] == [8, 8]
    assert [
        example.trajectory_index for batch in batches for example in batch.examples
    ] == [example.trajectory_index for batch in repeated for example in batch.examples]
    assert sorted(
        example.trajectory_index for batch in batches for example in batch.examples
    ) == list(range(16))
    assert all(
        example.metadata.get("group_advantage_prepared") is True
        and "token_advantages" not in example.metadata
        for batch in batches
        for example in batch.examples
    )


def test_rlinf_worker_batches_preserve_global_group_advantages() -> None:
    pytest.importorskip("torch")
    config = EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_action_token_distributed_smoke.yaml"
    )
    backend = ActionTokenGRPOBackend(
        policy=object(),
        normalize_advantages=True,
        advantage_normalization_scope="group",
        advantage_std_unbiased=False,
        require_prompts=True,
        loss_aggregation="trajectory_mean",
    )
    groups = []
    for group_index in range(2):
        trajectories = []
        for trajectory_index, reward in enumerate((1.0, 1.0, 0.0, 0.0)):
            trajectory = EmbodiedTrajectory(task=f"test-{group_index}", reward=reward)
            trajectory.actions.append(
                Action(
                    step=0,
                    kind="token",
                    raw={"tokens": [trajectory_index], "prompt": "test"},
                    logprobs={"token_logprobs": [math.log(0.5)]},
                )
            )
            trajectories.append(trajectory)
        groups.append(EmbodiedTrajectoryGroup(trajectories))

    batches, _ = _prepare_gradient_batches(
        groups,
        config=config,
        backend=backend,
    )

    assert len(batches) == 2
    examples = [example for batch in batches for example in batch.examples]
    advantages = [
        float(example.metadata["token_advantages"][0]) for example in examples
    ]
    assert sorted(advantages) == pytest.approx(
        [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
    )
    assert sum(batch.positive_tokens for batch in batches) == 4
    assert sum(batch.negative_tokens for batch in batches) == 4
