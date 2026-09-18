from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from art_embodied import Action, EmbodiedTrajectory, EmbodiedTrajectoryGroup
from art_embodied.backends.action_token import (
    ActionTokenExample,
    ActionTokenGRPOBackend,
    _write_action_token_train_progress_record,
)
from art_embodied.config import (
    EmbodiedExperimentConfig,
    RlinfActorBatchScheduleConfig,
)
from art_embodied.conformance.rlinf import (
    RlinfActionTokenSubupdate,
    RlinfPreparedActionTokenUpdate,
    RlinfScheduledActionTokenBackend,
    _prune_action_token_checkpoints,
    partition_rlinf_actor_global_batches,
    prepare_rlinf_action_token_update,
)
from art_embodied.types import LocalTrainResult


def test_action_token_progress_log_honors_storage_byte_limit(tmp_path: Path) -> None:
    path = tmp_path / "diagnostics/progress.jsonl"
    record = {"event": "microbatch", "value": 1}
    encoded_size = len((json.dumps(record, sort_keys=True) + "\n").encode())

    _write_action_token_train_progress_record(
        record,
        progress_path=path,
        progress_max_bytes=encoded_size,
    )
    _write_action_token_train_progress_record(
        record,
        progress_path=path,
        progress_max_bytes=encoded_size,
    )

    assert path.stat().st_size == encoded_size
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def _config() -> EmbodiedExperimentConfig:
    path = (
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml"
    )
    return EmbodiedExperimentConfig.from_yaml(path)


def _example(
    *,
    group_index: int,
    trajectory_index: int,
    action_index: int,
    group_size: int,
) -> ActionTokenExample:
    return ActionTokenExample(
        task="task",
        trajectory_index=group_index * group_size + trajectory_index,
        action_index=action_index,
        step=action_index,
        tokens=[1],
        logprobs=[-0.5],
        reward=1.0,
        metadata={
            "group_index": group_index,
            "trajectory_index_in_group": trajectory_index,
            "group_size": group_size,
            "token_advantages": [1.0],
            "token_loss_mask": [True],
        },
    )


def test_rank_local_actor_batches_match_rlinf_process_topology() -> None:
    world_size = 4
    rollout_epochs = 2
    groups_per_process = 1
    group_size = 2
    policy_steps = 2
    optimizer_steps = 2
    actor_seed = 1234
    groups_per_epoch = world_size * groups_per_process
    examples = [
        _example(
            group_index=group_index,
            trajectory_index=trajectory_index,
            action_index=action_index,
            group_size=group_size,
        )
        for action_index in range(policy_steps)
        for group_index in range(rollout_epochs * groups_per_epoch)
        for trajectory_index in range(group_size)
    ]
    by_slot = {
        (
            int(item.metadata["group_index"]),
            int(item.metadata["trajectory_index_in_group"]),
            item.action_index,
        ): item
        for item in examples
    }
    actor_batch = RlinfActorBatchScheduleConfig(
        type="rlinf_actor_global_batch",
        global_batch_size=len(examples) // optimizer_steps,
        actor_seed=actor_seed,
        actor_world_size=world_size,
        rank_local_shuffle=True,
        groups_per_process_per_rollout_epoch=groups_per_process,
        action_chunk_size=1,
        strict_geometry=True,
        pre_update_alignment_guard="disabled",
    )

    partitions = partition_rlinf_actor_global_batches(
        examples,
        optimizer_steps_per_update=optimizer_steps,
        actor_batch=actor_batch,
        fixed_horizon_rows=len(examples),
    )

    expected = [[] for _ in range(optimizer_steps)]
    local_batch_size = rollout_epochs * groups_per_process * group_size
    local_rows = local_batch_size * policy_steps
    rows_per_step = local_rows // optimizer_steps
    for rank in range(world_size):
        local_slots = [
            by_slot[(group_index, trajectory_index, action_index)]
            for action_index in range(policy_steps)
            for rollout_epoch in range(rollout_epochs)
            for group_index in [
                rollout_epoch * groups_per_epoch + rank * groups_per_process
            ]
            for trajectory_index in range(group_size)
        ]
        generator = torch.Generator()
        generator.manual_seed(actor_seed + rank)
        order = torch.randperm(local_rows, generator=generator).tolist()
        shuffled = [local_slots[index] for index in order]
        for subupdate_index in range(optimizer_steps):
            start = subupdate_index * rows_per_step
            expected[subupdate_index].extend(shuffled[start : start + rows_per_step])

    assert partitions == expected
    assert sorted(id(item) for part in partitions for item in part) == sorted(
        id(item) for item in examples
    )


def test_full_update_preparation_preserves_virtual_fixed_horizon_denominators() -> None:
    raw = _config().model_dump(mode="python")
    raw["algorithm"]["group_size"] = 2
    raw["rollout"]["groups_per_update"] = 2
    raw["rollout"]["epochs_per_update"] = 1
    raw["rollout"]["minimum_completed_attempts_per_group"] = 2
    raw["rollout"]["max_episode_steps"] = 4
    raw["rollout"]["max_policy_steps"] = 2
    raw["training"]["optimizer_steps_per_update"] = 2
    raw["training"]["schedule"].update(
        {
            "global_batch_size": 4,
            "actor_world_size": 2,
            "groups_per_process_per_rollout_epoch": 1,
            "action_chunk_size": 2,
        }
    )
    config = EmbodiedExperimentConfig.model_validate(raw)

    def trajectory(success: bool, task: str) -> EmbodiedTrajectory:
        item = EmbodiedTrajectory(task=task, reward=5.0 if success else 0.0)
        rewards = [0.0, 5.0 if success else 0.0]
        item.actions.append(
            Action(
                step=0,
                kind="token",
                raw={"tokens": [1, 1], "prompt": task},
                logprobs={"token_logprobs": [math.log(0.5), math.log(0.5)]},
                metadata={
                    "primitive_rewards": rewards,
                    "primitive_loss_mask": [True, True],
                },
            )
        )
        return item

    groups = [
        EmbodiedTrajectoryGroup(
            [trajectory(True, f"task-{index}"), trajectory(False, f"task-{index}")]
        )
        for index in range(2)
    ]

    class Policy:
        def action_token_logprobs(self, examples):
            return [torch.zeros(len(example.tokens)) for example in examples]

    backend = ActionTokenGRPOBackend(
        policy=Policy(),
        require_prompts=True,
        action_advantage_mode="rlinf_action_level_cumulative",
        normalize_advantages=True,
        advantage_std_unbiased=True,
        rlinf_action_level_score_source="chunk_rewards",
        filter_rewards=True,
        reward_filter_mode="loss_mask",
        rewards_lower_bound=0.5,
        rewards_upper_bound=4.5,
        loss_aggregation="rlinf_masked_mean_ratio",
    )

    prepared = prepare_rlinf_action_token_update(
        groups,
        backend=backend,
        config=config,
    )

    assert len(prepared.examples) == 4
    assert prepared.fixed_horizon_rows == 8
    assert prepared.action_token_width == 2
    assert [item.denominator_examples for item in prepared.subupdates] == [4, 4]
    assert [item.denominator_tokens for item in prepared.subupdates] == [8, 8]
    assert sum(len(item.examples) for item in prepared.subupdates) == 4
    advantages = [
        float(example.metadata["token_advantages"][0]) for example in prepared.examples
    ]
    assert sum(value > 0.0 for value in advantages) == 2
    assert sum(value < 0.0 for value in advantages) == 2
    assert prepared.reward_filter_report["groups_kept"] == 2.0
    assert "_example_keep_mask" not in prepared.reward_filter_report
    assert "example_keep_mask" not in prepared.reward_filter_report
    assert "group_means" not in prepared.reward_filter_report
    assert prepared.reward_filter_report["group_reward_min"] == 2.5
    assert prepared.reward_filter_report["group_reward_max"] == 2.5


def test_scheduled_backend_executes_all_subupdates_and_saves_only_final(
    monkeypatch,
) -> None:
    config = _config()
    examples = tuple(
        _example(
            group_index=index,
            trajectory_index=0,
            action_index=0,
            group_size=1,
        )
        for index in range(4)
    )
    prepared = RlinfPreparedActionTokenUpdate(
        examples=examples,
        subupdates=tuple(
            RlinfActionTokenSubupdate(
                index=index,
                examples=(examples[index],),
                denominator_examples=16384,
                denominator_tokens=16384,
                positive_tokens=1,
                negative_tokens=0,
            )
            for index in range(4)
        ),
        reward_filter_report={"enabled": True},
        fixed_horizon_rows=65536,
        action_token_width=1,
    )

    def fake_prepare(groups, *, backend, config):
        return prepared

    monkeypatch.setattr(
        "art_embodied.conformance.rlinf.prepare_rlinf_action_token_update",
        fake_prepare,
    )

    class Backend:
        def __init__(self):
            self.checkpoint_dir = Path("checkpoints")
            self.pre_update_logprob_kl_tolerance = 0.01
            self.pre_update_ratio_tolerance = 0.02
            self.calls = []
            self.closed = False

        async def train(self, groups, **kwargs):
            self.calls.append(
                SimpleNamespace(
                    kwargs=kwargs,
                    checkpoint_dir=self.checkpoint_dir,
                    kl=self.pre_update_logprob_kl_tolerance,
                    ratio=self.pre_update_ratio_tolerance,
                )
            )
            return LocalTrainResult(
                step=len(self.calls),
                metrics={"embodied_action_token_grpo/optimizer_step_completed": 1.0},
                checkpoint_path=(
                    str(self.checkpoint_dir / "final")
                    if self.checkpoint_dir is not None
                    else None
                ),
            )

        async def close(self):
            self.closed = True

    backend = Backend()
    scheduled = RlinfScheduledActionTokenBackend(backend=backend, config=config)
    scheduled.update_step = 9

    result = asyncio.run(scheduled.train([]))

    assert len(backend.calls) == 4
    assert all(call.kl is None and call.ratio is None for call in backend.calls)
    assert all(call.checkpoint_dir is None for call in backend.calls[:-1])
    assert backend.calls[-1].checkpoint_dir == Path("checkpoints")
    assert [
        call.kwargs["_action_token_grpo_global_example_count"] for call in backend.calls
    ] == [16384] * 4
    assert result.step == 10
    assert result.checkpoint_path == "checkpoints/final"
    assert (
        result.metrics["embodied_action_token_schedule/optimizer_steps_completed"]
        == 4.0
    )


def test_checkpoint_retention_only_prunes_managed_sibling_directories(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()
    checkpoints = []
    for step in range(1, 5):
        path = root / f"action-token-grpo-step-{step}"
        path.mkdir()
        (path / "weights.bin").write_text(str(step), encoding="utf-8")
        checkpoints.append(path)
    unrelated = root / "manual-checkpoint"
    unrelated.mkdir()

    removed = _prune_action_token_checkpoints(
        checkpoint_root=root,
        current_checkpoint=str(checkpoints[-1]),
        keep_last=2,
    )

    assert removed == 2
    assert checkpoints[-1].is_dir()
    assert checkpoints[-2].is_dir()
    assert not checkpoints[0].exists()
    assert not checkpoints[1].exists()
    assert unrelated.is_dir()


def test_strict_fixed_horizon_rejects_variable_action_token_widths() -> None:
    examples = [
        _example(
            group_index=index,
            trajectory_index=0,
            action_index=0,
            group_size=1,
        )
        for index in range(2)
    ]
    examples[1].tokens.append(2)
    examples[1].logprobs.append(-0.5)

    from art_embodied.conformance import rlinf

    with pytest.raises(ValueError, match="constant action-token width"):
        rlinf._action_token_width(examples)
