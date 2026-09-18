"""Executed optimizer schedule for action-token embodied RL.

RLinf computes rewards and group-relative advantages over a complete rollout
update, reconstructs each actor rank's fixed-horizon tensor, shuffles locally,
and then performs sequential actor-global-batch optimizer steps. This module
keeps that execution contract out of simulator-specific runners.

The conformance target is RLinf release/v0.1 at commit
9df6dc80dc729a6caccab92dd676004e51b1d3a2 (Apache-2.0, Copyright 2025 The
RLinf Authors). The scheduling API and native ART backend integration are
ART-Embodied modifications and extensions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any

from art_embodied.config import (
    EmbodiedExperimentConfig,
    RlinfActorBatchScheduleConfig,
)
from art_embodied.trajectories import EmbodiedTrajectoryGroup
from art_embodied.types import LocalTrainResult, TrainResult

from ..backends.action_token import (
    ActionTokenExample,
    ActionTokenGRPOBackend,
    _example_advantage_sign_token_counts,
    _example_loss_denominator_count,
    prepare_action_token_examples,
)


@dataclass(frozen=True, slots=True)
class RlinfActionTokenSubupdate:
    """One actor-global optimizer batch with its frozen loss denominators."""

    index: int
    examples: tuple[ActionTokenExample, ...]
    denominator_examples: int
    denominator_tokens: int
    positive_tokens: int
    negative_tokens: int


@dataclass(frozen=True, slots=True)
class RlinfPreparedActionTokenUpdate:
    """Complete rollout update after global preparation and actor reshaping."""

    examples: tuple[ActionTokenExample, ...]
    subupdates: tuple[RlinfActionTokenSubupdate, ...]
    reward_filter_report: dict[str, Any]
    fixed_horizon_rows: int
    action_token_width: int


class RlinfScheduledActionTokenBackend:
    """Make one ``train`` call execute one complete rollout update.

    The wrapped backend still owns the GRPO/GSPO loss and optimizer. This class
    owns only full-update preparation, fixed-horizon actor batching, sequential
    optimizer steps, alignment-guard scope, and checkpoint cadence.
    """

    def __init__(
        self,
        *,
        backend: ActionTokenGRPOBackend,
        config: EmbodiedExperimentConfig,
    ) -> None:
        self.backend = backend
        self.config = config
        self.update_step = 0

    async def train(
        self,
        trajectory_groups: Sequence[EmbodiedTrajectoryGroup],
        **kwargs: Any,
    ) -> TrainResult:
        groups = list(trajectory_groups)
        prepared = prepare_rlinf_action_token_update(
            groups,
            backend=self.backend,
            config=self.config,
        )
        actor_batch = self.config.training.schedule
        if not isinstance(actor_batch, RlinfActorBatchScheduleConfig):
            raise TypeError(
                "RlinfScheduledActionTokenBackend requires an RLinf schedule"
            )

        original_checkpoint_dir = self.backend.checkpoint_dir
        original_kl_tolerance = self.backend.pre_update_logprob_kl_tolerance
        original_ratio_tolerance = self.backend.pre_update_ratio_tolerance
        results: list[TrainResult] = []
        checkpoint_this_update = (
            self.update_step + 1
        ) % self.config.training.checkpoint_every_updates == 0
        try:
            for subupdate in prepared.subupdates:
                is_first = subupdate.index == 0
                is_last = subupdate.index == len(prepared.subupdates) - 1
                guard_first = (
                    actor_batch.pre_update_alignment_guard == "first_subupdate"
                )
                self.backend.pre_update_logprob_kl_tolerance = (
                    original_kl_tolerance if is_first and guard_first else None
                )
                self.backend.pre_update_ratio_tolerance = (
                    original_ratio_tolerance if is_first and guard_first else None
                )
                self.backend.checkpoint_dir = (
                    original_checkpoint_dir
                    if is_last and checkpoint_this_update
                    else None
                )
                result = await self.backend.train(
                    groups,
                    _action_token_grpo_precomputed_examples=list(subupdate.examples),
                    _action_token_grpo_precomputed_examples_prepared=True,
                    _action_token_grpo_precomputed_reward_filter_report=(
                        prepared.reward_filter_report
                    ),
                    _action_token_grpo_global_example_count=(
                        subupdate.denominator_examples
                    ),
                    _action_token_grpo_global_token_count=(
                        subupdate.denominator_tokens
                    ),
                    _action_token_grpo_global_positive_token_count=(
                        subupdate.positive_tokens
                    ),
                    _action_token_grpo_global_negative_token_count=(
                        subupdate.negative_tokens
                    ),
                    optimizer_sub_update_index=float(subupdate.index),
                    optimizer_sub_updates_total=float(len(prepared.subupdates)),
                    **kwargs,
                )
                results.append(result)
        finally:
            self.backend.checkpoint_dir = original_checkpoint_dir
            self.backend.pre_update_logprob_kl_tolerance = original_kl_tolerance
            self.backend.pre_update_ratio_tolerance = original_ratio_tolerance

        if not results:
            raise RuntimeError("Action-token schedule produced no optimizer subupdates")
        self.update_step += 1
        update_result = _aggregate_update_result(
            update_step=self.update_step,
            results=results,
            prepared=prepared,
        )
        removed = _prune_action_token_checkpoints(
            checkpoint_root=original_checkpoint_dir,
            current_checkpoint=update_result.checkpoint_path,
            keep_last=self.config.storage.keep_last_checkpoints,
        )
        update_result.metrics["embodied_action_token_schedule/checkpoints_pruned"] = (
            float(removed)
        )
        return update_result

    async def close(self) -> None:
        await self.backend.close()


def prepare_rlinf_action_token_update(
    groups: Sequence[EmbodiedTrajectoryGroup],
    *,
    backend: ActionTokenGRPOBackend,
    config: EmbodiedExperimentConfig,
) -> RlinfPreparedActionTokenUpdate:
    """Prepare and partition one rollout update using RLinf v0.1 geometry."""

    actor_batch = config.training.schedule
    if not isinstance(actor_batch, RlinfActorBatchScheduleConfig):
        raise ValueError(
            "RLinf actor-batch preparation requires "
            "training.schedule.type='rlinf_actor_global_batch'"
        )
    expected_groups = (
        config.rollout.groups_per_update * config.rollout.epochs_per_update
    )
    if actor_batch.strict_geometry and len(groups) != expected_groups:
        raise ValueError(
            "Observed groups do not match the configured rollout update: "
            f"groups={len(groups)}, expected={expected_groups}"
        )

    examples, reward_filter_report = prepare_action_token_examples(
        groups,
        backend=backend,
        max_primitive_slots=config.rollout.max_episode_steps,
    )
    policy_steps = config.rollout.max_policy_steps
    fixed_horizon_rows = len(groups) * config.algorithm.group_size * policy_steps
    if len(examples) > fixed_horizon_rows:
        raise ValueError(
            "Observed action rows exceed the configured fixed horizon: "
            f"examples={len(examples)}, fixed_horizon_rows={fixed_horizon_rows}"
        )
    if actor_batch.strict_geometry:
        expected_rows = (
            config.training.optimizer_steps_per_update * actor_batch.global_batch_size
        )
        if fixed_horizon_rows != expected_rows:
            raise ValueError(
                "Executed fixed-horizon rows differ from configured actor "
                f"batches: rows={fixed_horizon_rows}, expected={expected_rows}"
            )

    partitions = partition_rlinf_actor_global_batches(
        examples,
        optimizer_steps_per_update=config.training.optimizer_steps_per_update,
        actor_batch=actor_batch,
        fixed_horizon_rows=fixed_horizon_rows,
    )
    virtual_rows = _split_count_evenly(fixed_horizon_rows, len(partitions))
    token_width = _action_token_width(examples)
    subupdates = tuple(
        _make_subupdate(
            index=index,
            examples=partition,
            denominator_examples=virtual_rows[index],
            token_width=token_width,
            loss_aggregation=config.algorithm.loss_aggregation,
        )
        for index, partition in enumerate(partitions)
    )
    if actor_batch.strict_geometry:
        for subupdate in subupdates:
            if subupdate.denominator_examples != actor_batch.global_batch_size:
                raise ValueError(
                    "RLinf subupdate denominator differs from actor global "
                    f"batch: subupdate={subupdate.index}, denominator="
                    f"{subupdate.denominator_examples}, expected="
                    f"{actor_batch.global_batch_size}"
                )
    return RlinfPreparedActionTokenUpdate(
        examples=tuple(examples),
        subupdates=subupdates,
        reward_filter_report=reward_filter_report,
        fixed_horizon_rows=fixed_horizon_rows,
        action_token_width=token_width,
    )


def partition_rlinf_actor_global_batches(
    examples: Sequence[ActionTokenExample],
    *,
    optimizer_steps_per_update: int,
    actor_batch: RlinfActorBatchScheduleConfig,
    fixed_horizon_rows: int,
) -> list[list[ActionTokenExample]]:
    """Reconstruct RLinf rank-local tensors and their seeded permutations."""

    if optimizer_steps_per_update <= 0:
        raise ValueError("optimizer_steps_per_update must be positive")
    if fixed_horizon_rows <= 0:
        raise ValueError("fixed_horizon_rows must be positive")
    if fixed_horizon_rows % actor_batch.actor_world_size != 0:
        raise ValueError("fixed_horizon_rows must be divisible by actor_world_size")

    if not actor_batch.rank_local_shuffle:
        return _globally_shuffled_partitions(
            examples,
            optimizer_steps_per_update=optimizer_steps_per_update,
            seed=actor_batch.actor_seed,
            fixed_horizon_rows=fixed_horizon_rows,
        )

    import torch

    rank_slots = _rank_local_time_major_slots(
        examples,
        fixed_horizon_rows=fixed_horizon_rows,
        actor_world_size=actor_batch.actor_world_size,
        groups_per_process=actor_batch.groups_per_process_per_rollout_epoch,
    )
    local_rows = fixed_horizon_rows // actor_batch.actor_world_size
    if local_rows % optimizer_steps_per_update != 0:
        raise ValueError(
            "Rank-local fixed horizon must be divisible by optimizer steps"
        )
    local_batch_rows = local_rows // optimizer_steps_per_update
    partitions: list[list[ActionTokenExample]] = [
        [] for _ in range(optimizer_steps_per_update)
    ]
    for rank, slots in enumerate(rank_slots):
        generator = torch.Generator()
        generator.manual_seed(actor_batch.actor_seed + rank)
        order = torch.randperm(local_rows, generator=generator).tolist()
        shuffled = [slots[index] for index in order]
        for subupdate_index in range(optimizer_steps_per_update):
            start = subupdate_index * local_batch_rows
            end = start + local_batch_rows
            partitions[subupdate_index].extend(
                examples[index] for index in shuffled[start:end] if index is not None
            )
    if sum(len(partition) for partition in partitions) != len(examples):
        raise ValueError("RLinf partition did not consume every observed example")
    return partitions


def _rank_local_time_major_slots(
    examples: Sequence[ActionTokenExample],
    *,
    fixed_horizon_rows: int,
    actor_world_size: int,
    groups_per_process: int,
) -> list[list[int | None]]:
    parsed: list[tuple[int, int, int, int, int]] = []
    max_group_index = -1
    group_sizes: set[int] = set()
    for compact_index, example in enumerate(examples):
        metadata = example.metadata
        try:
            group_index = int(metadata["group_index"])
            trajectory_index = int(metadata["trajectory_index_in_group"])
            group_size = int(metadata["group_size"])
            action_index = int(example.action_index)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "RLinf batching requires group_index, "
                "trajectory_index_in_group, group_size, and action_index"
            ) from exc
        if min(group_index, trajectory_index, action_index) < 0:
            raise ValueError("RLinf batching indices must be non-negative")
        if group_size <= 0 or trajectory_index >= group_size:
            raise ValueError("Invalid RLinf group geometry")
        parsed.append(
            (
                compact_index,
                group_index,
                trajectory_index,
                group_size,
                action_index,
            )
        )
        group_sizes.add(group_size)
        max_group_index = max(max_group_index, group_index)

    if len(group_sizes) != 1:
        raise ValueError("RLinf batching requires one constant group_size")
    group_size = next(iter(group_sizes))
    groups_per_epoch = actor_world_size * groups_per_process
    group_count = max_group_index + 1
    if group_count % groups_per_epoch != 0:
        raise ValueError("Observed groups do not fit the RLinf actor topology")
    rollout_epochs = group_count // groups_per_epoch
    global_trajectories = group_count * group_size
    if fixed_horizon_rows % global_trajectories != 0:
        raise ValueError("Fixed horizon is incompatible with observed groups")
    policy_steps = fixed_horizon_rows // global_trajectories
    local_trajectories = rollout_epochs * groups_per_process * group_size
    local_rows = local_trajectories * policy_steps
    slots: list[list[int | None]] = [
        [None] * local_rows for _ in range(actor_world_size)
    ]
    for compact_index, group_index, trajectory_index, _, action_index in parsed:
        if action_index >= policy_steps:
            raise ValueError("action_index exceeds the fixed policy horizon")
        rollout_epoch = group_index // groups_per_epoch
        group_in_epoch = group_index % groups_per_epoch
        rank = group_in_epoch // groups_per_process
        process_group = group_in_epoch % groups_per_process
        local_group = rollout_epoch * groups_per_process + process_group
        local_trajectory = local_group * group_size + trajectory_index
        slot = action_index * local_trajectories + local_trajectory
        if slots[rank][slot] is not None:
            raise ValueError("Duplicate example in one RLinf tensor slot")
        slots[rank][slot] = compact_index
    return slots


def _globally_shuffled_partitions(
    examples: Sequence[ActionTokenExample],
    *,
    optimizer_steps_per_update: int,
    seed: int,
    fixed_horizon_rows: int,
) -> list[list[ActionTokenExample]]:
    import torch

    slots = _global_time_major_slots(
        examples,
        fixed_horizon_rows=fixed_horizon_rows,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    order = torch.randperm(fixed_horizon_rows, generator=generator).tolist()
    shuffled = [slots[index] for index in order]
    partitions: list[list[ActionTokenExample]] = []
    for index in range(optimizer_steps_per_update):
        start = fixed_horizon_rows * index // optimizer_steps_per_update
        end = fixed_horizon_rows * (index + 1) // optimizer_steps_per_update
        partitions.append(
            [examples[item] for item in shuffled[start:end] if item is not None]
        )
    return partitions


def _global_time_major_slots(
    examples: Sequence[ActionTokenExample],
    *,
    fixed_horizon_rows: int,
) -> list[int | None]:
    group_count = max(int(item.metadata["group_index"]) for item in examples) + 1
    group_size = int(examples[0].metadata["group_size"])
    trajectories = group_count * group_size
    if fixed_horizon_rows % trajectories != 0:
        raise ValueError("Fixed horizon is incompatible with global trajectories")
    policy_steps = fixed_horizon_rows // trajectories
    slots: list[int | None] = [None] * fixed_horizon_rows
    for compact_index, example in enumerate(examples):
        group_index = int(example.metadata["group_index"])
        trajectory_index = int(example.metadata["trajectory_index_in_group"])
        if example.action_index >= policy_steps:
            raise ValueError("action_index exceeds the fixed policy horizon")
        batch_index = group_index * group_size + trajectory_index
        slot = example.action_index * trajectories + batch_index
        if slots[slot] is not None:
            raise ValueError("Duplicate example in one global tensor slot")
        slots[slot] = compact_index
    return slots


def _make_subupdate(
    *,
    index: int,
    examples: Sequence[ActionTokenExample],
    denominator_examples: int,
    token_width: int,
    loss_aggregation: str,
) -> RlinfActionTokenSubupdate:
    observed_tokens = sum(
        max(
            1,
            _example_loss_denominator_count(
                example,
                loss_aggregation=loss_aggregation,
            ),
        )
        for example in examples
    )
    denominator_tokens = (
        denominator_examples * token_width
        if loss_aggregation == "rlinf_masked_mean_ratio"
        else observed_tokens
    )
    signs = {"positive": 0, "negative": 0, "zero": 0}
    for example in examples:
        counts = _example_advantage_sign_token_counts(example)
        for name in signs:
            signs[name] += counts[name]
    return RlinfActionTokenSubupdate(
        index=index,
        examples=tuple(examples),
        denominator_examples=denominator_examples,
        denominator_tokens=denominator_tokens,
        positive_tokens=signs["positive"],
        negative_tokens=signs["negative"],
    )


def _action_token_width(examples: Sequence[ActionTokenExample]) -> int:
    widths = {
        len(example.logprobs or example.tokens)
        for example in examples
        if len(example.logprobs or example.tokens) > 0
    }
    if not widths:
        raise ValueError("Cannot infer action-token width from empty actions")
    if len(widths) != 1:
        raise ValueError(
            "RLinf fixed-horizon denominator requires one constant "
            f"action-token width, observed={sorted(widths)}"
        )
    return next(iter(widths))


def _split_count_evenly(total: int, parts: int) -> list[int]:
    if parts <= 0:
        raise ValueError("parts must be positive")
    return [
        total * (index + 1) // parts - total * index // parts for index in range(parts)
    ]


def _aggregate_update_result(
    *,
    update_step: int,
    results: Sequence[TrainResult],
    prepared: RlinfPreparedActionTokenUpdate,
) -> LocalTrainResult:
    final = results[-1]
    metrics = dict(final.metrics)
    completed = sum(
        result.metrics.get(
            "embodied_action_token_grpo/optimizer_step_completed",
            0.0,
        )
        for result in results
    )
    metrics.update(
        {
            "embodied_action_token_schedule/update": float(update_step),
            "embodied_action_token_schedule/subupdates_requested": float(
                len(prepared.subupdates)
            ),
            "embodied_action_token_schedule/subupdates_completed": float(len(results)),
            "embodied_action_token_schedule/optimizer_steps_completed": float(
                completed
            ),
            "embodied_action_token_schedule/examples_observed": float(
                len(prepared.examples)
            ),
            "embodied_action_token_schedule/fixed_horizon_rows": float(
                prepared.fixed_horizon_rows
            ),
            "embodied_action_token_schedule/action_token_width": float(
                prepared.action_token_width
            ),
        }
    )
    checkpoint_path = getattr(final, "checkpoint_path", None)
    return LocalTrainResult(
        step=update_step,
        metrics=metrics,
        checkpoint_path=(
            str(Path(checkpoint_path)) if checkpoint_path is not None else None
        ),
    )


def _prune_action_token_checkpoints(
    *,
    checkpoint_root: Path | None,
    current_checkpoint: str | None,
    keep_last: int,
) -> int:
    if checkpoint_root is None or current_checkpoint is None:
        return 0
    root = checkpoint_root.resolve()
    current = Path(current_checkpoint).resolve()
    if current.parent != root or not current.is_dir():
        return 0
    candidates = sorted(
        (
            path
            for path in root.glob("action-token-*-step-*")
            if path.is_dir() and not path.is_symlink()
        ),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    retained = {path.resolve() for path in candidates[: max(1, keep_last)]}
    retained.add(current)
    removed = 0
    for path in candidates:
        if path.resolve() in retained:
            continue
        shutil.rmtree(path)
        removed += 1
    return removed
