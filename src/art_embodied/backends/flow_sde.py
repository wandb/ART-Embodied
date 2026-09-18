"""Single-process Flow-SDE GRPO backend for LeRobot PI policies."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any

import torch

from art_embodied.checkpointing import CheckpointManager
from art_embodied.config import RlinfActorBatchScheduleConfig
from art_embodied.policies.flow_policy import FlowSDERollout
from art_embodied.trajectories import EmbodiedTrajectoryGroup
from art_embodied.types import LocalTrainResult, TrainResult

from .flow_sde_grpo import (
    chunk_logprobs,
    flow_sde_grpo_loss,
    flow_sde_reference_kl_loss,
    group_relative_advantages,
    primitive_normalized_chunk_abs_delta,
)
from .flow_sde_progress import (
    emit_flow_sde_alignment,
    emit_flow_sde_training_progress,
    emit_flow_sde_trust_region_stop,
)

TRANSIENT_FLOW_SDE_ROLLOUT_KEY = "_art_embodied_transient_flow_sde_rollout"
FLOW_SDE_REPLAY_SELECTED_KEY = "_art_embodied_flow_sde_replay_selected"
FLOW_SDE_ACTION_DIMENSION_MASK_KEY = "flow_sde_action_dimension_mask"

FlowSDEActionMask = tuple[bool, ...] | tuple[tuple[bool, ...], ...]


@dataclass(frozen=True, slots=True)
class FlowSDEExample:
    """One retained stochastic action chunk and its frozen training coordinates."""

    rollout: FlowSDERollout
    group_index: int
    trajectory_index: int
    action_index: int
    reward: float
    loss_mask: bool
    trajectory_primitive_steps: int
    action_mask: FlowSDEActionMask | None = None
    task_key: str = "__unassigned__"


def _flow_sde_microbatches(
    examples: list[FlowSDEExample],
    advantages: list[float],
    *,
    microbatch_size: int,
    separate_task_keys: bool = False,
) -> list[tuple[list[FlowSDEExample], list[float]]]:
    """Bucket replay by exact tensor geometry before forming microbatches.

    Flow-policy prompts can have different token lengths. Padding an already
    encoded prefix is not guaranteed to preserve its score, so gradient
    accumulation must keep incompatible replay shapes in separate forward
    calls. The global loss denominator remains unchanged by this partition.
    """

    if len(examples) != len(advantages):
        raise ValueError("Flow-SDE examples and advantages do not align")
    if microbatch_size < 1:
        raise ValueError("microbatch_size must be positive")
    buckets: dict[tuple[object, ...], list[tuple[FlowSDEExample, float]]] = {}
    for example, advantage in zip(examples, advantages, strict=True):
        signature = example.rollout.batch_signature()
        if separate_task_keys:
            signature = (example.task_key, *signature)
        buckets.setdefault(signature, []).append((example, advantage))
    batches = []
    for rows in buckets.values():
        for start in range(0, len(rows), microbatch_size):
            chunk = rows[start : start + microbatch_size]
            batches.append(
                (
                    [example for example, _advantage in chunk],
                    [advantage for _example, advantage in chunk],
                )
            )
    return batches


def precalculate_flow_sde_logprobs(
    policy: Any,
    examples: list[FlowSDEExample],
    advantages: list[float],
    *,
    microbatch_size: int,
    device: str,
) -> tuple[list[FlowSDEExample], dict[str, float]]:
    """Freeze old scores using the exact forward geometry used for training.

    Rollout and gradient workers can use different batch geometry. With
    low-precision transformer kernels that alone can create a non-unit ratio
    before any parameter update. This one-time rescore removes that numerical
    artifact while preserving sampled actions, states, and advantages.
    """

    refreshed: dict[int, FlowSDEExample] = {}
    delta_sum = 0.0
    delta_count = 0
    delta_max = 0.0
    scorer_batches = 0
    with torch.no_grad():
        for batch, _batch_advantages in _flow_sde_microbatches(
            examples,
            advantages,
            microbatch_size=microbatch_size,
        ):
            rollout = FlowSDERollout.concatenate(
                [example.rollout for example in batch]
            ).to(device)
            current = policy.flow_sde_logprobs(rollout).float().detach().cpu()
            previous_full = rollout.transition.old_logprobs.to(
                device="cpu", dtype=torch.float32
            )
            if (
                current.ndim != previous_full.ndim
                or current.shape[0] != previous_full.shape[0]
                or any(
                    current.shape[index] > previous_full.shape[index]
                    for index in range(1, current.ndim)
                )
            ):
                raise ValueError(
                    "Flow-SDE old-logprob rescore is incompatible with the "
                    "retained native chunk: "
                    f"current={tuple(current.shape)}, old={tuple(previous_full.shape)}"
                )
            scored_region = tuple(slice(0, int(size)) for size in current.shape[1:])
            previous = previous_full[(slice(None), *scored_region)]
            delta = (current - previous).abs()
            delta_sum += float(delta.sum())
            delta_count += int(delta.numel())
            delta_max = max(delta_max, float(delta.max()) if delta.numel() else 0.0)
            for example, row in zip(batch, current, strict=True):
                refreshed_old = example.rollout.transition.old_logprobs.clone()
                refreshed_old[(slice(None), *scored_region)] = row.unsqueeze(0).to(
                    device=refreshed_old.device,
                    dtype=refreshed_old.dtype,
                )
                refreshed[id(example)] = replace(
                    example,
                    rollout=example.rollout.with_old_logprobs(refreshed_old),
                )
            scorer_batches += 1
    if len(refreshed) != len(examples):
        raise RuntimeError(
            "Flow-SDE old-logprob rescore did not visit every example: "
            f"visited={len(refreshed)}, expected={len(examples)}"
        )
    return (
        [refreshed[id(example)] for example in examples],
        {
            "old_logprob_rescore_rows": float(len(examples)),
            "old_logprob_rescore_microbatches": float(scorer_batches),
            "old_logprob_rescore_previous_abs_delta_mean": (
                delta_sum / delta_count if delta_count else 0.0
            ),
            "old_logprob_rescore_previous_abs_delta_max": delta_max,
        },
    )


class FlowSDEGRPOBackend:
    """Train sampler-aligned Gaussian action chunks with group-relative GRPO."""

    def __init__(
        self,
        *,
        policy: Any,
        optimizer: Any | None,
        device: str,
        learning_rate: float,
        weight_decay: float,
        betas: tuple[float, float],
        epsilon: float,
        max_grad_norm: float | None,
        pre_update_logprob_kl_tolerance: float | None = None,
        pre_update_ratio_tolerance: float | None = None,
        microbatch_size: int,
        optimizer_steps_per_update: int,
        training_schedule: Any,
        max_episode_steps: int,
        group_size: int,
        advantage_epsilon: float,
        advantage_std_unbiased: bool,
        clip_epsilon_low: float,
        clip_epsilon_high: float,
        clip_ratio_c: float | None,
        filter_rewards: bool,
        rewards_lower_bound: float | None,
        rewards_upper_bound: float | None,
        checkpoint_dir: Path,
        config_fingerprint: str,
        resume_contract_fingerprint: str,
        keep_last_checkpoints: int = 2,
        retain_checkpoint_updates: Iterable[int] = (),
        diagnostics_dir: Path | None = None,
        precalculate_logprobs: bool = False,
        reference_kl_coefficient: float = 0.0,
        sft_replay_coefficient: float = 0.0,
    ) -> None:
        self.policy = policy
        self.device = device
        self.group_size = int(group_size)
        self.advantage_epsilon = float(advantage_epsilon)
        self.advantage_std_unbiased = bool(advantage_std_unbiased)
        self.clip_epsilon_low = float(clip_epsilon_low)
        self.clip_epsilon_high = float(clip_epsilon_high)
        self.clip_ratio_c = clip_ratio_c
        self.filter_rewards = bool(filter_rewards)
        self.rewards_lower_bound = rewards_lower_bound
        self.rewards_upper_bound = rewards_upper_bound
        self.max_grad_norm = max_grad_norm
        self.pre_update_logprob_kl_tolerance = (
            float(pre_update_logprob_kl_tolerance)
            if pre_update_logprob_kl_tolerance is not None
            else None
        )
        self.pre_update_ratio_tolerance = (
            float(pre_update_ratio_tolerance)
            if pre_update_ratio_tolerance is not None
            else None
        )
        if microbatch_size < 1:
            raise ValueError("microbatch_size must be positive")
        self.microbatch_size = int(microbatch_size)
        if optimizer_steps_per_update < 1:
            raise ValueError("optimizer_steps_per_update must be positive")
        self.optimizer_steps_per_update = int(optimizer_steps_per_update)
        self.training_schedule = training_schedule
        self.max_episode_steps = int(max_episode_steps)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config_fingerprint = config_fingerprint
        self.resume_contract_fingerprint = resume_contract_fingerprint
        if keep_last_checkpoints < 1:
            raise ValueError("keep_last_checkpoints must be positive")
        self.keep_last_checkpoints = int(keep_last_checkpoints)
        self.retain_checkpoint_updates = frozenset(
            int(update) for update in retain_checkpoint_updates
        )
        self.diagnostics_dir = (
            Path(diagnostics_dir) if diagnostics_dir is not None else None
        )
        self.precalculate_logprobs = bool(precalculate_logprobs)
        self.reference_kl_coefficient = float(reference_kl_coefficient)
        if (
            not math.isfinite(self.reference_kl_coefficient)
            or self.reference_kl_coefficient < 0.0
        ):
            raise ValueError("reference_kl_coefficient must be finite and non-negative")
        self.sft_replay_coefficient = float(sft_replay_coefficient)
        if (
            not math.isfinite(self.sft_replay_coefficient)
            or self.sft_replay_coefficient < 0.0
        ):
            raise ValueError("sft_replay_coefficient must be finite and non-negative")
        if self.reference_kl_coefficient > 0.0 and not callable(
            getattr(policy, "flow_sde_reference_logprobs", None)
        ):
            raise TypeError(
                "positive reference_kl_coefficient requires policy."
                "flow_sde_reference_logprobs"
            )
        parameters = [
            parameter for parameter in policy.parameters() if parameter.requires_grad
        ]
        if not parameters:
            raise ValueError("Flow-SDE backend requires trainable policy parameters")
        self.optimizer = optimizer or torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            betas=betas,
            eps=epsilon,
            weight_decay=weight_decay,
        )
        self.step = 0
        self._checkpoint_manager = CheckpointManager()

    async def train(
        self,
        trajectory_groups: Iterable[EmbodiedTrajectoryGroup],
        **kwargs: Any,
    ) -> TrainResult:
        del kwargs
        groups = list(trajectory_groups)
        replay_eligible, replay_selected = flow_sde_replay_selection_counts(groups)
        examples, advantages, kept_groups = prepare_flow_sde_examples(
            groups,
            group_size=self.group_size,
            advantage_epsilon=self.advantage_epsilon,
            advantage_std_unbiased=self.advantage_std_unbiased,
            filter_rewards=self.filter_rewards,
            rewards_lower_bound=self.rewards_lower_bound,
            rewards_upper_bound=self.rewards_upper_bound,
        )
        if not examples:
            raise ValueError("No valid Flow-SDE action chunks found in rollout update")

        self.policy.train()
        rescore_metrics: dict[str, float] = {}
        if self.precalculate_logprobs:
            rescore_started = time.perf_counter()
            examples, rescore_metrics = precalculate_flow_sde_logprobs(
                self.policy,
                examples,
                advantages,
                microbatch_size=self.microbatch_size,
                device=self.device,
            )
            rescore_metrics["old_logprob_rescore_elapsed_seconds"] = (
                time.perf_counter() - rescore_started
            )
        subupdates = _flow_sde_subupdates(
            examples,
            advantages,
            group_count=len(groups),
            group_size=self.group_size,
            optimizer_steps_per_update=self.optimizer_steps_per_update,
            schedule=self.training_schedule,
            max_episode_steps=self.max_episode_steps,
        )
        subupdate_metrics = []
        trust_region_stop: dict[str, Any] | None = None
        training_started = time.perf_counter()
        subupdate_audits: list[dict[str, Any]] | None = (
            [] if self.diagnostics_dir is not None else None
        )
        for subupdate_index, (
            sub_examples,
            sub_advantages,
            loss_denominator,
            length_normalized,
        ) in enumerate(subupdates):
            audit_rows: list[dict[str, torch.Tensor]] | None = (
                [] if subupdate_audits is not None else None
            )
            metrics = self._train_subupdate(
                sub_examples,
                sub_advantages,
                loss_denominator=loss_denominator,
                length_normalized=length_normalized,
                audit_rows=audit_rows,
                check_pre_update_alignment=subupdate_index == 0,
                subupdate_index=subupdate_index,
            )
            if subupdate_audits is not None:
                assert audit_rows is not None
                subupdate_audits.append(
                    _collate_flow_sde_audit_rows(
                        audit_rows,
                        subupdate_index=subupdate_index,
                        loss_denominator=loss_denominator,
                        length_normalized=length_normalized,
                        metrics=metrics,
                    )
                )
            if not bool(metrics.pop("optimizer_step_applied")):
                threshold = _flow_sde_max_approximate_kl(self.training_schedule)
                assert threshold is not None
                guard_scope = _flow_sde_kl_guard_scope(self.training_schedule)
                trust_region_stop = {
                    "approximate_kl": float(metrics["approximate_kl"]),
                    "approximate_kl_per_primitive": float(
                        metrics["approximate_kl_per_primitive"]
                    ),
                    "scope": guard_scope,
                    "before_subupdate": float(subupdate_index + 1),
                }
                emit_flow_sde_trust_region_stop(
                    update=self.step + 1,
                    applied_subupdates=len(subupdate_metrics),
                    planned_subupdates=len(subupdates),
                    approximate_kl=trust_region_stop["approximate_kl"],
                    approximate_kl_per_primitive=trust_region_stop[
                        "approximate_kl_per_primitive"
                    ],
                    scope=guard_scope,
                    threshold=threshold,
                )
                break
            subupdate_metrics.append(metrics)
            emit_flow_sde_training_progress(
                update=self.step + 1,
                completed=subupdate_index + 1,
                total=len(subupdates),
                metrics=metrics,
                started_at=training_started,
            )
        if subupdate_audits is not None:
            self._write_batch_audit(
                groups,
                examples=examples,
                advantages=advantages,
                kept_groups=kept_groups,
                subupdates=subupdate_audits,
            )
        self.step += 1
        checkpoint = self._save_checkpoint()
        rewards = torch.as_tensor([example.reward for example in examples])
        metric_count = float(len(subupdate_metrics))
        return LocalTrainResult(
            step=self.step,
            checkpoint_path=str(checkpoint),
            metrics={
                **{
                    f"embodied_flow_sde_grpo/{key}": value
                    for key, value in rescore_metrics.items()
                },
                "embodied_flow_sde_grpo/loss": sum(
                    item["loss"] for item in subupdate_metrics
                )
                / metric_count,
                "embodied_flow_sde_grpo/policy_loss": sum(
                    item["policy_loss"] for item in subupdate_metrics
                )
                / metric_count,
                "embodied_flow_sde_grpo/reference_kl": sum(
                    item["reference_kl"] for item in subupdate_metrics
                )
                / metric_count,
                "embodied_flow_sde_grpo/reference_kl_loss": sum(
                    item["reference_kl_loss"] for item in subupdate_metrics
                )
                / metric_count,
                "embodied_flow_sde_grpo/reference_kl_coefficient": (
                    self.reference_kl_coefficient
                ),
                "embodied_flow_sde_grpo/ratio_mean": sum(
                    item["ratio_mean"] for item in subupdate_metrics
                )
                / metric_count,
                "embodied_flow_sde_grpo/approximate_kl": sum(
                    item["approximate_kl"] for item in subupdate_metrics
                )
                / metric_count,
                "embodied_flow_sde_grpo/approximate_kl_per_primitive": sum(
                    item["approximate_kl_per_primitive"] for item in subupdate_metrics
                )
                / metric_count,
                "embodied_flow_sde_grpo/clip_fraction": sum(
                    item["clip_fraction"] for item in subupdate_metrics
                )
                / metric_count,
                "embodied_flow_sde_grpo/gradient_norm": sum(
                    item["gradient_norm"] for item in subupdate_metrics
                )
                / metric_count,
                "embodied_flow_sde_grpo/examples": float(len(examples)),
                "embodied_flow_sde_grpo/replay_actions_eligible": float(
                    replay_eligible
                ),
                "embodied_flow_sde_grpo/replay_actions_selected": float(
                    replay_selected
                ),
                "embodied_flow_sde_grpo/replay_selection_fraction": (
                    replay_selected / replay_eligible
                ),
                "embodied_flow_sde_grpo/groups": float(len(groups)),
                "embodied_flow_sde_grpo/groups_kept": float(kept_groups),
                "embodied_flow_sde_grpo/reward_mean": float(rewards.mean()),
                "embodied_flow_sde_grpo/microbatches": sum(
                    item["microbatches"] for item in subupdate_metrics
                ),
                "embodied_flow_sde_grpo/optimizer_subupdates": metric_count,
                "embodied_flow_sde_grpo/optimizer_subupdates_planned": float(
                    len(subupdates)
                ),
                "embodied_flow_sde_grpo/trust_region_early_stop": float(
                    trust_region_stop is not None
                ),
                "embodied_flow_sde_grpo/trust_region_stop_approximate_kl": (
                    trust_region_stop["approximate_kl"]
                    if trust_region_stop is not None
                    else 0.0
                ),
                "embodied_flow_sde_grpo/trust_region_stop_abs_approximate_kl": (
                    abs(trust_region_stop["approximate_kl"])
                    if trust_region_stop is not None
                    else 0.0
                ),
                "embodied_flow_sde_grpo/trust_region_stop_approximate_kl_per_primitive": (
                    trust_region_stop["approximate_kl_per_primitive"]
                    if trust_region_stop is not None
                    else 0.0
                ),
                "embodied_flow_sde_grpo/trust_region_stop_before_subupdate": (
                    trust_region_stop["before_subupdate"]
                    if trust_region_stop is not None
                    else 0.0
                ),
                "embodied_flow_sde_grpo/previous_abs_delta_mean": sum(
                    item["previous_abs_delta_mean"] for item in subupdate_metrics
                )
                / metric_count,
                "embodied_flow_sde_grpo/previous_abs_delta_max": max(
                    item["previous_abs_delta_max"] for item in subupdate_metrics
                ),
                "embodied_flow_sde_grpo/previous_ratio_mean": sum(
                    item["previous_ratio_mean"] for item in subupdate_metrics
                )
                / metric_count,
                "embodied_flow_sde_grpo/active_previous_abs_delta_mean": sum(
                    item["active_previous_abs_delta_mean"] for item in subupdate_metrics
                )
                / metric_count,
                "embodied_flow_sde_grpo/active_previous_abs_delta_max": max(
                    item["active_previous_abs_delta_max"] for item in subupdate_metrics
                ),
                "embodied_flow_sde_grpo/active_previous_ratio_mean": sum(
                    item["active_previous_ratio_mean"] for item in subupdate_metrics
                )
                / metric_count,
                # The first subupdate measures rollout/rescore alignment before
                # any optimizer movement. The legacy ``previous_*`` metrics
                # average all subupdates and therefore measure policy drift.
                "embodied_flow_sde_grpo/pre_update_alignment_abs_delta_mean": (
                    subupdate_metrics[0]["previous_abs_delta_mean"]
                ),
                "embodied_flow_sde_grpo/pre_update_alignment_abs_delta_max": (
                    subupdate_metrics[0]["previous_abs_delta_max"]
                ),
                "embodied_flow_sde_grpo/pre_update_alignment_ratio_mean": (
                    subupdate_metrics[0]["previous_ratio_mean"]
                ),
                "embodied_flow_sde_grpo/pre_update_active_alignment_abs_delta_mean": (
                    subupdate_metrics[0]["active_previous_abs_delta_mean"]
                ),
                "embodied_flow_sde_grpo/pre_update_active_alignment_abs_delta_max": (
                    subupdate_metrics[0]["active_previous_abs_delta_max"]
                ),
                "embodied_flow_sde_grpo/pre_update_active_alignment_ratio_mean": (
                    subupdate_metrics[0]["active_previous_ratio_mean"]
                ),
                "embodied_flow_sde_grpo/optimization_old_policy_abs_delta_mean": (
                    sum(item["previous_abs_delta_mean"] for item in subupdate_metrics)
                    / metric_count
                ),
            },
        )

    def _train_subupdate(
        self,
        examples: list[FlowSDEExample],
        advantages: list[float],
        *,
        loss_denominator: int,
        length_normalized: bool,
        audit_rows: list[dict[str, torch.Tensor]] | None = None,
        check_pre_update_alignment: bool = False,
        subupdate_index: int = 0,
    ) -> dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        if not examples:
            self.optimizer.step()
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "reference_kl": 0.0,
                "reference_kl_loss": 0.0,
                "ratio_mean": 0.0,
                "approximate_kl": 0.0,
                "approximate_kl_per_primitive": 0.0,
                "clip_fraction": 0.0,
                "gradient_norm": 0.0,
                "microbatches": 0.0,
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
                "optimizer_step_applied": 1.0,
            }
        total_valid = sum(example.loss_mask for example in examples)
        totals = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "reference_kl": 0.0,
            "reference_kl_loss": 0.0,
            "ratio_mean": 0.0,
            "approximate_kl": 0.0,
            "approximate_kl_per_primitive": 0.0,
            "clip_fraction": 0.0,
        }
        microbatches = 0
        alignment_sum = 0.0
        alignment_count = 0
        alignment_max = 0.0
        alignment_ratio_sum = 0.0
        active_alignment_sum = 0.0
        active_alignment_count = 0
        active_alignment_max = 0.0
        active_alignment_ratio_sum = 0.0
        primitive_alignment_sum = 0.0
        primitive_alignment_count = 0
        primitive_alignment_max = 0.0
        active_primitive_alignment_sum = 0.0
        active_primitive_alignment_count = 0
        active_primitive_alignment_max = 0.0
        for batch, batch_advantages in _flow_sde_microbatches(
            examples,
            advantages,
            microbatch_size=self.microbatch_size,
        ):
            rollout = FlowSDERollout.concatenate(
                [example.rollout for example in batch]
            ).to(self.device)
            current = self.policy.flow_sde_logprobs(rollout).float()
            old = rollout.transition.old_logprobs[
                :, : current.shape[1], : current.shape[2]
            ].to(device=current.device, dtype=torch.float32)
            loss_mask = torch.as_tensor(
                [example.loss_mask for example in batch],
                device=current.device,
                dtype=torch.bool,
            )
            action_mask = _flow_sde_action_mask_tensor(
                batch,
                horizon=current.shape[1],
                action_dim=current.shape[2],
                device=current.device,
            )
            current_chunks = chunk_logprobs(current.detach(), action_mask=action_mask)
            old_chunks = chunk_logprobs(old, action_mask=action_mask)
            chunk_log_ratio = current_chunks - old_chunks
            chunk_delta = chunk_log_ratio.abs()
            primitive_delta = primitive_normalized_chunk_abs_delta(
                current.detach(), old, action_mask=action_mask
            )
            chunk_ratio = torch.exp(chunk_log_ratio)
            alignment_sum += float(chunk_delta.sum().cpu())
            alignment_count += int(chunk_delta.numel())
            alignment_max = max(alignment_max, float(chunk_delta.max().cpu()))
            alignment_ratio_sum += float(chunk_ratio.sum().cpu())
            primitive_alignment_sum += float(primitive_delta.sum().cpu())
            primitive_alignment_count += int(primitive_delta.numel())
            primitive_alignment_max = max(
                primitive_alignment_max, float(primitive_delta.max().cpu())
            )
            active_delta = chunk_delta[loss_mask]
            active_ratio = chunk_ratio[loss_mask]
            active_primitive_delta = primitive_delta[loss_mask]
            if active_delta.numel():
                active_alignment_sum += float(active_delta.sum().cpu())
                active_alignment_count += int(active_delta.numel())
                active_alignment_max = max(
                    active_alignment_max, float(active_delta.max().cpu())
                )
                active_alignment_ratio_sum += float(active_ratio.sum().cpu())
                active_primitive_alignment_sum += float(
                    active_primitive_delta.sum().cpu()
                )
                active_primitive_alignment_count += int(active_primitive_delta.numel())
                active_primitive_alignment_max = max(
                    active_primitive_alignment_max,
                    float(active_primitive_delta.max().cpu()),
                )
            batch_valid = int(loss_mask.count_nonzero())
            row_weights = None
            if length_normalized:
                row_weights = torch.as_tensor(
                    [
                        self.max_episode_steps / example.trajectory_primitive_steps
                        for example in batch
                    ],
                    device=current.device,
                    dtype=torch.float32,
                )
            policy_loss, objective = flow_sde_grpo_loss(
                current,
                old,
                torch.as_tensor(
                    batch_advantages,
                    device=current.device,
                    dtype=torch.float32,
                ),
                loss_mask=loss_mask,
                action_mask=action_mask,
                row_weights=row_weights,
                loss_denominator=loss_denominator,
                clip_epsilon_low=self.clip_epsilon_low,
                clip_epsilon_high=self.clip_epsilon_high,
                clip_ratio_c=self.clip_ratio_c,
            )
            reference_kl = torch.zeros((), device=current.device)
            reference_kl_metric = torch.zeros((), device=current.device)
            if self.reference_kl_coefficient > 0.0:
                with torch.no_grad():
                    reference = self.policy.flow_sde_reference_logprobs(rollout).float()
                reference_kl, reference_objective = flow_sde_reference_kl_loss(
                    current,
                    reference,
                    loss_mask=loss_mask,
                    action_mask=action_mask,
                    row_weights=row_weights,
                    loss_denominator=loss_denominator,
                )
                reference_kl_metric = reference_objective.mean_per_primitive
            loss = policy_loss + self.reference_kl_coefficient * reference_kl
            if audit_rows is not None:
                audit_rows.append(
                    {
                        "current_chunk_logprobs": current_chunks.cpu(),
                        "old_chunk_logprobs": old_chunks.cpu(),
                        "action_mask": action_mask.detach().cpu(),
                        "advantages": torch.as_tensor(
                            batch_advantages, dtype=torch.float32
                        ),
                        "loss_mask": loss_mask.detach().cpu(),
                        "trajectory_primitive_steps": torch.as_tensor(
                            [example.trajectory_primitive_steps for example in batch],
                            dtype=torch.float32,
                        ),
                    }
                )
            loss.backward()
            totals["loss"] += float(loss.detach().cpu())
            totals["policy_loss"] += float(policy_loss.detach().cpu())
            totals["reference_kl_loss"] += float(reference_kl.detach().cpu())
            metric_weight = batch_valid / max(total_valid, 1)
            totals["reference_kl"] += (
                float(reference_kl_metric.detach().cpu()) * metric_weight
            )
            totals["ratio_mean"] += (
                float(objective.ratio_mean.detach().cpu()) * metric_weight
            )
            totals["approximate_kl"] += (
                float(objective.approximate_kl.detach().cpu()) * metric_weight
            )
            totals["approximate_kl_per_primitive"] += (
                float(objective.approximate_kl_per_primitive.detach().cpu())
                * metric_weight
            )
            totals["clip_fraction"] += (
                float(objective.clip_fraction.detach().cpu()) * metric_weight
            )
            microbatches += 1
        alignment_mean = alignment_sum / alignment_count if alignment_count else 0.0
        alignment_ratio_mean = (
            alignment_ratio_sum / alignment_count if alignment_count else 1.0
        )
        active_alignment_mean = (
            active_alignment_sum / active_alignment_count
            if active_alignment_count
            else 0.0
        )
        active_alignment_ratio_mean = (
            active_alignment_ratio_sum / active_alignment_count
            if active_alignment_count
            else 1.0
        )
        primitive_alignment_mean = (
            primitive_alignment_sum / primitive_alignment_count
            if primitive_alignment_count
            else 0.0
        )
        active_primitive_alignment_mean = (
            active_primitive_alignment_sum / active_primitive_alignment_count
            if active_primitive_alignment_count
            else 0.0
        )
        tolerance = self.pre_update_logprob_kl_tolerance
        ratio_tolerance = self.pre_update_ratio_tolerance
        if check_pre_update_alignment and (
            tolerance is not None or ratio_tolerance is not None
        ):
            passed = (
                math.isfinite(active_primitive_alignment_mean)
                and math.isfinite(active_alignment_ratio_mean)
                and (tolerance is None or active_primitive_alignment_mean <= tolerance)
                and (
                    ratio_tolerance is None
                    or abs(active_alignment_ratio_mean - 1.0) <= ratio_tolerance
                )
            )
            emit_flow_sde_alignment(
                update=self.step + 1,
                mean_abs_delta=active_primitive_alignment_mean,
                max_abs_delta=active_primitive_alignment_max,
                tolerance=tolerance if tolerance is not None else float("inf"),
                ratio_mean=active_alignment_ratio_mean,
                ratio_tolerance=ratio_tolerance,
                passed=passed,
            )
            if not passed:
                self.optimizer.zero_grad(set_to_none=True)
                raise RuntimeError(
                    "Flow-SDE rollout/rescore alignment failed before optimizer "
                    "step: mean_abs_delta_per_primitive="
                    f"{active_primitive_alignment_mean:.6g}, "
                    f"tolerance={tolerance}, "
                    f"ratio_mean={active_alignment_ratio_mean:.6g}, "
                    f"ratio_tolerance={ratio_tolerance}"
                )
        if _should_stop_flow_sde_before_optimizer(
            self.training_schedule,
            subupdate_index=subupdate_index,
            approximate_kl=totals["approximate_kl"],
            approximate_kl_per_primitive=totals["approximate_kl_per_primitive"],
        ):
            self.optimizer.zero_grad(set_to_none=True)
            return {
                **totals,
                "gradient_norm": float(_gradient_norm(self.policy.parameters())),
                "microbatches": float(microbatches),
                "previous_abs_delta_mean": alignment_mean,
                "previous_abs_delta_max": alignment_max,
                "previous_ratio_mean": alignment_ratio_mean,
                "active_previous_abs_delta_mean": (active_alignment_mean),
                "active_previous_abs_delta_max": active_alignment_max,
                "active_previous_ratio_mean": (active_alignment_ratio_mean),
                "previous_abs_delta_per_primitive_mean": primitive_alignment_mean,
                "previous_abs_delta_per_primitive_max": primitive_alignment_max,
                "active_previous_abs_delta_per_primitive_mean": (
                    active_primitive_alignment_mean
                ),
                "active_previous_abs_delta_per_primitive_max": (
                    active_primitive_alignment_max
                ),
                "optimizer_step_applied": 0.0,
            }
        grad_norm = _gradient_norm(self.policy.parameters())
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in self.policy.parameters()
                    if parameter.requires_grad
                ],
                self.max_grad_norm,
            )
        self.optimizer.step()
        return {
            **totals,
            "gradient_norm": float(grad_norm),
            "microbatches": float(microbatches),
            "previous_abs_delta_mean": alignment_mean,
            "previous_abs_delta_max": alignment_max,
            "previous_ratio_mean": alignment_ratio_mean,
            "active_previous_abs_delta_mean": (active_alignment_mean),
            "active_previous_abs_delta_max": active_alignment_max,
            "active_previous_ratio_mean": (active_alignment_ratio_mean),
            "previous_abs_delta_per_primitive_mean": primitive_alignment_mean,
            "previous_abs_delta_per_primitive_max": primitive_alignment_max,
            "active_previous_abs_delta_per_primitive_mean": (
                active_primitive_alignment_mean
            ),
            "active_previous_abs_delta_per_primitive_max": (
                active_primitive_alignment_max
            ),
            "optimizer_step_applied": 1.0,
        }

    def _write_batch_audit(
        self,
        groups: list[EmbodiedTrajectoryGroup],
        *,
        examples: list[FlowSDEExample],
        advantages: list[float],
        kept_groups: int,
        subupdates: list[dict[str, Any]],
    ) -> None:
        assert self.diagnostics_dir is not None
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        rewards = torch.tensor(
            [
                [float(trajectory.reward) for trajectory in group.trajectories]
                for group in groups
            ],
            dtype=torch.float32,
        )
        group_advantages = group_relative_advantages(
            rewards,
            epsilon=self.advantage_epsilon,
            std_unbiased=self.advantage_std_unbiased,
        )
        payload = {
            "schema_version": 1,
            "update": self.step + 1,
            "group_size": self.group_size,
            "max_episode_steps": self.max_episode_steps,
            "clip_epsilon_low": self.clip_epsilon_low,
            "clip_epsilon_high": self.clip_epsilon_high,
            "clip_ratio_c": self.clip_ratio_c,
            "rewards": rewards,
            "group_advantages": group_advantages,
            "kept_groups": kept_groups,
            "example_coordinates": torch.tensor(
                [
                    [
                        example.group_index,
                        example.trajectory_index,
                        example.action_index,
                    ]
                    for example in examples
                ],
                dtype=torch.int64,
            ),
            "example_advantages": torch.tensor(advantages, dtype=torch.float32),
            "subupdates": subupdates,
        }
        destination = self.diagnostics_dir / f"update-{self.step + 1:06d}.pt"
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(destination)

    async def close(self) -> None:
        return None

    def _save_checkpoint(self) -> Path:
        destination = self.checkpoint_dir / f"step-{self.step:06d}"

        def writer(staging: Path) -> None:
            self.policy.save_checkpoint(str(staging / "policy"))
            torch.save(
                {
                    "step": self.step,
                    "optimizer": self.optimizer.state_dict(),
                },
                staging / "art_embodied_training_state.pt",
            )
            (staging / "art_embodied_training_state.json").write_text(
                json.dumps({"step": self.step}, indent=2) + "\n",
                encoding="utf-8",
            )

        checkpoint = self._checkpoint_manager.publish(
            destination,
            writer=writer,
            config_fingerprint=self.config_fingerprint,
            resume_contract_fingerprint=self.resume_contract_fingerprint,
            metadata={"backend": "flow_sde_grpo", "step": self.step},
        )
        self._prune_checkpoints()
        return checkpoint

    def _prune_checkpoints(self) -> None:
        checkpoints: list[tuple[int, Path]] = []
        for path in self.checkpoint_dir.glob("step-*"):
            try:
                step = int(path.name.removeprefix("step-"))
            except ValueError:
                continue
            checkpoints.append((step, path))
        checkpoints.sort()
        unprotected = [
            (step, path)
            for step, path in checkpoints
            if step not in self.retain_checkpoint_updates
        ]
        for _step, stale in unprotected[: -self.keep_last_checkpoints]:
            shutil.rmtree(stale)


def prepare_flow_sde_examples(
    groups: list[EmbodiedTrajectoryGroup],
    *,
    group_size: int,
    advantage_epsilon: float,
    advantage_std_unbiased: bool,
    filter_rewards: bool,
    rewards_lower_bound: float | None,
    rewards_upper_bound: float | None,
) -> tuple[list[FlowSDEExample], list[float], int]:
    """Freeze group advantages and expand trajectories into action-chunk rows."""

    if not groups:
        raise ValueError("Flow-SDE GRPO requires at least one trajectory group")
    rewards = []
    for group in groups:
        if len(group.trajectories) != group_size:
            raise ValueError(
                "Flow-SDE GRPO requires complete same-reset groups: "
                f"got {len(group.trajectories)}, expected {group_size}"
            )
        rewards.append([float(trajectory.reward) for trajectory in group.trajectories])
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    group_advantages = group_relative_advantages(
        reward_tensor,
        epsilon=advantage_epsilon,
        std_unbiased=advantage_std_unbiased,
    )

    examples: list[FlowSDEExample] = []
    advantages: list[float] = []
    kept_groups = 0
    for group_index, group in enumerate(groups):
        group_mean = float(reward_tensor[group_index].mean())
        keep_group = not filter_rewards or _in_reward_filter(
            group_mean,
            lower=rewards_lower_bound,
            upper=rewards_upper_bound,
        )
        kept_groups += int(keep_group)
        for trajectory_index, trajectory in enumerate(group.trajectories):
            advantage = float(group_advantages[group_index, trajectory_index])
            primitive_steps = _trajectory_primitive_steps(trajectory)
            for action_index, action in enumerate(trajectory.actions):
                if action.metadata.get(FLOW_SDE_REPLAY_SELECTED_KEY) is False:
                    continue
                rollout = action.metadata.get(TRANSIENT_FLOW_SDE_ROLLOUT_KEY)
                if not isinstance(rollout, FlowSDERollout):
                    raise ValueError(
                        "Continuous action is missing its FlowSDERollout record"
                    )
                if rollout.inputs.batch_size != 1:
                    raise ValueError(
                        "Each trajectory action must retain one Flow-SDE sample; "
                        "split group-batched samples with rollout.select(index)"
                    )
                examples.append(
                    FlowSDEExample(
                        rollout=rollout,
                        group_index=group_index,
                        trajectory_index=trajectory_index,
                        action_index=action_index,
                        task_key=trajectory.task,
                        reward=float(trajectory.reward),
                        loss_mask=keep_group,
                        trajectory_primitive_steps=primitive_steps,
                        action_mask=_flow_sde_action_mask_from_metadata(
                            action.metadata,
                            maximum_horizon=int(
                                rollout.transition.old_logprobs.shape[1]
                            ),
                        ),
                    )
                )
                advantages.append(advantage)
    return examples, advantages, kept_groups


def flow_sde_replay_selection_counts(
    groups: Iterable[EmbodiedTrajectoryGroup],
) -> tuple[int, int]:
    """Count eligible and retained Flow-SDE rows for observability."""

    eligible = 0
    selected = 0
    for group in groups:
        for trajectory in group.trajectories:
            summary = trajectory.metadata.get("training_action_selection")
            if isinstance(summary, dict):
                eligible += int(summary["eligible"])
                selected += int(summary["selected"])
                continue
            for action in trajectory.actions:
                metadata = action.metadata
                rollout = metadata.get(TRANSIENT_FLOW_SDE_ROLLOUT_KEY)
                if FLOW_SDE_REPLAY_SELECTED_KEY not in metadata and not isinstance(
                    rollout, FlowSDERollout
                ):
                    continue
                eligible += 1
                selected += int(
                    metadata.get(FLOW_SDE_REPLAY_SELECTED_KEY) is not False
                    and isinstance(rollout, FlowSDERollout)
                )
    if eligible < 1 or selected < 1 or selected > eligible:
        raise ValueError(
            "Flow-SDE replay selection must retain between one and all eligible rows"
        )
    return eligible, selected


def _collate_flow_sde_audit_rows(
    rows: list[dict[str, torch.Tensor]],
    *,
    subupdate_index: int,
    loss_denominator: int,
    length_normalized: bool,
    metrics: dict[str, float],
) -> dict[str, Any]:
    if not rows:
        empty = torch.empty(0, dtype=torch.float32)
        return {
            "index": subupdate_index,
            "loss_denominator": loss_denominator,
            "length_normalized": length_normalized,
            "current_chunk_logprobs": empty,
            "old_chunk_logprobs": empty,
            "advantages": empty,
            "loss_mask": torch.empty(0, dtype=torch.bool),
            "action_mask": torch.empty((0, 0), dtype=torch.bool),
            "trajectory_primitive_steps": empty,
            "art_metrics": metrics,
        }
    keys = (
        "current_chunk_logprobs",
        "old_chunk_logprobs",
        "advantages",
        "loss_mask",
        "action_mask",
        "trajectory_primitive_steps",
    )
    return {
        "index": subupdate_index,
        "loss_denominator": loss_denominator,
        "length_normalized": length_normalized,
        **{key: torch.cat([row[key] for row in rows]) for key in keys},
        "art_metrics": metrics,
    }


def _flow_sde_subupdates(
    examples: list[FlowSDEExample],
    advantages: list[float],
    *,
    group_count: int,
    group_size: int,
    optimizer_steps_per_update: int,
    schedule: Any,
    max_episode_steps: int,
) -> list[tuple[list[FlowSDEExample], list[float], int, bool]]:
    if len(examples) != len(advantages):
        raise ValueError("Flow-SDE examples and advantages must align")
    if not isinstance(schedule, RlinfActorBatchScheduleConfig):
        if optimizer_steps_per_update != 1:
            raise ValueError(
                "Flow-SDE full-update training supports one optimizer step; use "
                "training.schedule.type='rlinf_actor_global_batch' for subupdates"
            )
        valid = max(sum(example.loss_mask for example in examples), 1)
        return [(examples, advantages, valid, False)]

    policy_steps = (
        max_episode_steps + schedule.action_chunk_size - 1
    ) // schedule.action_chunk_size
    fixed_horizon_rows = group_count * group_size * policy_steps
    if optimizer_steps_per_update % schedule.update_epochs:
        raise ValueError(
            "Flow-SDE optimizer steps must divide evenly across update epochs"
        )
    optimizer_steps_per_epoch = optimizer_steps_per_update // schedule.update_epochs
    if schedule.strict_geometry:
        expected = optimizer_steps_per_update * schedule.global_batch_size
        scheduled_rows = fixed_horizon_rows * schedule.update_epochs
        if scheduled_rows != expected:
            raise ValueError(
                "Flow-SDE RLinf rows differ from optimizer geometry: "
                f"rows={scheduled_rows}, expected={expected}"
            )
    epoch_partitions = _partition_flow_sde_actor_batches(
        examples,
        group_count=group_count,
        group_size=group_size,
        policy_steps=policy_steps,
        optimizer_steps_per_update=optimizer_steps_per_epoch,
        schedule=schedule,
    )
    # RLinf shuffles the rank-local rollout tensor once, then reuses the same
    # global-batch ordering for each PPO/GRPO update epoch.
    partitions = [
        list(partition)
        for _ in range(schedule.update_epochs)
        for partition in epoch_partitions
    ]
    return [
        (
            [examples[index] for index in partition],
            [advantages[index] for index in partition],
            schedule.global_batch_size,
            True,
        )
        for partition in partitions
    ]


def _flow_sde_max_approximate_kl(schedule: Any) -> float | None:
    value = getattr(schedule, "max_approximate_kl", None)
    return None if value is None else float(value)


def _flow_sde_kl_guard_scope(schedule: Any) -> str:
    return str(getattr(schedule, "approximate_kl_guard_scope", "joint_chunk"))


def _should_stop_flow_sde_before_optimizer(
    schedule: Any,
    *,
    subupdate_index: int,
    approximate_kl: float,
    approximate_kl_per_primitive: float | None = None,
) -> bool:
    """Stop before applying a gradient once old-policy KL is already unsafe.

    The KL is measured while rescoring the current minibatch before its
    optimizer step. The zero-based index therefore equals the number of steps
    already applied in this rollout update.
    """

    threshold = _flow_sde_max_approximate_kl(schedule)
    if threshold is None:
        return False
    minimum_steps = int(getattr(schedule, "min_optimizer_steps_before_kl_stop", 1))
    if subupdate_index < minimum_steps:
        return False
    scope = _flow_sde_kl_guard_scope(schedule)
    measured = (
        approximate_kl if scope == "joint_chunk" else approximate_kl_per_primitive
    )
    if measured is None:
        raise ValueError(
            "primitive_action KL guard requires approximate_kl_per_primitive"
        )
    if not math.isfinite(measured):
        raise FloatingPointError(
            "Flow-SDE approximate KL became non-finite before optimizer step"
        )
    # A minibatch Monte Carlo estimate of KL can be negative even when the
    # policy has moved substantially. Both signs indicate unsafe old-policy
    # displacement for a per-subupdate guard.
    return abs(measured) > threshold


def _flow_sde_action_mask_from_metadata(
    metadata: dict[str, Any],
    *,
    maximum_horizon: int,
) -> FlowSDEActionMask:
    """Recover the environment-confirmed time and action-dimension mask.

    ``primitive_loss_mask`` identifies which chunk rows reached the environment.
    Some policies also predict redundant action representations that are decoded
    but never applied by the environment.  In that case the policy adapter records
    ``flow_sde_action_dimension_mask`` so trajectory reward cannot be assigned to
    those non-causal output dimensions.
    """

    raw = metadata.get("primitive_loss_mask")
    if raw is None:
        return tuple(True for _ in range(maximum_horizon))
    if not isinstance(raw, list | tuple):
        raise ValueError("primitive_loss_mask must be a list or tuple")
    mask = tuple(bool(value) for value in raw)
    if len(mask) > maximum_horizon:
        raise ValueError(
            "primitive_loss_mask exceeds the retained native Flow-SDE horizon: "
            f"mask={len(mask)}, retained={maximum_horizon}"
        )
    if not any(mask):
        raise ValueError("Flow-SDE action chunk contains no executed primitive action")
    first_false = next(
        (index for index, keep in enumerate(mask) if not keep), len(mask)
    )
    if any(mask[first_false:]):
        raise ValueError("primitive_loss_mask must describe one executed prefix")
    raw_dimensions = metadata.get(FLOW_SDE_ACTION_DIMENSION_MASK_KEY)
    if raw_dimensions is None:
        return mask
    if not isinstance(raw_dimensions, list | tuple):
        raise ValueError(
            f"{FLOW_SDE_ACTION_DIMENSION_MASK_KEY} must be a list or tuple"
        )
    dimension_mask = tuple(bool(value) for value in raw_dimensions)
    if not dimension_mask or not any(dimension_mask):
        raise ValueError(
            f"{FLOW_SDE_ACTION_DIMENSION_MASK_KEY} must select at least one dimension"
        )
    return tuple(
        tuple(bool(keep_step and keep_dimension) for keep_dimension in dimension_mask)
        for keep_step in mask
    )


def _flow_sde_action_mask_tensor(
    examples: list[FlowSDEExample],
    *,
    horizon: int,
    action_dim: int,
    device: torch.device | str,
) -> torch.Tensor:
    rows = []
    for example in examples:
        mask = example.action_mask
        if mask is None:
            mask = tuple(True for _ in range(horizon))
        if len(mask) != horizon:
            raise ValueError(
                "Flow-SDE example action mask does not match rescored horizon: "
                f"mask={len(mask)}, horizon={horizon}"
            )
        if mask and isinstance(mask[0], tuple):
            dimension_rows = mask
            if any(len(row) != action_dim for row in dimension_rows):
                raise ValueError(
                    "Flow-SDE element mask action dimension does not match rescore: "
                    f"expected={action_dim}"
                )
        rows.append(mask)
    return torch.as_tensor(rows, device=device, dtype=torch.bool)


def _partition_flow_sde_actor_batches(
    examples: list[FlowSDEExample],
    *,
    group_count: int,
    group_size: int,
    policy_steps: int,
    optimizer_steps_per_update: int,
    schedule: RlinfActorBatchScheduleConfig,
) -> list[list[int]]:
    groups_per_epoch = (
        schedule.actor_world_size * schedule.groups_per_process_per_rollout_epoch
    )
    if group_count % groups_per_epoch:
        raise ValueError("Flow-SDE groups do not fit the RLinf actor topology")
    rollout_epochs = group_count // groups_per_epoch
    local_trajectories = (
        rollout_epochs * schedule.groups_per_process_per_rollout_epoch * group_size
    )
    local_rows = local_trajectories * policy_steps
    if local_rows % optimizer_steps_per_update:
        raise ValueError("Rank-local Flow-SDE rows do not divide into subupdates")
    slots: list[list[int | None]] = [
        [None] * local_rows for _ in range(schedule.actor_world_size)
    ]
    for index, example in enumerate(examples):
        if example.action_index >= policy_steps:
            raise ValueError("Flow-SDE action_index exceeds fixed policy horizon")
        rollout_epoch, group_in_epoch = divmod(example.group_index, groups_per_epoch)
        rank, process_group = divmod(
            group_in_epoch, schedule.groups_per_process_per_rollout_epoch
        )
        local_group = (
            rollout_epoch * schedule.groups_per_process_per_rollout_epoch
            + process_group
        )
        local_trajectory = local_group * group_size + example.trajectory_index
        slot = example.action_index * local_trajectories + local_trajectory
        if slots[rank][slot] is not None:
            raise ValueError("Duplicate Flow-SDE example in one RLinf tensor slot")
        slots[rank][slot] = index

    rows_per_rank_batch = local_rows // optimizer_steps_per_update
    partitions: list[list[int]] = [[] for _ in range(optimizer_steps_per_update)]
    for rank, rank_slots in enumerate(slots):
        if schedule.rank_local_shuffle:
            generator = torch.Generator()
            generator.manual_seed(schedule.actor_seed + rank)
            order = torch.randperm(local_rows, generator=generator).tolist()
            shuffled = [rank_slots[index] for index in order]
        else:
            shuffled = rank_slots
        for subupdate in range(optimizer_steps_per_update):
            start = subupdate * rows_per_rank_batch
            stop = start + rows_per_rank_batch
            partitions[subupdate].extend(
                index for index in shuffled[start:stop] if index is not None
            )
    if sum(map(len, partitions)) != len(examples):
        raise ValueError("Flow-SDE RLinf partition did not consume every chunk")
    return partitions


def _trajectory_primitive_steps(trajectory: Any) -> int:
    total = 0
    for action in trajectory.actions:
        explicit = action.metadata.get("primitive_loss_mask_sum")
        if explicit is not None:
            count = int(explicit)
        else:
            rollout = action.metadata.get(TRANSIENT_FLOW_SDE_ROLLOUT_KEY)
            if not isinstance(rollout, FlowSDERollout):
                raise ValueError(
                    "Flow-SDE action lacks both primitive mask and rollout evidence"
                )
            count = int(rollout.transition.old_logprobs.shape[1])
        if count < 0:
            raise ValueError("primitive_loss_mask_sum cannot be negative")
        total += count
    if total < 1:
        raise ValueError("Flow-SDE trajectory contains no executed primitive actions")
    return total


def _in_reward_filter(
    value: float,
    *,
    lower: float | None,
    upper: float | None,
) -> bool:
    if lower is None or upper is None:
        raise ValueError("reward filtering requires lower and upper bounds")
    return lower <= value <= upper


def _gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    norms = [
        parameter.grad.detach().float().norm()
        for parameter in parameters
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not norms:
        return torch.tensor(0.0)
    return torch.stack(norms).norm()
