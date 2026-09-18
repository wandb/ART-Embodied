"""Utilities and a scaffold backend for action-token VLA training."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
from pathlib import Path
import time
from typing import Any

import pydantic

from art_embodied.backends.action_token_advantages import (
    _advantage_sign_balance_scale_tensor,
    _coerce_bool_sequence,
    _coerce_float_sequence,
    _example_advantage_sign_token_counts,
    _example_advantages,
    _example_has_token_advantages,
    _example_loss_denominator_count,
    _example_objective_token_count,
    _example_rlinf_chunk_mean_denominator_count,
    _example_rlinf_masked_mean_ratio_denominator_count,
    _example_rlinf_masked_mean_ratio_scale,
    _example_token_advantage_tensor,
    _example_token_loss_mask_tensor,
    _examples_advantage_sign_token_counts,
    _examples_have_prepared_scalar_advantages,
    _examples_have_token_advantages,
    _has_policy_gradient_signal,
    _positive_int_or_none,
)
from art_embodied.backends.action_token_gradients import (
    _clear_trainable_gradients,
    _clip_grad_norm,
    _gradient_metrics,
    _named_trainable_parameter_iter,
    _optimizer_parameter_lrs,
    _parameter_metric_role,
    _parameter_update_metrics,
    _restore_trainable_parameter_snapshot,
    _save_gradient_payload,
    _trainable_gradient_payload,
    _trainable_parameter_snapshot,
    apply_action_token_gradient_payloads,
    load_action_token_gradient_payload,
)
from art_embodied.backends.action_token_progress import (
    _maybe_write_action_token_rescore_progress,
    _maybe_write_action_token_train_microbatch_start,
    _maybe_write_action_token_train_progress,
    _write_action_token_train_progress_record,
)
from art_embodied.backends.action_token_ratios import action_chunk_log_ratio
from art_embodied.backends.loss_scaling import (
    backward_policy_loss,
    unscale_policy_gradients,
)
from art_embodied.checkpointing import CheckpointManager
from art_embodied.trajectories import (
    Action,
    EmbodiedTrajectory,
    EmbodiedTrajectoryGroup,
    Observation,
)
from art_embodied.types import LocalTrainResult, TrainResult
from art_embodied.utils import make_json_safe

ActionToken = int | str
TRANSIENT_RLINF_ENV_OBS_METADATA_KEY = "_art_embodied_transient_rlinf_env_obs"
TRANSIENT_RLINF_FORWARD_INPUTS_METADATA_KEY = (
    "_art_embodied_transient_rlinf_forward_inputs"
)
_CHECKPOINT_MANAGER = CheckpointManager()


class ActionTokenExample(pydantic.BaseModel):
    """A trainable action-token segment extracted from an embodied trajectory."""

    task: str
    trajectory_index: int
    action_index: int
    step: int
    tokens: list[ActionToken]
    reward: float
    prompt: str | None = None
    observation: Observation | None = None
    decoded_action: Any | None = None
    logprobs: list[float] | None = None
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)


def prepare_action_token_examples(
    trajectory_groups: Iterable[EmbodiedTrajectoryGroup],
    *,
    backend: "ActionTokenGRPOBackend",
    max_primitive_slots: int | None = None,
) -> tuple[list[ActionTokenExample], dict[str, Any]]:
    """Prepare one complete rollout update before optimizer partitioning.

    Group-relative advantages and reward filtering must be computed over the
    full rollout update. Splitting groups first silently changes the GRPO
    objective, so scheduled backends use this function exactly once and pass
    the resulting examples to each sequential optimizer sub-update.
    """

    groups = list(trajectory_groups)
    _validate_group_relative_groups(
        groups,
        backend_name="prepare_action_token_examples",
    )
    require_rollout_logprobs = not (
        backend.precalculate_logprobs and backend.training_unit == "action"
    )
    if backend.training_unit == "trajectory":
        examples = extract_trajectory_action_token_examples(
            groups,
            require_logprobs=True,
            require_observations=backend.require_observations,
            require_prompts=backend.require_prompts,
        )
    else:
        examples = extract_action_token_examples(
            groups,
            require_logprobs=require_rollout_logprobs,
            require_observations=backend.require_observations,
            require_prompts=backend.require_prompts,
            score_source=backend.rlinf_action_level_score_source,
        )
    if not examples:
        raise ValueError("No action-token examples found in rollout update")

    reward_filter_report = _reward_filter_report(
        examples,
        enabled=backend.filter_rewards,
        lower=backend.rewards_lower_bound,
        upper=backend.rewards_upper_bound,
    )
    if backend.filter_rewards:
        examples = _apply_reward_filter_to_examples(
            examples,
            reward_filter_report=reward_filter_report,
            mode=backend.reward_filter_mode,
        )
    if backend.action_advantage_mode == "rlinf_action_level_cumulative":
        _attach_rlinf_action_level_token_advantages(
            examples,
            normalize=backend.normalize_advantages,
            group_std_unbiased=backend.advantage_std_unbiased,
            score_source=backend.rlinf_action_level_score_source,
            max_primitive_slots=max_primitive_slots,
            eps=backend.advantage_epsilon,
        )
        if backend.rlinf_action_level_mask_zero_variance_groups:
            reward_filter_report = {
                **reward_filter_report,
                "zero_variance_group_mask": (
                    _mask_rlinf_action_level_zero_variance_groups(
                        examples, eps=backend.advantage_epsilon
                    )
                ),
            }
        if backend.rlinf_action_level_extra_global_normalization:
            _apply_rlinf_action_level_extra_global_normalization(
                examples,
                mean=backend.rlinf_action_level_global_advantage_mean,
                scale=backend.rlinf_action_level_global_advantage_scale,
            )
    return list(examples), reward_filter_report


class NoopActionTokenBackend:
    """A no-op backend that validates action-token training data.

    This is the last CPU-only scaffold before wiring an actual OpenVLA-style
    trainer. It proves that ART-Embodied can collect action-token rollouts,
    bind them to trajectory rewards, and expose stable metrics for a backend.
    """

    def __init__(
        self,
        *,
        path: str | Path | None = None,
        require_logprobs: bool = False,
        require_observations: bool = False,
        require_prompts: bool = False,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.require_logprobs = require_logprobs
        self.require_observations = require_observations
        self.require_prompts = require_prompts
        self.step = 0

    async def train(
        self,
        trajectory_groups: Iterable[EmbodiedTrajectoryGroup],
        **kwargs: Any,
    ) -> TrainResult:
        groups = list(trajectory_groups)
        examples = extract_action_token_examples(
            groups,
            require_logprobs=self.require_logprobs,
            require_observations=self.require_observations,
            require_prompts=self.require_prompts,
        )
        self.step += 1

        token_counts = [float(len(example.tokens)) for example in examples]
        rewards = [example.reward for example in examples]
        metrics = {
            "embodied_action_token/groups": float(len(groups)),
            "embodied_action_token/examples": float(len(examples)),
            "embodied_action_token/tokens_mean": _mean(token_counts),
            "embodied_action_token/tokens_total": float(sum(token_counts)),
            "embodied_action_token/reward_mean": _mean(rewards),
            "embodied_action_token/with_logprobs": float(
                sum(1 for example in examples if example.logprobs is not None)
            ),
            "embodied_action_token/with_observations": float(
                sum(1 for example in examples if example.observation is not None)
            ),
            "embodied_action_token/with_prompts": float(
                sum(1 for example in examples if example.prompt is not None)
            ),
        }
        metrics.update(
            {key: float(value) for key, value in kwargs.items() if _is_number(value)}
        )

        checkpoint_path = None
        if self.path is not None:
            checkpoint_dir = self.path / f"noop-action-token-step-{self.step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / "README.txt").write_text(
                "NoopActionTokenBackend validates action-token examples only.\n",
                encoding="utf-8",
            )
            metrics["embodied_action_token/noop_checkpoint_written"] = 1.0
            checkpoint_path = str(checkpoint_dir)

        return LocalTrainResult(
            step=self.step,
            metrics=metrics,
            checkpoint_path=checkpoint_path,
        )

    async def close(self) -> None:
        return None


class ActionTokenGRPOBackend:
    """GRPO/GSPO-style policy-gradient backend for action-token trajectories.

    This backend is intentionally narrow and honest: it only supports policies
    that can recompute log-probabilities for the action tokens observed during
    rollout. That is the minimum contract needed for ART-style policy-gradient
    RL; reward-weighted behavior cloning should use a different backend.

    ``importance_sampling_level="token"`` uses token-level ratios, matching a
    compact GRPO-style objective. ``"sequence"`` uses a length-normalized
    sequence-level ratio (GSPO). Experimental ``"action_chunk"`` clips the
    unnormalized joint ratio of one sampled action chunk, while retaining a
    trajectory-relative advantage and a shared trajectory-count denominator.
    """

    def __init__(
        self,
        policy: Any,
        optimizer: Any | None = None,
        *,
        device: str | None = None,
        lr: float = 1e-5,
        optimizer_weight_decay: float = 0.0,
        optimizer_adam_beta1: float = 0.9,
        optimizer_adam_beta2: float = 0.999,
        optimizer_adam_eps: float = 1.0e-8,
        clip_epsilon: float = 0.2,
        clip_epsilon_low: float | None = None,
        clip_epsilon_high: float | None = None,
        clip_ratio_c: float | None = None,
        kl_coef: float = 0.0,
        normalize_advantages: bool = False,
        advantage_normalization_scope: str = "global",
        advantage_std_unbiased: bool = False,
        advantage_epsilon: float = 1.0e-8,
        filter_rewards: bool = False,
        reward_filter_mode: str = "drop_examples",
        rewards_lower_bound: float | None = None,
        rewards_upper_bound: float | None = None,
        importance_sampling_level: str = "token",
        max_grad_norm: float | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_config_fingerprint: str | None = None,
        checkpoint_resume_contract_fingerprint: str | None = None,
        require_observations: bool = False,
        require_prompts: bool = False,
        require_reference_logprobs: bool = False,
        reference_logprob_l2_coef: float = 0.0,
        training_unit: str = "action",
        action_advantage_mode: str = "example",
        rlinf_action_level_extra_global_normalization: bool = False,
        rlinf_action_level_mask_zero_variance_groups: bool = False,
        rlinf_action_level_score_source: str = "chunk_rewards",
        rlinf_action_level_global_advantage_mean: float | None = None,
        rlinf_action_level_global_advantage_scale: float | None = None,
        loss_aggregation: str = "trajectory_mean",
        precalculate_logprobs: bool = False,
        rollout_logprob_source: str = "rollout_action",
        logprob_eval_mode: bool = True,
        logprob_microbatch_size: int | None = None,
        train_logprob_microbatch_size: int | None = None,
        policy_step_loss_weights: Mapping[Any, Any] | None = None,
        action_dim_loss_weights: Mapping[Any, Any] | None = None,
        pre_update_logprob_kl_tolerance: float | None = None,
        pre_update_ratio_tolerance: float | None = None,
        skip_optimizer_step_without_policy_gradient_signal: bool = True,
        progress_path: str | Path | None = None,
        progress_every_microbatches: int = 128,
        progress_max_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self.policy = policy
        self.optimizer = optimizer
        self.device = device or _infer_policy_device(policy) or "cpu"
        self.lr = float(lr)
        self.optimizer_weight_decay = float(optimizer_weight_decay)
        self.optimizer_adam_beta1 = float(optimizer_adam_beta1)
        self.optimizer_adam_beta2 = float(optimizer_adam_beta2)
        self.optimizer_adam_eps = float(optimizer_adam_eps)
        self.clip_epsilon = float(clip_epsilon)
        self.clip_epsilon_low = float(
            clip_epsilon if clip_epsilon_low is None else clip_epsilon_low
        )
        self.clip_epsilon_high = float(
            clip_epsilon if clip_epsilon_high is None else clip_epsilon_high
        )
        if self.clip_epsilon_low < 0.0 or self.clip_epsilon_high < 0.0:
            raise ValueError(
                "clip_epsilon_low and clip_epsilon_high must be non-negative"
            )
        self.clip_ratio_c = float(clip_ratio_c) if clip_ratio_c is not None else None
        if self.clip_ratio_c is not None and self.clip_ratio_c <= 1.0:
            raise ValueError("clip_ratio_c must be greater than 1.0 when set")
        self.kl_coef = float(kl_coef)
        self.normalize_advantages = bool(normalize_advantages)
        if advantage_normalization_scope not in ("global", "group"):
            raise ValueError(
                "advantage_normalization_scope must be 'global' or 'group'"
            )
        self.advantage_normalization_scope = advantage_normalization_scope
        self.advantage_std_unbiased = bool(advantage_std_unbiased)
        self.advantage_epsilon = float(advantage_epsilon)
        if self.advantage_epsilon <= 0.0:
            raise ValueError("advantage_epsilon must be positive")
        self.filter_rewards = bool(filter_rewards)
        if reward_filter_mode not in ("drop_examples", "loss_mask"):
            raise ValueError(
                "reward_filter_mode must be 'drop_examples' or 'loss_mask'"
            )
        self.reward_filter_mode = reward_filter_mode
        self.rewards_lower_bound = (
            float(rewards_lower_bound) if rewards_lower_bound is not None else None
        )
        self.rewards_upper_bound = (
            float(rewards_upper_bound) if rewards_upper_bound is not None else None
        )
        if self.filter_rewards:
            if self.rewards_lower_bound is None or self.rewards_upper_bound is None:
                raise ValueError(
                    "filter_rewards=True requires rewards_lower_bound and rewards_upper_bound"
                )
            if self.rewards_lower_bound > self.rewards_upper_bound:
                raise ValueError("rewards_lower_bound must be <= rewards_upper_bound")
        if importance_sampling_level not in ("token", "sequence", "action_chunk"):
            raise ValueError(
                "importance_sampling_level must be 'token', 'sequence', or 'action_chunk'"
            )
        self.importance_sampling_level = importance_sampling_level
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir is not None else None
        )
        self.checkpoint_config_fingerprint = checkpoint_config_fingerprint
        self.checkpoint_resume_contract_fingerprint = (
            checkpoint_resume_contract_fingerprint
        )
        self.require_observations = require_observations
        self.require_prompts = require_prompts
        self.require_reference_logprobs = bool(require_reference_logprobs)
        self.reference_logprob_l2_coef = float(reference_logprob_l2_coef)
        if training_unit not in ("action", "trajectory"):
            raise ValueError("training_unit must be 'action' or 'trajectory'")
        self.training_unit = training_unit
        if action_advantage_mode not in ("example", "rlinf_action_level_cumulative"):
            raise ValueError(
                "action_advantage_mode must be 'example' or "
                "'rlinf_action_level_cumulative'"
            )
        if (
            action_advantage_mode == "rlinf_action_level_cumulative"
            and training_unit != "action"
        ):
            raise ValueError(
                "action_advantage_mode='rlinf_action_level_cumulative' requires "
                "training_unit='action'"
            )
        self.action_advantage_mode = action_advantage_mode
        self.rlinf_action_level_extra_global_normalization = bool(
            rlinf_action_level_extra_global_normalization
        )
        self.rlinf_action_level_mask_zero_variance_groups = bool(
            rlinf_action_level_mask_zero_variance_groups
        )
        if rlinf_action_level_score_source not in (
            "chunk_rewards",
            "trajectory_reward",
        ):
            raise ValueError(
                "rlinf_action_level_score_source must be 'chunk_rewards' or "
                "'trajectory_reward'"
            )
        self.rlinf_action_level_score_source = str(rlinf_action_level_score_source)
        if (
            self.rlinf_action_level_extra_global_normalization
            and action_advantage_mode != "rlinf_action_level_cumulative"
        ):
            raise ValueError(
                "rlinf_action_level_extra_global_normalization requires "
                "action_advantage_mode='rlinf_action_level_cumulative'"
            )
        if (
            self.rlinf_action_level_mask_zero_variance_groups
            and action_advantage_mode != "rlinf_action_level_cumulative"
        ):
            raise ValueError(
                "rlinf_action_level_mask_zero_variance_groups requires "
                "action_advantage_mode='rlinf_action_level_cumulative'"
            )
        self.rlinf_action_level_global_advantage_mean = (
            float(rlinf_action_level_global_advantage_mean)
            if rlinf_action_level_global_advantage_mean is not None
            else None
        )
        self.rlinf_action_level_global_advantage_scale = (
            float(rlinf_action_level_global_advantage_scale)
            if rlinf_action_level_global_advantage_scale is not None
            else None
        )
        if (self.rlinf_action_level_global_advantage_mean is None) != (
            self.rlinf_action_level_global_advantage_scale is None
        ):
            raise ValueError(
                "rlinf_action_level_global_advantage_mean and "
                "rlinf_action_level_global_advantage_scale must be provided together"
            )
        if (
            self.rlinf_action_level_global_advantage_scale is not None
            and self.rlinf_action_level_global_advantage_scale <= 0.0
        ):
            raise ValueError(
                "rlinf_action_level_global_advantage_scale must be positive"
            )
        if loss_aggregation not in (
            "trajectory_mean",
            "seq_mean_token_sum",
            "task_balanced_trajectory_mean",
            "token_mean",
            "rlinf_token_mean",
            "rlinf_chunk_mean",
            "rlinf_masked_mean_ratio",
            "advantage_sign_balanced_token_mean",
        ):
            raise ValueError(
                "loss_aggregation must be 'trajectory_mean', "
                "'seq_mean_token_sum', 'task_balanced_trajectory_mean', 'token_mean', "
                "'rlinf_token_mean', 'rlinf_chunk_mean', "
                "'rlinf_masked_mean_ratio', or 'advantage_sign_balanced_token_mean'"
            )
        if (
            loss_aggregation == "advantage_sign_balanced_token_mean"
            and importance_sampling_level != "token"
        ):
            raise ValueError(
                "loss_aggregation='advantage_sign_balanced_token_mean' requires "
                "importance_sampling_level='token'"
            )
        self.loss_aggregation = loss_aggregation
        if self.loss_aggregation == "task_balanced_trajectory_mean" and (
            self.importance_sampling_level != "token" or self.training_unit != "action"
        ):
            raise ValueError(
                "task_balanced_trajectory_mean currently requires "
                "importance_sampling_level='token' and training_unit='action'"
            )
        self.precalculate_logprobs = bool(precalculate_logprobs)
        if rollout_logprob_source not in (
            "rollout_action",
            "recomputed_current_policy",
        ):
            raise ValueError(
                "rollout_logprob_source must be 'rollout_action' or "
                "'recomputed_current_policy'"
            )
        self.rollout_logprob_source = str(rollout_logprob_source)
        if (
            self.precalculate_logprobs
            and self.rollout_logprob_source != "recomputed_current_policy"
        ):
            raise ValueError(
                "precalculate_logprobs requires recomputed_current_policy source"
            )
        self.logprob_eval_mode = bool(logprob_eval_mode)
        if logprob_microbatch_size is not None and int(logprob_microbatch_size) <= 0:
            raise ValueError(
                "logprob_microbatch_size must be a positive integer or None"
            )
        self.logprob_microbatch_size = (
            int(logprob_microbatch_size)
            if logprob_microbatch_size is not None
            else None
        )
        if (
            train_logprob_microbatch_size is not None
            and int(train_logprob_microbatch_size) <= 0
        ):
            raise ValueError(
                "train_logprob_microbatch_size must be a positive integer or None"
            )
        self.train_logprob_microbatch_size = (
            int(train_logprob_microbatch_size)
            if train_logprob_microbatch_size is not None
            else None
        )
        self.policy_step_loss_weights = _normalize_bucket_loss_weights(
            policy_step_loss_weights,
            key_prefix="policy_step_",
            label="policy_step_loss_weights",
        )
        self.action_dim_loss_weights = _normalize_bucket_loss_weights(
            action_dim_loss_weights,
            key_prefix="action_dim_",
            label="action_dim_loss_weights",
        )
        if self.importance_sampling_level == "action_chunk" and (
            self.training_unit != "action"
            or self.action_advantage_mode != "example"
            or rlinf_action_level_score_source != "trajectory_reward"
            or self.loss_aggregation != "seq_mean_token_sum"
            or self.filter_rewards
            or self.kl_coef
            or self.reference_logprob_l2_coef
            or self.policy_step_loss_weights
            or self.action_dim_loss_weights
            or rlinf_action_level_mask_zero_variance_groups
        ):
            raise ValueError(
                "action_chunk requires action examples with trajectory_reward, "
                "seq_mean_token_sum, scalar advantages, and no filters, KL penalties, "
                "or bucket weights"
            )
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
        self.skip_optimizer_step_without_policy_gradient_signal = bool(
            skip_optimizer_step_without_policy_gradient_signal
        )
        self.progress_path = Path(progress_path) if progress_path is not None else None
        self.progress_every_microbatches = int(progress_every_microbatches)
        self.progress_max_bytes = int(progress_max_bytes)
        if self.progress_every_microbatches <= 0:
            raise ValueError("progress_every_microbatches must be positive")
        if self.progress_max_bytes <= 0:
            raise ValueError("progress_max_bytes must be positive")
        self.step = 0

    async def train(
        self,
        trajectory_groups: Iterable[EmbodiedTrajectoryGroup],
        **kwargs: Any,
    ) -> TrainResult:
        gradient_only = bool(kwargs.pop("_action_token_grpo_return_gradients", False))
        gradient_output_path = kwargs.pop(
            "_action_token_grpo_gradient_output_path", None
        )
        precomputed_examples = kwargs.pop(
            "_action_token_grpo_precomputed_examples", None
        )
        precomputed_examples_prepared = bool(
            kwargs.pop("_action_token_grpo_precomputed_examples_prepared", False)
        )
        precomputed_reward_filter_report = kwargs.pop(
            "_action_token_grpo_precomputed_reward_filter_report",
            None,
        )
        gradient_direction_probe_cfg = kwargs.pop(
            "_action_token_grpo_gradient_direction_probe",
            None,
        )
        gradient_direction_probe_presence_metrics = (
            _gradient_direction_probe_presence_metrics(gradient_direction_probe_cfg)
        )
        global_example_count = kwargs.pop(
            "_action_token_grpo_global_example_count", None
        )
        global_token_count = kwargs.pop("_action_token_grpo_global_token_count", None)
        global_positive_token_count = kwargs.pop(
            "_action_token_grpo_global_positive_token_count", None
        )
        global_negative_token_count = kwargs.pop(
            "_action_token_grpo_global_negative_token_count", None
        )
        groups = list(trajectory_groups)
        if precomputed_examples is None:
            _validate_group_relative_groups(
                groups, backend_name="ActionTokenGRPOBackend"
            )
        if self.precalculate_logprobs and precomputed_examples is None:
            if not groups:
                raise ValueError("Old-logprob rescore requires trajectory groups")
            mode_owner = getattr(self.policy, "model", None)
            if mode_owner is None:
                mode_owner = self.policy
            previous_training = bool(getattr(mode_owner, "training", False))
            rescore_started = time.perf_counter()
            import torch

            try:
                _set_eval(self.policy) if self.logprob_eval_mode else _set_train(
                    self.policy
                )
                microbatch_size = int(
                    self.train_logprob_microbatch_size
                    or self.logprob_microbatch_size
                    or 1
                )
                # Match the autograd-enabled policy-gradient forward exactly.
                # The rescorer detaches every row before storing it.
                with torch.enable_grad():
                    rescore_report = refresh_action_token_logprobs(
                        self.policy,
                        groups,
                        device=self.device,
                        training_unit=self.training_unit,
                        microbatch_size=microbatch_size,
                        pad_to_batch_size=microbatch_size,
                        require_observations=self.require_observations,
                        require_prompts=self.require_prompts,
                        source=self.rollout_logprob_source,
                    )
            finally:
                _set_train(self.policy) if previous_training else _set_eval(self.policy)
            rescore_report["elapsed_seconds"] = time.perf_counter() - rescore_started
            for key, value in rescore_report.items():
                if _is_number(value):
                    kwargs[
                        "embodied_action_token_grpo/old_logprob_rescore_" + str(key)
                    ] = float(value)
        if precomputed_examples is None:
            if self.training_unit == "trajectory":
                examples = extract_trajectory_action_token_examples(
                    groups,
                    require_logprobs=True,
                    require_observations=self.require_observations,
                    require_prompts=self.require_prompts,
                )
            else:
                examples = extract_action_token_examples(
                    groups,
                    require_logprobs=True,
                    require_observations=self.require_observations,
                    require_prompts=self.require_prompts,
                    score_source=self.rlinf_action_level_score_source,
                )
        else:
            examples = list(precomputed_examples)
            if not precomputed_examples_prepared:
                raise ValueError(
                    "Precomputed action-token examples must be prepared with "
                    "reward filtering and RLinf action-level advantages before "
                    "calling ActionTokenGRPOBackend.train"
                )
        if not examples:
            raise ValueError(
                "No action-token examples found for ActionTokenGRPOBackend"
            )
        if precomputed_examples is None:
            reward_filter_report = _reward_filter_report(
                examples,
                enabled=self.filter_rewards,
                lower=self.rewards_lower_bound,
                upper=self.rewards_upper_bound,
            )
            if self.filter_rewards:
                examples = _apply_reward_filter_to_examples(
                    examples,
                    reward_filter_report=reward_filter_report,
                    mode=self.reward_filter_mode,
                )
        else:
            reward_filter_report = (
                precomputed_reward_filter_report
                if isinstance(precomputed_reward_filter_report, dict)
                else {
                    "enabled": self.filter_rewards,
                    "examples_before": len(examples),
                    "examples_after": len(examples),
                    "groups_total": 0,
                    "groups_kept": 0,
                    "groups_filtered": 0,
                }
            )
        if not examples:
            _clear_trainable_gradients(self.policy)
            gradient_payload = None
            if gradient_only:
                gradient_payload = _trainable_gradient_payload(self.policy)
                if gradient_output_path is not None:
                    _save_gradient_payload(
                        gradient_payload, Path(str(gradient_output_path))
                    )
            metrics = _empty_gradient_handoff_metrics(
                groups=groups,
                reward_filter_report=reward_filter_report,
                global_example_count=global_example_count,
                global_token_count=global_token_count,
                microbatch_size=self.logprob_microbatch_size or 1,
                logprob_eval_mode=self.logprob_eval_mode,
                gradient_payload=gradient_payload,
                gradient_handoff_worker=gradient_only,
                policy=self.policy,
            )
            metrics.update(
                {
                    key: float(value)
                    for key, value in kwargs.items()
                    if _is_number(value)
                }
            )
            if (
                _policy_gradient_algorithm_name(self.importance_sampling_level)
                == "gspo"
            ):
                metrics.update(
                    _alias_metric_prefix(
                        metrics,
                        source_prefix="embodied_action_token_grpo/",
                        target_prefix="embodied_action_token_gspo/",
                        drop_source=not gradient_only,
                    )
                )
            return LocalTrainResult(
                step=self.step, metrics=metrics, checkpoint_path=None
            )
        if (
            precomputed_examples is None
            and self.action_advantage_mode == "rlinf_action_level_cumulative"
        ):
            _attach_rlinf_action_level_token_advantages(
                examples,
                normalize=self.normalize_advantages,
                group_std_unbiased=self.advantage_std_unbiased,
                score_source=self.rlinf_action_level_score_source,
                eps=self.advantage_epsilon,
            )
            if self.rlinf_action_level_mask_zero_variance_groups:
                _mask_rlinf_action_level_zero_variance_groups(
                    examples, eps=self.advantage_epsilon
                )
            if self.rlinf_action_level_extra_global_normalization:
                _apply_rlinf_action_level_extra_global_normalization(
                    examples,
                    mean=self.rlinf_action_level_global_advantage_mean,
                    scale=self.rlinf_action_level_global_advantage_scale,
                )

        self._ensure_optimizer()
        if self.logprob_eval_mode:
            # Gradients still flow in eval mode; this keeps rollout-time and
            # train-time logprob conventions aligned by disabling dropout.
            _set_eval(self.policy)
        else:
            _set_train(self.policy)
        self.optimizer.zero_grad(set_to_none=True)

        import torch

        train_wall_start = time.perf_counter()
        logprob_forward_seconds = 0.0
        loss_backward_seconds = 0.0
        parameter_snapshot_seconds = 0.0
        optimizer_step_seconds = 0.0
        has_token_level_advantages = _examples_have_token_advantages(examples)
        has_prepared_scalar_advantages = _examples_have_prepared_scalar_advantages(
            examples
        )
        train_microbatch_size = (
            self.train_logprob_microbatch_size
            or self.logprob_microbatch_size
            or len(examples)
        )
        advantages = _example_advantages(
            examples,
            normalize=(
                self.normalize_advantages
                and not has_token_level_advantages
                and not has_prepared_scalar_advantages
            ),
            scope=self.advantage_normalization_scope,
            std_unbiased=self.advantage_std_unbiased,
            eps=self.advantage_epsilon,
            device=self.device,
        )
        has_policy_gradient_signal = _has_policy_gradient_signal(
            examples,
            advantages,
            device=self.device,
        )
        if (
            not has_policy_gradient_signal
            and self.skip_optimizer_step_without_policy_gradient_signal
        ):
            token_counts = [
                float(_example_objective_token_count(example)) for example in examples
            ]
            metrics = _grpo_metrics(
                groups=groups,
                examples=examples,
                loss=0.0,
                advantages=advantages.detach().cpu().tolist(),
                ratios=[],
                approx_kls=[],
                reference_l2_penalties=[],
                clip_hits=[],
                token_counts=token_counts,
                normalize_advantages=self.normalize_advantages,
                advantage_normalization_scope=self.advantage_normalization_scope,
                advantage_std_unbiased=self.advantage_std_unbiased,
                reward_filter_report=reward_filter_report,
                importance_sampling_level=self.importance_sampling_level,
                training_unit=self.training_unit,
                loss_aggregation=self.loss_aggregation,
                kl_coef=self.kl_coef,
                clip_epsilon=self.clip_epsilon,
                clip_epsilon_low=self.clip_epsilon_low,
                clip_epsilon_high=self.clip_epsilon_high,
                reference_logprob_l2_coef=self.reference_logprob_l2_coef,
            )
            if gradient_only:
                gradient_payload = _trainable_gradient_payload(self.policy)
                if gradient_output_path is not None:
                    _save_gradient_payload(
                        gradient_payload, Path(str(gradient_output_path))
                    )
                metrics["embodied_action_token_grpo/gradient_handoff_worker"] = 1.0
                metrics["embodied_action_token_grpo/gradient_handoff_tensors"] = float(
                    len(gradient_payload["gradients"])
                )
                metrics[
                    "embodied_action_token_grpo/global_loss_denominator_examples"
                ] = float(global_example_count or len(examples))
                metrics["embodied_action_token_grpo/global_loss_denominator_tokens"] = (
                    float(global_token_count or sum(token_counts))
                )
            metrics.update(_gradient_metrics(self.policy))
            metrics.update(gradient_direction_probe_presence_metrics)
            metrics["embodied_action_token_grpo/optimizer_step_completed"] = 0.0
            metrics[
                "embodied_action_token_grpo/optimizer_step_skipped_no_group_signal"
            ] = 1.0
            metrics[
                "embodied_action_token_grpo/optimizer_step_skipped_logprob_misalignment"
            ] = 0.0
            metrics["embodied_action_token_grpo/logprob_eval_mode"] = float(
                self.logprob_eval_mode
            )
            metrics["embodied_action_token_grpo/logprob_microbatch_size"] = float(
                train_microbatch_size
            )
            metrics["embodied_action_token_grpo/rollout_logprob_microbatch_size"] = (
                float(self.logprob_microbatch_size or len(examples))
            )
            metrics["embodied_action_token_grpo/train_logprob_microbatch_size"] = float(
                train_microbatch_size
            )
            metrics["embodied_action_token_grpo/logprob_microbatches"] = 0.0
            metrics["embodied_action_token_grpo/policy_parameters_updated"] = 0.0
            metrics.update(
                {
                    key: float(value)
                    for key, value in kwargs.items()
                    if _is_number(value)
                }
            )

            algorithm_name = _policy_gradient_algorithm_name(
                self.importance_sampling_level
            )
            if algorithm_name == "gspo":
                metrics.update(
                    _alias_metric_prefix(
                        metrics,
                        source_prefix="embodied_action_token_grpo/",
                        target_prefix="embodied_action_token_gspo/",
                        drop_source=not gradient_only,
                    )
                )
            return LocalTrainResult(
                step=self.step, metrics=metrics, checkpoint_path=None
            )

        alignment_guard = _maybe_guard_pre_update_logprob_alignment(
            policy=self.policy,
            examples=examples,
            device=self.device,
            importance_sampling_level=self.importance_sampling_level,
            training_unit=self.training_unit,
            loss_aggregation=self.loss_aggregation,
            microbatch_size=train_microbatch_size,
            kl_tolerance=self.pre_update_logprob_kl_tolerance,
            ratio_tolerance=self.pre_update_ratio_tolerance,
        )
        if (
            alignment_guard is not None
            and not alignment_guard["old_new_logprobs_aligned"]
        ):
            token_counts = [
                float(_example_objective_token_count(example)) for example in examples
            ]
            metrics = _grpo_metrics(
                groups=groups,
                examples=examples,
                loss=0.0,
                advantages=advantages.detach().cpu().tolist(),
                ratios=[],
                approx_kls=[],
                reference_l2_penalties=[],
                clip_hits=[],
                token_counts=token_counts,
                normalize_advantages=self.normalize_advantages,
                advantage_normalization_scope=self.advantage_normalization_scope,
                advantage_std_unbiased=self.advantage_std_unbiased,
                reward_filter_report=reward_filter_report,
                importance_sampling_level=self.importance_sampling_level,
                training_unit=self.training_unit,
                loss_aggregation=self.loss_aggregation,
                kl_coef=self.kl_coef,
                clip_epsilon=self.clip_epsilon,
                clip_epsilon_low=self.clip_epsilon_low,
                clip_epsilon_high=self.clip_epsilon_high,
                reference_logprob_l2_coef=self.reference_logprob_l2_coef,
            )
            _add_alignment_guard_metrics(
                metrics,
                alignment_guard,
                prefix="embodied_action_token_grpo",
            )
            if gradient_only:
                gradient_payload = _trainable_gradient_payload(self.policy)
                if gradient_output_path is not None:
                    _save_gradient_payload(
                        gradient_payload, Path(str(gradient_output_path))
                    )
                metrics["embodied_action_token_grpo/gradient_handoff_worker"] = 1.0
                metrics["embodied_action_token_grpo/gradient_handoff_tensors"] = float(
                    len(gradient_payload["gradients"])
                )
                metrics[
                    "embodied_action_token_grpo/global_loss_denominator_examples"
                ] = float(global_example_count or len(examples))
                metrics["embodied_action_token_grpo/global_loss_denominator_tokens"] = (
                    float(global_token_count or sum(token_counts))
                )
            metrics.update(_gradient_metrics(self.policy))
            metrics.update(gradient_direction_probe_presence_metrics)
            metrics["embodied_action_token_grpo/optimizer_step_completed"] = 0.0
            metrics[
                "embodied_action_token_grpo/optimizer_step_skipped_no_group_signal"
            ] = 0.0
            metrics[
                "embodied_action_token_grpo/optimizer_step_skipped_logprob_misalignment"
            ] = 1.0
            metrics["embodied_action_token_grpo/logprob_eval_mode"] = float(
                self.logprob_eval_mode
            )
            metrics["embodied_action_token_grpo/logprob_microbatch_size"] = float(
                train_microbatch_size
            )
            metrics["embodied_action_token_grpo/rollout_logprob_microbatch_size"] = (
                float(self.logprob_microbatch_size or len(examples))
            )
            metrics["embodied_action_token_grpo/train_logprob_microbatch_size"] = float(
                train_microbatch_size
            )
            metrics["embodied_action_token_grpo/logprob_microbatches"] = float(
                alignment_guard.get("logprob_microbatches", 0.0)
            )
            metrics["embodied_action_token_grpo/policy_parameters_updated"] = 0.0
            metrics.update(
                {
                    key: float(value)
                    for key, value in kwargs.items()
                    if _is_number(value)
                }
            )

            algorithm_name = _policy_gradient_algorithm_name(
                self.importance_sampling_level
            )
            if algorithm_name == "gspo":
                metrics.update(
                    _alias_metric_prefix(
                        metrics,
                        source_prefix="embodied_action_token_grpo/",
                        target_prefix="embodied_action_token_gspo/",
                        drop_source=not gradient_only,
                    )
                )
            return LocalTrainResult(
                step=self.step, metrics=metrics, checkpoint_path=None
            )

        losses = []
        ratios = []
        approx_kls = []
        reference_l2_penalties = []
        clip_hits = []
        token_counts = []
        objective_direction_stats = _new_objective_direction_stats()
        surrogate_weighted_objective_direction_stats = _new_objective_direction_stats()
        total_loss_value = 0.0
        microbatch_size = train_microbatch_size
        microbatch_count = 0
        total_token_count = sum(
            max(
                1,
                _example_loss_denominator_count(
                    example, loss_aggregation=self.loss_aggregation
                ),
            )
            for example in examples
        )
        loss_denominator_token_count = int(global_token_count or total_token_count)
        loss_denominator_token_count = max(1, loss_denominator_token_count)
        local_example_denominator = _loss_denominator_example_count(
            examples, loss_aggregation=self.loss_aggregation
        )
        loss_denominator_example_count = int(
            global_example_count or local_example_denominator
        )
        loss_denominator_example_count = max(1, loss_denominator_example_count)
        sign_balanced_positive_token_count = None
        sign_balanced_negative_token_count = None
        if self.loss_aggregation == "advantage_sign_balanced_token_mean":
            if (
                global_positive_token_count is not None
                and global_negative_token_count is not None
            ):
                sign_balanced_positive_token_count = int(global_positive_token_count)
                sign_balanced_negative_token_count = int(global_negative_token_count)
            else:
                sign_counts = _examples_advantage_sign_token_counts(
                    examples, advantages=advantages
                )
                sign_balanced_positive_token_count = int(sign_counts["positive"])
                sign_balanced_negative_token_count = int(sign_counts["negative"])
        streamed_trajectory_span_count = 0
        streamed_sequence_span_backward = _should_stream_trajectory_sequence_spans(
            examples,
            training_unit=self.training_unit,
            importance_sampling_level=self.importance_sampling_level,
            loss_aggregation=self.loss_aggregation,
        )
        streamed_token_span_backward = _should_stream_trajectory_token_spans(
            examples,
            training_unit=self.training_unit,
            importance_sampling_level=self.importance_sampling_level,
            loss_aggregation=self.loss_aggregation,
        )
        streamed_trajectory_span_backward = (
            streamed_sequence_span_backward or streamed_token_span_backward
        )

        if streamed_trajectory_span_backward and has_token_level_advantages:
            raise ValueError(
                "Trajectory-span streaming does not support per-token advantages yet; "
                "use training_unit='action' for RLinf action-level token advantages."
            )

        if streamed_sequence_span_backward:
            span_work = [
                (example_index, span_example)
                for example_index, example in enumerate(examples)
                for span_example in _iter_trajectory_action_span_examples(example)
            ]
            streamed_trajectory_span_count = len(span_work)
            sequence_stats = [
                {
                    "log_ratio_sum": 0.0,
                    "reference_l2_sum": 0.0,
                    "valid_tokens": 0,
                    "reference_tokens": 0,
                }
                for _ in examples
            ]
            sequence_rng_states: list[tuple[Any, list[Any] | None]] = []

            # GSPO needs one likelihood ratio for the complete trajectory. Score
            # spans without autograd first so the ratio and clipping branch are
            # fixed without retaining every VLA forward graph in GPU memory.
            with torch.no_grad():
                for microbatch_start in range(0, len(span_work), microbatch_size):
                    microbatch_items = span_work[
                        microbatch_start : microbatch_start + microbatch_size
                    ]
                    microbatch_examples = [
                        span_example
                        for _example_index, span_example in microbatch_items
                    ]
                    sequence_rng_states.append(
                        (
                            torch.random.get_rng_state(),
                            torch.cuda.get_rng_state_all()
                            if torch.cuda.is_available()
                            else None,
                        )
                    )
                    microbatch_count += 1
                    logprob_start = time.perf_counter()
                    microbatch_logprobs = _policy_action_token_logprobs(
                        self.policy,
                        microbatch_examples,
                        device=self.device,
                        pad_to_batch_size=microbatch_size,
                    )
                    logprob_forward_seconds += time.perf_counter() - logprob_start
                    for (example_index, span_example), span_logprobs in zip(
                        microbatch_items,
                        microbatch_logprobs,
                        strict=True,
                    ):
                        old_logprobs = _validated_old_action_token_logprobs(
                            span_example,
                            span_logprobs,
                            context=(
                                f"trajectory example {example_index} span "
                                f"{span_example.metadata.get('span_index')}"
                            ),
                        )
                        token_loss_mask = _example_token_loss_mask_tensor(
                            span_example,
                            token_count=int(span_logprobs.numel()),
                            device=str(span_logprobs.device),
                            dtype=span_logprobs.dtype,
                        )
                        valid_token_mask = token_loss_mask.detach() > 0
                        valid_count = int(valid_token_mask.sum().item())
                        if valid_count <= 0:
                            continue
                        stats = sequence_stats[example_index]
                        stats["log_ratio_sum"] += float(
                            (span_logprobs - old_logprobs)[valid_token_mask]
                            .double()
                            .sum()
                            .cpu()
                            .item()
                        )
                        stats["valid_tokens"] += valid_count
                        reference_logprobs = _validated_reference_action_token_logprobs(
                            span_example,
                            span_logprobs,
                            require=self.require_reference_logprobs,
                            context=(
                                f"trajectory example {example_index} span "
                                f"{span_example.metadata.get('span_index')}"
                            ),
                        )
                        if reference_logprobs is not None:
                            stats["reference_l2_sum"] += float(
                                (span_logprobs - reference_logprobs)[valid_token_mask]
                                .double()
                                .pow(2)
                                .sum()
                                .cpu()
                                .item()
                            )
                            stats["reference_tokens"] += valid_count

            sequence_backward_coefficients: list[float] = []
            for example_index, (example, advantage, stats) in enumerate(
                zip(examples, advantages, sequence_stats, strict=True)
            ):
                valid_tokens = int(stats["valid_tokens"])
                if valid_tokens <= 0:
                    raise ValueError(
                        "Sequence-level GSPO trajectory has no valid action tokens: "
                        f"example {example_index}"
                    )
                log_ratio_value = float(stats["log_ratio_sum"]) / float(valid_tokens)
                log_ratio_leaf = torch.tensor(
                    log_ratio_value,
                    dtype=torch.float64,
                    requires_grad=True,
                )
                ratio_leaf = torch.exp(log_ratio_leaf)
                clipped_ratio_leaf = torch.clamp(
                    ratio_leaf,
                    1.0 - self.clip_epsilon_low,
                    1.0 + self.clip_epsilon_high,
                )
                advantage_leaf = advantage.detach().double().cpu()
                surrogate_loss = torch.maximum(
                    -ratio_leaf * advantage_leaf,
                    -clipped_ratio_leaf * advantage_leaf,
                )
                if self.clip_ratio_c is not None:
                    dual_clip_loss = (
                        torch.sign(advantage_leaf) * self.clip_ratio_c * advantage_leaf
                    )
                    surrogate_loss = torch.minimum(surrogate_loss, dual_clip_loss)
                (surrogate_dlog_ratio,) = torch.autograd.grad(
                    surrogate_loss, log_ratio_leaf
                )
                sequence_backward_coefficients.append(
                    float(surrogate_dlog_ratio.item()) - self.kl_coef
                )

                ratio_value = ratio_leaf.detach().float()
                clipped_ratio_value = clipped_ratio_leaf.detach().float()
                approx_kl_value = -log_ratio_value
                reference_l2_value = None
                if int(stats["reference_tokens"]) > 0:
                    if int(stats["reference_tokens"]) != valid_tokens:
                        raise ValueError(
                            "Sequence-level GSPO requires reference logprobs for all "
                            "valid trajectory tokens when any are provided"
                        )
                    reference_l2_value = float(stats["reference_l2_sum"]) / float(
                        valid_tokens
                    )
                    reference_l2_penalties.append(
                        torch.tensor([reference_l2_value], dtype=torch.float32)
                    )
                example_loss_value = float(surrogate_loss.detach().item())
                if self.kl_coef:
                    example_loss_value += self.kl_coef * approx_kl_value
                if reference_l2_value is not None and self.reference_logprob_l2_coef:
                    example_loss_value += (
                        self.reference_logprob_l2_coef * reference_l2_value
                    )
                losses.append(example_loss_value)
                total_loss_value += example_loss_value / float(
                    loss_denominator_example_count
                )
                ratios.append(ratio_value.reshape(1))
                approx_kls.append(torch.tensor([approx_kl_value], dtype=torch.float32))
                clip_hits.append(
                    (ratio_value != clipped_ratio_value).float().reshape(1)
                )
                token_counts.append(float(valid_tokens))
                _accumulate_objective_direction_stats(
                    objective_direction_stats,
                    logprob_delta=torch.tensor([log_ratio_value]),
                    advantage=advantage.detach().cpu(),
                )

            # Recompute one action span at a time with autograd. The linear
            # surrogate below has exactly the same derivative at the current
            # policy as the full-trajectory GSPO loss, while each graph is freed
            # immediately after backward().
            for sequence_microbatch_index, microbatch_start in enumerate(
                range(0, len(span_work), microbatch_size)
            ):
                microbatch_items = span_work[
                    microbatch_start : microbatch_start + microbatch_size
                ]
                microbatch_examples = [
                    span_example for _example_index, span_example in microbatch_items
                ]
                microbatch_count += 1
                cpu_rng_state, cuda_rng_states = sequence_rng_states[
                    sequence_microbatch_index
                ]
                torch.random.set_rng_state(cpu_rng_state)
                if cuda_rng_states is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_states)
                _maybe_write_action_token_train_microbatch_start(
                    progress_path=self.progress_path,
                    progress_every_microbatches=self.progress_every_microbatches,
                    progress_max_bytes=self.progress_max_bytes,
                    microbatch_count=microbatch_count,
                    microbatch_start=microbatch_start,
                    total_items=len(span_work),
                    microbatch_size=microbatch_size,
                    streamed_trajectory_span_backward=True,
                )
                logprob_start = time.perf_counter()
                microbatch_logprobs = _policy_action_token_logprobs(
                    self.policy,
                    microbatch_examples,
                    device=self.device,
                    pad_to_batch_size=microbatch_size,
                )
                logprob_forward_seconds += time.perf_counter() - logprob_start
                microbatch_proxy_losses = []
                for (example_index, span_example), span_logprobs in zip(
                    microbatch_items,
                    microbatch_logprobs,
                    strict=True,
                ):
                    token_loss_mask = _example_token_loss_mask_tensor(
                        span_example,
                        token_count=int(span_logprobs.numel()),
                        device=str(span_logprobs.device),
                        dtype=span_logprobs.dtype,
                    )
                    valid_token_mask = token_loss_mask.detach() > 0
                    if not bool(valid_token_mask.any()):
                        continue
                    valid_tokens = int(sequence_stats[example_index]["valid_tokens"])
                    proxy_loss = (
                        sequence_backward_coefficients[example_index]
                        * span_logprobs[valid_token_mask].sum()
                        / float(valid_tokens)
                    )
                    if self.reference_logprob_l2_coef:
                        reference_logprobs = _validated_reference_action_token_logprobs(
                            span_example,
                            span_logprobs,
                            require=self.require_reference_logprobs,
                            context=(
                                f"trajectory example {example_index} span "
                                f"{span_example.metadata.get('span_index')}"
                            ),
                        )
                        if reference_logprobs is not None:
                            proxy_loss = proxy_loss + (
                                self.reference_logprob_l2_coef
                                * (span_logprobs - reference_logprobs)[valid_token_mask]
                                .pow(2)
                                .sum()
                                / float(valid_tokens)
                            )
                    microbatch_proxy_losses.append(
                        proxy_loss / float(loss_denominator_example_count)
                    )
                if not microbatch_proxy_losses:
                    continue
                backward_start = time.perf_counter()
                backward_policy_loss(self.policy, torch.stack(microbatch_proxy_losses).sum())
                loss_backward_seconds += time.perf_counter() - backward_start
                _maybe_write_action_token_train_progress(
                    progress_path=self.progress_path,
                    progress_every_microbatches=self.progress_every_microbatches,
                    progress_max_bytes=self.progress_max_bytes,
                    microbatch_count=microbatch_count,
                    microbatch_start=microbatch_start,
                    total_items=len(span_work),
                    microbatch_size=microbatch_size,
                    logprob_forward_seconds=logprob_forward_seconds,
                    loss_backward_seconds=loss_backward_seconds,
                    total_loss_value=total_loss_value,
                    streamed_trajectory_span_backward=True,
                )
        elif streamed_token_span_backward:
            span_work = [
                (example_index, span_example)
                for example_index, example in enumerate(examples)
                for span_example in _iter_trajectory_action_span_examples(example)
            ]
            streamed_trajectory_span_count = len(span_work)
            for microbatch_start in range(0, len(span_work), microbatch_size):
                microbatch_items = span_work[
                    microbatch_start : microbatch_start + microbatch_size
                ]
                microbatch_examples = [
                    span_example for _example_index, span_example in microbatch_items
                ]
                microbatch_count += 1
                _maybe_write_action_token_train_microbatch_start(
                    progress_path=self.progress_path,
                    progress_every_microbatches=self.progress_every_microbatches,
                    progress_max_bytes=self.progress_max_bytes,
                    microbatch_count=microbatch_count,
                    microbatch_start=microbatch_start,
                    total_items=len(span_work),
                    microbatch_size=microbatch_size,
                    streamed_trajectory_span_backward=streamed_trajectory_span_backward,
                )
                logprob_start = time.perf_counter()
                microbatch_logprobs = _policy_action_token_logprobs(
                    self.policy,
                    microbatch_examples,
                    device=self.device,
                    pad_to_batch_size=microbatch_size,
                )
                logprob_forward_seconds += time.perf_counter() - logprob_start
                microbatch_losses = []
                for (example_index, span_example), span_logprobs in zip(
                    microbatch_items,
                    microbatch_logprobs,
                    strict=True,
                ):
                    advantage = advantages[example_index]
                    if span_example.logprobs is None:
                        raise ValueError(
                            "Action-token GRPO requires old rollout logprobs"
                        )
                    old_logprobs = torch.as_tensor(
                        span_example.logprobs,
                        dtype=span_logprobs.dtype,
                        device=span_logprobs.device,
                    )
                    if old_logprobs.shape != span_logprobs.shape:
                        raise ValueError(
                            "New and old action-token logprobs must have the same shape for "
                            f"trajectory example {example_index} span "
                            f"{span_example.metadata.get('span_index')}: "
                            f"new={tuple(span_logprobs.shape)}, old={tuple(old_logprobs.shape)}"
                        )
                    approx_kl = old_logprobs - span_logprobs
                    reference_logprobs = _example_reference_logprobs(span_example)
                    reference_l2 = None
                    if reference_logprobs is None:
                        if self.require_reference_logprobs:
                            raise ValueError(
                                "Action-token GRPO/GSPO was configured with "
                                "require_reference_logprobs=True, but trajectory example "
                                f"{example_index} span {span_example.metadata.get('span_index')} "
                                "has no reference_logprobs"
                            )
                    else:
                        reference_logprobs = torch.as_tensor(
                            reference_logprobs,
                            dtype=span_logprobs.dtype,
                            device=span_logprobs.device,
                        )
                        if reference_logprobs.shape != span_logprobs.shape:
                            raise ValueError(
                                "Reference and current action-token logprobs must have the same "
                                "shape for trajectory spans: "
                                f"reference={tuple(reference_logprobs.shape)}, "
                                f"current={tuple(span_logprobs.shape)}"
                            )
                        reference_l2 = (
                            (span_logprobs - reference_logprobs).pow(2).mean()
                        )
                        reference_l2_penalties.append(reference_l2.detach().reshape(1))

                    ratio = torch.exp(span_logprobs - old_logprobs)
                    clipped_ratio = torch.clamp(
                        ratio,
                        1.0 - self.clip_epsilon_low,
                        1.0 + self.clip_epsilon_high,
                    )
                    token_loss = torch.maximum(
                        -ratio * advantage, -clipped_ratio * advantage
                    )
                    if self.clip_ratio_c is not None:
                        dual_clip_loss = (
                            torch.sign(advantage) * self.clip_ratio_c * advantage
                        )
                        token_loss = torch.minimum(token_loss, dual_clip_loss)
                    if self.kl_coef:
                        token_loss = token_loss + self.kl_coef * approx_kl
                    if reference_l2 is not None and self.reference_logprob_l2_coef:
                        token_loss = (
                            token_loss + self.reference_logprob_l2_coef * reference_l2
                        )

                    span_loss = token_loss.sum()
                    microbatch_losses.append(span_loss)
                    losses.append(float(span_loss.detach().cpu().item()))
                    ratios.append(ratio.detach())
                    approx_kls.append(approx_kl.detach())
                    clip_hits.append((ratio.detach() != clipped_ratio.detach()).float())
                    token_counts.append(float(span_logprobs.numel()))
                    _accumulate_objective_direction_stats(
                        objective_direction_stats,
                        logprob_delta=span_logprobs.detach() - old_logprobs.detach(),
                        advantage=advantage.detach(),
                    )
                microbatch_loss = torch.stack(microbatch_losses).sum()
                weighted_microbatch_loss = microbatch_loss / float(
                    loss_denominator_token_count
                )
                total_loss_value += float(
                    weighted_microbatch_loss.detach().cpu().item()
                )
                backward_start = time.perf_counter()
                backward_policy_loss(self.policy, weighted_microbatch_loss)
                loss_backward_seconds += time.perf_counter() - backward_start
                _maybe_write_action_token_train_progress(
                    progress_path=self.progress_path,
                    progress_every_microbatches=self.progress_every_microbatches,
                    progress_max_bytes=self.progress_max_bytes,
                    microbatch_count=microbatch_count,
                    microbatch_start=microbatch_start,
                    total_items=len(span_work),
                    microbatch_size=microbatch_size,
                    logprob_forward_seconds=logprob_forward_seconds,
                    loss_backward_seconds=loss_backward_seconds,
                    total_loss_value=total_loss_value,
                    streamed_trajectory_span_backward=streamed_trajectory_span_backward,
                )
        else:
            for microbatch_start in range(0, len(examples), microbatch_size):
                microbatch_examples = examples[
                    microbatch_start : microbatch_start + microbatch_size
                ]
                microbatch_count += 1
                _maybe_write_action_token_train_microbatch_start(
                    progress_path=self.progress_path,
                    progress_every_microbatches=self.progress_every_microbatches,
                    progress_max_bytes=self.progress_max_bytes,
                    microbatch_count=microbatch_count,
                    microbatch_start=microbatch_start,
                    total_items=len(examples),
                    microbatch_size=microbatch_size,
                    streamed_trajectory_span_backward=streamed_trajectory_span_backward,
                )
                logprob_start = time.perf_counter()
                microbatch_logprobs = _policy_action_token_logprobs(
                    self.policy,
                    microbatch_examples,
                    device=self.device,
                    pad_to_batch_size=microbatch_size,
                )
                logprob_forward_seconds += time.perf_counter() - logprob_start
                microbatch_losses = []
                for local_index, (example, example_logprobs) in enumerate(
                    zip(microbatch_examples, microbatch_logprobs, strict=True)
                ):
                    example_index = microbatch_start + local_index
                    if example.logprobs is None:
                        raise ValueError(
                            "Action-token GRPO requires old rollout logprobs"
                        )
                    old_logprobs = torch.as_tensor(
                        example.logprobs,
                        dtype=example_logprobs.dtype,
                        device=example_logprobs.device,
                    )
                    if old_logprobs.shape != example_logprobs.shape:
                        raise ValueError(
                            "New and old action-token logprobs must have the same shape for "
                            f"example {example_index}: new={tuple(example_logprobs.shape)}, "
                            f"old={tuple(old_logprobs.shape)}"
                        )
                    advantage = advantages[example_index]
                    approx_kl = old_logprobs - example_logprobs
                    reference_logprobs = _example_reference_logprobs(example)
                    reference_l2 = None
                    if reference_logprobs is None:
                        if self.require_reference_logprobs:
                            raise ValueError(
                                "Action-token GRPO/GSPO was configured with "
                                "require_reference_logprobs=True, but example "
                                f"{example_index} has no reference_logprobs"
                            )
                    else:
                        reference_logprobs = torch.as_tensor(
                            reference_logprobs,
                            dtype=example_logprobs.dtype,
                            device=example_logprobs.device,
                        )
                        if reference_logprobs.shape != example_logprobs.shape:
                            raise ValueError(
                                "Reference and current action-token logprobs must have the same "
                                f"shape for example {example_index}: "
                                f"reference={tuple(reference_logprobs.shape)}, "
                                f"current={tuple(example_logprobs.shape)}"
                            )
                        reference_l2 = (
                            (example_logprobs - reference_logprobs).pow(2).mean()
                        )
                        reference_l2_penalties.append(reference_l2.detach().reshape(1))
                    if self.importance_sampling_level in ("sequence", "action_chunk"):
                        if _example_has_token_advantages(example):
                            raise ValueError(
                                "Per-token RLinf action-level advantages require "
                                "importance_sampling_level='token'; event-level ratios "
                                "require a frozen scalar advantage."
                            )
                        log_ratio = (
                            action_chunk_log_ratio(
                                example_logprobs, old_logprobs, example
                            )
                            if self.importance_sampling_level == "action_chunk"
                            else (example_logprobs - old_logprobs).mean()
                        )
                        ratio = torch.exp(log_ratio)
                        clipped_ratio = torch.clamp(
                            ratio,
                            1.0 - self.clip_epsilon_low,
                            1.0 + self.clip_epsilon_high,
                        )
                        example_loss = torch.maximum(
                            -ratio * advantage, -clipped_ratio * advantage
                        )
                        if self.clip_ratio_c is not None:
                            dual_clip_loss = (
                                torch.sign(advantage) * self.clip_ratio_c * advantage
                            )
                            example_loss = torch.minimum(example_loss, dual_clip_loss)
                        if self.kl_coef:
                            example_loss = (
                                example_loss + self.kl_coef * approx_kl.mean()
                            )
                        if reference_l2 is not None and self.reference_logprob_l2_coef:
                            example_loss = (
                                example_loss
                                + self.reference_logprob_l2_coef * reference_l2
                            )
                        microbatch_losses.append(example_loss)
                        losses.append(float(example_loss.detach().cpu().item()))
                        ratios.append(ratio.detach().reshape(1))
                        approx_kls.append(-log_ratio.detach().reshape(1))
                        clip_hits.append(
                            (ratio.detach() != clipped_ratio.detach())
                            .float()
                            .reshape(1)
                        )
                        token_counts.append(float(example_logprobs.numel()))
                        _accumulate_objective_direction_stats(
                            objective_direction_stats,
                            logprob_delta=log_ratio.detach().reshape(1),
                            advantage=advantage.detach(),
                        )
                    else:
                        token_advantage = _example_token_advantage_tensor(
                            example,
                            fallback=advantage,
                            token_count=int(example_logprobs.numel()),
                            device=str(example_logprobs.device),
                            dtype=example_logprobs.dtype,
                        )
                        token_loss_mask = _example_token_loss_mask_tensor(
                            example,
                            token_count=int(example_logprobs.numel()),
                            device=str(example_logprobs.device),
                            dtype=example_logprobs.dtype,
                        )
                        bucket_loss_weight = _example_bucket_loss_weight_tensor(
                            example,
                            token_count=int(example_logprobs.numel()),
                            device=str(example_logprobs.device),
                            dtype=example_logprobs.dtype,
                            policy_step_loss_weights=self.policy_step_loss_weights,
                            action_dim_loss_weights=self.action_dim_loss_weights,
                            loss_aggregation=self.loss_aggregation,
                        )
                        valid_token_mask = token_loss_mask.detach() > 0
                        ratio = torch.exp(example_logprobs - old_logprobs)
                        clipped_ratio = torch.clamp(
                            ratio,
                            1.0 - self.clip_epsilon_low,
                            1.0 + self.clip_epsilon_high,
                        )
                        token_loss = torch.maximum(
                            -ratio * token_advantage,
                            -clipped_ratio * token_advantage,
                        )
                        if self.clip_ratio_c is not None:
                            dual_clip_loss = (
                                torch.sign(token_advantage)
                                * self.clip_ratio_c
                                * token_advantage
                            )
                            token_loss = torch.minimum(token_loss, dual_clip_loss)
                        if self.kl_coef:
                            token_loss = token_loss + self.kl_coef * approx_kl
                        if reference_l2 is not None and self.reference_logprob_l2_coef:
                            token_loss = (
                                token_loss
                                + self.reference_logprob_l2_coef * reference_l2
                            )
                        token_loss = token_loss * token_loss_mask * bucket_loss_weight
                        if self.loss_aggregation == "rlinf_masked_mean_ratio":
                            token_loss = (
                                token_loss
                                * _example_rlinf_masked_mean_ratio_scale(example)
                            )
                        elif (
                            self.loss_aggregation
                            == "advantage_sign_balanced_token_mean"
                        ):
                            token_loss = (
                                token_loss
                                * _advantage_sign_balance_scale_tensor(
                                    token_advantage,
                                    token_loss_mask,
                                    positive_token_count=sign_balanced_positive_token_count,
                                    negative_token_count=sign_balanced_negative_token_count,
                                    denominator_token_count=loss_denominator_token_count,
                                )
                            )
                        elif self.loss_aggregation == "task_balanced_trajectory_mean":
                            token_loss = token_loss * _example_task_balance_weight(
                                example
                            )
                        valid_token_count = torch.clamp(token_loss_mask.sum(), min=1.0)
                        if self.loss_aggregation in (
                            "token_mean",
                            "rlinf_token_mean",
                            "rlinf_chunk_mean",
                            "rlinf_masked_mean_ratio",
                            "advantage_sign_balanced_token_mean",
                            "seq_mean_token_sum",
                        ):
                            example_loss = token_loss.sum()
                        else:
                            example_loss = token_loss.sum() / valid_token_count
                        microbatch_losses.append(example_loss)
                        losses.append(float(example_loss.detach().cpu().item()))
                        ratios.append(ratio.detach()[valid_token_mask])
                        approx_kls.append(approx_kl.detach()[valid_token_mask])
                        clip_hits.append(
                            (ratio.detach() != clipped_ratio.detach()).float()[
                                valid_token_mask
                            ]
                        )
                        token_counts.append(
                            float(token_loss_mask.detach().sum().cpu().item())
                        )
                        diagnostic_advantage = token_advantage.detach()
                        if diagnostic_advantage.reshape(-1).numel() != 1:
                            diagnostic_advantage = diagnostic_advantage[
                                valid_token_mask
                            ]
                        _accumulate_objective_direction_stats(
                            objective_direction_stats,
                            logprob_delta=(
                                example_logprobs.detach() - old_logprobs.detach()
                            )[valid_token_mask],
                            advantage=diagnostic_advantage,
                        )

                if self.loss_aggregation in (
                    "token_mean",
                    "rlinf_token_mean",
                    "rlinf_chunk_mean",
                    "rlinf_masked_mean_ratio",
                    "advantage_sign_balanced_token_mean",
                ):
                    microbatch_loss = torch.stack(microbatch_losses).sum()
                    weighted_microbatch_loss = microbatch_loss / float(
                        loss_denominator_token_count
                    )
                else:
                    microbatch_loss = torch.stack(microbatch_losses).mean()
                    weighted_microbatch_loss = microbatch_loss * (
                        float(len(microbatch_examples))
                        / float(loss_denominator_example_count)
                    )
                total_loss_value += float(
                    weighted_microbatch_loss.detach().cpu().item()
                )
                backward_start = time.perf_counter()
                backward_policy_loss(self.policy, weighted_microbatch_loss)
                loss_backward_seconds += time.perf_counter() - backward_start
                _maybe_write_action_token_train_progress(
                    progress_path=self.progress_path,
                    progress_every_microbatches=self.progress_every_microbatches,
                    progress_max_bytes=self.progress_max_bytes,
                    microbatch_count=microbatch_count,
                    microbatch_start=microbatch_start,
                    total_items=len(examples),
                    microbatch_size=microbatch_size,
                    logprob_forward_seconds=logprob_forward_seconds,
                    loss_backward_seconds=loss_backward_seconds,
                    total_loss_value=total_loss_value,
                    streamed_trajectory_span_backward=streamed_trajectory_span_backward,
                )

        scaling_metrics = unscale_policy_gradients(self.policy)
        gradient_metrics = _gradient_metrics(self.policy) | scaling_metrics
        if gradient_only:
            gradient_payload = _trainable_gradient_payload(self.policy)
            if gradient_output_path is not None:
                _save_gradient_payload(
                    gradient_payload, Path(str(gradient_output_path))
                )
            metrics = _grpo_metrics(
                groups=groups,
                examples=examples,
                loss=total_loss_value,
                advantages=advantages.detach().cpu().tolist(),
                ratios=ratios,
                approx_kls=approx_kls,
                reference_l2_penalties=reference_l2_penalties,
                clip_hits=clip_hits,
                token_counts=token_counts,
                normalize_advantages=self.normalize_advantages,
                advantage_normalization_scope=self.advantage_normalization_scope,
                advantage_std_unbiased=self.advantage_std_unbiased,
                reward_filter_report=reward_filter_report,
                importance_sampling_level=self.importance_sampling_level,
                training_unit=self.training_unit,
                loss_aggregation=self.loss_aggregation,
                kl_coef=self.kl_coef,
                clip_epsilon=self.clip_epsilon,
                clip_epsilon_low=self.clip_epsilon_low,
                clip_epsilon_high=self.clip_epsilon_high,
                reference_logprob_l2_coef=self.reference_logprob_l2_coef,
                objective_direction_stats=objective_direction_stats,
            )
            metrics.update(
                _bucket_loss_weight_metrics(
                    examples,
                    policy_step_loss_weights=self.policy_step_loss_weights,
                    action_dim_loss_weights=self.action_dim_loss_weights,
                    prefix="embodied_action_token_grpo",
                )
            )
            metrics.update(gradient_metrics)
            metrics.update(gradient_direction_probe_presence_metrics)
            metrics["embodied_action_token_grpo/optimizer_step_completed"] = 0.0
            metrics[
                "embodied_action_token_grpo/optimizer_step_skipped_no_group_signal"
            ] = 0.0
            metrics[
                "embodied_action_token_grpo/optimizer_step_skipped_logprob_misalignment"
            ] = 0.0
            metrics["embodied_action_token_grpo/gradient_handoff_worker"] = 1.0
            metrics["embodied_action_token_grpo/gradient_handoff_tensors"] = float(
                len(gradient_payload["gradients"])
            )
            metrics["embodied_action_token_grpo/global_loss_denominator_examples"] = (
                float(loss_denominator_example_count)
            )
            metrics["embodied_action_token_grpo/global_loss_denominator_tokens"] = (
                float(loss_denominator_token_count)
            )
            metrics["embodied_action_token_grpo/logprob_eval_mode"] = float(
                self.logprob_eval_mode
            )
            metrics["embodied_action_token_grpo/logprob_microbatch_size"] = float(
                microbatch_size
            )
            metrics["embodied_action_token_grpo/rollout_logprob_microbatch_size"] = (
                float(self.logprob_microbatch_size or len(examples))
            )
            metrics["embodied_action_token_grpo/train_logprob_microbatch_size"] = float(
                microbatch_size
            )
            metrics["embodied_action_token_grpo/logprob_microbatches"] = float(
                microbatch_count
            )
            if alignment_guard is not None:
                _add_alignment_guard_metrics(
                    metrics,
                    alignment_guard,
                    prefix="embodied_action_token_grpo",
                )
            metrics["embodied_action_token_grpo/streamed_trajectory_span_backward"] = (
                float(streamed_trajectory_span_backward)
            )
            metrics["embodied_action_token_grpo/streamed_trajectory_span_examples"] = (
                float(streamed_trajectory_span_count)
            )
            metrics["embodied_action_token_grpo/streamed_sequence_two_pass"] = float(
                streamed_sequence_span_backward
            )
            metrics["embodied_action_token_grpo/train_total_seconds"] = float(
                time.perf_counter() - train_wall_start
            )
            metrics["embodied_action_token_grpo/train_logprob_forward_seconds"] = float(
                logprob_forward_seconds
            )
            metrics["embodied_action_token_grpo/train_loss_backward_seconds"] = float(
                loss_backward_seconds
            )
            metrics.update(
                {
                    key: float(value)
                    for key, value in kwargs.items()
                    if _is_number(value)
                }
            )
            algorithm_name = _policy_gradient_algorithm_name(
                self.importance_sampling_level
            )
            if algorithm_name == "gspo":
                metrics.update(
                    _alias_metric_prefix(
                        metrics,
                        source_prefix="embodied_action_token_grpo/",
                        target_prefix="embodied_action_token_gspo/",
                        drop_source=False,
                    )
                )
            return LocalTrainResult(
                step=self.step, metrics=metrics, checkpoint_path=None
            )
        if self.max_grad_norm is not None:
            _clip_grad_norm(self.policy, self.max_grad_norm)
        gradient_direction_probe_metrics = _gradient_direction_probe_metrics(
            self,
            groups=groups,
            config=gradient_direction_probe_cfg,
        )
        parameter_snapshot_start = time.perf_counter()
        parameter_snapshot = _trainable_parameter_snapshot(self.policy)
        parameter_snapshot_seconds = time.perf_counter() - parameter_snapshot_start
        optimizer_step_start = time.perf_counter()
        self.optimizer.step()
        optimizer_step_seconds = time.perf_counter() - optimizer_step_start
        parameter_update_metrics = _parameter_update_metrics(
            self.policy,
            parameter_snapshot,
            prefix="embodied_action_token_grpo",
        )
        self.step += 1

        metrics = _grpo_metrics(
            groups=groups,
            examples=examples,
            loss=total_loss_value,
            advantages=advantages.detach().cpu().tolist(),
            ratios=ratios,
            approx_kls=approx_kls,
            reference_l2_penalties=reference_l2_penalties,
            clip_hits=clip_hits,
            token_counts=token_counts,
            normalize_advantages=self.normalize_advantages,
            advantage_normalization_scope=self.advantage_normalization_scope,
            advantage_std_unbiased=self.advantage_std_unbiased,
            reward_filter_report=reward_filter_report,
            importance_sampling_level=self.importance_sampling_level,
            training_unit=self.training_unit,
            loss_aggregation=self.loss_aggregation,
            kl_coef=self.kl_coef,
            clip_epsilon=self.clip_epsilon,
            clip_epsilon_low=self.clip_epsilon_low,
            clip_epsilon_high=self.clip_epsilon_high,
            reference_logprob_l2_coef=self.reference_logprob_l2_coef,
            objective_direction_stats=objective_direction_stats,
        )
        metrics.update(
            _bucket_loss_weight_metrics(
                examples,
                policy_step_loss_weights=self.policy_step_loss_weights,
                action_dim_loss_weights=self.action_dim_loss_weights,
                prefix="embodied_action_token_grpo",
            )
        )
        metrics.update(gradient_metrics)
        metrics.update(gradient_direction_probe_presence_metrics)
        metrics.update(gradient_direction_probe_metrics)
        metrics.update(parameter_update_metrics)
        if alignment_guard is not None:
            _add_alignment_guard_metrics(
                metrics,
                alignment_guard,
                prefix="embodied_action_token_grpo",
            )
        metrics["embodied_action_token_grpo/optimizer_step_completed"] = 1.0
        metrics["embodied_action_token_grpo/optimizer_step_skipped_no_group_signal"] = (
            0.0
        )
        metrics[
            "embodied_action_token_grpo/optimizer_step_skipped_logprob_misalignment"
        ] = 0.0
        metrics["embodied_action_token_grpo/logprob_eval_mode"] = float(
            self.logprob_eval_mode
        )
        metrics["embodied_action_token_grpo/logprob_microbatch_size"] = float(
            microbatch_size
        )
        metrics["embodied_action_token_grpo/rollout_logprob_microbatch_size"] = float(
            self.logprob_microbatch_size or len(examples)
        )
        metrics["embodied_action_token_grpo/train_logprob_microbatch_size"] = float(
            microbatch_size
        )
        metrics["embodied_action_token_grpo/logprob_microbatches"] = float(
            microbatch_count
        )
        metrics["embodied_action_token_grpo/global_loss_denominator_examples"] = float(
            loss_denominator_example_count
        )
        metrics["embodied_action_token_grpo/global_loss_denominator_tokens"] = float(
            loss_denominator_token_count
        )
        metrics["embodied_action_token_grpo/streamed_trajectory_span_backward"] = float(
            streamed_trajectory_span_backward
        )
        metrics["embodied_action_token_grpo/streamed_trajectory_span_examples"] = float(
            streamed_trajectory_span_count
        )
        metrics["embodied_action_token_grpo/streamed_sequence_two_pass"] = float(
            streamed_sequence_span_backward
        )
        metrics["embodied_action_token_grpo/train_total_seconds"] = float(
            time.perf_counter() - train_wall_start
        )
        metrics["embodied_action_token_grpo/train_logprob_forward_seconds"] = float(
            logprob_forward_seconds
        )
        metrics["embodied_action_token_grpo/train_loss_backward_seconds"] = float(
            loss_backward_seconds
        )
        metrics["embodied_action_token_grpo/train_parameter_snapshot_seconds"] = float(
            parameter_snapshot_seconds
        )
        metrics["embodied_action_token_grpo/train_optimizer_step_seconds"] = float(
            optimizer_step_seconds
        )
        metrics.update(
            {key: float(value) for key, value in kwargs.items() if _is_number(value)}
        )

        checkpoint_path = None
        algorithm_name = _policy_gradient_algorithm_name(self.importance_sampling_level)
        if self.checkpoint_dir is not None:
            saved_path = _save_policy_checkpoint(
                self.policy,
                self.checkpoint_dir / f"action-token-{algorithm_name}-step-{self.step}",
                config_fingerprint=self.checkpoint_config_fingerprint,
                resume_contract_fingerprint=(
                    self.checkpoint_resume_contract_fingerprint
                ),
            )
            checkpoint_path = str(saved_path) if saved_path is not None else None
            metrics["embodied_action_token_grpo/checkpoint_written"] = float(
                checkpoint_path is not None
            )
        if algorithm_name == "gspo":
            metrics.update(
                _alias_metric_prefix(
                    metrics,
                    source_prefix="embodied_action_token_grpo/",
                    target_prefix="embodied_action_token_gspo/",
                    drop_source=True,
                )
            )

        return LocalTrainResult(
            step=self.step,
            metrics=metrics,
            checkpoint_path=checkpoint_path,
        )

    def probe_logprob_metrics(
        self,
        trajectory_groups: Iterable[EmbodiedTrajectoryGroup],
        **kwargs: Any,
    ) -> dict[str, float]:
        """Measure current/old action-token ratios without updating the policy.

        Distributed training uses worker metrics as a pre-apply guard between
        optimizer sub-updates. The final optimizer step has no
        subsequent sub-update to observe the already-moved policy, so checkpoint
        selection needs an explicit no-gradient probe before saving adapters.
        This method mirrors the train-time logprob and metric conventions but
        never calls backward() or optimizer.step().
        """

        groups = list(trajectory_groups)
        _validate_group_relative_groups(
            groups,
            backend_name="ActionTokenGRPOBackend.probe_logprob_metrics",
        )
        if self.training_unit == "trajectory":
            examples = extract_trajectory_action_token_examples(
                groups,
                require_logprobs=True,
                require_observations=self.require_observations,
                require_prompts=self.require_prompts,
            )
        else:
            examples = extract_action_token_examples(
                groups,
                require_logprobs=True,
                require_observations=self.require_observations,
                require_prompts=self.require_prompts,
                score_source=self.rlinf_action_level_score_source,
            )
        if not examples:
            raise ValueError(
                "No action-token examples found for ActionTokenGRPOBackend probe"
            )

        reward_filter_report = _reward_filter_report(
            examples,
            enabled=self.filter_rewards,
            lower=self.rewards_lower_bound,
            upper=self.rewards_upper_bound,
        )
        if self.filter_rewards:
            examples = _apply_reward_filter_to_examples(
                examples,
                reward_filter_report=reward_filter_report,
                mode=self.reward_filter_mode,
            )

        if self.action_advantage_mode == "rlinf_action_level_cumulative":
            _attach_rlinf_action_level_token_advantages(
                examples,
                normalize=self.normalize_advantages,
                group_std_unbiased=self.advantage_std_unbiased,
                score_source=self.rlinf_action_level_score_source,
                eps=self.advantage_epsilon,
            )
            if self.rlinf_action_level_mask_zero_variance_groups:
                _mask_rlinf_action_level_zero_variance_groups(
                    examples, eps=self.advantage_epsilon
                )
            if self.rlinf_action_level_extra_global_normalization:
                _apply_rlinf_action_level_extra_global_normalization(
                    examples,
                    mean=self.rlinf_action_level_global_advantage_mean,
                    scale=self.rlinf_action_level_global_advantage_scale,
                )

        if self.logprob_eval_mode:
            _set_eval(self.policy)
        else:
            _set_train(self.policy)

        import torch

        probe_start = time.perf_counter()
        ratios = []
        approx_kls = []
        reference_l2_penalties = []
        clip_hits = []
        token_counts = []
        objective_direction_stats = _new_objective_direction_stats()
        surrogate_weighted_objective_direction_stats = _new_objective_direction_stats()
        objective_direction_policy_step_stats: dict[str, dict[str, float]] = {}
        objective_direction_action_dim_stats: dict[str, dict[str, float]] = {}
        objective_direction_reward_bucket_stats: dict[str, dict[str, float]] = {}
        objective_direction_group_stats: dict[str, dict[str, float]] = {}
        surrogate_weighted_policy_step_stats: dict[str, dict[str, float]] = {}
        surrogate_weighted_action_dim_stats: dict[str, dict[str, float]] = {}
        surrogate_weighted_reward_bucket_stats: dict[str, dict[str, float]] = {}
        surrogate_weighted_group_stats: dict[str, dict[str, float]] = {}
        microbatch_count = 0
        logprob_forward_seconds = 0.0
        advantages = _example_advantages(
            examples,
            normalize=self.normalize_advantages
            and not _examples_have_token_advantages(examples)
            and not _examples_have_prepared_scalar_advantages(examples),
            scope=self.advantage_normalization_scope,
            std_unbiased=self.advantage_std_unbiased,
            eps=self.advantage_epsilon,
            device=self.device,
        )
        total_token_count = sum(
            max(
                1,
                _example_loss_denominator_count(
                    example, loss_aggregation=self.loss_aggregation
                ),
            )
            for example in examples
        )
        loss_denominator_token_count = max(1, int(total_token_count))
        loss_denominator_example_count = max(
            1,
            _loss_denominator_example_count(
                examples, loss_aggregation=self.loss_aggregation
            ),
        )
        sign_balanced_positive_token_count = None
        sign_balanced_negative_token_count = None
        if self.loss_aggregation == "advantage_sign_balanced_token_mean":
            sign_counts = _examples_advantage_sign_token_counts(
                examples, advantages=advantages
            )
            sign_balanced_positive_token_count = int(sign_counts["positive"])
            sign_balanced_negative_token_count = int(sign_counts["negative"])
        streamed_trajectory_span_probe = _should_stream_trajectory_token_spans(
            examples,
            training_unit=self.training_unit,
            importance_sampling_level=self.importance_sampling_level,
            loss_aggregation=self.loss_aggregation,
        )
        if streamed_trajectory_span_probe:
            work_items = [
                (example_index, span_example)
                for example_index, example in enumerate(examples)
                for span_example in _iter_trajectory_action_span_examples(example)
            ]
        else:
            work_items = [
                (example_index, example)
                for example_index, example in enumerate(examples)
            ]

        microbatch_size = self.logprob_microbatch_size or max(1, len(work_items))
        probe_surrogate_loss_numerator = 0.0
        probe_old_policy_surrogate_loss_numerator = 0.0
        probe_unclipped_surrogate_loss_numerator = 0.0
        probe_old_policy_unclipped_surrogate_loss_numerator = 0.0
        with torch.no_grad():
            for microbatch_start in range(0, len(work_items), microbatch_size):
                microbatch_items = work_items[
                    microbatch_start : microbatch_start + microbatch_size
                ]
                microbatch_examples = [
                    example for _example_index, example in microbatch_items
                ]
                logprob_start = time.perf_counter()
                microbatch_logprobs = _policy_action_token_logprobs(
                    self.policy,
                    microbatch_examples,
                    device=self.device,
                    pad_to_batch_size=microbatch_size,
                )
                logprob_forward_seconds += time.perf_counter() - logprob_start
                microbatch_count += 1
                for (example_index, example), example_logprobs in zip(
                    microbatch_items,
                    microbatch_logprobs,
                    strict=True,
                ):
                    if example.logprobs is None:
                        raise ValueError(
                            "Action-token GRPO probe requires old rollout logprobs"
                        )
                    old_logprobs = torch.as_tensor(
                        example.logprobs,
                        dtype=example_logprobs.dtype,
                        device=example_logprobs.device,
                    )
                    if old_logprobs.shape != example_logprobs.shape:
                        raise ValueError(
                            "New and old action-token logprobs must have the same shape for "
                            f"probe example {example_index}: new={tuple(example_logprobs.shape)}, "
                            f"old={tuple(old_logprobs.shape)}"
                        )
                    approx_kl = old_logprobs - example_logprobs
                    reference_logprobs = _example_reference_logprobs(example)
                    if reference_logprobs is not None:
                        reference_logprobs_tensor = torch.as_tensor(
                            reference_logprobs,
                            dtype=example_logprobs.dtype,
                            device=example_logprobs.device,
                        )
                        if reference_logprobs_tensor.shape != example_logprobs.shape:
                            raise ValueError(
                                "Reference and current action-token logprobs must have the same "
                                "shape for probe examples: "
                                f"reference={tuple(reference_logprobs_tensor.shape)}, "
                                f"current={tuple(example_logprobs.shape)}"
                            )
                        reference_l2_penalties.append(
                            (example_logprobs - reference_logprobs_tensor)
                            .pow(2)
                            .mean()
                            .reshape(1)
                        )
                    elif self.require_reference_logprobs:
                        raise ValueError(
                            "Action-token GRPO/GSPO probe requires reference_logprobs, "
                            f"but probe example {example_index} has none"
                        )

                    if self.importance_sampling_level in ("sequence", "action_chunk"):
                        if _example_has_token_advantages(example):
                            raise ValueError(
                                "Per-token RLinf action-level advantages require "
                                "importance_sampling_level='token'; event-level ratios "
                                "require a frozen scalar advantage."
                            )
                        log_ratio = (
                            action_chunk_log_ratio(
                                example_logprobs, old_logprobs, example
                            )
                            if self.importance_sampling_level == "action_chunk"
                            else (example_logprobs - old_logprobs).mean()
                        )
                        ratio = torch.exp(log_ratio)
                        clipped_ratio = torch.clamp(
                            ratio,
                            1.0 - self.clip_epsilon_low,
                            1.0 + self.clip_epsilon_high,
                        )
                        advantage = advantages[example_index]
                        unclipped_example_loss = -ratio * advantage
                        old_unclipped_example_loss = -advantage
                        example_loss = torch.maximum(
                            -ratio * advantage, -clipped_ratio * advantage
                        )
                        old_example_loss = -advantage
                        if self.clip_ratio_c is not None:
                            dual_clip_loss = (
                                torch.sign(advantage) * self.clip_ratio_c * advantage
                            )
                            example_loss = torch.minimum(example_loss, dual_clip_loss)
                            old_example_loss = torch.minimum(
                                old_example_loss, dual_clip_loss
                            )
                        if self.kl_coef:
                            kl_term = self.kl_coef * approx_kl.mean()
                            example_loss = example_loss + kl_term
                            unclipped_example_loss = unclipped_example_loss + kl_term
                        if (
                            reference_logprobs is not None
                            and self.reference_logprob_l2_coef
                        ):
                            old_reference_l2 = (
                                (old_logprobs - reference_logprobs_tensor).pow(2).mean()
                            )
                            current_reference_l2 = (
                                (example_logprobs - reference_logprobs_tensor)
                                .pow(2)
                                .mean()
                            )
                            current_reference_term = (
                                self.reference_logprob_l2_coef * current_reference_l2
                            )
                            old_reference_term = (
                                self.reference_logprob_l2_coef * old_reference_l2
                            )
                            example_loss = example_loss + current_reference_term
                            unclipped_example_loss = (
                                unclipped_example_loss + current_reference_term
                            )
                            old_example_loss = old_example_loss + old_reference_term
                            old_unclipped_example_loss = (
                                old_unclipped_example_loss + old_reference_term
                            )
                        probe_surrogate_loss_numerator += float(
                            example_loss.detach().cpu().item()
                        )
                        probe_old_policy_surrogate_loss_numerator += float(
                            old_example_loss.detach().cpu().item()
                        )
                        probe_unclipped_surrogate_loss_numerator += float(
                            unclipped_example_loss.detach().cpu().item()
                        )
                        probe_old_policy_unclipped_surrogate_loss_numerator += float(
                            old_unclipped_example_loss.detach().cpu().item()
                        )
                        ratios.append(ratio.detach().reshape(1))
                        approx_kls.append(-log_ratio.detach().reshape(1))
                        clip_hits.append(
                            (ratio.detach() != clipped_ratio.detach())
                            .float()
                            .reshape(1)
                        )
                        token_counts.append(float(example_logprobs.numel()))
                        _accumulate_objective_direction_stats(
                            objective_direction_stats,
                            logprob_delta=log_ratio.detach().reshape(1),
                            advantage=advantages[example_index].detach(),
                        )
                        policy_step = _example_policy_step(example)
                        if policy_step is not None:
                            _accumulate_objective_direction_bucket_stats(
                                objective_direction_policy_step_stats,
                                key=f"policy_step_{policy_step:02d}",
                                logprob_delta=log_ratio.detach().reshape(1),
                                advantage=advantages[example_index].detach(),
                            )
                        reward_bucket = _example_reward_bucket(example)
                        if reward_bucket is not None:
                            _accumulate_objective_direction_bucket_stats(
                                objective_direction_reward_bucket_stats,
                                key=reward_bucket,
                                logprob_delta=log_ratio.detach().reshape(1),
                                advantage=advantages[example_index].detach(),
                            )
                        group_bucket = _example_group_bucket(example)
                        if group_bucket is not None:
                            _accumulate_objective_direction_bucket_stats(
                                objective_direction_group_stats,
                                key=group_bucket,
                                logprob_delta=log_ratio.detach().reshape(1),
                                advantage=advantages[example_index].detach(),
                            )
                    else:
                        token_loss_mask = _example_token_loss_mask_tensor(
                            example,
                            token_count=int(example_logprobs.numel()),
                            device=str(example_logprobs.device),
                            dtype=example_logprobs.dtype,
                        )
                        bucket_loss_weight = _example_bucket_loss_weight_tensor(
                            example,
                            token_count=int(example_logprobs.numel()),
                            device=str(example_logprobs.device),
                            dtype=example_logprobs.dtype,
                            policy_step_loss_weights=self.policy_step_loss_weights,
                            action_dim_loss_weights=self.action_dim_loss_weights,
                            loss_aggregation=self.loss_aggregation,
                        )
                        valid_token_mask = token_loss_mask.detach() > 0
                        ratio = torch.exp(example_logprobs - old_logprobs)
                        clipped_ratio = torch.clamp(
                            ratio,
                            1.0 - self.clip_epsilon_low,
                            1.0 + self.clip_epsilon_high,
                        )
                        ratios.append(ratio.detach()[valid_token_mask])
                        approx_kls.append(approx_kl.detach()[valid_token_mask])
                        clip_hits.append(
                            (ratio.detach() != clipped_ratio.detach()).float()[
                                valid_token_mask
                            ]
                        )
                        token_counts.append(
                            float(token_loss_mask.detach().sum().cpu().item())
                        )
                        token_advantage = _example_token_advantage_tensor(
                            example,
                            fallback=advantages[example_index],
                            token_count=int(example_logprobs.numel()),
                            device=str(example_logprobs.device),
                            dtype=example_logprobs.dtype,
                        )
                        unclipped_token_loss = -ratio * token_advantage
                        old_unclipped_token_loss = -token_advantage
                        token_loss = torch.maximum(
                            -ratio * token_advantage,
                            -clipped_ratio * token_advantage,
                        )
                        old_token_loss = -token_advantage
                        if self.clip_ratio_c is not None:
                            dual_clip_loss = (
                                torch.sign(token_advantage)
                                * self.clip_ratio_c
                                * token_advantage
                            )
                            token_loss = torch.minimum(token_loss, dual_clip_loss)
                            old_token_loss = torch.minimum(
                                old_token_loss, dual_clip_loss
                            )
                        if self.kl_coef:
                            kl_term = self.kl_coef * approx_kl
                            token_loss = token_loss + kl_term
                            unclipped_token_loss = unclipped_token_loss + kl_term
                        if (
                            reference_logprobs is not None
                            and self.reference_logprob_l2_coef
                        ):
                            old_reference_l2 = (
                                (old_logprobs - reference_logprobs_tensor).pow(2).mean()
                            )
                            current_reference_l2 = (
                                (example_logprobs - reference_logprobs_tensor)
                                .pow(2)
                                .mean()
                            )
                            current_reference_term = (
                                self.reference_logprob_l2_coef * current_reference_l2
                            )
                            old_reference_term = (
                                self.reference_logprob_l2_coef * old_reference_l2
                            )
                            token_loss = token_loss + current_reference_term
                            unclipped_token_loss = (
                                unclipped_token_loss + current_reference_term
                            )
                            old_token_loss = old_token_loss + old_reference_term
                            old_unclipped_token_loss = (
                                old_unclipped_token_loss + old_reference_term
                            )
                        token_loss = token_loss * token_loss_mask * bucket_loss_weight
                        old_token_loss = (
                            old_token_loss * token_loss_mask * bucket_loss_weight
                        )
                        unclipped_token_loss = (
                            unclipped_token_loss * token_loss_mask * bucket_loss_weight
                        )
                        old_unclipped_token_loss = (
                            old_unclipped_token_loss
                            * token_loss_mask
                            * bucket_loss_weight
                        )
                        surrogate_direction_weight = (
                            token_loss_mask * bucket_loss_weight
                        )
                        if self.loss_aggregation == "rlinf_masked_mean_ratio":
                            scale = _example_rlinf_masked_mean_ratio_scale(example)
                            token_loss = token_loss * scale
                            old_token_loss = old_token_loss * scale
                            unclipped_token_loss = unclipped_token_loss * scale
                            old_unclipped_token_loss = old_unclipped_token_loss * scale
                            surrogate_direction_weight = (
                                surrogate_direction_weight * scale
                            )
                        elif (
                            self.loss_aggregation
                            == "advantage_sign_balanced_token_mean"
                        ):
                            balance_scale = _advantage_sign_balance_scale_tensor(
                                token_advantage,
                                token_loss_mask,
                                positive_token_count=sign_balanced_positive_token_count,
                                negative_token_count=sign_balanced_negative_token_count,
                                denominator_token_count=loss_denominator_token_count,
                            )
                            token_loss = token_loss * balance_scale
                            old_token_loss = old_token_loss * balance_scale
                            unclipped_token_loss = unclipped_token_loss * balance_scale
                            old_unclipped_token_loss = (
                                old_unclipped_token_loss * balance_scale
                            )
                            surrogate_direction_weight = (
                                surrogate_direction_weight * balance_scale
                            )
                        valid_token_count = torch.clamp(token_loss_mask.sum(), min=1.0)
                        if self.loss_aggregation in (
                            "token_mean",
                            "rlinf_token_mean",
                            "rlinf_chunk_mean",
                            "rlinf_masked_mean_ratio",
                            "advantage_sign_balanced_token_mean",
                            "seq_mean_token_sum",
                        ):
                            probe_surrogate_loss_numerator += float(
                                token_loss.sum().detach().cpu().item()
                            )
                            probe_old_policy_surrogate_loss_numerator += float(
                                old_token_loss.sum().detach().cpu().item()
                            )
                            probe_unclipped_surrogate_loss_numerator += float(
                                unclipped_token_loss.sum().detach().cpu().item()
                            )
                            probe_old_policy_unclipped_surrogate_loss_numerator += (
                                float(
                                    old_unclipped_token_loss.sum().detach().cpu().item()
                                )
                            )
                        else:
                            probe_surrogate_loss_numerator += float(
                                (token_loss.sum() / valid_token_count)
                                .detach()
                                .cpu()
                                .item()
                            )
                            probe_old_policy_surrogate_loss_numerator += float(
                                (old_token_loss.sum() / valid_token_count)
                                .detach()
                                .cpu()
                                .item()
                            )
                            probe_unclipped_surrogate_loss_numerator += float(
                                (unclipped_token_loss.sum() / valid_token_count)
                                .detach()
                                .cpu()
                                .item()
                            )
                            probe_old_policy_unclipped_surrogate_loss_numerator += (
                                float(
                                    (old_unclipped_token_loss.sum() / valid_token_count)
                                    .detach()
                                    .cpu()
                                    .item()
                                )
                            )
                        diagnostic_advantage = token_advantage.detach()
                        if diagnostic_advantage.reshape(-1).numel() != 1:
                            diagnostic_advantage = diagnostic_advantage[
                                valid_token_mask
                            ]
                        diagnostic_logprob_delta = (
                            example_logprobs.detach() - old_logprobs.detach()
                        )[valid_token_mask]
                        _accumulate_objective_direction_stats(
                            objective_direction_stats,
                            logprob_delta=diagnostic_logprob_delta,
                            advantage=diagnostic_advantage,
                        )
                        diagnostic_weight = surrogate_direction_weight.detach()[
                            valid_token_mask
                        ]
                        _accumulate_objective_direction_stats(
                            surrogate_weighted_objective_direction_stats,
                            logprob_delta=diagnostic_logprob_delta,
                            advantage=diagnostic_advantage,
                            weight=diagnostic_weight,
                        )
                        policy_step = _example_policy_step(example)
                        if policy_step is not None:
                            bucket_key = f"policy_step_{policy_step:02d}"
                            _accumulate_objective_direction_bucket_stats(
                                objective_direction_policy_step_stats,
                                key=bucket_key,
                                logprob_delta=diagnostic_logprob_delta,
                                advantage=diagnostic_advantage,
                            )
                            _accumulate_objective_direction_bucket_stats(
                                surrogate_weighted_policy_step_stats,
                                key=bucket_key,
                                logprob_delta=diagnostic_logprob_delta,
                                advantage=diagnostic_advantage,
                                weight=diagnostic_weight,
                            )
                        reward_bucket = _example_reward_bucket(example)
                        if reward_bucket is not None:
                            _accumulate_objective_direction_bucket_stats(
                                objective_direction_reward_bucket_stats,
                                key=reward_bucket,
                                logprob_delta=diagnostic_logprob_delta,
                                advantage=diagnostic_advantage,
                            )
                            _accumulate_objective_direction_bucket_stats(
                                surrogate_weighted_reward_bucket_stats,
                                key=reward_bucket,
                                logprob_delta=diagnostic_logprob_delta,
                                advantage=diagnostic_advantage,
                                weight=diagnostic_weight,
                            )
                        group_bucket = _example_group_bucket(example)
                        if group_bucket is not None:
                            _accumulate_objective_direction_bucket_stats(
                                objective_direction_group_stats,
                                key=group_bucket,
                                logprob_delta=diagnostic_logprob_delta,
                                advantage=diagnostic_advantage,
                            )
                            _accumulate_objective_direction_bucket_stats(
                                surrogate_weighted_group_stats,
                                key=group_bucket,
                                logprob_delta=diagnostic_logprob_delta,
                                advantage=diagnostic_advantage,
                                weight=diagnostic_weight,
                            )
                        action_dim = _example_action_dim(example)
                        if action_dim is not None and action_dim > 0:
                            token_positions = torch.arange(
                                int(example_logprobs.numel()),
                                device=example_logprobs.device,
                            )[valid_token_mask]
                            valid_dim_ids = token_positions % int(action_dim)
                            valid_dim_ids_cpu = valid_dim_ids.detach().cpu()
                            for dim_index in sorted(
                                {int(value) for value in valid_dim_ids_cpu.tolist()}
                            ):
                                dim_mask = valid_dim_ids == int(dim_index)
                                bucket_key = f"action_dim_{dim_index:02d}"
                                _accumulate_objective_direction_bucket_stats(
                                    objective_direction_action_dim_stats,
                                    key=bucket_key,
                                    logprob_delta=diagnostic_logprob_delta[dim_mask],
                                    advantage=diagnostic_advantage[dim_mask]
                                    if diagnostic_advantage.reshape(-1).numel() != 1
                                    else diagnostic_advantage,
                                )
                                _accumulate_objective_direction_bucket_stats(
                                    surrogate_weighted_action_dim_stats,
                                    key=bucket_key,
                                    logprob_delta=diagnostic_logprob_delta[dim_mask],
                                    advantage=diagnostic_advantage[dim_mask]
                                    if diagnostic_advantage.reshape(-1).numel() != 1
                                    else diagnostic_advantage,
                                    weight=diagnostic_weight[dim_mask],
                                )

        metrics = _grpo_metrics(
            groups=groups,
            examples=examples,
            loss=0.0,
            advantages=advantages.detach().cpu().tolist(),
            ratios=ratios,
            approx_kls=approx_kls,
            reference_l2_penalties=reference_l2_penalties,
            clip_hits=clip_hits,
            token_counts=token_counts,
            normalize_advantages=self.normalize_advantages,
            advantage_normalization_scope=self.advantage_normalization_scope,
            advantage_std_unbiased=self.advantage_std_unbiased,
            reward_filter_report=reward_filter_report,
            importance_sampling_level=self.importance_sampling_level,
            training_unit=self.training_unit,
            loss_aggregation=self.loss_aggregation,
            kl_coef=self.kl_coef,
            clip_epsilon=self.clip_epsilon,
            clip_epsilon_low=self.clip_epsilon_low,
            clip_epsilon_high=self.clip_epsilon_high,
            reference_logprob_l2_coef=self.reference_logprob_l2_coef,
            objective_direction_stats=objective_direction_stats,
        )
        metrics.update(
            _bucket_loss_weight_metrics(
                examples,
                policy_step_loss_weights=self.policy_step_loss_weights,
                action_dim_loss_weights=self.action_dim_loss_weights,
                prefix="embodied_action_token_grpo",
            )
        )
        metrics.update(
            _objective_direction_metrics(
                surrogate_weighted_objective_direction_stats,
                prefix="embodied_action_token_grpo/surrogate_weighted",
            )
        )
        metrics.update(
            _objective_direction_bucket_metrics(
                objective_direction_policy_step_stats,
                prefix="embodied_action_token_grpo/by_policy_step",
            )
        )
        metrics.update(
            _objective_direction_bucket_metrics(
                objective_direction_action_dim_stats,
                prefix="embodied_action_token_grpo/by_action_dim",
            )
        )
        metrics.update(
            _objective_direction_bucket_metrics(
                objective_direction_reward_bucket_stats,
                prefix="embodied_action_token_grpo/by_reward_bucket",
                max_buckets=16,
            )
        )
        metrics.update(
            _objective_direction_bucket_metrics(
                objective_direction_group_stats,
                prefix="embodied_action_token_grpo/by_group",
                max_buckets=128,
            )
        )
        metrics.update(
            _objective_direction_bucket_metrics(
                surrogate_weighted_policy_step_stats,
                prefix="embodied_action_token_grpo/surrogate_weighted/by_policy_step",
            )
        )
        metrics.update(
            _objective_direction_bucket_metrics(
                surrogate_weighted_action_dim_stats,
                prefix="embodied_action_token_grpo/surrogate_weighted/by_action_dim",
            )
        )
        metrics.update(
            _objective_direction_bucket_metrics(
                surrogate_weighted_reward_bucket_stats,
                prefix="embodied_action_token_grpo/surrogate_weighted/by_reward_bucket",
                max_buckets=16,
            )
        )
        metrics.update(
            _objective_direction_bucket_metrics(
                surrogate_weighted_group_stats,
                prefix="embodied_action_token_grpo/surrogate_weighted/by_group",
                max_buckets=128,
            )
        )
        metrics["embodied_action_token_grpo/probe_only"] = 1.0
        metrics["embodied_action_token_grpo/optimizer_step_completed"] = 0.0
        metrics["embodied_action_token_grpo/policy_parameters_updated"] = 0.0
        metrics["embodied_action_token_grpo/logprob_eval_mode"] = float(
            self.logprob_eval_mode
        )
        metrics["embodied_action_token_grpo/logprob_microbatch_size"] = float(
            microbatch_size
        )
        metrics["embodied_action_token_grpo/logprob_microbatches"] = float(
            microbatch_count
        )
        if self.loss_aggregation in (
            "token_mean",
            "rlinf_token_mean",
            "rlinf_chunk_mean",
            "rlinf_masked_mean_ratio",
            "advantage_sign_balanced_token_mean",
        ):
            surrogate_denominator = float(loss_denominator_token_count)
        else:
            surrogate_denominator = float(loss_denominator_example_count)
        probe_surrogate_loss = probe_surrogate_loss_numerator / surrogate_denominator
        probe_old_policy_surrogate_loss = (
            probe_old_policy_surrogate_loss_numerator / surrogate_denominator
        )
        probe_unclipped_surrogate_loss = (
            probe_unclipped_surrogate_loss_numerator / surrogate_denominator
        )
        probe_old_policy_unclipped_surrogate_loss = (
            probe_old_policy_unclipped_surrogate_loss_numerator / surrogate_denominator
        )
        probe_surrogate_loss_delta = (
            probe_surrogate_loss - probe_old_policy_surrogate_loss
        )
        probe_unclipped_surrogate_loss_delta = (
            probe_unclipped_surrogate_loss - probe_old_policy_unclipped_surrogate_loss
        )
        metrics["embodied_action_token_grpo/probe_surrogate_loss"] = float(
            probe_surrogate_loss
        )
        metrics["embodied_action_token_grpo/probe_old_policy_surrogate_loss"] = float(
            probe_old_policy_surrogate_loss
        )
        metrics["embodied_action_token_grpo/probe_surrogate_loss_delta"] = float(
            probe_surrogate_loss_delta
        )
        metrics["embodied_action_token_grpo/probe_unclipped_surrogate_loss"] = float(
            probe_unclipped_surrogate_loss
        )
        metrics[
            "embodied_action_token_grpo/probe_old_policy_unclipped_surrogate_loss"
        ] = float(probe_old_policy_unclipped_surrogate_loss)
        metrics["embodied_action_token_grpo/probe_unclipped_surrogate_loss_delta"] = (
            float(probe_unclipped_surrogate_loss_delta)
        )
        metrics[
            "embodied_action_token_grpo/probe_clipping_surrogate_loss_delta_gap"
        ] = float(probe_surrogate_loss_delta - probe_unclipped_surrogate_loss_delta)
        metrics["embodied_action_token_grpo/probe_total_seconds"] = float(
            time.perf_counter() - probe_start
        )
        metrics["embodied_action_token_grpo/probe_logprob_forward_seconds"] = float(
            logprob_forward_seconds
        )
        metrics.update(
            {key: float(value) for key, value in kwargs.items() if _is_number(value)}
        )
        if _policy_gradient_algorithm_name(self.importance_sampling_level) == "gspo":
            metrics.update(
                _alias_metric_prefix(
                    metrics,
                    source_prefix="embodied_action_token_grpo/",
                    target_prefix="embodied_action_token_gspo/",
                    drop_source=True,
                )
            )
        return metrics

    def probe_logprob_debug_samples(
        self,
        trajectory_groups: Iterable[EmbodiedTrajectoryGroup],
        *,
        max_examples: int = 3,
        max_tokens: int = 8,
    ) -> dict[str, Any]:
        """Return bounded old/current logprob rows for replay-alignment debugging.

        This is intentionally a compact diagnostic, not a training signal.  Long
        OpenVLA-OFT runs need a way to tell whether a replay mismatch comes from
        all tokens, a subset of action spans, or metadata/config drift without
        dumping a full retained rollout pickle into JSON logs.
        """

        groups = list(trajectory_groups)
        if self.training_unit == "trajectory":
            examples = extract_trajectory_action_token_examples(
                groups,
                require_logprobs=True,
                require_observations=self.require_observations,
                require_prompts=self.require_prompts,
            )
        else:
            examples = extract_action_token_examples(
                groups,
                require_logprobs=True,
                require_observations=self.require_observations,
                require_prompts=self.require_prompts,
                score_source=self.rlinf_action_level_score_source,
            )
        reward_filter_report = _reward_filter_report(
            examples,
            enabled=self.filter_rewards,
            lower=self.rewards_lower_bound,
            upper=self.rewards_upper_bound,
        )
        if self.filter_rewards:
            examples = _apply_reward_filter_to_examples(
                examples,
                reward_filter_report=reward_filter_report,
                mode=self.reward_filter_mode,
            )
        sample_count = max(0, min(int(max_examples), len(examples)))
        token_limit = max(0, int(max_tokens))
        sample_examples = examples[:sample_count]
        if not sample_examples:
            return {
                "examples_before_filter": int(reward_filter_report["examples_before"]),
                "examples_after_filter": int(len(examples)),
                "samples": [],
            }

        if self.logprob_eval_mode:
            _set_eval(self.policy)
        else:
            _set_train(self.policy)

        import torch

        microbatch_size = self.logprob_microbatch_size or max(1, len(sample_examples))
        rows: list[dict[str, Any]] = []
        with torch.no_grad():
            for microbatch_start in range(0, len(sample_examples), microbatch_size):
                microbatch_examples = sample_examples[
                    microbatch_start : microbatch_start + microbatch_size
                ]
                current_rows = _policy_action_token_logprobs(
                    self.policy,
                    microbatch_examples,
                    device=self.device,
                    pad_to_batch_size=microbatch_size,
                )
                for local_index, (example, current) in enumerate(
                    zip(microbatch_examples, current_rows, strict=True)
                ):
                    example_index = microbatch_start + local_index
                    old = torch.as_tensor(
                        example.logprobs,
                        dtype=current.dtype,
                        device=current.device,
                    )
                    if old.shape != current.shape:
                        rows.append(
                            {
                                "example_index": int(example_index),
                                "trajectory_index": int(example.trajectory_index),
                                "action_index": int(example.action_index),
                                "shape_mismatch": True,
                                "old_shape": list(old.shape),
                                "current_shape": list(current.shape),
                            }
                        )
                        continue
                    delta = (current - old).detach().float().cpu()
                    token_loss_mask = (
                        _example_token_loss_mask_tensor(
                            example,
                            token_count=int(current.numel()),
                            device=str(current.device),
                            dtype=current.dtype,
                        )
                        .detach()
                        .float()
                        .cpu()
                    )
                    rows.append(
                        {
                            "example_index": int(example_index),
                            "trajectory_index": int(example.trajectory_index),
                            "action_index": int(example.action_index),
                            "step": int(example.step),
                            "reward": float(example.reward),
                            "token_count": int(current.numel()),
                            "tokens_head": [
                                int(token) if isinstance(token, int) else str(token)
                                for token in example.tokens[:token_limit]
                            ],
                            "old_logprobs_head": _tensor_head(old, token_limit),
                            "current_logprobs_head": _tensor_head(current, token_limit),
                            "delta_head": _tensor_head(delta, token_limit),
                            "loss_mask_head": _tensor_head(
                                token_loss_mask, token_limit
                            ),
                            "delta_abs_mean": float(delta.abs().mean().item())
                            if delta.numel()
                            else 0.0,
                            "delta_abs_max": float(delta.abs().max().item())
                            if delta.numel()
                            else 0.0,
                            "metadata_keys": sorted(
                                str(key) for key in example.metadata.keys()
                            ),
                            "action_spans": make_json_safe(
                                example.metadata.get("action_spans")
                            )
                            if isinstance(example.metadata, dict)
                            else None,
                        }
                    )
        return {
            "examples_before_filter": int(reward_filter_report["examples_before"]),
            "examples_after_filter": int(len(examples)),
            "max_examples": int(max_examples),
            "max_tokens": int(max_tokens),
            "logprob_microbatch_size": int(microbatch_size),
            "samples": rows,
        }

    async def close(self) -> None:
        return None

    def _ensure_optimizer(self) -> None:
        if self.optimizer is not None:
            return
        import torch

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.lr,
            betas=(self.optimizer_adam_beta1, self.optimizer_adam_beta2),
            eps=self.optimizer_adam_eps,
            weight_decay=self.optimizer_weight_decay,
        )


class ActionTokenGSPOBackend(ActionTokenGRPOBackend):
    """Trajectory-level action-token backend with GSPO sequence ratios enabled.

    Equivalent to ``ActionTokenGRPOBackend(...,
    importance_sampling_level="sequence", training_unit="trajectory")``.
    One complete embodied rollout is one sequence: per-action token logprobs
    are concatenated, the log importance ratio is length-normalized across the
    complete trajectory, and the group-relative trajectory advantage weights
    that scalar ratio. The separate class keeps this contract explicit instead
    of silently treating each action chunk as an independent GSPO sequence.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("importance_sampling_level", "sequence")
        kwargs.setdefault("training_unit", "trajectory")
        super().__init__(*args, **kwargs)


def _validate_group_relative_groups(
    groups: Sequence[EmbodiedTrajectoryGroup],
    *,
    backend_name: str,
) -> None:
    singleton_indices = [index for index, group in enumerate(groups) if len(group) < 2]
    if singleton_indices:
        raise ValueError(
            f"{backend_name} requires every trajectory group to contain at least "
            "two comparable rollouts. Group-relative advantages are undefined "
            f"for singleton groups: {singleton_indices}"
        )


def extract_action_token_examples(
    trajectory_groups: Iterable[EmbodiedTrajectoryGroup],
    *,
    require_logprobs: bool = False,
    require_observations: bool = False,
    require_prompts: bool = False,
    score_source: str = "chunk_rewards",
) -> list[ActionTokenExample]:
    """Flatten grouped trajectories into auditable action-token training rows.

    Group and trajectory coordinates are retained in metadata so later
    advantage, filtering, and denominator transforms can reconstruct the
    original same-reset comparison geometry.
    """

    if score_source not in ("chunk_rewards", "trajectory_reward"):
        raise ValueError("score_source must be 'chunk_rewards' or 'trajectory_reward'")

    examples: list[ActionTokenExample] = []
    trajectory_index = 0
    for group_index, group in enumerate(trajectory_groups):
        group_examples: list[ActionTokenExample] = []
        trajectories = list(group)
        group_rewards = [float(trajectory.reward) for trajectory in trajectories]
        group_reward_mean = (
            sum(group_rewards) / len(group_rewards) if group_rewards else 0.0
        )
        for trajectory_index_in_group, trajectory in enumerate(trajectories):
            trajectory_example_start = len(group_examples)
            for action_index, action in enumerate(trajectory.actions):
                if action.kind != "token":
                    continue
                tokens = _extract_tokens(action)
                if not tokens:
                    continue
                logprobs = _extract_logprobs(action)
                if require_logprobs and logprobs is None:
                    raise ValueError(
                        "Action-token example is missing logprobs. "
                        "Set require_logprobs=False for rollout-only validation."
                    )
                observation = _find_observation_for_action(trajectory, action)
                if require_observations and observation is None:
                    raise ValueError(
                        "Action-token example is missing the source observation. "
                        "Record observations at the same step as actions or set "
                        "require_observations=False."
                    )
                prompt = _extract_prompt(action)
                if require_prompts and prompt is None:
                    raise ValueError(
                        "Action-token example is missing the source prompt. "
                        "Store the prompt in action.raw['prompt'] or set "
                        "require_prompts=False."
                    )
                if _is_fully_masked_action(action):
                    continue
                action_metadata = (
                    action.metadata if isinstance(action.metadata, dict) else {}
                )
                token_loss_mask = action_metadata.get("token_loss_mask")
                if token_loss_mask is not None:
                    if not isinstance(token_loss_mask, list | tuple):
                        raise ValueError(
                            "action token_loss_mask must be a list or tuple"
                        )
                    if len(token_loss_mask) != len(tokens):
                        raise ValueError(
                            "action token_loss_mask length must match action tokens"
                        )
                if score_source == "trajectory_reward":
                    action_reward = float(trajectory.reward)
                    action_reward_source = "trajectory_reward"
                else:
                    action_reward, action_reward_source = (
                        _action_level_reward_for_action(trajectory, action)
                    )
                group_examples.append(
                    ActionTokenExample(
                        task=trajectory.task,
                        trajectory_index=trajectory_index,
                        action_index=action_index,
                        step=action.step,
                        tokens=tokens,
                        reward=action_reward,
                        prompt=prompt,
                        observation=observation,
                        decoded_action=action.decoded,
                        logprobs=logprobs,
                        metadata={
                            "group_index": group_index,
                            "trajectory_index_in_group": trajectory_index_in_group,
                            "group_size": len(trajectories),
                            "group_reward_mean": group_reward_mean,
                            "group_advantage": (
                                float(action_reward) - group_reward_mean
                                if score_source == "trajectory_reward"
                                else float(action_reward)
                            ),
                            "trajectory_reward": float(trajectory.reward),
                            "action_reward": float(action_reward),
                            "action_reward_source": action_reward_source,
                            "trajectory_metadata": trajectory.metadata,
                            "action_metadata": action.metadata,
                            **(
                                {"token_loss_mask": list(token_loss_mask)}
                                if token_loss_mask is not None
                                else {}
                            ),
                            "observation_index": _find_observation_index_for_action(
                                trajectory, action
                            ),
                        },
                    )
                )
            trajectory_action_count = len(group_examples) - trajectory_example_start
            for example in group_examples[trajectory_example_start:]:
                example.metadata["trajectory_action_count"] = trajectory_action_count
            trajectory_index += 1
        group_action_reward_mean = (
            sum(float(example.reward) for example in group_examples)
            / len(group_examples)
            if group_examples
            else 0.0
        )
        for example in group_examples:
            example.metadata["group_action_reward_mean"] = float(
                group_action_reward_mean
            )
            if score_source != "trajectory_reward":
                example.metadata["group_advantage"] = float(example.reward) - float(
                    group_action_reward_mean
                )
        examples.extend(group_examples)
    return examples


def _action_level_reward_for_action(
    trajectory: EmbodiedTrajectory, action: Action
) -> tuple[float, str]:
    """Return the reward assigned to one token action/chunk.

    World-model VLA rollouts can attach multiple primitive-step RewardEvents to
    a single chunked token action. For RLinf-style ``reward_type=action_level``
    training, the example reward should be the sum of reward events that share
    the action's ``policy_step``. If no such events exist, fall back to the full
    trajectory reward to preserve compatibility with older examples.
    """

    policy_step = None
    if isinstance(action.metadata, dict) and "policy_step" in action.metadata:
        try:
            policy_step = int(action.metadata["policy_step"])
        except Exception:
            policy_step = None
    if policy_step is not None:
        matching_events = [
            event
            for event in trajectory.rewards
            if isinstance(event.metadata, dict)
            and int(event.metadata.get("policy_step", -1)) == policy_step
        ]
        if matching_events and any(
            "primitive_loss_mask" in event.metadata for event in matching_events
        ):
            values = [
                float(event.value)
                for event in matching_events
                if bool(event.metadata.get("primitive_loss_mask", True))
            ]
            return float(sum(values)), "reward_events_matching_policy_step_loss_masked"
        values = [float(event.value) for event in matching_events]
        if values:
            return float(sum(values)), "reward_events_matching_policy_step"
    return float(trajectory.reward), "trajectory_reward_fallback"


def _is_fully_masked_action(action: Action) -> bool:
    """Return True for fixed-shape rollout actions excluded by RLinf loss mask."""

    metadata = action.metadata if isinstance(action.metadata, dict) else {}
    if "primitive_loss_mask_sum" in metadata:
        try:
            return float(metadata["primitive_loss_mask_sum"]) <= 0.0
        except Exception:
            return False
    mask = metadata.get("primitive_loss_mask")
    if isinstance(mask, (list, tuple)):
        return not any(bool(value) for value in mask)
    return False


def extract_trajectory_action_token_examples(
    trajectory_groups: Iterable[EmbodiedTrajectoryGroup],
    *,
    require_logprobs: bool = False,
    require_observations: bool = False,
    require_prompts: bool = False,
) -> list[ActionTokenExample]:
    """Extract one action-token training example per trajectory.

    This is the trajectory-level unit needed for VLA-style GSPO benchmarks:
    all token actions in one rollout are concatenated into one likelihood span,
    while the reward and group-relative advantage remain trajectory-level.
    Policy adapters still receive the full ``ActionTokenExample`` and can use
    ``metadata['action_spans']`` / ``metadata['prompts']`` when they need to map
    the flat token sequence back to per-step observations.
    """

    examples: list[ActionTokenExample] = []
    trajectory_index = 0
    for group_index, group in enumerate(trajectory_groups):
        trajectories = list(group)
        group_rewards = [float(trajectory.reward) for trajectory in trajectories]
        group_reward_mean = (
            sum(group_rewards) / len(group_rewards) if group_rewards else 0.0
        )
        for trajectory_index_in_group, trajectory in enumerate(trajectories):
            tokens: list[ActionToken] = []
            logprobs: list[float] = []
            reference_logprobs: list[float] = []
            logprobs_seen = False
            logprobs_missing_seen = False
            reference_logprobs_seen = False
            reference_logprobs_missing_seen = False
            prompts: list[str | None] = []
            decoded_actions: list[Any] = []
            action_spans: list[dict[str, Any]] = []
            action_observations: list[Observation | None] = []
            first_observation: Observation | None = None
            first_step: int | None = None
            for action_index, action in enumerate(trajectory.actions):
                if action.kind != "token":
                    continue
                action_tokens = _extract_tokens(action)
                if not action_tokens:
                    continue
                action_logprobs = _extract_logprobs(action)
                if action_logprobs is None:
                    logprobs_missing_seen = True
                if require_logprobs and action_logprobs is None:
                    raise ValueError(
                        "Trajectory action-token example is missing logprobs. "
                        "Set require_logprobs=False for rollout-only validation."
                    )
                action_reference_logprobs = _extract_reference_logprobs_from_action(
                    action
                )
                if action_reference_logprobs is None:
                    reference_logprobs_missing_seen = True
                else:
                    reference_logprobs_seen = True
                logprob_convention = _action_token_logprob_convention(
                    token_count=len(action_tokens),
                    logprob_count=len(action_logprobs)
                    if action_logprobs is not None
                    else None,
                )
                if action_logprobs is not None and logprob_convention is None:
                    raise ValueError(
                        "Action-token logprobs must either match token count or use "
                        "the shifted next-token convention with one fewer logprob "
                        "than tokens"
                    )
                observation = _find_observation_for_action(trajectory, action)
                if first_observation is None and observation is not None:
                    first_observation = observation
                prompt = _extract_prompt(action)
                if require_prompts and prompt is None:
                    raise ValueError(
                        "Trajectory action-token example has an action missing a prompt. "
                        "Store prompts in action.raw['prompt'] or set require_prompts=False."
                    )
                start = len(tokens)
                logprob_start = len(logprobs)
                tokens.extend(action_tokens)
                if action_logprobs is not None:
                    logprobs_seen = True
                    logprobs.extend(action_logprobs)
                if action_reference_logprobs is not None:
                    reference_logprob_convention = _action_token_logprob_convention(
                        token_count=len(action_tokens),
                        logprob_count=len(action_reference_logprobs),
                    )
                    if reference_logprob_convention is None:
                        raise ValueError(
                            "Reference action-token logprobs must either match token count "
                            "or use the shifted next-token convention with one fewer logprob "
                            "than tokens"
                        )
                    reference_logprobs.extend(action_reference_logprobs)
                else:
                    reference_logprob_convention = None
                prompts.append(prompt)
                decoded_actions.append(action.decoded)
                action_observations.append(observation)
                span = {
                    "action_index": action_index,
                    "step": action.step,
                    "token_start": start,
                    "token_end": len(tokens),
                    "logprob_start": logprob_start,
                    "logprob_end": len(logprobs),
                    "token_count": len(action_tokens),
                    "logprob_count": len(action_logprobs)
                    if action_logprobs is not None
                    else None,
                    "logprob_convention": logprob_convention,
                    "observation_index": _find_observation_index_for_action(
                        trajectory, action
                    ),
                }
                span_action_metadata = _rlinf_action_level_span_metadata(
                    action.metadata
                )
                if span_action_metadata:
                    span["action_metadata"] = span_action_metadata
                if action_reference_logprobs is not None:
                    span["reference_logprob_count"] = len(action_reference_logprobs)
                    span["reference_logprob_convention"] = reference_logprob_convention
                action_spans.append(span)
                if first_step is None:
                    first_step = action.step
            if not tokens:
                trajectory_index += 1
                continue
            if require_observations and first_observation is None:
                raise ValueError(
                    "Trajectory action-token example is missing source observations. "
                    "Record observations at token-action steps or set require_observations=False."
                )
            if logprobs_seen and logprobs_missing_seen:
                raise ValueError(
                    "Trajectory-level action-token extraction requires either logprobs "
                    "for every token action or no logprobs for the trajectory"
                )
            metadata = {
                "training_unit": "trajectory",
                "group_index": group_index,
                "trajectory_index_in_group": trajectory_index_in_group,
                "group_size": len(trajectories),
                "group_reward_mean": group_reward_mean,
                "group_advantage": float(trajectory.reward) - group_reward_mean,
                "trajectory_metadata": trajectory.metadata,
                "action_count": len(action_spans),
                "action_spans": action_spans,
                "action_observations": action_observations,
                "prompts": prompts,
            }
            if reference_logprobs_seen and not reference_logprobs_missing_seen:
                metadata["reference_logprobs"] = reference_logprobs
            elif reference_logprobs_seen and reference_logprobs_missing_seen:
                metadata["reference_logprobs_mixed"] = True
            examples.append(
                ActionTokenExample(
                    task=trajectory.task,
                    trajectory_index=trajectory_index,
                    action_index=-1,
                    step=int(first_step if first_step is not None else 0),
                    tokens=tokens,
                    reward=trajectory.reward,
                    prompt=next((prompt for prompt in prompts if prompt), None),
                    observation=first_observation,
                    decoded_action=decoded_actions,
                    logprobs=logprobs if logprobs_seen else None,
                    metadata=metadata,
                )
            )
            trajectory_index += 1
    return examples


def _extract_tokens(action: Action) -> list[ActionToken]:
    raw = action.raw
    if isinstance(raw, dict) and "tokens" in raw:
        raw = raw["tokens"]
    if isinstance(raw, list | tuple):
        return [_coerce_token(token) for token in raw]
    return [_coerce_token(raw)]


def _extract_logprobs(action: Action) -> list[float] | None:
    logprobs = action.logprobs
    if logprobs is None:
        return None
    if isinstance(logprobs, dict):
        for key in ("token_logprobs", "logprobs", "values"):
            if key in logprobs:
                logprobs = logprobs[key]
                break
    if isinstance(logprobs, list | tuple):
        return [float(value) for value in logprobs]
    return [float(logprobs)]


def _extract_reference_logprobs_from_action(action: Action) -> list[float] | None:
    if not isinstance(action.metadata, dict):
        return None
    for key in (
        "reference_logprobs",
        "reference_token_logprobs",
        "static_reference_logprobs",
    ):
        if key in action.metadata:
            return _coerce_logprob_list(action.metadata[key])
    return None


_RLINF_ACTION_LEVEL_SPAN_METADATA_KEYS = (
    "primitive_rewards",
    "primitive_loss_mask",
    "primitive_loss_mask_sum",
    "policy_step",
)


def _rlinf_action_level_span_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    span_metadata = {
        key: make_json_safe(metadata[key])
        for key in _RLINF_ACTION_LEVEL_SPAN_METADATA_KEYS
        if key in metadata
    }
    # This key intentionally remains raw and transient. It carries the exact
    # RLinf env_obs dict used for rollout into logprob rescoring/training, then
    # the native runner scrubs it before any persistent logging or pickling.
    if TRANSIENT_RLINF_ENV_OBS_METADATA_KEY in metadata:
        span_metadata[TRANSIENT_RLINF_ENV_OBS_METADATA_KEY] = metadata[
            TRANSIENT_RLINF_ENV_OBS_METADATA_KEY
        ]
    if TRANSIENT_RLINF_FORWARD_INPUTS_METADATA_KEY in metadata:
        span_metadata[TRANSIENT_RLINF_FORWARD_INPUTS_METADATA_KEY] = metadata[
            TRANSIENT_RLINF_FORWARD_INPUTS_METADATA_KEY
        ]
    return span_metadata


def _example_reference_logprobs(example: ActionTokenExample) -> list[float] | None:
    metadata = example.metadata if isinstance(example.metadata, dict) else {}
    for key in (
        "reference_logprobs",
        "reference_token_logprobs",
        "static_reference_logprobs",
    ):
        if key in metadata:
            return _coerce_logprob_list(metadata[key])

    action_metadata = metadata.get("action_metadata")
    if isinstance(action_metadata, dict):
        for key in (
            "reference_logprobs",
            "reference_token_logprobs",
            "static_reference_logprobs",
        ):
            if key in action_metadata:
                return _coerce_logprob_list(action_metadata[key])
    return None


def _should_stream_trajectory_token_spans(
    examples: Sequence[ActionTokenExample],
    *,
    training_unit: str,
    importance_sampling_level: str,
    loss_aggregation: str,
) -> bool:
    """Return whether trajectory examples can be backpropagated one action span at a time.

    This is an exact memory-saving path for token-level GRPO with token-mean
    aggregation: every action token receives the same trajectory-level advantage
    as before, but the backend scores/backprops action spans in microbatches
    instead of asking the VLA policy adapter to concatenate a whole rollout's
    graphs.

    Sequence-level GSPO uses a separate two-pass streaming path because its
    likelihood ratio is defined over the full concatenated sequence.
    """

    if training_unit != "trajectory":
        return False
    if importance_sampling_level != "token" or loss_aggregation not in (
        "token_mean",
        "rlinf_token_mean",
        "rlinf_chunk_mean",
        "rlinf_masked_mean_ratio",
    ):
        return False
    return all(_trajectory_token_spans_are_streamable(example) for example in examples)


def _should_stream_trajectory_sequence_spans(
    examples: Sequence[ActionTokenExample],
    *,
    training_unit: str,
    importance_sampling_level: str,
    loss_aggregation: str,
) -> bool:
    """Use exact two-pass span streaming for trajectory-level GSPO."""

    if (
        training_unit != "trajectory"
        or importance_sampling_level != "sequence"
        or loss_aggregation != "trajectory_mean"
    ):
        return False
    return all(_trajectory_token_spans_are_streamable(example) for example in examples)


def _trajectory_token_spans_are_streamable(example: ActionTokenExample) -> bool:
    metadata = example.metadata if isinstance(example.metadata, dict) else {}
    if metadata.get("training_unit") != "trajectory":
        return False
    spans = metadata.get("action_spans")
    if not isinstance(spans, list) or not spans:
        return False
    if example.logprobs is None:
        return False
    reference_logprobs = _example_reference_logprobs(example)
    for span in spans:
        if not isinstance(span, dict):
            return False
        token_start = int(span.get("token_start", 0))
        token_end = int(span.get("token_end", token_start))
        logprob_start = int(span.get("logprob_start", token_start))
        logprob_end = int(span.get("logprob_end", token_end))
        if token_end <= token_start or logprob_end <= logprob_start:
            return False
        # The streaming path is exact when the policy recomputes one logprob per
        # rollout logprob. Shifted-next-token examples can be supported later, but
        # failing closed avoids silently changing the policy-gradient objective.
        if (token_end - token_start) != (logprob_end - logprob_start):
            return False
        if logprob_end > len(example.logprobs):
            return False
        if reference_logprobs is not None and logprob_end > len(reference_logprobs):
            return False
    return True


def _iter_trajectory_action_span_examples(
    example: ActionTokenExample,
) -> Iterable[ActionTokenExample]:
    metadata = example.metadata if isinstance(example.metadata, dict) else {}
    spans = metadata.get("action_spans") or []
    observations = metadata.get("action_observations") or []
    prompts = metadata.get("prompts") or []
    decoded_actions = (
        example.decoded_action if isinstance(example.decoded_action, list) else []
    )
    reference_logprobs = _example_reference_logprobs(example)
    token_loss_mask = metadata.get("token_loss_mask")
    if token_loss_mask is not None and not isinstance(token_loss_mask, list | tuple):
        raise ValueError("token_loss_mask must be a list or tuple")
    if example.logprobs is None:
        raise ValueError("Trajectory span streaming requires old rollout logprobs")

    for span_index, span in enumerate(spans):
        token_start = int(span.get("token_start", 0))
        token_end = int(span.get("token_end", token_start))
        logprob_start = int(span.get("logprob_start", token_start))
        logprob_end = int(span.get("logprob_end", token_end))
        span_reference_logprobs = (
            reference_logprobs[logprob_start:logprob_end]
            if reference_logprobs is not None
            else None
        )
        span_metadata = {
            "training_unit": "trajectory_span",
            "trajectory_training_unit": "trajectory",
            "parent_trajectory_index": example.trajectory_index,
            "span_index": span_index,
            "action_span": span,
        }
        action_metadata = span.get("action_metadata")
        if isinstance(action_metadata, dict):
            span_metadata["action_metadata"] = action_metadata
        if span_reference_logprobs is not None:
            span_metadata["reference_logprobs"] = span_reference_logprobs
        if token_loss_mask is not None:
            span_metadata["token_loss_mask"] = list(
                token_loss_mask[logprob_start:logprob_end]
            )

        yield ActionTokenExample(
            task=example.task,
            trajectory_index=example.trajectory_index,
            action_index=int(span.get("action_index", span_index)),
            step=int(span.get("step", example.step)),
            tokens=example.tokens[token_start:token_end],
            reward=example.reward,
            prompt=prompts[span_index] if span_index < len(prompts) else example.prompt,
            observation=observations[span_index]
            if span_index < len(observations)
            else example.observation,
            decoded_action=(
                decoded_actions[span_index]
                if span_index < len(decoded_actions)
                else None
            ),
            logprobs=example.logprobs[logprob_start:logprob_end],
            metadata=span_metadata,
        )


def refresh_action_token_logprobs(
    policy: Any,
    trajectory_groups: Iterable[EmbodiedTrajectoryGroup],
    *,
    device: str,
    training_unit: str = "trajectory",
    microbatch_size: int | None = None,
    pad_to_batch_size: int | None = None,
    require_observations: bool = True,
    require_prompts: bool = True,
    source: str = "pre_train_policy_rescore",
    progress_path: str | Path | None = None,
    progress_every_microbatches: int = 128,
    progress_max_bytes: int = 64 * 1024 * 1024,
) -> dict[str, Any]:
    """Recompute and replace rollout old logprobs before policy-gradient training.

    PPO-style objectives only require old logprobs under the policy that produced
    the trajectories; they do not have to be computed during the environment
    step itself.  VLA remote-code paths can be batch-size sensitive after LoRA
    updates, so long-running robotics rollouts may collect actions with a large
    rollout batch while the gradient path uses smaller microbatches.  This helper
    normalizes the stored old logprobs to the same scorer path used for training
    without changing the sampled actions or rewards.
    """

    groups = list(trajectory_groups)
    resolved_progress_path = Path(progress_path) if progress_path is not None else None
    if int(progress_every_microbatches) <= 0:
        raise ValueError("progress_every_microbatches must be positive")
    if int(progress_max_bytes) <= 0:
        raise ValueError("progress_max_bytes must be positive")
    flat_trajectories = [trajectory for group in groups for trajectory in group]
    if training_unit == "trajectory":
        examples = extract_trajectory_action_token_examples(
            groups,
            require_logprobs=True,
            require_observations=require_observations,
            require_prompts=require_prompts,
        )
    elif training_unit == "action":
        examples = extract_action_token_examples(
            groups,
            require_logprobs=True,
            require_observations=require_observations,
            require_prompts=require_prompts,
        )
    else:
        raise ValueError(
            "refresh_action_token_logprobs supports training_unit='trajectory' "
            f"or 'action', got {training_unit!r}"
        )

    if not examples:
        return {
            "enabled": True,
            "source": source,
            "training_unit": training_unit,
            "examples": 0,
            "actions_updated": 0,
            "tokens_updated": 0,
            "microbatch_size": 0,
            "pad_to_batch_size": 0,
        }

    batch_size = int(microbatch_size or len(examples))
    if batch_size <= 0:
        raise ValueError(
            "microbatch_size must be positive when refreshing action-token logprobs"
        )
    scorer_pad = int(pad_to_batch_size or batch_size)
    if scorer_pad <= 0:
        raise ValueError(
            "pad_to_batch_size must be positive when refreshing action-token logprobs"
        )

    actions_updated = 0
    tokens_updated = 0
    previous_delta_values: list[float] = []
    call_count = 0
    scorer_examples_count = len(examples)
    refresh_start = time.perf_counter()

    if training_unit == "trajectory":
        span_work: list[tuple[int, ActionTokenExample]] = []
        refreshed_by_parent: list[list[float | None]] = []
        for parent_index, example in enumerate(examples):
            if example.logprobs is None:
                raise ValueError(
                    "Trajectory old-logprob rescore requires rollout logprobs"
                )
            refreshed_by_parent.append([None for _ in example.logprobs])
            for span_example in _iter_trajectory_action_span_examples(example):
                span_work.append((parent_index, span_example))

        scorer_examples_count = len(span_work)
        for batch_start in range(0, len(span_work), batch_size):
            batch_items = span_work[batch_start : batch_start + batch_size]
            batch = [span_example for _parent_index, span_example in batch_items]
            _maybe_write_action_token_rescore_progress(
                progress_path=resolved_progress_path,
                progress_every_microbatches=progress_every_microbatches,
                progress_max_bytes=progress_max_bytes,
                event="action_token_logprob_rescore_microbatch_started",
                microbatch_count=call_count + 1,
                microbatch_start=batch_start,
                total_items=len(span_work),
                microbatch_size=batch_size,
                training_unit=training_unit,
                source=source,
                elapsed_seconds=time.perf_counter() - refresh_start,
            )
            refreshed_rows = _policy_action_token_logprobs(
                policy,
                batch,
                device=device,
                pad_to_batch_size=scorer_pad,
            )
            call_count += 1
            for (parent_index, span_example), row in zip(
                batch_items, refreshed_rows, strict=True
            ):
                refreshed = _logprob_row_to_float_list(row)
                span_metadata = (
                    span_example.metadata
                    if isinstance(span_example.metadata, dict)
                    else {}
                )
                span = (
                    span_metadata.get("action_span")
                    if isinstance(span_metadata, dict)
                    else None
                )
                if not isinstance(span, dict):
                    raise ValueError(
                        "Trajectory span rescore example is missing action_span metadata"
                    )
                logprob_start = int(
                    span.get("logprob_start", span.get("token_start", 0))
                )
                logprob_end = int(
                    span.get("logprob_end", span.get("token_end", logprob_start))
                )
                expected = logprob_end - logprob_start
                if len(refreshed) != expected:
                    raise ValueError(
                        "Refreshed trajectory-span logprob length does not match span: "
                        f"got {len(refreshed)}, expected {expected}"
                    )
                refreshed_by_parent[parent_index][logprob_start:logprob_end] = refreshed
            _maybe_write_action_token_rescore_progress(
                progress_path=resolved_progress_path,
                progress_every_microbatches=progress_every_microbatches,
                progress_max_bytes=progress_max_bytes,
                event="action_token_logprob_rescore_microbatch_completed",
                microbatch_count=call_count,
                microbatch_start=batch_start,
                total_items=len(span_work),
                microbatch_size=batch_size,
                training_unit=training_unit,
                source=source,
                elapsed_seconds=time.perf_counter() - refresh_start,
            )

        for example, refreshed in zip(examples, refreshed_by_parent, strict=True):
            if any(value is None for value in refreshed):
                raise ValueError(
                    "Trajectory old-logprob rescore did not cover every span logprob"
                )
            update = _write_refreshed_action_token_logprobs(
                example,
                [float(value) for value in refreshed],
                flat_trajectories,
                source=source,
                scorer_batch_size=batch_size,
                pad_to_batch_size=scorer_pad,
            )
            actions_updated += int(update["actions_updated"])
            tokens_updated += int(update["tokens_updated"])
            previous_delta_values.extend(update["previous_abs_delta_values"])
    else:
        for batch_start in range(0, len(examples), batch_size):
            batch = examples[batch_start : batch_start + batch_size]
            _maybe_write_action_token_rescore_progress(
                progress_path=resolved_progress_path,
                progress_every_microbatches=progress_every_microbatches,
                progress_max_bytes=progress_max_bytes,
                event="action_token_logprob_rescore_microbatch_started",
                microbatch_count=call_count + 1,
                microbatch_start=batch_start,
                total_items=len(examples),
                microbatch_size=batch_size,
                training_unit=training_unit,
                source=source,
                elapsed_seconds=time.perf_counter() - refresh_start,
            )
            refreshed_rows = _policy_action_token_logprobs(
                policy,
                batch,
                device=device,
                pad_to_batch_size=scorer_pad,
            )
            call_count += 1
            for example, row in zip(batch, refreshed_rows, strict=True):
                refreshed = _logprob_row_to_float_list(row)
                update = _write_refreshed_action_token_logprobs(
                    example,
                    refreshed,
                    flat_trajectories,
                    source=source,
                    scorer_batch_size=len(batch),
                    pad_to_batch_size=scorer_pad,
                )
                actions_updated += int(update["actions_updated"])
                tokens_updated += int(update["tokens_updated"])
                previous_delta_values.extend(update["previous_abs_delta_values"])
            _maybe_write_action_token_rescore_progress(
                progress_path=resolved_progress_path,
                progress_every_microbatches=progress_every_microbatches,
                progress_max_bytes=progress_max_bytes,
                event="action_token_logprob_rescore_microbatch_completed",
                microbatch_count=call_count,
                microbatch_start=batch_start,
                total_items=len(examples),
                microbatch_size=batch_size,
                training_unit=training_unit,
                source=source,
                elapsed_seconds=time.perf_counter() - refresh_start,
            )

    return {
        "enabled": True,
        "source": source,
        "training_unit": training_unit,
        "examples": len(examples),
        "scorer_examples": scorer_examples_count,
        "actions_updated": actions_updated,
        "tokens_updated": tokens_updated,
        "microbatch_size": batch_size,
        "pad_to_batch_size": scorer_pad,
        "policy_logprob_calls": call_count,
        "previous_abs_delta_mean": _mean(previous_delta_values),
        "previous_abs_delta_max": max(previous_delta_values)
        if previous_delta_values
        else 0.0,
    }


def rescore_action_token_examples(
    policy: Any,
    examples: Sequence[ActionTokenExample],
    *,
    device: str,
    training_unit: str,
    microbatch_size: int,
    pad_to_batch_size: int,
    source: str,
) -> dict[str, Any]:
    """Replace prepared-example old logprobs using the training scorer contract.

    This variant intentionally operates on prepared examples rather than source
    trajectories so local training workers can rescore disjoint shards without
    shipping complete trajectory groups back to the coordinator.
    """

    if microbatch_size <= 0 or pad_to_batch_size <= 0:
        raise ValueError("rescore microbatch and padding sizes must be positive")
    prepared = list(examples)
    previous_deltas: list[float] = []
    calls = 0

    if training_unit == "trajectory":
        work: list[tuple[int, ActionTokenExample]] = []
        refreshed_rows: list[list[float | None]] = []
        for parent_index, example in enumerate(prepared):
            if example.logprobs is None:
                raise ValueError("Trajectory rescore requires existing old logprobs")
            refreshed_rows.append([None] * len(example.logprobs))
            work.extend(
                (parent_index, span)
                for span in _iter_trajectory_action_span_examples(example)
            )
        for start in range(0, len(work), microbatch_size):
            items = work[start : start + microbatch_size]
            rows = _policy_action_token_logprobs(
                policy,
                [span for _parent, span in items],
                device=device,
                pad_to_batch_size=pad_to_batch_size,
            )
            calls += 1
            for (parent_index, span_example), row in zip(items, rows, strict=True):
                values = _logprob_row_to_float_list(row)
                span = span_example.metadata.get("action_span")
                if not isinstance(span, dict):
                    raise ValueError("Trajectory rescore span is missing metadata")
                start_index = int(span.get("logprob_start", span.get("token_start", 0)))
                end_index = int(
                    span.get("logprob_end", span.get("token_end", start_index))
                )
                if len(values) != end_index - start_index:
                    raise ValueError(
                        "Rescored trajectory span length does not match metadata"
                    )
                refreshed_rows[parent_index][start_index:end_index] = values
        for example, refreshed in zip(prepared, refreshed_rows, strict=True):
            if example.logprobs is None or any(value is None for value in refreshed):
                raise ValueError("Trajectory rescore did not cover every logprob")
            values = [float(value) for value in refreshed]
            previous_deltas.extend(
                abs(float(old) - new)
                for old, new in zip(example.logprobs, values, strict=True)
            )
            example.logprobs = values
            example.metadata["old_logprob_source"] = source
    elif training_unit == "action":
        for start in range(0, len(prepared), microbatch_size):
            batch = prepared[start : start + microbatch_size]
            rows = _policy_action_token_logprobs(
                policy,
                batch,
                device=device,
                pad_to_batch_size=pad_to_batch_size,
            )
            calls += 1
            for example, row in zip(batch, rows, strict=True):
                values = _logprob_row_to_float_list(row)
                previous = example.logprobs
                if previous is not None and len(values) != len(previous):
                    raise ValueError(
                        "Rescored action length does not match old logprobs"
                    )
                if previous is not None:
                    previous_deltas.extend(
                        abs(float(old) - new)
                        for old, new in zip(previous, values, strict=True)
                    )
                example.logprobs = values
                example.metadata["old_logprob_source"] = source
    else:
        raise ValueError(f"Unsupported rescore training_unit={training_unit!r}")

    return {
        "enabled": True,
        "source": source,
        "examples": len(prepared),
        "policy_logprob_calls": calls,
        "tokens_updated": sum(len(example.logprobs or ()) for example in prepared),
        "previous_abs_delta_mean": _mean(previous_deltas),
        "previous_abs_delta_max": max(previous_deltas) if previous_deltas else 0.0,
    }


def _logprob_row_to_float_list(row: Any) -> list[float]:
    if hasattr(row, "detach") and callable(row.detach):
        row = row.detach().float().cpu().tolist()
    if isinstance(row, list | tuple):
        return [float(value) for value in row]
    return [float(row)]


def _tensor_head(value: Any, limit: int) -> list[float]:
    if limit <= 0:
        return []
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().float().cpu().reshape(-1).tolist()
    if not isinstance(value, list | tuple):
        value = [value]
    return [float(item) for item in list(value)[: int(limit)]]


def _write_refreshed_action_token_logprobs(
    example: ActionTokenExample,
    refreshed_logprobs: list[float],
    trajectories: Sequence[EmbodiedTrajectory],
    *,
    source: str,
    scorer_batch_size: int,
    pad_to_batch_size: int,
) -> dict[str, Any]:
    if example.trajectory_index < 0 or example.trajectory_index >= len(trajectories):
        raise IndexError(
            f"Action-token example trajectory_index={example.trajectory_index} "
            f"is outside {len(trajectories)} trajectories"
        )
    trajectory = trajectories[example.trajectory_index]
    metadata = example.metadata if isinstance(example.metadata, dict) else {}
    spans = metadata.get("action_spans")
    if isinstance(spans, list) and spans:
        actions_updated = 0
        tokens_updated = 0
        deltas: list[float] = []
        for span in spans:
            if not isinstance(span, dict):
                continue
            action_index = int(span.get("action_index", -1))
            if action_index < 0 or action_index >= len(trajectory.actions):
                raise IndexError(
                    f"Action span index={action_index} is outside "
                    f"{len(trajectory.actions)} trajectory actions"
                )
            logprob_start = int(span.get("logprob_start", span.get("token_start", 0)))
            logprob_end = int(
                span.get("logprob_end", span.get("token_end", logprob_start))
            )
            span_logprobs = refreshed_logprobs[logprob_start:logprob_end]
            if not span_logprobs:
                continue
            delta = _replace_action_logprobs(
                trajectory.actions[action_index],
                span_logprobs,
                source=source,
                scorer_batch_size=scorer_batch_size,
                pad_to_batch_size=pad_to_batch_size,
            )
            deltas.extend(delta)
            actions_updated += 1
            tokens_updated += len(span_logprobs)
        return {
            "actions_updated": actions_updated,
            "tokens_updated": tokens_updated,
            "previous_abs_delta_values": deltas,
        }

    if example.action_index < 0 or example.action_index >= len(trajectory.actions):
        raise IndexError(
            f"Action-token example action_index={example.action_index} is outside "
            f"{len(trajectory.actions)} trajectory actions"
        )
    deltas = _replace_action_logprobs(
        trajectory.actions[example.action_index],
        refreshed_logprobs,
        source=source,
        scorer_batch_size=scorer_batch_size,
        pad_to_batch_size=pad_to_batch_size,
    )
    return {
        "actions_updated": 1,
        "tokens_updated": len(refreshed_logprobs),
        "previous_abs_delta_values": deltas,
    }


def _replace_action_logprobs(
    action: Action,
    refreshed_logprobs: list[float],
    *,
    source: str,
    scorer_batch_size: int,
    pad_to_batch_size: int,
) -> list[float]:
    previous = _extract_logprobs(action)
    deltas: list[float] = []
    if previous is not None and len(previous) == len(refreshed_logprobs):
        deltas = [
            abs(float(old) - float(new))
            for old, new in zip(previous, refreshed_logprobs, strict=True)
        ]
    action.logprobs = {"token_logprobs": list(refreshed_logprobs)}
    action.metadata["old_logprobs_rescore"] = {
        "source": source,
        "scorer_batch_size": int(scorer_batch_size),
        "pad_to_batch_size": int(pad_to_batch_size),
        "token_count": len(refreshed_logprobs),
        "previous_abs_delta_mean": _mean(deltas),
        "previous_abs_delta_max": max(deltas) if deltas else 0.0,
    }
    return deltas


def _coerce_logprob_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("token_logprobs", "logprobs", "values"):
            if key in value:
                value = value[key]
                break
    if isinstance(value, list | tuple):
        return [float(item) for item in value]
    return [float(value)]


def _action_token_logprob_convention(
    *,
    token_count: int,
    logprob_count: int | None,
) -> str | None:
    if logprob_count is None:
        return None
    if logprob_count == token_count:
        return "per_token"
    if token_count >= 2 and logprob_count == token_count - 1:
        return "shifted_next_token"
    return None


def _coerce_token(token: Any) -> ActionToken:
    if isinstance(token, int | str):
        return token
    if hasattr(token, "item") and callable(token.item):
        item = token.item()
        if isinstance(item, int | str):
            return item
    return str(token)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float | bool)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _gradient_direction_probe_presence_metrics(
    config: Any,
    *,
    prefix: str = "embodied_action_token_grpo",
) -> dict[str, float]:
    """Record whether train() received the gradient-direction probe config."""

    is_mapping = isinstance(config, Mapping)
    enabled = bool(config.get("enabled")) if is_mapping else False
    scales = config.get("scales") if is_mapping else None
    try:
        scale_count = len(scales) if scales is not None else 0
    except TypeError:
        scale_count = 0
    return {
        f"{prefix}/gradient_direction_probe_config_seen": float(config is not None),
        f"{prefix}/gradient_direction_probe_config_is_mapping": float(is_mapping),
        f"{prefix}/gradient_direction_probe_config_enabled": float(enabled),
        f"{prefix}/gradient_direction_probe_config_scale_count": float(scale_count),
    }


def _extract_prompt(action: Action) -> str | None:
    if isinstance(action.raw, dict) and isinstance(action.raw.get("prompt"), str):
        return action.raw["prompt"]
    return None


def _find_observation_for_action(
    trajectory: EmbodiedTrajectory, action: Action
) -> Observation | None:
    index = _find_observation_index_for_action(trajectory, action)
    if index is None:
        return None
    return trajectory.observations[index]


def _find_observation_index_for_action(
    trajectory: EmbodiedTrajectory, action: Action
) -> int | None:
    if isinstance(action.metadata, dict):
        explicit_index = action.metadata.get("observation_index")
        if isinstance(explicit_index, int) and 0 <= explicit_index < len(
            trajectory.observations
        ):
            return explicit_index
    for index, observation in enumerate(trajectory.observations):
        if observation.step == action.step:
            return index
    if 0 <= action.step < len(trajectory.observations):
        return action.step
    return None


def _policy_action_token_logprobs(
    policy: Any,
    examples: list[ActionTokenExample],
    *,
    device: str,
    pad_to_batch_size: int | None = None,
) -> list[Any]:
    """Call a policy-specific logprob method and normalize the result shape."""

    call_examples = examples
    original_count = len(examples)
    if (
        pad_to_batch_size is not None
        and original_count > 0
        and original_count < int(pad_to_batch_size)
    ):
        # Some VLA remote-code paths are not numerically batch-size invariant
        # after LoRA updates.  Keep the policy-gradient objective unchanged by
        # padding only the scorer call and slicing padded rows away before loss.
        call_examples = examples + [examples[-1]] * (
            int(pad_to_batch_size) - original_count
        )

    if hasattr(policy, "action_token_logprobs"):
        result = policy.action_token_logprobs(call_examples)
    elif hasattr(policy, "logprobs_for_action_tokens"):
        result = policy.logprobs_for_action_tokens(
            examples=call_examples,
            tokens=[example.tokens for example in call_examples],
            prompts=[example.prompt for example in call_examples],
            observations=[example.observation for example in call_examples],
        )
    elif callable(policy):
        result = policy(call_examples)
    else:
        raise TypeError(
            "ActionTokenGRPOBackend policy must provide action_token_logprobs, "
            "logprobs_for_action_tokens, or be callable on examples."
        )
    rows = _normalise_logprob_result(result, call_examples, device=device)
    return rows[:original_count]


def _validated_old_action_token_logprobs(
    example: ActionTokenExample,
    current_logprobs: Any,
    *,
    context: str,
) -> Any:
    import torch

    if example.logprobs is None:
        raise ValueError("Action-token GRPO requires old rollout logprobs")
    old_logprobs = torch.as_tensor(
        example.logprobs,
        dtype=current_logprobs.dtype,
        device=current_logprobs.device,
    )
    if old_logprobs.shape != current_logprobs.shape:
        raise ValueError(
            "New and old action-token logprobs must have the same shape for "
            f"{context}: new={tuple(current_logprobs.shape)}, "
            f"old={tuple(old_logprobs.shape)}"
        )
    return old_logprobs


def _validated_reference_action_token_logprobs(
    example: ActionTokenExample,
    current_logprobs: Any,
    *,
    require: bool,
    context: str,
) -> Any | None:
    import torch

    reference_logprobs = _example_reference_logprobs(example)
    if reference_logprobs is None:
        if require:
            raise ValueError(
                "Action-token GRPO/GSPO requires reference_logprobs, but "
                f"{context} has none"
            )
        return None
    reference = torch.as_tensor(
        reference_logprobs,
        dtype=current_logprobs.dtype,
        device=current_logprobs.device,
    )
    if reference.shape != current_logprobs.shape:
        raise ValueError(
            "Reference and current action-token logprobs must have the same shape for "
            f"{context}: reference={tuple(reference.shape)}, "
            f"current={tuple(current_logprobs.shape)}"
        )
    return reference


def _normalise_logprob_result(
    result: Any,
    examples: list[ActionTokenExample],
    *,
    device: str,
) -> list[Any]:
    import torch

    if isinstance(result, dict):
        for key in ("token_logprobs", "logprobs", "values"):
            if key in result:
                result = result[key]
                break
    if torch.is_tensor(result):
        if result.ndim == 1 and len(examples) == 1:
            return [result.to(device)]
        if result.ndim != 2 or result.shape[0] != len(examples):
            raise ValueError(
                "Tensor logprob result must be shape [batch, tokens] or [tokens] "
                "for a single example"
            )
        return [
            result[index, : len(example.tokens)].to(device)
            for index, example in enumerate(examples)
        ]
    if not isinstance(result, list | tuple):
        raise TypeError("Policy logprob result must be a tensor, list, tuple, or dict")
    if len(result) != len(examples):
        raise ValueError(
            f"Policy returned {len(result)} logprob rows for {len(examples)} examples"
        )
    rows = []
    for row in result:
        if torch.is_tensor(row):
            rows.append(row.to(device))
        else:
            rows.append(torch.as_tensor(row, dtype=torch.float32, device=device))
    return rows


def _maybe_guard_pre_update_logprob_alignment(
    *,
    policy: Any,
    examples: list[ActionTokenExample],
    device: str,
    importance_sampling_level: str,
    training_unit: str,
    loss_aggregation: str,
    microbatch_size: int,
    kl_tolerance: float | None,
    ratio_tolerance: float | None,
) -> dict[str, float | bool] | None:
    """Fail closed before optimizer.step when old/new logprobs are misaligned."""

    if kl_tolerance is None and ratio_tolerance is None:
        return None

    import torch

    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    guard_started = time.perf_counter()
    ratios: list[Any] = []
    approx_kls: list[Any] = []
    microbatches = 0
    streamed_trajectory_span_guard = _should_stream_trajectory_token_spans(
        examples,
        training_unit=training_unit,
        importance_sampling_level=importance_sampling_level,
        loss_aggregation=loss_aggregation,
    )
    guard_examples: list[ActionTokenExample]
    if streamed_trajectory_span_guard:
        guard_examples = [
            span_example
            for example in examples
            for span_example in _iter_trajectory_action_span_examples(example)
        ]
    else:
        guard_examples = examples
    # Validate the exact forward path that will construct the objective. A
    # no_grad teacher-forcing pass can differ for autoregressive policies.
    with torch.enable_grad():
        for microbatch_start in range(0, len(guard_examples), microbatch_size):
            microbatch_examples = guard_examples[
                microbatch_start : microbatch_start + microbatch_size
            ]
            microbatch_logprobs = _policy_action_token_logprobs(
                policy,
                microbatch_examples,
                device=device,
                pad_to_batch_size=microbatch_size,
            )
            microbatches += 1
            for local_index, (example, example_logprobs) in enumerate(
                zip(microbatch_examples, microbatch_logprobs, strict=True)
            ):
                example_index = microbatch_start + local_index
                if example.logprobs is None:
                    raise ValueError("Action-token GRPO requires old rollout logprobs")
                old_logprobs = torch.as_tensor(
                    example.logprobs,
                    dtype=example_logprobs.dtype,
                    device=example_logprobs.device,
                )
                if old_logprobs.shape != example_logprobs.shape:
                    raise ValueError(
                        "New and old action-token logprobs must have the same shape for "
                        f"example {example_index}: new={tuple(example_logprobs.shape)}, "
                        f"old={tuple(old_logprobs.shape)}"
                    )
                approx_kl = old_logprobs - example_logprobs
                token_loss_mask = _example_token_loss_mask_tensor(
                    example,
                    token_count=int(example_logprobs.numel()),
                    device=str(example_logprobs.device),
                    dtype=example_logprobs.dtype,
                )
                valid_token_mask = token_loss_mask.detach() > 0
                if not bool(valid_token_mask.any()):
                    continue
                # The guard only reports/rejects values; retaining its graph
                # across microbatches can consume the full GPU before the real
                # backward pass. Detach immediately after exercising the exact
                # autograd-enabled forward path.
                valid_log_ratio = (example_logprobs - old_logprobs)[
                    valid_token_mask
                ].detach()
                valid_approx_kl = approx_kl[valid_token_mask].detach()
                if importance_sampling_level == "sequence":
                    ratios.append(torch.exp(valid_log_ratio.mean()).reshape(1))
                    approx_kls.append(valid_approx_kl.mean().reshape(1))
                elif importance_sampling_level == "action_chunk":
                    log_ratio = action_chunk_log_ratio(
                        example_logprobs.detach(), old_logprobs.detach(), example
                    )
                    ratios.append(torch.exp(log_ratio).reshape(1))
                    approx_kls.append(-log_ratio.reshape(1))
                else:
                    ratios.append(torch.exp(valid_log_ratio).reshape(-1))
                    approx_kls.append(valid_approx_kl.reshape(-1))

    ratio_values = (
        torch.cat([row.detach().reshape(-1).cpu() for row in ratios])
        if ratios
        else torch.tensor([])
    )
    kl_values = (
        torch.cat([row.detach().reshape(-1).cpu() for row in approx_kls])
        if approx_kls
        else torch.tensor([])
    )
    approx_kl_mean = float(kl_values.mean().item()) if kl_values.numel() else 0.0
    approx_kl_abs_mean = (
        float(kl_values.abs().mean().item()) if kl_values.numel() else 0.0
    )
    ratio_mean = float(ratio_values.mean().item()) if ratio_values.numel() else 1.0
    ratio_min = float(ratio_values.min().item()) if ratio_values.numel() else 1.0
    ratio_max = float(ratio_values.max().item()) if ratio_values.numel() else 1.0
    approx_kl_abs_max = (
        float(kl_values.abs().max().item()) if kl_values.numel() else 0.0
    )
    kl_limit = float(kl_tolerance if kl_tolerance is not None else float("inf"))
    ratio_limit = float(
        ratio_tolerance if ratio_tolerance is not None else float("inf")
    )
    ratio_outside_fraction = 0.0
    if ratio_values.numel() and ratio_tolerance is not None:
        outside = (ratio_values < 1.0 - ratio_limit) | (
            ratio_values > 1.0 + ratio_limit
        )
        ratio_outside_fraction = float(outside.float().mean().item())

    # For long action-token trajectories, a strict min/max ratio gate is too
    # brittle: a handful of numerically different token logprobs can make
    # ratio_min/ratio_max look alarming even when aggregate old/new policy
    # agreement is excellent. Keep extrema as telemetry, but fail closed on
    # aggregate logprob/KL drift, which is the contract that matters before an
    # on-policy update.
    aligned = (
        abs(approx_kl_mean) <= kl_limit
        and approx_kl_abs_mean <= kl_limit
        and abs(ratio_mean - 1.0) <= ratio_limit
    )
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "guard_seconds": time.perf_counter() - guard_started,
        "old_new_logprobs_aligned": bool(aligned),
        "old_new_logprobs_evaluated": True,
        "approx_kl_mean": approx_kl_mean,
        "approx_kl_abs_mean": approx_kl_abs_mean,
        "approx_kl_abs_max": approx_kl_abs_max,
        "ratio_mean": ratio_mean,
        "ratio_min": ratio_min,
        "ratio_max": ratio_max,
        "ratio_outside_tolerance_fraction": ratio_outside_fraction,
        "ratio_extrema_enforced": False,
        "pre_update_logprob_kl_tolerance": kl_limit,
        "pre_update_ratio_tolerance": ratio_limit,
        "logprob_microbatches": float(microbatches),
        "logprob_alignment_guard_streamed_trajectory_spans": bool(
            streamed_trajectory_span_guard
        ),
    }


def _add_alignment_guard_metrics(
    metrics: dict[str, float],
    alignment_guard: Mapping[str, float | bool],
    *,
    prefix: str,
) -> None:
    """Publish pre-update alignment without overwriting objective telemetry."""

    compatibility_keys = {
        "old_new_logprobs_aligned",
        "old_new_logprobs_evaluated",
        "pre_update_logprob_kl_tolerance",
        "pre_update_ratio_tolerance",
        "ratio_extrema_enforced",
        "ratio_outside_tolerance_fraction",
        "logprob_alignment_guard_streamed_trajectory_spans",
    }
    for key, value in alignment_guard.items():
        if not _is_number(value):
            continue
        numeric = float(value)
        metrics[f"{prefix}/alignment_{key}"] = numeric
        if key in compatibility_keys:
            metrics[f"{prefix}/{key}"] = numeric


def _reward_filter_report(
    examples: list[ActionTokenExample],
    *,
    enabled: bool,
    lower: float | None,
    upper: float | None,
) -> dict[str, Any]:
    """Return RLinf-style group reward filtering decisions.

    RLinf filters whole prompt groups by the mean total reward in that group
    before applying the actor loss. ART-Embodied keeps this as a backend knob
    so sparse environment-reward runs can say exactly which groups contributed
    gradient.
    """

    group_to_traj_rewards: dict[Any, dict[Any, float]] = {}
    group_to_traj_chunk_reward_seen: dict[Any, dict[Any, bool]] = {}
    trajectory_reward_sources: dict[tuple[Any, Any], str] = {}
    for position, example in enumerate(examples):
        group_index = example.metadata.get("group_index", 0)
        trajectory_index = example.metadata.get("trajectory_index_in_group", position)
        trajectory_key = (group_index, trajectory_index)
        chunk_reward = _rlinf_masked_chunk_reward_sum_for_example(example)
        if chunk_reward is None:
            if not group_to_traj_chunk_reward_seen.setdefault(group_index, {}).get(
                trajectory_index, False
            ):
                trajectory_reward = example.metadata.get(
                    "trajectory_reward", example.reward
                )
                group_to_traj_rewards.setdefault(group_index, {})[trajectory_index] = (
                    float(trajectory_reward)
                )
                trajectory_reward_sources.setdefault(
                    trajectory_key, "trajectory_reward_fallback"
                )
            continue
        group_to_traj_chunk_reward_seen.setdefault(group_index, {})[
            trajectory_index
        ] = True
        trajectory_reward_sources[trajectory_key] = "masked_chunk_rewards"
        group_to_traj_rewards.setdefault(group_index, {})[trajectory_index] = (
            group_to_traj_rewards.setdefault(group_index, {}).get(trajectory_index, 0.0)
            + float(chunk_reward)
        )

    group_keep: dict[Any, bool] = {}
    group_means: dict[Any, float] = {}
    for group_index, rewards_by_traj in group_to_traj_rewards.items():
        rewards = list(rewards_by_traj.values())
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        group_means[group_index] = mean_reward
        if not enabled:
            group_keep[group_index] = True
        else:
            assert lower is not None and upper is not None
            group_keep[group_index] = lower <= mean_reward <= upper

    example_keep_mask = [
        bool(group_keep.get(example.metadata.get("group_index", 0), True))
        for example in examples
    ]
    kept_groups = sum(1 for keep in group_keep.values() if keep)
    filtered_groups = sum(1 for keep in group_keep.values() if not keep)
    kept_examples = sum(1 for keep in example_keep_mask if keep)
    kept_examples_list = [
        example
        for example, keep in zip(examples, example_keep_mask, strict=True)
        if keep
    ]
    report = {
        "enabled": bool(enabled),
        "lower": lower,
        "upper": upper,
        "groups_total": float(len(group_keep)),
        "groups_kept": float(kept_groups),
        "groups_filtered": float(filtered_groups),
        "examples_before": float(len(examples)),
        "examples_after": float(kept_examples),
        "action_spans_before": float(_action_span_count(examples)),
        "action_spans_after": float(_action_span_count(kept_examples_list)),
        "action_tokens_before": float(_action_logprob_token_count(examples)),
        "action_tokens_after": float(_action_logprob_token_count(kept_examples_list)),
        "group_reward_mean": (
            float(sum(group_means.values()) / len(group_means)) if group_means else 0.0
        ),
        "group_reward_min": float(min(group_means.values())) if group_means else 0.0,
        "group_reward_max": float(max(group_means.values())) if group_means else 0.0,
        "trajectory_reward_source_counts": {
            source: sum(
                1 for value in trajectory_reward_sources.values() if value == source
            )
            for source in sorted(set(trajectory_reward_sources.values()))
        },
    }
    if enabled:
        # The mask is an internal preparation artifact. It can contain tens of
        # thousands of booleans, so never retain it after filtering or copy it
        # into worker commands, JSONL diagnostics, W&B, or Weave.
        report["_example_keep_mask"] = example_keep_mask
    return report


def _rlinf_masked_chunk_reward_sum_for_example(
    example: ActionTokenExample,
) -> float | None:
    """Return RLinf's reward-filter contribution for one action-token example.

    RLinf filters whole prompt groups by the mean of per-rollout rewards after
    applying the embodied loss mask to the chunk reward tensor.  The trajectory
    scalar reward is usually equal for sparse terminal success, but using the
    masked chunk tensor here avoids silent divergence when a final action chunk
    contains post-terminal rewards or any other masked tail slots.
    """

    total = 0.0
    seen = False
    try:
        sources = _rlinf_action_level_sources_for_example(example)
    except ValueError:
        return None
    for source in sources:
        metadata = source["metadata"]
        raw_rewards = metadata.get("primitive_rewards")
        if raw_rewards is None:
            continue
        rewards = _coerce_float_sequence(raw_rewards, field="rlinf_chunk_rewards")
        raw_mask = metadata.get("primitive_loss_mask")
        if raw_mask is None:
            mask = [True for _ in rewards]
        else:
            mask = _coerce_bool_sequence(raw_mask, field="primitive_loss_mask")
            if len(mask) != len(rewards):
                raise ValueError(
                    "primitive_loss_mask and chunk rewards must have the same length for reward filtering: "
                    f"mask={len(mask)}, rewards={len(rewards)}"
                )
        total += sum(
            float(reward)
            for reward, keep in zip(rewards, mask, strict=True)
            if bool(keep)
        )
        seen = True
    return total if seen else None


def _apply_reward_filter_to_examples(
    examples: list[ActionTokenExample],
    *,
    reward_filter_report: dict[str, Any],
    mode: str,
) -> list[ActionTokenExample]:
    keep_mask = reward_filter_report.pop("_example_keep_mask", None)
    if not isinstance(keep_mask, list):
        raise ValueError(
            "enabled reward filtering requires an internal example keep mask"
        )
    if mode == "loss_mask":
        reward_filter_report.update(
            _apply_reward_filter_loss_mask_to_examples(
                examples,
                keep_mask,
            )
        )
        return list(examples)
    if mode == "drop_examples":
        return [
            example for example, keep in zip(examples, keep_mask, strict=True) if keep
        ]
    raise ValueError("reward_filter_mode must be 'drop_examples' or 'loss_mask'")


def _action_span_count(examples: Sequence[ActionTokenExample]) -> int:
    total = 0
    for example in examples:
        metadata = example.metadata if isinstance(example.metadata, dict) else {}
        spans = metadata.get("action_spans")
        if isinstance(spans, list) and spans:
            total += len(spans)
        else:
            total += 1
    return total


def _action_logprob_token_count(examples: Sequence[ActionTokenExample]) -> int:
    """Count objective rows rather than raw token slots."""

    total = 0
    for example in examples:
        metadata = example.metadata if isinstance(example.metadata, dict) else {}
        spans = metadata.get("action_spans")
        if isinstance(spans, list) and spans:
            for span in spans:
                if not isinstance(span, dict):
                    continue
                logprob_start = int(
                    span.get("logprob_start", span.get("token_start", 0))
                )
                logprob_end = int(
                    span.get("logprob_end", span.get("token_end", logprob_start))
                )
                total += max(0, logprob_end - logprob_start)
            continue
        if example.logprobs is not None:
            total += len(example.logprobs)
        else:
            total += len(example.tokens)
    return total


def _rlinf_action_level_sources_for_example(
    example: ActionTokenExample,
) -> list[dict[str, Any]]:
    """Return action-level metadata/logprob spans for RLinf-style advantages.

    ``extract_action_token_examples`` emits one example per policy action and
    stores action metadata directly on the example.  ``extract_trajectory_action_token_examples``
    emits one flattened trajectory example and stores the metadata on each
    action span.  RLinf action-level advantages need the latter shape for
    post-update trajectory-level diagnostics.
    """

    metadata = example.metadata if isinstance(example.metadata, dict) else {}
    logprob_count = (
        len(example.logprobs) if example.logprobs is not None else len(example.tokens)
    )
    action_metadata = metadata.get("action_metadata")
    if isinstance(action_metadata, dict):
        if logprob_count != len(example.tokens):
            raise ValueError(
                "RLinf action-level token advantages currently require one rollout "
                "logprob per action token"
            )
        return [
            {
                "metadata": action_metadata,
                "logprob_start": 0,
                "logprob_end": logprob_count,
            }
        ]

    spans = metadata.get("action_spans")
    if not isinstance(spans, list) or not spans:
        raise ValueError("RLinf action-level advantages require action metadata")

    sources: list[dict[str, Any]] = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        span_metadata = span.get("action_metadata")
        if not isinstance(span_metadata, dict):
            raise ValueError(
                "RLinf action-level trajectory advantages require span action metadata"
            )
        logprob_start = int(span.get("logprob_start", span.get("token_start", 0)) or 0)
        logprob_end = int(
            span.get("logprob_end", span.get("token_end", logprob_start))
            or logprob_start
        )
        if (
            logprob_start < 0
            or logprob_end > logprob_count
            or logprob_end < logprob_start
        ):
            raise ValueError(
                "RLinf action-level trajectory span logprob bounds are invalid: "
                f"start={logprob_start}, end={logprob_end}, logprobs={logprob_count}"
            )
        if logprob_end == logprob_start:
            continue
        sources.append(
            {
                "metadata": span_metadata,
                "logprob_start": logprob_start,
                "logprob_end": logprob_end,
            }
        )
    if not sources:
        raise ValueError(
            "RLinf action-level advantages require non-empty action metadata spans"
        )
    return sources


def _apply_reward_filter_loss_mask_to_examples(
    examples: Sequence[ActionTokenExample],
    keep_mask: Sequence[bool],
) -> dict[str, Any]:
    """Apply RLinf-style reward filtering by zeroing loss masks, not rows.

    RLinf keeps the fixed rollout tensor shape and intersects ``loss_mask`` with
    the group-level reward filter.  Dropping examples changes the actor global
    batch geometry and the ``masked_mean_ratio`` denominator, so parity paths
    must preserve rows and remove gradient contribution through masks only.
    """

    if len(examples) != len(keep_mask):
        raise ValueError(
            "reward filter keep mask length must match examples: "
            f"examples={len(examples)}, mask={len(keep_mask)}"
        )

    examples_masked = 0
    primitive_slots_masked = 0
    token_slots_masked = 0
    for example, keep in zip(examples, keep_mask, strict=True):
        metadata = example.metadata if isinstance(example.metadata, dict) else {}
        metadata["reward_filter_keep"] = bool(keep)
        metadata["reward_filter_mode"] = "loss_mask"
        if keep:
            continue

        examples_masked += 1
        for source in _rlinf_action_level_sources_for_example(example):
            action_metadata = source["metadata"]
            raw_mask = action_metadata.get("primitive_loss_mask")
            if isinstance(raw_mask, list | tuple):
                primitive_count = len(raw_mask)
            else:
                reward_values = action_metadata.get("primitive_rewards")
                primitive_count = (
                    len(reward_values) if isinstance(reward_values, list | tuple) else 0
                )
            if primitive_count <= 0:
                raise ValueError(
                    "reward_filter_mode=loss_mask requires primitive_loss_mask or "
                    "chunk rewards to infer primitive count"
                )
            action_metadata["reward_filter_keep"] = False
            action_metadata["reward_filter_mode"] = "loss_mask"
            primitive_slots_masked += primitive_count
            logprob_span = int(source["logprob_end"]) - int(source["logprob_start"])
            token_slots_masked += max(0, logprob_span)

    return {
        "mode": "loss_mask",
        "examples_masked": int(examples_masked),
        "primitive_slots_masked": int(primitive_slots_masked),
        "token_slots_masked": int(token_slots_masked),
    }


def _attach_rlinf_action_level_token_advantages(
    examples: Sequence[ActionTokenExample],
    *,
    normalize: bool,
    group_std_unbiased: bool,
    score_source: str = "chunk_rewards",
    max_primitive_slots: int | None = None,
    eps: float = 1.0e-6,
) -> None:
    """Attach RLinf-style action-level advantages to action-token examples.

    RLinf's embodied OpenVLA-OFT path uses ``reward_type=action_level`` and
    ``logprob_type=token_level``. The reward model emits one scalar per
    primitive action step, while OpenVLA-OFT scores a fixed action chunk as a
    flat token sequence. RLinf first sums those primitive rewards into one
    trajectory score per rollout, computes GRPO over the same-trial group, then
    broadcasts that trajectory-level advantage back over every valid primitive
    action step and its action-dimension tokens.
    """

    if score_source not in ("chunk_rewards", "trajectory_reward"):
        raise ValueError("score_source must be 'chunk_rewards' or 'trajectory_reward'")

    records_by_group: dict[Any, list[dict[str, Any]]] = {}
    examples_by_group_traj: dict[tuple[Any, Any], list[ActionTokenExample]] = {}
    for example in examples:
        group_index = example.metadata.get("group_index")
        trajectory_index = example.metadata.get("trajectory_index_in_group")
        examples_by_group_traj.setdefault((group_index, trajectory_index), []).append(
            example
        )

    for (
        group_index,
        trajectory_index,
    ), trajectory_examples in examples_by_group_traj.items():
        trajectory_examples.sort(
            key=lambda item: (int(item.action_index), int(item.step))
        )
        trajectory_records: list[dict[str, Any]] = []
        primitive_position = 0
        for example in trajectory_examples:
            for action_source in _rlinf_action_level_sources_for_example(example):
                action_metadata = action_source["metadata"]
                action_logprob_start = int(action_source["logprob_start"])
                action_logprob_end = int(action_source["logprob_end"])
                chunk_reward_field = "primitive_rewards"
                raw_chunk_rewards = action_metadata.get(chunk_reward_field)
                chunk_rewards = _coerce_float_sequence(
                    raw_chunk_rewards,
                    field=chunk_reward_field,
                )
                chunk_mask = _coerce_bool_sequence(
                    action_metadata.get("primitive_loss_mask"),
                    field="primitive_loss_mask",
                )
                if len(chunk_rewards) != len(chunk_mask):
                    raise ValueError(
                        f"{chunk_reward_field} and primitive_loss_mask must "
                        f"have the same length: rewards={len(chunk_rewards)}, mask={len(chunk_mask)}"
                    )
                if not chunk_rewards:
                    raise ValueError(
                        "RLinf action-level advantages require non-empty chunk rewards"
                    )
                logprob_count = action_logprob_end - action_logprob_start
                if logprob_count <= 0:
                    raise ValueError(
                        "RLinf action-level advantages require non-empty action logprobs"
                    )
                if logprob_count % len(chunk_rewards) != 0:
                    raise ValueError(
                        "Action-token count must be divisible by primitive chunk reward count: "
                        f"tokens={logprob_count}, rewards={len(chunk_rewards)}"
                    )
                action_dim = logprob_count // len(chunk_rewards)
                reward_filter_keep = bool(
                    action_metadata.get("reward_filter_keep", True)
                )
                for chunk_index, (reward, keep) in enumerate(
                    zip(chunk_rewards, chunk_mask, strict=True)
                ):
                    pre_filter_keep = bool(keep)
                    loss_keep = pre_filter_keep and reward_filter_keep
                    token_start = action_logprob_start + chunk_index * action_dim
                    trajectory_records.append(
                        {
                            "example": example,
                            "group_index": group_index,
                            "trajectory_index_in_group": trajectory_index,
                            "primitive_position": primitive_position,
                            "chunk_index": chunk_index,
                            "token_start": token_start,
                            "token_end": token_start + action_dim,
                            "masked_reward": float(reward) if pre_filter_keep else 0.0,
                            "pre_filter_loss_mask": pre_filter_keep,
                            "loss_mask": loss_keep,
                            "reward_filter_keep": reward_filter_keep,
                        }
                    )
                    primitive_position += 1

        # RLinf's calculate_scores returns one final cumulative score per rollout,
        # not a different return-to-go per primitive step. Invalid primitive steps
        # are later zeroed by the loss mask. Strict RLinf parity uses the rollout
        # chunk reward tensor for both dense and native sparse-success paths.
        # The trajectory_reward source is retained only for diagnostics or
        # non-parity adapters where a trajectory-level proxy is intentional.
        if score_source == "trajectory_reward":
            trajectory_metadata = (
                trajectory_examples[0].metadata
                if trajectory_examples
                and isinstance(trajectory_examples[0].metadata, dict)
                else {}
            )
            trajectory_score = float(
                trajectory_metadata.get(
                    "trajectory_reward",
                    trajectory_examples[0].reward if trajectory_examples else 0.0,
                )
            )
        else:
            trajectory_score = sum(
                float(record["masked_reward"]) for record in trajectory_records
            )
        for record in trajectory_records:
            record["score"] = (
                float(trajectory_score)
                if bool(record.get("pre_filter_loss_mask", record["loss_mask"]))
                else 0.0
            )
        records_by_group.setdefault(group_index, []).extend(trajectory_records)

    for group_records in records_by_group.values():
        records_by_trajectory: dict[Any, list[dict[str, Any]]] = {}
        for record in group_records:
            records_by_trajectory.setdefault(
                record["trajectory_index_in_group"], []
            ).append(record)
        trajectory_items = sorted(
            records_by_trajectory.items(), key=lambda item: item[0]
        )
        trajectory_observed_slots = {
            trajectory_index: len(trajectory_records)
            for trajectory_index, trajectory_records in trajectory_items
        }
        group_max_primitive_slots = max(trajectory_observed_slots.values(), default=1)
        fixed_max_primitive_slots = _positive_int_or_none(max_primitive_slots)
        if fixed_max_primitive_slots is not None:
            group_max_primitive_slots = max(
                group_max_primitive_slots, fixed_max_primitive_slots
            )
        trajectory_scores = [
            next(
                (
                    float(record["score"])
                    for record in trajectory_records
                    if bool(record.get("pre_filter_loss_mask", record["loss_mask"]))
                ),
                0.0,
            )
            for _trajectory_index, trajectory_records in trajectory_items
        ]
        if len(trajectory_scores) <= 1:
            trajectory_advantages = [0.0 for _ in trajectory_scores]
        else:
            mean_value = sum(trajectory_scores) / len(trajectory_scores)
            denom = (
                (len(trajectory_scores) - 1)
                if group_std_unbiased and len(trajectory_scores) > 1
                else len(trajectory_scores)
            )
            variance = sum(
                (value - mean_value) ** 2 for value in trajectory_scores
            ) / max(1, denom)
            std_value = variance**0.5
            trajectory_advantages = [
                (value - mean_value) / (std_value + float(eps))
                if std_value > 1.0e-12
                else 0.0
                for value in trajectory_scores
            ]
        for (trajectory_index, trajectory_records), advantage in zip(
            trajectory_items, trajectory_advantages, strict=True
        ):
            trajectory_records = sorted(
                trajectory_records,
                key=lambda item: int(item.get("primitive_position", 0)),
            )
            valid_primitive_count = sum(
                1
                for record in trajectory_records
                if bool(record.get("pre_filter_loss_mask", record["loss_mask"]))
            )
            observed_slots = int(
                trajectory_observed_slots.get(trajectory_index, len(trajectory_records))
            )
            extra_slots = max(0, int(group_max_primitive_slots) - observed_slots)
            for record_index, record in enumerate(trajectory_records):
                record["trajectory_valid_primitive_count"] = int(valid_primitive_count)
                record["trajectory_observed_primitive_slots"] = int(observed_slots)
                record["trajectory_primitive_slots"] = int(group_max_primitive_slots)
                # ART stores action chunks as separate examples, unlike RLinf's
                # fixed-horizon tensors.  Allocate the missing invalid tail to
                # the final observed primitive so masked_mean_ratio can still
                # reproduce the fixed-horizon denominator without fabricating
                # fake examples.
                record["possible_extra_primitive_slots"] = (
                    int(extra_slots)
                    if record_index == len(trajectory_records) - 1
                    else 0
                )
                record["advantage"] = (
                    float(advantage)
                    if bool(record.get("pre_filter_loss_mask", record["loss_mask"]))
                    else 0.0
                )

    example_records: dict[int, list[dict[str, Any]]] = {}
    for group_records in records_by_group.values():
        for record in group_records:
            example_records.setdefault(id(record["example"]), []).append(record)

    for example in examples:
        logprob_count = (
            len(example.logprobs)
            if example.logprobs is not None
            else len(example.tokens)
        )
        token_advantages = [0.0 for _ in range(logprob_count)]
        token_loss_mask = [0.0 for _ in range(logprob_count)]
        primitive_scores: list[float] = []
        primitive_advantages: list[float] = []
        primitive_loss_mask: list[bool] = []
        trajectory_valid_primitive_count = 0
        trajectory_primitive_slots = 0
        trajectory_observed_primitive_slots = 0
        possible_extra_primitive_slots = 0
        action_dim = 0
        for record in sorted(
            example_records.get(id(example), []),
            key=lambda item: int(item.get("primitive_position", item["chunk_index"])),
        ):
            advantage = float(record.get("advantage", 0.0))
            keep = bool(record["loss_mask"])
            start = int(record["token_start"])
            end = int(record["token_end"])
            action_dim = max(action_dim, end - start)
            trajectory_valid_primitive_count = int(
                record.get("trajectory_valid_primitive_count")
                or trajectory_valid_primitive_count
            )
            trajectory_primitive_slots = int(
                record.get("trajectory_primitive_slots") or trajectory_primitive_slots
            )
            trajectory_observed_primitive_slots = int(
                record.get("trajectory_observed_primitive_slots")
                or trajectory_observed_primitive_slots
            )
            possible_extra_primitive_slots += int(
                record.get("possible_extra_primitive_slots") or 0
            )
            primitive_scores.append(float(record.get("score", 0.0)))
            primitive_advantages.append(advantage)
            primitive_loss_mask.append(keep)
            for token_index in range(start, end):
                token_advantages[token_index] = advantage if keep else 0.0
                token_loss_mask[token_index] = 1.0 if keep else 0.0
        example.metadata["action_advantage_mode"] = "rlinf_action_level_cumulative"
        example.metadata["token_advantages"] = token_advantages
        example.metadata["token_loss_mask"] = token_loss_mask
        example.metadata["rlinf_action_level_primitive_scores"] = primitive_scores
        example.metadata["rlinf_action_level_trajectory_score"] = (
            float(primitive_scores[0]) if primitive_scores else 0.0
        )
        example.metadata["rlinf_action_level_primitive_advantages"] = (
            primitive_advantages
        )
        example.metadata["rlinf_action_level_primitive_loss_mask"] = primitive_loss_mask
        example.metadata["rlinf_action_level_trajectory_loss_mask_sum"] = int(
            trajectory_valid_primitive_count
        )
        example.metadata["rlinf_action_level_trajectory_primitive_slots"] = int(
            trajectory_primitive_slots
        )
        example.metadata["rlinf_action_level_observed_primitive_slots"] = int(
            trajectory_observed_primitive_slots
        )
        example.metadata["rlinf_action_level_possible_extra_primitive_slots"] = int(
            possible_extra_primitive_slots
        )
        example.metadata["rlinf_action_level_action_dim"] = int(action_dim)
        example.metadata["rlinf_action_level_extra_global_normalization"] = False
        valid_advantages = [
            value
            for value, keep in zip(
                primitive_advantages, primitive_loss_mask, strict=True
            )
            if keep
        ]
        example.metadata["group_advantage"] = _mean(valid_advantages)


def _mask_rlinf_action_level_zero_variance_groups(
    examples: Sequence[ActionTokenExample],
    *,
    eps: float = 1.0e-8,
) -> dict[str, Any]:
    """Remove zero-relative-signal groups from the action-token objective.

    RLinf's second masked normalization is meaningful when the actor batch is
    already made of examples with useful reward variation.  With native sparse
    success rewards, all-fail groups have no GRPO signal; if we include them in
    a global normalization, their originally-zero advantages are shifted away
    from zero and start driving policy updates.  That violates the first
    principle of group-relative policy gradients: a group with identical
    returns should contribute no gradient.
    """

    trajectory_scores_by_group: dict[Any, dict[Any, float]] = {}
    example_group_keys: dict[int, Any] = {}
    for example in examples:
        metadata = example.metadata if isinstance(example.metadata, dict) else {}
        group_index = metadata.get("group_index")
        trajectory_index = metadata.get("trajectory_index_in_group")
        example_group_keys[id(example)] = group_index
        primitive_scores = metadata.get("rlinf_action_level_primitive_scores")
        primitive_loss_mask = metadata.get("rlinf_action_level_primitive_loss_mask")
        if not isinstance(primitive_scores, list | tuple):
            continue
        if primitive_loss_mask is None:
            primitive_loss_mask = [True for _ in primitive_scores]
        if not isinstance(primitive_loss_mask, list | tuple):
            raise ValueError(
                "rlinf_action_level_primitive_loss_mask must be a list or tuple"
            )
        if len(primitive_scores) != len(primitive_loss_mask):
            raise ValueError(
                "rlinf_action_level_primitive_scores and "
                "rlinf_action_level_primitive_loss_mask must have the same length"
            )
        valid_scores = [
            float(score)
            for score, keep in zip(primitive_scores, primitive_loss_mask, strict=True)
            if bool(keep)
        ]
        explicit_trajectory_score = metadata.get("rlinf_action_level_trajectory_score")
        trajectory_score = (
            float(explicit_trajectory_score)
            if explicit_trajectory_score is not None
            else (valid_scores[0] if valid_scores else 0.0)
        )
        trajectory_scores_by_group.setdefault(group_index, {})[trajectory_index] = (
            float(trajectory_score)
        )

    zero_variance_groups: set[Any] = set()
    for group_index, scores_by_trajectory in trajectory_scores_by_group.items():
        scores = list(scores_by_trajectory.values())
        if len(scores) <= 1 or (max(scores) - min(scores)) <= float(eps):
            zero_variance_groups.add(group_index)

    examples_masked = 0
    tokens_masked = 0
    primitives_masked = 0
    for example in examples:
        metadata = example.metadata if isinstance(example.metadata, dict) else {}
        group_index = example_group_keys.get(id(example), metadata.get("group_index"))
        masked = group_index in zero_variance_groups
        token_advantages = metadata.get("token_advantages")
        token_loss_mask = metadata.get("token_loss_mask")
        primitive_advantages = metadata.get("rlinf_action_level_primitive_advantages")
        primitive_loss_mask = metadata.get("rlinf_action_level_primitive_loss_mask")
        if masked:
            examples_masked += 1
            if isinstance(token_advantages, list | tuple):
                if token_loss_mask is None:
                    token_loss_mask = [True for _ in token_advantages]
                if not isinstance(token_loss_mask, list | tuple):
                    raise ValueError("token_loss_mask must be a list or tuple")
                if len(token_advantages) != len(token_loss_mask):
                    raise ValueError(
                        "token_advantages and token_loss_mask must have the same length"
                    )
                tokens_masked += sum(1 for keep in token_loss_mask if bool(keep))
                metadata["token_advantages"] = [0.0 for _ in token_advantages]
                metadata["token_loss_mask"] = [False for _ in token_loss_mask]
            if isinstance(primitive_advantages, list | tuple):
                if primitive_loss_mask is None:
                    primitive_loss_mask = [True for _ in primitive_advantages]
                if not isinstance(primitive_loss_mask, list | tuple):
                    raise ValueError(
                        "rlinf_action_level_primitive_loss_mask must be a list or tuple"
                    )
                if len(primitive_advantages) != len(primitive_loss_mask):
                    raise ValueError(
                        "rlinf_action_level_primitive_advantages and "
                        "rlinf_action_level_primitive_loss_mask must have the same length"
                    )
                primitives_masked += sum(
                    1 for keep in primitive_loss_mask if bool(keep)
                )
                metadata["rlinf_action_level_primitive_advantages"] = [
                    0.0 for _ in primitive_advantages
                ]
                metadata["rlinf_action_level_primitive_loss_mask"] = [
                    False for _ in primitive_loss_mask
                ]
            metadata["group_advantage"] = 0.0
        metadata["rlinf_action_level_mask_zero_variance_groups"] = True
        metadata["rlinf_action_level_zero_variance_group_masked"] = bool(masked)

    return {
        "applied": True,
        "groups_total": int(len(trajectory_scores_by_group)),
        "groups_masked": int(len(zero_variance_groups)),
        "examples_masked": int(examples_masked),
        "tokens_masked": int(tokens_masked),
        "primitives_masked": int(primitives_masked),
    }


def _rlinf_action_level_global_advantage_stats(
    examples: Sequence[ActionTokenExample],
    *,
    eps: float = 1.0e-5,
) -> dict[str, Any]:
    """Return RLinf-style post-GRPO global masked advantage stats.

    RLinf's FSDP actor first computes group-relative GRPO advantages and then,
    when ``algorithm.normalize_advantages`` is true, applies
    ``masked_normalization`` over the full actor batch.  ART stores VLA action
    chunks as examples, so this helper computes the equivalent global mean and
    ``sqrt(var) + eps`` scale over valid action-token entries.
    """

    values: list[float] = []
    for example in examples:
        metadata = example.metadata if isinstance(example.metadata, dict) else {}
        token_advantages = metadata.get("token_advantages")
        token_loss_mask = metadata.get("token_loss_mask")
        if not isinstance(token_advantages, list | tuple):
            continue
        if token_loss_mask is None:
            token_loss_mask = [True for _ in token_advantages]
        if not isinstance(token_loss_mask, list | tuple):
            raise ValueError("token_loss_mask must be a list or tuple")
        if len(token_advantages) != len(token_loss_mask):
            raise ValueError(
                "token_advantages and token_loss_mask must have the same length"
            )
        values.extend(
            float(advantage)
            for advantage, keep in zip(token_advantages, token_loss_mask, strict=True)
            if bool(keep)
        )
    if not values:
        return {
            "applied": False,
            "count": 0,
            "mean": 0.0,
            "variance": 0.0,
            "scale": 1.0,
            "eps": float(eps),
        }
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    scale = math.sqrt(max(0.0, variance)) + float(eps)
    return {
        "applied": True,
        "count": int(len(values)),
        "mean": float(mean),
        "variance": float(variance),
        "scale": float(scale),
        "eps": float(eps),
    }


def _apply_rlinf_action_level_extra_global_normalization(
    examples: Sequence[ActionTokenExample],
    *,
    mean: float | None = None,
    scale: float | None = None,
    eps: float = 1.0e-5,
) -> dict[str, Any]:
    """Apply RLinf's second masked advantage normalization to token advantages."""

    stats = _rlinf_action_level_global_advantage_stats(examples, eps=eps)
    if mean is None and scale is None:
        mean = float(stats["mean"])
        scale = float(stats["scale"])
    elif mean is None or scale is None:
        raise ValueError("mean and scale must be provided together")
    else:
        mean = float(mean)
        scale = float(scale)
        if scale <= 0.0:
            raise ValueError("scale must be positive")
        stats = {
            **stats,
            "external_stats": True,
            "mean": mean,
            "scale": scale,
        }
    for example in examples:
        metadata = example.metadata if isinstance(example.metadata, dict) else {}
        token_advantages = metadata.get("token_advantages")
        token_loss_mask = metadata.get("token_loss_mask")
        if not isinstance(token_advantages, list | tuple):
            continue
        if token_loss_mask is None:
            token_loss_mask = [True for _ in token_advantages]
        if not isinstance(token_loss_mask, list | tuple):
            raise ValueError("token_loss_mask must be a list or tuple")
        if len(token_advantages) != len(token_loss_mask):
            raise ValueError(
                "token_advantages and token_loss_mask must have the same length"
            )
        normalized = [
            ((float(advantage) - mean) / scale) if bool(keep) else 0.0
            for advantage, keep in zip(token_advantages, token_loss_mask, strict=True)
        ]
        metadata["token_advantages"] = normalized
        metadata["rlinf_action_level_extra_global_normalization"] = True
        metadata["rlinf_action_level_global_advantage_mean"] = float(mean)
        metadata["rlinf_action_level_global_advantage_scale"] = float(scale)
        valid_advantages = [
            advantage
            for advantage, keep in zip(normalized, token_loss_mask, strict=True)
            if bool(keep)
        ]
        metadata["group_advantage"] = _mean(valid_advantages)
    return stats


def _new_objective_direction_stats() -> dict[str, float]:
    return {
        "positive_count": 0.0,
        "positive_logprob_delta_sum": 0.0,
        "positive_ratio_sum": 0.0,
        "positive_logprob_deltas": [],
        "positive_ratios": [],
        "negative_count": 0.0,
        "negative_logprob_delta_sum": 0.0,
        "negative_ratio_sum": 0.0,
        "negative_logprob_deltas": [],
        "negative_ratios": [],
        "zero_count": 0.0,
        "nonzero_count": 0.0,
        "signed_product_sum": 0.0,
        "signed_product_positive_count": 0.0,
        "signed_product_negative_count": 0.0,
        "signed_products": [],
    }


def _example_policy_step(example: ActionTokenExample) -> int | None:
    metadata = example.metadata if isinstance(example.metadata, dict) else {}
    action_metadata = metadata.get("action_metadata")
    if isinstance(action_metadata, dict) and "policy_step" in action_metadata:
        try:
            return int(action_metadata["policy_step"])
        except Exception:
            return None
    return None


def _example_reward_bucket(example: ActionTokenExample) -> str | None:
    metadata = example.metadata if isinstance(example.metadata, dict) else {}
    raw_reward = metadata.get("trajectory_reward", example.reward)
    try:
        reward = float(raw_reward)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(reward):
        return None
    return f"reward_{_safe_metric_bucket_value(reward)}"


def _example_group_bucket(example: ActionTokenExample) -> str | None:
    metadata = example.metadata if isinstance(example.metadata, dict) else {}
    raw_group_index = metadata.get("group_index")
    if raw_group_index is None:
        return None
    try:
        group_index = int(raw_group_index)
    except (TypeError, ValueError):
        return f"group_{_safe_metric_bucket_label(str(raw_group_index))}"
    return f"group_{group_index:04d}"


def _safe_metric_bucket_value(value: float) -> str:
    if abs(value - round(value)) < 1.0e-8:
        return str(int(round(value))).replace("-", "neg_")
    return _safe_metric_bucket_label(f"{value:.6g}")


def _safe_metric_bucket_label(value: str) -> str:
    chars = []
    for char in str(value):
        if char.isalnum():
            chars.append(char)
        elif char == "-":
            chars.append("neg")
        else:
            chars.append("_")
    label = "".join(chars).strip("_")
    return label or "unknown"


def _example_action_dim(example: ActionTokenExample) -> int | None:
    metadata = example.metadata if isinstance(example.metadata, dict) else {}
    action_dim = _positive_int_or_none(metadata.get("rlinf_action_level_action_dim"))
    if action_dim is not None:
        return int(action_dim)
    action_metadata = metadata.get("action_metadata")
    if isinstance(action_metadata, dict):
        mask = action_metadata.get("primitive_loss_mask")
        token_count = (
            len(example.logprobs)
            if example.logprobs is not None
            else len(example.tokens)
        )
        if isinstance(mask, list | tuple) and mask:
            if token_count % len(mask) == 0:
                return int(token_count // len(mask))
    return None


def _normalize_bucket_loss_weights(
    mapping: Mapping[Any, Any] | None,
    *,
    key_prefix: str,
    label: str,
) -> dict[int, float]:
    """Normalize optional YAML bucket weights to integer-indexed weights."""

    if mapping is None:
        return {}
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{label} must be a mapping when provided")
    weights: dict[int, float] = {}
    for raw_key, raw_value in mapping.items():
        index = _bucket_index_from_key(raw_key, key_prefix=key_prefix, label=label)
        try:
            weight = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}[{raw_key!r}] must be numeric") from exc
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"{label}[{raw_key!r}] must be finite and non-negative")
        weights[index] = weight
    return weights


def _bucket_index_from_key(raw_key: Any, *, key_prefix: str, label: str) -> int:
    if isinstance(raw_key, bool):
        raise ValueError(
            f"{label} keys must be integer indexes or {key_prefix}XX strings"
        )
    if isinstance(raw_key, int):
        if raw_key < 0:
            raise ValueError(f"{label} keys must be non-negative")
        return int(raw_key)
    text = str(raw_key)
    if text.startswith(key_prefix):
        text = text[len(key_prefix) :]
    try:
        index = int(text)
    except ValueError as exc:
        raise ValueError(
            f"{label} keys must be integer indexes or {key_prefix}XX strings"
        ) from exc
    if index < 0:
        raise ValueError(f"{label} keys must be non-negative")
    return int(index)


def _example_bucket_loss_weight_tensor(
    example: ActionTokenExample,
    *,
    token_count: int,
    device: str,
    dtype: Any,
    policy_step_loss_weights: Mapping[int, float],
    action_dim_loss_weights: Mapping[int, float],
    loss_aggregation: str = "token_mean",
) -> Any:
    """Return per-token diagnostic bucket weights.

    The default is all ones.  Configured policy-step weights scale the whole
    action chunk; action-dimension weights scale token positions modulo the
    inferred action dimension.  Denominators are intentionally not renormalized,
    so a zero-weighted bucket truly reduces effective update mass.
    """

    import torch

    weights = torch.ones(int(token_count), dtype=dtype, device=device)
    if loss_aggregation == "trajectory_mean":
        action_count = _positive_int_or_none(
            example.metadata.get("trajectory_action_count")
        )
        if action_count is not None:
            weights = weights / float(action_count)
    policy_step = _example_policy_step(example)
    if policy_step is not None and int(policy_step) in policy_step_loss_weights:
        weights = weights * float(policy_step_loss_weights[int(policy_step)])
    action_dim = _example_action_dim(example)
    if action_dim is not None and action_dim > 0 and action_dim_loss_weights:
        token_positions = torch.arange(int(token_count), device=device)
        dim_ids = token_positions % int(action_dim)
        for dim_index, dim_weight in action_dim_loss_weights.items():
            weights = torch.where(
                dim_ids == int(dim_index),
                weights * float(dim_weight),
                weights,
            )
    return weights


def _loss_denominator_example_count(
    examples: Sequence[ActionTokenExample], *, loss_aggregation: str
) -> int:
    """Return the number of equal-weight units in an example-mean objective."""

    if (
        loss_aggregation in ("trajectory_mean", "seq_mean_token_sum")
        and examples
        and all(
            _positive_int_or_none(example.metadata.get("trajectory_action_count"))
            is not None
            for example in examples
        )
    ):
        return len({int(example.trajectory_index) for example in examples})
    return len(examples)


def _bucket_loss_weight_metrics(
    examples: Sequence[ActionTokenExample],
    *,
    policy_step_loss_weights: Mapping[int, float],
    action_dim_loss_weights: Mapping[int, float],
    prefix: str,
) -> dict[str, float]:
    import torch

    metrics: dict[str, float] = {
        f"{prefix}/policy_step_loss_weights_configured": float(
            len(policy_step_loss_weights)
        ),
        f"{prefix}/action_dim_loss_weights_configured": float(
            len(action_dim_loss_weights)
        ),
    }
    if not policy_step_loss_weights and not action_dim_loss_weights:
        metrics[f"{prefix}/bucket_loss_weights_enabled"] = 0.0
        return metrics

    token_total = 0.0
    weighted_token_total = 0.0
    zero_weight_tokens = 0.0
    for example in examples:
        token_count = (
            len(example.logprobs)
            if example.logprobs is not None
            else len(example.tokens)
        )
        weights = _example_bucket_loss_weight_tensor(
            example,
            token_count=int(token_count),
            device="cpu",
            dtype=torch.float32,
            policy_step_loss_weights=policy_step_loss_weights,
            action_dim_loss_weights=action_dim_loss_weights,
        )
        token_total += float(token_count)
        weighted_token_total += float(weights.sum().item())
        zero_weight_tokens += float((weights <= 0.0).float().sum().item())
    metrics.update(
        {
            f"{prefix}/bucket_loss_weights_enabled": 1.0,
            f"{prefix}/bucket_loss_weight_tokens": token_total,
            f"{prefix}/bucket_loss_weighted_tokens": weighted_token_total,
            f"{prefix}/bucket_loss_zero_weight_tokens": zero_weight_tokens,
            f"{prefix}/bucket_loss_weighted_token_fraction": (
                weighted_token_total / token_total if token_total > 0.0 else 0.0
            ),
            f"{prefix}/bucket_loss_zero_weight_token_fraction": (
                zero_weight_tokens / token_total if token_total > 0.0 else 0.0
            ),
        }
    )
    return metrics


def _accumulate_objective_direction_bucket_stats(
    bucket_stats: dict[str, dict[str, float]],
    *,
    key: str,
    logprob_delta: Any,
    advantage: Any,
    weight: Any | None = None,
) -> None:
    stats = bucket_stats.setdefault(str(key), _new_objective_direction_stats())
    _accumulate_objective_direction_stats(
        stats,
        logprob_delta=logprob_delta,
        advantage=advantage,
        weight=weight,
    )


def _accumulate_objective_direction_stats(
    stats: dict[str, float],
    *,
    logprob_delta: Any,
    advantage: Any,
    weight: Any | None = None,
) -> None:
    """Track whether the current policy moved in the advantage direction.

    For positive advantages, logprob should increase.  For negative advantages,
    logprob should decrease.  This diagnostic is intentionally independent of
    the loss implementation so post-update probes can expose sign errors,
    overshoot, or objectives that move the policy without improving behavior.
    """

    import torch

    deltas = torch.as_tensor(logprob_delta).detach().float().reshape(-1).cpu()
    advantages = torch.as_tensor(advantage).detach().float().reshape(-1).cpu()
    if deltas.numel() == 0:
        return
    if advantages.numel() == 1 and deltas.numel() > 1:
        advantages = advantages.expand_as(deltas)
    if advantages.numel() != deltas.numel():
        raise ValueError(
            "objective-direction diagnostic requires advantage and logprob_delta "
            f"to align: advantage={advantages.numel()} delta={deltas.numel()}"
        )
    if weight is None:
        weights = torch.ones_like(deltas)
    else:
        weights = torch.as_tensor(weight).detach().float().reshape(-1).cpu()
        if weights.numel() == 1 and deltas.numel() > 1:
            weights = weights.expand_as(deltas)
        if weights.numel() != deltas.numel():
            raise ValueError(
                "objective-direction diagnostic requires weight and logprob_delta "
                f"to align: weight={weights.numel()} delta={deltas.numel()}"
            )

    pos_mask = advantages > 1.0e-8
    neg_mask = advantages < -1.0e-8
    zero_mask = ~(pos_mask | neg_mask)

    def add_bucket(name: str, mask: Any) -> None:
        if not bool(mask.any().item()):
            return
        bucket_deltas = deltas[mask]
        bucket_weights = weights[mask]
        bucket_weight_sum = float(bucket_weights.sum().item())
        if bucket_weight_sum <= 0.0:
            return
        stats[f"{name}_count"] += bucket_weight_sum
        stats[f"{name}_logprob_delta_sum"] += float(
            (bucket_deltas * bucket_weights).sum().item()
        )
        bucket_ratios = torch.exp(bucket_deltas)
        stats[f"{name}_ratio_sum"] += float(
            (bucket_ratios * bucket_weights).sum().item()
        )
        stats.setdefault(f"{name}_logprob_deltas", []).extend(
            float(value) for value in bucket_deltas.tolist()
        )
        stats.setdefault(f"{name}_ratios", []).extend(
            float(value) for value in bucket_ratios.tolist()
        )

    add_bucket("positive", pos_mask)
    add_bucket("negative", neg_mask)
    stats["zero_count"] += float(weights[zero_mask].sum().item())

    nonzero_mask = pos_mask | neg_mask
    if bool(nonzero_mask.any().item()):
        products = advantages[nonzero_mask] * deltas[nonzero_mask]
        product_weights = weights[nonzero_mask]
        product_weight_sum = float(product_weights.sum().item())
        if product_weight_sum <= 0.0:
            return
        weighted_products = products * product_weights
        stats["nonzero_count"] += product_weight_sum
        stats["signed_product_sum"] += float(weighted_products.sum().item())
        stats["signed_product_positive_count"] += float(
            product_weights[products > 0.0].sum().item()
        )
        stats["signed_product_negative_count"] += float(
            product_weights[products < 0.0].sum().item()
        )
        stats.setdefault("signed_products", []).extend(
            float(value) for value in products.tolist()
        )


def _quantile_metrics_from_values(
    values: Any,
    *,
    prefix: str,
    names: tuple[str, ...] = ("p01", "p05", "p50", "p95", "p99"),
    quantiles: tuple[float, ...] = (0.01, 0.05, 0.50, 0.95, 0.99),
) -> dict[str, float]:
    import torch

    tensor = torch.as_tensor(values).detach().float().reshape(-1).cpu()
    if tensor.numel() == 0:
        return {f"{prefix}_{name}": 0.0 for name in names}
    if tensor.numel() == 1:
        value = float(tensor.item())
        return {f"{prefix}_{name}": value for name in names}
    qs = torch.tensor(quantiles, dtype=torch.float32)
    quantile_values = torch.quantile(tensor, qs)
    return {
        f"{prefix}_{name}": float(value.item())
        for name, value in zip(names, quantile_values, strict=True)
    }


def _objective_direction_metrics(
    stats: dict[str, float] | None,
    *,
    prefix: str,
) -> dict[str, float]:
    stats = stats or _new_objective_direction_stats()

    def safe_mean(sum_key: str, count_key: str, default: float = 0.0) -> float:
        count = float(stats.get(count_key) or 0.0)
        if count <= 0.0:
            return default
        return float(stats.get(sum_key) or 0.0) / count

    nonzero_count = float(stats.get("nonzero_count") or 0.0)
    positive_logprob_delta_mean = safe_mean(
        "positive_logprob_delta_sum",
        "positive_count",
    )
    negative_logprob_delta_mean = safe_mean(
        "negative_logprob_delta_sum",
        "negative_count",
    )
    positive_ratio_mean = safe_mean(
        "positive_ratio_sum",
        "positive_count",
        default=1.0,
    )
    negative_ratio_mean = safe_mean(
        "negative_ratio_sum",
        "negative_count",
        default=1.0,
    )
    metrics = {
        f"{prefix}/objective_direction_positive_tokens": float(
            stats.get("positive_count") or 0.0
        ),
        f"{prefix}/objective_direction_negative_tokens": float(
            stats.get("negative_count") or 0.0
        ),
        f"{prefix}/objective_direction_zero_tokens": float(
            stats.get("zero_count") or 0.0
        ),
        f"{prefix}/objective_direction_nonzero_tokens": nonzero_count,
        f"{prefix}/objective_direction_positive_logprob_delta_mean": positive_logprob_delta_mean,
        f"{prefix}/objective_direction_negative_logprob_delta_mean": negative_logprob_delta_mean,
        f"{prefix}/objective_direction_positive_minus_negative_logprob_delta_mean": (
            positive_logprob_delta_mean - negative_logprob_delta_mean
        ),
        f"{prefix}/objective_direction_positive_ratio_mean": positive_ratio_mean,
        f"{prefix}/objective_direction_negative_ratio_mean": negative_ratio_mean,
        f"{prefix}/objective_direction_positive_minus_negative_ratio_mean": (
            positive_ratio_mean - negative_ratio_mean
        ),
        f"{prefix}/objective_direction_advantage_logprob_delta_product_mean": safe_mean(
            "signed_product_sum",
            "nonzero_count",
        ),
        f"{prefix}/objective_direction_sign_agreement_fraction": (
            float(stats.get("signed_product_positive_count") or 0.0) / nonzero_count
            if nonzero_count > 0.0
            else 0.0
        ),
        f"{prefix}/objective_direction_sign_disagreement_fraction": (
            float(stats.get("signed_product_negative_count") or 0.0) / nonzero_count
            if nonzero_count > 0.0
            else 0.0
        ),
    }
    metrics.update(
        _quantile_metrics_from_values(
            stats.get("positive_logprob_deltas") or [],
            prefix=f"{prefix}/objective_direction_positive_logprob_delta",
        )
    )
    metrics.update(
        _quantile_metrics_from_values(
            stats.get("negative_logprob_deltas") or [],
            prefix=f"{prefix}/objective_direction_negative_logprob_delta",
        )
    )
    metrics.update(
        _quantile_metrics_from_values(
            stats.get("signed_products") or [],
            prefix=f"{prefix}/objective_direction_advantage_logprob_delta_product",
        )
    )
    return metrics


def _objective_direction_bucket_metrics(
    bucket_stats: dict[str, dict[str, float]],
    *,
    prefix: str,
    max_buckets: int = 64,
) -> dict[str, float]:
    """Return compact per-bucket objective-direction metrics.

    Full quantiles for every policy step and action dimension produce noisy
    logs.  Keep only the fields needed to compare where public RLinf and an ART
    candidate move on the same retained shard.
    """

    selected_suffixes = (
        "objective_direction_positive_tokens",
        "objective_direction_negative_tokens",
        "objective_direction_positive_ratio_mean",
        "objective_direction_negative_ratio_mean",
        "objective_direction_positive_minus_negative_ratio_mean",
        "objective_direction_advantage_logprob_delta_product_mean",
        "objective_direction_sign_agreement_fraction",
    )
    metrics: dict[str, float] = {}
    for index, key in enumerate(sorted(bucket_stats)):
        if index >= int(max_buckets):
            metrics[f"{prefix}/truncated_buckets"] = float(
                max(0, len(bucket_stats) - index)
            )
            break
        raw_metrics = _objective_direction_metrics(
            bucket_stats[key],
            prefix=f"{prefix}/{key}",
        )
        for metric_key, value in raw_metrics.items():
            if any(metric_key.endswith(suffix) for suffix in selected_suffixes):
                metrics[metric_key] = float(value)
    metrics[f"{prefix}/bucket_count"] = float(len(bucket_stats))
    return metrics


def _grpo_metrics(
    *,
    groups: list[EmbodiedTrajectoryGroup],
    examples: list[ActionTokenExample],
    loss: float,
    advantages: list[float],
    ratios: list[Any],
    approx_kls: list[Any],
    reference_l2_penalties: list[Any],
    clip_hits: list[Any],
    token_counts: list[float],
    normalize_advantages: bool,
    advantage_normalization_scope: str,
    advantage_std_unbiased: bool,
    reward_filter_report: dict[str, Any] | None,
    importance_sampling_level: str,
    training_unit: str,
    loss_aggregation: str,
    kl_coef: float,
    clip_epsilon: float,
    clip_epsilon_low: float,
    clip_epsilon_high: float,
    reference_logprob_l2_coef: float,
    objective_direction_stats: dict[str, float] | None = None,
) -> dict[str, float]:
    import torch

    ratio_values = (
        torch.cat([row.reshape(-1).cpu() for row in ratios])
        if ratios
        else torch.tensor([])
    )
    kl_values = (
        torch.cat([row.reshape(-1).cpu() for row in approx_kls])
        if approx_kls
        else torch.tensor([])
    )
    clip_values = (
        torch.cat([row.reshape(-1).cpu() for row in clip_hits])
        if clip_hits
        else torch.tensor([])
    )
    reference_l2_values = (
        torch.cat([row.reshape(-1).cpu() for row in reference_l2_penalties])
        if reference_l2_penalties
        else torch.tensor([])
    )
    extra_global_norm_examples = [
        example
        for example in examples
        if bool(
            (example.metadata if isinstance(example.metadata, dict) else {}).get(
                "rlinf_action_level_extra_global_normalization",
                False,
            )
        )
    ]
    extra_global_norm_metadata = (
        extra_global_norm_examples[0].metadata
        if extra_global_norm_examples
        and isinstance(extra_global_norm_examples[0].metadata, dict)
        else {}
    )
    zero_variance_mask_examples = [
        example
        for example in examples
        if bool(
            (example.metadata if isinstance(example.metadata, dict) else {}).get(
                "rlinf_action_level_mask_zero_variance_groups",
                False,
            )
        )
    ]
    zero_variance_masked_examples = [
        example
        for example in zero_variance_mask_examples
        if bool(
            (example.metadata if isinstance(example.metadata, dict) else {}).get(
                "rlinf_action_level_zero_variance_group_masked",
                False,
            )
        )
    ]
    metrics = {
        "embodied_action_token_grpo/groups": float(len(groups)),
        "embodied_action_token_grpo/examples": float(len(examples)),
        "embodied_action_token_grpo/tokens_total": float(sum(token_counts)),
        "embodied_action_token_grpo/tokens_mean": _mean(token_counts),
        "embodied_action_token_grpo/reward_mean": _mean(
            [example.reward for example in examples]
        ),
        "embodied_action_token_grpo/advantage_mean": _mean(advantages),
        "embodied_action_token_grpo/advantage_min": min(advantages)
        if advantages
        else 0.0,
        "embodied_action_token_grpo/advantage_max": max(advantages)
        if advantages
        else 0.0,
        "embodied_action_token_grpo/normalize_advantages": float(normalize_advantages),
        "embodied_action_token_grpo/advantage_normalization_scope_group": float(
            advantage_normalization_scope == "group"
        ),
        "embodied_action_token_grpo/advantage_std_unbiased": float(
            advantage_std_unbiased
        ),
        "embodied_action_token_grpo/rlinf_action_level_extra_global_normalization": float(
            bool(extra_global_norm_examples)
        ),
        "embodied_action_token_grpo/rlinf_action_level_extra_global_normalization_examples": float(
            len(extra_global_norm_examples)
        ),
        "embodied_action_token_grpo/rlinf_action_level_global_advantage_mean": float(
            extra_global_norm_metadata.get("rlinf_action_level_global_advantage_mean")
            or 0.0
        ),
        "embodied_action_token_grpo/rlinf_action_level_global_advantage_scale": float(
            extra_global_norm_metadata.get("rlinf_action_level_global_advantage_scale")
            or 0.0
        ),
        "embodied_action_token_grpo/rlinf_action_level_mask_zero_variance_groups": float(
            bool(zero_variance_mask_examples)
        ),
        "embodied_action_token_grpo/rlinf_action_level_zero_variance_group_mask_examples": float(
            len(zero_variance_mask_examples)
        ),
        "embodied_action_token_grpo/rlinf_action_level_zero_variance_group_masked_examples": float(
            len(zero_variance_masked_examples)
        ),
        "embodied_action_token_grpo/sequence_importance_sampling": float(
            importance_sampling_level == "sequence"
        ),
        "embodied_action_token_grpo/action_chunk_importance_sampling": float(
            importance_sampling_level == "action_chunk"
        ),
        "embodied_action_token_grpo/trajectory_training_unit": float(
            training_unit == "trajectory"
        ),
        "embodied_action_token_grpo/loss_aggregation_token_mean": float(
            loss_aggregation == "token_mean"
        ),
        "embodied_action_token_grpo/loss_aggregation_seq_mean_token_sum": float(
            loss_aggregation == "seq_mean_token_sum"
        ),
        "embodied_action_token_grpo/loss_aggregation_task_balanced_trajectory_mean": float(
            loss_aggregation == "task_balanced_trajectory_mean"
        ),
        "embodied_action_token_grpo/loss_aggregation_rlinf_token_mean": float(
            loss_aggregation == "rlinf_token_mean"
        ),
        "embodied_action_token_grpo/loss_aggregation_rlinf_chunk_mean": float(
            loss_aggregation == "rlinf_chunk_mean"
        ),
        "embodied_action_token_grpo/loss_aggregation_rlinf_masked_mean_ratio": float(
            loss_aggregation == "rlinf_masked_mean_ratio"
        ),
        "embodied_action_token_grpo/loss_aggregation_advantage_sign_balanced_token_mean": float(
            loss_aggregation == "advantage_sign_balanced_token_mean"
        ),
        "embodied_action_token_grpo/clip_epsilon": float(clip_epsilon),
        "embodied_action_token_grpo/clip_epsilon_low": float(clip_epsilon_low),
        "embodied_action_token_grpo/clip_epsilon_high": float(clip_epsilon_high),
        "embodied_action_token_grpo/clip_fraction": float(clip_values.mean().item())
        if clip_values.numel()
        else 0.0,
        "embodied_action_token_grpo/kl_coef": float(kl_coef),
        "embodied_action_token_grpo/reference_logprob_l2_coef": float(
            reference_logprob_l2_coef
        ),
        "embodied_action_token_grpo/reference_logprob_l2_examples": float(
            reference_l2_values.numel()
        ),
        "embodied_action_token_grpo/reference_logprob_l2_example_fraction": (
            float(reference_l2_values.numel()) / float(len(examples))
            if examples
            else 0.0
        ),
        "embodied_action_token_grpo/reference_logprob_l2_mean": float(
            reference_l2_values.mean().item()
        )
        if reference_l2_values.numel()
        else 0.0,
        "embodied_action_token_grpo/reference_logprob_l2_max": float(
            reference_l2_values.max().item()
        )
        if reference_l2_values.numel()
        else 0.0,
        "embodied_action_token_grpo/reference_logprob_l2_std": float(
            reference_l2_values.std(unbiased=False).item()
        )
        if reference_l2_values.numel()
        else 0.0,
        "embodied_action_token_grpo/reward_filter_enabled": float(
            bool((reward_filter_report or {}).get("enabled", False))
        ),
        "embodied_action_token_grpo/reward_filter_lower_bound": float(
            (reward_filter_report or {}).get("lower") or 0.0
        ),
        "embodied_action_token_grpo/reward_filter_upper_bound": float(
            (reward_filter_report or {}).get("upper") or 0.0
        ),
        "embodied_action_token_grpo/reward_filter_groups_total": float(
            (reward_filter_report or {}).get("groups_total", 0.0)
        ),
        "embodied_action_token_grpo/reward_filter_groups_kept": float(
            (reward_filter_report or {}).get("groups_kept", 0.0)
        ),
        "embodied_action_token_grpo/reward_filter_groups_filtered": float(
            (reward_filter_report or {}).get("groups_filtered", 0.0)
        ),
        "embodied_action_token_grpo/reward_filter_examples_before": float(
            (reward_filter_report or {}).get("examples_before", len(examples))
        ),
        "embodied_action_token_grpo/reward_filter_examples_after": float(
            (reward_filter_report or {}).get("examples_after", len(examples))
        ),
        "embodied_action_token_grpo/reward_filter_action_spans_before": float(
            (reward_filter_report or {}).get(
                "action_spans_before", _action_span_count(examples)
            )
        ),
        "embodied_action_token_grpo/reward_filter_action_spans_after": float(
            (reward_filter_report or {}).get(
                "action_spans_after", _action_span_count(examples)
            )
        ),
        "embodied_action_token_grpo/reward_filter_action_tokens_before": float(
            (reward_filter_report or {}).get(
                "action_tokens_before", _action_logprob_token_count(examples)
            )
        ),
        "embodied_action_token_grpo/reward_filter_action_tokens_after": float(
            (reward_filter_report or {}).get(
                "action_tokens_after", _action_logprob_token_count(examples)
            )
        ),
        "embodied_action_token_grpo/action_spans_after_filter": float(
            _action_span_count(examples)
        ),
        "embodied_action_token_grpo/action_tokens_after_filter": float(
            _action_logprob_token_count(examples)
        ),
        "embodied_action_token_grpo/approx_kl_mean": float(kl_values.mean().item())
        if kl_values.numel()
        else 0.0,
        "embodied_action_token_grpo/approx_kl_abs_mean": float(
            kl_values.abs().mean().item()
        )
        if kl_values.numel()
        else 0.0,
        "embodied_action_token_grpo/approx_kl_min": float(kl_values.min().item())
        if kl_values.numel()
        else 0.0,
        "embodied_action_token_grpo/approx_kl_max": float(kl_values.max().item())
        if kl_values.numel()
        else 0.0,
        "embodied_action_token_grpo/approx_kl_std": float(
            kl_values.std(unbiased=False).item()
        )
        if kl_values.numel()
        else 0.0,
        "embodied_action_token_grpo/ratio_mean": float(ratio_values.mean().item())
        if ratio_values.numel()
        else 0.0,
        "embodied_action_token_grpo/ratio_min": float(ratio_values.min().item())
        if ratio_values.numel()
        else 0.0,
        "embodied_action_token_grpo/ratio_max": float(ratio_values.max().item())
        if ratio_values.numel()
        else 0.0,
        "embodied_action_token_grpo/ratio_std": float(
            ratio_values.std(unbiased=False).item()
        )
        if ratio_values.numel()
        else 0.0,
        "embodied_action_token_grpo/clip_low_fraction": float(
            (ratio_values < (1.0 - clip_epsilon_low)).float().mean().item()
        )
        if ratio_values.numel()
        else 0.0,
        "embodied_action_token_grpo/clip_high_fraction": float(
            (ratio_values > (1.0 + clip_epsilon_high)).float().mean().item()
        )
        if ratio_values.numel()
        else 0.0,
        "embodied_action_token_grpo/loss": float(loss),
        "embodied_action_token_grpo/true_policy_gradient": 1.0,
    }
    if kl_values.numel():
        metrics.update(
            _quantile_metrics_from_values(
                kl_values,
                prefix="embodied_action_token_grpo/approx_kl",
            )
        )
        metrics.update(
            _quantile_metrics_from_values(
                kl_values.abs(),
                prefix="embodied_action_token_grpo/approx_kl_abs",
            )
        )
        metrics.update(
            _quantile_metrics_from_values(
                -kl_values,
                prefix="embodied_action_token_grpo/logprob_delta",
            )
        )
    if ratio_values.numel():
        metrics.update(
            _quantile_metrics_from_values(
                ratio_values,
                prefix="embodied_action_token_grpo/ratio",
            )
        )
    metrics.update(
        _group_signal_metrics(
            groups=groups,
            examples=examples,
            advantages=advantages,
            prefix="embodied_action_token_grpo",
        )
    )
    metrics.update(_task_balance_metrics(examples, prefix="embodied_action_token_grpo"))
    metrics.update(
        _objective_direction_metrics(
            objective_direction_stats,
            prefix="embodied_action_token_grpo",
        )
    )
    return metrics


def _example_task_balance_weight(example: ActionTokenExample) -> float:
    metadata = example.metadata if isinstance(example.metadata, dict) else {}
    value = metadata.get("task_balance_weight")
    if value is None:
        raise ValueError(
            "task_balanced_trajectory_mean requires coordinator-prepared "
            "task_balance_weight metadata"
        )
    weight = float(value)
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError(f"Invalid task_balance_weight={value!r}")
    return weight


def _task_balance_metrics(
    examples: Sequence[ActionTokenExample], *, prefix: str
) -> dict[str, float]:
    tasks: dict[int, dict[str, float]] = {}
    for example in examples:
        metadata = example.metadata if isinstance(example.metadata, dict) else {}
        task_index = int(metadata.get("task_balance_index", -1))
        row = tasks.setdefault(
            task_index,
            {
                "examples": 0.0,
                "signal_examples": 0.0,
                "global_examples": float(metadata.get("task_balance_task_examples", 0)),
                "global_signal_examples": float(
                    metadata.get("task_balance_task_signal_examples", 0)
                ),
                "groups": float(metadata.get("task_balance_task_groups", 0)),
                "signal_groups": float(
                    metadata.get("task_balance_task_signal_groups", 0)
                ),
            },
        )
        row["examples"] += 1.0
        advantage = metadata.get("group_advantage", example.reward)
        if abs(float(advantage)) > 1.0e-8:
            row["signal_examples"] += 1.0
    declared_task_counts = [
        int(example.metadata.get("task_balance_task_count", 0)) for example in examples
    ]
    metrics = {
        f"{prefix}/task_count": float(max(declared_task_counts, default=len(tasks)))
    }
    for task_index in sorted(tasks):
        row = tasks[task_index]
        task_prefix = f"{prefix}/by_task/{task_index:02d}"
        metrics[f"{task_prefix}/examples"] = row["examples"]
        metrics[f"{task_prefix}/signal_examples"] = row["signal_examples"]
        metrics[f"{task_prefix}/groups"] = row["groups"]
        metrics[f"{task_prefix}/signal_groups"] = row["signal_groups"]
    if tasks:
        counts = [row["global_examples"] for row in tasks.values()]
        signal_counts = [row["global_signal_examples"] for row in tasks.values()]
        group_counts = [row["groups"] for row in tasks.values()]
        signal_group_counts = [row["signal_groups"] for row in tasks.values()]
        metrics[f"{prefix}/task_examples_min"] = min(counts)
        metrics[f"{prefix}/task_examples_max"] = max(counts)
        metrics[f"{prefix}/task_signal_examples_min"] = min(signal_counts)
        metrics[f"{prefix}/task_signal_examples_max"] = max(signal_counts)
        metrics[f"{prefix}/task_groups_min"] = min(group_counts)
        metrics[f"{prefix}/task_groups_max"] = max(group_counts)
        metrics[f"{prefix}/task_signal_groups_min"] = min(signal_group_counts)
        metrics[f"{prefix}/task_signal_groups_max"] = max(signal_group_counts)
    return metrics


def _group_signal_metrics(
    *,
    groups: Sequence[EmbodiedTrajectoryGroup],
    examples: Sequence[ActionTokenExample],
    advantages: Sequence[float],
    prefix: str,
) -> dict[str, float]:
    """Expose whether grouped rollouts actually contain a learning signal.

    These diagnostics are deliberately task-agnostic.  They answer the first
    questions that matter when a trajectory-level robotics run is flat:

    - do comparable rollouts in the same group have different rewards?
    - did reward filtering leave examples with non-zero advantages?
    - after token/action masks, how many objective tokens still carry signal?
    """

    trajectory_metrics = _trajectory_group_signal_metrics(groups, prefix=prefix)
    groups_with_signal = int(
        trajectory_metrics[f"{prefix}/groups_with_reward_variance"]
    )
    trajectory_rewards = [
        float(trajectory.reward) for group in groups for trajectory in group
    ]

    # Distributed gradient workers receive flattened examples rather than the
    # original trajectory groups. Recover one reward per trajectory so this
    # diagnostic remains valid in distributed runs.
    if not trajectory_rewards:
        rewards_by_trajectory: dict[tuple[Any, Any], float] = {}
        for example in examples:
            metadata = example.metadata if isinstance(example.metadata, dict) else {}
            trajectory_key = (
                metadata.get("group_index", 0),
                metadata.get("trajectory_index_in_group", example.trajectory_index),
            )
            rewards_by_trajectory.setdefault(
                trajectory_key,
                float(metadata.get("trajectory_reward", example.reward)),
            )
        trajectory_rewards.extend(rewards_by_trajectory.values())

    examples_by_group: dict[Any, list[ActionTokenExample]] = {}
    for example in examples:
        group_index = example.metadata.get("group_index", 0)
        examples_by_group.setdefault(group_index, []).append(example)

    action_group_ranges: list[float] = []
    examples_in_action_signal_groups = 0
    for group_examples in examples_by_group.values():
        rewards = [float(example.reward) for example in group_examples]
        if not rewards:
            continue
        reward_range = max(rewards) - min(rewards)
        action_group_ranges.append(reward_range)
        if abs(reward_range) > 1e-8:
            examples_in_action_signal_groups += len(group_examples)

    action_groups_with_signal = sum(
        1 for value in action_group_ranges if abs(value) > 1e-8
    )
    zero_advantages = sum(1 for value in advantages if abs(float(value)) <= 1e-8)
    examples_in_signal_groups = sum(
        1
        for example in examples
        if abs(float(example.metadata.get("group_advantage", 0.0))) > 1e-8
    )
    action_reward_positive_examples = sum(
        1 for example in examples if float(example.reward) > 0.0
    )
    trajectory_reward_positive_examples = sum(
        1 for value in trajectory_rewards if value > 0.0
    )
    grammar_flags = [
        bool(example.metadata.get("action_metadata", {}).get("action_grammar_valid"))
        for example in examples
        if "action_grammar_valid" in example.metadata.get("action_metadata", {})
    ]
    discarded_after_termination = [
        float(
            example.metadata.get("action_metadata", {}).get(
                "post_termination_tokens_discarded", 0
            )
        )
        for example in examples
        if "post_termination_tokens_discarded"
        in example.metadata.get("action_metadata", {})
    ]

    objective_tokens_total = 0
    objective_tokens_kept = 0
    objective_advantage_nonzero_tokens = 0
    objective_advantage_abs_sum = 0.0
    objective_advantage_sum = 0.0
    objective_advantage_sumsq = 0.0
    objective_positive_advantage_tokens = 0
    objective_negative_advantage_tokens = 0
    objective_positive_advantage_abs_sum = 0.0
    objective_negative_advantage_abs_sum = 0.0
    token_advantage_metadata_examples = 0
    token_loss_mask_metadata_examples = 0

    for example in examples:
        metadata = example.metadata if isinstance(example.metadata, dict) else {}
        token_advantages = metadata.get("token_advantages")
        token_loss_mask = metadata.get("token_loss_mask")
        raw_token_count = _action_logprob_token_count([example])
        if isinstance(token_advantages, list | tuple):
            token_advantage_metadata_examples += 1
            if token_loss_mask is None:
                token_loss_mask = [True for _ in token_advantages]
            if not isinstance(token_loss_mask, list | tuple):
                raise ValueError("token_loss_mask must be a list or tuple")
            if len(token_advantages) != len(token_loss_mask):
                raise ValueError(
                    "token_advantages and token_loss_mask must have the same length"
                )
            token_loss_mask_metadata_examples += 1
            objective_tokens_total += len(token_advantages)
            for value, keep in zip(token_advantages, token_loss_mask, strict=True):
                if not bool(keep):
                    continue
                advantage_value = float(value)
                objective_tokens_kept += 1
                objective_advantage_abs_sum += abs(advantage_value)
                objective_advantage_sum += advantage_value
                objective_advantage_sumsq += advantage_value * advantage_value
                if abs(advantage_value) > 1e-8:
                    objective_advantage_nonzero_tokens += 1
                    if advantage_value > 0.0:
                        objective_positive_advantage_tokens += 1
                        objective_positive_advantage_abs_sum += advantage_value
                    elif advantage_value < 0.0:
                        objective_negative_advantage_tokens += 1
                        objective_negative_advantage_abs_sum += abs(advantage_value)
            continue

        # Generic action-token GRPO/GSPO path: every objective token in the
        # example receives the same group-relative advantage.
        kept_count = _example_objective_token_count(example)
        objective_tokens_total += max(raw_token_count, kept_count)
        objective_tokens_kept += kept_count
        advantage_value = float(metadata.get("group_advantage", 0.0))
        objective_advantage_abs_sum += abs(advantage_value) * kept_count
        objective_advantage_sum += advantage_value * kept_count
        objective_advantage_sumsq += (advantage_value * advantage_value) * kept_count
        if abs(advantage_value) > 1e-8:
            objective_advantage_nonzero_tokens += kept_count
            if advantage_value > 0.0:
                objective_positive_advantage_tokens += kept_count
                objective_positive_advantage_abs_sum += (
                    abs(advantage_value) * kept_count
                )
            elif advantage_value < 0.0:
                objective_negative_advantage_tokens += kept_count
                objective_negative_advantage_abs_sum += (
                    abs(advantage_value) * kept_count
                )

    objective_advantage_mean = (
        objective_advantage_sum / float(objective_tokens_kept)
        if objective_tokens_kept
        else 0.0
    )
    objective_advantage_variance = (
        objective_advantage_sumsq / float(objective_tokens_kept)
        - objective_advantage_mean**2
        if objective_tokens_kept
        else 0.0
    )
    objective_advantage_std = max(0.0, objective_advantage_variance) ** 0.5

    return {
        **trajectory_metrics,
        f"{prefix}/action_groups_with_reward_variance": float(
            action_groups_with_signal
        ),
        f"{prefix}/action_groups_with_zero_reward_variance": float(
            max(0, len(action_group_ranges) - action_groups_with_signal)
        ),
        f"{prefix}/action_group_reward_range_mean": _mean(action_group_ranges),
        f"{prefix}/action_effective_group_fraction": (
            float(action_groups_with_signal) / float(len(action_group_ranges))
            if action_group_ranges
            else 0.0
        ),
        f"{prefix}/action_examples_in_signal_groups": float(
            examples_in_action_signal_groups
        ),
        f"{prefix}/nonzero_advantage_examples": float(examples_in_signal_groups),
        f"{prefix}/zero_advantage_fraction": (
            float(zero_advantages) / float(len(advantages)) if advantages else 0.0
        ),
        f"{prefix}/action_reward_positive_fraction": (
            float(action_reward_positive_examples) / float(len(examples))
            if examples
            else 0.0
        ),
        f"{prefix}/trajectory_reward_positive_fraction": (
            float(trajectory_reward_positive_examples) / float(len(trajectory_rewards))
            if trajectory_rewards
            else 0.0
        ),
        f"{prefix}/action_grammar_invalid_fraction": (
            float(sum(not valid for valid in grammar_flags)) / float(len(grammar_flags))
            if grammar_flags
            else 0.0
        ),
        f"{prefix}/post_termination_tokens_discarded_mean": _mean(
            discarded_after_termination
        ),
        f"{prefix}/objective_tokens_total_before_mask": float(objective_tokens_total),
        f"{prefix}/objective_tokens_kept_after_mask": float(objective_tokens_kept),
        f"{prefix}/objective_token_loss_mask_kept_fraction": (
            float(objective_tokens_kept) / float(objective_tokens_total)
            if objective_tokens_total
            else 0.0
        ),
        f"{prefix}/objective_token_advantage_nonzero_fraction": (
            float(objective_advantage_nonzero_tokens) / float(objective_tokens_kept)
            if objective_tokens_kept
            else 0.0
        ),
        f"{prefix}/objective_token_advantage_abs_mean": (
            objective_advantage_abs_sum / float(objective_tokens_kept)
            if objective_tokens_kept
            else 0.0
        ),
        f"{prefix}/objective_token_advantage_mean": objective_advantage_mean,
        f"{prefix}/objective_token_advantage_std": objective_advantage_std,
        f"{prefix}/objective_token_positive_advantage_fraction": (
            float(objective_positive_advantage_tokens) / float(objective_tokens_kept)
            if objective_tokens_kept
            else 0.0
        ),
        f"{prefix}/objective_token_negative_advantage_fraction": (
            float(objective_negative_advantage_tokens) / float(objective_tokens_kept)
            if objective_tokens_kept
            else 0.0
        ),
        f"{prefix}/objective_token_positive_advantage_abs_sum": objective_positive_advantage_abs_sum,
        f"{prefix}/objective_token_negative_advantage_abs_sum": objective_negative_advantage_abs_sum,
        f"{prefix}/objective_token_positive_negative_abs_ratio": (
            objective_positive_advantage_abs_sum / objective_negative_advantage_abs_sum
            if objective_negative_advantage_abs_sum > 0.0
            else 0.0
        ),
        f"{prefix}/token_advantage_metadata_example_fraction": (
            float(token_advantage_metadata_examples) / float(len(examples))
            if examples
            else 0.0
        ),
        f"{prefix}/token_loss_mask_metadata_example_fraction": (
            float(token_loss_mask_metadata_examples) / float(len(examples))
            if examples
            else 0.0
        ),
        f"{prefix}/group_relative_signal_available": float(
            groups_with_signal > 0 or action_groups_with_signal > 0
        ),
    }


def _trajectory_group_signal_metrics(
    groups: Sequence[EmbodiedTrajectoryGroup],
    *,
    prefix: str,
) -> dict[str, float]:
    """Compute metrics that require complete, unsharded trajectory groups."""

    group_ranges: list[float] = []
    trajectory_rewards: list[float] = []
    for group in groups:
        rewards = [float(trajectory.reward) for trajectory in group]
        if not rewards:
            continue
        trajectory_rewards.extend(rewards)
        group_ranges.append(max(rewards) - min(rewards))

    nonempty_groups = len(group_ranges)
    groups_with_signal = sum(1 for value in group_ranges if abs(value) > 1e-8)
    positive_trajectories = sum(1 for reward in trajectory_rewards if reward > 0.0)
    return {
        f"{prefix}/groups_nonempty": float(nonempty_groups),
        f"{prefix}/groups_with_reward_variance": float(groups_with_signal),
        f"{prefix}/groups_with_zero_reward_variance": float(
            max(0, nonempty_groups - groups_with_signal)
        ),
        f"{prefix}/group_reward_range_mean": _mean(group_ranges),
        f"{prefix}/effective_group_fraction": (
            float(groups_with_signal) / float(nonempty_groups)
            if nonempty_groups
            else 0.0
        ),
        f"{prefix}/trajectory_reward_positive_fraction": (
            float(positive_trajectories) / float(len(trajectory_rewards))
            if trajectory_rewards
            else 0.0
        ),
    }


def _empty_gradient_handoff_metrics(
    *,
    groups: Sequence[EmbodiedTrajectoryGroup],
    reward_filter_report: dict[str, Any] | None,
    global_example_count: Any,
    global_token_count: Any,
    microbatch_size: int,
    logprob_eval_mode: bool,
    gradient_payload: dict[str, Any] | None,
    gradient_handoff_worker: bool,
    policy: Any,
) -> dict[str, float]:
    metrics = {
        "embodied_action_token_grpo/groups": float(len(groups)),
        "embodied_action_token_grpo/examples": 0.0,
        "embodied_action_token_grpo/tokens_total": 0.0,
        "embodied_action_token_grpo/tokens_mean": 0.0,
        "embodied_action_token_grpo/reward_mean": 0.0,
        "embodied_action_token_grpo/advantage_mean": 0.0,
        "embodied_action_token_grpo/advantage_min": 0.0,
        "embodied_action_token_grpo/advantage_max": 0.0,
        "embodied_action_token_grpo/group_relative_signal_available": 0.0,
        "embodied_action_token_grpo/true_policy_gradient": 1.0,
        "embodied_action_token_grpo/loss": 0.0,
        "embodied_action_token_grpo/gradient_handoff_worker": float(
            gradient_handoff_worker
        ),
        "embodied_action_token_grpo/gradient_handoff_tensors": float(
            len((gradient_payload or {}).get("gradients") or {})
        ),
        "embodied_action_token_grpo/global_loss_denominator_examples": float(
            global_example_count or 0
        ),
        "embodied_action_token_grpo/global_loss_denominator_tokens": float(
            global_token_count or 0
        ),
        "embodied_action_token_grpo/optimizer_step_completed": 0.0,
        "embodied_action_token_grpo/optimizer_step_skipped_no_group_signal": 1.0,
        "embodied_action_token_grpo/optimizer_step_skipped_logprob_misalignment": 0.0,
        "embodied_action_token_grpo/logprob_eval_mode": float(logprob_eval_mode),
        "embodied_action_token_grpo/logprob_microbatch_size": float(microbatch_size),
        "embodied_action_token_grpo/logprob_microbatches": 0.0,
        "embodied_action_token_grpo/policy_parameters_updated": 0.0,
    }
    metrics.update(_gradient_metrics(policy))
    if reward_filter_report is not None:
        for key, value in reward_filter_report.items():
            if _is_number(value):
                metrics[f"embodied_action_token_grpo/reward_filter_{key}"] = float(
                    value
                )
    return metrics


def _gradient_direction_probe_metrics(
    backend: Any,
    *,
    groups: list[EmbodiedTrajectoryGroup],
    config: Any,
    prefix: str = "embodied_action_token_grpo",
) -> dict[str, float]:
    """Probe whether the raw gradient direction improves the stated objective.

    This is a diagnostic, not an optimizer.  It temporarily applies tiny
    SGD-style steps along the current gradient, measures the same no-gradient
    probe used by the trust-region gate, and restores the exact pre-probe
    trainable-parameter snapshot.  If this improves the surrogate but the real
    optimizer step does not, the bug is likely in optimizer/update conversion.
    If this also worsens the surrogate, the issue is in the loss/probe contract
    or the gradient itself.
    """

    if not isinstance(config, Mapping) or not bool(config.get("enabled")):
        return {}
    import torch

    policy = getattr(backend, "policy", None)
    if policy is None or not hasattr(policy, "named_parameters"):
        return {
            f"{prefix}/gradient_direction_probe_enabled": 1.0,
            f"{prefix}/gradient_direction_probe_available": 0.0,
        }

    raw_scales = config.get("scales")
    if raw_scales in (None, ""):
        raw_scales = (1.0,)
    scales = [float(scale) for scale in raw_scales]
    if not scales:
        return {
            f"{prefix}/gradient_direction_probe_enabled": 1.0,
            f"{prefix}/gradient_direction_probe_available": 0.0,
        }
    max_groups = config.get("max_groups")
    probe_groups = list(groups)
    if max_groups not in (None, ""):
        max_groups_int = max(1, int(max_groups))
        probe_groups = probe_groups[:max_groups_int]

    named_parameters = [
        (name, parameter)
        for name, parameter in policy.named_parameters()
        if bool(getattr(parameter, "requires_grad", False))
    ]
    param_lrs = _optimizer_parameter_lrs(
        getattr(backend, "optimizer", None),
        default_lr=float(getattr(backend, "lr", 0.0) or 0.0),
    )
    grad_tensors = [
        (name, parameter)
        for name, parameter in named_parameters
        if getattr(parameter, "grad", None) is not None
    ]
    metrics: dict[str, float] = {
        f"{prefix}/gradient_direction_probe_enabled": 1.0,
        f"{prefix}/gradient_direction_probe_available": float(
            bool(probe_groups and grad_tensors)
        ),
        f"{prefix}/gradient_direction_probe_groups": float(len(probe_groups)),
        f"{prefix}/gradient_direction_probe_scales": float(len(scales)),
        f"{prefix}/gradient_direction_probe_trainable_tensors": float(
            len(named_parameters)
        ),
        f"{prefix}/gradient_direction_probe_grad_tensors": float(len(grad_tensors)),
    }
    if not probe_groups or not grad_tensors:
        return metrics

    snapshot = _trainable_parameter_snapshot(policy)
    start = time.perf_counter()
    best_advprod: float | None = None
    best_surrogate_delta: float | None = None
    improved_count = 0
    try:
        for index, scale in enumerate(scales):
            _restore_trainable_parameter_snapshot(policy, snapshot)
            expected_delta_norm_sq = 0.0
            expected_delta_abs_max = 0.0
            with torch.no_grad():
                for _name, parameter in grad_tensors:
                    grad = getattr(parameter, "grad", None)
                    if grad is None:
                        continue
                    lr = float(
                        param_lrs.get(id(parameter), getattr(backend, "lr", 0.0) or 0.0)
                    )
                    alpha = -lr * float(scale)
                    delta = grad.detach().float() * float(alpha)
                    if delta.numel():
                        expected_delta_norm_sq += float(delta.pow(2).sum().item())
                        expected_delta_abs_max = max(
                            expected_delta_abs_max,
                            float(delta.abs().max().item()),
                        )
                    parameter.add_(grad, alpha=alpha)

            probe = backend.probe_logprob_metrics(
                probe_groups,
                gradient_direction_probe=1.0,
                gradient_direction_probe_scale=float(scale),
            )
            key = f"{prefix}/gradient_direction_probe_scale_{index:02d}"
            probe_prefix = "embodied_action_token_grpo"
            kl_abs = _optional_float(probe.get(f"{probe_prefix}/approx_kl_abs_mean"))
            advprod = _optional_float(
                probe.get(
                    f"{probe_prefix}/objective_direction_advantage_logprob_delta_product_mean"
                )
            )
            weighted_advprod = _optional_float(
                probe.get(
                    f"{probe_prefix}/surrogate_weighted/objective_direction_advantage_logprob_delta_product_mean"
                )
            )
            pos_ratio = _optional_float(
                probe.get(f"{probe_prefix}/objective_direction_positive_ratio_mean")
            )
            neg_ratio = _optional_float(
                probe.get(f"{probe_prefix}/objective_direction_negative_ratio_mean")
            )
            weighted_pos_ratio = _optional_float(
                probe.get(
                    f"{probe_prefix}/surrogate_weighted/objective_direction_positive_ratio_mean"
                )
            )
            weighted_neg_ratio = _optional_float(
                probe.get(
                    f"{probe_prefix}/surrogate_weighted/objective_direction_negative_ratio_mean"
                )
            )
            surrogate_delta = _optional_float(
                probe.get(f"{probe_prefix}/probe_surrogate_loss_delta")
            )
            unclipped_surrogate_delta = _optional_float(
                probe.get(f"{probe_prefix}/probe_unclipped_surrogate_loss_delta")
            )
            clipping_surrogate_gap = _optional_float(
                probe.get(f"{probe_prefix}/probe_clipping_surrogate_loss_delta_gap")
            )
            metrics[f"{key}/scale"] = float(scale)
            metrics[f"{key}/expected_raw_sgd_delta_norm"] = float(
                expected_delta_norm_sq**0.5
            )
            metrics[f"{key}/expected_raw_sgd_delta_abs_max"] = float(
                expected_delta_abs_max
            )
            if kl_abs is not None:
                metrics[f"{key}/approx_kl_abs_mean"] = float(kl_abs)
            if advprod is not None:
                metrics[
                    f"{key}/objective_direction_advantage_logprob_delta_product_mean"
                ] = float(advprod)
                best_advprod = (
                    advprod if best_advprod is None else max(best_advprod, advprod)
                )
            if weighted_advprod is not None:
                metrics[
                    f"{key}/surrogate_weighted_objective_direction_advantage_logprob_delta_product_mean"
                ] = float(weighted_advprod)
            if pos_ratio is not None:
                metrics[f"{key}/objective_direction_positive_ratio_mean"] = float(
                    pos_ratio
                )
            if neg_ratio is not None:
                metrics[f"{key}/objective_direction_negative_ratio_mean"] = float(
                    neg_ratio
                )
            if weighted_pos_ratio is not None:
                metrics[
                    f"{key}/surrogate_weighted_objective_direction_positive_ratio_mean"
                ] = float(weighted_pos_ratio)
            if weighted_neg_ratio is not None:
                metrics[
                    f"{key}/surrogate_weighted_objective_direction_negative_ratio_mean"
                ] = float(weighted_neg_ratio)
            if surrogate_delta is not None:
                metrics[f"{key}/probe_surrogate_loss_delta"] = float(surrogate_delta)
                best_surrogate_delta = (
                    surrogate_delta
                    if best_surrogate_delta is None
                    else min(best_surrogate_delta, surrogate_delta)
                )
            if unclipped_surrogate_delta is not None:
                metrics[f"{key}/probe_unclipped_surrogate_loss_delta"] = float(
                    unclipped_surrogate_delta
                )
            if clipping_surrogate_gap is not None:
                metrics[f"{key}/probe_clipping_surrogate_loss_delta_gap"] = float(
                    clipping_surrogate_gap
                )
            improved = (
                advprod is not None
                and advprod > 0.0
                and surrogate_delta is not None
                and surrogate_delta < 0.0
            )
            metrics[f"{key}/improved_surrogate_and_direction"] = float(improved)
            improved_count += int(improved)
    except Exception:
        metrics[f"{prefix}/gradient_direction_probe_failed"] = 1.0
        raise
    finally:
        _restore_trainable_parameter_snapshot(policy, snapshot)

    metrics[f"{prefix}/gradient_direction_probe_improved_scales"] = float(
        improved_count
    )
    if best_advprod is not None:
        metrics[
            f"{prefix}/gradient_direction_probe_best_advantage_logprob_delta_product_mean"
        ] = float(best_advprod)
    if best_surrogate_delta is not None:
        metrics[f"{prefix}/gradient_direction_probe_best_surrogate_loss_delta"] = float(
            best_surrogate_delta
        )
    metrics[f"{prefix}/gradient_direction_probe_seconds"] = float(
        time.perf_counter() - start
    )
    return metrics


def _infer_policy_device(policy: Any) -> str | None:
    if not hasattr(policy, "parameters"):
        return None
    try:
        first_param = next(policy.parameters())
    except StopIteration:
        return None
    except Exception:
        return None
    return str(first_param.device)


def _set_train(policy: Any) -> None:
    if hasattr(policy, "train"):
        policy.train()


def _set_eval(policy: Any) -> None:
    if hasattr(policy, "eval"):
        policy.eval()


def _policy_gradient_algorithm_name(importance_sampling_level: str) -> str:
    return "gspo" if importance_sampling_level in ("sequence", "trajectory") else "grpo"


def _alias_metric_prefix(
    metrics: dict[str, float],
    *,
    source_prefix: str,
    target_prefix: str,
    drop_source: bool = False,
) -> dict[str, float]:
    """Return public aliases and optionally hide the internal metric namespace."""

    aliases = {
        target_prefix + key.removeprefix(source_prefix): value
        for key, value in metrics.items()
        if key.startswith(source_prefix)
    }
    if drop_source:
        for key in tuple(metrics):
            if key.startswith(source_prefix):
                del metrics[key]
    return aliases


def _save_policy_checkpoint(
    policy: Any,
    path: Path,
    *,
    config_fingerprint: str | None = None,
    resume_contract_fingerprint: str | None = None,
) -> Path | None:
    save_checkpoint = getattr(policy, "save_checkpoint", None)
    save_pretrained = getattr(policy, "save_pretrained", None)
    private_save_pretrained = getattr(policy, "_save_pretrained", None)
    state_dict = getattr(policy, "state_dict", None)
    if not any(
        callable(candidate)
        for candidate in (
            save_checkpoint,
            save_pretrained,
            private_save_pretrained,
            state_dict,
        )
    ):
        return None

    def write_checkpoint(staging: Path) -> None:
        if callable(save_checkpoint):
            result = save_checkpoint(str(staging))
            if isinstance(result, Mapping) and result.get("path"):
                reported = Path(str(result["path"]))
                if reported.resolve() != staging.resolve():
                    raise ValueError(
                        "policy.save_checkpoint(path) must publish into the "
                        f"requested directory: requested={staging}, reported={reported}"
                    )
            return
        if callable(save_pretrained):
            save_pretrained(staging)
            return
        if callable(private_save_pretrained):
            private_save_pretrained(staging)
            return
        import torch

        torch.save(state_dict(), staging / "policy_state.pt")

    return _CHECKPOINT_MANAGER.publish(
        path,
        writer=write_checkpoint,
        config_fingerprint=config_fingerprint or "unscoped-direct-backend",
        resume_contract_fingerprint=(
            resume_contract_fingerprint or "unscoped-direct-backend"
        ),
        metadata={
            "backend": "action_token_direct",
            "contains_training_state": False,
        },
    )
