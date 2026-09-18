"""Strict, visible experiment configuration for embodied RL.

Embodied experiments are unusually sensitive to details such as environment
reset behavior, action processors, and evaluation sampling.  These models make
those details part of the serialized experiment contract instead of hiding
them in environment variables or runner defaults.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal

import pydantic
import yaml


class _StrictConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


class ExperimentIdentity(_StrictConfig):
    """Stable run identity and seed recorded with every experiment artifact."""

    project: str = pydantic.Field(min_length=1)
    run: str = pydantic.Field(min_length=1)
    seed: int
    tags: list[str]


class LoraWarmStartSourceConfig(_StrictConfig):
    """Immutable provenance for one learned single-task LoRA source."""

    source_kind: Literal["validated_grpo", "partition_sft"] = "validated_grpo"
    checkpoint: Path
    source_config: Path
    checkpoint_manifest_sha256: str = pydantic.Field(pattern=r"^[0-9a-f]{64}$")
    source_config_sha256: str = pydantic.Field(pattern=r"^[0-9a-f]{64}$")
    development_adjudication: Path | None = None
    development_adjudication_sha256: str | None = pydantic.Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    sealed_adjudication: Path | None = None
    sealed_adjudication_sha256: str | None = pydantic.Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    sft_completion: Path | None = None
    sft_completion_sha256: str | None = pydantic.Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    final_step: int = pydantic.Field(ge=1)

    @pydantic.model_validator(mode="after")
    def require_source_evidence(self) -> "LoraWarmStartSourceConfig":
        grpo_fields = (
            self.development_adjudication,
            self.development_adjudication_sha256,
            self.sealed_adjudication,
            self.sealed_adjudication_sha256,
        )
        sft_fields = (self.sft_completion, self.sft_completion_sha256)
        if self.source_kind == "validated_grpo":
            if any(value is None for value in grpo_fields):
                raise ValueError(
                    "validated_grpo warm starts require development and sealed "
                    "adjudications with SHA256 digests"
                )
            if any(value is not None for value in sft_fields):
                raise ValueError(
                    "validated_grpo warm starts cannot declare SFT completion evidence"
                )
        elif self.source_kind == "partition_sft":
            if any(value is None for value in sft_fields):
                raise ValueError(
                    "partition_sft warm starts require an SFT completion manifest "
                    "and SHA256 digest"
                )
            if any(value is not None for value in grpo_fields):
                raise ValueError(
                    "partition_sft warm starts cannot declare GRPO adjudications"
                )
        return self


class LoraCompositionAdmissionConfig(_StrictConfig):
    """Immutable proof that a composed policy passed its pre-optimizer gates."""

    checkpoint: Path
    checkpoint_manifest_sha256: str = pydantic.Field(pattern=r"^[0-9a-f]{64}$")
    retention_adjudication: Path
    retention_adjudication_sha256: str = pydantic.Field(pattern=r"^[0-9a-f]{64}$")


class LoraRankPartitionConfig(_StrictConfig):
    """Disjoint trainable rank blocks with explicit forward routing."""

    mode: Literal["task_all_active"]
    task_keys: list[str] = pydantic.Field(min_length=2)
    allocation: Literal["balanced"] = "balanced"
    forward_routing: Literal["all_active", "task_owned"] = "all_active"
    warm_start_sources: dict[str, LoraWarmStartSourceConfig] | None = None
    composition_admission: LoraCompositionAdmissionConfig | None = None

    @pydantic.model_validator(mode="after")
    def require_unique_task_keys(self) -> "LoraRankPartitionConfig":
        normalized = [value.strip() for value in self.task_keys]
        if any(not value for value in normalized):
            raise ValueError("LoRA rank-partition task keys cannot be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("LoRA rank-partition task keys must be unique")
        if normalized != self.task_keys:
            raise ValueError(
                "LoRA rank-partition task keys cannot have surrounding whitespace"
            )
        if self.warm_start_sources is not None:
            checkpoint_keys = set(self.warm_start_sources)
            task_keys = set(self.task_keys)
            if checkpoint_keys != task_keys:
                raise ValueError(
                    "LoRA rank-partition warm-start checkpoints must cover every "
                    "task key exactly; "
                    f"missing={sorted(task_keys - checkpoint_keys)}, "
                    f"unexpected={sorted(checkpoint_keys - task_keys)}"
                )
            normalized_paths = [
                str(source.checkpoint.expanduser().resolve(strict=False))
                for source in self.warm_start_sources.values()
            ]
            if len(set(normalized_paths)) != len(normalized_paths):
                raise ValueError(
                    "LoRA rank-partition warm-start checkpoints must be unique per task"
                )
        if self.composition_admission is not None and self.warm_start_sources is None:
            raise ValueError("LoRA composition admission requires warm_start_sources")
        return self


class LoraConfig(_StrictConfig):
    """Parameter-efficient adaptation surface for the policy under training."""

    enabled: bool
    rank: int = pydantic.Field(ge=1)
    alpha: int = pydantic.Field(ge=1)
    dropout: float = pydantic.Field(ge=0.0, lt=1.0)
    init: Literal["default", "gaussian"]
    target_modules: list[str]
    rank_partition: LoraRankPartitionConfig | None = None

    @pydantic.model_validator(mode="after")
    def require_targets_when_enabled(self) -> "LoraConfig":
        if self.enabled and not self.target_modules:
            raise ValueError(
                "policy.lora.target_modules cannot be empty when LoRA is enabled"
            )
        if self.rank_partition is not None:
            if not self.enabled:
                raise ValueError("LoRA rank partitioning requires lora.enabled=true")
            if self.rank < len(self.rank_partition.task_keys):
                raise ValueError(
                    "LoRA rank partitioning requires at least one rank per task"
                )
            if self.alpha != self.rank:
                raise ValueError(
                    "Experimental task_all_active partitioning requires alpha=rank "
                    "so every independently sized rank block has identical scaling"
                )
        return self


class GenerationConfig(_StrictConfig):
    """Sampling policy for one rollout, training, or evaluation phase."""

    do_sample: bool
    temperature: float = pydantic.Field(gt=0.0)
    top_p: float = pydantic.Field(gt=0.0, le=1.0)


class OpenVLALoadConfig(_StrictConfig):
    """OpenVLA-OFT loading choices that can alter action-token semantics."""

    runtime_contract: Literal["openvla_oft_v01", "unchecked"] = "openvla_oft_v01"
    robot_platform: Literal["libero", "aloha", "bridge"]
    attn_implementation: str | None
    peft_adapter_path: str | None
    dataset_statistics_path: str | None
    logprob_batch_size: int | None = pydantic.Field(default=None, ge=1)
    strict_batched_logprobs: bool
    max_prompt_length: int | None = pydantic.Field(default=None, ge=1)
    prompt_template: str | None
    lowercase_instruction: bool
    num_images_in_input: int | None = pydantic.Field(default=None, ge=1)
    use_proprio: bool | None
    action_dim: int = pydantic.Field(ge=1)
    num_action_chunks: int = pydantic.Field(ge=1)
    model_loader: Literal["native", "transformers", "rlinf"]

    @pydantic.model_validator(mode="after")
    def require_validated_attention_backend(self) -> "OpenVLALoadConfig":
        if (
            self.runtime_contract == "openvla_oft_v01"
            and self.attn_implementation is not None
        ):
            raise ValueError(
                "OpenVLA-OFT v0.1 requires the checkpoint/default attention "
                "selection; explicit attention overrides change its "
                "bidirectional action-token logits. Set "
                "policy.load_kwargs.attn_implementation to null, "
                "or select runtime_contract='unchecked' for unsupported diagnostics."
            )
        if self.model_loader == "rlinf" and self.robot_platform != "libero":
            raise ValueError(
                "The optional RLinf conformance loader currently implements the "
                "LIBERO OpenVLA-OFT contract only; use model_loader='native' for "
                f"robot_platform={self.robot_platform!r}"
            )
        return self


class PIFlowLoadConfig(_StrictConfig):
    """LeRobot PI0/PI0.5 model and execution-shape assertions."""

    runtime_contract: Literal["lerobot_v060"]
    model_format: Literal["lerobot", "rlinf_openpi_safetensors"] = "lerobot"
    execution_horizon: int = pydantic.Field(ge=1)
    action_dim: int = pydantic.Field(ge=1)
    processor_path: str | None = None
    processor_revision: str | None = None
    model_chunk_size: int | None = pydantic.Field(default=None, ge=1)
    normalization_stats_file: str | None = None
    discrete_state_input: bool | None = None
    extra_delta_transform: bool | None = None
    observation_key_map: dict[str, str] = pydantic.Field(default_factory=dict)
    strict_weights: bool
    compile_model: bool
    gradient_checkpointing: bool
    train_expert_only: bool

    @pydantic.model_validator(mode="after")
    def _validate_model_source_contract(self) -> "PIFlowLoadConfig":
        if any(
            not source or not target
            for source, target in self.observation_key_map.items()
        ):
            raise ValueError("observation_key_map keys and values must be non-empty")
        if len(set(self.observation_key_map.values())) != len(self.observation_key_map):
            raise ValueError("observation_key_map targets must be unique")
        if self.model_format == "rlinf_openpi_safetensors" and not self.processor_path:
            raise ValueError(
                "rlinf_openpi_safetensors requires processor_path to provide the "
                "LeRobot architecture and processor contract"
            )
        if self.model_format == "rlinf_openpi_safetensors":
            if self.model_chunk_size is None:
                raise ValueError(
                    "rlinf_openpi_safetensors requires model_chunk_size because "
                    "raw OpenPI checkpoints do not carry a LeRobot config"
                )
            if not self.normalization_stats_file:
                raise ValueError(
                    "rlinf_openpi_safetensors requires normalization_stats_file; "
                    "checkpoint weights and state/action coordinates are one contract"
                )
            if self.discrete_state_input is None:
                raise ValueError(
                    "rlinf_openpi_safetensors requires discrete_state_input because "
                    "raw OpenPI checkpoints do not carry their prompt/state contract"
                )
            if self.extra_delta_transform is None:
                raise ValueError(
                    "rlinf_openpi_safetensors requires extra_delta_transform because "
                    "older PI0 LIBERO checkpoints and PI0.5 use different action "
                    "coordinate contracts"
                )
        elif (
            self.discrete_state_input is not None
            or self.extra_delta_transform is not None
        ):
            raise ValueError(
                "discrete_state_input and extra_delta_transform are owned by the "
                "serialized LeRobot processor when model_format='lerobot'"
            )
        return self


class SmolVLAFlowLoadConfig(_StrictConfig):
    """LeRobot SmolVLA model and execution-shape assertions."""

    runtime_contract: Literal["lerobot_v060"]
    execution_horizon: int = pydantic.Field(ge=1)
    action_dim: int = pydantic.Field(ge=1)
    observation_key_map: dict[str, str]
    strict_weights: bool
    compile_model: bool
    gradient_checkpointing: Literal[False]
    train_expert_only: bool

    @pydantic.model_validator(mode="after")
    def _validate_observation_key_map(self) -> "SmolVLAFlowLoadConfig":
        if any(
            not source or not target
            for source, target in self.observation_key_map.items()
        ):
            raise ValueError(
                "SmolVLA observation_key_map keys and values cannot be empty"
            )
        targets = list(self.observation_key_map.values())
        if len(set(targets)) != len(targets):
            raise ValueError("SmolVLA observation_key_map targets must be unique")
        return self


class PI0FastSFTAnchorConfig(_StrictConfig):
    """Task-balanced native-SFT anchor for categorical pi0-FAST GRPO."""

    coefficient: float = pydantic.Field(gt=0.0)
    dataset_repo_id: str = pydantic.Field(min_length=1)
    dataset_revision: str = pydantic.Field(pattern=r"^[0-9a-f]{40}$")
    task_indices: list[int] = pydantic.Field(min_length=1)
    seed: int
    samples_per_task: Literal[1] = 1

    @pydantic.model_validator(mode="after")
    def require_unique_tasks(self) -> "PI0FastSFTAnchorConfig":
        if len(set(self.task_indices)) != len(self.task_indices):
            raise ValueError("pi0-FAST SFT anchor task_indices must be unique")
        if any(task < 0 for task in self.task_indices):
            raise ValueError("pi0-FAST SFT anchor task_indices must be non-negative")
        return self


class PI0FastLoadConfig(_StrictConfig):
    """LeRobot pi0-FAST autoregressive action-token contract."""

    runtime_contract: Literal["lerobot_v060"]
    execution_horizon: int = pydantic.Field(ge=1)
    action_dim: int = pydantic.Field(ge=1)
    max_decoding_steps: int = pydantic.Field(ge=1)
    action_tokenizer_revision: str = pydantic.Field(pattern=r"^[0-9a-f]{40}$")
    observation_key_map: dict[str, str]
    strict_weights: bool
    compile_model: bool
    gradient_checkpointing: bool
    use_kv_cache: Literal[True]
    model_compute_dtype: Literal["checkpoint", "float32", "fp16_residual"] = (
        "checkpoint"
    )
    training_loss_scale: float = pydantic.Field(default=1.0, ge=1, allow_inf_nan=False)
    training_logprob_mode: Literal["kv", "full_sequence"] = "kv"
    invalid_action_handling: Literal["raise", "terminate_episode"] = "raise"
    action_decoder: Literal["strict", "native"] = "strict"
    rl_token_scope: Literal["generated_sequence", "fast_payload"] = "generated_sequence"
    sft_anchor: PI0FastSFTAnchorConfig | None = None
    warm_start_checkpoint: Path | None = None

    @pydantic.model_validator(mode="after")
    def _validate_observation_key_map(self) -> "PI0FastLoadConfig":
        import math

        if not math.log2(self.training_loss_scale).is_integer():
            raise ValueError("FAST training_loss_scale must be a power of two")
        if self.model_compute_dtype == "fp16_residual":
            if self.compile_model:
                raise ValueError("fp16_residual is not qualified with compile_model")
        elif self.training_loss_scale != 1:
            raise ValueError("Loss scaling is only supported by fp16_residual")
        if (
            self.action_decoder == "native"
            and self.rl_token_scope != "generated_sequence"
        ):
            raise ValueError("Native decoding requires generated_sequence token scope")
        if (
            self.invalid_action_handling == "terminate_episode"
            and self.rl_token_scope != "generated_sequence"
        ):
            raise ValueError(
                "terminate_episode requires generated_sequence token scope"
            )
        if any(
            not source or not target
            for source, target in self.observation_key_map.items()
        ):
            raise ValueError(
                "pi0-FAST observation_key_map keys and values cannot be empty"
            )
        targets = list(self.observation_key_map.values())
        if len(set(targets)) != len(targets):
            raise ValueError("pi0-FAST observation_key_map targets must be unique")
        return self


class GR00TFlowLoadConfig(_StrictConfig):
    """NVIDIA GR00T N1.5 model and embodiment contract.

    N1.7 has a materially different processor and action-head API. It will use
    a separate runtime contract rather than silently passing through this one.
    """

    runtime_contract: Literal["nvidia_n1d5"]
    model_format: Literal["nvidia_n1d5"]
    embodiment_tag: str = pydantic.Field(min_length=1)
    data_config: str = pydantic.Field(min_length=1)
    execution_horizon: int = pydantic.Field(ge=1)
    action_dim: int = pydantic.Field(ge=1)
    model_action_horizon: int = pydantic.Field(ge=1)
    language_padding_length: int = pydantic.Field(ge=1)
    disable_dropout: Literal[True]
    compile_model: Literal[False]
    gradient_checkpointing: Literal[False]
    train_expert_only: Literal[True]


class GR00TN17ActionComponentConfig(_StrictConfig):
    """One processor action component and its physical execution contract."""

    key: str = pydantic.Field(min_length=1)
    size: int = pydantic.Field(ge=1)
    executed: bool
    execution_order: int | None = pydantic.Field(default=None, ge=0)


class GR00TN17SFTReplayConfig(_StrictConfig):
    """Deterministic native-SFT auxiliary batches for N1.7 gradient workers."""

    coefficient: float = pydantic.Field(ge=0.0)
    dataset_root: Path
    task_ids: list[str] = pydantic.Field(min_length=1)
    seed: int
    samples_per_worker: Literal[1] = 1
    sampling_strategy: Literal["worker_task_balanced"] = "worker_task_balanced"
    processor_mode: Literal["eval"] = "eval"
    dataset_prefix: str = pydantic.Field(default="gr1_unified", min_length=1)

    @pydantic.model_validator(mode="after")
    def require_unique_tasks(self) -> "GR00TN17SFTReplayConfig":
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("policy.load_kwargs.sft_replay.task_ids must be unique")
        return self


class GR00TN17FlowLoadConfig(_StrictConfig):
    """Pinned NVIDIA N1.7 model, processor, and embodiment horizon contract."""

    runtime_contract: Literal["nvidia_n1d7"]
    model_format: Literal["nvidia_n1d7"]
    checkpoint_subfolder: str = pydantic.Field(min_length=1)
    embodiment_tag: str = pydantic.Field(min_length=1)
    execution_horizon: int = pydantic.Field(ge=1)
    processor_action_horizon: int = pydantic.Field(ge=1)
    action_dim: int = pydantic.Field(ge=1)
    execution_action_dim: int = pydantic.Field(ge=1)
    action_components: tuple[GR00TN17ActionComponentConfig, ...] = pydantic.Field(
        min_length=1
    )
    model_action_horizon: int = pydantic.Field(ge=1)
    disable_dropout: Literal[True]
    compile_model: Literal[False]
    gradient_checkpointing: Literal[False]
    train_expert_only: Literal[True]
    sft_replay: GR00TN17SFTReplayConfig | None = None
    gradient_aggregation: Literal["sum", "task_pcgrad"] = "sum"

    @pydantic.model_validator(mode="after")
    def validate_action_layout(self) -> "GR00TN17FlowLoadConfig":
        keys = [component.key for component in self.action_components]
        if len(keys) != len(set(keys)):
            raise ValueError("GR00T N1.7 action component keys must be unique")
        total = sum(component.size for component in self.action_components)
        if total != self.action_dim:
            raise ValueError(
                "GR00T N1.7 action_components must cover action_dim exactly: "
                f"components={total}, action_dim={self.action_dim}"
            )
        executed = sum(
            component.size for component in self.action_components if component.executed
        )
        if executed != self.execution_action_dim:
            raise ValueError(
                "GR00T N1.7 executed action_components must cover "
                "execution_action_dim exactly: "
                f"components={executed}, execution_action_dim={self.execution_action_dim}"
            )
        execution_orders = []
        for component in self.action_components:
            if component.executed and component.execution_order is None:
                raise ValueError(
                    f"Executed action component {component.key!r} needs execution_order"
                )
            if not component.executed and component.execution_order is not None:
                raise ValueError(
                    f"Non-executed action component {component.key!r} cannot set "
                    "execution_order"
                )
            if component.execution_order is not None:
                execution_orders.append(component.execution_order)
        if sorted(execution_orders) != list(range(len(execution_orders))):
            raise ValueError(
                "GR00T N1.7 execution_order values must be contiguous from zero"
            )
        return self


class PolicyConfig(_StrictConfig):
    """Policy source, generation contracts, and trainable parameter surface."""

    type: str = pydantic.Field(min_length=1)
    path: str = pydantic.Field(min_length=1)
    revision: str | None
    device: str = pydantic.Field(min_length=1)
    dtype: str = pydantic.Field(min_length=1)
    trust_remote_code: bool
    unnorm_key: str | None
    # Built-in policy factories validate this mapping against their own strict
    # schema. Keeping the generic experiment contract open lets users inject a
    # custom LeRobot policy without changing ART's core config model.
    load_kwargs: dict[str, Any]
    rollout_generation: GenerationConfig
    train_generation: GenerationConfig
    evaluation_generation: GenerationConfig
    trainable_parameter_strategy: str = pydantic.Field(min_length=1)
    force_trainable_float32: bool
    lora: LoraConfig


class EnvironmentConfig(_StrictConfig):
    """User-owned environment factory and its serialized adapter contracts."""

    type: str = pydantic.Field(min_length=1)
    task: str = pydantic.Field(min_length=1)
    robot_type: str | None
    kwargs: dict[str, Any]
    reset: dict[str, Any]
    observation_processor: dict[str, Any]
    action_processor: dict[str, Any]


class RewardConfig(_StrictConfig):
    """Reward source and scaling applied before advantage construction."""

    type: Literal["environment", "custom", "reward_model"]
    name: str = pydantic.Field(min_length=1)
    terminal_only: bool
    scale: float
    kwargs: dict[str, Any]


class AdvantageConfig(_StrictConfig):
    """Normalization and masking rules for group-relative advantages."""

    scope: Literal["group", "global"]
    normalize: bool
    extra_global_normalization: bool
    mask_zero_variance_groups: bool
    std_unbiased: bool
    epsilon: float = pydantic.Field(gt=0.0)


class FlowSDEConfig(_StrictConfig):
    """Sampler probability contract for PI-style flow policies."""

    noise_level: float = pydantic.Field(gt=0.0)
    num_denoise_steps: int = pydantic.Field(ge=1)
    stochastic_transitions_per_sample: Literal[1]
    selected_step_sampling: Literal["uniform"]
    joint_logprob: Literal[False]


class AlgorithmConfig(_StrictConfig):
    """Executed GRPO or GSPO objective geometry, not merely an experiment label."""

    type: Literal["grpo", "gspo"]
    flow_sde: FlowSDEConfig | None
    group_size: int = pydantic.Field(ge=2)
    clip_epsilon_low: float = pydantic.Field(ge=0.0)
    clip_epsilon_high: float = pydantic.Field(ge=0.0)
    kl_coefficient: float = pydantic.Field(ge=0.0)
    clip_ratio_c: float | None = pydantic.Field(default=None, gt=1.0)
    importance_sampling_level: Literal["token", "sequence", "action_chunk"]
    training_unit: Literal["action", "trajectory"]
    action_advantage_mode: Literal["example", "rlinf_action_level_cumulative"]
    score_source: Literal["trajectory_reward", "chunk_rewards"]
    pad_fixed_horizon_examples: bool
    filter_rewards: bool
    reward_filter_mode: Literal["drop_examples", "loss_mask"]
    rewards_lower_bound: float | None
    rewards_upper_bound: float | None
    loss_aggregation: Literal[
        "token_mean",
        "trajectory_mean",
        "seq_mean_token_sum",
        "task_balanced_trajectory_mean",
        "rlinf_chunk_mean",
        "rlinf_masked_mean_ratio",
    ]
    advantage: AdvantageConfig
    precalculate_logprobs: bool = False
    rollout_logprob_source: Literal["rollout_action", "recomputed_current_policy"] = (
        "rollout_action"
    )
    logprob_eval_mode: bool
    logprob_microbatch_size: int | None = pydantic.Field(default=None, ge=1)
    train_logprob_microbatch_size: int | None = pydantic.Field(default=None, ge=1)
    pre_update_logprob_kl_tolerance: float | None = pydantic.Field(default=None, ge=0.0)
    pre_update_ratio_tolerance: float | None = pydantic.Field(default=None, ge=0.0)
    skip_optimizer_step_without_policy_gradient_signal: bool

    @pydantic.model_validator(mode="after")
    def validate_algorithm_contract(self) -> "AlgorithmConfig":
        if self.importance_sampling_level == "action_chunk" and (
            self.type != "grpo"
            or self.flow_sde is not None
            or self.training_unit != "action"
            or self.action_advantage_mode != "example"
            or self.score_source != "trajectory_reward"
            or self.loss_aggregation != "seq_mean_token_sum"
            or self.pad_fixed_horizon_examples
            or self.filter_rewards
            or self.kl_coefficient != 0
        ):
            raise ValueError(
                "action_chunk requires action-token GRPO, action examples, scalar "
                "trajectory_reward advantages, seq_mean_token_sum, and no padding, "
                "reward filters, or KL penalty"
            )
        if self.loss_aggregation == "task_balanced_trajectory_mean" and (
            self.type != "grpo"
            or self.importance_sampling_level != "token"
            or self.training_unit != "action"
        ):
            raise ValueError(
                "task_balanced_trajectory_mean currently requires action-level "
                "token-importance GRPO"
            )
        if self.type == "gspo":
            if not self.precalculate_logprobs:
                raise ValueError(
                    "GSPO requires precalculate_logprobs=true so old logprobs "
                    "use the same scorer geometry as training"
                )
            if self.rollout_logprob_source != "recomputed_current_policy":
                raise ValueError(
                    "GSPO requires rollout_logprob_source='recomputed_current_policy'"
                )
            if self.importance_sampling_level != "sequence":
                raise ValueError("GSPO requires importance_sampling_level='sequence'")
            if self.training_unit != "trajectory":
                raise ValueError("GSPO requires training_unit='trajectory'")
            if self.action_advantage_mode != "example":
                raise ValueError("GSPO requires action_advantage_mode='example'")
            if self.score_source != "trajectory_reward":
                raise ValueError("GSPO requires score_source='trajectory_reward'")
            if self.loss_aggregation != "trajectory_mean":
                raise ValueError("GSPO requires loss_aggregation='trajectory_mean'")
            if self.filter_rewards:
                raise ValueError(
                    "GSPO does not yet support reward filtering; set "
                    "filter_rewards=false"
                )
            alignment_tolerances = {
                "pre_update_logprob_kl_tolerance": (
                    self.pre_update_logprob_kl_tolerance
                ),
                "pre_update_ratio_tolerance": self.pre_update_ratio_tolerance,
            }
            for name, value in alignment_tolerances.items():
                if value is None:
                    raise ValueError(f"GSPO requires {name} to be configured")
                if value > self.clip_epsilon_low:
                    raise ValueError(
                        f"GSPO requires {name} <= clip_epsilon_low so the "
                        "alignment guard is stricter than sequence clipping"
                    )
        if (
            self.precalculate_logprobs
            and self.rollout_logprob_source != "recomputed_current_policy"
        ):
            raise ValueError(
                "precalculate_logprobs=true requires rollout_logprob_source="
                "'recomputed_current_policy'"
            )
        if self.action_advantage_mode == "rlinf_action_level_cumulative":
            if self.training_unit != "action":
                raise ValueError(
                    "rlinf_action_level_cumulative requires training_unit='action'"
                )
            if self.score_source != "chunk_rewards":
                raise ValueError(
                    "rlinf_action_level_cumulative requires score_source='chunk_rewards'"
                )
        if self.filter_rewards:
            if self.rewards_lower_bound is None or self.rewards_upper_bound is None:
                raise ValueError(
                    "filter_rewards=true requires rewards_lower_bound and rewards_upper_bound"
                )
            if self.rewards_lower_bound > self.rewards_upper_bound:
                raise ValueError("rewards_lower_bound must be <= rewards_upper_bound")
        return self


class ActionPayloadConfig(_StrictConfig):
    """Evidence each rollout action must retain for the selected objective."""

    kind: Literal["token", "continuous"]
    require_old_logprobs: bool
    require_prompt: bool
    require_observation: bool
    trainable_action_selection: Literal["all", "uniform_grid"] = "all"
    max_trainable_actions_per_trajectory: int | None = pydantic.Field(
        default=None, ge=1
    )

    @pydantic.model_validator(mode="after")
    def validate_trainable_action_selection(self) -> "ActionPayloadConfig":
        if self.trainable_action_selection == "all":
            if self.max_trainable_actions_per_trajectory is not None:
                raise ValueError(
                    "max_trainable_actions_per_trajectory requires "
                    "trainable_action_selection='uniform_grid'"
                )
        elif self.max_trainable_actions_per_trajectory is None:
            raise ValueError(
                "trainable_action_selection='uniform_grid' requires "
                "max_trainable_actions_per_trajectory"
            )
        return self


class RolloutConfig(_StrictConfig):
    """Sampling volume, concurrency, horizon, and failure semantics per update."""

    groups_per_update: int = pydantic.Field(ge=1)
    epochs_per_update: int = pydantic.Field(ge=1)
    workers: int = pydantic.Field(ge=1)
    failure_policy: Literal["fail_update", "keep_partial"]
    minimum_completed_attempts_per_group: int = pydantic.Field(ge=2)
    # Environment horizon and policy-decision horizon differ for policies that
    # emit action chunks. Both are explicit so one cannot silently be used as
    # the other by rollout and training code.
    max_episode_steps: int = pydantic.Field(ge=1)
    max_policy_steps: int = pydantic.Field(ge=1)
    # During training only, execute this many leader-sampled action chunks in
    # every same-reset group before independently sampling each replica.
    shared_prefix_action_chunks: int = pydantic.Field(default=0, ge=0)
    temperature: float = pydantic.Field(gt=0.0)
    deterministic: bool
    action_payload: ActionPayloadConfig

    @pydantic.model_validator(mode="after")
    def validate_shared_prefix_horizon(self) -> "RolloutConfig":
        if self.shared_prefix_action_chunks >= self.max_policy_steps:
            raise ValueError(
                "rollout.shared_prefix_action_chunks must be smaller than "
                "rollout.max_policy_steps so every training trajectory has an "
                "independently sampled suffix"
            )
        return self

    def trajectories_per_update(self, *, group_size: int) -> int:
        return group_size * self.groups_per_update * self.epochs_per_update


class OptimizerConfig(_StrictConfig):
    """Optimizer hyperparameters consumed by the built-in training backend."""

    # The built-in action-token backend currently owns AdamW state and resume
    # semantics. Do not advertise optimizers that the runtime silently ignores.
    type: Literal["adamw"]
    learning_rate: float = pydantic.Field(gt=0.0)
    weight_decay: float = pydantic.Field(ge=0.0)
    beta1: float = pydantic.Field(gt=0.0, lt=1.0)
    beta2: float = pydantic.Field(gt=0.0, lt=1.0)
    epsilon: float = pydantic.Field(gt=0.0)
    max_grad_norm: float | None = pydantic.Field(default=None, gt=0.0)


class FullUpdateScheduleConfig(_StrictConfig):
    """Repeated optimizer epochs over the complete prepared rollout update."""

    type: Literal["full_update"]
    update_epochs: int = pydantic.Field(default=1, ge=1)


class TrajectoryMinibatchScheduleConfig(_StrictConfig):
    """Shuffle complete trajectories into sequential policy subupdates."""

    type: Literal["trajectory_minibatch"]
    minibatch_trajectories: int = pydantic.Field(ge=1)
    shuffle_seed: int


class RlinfActorBatchScheduleConfig(_StrictConfig):
    """Executed RLinf actor-batch geometry, not merely documentation.

    ``actor_world_size`` describes the rollout tensor topology reconstructed
    before shuffling. It is independent of the number of GPUs used to execute
    the optimizer step.
    """

    type: Literal["rlinf_actor_global_batch"]
    global_batch_size: int = pydantic.Field(ge=1)
    actor_seed: int
    actor_world_size: int = pydantic.Field(ge=1)
    rank_local_shuffle: bool
    groups_per_process_per_rollout_epoch: int = pydantic.Field(ge=1)
    action_chunk_size: int = pydantic.Field(ge=1)
    update_epochs: int = pydantic.Field(default=1, ge=1)
    strict_geometry: bool
    pre_update_alignment_guard: Literal["first_subupdate", "disabled"]
    max_approximate_kl: float | None = pydantic.Field(default=None, gt=0.0)
    approximate_kl_guard_scope: Literal["joint_chunk", "primitive_action"] = (
        "joint_chunk"
    )
    min_optimizer_steps_before_kl_stop: int = pydantic.Field(default=1, ge=1)


class TrainingConfig(_StrictConfig):
    """Optimizer schedule, checkpoint cadence, and microbatch execution shape."""

    updates: int = pydantic.Field(ge=1)
    optimizer_steps_per_update: int = pydantic.Field(ge=1)
    microbatch_size: int = pydantic.Field(ge=1)
    log_action_token_progress: bool
    action_token_progress_every_microbatches: int = pydantic.Field(ge=1)
    schedule: Annotated[
        FullUpdateScheduleConfig
        | TrajectoryMinibatchScheduleConfig
        | RlinfActorBatchScheduleConfig,
        pydantic.Field(discriminator="type"),
    ]
    checkpoint_every_updates: int = pydantic.Field(ge=1)
    optimizer: OptimizerConfig

    @pydantic.model_validator(mode="after")
    def validate_schedule(self) -> "TrainingConfig":
        if isinstance(self.schedule, FullUpdateScheduleConfig) and (
            self.optimizer_steps_per_update != self.schedule.update_epochs
        ):
            raise ValueError(
                "training.schedule.type='full_update' requires "
                "optimizer_steps_per_update == schedule.update_epochs; choose "
                "trajectory_minibatch for trajectory-reward updates or rlinf_actor_global_batch "
                "for action-level GRPO subupdates"
            )
        return self


class PreTrainingSuccessGateConfig(_StrictConfig):
    """Require an informative initialized policy before optimizer work begins."""

    minimum_success_rate: float = pydantic.Field(ge=0.0, le=1.0)
    maximum_success_rate: float = pydantic.Field(ge=0.0, le=1.0)

    @pydantic.model_validator(mode="after")
    def require_nonempty_interval(self) -> "PreTrainingSuccessGateConfig":
        if self.minimum_success_rate >= self.maximum_success_rate:
            raise ValueError(
                "minimum_success_rate must be smaller than maximum_success_rate"
            )
        return self


class EvaluationConfig(_StrictConfig):
    """Fixed evaluation population, data role, cadence, and selection policy."""

    enabled: bool
    evaluate_before_training: bool = False
    evaluate_after_first_update: bool = False
    pre_training_success_gate: PreTrainingSuccessGateConfig | None = None
    split: Literal["train_matched", "held_out"]
    data_role: Literal["diagnostic", "development", "sealed_test"]
    runtime: Literal["native_lerobot"]
    baseline_outcomes_path: Path | None
    baseline_wait_timeout_seconds: int = pydantic.Field(default=0, ge=0)
    every_updates: int = pydantic.Field(ge=1)
    episodes: int = pydantic.Field(ge=1)
    seeds: list[int]
    deterministic: bool
    temperature: float = pydantic.Field(gt=0.0)
    fixed_scenarios: list[str]
    checkpoint_selection: Literal["evaluation_success", "last"]
    kwargs: dict[str, Any]

    @pydantic.model_validator(mode="after")
    def require_fixed_evaluation_contract(self) -> "EvaluationConfig":
        if self.evaluate_after_first_update and not self.enabled:
            raise ValueError(
                "evaluation.evaluate_after_first_update requires evaluation.enabled=true"
            )
        if self.evaluate_before_training and not self.enabled:
            raise ValueError(
                "evaluation.evaluate_before_training requires evaluation.enabled=true"
            )
        if (
            self.pre_training_success_gate is not None
            and not self.evaluate_before_training
        ):
            raise ValueError(
                "evaluation.pre_training_success_gate requires "
                "evaluation.evaluate_before_training=true"
            )
        if not self.enabled and self.baseline_outcomes_path is not None:
            raise ValueError(
                "evaluation.baseline_outcomes_path requires evaluation.enabled=true"
            )
        if self.enabled and not self.seeds:
            raise ValueError(
                "evaluation.seeds cannot be empty when evaluation is enabled"
            )
        if self.enabled and not self.fixed_scenarios:
            raise ValueError(
                "evaluation.fixed_scenarios cannot be empty when evaluation is enabled"
            )
        if self.split == "train_matched" and self.data_role != "diagnostic":
            raise ValueError(
                "evaluation.split='train_matched' is diagnostic data and requires "
                "evaluation.data_role='diagnostic'"
            )
        if self.split == "held_out" and self.data_role == "diagnostic":
            raise ValueError(
                "evaluation.split='held_out' requires evaluation.data_role to be "
                "'development' or 'sealed_test'"
            )
        if self.data_role == "sealed_test" and self.checkpoint_selection != "last":
            raise ValueError(
                "sealed evaluation cannot select a checkpoint from its own outcomes; "
                "set evaluation.checkpoint_selection='last' after freezing the "
                "method and checkpoint selection rule"
            )
        return self


class WandbConfig(_StrictConfig):
    """W&B run ownership and bounded logging behavior for one coordinator."""

    enabled: bool
    entity: str | None
    project: str = pydantic.Field(min_length=1)
    connection: Literal["primary", "shared_primary", "shared_worker", "resume"] = (
        "primary"
    )
    run_id: str | None = None
    resume: Literal["allow", "must", "never", "auto"] | None = None
    writer_label: str | None = None
    console_multipart: bool = True
    console_chunk_max_bytes: int = pydantic.Field(default=1_048_576, ge=0)
    console_chunk_max_seconds: int = pydantic.Field(default=60, ge=0)
    log_system_metrics: bool = True
    native_update_steps: bool = False
    save_code: bool = False
    group: str | None
    job_type: str = pydantic.Field(min_length=1)
    mode: Literal["online", "offline", "disabled"]
    log_model_artifacts: bool
    log_input_model_artifact: bool = False
    input_model_artifact_ref: str | None = pydantic.Field(
        default=None,
        min_length=1,
    )
    log_evaluation_artifacts: bool
    log_evaluation_table: bool
    max_evaluation_table_rows: int = pydantic.Field(ge=0)
    log_rollout_progress: bool
    rollout_progress_every_groups: int = pydantic.Field(ge=1)
    log_evaluation_progress: bool
    evaluation_progress_every_episodes: int = pydantic.Field(ge=1)

    @pydantic.model_validator(mode="after")
    def validate_run_attachment(self) -> "WandbConfig":
        if self.native_update_steps and self.connection.startswith("shared_"):
            raise ValueError("native_update_steps requires a single W&B history writer")
        if self.log_input_model_artifact and not self.enabled:
            raise ValueError(
                "wandb.log_input_model_artifact=true requires wandb.enabled=true"
            )
        if self.input_model_artifact_ref is not None:
            if not self.log_input_model_artifact:
                raise ValueError(
                    "wandb.input_model_artifact_ref requires "
                    "log_input_model_artifact=true"
                )
            version = self.input_model_artifact_ref.rpartition(":")[2]
            if not version.startswith("v") or not version[1:].isdigit():
                raise ValueError(
                    "wandb.input_model_artifact_ref must pin an immutable :vN version"
                )
        if self.connection == "resume":
            if self.run_id is None or self.resume is None:
                raise ValueError(
                    "wandb connection='resume' requires both run_id and resume; "
                    "this mode is reserved for recovery of an interrupted W&B writer"
                )
        elif self.resume is not None:
            raise ValueError(
                "wandb.resume is only valid with connection='resume'; use "
                "connection='primary' for a new coordinator. Evaluation must be "
                "logged by that coordinator rather than attached post hoc."
            )
        if self.connection == "primary" and self.run_id is not None:
            raise ValueError(
                "wandb primary creates a new coordinator run and cannot set run_id; "
                "use connection='resume' only to recover an interrupted writer"
            )
        if self.connection == "shared_worker" and self.run_id is None:
            raise ValueError("wandb shared_worker requires an existing run_id")
        if self.mode != "online" and (
            self.connection not in {"primary", "shared_primary"}
            or self.run_id is not None
        ):
            raise ValueError(
                "W&B run attachment requires mode='online'; offline/disabled mode "
                "can only create a local primary run without run_id"
            )
        if self.console_multipart and (
            self.console_chunk_max_bytes == 0 and self.console_chunk_max_seconds == 0
        ):
            raise ValueError(
                "multipart console logging requires a byte or time rollover limit "
                "so logs remain visible while a long run is active"
            )
        return self


class WeaveConfig(_StrictConfig):
    """Weave trace sampling and optional local cache bounds."""

    enabled: bool
    project: str = pydantic.Field(min_length=1)
    trace_trajectories: bool
    max_groups_per_update: int = pydantic.Field(ge=0)
    max_trajectories_per_group: int = pydantic.Field(ge=0)
    max_evaluation_trajectories: int = pydantic.Field(ge=0)
    use_server_cache: bool
    server_cache_dir: Path | None
    server_cache_size_mb: int = pydantic.Field(ge=0)

    @pydantic.model_validator(mode="after")
    def validate_server_cache(self) -> "WeaveConfig":
        if self.use_server_cache:
            if self.server_cache_dir is None:
                raise ValueError(
                    "weave.server_cache_dir is required when "
                    "weave.use_server_cache=true"
                )
            if self.server_cache_size_mb < 1:
                raise ValueError(
                    "weave.server_cache_size_mb must be positive when "
                    "weave.use_server_cache=true"
                )
        return self


class LookaheadPreviewConfig(_StrictConfig):
    """Optional filmstrip rendering of an action chunk's unused tail."""

    enabled: bool = False
    videos_per_update: int = pydantic.Field(default=1, ge=0)
    videos_per_evaluation: int = pydantic.Field(default=1, ge=0)
    max_future_frames: int = pydantic.Field(default=10, ge=1)
    future_stride: int = pydantic.Field(default=4, ge=1)
    max_panels: int = pydantic.Field(default=4, ge=1, le=6)


class ObservabilityConfig(_StrictConfig):
    """Failure policy and budgets shared by metrics, traces, and media."""

    delivery_failure_policy: Literal["best_effort", "fail_run"] = "best_effort"
    wandb: WandbConfig
    weave: WeaveConfig
    videos_per_update: int = pydantic.Field(ge=0)
    videos_per_evaluation: int = pydantic.Field(ge=0)
    require_train_video: bool
    require_evaluation_video: bool
    video_fps: int = pydantic.Field(ge=1)
    lookahead_preview: LookaheadPreviewConfig = pydantic.Field(
        default_factory=LookaheadPreviewConfig
    )


class StorageConfig(_StrictConfig):
    """Checkpoint retention, resume provenance, and local storage bounds."""

    output_dir: Path
    keep_last_checkpoints: int = pydantic.Field(ge=1)
    retain_checkpoint_updates: list[int] = pydantic.Field(default_factory=list)
    save_training_state: bool = True
    resume_from_checkpoint: Path | None = None
    allow_legacy_checkpoint_resume: bool = False
    max_log_file_mb: int = pydantic.Field(ge=1)
    retain_rollout_payloads: bool

    @pydantic.field_validator("retain_checkpoint_updates")
    @classmethod
    def validate_retained_checkpoint_updates(cls, value: list[int]) -> list[int]:
        if any(int(step) < 1 for step in value):
            raise ValueError("retain_checkpoint_updates must contain positive updates")
        if len(value) != len(set(value)):
            raise ValueError("retain_checkpoint_updates must be unique")
        return sorted(int(step) for step in value)


class RolloutExecutionConfig(_StrictConfig):
    """How episode rollouts execute independently of resource allocation."""

    mode: Literal["in_process", "local_process"]
    actor_factory: str | None
    actor_kwargs: dict[str, Any]
    actors_per_device: int = pydantic.Field(ge=1)
    group_batching: bool
    lifecycle: Literal["persistent", "per_update", "cpu_offload"]
    policy_sync: Literal["shared", "checkpoint"]
    inference_mode: Literal["embedded", "batched_server"]
    inference_factory: str | None
    inference_replicas_per_device: int = pydantic.Field(ge=1)
    inference_max_batch_size: int = pydantic.Field(ge=1)
    inference_max_wait_ms: float = pydantic.Field(ge=0.0)
    startup_timeout_seconds: int = pydantic.Field(ge=1)
    request_timeout_seconds: int = pydantic.Field(ge=1)
    actor_python_executable: str | None = None
    inference_python_executable: str | None = None

    @pydantic.field_validator(
        "actor_python_executable",
        "inference_python_executable",
    )
    @classmethod
    def normalize_python_executable(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("rollout Python executables cannot be empty")
        return normalized

    @pydantic.model_validator(mode="after")
    def validate_execution_mode(self) -> "RolloutExecutionConfig":
        if self.mode == "in_process":
            if (
                self.actor_python_executable is not None
                or self.inference_python_executable is not None
            ):
                raise ValueError(
                    "in-process rollout cannot configure child-process Python "
                    "executables"
                )
            if self.group_batching:
                raise ValueError(
                    "runtime.rollout_execution.group_batching must be false for "
                    "mode='in_process'"
                )
            if self.actor_factory is not None:
                raise ValueError(
                    "runtime.rollout_execution.actor_factory must be null for "
                    "mode='in_process'"
                )
            if self.actor_kwargs:
                raise ValueError(
                    "runtime.rollout_execution.actor_kwargs must be empty for "
                    "mode='in_process'"
                )
            if self.lifecycle != "per_update":
                raise ValueError(
                    "runtime.rollout_execution.lifecycle must be 'per_update' "
                    "for mode='in_process'"
                )
            if self.policy_sync != "shared":
                raise ValueError(
                    "runtime.rollout_execution.policy_sync must be 'shared' "
                    "for mode='in_process'"
                )
            if self.inference_mode != "embedded" or self.inference_factory is not None:
                raise ValueError(
                    "in-process rollout requires inference_mode='embedded' "
                    "and inference_factory=null"
                )
            if self.inference_replicas_per_device != 1:
                raise ValueError(
                    "in-process rollout requires inference_replicas_per_device=1"
                )
            return self

        if not self.actor_factory:
            raise ValueError(
                "runtime.rollout_execution.actor_factory is required for "
                "mode='local_process'"
            )
        if ":" not in self.actor_factory:
            raise ValueError(
                "runtime.rollout_execution.actor_factory must use the "
                "'module:callable' form"
            )
        if self.policy_sync != "checkpoint":
            raise ValueError(
                "runtime.rollout_execution.policy_sync must be 'checkpoint' "
                "for mode='local_process'"
            )
        if self.inference_mode == "embedded":
            if self.inference_factory is not None:
                raise ValueError(
                    "embedded local-process inference requires inference_factory=null"
                )
            if self.inference_replicas_per_device != 1:
                raise ValueError(
                    "embedded local-process inference requires "
                    "inference_replicas_per_device=1"
                )
            if self.inference_python_executable is not None:
                raise ValueError(
                    "embedded local-process inference cannot configure "
                    "inference_python_executable"
                )
        else:
            if not self.inference_factory or ":" not in self.inference_factory:
                raise ValueError(
                    "batched-server inference requires inference_factory in "
                    "the 'module:callable' form"
                )
            if self.inference_replicas_per_device > self.actors_per_device:
                raise ValueError(
                    "batched-server inference_replicas_per_device cannot exceed "
                    "actors_per_device"
                )
        return self


class RuntimeConfig(_StrictConfig):
    """Device allocation and worker lifecycle for the local backend."""

    backend: Literal["local"]
    rollout_devices: list[str]
    training_devices: list[str]
    worker_python_executable: str | None = None
    rollout_execution: RolloutExecutionConfig
    distributed_training: bool
    training_worker_lifecycle: Literal["per_update", "cpu_offload"] = "per_update"
    worker_handoff_dir: Path
    worker_timeout_seconds: int = pydantic.Field(ge=1)
    max_worker_handoff_mb: int = pydantic.Field(ge=1)
    keep_worker_handoffs: bool

    @pydantic.field_validator("worker_python_executable")
    @classmethod
    def normalize_worker_python_executable(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("runtime.worker_python_executable cannot be empty")
        return normalized

    @pydantic.model_validator(mode="after")
    def require_devices(self) -> "RuntimeConfig":
        if not self.rollout_devices:
            raise ValueError("runtime.rollout_devices cannot be empty")
        if not self.training_devices:
            raise ValueError("runtime.training_devices cannot be empty")
        if self.distributed_training and len(self.training_devices) < 2:
            raise ValueError(
                "runtime.distributed_training=true requires at least two "
                "runtime.training_devices"
            )
        if not self.distributed_training and len(self.training_devices) != 1:
            raise ValueError(
                "runtime.distributed_training=false requires exactly one "
                "runtime.training_devices entry; extra devices would be ignored"
            )
        return self


class EmbodiedExperimentConfig(_StrictConfig):
    """Complete immutable experiment contract loaded from one resolved YAML."""

    schema_version: Literal[1]
    experiment: ExperimentIdentity
    policy: PolicyConfig
    environment: EnvironmentConfig
    reward: RewardConfig
    algorithm: AlgorithmConfig
    rollout: RolloutConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    observability: ObservabilityConfig
    storage: StorageConfig
    runtime: RuntimeConfig

    @pydantic.model_validator(mode="after")
    def validate_executed_training_geometry(
        self,
        info: pydantic.ValidationInfo,
    ) -> "EmbodiedExperimentConfig":
        observability = self.observability
        wandb = observability.wandb
        process_role = (
            info.context.get("art_embodied_process_role")
            if isinstance(info.context, dict)
            else None
        )
        if (
            self.policy.lora.rank_partition is not None
            and not self.runtime.distributed_training
            and process_role != "distributed_gradient_worker"
        ):
            raise ValueError(
                "Experimental task-partitioned LoRA currently requires "
                "runtime.distributed_training=true so task-routed worker gradients "
                "and coordinator optimizer ownership remain explicit"
            )
        rank_partition = self.policy.lora.rank_partition
        if rank_partition is not None and rank_partition.warm_start_sources is not None:
            source_kinds = {
                source.source_kind
                for source in rank_partition.warm_start_sources.values()
            }
            expected_kind = {
                "gr00t_n1d7": "validated_grpo",
                "pi0_fast": "partition_sft",
            }.get(self.policy.type)
            if expected_kind is None or source_kinds != {expected_kind}:
                raise ValueError(
                    "Task-partitioned warm-start source kind does not match the "
                    f"policy: policy={self.policy.type!r}, sources={sorted(source_kinds)}"
                )
        if (
            wandb.enabled
            and self.evaluation.enabled
            and wandb.connection in {"shared_primary", "shared_worker"}
        ):
            raise ValueError(
                "W&B shared mode cannot satisfy ART-Embodied's native-step "
                "evaluation contract: pre-training validation must share Step 0 "
                "with the first train rollout, later validation must share the "
                "corresponding policy-version step, and evaluation Artifacts must "
                "remain uploadable. Use connection='primary' so the coordinator "
                "owns both training and evaluation logging, or connection='resume' "
                "only when recovering that same interrupted coordinator."
            )
        if self.policy.type in {"pi0", "pi05"}:
            self._validate_pi_flow_contract()
        elif self.policy.type == "pi0_fast":
            self._validate_pi0_fast_contract()
        elif self.policy.type == "smolvla":
            self._validate_smolvla_flow_contract()
        elif self.policy.type == "gr00t_n1d5":
            self._validate_gr00t_n1d5_flow_contract()
        elif self.policy.type == "gr00t_n1d7":
            self._validate_gr00t_n1d7_flow_contract()
        self._validate_sft_replay_contract(process_role=process_role)
        self._validate_task_gradient_aggregation_contract(process_role=process_role)
        if (
            self.rollout.action_payload.kind == "token"
            and not self.rollout.action_payload.require_old_logprobs
        ):
            raise ValueError(
                "Built-in action-token GRPO/GSPO requires "
                "rollout.action_payload.require_old_logprobs=true"
            )
        configured_train_microbatch = self.algorithm.train_logprob_microbatch_size
        if (
            configured_train_microbatch is not None
            and configured_train_microbatch != self.training.microbatch_size
        ):
            raise ValueError(
                "training.microbatch_size and "
                "algorithm.train_logprob_microbatch_size must match; the former "
                "is the canonical optimizer/scorer microbatch and the latter is "
                "a schema-v1 compatibility assertion"
            )
        resume_checkpoint = self.storage.resume_from_checkpoint
        if resume_checkpoint is not None:
            if self.evaluation.evaluate_before_training:
                raise ValueError(
                    "evaluation.evaluate_before_training cannot be used while "
                    "resuming a trained checkpoint; the resumed policy is not "
                    "the SFT policy version zero"
                )
            measured_baseline = (
                self.storage.output_dir
                / "evaluation"
                / "update_000000_episode_outcomes.json"
            )
            if (
                self.evaluation.enabled
                and self.evaluation.baseline_outcomes_path is None
                and measured_baseline.is_file()
            ):
                raise ValueError(
                    "checkpoint resume found a measured Step-0 baseline but "
                    "evaluation.baseline_outcomes_path is unset; reference "
                    f"{measured_baseline} so periodic evaluation remains paired"
                )
            if (
                wandb.enabled
                and wandb.mode == "online"
                and not (
                    wandb.connection == "resume"
                    and wandb.run_id is not None
                    and wandb.resume == "must"
                )
            ):
                raise ValueError(
                    "online W&B checkpoint recovery must attach to the same run: "
                    "set observability.wandb.connection='resume', an explicit "
                    "run_id, and resume='must'. Start a separate experiment "
                    "instead of silently fragmenting one learning curve."
                )
            if not self.runtime.distributed_training:
                raise ValueError(
                    "storage.resume_from_checkpoint currently requires "
                    "runtime.distributed_training=true"
                )
            if not resume_checkpoint.is_dir():
                raise ValueError(
                    "storage.resume_from_checkpoint must be an existing checkpoint "
                    f"directory: {resume_checkpoint}"
                )
            if not (resume_checkpoint / "art_embodied_training_state.pt").is_file():
                raise ValueError(
                    "storage.resume_from_checkpoint is missing "
                    "art_embodied_training_state.pt"
                )
        if observability.require_train_video:
            if not observability.wandb.enabled:
                raise ValueError(
                    "observability.require_train_video=true requires W&B logging"
                )
            if observability.videos_per_update < 1:
                raise ValueError(
                    "observability.require_train_video=true requires "
                    "videos_per_update >= 1"
                )
        if observability.require_evaluation_video:
            if not observability.wandb.enabled:
                raise ValueError(
                    "observability.require_evaluation_video=true requires W&B logging"
                )
            if not self.evaluation.enabled:
                raise ValueError(
                    "observability.require_evaluation_video=true requires "
                    "evaluation.enabled=true"
                )
            if observability.videos_per_evaluation < 1:
                raise ValueError(
                    "observability.require_evaluation_video=true requires "
                    "videos_per_evaluation >= 1"
                )
        if self.rollout.action_payload.kind == "token":
            self._validate_action_token_generation_contract()
        if (
            self.rollout.minimum_completed_attempts_per_group
            > self.algorithm.group_size
        ):
            raise ValueError(
                "rollout.minimum_completed_attempts_per_group cannot exceed "
                "algorithm.group_size"
            )
        rollout_execution = self.runtime.rollout_execution
        if self.rollout.shared_prefix_action_chunks:
            if self.environment.type != "robocasa_gr1_tabletop":
                raise ValueError(
                    "rollout.shared_prefix_action_chunks currently requires "
                    "environment.type='robocasa_gr1_tabletop'"
                )
            if self.policy.type != "gr00t_n1d7" or self.algorithm.type != "grpo":
                raise ValueError(
                    "rollout.shared_prefix_action_chunks currently requires "
                    "GR00T N1.7 GRPO"
                )
            if self.algorithm.group_size < 2 or not rollout_execution.group_batching:
                raise ValueError(
                    "rollout.shared_prefix_action_chunks requires group_size >= 2 "
                    "and runtime.rollout_execution.group_batching=true"
                )
        if (
            rollout_execution.group_batching
            and self.rollout.failure_policy != "fail_update"
        ):
            raise ValueError(
                "runtime.rollout_execution.group_batching=true requires "
                "rollout.failure_policy='fail_update' because one group-native "
                "actor request is atomic"
            )
        if rollout_execution.mode == "local_process":
            expected_workers = (
                len(self.runtime.rollout_devices) * rollout_execution.actors_per_device
            )
            if self.rollout.workers != expected_workers:
                raise ValueError(
                    "local-process rollout requires rollout.workers to equal "
                    "len(runtime.rollout_devices) * "
                    "runtime.rollout_execution.actors_per_device: "
                    f"workers={self.rollout.workers}, expected={expected_workers}"
                )
            if rollout_execution.lifecycle == "persistent":
                overlap = set(self.runtime.rollout_devices).intersection(
                    self.runtime.training_devices
                )
                if overlap:
                    raise ValueError(
                        "persistent rollout actors must use devices disjoint "
                        "from runtime.training_devices; use lifecycle='per_update' "
                        "to time-share devices: " + ", ".join(sorted(overlap))
                    )
        schedule = self.training.schedule
        if (
            isinstance(schedule, RlinfActorBatchScheduleConfig)
            and schedule.strict_geometry
            and self.rollout.action_payload.trainable_action_selection != "all"
        ):
            raise ValueError(
                "strict RLinf actor-batch geometry requires every action row; "
                "set rollout.action_payload.trainable_action_selection='all'"
            )
        uses_rlinf_fixed_horizon = isinstance(schedule, RlinfActorBatchScheduleConfig)
        if self.algorithm.pad_fixed_horizon_examples != uses_rlinf_fixed_horizon:
            expected = str(uses_rlinf_fixed_horizon).lower()
            raise ValueError(
                "algorithm.pad_fixed_horizon_examples must be "
                f"{expected} for training.schedule.type={schedule.type!r}; "
                "the value is an executed-schedule assertion, not an "
                "independent padding switch"
            )
        if self.algorithm.type == "gspo" and not isinstance(
            schedule,
            FullUpdateScheduleConfig | TrajectoryMinibatchScheduleConfig,
        ):
            raise ValueError(
                "GSPO requires training.schedule.type='full_update' or "
                "'trajectory_minibatch'; RLinf actor-batch subupdates produce "
                "action-level advantages and are not sequence-GSPO compatible"
            )
        if isinstance(schedule, TrajectoryMinibatchScheduleConfig):
            if self.algorithm.type == "grpo":
                if (
                    self.rollout.action_payload.kind != "token"
                    or self.algorithm.score_source != "trajectory_reward"
                    or self.algorithm.action_advantage_mode != "example"
                    or self.algorithm.loss_aggregation != "seq_mean_token_sum"
                    or self.algorithm.filter_rewards
                    or self.policy.lora.rank_partition is not None
                    or self.rollout.failure_policy != "fail_update"
                    or self.rollout.minimum_completed_attempts_per_group
                    != self.algorithm.group_size
                ):
                    raise ValueError(
                        "GRPO trajectory_minibatch requires complete token "
                        "trajectories with trajectory_reward, example advantages, "
                        "seq_mean_token_sum, no reward filtering, and no rank partition"
                    )
            if not self.runtime.distributed_training:
                raise ValueError(
                    "training.schedule.type='trajectory_minibatch' currently "
                    "requires runtime.distributed_training=true"
                )
            scheduled_trajectories = (
                self.training.optimizer_steps_per_update
                * schedule.minibatch_trajectories
            )
            if scheduled_trajectories != self.trajectories_per_update:
                raise ValueError(
                    "Trajectory minibatches must consume every rollout "
                    "trajectory exactly once: optimizer_steps_per_update * "
                    f"minibatch_trajectories={scheduled_trajectories}, "
                    f"trajectories_per_update={self.trajectories_per_update}"
                )
            return self
        if (
            isinstance(schedule, FullUpdateScheduleConfig)
            and schedule.update_epochs > 1
            and not self.runtime.distributed_training
            and process_role != "distributed_gradient_worker"
        ):
            raise ValueError(
                "full_update update_epochs > 1 currently requires "
                "runtime.distributed_training=true"
            )
        if not isinstance(schedule, RlinfActorBatchScheduleConfig):
            return self
        if not schedule.strict_geometry:
            return self

        if self.algorithm.training_unit != "action":
            raise ValueError(
                "strict RLinf actor-batch geometry requires "
                "algorithm.training_unit='action'; trajectory-level extraction "
                "does not preserve the action/time indices needed to reconstruct "
                "rank-local actor batches"
            )

        if self.rollout.failure_policy != "fail_update":
            raise ValueError(
                "strict RLinf geometry requires rollout.failure_policy='fail_update'"
            )
        if (
            self.rollout.minimum_completed_attempts_per_group
            != self.algorithm.group_size
        ):
            raise ValueError(
                "strict RLinf geometry requires "
                "rollout.minimum_completed_attempts_per_group to equal "
                "algorithm.group_size"
            )

        expected_policy_steps = (
            self.rollout.max_episode_steps + schedule.action_chunk_size - 1
        ) // schedule.action_chunk_size
        if self.rollout.max_policy_steps != expected_policy_steps:
            raise ValueError(
                "strict RLinf geometry requires rollout.max_policy_steps to "
                "equal ceil(rollout.max_episode_steps / "
                "training.schedule.action_chunk_size): "
                f"max_policy_steps={self.rollout.max_policy_steps}, "
                f"expected={expected_policy_steps}"
            )
        groups_per_rollout_epoch = (
            schedule.actor_world_size * schedule.groups_per_process_per_rollout_epoch
        )
        if groups_per_rollout_epoch != self.rollout.groups_per_update:
            raise ValueError(
                "RLinf actor topology must reconstruct one rollout epoch: "
                f"actor_world_size * groups_per_process="
                f"{groups_per_rollout_epoch}, groups_per_update="
                f"{self.rollout.groups_per_update}"
            )

        policy_steps = self.rollout.max_policy_steps
        fixed_horizon_rows = self.trajectories_per_update * policy_steps
        optimizer_rows = (
            self.training.optimizer_steps_per_update * schedule.global_batch_size
        )
        scheduled_rows = fixed_horizon_rows * schedule.update_epochs
        if scheduled_rows != optimizer_rows:
            raise ValueError(
                "RLinf fixed-horizon rows times update epochs must equal the "
                "executed optimizer batches: trajectories_per_update * "
                f"policy_steps * update_epochs={scheduled_rows}, "
                f"optimizer_steps * global_batch_size={optimizer_rows}"
            )
        return self

    def _validate_pi_flow_contract(self) -> None:
        """Fail closed when PI sampler and executed-action geometry diverge."""

        if self.algorithm.flow_sde is None:
            raise ValueError("PI0/PI0.5 training requires algorithm.flow_sde")
        if self.rollout.action_payload.kind != "continuous":
            raise ValueError("PI0/PI0.5 Flow-SDE requires a continuous action payload")

        load = PIFlowLoadConfig.model_validate(self.policy.load_kwargs)
        schedule = self.training.schedule
        if (
            isinstance(schedule, RlinfActorBatchScheduleConfig)
            and schedule.strict_geometry
            and not self.runtime.rollout_execution.group_batching
        ):
            raise ValueError(
                "strict RLinf PI Flow-SDE geometry requires "
                "runtime.rollout_execution.group_batching=true so one sampled "
                "denoise index is shared by every same-reset group"
            )
        execution_horizon = load.execution_horizon
        environment_horizon = self._environment_execution_horizon()
        if environment_horizon != execution_horizon:
            raise ValueError(
                "PI execution_horizon must equal environment.kwargs.action_chunk_size: "
                f"execution_horizon={execution_horizon}, "
                f"action_chunk_size={environment_horizon!r}"
            )
        if (
            isinstance(schedule, RlinfActorBatchScheduleConfig)
            and schedule.action_chunk_size != execution_horizon
        ):
            raise ValueError(
                "PI execution_horizon must equal training.schedule.action_chunk_size: "
                f"execution_horizon={execution_horizon}, "
                f"action_chunk_size={schedule.action_chunk_size}"
            )

        if load.model_format == "rlinf_openpi_safetensors":
            # RLinf v0.1 constructs these raw OpenPI checkpoints from Pi0Config.
            # The prediction horizon is architecture/runtime geometry, not the
            # smaller number of leading actions executed before replanning.
            expected_model_horizon = 50 if self.policy.type == "pi0" else 10
            if load.model_chunk_size != expected_model_horizon:
                raise ValueError(
                    f"RLinf v0.1 {self.policy.type} requires model_chunk_size="
                    f"{expected_model_horizon}; execution_horizon controls how many "
                    "leading actions reach the environment and must not shorten the "
                    "model prediction horizon"
                )

    def _validate_smolvla_flow_contract(self) -> None:
        """Fail closed when SmolVLA sampler and execution geometry diverge."""

        if self.algorithm.type != "grpo":
            raise ValueError("SmolVLA Flow-SDE currently supports GRPO only")
        if self.algorithm.flow_sde is None:
            raise ValueError("SmolVLA training requires algorithm.flow_sde")
        if self.rollout.action_payload.kind != "continuous":
            raise ValueError("SmolVLA Flow-SDE requires a continuous action payload")
        if self.policy.dtype != "checkpoint":
            raise ValueError(
                "SmolVLA preserves the serialized checkpoint dtype; set "
                "policy.dtype='checkpoint'"
            )
        load = SmolVLAFlowLoadConfig.model_validate(self.policy.load_kwargs)
        schedule = self.training.schedule
        if (
            isinstance(schedule, RlinfActorBatchScheduleConfig)
            and schedule.strict_geometry
            and not self.runtime.rollout_execution.group_batching
        ):
            raise ValueError(
                "strict SmolVLA Flow-SDE geometry requires "
                "runtime.rollout_execution.group_batching=true"
            )
        environment_horizon = self._environment_execution_horizon()
        if environment_horizon != load.execution_horizon:
            raise ValueError(
                "SmolVLA execution_horizon must equal "
                "environment.kwargs.action_chunk_size: "
                f"execution_horizon={load.execution_horizon}, "
                f"action_chunk_size={environment_horizon!r}"
            )
        if (
            isinstance(schedule, RlinfActorBatchScheduleConfig)
            and schedule.action_chunk_size != load.execution_horizon
        ):
            raise ValueError(
                "SmolVLA execution_horizon must equal "
                "training.schedule.action_chunk_size: "
                f"execution_horizon={load.execution_horizon}, "
                f"action_chunk_size={schedule.action_chunk_size}"
            )

    def _validate_pi0_fast_contract(self) -> None:
        """Fail closed unless rollout evidence matches pi0-FAST tokens."""

        if self.algorithm.flow_sde is not None:
            raise ValueError(
                "pi0-FAST is categorical and requires algorithm.flow_sde=null"
            )
        if self.rollout.action_payload.kind != "token":
            raise ValueError("pi0-FAST requires rollout.action_payload.kind='token'")
        if not self.rollout.action_payload.require_old_logprobs:
            raise ValueError("pi0-FAST training requires rollout old token logprobs")
        if not self.rollout.action_payload.require_observation:
            raise ValueError("pi0-FAST teacher-forced rescoring requires observations")
        if not self.rollout.action_payload.require_prompt:
            raise ValueError("pi0-FAST teacher-forced rescoring requires prompts")
        if self.policy.dtype != "checkpoint":
            raise ValueError(
                "pi0-FAST preserves checkpoint dtype; set dtype='checkpoint'"
            )
        load = PI0FastLoadConfig.model_validate(self.policy.load_kwargs)
        if (
            load.rl_token_scope == "fast_payload"
            and self.algorithm.training_unit != "action"
        ):
            raise ValueError(
                "pi0-FAST fast_payload RL token scope requires "
                "algorithm.training_unit='action'"
            )
        environment_horizon = self._environment_execution_horizon()
        if environment_horizon != load.execution_horizon:
            raise ValueError(
                "pi0-FAST execution_horizon must equal "
                "environment.kwargs.action_chunk_size: "
                f"execution_horizon={load.execution_horizon}, "
                f"action_chunk_size={environment_horizon!r}"
            )
        for name, generation in (
            ("rollout_generation", self.policy.rollout_generation),
            ("train_generation", self.policy.train_generation),
            ("evaluation_generation", self.policy.evaluation_generation),
        ):
            if generation.top_p != 1.0:
                raise ValueError(
                    f"pi0-FAST policy.{name}.top_p must be 1.0 until nucleus "
                    "sampling and its exact rescore contract are implemented"
                )

    def _validate_gr00t_n1d5_flow_contract(self) -> None:
        """Fail closed unless YAML describes the audited N1.5 Flow-SDE path."""

        if self.algorithm.type != "grpo":
            raise ValueError("GR00T N1.5 Flow-SDE currently supports GRPO only")
        if self.algorithm.flow_sde is None:
            raise ValueError("GR00T N1.5 training requires algorithm.flow_sde")
        if self.rollout.action_payload.kind != "continuous":
            raise ValueError("GR00T N1.5 Flow-SDE requires continuous actions")
        if self.policy.dtype != "bfloat16":
            raise ValueError("GR00T N1.5 parity requires policy.dtype='bfloat16'")
        load = GR00TFlowLoadConfig.model_validate(self.policy.load_kwargs)
        if load.execution_horizon > load.model_action_horizon:
            raise ValueError(
                "GR00T execution_horizon cannot exceed model_action_horizon"
            )
        environment_horizon = self._environment_execution_horizon()
        if environment_horizon != load.execution_horizon:
            raise ValueError(
                "GR00T execution_horizon must equal "
                "environment.kwargs.action_chunk_size: "
                f"execution_horizon={load.execution_horizon}, "
                f"action_chunk_size={environment_horizon!r}"
            )
        schedule = self.training.schedule
        if isinstance(schedule, RlinfActorBatchScheduleConfig):
            if schedule.action_chunk_size != load.execution_horizon:
                raise ValueError(
                    "GR00T execution_horizon must equal "
                    "training.schedule.action_chunk_size"
                )
            if (
                schedule.strict_geometry
                and not self.runtime.rollout_execution.group_batching
            ):
                raise ValueError(
                    "strict GR00T Flow-SDE geometry requires group_batching=true"
                )

    def _validate_gr00t_n1d7_flow_contract(self) -> None:
        """Fail closed unless YAML describes the audited N1.7 Flow-SDE path."""

        if self.algorithm.type != "grpo":
            raise ValueError("GR00T N1.7 Flow-SDE currently supports GRPO only")
        if self.algorithm.flow_sde is None:
            raise ValueError("GR00T N1.7 training requires algorithm.flow_sde")
        if self.rollout.action_payload.kind != "continuous":
            raise ValueError("GR00T N1.7 Flow-SDE requires continuous actions")
        if self.policy.dtype != "bfloat16":
            raise ValueError("GR00T N1.7 requires policy.dtype='bfloat16'")
        load = GR00TN17FlowLoadConfig.model_validate(self.policy.load_kwargs)
        if load.execution_horizon > load.processor_action_horizon:
            raise ValueError(
                "GR00T N1.7 execution_horizon cannot exceed processor_action_horizon"
            )
        if load.processor_action_horizon > load.model_action_horizon:
            raise ValueError(
                "GR00T N1.7 processor_action_horizon cannot exceed model_action_horizon"
            )
        environment_horizon = self._environment_execution_horizon()
        if environment_horizon != load.execution_horizon:
            raise ValueError(
                "GR00T N1.7 execution_horizon must equal "
                "environment.kwargs.action_chunk_size"
            )
        schedule = self.training.schedule
        if isinstance(schedule, RlinfActorBatchScheduleConfig):
            if schedule.action_chunk_size != load.execution_horizon:
                raise ValueError(
                    "GR00T N1.7 execution_horizon must equal "
                    "training.schedule.action_chunk_size"
                )
            if (
                schedule.strict_geometry
                and not self.runtime.rollout_execution.group_batching
            ):
                raise ValueError(
                    "strict GR00T N1.7 Flow-SDE geometry requires group_batching=true"
                )

    def _validate_sft_replay_contract(self, *, process_role: str | None) -> None:
        pi0_anchor = (
            PI0FastLoadConfig.model_validate(self.policy.load_kwargs).sft_anchor
            if self.policy.type == "pi0_fast"
            else None
        )
        if pi0_anchor is not None:
            is_gradient_worker = process_role == "distributed_gradient_worker"
            if not self.runtime.distributed_training and not is_gradient_worker:
                raise ValueError(
                    "pi0-FAST SFT anchor requires distributed gradient workers"
                )
            if self.policy.lora.rank_partition is not None:
                raise ValueError(
                    "pi0-FAST SFT anchor is incompatible with task-partitioned LoRA"
                )
            if self.algorithm.loss_aggregation != "task_balanced_trajectory_mean":
                raise ValueError(
                    "pi0-FAST SFT anchor requires task_balanced_trajectory_mean"
                )
            environment_tasks = (
                self.environment.kwargs.get("sft_anchor_task_ids")
                if self.environment.type == "libero_plus"
                else self.environment.kwargs.get("task_ids")
            )
            if environment_tasks != pi0_anchor.task_indices:
                raise ValueError(
                    "pi0-FAST SFT anchor task_indices must exactly match the ordered "
                    "environment task_ids"
                )
            if self.algorithm.kl_coefficient != 0.0:
                raise ValueError(
                    "pi0-FAST SFT anchor and reference KL cannot be enabled together"
                )
        replay = (
            GR00TN17FlowLoadConfig.model_validate(self.policy.load_kwargs).sft_replay
            if self.policy.type == "gr00t_n1d7"
            else None
        )
        if replay is None or replay.coefficient == 0.0:
            return
        if self.policy.type != "gr00t_n1d7":
            raise ValueError(
                "positive policy.load_kwargs.sft_replay requires "
                "policy.type='gr00t_n1d7'"
            )
        is_gradient_worker = process_role == "distributed_gradient_worker"
        if not self.runtime.distributed_training and not is_gradient_worker:
            raise ValueError(
                "positive policy.load_kwargs.sft_replay requires distributed "
                "gradient workers"
            )
        if self.policy.lora.rank_partition is not None:
            raise ValueError(
                "SFT replay is not yet compatible with task-partitioned LoRA"
            )
        if self.algorithm.kl_coefficient != 0.0:
            raise ValueError(
                "SFT replay and reference KL cannot be enabled together; test one "
                "training factor at a time"
            )
        worker_count = len(self.runtime.training_devices)
        if not is_gradient_worker and worker_count != len(replay.task_ids):
            raise ValueError(
                "worker_task_balanced SFT replay requires one training worker per "
                f"task: workers={worker_count}, tasks={len(replay.task_ids)}"
            )
        environment_tasks = self.environment.kwargs.get("task_ids")
        if environment_tasks != replay.task_ids:
            raise ValueError(
                "policy.load_kwargs.sft_replay.task_ids must exactly match the ordered "
                "environment.kwargs.task_ids development panel"
            )

    def _validate_task_gradient_aggregation_contract(
        self,
        *,
        process_role: str | None,
    ) -> None:
        if self.policy.type != "gr00t_n1d7":
            return
        load = GR00TN17FlowLoadConfig.model_validate(self.policy.load_kwargs)
        if load.gradient_aggregation == "sum":
            return
        if load.gradient_aggregation != "task_pcgrad":
            raise ValueError(
                "Unsupported policy.load_kwargs.gradient_aggregation: "
                f"{load.gradient_aggregation!r}"
            )
        if self.algorithm.flow_sde is None:
            raise ValueError("task_pcgrad currently requires GR00T N1.7 Flow-SDE")
        if self.policy.lora.rank_partition is not None:
            raise ValueError(
                "task_pcgrad and task-partitioned LoRA cannot be enabled together"
            )
        replay = load.sft_replay
        if replay is not None and replay.coefficient > 0.0:
            raise ValueError(
                "task_pcgrad and SFT replay cannot be enabled together; test one "
                "training factor at a time"
            )
        if process_role == "distributed_gradient_worker":
            return
        if not self.runtime.distributed_training:
            raise ValueError("task_pcgrad requires runtime.distributed_training=true")
        raw_tasks = self.environment.kwargs.get("task_ids")
        if not isinstance(raw_tasks, list) or len(raw_tasks) < 2:
            raise ValueError(
                "task_pcgrad requires environment.kwargs.task_ids with at least "
                "two ordered tasks"
            )
        task_keys = [str(value).strip() for value in raw_tasks]
        if any(not value for value in task_keys) or len(set(task_keys)) != len(
            task_keys
        ):
            raise ValueError("task_pcgrad task_ids must be non-empty and unique")
        if len(self.runtime.training_devices) != len(task_keys):
            raise ValueError(
                "task_pcgrad requires one gradient worker per task: "
                f"workers={len(self.runtime.training_devices)}, tasks={len(task_keys)}"
            )

    def _environment_execution_horizon(self) -> int | None:
        """Normalize simulator-specific names for the executed chunk length."""

        values = self.environment.kwargs
        action_chunk_size = values.get("action_chunk_size")
        execution_horizon = values.get("execution_horizon")
        if (
            action_chunk_size is not None
            and execution_horizon is not None
            and int(action_chunk_size) != int(execution_horizon)
        ):
            raise ValueError(
                "environment action_chunk_size and execution_horizon disagree: "
                f"action_chunk_size={action_chunk_size}, "
                f"execution_horizon={execution_horizon}"
            )
        value = (
            execution_horizon if execution_horizon is not None else action_chunk_size
        )
        return None if value is None else int(value)

    def _validate_action_token_generation_contract(self) -> None:
        """Validate categorical sampling controls only for token policies.

        Flow and diffusion policies expose exploration through their native
        probability model (for example Flow-SDE noise), not token temperature
        or top-p. Applying this contract to continuous policies would validate
        fields that their sampler never consumes.
        """

        rollout_generation = self.policy.rollout_generation
        train_generation = self.policy.train_generation
        evaluation_generation = self.policy.evaluation_generation
        if rollout_generation != train_generation:
            raise ValueError(
                "Action-token on-policy training requires identical "
                "policy.rollout_generation and policy.train_generation"
            )
        if self.rollout.temperature != rollout_generation.temperature:
            raise ValueError(
                "rollout.temperature must equal policy.rollout_generation.temperature"
            )
        if self.rollout.deterministic == rollout_generation.do_sample:
            raise ValueError(
                "rollout.deterministic must be the inverse of "
                "policy.rollout_generation.do_sample"
            )
        if self.evaluation.temperature != evaluation_generation.temperature:
            raise ValueError(
                "evaluation.temperature must equal "
                "policy.evaluation_generation.temperature"
            )
        if self.evaluation.deterministic == evaluation_generation.do_sample:
            raise ValueError(
                "evaluation.deterministic must be the inverse of "
                "policy.evaluation_generation.do_sample"
            )

    @property
    def trajectories_per_update(self) -> int:
        return self.rollout.trajectories_per_update(
            group_size=self.algorithm.group_size
        )

    def execution_summary(self) -> dict[str, Any]:
        """Return the executed geometry without loading model dependencies."""

        schedule = self.training.schedule
        execution = self.runtime.rollout_execution
        rollout_device_count = len(self.runtime.rollout_devices)
        training_device_count = len(self.runtime.training_devices)
        if execution.mode == "local_process":
            rollout_actor_count = rollout_device_count * execution.actors_per_device
            rollout_active_actor_count = int(
                execution.actor_kwargs.get(
                    "max_concurrent_rollouts",
                    rollout_actor_count,
                )
            )
            rollout_model_replicas = (
                rollout_actor_count
                if execution.inference_mode == "embedded"
                else (rollout_device_count * execution.inference_replicas_per_device)
            )
            rollout_environment_slots = rollout_actor_count * (
                self.algorithm.group_size if execution.group_batching else 1
            )
            rollout_active_environment_slots = rollout_active_actor_count * (
                self.algorithm.group_size if execution.group_batching else 1
            )
        else:
            rollout_actor_count = 1
            rollout_active_actor_count = 1
            rollout_model_replicas = 1
            rollout_environment_slots = self.rollout.workers
            rollout_active_environment_slots = rollout_environment_slots
        shared_devices = sorted(
            set(self.runtime.rollout_devices).intersection(
                self.runtime.training_devices
            )
        )
        serial_phase_device_reuse = bool(
            shared_devices and execution.lifecycle in {"per_update", "cpu_offload"}
        )
        paired_baseline_source = (
            "external_outcomes"
            if self.evaluation.baseline_outcomes_path is not None
            else (
                "measured_step_zero"
                if self.evaluation.enabled and self.evaluation.evaluate_before_training
                else None
            )
        )
        fixed_horizon_rows: int | None = None
        optimizer_rows: int | None = None
        if isinstance(schedule, RlinfActorBatchScheduleConfig):
            policy_steps = self.rollout.max_policy_steps
            fixed_horizon_rows = self.trajectories_per_update * policy_steps
            optimizer_rows = (
                self.training.optimizer_steps_per_update * schedule.global_batch_size
            )
        elif isinstance(schedule, TrajectoryMinibatchScheduleConfig):
            fixed_horizon_rows = self.trajectories_per_update
            optimizer_rows = (
                self.training.optimizer_steps_per_update
                * schedule.minibatch_trajectories
            )
        return {
            "config_fingerprint": self.fingerprint,
            "policy_type": self.policy.type,
            "algorithm": self.algorithm.type,
            "flow_sde_contract": (
                self.algorithm.flow_sde.model_dump(mode="json")
                if self.algorithm.flow_sde is not None
                else None
            ),
            "training_unit": self.algorithm.training_unit,
            "precalculate_logprobs": self.algorithm.precalculate_logprobs,
            "rollout_logprob_source": self.algorithm.rollout_logprob_source,
            "schedule": schedule.type,
            "updates": self.training.updates,
            "groups_per_update": (
                self.rollout.groups_per_update * self.rollout.epochs_per_update
            ),
            "trajectories_per_update": self.trajectories_per_update,
            "total_trajectories": (
                self.trajectories_per_update * self.training.updates
            ),
            "optimizer_steps_per_update": (self.training.optimizer_steps_per_update),
            "sft_replay": (
                GR00TN17FlowLoadConfig.model_validate(
                    self.policy.load_kwargs
                ).sft_replay.model_dump(mode="json")
                if self.policy.type == "gr00t_n1d7"
                and GR00TN17FlowLoadConfig.model_validate(
                    self.policy.load_kwargs
                ).sft_replay
                is not None
                else None
            ),
            "sft_anchor": (
                PI0FastLoadConfig.model_validate(
                    self.policy.load_kwargs
                ).sft_anchor.model_dump(mode="json")
                if self.policy.type == "pi0_fast"
                and PI0FastLoadConfig.model_validate(self.policy.load_kwargs).sft_anchor
                is not None
                else None
            ),
            "pi0_fast_rl_token_scope": (
                PI0FastLoadConfig.model_validate(self.policy.load_kwargs).rl_token_scope
                if self.policy.type == "pi0_fast"
                else None
            ),
            "optimizer_update_epochs": (
                schedule.update_epochs
                if isinstance(
                    schedule,
                    FullUpdateScheduleConfig | RlinfActorBatchScheduleConfig,
                )
                else 1
            ),
            "action_token_progress_enabled": (self.training.log_action_token_progress),
            "action_token_progress_every_microbatches": (
                self.training.action_token_progress_every_microbatches
            ),
            "max_log_file_mb": self.storage.max_log_file_mb,
            "fixed_horizon_rows": fixed_horizon_rows,
            "optimizer_rows": optimizer_rows,
            "evaluation_enabled": self.evaluation.enabled,
            "evaluation_before_training": self.evaluation.evaluate_before_training,
            "evaluation_after_first_update": self.evaluation.evaluate_after_first_update,
            "evaluation_data_role": self.evaluation.data_role,
            "paired_evaluation_enabled": (paired_baseline_source is not None),
            "paired_baseline_source": paired_baseline_source,
            "baseline_outcomes_path": (
                str(self.evaluation.baseline_outcomes_path)
                if self.evaluation.baseline_outcomes_path is not None
                else None
            ),
            "evaluation_every_updates": self.evaluation.every_updates,
            "evaluation_episodes": self.evaluation.episodes,
            "wandb_enabled": self.observability.wandb.enabled,
            "wandb_train_video_required": self.observability.require_train_video,
            "wandb_evaluation_video_required": (
                self.observability.require_evaluation_video
            ),
            "videos_per_update": self.observability.videos_per_update,
            "videos_per_evaluation": self.observability.videos_per_evaluation,
            "rollout_workers": self.rollout.workers,
            "rollout_max_environment_steps": self.rollout.max_episode_steps,
            "rollout_max_policy_steps": self.rollout.max_policy_steps,
            "rollout_shared_prefix_action_chunks": (
                self.rollout.shared_prefix_action_chunks
            ),
            "rollout_execution_mode": self.runtime.rollout_execution.mode,
            "rollout_actor_factory": self.runtime.rollout_execution.actor_factory,
            "rollout_actor_kwargs": self.runtime.rollout_execution.actor_kwargs,
            "rollout_actors_per_device": (
                self.runtime.rollout_execution.actors_per_device
            ),
            "rollout_group_batching": (self.runtime.rollout_execution.group_batching),
            "rollout_actor_lifecycle": self.runtime.rollout_execution.lifecycle,
            "rollout_policy_sync": self.runtime.rollout_execution.policy_sync,
            "rollout_inference_mode": (self.runtime.rollout_execution.inference_mode),
            "rollout_inference_factory": (
                self.runtime.rollout_execution.inference_factory
            ),
            "rollout_inference_replicas_per_device": (
                self.runtime.rollout_execution.inference_replicas_per_device
            ),
            "rollout_inference_max_batch_size": (
                self.runtime.rollout_execution.inference_max_batch_size
            ),
            "rollout_inference_max_wait_ms": (
                self.runtime.rollout_execution.inference_max_wait_ms
            ),
            "rollout_actor_python_executable": (
                self.runtime.rollout_execution.actor_python_executable
            ),
            "rollout_inference_python_executable": (
                self.runtime.rollout_execution.inference_python_executable
            ),
            "rollout_devices": list(self.runtime.rollout_devices),
            "rollout_device_count": rollout_device_count,
            "rollout_actor_count": rollout_actor_count,
            "rollout_active_actor_count": rollout_active_actor_count,
            "rollout_model_replicas": rollout_model_replicas,
            "rollout_environment_slots": rollout_environment_slots,
            "rollout_active_environment_slots": rollout_active_environment_slots,
            "training_devices": list(self.runtime.training_devices),
            "worker_python_executable": self.runtime.worker_python_executable,
            "training_device_count": training_device_count,
            "training_model_replicas": (
                training_device_count if self.runtime.distributed_training else 1
            ),
            "shared_rollout_training_devices": shared_devices,
            "rollout_training_time_shared": serial_phase_device_reuse,
            "serial_phase_device_reuse": serial_phase_device_reuse,
            "serial_phase_order": (
                [
                    *(
                        ["initial_evaluation"]
                        if self.evaluation.enabled
                        and self.evaluation.evaluate_before_training
                        else []
                    ),
                    "train_rollout",
                    "training",
                    *(["periodic_evaluation"] if self.evaluation.enabled else []),
                ]
                if serial_phase_device_reuse
                else []
            ),
            "distributed_training": self.runtime.distributed_training,
            "training_worker_lifecycle": self.runtime.training_worker_lifecycle,
            "worker_handoff_dir": str(self.runtime.worker_handoff_dir),
            "worker_timeout_seconds": self.runtime.worker_timeout_seconds,
            "max_worker_handoff_mb": self.runtime.max_worker_handoff_mb,
            "keep_worker_handoffs": self.runtime.keep_worker_handoffs,
        }

    def consumption_report(self) -> dict[str, Any]:
        """Return runtime ownership for every resolved public config field."""

        from art_embodied.config_contract import config_consumption_report

        return config_consumption_report(self)

    @property
    def fingerprint(self) -> str:
        document = self.model_dump(mode="json")
        # Preserve identities frozen before shared-prefix sampling existed.
        # A positive value remains behavior-defining and changes the fingerprint.
        if document["rollout"].get("shared_prefix_action_chunks") == 0:
            document["rollout"].pop("shared_prefix_action_chunks")
        # The immutable W&B locator changes only lineage delivery. Excluding it
        # preserves behavior fingerprints and frozen evaluation manifests.
        document["observability"]["wandb"].pop("input_model_artifact_ref", None)
        if not document["observability"]["wandb"].get("native_update_steps"):
            document["observability"]["wandb"].pop("native_update_steps", None)
        payload = json.dumps(document, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def resume_contract_fingerprint(self) -> str:
        """Fingerprint fields that must match for an exact optimizer resume.

        Operational fields such as output paths, telemetry, device allocation,
        checkpoint cadence, and the final update target may change after a
        preemption. Policy, rollout, reward, objective, and optimizer math may
        not.
        """

        policy = self.policy.model_dump(mode="json")
        policy.pop("device", None)
        training = self.training.model_dump(mode="json")
        for key in (
            "updates",
            "checkpoint_every_updates",
            "log_action_token_progress",
            "action_token_progress_every_microbatches",
        ):
            training.pop(key, None)
        environment = self.environment.model_dump(mode="json")
        # A sealed evaluation bank is evidence attached to a policy, not part
        # of the rollout or optimizer contract. It may be added after a
        # preemption without changing the continued training computation.
        environment["kwargs"].pop("evaluation_state_manifest", None)
        rollout = self.rollout.model_dump(mode="json")
        if rollout.get("shared_prefix_action_chunks") == 0:
            rollout.pop("shared_prefix_action_chunks")
        payload = {
            "schema_version": self.schema_version,
            "experiment_seed": self.experiment.seed,
            "policy": policy,
            "environment": environment,
            "reward": self.reward.model_dump(mode="json"),
            "algorithm": self.algorithm.model_dump(mode="json"),
            "rollout": rollout,
            "training": training,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EmbodiedExperimentConfig":
        source = Path(path)
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(
                f"Embodied experiment YAML must contain a mapping: {source}"
            )
        return cls.model_validate(raw)

    def to_yaml(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            yaml.safe_dump(
                self.model_dump(mode="json"),
                sort_keys=False,
                allow_unicode=False,
            ),
            encoding="utf-8",
        )
        return destination
