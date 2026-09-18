"""Backend construction from the explicit embodied experiment contract."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

from art_embodied.config import (
    EmbodiedExperimentConfig,
    FullUpdateScheduleConfig,
    GR00TN17FlowLoadConfig,
)
from art_embodied.policy_capabilities import BackendRequirements

from ..conformance.rlinf import RlinfScheduledActionTokenBackend
from .action_token import ActionTokenGRPOBackend, ActionTokenGSPOBackend

BackendFactory: TypeAlias = Callable[..., Any]

_BACKEND_FACTORIES: dict[tuple[str, str], BackendFactory] = {}
_BACKEND_REQUIREMENTS: dict[tuple[str, str], BackendRequirements] = {}


def register_embodied_backend(
    *,
    action_kind: str,
    algorithm_type: str,
    factory: BackendFactory,
    requirements: BackendRequirements | None = None,
    replace: bool = False,
) -> None:
    """Register a backend for one action representation and algorithm.

    This is the boundary for future native flow/diffusion objectives. Such a
    backend can be added without teaching the generic runner how that policy
    computes likelihoods or denoising losses.
    """

    key = (action_kind.strip(), algorithm_type.strip())
    if not all(key):
        raise ValueError("action_kind and algorithm_type cannot be empty")
    if not callable(factory):
        raise TypeError("backend factory must be callable")
    if key in _BACKEND_FACTORIES and not replace:
        raise ValueError(
            "embodied backend already registered for "
            f"action_kind={key[0]!r}, algorithm_type={key[1]!r}"
        )
    _BACKEND_FACTORIES[key] = factory
    if requirements is not None:
        _BACKEND_REQUIREMENTS[key] = requirements
    elif replace:
        _BACKEND_REQUIREMENTS.pop(key, None)


def registered_embodied_backends() -> tuple[tuple[str, str], ...]:
    """Return registered ``(action_kind, algorithm_type)`` capabilities."""

    return tuple(sorted(_BACKEND_FACTORIES))


def make_embodied_backend(
    config: EmbodiedExperimentConfig,
    *,
    policy: Any,
    optimizer: Any | None = None,
) -> Any:
    """Construct the backend selected by action representation and algorithm."""

    key = (config.rollout.action_payload.kind, config.algorithm.type)
    try:
        factory = _BACKEND_FACTORIES[key]
    except KeyError as exc:
        available = ", ".join(
            f"{kind}/{algorithm}" for kind, algorithm in registered_embodied_backends()
        )
        raise ValueError(
            "No embodied backend registered for "
            f"action_kind={key[0]!r}, algorithm_type={key[1]!r}. "
            f"Registered capabilities: {available or '<none>'}"
        ) from exc
    requirements = _BACKEND_REQUIREMENTS.get(key)
    if requirements is not None:
        # Lazy import avoids a cycle: OpenVLA's policy implementation imports
        # action-token backend data structures during policy registration.
        from art_embodied.policies.factory import policy_capabilities

        requirements.validate(policy_capabilities(config))
    return factory(config, policy=policy, optimizer=optimizer)


def make_action_token_backend(
    config: EmbodiedExperimentConfig,
    *,
    policy: Any,
    optimizer: Any | None = None,
) -> Any:
    """Build an action-token backend without runner-specific hidden defaults."""

    if config.rollout.action_payload.kind != "token":
        raise ValueError(
            "Action-token GRPO/GSPO requires rollout.action_payload.kind='token'"
        )
    if not callable(getattr(policy, "action_token_logprobs", None)):
        raise TypeError(
            "Action-token GRPO/GSPO requires policy.action_token_logprobs(...)"
        )
    backend_type = (
        ActionTokenGSPOBackend
        if config.algorithm.type == "gspo"
        else ActionTokenGRPOBackend
    )
    optimizer_config = config.training.optimizer
    backend = backend_type(
        policy=policy,
        optimizer=optimizer,
        device=config.policy.device,
        lr=optimizer_config.learning_rate,
        optimizer_weight_decay=optimizer_config.weight_decay,
        optimizer_adam_beta1=optimizer_config.beta1,
        optimizer_adam_beta2=optimizer_config.beta2,
        optimizer_adam_eps=optimizer_config.epsilon,
        clip_epsilon=config.algorithm.clip_epsilon_low,
        clip_epsilon_low=config.algorithm.clip_epsilon_low,
        clip_epsilon_high=config.algorithm.clip_epsilon_high,
        clip_ratio_c=config.algorithm.clip_ratio_c,
        kl_coef=config.algorithm.kl_coefficient,
        normalize_advantages=config.algorithm.advantage.normalize,
        advantage_normalization_scope=config.algorithm.advantage.scope,
        advantage_std_unbiased=config.algorithm.advantage.std_unbiased,
        advantage_epsilon=config.algorithm.advantage.epsilon,
        filter_rewards=config.algorithm.filter_rewards,
        reward_filter_mode=config.algorithm.reward_filter_mode,
        rewards_lower_bound=config.algorithm.rewards_lower_bound,
        rewards_upper_bound=config.algorithm.rewards_upper_bound,
        importance_sampling_level=config.algorithm.importance_sampling_level,
        max_grad_norm=optimizer_config.max_grad_norm,
        checkpoint_dir=config.storage.output_dir / "checkpoints",
        checkpoint_config_fingerprint=config.fingerprint,
        checkpoint_resume_contract_fingerprint=config.resume_contract_fingerprint,
        require_observations=config.rollout.action_payload.require_observation,
        require_prompts=config.rollout.action_payload.require_prompt,
        training_unit=config.algorithm.training_unit,
        action_advantage_mode=config.algorithm.action_advantage_mode,
        rlinf_action_level_extra_global_normalization=(
            config.algorithm.advantage.extra_global_normalization
            if config.algorithm.action_advantage_mode == "rlinf_action_level_cumulative"
            else False
        ),
        rlinf_action_level_mask_zero_variance_groups=(
            config.algorithm.advantage.mask_zero_variance_groups
            if config.algorithm.action_advantage_mode == "rlinf_action_level_cumulative"
            else False
        ),
        rlinf_action_level_score_source=config.algorithm.score_source,
        loss_aggregation=config.algorithm.loss_aggregation,
        precalculate_logprobs=config.algorithm.precalculate_logprobs,
        rollout_logprob_source=config.algorithm.rollout_logprob_source,
        logprob_eval_mode=config.algorithm.logprob_eval_mode,
        logprob_microbatch_size=config.algorithm.logprob_microbatch_size,
        train_logprob_microbatch_size=config.training.microbatch_size,
        pre_update_logprob_kl_tolerance=(
            config.algorithm.pre_update_logprob_kl_tolerance
        ),
        pre_update_ratio_tolerance=config.algorithm.pre_update_ratio_tolerance,
        skip_optimizer_step_without_policy_gradient_signal=(
            config.algorithm.skip_optimizer_step_without_policy_gradient_signal
        ),
        progress_path=(
            config.storage.output_dir / "diagnostics/action_token_progress.jsonl"
            if config.training.log_action_token_progress
            else None
        ),
        progress_every_microbatches=(
            config.training.action_token_progress_every_microbatches
        ),
        progress_max_bytes=config.storage.max_log_file_mb * 1024 * 1024,
    )
    if config.runtime.distributed_training:
        from .local_process import LocalProcessActionTokenBackend

        return LocalProcessActionTokenBackend(
            config=config,
            policy=policy,
            backend=backend,
        )
    if isinstance(config.training.schedule, FullUpdateScheduleConfig):
        return backend
    return RlinfScheduledActionTokenBackend(backend=backend, config=config)


def make_flow_sde_backend(
    config: EmbodiedExperimentConfig,
    *,
    policy: Any,
    optimizer: Any | None = None,
) -> Any:
    """Build the sampler-aligned PI0/PI0.5 GRPO backend."""

    from .flow_sde import FlowSDEGRPOBackend

    if config.algorithm.flow_sde is None:
        raise ValueError("continuous Flow-SDE GRPO requires algorithm.flow_sde")
    if config.algorithm.kl_coefficient > 0.0 and not callable(
        getattr(policy, "flow_sde_reference_logprobs", None)
    ):
        raise TypeError(
            "continuous Flow-SDE algorithm.kl_coefficient requires "
            "policy.flow_sde_reference_logprobs"
        )
    optimizer_config = config.training.optimizer
    n1d7_replay = (
        GR00TN17FlowLoadConfig.model_validate(config.policy.load_kwargs).sft_replay
        if config.policy.type == "gr00t_n1d7"
        else None
    )
    backend = FlowSDEGRPOBackend(
        policy=policy,
        optimizer=optimizer,
        device=config.policy.device,
        learning_rate=optimizer_config.learning_rate,
        weight_decay=optimizer_config.weight_decay,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
        epsilon=optimizer_config.epsilon,
        max_grad_norm=optimizer_config.max_grad_norm,
        pre_update_logprob_kl_tolerance=(
            config.algorithm.pre_update_logprob_kl_tolerance
        ),
        pre_update_ratio_tolerance=config.algorithm.pre_update_ratio_tolerance,
        microbatch_size=config.training.microbatch_size,
        optimizer_steps_per_update=config.training.optimizer_steps_per_update,
        training_schedule=config.training.schedule,
        max_episode_steps=config.rollout.max_episode_steps,
        group_size=config.algorithm.group_size,
        advantage_epsilon=config.algorithm.advantage.epsilon,
        advantage_std_unbiased=config.algorithm.advantage.std_unbiased,
        clip_epsilon_low=config.algorithm.clip_epsilon_low,
        clip_epsilon_high=config.algorithm.clip_epsilon_high,
        clip_ratio_c=config.algorithm.clip_ratio_c,
        filter_rewards=config.algorithm.filter_rewards,
        rewards_lower_bound=config.algorithm.rewards_lower_bound,
        rewards_upper_bound=config.algorithm.rewards_upper_bound,
        checkpoint_dir=config.storage.output_dir / "checkpoints",
        config_fingerprint=config.fingerprint,
        resume_contract_fingerprint=config.resume_contract_fingerprint,
        keep_last_checkpoints=config.storage.keep_last_checkpoints,
        retain_checkpoint_updates=config.storage.retain_checkpoint_updates,
        diagnostics_dir=(
            config.storage.output_dir / "diagnostics/flow_sde_batches"
            if config.storage.retain_rollout_payloads
            else None
        ),
        precalculate_logprobs=config.algorithm.precalculate_logprobs,
        reference_kl_coefficient=config.algorithm.kl_coefficient,
        sft_replay_coefficient=(
            n1d7_replay.coefficient if n1d7_replay is not None else 0.0
        ),
    )
    if config.runtime.distributed_training:
        from .flow_sde_local_process import LocalProcessFlowSDEBackend

        return LocalProcessFlowSDEBackend(
            config=config,
            policy=policy,
            backend=backend,
        )
    return backend


register_embodied_backend(
    action_kind="token",
    algorithm_type="grpo",
    factory=make_action_token_backend,
    requirements=BackendRequirements(
        action_kind="token",
        probability_models=frozenset({"categorical_tokens"}),
        exact_logprobs=True,
        teacher_forced_rescore=True,
        rng_replay=True,
    ),
)
register_embodied_backend(
    action_kind="continuous",
    algorithm_type="grpo",
    factory=make_flow_sde_backend,
    requirements=BackendRequirements(
        action_kind="continuous",
        probability_models=frozenset({"gaussian_flow_sde"}),
        exact_logprobs=True,
        teacher_forced_rescore=True,
        rng_replay=True,
    ),
)
register_embodied_backend(
    action_kind="token",
    algorithm_type="gspo",
    factory=make_action_token_backend,
    requirements=BackendRequirements(
        action_kind="token",
        probability_models=frozenset({"categorical_tokens"}),
        exact_logprobs=True,
        teacher_forced_rescore=True,
        rng_replay=True,
    ),
)
