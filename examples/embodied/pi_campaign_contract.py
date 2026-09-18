"""Fail-closed launch contracts for calibrated PI Flow-SDE campaigns."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from art_embodied import EmbodiedExperimentConfig


@dataclass(frozen=True)
class PICampaignProfile:
    """One empirically admitted sampler and experiment geometry."""

    policy_type: str
    policy_path: str
    environment_task: str
    run_token: str
    denoise_steps: int
    noise_level: float
    optimizer_steps_per_update: int
    update_epochs: int
    model_chunk_size: int
    execution_horizon: int
    max_episode_steps: int
    max_policy_steps: int
    trainable_parameter_strategy: str


PROFILES = {
    "pi0-spatial-rlinf-reference-v1": PICampaignProfile(
        policy_type="pi0",
        policy_path="RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT",
        environment_task="libero_spatial",
        run_token="k4-n05",
        denoise_steps=4,
        noise_level=0.5,
        optimizer_steps_per_update=24,
        update_epochs=1,
        model_chunk_size=50,
        execution_horizon=5,
        max_episode_steps=240,
        max_policy_steps=48,
        trainable_parameter_strategy="pi0_action_expert_lora",
    ),
    "pi05-long-rlinf-reference-v1": PICampaignProfile(
        policy_type="pi05",
        policy_path="RLinf/RLinf-Pi05-SFT",
        environment_task="libero_10",
        run_token="k4-n05",
        denoise_steps=4,
        noise_level=0.5,
        optimizer_steps_per_update=48,
        update_epochs=2,
        model_chunk_size=10,
        execution_horizon=10,
        max_episode_steps=480,
        max_policy_steps=48,
        trainable_parameter_strategy="pi05_action_expert_lora",
    ),
    "pi0-spatial-behavioral-v1": PICampaignProfile(
        policy_type="pi0",
        policy_path="RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT",
        environment_task="libero_spatial",
        run_token="k8-n03",
        denoise_steps=8,
        noise_level=0.3,
        optimizer_steps_per_update=24,
        update_epochs=1,
        model_chunk_size=50,
        execution_horizon=5,
        max_episode_steps=240,
        max_policy_steps=48,
        trainable_parameter_strategy="pi0_action_expert_lora",
    ),
    "pi05-long-behavioral-v1": PICampaignProfile(
        policy_type="pi05",
        policy_path="RLinf/RLinf-Pi05-SFT",
        environment_task="libero_10",
        run_token="k16-n03",
        denoise_steps=16,
        noise_level=0.3,
        optimizer_steps_per_update=48,
        update_epochs=2,
        model_chunk_size=10,
        execution_horizon=10,
        max_episode_steps=480,
        max_policy_steps=48,
        trainable_parameter_strategy="pi05_action_expert_lora",
    ),
}


def audit_pi_campaign_contract(
    config: EmbodiedExperimentConfig,
    *,
    profile_name: str,
    claim_facing: bool = False,
    expected_updates: int | None = None,
) -> dict[str, Any]:
    """Check the complete claim-facing contract before allocating a node."""

    profile = PROFILES[profile_name]
    summary = config.execution_summary()
    checks: list[dict[str, Any]] = []

    def check(path: str, actual: Any, expected: Any) -> None:
        if isinstance(expected, float):
            passed = isinstance(actual, (float, int)) and math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1e-12
            )
        else:
            passed = actual == expected
        checks.append(
            {
                "path": path,
                "actual": actual,
                "expected": expected,
                "passed": passed,
            }
        )

    flow_sde = config.algorithm.flow_sde
    load_kwargs = config.policy.load_kwargs
    optimizer = config.training.optimizer
    schedule = config.training.schedule
    check("policy.type", config.policy.type, profile.policy_type)
    check("policy.path", config.policy.path, profile.policy_path)
    check(
        "policy.load_kwargs.model_chunk_size",
        load_kwargs.get("model_chunk_size"),
        profile.model_chunk_size,
    )
    check(
        "policy.load_kwargs.execution_horizon",
        load_kwargs.get("execution_horizon"),
        profile.execution_horizon,
    )
    check(
        "policy.trainable_parameter_strategy",
        config.policy.trainable_parameter_strategy,
        profile.trainable_parameter_strategy,
    )
    check("policy.force_trainable_float32", config.policy.force_trainable_float32, True)
    check("policy.lora.enabled", config.policy.lora.enabled, True)
    check("policy.lora.rank", config.policy.lora.rank, 32)
    check("policy.lora.alpha", config.policy.lora.alpha, 32)
    check("policy.lora.dropout", config.policy.lora.dropout, 0.0)
    check("policy.lora.init", config.policy.lora.init, "gaussian")
    check("environment.task", config.environment.task, profile.environment_task)
    check(
        "experiment.run.contains_sampler_contract",
        profile.run_token in config.experiment.run,
        True,
    )
    check(
        "experiment.tags.contains_sampler_contract",
        f"flow-sde-{profile.run_token}" in config.experiment.tags,
        True,
    )
    check(
        "algorithm.flow_sde.num_denoise_steps",
        flow_sde.num_denoise_steps if flow_sde is not None else None,
        profile.denoise_steps,
    )
    check(
        "algorithm.flow_sde.noise_level",
        flow_sde.noise_level if flow_sde is not None else None,
        profile.noise_level,
    )
    check(
        "algorithm.flow_sde.stochastic_transitions_per_sample",
        flow_sde.stochastic_transitions_per_sample if flow_sde is not None else None,
        1,
    )
    check(
        "algorithm.flow_sde.selected_step_sampling",
        flow_sde.selected_step_sampling if flow_sde is not None else None,
        "uniform",
    )
    check(
        "algorithm.flow_sde.joint_logprob",
        flow_sde.joint_logprob if flow_sde is not None else None,
        False,
    )
    check("algorithm.type", config.algorithm.type, "grpo")
    check("algorithm.group_size", config.algorithm.group_size, 8)
    check("algorithm.clip_epsilon_low", config.algorithm.clip_epsilon_low, 0.2)
    check("algorithm.clip_epsilon_high", config.algorithm.clip_epsilon_high, 0.2)
    check("algorithm.kl_coefficient", config.algorithm.kl_coefficient, 0.0)
    check("algorithm.clip_ratio_c", config.algorithm.clip_ratio_c, 3.0)
    check("algorithm.importance_sampling_level", config.algorithm.importance_sampling_level, "token")
    check("algorithm.training_unit", config.algorithm.training_unit, "action")
    check("algorithm.action_advantage_mode", config.algorithm.action_advantage_mode, "example")
    check("algorithm.score_source", config.algorithm.score_source, "trajectory_reward")
    check("algorithm.pad_fixed_horizon_examples", config.algorithm.pad_fixed_horizon_examples, True)
    check("algorithm.filter_rewards", config.algorithm.filter_rewards, True)
    check("algorithm.reward_filter_mode", config.algorithm.reward_filter_mode, "loss_mask")
    check("algorithm.rewards_lower_bound", config.algorithm.rewards_lower_bound, 0.1)
    check("algorithm.rewards_upper_bound", config.algorithm.rewards_upper_bound, 0.9)
    check("algorithm.loss_aggregation", config.algorithm.loss_aggregation, "rlinf_chunk_mean")
    check("algorithm.advantage.scope", config.algorithm.advantage.scope, "group")
    check("algorithm.advantage.normalize", config.algorithm.advantage.normalize, True)
    check(
        "algorithm.advantage.extra_global_normalization",
        config.algorithm.advantage.extra_global_normalization,
        False,
    )
    check("algorithm.advantage.std_unbiased", config.algorithm.advantage.std_unbiased, True)
    check("algorithm.advantage.epsilon", config.algorithm.advantage.epsilon, 1.0e-6)
    check("rollout.groups_per_update", config.rollout.groups_per_update, 8)
    check("rollout.epochs_per_update", config.rollout.epochs_per_update, 16)
    check("rollout.trajectories_per_update", config.trajectories_per_update, 1024)
    check("rollout.max_episode_steps", config.rollout.max_episode_steps, profile.max_episode_steps)
    check("rollout.max_policy_steps", config.rollout.max_policy_steps, profile.max_policy_steps)
    check("rollout.temperature", config.rollout.temperature, 1.0)
    check("rollout.deterministic", config.rollout.deterministic, False)
    check(
        "training.optimizer_steps_per_update",
        config.training.optimizer_steps_per_update,
        profile.optimizer_steps_per_update,
    )
    check("training.schedule.type", schedule.type, "rlinf_actor_global_batch")
    check("training.schedule.global_batch_size", schedule.global_batch_size, 2048)
    check("training.schedule.actor_seed", schedule.actor_seed, 42)
    check("training.schedule.update_epochs", schedule.update_epochs, profile.update_epochs)
    check("training.optimizer.type", optimizer.type, "adamw")
    check("training.optimizer.learning_rate", optimizer.learning_rate, 1.0e-4)
    check("training.optimizer.weight_decay", optimizer.weight_decay, 0.0)
    check("training.optimizer.beta1", optimizer.beta1, 0.9)
    check("training.optimizer.beta2", optimizer.beta2, 0.95)
    check("training.optimizer.epsilon", optimizer.epsilon, 1.0e-5)
    check("training.optimizer.max_grad_norm", optimizer.max_grad_norm, 1.0)
    if expected_updates is not None:
        check("training.updates", config.training.updates, expected_updates)
    check("evaluation.enabled", config.evaluation.enabled, True)
    check(
        "evaluation.evaluate_before_training",
        config.evaluation.evaluate_before_training,
        True,
    )
    check("evaluation.every_updates", config.evaluation.every_updates, 5)
    check("evaluation.episodes", config.evaluation.episodes, 100)
    if claim_facing:
        check("evaluation.data_role", config.evaluation.data_role, "development")
    check("evaluation.runtime", config.evaluation.runtime, "native_lerobot")
    check("evaluation.fixed_scenarios", len(config.evaluation.fixed_scenarios), 100)
    check("runtime.distributed_training", config.runtime.distributed_training, True)
    check("runtime.rollout_model_replicas", summary["rollout_model_replicas"], 8)
    check("runtime.training_model_replicas", summary["training_model_replicas"], 8)
    check(
        "observability.delivery_failure_policy",
        config.observability.delivery_failure_policy,
        "fail_run",
    )
    check("observability.wandb.enabled", config.observability.wandb.enabled, True)
    check("observability.wandb.mode", config.observability.wandb.mode, "online")
    check(
        "observability.wandb.project_matches_experiment",
        config.observability.wandb.project,
        config.experiment.project,
    )
    check(
        "observability.wandb.connection",
        config.observability.wandb.connection,
        "primary",
    )
    check(
        "observability.wandb.log_model_artifacts",
        config.observability.wandb.log_model_artifacts,
        True,
    )
    check(
        "observability.wandb.log_evaluation_artifacts",
        config.observability.wandb.log_evaluation_artifacts,
        True,
    )
    check("observability.weave.enabled", config.observability.weave.enabled, True)
    check(
        "observability.weave.project_matches_experiment",
        config.observability.weave.project,
        config.experiment.project,
    )
    check(
        "observability.weave.trace_trajectories",
        config.observability.weave.trace_trajectories,
        True,
    )
    check(
        "observability.require_train_video",
        config.observability.require_train_video,
        True,
    )
    check(
        "observability.require_evaluation_video",
        config.observability.require_evaluation_video,
        True,
    )
    failed = [row for row in checks if not row["passed"]]
    return {
        "schema_version": 1,
        "profile": profile_name,
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "config_fingerprint": summary["config_fingerprint"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--claim-facing", action="store_true")
    parser.add_argument("--expected-updates", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EmbodiedExperimentConfig.from_yaml(args.config)
    report = audit_pi_campaign_contract(
        config,
        profile_name=args.profile,
        claim_facing=args.claim_facing,
        expected_updates=args.expected_updates,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
