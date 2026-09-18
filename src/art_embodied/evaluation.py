"""Fixed, repeatable held-out evaluation for embodied policies."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
from statistics import fmean
import time
from typing import Any

from .config import EmbodiedExperimentConfig
from .evidence import evaluation_evidence_identity, require_resolved_sealed_identity
from .experiment import (
    EmbodiedScenario,
    EvaluationResult,
    ExperimentProgress,
    ProgressLogger,
    RolloutContext,
    RolloutFunction,
    _finish_rollout,
    _prepare_rollout,
    _stable_seed,
)
from .trajectories import EmbodiedTrajectory
from .types import TrainResult


@dataclass(frozen=True, slots=True)
class EvaluationEpisode:
    """One fully specified episode in a fixed evaluation plan."""

    index: int
    scenario: EmbodiedScenario
    environment_seed: int
    policy_seed: int


class FixedScenarioEvaluator:
    """Evaluate the current policy on an immutable scenario/seed plan.

    The caller keeps ownership of the simulator and policy through ``rollout``.
    ART only fixes episode selection and seeds, bounds concurrency, aggregates
    metrics, and returns trajectories for observability. Failed episodes count
    as unsuccessful instead of silently shrinking the denominator.
    """

    def __init__(
        self,
        *,
        config: EmbodiedExperimentConfig,
        scenarios: Sequence[EmbodiedScenario],
        rollout: RolloutFunction,
        success_metric: str = "success",
        workers: int | None = None,
        log_progress: ProgressLogger | None = None,
        measured_baseline_path: str | Path | None = None,
    ) -> None:
        if not config.evaluation.enabled:
            raise ValueError("FixedScenarioEvaluator requires evaluation.enabled=true")
        if not scenarios:
            raise ValueError("FixedScenarioEvaluator requires evaluation scenarios")
        if not success_metric:
            raise ValueError("success_metric cannot be empty")
        self.config = config
        self.rollout = rollout
        self.success_metric = success_metric
        self.log_progress = log_progress
        self.workers = int(workers or config.rollout.workers)
        if self.workers < 1:
            raise ValueError("evaluation workers must be positive")

        by_id = {scenario.id: scenario for scenario in scenarios}
        if len(by_id) != len(scenarios):
            raise ValueError("evaluation scenario ids must be unique")
        missing = [
            scenario_id
            for scenario_id in config.evaluation.fixed_scenarios
            if scenario_id not in by_id
        ]
        if missing:
            raise ValueError(
                "evaluation.fixed_scenarios are missing from the supplied scenarios: "
                + ", ".join(missing)
            )
        self.scenarios = tuple(
            by_id[scenario_id] for scenario_id in config.evaluation.fixed_scenarios
        )
        self.plan = self._build_plan()
        self._measured_baseline_path = (
            Path(measured_baseline_path).expanduser().resolve()
            if measured_baseline_path is not None
            else None
        )
        if (
            self._measured_baseline_path is not None
            and not self._measured_baseline_path.is_file()
        ):
            raise FileNotFoundError(
                "Measured same-run baseline outcome report is missing: "
                f"{self._measured_baseline_path}"
            )

    def _build_plan(self) -> tuple[EvaluationEpisode, ...]:
        seeds = self.config.evaluation.seeds
        seed_contract = self.config.evaluation.kwargs.get("seed_contract", {})
        if not isinstance(seed_contract, dict):
            raise TypeError("evaluation.kwargs.seed_contract must be a mapping")
        episodes: list[EvaluationEpisode] = []
        for episode_index in range(self.config.evaluation.episodes):
            scenario = self.scenarios[episode_index % len(self.scenarios)]
            repetition = episode_index // len(self.scenarios)
            configured_seed = seeds[repetition % len(seeds)]
            environment_seed = _evaluation_seed(
                seed_contract=seed_contract,
                kind="environment",
                configured_seed=configured_seed,
                scenario_id=scenario.id,
                repetition=repetition,
            )
            policy_seed = _evaluation_seed(
                seed_contract=seed_contract,
                kind="policy",
                configured_seed=configured_seed,
                scenario_id=scenario.id,
                repetition=repetition,
            )
            episodes.append(
                EvaluationEpisode(
                    index=episode_index,
                    scenario=scenario,
                    environment_seed=environment_seed,
                    policy_seed=policy_seed,
                )
            )
        return tuple(episodes)

    async def __call__(
        self,
        step: int,
        train_result: TrainResult,
        config: EmbodiedExperimentConfig,
    ) -> EvaluationResult:
        if config.fingerprint != self.config.fingerprint:
            raise ValueError("Evaluator config does not match experiment config")
        identity = evaluation_evidence_identity(
            policy_path=config.policy.path,
            policy_revision=config.policy.revision,
            adapter_path=config.policy.load_kwargs.get("peft_adapter_path"),
            checkpoint_path=getattr(train_result, "checkpoint_path", None),
            evaluation_manifest_path=_evaluation_manifest_path(config),
            step=step,
        )
        if config.evaluation.data_role == "sealed_test":
            require_resolved_sealed_identity(identity)
        await _prepare_rollout(self.rollout, update=step)
        semaphore = asyncio.Semaphore(self.workers)

        async def run_episode(
            episode: EvaluationEpisode,
        ) -> tuple[int, EmbodiedTrajectory | Exception]:
            context = RolloutContext(
                update=step,
                group_index=episode.index,
                attempt_index=0,
                environment_seed=episode.environment_seed,
                policy_seed=episode.policy_seed,
                config_fingerprint=config.fingerprint,
            )
            try:
                async with semaphore:
                    trajectory = await self.rollout(episode.scenario, context)
            except Exception as exc:
                return episode.index, exc
            trajectory.metadata.setdefault("split", config.evaluation.split)
            trajectory.metadata.setdefault(
                "evaluation_data_role", config.evaluation.data_role
            )
            trajectory.metadata.setdefault("evaluation_episode", episode.index)
            trajectory.metadata.setdefault("scenario_id", episode.scenario.id)
            trajectory.metadata.setdefault("environment_seed", episode.environment_seed)
            trajectory.metadata.setdefault("policy_seed", episode.policy_seed)
            trajectory.metadata.setdefault("config_fingerprint", config.fingerprint)
            return episode.index, trajectory

        tasks = [asyncio.create_task(run_episode(item)) for item in self.plan]
        try:
            indexed_outcomes: list[tuple[int, EmbodiedTrajectory | Exception]] = []
            for task in asyncio.as_completed(tasks):
                indexed_outcomes.append(await task)
                await self._emit_progress(
                    step=step,
                    indexed_outcomes=indexed_outcomes,
                )
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            await _finish_rollout(self.rollout, update=step)
        outcomes = [
            outcome
            for _index, outcome in sorted(indexed_outcomes, key=lambda item: item[0])
        ]
        trajectories = tuple(
            item for item in outcomes if isinstance(item, EmbodiedTrajectory)
        )
        outcome_rows = self._episode_outcome_rows(outcomes)
        metrics = _summarize_outcome_rows(outcome_rows)
        outcome_path = _write_evaluation_outcomes(
            config=self.config,
            step=step,
            rows=outcome_rows,
            protected_path=self.config.evaluation.baseline_outcomes_path,
        )
        artifacts = {"episode_outcomes_json": str(outcome_path)}
        if not trajectories:
            raise RuntimeError(
                "All fixed evaluation episodes failed before producing a trajectory; "
                f"inspect {outcome_path} instead of interpreting success_rate=0"
            )
        baseline_path = (
            self.config.evaluation.baseline_outcomes_path
            or self._measured_baseline_path
        )
        if (
            baseline_path is None
            and step == 0
            and self.config.evaluation.evaluate_before_training
        ):
            # The policy loaded before update zero is the actual SFT warm start.
            # Pin its immutable outcome report so later evaluations in this run
            # produce paired statistics without a separate baseline job.
            self._measured_baseline_path = outcome_path
            baseline_path = outcome_path
        compare_to_baseline = (
            baseline_path is not None
            and Path(baseline_path).expanduser().resolve() != outcome_path.resolve()
        )
        if compare_to_baseline:
            await _wait_for_outcome_report(
                baseline_path,
                timeout_seconds=(self.config.evaluation.baseline_wait_timeout_seconds),
            )
            paired_metrics = compare_paired_evaluation_reports(
                baseline_path,
                outcome_path,
            )
            metrics.update(
                {f"paired/{name}": value for name, value in paired_metrics.items()}
            )
            artifacts["baseline_episode_outcomes_json"] = str(baseline_path)
        evidence_path = _write_evaluation_evidence(
            config=self.config,
            step=step,
            outcome_path=outcome_path,
            baseline_path=baseline_path if compare_to_baseline else None,
            metrics=metrics,
            identity=identity,
        )
        artifacts["evaluation_evidence_json"] = str(evidence_path)
        return EvaluationResult(
            step=step,
            metrics=metrics,
            artifacts=artifacts,
            trajectories=trajectories,
        )

    async def _emit_progress(
        self,
        *,
        step: int,
        indexed_outcomes: Sequence[tuple[int, EmbodiedTrajectory | Exception]],
    ) -> None:
        if self.log_progress is None:
            return
        completed_trajectories = [
            outcome
            for _index, outcome in indexed_outcomes
            if isinstance(outcome, EmbodiedTrajectory)
        ]
        successes = [
            self._success_value(trajectory) for trajectory in completed_trajectories
        ]
        completed = len(indexed_outcomes)
        success_count = sum(successes)
        metrics: dict[str, float | int] = {
            "evaluation_completed_episodes": completed,
            "evaluation_failed_episodes": completed - len(completed_trajectories),
            "evaluation_success_count": success_count,
            "evaluation_success_denominator": completed,
            "evaluation_success_rate": success_count / completed,
        }
        await self.log_progress(
            ExperimentProgress(
                update=step,
                phase="evaluation",
                status="progress",
                completed=completed,
                total=len(self.plan),
                metrics=metrics,
            ),
            self.config,
        )

    def _success_value(self, trajectory: EmbodiedTrajectory) -> float:
        value: Any = trajectory.metrics.get(self.success_metric)
        if not isinstance(value, bool | int | float):
            raise ValueError(
                "Evaluation trajectory must expose a numeric/bool metric named "
                f"{self.success_metric!r}; task={trajectory.task!r}"
            )
        return float(value)

    def _episode_outcome_rows(
        self,
        outcomes: Sequence[EmbodiedTrajectory | Exception],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for episode, outcome in zip(self.plan, outcomes, strict=True):
            row: dict[str, Any] = {
                "episode": int(episode.index),
                "scenario_id": episode.scenario.id,
                "task": episode.scenario.task,
                "scenario_payload": _json_safe_value(episode.scenario.payload),
                "environment_seed": int(episode.environment_seed),
                "policy_seed": int(episode.policy_seed),
                "completed": isinstance(outcome, EmbodiedTrajectory),
                "success": 0.0,
                "reward": 0.0,
                "error_type": None,
                "error": None,
            }
            if isinstance(outcome, EmbodiedTrajectory):
                row["success"] = self._success_value(outcome)
                row["reward"] = float(outcome.reward)
                row["policy_termination_reason"] = outcome.metadata.get(
                    "policy_termination_reason"
                )
                action_chunks = [action.decoded for action in outcome.actions]
                row["action_count"] = len(action_chunks)
                row["action_chunk_sha256s"] = [
                    _canonical_json_sha256(chunk) for chunk in action_chunks
                ]
                row["action_sequence_sha256"] = _canonical_json_sha256(action_chunks)
                row["episode_steps"] = int(
                    outcome.metrics.get("episode_steps", len(outcome.actions))
                )
                row["duration_seconds"] = outcome.metrics.get("duration")
                row["reset_info"] = _json_safe_value(
                    outcome.metadata.get("reset_info", {})
                )
            else:
                row["error_type"] = type(outcome).__name__
                row["error"] = str(outcome)[:500]
                row["episode_steps"] = 0
                row["action_count"] = 0
                row["action_chunk_sha256s"] = []
                row["action_sequence_sha256"] = None
                row["duration_seconds"] = None
                row["reset_info"] = None
            rows.append(row)
        return rows


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        _json_safe_value(value),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _evaluation_seed(
    *,
    seed_contract: dict[str, Any],
    kind: str,
    configured_seed: int,
    scenario_id: str,
    repetition: int,
) -> int:
    mode = str(seed_contract.get(f"{kind}_mode", "derived"))
    if mode == "derived":
        return _stable_seed(
            configured_seed,
            scenario_id,
            repetition,
            f"held-out-{kind}",
        )
    if mode == "configured":
        return int(configured_seed)
    if mode == "fixed":
        key = f"fixed_{kind}_seed"
        if key not in seed_contract:
            raise ValueError(
                f"evaluation.kwargs.seed_contract.{key} is required for fixed mode"
            )
        return int(seed_contract[key])
    raise ValueError(
        f"Unsupported evaluation {kind}_mode={mode!r}; "
        "expected derived, configured, or fixed"
    )


def _json_safe_value(value: Any) -> Any:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)
    return value


def compare_paired_evaluation_reports(
    baseline_path: str | Path,
    candidate_path: str | Path,
) -> dict[str, float]:
    """Compare two fixed-evaluation reports episode by episode.

    The pairing key includes scenario and both environment/policy seeds. A
    mismatch fails closed rather than comparing two different reset streams.
    """

    baseline = _load_evaluation_outcomes(baseline_path)
    candidate = _load_evaluation_outcomes(candidate_path)
    baseline_rows = baseline["episodes"]
    candidate_rows = candidate["episodes"]
    if len(baseline_rows) != len(candidate_rows):
        raise ValueError(
            "Paired evaluation reports have different episode counts: "
            f"baseline={len(baseline_rows)}, candidate={len(candidate_rows)}"
        )

    improved = 0
    regressed = 0
    unchanged_success = 0
    unchanged_failure = 0
    baseline_successes = 0
    candidate_successes = 0
    paired_differences: list[float] = []
    task_differences: dict[str, list[float]] = {}
    task_baseline_successes: dict[str, int] = {}
    task_candidate_successes: dict[str, int] = {}
    task_episode_counts: dict[str, int] = {}
    pairing_fields = ("episode", "scenario_id", "environment_seed", "policy_seed")
    for baseline_row, candidate_row in zip(
        baseline_rows,
        candidate_rows,
        strict=True,
    ):
        baseline_key = tuple(baseline_row.get(field) for field in pairing_fields)
        candidate_key = tuple(candidate_row.get(field) for field in pairing_fields)
        if baseline_key != candidate_key:
            raise ValueError(
                "Paired evaluation episode mismatch: "
                f"baseline={baseline_key!r}, candidate={candidate_key!r}"
            )
        baseline_success = _binary_success(baseline_row.get("success"))
        candidate_success = _binary_success(candidate_row.get("success"))
        baseline_successes += baseline_success
        candidate_successes += candidate_success
        difference = float(candidate_success - baseline_success)
        paired_differences.append(difference)
        task = str(baseline_row.get("task") or baseline_row.get("scenario_id"))
        candidate_task = str(
            candidate_row.get("task") or candidate_row.get("scenario_id")
        )
        if task != candidate_task:
            raise ValueError(
                "Paired evaluation task mismatch: "
                f"baseline={task!r}, candidate={candidate_task!r}"
            )
        task_differences.setdefault(task, []).append(difference)
        task_baseline_successes[task] = (
            task_baseline_successes.get(task, 0) + baseline_success
        )
        task_candidate_successes[task] = (
            task_candidate_successes.get(task, 0) + candidate_success
        )
        task_episode_counts[task] = task_episode_counts.get(task, 0) + 1
        if not baseline_success and candidate_success:
            improved += 1
        elif baseline_success and not candidate_success:
            regressed += 1
        elif baseline_success:
            unchanged_success += 1
        else:
            unchanged_failure += 1

    episodes = len(baseline_rows)
    lift_ci_low, lift_ci_high = _paired_bootstrap_interval(paired_differences)
    task_ci_low, task_ci_high = _task_stratified_bootstrap_interval(task_differences)
    task_baseline_rates = {
        task: task_baseline_successes[task] / task_episode_counts[task]
        for task in task_episode_counts
    }
    task_candidate_rates = {
        task: task_candidate_successes[task] / task_episode_counts[task]
        for task in task_episode_counts
    }
    metrics = {
        "episodes": float(episodes),
        "baseline_success_count": float(baseline_successes),
        "baseline_success_rate": baseline_successes / episodes,
        "candidate_success_count": float(candidate_successes),
        "candidate_success_rate": candidate_successes / episodes,
        "success_rate_lift": (candidate_successes - baseline_successes) / episodes,
        "success_rate_lift_ci95_low": lift_ci_low,
        "success_rate_lift_ci95_high": lift_ci_high,
        "improved_pairs": float(improved),
        "regressed_pairs": float(regressed),
        "unchanged_success_pairs": float(unchanged_success),
        "unchanged_failure_pairs": float(unchanged_failure),
        "mcnemar_exact_p_value": _mcnemar_exact_p_value(improved, regressed),
        "task_count": float(len(task_episode_counts)),
        "baseline_task_macro_success_rate": fmean(task_baseline_rates.values()),
        "candidate_task_macro_success_rate": fmean(task_candidate_rates.values()),
        "task_macro_success_rate_lift": (
            fmean(task_candidate_rates.values()) - fmean(task_baseline_rates.values())
        ),
        "task_macro_success_rate_lift_ci95_low": task_ci_low,
        "task_macro_success_rate_lift_ci95_high": task_ci_high,
    }
    for task in sorted(task_episode_counts):
        segment = _metric_segment(task)
        metrics[f"task/{segment}/episodes"] = float(task_episode_counts[task])
        metrics[f"task/{segment}/baseline_success_rate"] = task_baseline_rates[task]
        metrics[f"task/{segment}/candidate_success_rate"] = task_candidate_rates[task]
        metrics[f"task/{segment}/success_rate_lift"] = (
            task_candidate_rates[task] - task_baseline_rates[task]
        )
    return metrics


def summarize_evaluation_report(path: str | Path) -> dict[str, float]:
    """Regenerate aggregate, scenario, and task metrics from raw outcomes."""

    payload = _load_evaluation_outcomes(path)
    return _summarize_outcome_rows(payload["episodes"])


def validate_paired_evaluation_configs(
    baseline: EmbodiedExperimentConfig,
    candidate: EmbodiedExperimentConfig,
    *,
    baseline_step: int = 0,
) -> dict[str, Any]:
    """Fail closed unless two configs define the same native evaluation.

    Model/checkpoint identity is the treatment and may differ, as may training
    geometry and LoRA state. The policy inference runtime, environment,
    reset/processor contract, reward interpretation, episode horizon, and
    fixed evaluation plan may not.
    """

    if baseline_step < 0:
        raise ValueError("baseline_step must be non-negative")
    if not baseline.evaluation.enabled or not candidate.evaluation.enabled:
        raise ValueError("Both paired-evaluation configs must enable evaluation")
    if baseline.evaluation.baseline_outcomes_path is not None:
        raise ValueError(
            "The baseline config cannot reference another baseline outcome report"
        )

    baseline_report = (
        baseline.storage.output_dir
        / "evaluation"
        / f"update_{baseline_step:06d}_episode_outcomes.json"
    )
    configured_report = candidate.evaluation.baseline_outcomes_path
    if configured_report is None:
        raise ValueError(
            "The candidate config must set evaluation.baseline_outcomes_path"
        )

    mismatches: list[str] = []
    baseline_path = baseline_report.expanduser().resolve()
    candidate_path = configured_report.expanduser().resolve()
    if baseline_path != candidate_path:
        mismatches.append(
            "evaluation.baseline_outcomes_path: "
            f"expected {baseline_report}, got {configured_report}"
        )

    baseline_contract = _paired_evaluation_contract(baseline)
    candidate_contract = _paired_evaluation_contract(candidate)
    for field in baseline_contract:
        if baseline_contract[field] != candidate_contract[field]:
            mismatches.append(
                f"{field}: baseline={baseline_contract[field]!r}, "
                f"candidate={candidate_contract[field]!r}"
            )

    baseline_wandb = baseline.observability.wandb
    candidate_wandb = candidate.observability.wandb
    if baseline_wandb.enabled and candidate_wandb.enabled:
        if baseline_wandb.project != candidate_wandb.project:
            mismatches.append(
                "observability.wandb.project: "
                f"baseline={baseline_wandb.project!r}, "
                f"candidate={candidate_wandb.project!r}"
            )
        if baseline_wandb.group != candidate_wandb.group:
            mismatches.append(
                "observability.wandb.group: "
                f"baseline={baseline_wandb.group!r}, "
                f"candidate={candidate_wandb.group!r}"
            )

    if mismatches:
        raise ValueError(
            "Baseline and candidate do not define an identical paired evaluation:\n- "
            + "\n- ".join(mismatches)
        )

    return {
        "baseline_config_fingerprint": baseline.fingerprint,
        "candidate_config_fingerprint": candidate.fingerprint,
        "baseline_policy_path": baseline.policy.path,
        "baseline_policy_revision": baseline.policy.revision,
        "candidate_policy_path": candidate.policy.path,
        "candidate_policy_revision": candidate.policy.revision,
        "baseline_report_path": str(baseline_report),
        "baseline_step": baseline_step,
        "episodes": baseline.evaluation.episodes,
        "fixed_scenarios": list(baseline.evaluation.fixed_scenarios),
        "seeds": list(baseline.evaluation.seeds),
        "runtime": baseline.evaluation.runtime,
        "split": baseline.evaluation.split,
    }


def create_paired_evaluation_candidate(
    baseline: EmbodiedExperimentConfig,
    *,
    run: str,
    output_dir: str | Path,
    peft_adapter_path: str | Path | None = None,
    policy_path: str | None = None,
    policy_revision: str | None = None,
    baseline_step: int = 0,
    baseline_wait_timeout_seconds: int = 0,
) -> EmbodiedExperimentConfig:
    """Derive a claim-facing candidate without changing evaluation controls.

    Exactly one treatment is selected: a LoRA adapter on the baseline model or
    a complete candidate checkpoint. The returned config is validated against
    the baseline before it is exposed to callers.
    """

    if not run:
        raise ValueError("run cannot be empty")
    if baseline_step < 0:
        raise ValueError("baseline_step must be non-negative")
    if baseline_wait_timeout_seconds < 0:
        raise ValueError("baseline_wait_timeout_seconds must be non-negative")
    if baseline.evaluation.baseline_outcomes_path is not None:
        raise ValueError("baseline cannot reference another outcome report")
    has_adapter = peft_adapter_path is not None
    has_policy = policy_path is not None
    if has_adapter == has_policy:
        raise ValueError("Specify exactly one of peft_adapter_path or policy_path")
    if policy_revision is not None and not has_policy:
        raise ValueError("policy_revision requires policy_path")

    candidate_output = Path(output_dir).expanduser()
    if candidate_output.resolve() == baseline.storage.output_dir.expanduser().resolve():
        raise ValueError("candidate output_dir must differ from baseline output_dir")

    load_kwargs = dict(baseline.policy.load_kwargs)
    if has_adapter:
        assert peft_adapter_path is not None
        load_kwargs["peft_adapter_path"] = str(Path(peft_adapter_path).expanduser())
        candidate_policy_path = baseline.policy.path
        candidate_revision = baseline.policy.revision
    else:
        load_kwargs["peft_adapter_path"] = None
        candidate_policy_path = str(policy_path)
        candidate_revision = policy_revision

    baseline_report = (
        baseline.storage.output_dir
        / "evaluation"
        / f"update_{baseline_step:06d}_episode_outcomes.json"
    )
    candidate = baseline.model_copy(
        update={
            "experiment": baseline.experiment.model_copy(update={"run": run}),
            "policy": baseline.policy.model_copy(
                update={
                    "path": candidate_policy_path,
                    "revision": candidate_revision,
                    "load_kwargs": load_kwargs,
                }
            ),
            "evaluation": baseline.evaluation.model_copy(
                update={
                    "baseline_outcomes_path": baseline_report,
                    "baseline_wait_timeout_seconds": baseline_wait_timeout_seconds,
                }
            ),
            "storage": baseline.storage.model_copy(
                update={"output_dir": candidate_output}
            ),
        }
    )
    validate_paired_evaluation_configs(
        baseline,
        candidate,
        baseline_step=baseline_step,
    )
    return candidate


def _paired_evaluation_contract(config: EmbodiedExperimentConfig) -> dict[str, Any]:
    policy = config.policy.model_dump(mode="json")
    # Model/checkpoint identity and adapter weights are the treatment. Keep the
    # loader/runtime contract fixed so a weight change cannot hide an inference
    # protocol change. Logprob batching only affects training and rescoring.
    policy.pop("path")
    policy.pop("revision")
    load_kwargs = dict(policy["load_kwargs"])
    load_kwargs.pop("peft_adapter_path", None)
    load_kwargs.pop("logprob_batch_size", None)
    policy["load_kwargs"] = load_kwargs
    policy.pop("rollout_generation")
    policy.pop("train_generation")
    # These look like training-only controls, but policy construction currently
    # consumes them before evaluation. Raw-parameter selection can promote a
    # subset of base weights to FP32, while LoRA selection controls which
    # adapter tensors are attached and promoted. Both can change native actions,
    # so paired evaluation must fail closed when they differ.

    evaluation = config.evaluation.model_dump(mode="json")
    # Scheduling the baseline before optimization does not change the native
    # evaluation itself. Standalone baselines and training-integrated
    # candidates intentionally differ here while preserving identical policy,
    # scenario, seed, and sampler contracts.
    evaluation.pop("evaluate_before_training")
    evaluation.pop("baseline_outcomes_path")
    evaluation.pop("baseline_wait_timeout_seconds")
    evaluation.pop("every_updates")
    evaluation.pop("checkpoint_selection")
    evaluation_kwargs = dict(evaluation["kwargs"])
    # This is a fail-closed CLI loading guard for trained candidates, not part
    # of the native scenario/seed/sampler treatment contract.
    evaluation_kwargs.pop("require_policy_checkpoint", None)
    evaluation["kwargs"] = evaluation_kwargs

    return {
        "policy_evaluation": policy,
        "environment": config.environment.model_dump(mode="json"),
        "reward": config.reward.model_dump(mode="json"),
        "rollout.max_episode_steps": config.rollout.max_episode_steps,
        "rollout.max_policy_steps": config.rollout.max_policy_steps,
        "evaluation": evaluation,
    }


async def _wait_for_outcome_report(
    path: str | Path,
    *,
    timeout_seconds: int,
) -> None:
    """Wait for a concurrently scheduled baseline without busy-waiting."""

    source = Path(path).expanduser()
    if source.is_file() or timeout_seconds == 0:
        return
    deadline = time.monotonic() + timeout_seconds
    while not source.is_file():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out waiting for paired baseline outcome report: {source}"
            )
        await asyncio.sleep(min(1.0, remaining))


def _write_evaluation_outcomes(
    *,
    config: EmbodiedExperimentConfig,
    step: int,
    rows: list[dict[str, Any]],
    protected_path: str | Path | None = None,
) -> Path:
    path = (
        config.storage.output_dir
        / "evaluation"
        / f"update_{int(step):06d}_episode_outcomes.json"
    )
    if (
        protected_path is not None
        and Path(protected_path).expanduser().resolve() == path.resolve()
    ):
        raise ValueError(
            "Candidate evaluation output would overwrite its paired baseline "
            f"artifact: {path}"
        )
    payload = {
        "schema_version": 1,
        "config_fingerprint": config.fingerprint,
        "step": int(step),
        "split": config.evaluation.split,
        "data_role": config.evaluation.data_role,
        "episodes": rows,
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    max_bytes = config.storage.max_log_file_mb * 1024 * 1024
    if len(encoded) > max_bytes:
        raise RuntimeError(
            "Evaluation outcome report exceeds storage.max_log_file_mb: "
            f"size={len(encoded)}, max={max_bytes}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == encoded:
            return path
        raise FileExistsError(
            "Refusing to overwrite an existing fixed-evaluation report with "
            "different outcomes. Use a distinct storage.output_dir or "
            f"evaluation step: {path}"
        )
    temporary = path.with_suffix(".json.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return path


def _write_evaluation_evidence(
    *,
    config: EmbodiedExperimentConfig,
    step: int,
    outcome_path: Path,
    baseline_path: str | Path | None,
    metrics: dict[str, float],
    identity: dict[str, Any],
) -> Path:
    """Write the machine-regenerable index for one fixed evaluation."""

    path = outcome_path.with_name(f"update_{int(step):06d}_evidence.json")
    baseline = Path(baseline_path).expanduser() if baseline_path is not None else None
    payload = {
        "schema_version": 2,
        "kind": "art_embodied_evaluation_evidence",
        "step": int(step),
        "config_fingerprint": config.fingerprint,
        "resolved_config": config.model_dump(mode="json"),
        "config_consumption": config.consumption_report(),
        "identity": identity,
        "policy": {
            "type": config.policy.type,
            "path": config.policy.path,
            "revision": config.policy.revision,
            "adapter_path": config.policy.load_kwargs.get("peft_adapter_path"),
        },
        "evaluation": {
            "split": config.evaluation.split,
            "data_role": config.evaluation.data_role,
            "runtime": config.evaluation.runtime,
            "episodes": config.evaluation.episodes,
            "fixed_scenarios": list(config.evaluation.fixed_scenarios),
            "seeds": list(config.evaluation.seeds),
        },
        "outcomes": {
            "path": outcome_path.name,
            "sha256": _file_sha256(outcome_path),
        },
        "baseline_outcomes": (
            {
                "path": str(baseline),
                "sha256": _file_sha256(baseline),
            }
            if baseline is not None
            else None
        ),
        "metrics": dict(sorted(metrics.items())),
        "statistics": {
            "paired_bootstrap_samples": 10_000,
            "paired_bootstrap_seed": 20260720,
            "confidence_level": 0.95,
        },
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    max_bytes = config.storage.max_log_file_mb * 1024 * 1024
    if len(encoded) > max_bytes:
        raise RuntimeError(
            "Evaluation evidence exceeds storage.max_log_file_mb: "
            f"size={len(encoded)}, max={max_bytes}"
        )
    if path.exists():
        if path.read_bytes() == encoded:
            return path
        raise FileExistsError(
            f"Refusing to overwrite an existing evaluation evidence bundle: {path}"
        )
    temporary = path.with_suffix(".json.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return path


def _evaluation_manifest_path(config: EmbodiedExperimentConfig) -> str | None:
    sealed_value = config.evaluation.kwargs.get("sealed_manifest_path")
    replication_value = config.evaluation.kwargs.get("replication_manifest_path")
    configured = [
        ("sealed", sealed_value),
        ("replication", replication_value),
    ]
    selected = [(label, value) for label, value in configured if value is not None]
    if len(selected) > 1:
        raise ValueError(
            "evaluation may reference either a sealed or development replication "
            "manifest, not both"
        )
    if selected:
        label, value = selected[0]
        key = f"{label}_manifest_path"
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"evaluation.kwargs.{key} must be a non-empty path")
        source = Path(value).expanduser()
        if not source.is_file():
            raise FileNotFoundError(
                f"{label.title()} evaluation manifest is missing: {source}"
            )
        payload = json.loads(source.read_text(encoding="utf-8"))
        expected_kind = (
            "art_embodied_sealed_evaluation_manifest"
            if label == "sealed"
            else "art_embodied_development_replication_manifest"
        )
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != 1
            or payload.get("kind") != expected_kind
        ):
            raise ValueError(f"Invalid {label} evaluation manifest: {source}")
        expected = {
            "split": config.evaluation.split,
            "data_role": config.evaluation.data_role,
            "episodes": config.evaluation.episodes,
            "seeds": config.evaluation.seeds,
            "fixed_scenarios": config.evaluation.fixed_scenarios,
            "seed_contract": config.evaluation.kwargs.get("seed_contract", {}),
        }
        mismatches = {
            key: {"manifest": payload.get(key), "config": value}
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"{label.title()} evaluation manifest does not match the executable "
                "config: " + json.dumps(mismatches, sort_keys=True)
            )
        excluded_key = (
            "excluded_development_seeds"
            if label == "sealed"
            else "excluded_prior_seeds"
        )
        excluded = payload.get(excluded_key, [])
        if not isinstance(excluded, list) or any(
            not isinstance(value, int) for value in excluded
        ):
            raise ValueError(f"{label} manifest {excluded_key} must be integer seeds")
        overlap = sorted(set(excluded).intersection(config.evaluation.seeds))
        if overlap:
            raise ValueError(
                f"{label.title()} evaluation seeds overlap excluded prior seeds: "
                + ", ".join(map(str, overlap))
            )
        return str(source)
    value = config.environment.kwargs.get("evaluation_state_manifest")
    if value is None:
        if config.evaluation.data_role == "sealed_test":
            raise ValueError(
                "sealed_test requires evaluation.kwargs.sealed_manifest_path or "
                "environment.kwargs.evaluation_state_manifest"
            )
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "environment.kwargs.evaluation_state_manifest must be a non-empty "
            "path string when configured"
        )
    return value


def _load_evaluation_outcomes(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Evaluation outcome report not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported evaluation outcome schema: {source}")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"Evaluation outcome report has no episodes: {source}")
    return payload


def _binary_success(value: Any) -> int:
    numeric = float(value)
    if numeric not in (0.0, 1.0):
        raise ValueError(
            f"Paired success comparison requires binary success values, got {numeric}"
        )
    return int(numeric)


def _success_rates_by_field(
    rows: Sequence[dict[str, Any]],
    *,
    field: str,
) -> dict[str, tuple[float, int]]:
    counts: dict[str, list[float | int]] = {}
    for row in rows:
        key = str(row.get(field) or row.get("scenario_id") or "unnamed")
        values = counts.setdefault(key, [0.0, 0])
        values[0] += _binary_success(row.get("success"))
        values[1] += 1
    return {key: (float(values[0]), int(values[1])) for key, values in counts.items()}


def _summarize_outcome_rows(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Evaluation summary requires at least one episode")
    episodes = len(rows)
    completed_rows = [row for row in rows if bool(row.get("completed", True))]
    successes = [_binary_success(row.get("success")) for row in rows]
    rewards = [float(row.get("reward", 0.0)) for row in rows]
    metrics = {
        "episodes": float(episodes),
        "completed_episodes": float(len(completed_rows)),
        "failed_episodes": float(episodes - len(completed_rows)),
        "completion_rate": len(completed_rows) / episodes,
        "reward_mean": sum(rewards) / episodes,
        "success_rate": sum(successes) / episodes,
    }
    if completed_rows:
        metrics["completed_reward_mean"] = fmean(
            float(row.get("reward", 0.0)) for row in completed_rows
        )
    for field, prefix in (("scenario_id", "scenario"), ("task", "task")):
        rates = _success_rates_by_field(rows, field=field)
        metric_segments: dict[str, str] = {}
        for key, (success_count, episode_count) in rates.items():
            segment = _metric_segment(key)
            previous = metric_segments.setdefault(segment, key)
            if previous != key:
                raise ValueError(
                    f"Evaluation {field} values collide after metric normalization: "
                    f"{previous!r} and {key!r}"
                )
            metrics[f"{prefix}/{segment}/episodes"] = float(episode_count)
            metrics[f"{prefix}/{segment}/success_rate"] = success_count / episode_count
        metrics[f"{prefix}_macro_success_rate"] = fmean(
            success_count / episode_count
            for success_count, episode_count in rates.values()
        )
    for payload_field, prefix in (
        ("perturbation_category", "category"),
        ("perturbation_difficulty", "difficulty"),
    ):
        grouped: dict[str, list[int]] = {}
        for row, success in zip(rows, successes, strict=True):
            scenario_payload = row.get("scenario_payload")
            if not isinstance(scenario_payload, dict):
                continue
            value = scenario_payload.get(payload_field)
            if value is None:
                continue
            grouped.setdefault(str(value), []).append(success)
        for value, group_successes in grouped.items():
            segment = _metric_segment(value)
            metrics[f"{prefix}/{segment}/episodes"] = float(len(group_successes))
            metrics[f"{prefix}/{segment}/success_rate"] = sum(group_successes) / len(
                group_successes
            )
    return metrics


def _paired_bootstrap_interval(
    differences: Sequence[float],
    *,
    samples: int = 10_000,
    seed: int = 20260720,
) -> tuple[float, float]:
    if not differences:
        raise ValueError("Paired bootstrap requires at least one episode")
    generator = random.Random(seed)
    count = len(differences)
    estimates = sorted(
        sum(differences[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    )
    return (_percentile(estimates, 0.025), _percentile(estimates, 0.975))


def _task_stratified_bootstrap_interval(
    differences_by_task: dict[str, list[float]],
    *,
    samples: int = 10_000,
    seed: int = 20260720,
) -> tuple[float, float]:
    if not differences_by_task:
        raise ValueError("Task-stratified bootstrap requires at least one task")
    generator = random.Random(seed)
    task_rows = [differences_by_task[key] for key in sorted(differences_by_task)]
    estimates = []
    for _ in range(samples):
        task_means = []
        for differences in task_rows:
            count = len(differences)
            task_means.append(
                sum(differences[generator.randrange(count)] for _ in range(count))
                / count
            )
        estimates.append(fmean(task_means))
    estimates.sort()
    return (_percentile(estimates, 0.025), _percentile(estimates, 0.975))


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("Percentile requires at least one value")
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mcnemar_exact_p_value(improved: int, regressed: int) -> float:
    discordant = int(improved) + int(regressed)
    if discordant == 0:
        return 1.0
    tail = min(int(improved), int(regressed))
    probability = sum(math.comb(discordant, index) for index in range(tail + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * probability)


def _metric_segment(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return normalized or "unnamed"
