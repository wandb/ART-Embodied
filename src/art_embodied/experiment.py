"""ART-style orchestration for LeRobot-owned embodied rollouts.

The orchestration deliberately knows nothing about a particular simulator or
policy family.  A caller supplies scenarios and a rollout function using its
native LeRobot environment.  ART supplies deterministic grouping, bounded
concurrency, backend training, evaluation cadence, and logging hooks.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
import time
from typing import Any, Literal, Protocol

import pydantic

from .config import EmbodiedExperimentConfig
from .trajectories import EmbodiedTrajectory, EmbodiedTrajectoryGroup
from .types import TrainResult

_TRAINING_PROGRESS_INTERVAL_SECONDS = 60.0


class EmbodiedScenario(pydantic.BaseModel):
    """A replayable task/reset specification owned by the user environment."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    id: str = pydantic.Field(min_length=1)
    task: str = pydantic.Field(min_length=1)
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RolloutContext:
    """Deterministic coordinates for one policy attempt within an update."""

    update: int
    group_index: int
    attempt_index: int
    environment_seed: int
    policy_seed: int
    config_fingerprint: str


class RolloutFunction(Protocol):
    """Produce one trajectory from a native environment and policy."""

    def __call__(
        self,
        scenario: EmbodiedScenario,
        context: RolloutContext,
    ) -> Awaitable[EmbodiedTrajectory]: ...


class GroupRolloutFunction(Protocol):
    """Optional vectorized rollout interface for a same-reset comparison group."""

    def rollout_group(
        self,
        scenario: EmbodiedScenario,
        contexts: Sequence[RolloutContext],
    ) -> Awaitable[Sequence[EmbodiedTrajectory]]: ...


class EmbodiedTrainingBackend(Protocol):
    """Minimal lifecycle implemented by an embodied policy optimizer."""

    async def train(
        self,
        trajectory_groups: Sequence[EmbodiedTrajectoryGroup],
        **kwargs: Any,
    ) -> TrainResult: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Metrics and evidence produced by a fixed policy-version evaluation."""

    step: int
    metrics: dict[str, float]
    artifacts: dict[str, str]
    trajectories: tuple[EmbodiedTrajectory, ...] = ()


EvaluationFunction = Callable[
    [int, TrainResult, EmbodiedExperimentConfig],
    Awaitable[EvaluationResult],
]
StepLogger = Callable[
    [
        int,
        Sequence[EmbodiedTrajectoryGroup],
        TrainResult,
        EvaluationResult | None,
        EmbodiedExperimentConfig,
    ],
    Awaitable[None],
]
RolloutLogger = Callable[
    [
        int,
        Sequence[EmbodiedTrajectoryGroup],
        EmbodiedExperimentConfig,
    ],
    Awaitable[None],
]


@dataclass(frozen=True, slots=True)
class ExperimentProgress:
    """One live progress event emitted before an update is fully committed."""

    update: int
    phase: Literal[
        "initialization", "update", "rollout", "training", "evaluation", "logging"
    ]
    status: Literal["started", "progress", "completed", "failed"]
    completed: int | None = None
    total: int | None = None
    metrics: dict[str, float | int] | None = None
    message: str | None = None


ProgressLogger = Callable[
    [ExperimentProgress, EmbodiedExperimentConfig],
    Awaitable[None],
]


@dataclass(frozen=True, slots=True)
class ExperimentStepResult:
    """Committed training and optional evaluation result for one update."""

    update: int
    training: TrainResult
    evaluation: EvaluationResult | None


class EmbodiedExperiment:
    """Collect grouped rollouts and train an embodied ART backend.

    Environment reset behavior remains inside ``rollout`` and must be described
    in ``config.environment.reset``.  The same environment seed is used for all
    attempts in a group, while each attempt gets a distinct policy seed.  This
    gives group-relative algorithms comparable initial conditions without
    forcing a simulator-specific reset API into ART.
    """

    def __init__(
        self,
        *,
        config: EmbodiedExperimentConfig,
        scenarios: Sequence[EmbodiedScenario],
        rollout: RolloutFunction,
        backend: EmbodiedTrainingBackend,
        evaluate: EvaluationFunction | None = None,
        log_rollout: RolloutLogger | None = None,
        log_step: StepLogger | None = None,
        log_progress: ProgressLogger | None = None,
    ) -> None:
        if not scenarios:
            raise ValueError("EmbodiedExperiment requires at least one scenario")
        if config.evaluation.enabled and evaluate is None:
            raise ValueError(
                "evaluation.enabled=true requires an explicit evaluation function"
            )
        self.config = config
        self.scenarios = tuple(scenarios)
        self.rollout = rollout
        self.backend = backend
        self.evaluate = evaluate
        self.log_rollout = log_rollout
        self.log_step = log_step
        self.log_progress = log_progress

    async def collect(self, *, update: int) -> list[EmbodiedTrajectoryGroup]:
        rollout_started_at = time.monotonic()
        await _prepare_rollout(self.rollout, update=update)
        try:
            return await self._collect_prepared(
                update=update,
                rollout_started_at=rollout_started_at,
            )
        finally:
            await _finish_rollout(self.rollout, update=update)

    async def _collect_prepared(
        self,
        *,
        update: int,
        rollout_started_at: float,
    ) -> list[EmbodiedTrajectoryGroup]:
        semaphore = asyncio.Semaphore(self.config.rollout.workers)
        group_count = (
            self.config.rollout.groups_per_update
            * self.config.rollout.epochs_per_update
        )

        async def collect_group(group_index: int) -> EmbodiedTrajectoryGroup:
            scenario_index = (update * group_count + group_index) % len(self.scenarios)
            scenario = self.scenarios[scenario_index]
            environment_seed = _stable_seed(
                self.config.experiment.seed,
                update,
                group_index,
                "environment",
            )

            contexts = tuple(
                RolloutContext(
                    update=update,
                    group_index=group_index,
                    attempt_index=attempt_index,
                    environment_seed=environment_seed,
                    policy_seed=_stable_seed(
                        self.config.experiment.seed,
                        update,
                        group_index,
                        attempt_index,
                        "policy",
                    ),
                    config_fingerprint=self.config.fingerprint,
                )
                for attempt_index in range(self.config.algorithm.group_size)
            )

            async def collect_attempt(
                context: RolloutContext,
            ) -> EmbodiedTrajectory | Exception:
                try:
                    async with semaphore:
                        trajectory = await self.rollout(scenario, context)
                except Exception as exc:
                    return exc
                _finalize_rollout_metadata(
                    trajectory,
                    scenario=scenario,
                    context=context,
                )
                self._discard_unrequired_observation_values(trajectory)
                return trajectory

            collect_group_batch = getattr(self.rollout, "rollout_group", None)
            if callable(collect_group_batch) and bool(
                getattr(self.rollout, "supports_group_rollout", False)
            ):
                async with semaphore:
                    batch_result = collect_group_batch(scenario, contexts)
                    if inspect.isawaitable(batch_result):
                        batch_result = await batch_result
                attempts = list(batch_result)
                if len(attempts) != len(contexts):
                    raise RuntimeError(
                        "rollout_group returned the wrong number of trajectories: "
                        f"returned={len(attempts)}, expected={len(contexts)}"
                    )
                for trajectory, context in zip(attempts, contexts, strict=True):
                    if not isinstance(trajectory, EmbodiedTrajectory):
                        raise TypeError(
                            "rollout_group must return EmbodiedTrajectory values, "
                            f"got {type(trajectory).__name__}"
                        )
                    _finalize_rollout_metadata(
                        trajectory,
                        scenario=scenario,
                        context=context,
                    )
                    self._discard_unrequired_observation_values(trajectory)
            else:
                attempts = await asyncio.gather(
                    *(collect_attempt(context) for context in contexts)
                )
            group = EmbodiedTrajectoryGroup(
                attempts,
                metadata={
                    "scenario_id": scenario.id,
                    "scenario_index": scenario_index,
                    "update": update,
                    "group_index": group_index,
                    "environment_seed": environment_seed,
                    "config_fingerprint": self.config.fingerprint,
                },
            )
            if not group.trajectories:
                raise RuntimeError(
                    f"All rollouts failed for scenario={scenario.id!r}, "
                    f"update={update}, group={group_index}"
                )
            if self.config.rollout.failure_policy == "fail_update" and group.exceptions:
                raise RuntimeError(
                    "A rollout failed under failure_policy='fail_update': "
                    f"scenario={scenario.id!r}, update={update}, "
                    f"group={group_index}, completed={len(group.trajectories)}, "
                    f"failed={len(group.exceptions)}"
                )
            minimum = self.config.rollout.minimum_completed_attempts_per_group
            if len(group.trajectories) < minimum:
                raise RuntimeError(
                    "Too few completed attempts for a trainable group: "
                    f"scenario={scenario.id!r}, update={update}, "
                    f"group={group_index}, completed={len(group.trajectories)}, "
                    f"required={minimum}"
                )
            return group

        tasks = [
            asyncio.create_task(collect_group(group_index))
            for group_index in range(group_count)
        ]
        completed_groups: list[EmbodiedTrajectoryGroup] = []
        try:
            for task in asyncio.as_completed(tasks):
                group = await task
                completed_groups.append(group)
                metrics = _rollout_progress_metrics(completed_groups)
                metrics.update(
                    _rollout_throughput_metrics(
                        completed_groups,
                        elapsed_seconds=time.monotonic() - rollout_started_at,
                    )
                )
                await self._emit_progress(
                    ExperimentProgress(
                        update=update + 1,
                        phase="rollout",
                        status="progress",
                        completed=len(completed_groups),
                        total=group_count,
                        metrics=metrics,
                    )
                )
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return sorted(
            completed_groups,
            key=lambda group: int(group.metadata["group_index"]),
        )

    def _discard_unrequired_observation_values(
        self,
        trajectory: EmbodiedTrajectory,
    ) -> None:
        if (
            self.config.storage.retain_rollout_payloads
            or self.config.rollout.action_payload.require_observation
        ):
            return
        trajectory.discard_observation_values()

    async def train_step(self, *, update: int) -> ExperimentStepResult:
        completed_update = update + 1
        await self._emit_progress(
            ExperimentProgress(
                update=completed_update,
                phase="update",
                status="started",
            )
        )
        try:
            group_count = (
                self.config.rollout.groups_per_update
                * self.config.rollout.epochs_per_update
            )
            await self._emit_progress(
                ExperimentProgress(
                    update=completed_update,
                    phase="rollout",
                    status="started",
                    completed=0,
                    total=group_count,
                )
            )
            rollout_started_at = time.monotonic()
            groups = await self.collect(update=update)
            rollout_elapsed_seconds = time.monotonic() - rollout_started_at
            rollout_lifecycle_metrics = _rollout_lifecycle_metrics(self.rollout)
            rollout_metrics = _rollout_progress_metrics(groups)
            rollout_metrics.update(
                _rollout_throughput_metrics(
                    groups,
                    elapsed_seconds=rollout_elapsed_seconds,
                )
            )
            rollout_memory_metrics = _process_memory_metrics()
            if self.log_rollout is not None:
                # ``update`` is the policy version that generated these
                # trajectories. Log before optimization so initial SFT
                # behavior is correctly placed at policy version zero.
                await self.log_rollout(update, groups, self.config)
            await self._emit_progress(
                ExperimentProgress(
                    update=completed_update,
                    phase="rollout",
                    status="completed",
                    completed=group_count,
                    total=group_count,
                    metrics=rollout_metrics,
                )
            )
            await self._emit_progress(
                ExperimentProgress(
                    update=completed_update,
                    phase="training",
                    status="started",
                )
            )
            training_started_at = time.monotonic()
            train_result = await self._train_with_progress(
                groups,
                completed_update=completed_update,
                started_at=training_started_at,
            )
            training_elapsed_seconds = time.monotonic() - training_started_at
            if not self.config.storage.retain_rollout_payloads:
                _discard_transient_rollout_payloads(groups)
            training_memory_metrics = _process_memory_metrics()
            train_result.metrics.setdefault(
                "rollout/elapsed_seconds",
                rollout_elapsed_seconds,
            )
            train_result.metrics.setdefault(
                "rollout/groups_per_second",
                float(rollout_metrics["groups_per_second"]),
            )
            train_result.metrics.setdefault(
                "rollout/trajectories_per_second",
                float(rollout_metrics["trajectories_per_second"]),
            )
            for key, value in rollout_lifecycle_metrics.items():
                train_result.metrics.setdefault(f"rollout/{key}", value)
            for key, value in rollout_memory_metrics.items():
                train_result.metrics.setdefault(f"rollout/coordinator_{key}", value)
            for key, value in training_memory_metrics.items():
                train_result.metrics.setdefault(f"training/coordinator_{key}", value)
            lifecycle_seconds = sum(
                rollout_lifecycle_metrics.get(key, 0.0)
                for key in ("prepare_seconds", "finish_seconds")
            )
            train_result.metrics.setdefault(
                "rollout/collection_seconds",
                max(0.0, rollout_elapsed_seconds - lifecycle_seconds),
            )
            train_result.metrics.setdefault(
                "training/elapsed_seconds",
                training_elapsed_seconds,
            )
            await self._emit_progress(
                ExperimentProgress(
                    update=completed_update,
                    phase="training",
                    status="completed",
                    metrics={
                        str(key): float(value)
                        for key, value in train_result.metrics.items()
                        if isinstance(value, int | float | bool)
                    },
                )
            )
            evaluation = None
            if self.config.evaluation.enabled and (
                completed_update % self.config.evaluation.every_updates == 0
                or (
                    completed_update == 1
                    and self.config.evaluation.evaluate_after_first_update
                )
            ):
                assert self.evaluate is not None
                await self._emit_progress(
                    ExperimentProgress(
                        update=completed_update,
                        phase="evaluation",
                        status="started",
                        completed=0,
                        total=self.config.evaluation.episodes,
                    )
                )
                evaluation = await self.evaluate(
                    completed_update,
                    train_result,
                    self.config,
                )
                await self._emit_progress(
                    ExperimentProgress(
                        update=completed_update,
                        phase="evaluation",
                        status="completed",
                        completed=self.config.evaluation.episodes,
                        total=self.config.evaluation.episodes,
                        metrics={
                            str(key): float(value)
                            for key, value in evaluation.metrics.items()
                        },
                    )
                )
            if self.log_step is not None:
                await self._emit_progress(
                    ExperimentProgress(
                        update=completed_update,
                        phase="logging",
                        status="started",
                    )
                )
                await self.log_step(
                    completed_update,
                    groups,
                    train_result,
                    evaluation,
                    self.config,
                )
                await self._emit_progress(
                    ExperimentProgress(
                        update=completed_update,
                        phase="logging",
                        status="completed",
                    )
                )
            await self._emit_progress(
                ExperimentProgress(
                    update=completed_update,
                    phase="update",
                    status="completed",
                )
            )
        except BaseException as exc:
            await self._emit_progress(
                ExperimentProgress(
                    update=completed_update,
                    phase="update",
                    status="failed",
                    message=f"{type(exc).__name__}: {exc}",
                )
            )
            raise
        returned_evaluation = evaluation
        if (
            returned_evaluation is not None
            and not self.config.storage.retain_rollout_payloads
        ):
            returned_evaluation = EvaluationResult(
                step=returned_evaluation.step,
                metrics=returned_evaluation.metrics,
                artifacts=returned_evaluation.artifacts,
            )
        return ExperimentStepResult(
            update=completed_update,
            training=train_result,
            evaluation=returned_evaluation,
        )

    async def _train_with_progress(
        self,
        groups: Sequence[EmbodiedTrajectoryGroup],
        *,
        completed_update: int,
        started_at: float,
    ) -> TrainResult:
        """Keep observers live while a long-running backend update is in flight."""

        training_task = asyncio.create_task(self.backend.train(groups))
        try:
            while True:
                done, _ = await asyncio.wait(
                    (training_task,),
                    timeout=_TRAINING_PROGRESS_INTERVAL_SECONDS,
                )
                if training_task in done:
                    return await training_task
                await self._emit_progress(
                    ExperimentProgress(
                        update=completed_update,
                        phase="training",
                        status="progress",
                        metrics={
                            "training/elapsed_seconds": time.monotonic() - started_at
                        },
                        message="Training backend is active",
                    )
                )
        finally:
            if not training_task.done():
                training_task.cancel()
                await asyncio.gather(training_task, return_exceptions=True)

    async def _emit_progress(self, event: ExperimentProgress) -> None:
        if self.log_progress is not None:
            await self.log_progress(event, self.config)

    async def run(self, *, close_backend: bool = True) -> list[ExperimentStepResult]:
        results: list[ExperimentStepResult] = []
        start_update = int(getattr(self.backend, "update_step", 0))
        if not 0 <= start_update <= self.config.training.updates:
            raise ValueError(
                "Backend update_step is outside the configured training range: "
                f"update_step={start_update}, updates={self.config.training.updates}"
            )
        try:
            for update in range(start_update, self.config.training.updates):
                # Operator-requested stop at an update boundary: return normally
                # so the owner closes observers and waits for W&B uploads. Do
                # not cancel the process on a logging-enqueued progress message.
                if (Path(self.config.storage.output_dir) / "STOP_REQUESTED").exists():
                    break
                results.append(await self.train_step(update=update))
        finally:
            if close_backend:
                await self.backend.close()
        return results


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.blake2b(
        "\x1f".join(str(part) for part in parts).encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


def _discard_transient_rollout_payloads(
    groups: Sequence[EmbodiedTrajectoryGroup],
) -> int:
    """Release optimizer-only payloads before evaluation and telemetry."""

    removed = 0
    for group in groups:
        for trajectory in group.trajectories:
            records = (
                trajectory,
                *trajectory.observations,
                *trajectory.actions,
                *trajectory.rewards,
                *trajectory.tool_calls,
            )
            for record in records:
                metadata = getattr(record, "metadata", None)
                if not isinstance(metadata, dict):
                    continue
                keys = [
                    key
                    for key in metadata
                    if str(key).startswith("_art_embodied_transient_")
                ]
                for key in keys:
                    metadata.pop(key, None)
                    removed += 1
    return removed


def _finalize_rollout_metadata(
    trajectory: EmbodiedTrajectory,
    *,
    scenario: EmbodiedScenario,
    context: RolloutContext,
) -> None:
    trajectory.metadata.setdefault("scenario_id", scenario.id)
    trajectory.metadata.setdefault("environment_seed", context.environment_seed)
    trajectory.metadata.setdefault("policy_seed", context.policy_seed)
    trajectory.metadata.setdefault("attempt_index", context.attempt_index)
    trajectory.metadata.setdefault("config_fingerprint", context.config_fingerprint)


async def _prepare_rollout(rollout: RolloutFunction, *, update: int) -> None:
    prepare = getattr(rollout, "prepare_update", None)
    if not callable(prepare):
        return
    result = prepare(update=update)
    if inspect.isawaitable(result):
        await result


async def _finish_rollout(rollout: RolloutFunction, *, update: int) -> None:
    finish = getattr(rollout, "finish_update", None)
    if not callable(finish):
        return
    result = finish(update=update)
    if inspect.isawaitable(result):
        await result


def _rollout_lifecycle_metrics(rollout: RolloutFunction) -> dict[str, float]:
    raw = getattr(rollout, "lifecycle_metrics", None)
    if not isinstance(raw, dict):
        return {}
    return {
        str(key): float(value)
        for key, value in raw.items()
        if isinstance(value, int | float)
    }


def _process_memory_metrics(
    status_path: Path = Path("/proc/self/status"),
) -> dict[str, float]:
    """Read coordinator RSS without introducing a process-monitor dependency."""

    try:
        lines = status_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}

    fields = {
        "VmRSS": "resident_memory_mb",
        "VmHWM": "peak_resident_memory_mb",
    }
    metrics: dict[str, float] = {}
    for line in lines:
        name, separator, raw_value = line.partition(":")
        metric_name = fields.get(name)
        if not separator or metric_name is None:
            continue
        parts = raw_value.split()
        if not parts:
            continue
        try:
            value = float(parts[0])
        except ValueError:
            continue
        unit = parts[1].lower() if len(parts) > 1 else "bytes"
        scale = {
            "bytes": 1.0 / (1024 * 1024),
            "b": 1.0 / (1024 * 1024),
            "kb": 1.0 / 1024,
            "mb": 1.0,
        }.get(unit)
        if scale is not None:
            metrics[metric_name] = value * scale
    return metrics


def _rollout_progress_metrics(
    groups: Sequence[EmbodiedTrajectoryGroup],
) -> dict[str, float | int]:
    trajectories = [trajectory for group in groups for trajectory in group]
    rewards = [float(trajectory.reward) for trajectory in trajectories]
    successes = [
        bool(trajectory.metrics["success"])
        for trajectory in trajectories
        if "success" in trajectory.metrics
    ]
    metrics: dict[str, float | int] = {
        "groups_completed": len(groups),
        "trajectories_completed": len(trajectories),
    }
    if rewards:
        metrics["reward_mean"] = sum(rewards) / len(rewards)
    if successes:
        metrics["success_count"] = sum(successes)
        metrics["success_denominator"] = len(successes)
        metrics["success_rate"] = sum(successes) / len(successes)
    return metrics


def _rollout_throughput_metrics(
    groups: Sequence[EmbodiedTrajectoryGroup],
    *,
    elapsed_seconds: float,
) -> dict[str, float]:
    elapsed = max(float(elapsed_seconds), 1e-12)
    trajectories = sum(len(group.trajectories) for group in groups)
    return {
        "elapsed_seconds": float(elapsed_seconds),
        "groups_per_second": len(groups) / elapsed,
        "trajectories_per_second": trajectories / elapsed,
    }
