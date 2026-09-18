"""Measure one YAML-defined rollout collection without an optimizer update."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import time
from typing import Any, Literal

from art_embodied import (
    EmbodiedExperimentConfig,
    ExperimentProgress,
    WandbWeaveObserver,
    make_policy,
)
from art_embodied.experiment import EmbodiedExperiment, EmbodiedTrajectoryGroup
from art_embodied.rollout_process import LocalProcessRolloutPool
from art_embodied.runner import validate_runtime_device_availability
from art_embodied.types import LocalTrainResult
from examples.embodied.libero.components import (
    LiberoSettings,
    build_train_scenarios,
    prepare_libero_runtime_paths,
    validate_libero_runtime_imports,
    validate_libero_task_assets,
)


class _UnusedBackend:
    async def train(
        self, trajectory_groups: list[EmbodiedTrajectoryGroup], **_: Any
    ) -> LocalTrainResult:
        raise RuntimeError("rollout benchmark must not call the training backend")

    async def close(self) -> None:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument(
        "--phase",
        choices=("train", "eval"),
        default="train",
        help=(
            "Sampler phase applied to the training scenario stream. 'eval' "
            "isolates native-policy behavior without changing reset geometry."
        ),
    )
    return parser.parse_args()


async def run(
    config_path: Path,
    output_path: Path,
    *,
    updates: int = 1,
    phase: Literal["train", "eval"] = "train",
) -> dict[str, Any]:
    if updates <= 0:
        raise ValueError("updates must be positive")
    config = EmbodiedExperimentConfig.from_yaml(config_path)
    prepare_libero_runtime_paths()
    validate_libero_task_assets(LiberoSettings.from_config(config))
    if config.evaluation.enabled:
        raise ValueError("rollout benchmark requires evaluation.enabled=false")
    if config.runtime.rollout_execution.mode != "local_process":
        raise ValueError("rollout benchmark requires local-process actors")
    validate_runtime_device_availability(
        config,
        include_training_devices=False,
    )
    observer = WandbWeaveObserver.start(config)
    pool: LocalProcessRolloutPool | None = None
    run_error: BaseException | None = None
    benchmark_started = time.monotonic()
    update_results: list[dict[str, Any]] = []
    try:
        await observer.log_progress(
            ExperimentProgress(
                update=0,
                phase="initialization",
                status="started",
                message="Loading policy for rollout-only group-signal gate",
            ),
            config,
        )
        validate_libero_runtime_imports()
        policy = make_policy(config)
        pool = LocalProcessRolloutPool(config=config, policy=policy)
        rollout = pool.for_phase(phase)
        experiment = EmbodiedExperiment(
            config=config,
            scenarios=build_train_scenarios(config),
            rollout=rollout,
            backend=_UnusedBackend(),
            log_progress=observer.log_progress,
        )
        await observer.log_progress(
            ExperimentProgress(
                update=0,
                phase="initialization",
                status="completed",
                message="Policy loaded; optimizer is disabled",
            ),
            config,
        )
        for update in range(updates):
            started = time.monotonic()
            groups = await experiment.collect(update=update)
            elapsed = time.monotonic() - started
            await observer.log_rollout(update, groups, config)
            lifecycle_metrics = rollout.lifecycle_metrics
            trajectories = [
                trajectory for group in groups for trajectory in group.trajectories
            ]
            successes = sum(
                float(trajectory.reward or 0.0) > 0.0 for trajectory in trajectories
            )
            update_results.append(
                {
                    "update": update,
                    "groups": len(groups),
                    "trajectories": len(trajectories),
                    "successes": int(successes),
                    "success_rate": (
                        successes / len(trajectories) if trajectories else 0.0
                    ),
                    "group_outcomes": [
                        {
                            "group_index": int(group.metadata["group_index"]),
                            "scenario_id": str(group.metadata["scenario_id"]),
                            "successes": sum(
                                float(trajectory.reward or 0.0) > 0.0
                                for trajectory in group.trajectories
                            ),
                            "trajectories": len(group.trajectories),
                        }
                        for group in groups
                    ],
                    "elapsed_seconds": elapsed,
                    "groups_per_second": len(groups) / elapsed if elapsed else 0.0,
                    "trajectories_per_second": (
                        len(trajectories) / elapsed if elapsed else 0.0
                    ),
                    "lifecycle": lifecycle_metrics,
                    "collection_seconds": max(
                        0.0,
                        elapsed
                        - lifecycle_metrics.get("prepare_seconds", 0.0)
                        - lifecycle_metrics.get("finish_seconds", 0.0),
                    ),
                }
            )
            await observer.log_progress(
                ExperimentProgress(
                    update=update,
                    phase="rollout_only_gate",
                    status="completed",
                    completed=len(groups),
                    total=len(groups),
                    message="Rollouts logged without an optimizer update",
                    metrics={
                        "success_count": int(successes),
                        "success_denominator": len(trajectories),
                        "success_rate": (
                            successes / len(trajectories) if trajectories else 0.0
                        ),
                    },
                ),
                config,
            )
    except BaseException as exc:
        run_error = exc
        await observer.log_progress(
            ExperimentProgress(
                update=0,
                phase="rollout_only_gate",
                status="failed",
                message=f"{type(exc).__name__}: {exc}",
            ),
            config,
        )
        raise
    finally:
        if pool is not None:
            await pool.close()
        observer.close(exit_code=1 if run_error is not None else 0)
    elapsed = time.monotonic() - benchmark_started
    group_count = sum(item["groups"] for item in update_results)
    trajectory_count = sum(item["trajectories"] for item in update_results)
    successes = sum(item["successes"] for item in update_results)
    result = {
        "config_fingerprint": config.fingerprint,
        "phase": phase,
        "execution": config.execution_summary(),
        "updates": update_results,
        "groups": group_count,
        "trajectories": trajectory_count,
        "successes": int(successes),
        "success_rate": successes / trajectory_count if trajectory_count else 0.0,
        "elapsed_seconds": elapsed,
        "groups_per_second": group_count / elapsed if elapsed else 0.0,
        "trajectories_per_second": (trajectory_count / elapsed if elapsed else 0.0),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        run(
            args.config,
            args.output,
            updates=args.updates,
            phase=args.phase,
        )
    )
