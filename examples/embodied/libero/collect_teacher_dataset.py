"""Collect successful VLA rollouts into a supervised LeRobotDataset."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from art_embodied import EmbodiedExperimentConfig, make_policy
from art_embodied.experiment import RolloutContext
from art_embodied.integrations.pi_flow_sde import PIFlowSDEPolicyAdapter
from examples.embodied.libero.dataset_export import (
    LiberoDatasetExportSpec,
    LiberoTrajectoryDatasetWriter,
)
from examples.embodied.libero.environment import (
    LiberoTaskCatalog,
    prepare_libero_runtime_paths,
    validate_libero_task_assets,
)
from examples.embodied.libero.records import build_train_scenarios
from examples.embodied.libero.rollout import rollout_libero_group
from examples.embodied.libero.settings import LiberoSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--successes-per-task", type=int, default=20)
    parser.add_argument("--max-groups", type=int, default=128)
    parser.add_argument(
        "--sampling-mode",
        choices=("eval", "train"),
        default="eval",
        help="Use native deployment ODE by default; train selects exploratory Flow-SDE.",
    )
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> dict[str, Any]:
    config = EmbodiedExperimentConfig.from_yaml(args.config)
    settings = LiberoSettings.from_config(config)
    target_tasks = settings.training_task_ids or settings.task_ids
    if config.policy.type not in {"pi0", "pi05"}:
        raise ValueError("Teacher collection currently requires a PI0/PI0.5 policy")
    if settings.init_state_selection != "partitioned_random_reset":
        raise ValueError("Teacher collection requires partitioned_random_reset")
    if args.successes_per_task <= 0 or args.max_groups <= 0:
        raise ValueError("Collection limits must be positive")
    if args.dataset_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite teacher dataset: {args.dataset_root}"
        )
    # Direct collection does not pass through the normal rollout-worker
    # launcher, so establish the same headless MuJoCo rendering contract before
    # importing LIBERO's robosuite environment.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    wandb_run = _start_wandb(config, args) if not args.no_wandb else None
    if wandb_run is not None:
        wandb_run.summary["status"] = "initializing"
    try:
        prepare_libero_runtime_paths()
        validate_libero_task_assets(settings)
        policy = make_policy(config)
        checkpoint = _policy_snapshot_path(args.policy_checkpoint)
        policy.load_checkpoint({"path": str(checkpoint)})
        policy.eval()
        policy_adapter = PIFlowSDEPolicyAdapter(
            policy=policy,
            robot_type="panda",
            sampling_mode=args.sampling_mode,
        )
        catalog = LiberoTaskCatalog(settings)
        scenarios = build_train_scenarios(config)
        writer = LiberoTrajectoryDatasetWriter(
            LiberoDatasetExportSpec(
                repo_id=args.dataset_repo_id,
                root=args.dataset_root,
                fps=10,
                image_height=256,
                image_width=256,
                source_image_height=settings.observation_height,
                source_image_width=settings.observation_width,
            ),
            provenance={
                "config": str(args.config.resolve()),
                "config_fingerprint": config.fingerprint,
                "policy_checkpoint": str(checkpoint.resolve()),
                "policy_checkpoint_sha256": _checkpoint_sha256(checkpoint),
                "training_task_ids": list(target_tasks),
                "training_trial_ids": list(settings.training_trial_ids or ()),
                "sampling_mode": args.sampling_mode,
                "dataset_image_size": [256, 256],
                "source_observation_size": [
                    settings.observation_height,
                    settings.observation_width,
                ],
            },
        )
    except BaseException:
        if wandb_run is not None:
            wandb_run.summary["status"] = "failed_during_initialization"
            wandb_run.finish(exit_code=1)
        raise
    if wandb_run is not None:
        wandb_run.summary["status"] = "collecting"
    successes = {task_id: 0 for task_id in target_tasks}
    attempted = {task_id: 0 for task_id in target_tasks}
    exported_frames = 0
    completed_groups = 0
    try:
        for group_index, scenario in enumerate(scenarios[: args.max_groups]):
            task_id = int(scenario.payload["task_id"])
            if successes[task_id] >= args.successes_per_task:
                continue
            contexts = _rollout_contexts(
                config=config,
                group_index=group_index,
                scenario_id=scenario.id,
                attempts=(
                    1
                    if args.sampling_mode == "eval"
                    else config.algorithm.group_size
                ),
            )
            trajectories = await rollout_libero_group(
                config=config,
                policy=policy,
                catalog=catalog,
                settings=settings,
                scenario=scenario,
                contexts=contexts,
                phase=args.sampling_mode,
                embedded_batch_predictor=policy_adapter.predict_batch,
                embedded_batch_reset=lambda seed: policy_adapter.reset(seed=seed),
            )
            completed_groups += 1
            for trajectory in trajectories:
                attempted[task_id] += 1
                if not bool(trajectory.metrics.get("success", False)):
                    continue
                if successes[task_id] >= args.successes_per_task:
                    continue
                exported_frames += writer.add_successful_trajectory(trajectory)
                successes[task_id] += 1
            metrics = _collection_metrics(
                completed_groups=completed_groups,
                successes=successes,
                attempted=attempted,
                exported_frames=exported_frames,
            )
            print(json.dumps(metrics, sort_keys=True), flush=True)
            if wandb_run is not None:
                wandb_run.log(metrics)
            if all(value >= args.successes_per_task for value in successes.values()):
                break
        if not all(value >= args.successes_per_task for value in successes.values()):
            raise RuntimeError(
                "Teacher collection exhausted max_groups before meeting targets: "
                f"successes={successes}, attempted={attempted}"
            )
        report = writer.finalize()
        result = {
            "status": "completed",
            "successes_by_task": {
                f"task_{task_id:02d}": value
                for task_id, value in successes.items()
            },
            "attempted_by_task": {
                f"task_{task_id:02d}": value
                for task_id, value in attempted.items()
            },
            "completed_groups": completed_groups,
            "dataset": str(args.dataset_root.resolve()),
            "exported_episodes": report.exported_episodes,
            "exported_frames": report.exported_frames,
        }
        if wandb_run is not None:
            import wandb

            artifact = wandb.Artifact(
                name=f"{config.experiment.run}-teacher-dataset",
                type="dataset",
                metadata=result,
            )
            artifact.add_dir(str(args.dataset_root))
            wandb_run.log_artifact(artifact, aliases=["latest", "complete"])
            wandb_run.summary.update(result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return result
    except BaseException:
        writer.abort()
        if wandb_run is not None:
            wandb_run.summary["status"] = "failed_during_collection"
            wandb_run.finish(exit_code=1)
            wandb_run = None
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def _rollout_contexts(
    *,
    config: EmbodiedExperimentConfig,
    group_index: int,
    scenario_id: str,
    attempts: int,
) -> tuple[RolloutContext, ...]:
    environment_seed = _stable_seed(config.experiment.seed, group_index, scenario_id)
    return tuple(
        RolloutContext(
            update=0,
            group_index=group_index,
            attempt_index=attempt_index,
            environment_seed=environment_seed,
            policy_seed=_stable_seed(
                config.experiment.seed,
                group_index,
                attempt_index,
                "policy",
            ),
            config_fingerprint=config.fingerprint,
        )
        for attempt_index in range(attempts)
    )


def _collection_metrics(
    *,
    completed_groups: int,
    successes: dict[int, int],
    attempted: dict[int, int],
    exported_frames: int,
) -> dict[str, int | float]:
    total_attempted = sum(attempted.values())
    total_successes = sum(successes.values())
    metrics: dict[str, int | float] = {
        "collection/group": completed_groups,
        "collection/attempted_trajectories": total_attempted,
        "collection/exported_episodes": total_successes,
        "collection/exported_frames": exported_frames,
        "collection/success_rate": (
            total_successes / total_attempted if total_attempted else 0.0
        ),
    }
    for task_id in successes:
        metrics[f"collection_tasks/task_{task_id:02d}_exported"] = successes[task_id]
        metrics[f"collection_tasks/task_{task_id:02d}_attempted"] = attempted[task_id]
    return metrics


def _start_wandb(config: EmbodiedExperimentConfig, args: argparse.Namespace) -> Any:
    import wandb

    wandb_config = config.observability.wandb
    run = wandb.init(
        entity=wandb_config.entity,
        project=wandb_config.project,
        name=config.experiment.run,
        job_type="teacher-dataset-collection",
        tags=[*config.experiment.tags, "teacher-dataset"],
        config={
            "experiment": config.model_dump(mode="json"),
            "dataset_repo_id": args.dataset_repo_id,
            "successes_per_task": args.successes_per_task,
            "max_groups": args.max_groups,
            "sampling_mode": args.sampling_mode,
        },
    )
    return run


def _policy_snapshot_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if (resolved / "art_embodied_pi_snapshot.json").is_file():
        return resolved
    policy = resolved / "policy"
    if (policy / "art_embodied_pi_snapshot.json").is_file():
        return policy
    raise FileNotFoundError(f"PI policy snapshot not found under {resolved}")


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode())
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
