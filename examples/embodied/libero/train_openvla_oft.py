"""Run LIBERO VLA training and fixed evaluation through public ART APIs."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from art_embodied import (
    EmbodiedExperimentConfig,
    ExperimentProgress,
    WandbWeaveObserver,
    make_policy,
    run_lerobot_evaluation,
    run_lerobot_experiment,
    validate_runtime_device_availability,
)
from art_embodied.compatibility import (
    require_compatible_runtime,
    runtime_profile_for_policy,
)
from examples.embodied.libero.components import (
    LiberoSettings,
    build_evaluation_scenarios,
    build_train_scenarios,
    prepare_libero_runtime_paths,
    validate_libero_runtime_imports,
    validate_libero_task_assets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Complete ART-Embodied experiment YAML.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help=(
            "Run the YAML-defined fixed evaluation without constructing the "
            "training backend."
        ),
    )
    parser.add_argument(
        "--evaluation-step",
        type=int,
        default=0,
        help="Non-negative checkpoint/update label used in evaluation logs.",
    )
    parser.add_argument(
        "--native-wandb-step",
        action="store_true",
        help="Use the evaluation update as native W&B Step in a fresh evaluation-only run.",
    )
    parser.add_argument(
        "--checkpoint-role",
        choices=("candidate", "sft_baseline"),
        default="candidate",
        help="Checkpoint role recorded in evaluation artifacts.",
    )
    parser.add_argument(
        "--policy-checkpoint",
        type=Path,
        help=(
            "Local ART-Embodied policy snapshot to load before --evaluate-only. "
            "This evaluates a saved adapter without resuming the optimizer."
        ),
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Validate the YAML, LIBERO assets, and scenario matrices without "
            "checking GPU availability or loading the policy."
        ),
    )
    return parser.parse_args()


async def run(
    config_path: Path,
    *,
    evaluate_only: bool = False,
    evaluation_step: int = 0,
    preflight: bool = False,
    policy_checkpoint: Path | None = None,
    native_wandb_step: bool = False,
    checkpoint_role: str = "candidate",
) -> None:
    if native_wandb_step and not evaluate_only:
        raise ValueError("--native-wandb-step requires --evaluate-only")
    if checkpoint_role not in {"candidate", "sft_baseline"}:
        raise ValueError("Unsupported evaluation checkpoint role")
    if checkpoint_role != "candidate" and not evaluate_only:
        raise ValueError("--checkpoint-role requires --evaluate-only")
    if evaluate_only and evaluation_step < 0:
        raise ValueError("evaluation step cannot be negative")
    if policy_checkpoint is not None and not evaluate_only:
        raise ValueError("--policy-checkpoint requires --evaluate-only")
    config = EmbodiedExperimentConfig.from_yaml(config_path)
    if native_wandb_step:
        wandb_config = config.observability.wandb
        if wandb_config.connection != "primary":
            raise ValueError("Native-step evaluation requires a fresh primary W&B run")
        if wandb_config.native_update_steps:
            raise ValueError(
                "Standalone evaluation must disable the contiguous training writer "
                "(observability.wandb.native_update_steps=false); "
                "--native-wandb-step aligns the standalone evaluation row."
            )
    _validate_required_policy_checkpoint(config, policy_checkpoint=policy_checkpoint)
    # This process owns the ART model lifecycle even when policy workers run in
    # isolated environments. Check the lightweight package contract before a
    # source snapshot consumes GPU time or downloads policy weights.
    profile = runtime_profile_for_policy(config.policy.type)
    require_compatible_runtime(profile=profile)
    integration = _integration(config.environment.type)
    integration[0]()
    settings = integration[1].from_config(config)
    asset_summary = integration[2](settings)
    evaluation_scenarios = integration[3](config)
    train_scenarios = [] if evaluate_only else integration[4](config)
    if preflight:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "config": str(config_path),
                    "mode": "evaluation" if evaluate_only else "training",
                    "policy_checkpoint": (
                        str(policy_checkpoint.resolve())
                        if policy_checkpoint is not None
                        else None
                    ),
                    "assets": asset_summary,
                    "train_scenarios": len(train_scenarios),
                    "evaluation_scenarios": len(evaluation_scenarios),
                    "execution": config.execution_summary(),
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        return
    validate_runtime_device_availability(
        config,
        include_training_devices=not evaluate_only,
    )
    # Start telemetry before loading a multi-billion-parameter policy. Users
    # must be able to discover a scheduled run and inspect startup failures
    # rather than waiting several minutes for the first evaluation event.
    observer = WandbWeaveObserver.start(config)
    initialization_update = _initialization_update(
        config,
        evaluate_only=evaluate_only,
        evaluation_step=evaluation_step,
    )
    run_error: BaseException | None = None
    try:
        await observer.log_progress(
            ExperimentProgress(
                update=initialization_update,
                phase="initialization",
                status="started",
                message="Loading policy",
            ),
            config,
        )
        try:
            # LIBERO initializes EGL while importing its environment classes.
            # Keep that runtime check on the allocated GPU node so the pure
            # --preflight path remains usable from a CPU-only login node.
            integration[5]()
            policy = make_policy(config)
            if policy_checkpoint is not None:
                _load_policy_checkpoint(policy, policy_checkpoint)
        except BaseException as exc:
            await observer.log_progress(
                ExperimentProgress(
                    update=initialization_update,
                    phase="initialization",
                    status="failed",
                    message=f"{type(exc).__name__}: {exc}",
                ),
                config,
            )
            raise
        await observer.log_progress(
            ExperimentProgress(
                update=initialization_update,
                phase="initialization",
                status="completed",
                message="Policy loaded",
            ),
            config,
        )
        if evaluate_only:
            await run_lerobot_evaluation(
                config=config,
                policy=policy,
                evaluation_scenarios=evaluation_scenarios,
                step=evaluation_step,
                checkpoint_path=policy_checkpoint,
                observer=observer,
                use_native_wandb_step=native_wandb_step,
                checkpoint_role=checkpoint_role,
            )
            return
        await run_lerobot_experiment(
            config=config,
            policy=policy,
            train_scenarios=train_scenarios,
            evaluation_scenarios=(
                evaluation_scenarios if config.evaluation.enabled else None
            ),
            observer=observer,
        )
    except BaseException as exc:
        run_error = exc
        raise
    finally:
        observer.close(exit_code=1 if run_error is not None else 0)


def _initialization_update(
    config: EmbodiedExperimentConfig,
    *,
    evaluate_only: bool,
    evaluation_step: int,
) -> int:
    """Return the policy version represented while the entrypoint initializes."""

    if evaluate_only:
        if evaluation_step < 0:
            raise ValueError("evaluation_step must be non-negative")
        return int(evaluation_step)
    checkpoint = config.storage.resume_from_checkpoint
    if checkpoint is None:
        return 0
    state_path = checkpoint / "art_embodied_training_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot resolve initialization update from checkpoint: {state_path}"
        ) from exc
    step = None
    if isinstance(state, dict):
        step = state.get("update_step", state.get("step"))
    if not isinstance(step, int) or step < 0:
        raise ValueError(
            "Checkpoint training-state metadata must contain a non-negative "
            f"integer update_step (or legacy step): {state_path}"
        )
    return step


def _validate_required_policy_checkpoint(
    config: EmbodiedExperimentConfig,
    *,
    policy_checkpoint: Path | None,
) -> None:
    """Fail closed when a candidate evaluation requires an explicit snapshot."""

    required = config.evaluation.kwargs.get("require_policy_checkpoint", False)
    if not isinstance(required, bool):
        raise TypeError("evaluation.kwargs.require_policy_checkpoint must be a boolean")
    if required and policy_checkpoint is None:
        raise ValueError(
            "This evaluation requires --policy-checkpoint so the SFT base cannot "
            "be mislabeled as the trained candidate"
        )


def _load_policy_checkpoint(policy: object, checkpoint: Path) -> None:
    """Restore one complete policy snapshot without constructing an optimizer."""

    resolved = checkpoint.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Policy checkpoint directory is missing: {resolved}")
    load = getattr(policy, "load_checkpoint", None)
    if not callable(load):
        raise TypeError("Policy checkpoint evaluation requires policy.load_checkpoint")
    load({"path": str(resolved)})


def _integration(environment_type: str):
    """Resolve one LIBERO provider without importing both providers."""

    if environment_type == "libero":
        return (
            prepare_libero_runtime_paths,
            LiberoSettings,
            validate_libero_task_assets,
            build_evaluation_scenarios,
            build_train_scenarios,
            validate_libero_runtime_imports,
        )
    if environment_type == "libero_plus":
        from examples.embodied.libero_plus import (
            LiberoPlusSettings,
            prepare_libero_plus_runtime_paths,
            validate_libero_plus_runtime_imports,
            validate_libero_plus_task_assets,
        )
        from examples.embodied.libero_plus import (
            build_evaluation_scenarios as build_libero_plus_evaluation_scenarios,
        )
        from examples.embodied.libero_plus import (
            build_train_scenarios as build_libero_plus_train_scenarios,
        )

        return (
            prepare_libero_plus_runtime_paths,
            LiberoPlusSettings,
            validate_libero_plus_task_assets,
            build_libero_plus_evaluation_scenarios,
            build_libero_plus_train_scenarios,
            validate_libero_plus_runtime_imports,
        )
    raise ValueError(
        "The LIBERO entrypoint requires environment.type='libero' or 'libero_plus'"
    )


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        run(
            args.config,
            evaluate_only=args.evaluate_only,
            evaluation_step=args.evaluation_step,
            preflight=args.preflight,
            policy_checkpoint=args.policy_checkpoint,
            native_wandb_step=args.native_wandb_step,
            checkpoint_role=args.checkpoint_role,
        )
    )
