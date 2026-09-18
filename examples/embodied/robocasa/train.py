"""Run a RoboCasa GR-1 task through ART-Embodied's public lifecycle."""

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

from .records import build_evaluation_scenarios, build_train_scenarios
from .settings import RoboCasaSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--evaluation-step", type=int, default=0)
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


async def run(
    config_path: Path,
    *,
    evaluate_only: bool = False,
    evaluation_step: int = 0,
    preflight: bool = False,
    policy_checkpoint: Path | None = None,
) -> None:
    if policy_checkpoint is not None and not evaluate_only:
        raise ValueError("--policy-checkpoint requires --evaluate-only")
    if evaluation_step < 0:
        raise ValueError("--evaluation-step must be non-negative")

    config = EmbodiedExperimentConfig.from_yaml(config_path)
    _validate_evaluation_checkpoint_request(
        config,
        evaluate_only=evaluate_only,
        policy_checkpoint=policy_checkpoint,
    )
    require_compatible_runtime(profile=runtime_profile_for_policy(config.policy.type))
    settings = RoboCasaSettings.from_config(config)
    evaluation_scenarios = build_evaluation_scenarios(config)
    train_scenarios = [] if evaluate_only else build_train_scenarios(config)
    if preflight:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "config": str(config_path),
                    "mode": "evaluation" if evaluate_only else "training",
                    "task": ",".join(task.id for task in settings.tasks),
                    "instruction": "environment-provided",
                    "robot": "GR1ArmsAndWaistFourierHands",
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
    observer = WandbWeaveObserver.start(config)
    initialization_update = evaluation_step if evaluate_only else 0
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
            )
        else:
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


def _load_policy_checkpoint(policy: object, checkpoint: Path) -> None:
    resolved = checkpoint.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Policy checkpoint directory is missing: {resolved}")
    load_path = resolved
    marker_path = resolved / "art_embodied_checkpoint_complete.json"
    policy_path = resolved / "policy"
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        from art_embodied.checkpointing import CheckpointManager

        CheckpointManager().validate(
            resolved,
            expected_resume_contract_fingerprint=marker["resume_contract_fingerprint"],
            require_training_state=False,
        )
        if not policy_path.is_dir():
            raise ValueError(
                "Transactional evaluation checkpoint has no policy directory: "
                f"{resolved}"
            )
        load_path = policy_path
    load = getattr(policy, "load_checkpoint", None)
    if not callable(load):
        raise TypeError("Policy checkpoint evaluation requires policy.load_checkpoint")
    load({"path": str(load_path)})


def _validate_evaluation_checkpoint_request(
    config: EmbodiedExperimentConfig,
    *,
    evaluate_only: bool,
    policy_checkpoint: Path | None,
) -> None:
    required = config.evaluation.kwargs.get("require_policy_checkpoint", False)
    if not isinstance(required, bool):
        raise TypeError("evaluation.kwargs.require_policy_checkpoint must be a boolean")
    if required and (not evaluate_only or policy_checkpoint is None):
        raise ValueError(
            "This evaluation config requires --evaluate-only and an explicit "
            "--policy-checkpoint"
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
        )
    )
