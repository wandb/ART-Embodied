"""Evaluate a sealed checkpoint in the isolated, checked pi0-FAST MuJoCo 3.3 runtime."""

import argparse
import asyncio
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

from art_embodied import (
    EmbodiedExperimentConfig,
    ExperimentProgress,
    WandbWeaveObserver,
    make_policy,
    run_lerobot_evaluation,
    validate_runtime_device_availability,
)
from art_embodied.checkpointing import CheckpointManager
from art_embodied.compatibility import require_compatible_runtime
from examples.embodied.libero.train_openvla_oft import (
    _integration,
    _load_policy_checkpoint,
)
from examples.embodied.pi0_fast_long_restart import check_runtime, isolated_art_runtime
from examples.embodied.pi0_fast_spatial_teacher_control import diagnostic_cuda_mapping


def validate_inputs(config, checkpoint, step, role):
    if config.policy.type != "pi0_fast" or config.environment.type != "libero":
        raise ValueError("This isolated entrypoint supports pi0_fast with LIBERO only")
    if not config.evaluation.enabled or config.evaluation.data_role != "sealed_test":
        raise ValueError("Require an enabled sealed_test configuration")
    if config.evaluation.checkpoint_selection != "last":
        raise ValueError("Sealed evaluation requires last-checkpoint selection")
    if role not in {"sft_baseline", "candidate"} or type(step) is not int or step < 0:
        raise ValueError("Invalid checkpoint role or update")
    if (role == "sft_baseline") != (step == 0):
        raise ValueError("Baseline must be step 0; GRPO candidate must be positive")
    wandb = config.observability.wandb
    if not wandb.enabled or wandb.mode != "online" or wandb.connection != "primary":
        raise ValueError("Sealed evaluation requires a fresh online primary W&B run")
    if wandb.native_update_steps:
        raise ValueError(
            "Disable the contiguous training writer for standalone evaluation"
        )
    if config.storage.resume_from_checkpoint is not None:
        raise ValueError("Load the selected policy explicitly; do not resume training")
    checkpoint = checkpoint.expanduser().resolve()
    snapshot = json.loads(
        (checkpoint / "art_embodied_pi0_fast_snapshot.json").read_text()
    )
    expected = {
        "family": "pi0_fast",
        "model_id": config.policy.path,
        "revision": config.policy.revision,
        "model_compute_dtype": config.policy.load_kwargs["model_compute_dtype"],
        "training_loss_scale": config.policy.load_kwargs["training_loss_scale"],
    }
    if any(snapshot.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "Selected snapshot differs from the configured model/precision"
        )
    container = checkpoint if role == "candidate" else checkpoint.parent
    verified = CheckpointManager().validate_payload(container)
    if role == "candidate" and verified.manifest["metadata"].get("update_step") != step:
        raise ValueError("Candidate checkpoint update differs from evaluation step")
    return checkpoint, snapshot


def validate_isolated_simulator(validate_imports):
    from examples.embodied.libero import environment

    # The public LIBERO profile remains 3.8.1. This experiment checks its exact
    # 3.3.0 package AND loaded-library contract instead, only during this call.
    check_runtime()
    original = environment._validate_libero_mujoco_abi
    environment._validate_libero_mujoco_abi = check_runtime
    try:
        validate_imports()
    finally:
        environment._validate_libero_mujoco_abi = original


async def run(config_path, *, checkpoint, step, role, preflight=False):
    config = EmbodiedExperimentConfig.from_yaml(config_path)
    checkpoint, snapshot = validate_inputs(config, checkpoint, step, role)
    require_compatible_runtime(profile="control")
    runtime = check_runtime()
    integration = _integration(config.environment.type)
    with redirect_stdout(sys.stderr):
        integration[0]()
        assets = integration[2](integration[1].from_config(config))
        scenarios = integration[3](config)
    if preflight:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "preflight_only": True,
                    "sealed_policy_evaluated": False,
                    "runtime": runtime,
                    "snapshot": snapshot,
                    "checkpoint": str(checkpoint),
                    "step": step,
                    "role": role,
                    "config_fingerprint": config.fingerprint,
                    "evaluation_scenarios": len(scenarios),
                    "assets": assets,
                },
                indent=2,
                default=str,
            )
        )
        return
    validate_runtime_device_availability(config, include_training_devices=False)
    validate_isolated_simulator(integration[5])
    observer = WandbWeaveObserver.start(config)
    error = None
    try:
        await observer.log_progress(
            ExperimentProgress(
                update=step,
                phase="initialization",
                status="started",
                message="Loading sealed checkpoint",
            ),
            config,
        )
        with isolated_art_runtime(), diagnostic_cuda_mapping():
            policy = make_policy(config)
            _load_policy_checkpoint(policy, checkpoint)
            await observer.log_progress(
                ExperimentProgress(
                    update=step,
                    phase="initialization",
                    status="completed",
                    message="Sealed checkpoint loaded",
                ),
                config,
            )
            await run_lerobot_evaluation(
                config=config,
                policy=policy,
                evaluation_scenarios=scenarios,
                step=step,
                checkpoint_path=checkpoint,
                observer=observer,
                checkpoint_role=role,
                use_native_wandb_step=True,
            )
    except BaseException as exc:
        error = exc
        raise
    finally:
        observer.close(exit_code=1 if error else 0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-step", type=int, required=True)
    parser.add_argument(
        "--checkpoint-role", choices=("sft_baseline", "candidate"), required=True
    )
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        run(
            args.config,
            checkpoint=args.policy_checkpoint,
            step=args.evaluation_step,
            role=args.checkpoint_role,
            preflight=args.preflight,
        )
    )
