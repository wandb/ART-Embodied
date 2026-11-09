"""Commit a saved DROID update after logging failed, without repeating training.

Requires the completed checkpoint and original trajectories. Refuses to
overwrite history, skip a step, or fill in missing evaluation results.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from art_embodied.config import EmbodiedExperimentConfig
from art_embodied.integrations.isaaclab_reset_bank import IsaacLabResetBank
from art_embodied.observability import WandbWeaveObserver
from art_embodied.types import LocalTrainResult
from art_embodied.utils import write_json_atomic
from examples.embodied.isaaclab.droid_environment import DroidSceneConfig
from examples.embodied.isaaclab.run_droid import (
    checkpoint_version,
    read_evaluation,
    read_logging_groups,
    verify_history,
    wandb_run_path,
)


async def restore_rollout(observer, *, policy_version, groups, config):
    """Restage original rollout evidence without replacing its provenance."""
    path = (
        Path(config.storage.output_dir)
        / "rollout-evidence"
        / f"policy-version-{policy_version:06d}.json"
    )
    if not path.is_file():
        await observer.log_rollout(policy_version, groups, config)
        return
    if config.fingerprint != observer.config.fingerprint:
        raise ValueError("Observer config does not match recovery config")
    original = json.loads(path.read_text())
    fingerprint = original.get("config_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("Saved rollout evidence has no source config fingerprint")
    # Resume changes operational settings. Keep the collection-time fingerprint,
    # while the shared writer still checks every other field and never overwrites.
    evidence = observer._persist_rollout_group_evidence(
        policy_version, groups, source_config_fingerprint=fingerprint
    )
    if observer.wandb_run is not None:
        observer._log_wandb_rollout(policy_version, groups, evidence=evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    args = parser.parse_args()
    config = EmbodiedExperimentConfig.model_validate_json(args.config.read_text())
    settings = config.observability.wandb
    checkpoint = config.storage.resume_from_checkpoint
    if (
        not checkpoint
        or settings.connection != "resume"
        or settings.resume != "must"
        or not settings.run_id
    ):
        parser.error(
            "Use the completed checkpoint and the existing run with resume=must"
        )
    step = checkpoint_version(checkpoint, config)
    output = Path(config.storage.output_dir)
    result_dir = output / f"update-{step:06d}"
    data = json.loads((result_dir / "result.json").read_text())
    if (
        data["step"] != step
        or Path(data["checkpoint_path"]).resolve() != Path(checkpoint).resolve()
    ):
        raise ValueError("Saved result and checkpoint differ")
    groups = read_logging_groups(
        result_dir / "trace-groups.json",
        expected_groups=data["groups"],
        expected_trajectories=data["trajectories"],
        expected_chunks=data["chunks"],
    )
    if sum(t.reward for g in groups for t in g) != data["successes"] or any(
        t.initial_policy_version != step - 1 or t.final_policy_version != step - 1
        for g in groups
        for t in g
    ):
        raise ValueError("Saved behavior rewards or policy versions differ")
    expected_path = output / "history-contract.json"
    expected = json.loads(expected_path.read_text())
    if set(expected) != {str(i) for i in range(step)}:
        raise ValueError("Recovery requires exactly one completed, uncommitted update")
    run_path = f"{settings.entity}/{settings.project}/{settings.run_id}"
    verify_history(run_path, last_step=step - 1, expected=expected)
    evaluation = None
    if step % config.evaluation.every_updates == 0:
        scene = DroidSceneConfig(
            **{
                k: config.environment.kwargs[k]
                for k in ("pick_up_object", "destination_location", "instruction")
            }
        )
        bank = IsaacLabResetBank.load(
            args.banks / "development",
            contract=scene.reset_contract(),
            split="development",
        )
        if bank.fingerprint != config.evaluation.kwargs["reset_bank_sha256"]:
            raise ValueError("Development bank differs from the configured bank")
        directories = sorted(
            (output / f"development-{step:06d}").glob("actor-*/rollout")
        )
        evaluation = read_evaluation(
            directories, version=step, config=config, bank=bank
        )
    metrics = dict(data["metrics"])
    metrics.update(
        {
            "performance/training_wall_seconds": data["wall_seconds"],
            "performance/backend_update_seconds": data["backend_update_seconds"],
            "performance/rollout_chunks": data["chunks"],
            "performance/rollout_trajectories": data["trajectories"],
        }
    )
    collection = output / f"train-{step - 1:06d}/collection.json"
    if collection.exists():
        metrics["performance/rollout_wall_seconds"] = json.loads(
            collection.read_text()
        )["wall_seconds"]
    if evaluation is not None:
        metrics["performance/evaluation_wall_seconds"] = json.loads(
            (output / f"development-{step:06d}/collection.json").read_text()
        )["wall_seconds"]
    observer = WandbWeaveObserver.start(config)
    exit_code = 1
    try:
        if wandb_run_path(observer.wandb_run) != run_path:
            raise ValueError("Recovery connected to a different run")
        observer._native_update_history.require_next_step(step)
        asyncio.run(
            restore_rollout(
                observer, policy_version=step - 1, groups=groups, config=config
            )
        )
        asyncio.run(
            observer.log_step(
                step,
                groups,
                LocalTrainResult(
                    step=step, metrics=metrics, checkpoint_path=str(checkpoint)
                ),
                evaluation,
                config,
            )
        )
        if observer.weave_client is not None:
            observer.weave_client.flush()
        expected[str(step)] = {
            "metrics": {"train/success_rate": data["successes"] / data["trajectories"]},
            "media": data["trajectories"]
            + (config.evaluation.episodes if evaluation else 0),
        }
        if evaluation:
            expected[str(step)]["metrics"]["validation/success_rate"] = (
                evaluation.metrics["success_rate"]
            )
        write_json_atomic(expected_path, expected)
        exit_code = 0
    finally:
        observer.close(exit_code=exit_code)
    verified = verify_history(run_path, last_step=step, expected=expected)
    write_json_atomic(output / f"verified-{step:06d}.json", verified)
    print(json.dumps(verified), flush=True)


if __name__ == "__main__":
    main()
