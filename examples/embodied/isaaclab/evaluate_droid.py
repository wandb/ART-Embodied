"""Paired native-ODE evaluation of the initial DROID policy and one checkpoint."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import hashlib
from itertools import combinations
import json
from pathlib import Path
import subprocess
import time

from art_embodied.config import EmbodiedExperimentConfig
from art_embodied.evaluation import (
    _write_evaluation_evidence,
    _write_evaluation_outcomes,
    compare_paired_evaluation_reports,
)
from art_embodied.evidence import (
    evaluation_evidence_identity,
    require_resolved_sealed_identity,
)
from art_embodied.integrations.isaaclab_reset_bank import IsaacLabResetBank
from art_embodied.observability import WandbWeaveObserver
from art_embodied.utils import make_json_safe, write_json_atomic
from examples.embodied.isaaclab.droid_environment import DroidSceneConfig
from examples.embodied.isaaclab.run_droid import (
    DroidRunner,
    _wandb_history,
    checkpoint_version,
    read_evaluation,
    read_wandb,
    wandb_run_path,
)


def evaluation_config(training, *, banks, output, split):
    if split not in {"development", "sealed"}:
        raise ValueError("Select development or sealed evaluation")
    if training.evaluation.data_role != "development":
        raise ValueError("Start from the original development training configuration")
    if training.policy.type != "openpi_pi05_droid":
        raise ValueError("This recipe evaluates the OpenPI DROID policy")
    scene = DroidSceneConfig(
        **{
            key: training.environment.kwargs[key]
            for key in ("pick_up_object", "destination_location", "instruction")
        }
    )
    loaded = {
        name: IsaacLabResetBank.load(
            banks / name, contract=scene.reset_contract(), split=name
        )
        for name in ("train", "development", "sealed")
    }
    for first, second in combinations(loaded.values(), 2):
        first.assert_disjoint(second)
    if (
        loaded["train"].fingerprint
        != training.environment.kwargs["train_reset_bank_sha256"]
        or loaded["development"].fingerprint
        != training.evaluation.kwargs["reset_bank_sha256"]
    ):
        raise ValueError(
            "Training/development banks differ from the original experiment"
        )
    counts = {"train": 512, "development": 128, "sealed": 256}
    if any(len(loaded[name]) != count for name, count in counts.items()):
        raise ValueError("Use the reference recipe's 512/128/256 disjoint reset banks")
    raw = training.model_dump(mode="json")
    raw["experiment"]["run"] += f"-{split}-paired"
    raw["experiment"]["tags"] += ["evaluation-only", split]
    raw["storage"].update(output_dir=str(output), resume_from_checkpoint=None)
    raw["evaluation"].update(
        data_role="sealed_test" if split == "sealed" else "development",
        runtime="native_openpi",
        episodes=counts[split],
        seeds=[33000] if split == "sealed" else training.evaluation.seeds,
        fixed_scenarios=[f"{split}-{index}" for index in range(counts[split])],
        baseline_outcomes_path=None,
        checkpoint_selection="last",
    )
    raw["evaluation"]["kwargs"]["reset_bank_sha256"] = loaded[split].fingerprint
    raw["observability"]["videos_per_evaluation"] = counts[split]
    raw["observability"]["wandb"].update(
        connection="primary",
        run_id=None,
        resume=None,
        native_update_steps=False,
        job_type="paired-evaluation",
        log_evaluation_artifacts=True,
        max_evaluation_table_rows=counts[split],
    )
    config = EmbodiedExperimentConfig.model_validate(raw)
    if config.resume_contract_fingerprint != training.resume_contract_fingerprint:
        raise ValueError("Evaluation changed the policy or training contract")
    if (
        not config.observability.wandb.enabled
        or config.observability.wandb.mode != "online"
    ):
        raise ValueError("Paired evaluation requires online W&B logging")
    return config, loaded[split]


def outcome_rows(evaluation):
    rows = []
    for trajectory in sorted(
        evaluation.trajectories, key=lambda t: t.metadata["bank_index"]
    ):
        metadata = trajectory.metadata
        actions = make_json_safe([action.raw for action in trajectory.actions])
        rows.append(
            {
                "episode": metadata["bank_index"],
                "scenario_id": metadata["scenario_id"],
                "task": trajectory.task,
                "environment_seed": metadata["environment_seed"],
                "policy_seed": metadata["policy_seed"],
                "reset_sha256": metadata["reset_sha256"],
                "success": trajectory.metrics["success"],
                "reward": trajectory.reward,
                "completed": True,
                "termination_reason": metadata["termination_reason"],
                "episode_steps": trajectory.metrics["environment_steps"],
                "action_count": len(actions),
                "action_sequence_sha256": hashlib.sha256(
                    json.dumps(actions, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
    return rows


def record_evidence(evaluation, *, config, bank_path, checkpoint, baseline=None):
    rows = outcome_rows(evaluation)
    count = config.evaluation.episodes
    if (
        len(rows) != count
        or [row["scenario_id"] for row in rows] != config.evaluation.fixed_scenarios
    ):
        raise ValueError("Evaluation outcomes differ from the fixed episode population")
    successes = sum(row["success"] for row in rows)
    if any(row["success"] not in (0, 1) for row in rows) or (
        evaluation.metrics.get("episodes") != count
        or evaluation.metrics.get("successes") != successes
        or evaluation.metrics.get("success_rate") != successes / count
    ):
        raise ValueError("Evaluation metrics differ from the episode outcomes")
    if (checkpoint is None and evaluation.step != 0) or (
        checkpoint is not None
        and checkpoint_version(checkpoint, config) != evaluation.step
    ):
        raise ValueError("Evaluation update differs from the selected checkpoint")
    identity = evaluation_evidence_identity(
        policy_path=config.policy.path,
        policy_revision=config.policy.revision,
        adapter_path=None,
        checkpoint_path=str(checkpoint) if checkpoint else None,
        evaluation_manifest_path=str(bank_path),
        step=evaluation.step,
    )
    if config.evaluation.data_role == "sealed_test":
        require_resolved_sealed_identity(identity)
    outcomes = _write_evaluation_outcomes(
        config=config,
        step=evaluation.step,
        rows=rows,
        protected_path=baseline,
    )
    metrics = dict(evaluation.metrics)
    if baseline is not None:
        metrics.update(compare_paired_evaluation_reports(baseline, outcomes))
    evidence = _write_evaluation_evidence(
        config=config,
        step=evaluation.step,
        outcome_path=outcomes,
        baseline_path=baseline,
        metrics=metrics,
        identity=identity,
    )
    return replace(
        evaluation,
        metrics=metrics,
        artifacts={
            **evaluation.artifacts,
            "episode_outcomes_json": str(outcomes),
            "evaluation_evidence_json": str(evidence),
        },
    )


def verify_campaign(run_path, expected, *, namespace, timeout=1800):
    steps = list(expected)
    deadline = time.monotonic() + timeout
    actual = []
    while time.monotonic() < deadline:
        run, rows = read_wandb(
            lambda: _wandb_history(run_path), deadline=deadline, label="history"
        )
        actual = [row["_step"] for row in rows]
        if actual != steps[: len(actual)]:
            raise RuntimeError(f"Unexpected evaluation-only history steps: {actual}")
        if actual == steps:
            files = read_wandb(
                lambda: {file.name: file.size for file in run.files(per_page=500)},
                deadline=deadline,
                label="media files",
            )
            complete = True
            for row, step in zip(rows, steps, strict=True):
                contract = expected[step]
                if (
                    row.get("experiment/update") != step
                    or row.get(f"{namespace}/success_rate") != contract["success_rate"]
                ):
                    raise RuntimeError(
                        "Evaluation value or native step differs from measured results"
                    )
                media = [
                    value
                    for key, value in row.items()
                    if key.startswith("media/simulation/eval/")
                    and isinstance(value, dict)
                ]
                if len(media) != contract["episodes"]:
                    raise RuntimeError(
                        "Evaluation media count differs from episode count"
                    )
                complete &= all(files.get(value["path"], 0) > 0 for value in media)
            artifacts = read_wandb(
                lambda: {
                    artifact.metadata.get("update"): artifact.name
                    for artifact in run.logged_artifacts()
                    if artifact.type == "evaluation"
                },
                deadline=deadline,
                label="evaluation artifacts",
            )
            if complete and all(step in artifacts for step in steps):
                return {
                    "native_steps": actual,
                    "evaluation_artifacts": artifacts,
                    "ui_checked": False,
                }
        time.sleep(10)
    raise TimeoutError(f"Evaluation history/media/artifacts are incomplete: {actual}")


def runtime_inventory(python):
    probe = """
import json,sys
from importlib.metadata import version,PackageNotFoundError
versions={}
for name in ('openpi','openpi-client','torch','numpy','transformers','peft','isaaclab','isaaclab-arena','newton','mujoco','mujoco-warp','warp-lang','nvidia-curobo'):
    try: versions[name]=version(name)
    except PackageNotFoundError: pass
print(json.dumps({'python':sys.version,'packages':versions}))
"""
    return json.loads(
        subprocess.check_output([python, "-c", probe], text=True, timeout=60)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="Original training configuration"
    )
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "sealed"), required=True)
    parser.add_argument(
        "--checkpoint", type=Path, help="One checkpoint selected before evaluation"
    )
    parser.add_argument("--policy-python")
    parser.add_argument("--sim-python")
    parser.add_argument("--port", type=int, default=8900)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Prepare and check inputs without evaluating any policy",
    )
    args = parser.parse_args()
    training = EmbodiedExperimentConfig.model_validate_json(args.config.read_text())
    config, bank = evaluation_config(
        training, banks=args.banks, output=args.output_dir, split=args.split
    )
    step = checkpoint_version(args.checkpoint, training) if args.checkpoint else None
    if step is not None and step <= 0:
        parser.error("The selected candidate must be a trained checkpoint")
    if not args.preflight and (
        step is None or not args.policy_python or not args.sim_python
    ):
        parser.error("Execution requires checkpoint, policy-python and sim-python")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "evaluation-config.json"
    encoded = config.model_dump(mode="json")
    if config_path.exists() and json.loads(config_path.read_text()) != encoded:
        raise ValueError("Evaluation configuration changed after preparation")
    if (args.output_dir / "run.json").exists():
        raise FileExistsError(
            "This evaluation has already started; do not overwrite it"
        )
    write_json_atomic(config_path, encoded)
    if args.preflight:
        print(
            json.dumps(
                {
                    "preflight_only": True,
                    "policy_executions": 0,
                    "episodes": len(bank),
                    "bank_sha256": bank.fingerprint,
                    "candidate_update": step,
                }
            )
        )
        return
    inventory = {
        name: runtime_inventory(python)
        for name, python in (
            ("policy", args.policy_python),
            ("simulator", args.sim_python),
        )
    }
    runtime_path = args.output_dir / "runtimes.json"
    write_json_atomic(runtime_path, inventory)
    runner = DroidRunner(args, config)
    observer = WandbWeaveObserver.start(config)
    path = wandb_run_path(observer.wandb_run)
    write_json_atomic(
        args.output_dir / "run.json",
        {"path": path, "candidate_update": step, "checkpoint": str(args.checkpoint)},
    )
    expected, baseline = {}, None
    exit_code = 1
    try:
        for version, checkpoint in ((0, None), (step, args.checkpoint)):
            roots, seconds = runner.collect(
                version=version,
                checkpoint=checkpoint,
                training=False,
                evaluation_split=args.split,
            )
            evaluation = read_evaluation(
                [root / "rollout" for root in roots],
                version=version,
                config=config,
                bank=bank,
                split=args.split,
            )
            evaluation = replace(
                evaluation,
                metrics={**evaluation.metrics, "evaluation_wall_seconds": seconds},
                artifacts={
                    **evaluation.artifacts,
                    "runtime_inventory_json": str(runtime_path),
                },
            )
            evaluation = record_evidence(
                evaluation,
                config=config,
                bank_path=args.banks / args.split,
                checkpoint=checkpoint,
                baseline=baseline,
            )
            asyncio.run(
                observer.log_evaluation_checkpoint(
                    version,
                    evaluation,
                    config,
                    checkpoint_role="initial_policy" if version == 0 else "candidate",
                    use_native_wandb_step=True,
                )
            )
            if observer.weave_client is not None:
                observer.weave_client.flush()
            expected[version] = {
                "success_rate": evaluation.metrics["success_rate"],
                "episodes": len(bank),
            }
            verified = verify_campaign(
                path,
                expected,
                namespace="test" if args.split == "sealed" else "validation",
            )
            write_json_atomic(
                args.output_dir / f"verified-{version:06d}.json", verified
            )
            if version == 0:
                baseline = evaluation.artifacts["episode_outcomes_json"]
        exit_code = 0
    finally:
        observer.close(exit_code=exit_code)


if __name__ == "__main__":
    main()
