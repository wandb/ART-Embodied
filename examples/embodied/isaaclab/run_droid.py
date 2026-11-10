"""Run DROID collection, distributed updates, and W&B logging in separate runtimes."""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
from urllib.error import URLError
from urllib.request import urlopen

from art_embodied.checkpointing import CheckpointManager
from art_embodied.config import EmbodiedExperimentConfig
from art_embodied.experiment import EvaluationResult, ExperimentProgress
from art_embodied.observability import WandbWeaveObserver
from art_embodied.trajectories import (
    Action,
    EmbodiedTrajectory,
    EmbodiedTrajectoryGroup,
    MediaRef,
)
from art_embodied.types import LocalTrainResult
from art_embodied.utils import make_json_safe, write_json_atomic


def assigned_devices(devices, visible=None):
    """Translate logical device IDs without escaping the scheduler allocation."""
    allocated = visible.split(",") if visible else None
    result = []
    for device in devices:
        if not device.startswith("cuda:") or not device[5:].isdigit():
            raise ValueError("DROID workers require explicit CUDA device IDs")
        index = int(device[5:])
        if allocated is not None and index >= len(allocated):
            raise ValueError("Configured GPU lies outside CUDA_VISIBLE_DEVICES")
        result.append(allocated[index] if allocated else str(index))
    if len(set(result)) != len(result):
        raise ValueError("Collection devices must be distinct")
    return result


def partition_indices(indices, *, group_size, batch_size, devices):
    if group_size < 1 or batch_size % group_size or devices < 1:
        raise ValueError("Invalid vector collection geometry")
    per_batch = batch_size // group_size
    if not indices or len(indices) % per_batch or len(set(indices)) != len(indices):
        raise ValueError("Distinct reset indices must fill vector batches")
    batches = [indices[i : i + per_batch] for i in range(0, len(indices), per_batch)]
    count = min(devices, len(batches))
    return [sum(batches[rank::count], []) for rank in range(count)]


def train_indices(size, count, *, seed, version):
    if count > size or count < 1:
        raise ValueError("Training bank is smaller than the requested distinct groups")
    return random.Random(f"{seed}:{version}").sample(range(size), count)


def checkpoint_version(path, config):
    checked = CheckpointManager().validate(
        path, expected_resume_contract_fingerprint=config.resume_contract_fingerprint
    )
    return int(checked.manifest["metadata"]["step"])


def update_limit(*, completed, total, stop_after=None):
    end = total if stop_after is None else stop_after
    if not 0 <= completed < end <= total:
        raise ValueError(
            "Stop update must follow the checkpoint and not exceed the training target"
        )
    return end


def resume_configuration(config, *, checkpoint, run_id):
    raw = config.model_dump(mode="json")
    raw["storage"]["resume_from_checkpoint"] = str(checkpoint)
    raw["evaluation"]["evaluate_before_training"] = False
    raw["observability"]["wandb"].update(
        connection="resume", resume="must", run_id=run_id
    )
    resumed = EmbodiedExperimentConfig.model_validate(raw)
    if resumed.resume_contract_fingerprint != config.resume_contract_fingerprint:
        raise ValueError("Resume configuration changed the training contract")
    return resumed


def wandb_run_path(run):
    parts = run.path.split("/") if isinstance(run.path, str) else list(run.path)
    if len(parts) != 3 or any(
        not isinstance(p, str) or not p or "/" in p for p in parts
    ):
        raise ValueError("Invalid W&B entity/project/run path")
    return "/".join(parts)


def read_logging_groups(
    path, *, expected_groups, expected_trajectories, expected_chunks
):
    groups = []
    for data in json.loads(Path(path).read_text()):
        if data.get("exceptions"):
            raise ValueError(
                "Completed DROID updates cannot contain collection exceptions"
            )
        # The group's constructor accepts trajectory objects, not JSON dictionaries.
        groups.append(
            EmbodiedTrajectoryGroup(
                [EmbodiedTrajectory.model_validate(t) for t in data["trajectories"]],
                metadata=data.get("metadata"),
                metrics=data.get("metrics"),
                logs=data.get("logs"),
            )
        )
    if (
        len(groups) != expected_groups
        or sum(len(g) for g in groups) != expected_trajectories
        or sum(len(t.actions) for g in groups for t in g) != expected_chunks
    ):
        raise ValueError("Logging handoff lost groups, trajectories, or action chunks")
    return groups


def read_wandb(operation, *, deadline, label):
    """Retry API reads, never writes or validation of the returned evidence."""
    from requests.exceptions import RequestException
    import wandb

    while True:
        try:
            return operation()
        except (wandb.errors.CommError, RequestException, TypeError) as error:
            if isinstance(error, wandb.errors.AuthenticationError):
                raise
            if isinstance(error, TypeError) and str(error) != (
                "'NoneType' object is not subscriptable"
            ):
                raise
            # The SDK's file paginator can raise this TypeError for a null
            # GraphQL page. Restart the read; never accept a partial file list.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"W&B {label} readback did not recover") from error
            print(
                f"[ART-Embodied] W&B {label} read failed "
                f"({type(error).__name__}); retrying without writing history",
                flush=True,
            )
            time.sleep(min(10, remaining))


def verify_history(run_path, *, last_step, expected, timeout=1800):
    """Read back committed scalar rows, media files, and model artifacts."""
    deadline = time.monotonic() + timeout
    actual = None
    while time.monotonic() < deadline:
        run, rows = read_wandb(
            lambda: _wandb_history(run_path), deadline=deadline, label="history"
        )
        actual = [r["_step"] for r in rows]
        if actual and (actual != list(range(len(actual))) or actual[-1] > last_step):
            raise RuntimeError(f"Unexpected native W&B steps: {actual}")
        if actual == list(range(last_step + 1)):
            if [r.get("experiment/update") for r in rows] != actual:
                raise RuntimeError("Custom and native W&B update axes differ")
            files = read_wandb(
                lambda: {f.name: f.size for f in run.files(per_page=500)},
                deadline=deadline,
                label="media files",
            )
            complete = True
            for step, contract in expected.items():
                row = rows[int(step)]
                for metric, value in contract["metrics"].items():
                    if row.get(metric) != value:
                        raise RuntimeError(f"W&B metric mismatch at {step}: {metric}")
                media = [
                    v
                    for k, v in row.items()
                    if k.startswith("media/simulation/") and isinstance(v, dict)
                ]
                if len(media) != contract["media"]:
                    raise RuntimeError(
                        f"W&B media count mismatch at {step}: {len(media)}"
                    )
                complete &= all(files.get(v["path"], 0) > 0 for v in media)
            artifacts = read_wandb(
                lambda: {
                    artifact.metadata.get("update"): artifact.name
                    for artifact in run.logged_artifacts()
                    if artifact.type == "model"
                },
                deadline=deadline,
                label="model artifacts",
            )
            if complete and all(step in artifacts for step in range(1, last_step + 1)):
                return {
                    "native_steps": actual,
                    "model_artifacts": [artifacts[s] for s in range(1, last_step + 1)],
                    "model_artifacts_by_update": {
                        s: artifacts[s] for s in range(1, last_step + 1)
                    },
                    "ui_checked": False,
                }
        time.sleep(10)
    raise TimeoutError(f"W&B history/media delivery incomplete: {actual}")


def _wandb_history(run_path):
    import wandb

    run = wandb.Api(timeout=60).run(run_path)
    return run, list(run.scan_history())


def read_evaluation(directories, *, version, config, bank, split="development"):
    if split not in {"development", "sealed"}:
        raise ValueError("Fixed evaluation requires a held-out reset bank")
    trajectories = []
    seen = set()
    artifacts = {}
    for actor, directory in enumerate(directories):
        report = json.loads((directory / "report.json").read_text())
        if (
            report["model_server"]["mode"] != "ode"
            or report["optimizer_updates"] != version
        ):
            raise ValueError("Evaluation used an unexpected sampler or policy")
        if (
            report["split"] != split
            or not report["initial_state_and_camera_replay_exact"]
        ):
            raise ValueError("Invalid fixed evaluation reset")
        chunks = json.loads((directory / "actions.json").read_text())
        episodes = json.loads((directory / "episodes.json").read_text())
        if len(episodes) != report["episodes"]:
            raise ValueError("Incomplete evaluation episodes")
        for episode in episodes:
            index = episode["bank_index"]
            if index in seen or episode["reset_sha256"] != bank.hashes[index]:
                raise ValueError("Evaluation repeated or changed a fixed initial state")
            seen.add(index)
            actions = []
            for chunk in chunks:
                if (
                    chunk["batch"] == episode["batch"]
                    and episode["row"] in chunk["active_rows"]
                ):
                    row = chunk["active_rows"].index(episode["row"])
                    actions.append(
                        Action(
                            step=chunk["chunk"],
                            kind="continuous",
                            raw=chunk["actions"][row],
                            metadata={
                                "primitive_loss_mask": chunk["execution_mask"][row]
                            },
                        )
                    )
            trajectories.append(
                EmbodiedTrajectory(
                    task=config.environment.kwargs["instruction"],
                    reward=float(episode["success"]),
                    initial_policy_version=version,
                    final_policy_version=version,
                    metrics={
                        "success": float(episode["success"]),
                        "environment_steps": episode["env_steps"],
                    },
                    metadata={
                        "scenario_id": f"{split}-{index}",
                        "bank_index": index,
                        "environment_seed": episode["environment_seed"],
                        "reset_sha256": episode["reset_sha256"],
                        "reset_state": make_json_safe(bank.select([index])),
                        "policy_seed": episode["policy_seed_base"],
                        "termination_reason": episode["termination_reason"],
                    },
                    actions=actions,
                    media=[MediaRef.model_validate(m) for m in episode["media"]],
                )
            )
        artifacts[f"episodes-{actor}"] = str(directory / "episodes.json")
    if seen != set(range(config.evaluation.episodes)):
        raise ValueError("Evaluation did not cover the fixed held-out set")
    successes = sum(t.reward for t in trajectories)
    return EvaluationResult(
        step=version,
        metrics={
            "success_rate": successes / len(trajectories),
            "episodes": len(trajectories),
            "successes": successes,
        },
        artifacts=artifacts,
        trajectories=tuple(trajectories),
    )


class DroidRunner:
    def __init__(self, args, config):
        if (
            config.policy.type != "openpi_pi05_droid"
            or config.environment.type != "isaaclab"
            or config.environment.kwargs.get("physics_condition")
            != "native_mujoco_warp_fp32_panda_armature_0.1"
            or config.environment.kwargs.get("action_chunk_size") != 15
            or config.rollout.max_episode_steps != 1050
            or config.rollout.max_policy_steps != 70
        ):
            raise ValueError(
                "This DROID collection recipe requires Newton/Warp, Panda armature "
                "0.1, 15-action chunks, and a 1050-step (70-chunk) horizon"
            )
        if config.environment.kwargs["num_envs"] != config.training.microbatch_size:
            raise ValueError("DROID collection and replay must use the same batch size")
        self.args, self.config = args, config
        self.output = Path(config.storage.output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.devices = assigned_devices(
            config.runtime.rollout_devices, os.environ.get("CUDA_VISIBLE_DEVICES")
        )
        self.batch_size = int(config.environment.kwargs["num_envs"])

    def collect(self, *, version, checkpoint, training, evaluation_split="development"):
        config = self.config
        if evaluation_split not in {"development", "sealed"} or (
            training and evaluation_split != "development"
        ):
            raise ValueError("Sealed states cannot be used for training")
        split = "train" if training else evaluation_split
        manifest = json.loads((self.args.banks / split / "manifest.json").read_text())
        size = len(manifest["reset_hashes"])
        indices = (
            train_indices(
                size,
                config.rollout.groups_per_update,
                seed=config.experiment.seed,
                version=version,
            )
            if training
            else list(range(config.evaluation.episodes))
        )
        group_size = config.algorithm.group_size if training else 1
        partitions = partition_indices(
            indices,
            group_size=group_size,
            batch_size=self.batch_size,
            devices=len(self.devices),
        )
        directory = self.output / f"{split}-{version:06d}"
        directory.mkdir(exist_ok=False)
        config_path = directory / "config.json"
        inference_config = config.model_dump(mode="json")
        # Collection servers restore the explicit adapter below, not training
        # state from the original (possibly pruned) resume checkpoint.
        inference_config["storage"]["resume_from_checkpoint"] = None
        write_json_atomic(config_path, inference_config)
        seed = (
            config.experiment.seed * 1000 + version * 10
            if training
            else config.evaluation.seeds[0]
        )
        mode = "sde" if training else "ode"

        def actor(rank):
            root = directory / f"actor-{rank}"
            root.mkdir()
            selection = root / "indices.json"
            write_json_atomic(selection, partitions[rank])
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.devices[rank])
            port = self.args.port + rank
            server_command = [
                self.args.policy_python,
                "-u",
                "-m",
                "examples.embodied.isaaclab.openpi_server",
                "--checkpoint",
                config.policy.path,
                "--config-name",
                config.policy.load_kwargs["config_name"],
                "--config",
                str(config_path),
                "--mode",
                mode,
                "--port",
                str(port),
                "--batch-size",
                str(self.batch_size),
                "--noise-level",
                str(config.algorithm.flow_sde.noise_level),
                "--denoising-steps",
                str(config.algorithm.flow_sde.num_denoise_steps),
                "--policy-version",
                str(version),
            ]
            if checkpoint:
                server_command += [
                    "--adapter-checkpoint",
                    str(Path(checkpoint) / "policy"),
                ]
            if training:
                server_command += ["--record-dir", str(root / "records")]
            with (root / "server.log").open("w") as log:
                server = subprocess.Popen(
                    server_command, env=env, stdout=log, stderr=subprocess.STDOUT
                )
                try:
                    deadline = time.monotonic() + 900
                    while True:
                        if server.poll() is not None:
                            raise RuntimeError(
                                f"Policy server failed: {root / 'server.log'}"
                            )
                        try:
                            with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2):
                                break
                        except (URLError, TimeoutError):
                            if time.monotonic() > deadline:
                                raise TimeoutError("Policy server startup timed out")
                            time.sleep(2)
                    command = [
                        self.args.sim_python,
                        "-u",
                        "-m",
                        "examples.embodied.isaaclab.droid_evaluate",
                        "--output-dir",
                        str(root / "rollout"),
                        "--port",
                        str(port),
                        "--mode",
                        mode,
                        "--num-envs",
                        str(self.batch_size),
                        "--episodes",
                        str(len(partitions[rank]) * group_size),
                        "--seed",
                        str(seed),
                        "--reset-bank",
                        str(self.args.banks / split),
                        "--split",
                        split,
                        "--bank-indices",
                        str(selection),
                        "--group-size",
                        str(group_size),
                        "--policy-version",
                        str(version),
                        "--pick-up-object",
                        config.environment.kwargs["pick_up_object"],
                        "--destination-location",
                        config.environment.kwargs["destination_location"],
                        "--instruction",
                        config.environment.kwargs["instruction"],
                        "--viz",
                        "none",
                    ]
                    with (root / "simulator.log").open("w") as simlog:
                        subprocess.run(
                            command,
                            env=env,
                            stdout=simlog,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=config.runtime.worker_timeout_seconds,
                        )
                finally:
                    server.terminate()
                    try:
                        server.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait()
            return root

        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=len(partitions)) as pool:
            roots = list(pool.map(actor, range(len(partitions))))
        seconds = time.monotonic() - start
        write_json_atomic(
            directory / "collection.json",
            {
                "wall_seconds": seconds,
                "policy_version": version,
                "split": split,
                "actors": len(roots),
                "episodes": len(indices) * group_size,
            },
        )
        return roots, seconds

    def update(self, roots, *, version, checkpoint):
        raw = self.config.model_dump(mode="json")
        raw["storage"]["resume_from_checkpoint"] = (
            str(checkpoint) if checkpoint else None
        )
        configuration = self.output / f"update-{version + 1:06d}.json"
        write_json_atomic(configuration, raw)
        result_dir = self.output / f"update-{version + 1:06d}"
        command = [
            self.args.policy_python,
            "-u",
            "-m",
            "examples.embodied.isaaclab.train_update",
            "--config",
            str(configuration),
            "--reset-bank",
            str(self.args.banks / "train"),
            "--result-dir",
            str(result_dir),
        ]
        for root in roots:
            command += [
                "--rollout-dir",
                str(root / "rollout"),
                "--record-dir",
                str(root / "records"),
            ]
        with (self.output / f"update-{version + 1:06d}.log").open("w") as log:
            subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=self.config.runtime.worker_timeout_seconds,
            )
        result = json.loads((result_dir / "result.json").read_text())
        if (
            result["step"] != version + 1
            or checkpoint_version(result["checkpoint_path"], self.config) != version + 1
        ):
            raise ValueError("Backend checkpoint does not match the completed update")
        groups = read_logging_groups(
            result_dir / "trace-groups.json",
            expected_groups=result["groups"],
            expected_trajectories=result["trajectories"],
            expected_chunks=result["chunks"],
        )
        return result, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--policy-python", required=True)
    parser.add_argument("--sim-python", required=True)
    parser.add_argument("--port", type=int, default=8900)
    parser.add_argument("--verify-resume-at", type=int, default=1)
    parser.add_argument(
        "--stop-after-update",
        type=int,
        help="Exit normally after this update is saved and its W&B delivery verified",
    )
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--run-id", help="Existing W&B run ID, required with --resume-from"
    )
    parser.add_argument(
        "--resume-baseline",
        action="store_true",
        help="Continue an existing run that committed only its initial evaluation",
    )
    args = parser.parse_args()
    if bool(args.resume_from) != bool(args.run_id):
        parser.error("--resume-from and --run-id must be supplied together")
    config = EmbodiedExperimentConfig.model_validate_json(args.config.read_text())
    if args.resume_from:
        config = resume_configuration(
            config, checkpoint=args.resume_from, run_id=args.run_id
        )
    if (
        not config.observability.wandb.enabled
        or not config.observability.wandb.native_update_steps
    ):
        parser.error("This lifecycle requires W&B with native update steps")
    if config.evaluation.data_role != "development" or (
        not config.storage.resume_from_checkpoint
        and not config.evaluation.evaluate_before_training
    ):
        parser.error("A fixed development baseline is required")
    runner = DroidRunner(args, config)
    from art_embodied.integrations.isaaclab_reset_bank import IsaacLabResetBank
    from examples.embodied.isaaclab.droid_environment import DroidSceneConfig

    scene = DroidSceneConfig(
        **{
            k: config.environment.kwargs[k]
            for k in ("pick_up_object", "destination_location", "instruction")
        }
    )
    banks = {
        split: IsaacLabResetBank.load(
            args.banks / split, contract=scene.reset_contract(), split=split
        )
        for split in ("train", "development")
    }
    if (
        banks["train"].fingerprint
        != config.environment.kwargs["train_reset_bank_sha256"]
        or banks["development"].fingerprint
        != config.evaluation.kwargs["reset_bank_sha256"]
    ):
        raise ValueError("Reset banks differ from the configured fingerprints")
    checkpoint = config.storage.resume_from_checkpoint
    if args.resume_baseline and checkpoint:
        parser.error("resume-baseline cannot be combined with a trained checkpoint")
    version = checkpoint_version(checkpoint, config) if checkpoint else 0
    final_update = update_limit(
        completed=version,
        total=config.training.updates,
        stop_after=args.stop_after_update,
    )
    evidence_path = runner.output / "history-contract.json"
    resuming = bool(checkpoint) or args.resume_baseline
    expected = json.loads(evidence_path.read_text()) if resuming else {}
    if args.resume_baseline:
        original = EmbodiedExperimentConfig.model_validate_json(
            (runner.output / "development-000000/config.json").read_text()
        )
        if (
            set(expected) != {"0"}
            or original.resume_contract_fingerprint
            != config.resume_contract_fingerprint
        ):
            raise ValueError(
                "Baseline resume requires an unchanged experiment and only step 0"
            )
    if resuming:
        if (
            not config.observability.wandb.run_id
            or config.observability.wandb.connection != "resume"
        ):
            raise ValueError("Checkpoint resume requires the existing W&B run ID")
        verified = verify_history(
            f"{config.observability.wandb.entity}/{config.observability.wandb.project}/{config.observability.wandb.run_id}",
            last_step=version,
            expected=expected,
        )
        write_json_atomic(runner.output / f"verified-{version:06d}.json", verified)
    elif (runner.output / "run.json").exists():
        raise FileExistsError(
            "Output directory already belongs to a run; resume explicitly"
        )
    observer = WandbWeaveObserver.start(config)
    run_path = wandb_run_path(observer.wandb_run)
    write_json_atomic(
        runner.output / "run.json", {"id": observer.wandb_run.id, "path": run_path}
    )
    exit_code = 1
    try:
        if resuming:
            observer._native_update_history.require_next_step(version + 1)
        if not resuming:
            roots, seconds = runner.collect(version=0, checkpoint=None, training=False)
            evaluation = read_evaluation(
                [r / "rollout" for r in roots],
                version=0,
                config=config,
                bank=banks["development"],
            )
            asyncio.run(observer.log_initial_evaluation(evaluation, config))
            expected["0"] = {
                "metrics": {
                    "validation/success_rate": evaluation.metrics["success_rate"]
                },
                "media": config.evaluation.episodes,
            }
            write_json_atomic(evidence_path, expected)
            write_json_atomic(
                runner.output / "verified-000000.json",
                verify_history(run_path, last_step=0, expected=expected),
            )
        for version in range(version, final_update):
            step = version + 1
            asyncio.run(
                observer.log_progress(
                    ExperimentProgress(
                        phase="rollout",
                        status="started",
                        update=step,
                        total=config.rollout.groups_per_update,
                    ),
                    config,
                )
            )
            roots, rollout_seconds = runner.collect(
                version=version, checkpoint=checkpoint, training=True
            )
            asyncio.run(
                observer.log_progress(
                    ExperimentProgress(
                        phase="rollout",
                        status="completed",
                        update=step,
                        completed=config.rollout.groups_per_update,
                        total=config.rollout.groups_per_update,
                    ),
                    config,
                )
            )
            asyncio.run(
                observer.log_progress(
                    ExperimentProgress(phase="training", status="started", update=step),
                    config,
                )
            )
            data, groups = runner.update(roots, version=version, checkpoint=checkpoint)
            checkpoint = data["checkpoint_path"]
            # The subprocess has restored the backend before recording a resume.
            asyncio.run(
                observer.log_progress(
                    ExperimentProgress(phase="update", status="started", update=step),
                    config,
                )
            )
            asyncio.run(observer.log_rollout(version, groups, config))
            evaluation = None
            evaluation_seconds = 0
            if step % config.evaluation.every_updates == 0:
                eval_roots, evaluation_seconds = runner.collect(
                    version=step, checkpoint=checkpoint, training=False
                )
                evaluation = read_evaluation(
                    [r / "rollout" for r in eval_roots],
                    version=step,
                    config=config,
                    bank=banks["development"],
                )
            metrics = dict(data["metrics"])
            metrics.update(
                {
                    "performance/rollout_wall_seconds": rollout_seconds,
                    "performance/training_wall_seconds": data["wall_seconds"],
                    "performance/backend_update_seconds": data[
                        "backend_update_seconds"
                    ],
                    "performance/evaluation_wall_seconds": evaluation_seconds,
                    "performance/rollout_chunks": data["chunks"],
                    "performance/rollout_trajectories": data["trajectories"],
                }
            )
            asyncio.run(
                observer.log_step(
                    step,
                    groups,
                    LocalTrainResult(
                        step=step, metrics=metrics, checkpoint_path=checkpoint
                    ),
                    evaluation,
                    config,
                )
            )
            if observer.weave_client is not None:
                observer.weave_client.flush()
            expected[str(step)] = {
                "metrics": {
                    "train/success_rate": data["successes"] / data["trajectories"]
                },
                "media": data["trajectories"]
                + (config.evaluation.episodes if evaluation else 0),
            }
            if evaluation:
                expected[str(step)]["metrics"]["validation/success_rate"] = (
                    evaluation.metrics["success_rate"]
                )
            write_json_atomic(evidence_path, expected)
            checked = verify_history(run_path, last_step=step, expected=expected)
            write_json_atomic(runner.output / f"verified-{step:06d}.json", checked)
            asyncio.run(
                observer.log_progress(
                    ExperimentProgress(phase="update", status="completed", update=step),
                    config,
                )
            )
            if step == args.verify_resume_at:
                observer.close()
                verify_history(run_path, last_step=step, expected=expected)
                config = resume_configuration(
                    config, checkpoint=checkpoint, run_id=run_path.split("/")[-1]
                )
                runner.config = config
                write_json_atomic(
                    runner.output / "resume-config.json", config.model_dump(mode="json")
                )
                observer = WandbWeaveObserver.start(config)
                observer._native_update_history.require_next_step(step + 1)
            if not config.storage.retain_rollout_payloads:
                for root in roots:
                    shutil.rmtree(root / "records")
            print(
                json.dumps(
                    {
                        "completed_update": step,
                        "checkpoint": checkpoint,
                        "wandb": run_path,
                    }
                ),
                flush=True,
            )
        exit_code = 0
    finally:
        observer.close(exit_code=exit_code)


if __name__ == "__main__":
    main()
