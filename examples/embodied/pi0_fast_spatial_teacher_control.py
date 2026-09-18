"""Paired full-dataset Spatial teacher controls with closed-loop evaluation.

Not GRPO qualification: hold data, weights, optimizer and evaluation fixed;
change only the native SFT versus sampler teacher-likelihood forward.
"""

import argparse
import asyncio
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time


def verify_wandb_identity(viewer):
    """Check the operator-selected account before creating a diagnostic run."""
    expected = os.environ.get("ART_EMBODIED_EXPECTED_WANDB_EMAIL")
    if not expected:
        raise ValueError("Set ART_EMBODIED_EXPECTED_WANDB_EMAIL before running")
    email = viewer.get("email") if isinstance(viewer, dict) else viewer.email
    if email != expected:
        raise ValueError("Unexpected W&B account")


@contextmanager
def diagnostic_cuda_mapping():
    """Scope nested-worker visibility to this isolated diagnostic process only."""
    import art_embodied.rollout_process as rollout_process

    original = rollout_process._worker_environment
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if not devices or any(not device.isdigit() for device in devices):
        raise ValueError("Diagnostic requires explicit numeric Slurm GPU allocation")

    def environment(device):
        if device == "cpu":
            return original(device)
        index = int(device.removeprefix("cuda:"))
        if not 0 <= index < len(devices):
            raise ValueError("Diagnostic worker is outside the allocated GPUs")
        return original(f"cuda:{devices[index]}")

    rollout_process._worker_environment = environment
    try:
        yield
    finally:
        rollout_process._worker_environment = original


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def configure(spec, output, *, evaluation_manifest=None, rollout_gpus=1):
    """Preserve the existing task, reset states and policy; isolate worker IO."""
    spec = deepcopy(spec)
    raw = spec["config"]
    if raw["policy"]["type"] != "pi0_fast":
        raise ValueError("This diagnostic must not operate on another policy family")
    if raw["environment"]["kwargs"]["task_ids"] != [5]:
        raise ValueError("Require the existing Spatial task 5 control")
    if raw["policy"]["lora"]["rank_partition"] is not None:
        raise ValueError("No task routing or rank partitioning in this control")
    if not 1 <= rollout_gpus <= 8:
        raise ValueError("Require 1..8 GPUs on one node")
    raw["runtime"]["rollout_devices"] = [f"cuda:{i}" for i in range(rollout_gpus)]
    raw["runtime"]["rollout_execution"]["actors_per_device"] = 2
    raw["rollout"]["workers"] = 2 * rollout_gpus
    raw["runtime"]["worker_handoff_dir"] = str((output / "handoffs").resolve())
    raw["storage"]["output_dir"] = str((output / "runtime").resolve())
    raw["observability"]["wandb"]["enabled"] = False
    raw["observability"]["weave"]["enabled"] = False
    # The outer diagnostic writer logs SFT and evaluation together. This local
    # evaluator must not start a competing W&B writer or demand RL train video.
    raw["observability"]["require_train_video"] = False
    raw["observability"]["require_evaluation_video"] = False
    raw["evaluation"]["pre_training_success_gate"] = None
    if evaluation_manifest is not None:
        from examples.embodied.libero.state_manifest import load_state_manifest

        manifest = load_state_manifest(
            evaluation_manifest, expected_suite_name="libero_spatial"
        )
        if len(manifest.entries) != 100 or {e.task_id for e in manifest.entries} != {5}:
            raise ValueError("Require all 100 task5 development states")
        raw["environment"]["kwargs"]["evaluation_state_manifest"] = str(manifest.path)
        raw["evaluation"]["fixed_scenarios"] = [e.id for e in manifest.entries]
    return spec


def check_inventory(inventory, data):
    if inventory["dataset_task_index"] != 32 or inventory["episode_count"] != 39:
        raise ValueError("Wrong teacher population")
    if len(data) != inventory["frame_count"]:
        raise ValueError("Incomplete teacher frames")
    if set(map(int, data.hf_dataset["episode_index"])) != set(inventory["episodes"]):
        raise ValueError("Teacher episode population mismatch")


def verify_run(run, records, output, artifact, *, media_required):
    import wandb

    from examples.embodied.wandb_verified_recovery import assert_history

    deadline = time.monotonic() + 120
    while True:
        try:
            remote = wandb.Api(timeout=20).run(run.path)
            rows = list(remote.scan_history(page_size=1000, max_step=1000))
            assert_history(rows, records)
            if media_required and "media/evaluation" not in rows[0]:
                raise ValueError("Initial evaluation video absent")
            if media_required:
                media = rows[0]["media/evaluation"]
                downloaded = remote.file(media["path"]).download(
                    root=str(output / "remote-media"), replace=True
                )
                media_path = Path(downloaded.name)
                downloaded.close()
                if (
                    hashlib.sha256(media_path.read_bytes()).hexdigest()
                    != media["sha256"]
                ):
                    raise ValueError("Evaluation video readback mismatch")
            saved = wandb.Api(timeout=20).artifact(artifact)
            path = saved.get_entry("adapter_model.safetensors").download(
                root=str(output / "remote-checkpoint")
            )
            local = output / f"checkpoint-{records[-1]['experiment/update']:04d}"
            if (
                hashlib.sha256(Path(path).read_bytes()).digest()
                != hashlib.sha256(
                    (local / "adapter_model.safetensors").read_bytes()
                ).digest()
            ):
                raise ValueError("Model artifact readback mismatch")
            write(
                output / f"verified-{len(records):04d}.json",
                {
                    "rows": len(rows),
                    "artifact": artifact,
                    "history_verified": True,
                    "artifact_verified": True,
                    "rendering_verified": False,
                },
            )
            return
        except Exception:
            if time.monotonic() >= deadline:
                raise
            time.sleep(5)


def finish_verified(run, records, output, artifact, *, already_recovered):
    """Verify closed delivery; only recover a complete, never-resumed writer."""
    from examples.embodied.wandb_verified_recovery import recover_closed

    transaction = Path(run.dir).parent / f"run-{run.id}.wandb"
    run.finish()
    try:
        verify_run(run, records, output, artifact, media_required=True)
        return {"postfinish_recovery_used": False}
    except Exception:
        if already_recovered:
            raise
        incident = recover_closed(
            run.path, transaction, records, output / "postfinish-recovery"
        )
        verify_run(run, records, output, artifact, media_required=True)
        run_record = {
            "postfinish_recovery_used": incident["recovery_sync_used"],
            "normal_live_delivery_accepted": False,
        }
        write(output / "postfinish-delivery.json", run_record)
        return run_record


def main(args):
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    import torch
    import wandb

    from art_embodied.backends.action_token_worker import _build_runtime
    from art_embodied.runner import run_lerobot_evaluation
    from examples.embodied.libero.components import build_evaluation_scenarios
    from examples.embodied.libero_plus.train_pi0_fast_language_sft import (
        _infinite_batches,
        _make_dataloader,
        _prepare_batch,
    )
    from examples.embodied.pi0_fast_pipeline_audit import snapshot_comparison
    from examples.embodied.pi0_fast_teacher_learning_control import update
    from examples.embodied.wandb_verified_recovery import recover_closed

    if os.environ.get("SLURM_JOB_NUM_NODES") != "1":
        raise ValueError("Require exactly one allocated Slurm node")
    viewer = wandb.Api(timeout=30).viewer
    verify_wandb_identity(viewer)
    args.output.mkdir(parents=True, exist_ok=False)
    initial = args.workers / "snapshots/update-0000/initial"
    spec = configure(
        json.loads((args.workers / "worker-00/bootstrap.json").read_text()),
        args.output,
        evaluation_manifest=args.evaluation_manifest,
        rollout_gpus=args.rollout_gpus,
    )
    spec["policy_snapshot"] = str(initial.resolve())
    inventory = json.loads(args.inventory.read_text())
    plan = {
        "scope": __doc__,
        "arm": args.arm,
        "updates": args.updates,
        "evaluation_updates": [0, 25, 50, args.updates],
        "evaluation_episodes": 100,
        "microbatch": 4,
        "accumulation": 8,
        "initial_sha256": hashlib.sha256(
            (initial / "adapter_model.safetensors").read_bytes()
        ).hexdigest(),
        "inventory": inventory,
        "teacher_temperature": 1.0,
        "seed": 1701,
        "mode": "train",
        "sealed_accessed": False,
        "evaluation_manifest": str(args.evaluation_manifest)
        if args.evaluation_manifest
        else None,
        "reset_experiment": args.evaluation_manifest is not None,
        "sealed_plan": "No candidate promotion in this diagnostic. Reuse the existing untouched Spatial sealed manifest only after a GRPO candidate qualifies.",
    }
    write(args.output / "plan.json", plan)
    torch.manual_seed(1701)
    config, policy, _ = _build_runtime(spec)
    snapshot_comparison(policy, initial)
    root = Path(inventory["root"])
    meta = LeRobotDatasetMetadata(
        "lerobot/libero", root=root, revision=inventory["revision"]
    )
    data = LeRobotDataset(
        "lerobot/libero",
        root=root,
        revision=inventory["revision"],
        episodes=inventory["episodes"],
        return_uint8=True,
        video_backend="pyav",
        delta_timestamps=resolve_delta_timestamps(policy.config, meta),
    )
    check_inventory(inventory, data)
    scenarios = build_evaluation_scenarios(config)
    if len(scenarios) != 100 or {s.task for s in scenarios} != {inventory["task"]}:
        raise ValueError("Teacher language or evaluation population mismatch")
    write(
        args.output / "evaluation-scenarios.json",
        [s.model_dump(mode="json") for s in scenarios],
    )
    preset = policy.config.get_optimizer_preset()
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=preset.lr,
        betas=preset.betas,
        eps=preset.eps,
        weight_decay=preset.weight_decay,
    )
    scheduler = policy.config.get_scheduler_preset().build(optimizer, args.updates)
    plan["optimizer"] = {
        "lr": preset.lr,
        "betas": preset.betas,
        "eps": preset.eps,
        "weight_decay": preset.weight_decay,
        "clip_norm": preset.grad_clip_norm,
    }
    plan["scheduler"] = str(policy.config.get_scheduler_preset())
    write(args.output / "plan.json", plan)
    batches = _infinite_batches(
        _make_dataloader(data, batch_size=4, workers=4, seed=1701)
    )
    run = wandb.init(
        entity="wandb-japan",
        project="art-embodied-pi0-fast-spatial",
        name=f"pi0-fast-spatial-teacher-{os.environ['SLURM_JOB_ID']}-{args.arm}",
        group=f"pi0-fast-spatial-teacher-{os.environ['SLURM_JOB_ID']}",
        job_type="full-task-sft-path-diagnostic",
        config=plan,
        tags=["diagnostic-not-grpo", "spatial-task5", args.arm],
    )
    run.define_metric("experiment/update")
    for section in ("train", "validation", "optimization", "performance", "media"):
        run.define_metric(f"{section}/*", step_metric="experiment/update")
    write(args.output / "run.json", {"id": run.id, "url": run.url})
    records, batch_indices = [], []
    recovered = False
    baseline = None

    def checkpoint(step):
        path = args.output / f"checkpoint-{step:04d}"
        policy.save_checkpoint(path)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
            },
            path / "optimizer.pt",
        )
        return path

    def upload(path):
        artifact = wandb.Artifact(
            f"pi0-fast-spatial-teacher-{run.id}-{path.name}", type="model-diagnostic"
        )
        artifact.add_dir(str(path))
        artifact.add_file(str(args.output / "plan.json"))
        return run.log_artifact(artifact).wait().qualified_name

    def evaluate(step, path):
        nonlocal baseline
        tick = time.monotonic()
        policy.eval()
        policy.to("cpu")
        with diagnostic_cuda_mapping():
            result = asyncio.run(
                run_lerobot_evaluation(
                    config=config,
                    policy=policy,
                    evaluation_scenarios=scenarios,
                    step=step,
                    checkpoint_path=path,
                    measured_baseline_path=baseline,
                )
            ).evaluation
        policy.to("cuda:0")
        policy.train()
        if result.metrics["completed_episodes"] != 100:
            raise ValueError("Physical evaluation did not complete all 100 episodes")
        if baseline is None:
            baseline = result.artifacts["episode_outcomes_json"]
        write(
            args.output / f"evaluation-{step:04d}.json",
            {"metrics": result.metrics, "artifacts": result.artifacts},
        )
        videos = [
            Path(media.metadata["path"])
            for trajectory in result.trajectories
            for media in trajectory.media
            if media.kind == "video" and "path" in media.metadata
        ]
        metrics = {f"validation/{k}": v for k, v in result.metrics.items()}
        metrics["performance/evaluation_seconds"] = time.monotonic() - tick
        media = (
            {
                "media/evaluation": wandb.Video(
                    str(videos[0]), format=videos[0].suffix.lstrip(".")
                )
            }
            if videos
            else {}
        )
        if not media:
            raise ValueError("Evaluation produced no video")
        artifact = wandb.Artifact(
            f"pi0-fast-spatial-eval-{run.id}-u{step}", type="evaluation"
        )
        for name, source in result.artifacts.items():
            artifact.add_file(source, name=f"{name}.json")
        run.log_artifact(artifact).wait()
        return metrics, media

    try:
        for step in range(args.updates + 1):
            row, media = {"experiment/update": step}, {}
            if step:
                tick = time.monotonic()
                prepared, indices = [], []
                for _ in range(8):
                    batch = next(batches)
                    if set(batch["task"]) != {inventory["task"]}:
                        raise ValueError("Unexpected teacher instruction")
                    indices.extend(batch["index"].tolist())
                    prepared.append(
                        _prepare_batch(batch, policy, list(meta.camera_keys))
                    )
                lr = optimizer.param_groups[0]["lr"]
                loss, norm = update(
                    policy, prepared, args.arm, optimizer, preset.grad_clip_norm
                )
                scheduler.step()
                del prepared
                batch_indices.append(indices)
                write(args.output / "batch-indices.json", batch_indices)
                row.update(
                    {
                        "train/loss": loss,
                        "train/ce_loss": loss,
                        "optimization/grad_norm": norm,
                        "optimization/learning_rate": lr,
                        "performance/training_seconds": time.monotonic() - tick,
                    }
                )
            path = checkpoint(step) if step in (0, 1, 25, 50, args.updates) else None
            if step in plan["evaluation_updates"]:
                metrics, media = evaluate(step, path)
                row.update(metrics)
            records.append(row)
            write(args.output / "records.json", records)
            run.log(row | media)
            print(json.dumps(row), flush=True)
            if path is not None:
                artifact = upload(path)
            if step in (1, 25, 50, args.updates):
                try:
                    verify_run(run, records, args.output, artifact, media_required=True)
                except Exception:
                    # Recovery is bounded to the complete initial writer, not a partial resume.
                    if recovered:
                        raise
                    transaction = Path(run.dir).parent / f"run-{run.id}.wandb"
                    identity = run.path
                    run.finish()
                    recover_closed(
                        identity, transaction, records, args.output / "recovery"
                    )
                    run = wandb.init(
                        entity="wandb-japan",
                        project="art-embodied-pi0-fast-spatial",
                        id=identity.split("/")[-1],
                        resume="must",
                    )
                    if run.step != len(records):
                        raise ValueError("Recovered writer resume offset mismatch")
                    recovered = True
                    run.summary["diagnostics/recovery_sync_used"] = True
                    verify_run(run, records, args.output, artifact, media_required=True)
                if step == 1:
                    write(
                        args.output / "FIRST_UPDATE_VERIFIED.json",
                        {"run": run.path, "recovery_used": recovered},
                    )
        delivery = finish_verified(
            run, records, args.output, artifact, already_recovered=recovered
        )
        write(
            args.output / "complete.json",
            {
                "updates": args.updates,
                "recovery_used": recovered or delivery["postfinish_recovery_used"],
                **delivery,
                "root_cause_confirmed": False,
            },
        )
    except BaseException as exc:
        write(
            args.output / "failure.json",
            {
                "error": repr(exc),
                "last_logged_update": records[-1]["experiment/update"]
                if records
                else None,
            },
        )
        run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("workers", "inventory", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--arm", choices=("native", "sampler"), required=True)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument("--rollout-gpus", type=int, default=1)
    args = parser.parse_args()
    if args.updates != 100:
        parser.error("This preregistered control uses 100 SFT updates")
    main(args)
