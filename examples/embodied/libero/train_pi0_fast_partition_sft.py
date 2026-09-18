"""Train one task-owned pi0-FAST LoRA block with immutable evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import yaml

from art_embodied import EmbodiedExperimentConfig, make_policy
from art_embodied.checkpointing import CheckpointManager

DATASET_REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--task-index", type=int, choices=range(10), required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--entity", default="wandb-japan")
    parser.add_argument(
        "--project", default="art-embodied-pi0-fast-long-task-partition-sft"
    )
    parser.add_argument("--dataset-repo-id", default="lerobot/libero")
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    return parser.parse_args()


def discover_task_episodes(data_root: Path, task_index: int) -> tuple[int, ...]:
    """Return every episode carrying one task index from pinned parquet data."""

    import pandas as pd

    episodes: set[int] = set()
    for parquet_path in sorted(data_root.glob("data/**/*.parquet")):
        frame = pd.read_parquet(parquet_path, columns=["episode_index", "task_index"])
        selected = frame.loc[frame["task_index"] == task_index, "episode_index"]
        episodes.update(int(value) for value in selected.unique())
    if not episodes:
        raise ValueError(f"No dataset episodes found for task_index={task_index}")
    return tuple(sorted(episodes))


def task_language(data_root: Path, task_index: int) -> str:
    import pandas as pd

    tasks = pd.read_parquet(data_root / "meta" / "tasks.parquet")
    row = tasks.loc[tasks["task_index"] == task_index]
    if len(row) != 1:
        raise ValueError(f"Expected exactly one task row for task_index={task_index}")
    # LeRobot v3 stores the natural-language task as the parquet index and the
    # integer task_index as its only column.
    value = str(row.index[0])
    if not value:
        raise ValueError(f"Dataset task language is empty for task_index={task_index}")
    return value


def materialize_source_config(
    base_config: Path,
    destination: Path,
    *,
    task_index: int,
    rank: int,
    steps: int,
    project: str,
) -> EmbodiedExperimentConfig:
    """Freeze the exact ART policy surface used by one independent SFT job."""

    raw = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    raw["experiment"]["project"] = project
    raw["experiment"]["run"] = (
        f"pi0-fast-libero-long-task-{task_index:02d}-partition-sft-r{rank}-u{steps}"
    )
    raw["experiment"]["tags"] = [
        "pi0-fast",
        "libero-long",
        "partition-sft",
        f"task-{task_index:02d}",
        f"lora-r{rank}",
        "one-node",
        "candidate",
    ]
    raw["policy"]["lora"].update({"rank": rank, "alpha": rank, "rank_partition": None})
    raw["environment"]["kwargs"]["task_ids"] = [task_index]
    raw["training"]["updates"] = steps
    raw["runtime"]["training_devices"] = ["cuda"]
    raw["runtime"]["distributed_training"] = False
    raw["storage"]["resume_from_checkpoint"] = None
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return EmbodiedExperimentConfig.from_yaml(destination)


def _dataset_snapshot(repo_id: str, revision: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=["meta/**", "data/**"],
        )
    )


def _make_dataset(
    *, repo_id: str, revision: str, episodes: tuple[int, ...], policy: Any
) -> Any:
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import (
        LeRobotDataset,
        LeRobotDatasetMetadata,
    )

    metadata = LeRobotDatasetMetadata(repo_id, revision=revision)
    return LeRobotDataset(
        repo_id,
        episodes=list(episodes),
        revision=revision,
        delta_timestamps=resolve_delta_timestamps(policy.config, metadata),
        return_uint8=True,
        video_backend="pyav",
    )


def _make_dataloader(dataset: Any, *, batch_size: int, workers: int, seed: int) -> Any:
    from lerobot.datasets import EpisodeAwareSampler
    from lerobot.utils.collate import lerobot_collate_fn
    import torch

    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=dataset.episodes,
        drop_n_last_frames=0,
        shuffle=True,
        seed=seed,
        absolute_to_relative_idx=dataset.absolute_to_relative_idx,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=(lerobot_collate_fn if dataset.meta.has_language_columns else None),
        prefetch_factor=2 if workers else None,
        persistent_workers=bool(workers),
    )


def _infinite_batches(dataloader: Any):
    while True:
        yield from dataloader


def _prepare_batch(batch: dict[str, Any], policy: Any, camera_keys: list[str]) -> Any:
    import torch

    for camera_key in camera_keys:
        value = batch.get(camera_key)
        if value is not None and value.dtype == torch.uint8:
            batch[camera_key] = value.float().div_(255.0)
    return policy.preprocessor(batch)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _history_has_steps(
    *, entity: str, project: str, run_id: str, required_steps: set[int]
) -> bool:
    import wandb

    path = f"{entity}/{project}/{run_id}"
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            api_run = wandb.Api(timeout=30).run(path)
            observed = {
                int(row["experiment/update"])
                for row in api_run.scan_history(
                    keys=["experiment/update", "train/loss"]
                )
                if row.get("experiment/update") is not None
                and row.get("train/loss") is not None
            }
            if required_steps.issubset(observed):
                return True
        except Exception:
            pass
        time.sleep(10)
    return False


def train(args: argparse.Namespace) -> dict[str, Any]:
    from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
    import torch
    import wandb

    if args.rank < 1 or args.steps < 1:
        raise ValueError("rank and steps must be positive")
    if args.microbatch_size < 1 or args.gradient_accumulation < 1:
        raise ValueError("microbatch and accumulation must be positive")
    task_root = args.output_root / f"task-{args.task_index:02d}"
    if task_root.exists():
        raise FileExistsError(f"Refusing to overwrite SFT source: {task_root}")
    task_root.mkdir(parents=True)
    source_config_path = task_root / "source.yaml"
    config = materialize_source_config(
        args.base_config,
        source_config_path,
        task_index=args.task_index,
        rank=args.rank,
        steps=args.steps,
        project=args.project,
    )
    snapshot = _dataset_snapshot(args.dataset_repo_id, args.dataset_revision)
    episodes = discover_task_episodes(snapshot, args.task_index)
    task_key = task_language(snapshot, args.task_index)

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name=config.experiment.run,
        group="pi0-fast-libero-long-task-partition-sft-r128",
        job_type="supervised-warm-start",
        save_code=config.observability.wandb.save_code,
        settings=wandb.Settings(
            x_disable_stats=not config.observability.wandb.log_system_metrics
        ),
        config={
            "stage": "partition_sft",
            "task_index": args.task_index,
            "task_key": task_key,
            "rank": args.rank,
            "steps": args.steps,
            "microbatch_size": args.microbatch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_batch_size": (args.microbatch_size * args.gradient_accumulation),
            "dataset_repo_id": args.dataset_repo_id,
            "dataset_revision": args.dataset_revision,
            "episodes": list(episodes),
            "source_config_fingerprint": config.fingerprint,
        },
        tags=list(config.experiment.tags),
    )
    if run is None:
        raise RuntimeError("wandb.init returned no Run")
    update_metric = run.define_metric("experiment/update", hidden=True)
    for namespace in ("train/*", "optimization/*", "performance/*", "monitor/*"):
        run.define_metric(namespace, step_metric=update_metric)
    run.log(
        {
            "experiment/update": 0,
            "monitor/initialized": 1,
            "monitor/task_index": args.task_index,
        }
    )

    error: BaseException | None = None
    final_loss = math.nan
    checkpoint: Path | None = None
    first_readback = False
    try:
        policy = make_policy(config)
        dataset = _make_dataset(
            repo_id=args.dataset_repo_id,
            revision=args.dataset_revision,
            episodes=episodes,
            policy=policy,
        )
        dataloader = _make_dataloader(
            dataset,
            batch_size=args.microbatch_size,
            workers=args.num_workers,
            seed=config.experiment.seed + args.task_index,
        )
        batches = _infinite_batches(dataloader)
        trainable = [
            parameter for parameter in policy.parameters() if parameter.requires_grad
        ]
        if not trainable:
            raise RuntimeError("Partition SFT policy has no trainable parameters")
        native_optimizer = policy.config.get_optimizer_preset()
        optimizer = torch.optim.AdamW(
            trainable,
            lr=native_optimizer.lr,
            betas=native_optimizer.betas,
            eps=native_optimizer.eps,
            weight_decay=native_optimizer.weight_decay,
        )
        native_scheduler = policy.config.get_scheduler_preset()
        scheduler = CosineDecayWithWarmupSchedulerConfig(
            peak_lr=native_scheduler.peak_lr,
            decay_lr=native_scheduler.decay_lr,
            num_warmup_steps=native_scheduler.num_warmup_steps,
            num_decay_steps=native_scheduler.num_decay_steps,
        ).build(optimizer, args.steps)
        optimizer.zero_grad(set_to_none=True)
        policy.train()
        for update in range(1, args.steps + 1):
            started = time.perf_counter()
            accumulated_loss = 0.0
            accumulated_ce = 0.0
            for _ in range(args.gradient_accumulation):
                batch = _prepare_batch(
                    next(batches), policy, list(dataset.meta.camera_keys)
                )
                loss, details = policy.policy(batch)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Non-finite SFT loss at update {update}")
                (loss / args.gradient_accumulation).backward()
                accumulated_loss += float(loss.detach().cpu())
                accumulated_ce += float(details["ce_loss"])
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                native_optimizer.grad_clip_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            final_loss = accumulated_loss / args.gradient_accumulation
            run.log(
                {
                    "experiment/update": update,
                    "train/loss": final_loss,
                    "train/ce_loss": accumulated_ce / args.gradient_accumulation,
                    "optimization/learning_rate": optimizer.param_groups[0]["lr"],
                    "optimization/grad_norm": float(grad_norm.detach().cpu()),
                    "performance/update_seconds": time.perf_counter() - started,
                    "performance/effective_examples": (
                        args.microbatch_size * args.gradient_accumulation
                    ),
                }
            )
            if update == 1:
                first_readback = _history_has_steps(
                    entity=args.entity,
                    project=args.project,
                    run_id=run.id,
                    required_steps={1},
                )
                if not first_readback:
                    raise RuntimeError("W&B could not read back SFT update 1")

        checkpoint = CheckpointManager().publish(
            task_root / f"checkpoint-step-{args.steps:06d}",
            writer=lambda staging: policy.save_checkpoint(staging / "policy"),
            config_fingerprint=config.fingerprint,
            resume_contract_fingerprint=config.resume_contract_fingerprint,
            metadata={
                "backend": "pi0_fast_partition_sft",
                "step": args.steps,
                "task_index": args.task_index,
                "task_key": task_key,
                "dataset_revision": args.dataset_revision,
            },
        )
        run.summary.update(
            {
                "train/final_loss": final_loss,
                "monitor/checkpoint_published": 1,
                "monitor/first_history_readback_verified": int(first_readback),
            }
        )
    except BaseException as exc:
        error = exc
        raise
    finally:
        run.finish(exit_code=1 if error is not None else 0)

    assert checkpoint is not None
    final_readback = _history_has_steps(
        entity=args.entity,
        project=args.project,
        run_id=run.id,
        required_steps={1, args.steps},
    )
    if not final_readback:
        raise RuntimeError("W&B could not read back first and final SFT updates")
    marker = checkpoint / "art_embodied_checkpoint_complete.json"
    completion = {
        "schema_version": 1,
        "kind": "pi0_fast_partition_sft_completion",
        "status": "passed",
        "task_index": args.task_index,
        "task_key": task_key,
        "rank": args.rank,
        "final_step": args.steps,
        "final_loss": final_loss,
        "finite_loss_verified": math.isfinite(final_loss),
        "dataset": {
            "repo_id": args.dataset_repo_id,
            "revision": args.dataset_revision,
            "episodes": list(episodes),
        },
        "source_config": {
            "path": str(source_config_path.resolve()),
            "sha256": _sha256(source_config_path),
        },
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "marker_sha256": _sha256(marker),
        },
        "wandb": {
            "run_id": run.id,
            "run_url": run.url,
            "history_readback_verified": True,
        },
    }
    completion_path = task_root / "sft-completion.json"
    completion_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**completion, "sft_completion_sha256": _sha256(completion_path)}


def main() -> None:
    print(json.dumps(train(_parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
