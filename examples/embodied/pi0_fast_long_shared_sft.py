"""Synchronous shared-LoRA native SFT on all ten Long tasks, then publication.

Set ART_EMBODIED_SFT_DATASET_ROOT to the downloaded dataset snapshot and
ART_EMBODIED_EXPECTED_WANDB_EMAIL to the account authorized for logging.
"""

import argparse
import asyncio
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np

from art_embodied import EmbodiedExperimentConfig, make_policy
from art_embodied.backends.loss_scaling import (
    backward_policy_loss,
    unscale_policy_gradients,
)
from art_embodied.checkpointing import CheckpointManager
from examples.embodied.pi0_fast_long_restart import check_runtime
from examples.embodied.pi0_fast_spatial_teacher_control import (
    diagnostic_cuda_mapping,
    finish_verified,
    verify_run,
    verify_wandb_identity,
    write,
)


def sft_dataset_root():
    configured = os.environ.get("ART_EMBODIED_SFT_DATASET_ROOT")
    if not configured:
        raise ValueError("Set ART_EMBODIED_SFT_DATASET_ROOT to the LIBERO dataset snapshot")
    root = Path(configured).expanduser().resolve()
    if not (root / "meta/tasks.parquet").is_file():
        raise ValueError("LIBERO dataset snapshot must contain meta/tasks.parquet")
    return root


def normalized_language(text):
    return " ".join(str(text).lower().replace("_", " ").split())


def match_tasks(languages, dataset_tasks):
    matches = {}
    for task_id, language in languages.items():
        candidates = [
            int(index)
            for text, index in dataset_tasks
            if normalized_language(text) == normalized_language(language)
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Task {task_id}: require one exact language match, found {candidates}"
            )
        matches[int(task_id)] = candidates[0]
    if len(set(matches.values())) != len(matches):
        raise ValueError("Multiple simulator tasks map to one teacher task")
    return matches


class BalancedBatches:
    def __init__(self, rows, *, steps, rank, seed):
        self.rows, self.steps, self.rank, self.seed = rows, steps, rank, seed
        if len(rows) != 10 or any(len(v) == 0 for v in rows.values()):
            raise ValueError("Every Long task requires teacher frames")

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.rank)
        for _ in range(self.steps):
            selected = [int(rng.choice(self.rows[k])) for k in sorted(self.rows)]
            rng.shuffle(selected)
            for start in range(0, 10, 2):
                yield selected[start : start + 2]

    def __len__(self):
        return self.steps * 5


def main(root, steps):
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.utils.collate import lerobot_collate_fn
    import pandas as pd
    import torch
    import torch.distributed as dist
    import wandb

    from art_embodied.runner import run_lerobot_evaluation
    from examples.embodied.libero.components import build_evaluation_scenarios
    from examples.embodied.libero.environment import LiberoTaskCatalog
    from examples.embodied.libero.settings import LiberoSettings
    from examples.embodied.libero.train_pi0_fast_partition_sft import _prepare_batch
    from examples.embodied.pi0_fast_teacher_learning_control import (
        objective,
        token_counts,
    )

    data_root = sft_dataset_root()
    check_runtime()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if world != 8 or os.environ.get("SLURM_JOB_NUM_NODES") != "1":
        raise ValueError("Require one eight-GPU node")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(hours=2))
    seed = 20260909
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = EmbodiedExperimentConfig.from_yaml(root / "sft.yaml")
    config = config.model_copy(
        update={"policy": config.policy.model_copy(update={"device": f"cuda:{rank}"})}
    )
    out = root / "sft"
    out.mkdir(exist_ok=True)
    run = None
    if rank == 0:
        viewer = wandb.Api(timeout=30).viewer
        verify_wandb_identity(viewer)
        run = wandb.init(
            entity="wandb-japan",
            project="art-embodied-pi0-fast-long",
            name=root.name + "-sft",
            group=root.name,
            job_type="multitask-shared-sft",
            config=json.loads((root / "plan.json").read_text()),
        )
        run.define_metric("experiment/update")
        for section in ("train", "validation", "optimization", "performance", "media"):
            run.define_metric(f"{section}/*", step_metric="experiment/update")
        write(out / "run.json", {"id": run.id, "url": run.url})
    catalog = LiberoTaskCatalog(LiberoSettings.from_config(config))
    tasks = pd.read_parquet(data_root / "meta/tasks.parquet")
    mapping = match_tasks(
        {k: v.language for k, v in catalog.tasks.items()},
        [(text, row["task_index"]) for text, row in tasks.iterrows()],
    )
    episode_ids = set()
    for file in sorted(data_root.glob("data/**/*.parquet")):
        frame = pd.read_parquet(file, columns=["task_index", "episode_index"])
        episode_ids.update(
            map(
                int,
                frame.loc[
                    frame.task_index.isin(mapping.values()), "episode_index"
                ].unique(),
            )
        )
    policy = make_policy(config)
    params = [p for p in policy.parameters() if p.requires_grad]
    for p in params:
        dist.broadcast(p.data, src=0)
    meta = LeRobotDatasetMetadata(
        "lerobot/libero", root=data_root, revision=data_root.name
    )
    dataset = LeRobotDataset(
        "lerobot/libero",
        root=data_root,
        revision=data_root.name,
        episodes=sorted(episode_ids),
        delta_timestamps=resolve_delta_timestamps(policy.config, meta),
        return_uint8=True,
        video_backend="pyav",
    )
    task_rows = np.asarray(dataset.hf_dataset["task_index"]).reshape(-1)
    rows = {task: np.flatnonzero(task_rows == index) for task, index in mapping.items()}
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=BalancedBatches(rows, steps=steps, rank=rank, seed=seed),
        num_workers=2,
        multiprocessing_context="spawn",
        persistent_workers=True,
        pin_memory=True,
        collate_fn=lerobot_collate_fn if meta.has_language_columns else None,
    )
    batches = iter(loader)
    preset = policy.config.get_optimizer_preset()
    optimizer = torch.optim.AdamW(
        params,
        lr=preset.lr,
        betas=preset.betas,
        eps=preset.eps,
        weight_decay=preset.weight_decay,
    )
    scheduler = policy.config.get_scheduler_preset().build(optimizer, steps)
    records = []
    artifact = None
    if rank == 0:
        plan = json.loads((root / "plan.json").read_text()) | {
            "task_mapping": mapping,
            "frames_per_task": {k: len(v) for k, v in rows.items()},
            "optimizer": str(preset),
            "scheduler": str(policy.config.get_scheduler_preset()),
            "steps": steps,
            "dataset_revision": data_root.name,
        }
        write(out / "plan.json", plan)
        run.config.update(plan)

    def save(step):
        path = out / f"checkpoint-{step:04d}"
        policy.save_checkpoint(path)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
            },
            path / "optimizer.pt",
        )
        a = wandb.Artifact(
            f"pi0-fast-long-sft-{run.id}-u{step}", type="model-diagnostic"
        )
        a.add_dir(str(path))
        a.add_file(str(out / "plan.json"))
        for evidence in ("production-conformance.json", "source.sha256"):
            if (root / evidence).exists():
                a.add_file(str(root / evidence), name=evidence)
        return path, run.log_artifact(a).wait().qualified_name

    try:
        for step in range(steps + 1):
            row = {"experiment/update": step}
            media = {}
            if step:
                tick = time.monotonic()
                policy.train()
                prepared = []
                seen = []
                for _ in range(5):
                    batch = next(batches)
                    seen.extend(map(int, batch["task_index"].reshape(-1).tolist()))
                    prepared.append(
                        _prepare_batch(batch, policy, list(meta.camera_keys))
                    )
                if sorted(seen) != sorted(mapping.values()):
                    raise ValueError("SFT rank batch is not task-balanced")
                counts = token_counts(policy, prepared)
                total = torch.tensor(
                    sum(counts), device=f"cuda:{rank}", dtype=torch.float64
                )
                dist.all_reduce(total)
                optimizer.zero_grad(set_to_none=True)
                loss_sum = torch.zeros((), device=f"cuda:{rank}", dtype=torch.float64)
                for batch, count in zip(prepared, counts, strict=True):
                    loss = objective(policy, batch, "native") * (count / total.item())
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite SFT loss")
                    backward_policy_loss(policy, loss)
                    loss_sum += loss.detach().double()
                del prepared
                row.update(unscale_policy_gradients(policy))
                # Each rank already divided by the global token count: SUM, not mean.
                for p in params:
                    if p.grad is None or not torch.isfinite(p.grad).all():
                        raise ValueError("Missing/nonfinite SFT gradient")
                    dist.all_reduce(p.grad)
                dist.all_reduce(loss_sum)
                norm = torch.nn.utils.clip_grad_norm_(
                    params, preset.grad_clip_norm, error_if_nonfinite=True
                )
                lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                row.update(
                    {
                        "train/loss": float(loss_sum),
                        "train/ce_loss": float(loss_sum),
                        "optimization/grad_norm": float(norm),
                        "optimization/learning_rate": lr,
                        "performance/training_seconds": time.monotonic() - tick,
                        "train/frames_per_task": 8,
                    }
                )
            path = None
            if rank == 0 and step in (0, 1, steps):
                path, artifact = save(step)
            if step in (0, steps):
                policy.to("cpu")
                torch.cuda.empty_cache()
                dist.barrier()
                if rank == 0:
                    tick = time.monotonic()
                    with diagnostic_cuda_mapping():
                        result = asyncio.run(
                            run_lerobot_evaluation(
                                config=config,
                                policy=policy,
                                evaluation_scenarios=build_evaluation_scenarios(config),
                                step=step,
                                checkpoint_path=path,
                            )
                        ).evaluation
                    if result.metrics["completed_episodes"] != 100:
                        raise ValueError("Incomplete SFT dev evaluation")
                    write(
                        out / f"evaluation-{step:04d}.json",
                        {"metrics": result.metrics, "artifacts": result.artifacts},
                    )
                    row.update(
                        {f"validation/{k}": v for k, v in result.metrics.items()}
                    )
                    row["performance/evaluation_seconds"] = time.monotonic() - tick
                    videos = [
                        Path(m.metadata["path"])
                        for t in result.trajectories
                        for m in t.media
                        if m.kind == "video" and "path" in m.metadata
                    ]
                    if not videos:
                        raise ValueError("Missing evaluation video")
                    media["media/evaluation"] = wandb.Video(
                        str(videos[0]), format="mp4"
                    )
                    a = wandb.Artifact(
                        f"pi0-fast-long-sft-eval-{run.id}-u{step}", type="evaluation"
                    )
                    for name, file in result.artifacts.items():
                        a.add_file(file, name=name + ".json")
                    run.log_artifact(a).wait()
                dist.barrier()
                policy.to(f"cuda:{rank}")
            if rank == 0:
                records.append(row)
                write(out / "records.json", records)
                run.log(row | media)
                print(json.dumps(row), flush=True)
                if step in (1, steps):
                    verify_run(run, records, out, artifact, media_required=True)
                    if step == 1:
                        write(out / "FIRST_UPDATE_VERIFIED.json", {"run": run.path})
            dist.barrier()
        if rank == 0:
            delivery = finish_verified(
                run, records, out, artifact, already_recovered=False
            )
            grpo_config = EmbodiedExperimentConfig.from_yaml(root / "grpo.yaml")
            CheckpointManager().publish(
                root / "sft-warm-start",
                writer=lambda staging: shutil.copytree(
                    out / f"checkpoint-{steps:04d}", staging / "policy"
                ),
                config_fingerprint=grpo_config.fingerprint,
                resume_contract_fingerprint=grpo_config.resume_contract_fingerprint,
                metadata={
                    "backend": "pi0_fast_language_sft",
                    "step": steps,
                    "task_mapping": mapping,
                    "source": "shared native multitask SFT, no routing",
                },
            )
            write(
                out / "complete.json",
                {"updates": steps, "tasks": 10, "delivery": delivery},
            )
        dist.barrier()
    except BaseException as exc:
        if rank == 0:
            write(out / "failure.json", {"error": repr(exc)})
            if run is not None:
                run.finish(exit_code=1)
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()
    main(args.root.resolve(), args.steps)
