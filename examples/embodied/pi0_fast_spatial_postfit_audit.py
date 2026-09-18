"""Read-only fixed-teacher likelihood and action audit after Spatial SFT.

Unseen frames are from the same teacher episodes, not an independent episode
split or a sealed test. No optimizer, simulator, or production policy edits.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

from examples.embodied.pi0_fast_spatial_teacher_control import (
    verify_wandb_identity,
    write,
)


def select_panel(all_indices, used, count=16):
    universe = set(map(int, all_indices))
    if len(universe) != len(all_indices) or not used <= universe:
        raise ValueError("Invalid teacher index population")
    rng = random.Random(20260909)
    result = {}
    for group, choices in (
        ("seen", universe & used),
        ("unseen_frame", universe - used),
    ):
        if len(choices) < count:
            raise ValueError(f"Insufficient {group} frames")
        result[group] = sorted(rng.sample(sorted(choices), count))
    return result


def aggregate(rows):
    if not rows:
        raise ValueError("Empty fixed panel")
    tokens = sum(row["target_tokens"] for row in rows)
    if tokens <= 0:
        raise ValueError("Missing teacher tokens")
    result = {
        f"{mode}_ce": sum(r[f"{mode}_ce"] * r["target_tokens"] for r in rows) / tokens
        for mode in ("native", "sampler")
    }
    result["teacher_roundtrip_mae"] = sum(
        r["teacher_roundtrip_mae"] for r in rows
    ) / len(rows)
    valid = [r for r in rows if r["prediction_error"] is None]
    result["prediction_decode_failures"] = len(rows) - len(valid)
    if any("token_detail" in row for row in rows):
        from examples.embodied.pi0_fast_teacher_token_audit import aggregate_details

        result.update(aggregate_details(rows))
    if valid:
        result["greedy_action_mae_valid_only"] = sum(
            r["greedy_action_mae"] for r in valid
        ) / len(valid)
    return result


def teacher_decode_tokens(native, batch):
    import torch

    from examples.embodied.pi0_fast_teacher_path_control import teacher_rows

    rows = teacher_rows(native, batch)
    if len(rows) != 1:
        raise ValueError("Postfit audit decodes one teacher frame at a time")
    # Generation starts after conditioning BOS. The deployment decoder expects
    # that generated suffix, not the padded SFT input (BOS + target tokens).
    device = next(
        value.device for value in batch.values() if isinstance(value, torch.Tensor)
    )
    return torch.tensor(rows, dtype=torch.long, device=device)


def verify(run, records, result_path, artifact, output):
    import wandb

    from examples.embodied.wandb_verified_recovery import assert_history

    deadline = time.monotonic() + 120
    while True:
        try:
            remote = wandb.Api(timeout=20).run(run.path)
            rows = list(remote.scan_history(page_size=1000, max_step=1000))
            assert_history(rows, records)
            if "media/teacher_observation" not in rows[0]:
                raise ValueError("Teacher image absent")
            image = rows[0]["media/teacher_observation"]
            file = remote.file(image["path"]).download(
                root=str(output / "remote-media"), replace=True
            )
            path = Path(file.name)
            file.close()
            if hashlib.sha256(path.read_bytes()).hexdigest() != image["sha256"]:
                raise ValueError("Teacher image mismatch")
            saved = wandb.Api(timeout=20).artifact(artifact)
            path = saved.get_entry(result_path.name).download(
                root=str(output / "remote")
            )
            if Path(path).read_bytes() != result_path.read_bytes():
                raise ValueError("Result artifact mismatch")
            write(
                output / f"verified-{len(records)}.json",
                {
                    "rows": len(rows),
                    "history_verified": True,
                    "media_verified": True,
                    "artifact_verified": True,
                    "rendering_verified": False,
                },
            )
            return
        except Exception:
            if time.monotonic() >= deadline:
                raise
            time.sleep(5)


def close_with_verification(run, records, result_path, artifact_name, output):
    import wandb

    from examples.embodied.wandb_verified_recovery import recover_closed

    run.finish()
    recovered, retried = False, False
    try:
        verify(run, records, result_path, artifact_name, output)
    except Exception:
        # Only a complete, closed, never-resumed diagnostic transaction. The
        # recovery helper rejects live writers, partial journals and stale data.
        retried = True
        transaction = Path(run.dir).parent / f"run-{run.id}.wandb"
        repair = recover_closed(run.path, transaction, records, output / "recovery")
        recovered = repair["recovery_sync_used"]
        verify(run, records, result_path, artifact_name, output / "recovery-final")
        remote = wandb.Api(timeout=20).run(run.path)
        remote.summary["diagnostics/recovery_sync_used"] = recovered
        remote.summary["diagnostics/normal_live_delivery_accepted"] = False
        remote.summary.update()
    return {
        "recovery_sync_used": recovered,
        "post_finish_retry_required": retried,
        "normal_live_delivery_accepted": not retried,
    }


def main(args):
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    import torch
    import wandb

    from art_embodied.backends.action_token_worker import _build_runtime
    from art_embodied.policies.pi0_fast_sft_anchor import _collate_one
    from examples.embodied.libero_plus.train_pi0_fast_language_sft import _prepare_batch
    from examples.embodied.pi0_fast_pipeline_audit import snapshot_comparison
    from examples.embodied.pi0_fast_spatial_teacher_control import check_inventory
    from examples.embodied.pi0_fast_spatial_teacher_summary import summarize
    from examples.embodied.pi0_fast_teacher_learning_control import objective
    from examples.embodied.pi0_fast_teacher_path_control import teacher_rows

    if os.environ.get("SLURM_JOB_NUM_NODES") != "1":
        raise ValueError("Require one Slurm node")
    summary = summarize(args.source)
    if not all(a["complete"] and not a["failure"] for a in summary["arms"].values()):
        raise ValueError("Both source SFT experiments must have completed")
    viewer = wandb.Api(timeout=30).viewer
    verify_wandb_identity(viewer)
    args.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((args.source / "native/plan.json").read_text())
    inventory = plan["inventory"]
    spec = json.loads((args.workers / "worker-00/bootstrap.json").read_text())
    if (
        spec["config"]["policy"]["type"] != "pi0_fast"
        or spec["config"]["policy"]["lora"]["rank_partition"] is not None
    ):
        raise ValueError("FAST shared LoRA only")
    initial = args.source / "native/checkpoint-0000"
    spec["policy_snapshot"] = str(initial.resolve())
    torch.manual_seed(1701)
    _, policy, _ = _build_runtime(spec)
    policy.eval()
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
    indices = list(map(int, data.hf_dataset["index"]))
    used = {
        i
        for batch in json.loads((args.source / "native/batch-indices.json").read_text())
        for i in batch
    }
    panel = select_panel(indices, used)
    positions = {idx: i for i, idx in enumerate(indices)}
    metadata = {
        "scope": __doc__,
        "panel": panel,
        "source": str(args.source.resolve()),
        "source_comparison": summary,
        "selection_seed": 20260909,
        "optimization_updates": 0,
        "sealed_accessed": False,
        "greedy_temperature": 0,
        "teacher_temperature": 1,
        "scoring_mode": "eval with autograd, immediately detached",
        "token_details": getattr(args, "token_details", False),
    }
    write(args.output / "plan.json", metadata)
    run = wandb.init(
        entity="wandb-japan",
        project="art-embodied-pi0-fast-spatial",
        name=f"pi0-fast-spatial-postfit-{os.environ['SLURM_JOB_ID']}",
        job_type="fixed-teacher-checkpoint-audit",
        config=metadata,
        tags=["diagnostic-not-grpo", "read-only", "spatial-task5"],
    )
    run.define_metric("diagnostics/stage")
    for namespace in ("validation/*", "performance/*", "monitor/*", "media/*"):
        run.define_metric(namespace, step_metric="diagnostics/stage")
    write(args.output / "run.json", {"id": run.id, "url": run.url})
    records = []
    cases = [
        ("initial", initial, 0),
        ("native", args.source / "native/checkpoint-0100", 100),
        ("sampler", args.source / "sampler/checkpoint-0100", 100),
    ]
    try:
        for stage, (name, path, checkpoint_update) in enumerate(cases):
            tick = time.monotonic()
            policy.load_checkpoint(path)
            policy.eval()
            before = snapshot_comparison(policy, path)
            results = {}
            media = None
            for group, selected in panel.items():
                rows = []
                for idx in selected:
                    sample = data[positions[idx]]
                    if sample["task"] != inventory["task"]:
                        raise ValueError("Wrong teacher instruction")
                    target = sample["action"].detach().cpu().clone()
                    if media is None:
                        media = wandb.Image(
                            sample[meta.camera_keys[0]].permute(1, 2, 0).numpy(),
                            caption=f"Fixed teacher frame {idx}",
                        )
                    batch = _prepare_batch(
                        _collate_one(sample, data), policy, list(meta.camera_keys)
                    )
                    row = {
                        "index": idx,
                        "episode_index": int(sample["episode_index"]),
                        "target_tokens": len(teacher_rows(policy.policy, batch)[0]),
                    }
                    for mode in ("native", "sampler"):
                        row[f"{mode}_ce"] = float(
                            objective(policy, batch, mode).detach()
                        )
                    if getattr(args, "token_details", False):
                        import math

                        from examples.embodied.pi0_fast_teacher_token_audit import (
                            measure,
                        )

                        row["token_detail"] = measure(policy, batch)
                        for mode in ("native", "sampler"):
                            losses = row["token_detail"][f"{mode}_nll"]
                            if not math.isclose(
                                sum(losses) / len(losses),
                                row[f"{mode}_ce"],
                                abs_tol=2e-5,
                                rel_tol=2e-5,
                            ):
                                raise ValueError(
                                    "Token diagnostic changed forward result"
                                )
                    with torch.no_grad():
                        chunks, errors = policy.decode_action_tokens_native(
                            teacher_decode_tokens(policy.policy, batch)
                        )
                        if errors[0] is not None:
                            write(
                                args.output / "teacher-decode-failure.json",
                                row | {"error": errors[0]},
                            )
                            raise ValueError(
                                f"Teacher tokens cannot be decoded: {errors[0]}"
                            )
                        row["teacher_roundtrip_mae"] = float(
                            (chunks[0].cpu() - target).abs().mean()
                        )
                        generated = policy.sample_action_tokens(batch, temperature=0)
                        chunks, errors = policy.decode_action_tokens_native(generated)
                        row["prediction_error"] = errors[0]
                        if getattr(args, "token_details", False):
                            row["generated_tokens"] = generated.detach().cpu().tolist()
                            row["target_actions"] = target.tolist()
                        if errors[0] is None:
                            row["greedy_action_mae"] = float(
                                (chunks[0].cpu() - target).abs().mean()
                            )
                            if getattr(args, "token_details", False):
                                row["predicted_actions"] = chunks[0].cpu().tolist()
                                row["action_mae_per_dimension"] = (
                                    (chunks[0].cpu() - target)
                                    .abs()
                                    .mean(dim=0)
                                    .tolist()
                                )
                    json.dumps(row, allow_nan=False)
                    rows.append(row)
                    print(
                        json.dumps(
                            {"case": name, "group": group, "completed": len(rows)}
                        ),
                        flush=True,
                    )
                results[group] = {"rows": rows, "metrics": aggregate(rows)}
            after = snapshot_comparison(policy, path)
            if before != after or any(p.grad is not None for p in policy.parameters()):
                raise ValueError("Read-only audit changed model state")
            result_path = args.output / f"result-{name}.json"
            write(
                result_path,
                {
                    "case": name,
                    "checkpoint_update": checkpoint_update,
                    "snapshot": after,
                    "groups": results,
                    "root_cause_confirmed": False,
                },
            )
            row = {
                "diagnostics/stage": stage,
                "monitor/checkpoint_update": checkpoint_update,
                "performance/audit_seconds": time.monotonic() - tick,
            }
            for group, result in results.items():
                row.update(
                    {f"validation/{group}_{k}": v for k, v in result["metrics"].items()}
                )
            records.append(row)
            write(args.output / "records.json", records)
            run.log(row | {"media/teacher_observation": media})
            artifact = wandb.Artifact(
                f"pi0-fast-postfit-{run.id}-{name}", type="diagnostic"
            )
            artifact.add_file(str(result_path))
            artifact.add_file(str(args.output / "plan.json"))
            artifact.add_file(str(path / "adapter_model.safetensors"))
            artifact_name = run.log_artifact(artifact).wait().qualified_name
            verify(run, records, result_path, artifact_name, args.output)
        delivery = close_with_verification(
            run, records, result_path, artifact_name, args.output
        )
        write(
            args.output / "complete.json",
            {
                "optimization_updates": 0,
                "root_cause_confirmed": False,
                **delivery,
            },
        )
    except BaseException as exc:
        write(args.output / "failure.json", {"error": repr(exc)})
        run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "workers", "output"):
        p.add_argument(f"--{key}", type=Path, required=True)
    p.add_argument("--token-details", action="store_true")
    main(p.parse_args())
