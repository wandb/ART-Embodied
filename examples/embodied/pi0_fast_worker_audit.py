"""Replay a trusted retained GRPO gradient job without an optimizer update."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import time

from art_embodied.backends.action_token_gradients import (
    load_action_token_gradient_payload,
)
from art_embodied.backends.action_token_worker import _build_runtime, _run_gradient_job


def make_audit_spec(worker: Path, output: Path) -> tuple[dict, dict]:
    spec = json.loads((worker / "bootstrap.json").read_text())
    job = worker / "update-0000-job-0000"
    reference = json.loads((job / "result.json").read_text())
    config = spec["config"]
    if config["policy"]["type"] != "pi0_fast":
        raise ValueError("This audit requires pi0-FAST")
    if config["policy"]["load_kwargs"].get("model_compute_dtype") != "float32":
        raise ValueError("This audit requires retained FP32 evidence")
    if config["training"]["schedule"]["type"] != "full_update":
        raise ValueError("This audit requires the retained full-update schedule")
    if config["algorithm"]["loss_aggregation"] != "trajectory_mean":
        raise ValueError("This audit reconstructs trajectory_mean denominators only")
    if config["policy"]["load_kwargs"].get("sft_anchor") is not None:
        raise ValueError("This audit does not reconstruct auxiliary SFT gradients")
    config["storage"]["output_dir"] = str(output)
    config["policy"]["load_kwargs"]["training_logprob_mode"] = "full_sequence"
    metrics = reference["metrics"]
    prefix = "embodied_action_token_grpo/"
    spec.update(
        examples_path=str(job / "examples.pkl"),
        gradient_path=str(output / "gradients.pt"),
        global_example_count=int(metrics[prefix + "global_loss_denominator_examples"]),
        global_token_count=int(metrics[prefix + "global_loss_denominator_tokens"]),
        global_positive_token_count=None,
        global_negative_token_count=None,
        enforce_pre_update_alignment=True,
    )
    return spec, reference


def compare_gradients(reference: dict, candidate: dict) -> dict:
    import torch

    old, new = reference["gradients"], candidate["gradients"]
    if set(old) != set(new):
        raise ValueError("Gradient keys differ")
    norm2 = delta2 = dot = new_norm2 = 0.0
    for name, value in old.items():
        left, right = value.double(), new[name].double()
        if left.shape != right.shape:
            raise ValueError(f"Gradient shape differs: {name}")
        if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
            raise ValueError(f"Nonfinite gradient: {name}")
        norm2 += float(left.square().sum())
        new_norm2 += float(right.square().sum())
        delta2 += float((left - right).square().sum())
        dot += float((left * right).sum())
    if norm2 == 0 or new_norm2 == 0:
        raise ValueError("Zero gradient is not a meaningful conformance reference")
    return {
        "gradient_relative_l2": (delta2 / norm2) ** 0.5,
        "gradient_cosine": dot / (norm2 * new_norm2) ** 0.5,
        "gradient_tensors": len(old),
        "all_finite": True,
        "missing_gradient_names_match": sorted(reference["missing_gradients"])
        == sorted(candidate["missing_gradients"]),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    spec, reference = make_audit_spec(args.worker, args.output)

    import torch
    import wandb

    run = wandb.init(
        entity="wandb-japan",
        project="art-embodied-pi0-fast-spatial",
        name="pi0-fast-full-shard-grpo-scorer-audit",
        job_type="performance-audit",
        config={
            "worker": str(args.worker),
            "policy_snapshot": spec["policy_snapshot"],
            "training_logprob_mode": "full_sequence",
            "optimizer_steps": 0,
            "source_config": spec["config"],
        },
    )
    run.log({"audit/started": 1, "audit/optimizer_steps": 0})
    backend = None
    try:
        start = time.perf_counter()
        config, policy, backend = _build_runtime(spec)
        startup_seconds = time.perf_counter() - start
        # Capture all trainable weights to verify that gradient-only really is read-only.
        before = {
            name: p.detach().cpu().clone()
            for name, p in policy.named_parameters()
            if p.requires_grad
        }
        result, _ = await _run_gradient_job(
            spec,
            policy=policy,
            backend=backend,
            config=config,
            current_snapshot=spec["policy_snapshot"],
        )
        for name, p in policy.named_parameters():
            if name in before and not torch.equal(before[name], p.detach().cpu()):
                raise RuntimeError(f"Gradient audit changed weights: {name}")
        conformance = compare_gradients(
            load_action_token_gradient_payload(
                args.worker / "update-0000-job-0000/gradients.pt"
            ),
            load_action_token_gradient_payload(spec["gradient_path"]),
        )
        metrics = result["metrics"]
        prefix = "embodied_action_token_grpo/"
        conformance["loss_abs_difference"] = abs(
            metrics[prefix + "loss"] - reference["metrics"][prefix + "loss"]
        )
        result.update(
            wandb_url=run.url,
            conformance=conformance,
            startup_seconds=startup_seconds,
            reference_elapsed_seconds=reference["elapsed_seconds"],
            trainable_weights_unchanged=True,
            optimizer_steps=0,
        )
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        run.log(
            {f"conformance/{k}": v for k, v in conformance.items()}
            | {
                f"performance/{k.removeprefix(prefix)}": v
                for k, v in metrics.items()
                if "seconds" in k
            }
            | {
                "performance/startup_seconds": startup_seconds,
                "performance/worker_seconds": result["elapsed_seconds"],
                "audit/optimizer_steps": 0,
                "audit/weights_unchanged": True,
            }
        )
        artifact = wandb.Artifact("pi0-fast-full-shard-scorer-audit", type="benchmark")
        artifact.add_file(str(args.output / "result.json"))
        run.log_artifact(artifact)
        run.summary["audit/completed"] = True
    except Exception:
        run.finish(exit_code=1)
        raise
    else:
        run.finish()
    finally:
        if backend is not None:
            await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
