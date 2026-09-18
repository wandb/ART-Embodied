"""CPU-only counterfactual fourth AdamW step with shared optimizer history.

This is not four steps of chunk-ratio training. Earlier gradients and weights
are fixed to the original run. Directional gains are not success-rate gains.
"""

import argparse
import json
import math
from pathlib import Path
import time

from examples.embodied.pi0_fast_optimizer_calibration_audit import adam_delta


def comparison(a, b):
    import torch

    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Nonfinite vector")
    if a.norm() == 0 or b.norm() == 0:
        raise ValueError("Zero vector cannot define direction")
    return {
        "cosine": float(a @ b / (a.norm() * b.norm())),
        "relative_l2": float((a - b).norm() / a.norm()),
        "first_norm": float(a.norm()),
        "second_norm": float(b.norm()),
    }


def counterfactual(weight, first, second, gradients, step):
    deltas = {}
    for name, raw in gradients.items():
        grad = raw * min(1.0, 1.0 / (float(raw.norm()) + 1e-6))
        deltas[name], _, _, _ = adam_delta(
            weight, grad, first, second, step, lr=2e-6, eps=1e-5
        )
    return deltas


def aggregate(paths, names):
    import torch

    total = None
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        values = payload["gradients"]
        if set(values) != set(names) or payload.get("missing_gradients"):
            raise ValueError(f"Incomplete gradient surface: {path}")
        flat = torch.cat([values[n].double().flatten() for n in names])
        if not torch.isfinite(flat).all():
            raise ValueError(f"Nonfinite gradient: {path}")
        total = flat if total is None else total + flat
    if total is None:
        raise ValueError("No worker gradients")
    return total


def audit(args, record):
    from safetensors.torch import load_file
    import torch

    def original(job):
        return [
            args.workers / f"worker-{i:02d}/update-0000-job-{job:04d}/gradients.pt"
            for i in range(8)
        ]

    names = sorted(torch.load(original(0)[0], weights_only=False)["gradients"])
    keys = [
        n.removeprefix("model.").replace(".default.weight", ".weight") for n in names
    ]
    initial = load_file(
        args.workers / "snapshots/update-0000/initial/adapter_model.safetensors"
    )
    if set(keys) != set(initial):
        raise ValueError("Checkpoint and gradient surfaces differ")

    def flatten(path):
        state = load_file(path / "adapter_model.safetensors")
        if set(state) != set(keys):
            raise ValueError("Snapshot parameter surface differs")
        return torch.cat([state[k].double().flatten() for k in keys])

    weight = torch.cat([initial[k].double().flatten() for k in keys])
    first, second = torch.zeros_like(weight), torch.zeros_like(weight)
    reconstruction = []
    for job in range(3):
        raw = aggregate(original(job), names)
        grad = raw * min(1.0, 1.0 / (float(raw.norm()) + 1e-6))
        delta, first, second, _ = adam_delta(
            weight, grad, first, second, job + 1, lr=2e-6, eps=1e-5
        )
        actual = flatten(args.workers / f"snapshots/job-{job + 1:04d}")
        error = float((weight + delta - actual).norm() / delta.norm())
        if not math.isfinite(error) or error > 0.01:
            raise ValueError(f"Optimizer reconstruction failed: {error}")
        reconstruction.append(error)
        record(
            {"diagnostics/stage": job + 1, "diagnostics/reconstruction_error": error}
        )
        weight = actual

    gradients = {"stored": aggregate(original(3), names)}
    for name in ("token", "action_chunk"):
        gradients[name] = aggregate(
            [args.gradients / f"{name}-{i}.pt" for i in range(8)], names
        )
    reproduced = comparison(gradients["stored"], gradients["token"])
    if reproduced["relative_l2"] > 0.005:
        raise ValueError("Original gradient was not reproduced")
    deltas = counterfactual(weight, first, second, gradients, 4)
    actual = flatten(args.checkpoint) - weight
    update_control = comparison(actual, deltas["stored"])
    if update_control["relative_l2"] > 0.01:
        raise ValueError("Fourth optimizer update was not reproduced")
    result = {
        "scope": __doc__,
        "reconstruction_relative_errors": reconstruction,
        "stored_vs_token_gradient": reproduced,
        "actual_vs_reconstructed_update": update_control,
        "token_vs_chunk_gradient": comparison(
            gradients["token"], gradients["action_chunk"]
        ),
        "token_vs_chunk_update": comparison(deltas["token"], deltas["action_chunk"]),
        "first_order_surrogate_gains": {
            objective: {
                update: float(-grad @ delta) for update, delta in deltas.items()
            }
            for objective, grad in gradients.items()
        },
        "sources": {k: str(v.resolve()) for k, v in vars(args).items()},
        "root_cause_confirmed": False,
        "new_rollouts": 0,
        "model_forward_passes": 0,
        "training_updates": 0,
    }
    record(
        {
            "diagnostics/stage": 4,
            **{
                f"diagnostics/{name}_{key}": value
                for name in ("token_vs_chunk_gradient", "token_vs_chunk_update")
                for key, value in result[name].items()
            },
        }
    )
    return result


def verify_rows(actual, expected):
    if len(actual) != len(expected):
        raise ValueError(f"Expected {len(expected)} history rows, found {len(actual)}")
    for row, reference in zip(actual, expected, strict=True):
        for key, value in reference.items():
            observed = row.get(key)
            if not isinstance(observed, int | float) or not math.isclose(
                observed, value, rel_tol=1e-10, abs_tol=1e-12
            ):
                raise ValueError(f"History mismatch: {key}: {observed} != {value}")


def main():
    import wandb

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workers", "gradients", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    run = wandb.init(
        entity="wandb-japan",
        project="art-embodied-pi0-fast-spatial",
        name="pi0-fast-chunk-optimizer-counterfactual",
        job_type="objective-diagnostic",
        config={"scope": __doc__, "gpu_count": 0},
    )
    run.define_metric("diagnostics/stage")
    run.define_metric("diagnostics/*", step_metric="diagnostics/stage")
    (args.output / "run.json").write_text(
        json.dumps({"id": run.id, "url": run.url, "dir": run.dir})
    )
    records = []

    def record(row):
        records.append(row)
        (args.output / "history.json").write_text(json.dumps(records, indent=2))
        run.log(row)

    try:
        record({"diagnostics/stage": 0})
        result = audit(args, record)
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        artifact = wandb.Artifact(
            f"pi0-fast-chunk-optimizer-{run.id}", type="diagnostic"
        )
        artifact.add_dir(str(args.output))
        artifact.add_file(__file__, name="audit.py")
        run.log_artifact(artifact).wait()
    except BaseException:
        run.finish(exit_code=1)
        raise
    run.finish()
    deadline = time.monotonic() + 120
    while True:
        try:
            remote = wandb.Api(timeout=30).run(
                f"wandb-japan/art-embodied-pi0-fast-spatial/{run.id}"
            )
            if remote.state != "finished":
                raise ValueError(f"Unexpected run state: {remote.state}")
            verify_rows(list(remote.scan_history()), records)
            saved = next(a for a in remote.logged_artifacts() if a.type == "diagnostic")
            if saved.state != "COMMITTED":
                raise ValueError("Artifact is not committed")
            path = saved.get_entry("result.json").download(
                root=str(args.output / "remote-result")
            )
            if json.loads(Path(path).read_text()) != result:
                raise ValueError("Downloaded artifact differs")
            status = {
                "history_verified": True,
                "artifact_verified": True,
                "rendering_verified": False,
                "recovery_sync_used": False,
            }
        except Exception as error:
            status = {
                "history_verified": False,
                "error": str(error),
                "rendering_verified": False,
            }
            (args.output / "remote-verification.json").write_text(
                json.dumps(status, indent=2)
            )
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Ordinary online delivery failed; no recovery sync attempted"
                ) from error
            time.sleep(10)
            continue
        (args.output / "remote-verification.json").write_text(
            json.dumps(status, indent=2)
        )
        break
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
