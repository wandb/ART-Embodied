"""Read-only diagnosis of native FAST score-sum updates on retained trajectories.

Teacher-forced likelihood and fixed-observation greedy probes are not new
rollouts or evidence of improved success. Only development evaluations measure
closed-loop success here; sealed data is never accessed.
Likelihood forwards enable autograd to match training, but never backpropagate.
"""

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np


def training_path_scores(policy, examples):
    import torch

    # no_grad can dispatch a different attention kernel than the training path.
    # Detach immediately: matching that forward does not require a backward.
    with torch.enable_grad():
        scores = policy.action_token_logprobs(examples)
        return [score.detach().cpu().double().numpy() for score in scores]


def token_statistics(old, baseline, current, advantage, mask, low=0.2, high=0.28):
    arrays = [np.asarray(x) for x in (old, baseline, current, advantage, mask)]
    if len({x.shape for x in arrays}) != 1:
        raise ValueError("Token/advantage/mask shapes differ")
    old, baseline, current, advantage, mask = arrays
    mask = mask.astype(bool)
    if not all(np.isfinite(x[mask]).all() for x in arrays[:4]):
        raise ValueError("Nonfinite active token")
    a = advantage[mask].astype(np.float64)
    before = baseline[mask].astype(np.float64)
    after = current[mask].astype(np.float64)
    reference = old[mask].astype(np.float64)
    delta = after - before
    rb, ra = np.exp(before - reference), np.exp(after - reference)
    if not np.isfinite(ra).all() or not np.isfinite(rb).all():
        raise ValueError("Nonfinite importance ratio")
    surrogate = lambda r: np.minimum(a * r, a * np.clip(r, 1 - low, 1 + high))
    return {
        "tokens": int(mask.sum()),
        "logprob_delta_sum": float(delta.sum()),
        "advantage_logprob_delta_sum": float((a * delta).sum()),
        "clipped_surrogate_gain_sum": float((surrogate(ra) - surrogate(rb)).sum()),
        "sign_agreement_tokens": int(((a * delta) > 0).sum()),
        "signal_tokens": int((a != 0).sum()),
        "baseline_old_abs_delta_max": float(np.abs(before - reference).max(initial=0)),
    }


def summarize(rows):
    if not rows:
        raise ValueError("No trajectories")
    positive = [r for r in rows if r["advantage"] > 0]
    negative = [r for r in rows if r["advantage"] < 0]
    return {
        "trajectories": len(rows),
        "positive_trajectories": len(positive),
        "negative_trajectories": len(negative),
        "score_sum_gain_per_trajectory": sum(
            r["advantage_logprob_delta_sum"] for r in rows
        )
        / len(rows),
        "clipped_surrogate_gain_per_trajectory": sum(
            r["clipped_surrogate_gain_sum"] for r in rows
        )
        / len(rows),
        "positive_sequence_logprob_delta_mean": sum(
            r["logprob_delta_sum"] for r in positive
        )
        / max(1, len(positive)),
        "negative_sequence_logprob_delta_mean": sum(
            r["logprob_delta_sum"] for r in negative
        )
        / max(1, len(negative)),
        "trajectory_sign_agreement": sum(
            r["advantage"] * r["logprob_delta_sum"] > 0 for r in positive + negative
        )
        / max(1, len(positive) + len(negative)),
        "baseline_old_abs_delta_max": max(
            r["baseline_old_abs_delta_max"] for r in rows
        ),
    }


def cpu_direction(root, first_checkpoint):
    from safetensors.torch import load_file
    import torch

    paths = (
        [root / "snapshots/update-0000/initial"]
        + [root / f"snapshots/job-{j:04d}" for j in range(1, 4)]
        + [first_checkpoint]
    )
    weights = [load_file(p / "adapter_model.safetensors") for p in paths]
    results = []
    for j in range(4):
        payloads = [
            torch.load(
                root / f"worker-{i:02d}/update-0000-job-{j:04d}/gradients.pt",
                map_location="cpu",
                weights_only=False,
            )["gradients"]
            for i in range(8)
        ]
        dot = gn = dn = 0.0
        for name in payloads[0]:
            key = name.removeprefix("model.").replace(".default.weight", ".weight")
            g = sum(p[name].double() for p in payloads)
            d = weights[j + 1][key].double() - weights[j][key].double()
            if not torch.isfinite(g).all() or not torch.isfinite(d).all():
                raise ValueError("Nonfinite gradient/weight change")
            dot += float((g * d).sum())
            gn += float(g.square().sum())
            dn += float(d.square().sum())
        results.append(
            {
                "minibatch": j,
                "predicted_surrogate_gain_first_order": -dot,
                "ascent_direction_cosine": -dot / math.sqrt(gn * dn),
                "gradient_norm": math.sqrt(gn),
                "update_norm": math.sqrt(dn),
            }
        )
    return results


def worker(args):
    import torch

    from art_embodied.backends.action_token_worker import _build_runtime
    from art_embodied.integrations.pi0_fast import _concatenate_processed_batches

    spec = json.loads(
        (args.workers / f"worker-{args.worker:02d}/bootstrap.json").read_text()
    )
    config, policy, _ = _build_runtime(spec)
    if config.algorithm.loss_aggregation != "seq_mean_token_sum":
        raise ValueError("This diagnostic only supports the score-sum objective")
    policy.eval()
    examples = []
    for j in range(4):
        with (
            args.workers
            / f"worker-{args.worker:02d}/update-0000-job-{j:04d}/examples.pkl"
        ).open("rb") as handle:
            examples.extend(pickle.load(handle))
    selected = []
    for sign in (-1, 1):
        pool = [
            i
            for i, e in enumerate(examples)
            if sign * e.metadata["group_advantage"] > 0
        ]
        if len(pool) < 4:
            raise ValueError("Not enough signed contexts")
        selected.extend(pool[k * len(pool) // 4] for k in range(4))
    probe_batches = [
        _concatenate_processed_batches(
            [
                policy.preprocess_observation(
                    examples[i].observation.value,
                    task=examples[i].prompt,
                    robot_type="panda",
                )
                for i in selected[k : k + 2]
            ]
        )
        for k in range(0, len(selected), 2)
    ]

    @torch.no_grad()
    def greedy():
        values = []
        for batch in probe_batches:
            tokens = policy.sample_action_tokens(batch, temperature=0)
            decoded, errors = policy.decode_action_tokens_native(tokens)
            for t, d, error in zip(tokens, decoded, errors, strict=True):
                values.append(
                    {
                        "tokens": t.cpu().tolist(),
                        "action": None if error else d.detach().cpu().tolist(),
                        "error": error,
                    }
                )
        return values

    output = {"worker": args.worker, "examples": len(examples), "stages": {}}
    baseline = []
    stages = [
        ("initial", args.workers / "snapshots/update-0000/initial"),
        ("full_update_1", args.control_checkpoint),
        ("minibatch_1", args.first_checkpoint),
        ("minibatch_10", args.candidate),
    ]
    for label, path in stages:
        policy.load_checkpoint({"path": str(path.resolve())})
        policy.eval()
        rows = defaultdict(lambda: defaultdict(float))
        started = time.monotonic()
        with torch.no_grad():
            for offset in range(0, len(examples), 2):
                batch = examples[offset : offset + 2]
                scores = training_path_scores(policy, batch)
                for i, (e, score) in enumerate(
                    zip(batch, scores, strict=True), start=offset
                ):
                    value = score
                    if label == "initial":
                        baseline.append(value)
                    stats = token_statistics(
                        e.logprobs,
                        baseline[i],
                        value,
                        e.metadata["token_advantages"],
                        e.metadata["token_loss_mask"],
                        config.algorithm.clip_epsilon_low,
                        config.algorithm.clip_epsilon_high,
                    )
                    if stats["baseline_old_abs_delta_max"] > 0.02:
                        delta = np.abs(baseline[i] - np.asarray(e.logprobs))
                        k = int(np.argmax(delta))
                        raise ValueError(
                            f"Initial alignment failed: trajectory={e.trajectory_index} "
                            f"example={i} token_index={k} token_id={e.tokens[k]} "
                            f"old={e.logprobs[k]} baseline={baseline[i][k]}"
                        )
                    row = rows[e.trajectory_index]
                    row["advantage"] = float(e.metadata["group_advantage"])
                    for key, amount in stats.items():
                        if key.endswith("_max"):
                            row[key] = max(row[key], amount)
                        else:
                            row[key] += amount
                if offset % 256 == 0:
                    print(f"{label}: {offset}/{len(examples)}", flush=True)
        output["stages"][label] = {
            "trajectories": dict(rows),
            "greedy": greedy(),
            "seconds": time.monotonic() - started,
        }
        (args.output / f"worker-{args.worker:02d}.json").write_text(json.dumps(output))
    print("All score-only stages complete", flush=True)


def parent(args):
    import wandb

    for checkpoint in (args.first_checkpoint, args.control_checkpoint, args.candidate):
        marker = json.loads(
            (checkpoint / "art_embodied_checkpoint_complete.json").read_text()
        )
        if not marker["complete"]:
            raise ValueError(f"Incomplete checkpoint: {checkpoint}")
    args.output.mkdir(parents=True, exist_ok=False)
    cpu = cpu_direction(args.workers, args.first_checkpoint)
    (args.output / "cpu-direction.json").write_text(json.dumps(cpu, indent=2))
    run = wandb.init(
        entity="wandb-japan",
        project="art-embodied-pi0-fast-spatial",
        name="pi0-fast-native-score-sum-autograd-direction-diagnostic",
        job_type="objective-diagnostic",
        config={
            "workers": str(args.workers),
            "candidate": str(args.candidate),
            "first_checkpoint": str(args.first_checkpoint),
            "control_checkpoint": str(args.control_checkpoint),
            "new_rollout_episodes": 0,
            "optimizer_updates": 0,
            "likelihood_autograd_enabled": True,
            "scope": __doc__,
        },
    )
    run.define_metric("diagnostics/stage")
    run.define_metric("diagnostics/*", step_metric="diagnostics/stage")
    run.log({"diagnostics/stage": 0, "diagnostics/started": 1})
    (args.output / "run.json").write_text(json.dumps({"url": run.url, "id": run.id}))
    processes = []
    files = []
    try:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
        if len(visible) != 8:
            raise ValueError("Diagnostic requires exactly eight allocated GPUs")
        for i in range(8):
            log = (args.output / f"worker-{i:02d}.log").open("w")
            files.append(log)
            env = os.environ | {"CUDA_VISIBLE_DEVICES": visible[i]}
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "examples.embodied.pi0_fast_score_sum_direction",
                        *sys.argv[1:],
                        "--worker",
                        str(i),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        deadline = time.monotonic() + 1500
        while any(p.poll() is None for p in processes):
            if time.monotonic() > deadline or any(
                p.poll() not in (None, 0) for p in processes
            ):
                raise RuntimeError(
                    "Diagnostic worker failed or exceeded 25-minute bound"
                )
            time.sleep(5)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError("Diagnostic worker failed")
        payloads = [
            json.loads((args.output / f"worker-{i:02d}.json").read_text())
            for i in range(8)
        ]
        results = {}
        for index, label in enumerate(payloads[0]["stages"]):
            merged = defaultdict(lambda: defaultdict(float))
            probes = []
            for p in payloads:
                stage = p["stages"][label]
                for tid, values in stage["trajectories"].items():
                    for k, v in values.items():
                        if k == "advantage":
                            merged[tid][k] = v
                        elif k.endswith("_max"):
                            merged[tid][k] = max(merged[tid][k], v)
                        else:
                            merged[tid][k] += v
                probes.extend(
                    zip(p["stages"]["initial"]["greedy"], stage["greedy"], strict=True)
                )
            if len(merged) != 512:
                raise ValueError("Incomplete trajectory population")
            metrics = summarize(list(merged.values()))
            metrics["greedy_probe_contexts"] = len(probes)
            metrics["greedy_token_changed_fraction"] = sum(
                a["tokens"] != b["tokens"] for a, b in probes
            ) / len(probes)
            valid = [
                (np.asarray(a["action"]), np.asarray(b["action"]))
                for a, b in probes
                if a["action"] is not None and b["action"] is not None
            ]
            metrics["greedy_valid_pair_contexts"] = len(valid)
            metrics["greedy_action_changed_fraction"] = sum(
                not np.array_equal(a, b) for a, b in valid
            ) / max(1, len(valid))
            metrics["greedy_executed_prefix_changed_fraction"] = sum(
                not np.array_equal(a[:10], b[:10]) for a, b in valid
            ) / max(1, len(valid))
            results[label] = metrics
            run.log(
                {
                    "diagnostics/stage": index + 1,
                    **{f"diagnostics/{k}": v for k, v in metrics.items()},
                }
            )
        result = {
            "scope": __doc__,
            "cpu_first_order": cpu,
            "results": results,
            "wandb_url": run.url,
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        artifact = wandb.Artifact(
            "pi0-fast-native-score-sum-direction", type="diagnostic"
        )
        artifact.add_dir(str(args.output))
        run.log_artifact(artifact).wait()
        run.summary["diagnostics/completed"] = True
        run.finish()
    except BaseException:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        run.finish(exit_code=1)
        raise
    finally:
        for f in files:
            f.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=Path, required=True)
    parser.add_argument("--first-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    if args.cpu_only:
        args.output.mkdir(parents=True, exist_ok=True)
        result = cpu_direction(args.workers, args.first_checkpoint)
        (args.output / "cpu-direction.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
    elif args.worker is not None:
        worker(args)
    else:
        parent(args)


if __name__ == "__main__":
    main()
