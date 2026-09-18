"""CPU checks of original rollout records, training rows and snapshot files."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


def observation_equal(left, right):
    return set(left) == set(right) and all(
        np.array_equal(left[k], right[k]) for k in left
    )


def observation_digest(values):
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        array = np.asarray(value)
        digest.update(f"{key}:{array.dtype}:{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def check_optimizer_handoffs(training, experiment):
    from safetensors.torch import load_file
    import torch

    from examples.embodied.pi0_fast_optimizer_calibration_audit import adam_delta

    initial = load_file(
        training / "snapshots/update-0000/initial/adapter_model.safetensors"
    )
    first_payload = torch.load(
        training / "worker-00/update-0000-job-0000/gradients.pt",
        map_location="cpu",
        weights_only=False,
    )
    names = list(first_payload["gradients"])
    keys = [
        n.removeprefix("model.").replace(".default.weight", ".weight") for n in names
    ]
    weight = torch.cat([initial[k].double().flatten() for k in keys])
    m, v = torch.zeros_like(weight), torch.zeros_like(weight)
    rows = []
    for job in range(4):
        raw = torch.zeros_like(weight)
        for worker in range(8):
            payload = torch.load(
                training
                / f"worker-{worker:02d}/update-0000-job-{job:04d}/gradients.pt",
                map_location="cpu",
                weights_only=False,
            )
            raw += torch.cat(
                [payload["gradients"][n].double().flatten() for n in names]
            )
        grad = raw * min(1.0, 1.0 / (float(raw.norm()) + 1e-6))
        delta, m, v, _ = adam_delta(weight, grad, m, v, job + 1, lr=2e-6, eps=1e-5)
        target_path = (
            training / f"snapshots/job-{job + 1:04d}"
            if job < 3
            else experiment / "checkpoints/step_000001"
        )
        target = load_file(target_path / "adapter_model.safetensors")
        expected = weight + delta
        actual = torch.cat([target[k].double().flatten() for k in keys])
        error = float((expected - actual).norm() / delta.norm())
        if not torch.isfinite(expected).all() or error > 0.01:
            raise ValueError(
                f"Optimizer-to-next-worker mismatch at minibatch {job}: {error}"
            )
        rows.append(
            {
                "minibatch": job,
                "relative_update_error": error,
                "max_weight_error": float((expected - actual).abs().max()),
                "target": str(target_path),
            }
        )
        # Continue from the observed FP32 weights, retaining independent moments.
        weight = actual
    return rows


def check(handoffs, experiment):
    from safetensors.torch import load_file
    import torch

    training = handoffs / "training-workers-xfa5130m"
    job = training / "worker-00/update-0000-job-0000"
    with (job / "examples.pkl").open("rb") as f:
        examples = pickle.load(f)
    selected = defaultdict(list)
    for e in examples:
        selected[int(e.metadata["group_index"])].append(e)
    evidence = json.loads(
        (experiment / "rollout-evidence/policy-version-000000.json").read_text()
    )
    groups = {int(g["group_index"]): g for g in evidence["groups"]}
    matched = 0
    initial_equal = 0
    trajectories = 0
    actors = handoffs / "rollout-actors-4qly9v9z"
    sources = []
    original_rows = {}
    for request in sorted(actors.glob("worker-*/job-*/request.pkl")):
        with request.open("rb") as f:
            q = pickle.load(f)
        gid = int(q["contexts"][0]["group_index"])
        if q["phase"] != "train" or q["contexts"][0]["update"] != 0:
            raise ValueError("Unexpected source phase or update")
        path = request.parent / "trajectories.pkl"
        with path.open("rb") as f:
            ts = pickle.load(f)
        if len(ts) != 8 or [t.reward for t in ts] != groups[gid]["rewards"]:
            raise ValueError("Original rollout rewards disagree with evidence")
        first = ts[0].observations[0].value
        if not all(observation_equal(first, t.observations[0].value) for t in ts):
            raise ValueError(f"Group {gid} did not start at identical observations")
        initial_equal += 1
        trajectories += len(ts)
        for attempt, t in enumerate(ts):
            if t.reward != float(t.metrics["success"]):
                raise ValueError("Terminal reward and success disagree")
            primitive_success = any(
                any(float(r) > 0 for r in a.metadata.get("primitive_rewards", []))
                for a in t.actions
            )
            if bool(t.metrics["success"]) != primitive_success:
                raise ValueError("Primitive success and terminal reward disagree")
            for index, action in enumerate(t.actions):
                original_rows[(gid, attempt, index)] = (
                    observation_digest(t.observations[index].value),
                    tuple(action.raw["tokens"]),
                    tuple(action.logprobs),
                    action.raw["prompt"],
                    action.step,
                )
        for e in selected[gid]:
            t = ts[int(e.metadata["trajectory_index_in_group"])]
            a = t.actions[e.action_index]
            o = t.observations[e.action_index]
            if not observation_equal(e.observation.value, o.value):
                raise ValueError("Worker input differs from original observation")
            if list(e.tokens) != a.raw["tokens"] or list(e.logprobs) != a.logprobs:
                raise ValueError("Worker tokens/logprobs differ from original action")
            if e.prompt != a.raw["prompt"] or e.step != a.step:
                raise ValueError("Worker prompt/step differs from original action")
            matched += 1
        sources.append(str(path))
        del ts
    if matched != len(examples) or initial_equal != 64:
        raise ValueError("Incomplete original-record coverage")
    from examples.embodied.pi0_fast_pipeline_audit import validate_example

    seen = set()
    minibatches = defaultdict(set)
    for path in sorted(training.glob("worker-*/update-0000-job-*/examples.pkl")):
        job_index = int(path.parent.name.rsplit("-", 1)[-1])
        with path.open("rb") as f:
            shard = pickle.load(f)
        for e in shard:
            validate_example(e, groups)
            key = (
                int(e.metadata["group_index"]),
                int(e.metadata["trajectory_index_in_group"]),
                e.action_index,
            )
            row = (
                observation_digest(e.observation.value),
                tuple(e.tokens),
                tuple(e.logprobs),
                e.prompt,
                e.step,
            )
            if key in seen or original_rows[key] != row:
                raise ValueError(f"Duplicate or mismatched training row: {key}")
            seen.add(key)
            minibatches[job_index].add(key[:2])
        del shard
    if seen != set(original_rows):
        raise ValueError("Training rows do not exactly cover rollout actions")
    if len(minibatches) != 4 or any(len(v) != 128 for v in minibatches.values()):
        raise ValueError("Unexpected trajectory minibatch membership")
    if len(set.union(*minibatches.values())) != 512:
        raise ValueError("Trajectory minibatches overlap")
    # Compare independently saved coordinator, training and rollout snapshots.
    paths = defaultdict(list)
    for p in handoffs.glob(
        "training-workers-*/snapshots/update-*/initial/adapter_model.safetensors"
    ):
        paths[int(p.parent.parent.name.removeprefix("update-"))].append(p)
    for p in handoffs.glob(
        "rollout-actors-*/snapshots/update-*/adapter_model.safetensors"
    ):
        paths[int(p.parent.name.removeprefix("update-"))].append(p)
    for p in (experiment / "checkpoints").glob("step_*/adapter_model.safetensors"):
        paths[int(p.parent.name.removeprefix("step_"))].append(p)
    snapshots = []
    for update, entries in sorted(paths.items()):
        if len(entries) < 2:
            continue
        reference = load_file(entries[0])
        for p in entries[1:]:
            other = load_file(p)
            if set(other) != set(reference) or not all(
                torch.equal(other[k], reference[k]) for k in other
            ):
                raise ValueError(f"Snapshot values disagree at update {update}: {p}")
        snapshots.append(
            {
                "update": update,
                "copies": len(entries),
                "bitwise_equal": True,
                "paths": [str(p) for p in entries],
            }
        )
    return {
        "all_worker_rows_matched": len(seen),
        "minibatch_trajectory_counts": {str(k): len(v) for k, v in minibatches.items()},
        "optimizer_handoffs": check_optimizer_handoffs(training, experiment),
        "matched_worker_examples": matched,
        "original_trajectories": trajectories,
        "identical_initial_observation_groups": initial_equal,
        "snapshots": snapshots,
        "source_paths": sources,
        "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoffs", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.handoffs, args.experiment)
    args.output.write_text(json.dumps(result, indent=2))
    print({k: v for k, v in result.items() if k not in ("snapshots", "source_paths")})
