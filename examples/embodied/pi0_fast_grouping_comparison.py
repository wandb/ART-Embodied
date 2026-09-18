"""Same 256 trajectories: two independent G8 advantages versus one pooled G16.

Compare both gradients on a third independent G8 per start. No optimizer steps.
"""

import copy
import json
from pathlib import Path


def pooling_plan(base, source, all_old_seeds):
    from examples.embodied.pi0_fast_repeat_gradient_audit import new_seeds

    previous = json.loads((source / "plan.json").read_text())
    if previous != base:
        raise ValueError("Pooling requires the exact verified source plan")
    verification = json.loads((source / "remote-verification.json").read_text())
    if not all(
        verification[k]
        for k in ("history_verified", "artifact_verified", "video_uploaded")
    ):
        raise ValueError("Pooling source has not passed observability checks")
    result = copy.deepcopy(base)
    used = all_old_seeds + [
        s for g in previous["groups"] for s in g["new_policy_seeds"]
    ]
    for group in result["groups"]:
        gid = group["group_index"]
        pair = json.loads((source / f"pair-{gid}.json").read_text())
        if not pair["policy_weights_unchanged"]:
            raise ValueError("Source policy was modified")
        group["pooled_policy_seeds"] = group["new_policy_seeds"]
        group["new_policy_seeds"] = new_seeds(gid + 64, used)
        used.extend(group["new_policy_seeds"])
    result.update(
        scope=__doc__,
        pooling_source=str(source.resolve()),
        fit_trajectories=256,
        probe_trajectories=128,
        fit_groupings=["2 x G8 per start", "1 x G16 per start"],
    )
    return result


def average_payloads(a, b):
    if (
        set(a["gradients"]) != set(b["gradients"])
        or a["missing_gradients"]
        or b["missing_gradients"]
    ):
        raise ValueError("Gradient surfaces differ or are incomplete")
    result = copy.deepcopy(a)
    for name, value in a["gradients"].items():
        other = b["gradients"][name]
        if value.shape != other.shape:
            raise ValueError("Gradient shapes differ")
        result["gradients"][name] = (value.double() + other.double()) / 2
    return result


def separate_gradient(source, gid, output):
    import torch

    from examples.embodied.pi0_fast_repeat_gradient_audit import vector

    payloads = [
        torch.load(source / f"{side}-{gid}.pt", map_location="cpu", weights_only=False)
        for side in ("original", "repeated")
    ]
    result = average_payloads(*payloads)
    flat = vector(result)
    torch.save(result, output / f"separate-{gid}.pt")
    return flat


def compare(output, workers):
    from safetensors.torch import load_file
    import torch

    from examples.embodied.pi0_fast_optimizer_calibration_audit import adam_delta
    from examples.embodied.pi0_fast_repeat_gradient_audit import cosine, vector

    gradients = {}
    names = None
    for side in ("original", "separate", "repeated"):
        values = []
        for gid in range(16):
            payload = torch.load(
                output / f"{side}-{gid}.pt", map_location="cpu", weights_only=False
            )
            current = sorted(payload["gradients"])
            names = current if names is None else names
            if current != names:
                raise ValueError("Comparison parameter surfaces differ")
            values.append(vector(payload))
        gradients[side] = sum(values) / 16
    state = load_file(
        workers / "snapshots/update-0000/initial/adapter_model.safetensors"
    )
    keys = [
        n.removeprefix("model.").replace(".default.weight", ".weight") for n in names
    ]
    if set(keys) != set(state):
        raise ValueError("Checkpoint surface differs")
    weight = torch.cat([state[k].double().flatten() for k in keys])
    zeros = torch.zeros_like(weight)
    pooled, separate, probe = (
        gradients[k] for k in ("original", "separate", "repeated")
    )
    result = {
        "pooled_vs_separate_cosine": cosine(pooled, separate),
        "pooled_probe_cosine": cosine(pooled, probe),
        "separate_probe_cosine": cosine(separate, probe),
        "caution": "First-order surrogate predictions, not measured return gains; one independent probe realization.",
    }
    for name, grad in (("pooled", pooled), ("separate", separate)):
        clipped = grad * min(1.0, 1.0 / (float(grad.norm()) + 1e-6))
        delta, _, _, _ = adam_delta(weight, clipped, zeros, zeros, 1, lr=2e-6, eps=1e-5)
        for key, target in (
            ("probe", probe),
            ("pooled_fit", pooled),
            ("separate_fit", separate),
        ):
            result[f"{name}_{key}_first_order_gain"] = float(-target @ delta)
        result[f"{name}_update_norm"] = float(delta.norm())
    return result
