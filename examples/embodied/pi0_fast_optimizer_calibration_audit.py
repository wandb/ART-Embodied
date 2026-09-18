"""CPU replay of saved gradients, not a new policy-training experiment.

Counterfactual updates hold gradient and optimizer history fixed. Their predicted
gain is a local first-order quantity, not a measured success-rate improvement.
"""

import argparse
import json
from pathlib import Path


def adam_delta(
    weight, grad, first, second, step, *, lr, eps, beta1=0.9, beta2=0.999, wd=0.01
):
    m = beta1 * first + (1 - beta1) * grad
    v = beta2 * second + (1 - beta2) * grad.square()
    rms = (v / (1 - beta2**step)).sqrt()
    adaptive = -lr * (m / (1 - beta1**step)) / (rms + eps)
    return adaptive - lr * wd * weight, m, v, rms


def main():
    from safetensors.torch import load_file
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    initial = load_file(
        args.workers / "snapshots/update-0000/initial/adapter_model.safetensors"
    )
    target = load_file(args.checkpoint / "adapter_model.safetensors")
    names = list(
        torch.load(
            args.workers / "worker-00/update-0000-job-0000/gradients.pt",
            map_location="cpu",
            weights_only=False,
        )["gradients"]
    )
    keys = [
        n.removeprefix("model.").replace(".default.weight", ".weight") for n in names
    ]
    if set(keys) != set(initial):
        raise ValueError("Gradient and checkpoint parameters differ")
    weight = torch.cat([initial[k].double().flatten() for k in keys])
    first, second = torch.zeros_like(weight), torch.zeros_like(weight)
    role_a = torch.cat(
        [
            torch.full((initial[k].numel(),), "lora_A" in k, dtype=torch.bool)
            for k in keys
        ]
    )
    rows = []
    for job in range(4):
        shards = [
            torch.load(
                args.workers / f"worker-{i:02d}/update-0000-job-{job:04d}/gradients.pt",
                map_location="cpu",
                weights_only=False,
            )["gradients"]
            for i in range(8)
        ]
        raw = torch.cat([sum(p[n].double() for p in shards).flatten() for n in names])
        clip = min(1.0, 1.0 / (float(raw.norm()) + 1e-6))
        grad = raw * clip
        variants = []
        for lr, eps in (
            (2e-7, 1e-5),
            (2e-6, 1e-5),
            (1e-5, 1e-5),
            (2.5e-5, 1e-5),
            (2e-6, 1e-8),
        ):
            d, m, v, rms = adam_delta(
                weight, grad, first, second, job + 1, lr=lr, eps=eps
            )
            role_metrics = {}
            for role, mask in (("A", role_a), ("B", ~role_a)):
                nonzero = rms[mask] > 0
                attenuation = rms[mask][nonzero] / (rms[mask][nonzero] + eps)
                role_metrics[role] = {
                    "delta_norm": float(d[mask].norm()),
                    "rms_median": float(rms[mask].median()),
                    "epsilon_attenuation_median_nonzero": float(attenuation.median())
                    if nonzero.any()
                    else None,
                    "predicted_surrogate_gain": float(-(raw[mask] * d[mask]).sum()),
                }
            variants.append(
                {
                    "lr": lr,
                    "epsilon": eps,
                    "delta_norm": float(d.norm()),
                    "predicted_surrogate_gain": float(-(raw * d).sum()),
                    "roles": role_metrics,
                }
            )
        d, first, second, _ = adam_delta(
            weight, grad, first, second, job + 1, lr=2e-6, eps=1e-5
        )
        weight = weight + d
        rows.append(
            {
                "minibatch": job,
                "raw_gradient_norm": float(raw.norm()),
                "clip_scale": clip,
                "variants": variants,
            }
        )
    actual = torch.cat([target[k].double().flatten() for k in keys])
    base = torch.cat([initial[k].double().flatten() for k in keys])
    result = {
        "scope": __doc__,
        "rows": rows,
        "replay_vs_actual_delta_relative_error": float(
            (weight - actual).norm() / (actual - base).norm()
        ),
        "replay_vs_actual_weight_abs_max": float((weight - actual).abs().max()),
    }
    # A double-precision independent equation need not reproduce float32 rounding exactly.
    if result["replay_vs_actual_delta_relative_error"] > 0.01:
        raise ValueError(
            f"Independent Adam replay did not match actual weights: {result}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
