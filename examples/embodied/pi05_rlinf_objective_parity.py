"""Compare ART-Embodied Flow-SDE GRPO math with an RLinf checkout.

This diagnostic intentionally imports RLinf's implementation instead of
copying its equations. It provides a same-tensor conformance gate without
making RLinf a runtime dependency of ART-Embodied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from art_embodied.backends.flow_sde_grpo import (
    flow_sde_grpo_loss,
    group_relative_advantages,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rlinf-source",
        type=Path,
        required=True,
        help="RLinf source checkout containing the rlinf package.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def run(rlinf_source: Path) -> dict[str, object]:
    source = rlinf_source.expanduser().resolve()
    if not (source / "rlinf/algorithms/losses.py").is_file():
        raise FileNotFoundError(f"Not an RLinf source checkout: {source}")
    sys.path.insert(0, str(source))
    from rlinf.algorithms.advantages import compute_grpo_advantages
    from rlinf.algorithms.losses import compute_ppo_actor_loss

    rewards = torch.tensor(
        [
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    loss_mask = torch.tensor(
        [
            [True] * 16,
            [True] * 8 + [False] * 8,
            [False] * 16,
        ]
    )
    art_group_advantage = group_relative_advantages(
        rewards,
        epsilon=1.0e-6,
        std_unbiased=True,
    )
    rlinf_advantage, _ = compute_grpo_advantages(
        rewards=rewards.flatten(),
        loss_mask=loss_mask,
        group_size=8,
    )
    art_advantage = art_group_advantage.flatten().unsqueeze(0) * loss_mask

    old = torch.tensor([-1.2, -0.7, -1.5, -0.9], dtype=torch.float32)
    art_current = (old + torch.tensor([0.03, -0.04, 0.4, -0.5])).reshape(
        -1, 1, 1
    )
    art_current.requires_grad_(True)
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5], dtype=torch.float32)
    row_mask = torch.tensor([True, True, True, False])
    trajectory_steps = torch.tensor([120.0, 60.0, 240.0, 120.0])
    art_loss, art_metrics = flow_sde_grpo_loss(
        art_current,
        old.reshape(-1, 1, 1),
        advantages,
        loss_mask=row_mask,
        row_weights=240.0 / trajectory_steps,
        loss_denominator=4,
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.2,
        clip_ratio_c=3.0,
    )
    art_loss.backward()
    art_gradient = art_current.grad.detach().reshape(-1)

    rlinf_current = art_current.detach().reshape(-1).requires_grad_(True)
    rlinf_loss, rlinf_metrics = compute_ppo_actor_loss(
        logprobs=rlinf_current,
        old_logprobs=old,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        advantages=advantages,
        loss_mask=row_mask,
        clip_ratio_c=3.0,
        max_episode_steps=240,
        loss_mask_sum=trajectory_steps,
    )
    rlinf_loss.backward()
    rlinf_gradient = rlinf_current.grad.detach()

    deltas = {
        "advantage_max_abs": float((art_advantage - rlinf_advantage).abs().max()),
        "loss_abs": float((art_loss.detach() - rlinf_loss.detach()).abs()),
        "gradient_max_abs": float((art_gradient - rlinf_gradient).abs().max()),
        "ratio_mean_abs": float(
            (
                art_metrics.ratio_mean.detach()
                - rlinf_metrics["actor/ratio"].detach()
            ).abs()
        ),
        "approximate_kl_abs": float(
            (
                art_metrics.approximate_kl.detach()
                - rlinf_metrics["actor/approx_kl"].detach()
            ).abs()
        ),
        "clip_fraction_abs": float(
            (
                art_metrics.clip_fraction.detach()
                - rlinf_metrics["actor/clip_fraction"].detach()
            ).abs()
        ),
    }
    tolerance = 1.0e-7
    return {
        "status": "ok" if max(deltas.values()) <= tolerance else "mismatch",
        "rlinf_source": str(source),
        "tolerance": tolerance,
        "deltas": deltas,
        "art_loss": float(art_loss.detach()),
        "rlinf_loss": float(rlinf_loss.detach()),
    }


def main() -> None:
    args = parse_args()
    result = run(args.rlinf_source)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
