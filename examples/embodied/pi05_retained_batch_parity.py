"""Replay a retained ART Flow-SDE batch through RLinf's public loss math."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types

import torch

from art_embodied.backends.flow_sde_grpo import flow_sde_grpo_loss


def _load_rlinf_math(source: Path):
    """Load RLinf's math files without importing its optional runtime stack."""

    def register(_name):
        return lambda function: function

    def masked_mean(values, mask, axis=None):
        if mask is None:
            return values.mean(axis=axis)
        if (~mask).all():
            return (values * mask).sum(axis=axis)
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)

    def masked_mean_ratio(values, mask, loss_mask_ratio):
        return (values / loss_mask_ratio * mask).mean()

    stubs = {
        "rlinf": types.ModuleType("rlinf"),
        "rlinf.algorithms": types.ModuleType("rlinf.algorithms"),
        "rlinf.algorithms.registry": types.ModuleType("rlinf.algorithms.registry"),
        "rlinf.algorithms.utils": types.ModuleType("rlinf.algorithms.utils"),
        "rlinf.utils": types.ModuleType("rlinf.utils"),
        "rlinf.utils.utils": types.ModuleType("rlinf.utils.utils"),
    }
    stubs["rlinf.algorithms.registry"].register_advantage = register
    stubs["rlinf.algorithms.registry"].register_policy_loss = register
    stubs["rlinf.algorithms.utils"].huber_loss = lambda error, delta: torch.where(
        error.abs() < delta,
        0.5 * error**2,
        delta * (error.abs() - 0.5 * delta),
    )
    stubs["rlinf.algorithms.utils"].kl_penalty = lambda *_args, **_kwargs: None
    stubs["rlinf.algorithms.utils"].safe_normalize = lambda values, **_kwargs: values
    stubs["rlinf.utils.utils"].masked_mean = masked_mean
    stubs["rlinf.utils.utils"].masked_mean_ratio = masked_mean_ratio
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        modules = []
        for module_name, relative_path in (
            ("_art_embodied_rlinf_advantages", "rlinf/algorithms/advantages.py"),
            ("_art_embodied_rlinf_losses", "rlinf/algorithms/losses.py"),
        ):
            spec = importlib.util.spec_from_file_location(
                module_name, source / relative_path
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Unable to load RLinf source: {relative_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules.append(module)
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return modules[0].compute_grpo_advantages, modules[1].compute_ppo_actor_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--rlinf-source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    return parser.parse_args()


def run(
    audit_path: Path,
    *,
    rlinf_source: Path,
    tolerance: float,
) -> dict[str, object]:
    source = rlinf_source.expanduser().resolve()
    if not (source / "rlinf/algorithms/losses.py").is_file():
        raise FileNotFoundError(f"Not an RLinf source checkout: {source}")
    compute_grpo_advantages, compute_ppo_actor_loss = _load_rlinf_math(source)

    audit = torch.load(
        audit_path.expanduser().resolve(), map_location="cpu", weights_only=True
    )
    rewards = audit["rewards"].float()
    art_advantages = audit["group_advantages"].float()
    reference_advantages, _ = compute_grpo_advantages(
        rewards=rewards.flatten(),
        loss_mask=torch.ones((1, rewards.numel()), dtype=torch.bool),
        group_size=int(audit["group_size"]),
    )
    deltas: dict[str, float] = {
        "advantage_max_abs": float(
            (art_advantages.flatten() - reference_advantages.flatten()).abs().max()
        )
    }
    subupdate_results = []
    for subupdate in audit["subupdates"]:
        current = subupdate["current_chunk_logprobs"].float()
        old = subupdate["old_chunk_logprobs"].float()
        advantages = subupdate["advantages"].float()
        mask = subupdate["loss_mask"].bool()
        trajectory_steps = subupdate["trajectory_primitive_steps"].float()
        kwargs = {}
        if subupdate["length_normalized"]:
            kwargs = {
                "max_episode_steps": int(audit["max_episode_steps"]),
                "loss_mask_sum": trajectory_steps,
            }
        reference_loss, reference_metrics = compute_ppo_actor_loss(
            logprobs=current,
            old_logprobs=old,
            clip_ratio_low=float(audit["clip_epsilon_low"]),
            clip_ratio_high=float(audit["clip_epsilon_high"]),
            advantages=advantages,
            loss_mask=mask,
            clip_ratio_c=audit["clip_ratio_c"],
            **kwargs,
        )
        # RLinf pads each actor batch to the configured global batch before
        # ``masked_mean_ratio`` takes its mean. ART omits those inert rows and
        # carries the same denominator explicitly.
        reference_loss = reference_loss * (
            current.numel() / int(subupdate["loss_denominator"])
        )
        art_loss, art_objective = flow_sde_grpo_loss(
            current[:, None, None],
            old[:, None, None],
            advantages,
            loss_mask=mask,
            row_weights=(
                int(audit["max_episode_steps"]) / trajectory_steps
                if subupdate["length_normalized"]
                else None
            ),
            loss_denominator=int(subupdate["loss_denominator"]),
            clip_epsilon_low=float(audit["clip_epsilon_low"]),
            clip_epsilon_high=float(audit["clip_epsilon_high"]),
            clip_ratio_c=audit["clip_ratio_c"],
        )
        art_metrics = subupdate["art_metrics"]
        item_deltas = {
            "reference_loss_abs": abs(float(reference_loss) - float(art_loss)),
            "ratio_mean_abs": abs(
                float(reference_metrics["actor/ratio"])
                - float(art_objective.ratio_mean)
            ),
            "approximate_kl_abs": abs(
                float(reference_metrics["actor/approx_kl"])
                - float(art_objective.approximate_kl)
            ),
            "clip_fraction_abs": abs(
                float(reference_metrics["actor/clip_fraction"])
                - float(art_objective.clip_fraction)
            ),
            "executed_loss_abs": abs(float(art_loss) - float(art_metrics["loss"])),
            "executed_ratio_mean_abs": abs(
                float(art_objective.ratio_mean) - float(art_metrics["ratio_mean"])
            ),
            "executed_approximate_kl_abs": abs(
                float(art_objective.approximate_kl)
                - float(art_metrics["approximate_kl"])
            ),
            "executed_clip_fraction_abs": abs(
                float(art_objective.clip_fraction)
                - float(art_metrics["clip_fraction"])
            ),
        }
        index = int(subupdate["index"])
        deltas.update(
            {f"subupdate_{index}/{key}": value for key, value in item_deltas.items()}
        )
        subupdate_results.append(
            {
                "index": index,
                "rows": int(current.numel()),
                "valid_rows": int(mask.count_nonzero()),
                "deltas": item_deltas,
            }
        )
    maximum = max(deltas.values(), default=0.0)
    return {
        "status": "ok" if maximum <= tolerance else "mismatch",
        "audit": str(audit_path.expanduser().resolve()),
        "rlinf_source": str(source),
        "tolerance": tolerance,
        "max_delta": maximum,
        "deltas": deltas,
        "subupdates": subupdate_results,
    }


def main() -> None:
    args = parse_args()
    result = run(
        args.audit,
        rlinf_source=args.rlinf_source,
        tolerance=args.tolerance,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
