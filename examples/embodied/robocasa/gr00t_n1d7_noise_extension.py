"""Extend an inconclusive GR00T N1.7 ODE/SDE panel with new reset seeds."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from art_embodied import EmbodiedExperimentConfig

from . import train
from .gr00t_n1d7_sampler_calibration import (
    _comparison_bundle,
    _condition_config,
    _condition_paths,
    _equivalence_decision,
    _load_completed_rows,
    _sha256,
    _write_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prior-ode-outcomes", type=Path, required=True)
    parser.add_argument("--prior-sde-outcomes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--noise-level", type=float, default=0.1)
    parser.add_argument("--seed-start", type=int, default=416)
    parser.add_argument("--seed-count", type=int, default=16)
    parser.add_argument("--policy-seed-repetitions", type=int, default=8)
    parser.add_argument("--maximum-absolute-gap", type=float, default=0.10)
    parser.add_argument(
        "--wandb-group",
        default="cuttingboard-pan-k4-noise01-extension-v1",
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def _verify_preregistration(
    path: Path,
    *,
    config: Path,
    prior_ode: Path,
    prior_sde: Path,
    noise_level: float,
    seeds: list[int],
    repetitions: int,
    maximum_absolute_gap: float,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frozen = payload.get("frozen_inputs", {})
    design = payload.get("extension_design", {})
    gate = payload.get("equivalence_gate", {})
    expected = {
        "base_config_sha256": _sha256(config),
        "prior_ode_outcomes_sha256": _sha256(prior_ode),
        "prior_sde_outcomes_sha256": _sha256(prior_sde),
        "noise_level": noise_level,
        "new_environment_seeds": seeds,
        "policy_seed_repetitions_per_environment_seed": repetitions,
        "maximum_absolute_success_rate_gap": maximum_absolute_gap,
    }
    actual = {
        "base_config_sha256": frozen.get("base_config_sha256"),
        "prior_ode_outcomes_sha256": frozen.get("prior_ode_outcomes_sha256"),
        "prior_sde_outcomes_sha256": frozen.get("prior_sde_outcomes_sha256"),
        "noise_level": design.get("noise_level"),
        "new_environment_seeds": design.get("new_environment_seeds"),
        "policy_seed_repetitions_per_environment_seed": design.get(
            "policy_seed_repetitions_per_environment_seed"
        ),
        "maximum_absolute_success_rate_gap": gate.get(
            "maximum_absolute_success_rate_gap"
        ),
    }
    mismatches = {
        key: {"expected": value, "actual": actual[key]}
        for key, value in expected.items()
        if actual[key] != value
    }
    if mismatches:
        raise ValueError(f"Noise extension preregistration mismatch: {mismatches}")


def _merge_outcomes(prior: Path, extension: Path, destination: Path) -> Path:
    prior_payload = json.loads(prior.read_text(encoding="utf-8"))
    extension_payload = json.loads(extension.read_text(encoding="utf-8"))
    prior_rows = _load_completed_rows(prior)
    extension_rows = _load_completed_rows(extension)
    prior_keys = {
        (row["scenario_id"], row["environment_seed"], row["policy_seed"])
        for row in prior_rows
    }
    extension_keys = {
        (row["scenario_id"], row["environment_seed"], row["policy_seed"])
        for row in extension_rows
    }
    overlap = prior_keys & extension_keys
    if overlap:
        raise ValueError(f"Noise extension duplicates {len(overlap)} prior pairs")
    offset = len(prior_rows)
    renumbered = [
        {**row, "episode": offset + index} for index, row in enumerate(extension_rows)
    ]
    merged = {
        **prior_payload,
        "config_fingerprint": "aggregate-preregistered-panel",
        "episodes": [*prior_rows, *renumbered],
        "aggregate_sources": [
            {"path": str(prior.resolve()), "sha256": _sha256(prior)},
            {"path": str(extension.resolve()), "sha256": _sha256(extension)},
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


async def run(args: argparse.Namespace) -> None:
    if not 0.0 < args.noise_level < 1.0:
        raise ValueError("--noise-level must be in (0, 1)")
    config = args.config.expanduser().resolve()
    prior_ode = args.prior_ode_outcomes.expanduser().resolve()
    prior_sde = args.prior_sde_outcomes.expanduser().resolve()
    preregistration = args.preregistration.expanduser().resolve()
    for path in (config, prior_ode, prior_sde, preregistration):
        if not path.is_file():
            raise FileNotFoundError(path)
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    _verify_preregistration(
        preregistration,
        config=config,
        prior_ode=prior_ode,
        prior_sde=prior_sde,
        noise_level=args.noise_level,
        seeds=seeds,
        repetitions=args.policy_seed_repetitions,
        maximum_absolute_gap=args.maximum_absolute_gap,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base = EmbodiedExperimentConfig.from_yaml(config)
    definitions = (("extension-ode", "native-ode"), ("extension-sde", "flow-sde"))
    conditions: dict[str, dict[str, Path]] = {}
    configs: dict[str, EmbodiedExperimentConfig] = {}
    for label, sampler in definitions:
        paths = _condition_paths(output_dir, label, 0)
        condition = _condition_config(
            base,
            label=label,
            output_dir=output_dir / label,
            seeds=seeds,
            policy_seed_repetitions=args.policy_seed_repetitions,
            sampler=sampler,
            candidate=False,
            preregistration=preregistration,
            noise_level=args.noise_level,
            group=args.wandb_group,
        )
        _write_config(condition, paths["config"])
        conditions[label] = paths
        configs[label] = condition

    preflight = {
        "status": "ok",
        "mode": "preflight" if args.preflight else "execution",
        "optimizer_steps": 0,
        "noise_level": args.noise_level,
        "wandb_group": args.wandb_group,
        "new_environment_seeds": seeds,
        "new_episodes_per_condition": args.seed_count * args.policy_seed_repetitions,
        "aggregate_episodes_per_condition": len(_load_completed_rows(prior_ode))
        + args.seed_count * args.policy_seed_repetitions,
        "conditions": {
            label: {
                "config": str(conditions[label]["config"]),
                "fingerprint": configs[label].fingerprint,
            }
            for label, _ in definitions
        },
    }
    (output_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
    if args.preflight:
        return

    if not args.report_only:
        for label, _ in definitions:
            paths = conditions[label]
            if paths["outcomes"].is_file() and paths["evidence"].is_file():
                _load_completed_rows(paths["outcomes"])
                print(f"[noise-extension] reuse {label}", flush=True)
                continue
            print(f"[noise-extension] start {label}", flush=True)
            await train.run(paths["config"], evaluate_only=True, evaluation_step=0)

    aggregate_ode = _merge_outcomes(
        prior_ode,
        conditions["extension-ode"]["outcomes"],
        output_dir / "aggregate-ode-outcomes.json",
    )
    aggregate_sde = _merge_outcomes(
        prior_sde,
        conditions["extension-sde"]["outcomes"],
        output_dir / "aggregate-sde-outcomes.json",
    )
    comparison = _comparison_bundle(aggregate_ode, aggregate_sde)
    decision = _equivalence_decision(
        comparison["reset_clustered"],
        maximum_absolute_gap=args.maximum_absolute_gap,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "gr00t_n1d7_robocasa_flow_sde_noise_extension",
        "sealed_eligible": False,
        "development_lift_claim_eligible": False,
        "optimizer_steps": 0,
        "noise_level": args.noise_level,
        "decision": decision,
        "training_admission": decision == "pass",
        "automatic_further_extension_allowed": False,
        "comparison": comparison,
        "aggregate_ode": {
            "path": str(aggregate_ode),
            "sha256": _sha256(aggregate_ode),
        },
        "aggregate_sde": {
            "path": str(aggregate_sde),
            "sha256": _sha256(aggregate_sde),
        },
        "preregistration": {
            "path": str(preregistration),
            "sha256": _sha256(preregistration),
        },
    }
    report_path = output_dir / "noise-extension-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if decision != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
