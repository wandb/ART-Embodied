"""Screen GR00T N1.7 Flow-SDE noise against a frozen native-ODE panel."""

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
    parser.add_argument("--reference-ode-outcomes", type=Path, required=True)
    parser.add_argument("--reference-ode-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument(
        "--noise-levels", type=float, nargs="+", default=[0.1, 0.2, 0.3]
    )
    parser.add_argument("--seed-start", type=int, default=400)
    parser.add_argument("--seed-count", type=int, default=16)
    parser.add_argument("--policy-seed-repetitions", type=int, default=8)
    parser.add_argument("--maximum-absolute-gap", type=float, default=0.10)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def _label(noise_level: float) -> str:
    return f"sde-noise-{noise_level:.2f}".replace(".", "p")


def _verify_preregistration(
    path: Path,
    *,
    config: Path,
    reference_outcomes: Path,
    reference_evidence: Path,
    noise_levels: list[float],
    seeds: list[int],
    policy_seed_repetitions: int,
    maximum_absolute_gap: float,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frozen = payload.get("frozen_inputs", {})
    design = payload.get("paired_design", {})
    gate = payload.get("selection_gate", {})
    expected = {
        "base_config_sha256": _sha256(config),
        "reference_ode_outcomes_sha256": _sha256(reference_outcomes),
        "reference_ode_evidence_sha256": _sha256(reference_evidence),
        "noise_levels": noise_levels,
        "environment_seeds": seeds,
        "policy_seed_repetitions_per_environment_seed": policy_seed_repetitions,
        "maximum_absolute_success_rate_gap": maximum_absolute_gap,
    }
    actual = {
        "base_config_sha256": frozen.get("base_config_sha256"),
        "reference_ode_outcomes_sha256": frozen.get("reference_ode_outcomes_sha256"),
        "reference_ode_evidence_sha256": frozen.get("reference_ode_evidence_sha256"),
        "noise_levels": design.get("noise_levels"),
        "environment_seeds": design.get("environment_seeds"),
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
        raise ValueError(f"Noise calibration preregistration mismatch: {mismatches}")
    return payload


def _build_report(
    *,
    config: Path,
    preregistration: Path,
    reference_outcomes: Path,
    reference_evidence: Path,
    conditions: dict[float, dict[str, Path]],
    maximum_absolute_gap: float,
) -> dict[str, Any]:
    _load_completed_rows(reference_outcomes)
    comparisons: dict[str, Any] = {}
    decisions: dict[str, str] = {}
    for noise_level, paths in sorted(conditions.items()):
        _load_completed_rows(paths["outcomes"])
        comparison = _comparison_bundle(reference_outcomes, paths["outcomes"])
        label = _label(noise_level)
        comparisons[label] = comparison
        decisions[label] = _equivalence_decision(
            comparison["reset_clustered"],
            maximum_absolute_gap=maximum_absolute_gap,
        )
    passing = [
        noise for noise in sorted(conditions) if decisions[_label(noise)] == "pass"
    ]
    selected = max(passing) if passing else None
    return {
        "schema_version": 1,
        "kind": "gr00t_n1d7_robocasa_flow_sde_noise_calibration",
        "sealed_eligible": False,
        "development_lift_claim_eligible": False,
        "optimizer_steps": 0,
        "base_config": {"path": str(config.resolve()), "sha256": _sha256(config)},
        "preregistration": {
            "path": str(preregistration.resolve()),
            "sha256": _sha256(preregistration),
        },
        "reference_ode": {
            "outcomes": str(reference_outcomes.resolve()),
            "outcomes_sha256": _sha256(reference_outcomes),
            "evidence": str(reference_evidence.resolve()),
            "evidence_sha256": _sha256(reference_evidence),
        },
        "maximum_absolute_gap": maximum_absolute_gap,
        "decisions": decisions,
        "selected_noise_level": selected,
        "training_admission": selected is not None,
        "selection_rule": "highest tested noise level with an equivalence pass",
        "comparisons": comparisons,
        "conditions": {
            _label(noise): {
                "noise_level": noise,
                "config": str(paths["config"]),
                "config_sha256": _sha256(paths["config"]),
                "outcomes": str(paths["outcomes"]),
                "outcomes_sha256": _sha256(paths["outcomes"]),
                "evidence": str(paths["evidence"]),
                "evidence_sha256": _sha256(paths["evidence"]),
            }
            for noise, paths in sorted(conditions.items())
        },
    }


async def run(args: argparse.Namespace) -> None:
    if args.seed_count < 2 or args.policy_seed_repetitions < 2:
        raise ValueError("Noise calibration requires at least two resets and repeats")
    noise_levels = sorted(set(args.noise_levels))
    if not noise_levels or any(not 0.0 < level < 1.0 for level in noise_levels):
        raise ValueError("--noise-levels must contain unique values in (0, 1)")
    config = args.config.expanduser().resolve()
    preregistration = args.preregistration.expanduser().resolve()
    reference_outcomes = args.reference_ode_outcomes.expanduser().resolve()
    reference_evidence = args.reference_ode_evidence.expanduser().resolve()
    for path in (config, preregistration, reference_outcomes, reference_evidence):
        if not path.is_file():
            raise FileNotFoundError(path)
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    _verify_preregistration(
        preregistration,
        config=config,
        reference_outcomes=reference_outcomes,
        reference_evidence=reference_evidence,
        noise_levels=noise_levels,
        seeds=seeds,
        policy_seed_repetitions=args.policy_seed_repetitions,
        maximum_absolute_gap=args.maximum_absolute_gap,
    )
    base = EmbodiedExperimentConfig.from_yaml(config)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions: dict[float, dict[str, Path]] = {}
    configs: dict[float, EmbodiedExperimentConfig] = {}
    for noise_level in noise_levels:
        label = _label(noise_level)
        paths = _condition_paths(output_dir, label, 0)
        condition = _condition_config(
            base,
            label=label,
            output_dir=output_dir / label,
            seeds=seeds,
            policy_seed_repetitions=args.policy_seed_repetitions,
            sampler="flow-sde",
            candidate=False,
            preregistration=preregistration,
            noise_level=noise_level,
            group="cuttingboard-pan-k4-noise-screen-v1",
        )
        _write_config(condition, paths["config"])
        conditions[noise_level] = paths
        configs[noise_level] = condition

    preflight = {
        "status": "ok",
        "mode": "preflight" if args.preflight else "execution",
        "optimizer_steps": 0,
        "reference_ode_outcomes_sha256": _sha256(reference_outcomes),
        "episodes_per_condition": args.seed_count * args.policy_seed_repetitions,
        "environment_seeds": seeds,
        "policy_seed_repetitions": args.policy_seed_repetitions,
        "conditions": {
            _label(noise): {
                "noise_level": noise,
                "config": str(conditions[noise]["config"]),
                "fingerprint": configs[noise].fingerprint,
            }
            for noise in noise_levels
        },
    }
    (output_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
    if args.preflight:
        return

    if not args.report_only:
        for noise_level in noise_levels:
            paths = conditions[noise_level]
            if paths["outcomes"].is_file() and paths["evidence"].is_file():
                _load_completed_rows(paths["outcomes"])
                print(f"[noise-calibration] reuse {_label(noise_level)}", flush=True)
                continue
            print(f"[noise-calibration] start {_label(noise_level)}", flush=True)
            await train.run(paths["config"], evaluate_only=True, evaluation_step=0)

    report = _build_report(
        config=config,
        preregistration=preregistration,
        reference_outcomes=reference_outcomes,
        reference_evidence=reference_evidence,
        conditions=conditions,
        maximum_absolute_gap=args.maximum_absolute_gap,
    )
    report_path = output_dir / "noise-calibration-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["training_admission"]:
        raise SystemExit(2)


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
