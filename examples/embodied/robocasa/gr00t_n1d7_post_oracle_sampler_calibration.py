"""Revalidate the admitted GR00T sampler after official-parity fixes."""

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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--noise-level", type=float, default=0.1)
    parser.add_argument("--seed-start", type=int, default=400)
    parser.add_argument("--seed-count", type=int, default=64)
    parser.add_argument("--policy-seed-repetitions", type=int, default=8)
    parser.add_argument("--maximum-absolute-gap", type=float, default=0.10)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def _required_evidence_paths(repository_root: Path) -> dict[str, Path]:
    return {
        "oracle_manifest_sha256": repository_root
        / "examples/embodied/robocasa/oracle_parity_manifest.json",
        "oracle_frontier8_evidence_sha256": repository_root
        / "outputs/conformance/gr00t-n17-robocasa-frontier8-oracle-evidence-20260819.json",
        "art_oracle_footprint_outcomes_sha256": repository_root
        / "outputs/gr00t-n1d7-robocasa-development/"
        "frontier8-oracle-footprint-sft-eval/evaluation/"
        "update_000000_episode_outcomes.json",
        "art_oracle_footprint_evidence_sha256": repository_root
        / "outputs/gr00t-n1d7-robocasa-development/"
        "frontier8-oracle-footprint-sft-eval/evaluation/"
        "update_000000_evidence.json",
        "sft_completion_marker_sha256": repository_root
        / "outputs/gr00t-n1d7-robocasa-sft/"
        "gr00t-n1d7-robocasa-gr1-tabletop-sft-u60000/"
        "checkpoint-60000/art_embodied_sft_complete.json",
    }


def _verify_preregistration(
    path: Path,
    *,
    config: Path,
    noise_level: float,
    seed_start: int,
    seed_count: int,
    repetitions: int,
    maximum_absolute_gap: float,
    repository_root: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frozen = payload.get("frozen_inputs", {})
    design = payload.get("paired_design", {})
    gate = payload.get("equivalence_gate", {})
    expected: dict[str, Any] = {
        "base_config_sha256": _sha256(config),
        "noise_level": noise_level,
        "denoise_steps": 4,
        "execution_horizon": 8,
        "maximum_environment_steps": 720,
        "environment_seed_start": seed_start,
        "environment_seed_count": seed_count,
        "policy_seed_repetitions_per_environment_seed": repetitions,
        "episodes_per_condition": seed_count * repetitions,
        "maximum_absolute_success_rate_gap": maximum_absolute_gap,
    }
    actual: dict[str, Any] = {
        "base_config_sha256": frozen.get("base_config_sha256"),
        "noise_level": frozen.get("noise_level"),
        "denoise_steps": frozen.get("denoise_steps"),
        "execution_horizon": frozen.get("execution_horizon"),
        "maximum_environment_steps": frozen.get("maximum_environment_steps"),
        "environment_seed_start": design.get("environment_seed_start"),
        "environment_seed_count": design.get("environment_seed_count"),
        "policy_seed_repetitions_per_environment_seed": design.get(
            "policy_seed_repetitions_per_environment_seed"
        ),
        "episodes_per_condition": design.get("episodes_per_condition"),
        "maximum_absolute_success_rate_gap": gate.get(
            "maximum_absolute_success_rate_gap"
        ),
    }
    evidence = _required_evidence_paths(repository_root)
    for key, evidence_path in evidence.items():
        if not evidence_path.is_file():
            raise FileNotFoundError(
                f"Missing frozen admission evidence: {evidence_path}"
            )
        expected[key] = _sha256(evidence_path)
        actual[key] = frozen.get(key)
    implementation = frozen.get("implementation_sha256", {})
    if not isinstance(implementation, dict) or not implementation:
        raise ValueError("Preregistration must freeze implementation_sha256")
    for relative, digest in implementation.items():
        source = repository_root / relative
        key = f"implementation_sha256.{relative}"
        if not source.is_file():
            raise FileNotFoundError(f"Missing frozen implementation source: {source}")
        expected[key] = _sha256(source)
        actual[key] = digest
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Post-oracle sampler preregistration mismatch: {mismatches}")
    return payload


def _without_high_volume_observability(
    config: EmbodiedExperimentConfig,
) -> EmbodiedExperimentConfig:
    raw = config.model_dump(mode="json")
    raw["observability"].update(
        {
            "videos_per_update": 0,
            "videos_per_evaluation": 0,
            "require_train_video": False,
            "require_evaluation_video": False,
        }
    )
    raw["observability"]["lookahead_preview"].update(
        {"videos_per_update": 0, "videos_per_evaluation": 0}
    )
    raw["observability"]["weave"].update(
        {
            "enabled": False,
            "trace_trajectories": False,
            "max_groups_per_update": 0,
            "max_trajectories_per_group": 0,
            "max_evaluation_trajectories": 0,
        }
    )
    return EmbodiedExperimentConfig.model_validate(raw)


def _condition_evidence(paths: dict[str, dict[str, Path]]) -> dict[str, Any]:
    return {
        label: {
            "config": str(items["config"]),
            "config_sha256": _sha256(items["config"]),
            "outcomes": str(items["outcomes"]),
            "outcomes_sha256": _sha256(items["outcomes"]),
            "evidence": str(items["evidence"]),
            "evidence_sha256": _sha256(items["evidence"]),
        }
        for label, items in paths.items()
    }


async def run(args: argparse.Namespace) -> None:
    if args.seed_count < 2 or args.policy_seed_repetitions < 2:
        raise ValueError("Calibration requires at least two reset and policy seeds")
    if not 0.0 < args.noise_level < 1.0:
        raise ValueError("--noise-level must be in (0, 1)")
    if not 0.0 < args.maximum_absolute_gap < 1.0:
        raise ValueError("--maximum-absolute-gap must be in (0, 1)")
    config_path = args.config.expanduser().resolve()
    preregistration = args.preregistration.expanduser().resolve()
    for required in (config_path, preregistration):
        if not required.is_file():
            raise FileNotFoundError(required)
    repository_root = Path(__file__).parents[3].resolve()
    registration = _verify_preregistration(
        preregistration,
        config=config_path,
        noise_level=args.noise_level,
        seed_start=args.seed_start,
        seed_count=args.seed_count,
        repetitions=args.policy_seed_repetitions,
        maximum_absolute_gap=args.maximum_absolute_gap,
        repository_root=repository_root,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base = EmbodiedExperimentConfig.from_yaml(config_path)
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    definitions = (("post-oracle-ode", "native-ode"), ("post-oracle-sde", "flow-sde"))
    paths_by_label: dict[str, dict[str, Path]] = {}
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
            project="art-embodied-gr00t-n1d7-robocasa-sampler-calibration",
            group="cuttingboard-pan-k4-noise01-post-oracle-v1",
        )
        condition = _without_high_volume_observability(condition)
        _write_config(condition, paths["config"])
        paths_by_label[label] = paths
        configs[label] = condition
    preflight = {
        "status": "ok",
        "mode": "preflight" if args.preflight else "execution",
        "optimizer_steps": 0,
        "noise_level": args.noise_level,
        "environment_seeds": seeds,
        "policy_seed_repetitions": args.policy_seed_repetitions,
        "episodes_per_condition": args.seed_count * args.policy_seed_repetitions,
        "maximum_absolute_gap": args.maximum_absolute_gap,
        "preregistration": {
            "path": str(preregistration),
            "sha256": _sha256(preregistration),
            "name": registration.get("name"),
        },
        "conditions": {
            label: {
                "config": str(paths_by_label[label]["config"]),
                "fingerprint": configs[label].fingerprint,
                "sampler": sampler,
            }
            for label, sampler in definitions
        },
    }
    (output_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
    if args.preflight:
        return
    if not args.report_only:
        for label, _sampler in definitions:
            paths = paths_by_label[label]
            if paths["outcomes"].is_file() and paths["evidence"].is_file():
                _load_completed_rows(paths["outcomes"])
                print(f"[post-oracle-calibration] reuse {label}", flush=True)
                continue
            print(f"[post-oracle-calibration] start {label}", flush=True)
            await train.run(paths["config"], evaluate_only=True, evaluation_step=0)
    for paths in paths_by_label.values():
        _load_completed_rows(paths["outcomes"])
    comparison = _comparison_bundle(
        paths_by_label["post-oracle-ode"]["outcomes"],
        paths_by_label["post-oracle-sde"]["outcomes"],
    )
    decision = _equivalence_decision(
        comparison["reset_clustered"],
        maximum_absolute_gap=args.maximum_absolute_gap,
    )
    report = {
        "schema_version": 1,
        "kind": "gr00t_n1d7_robocasa_post_oracle_sampler_calibration",
        "sealed_eligible": False,
        "development_lift_claim_eligible": False,
        "optimizer_steps": 0,
        "noise_level": args.noise_level,
        "decision": decision,
        "training_admission": decision == "pass",
        "comparison": comparison,
        "conditions": _condition_evidence(paths_by_label),
        "preregistration": preflight["preregistration"],
    }
    report_path = output_dir / "post-oracle-sampler-calibration-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if decision != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
