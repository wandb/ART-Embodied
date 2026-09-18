"""Run a paired ODE/Flow-SDE calibration for GR00T N1.7 on RoboCasa."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import fmean
from typing import Any, Sequence

from art_embodied import EmbodiedExperimentConfig
from art_embodied.evaluation import compare_paired_evaluation_reports

from . import train

_BOOTSTRAP_SAMPLES = 10_000
_BOOTSTRAP_SEED = 20260819


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=400)
    parser.add_argument("--seed-count", type=int, default=16)
    parser.add_argument("--policy-seed-repetitions", type=int, default=8)
    parser.add_argument("--maximum-absolute-gap", type=float, default=0.10)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    marker = resolved / "art_embodied_checkpoint_complete.json"
    adapter = resolved / "policy" / "adapter_model.safetensors"
    if not marker.is_file():
        raise FileNotFoundError(f"Incomplete candidate checkpoint: {marker}")
    if not adapter.is_file():
        raise FileNotFoundError(f"Candidate adapter is missing: {adapter}")
    return {
        "path": str(resolved),
        "checkpoint_complete_sha256": _sha256(marker),
        "adapter_model_sha256": _sha256(adapter),
    }


def _verify_preregistration(
    path: Path,
    *,
    config_path: Path,
    checkpoint_identity: dict[str, Any],
    seeds: list[int],
    policy_seed_repetitions: int,
    maximum_absolute_gap: float,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frozen = payload.get("frozen_inputs", {})
    design = payload.get("paired_design", {})
    gate = payload.get("equivalence_gate", {})
    expected = {
        "base_config_sha256": _sha256(config_path),
        "candidate_checkpoint_complete_sha256": checkpoint_identity[
            "checkpoint_complete_sha256"
        ],
        "candidate_adapter_sha256": checkpoint_identity["adapter_model_sha256"],
    }
    mismatches = {
        key: {"expected": value, "actual": frozen.get(key)}
        for key, value in expected.items()
        if frozen.get(key) != value
    }
    if design.get("environment_seeds") != seeds:
        mismatches["environment_seeds"] = {
            "expected": seeds,
            "actual": design.get("environment_seeds"),
        }
    if design.get("policy_seed_repetitions_per_environment_seed") != (
        policy_seed_repetitions
    ):
        mismatches["policy_seed_repetitions"] = {
            "expected": policy_seed_repetitions,
            "actual": design.get("policy_seed_repetitions_per_environment_seed"),
        }
    if gate.get("maximum_absolute_success_rate_gap") != maximum_absolute_gap:
        mismatches["maximum_absolute_gap"] = {
            "expected": maximum_absolute_gap,
            "actual": gate.get("maximum_absolute_success_rate_gap"),
        }
    if mismatches:
        raise ValueError(f"Sampler calibration preregistration mismatch: {mismatches}")


def _condition_config(
    base: EmbodiedExperimentConfig,
    *,
    label: str,
    output_dir: Path,
    seeds: list[int],
    policy_seed_repetitions: int,
    sampler: str,
    candidate: bool,
    preregistration: Path,
    noise_level: float | None = None,
    project: str = "art-embodied-gr00t-n1d7-robocasa-sampler-calibration",
    group: str = "cuttingboard-pan-k4-noise05-v1",
) -> EmbodiedExperimentConfig:
    raw = base.model_dump(mode="json")
    stochastic = sampler == "flow-sde"
    if noise_level is not None:
        raw["algorithm"]["flow_sde"]["noise_level"] = noise_level
    raw["experiment"].update(
        {
            "project": project,
            "run": f"gr00t-n1d7-robocasa-sampler-calibration-{label}",
            "tags": [
                "gr00t-n1d7",
                "robocasa",
                "sampler-calibration",
                "diagnostic-only",
                label,
            ],
        }
    )
    raw["policy"]["evaluation_generation"]["do_sample"] = stochastic
    raw["evaluation"].update(
        {
            "evaluate_before_training": False,
            "pre_training_success_gate": None,
            "split": "train_matched",
            "data_role": "diagnostic",
            "baseline_outcomes_path": None,
            "episodes": len(seeds) * policy_seed_repetitions,
            "seeds": seeds,
            "deterministic": not stochastic,
            "checkpoint_selection": "last",
            "kwargs": {
                "action_sampling": (
                    "gaussian_flow_sde" if stochastic else "native_flow_ode"
                ),
                "require_policy_checkpoint": candidate,
                "sampler_calibration_preregistration": str(
                    preregistration.expanduser().resolve()
                ),
                "seed_contract": {
                    "environment_mode": "configured",
                    "fixed_environment_seed": 0,
                    "policy_mode": "derived",
                    "fixed_policy_seed": 0,
                },
            },
        }
    )
    wandb = raw["observability"]["wandb"]
    wandb.update(
        {
            "project": project,
            "group": group,
            "job_type": "sampler-calibration",
            "run_id": None,
            "resume": None,
            "log_model_artifacts": False,
            "max_evaluation_table_rows": len(seeds) * policy_seed_repetitions,
            "evaluation_progress_every_episodes": len(seeds),
        }
    )
    raw["observability"]["weave"]["enabled"] = False
    raw["observability"].update(
        {
            "videos_per_update": 0,
            "videos_per_evaluation": 4,
            "require_train_video": False,
            "require_evaluation_video": True,
        }
    )
    raw["storage"].update(
        {
            "output_dir": str(output_dir),
            "resume_from_checkpoint": None,
            "retain_checkpoint_updates": [],
        }
    )
    return EmbodiedExperimentConfig.model_validate(raw)


def _condition_paths(output_dir: Path, label: str, step: int) -> dict[str, Path]:
    root = output_dir / label
    stem = root / "evaluation" / f"update_{step:06d}"
    return {
        "config": output_dir / "configs" / f"{label}.yaml",
        "outcomes": stem.with_name(stem.name + "_episode_outcomes.json"),
        "evidence": stem.with_name(stem.name + "_evidence.json"),
    }


def _write_config(config: EmbodiedExperimentConfig, path: Path) -> None:
    encoded = config.to_yaml(path).read_bytes()
    reparsed = EmbodiedExperimentConfig.from_yaml(path)
    if reparsed.model_dump(mode="json") != config.model_dump(mode="json"):
        raise RuntimeError(f"Generated config does not round-trip: {path}")
    if path.read_bytes() != encoded:
        raise RuntimeError(f"Generated config changed during verification: {path}")


def _load_completed_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("episodes")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Evaluation outcomes contain no episodes: {path}")
    failures = [row for row in rows if not row.get("completed")]
    if failures:
        raise RuntimeError(
            f"Sampler calibration is invalid because {len(failures)} episodes failed: "
            f"{path}"
        )
    return rows


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _clustered_comparison(
    baseline_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    baseline = _load_completed_rows(baseline_path)
    candidate = _load_completed_rows(candidate_path)
    if len(baseline) != len(candidate):
        raise ValueError("Clustered paired reports have different episode counts")
    differences_by_reset: dict[int, list[float]] = {}
    fields = ("episode", "scenario_id", "environment_seed", "policy_seed")
    for left, right in zip(baseline, candidate, strict=True):
        if tuple(left.get(key) for key in fields) != tuple(
            right.get(key) for key in fields
        ):
            raise ValueError("Clustered paired reports use different episode plans")
        environment_seed = int(left["environment_seed"])
        differences_by_reset.setdefault(environment_seed, []).append(
            float(right["success"]) - float(left["success"])
        )
    cluster_means = {
        seed: fmean(values) for seed, values in sorted(differences_by_reset.items())
    }
    generator = random.Random(_BOOTSTRAP_SEED)
    means = list(cluster_means.values())
    estimates = [
        fmean(means[generator.randrange(len(means))] for _ in means)
        for _ in range(_BOOTSTRAP_SAMPLES)
    ]
    return {
        "reset_clusters": len(means),
        "episodes_per_reset": sorted({len(v) for v in differences_by_reset.values()}),
        "success_rate_difference": fmean(means),
        "cluster_bootstrap_ci95_low": _percentile(estimates, 0.025),
        "cluster_bootstrap_ci95_high": _percentile(estimates, 0.975),
        "cluster_bootstrap_samples": _BOOTSTRAP_SAMPLES,
        "cluster_bootstrap_seed": _BOOTSTRAP_SEED,
        "per_environment_seed_difference": {
            str(seed): value for seed, value in cluster_means.items()
        },
    }


def _equivalence_decision(
    clustered: dict[str, Any],
    *,
    maximum_absolute_gap: float,
) -> str:
    point = float(clustered["success_rate_difference"])
    low = float(clustered["cluster_bootstrap_ci95_low"])
    high = float(clustered["cluster_bootstrap_ci95_high"])
    if (
        abs(point) <= maximum_absolute_gap
        and low >= -maximum_absolute_gap
        and high <= maximum_absolute_gap
    ):
        return "pass"
    if (
        high < -maximum_absolute_gap
        or low > maximum_absolute_gap
        or abs(point) >= 2.0 * maximum_absolute_gap
    ):
        return "reject"
    return "inconclusive"


def _comparison_bundle(baseline: Path, candidate: Path) -> dict[str, Any]:
    return {
        "episode_paired": compare_paired_evaluation_reports(baseline, candidate),
        "reset_clustered": _clustered_comparison(baseline, candidate),
    }


def _condition_evidence(
    condition_paths: dict[str, dict[str, Path]],
) -> dict[str, Any]:
    return {
        label: {
            "config": str(paths["config"]),
            "config_sha256": _sha256(paths["config"]),
            "outcomes": str(paths["outcomes"]),
            "outcomes_sha256": _sha256(paths["outcomes"]),
            "evidence": str(paths["evidence"]),
            "evidence_sha256": _sha256(paths["evidence"]),
        }
        for label, paths in condition_paths.items()
    }


def _build_baseline_report(
    *,
    base_config_path: Path,
    checkpoint_identity: dict[str, Any],
    preregistration: Path,
    maximum_absolute_gap: float,
    condition_paths: dict[str, dict[str, Path]],
) -> dict[str, Any]:
    baseline_paths = {
        label: condition_paths[label] for label in ("baseline-ode", "baseline-sde")
    }
    for paths in baseline_paths.values():
        _load_completed_rows(paths["outcomes"])
    comparison = _comparison_bundle(
        baseline_paths["baseline-ode"]["outcomes"],
        baseline_paths["baseline-sde"]["outcomes"],
    )
    decision = _equivalence_decision(
        comparison["reset_clustered"],
        maximum_absolute_gap=maximum_absolute_gap,
    )
    return {
        "schema_version": 1,
        "kind": "gr00t_n1d7_robocasa_ode_sde_sampler_calibration",
        "sealed_eligible": False,
        "development_lift_claim_eligible": False,
        "base_config": {
            "path": str(base_config_path.expanduser().resolve()),
            "sha256": _sha256(base_config_path),
        },
        "candidate_checkpoint": checkpoint_identity,
        "preregistration": {
            "path": str(preregistration.expanduser().resolve()),
            "sha256": _sha256(preregistration),
        },
        "maximum_absolute_gap": maximum_absolute_gap,
        "baseline_sampler_equivalence_decision": decision,
        "training_admission": decision == "pass",
        "learning_trend_interpretation": (
            "prohibited: the candidate has only five completed updates; candidate "
            "comparisons diagnose sampler stability, not learning or generalization"
        ),
        "comparisons": {"baseline_sde_minus_ode": comparison},
        "conditions": _condition_evidence(baseline_paths),
    }


def _build_report(
    *,
    base_config_path: Path,
    checkpoint_identity: dict[str, Any],
    preregistration: Path,
    maximum_absolute_gap: float,
    condition_paths: dict[str, dict[str, Path]],
) -> dict[str, Any]:
    outcomes = {label: paths["outcomes"] for label, paths in condition_paths.items()}
    for path in outcomes.values():
        _load_completed_rows(path)
    baseline_report = _build_baseline_report(
        base_config_path=base_config_path,
        checkpoint_identity=checkpoint_identity,
        preregistration=preregistration,
        maximum_absolute_gap=maximum_absolute_gap,
        condition_paths=condition_paths,
    )
    baseline_protocol = baseline_report["comparisons"]["baseline_sde_minus_ode"]
    candidate_protocol = _comparison_bundle(
        outcomes["candidate-ode"], outcomes["candidate-sde"]
    )
    baseline_decision = _equivalence_decision(
        baseline_protocol["reset_clustered"],
        maximum_absolute_gap=maximum_absolute_gap,
    )
    return {
        **baseline_report,
        "baseline_sampler_equivalence_decision": baseline_decision,
        "training_admission": baseline_decision == "pass",
        "comparisons": {
            "baseline_sde_minus_ode": baseline_protocol,
            "candidate_sde_minus_ode": candidate_protocol,
            "candidate_minus_baseline_under_ode": _comparison_bundle(
                outcomes["baseline-ode"], outcomes["candidate-ode"]
            ),
            "candidate_minus_baseline_under_sde": _comparison_bundle(
                outcomes["baseline-sde"], outcomes["candidate-sde"]
            ),
        },
        "conditions": _condition_evidence(condition_paths),
    }


def _write_report(report: dict[str, Any], output_dir: Path) -> Path:
    report_path = output_dir / "sampler-calibration-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report_path


async def run(args: argparse.Namespace) -> None:
    if args.seed_count < 2:
        raise ValueError("--seed-count must be at least 2")
    if args.policy_seed_repetitions < 2:
        raise ValueError("--policy-seed-repetitions must be at least 2")
    if not 0.0 < args.maximum_absolute_gap < 1.0:
        raise ValueError("--maximum-absolute-gap must be in (0, 1)")
    preregistration = args.preregistration.expanduser().resolve()
    if not preregistration.is_file():
        raise FileNotFoundError(f"Missing preregistration: {preregistration}")
    checkpoint_identity = _checkpoint_identity(args.candidate_checkpoint)
    base = EmbodiedExperimentConfig.from_yaml(args.config)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    _verify_preregistration(
        preregistration,
        config_path=args.config,
        checkpoint_identity=checkpoint_identity,
        seeds=seeds,
        policy_seed_repetitions=args.policy_seed_repetitions,
        maximum_absolute_gap=args.maximum_absolute_gap,
    )
    definitions = (
        ("baseline-ode", "native-ode", False, 0),
        ("baseline-sde", "flow-sde", False, 0),
        ("candidate-ode", "native-ode", True, 5),
        ("candidate-sde", "flow-sde", True, 5),
    )
    paths_by_label: dict[str, dict[str, Path]] = {}
    configs: dict[str, EmbodiedExperimentConfig] = {}
    for label, sampler, candidate, step in definitions:
        paths = _condition_paths(output_dir, label, step)
        config = _condition_config(
            base,
            label=label,
            output_dir=output_dir / label,
            seeds=seeds,
            policy_seed_repetitions=args.policy_seed_repetitions,
            sampler=sampler,
            candidate=candidate,
            preregistration=preregistration,
        )
        _write_config(config, paths["config"])
        paths_by_label[label] = paths
        configs[label] = config

    preflight = {
        "status": "ok",
        "mode": "preflight" if args.preflight else "execution",
        "episodes_per_condition": args.seed_count * args.policy_seed_repetitions,
        "environment_seeds": seeds,
        "policy_seed_repetitions": args.policy_seed_repetitions,
        "maximum_absolute_gap": args.maximum_absolute_gap,
        "checkpoint": checkpoint_identity,
        "conditions": {
            label: {
                "config": str(paths_by_label[label]["config"]),
                "fingerprint": configs[label].fingerprint,
                "sampler": sampler,
                "candidate": candidate,
                "step": step,
            }
            for label, sampler, candidate, step in definitions
        },
    }
    preflight_path = output_dir / "preflight.json"
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
    if args.preflight:
        return

    if not args.report_only:
        for label, _sampler, candidate, step in definitions:
            paths = paths_by_label[label]
            if paths["outcomes"].is_file() and paths["evidence"].is_file():
                _load_completed_rows(paths["outcomes"])
                print(f"[sampler-calibration] reuse completed {label}", flush=True)
                continue
            print(f"[sampler-calibration] start {label}", flush=True)
            await train.run(
                paths["config"],
                evaluate_only=True,
                evaluation_step=step,
                policy_checkpoint=(
                    Path(checkpoint_identity["path"]) if candidate else None
                ),
            )
            if label == "baseline-sde":
                baseline_report = _build_baseline_report(
                    base_config_path=args.config,
                    checkpoint_identity=checkpoint_identity,
                    preregistration=preregistration,
                    maximum_absolute_gap=args.maximum_absolute_gap,
                    condition_paths=paths_by_label,
                )
                if baseline_report["baseline_sampler_equivalence_decision"] == (
                    "reject"
                ):
                    baseline_report["candidate_conditions_skipped"] = (
                        "Baseline sampler equivalence was rejected; candidate "
                        "conditions cannot alter training admission."
                    )
                    _write_report(baseline_report, output_dir)
                    raise SystemExit(2)

    candidate_complete = all(
        paths_by_label[label]["outcomes"].is_file()
        and paths_by_label[label]["evidence"].is_file()
        for label in ("candidate-ode", "candidate-sde")
    )
    if not candidate_complete:
        baseline_report = _build_baseline_report(
            base_config_path=args.config,
            checkpoint_identity=checkpoint_identity,
            preregistration=preregistration,
            maximum_absolute_gap=args.maximum_absolute_gap,
            condition_paths=paths_by_label,
        )
        if baseline_report["baseline_sampler_equivalence_decision"] != "reject":
            raise RuntimeError(
                "Candidate conditions are missing but baseline calibration did not "
                "produce a preregistered rejection"
            )
        baseline_report["candidate_conditions_skipped"] = (
            "Baseline sampler equivalence was rejected; candidate conditions cannot "
            "alter training admission."
        )
        _write_report(baseline_report, output_dir)
        raise SystemExit(2)

    report = _build_report(
        base_config_path=args.config,
        checkpoint_identity=checkpoint_identity,
        preregistration=preregistration,
        maximum_absolute_gap=args.maximum_absolute_gap,
        condition_paths=paths_by_label,
    )
    _write_report(report, output_dir)
    if report["baseline_sampler_equivalence_decision"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
