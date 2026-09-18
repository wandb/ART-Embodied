"""Plan and adjudicate an independent paired development replication."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import comb, exp, lgamma, log
from pathlib import Path
from typing import Any


def exact_mcnemar_power(
    episodes: int,
    *,
    improved_probability: float,
    regressed_probability: float,
    alpha: float = 0.05,
) -> float:
    """Return exact two-sided McNemar power under a multinomial alternative."""

    if episodes < 1:
        raise ValueError("episodes must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if improved_probability < 0 or regressed_probability < 0:
        raise ValueError("discordant outcome probabilities cannot be negative")
    discordant_probability = improved_probability + regressed_probability
    if not 0 < discordant_probability <= 1:
        raise ValueError("discordant outcome probabilities must sum to (0, 1]")
    conditional_improvement = improved_probability / discordant_probability

    power = 0.0
    for discordant in range(episodes + 1):
        discordant_mass = _binomial_pmf(
            discordant,
            episodes,
            discordant_probability,
        )
        for improved in range(discordant // 2 + 1, discordant + 1):
            regressed = discordant - improved
            if _mcnemar_exact_p_value(improved, regressed) > alpha:
                continue
            power += discordant_mass * _binomial_pmf(
                improved,
                discordant,
                conditional_improvement,
            )
    return power


def minimum_balanced_episode_count(
    *,
    improved_probability: float,
    regressed_probability: float,
    task_count: int,
    target_power: float = 0.8,
    alpha: float = 0.05,
    minimum_episodes: int = 1,
    maximum_episodes: int = 4096,
) -> tuple[int, float]:
    """Find the smallest task-balanced panel meeting exact McNemar power."""

    if task_count < 1:
        raise ValueError("task_count must be positive")
    if not 0 < target_power < 1:
        raise ValueError("target_power must be between zero and one")
    start = max(task_count, minimum_episodes)
    start += (-start) % task_count
    for episodes in range(start, maximum_episodes + 1, task_count):
        power = exact_mcnemar_power(
            episodes,
            improved_probability=improved_probability,
            regressed_probability=regressed_probability,
            alpha=alpha,
        )
        if power >= target_power:
            return episodes, power
    raise ValueError(
        "No task-balanced panel reaches the requested power within "
        f"{maximum_episodes} episodes"
    )


def plan_from_evidence(
    evidence_path: Path,
    *,
    task_count: int,
    target_power: float,
    alpha: float,
) -> dict[str, Any]:
    """Derive a replication panel from an already selected development result."""

    evidence = _read_json(evidence_path)
    metrics = evidence.get("metrics", {})
    completed = _required_int(metrics, "completed_episodes")
    improved = _required_int(metrics, "paired/improved_pairs")
    regressed = _required_int(metrics, "paired/regressed_pairs")
    improved_probability = improved / completed
    regressed_probability = regressed / completed
    minimum_episodes, achieved_power = minimum_balanced_episode_count(
        improved_probability=improved_probability,
        regressed_probability=regressed_probability,
        task_count=task_count,
        target_power=target_power,
        alpha=alpha,
        minimum_episodes=completed,
    )
    return {
        "schema_version": 1,
        "source_evidence": str(evidence_path),
        "source_evidence_sha256": _sha256(evidence_path),
        "source_completed_episodes": completed,
        "source_improved_pairs": improved,
        "source_regressed_pairs": regressed,
        "assumed_improved_probability": improved_probability,
        "assumed_regressed_probability": regressed_probability,
        "alpha": alpha,
        "target_power": target_power,
        "task_count": task_count,
        "minimum_balanced_episodes": minimum_episodes,
        "minimum_balanced_power": achieved_power,
    }


def adjudicate_replication(
    *,
    manifest_path: Path,
    baseline_evidence_path: Path,
    candidate_evidence_path: Path,
) -> dict[str, Any]:
    """Apply the preregistered gate to independent replication evidence only."""

    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "art_embodied_development_replication_manifest"
    ):
        raise ValueError(f"Invalid development replication manifest: {manifest_path}")
    baseline = _read_json(baseline_evidence_path)
    candidate = _read_json(candidate_evidence_path)
    baseline_episodes = _verified_outcomes(baseline_evidence_path, baseline)
    candidate_episodes = _verified_outcomes(candidate_evidence_path, candidate)
    expected_manifest_sha256 = _sha256(manifest_path)
    for label, evidence in (("baseline", baseline), ("candidate", candidate)):
        identity = evidence.get("identity", {}).get("evaluation_manifest")
        if not isinstance(identity, dict) or identity.get("sha256") != (
            expected_manifest_sha256
        ):
            raise ValueError(
                f"{label} evidence is not bound to the preregistered manifest"
            )

    baseline_outcomes = baseline.get("outcomes", {})
    candidate_baseline = candidate.get("baseline_outcomes", {})
    if candidate_baseline.get("sha256") != baseline_outcomes.get("sha256"):
        raise ValueError("Candidate evidence does not reference the measured baseline")

    checkpoint = (
        candidate.get("identity", {}).get("policy", {}).get("evaluated_state", {})
    )
    checkpoint_artifact = checkpoint.get("artifact", {})
    expected_checkpoint = manifest.get("candidate", {}).get("policy_snapshot_sha256")
    if (
        checkpoint.get("status") != "checkpoint"
        or checkpoint_artifact.get("sha256") != expected_checkpoint
    ):
        raise ValueError("Candidate evidence does not identify the frozen checkpoint")

    episodes = int(manifest["episodes"])
    if len(baseline_episodes) != episodes or len(candidate_episodes) != episodes:
        raise ValueError(
            "Outcome files do not contain the preregistered episode count: "
            f"baseline={len(baseline_episodes)}, "
            f"candidate={len(candidate_episodes)}, expected={episodes}"
        )
    metrics = candidate.get("metrics", {})
    task_metrics = _paired_task_metrics(baseline_episodes, candidate_episodes)
    computed_improved = sum(row["improved_pairs"] for row in task_metrics)
    computed_regressed = sum(row["regressed_pairs"] for row in task_metrics)
    if computed_improved != _required_int(
        metrics, "paired/improved_pairs"
    ) or computed_regressed != _required_int(metrics, "paired/regressed_pairs"):
        raise ValueError("Candidate paired metrics do not match the outcome files")
    gate_checks = {
        "baseline_completed": (
            _required_int(baseline.get("metrics", {}), "completed_episodes") == episodes
            and _required_int(baseline.get("metrics", {}), "failed_episodes") == 0
        ),
        "candidate_completed": (
            _required_int(metrics, "completed_episodes") == episodes
            and _required_int(metrics, "failed_episodes") == 0
        ),
        "positive_lift": _required_float(metrics, "paired/success_rate_lift") > 0,
        "ci95_lower_bound_positive": (
            _required_float(metrics, "paired/success_rate_lift_ci95_low") > 0
        ),
        "improved_pairs_exceed_regressed": (
            _required_int(metrics, "paired/improved_pairs")
            > _required_int(metrics, "paired/regressed_pairs")
        ),
        "mcnemar_exact_p_at_most_0_05": (
            _required_float(metrics, "paired/mcnemar_exact_p_value") <= 0.05
        ),
    }
    return {
        "schema_version": 1,
        "kind": "art_embodied_development_replication_adjudication",
        "manifest": str(manifest_path),
        "manifest_sha256": expected_manifest_sha256,
        "baseline_evidence": str(baseline_evidence_path),
        "candidate_evidence": str(candidate_evidence_path),
        "replication_metrics": {
            key: metrics[key]
            for key in (
                "paired/baseline_success_count",
                "paired/candidate_success_count",
                "paired/success_rate_lift",
                "paired/success_rate_lift_ci95_low",
                "paired/success_rate_lift_ci95_high",
                "paired/improved_pairs",
                "paired/regressed_pairs",
                "paired/mcnemar_exact_p_value",
            )
        },
        "task_metrics": task_metrics,
        "gate_checks": gate_checks,
        "sealed_eligible": all(gate_checks.values()),
    }


def _verified_outcomes(
    evidence_path: Path,
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    record = evidence.get("outcomes", {})
    raw_path = record.get("path")
    expected_sha256 = record.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(expected_sha256, str):
        raise ValueError(f"Evidence has no content-addressed outcomes: {evidence_path}")
    configured = Path(raw_path)
    candidates = (
        (configured,)
        if configured.is_absolute()
        else (evidence_path.parent / configured, configured)
    )
    outcome_path = next((path for path in candidates if path.is_file()), None)
    if outcome_path is None:
        raise ValueError(f"Evidence outcome file does not exist: {raw_path}")
    if _sha256(outcome_path) != expected_sha256:
        raise ValueError(f"Evidence outcome SHA does not match: {outcome_path}")
    payload = _read_json(outcome_path)
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not all(
        isinstance(episode, dict) for episode in episodes
    ):
        raise ValueError(f"Outcome file has no episode records: {outcome_path}")
    return episodes


def _paired_task_metrics(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    task_pairs: dict[str, list[tuple[int, int]]] = {}
    for baseline_episode, candidate_episode in zip(baseline, candidate, strict=True):
        identity_keys = ("episode", "scenario_id", "environment_seed", "policy_seed")
        if any(
            baseline_episode.get(key) != candidate_episode.get(key)
            for key in identity_keys
        ):
            raise ValueError("Baseline and candidate outcome episodes are not paired")
        task = baseline_episode.get("task")
        if not isinstance(task, str) or candidate_episode.get("task") != task:
            raise ValueError("Baseline and candidate outcome tasks are not paired")
        baseline_success = int(bool(baseline_episode.get("success")))
        candidate_success = int(bool(candidate_episode.get("success")))
        task_pairs.setdefault(task, []).append((baseline_success, candidate_success))

    rows: list[dict[str, Any]] = []
    for task, pairs in sorted(task_pairs.items()):
        baseline_successes = sum(item[0] for item in pairs)
        candidate_successes = sum(item[1] for item in pairs)
        improved = sum(item == (0, 1) for item in pairs)
        regressed = sum(item == (1, 0) for item in pairs)
        rows.append(
            {
                "task": task,
                "episodes": len(pairs),
                "baseline_successes": baseline_successes,
                "candidate_successes": candidate_successes,
                "success_rate_lift": (
                    (candidate_successes - baseline_successes) / len(pairs)
                ),
                "improved_pairs": improved,
                "regressed_pairs": regressed,
                "mcnemar_exact_p_value": _mcnemar_exact_p_value(
                    improved,
                    regressed,
                ),
            }
        )
    return rows


def _mcnemar_exact_p_value(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    tail = min(improved, regressed)
    probability = sum(comb(discordant, index) for index in range(tail + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * probability)


def _binomial_pmf(successes: int, trials: int, probability: float) -> float:
    if probability == 0:
        return float(successes == 0)
    if probability == 1:
        return float(successes == trials)
    log_probability = (
        lgamma(trials + 1)
        - lgamma(successes + 1)
        - lgamma(trials - successes + 1)
        + successes * log(probability)
        + (trials - successes) * log(1 - probability)
    )
    return exp(log_probability)


def _required_int(values: dict[str, Any], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"Evidence metric is missing or non-numeric: {key}")
    return int(value)


def _required_float(values: dict[str, Any], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"Evidence metric is missing or non-numeric: {key}")
    return float(value)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--evidence", type=Path, required=True)
    plan.add_argument("--task-count", type=int, required=True)
    plan.add_argument("--target-power", type=float, default=0.8)
    plan.add_argument("--alpha", type=float, default=0.05)
    plan.add_argument("--output", type=Path)
    adjudicate = subparsers.add_parser("adjudicate")
    adjudicate.add_argument("--manifest", type=Path, required=True)
    adjudicate.add_argument("--baseline-evidence", type=Path, required=True)
    adjudicate.add_argument("--candidate-evidence", type=Path, required=True)
    adjudicate.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "plan":
        report = plan_from_evidence(
            args.evidence,
            task_count=args.task_count,
            target_power=args.target_power,
            alpha=args.alpha,
        )
    else:
        report = adjudicate_replication(
            manifest_path=args.manifest,
            baseline_evidence_path=args.baseline_evidence,
            candidate_evidence_path=args.candidate_evidence,
        )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(args.output)
    print(encoded, end="")


if __name__ == "__main__":
    main()
