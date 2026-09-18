"""Adjudicate the preregistered GR00T one-task update-100 continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from art_embodied import EmbodiedExperimentConfig
from art_embodied.evaluation import compare_paired_evaluation_reports

PARENT_EVALUATION_UPDATES = tuple(range(0, 21, 2))
CONTINUATION_EVALUATION_UPDATES = tuple(range(25, 101, 5))
EVALUATION_UPDATES = PARENT_EVALUATION_UPDATES + CONTINUATION_EVALUATION_UPDATES
DURABILITY_UPDATES = (90, 95, 100)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _linear_slope(points: list[tuple[int, float]]) -> float:
    x_mean = sum(point[0] for point in points) / len(points)
    y_mean = sum(point[1] for point in points) / len(points)
    denominator = sum((point[0] - x_mean) ** 2 for point in points)
    return sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator


def _evaluation_point(path: Path, update: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("episodes")
    if not isinstance(rows, list) or len(rows) != 64:
        raise ValueError(f"Update {update} must contain exactly 64 episodes")
    completed = sum(bool(row.get("completed")) for row in rows)
    successes = sum(int(float(row.get("success", 0.0))) for row in rows)
    return {
        "update": update,
        "episodes": len(rows),
        "completed_episodes": completed,
        "successes": successes,
        "success_rate": successes / len(rows),
        "outcomes": str(path),
        "outcomes_sha256": _sha256(path),
    }


def _training_point(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = payload.get("groups")
    if not isinstance(groups, list) or len(groups) != 32:
        raise ValueError(f"Rollout evidence must contain exactly 32 groups: {path}")
    trajectories = sum(int(group["completed_trajectories"]) for group in groups)
    failures = sum(int(group["failed_trajectories"]) for group in groups)
    successes = sum(int(group["success_count"]) for group in groups)
    if trajectories != 256 or failures:
        raise ValueError(
            f"Rollout evidence is incomplete: trajectories={trajectories}, "
            f"failures={failures}, path={path}"
        )
    policy_version = int(payload["policy_version"])
    return {
        "policy_version": policy_version,
        "training_update": policy_version + 1,
        "trajectories": trajectories,
        "successes": successes,
        "success_rate": successes / trajectories,
        "mixed_reward_groups": int(payload["summary"]["mixed_reward_groups"]),
        "evidence": str(path),
        "evidence_sha256": _sha256(path),
    }


def adjudicate(
    *,
    config_path: Path,
    preregistration_path: Path,
) -> dict[str, Any]:
    config = EmbodiedExperimentConfig.from_yaml(config_path)
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    frozen = preregistration["frozen_inputs"]
    if frozen["config_sha256"] != _sha256(config_path):
        raise ValueError("The continuation config differs from its preregistration")
    if frozen["resume_contract_fingerprint"] != config.resume_contract_fingerprint:
        raise ValueError("The continuation changed the frozen training contract")
    if config.training.updates != 100 or config.storage.resume_from_checkpoint is None:
        raise ValueError("Adjudication requires the frozen update-20 to update-100 run")

    output_dir = config.storage.output_dir
    evaluation_points = [
        _evaluation_point(
            output_dir / "evaluation" / f"update_{update:06d}_episode_outcomes.json",
            update,
        )
        for update in EVALUATION_UPDATES
    ]
    training_points = [
        _training_point(
            output_dir
            / "rollout-evidence"
            / f"policy-version-{policy_version:06d}.json"
        )
        for policy_version in range(100)
    ]
    baseline = evaluation_points[0]
    endpoint = evaluation_points[-1]
    paired = compare_paired_evaluation_reports(
        baseline["outcomes"],
        endpoint["outcomes"],
    )
    points_by_update = {point["update"]: point for point in evaluation_points}
    durability = [points_by_update[update] for update in DURABILITY_UPDATES]
    durability_above_baseline = sum(
        point["success_rate"] > baseline["success_rate"] for point in durability
    )
    checks = {
        "all_evaluations_complete": all(
            point["completed_episodes"] == 64 for point in evaluation_points
        ),
        "all_100_training_rollouts_complete": len(training_points) == 100,
        "update100_lift_positive": paired["success_rate_lift"] > 0.0,
        "update100_ci95_lower_positive": paired["success_rate_lift_ci95_low"] > 0.0,
        "update100_improved_exceeds_regressed": paired["improved_pairs"]
        > paired["regressed_pairs"],
        "update100_mcnemar_at_most_005": paired["mcnemar_exact_p_value"] <= 0.05,
        "durability_two_of_last_three_above_baseline": durability_above_baseline >= 2,
    }
    evaluation_slope = _linear_slope(
        [(point["update"], point["success_rate"]) for point in evaluation_points]
    )
    training_slope = _linear_slope(
        [(point["training_update"], point["success_rate"]) for point in training_points]
    )
    return {
        "schema_version": 1,
        "kind": "gr00t_n1d7_robocasa_one_task_u100_adjudication",
        "status": "passed" if all(checks.values()) else "rejected",
        "development_lift_established": all(checks.values()),
        "original_eight_task_claim_eligible": False,
        "sealed_eligible": all(checks.values()),
        "selected_checkpoint": (
            str(output_dir / "checkpoints" / "step-000100")
            if all(checks.values())
            else None
        ),
        "config": {
            "path": str(config_path),
            "sha256": _sha256(config_path),
            "fingerprint": config.fingerprint,
            "resume_contract_fingerprint": config.resume_contract_fingerprint,
        },
        "preregistration": {
            "path": str(preregistration_path),
            "sha256": _sha256(preregistration_path),
        },
        "gate_checks": checks,
        "update100_paired": paired,
        "durability": {
            "updates": list(DURABILITY_UPDATES),
            "points_above_baseline": durability_above_baseline,
            "points": durability,
        },
        "evaluation_curve": evaluation_points,
        "training_curve": training_points,
        "health": {
            "native_ode_success_rate_slope_per_update": evaluation_slope,
            "stochastic_training_success_rate_slope_per_update": training_slope,
            "directionally_aligned": evaluation_slope * training_slope > 0.0,
            "interpretation": (
                "Directional alignment is a rule-of-thumb diagnostic, not a gate; "
                "the training and evaluation reset distributions differ."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = adjudicate(
        config_path=args.config,
        preregistration_path=args.preregistration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="", flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
