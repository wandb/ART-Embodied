"""Adjudicate the frozen paired sealed test for the update-100 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from art_embodied.checkpointing import CheckpointManager
from art_embodied.evaluation import compare_paired_evaluation_reports


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_outcomes(
    path: Path,
    *,
    expected_step: int,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload.get("episodes")
    expected_episodes = int(manifest["episodes"])
    expected_scenario = manifest["fixed_scenarios"][0]
    expected_seeds = list(manifest["seeds"])
    if (
        payload.get("schema_version") != 1
        or payload.get("data_role") != manifest["data_role"]
        or payload.get("split") != manifest["split"]
        or payload.get("step") != expected_step
        or not isinstance(episodes, list)
        or len(episodes) != expected_episodes
    ):
        raise ValueError(f"Invalid sealed outcomes: {path}")
    for index, (row, expected_seed) in enumerate(
        zip(episodes, expected_seeds, strict=True)
    ):
        success = row.get("success")
        if (
            row.get("episode") != index
            or row.get("scenario_id") != expected_scenario
            or row.get("environment_seed") != expected_seed
            or not isinstance(row.get("policy_seed"), int)
            or row.get("completed") is not True
            or bool(row.get("error"))
        ):
            raise ValueError(
                f"Sealed outcome row {index} differs from the frozen panel: {path}"
            )
        try:
            numeric_success = float(success)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Sealed outcome row {index} has non-numeric success: {path}"
            ) from exc
        if numeric_success not in (0.0, 1.0):
            raise ValueError(
                f"Sealed outcome row {index} has non-binary success: {path}"
            )
    return payload


def _load_evidence(
    path: Path,
    *,
    outcomes_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    evidence = json.loads(path.read_text(encoding="utf-8"))
    outcomes = evidence.get("outcomes") or {}
    manifest = evidence.get("identity", {}).get("evaluation_manifest") or {}
    referenced_outcomes = (path.parent / str(outcomes.get("path", ""))).resolve()
    if (
        evidence.get("schema_version") != 2
        or evidence.get("kind") != "art_embodied_evaluation_evidence"
        or referenced_outcomes != outcomes_path.resolve()
        or outcomes.get("sha256") != _sha256(outcomes_path)
        or manifest.get("sha256") != _sha256(manifest_path)
    ):
        raise ValueError(
            f"Sealed evaluation evidence is not provenance-complete: {path}"
        )
    return evidence


def _validate_candidate_checkpoint(
    *,
    checkpoint_path: Path,
    candidate_evidence: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = checkpoint_path.expanduser().resolve()
    expected_checkpoint = Path(manifest["candidate_checkpoint"]).expanduser().resolve()
    if checkpoint != expected_checkpoint:
        raise ValueError(
            "Sealed candidate checkpoint differs from the frozen selection"
        )
    marker_path = checkpoint / "art_embodied_checkpoint_complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    CheckpointManager().validate(
        checkpoint,
        expected_resume_contract_fingerprint=manifest["resume_contract_fingerprint"],
        require_training_state=False,
    )
    artifact = (
        candidate_evidence.get("identity", {})
        .get("policy", {})
        .get("evaluated_state", {})
        .get("artifact", {})
    )
    if (
        marker.get("metadata", {}).get("step") != 100
        or artifact.get("kind") != "transactional_checkpoint"
        or Path(artifact.get("path", "")).expanduser().resolve() != checkpoint
        or artifact.get("marker_sha256") != _sha256(marker_path)
        or artifact.get("resume_contract_fingerprint")
        != manifest["resume_contract_fingerprint"]
    ):
        raise ValueError(
            "Sealed candidate evidence is not bound to the frozen update-100 marker"
        )
    return {
        "path": str(checkpoint),
        "marker_path": str(marker_path),
        "marker_sha256": _sha256(marker_path),
    }


def adjudicate(
    *,
    manifest_path: Path,
    baseline_path: Path,
    candidate_path: Path,
    baseline_evidence_path: Path,
    candidate_evidence_path: Path,
    candidate_checkpoint_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "art_embodied_sealed_evaluation_manifest"
        or manifest.get("sealed_outcomes_observed") is not False
        or manifest.get("latest_completed_development_update_when_frozen") != 8
    ):
        raise ValueError("The sealed manifest was not frozen before candidate outcomes")
    if (
        manifest.get("episodes") != len(manifest.get("seeds", []))
        or len(manifest.get("fixed_scenarios", [])) != 1
    ):
        raise ValueError("The sealed manifest does not define one exact episode panel")
    baseline = _load_outcomes(baseline_path, expected_step=0, manifest=manifest)
    candidate = _load_outcomes(candidate_path, expected_step=100, manifest=manifest)
    baseline_evidence = _load_evidence(
        baseline_evidence_path,
        outcomes_path=baseline_path,
        manifest_path=manifest_path,
    )
    candidate_evidence = _load_evidence(
        candidate_evidence_path,
        outcomes_path=candidate_path,
        manifest_path=manifest_path,
    )
    if candidate_evidence.get("baseline_outcomes", {}).get("sha256") != _sha256(
        baseline_path
    ):
        raise ValueError("Sealed candidate evidence used a different paired baseline")
    checkpoint_identity = _validate_candidate_checkpoint(
        checkpoint_path=candidate_checkpoint_path,
        candidate_evidence=candidate_evidence,
        manifest=manifest,
    )
    baseline_rows = baseline["episodes"]
    candidate_rows = candidate["episodes"]
    completed_baseline = sum(bool(row.get("completed")) for row in baseline_rows)
    completed_candidate = sum(bool(row.get("completed")) for row in candidate_rows)
    paired = compare_paired_evaluation_reports(baseline_path, candidate_path)
    checks = {
        "baseline_192_complete": completed_baseline == 192,
        "candidate_192_complete": completed_candidate == 192,
        "success_rate_lift_positive": paired["success_rate_lift"] > 0.0,
        "paired_lift_ci95_lower_positive": (paired["success_rate_lift_ci95_low"] > 0.0),
        "improved_pairs_exceed_regressed": (
            paired["improved_pairs"] > paired["regressed_pairs"]
        ),
        "mcnemar_exact_p_at_most_005": (paired["mcnemar_exact_p_value"] <= 0.05),
    }
    return {
        "schema_version": 1,
        "kind": "gr00t_n1d7_robocasa_one_task_u100_sealed_adjudication",
        "status": "passed" if all(checks.values()) else "rejected",
        "claim": (
            "The fixed update-100 candidate generalizes to the preregistered "
            "single-task sealed seed panel."
            if all(checks.values())
            else "The fixed update-100 candidate did not pass the sealed panel."
        ),
        "no_post_sealed_tuning": True,
        "manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
        "baseline": {
            "path": str(baseline_path),
            "sha256": _sha256(baseline_path),
        },
        "candidate": {
            "path": str(candidate_path),
            "sha256": _sha256(candidate_path),
        },
        "baseline_evidence": {
            "path": str(baseline_evidence_path),
            "sha256": _sha256(baseline_evidence_path),
        },
        "candidate_evidence": {
            "path": str(candidate_evidence_path),
            "sha256": _sha256(candidate_evidence_path),
        },
        "candidate_checkpoint": checkpoint_identity,
        "gate_checks": checks,
        "paired": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-evidence", type=Path, required=True)
    parser.add_argument("--candidate-evidence", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = adjudicate(
        manifest_path=args.manifest,
        baseline_path=args.baseline,
        candidate_path=args.candidate,
        baseline_evidence_path=args.baseline_evidence,
        candidate_evidence_path=args.candidate_evidence,
        candidate_checkpoint_path=args.candidate_checkpoint,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="", flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
