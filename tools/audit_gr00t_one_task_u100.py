#!/usr/bin/env python3
"""Independently audit the frozen GR00T one-task update-100 evidence chain."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from art_embodied import EmbodiedExperimentConfig
from art_embodied.checkpointing import CheckpointManager
from art_embodied.evaluation import _evaluation_seed, compare_paired_evaluation_reports
from art_embodied.evidence import _artifact_identity

PARENT_UPDATES = tuple(range(0, 21, 2))
CONTINUATION_UPDATES = tuple(range(25, 101, 5))
EVALUATION_UPDATES = PARENT_UPDATES + CONTINUATION_UPDATES
DURABILITY_UPDATES = (90, 95, 100)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _require_hash(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"{label} differs from its frozen identity: {path}")


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _validate_contract(
    *,
    config_path: Path,
    preregistration_path: Path,
) -> tuple[
    EmbodiedExperimentConfig,
    EmbodiedExperimentConfig,
    dict[str, Any],
    dict[str, Any],
]:
    preregistration = _load_object(preregistration_path)
    if (
        preregistration.get("schema_version") != 1
        or preregistration.get("kind")
        != "art_embodied_development_experiment_preregistration"
        or preregistration.get("decision_time", {}).get(
            "primary_update20_outcome_observed"
        )
        is not False
    ):
        raise ValueError("The continuation was not prospectively frozen")

    frozen = preregistration.get("frozen_inputs") or {}
    parent_path = Path(str(frozen.get("parent_config", "")))
    parent_preregistration_path = Path(str(frozen.get("parent_preregistration", "")))
    _require_hash(config_path, frozen["config_sha256"], label="Continuation config")
    _require_hash(parent_path, frozen["parent_config_sha256"], label="Parent config")
    _require_hash(
        parent_preregistration_path,
        frozen["parent_preregistration_sha256"],
        label="Parent preregistration",
    )
    for source, expected in frozen["implementation_sha256"].items():
        _require_hash(Path(source), expected, label="Frozen implementation")

    config = EmbodiedExperimentConfig.from_yaml(config_path)
    parent = EmbodiedExperimentConfig.from_yaml(parent_path)
    parent_preregistration = _load_object(parent_preregistration_path)
    expected_task = preregistration["scope"]["task"]
    expected_scenario = f"robocasa-gr1/{expected_task}/eval"
    expected_seeds = list(range(400, 464))
    checkpoint = _resolved(frozen["expected_resume_checkpoint"])
    if (
        config.training.updates != 100
        or parent.training.updates != 20
        or config.resume_contract_fingerprint != frozen["resume_contract_fingerprint"]
        or config.resume_contract_fingerprint != parent.resume_contract_fingerprint
        or config.storage.output_dir.resolve() != parent.storage.output_dir.resolve()
        or config.storage.resume_from_checkpoint is None
        or config.storage.resume_from_checkpoint.resolve() != checkpoint
        or config.environment.kwargs.get("task_ids") != [expected_task]
        or parent.environment.kwargs.get("task_ids") != [expected_task]
        or config.evaluation.fixed_scenarios != [expected_scenario]
        or parent.evaluation.fixed_scenarios != [expected_scenario]
        or config.evaluation.seeds != expected_seeds
        or parent.evaluation.seeds != expected_seeds
        or config.evaluation.episodes != 64
        or parent.evaluation.episodes != 64
        or config.evaluation.every_updates != 5
        or parent.evaluation.every_updates != 2
        or config.evaluation.evaluate_before_training
        or not parent.evaluation.evaluate_before_training
        or config.evaluation.data_role != "development"
        or parent.evaluation.data_role != "development"
        or not config.evaluation.deterministic
        or not parent.evaluation.deterministic
        or config.evaluation.checkpoint_selection != "last"
    ):
        raise ValueError("Executable configs do not implement the frozen u100 plan")

    parent_frozen = parent_preregistration.get("frozen_inputs") or {}
    sft_checkpoint = Path(parent_frozen["sft_checkpoint"])
    sft_marker = sft_checkpoint / "art_embodied_sft_complete.json"
    if (
        parent_frozen.get("config_sha256") != _sha256(parent_path)
        or parent_frozen.get("sft_completion_marker_sha256")
        != frozen["sft_completion_marker_sha256"]
    ):
        raise ValueError("Parent SFT/config identity differs from the frozen chain")
    _require_hash(
        sft_marker,
        frozen["sft_completion_marker_sha256"],
        label="Frozen SFT completion marker",
    )
    return config, parent, preregistration, parent_preregistration


def _expected_policy_seeds(
    config: EmbodiedExperimentConfig,
    *,
    scenario_id: str,
) -> list[int]:
    seed_contract = config.evaluation.kwargs.get("seed_contract", {})
    return [
        _evaluation_seed(
            seed_contract=seed_contract,
            kind="policy",
            configured_seed=seed,
            scenario_id=scenario_id,
            repetition=episode,
        )
        for episode, seed in enumerate(config.evaluation.seeds)
    ]


def _base_artifact(evidence: dict[str, Any]) -> dict[str, Any]:
    artifact = (
        evidence.get("identity", {})
        .get("policy", {})
        .get("base", {})
        .get("local_artifact", {})
    )
    if not isinstance(artifact, dict):
        raise ValueError("Evaluation evidence has no local base artifact")
    return artifact


def _validate_evaluation(
    *,
    output_dir: Path,
    update: int,
    config: EmbodiedExperimentConfig,
    expected_task: str,
    expected_base: dict[str, Any] | None,
    baseline_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    outcomes_path = (
        output_dir / "evaluation" / f"update_{update:06d}_episode_outcomes.json"
    )
    evidence_path = output_dir / "evaluation" / f"update_{update:06d}_evidence.json"
    outcomes = _load_object(outcomes_path)
    evidence = _load_object(evidence_path)
    scenario = config.evaluation.fixed_scenarios[0]
    policy_seeds = _expected_policy_seeds(config, scenario_id=scenario)
    rows = outcomes.get("episodes")
    if not isinstance(rows, list) or len(rows) != 64:
        raise ValueError(f"Update {update} does not contain exactly 64 episodes")
    for episode, (row, environment_seed, policy_seed) in enumerate(
        zip(rows, config.evaluation.seeds, policy_seeds, strict=True)
    ):
        try:
            success = float(row.get("success"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid success at update {update}, row {episode}"
            ) from exc
        if (
            row.get("episode") != episode
            or row.get("task") != expected_task
            or row.get("scenario_id") != scenario
            or row.get("environment_seed") != environment_seed
            or row.get("policy_seed") != policy_seed
            or row.get("completed") is not True
            or bool(row.get("error"))
            or success not in (0.0, 1.0)
        ):
            raise ValueError(
                f"Update {update} row {episode} violates the fixed evaluation plan"
            )

    referenced_outcomes = (
        evidence_path.parent / str(evidence.get("outcomes", {}).get("path", ""))
    ).resolve()
    if (
        outcomes.get("schema_version") != 1
        or outcomes.get("config_fingerprint") != config.fingerprint
        or outcomes.get("data_role") != "development"
        or outcomes.get("split") != config.evaluation.split
        or outcomes.get("step") != update
        or evidence.get("schema_version") != 2
        or evidence.get("kind") != "art_embodied_evaluation_evidence"
        or evidence.get("step") != update
        or evidence.get("config_fingerprint") != config.fingerprint
        or referenced_outcomes != outcomes_path.resolve()
        or evidence.get("outcomes", {}).get("sha256") != _sha256(outcomes_path)
    ):
        raise ValueError(f"Update {update} evaluation evidence is incomplete")

    base = _base_artifact(evidence)
    if expected_base is not None and base != expected_base:
        raise ValueError(f"Update {update} evaluated a different SFT base")
    state = evidence.get("identity", {}).get("policy", {}).get("evaluated_state", {})
    artifact = state.get("artifact", {})
    if update == 0:
        if state.get("status") != "immutable_local_base" or artifact != base:
            raise ValueError("Update 0 is not the immutable SFT base")
        if evidence.get("baseline_outcomes") is not None:
            raise ValueError("Update 0 unexpectedly references another baseline")
    else:
        expected_checkpoint = (
            output_dir / "checkpoints" / f"step-{update:06d}"
        ).resolve()
        baseline = evidence.get("baseline_outcomes") or {}
        if (
            state.get("status") != "checkpoint"
            or artifact.get("kind") != "transactional_checkpoint"
            or _resolved(str(artifact.get("path", ""))) != expected_checkpoint
            or artifact.get("config_fingerprint") != config.fingerprint
            or artifact.get("resume_contract_fingerprint")
            != config.resume_contract_fingerprint
            or not isinstance(artifact.get("marker_sha256"), str)
            or len(artifact["marker_sha256"]) != 64
            or baseline.get("sha256") != _sha256(baseline_path)
        ):
            raise ValueError(f"Update {update} evidence identifies the wrong policy")
    return outcomes, evidence


def _validate_rollout(
    *,
    path: Path,
    policy_version: int,
    config: EmbodiedExperimentConfig,
    expected_task: str,
) -> dict[str, Any]:
    payload = _load_object(path)
    groups = payload.get("groups")
    if (
        payload.get("schema_version") != 1
        or payload.get("config_fingerprint") != config.fingerprint
        or payload.get("policy_version") != policy_version
        or not isinstance(groups, list)
        or len(groups) != 32
    ):
        raise ValueError(
            f"Policy version {policy_version} has invalid rollout evidence"
        )
    scenario = f"robocasa-gr1/{expected_task}/train"
    trajectories = failures = successes = mixed = 0
    for group_index, group in enumerate(groups):
        rewards = group.get("rewards")
        success_values = group.get("successes")
        completed = int(group.get("completed_trajectories", -1))
        failed = int(group.get("failed_trajectories", -1))
        success_count = int(group.get("success_count", -1))
        if (
            group.get("group_index") != group_index
            or group.get("policy_version") != policy_version
            or group.get("task") != expected_task
            or group.get("scenario_id") != scenario
            or completed != 8
            or failed != 0
            or not isinstance(group.get("environment_seed"), int)
            or not isinstance(rewards, list)
            or len(rewards) != 8
            or any(float(value) not in (0.0, 0.1, 0.75, 1.0) for value in rewards)
            or not isinstance(success_values, list)
            or len(success_values) != 8
            or any(float(value) not in (0.0, 1.0) for value in success_values)
            or success_count != sum(int(value) for value in success_values)
        ):
            raise ValueError(
                f"Policy version {policy_version}, group {group_index} is incomplete"
            )
        trajectories += completed
        failures += failed
        successes += success_count
        mixed += int(group.get("signal_class") == "mixed")
    summary = payload.get("summary") or {}
    if (
        trajectories != 256
        or failures
        or summary.get("groups") != 32
        or summary.get("trajectories") != 256
        or summary.get("mixed_reward_groups") != mixed
    ):
        raise ValueError(f"Policy version {policy_version} rollout is inconsistent")
    return {
        "policy_version": policy_version,
        "successes": successes,
        "success_rate": successes / trajectories,
        "mixed_reward_groups": mixed,
        "sha256": _sha256(path),
    }


def audit(
    *,
    config_path: Path,
    preregistration_path: Path,
    adjudication_path: Path,
) -> dict[str, Any]:
    config, parent, preregistration, parent_preregistration = _validate_contract(
        config_path=config_path,
        preregistration_path=preregistration_path,
    )
    output_dir = config.storage.output_dir
    expected_task = preregistration["scope"]["task"]
    baseline_path = output_dir / "evaluation" / "update_000000_episode_outcomes.json"

    evaluations: list[dict[str, Any]] = []
    expected_base: dict[str, Any] | None = None
    for update in EVALUATION_UPDATES:
        point_config = parent if update <= 20 else config
        outcomes, evidence = _validate_evaluation(
            output_dir=output_dir,
            update=update,
            config=point_config,
            expected_task=expected_task,
            expected_base=expected_base,
            baseline_path=baseline_path,
        )
        if expected_base is None:
            expected_base = _base_artifact(evidence)
            parent_frozen = parent_preregistration["frozen_inputs"]
            if (
                _resolved(str(expected_base.get("path", "")))
                != _resolved(parent_frozen["sft_checkpoint"])
                or expected_base.get("kind") != "directory"
                or not isinstance(expected_base.get("sha256"), str)
                or len(expected_base["sha256"]) != 64
            ):
                raise ValueError("Update 0 does not identify the frozen SFT checkpoint")
            if expected_base != _artifact_identity(Path(config.policy.path)):
                raise ValueError(
                    "The current frozen SFT directory differs from update 0"
                )
        evaluations.append(
            {
                "update": update,
                "successes": sum(
                    int(float(row["success"])) for row in outcomes["episodes"]
                ),
                "outcomes_sha256": _sha256(
                    output_dir
                    / "evaluation"
                    / f"update_{update:06d}_episode_outcomes.json"
                ),
                "evidence_sha256": _sha256(
                    output_dir / "evaluation" / f"update_{update:06d}_evidence.json"
                ),
            }
        )

    rollouts = [
        _validate_rollout(
            path=output_dir
            / "rollout-evidence"
            / f"policy-version-{policy_version:06d}.json",
            policy_version=policy_version,
            config=parent if policy_version < 20 else config,
            expected_task=expected_task,
        )
        for policy_version in range(100)
    ]

    retained_updates = tuple(range(2, 21, 2)) + (25, 50, 75, 100)
    evidence_by_update = {
        update: _load_object(
            output_dir / "evaluation" / f"update_{update:06d}_evidence.json"
        )
        for update in retained_updates
    }
    for update in retained_updates:
        point_config = parent if update <= 20 else config
        retained = output_dir / "checkpoints" / f"step-{update:06d}"
        CheckpointManager().validate(
            retained,
            expected_resume_contract_fingerprint=(
                point_config.resume_contract_fingerprint
            ),
            require_training_state=True,
        )
        marker_path = retained / "art_embodied_checkpoint_complete.json"
        artifact = evidence_by_update[update]["identity"]["policy"]["evaluated_state"][
            "artifact"
        ]
        if artifact.get("marker_sha256") != _sha256(marker_path):
            raise ValueError(
                f"Retained update-{update} checkpoint differs from its evaluation"
            )

    endpoint = output_dir / "evaluation" / "update_000100_episode_outcomes.json"
    paired = compare_paired_evaluation_reports(baseline_path, endpoint)
    points = {point["update"]: point for point in evaluations}
    durability_above_baseline = sum(
        points[update]["successes"] > points[0]["successes"]
        for update in DURABILITY_UPDATES
    )
    independent_checks = {
        "all_27_evaluations_verified": len(evaluations) == len(EVALUATION_UPDATES),
        "all_100_rollouts_verified": len(rollouts) == 100,
        "update100_lift_positive": paired["success_rate_lift"] > 0.0,
        "update100_ci95_lower_positive": paired["success_rate_lift_ci95_low"] > 0.0,
        "update100_improved_exceeds_regressed": paired["improved_pairs"]
        > paired["regressed_pairs"],
        "update100_mcnemar_at_most_005": paired["mcnemar_exact_p_value"] <= 0.05,
        "durability_two_of_last_three_above_baseline": durability_above_baseline >= 2,
    }

    checkpoint = output_dir / "checkpoints" / "step-000100"
    CheckpointManager().validate(
        checkpoint,
        expected_resume_contract_fingerprint=config.resume_contract_fingerprint,
        require_training_state=True,
    )
    marker_path = checkpoint / "art_embodied_checkpoint_complete.json"
    marker = _load_object(marker_path)
    endpoint_evidence = _load_object(
        output_dir / "evaluation" / "update_000100_evidence.json"
    )
    endpoint_artifact = endpoint_evidence["identity"]["policy"]["evaluated_state"][
        "artifact"
    ]
    if (
        marker.get("metadata", {}).get("step") != 100
        or marker.get("config_fingerprint") != config.fingerprint
        or endpoint_artifact.get("marker_sha256") != _sha256(marker_path)
    ):
        raise ValueError(
            "The update-100 checkpoint differs from evaluated policy state"
        )

    adjudication = _load_object(adjudication_path)
    expected_selected = str(checkpoint)
    if (
        adjudication.get("schema_version") != 1
        or adjudication.get("kind") != "gr00t_n1d7_robocasa_one_task_u100_adjudication"
        or adjudication.get("status") != "passed"
        or adjudication.get("sealed_eligible") is not True
        or adjudication.get("selected_checkpoint") != expected_selected
        or not all(adjudication.get("gate_checks", {}).values())
    ):
        raise ValueError(
            "Primary development adjudication did not admit sealed evaluation"
        )
    reported_paired = adjudication.get("update100_paired") or {}
    for key, value in paired.items():
        if reported_paired.get(key) != value:
            raise ValueError(f"Primary adjudication differs on paired metric {key}")
    if not all(independent_checks.values()):
        raise ValueError(f"Independent u100 gate rejected: {independent_checks}")

    return {
        "schema_version": 1,
        "kind": "gr00t_n1d7_robocasa_one_task_u100_independent_evidence_audit",
        "status": "passed",
        "sealed_eligible": True,
        "no_policy_or_outcome_mutation": True,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "preregistration": {
            "path": str(preregistration_path),
            "sha256": _sha256(preregistration_path),
        },
        "primary_adjudication": {
            "path": str(adjudication_path),
            "sha256": _sha256(adjudication_path),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "marker_sha256": _sha256(marker_path),
        },
        "checks": independent_checks,
        "paired_update0_update100": paired,
        "evaluations": evaluations,
        "rollouts": rollouts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite evidence audit: {args.output}")
    report = audit(
        config_path=args.config,
        preregistration_path=args.preregistration,
        adjudication_path=args.adjudication,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
