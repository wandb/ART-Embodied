from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from art_embodied.config import EmbodiedExperimentConfig
from art_embodied.evaluation import (
    FixedScenarioEvaluator,
    _evaluation_manifest_path,
    _wait_for_outcome_report,
    compare_paired_evaluation_reports,
    create_paired_evaluation_candidate,
    summarize_evaluation_report,
    validate_paired_evaluation_configs,
)
from art_embodied.experiment import EmbodiedScenario
from art_embodied.trajectories import EmbodiedTrajectory
from art_embodied.types import LocalTrainResult


def _config(tmp_path: Path, *, episodes: int = 4) -> EmbodiedExperimentConfig:
    source = (
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml"
    )
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["storage"]["output_dir"] = str(tmp_path / "run")
    raw["evaluation"]["enabled"] = True
    raw["evaluation"]["episodes"] = episodes
    raw["evaluation"]["fixed_scenarios"] = ["held-out-0", "held-out-1"]
    raw["evaluation"]["seeds"] = [11, 17]
    raw["evaluation"]["split"] = "held_out"
    raw["evaluation"]["data_role"] = "development"
    raw["evaluation"]["baseline_outcomes_path"] = None
    return EmbodiedExperimentConfig.model_validate(raw)


def _scenarios() -> list[EmbodiedScenario]:
    return [
        EmbodiedScenario(id="held-out-0", task="pick", payload={}),
        EmbodiedScenario(id="held-out-1", task="place", payload={}),
    ]


def test_paired_config_validation_rejects_environment_drift(tmp_path: Path) -> None:
    baseline = _config(tmp_path)
    baseline_report = (
        baseline.storage.output_dir / "evaluation/update_000000_episode_outcomes.json"
    )
    candidate = baseline.model_copy(
        update={
            "evaluation": baseline.evaluation.model_copy(
                update={"baseline_outcomes_path": baseline_report}
            ),
            "environment": baseline.environment.model_copy(
                update={"task": "different-held-out-suite"}
            ),
        }
    )

    with pytest.raises(ValueError, match="environment"):
        validate_paired_evaluation_configs(baseline, candidate)


def test_paired_config_validation_allows_model_identity_as_treatment(
    tmp_path: Path,
) -> None:
    baseline = _config(tmp_path)
    baseline_report = (
        baseline.storage.output_dir / "evaluation/update_000000_episode_outcomes.json"
    )
    candidate = baseline.model_copy(
        update={
            "evaluation": baseline.evaluation.model_copy(
                update={"baseline_outcomes_path": baseline_report}
            ),
            "policy": baseline.policy.model_copy(
                update={
                    "path": "organization/trained-policy",
                    "revision": "candidate-revision",
                }
            ),
        }
    )

    summary = validate_paired_evaluation_configs(baseline, candidate)

    assert summary["baseline_policy_path"] == baseline.policy.path
    assert summary["baseline_policy_revision"] == baseline.policy.revision
    assert summary["candidate_policy_path"] == "organization/trained-policy"
    assert summary["candidate_policy_revision"] == "candidate-revision"


def test_paired_config_validation_allows_candidate_checkpoint_loading_guard(
    tmp_path: Path,
) -> None:
    baseline = _config(tmp_path)
    baseline_report = (
        baseline.storage.output_dir / "evaluation/update_000000_episode_outcomes.json"
    )
    candidate_kwargs = dict(baseline.evaluation.kwargs)
    candidate_kwargs["require_policy_checkpoint"] = True
    candidate = baseline.model_copy(
        update={
            "evaluation": baseline.evaluation.model_copy(
                update={
                    "baseline_outcomes_path": baseline_report,
                    "kwargs": candidate_kwargs,
                }
            )
        }
    )

    summary = validate_paired_evaluation_configs(baseline, candidate)

    assert summary["episodes"] == baseline.evaluation.episodes


def test_development_replication_manifest_freezes_independent_panel(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    manifest = tmp_path / "replication.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "art_embodied_development_replication_manifest",
                "split": config.evaluation.split,
                "data_role": config.evaluation.data_role,
                "episodes": config.evaluation.episodes,
                "seeds": config.evaluation.seeds,
                "fixed_scenarios": config.evaluation.fixed_scenarios,
                "seed_contract": config.evaluation.kwargs["seed_contract"],
                "excluded_prior_seeds": [10, 12],
            }
        ),
        encoding="utf-8",
    )
    kwargs = dict(config.evaluation.kwargs)
    kwargs["replication_manifest_path"] = str(manifest)
    replicated = config.model_copy(
        update={"evaluation": config.evaluation.model_copy(update={"kwargs": kwargs})}
    )

    assert _evaluation_manifest_path(replicated) == str(manifest)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["excluded_prior_seeds"].append(config.evaluation.seeds[0])
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap excluded prior seeds"):
        _evaluation_manifest_path(replicated)


def test_paired_config_validation_rejects_inference_runtime_drift(
    tmp_path: Path,
) -> None:
    baseline = _config(tmp_path)
    baseline_report = (
        baseline.storage.output_dir / "evaluation/update_000000_episode_outcomes.json"
    )
    load_kwargs = dict(baseline.policy.load_kwargs)
    load_kwargs["attn_implementation"] = "flash_attention_2"
    candidate = baseline.model_copy(
        update={
            "evaluation": baseline.evaluation.model_copy(
                update={"baseline_outcomes_path": baseline_report}
            ),
            "policy": baseline.policy.model_copy(update={"load_kwargs": load_kwargs}),
        }
    )

    with pytest.raises(ValueError, match="attn_implementation"):
        validate_paired_evaluation_configs(baseline, candidate)


def test_paired_config_validation_rejects_trainable_dtype_drift(
    tmp_path: Path,
) -> None:
    baseline = _config(tmp_path)
    baseline_report = (
        baseline.storage.output_dir / "evaluation/update_000000_episode_outcomes.json"
    )
    candidate = baseline.model_copy(
        update={
            "evaluation": baseline.evaluation.model_copy(
                update={"baseline_outcomes_path": baseline_report}
            ),
            "policy": baseline.policy.model_copy(
                update={
                    "force_trainable_float32": (
                        not baseline.policy.force_trainable_float32
                    )
                }
            ),
        }
    )

    with pytest.raises(ValueError, match="force_trainable_float32"):
        validate_paired_evaluation_configs(baseline, candidate)


def test_create_paired_evaluation_candidate_preserves_controls(
    tmp_path: Path,
) -> None:
    baseline = _config(tmp_path / "baseline")
    adapter = tmp_path / "adapter"
    adapter.mkdir()

    candidate = create_paired_evaluation_candidate(
        baseline,
        run="trained-lora-evaluation",
        output_dir=tmp_path / "candidate",
        peft_adapter_path=adapter,
    )

    assert candidate.experiment.run == "trained-lora-evaluation"
    assert candidate.policy.path == baseline.policy.path
    assert candidate.policy.revision == baseline.policy.revision
    assert candidate.policy.load_kwargs["peft_adapter_path"] == str(adapter)
    assert (
        candidate.policy.load_kwargs["attn_implementation"]
        == (baseline.policy.load_kwargs["attn_implementation"])
    )
    assert candidate.evaluation.baseline_outcomes_path == (
        baseline.storage.output_dir / "evaluation/update_000000_episode_outcomes.json"
    )
    assert candidate.environment == baseline.environment


def test_create_paired_evaluation_candidate_requires_one_treatment(
    tmp_path: Path,
) -> None:
    baseline = _config(tmp_path / "baseline")

    with pytest.raises(ValueError, match="exactly one"):
        create_paired_evaluation_candidate(
            baseline,
            run="invalid",
            output_dir=tmp_path / "candidate",
        )


def test_fixed_evaluator_has_stable_plan_and_keeps_failures_in_denominator(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    seen = []
    progress = []

    async def rollout(scenario, context):
        seen.append((scenario.id, context.environment_seed, context.policy_seed))
        if context.group_index == 3:
            raise RuntimeError("simulator reset failed")
        return EmbodiedTrajectory(
            task=scenario.task,
            reward=float(context.group_index + 1),
            metrics={"success": context.group_index in (0, 2)},
        )

    async def log_progress(event, logged_config):
        assert logged_config.fingerprint == config.fingerprint
        progress.append(event)

    evaluator = FixedScenarioEvaluator(
        config=config,
        scenarios=_scenarios(),
        rollout=rollout,
        workers=2,
        log_progress=log_progress,
    )
    result = asyncio.run(evaluator(10, LocalTrainResult(step=10), config))

    assert result.metrics["episodes"] == 4.0
    assert result.metrics["completed_episodes"] == 3.0
    assert result.metrics["failed_episodes"] == 1.0
    assert result.metrics["success_rate"] == 0.5
    assert result.metrics["reward_mean"] == 1.5
    assert result.metrics["scenario_macro_success_rate"] == 0.5
    assert result.metrics["scenario/held-out-0/episodes"] == 2.0
    assert result.metrics["scenario/held-out-0/success_rate"] == 1.0
    assert result.metrics["scenario/held-out-1/episodes"] == 2.0
    assert result.metrics["scenario/held-out-1/success_rate"] == 0.0
    assert result.metrics["task/pick/episodes"] == 2.0
    assert result.metrics["task/pick/success_rate"] == 1.0
    assert result.metrics["task/place/episodes"] == 2.0
    assert result.metrics["task/place/success_rate"] == 0.0
    assert result.metrics["task_macro_success_rate"] == 0.5
    assert len(result.trajectories) == 3
    assert len(seen) == 4
    assert len(set(seen)) == 4
    assert [event.completed for event in progress] == [1, 2, 3, 4]
    assert progress[-1].metrics["evaluation_success_rate"] == 0.5
    assert progress[-1].metrics["evaluation_failed_episodes"] == 1
    outcomes_path = Path(result.artifacts["episode_outcomes_json"])
    outcomes = json.loads(outcomes_path.read_text(encoding="utf-8"))
    assert len(outcomes["episodes"]) == 4
    assert outcomes["episodes"][3]["completed"] is False
    assert outcomes["episodes"][3]["error_type"] == "RuntimeError"
    assert outcomes["episodes"][0]["scenario_payload"] == {}
    assert outcomes["episodes"][0]["reset_info"] == {}
    assert outcomes["episodes"][0]["action_count"] == 0
    assert outcomes["episodes"][0]["action_chunk_sha256s"] == []
    assert len(outcomes["episodes"][0]["action_sequence_sha256"]) == 64
    assert outcomes["episodes"][3]["action_sequence_sha256"] is None
    assert {row["environment_seed"] for row in outcomes["episodes"]} == {0}
    evidence_path = Path(result.artifacts["evaluation_evidence_json"])
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["kind"] == "art_embodied_evaluation_evidence"
    assert evidence["schema_version"] == 2
    assert evidence["config_fingerprint"] == config.fingerprint
    assert evidence["resolved_config"]["evaluation"]["data_role"] == "development"
    assert evidence["config_consumption"]["unowned_fields"] == []
    assert evidence["evaluation"]["data_role"] == "development"
    assert evidence["outcomes"]["path"] == outcomes_path.name
    assert (
        evidence["outcomes"]["sha256"]
        == hashlib.sha256(outcomes_path.read_bytes()).hexdigest()
    )
    assert evidence["metrics"]["task_macro_success_rate"] == 0.5
    regenerated = summarize_evaluation_report(outcomes_path)
    assert regenerated == result.metrics

    same_plan = FixedScenarioEvaluator(
        config=config,
        scenarios=_scenarios(),
        rollout=rollout,
    )
    assert same_plan.plan == evaluator.plan


def test_fixed_evaluator_fails_closed_when_every_episode_errors(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    async def failing_rollout(scenario, context):
        del scenario, context
        raise ModuleNotFoundError("missing simulator dependency")

    evaluator = FixedScenarioEvaluator(
        config=config,
        scenarios=_scenarios(),
        rollout=failing_rollout,
    )

    with pytest.raises(RuntimeError, match="All fixed evaluation episodes failed"):
        asyncio.run(evaluator(0, LocalTrainResult(step=0), config))

    report = (
        config.storage.output_dir / "evaluation/update_000000_episode_outcomes.json"
    )
    outcomes = json.loads(report.read_text(encoding="utf-8"))
    assert len(outcomes["episodes"]) == config.evaluation.episodes
    assert {row["error_type"] for row in outcomes["episodes"]} == {
        "ModuleNotFoundError"
    }


def test_sealed_evaluator_requires_frozen_manifest_before_rollout(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path).model_copy(
        update={
            "evaluation": _config(tmp_path).evaluation.model_copy(
                update={"data_role": "sealed_test", "checkpoint_selection": "last"}
            )
        }
    )
    calls = 0

    async def rollout(scenario, context):
        nonlocal calls
        del scenario, context
        calls += 1
        raise AssertionError("rollout must not start")

    evaluator = FixedScenarioEvaluator(
        config=config,
        scenarios=_scenarios(),
        rollout=rollout,
    )

    with pytest.raises(ValueError, match="sealed_manifest_path"):
        asyncio.run(evaluator(1, LocalTrainResult(step=1), config))
    assert calls == 0


def test_sealed_evaluator_rejects_manifest_config_drift_before_rollout(
    tmp_path: Path,
) -> None:
    base = _config(tmp_path)
    manifest = tmp_path / "sealed-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "art_embodied_sealed_evaluation_manifest",
                "split": base.evaluation.split,
                "data_role": "sealed_test",
                "episodes": base.evaluation.episodes,
                "seeds": [999],
                "fixed_scenarios": base.evaluation.fixed_scenarios,
                "seed_contract": base.evaluation.kwargs.get("seed_contract", {}),
                "excluded_development_seeds": [0, 1, 2, 3],
            }
        ),
        encoding="utf-8",
    )
    config = base.model_copy(
        update={
            "evaluation": base.evaluation.model_copy(
                update={
                    "data_role": "sealed_test",
                    "checkpoint_selection": "last",
                    "kwargs": {
                        **base.evaluation.kwargs,
                        "sealed_manifest_path": str(manifest),
                    },
                }
            )
        }
    )
    calls = 0

    async def rollout(scenario, context):
        nonlocal calls
        del scenario, context
        calls += 1
        raise AssertionError("rollout must not start")

    evaluator = FixedScenarioEvaluator(
        config=config,
        scenarios=_scenarios(),
        rollout=rollout,
    )

    with pytest.raises(ValueError, match="does not match the executable config"):
        asyncio.run(evaluator(1, LocalTrainResult(step=1), config))
    assert calls == 0


def test_fixed_evaluator_reports_paired_lift_against_baseline(
    tmp_path: Path,
) -> None:
    baseline_config = _config(tmp_path / "baseline")

    async def baseline_rollout(scenario, context):
        return EmbodiedTrajectory(
            task=scenario.task,
            reward=float(context.group_index == 0),
            metrics={"success": context.group_index == 0},
        )

    baseline = asyncio.run(
        FixedScenarioEvaluator(
            config=baseline_config,
            scenarios=_scenarios(),
            rollout=baseline_rollout,
        )(1, LocalTrainResult(step=1), baseline_config)
    )
    baseline_path = Path(baseline.artifacts["episode_outcomes_json"])

    raw = baseline_config.model_dump(mode="python")
    raw["storage"]["output_dir"] = tmp_path / "candidate"
    raw["evaluation"]["baseline_outcomes_path"] = baseline_path
    candidate_config = EmbodiedExperimentConfig.model_validate(raw)

    async def candidate_rollout(scenario, context):
        return EmbodiedTrajectory(
            task=scenario.task,
            reward=float(context.group_index in (0, 1)),
            metrics={"success": context.group_index in (0, 1)},
        )

    candidate = asyncio.run(
        FixedScenarioEvaluator(
            config=candidate_config,
            scenarios=_scenarios(),
            rollout=candidate_rollout,
        )(2, LocalTrainResult(step=2), candidate_config)
    )

    assert candidate.metrics["paired/baseline_success_rate"] == 0.25
    assert candidate.metrics["paired/candidate_success_rate"] == 0.5
    assert candidate.metrics["paired/success_rate_lift"] == 0.25
    assert candidate.metrics["paired/success_rate_lift_ci95_low"] <= 0.25
    assert candidate.metrics["paired/success_rate_lift_ci95_high"] >= 0.25
    assert candidate.metrics["paired/task_count"] == 2.0
    assert candidate.metrics["paired/baseline_task_macro_success_rate"] == 0.25
    assert candidate.metrics["paired/candidate_task_macro_success_rate"] == 0.5
    assert candidate.metrics["paired/task_macro_success_rate_lift"] == 0.25
    assert candidate.metrics["paired/task/pick/episodes"] == 2.0
    assert candidate.metrics["paired/task/pick/success_rate_lift"] == 0.0
    assert candidate.metrics["paired/task/place/success_rate_lift"] == 0.5
    assert candidate.metrics["paired/improved_pairs"] == 1.0
    assert candidate.metrics["paired/regressed_pairs"] == 0.0
    direct = compare_paired_evaluation_reports(
        baseline_path,
        candidate.artifacts["episode_outcomes_json"],
    )
    assert direct == {
        key.removeprefix("paired/"): value
        for key, value in candidate.metrics.items()
        if key.startswith("paired/")
    }
    evidence = json.loads(
        Path(candidate.artifacts["evaluation_evidence_json"]).read_text(
            encoding="utf-8"
        )
    )
    assert (
        evidence["baseline_outcomes"]["sha256"]
        == hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    )
    assert evidence["statistics"]["paired_bootstrap_samples"] == 10_000


def test_fixed_evaluator_reuses_measured_step_zero_as_paired_baseline(
    tmp_path: Path,
) -> None:
    raw = _config(tmp_path).model_dump(mode="python")
    raw["evaluation"]["evaluate_before_training"] = True
    raw["evaluation"]["baseline_outcomes_path"] = None
    config = EmbodiedExperimentConfig.model_validate(raw)
    policy_version = 0

    async def rollout(scenario, context):
        del scenario
        success = context.group_index == 0 or (
            policy_version == 1 and context.group_index == 1
        )
        return EmbodiedTrajectory(
            task="pick",
            reward=float(success),
            metrics={"success": success},
        )

    evaluator = FixedScenarioEvaluator(
        config=config,
        scenarios=_scenarios(),
        rollout=rollout,
    )
    baseline = asyncio.run(evaluator(0, LocalTrainResult(step=0), config))
    assert "paired/success_rate_lift" not in baseline.metrics
    baseline_path = Path(baseline.artifacts["episode_outcomes_json"])

    policy_version = 1
    candidate = asyncio.run(evaluator(1, LocalTrainResult(step=1), config))

    assert candidate.metrics["paired/baseline_success_rate"] == 0.25
    assert candidate.metrics["paired/candidate_success_rate"] == 0.5
    assert candidate.metrics["paired/success_rate_lift"] == 0.25
    assert candidate.artifacts["baseline_episode_outcomes_json"] == str(baseline_path)
    evidence = json.loads(
        Path(candidate.artifacts["evaluation_evidence_json"]).read_text(
            encoding="utf-8"
        )
    )
    assert evidence["baseline_outcomes"]["path"] == str(baseline_path)


def test_fixed_evaluator_accepts_same_run_baseline_from_previous_instance(
    tmp_path: Path,
) -> None:
    raw = _config(tmp_path).model_dump(mode="python")
    raw["evaluation"]["evaluate_before_training"] = True
    config = EmbodiedExperimentConfig.model_validate(raw)

    async def baseline_rollout(scenario, context):
        return EmbodiedTrajectory(
            task=scenario.task,
            metrics={"success": context.group_index == 0},
        )

    baseline = asyncio.run(
        FixedScenarioEvaluator(
            config=config,
            scenarios=_scenarios(),
            rollout=baseline_rollout,
        )(0, LocalTrainResult(step=0), config)
    )
    baseline_path = Path(baseline.artifacts["episode_outcomes_json"])

    async def candidate_rollout(scenario, context):
        return EmbodiedTrajectory(
            task=scenario.task,
            metrics={"success": context.group_index in (0, 1)},
        )

    candidate = asyncio.run(
        FixedScenarioEvaluator(
            config=config,
            scenarios=_scenarios(),
            rollout=candidate_rollout,
            measured_baseline_path=baseline_path,
        )(25, LocalTrainResult(step=25), config)
    )

    assert candidate.metrics["paired/baseline_success_rate"] == 0.25
    assert candidate.metrics["paired/candidate_success_rate"] == 0.5
    assert candidate.metrics["paired/success_rate_lift"] == 0.25
    assert candidate.artifacts["baseline_episode_outcomes_json"] == str(baseline_path)


def test_concurrent_candidate_can_wait_for_baseline_report(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"

    async def run() -> None:
        waiter = asyncio.create_task(
            _wait_for_outcome_report(baseline_path, timeout_seconds=2)
        )
        await asyncio.sleep(0)
        baseline_path.write_text("{}")
        await waiter

    asyncio.run(run())


def test_fixed_evaluator_requires_explicit_success_metric(tmp_path: Path) -> None:
    config = _config(tmp_path, episodes=1)

    async def rollout(scenario, context):
        del context
        return EmbodiedTrajectory(task=scenario.task, reward=1.0)

    evaluator = FixedScenarioEvaluator(
        config=config,
        scenarios=_scenarios(),
        rollout=rollout,
    )
    with pytest.raises(ValueError, match="metric named 'success'"):
        asyncio.run(evaluator(1, LocalTrainResult(step=1), config))


def test_fixed_evaluator_fails_closed_on_missing_scenarios(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="held-out-1"):
        FixedScenarioEvaluator(
            config=config,
            scenarios=_scenarios()[:1],
            rollout=lambda scenario, context: None,
        )


def test_fixed_evaluator_propagates_cancellation(tmp_path: Path) -> None:
    config = _config(tmp_path, episodes=1)

    async def cancelled_rollout(scenario, context):
        del scenario, context
        raise asyncio.CancelledError

    evaluator = FixedScenarioEvaluator(
        config=config,
        scenarios=_scenarios(),
        rollout=cancelled_rollout,
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(evaluator(1, LocalTrainResult(step=1), config))


def test_fixed_evaluator_refuses_to_overwrite_different_outcomes(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, episodes=1)
    calls = 0

    async def rollout(scenario, context):
        nonlocal calls
        del context
        calls += 1
        return EmbodiedTrajectory(
            task=scenario.task,
            reward=float(calls == 1),
            metrics={"success": calls == 1},
        )

    evaluator = FixedScenarioEvaluator(
        config=config,
        scenarios=_scenarios(),
        rollout=rollout,
    )
    asyncio.run(evaluator(1, LocalTrainResult(step=1), config))

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        asyncio.run(evaluator(1, LocalTrainResult(step=1), config))
