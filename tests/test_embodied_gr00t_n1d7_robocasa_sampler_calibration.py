from __future__ import annotations

import json
from pathlib import Path

from art_embodied import EmbodiedExperimentConfig
from examples.embodied.robocasa.adjudicate_one_task_u20 import (
    _evaluation_point,
    _linear_slope,
    _training_point,
)
from examples.embodied.robocasa.gr00t_n1d7_noise_calibration import (
    _build_report as _build_noise_report,
)
from examples.embodied.robocasa.gr00t_n1d7_noise_extension import _merge_outcomes
from examples.embodied.robocasa.gr00t_n1d7_post_oracle_sampler_calibration import (
    _without_high_volume_observability,
)
from examples.embodied.robocasa.gr00t_n1d7_sampler_calibration import (
    _clustered_comparison,
    _condition_config,
    _equivalence_decision,
)

ROOT = Path(__file__).parents[1]
CONFIG = (
    ROOT / "examples/embodied/"
    "gr00t_n1d7_robocasa_gr1_tabletop_cuttingboard_pan_single_task_development.yaml"
)


def test_calibration_configs_pair_reset_and_policy_seeds(tmp_path: Path) -> None:
    base = EmbodiedExperimentConfig.from_yaml(CONFIG)
    preregistration = tmp_path / "preregistered.json"
    preregistration.write_text("{}\n", encoding="utf-8")

    ode = _condition_config(
        base,
        label="baseline-ode",
        output_dir=tmp_path / "ode",
        seeds=[400, 401],
        policy_seed_repetitions=8,
        sampler="native-ode",
        candidate=False,
        preregistration=preregistration,
    )
    sde = _condition_config(
        base,
        label="baseline-sde",
        output_dir=tmp_path / "sde",
        seeds=[400, 401],
        policy_seed_repetitions=8,
        sampler="flow-sde",
        candidate=False,
        preregistration=preregistration,
    )

    assert ode.evaluation.episodes == sde.evaluation.episodes == 16
    assert ode.evaluation.seeds == sde.evaluation.seeds == [400, 401]
    assert ode.evaluation.split == sde.evaluation.split == "train_matched"
    assert ode.evaluation.data_role == sde.evaluation.data_role == "diagnostic"
    assert ode.evaluation.kwargs["seed_contract"] == {
        "environment_mode": "configured",
        "fixed_environment_seed": 0,
        "policy_mode": "derived",
        "fixed_policy_seed": 0,
    }
    assert ode.evaluation.deterministic is True
    assert ode.policy.evaluation_generation.do_sample is False
    assert ode.evaluation.kwargs["action_sampling"] == "native_flow_ode"
    assert sde.evaluation.deterministic is False
    assert sde.policy.evaluation_generation.do_sample is True
    assert sde.evaluation.kwargs["action_sampling"] == "gaussian_flow_sde"
    assert base.evaluation.split == "held_out"
    assert base.evaluation.data_role == "development"


def test_post_oracle_calibration_disables_high_volume_observability(
    tmp_path: Path,
) -> None:
    base = EmbodiedExperimentConfig.from_yaml(CONFIG)
    preregistration = tmp_path / "preregistered.json"
    preregistration.write_text("{}\n", encoding="utf-8")
    condition = _condition_config(
        base,
        label="post-oracle-sde",
        output_dir=tmp_path / "sde",
        seeds=[400, 401],
        policy_seed_repetitions=2,
        sampler="flow-sde",
        candidate=False,
        preregistration=preregistration,
        noise_level=0.1,
    )

    condition = _without_high_volume_observability(condition)

    assert condition.algorithm.flow_sde.noise_level == 0.1
    assert condition.evaluation.kwargs["action_sampling"] == "gaussian_flow_sde"
    assert condition.observability.wandb.enabled is True
    assert condition.observability.wandb.log_evaluation_artifacts is True
    assert condition.observability.weave.enabled is False
    assert condition.observability.videos_per_evaluation == 0
    assert condition.observability.require_evaluation_video is False


def test_u20_adjudication_extracts_complete_curves(tmp_path: Path) -> None:
    outcomes = tmp_path / "outcomes.json"
    outcomes.write_text(
        json.dumps(
            {
                "episodes": [
                    {"completed": True, "success": float(index % 2 == 0)}
                    for index in range(64)
                ]
            }
        ),
        encoding="utf-8",
    )
    evaluation = _evaluation_point(outcomes, 20)
    assert evaluation["completed_episodes"] == 64
    assert evaluation["successes"] == 32
    assert evaluation["success_rate"] == 0.5

    rollout = tmp_path / "rollout.json"
    rollout.write_text(
        json.dumps(
            {
                "policy_version": 19,
                "summary": {"mixed_reward_groups": 24},
                "groups": [
                    {
                        "completed_trajectories": 8,
                        "failed_trajectories": 0,
                        "success_count": 4,
                    }
                    for _ in range(32)
                ],
            }
        ),
        encoding="utf-8",
    )
    training = _training_point(rollout)
    assert training["training_update"] == 20
    assert training["trajectories"] == 256
    assert training["success_rate"] == 0.5
    assert _linear_slope([(0, 0.25), (10, 0.5), (20, 0.75)]) == 0.025


def test_clustered_comparison_resamples_resets_not_episodes(tmp_path: Path) -> None:
    baseline_rows = []
    candidate_rows = []
    outcomes = ((400, 0.0), (400, 1.0), (401, 1.0), (401, 1.0))
    candidate_successes = (1.0, 1.0, 0.0, 1.0)
    for episode, ((environment_seed, success), candidate_success) in enumerate(
        zip(outcomes, candidate_successes, strict=True)
    ):
        common = {
            "episode": episode,
            "scenario_id": "scenario",
            "environment_seed": environment_seed,
            "policy_seed": 1000 + episode,
            "completed": True,
        }
        baseline_rows.append({**common, "success": success})
        candidate_rows.append({**common, "success": candidate_success})
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps({"episodes": baseline_rows}), encoding="utf-8")
    candidate.write_text(json.dumps({"episodes": candidate_rows}), encoding="utf-8")

    result = _clustered_comparison(baseline, candidate)

    assert result["reset_clusters"] == 2
    assert result["episodes_per_reset"] == [2]
    assert result["success_rate_difference"] == 0.0
    assert result["per_environment_seed_difference"] == {"400": 0.5, "401": -0.5}


def test_sampler_equivalence_has_pass_reject_and_inconclusive() -> None:
    assert (
        _equivalence_decision(
            {
                "success_rate_difference": 0.01,
                "cluster_bootstrap_ci95_low": -0.05,
                "cluster_bootstrap_ci95_high": 0.07,
            },
            maximum_absolute_gap=0.10,
        )
        == "pass"
    )
    assert (
        _equivalence_decision(
            {
                "success_rate_difference": -0.25,
                "cluster_bootstrap_ci95_low": -0.35,
                "cluster_bootstrap_ci95_high": -0.15,
            },
            maximum_absolute_gap=0.10,
        )
        == "reject"
    )
    assert (
        _equivalence_decision(
            {
                "success_rate_difference": -0.08,
                "cluster_bootstrap_ci95_low": -0.18,
                "cluster_bootstrap_ci95_high": 0.02,
            },
            maximum_absolute_gap=0.10,
        )
        == "inconclusive"
    )


def test_noise_calibration_selects_highest_passing_noise(tmp_path: Path) -> None:
    common = {
        "episode": 0,
        "scenario_id": "scenario",
        "environment_seed": 400,
        "policy_seed": 1000,
        "completed": True,
    }
    reference = tmp_path / "reference.json"
    evidence = tmp_path / "evidence.json"
    config = tmp_path / "config.yaml"
    preregistration = tmp_path / "preregistered.json"
    reference.write_text(
        json.dumps({"schema_version": 1, "episodes": [{**common, "success": 1.0}]}),
        encoding="utf-8",
    )
    evidence.write_text("{}\n", encoding="utf-8")
    config.write_text("config\n", encoding="utf-8")
    preregistration.write_text("{}\n", encoding="utf-8")
    conditions = {}
    for noise, success in ((0.1, 1.0), (0.2, 1.0), (0.3, 0.0)):
        root = tmp_path / str(noise)
        root.mkdir()
        paths = {
            "config": root / "config.yaml",
            "outcomes": root / "outcomes.json",
            "evidence": root / "evidence.json",
        }
        paths["config"].write_text("config\n", encoding="utf-8")
        paths["outcomes"].write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "episodes": [{**common, "success": success}],
                }
            ),
            encoding="utf-8",
        )
        paths["evidence"].write_text("{}\n", encoding="utf-8")
        conditions[noise] = paths

    report = _build_noise_report(
        config=config,
        preregistration=preregistration,
        reference_outcomes=reference,
        reference_evidence=evidence,
        conditions=conditions,
        maximum_absolute_gap=0.10,
    )

    assert report["selected_noise_level"] == 0.2
    assert report["training_admission"] is True


def test_noise_extension_merges_and_renumbers_new_pairs(tmp_path: Path) -> None:
    prior = tmp_path / "prior.json"
    extension = tmp_path / "extension.json"
    destination = tmp_path / "aggregate.json"
    common = {"schema_version": 1, "split": "train_matched"}
    prior.write_text(
        json.dumps(
            {
                **common,
                "episodes": [
                    {
                        "episode": 0,
                        "scenario_id": "scenario",
                        "environment_seed": 400,
                        "policy_seed": 10,
                        "completed": True,
                        "success": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    extension.write_text(
        json.dumps(
            {
                **common,
                "episodes": [
                    {
                        "episode": 0,
                        "scenario_id": "scenario",
                        "environment_seed": 416,
                        "policy_seed": 20,
                        "completed": True,
                        "success": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    _merge_outcomes(prior, extension, destination)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert [row["episode"] for row in payload["episodes"]] == [0, 1]
    assert len(payload["aggregate_sources"]) == 2
