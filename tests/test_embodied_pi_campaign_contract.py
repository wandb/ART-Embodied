from __future__ import annotations

from pathlib import Path

import pytest

from art_embodied import EmbodiedExperimentConfig
from examples.embodied.pi_campaign_contract import audit_pi_campaign_contract

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("relative_path", "profile"),
    [
        (
            "examples/embodied/pi0_libero_spatial_flow_sde_grpo_1024.yaml",
            "pi0-spatial-behavioral-v1",
        ),
        (
            "examples/embodied/pi05_libero_long_flow_sde_grpo_1024.yaml",
            "pi05-long-behavioral-v1",
        ),
    ],
)
def test_calibrated_pi_campaign_recipes_pass_contract(
    relative_path: str,
    profile: str,
) -> None:
    config = EmbodiedExperimentConfig.from_yaml(ROOT / relative_path)

    report = audit_pi_campaign_contract(config, profile_name=profile)

    assert report["passed"] is True
    assert report["failed_checks"] == []


@pytest.mark.parametrize(
    ("relative_path", "profile"),
    [
        (
            "examples/embodied/"
            "pi0_libero_spatial_flow_sde_grpo_rlinf_reference_1024.yaml",
            "pi0-spatial-rlinf-reference-v1",
        ),
        (
            "examples/embodied/"
            "pi05_libero_long_flow_sde_grpo_rlinf_reference_1024.yaml",
            "pi05-long-rlinf-reference-v1",
        ),
    ],
)
def test_rlinf_reference_sampler_profiles_pass_contract(
    relative_path: str,
    profile: str,
) -> None:
    config = EmbodiedExperimentConfig.from_yaml(ROOT / relative_path)

    report = audit_pi_campaign_contract(config, profile_name=profile)

    assert report["passed"] is True
    assert report["failed_checks"] == []


def test_campaign_contract_rejects_sampler_name_that_disagrees_with_yaml() -> None:
    config = EmbodiedExperimentConfig.from_yaml(
        ROOT / "examples/embodied/pi0_libero_spatial_flow_sde_grpo_1024.yaml"
    )
    assert config.algorithm.flow_sde is not None
    broken = config.model_copy(
        update={
            "algorithm": config.algorithm.model_copy(
                update={
                    "flow_sde": config.algorithm.flow_sde.model_copy(
                        update={"num_denoise_steps": 4, "noise_level": 0.5}
                    )
                }
            )
        }
    )

    report = audit_pi_campaign_contract(
        broken,
        profile_name="pi0-spatial-behavioral-v1",
    )

    assert report["passed"] is False
    assert {
        row["path"] for row in report["failed_checks"]
    } == {
        "algorithm.flow_sde.num_denoise_steps",
        "algorithm.flow_sde.noise_level",
    }


def test_campaign_contract_rejects_unlabeled_sampler_contract() -> None:
    config = EmbodiedExperimentConfig.from_yaml(
        ROOT / "examples/embodied/pi05_libero_long_flow_sde_grpo_1024.yaml"
    )
    broken = config.model_copy(
        update={
                "experiment": config.experiment.model_copy(
                update={"run": "pi05-long-grpo", "tags": ["flow-sde"]}
            )
        }
    )

    report = audit_pi_campaign_contract(
        broken,
        profile_name="pi05-long-behavioral-v1",
    )

    assert report["passed"] is False
    assert {
        row["path"] for row in report["failed_checks"]
    } == {
        "experiment.run.contains_sampler_contract",
        "experiment.tags.contains_sampler_contract",
    }


def test_campaign_contract_rejects_optimizer_semantic_drift() -> None:
    config = EmbodiedExperimentConfig.from_yaml(
        ROOT
        / "examples/embodied/"
        "pi05_libero_long_flow_sde_grpo_rlinf_reference_1024.yaml"
    )
    broken = config.model_copy(
        update={
            "training": config.training.model_copy(
                update={
                    "optimizer": config.training.optimizer.model_copy(
                        update={"epsilon": 1.0e-8, "weight_decay": 0.01}
                    )
                }
            )
        }
    )

    report = audit_pi_campaign_contract(
        broken,
        profile_name="pi05-long-rlinf-reference-v1",
    )

    assert report["passed"] is False
    assert {
        row["path"] for row in report["failed_checks"]
    } == {
        "training.optimizer.epsilon",
        "training.optimizer.weight_decay",
    }


def test_campaign_contract_rejects_split_observability_projects() -> None:
    config = EmbodiedExperimentConfig.from_yaml(
        ROOT / "examples/embodied/pi0_libero_spatial_flow_sde_grpo_1024.yaml"
    )
    broken = config.model_copy(
        update={
            "observability": config.observability.model_copy(
                update={
                    "wandb": config.observability.wandb.model_copy(
                        update={"project": "wrong-wandb-project"}
                    ),
                    "weave": config.observability.weave.model_copy(
                        update={"project": "wrong-weave-project"}
                    ),
                }
            )
        }
    )

    report = audit_pi_campaign_contract(
        broken,
        profile_name="pi0-spatial-behavioral-v1",
    )

    assert report["passed"] is False
    assert {
        row["path"] for row in report["failed_checks"]
    } == {
        "observability.wandb.project_matches_experiment",
        "observability.weave.project_matches_experiment",
    }
