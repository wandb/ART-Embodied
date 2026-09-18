from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
DEVELOPMENT_CONFIG = ROOT / (
    "examples/embodied/gr00t_n1d7_robocasa_gr1_tabletop_"
    "cuttingboard_pan_noise01_u20_development.yaml"
)
CONTINUATION_CONFIG = ROOT / (
    "examples/embodied/robocasa/development_experiments/"
    "gr1_tabletop_cuttingboard_pan_noise01_u100_continuation_v1.yaml"
)
SEALED_MANIFEST = ROOT / (
    "examples/embodied/robocasa/sealed_tests/gr1_tabletop_cuttingboard_pan_u100_v1.json"
)


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_single_task_development_recipe_is_the_validated_flow_sde_contract() -> None:
    config = _yaml(DEVELOPMENT_CONFIG)

    assert config["policy"]["type"] == "gr00t_n1d7"
    assert config["policy"]["lora"]["rank"] == 64
    assert config["policy"]["lora"]["alpha"] == 64
    assert config["algorithm"]["group_size"] == 8
    assert config["algorithm"]["flow_sde"] == {
        "noise_level": 0.1,
        "num_denoise_steps": 4,
        "stochastic_transitions_per_sample": 1,
        "selected_step_sampling": "uniform",
        "joint_logprob": False,
    }
    assert config["training"]["updates"] == 20
    assert config["training"]["optimizer"]["learning_rate"] == 3e-5
    assert len(config["evaluation"]["seeds"]) == 64


def test_u100_continuation_changes_only_the_preregistered_training_horizon() -> None:
    development = _yaml(DEVELOPMENT_CONFIG)
    continuation = _yaml(CONTINUATION_CONFIG)

    assert continuation["training"]["updates"] == 100
    assert continuation["storage"]["resume_from_checkpoint"].endswith(
        "checkpoints/step-000020"
    )
    assert continuation["storage"]["output_dir"] == development["storage"]["output_dir"]
    assert continuation["policy"] == development["policy"]
    assert continuation["algorithm"] == development["algorithm"]
    assert continuation["rollout"] == development["rollout"]
    assert continuation["evaluation"]["seeds"] == development["evaluation"]["seeds"]


def test_sealed_panel_is_frozen_disjoint_and_hash_bound() -> None:
    manifest = json.loads(SEALED_MANIFEST.read_text(encoding="utf-8"))
    sealed_seeds = manifest["seeds"]
    development_seeds = manifest["excluded_development_seeds"]

    assert manifest["data_role"] == "sealed_test"
    assert manifest["episodes"] == 192
    assert sealed_seeds == list(range(10000, 10192))
    assert len(sealed_seeds) == len(set(sealed_seeds))
    assert set(sealed_seeds).isdisjoint(development_seeds)
    assert manifest["sealed_outcomes_observed"] is False
    assert manifest["policy_selection_uses_sealed_outcomes"] is False

    for path_key, digest_key in (
        ("baseline_config", "baseline_config_sha256"),
        ("candidate_config", "candidate_config_sha256"),
        ("candidate_preregistration", "candidate_preregistration_sha256"),
        ("continuation_config", "continuation_config_sha256"),
    ):
        assert _sha256(ROOT / manifest[path_key]) == manifest[digest_key]
