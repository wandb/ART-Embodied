from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from art_embodied.config import EmbodiedExperimentConfig
from art_embodied.experiment import EvaluationResult
from art_embodied.trajectories import Action
from examples.embodied.pi0_fast_sampler_calibration import (
    _condition_config,
    _sampler_diagnostics,
    _temperature_label,
)

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "examples/embodied/pi0_fast_libero_spatial_grpo_development_h100.yaml"


def test_pi0_fast_sampler_calibration_changes_only_decoder_treatment(
    tmp_path: Path,
) -> None:
    base = EmbodiedExperimentConfig.from_yaml(CONFIG)
    sampled = _condition_config(
        base,
        output_dir=tmp_path,
        label="sample-t0p5",
        do_sample=True,
        temperature=0.5,
        episodes=None,
    )

    assert sampled.policy.path == base.policy.path
    assert sampled.policy.revision == base.policy.revision
    assert sampled.environment == base.environment
    assert sampled.evaluation.fixed_scenarios == base.evaluation.fixed_scenarios
    assert sampled.evaluation.seeds == base.evaluation.seeds
    assert sampled.evaluation.episodes == base.evaluation.episodes
    assert sampled.evaluation.deterministic is False
    assert sampled.policy.evaluation_generation.do_sample is True
    assert sampled.policy.evaluation_generation.temperature == pytest.approx(0.5)
    assert sampled.evaluation.temperature == pytest.approx(0.5)
    assert sampled.evaluation.pre_training_success_gate is None
    assert sampled.observability.wandb.enabled is True
    assert sampled.observability.wandb.job_type == "sampler-calibration"
    assert sampled.storage.output_dir == tmp_path / "sample-t0p5"


def test_pi0_fast_sampler_calibration_builds_greedy_baseline(
    tmp_path: Path,
) -> None:
    base = EmbodiedExperimentConfig.from_yaml(CONFIG)
    greedy = _condition_config(
        base,
        output_dir=tmp_path,
        label="greedy",
        do_sample=False,
        temperature=1.0,
        episodes=20,
    )

    assert greedy.evaluation.episodes == 20
    assert greedy.evaluation.deterministic is True
    assert greedy.policy.evaluation_generation.do_sample is False
    assert greedy.evaluation.kwargs["seed_contract"]["policy_mode"] == "derived"
    assert base.evaluation.episodes == 100


@pytest.mark.parametrize(
    ("temperature", "label"),
    [(1.0, "sample-t1"), (0.5, "sample-t0p5"), (0.25, "sample-t0p25")],
)
def test_pi0_fast_sampler_temperature_label(
    temperature: float,
    label: str,
) -> None:
    assert _temperature_label(temperature) == label


def test_pi0_fast_sampler_diagnostics_report_grammar_health() -> None:
    valid = Action(
        step=0,
        kind="token",
        raw={"tokens": [1]},
        metadata={
            "action_grammar_valid": True,
            "generated_token_count": 10,
            "post_termination_tokens_discarded": 2,
        },
    )
    invalid = Action(
        step=1,
        kind="token",
        raw={"tokens": [2]},
        metadata={
            "action_grammar_valid": False,
            "generated_token_count": 20,
            "post_termination_tokens_discarded": 4,
        },
    )
    evaluation = EvaluationResult(
        step=0,
        metrics={"success_rate": 0.0},
        artifacts={},
        trajectories=(
            SimpleNamespace(actions=[valid]),
            SimpleNamespace(actions=[valid, invalid]),
        ),
    )

    assert _sampler_diagnostics(evaluation) == {
        "action_chunks": 3.0,
        "action_grammar_valid_rate": pytest.approx(2 / 3),
        "episodes_with_invalid_action_rate": 0.5,
        "generated_token_count_mean": pytest.approx(40 / 3),
        "post_termination_tokens_discarded_mean": pytest.approx(8 / 3),
    }
