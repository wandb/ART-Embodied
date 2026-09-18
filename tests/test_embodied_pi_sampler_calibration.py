from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from art_embodied.config import EmbodiedExperimentConfig
from examples.embodied.flow_sde_sampler_calibration import (
    _diagnostic_config,
    _load_checkpoint,
    _sampler_gap_passes,
)

ROOT = Path(__file__).parents[1]
PI0_CONFIG = (
    ROOT
    / "examples/embodied/pi0_libero_spatial_flow_sde_grpo_rlinf_positive_control.yaml"
)
SMOLVLA_CONFIG = (
    ROOT
    / "examples/embodied/smolvla_libero_10_sft_baseline_eval_h100.yaml"
)


def test_sampler_calibration_preserves_plan_while_changing_noise(tmp_path: Path) -> None:
    base = EmbodiedExperimentConfig.from_yaml(PI0_CONFIG)
    original_seed_contract = dict(base.evaluation.kwargs["seed_contract"])
    output = tmp_path / "calibration"

    diagnostic = _diagnostic_config(
        base,
        output_dir=output,
        noise_level=0.2,
        denoise_steps=8,
        episodes=10,
        policy_seed_repetitions=1,
    )

    assert diagnostic.algorithm.flow_sde is not None
    assert diagnostic.algorithm.flow_sde.noise_level == 0.2
    assert diagnostic.algorithm.flow_sde.num_denoise_steps == 8
    assert diagnostic.storage.output_dir == output
    assert diagnostic.evaluation.episodes == 10
    assert diagnostic.evaluation.seeds == base.evaluation.seeds
    assert diagnostic.evaluation.fixed_scenarios == base.evaluation.fixed_scenarios
    assert diagnostic.evaluation.kwargs["seed_contract"] == {
        **original_seed_contract,
        "policy_mode": "derived",
    }
    assert base.evaluation.kwargs["seed_contract"] == original_seed_contract
    assert diagnostic.evaluation.baseline_outcomes_path is None
    assert diagnostic.evaluation.data_role == "diagnostic"
    assert base.algorithm.flow_sde.noise_level == 0.5
    assert diagnostic.policy == base.policy


def test_sampler_calibration_repeats_fixed_plan_with_new_policy_seeds(
    tmp_path: Path,
) -> None:
    base = EmbodiedExperimentConfig.from_yaml(PI0_CONFIG)

    diagnostic = _diagnostic_config(
        base,
        output_dir=tmp_path / "calibration",
        noise_level=None,
        denoise_steps=None,
        episodes=None,
        policy_seed_repetitions=5,
    )

    assert diagnostic.evaluation.episodes == 5 * len(
        base.evaluation.fixed_scenarios
    )


def test_sampler_calibration_accepts_smolvla_contract(tmp_path: Path) -> None:
    base = EmbodiedExperimentConfig.from_yaml(SMOLVLA_CONFIG)

    diagnostic = _diagnostic_config(
        base,
        output_dir=tmp_path / "smolvla-calibration",
        noise_level=0.15,
        denoise_steps=12,
        episodes=20,
    )

    assert diagnostic.policy.type == "smolvla"
    assert diagnostic.algorithm.flow_sde is not None
    assert diagnostic.algorithm.flow_sde.noise_level == 0.15
    assert diagnostic.algorithm.flow_sde.num_denoise_steps == 12
    assert diagnostic.evaluation.episodes == 20
    assert diagnostic.evaluation.fixed_scenarios == base.evaluation.fixed_scenarios


def test_sampler_calibration_loads_requested_policy_snapshot(tmp_path: Path) -> None:
    loaded: list[Path] = []
    policy = SimpleNamespace(load_checkpoint=loaded.append)
    snapshot = tmp_path / "adapter"

    _load_checkpoint(policy, snapshot, policy_type="smolvla")

    assert loaded == [snapshot.resolve()]


@pytest.mark.parametrize(
    ("lift", "limit", "expected"),
    [
        (-0.10, 0.10, True),
        (0.10, 0.10, True),
        (-0.10001, 0.10, False),
        (0.4, None, True),
    ],
)
def test_sampler_calibration_gap_gate(
    lift: float,
    limit: float | None,
    expected: bool,
) -> None:
    assert (
        _sampler_gap_passes(
            {"success_rate_lift": lift},
            maximum_absolute_gap=limit,
        )
        is expected
    )
