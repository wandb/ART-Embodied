from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from examples.embodied.pi_cross_runtime_compare import compare_bundles
from examples.embodied.pi_cross_runtime_dump import postprocess_art_actions


def _bundle(offset: float = 0.0):
    tensor = torch.tensor([[1.0 + offset, 2.0]])
    return {
        "schema_version": 1,
        "runtime": "test",
        "images": [tensor],
        "image_masks": [torch.tensor([[True, False]])],
        "language_tokens": torch.tensor([[1, 2]]),
        "language_masks": torch.tensor([[True, True]]),
        "model_state": tensor,
        "initial_noise": tensor,
        "flow_states": tensor,
        "velocities": tensor,
        "normalized_actions": tensor,
        "native_actions": tensor,
    }


def test_compare_bundles_reports_exact_and_nonzero_deltas():
    exact = compare_bundles(_bundle(), _bundle())
    assert exact["comparisons"]["velocities"]["max_abs"] == 0.0

    changed = compare_bundles(_bundle(), _bundle(offset=0.25))
    assert changed["comparisons"]["velocities"]["max_abs"] == 0.25
    assert changed["comparisons"]["images"]["items"][0]["max_abs"] == 0.25


def test_art_cross_runtime_postprocess_uses_environment_action_width():
    class Policy:
        action_dim = 7
        config = type("Config", (), {"use_relative_actions": False})()

        @staticmethod
        def postprocessor(actions):
            return actions

    normalized = torch.zeros(2, 50, 32)
    native = postprocess_art_actions(
        Policy(),
        normalized,
        prepared_rows=[
            {"observation.state": torch.zeros(1, 7)},
            {"observation.state": torch.zeros(1, 7)},
        ],
    )

    assert native.shape == (2, 50, 7)


def test_compare_bundles_compares_common_native_execution_horizon():
    art = _bundle()
    rlinf = _bundle()
    art["native_actions"] = torch.arange(2 * 5 * 7).reshape(2, 5, 7)
    rlinf["native_actions"] = art["native_actions"][:, :2].clone()

    comparison = compare_bundles(art, rlinf)

    assert comparison["comparisons"]["native_actions"]["shape_match"] is False
    executed = comparison["comparisons"]["native_actions_executed"]
    assert executed["shape"] == [2, 2, 7]
    assert executed["max_abs"] == 0.0
