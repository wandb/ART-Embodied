from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from examples.embodied.pi_flow_sde_action_gap import action_gap_summary


def test_action_gap_summary_separates_bias_and_stochastic_variance() -> None:
    ode = np.zeros((2, 2, 3, 2), dtype=np.float64)
    sde = np.stack(
        (np.full((2, 3, 2), -1.0), np.full((2, 3, 2), 3.0)), axis=1
    )

    summary = action_gap_summary(ode, sde)

    assert summary["states"] == 2
    assert summary["sde_samples_per_state"] == 2
    assert summary["horizon"] == 3
    assert summary["action_dim"] == 2
    assert summary["sde_mean_vs_ode_bias_rmse"] == pytest.approx(1.0)
    assert summary["paired_sde_vs_ode_rmse"] == pytest.approx(np.sqrt(5.0))
    assert summary["ode_within_state_std_rms"] == 0.0
    assert summary["sde_within_state_std_rms"] == pytest.approx(2.0)
    assert len(summary["per_action_dimension"]) == 2


def test_action_gap_summary_rejects_incompatible_shapes() -> None:
    with pytest.raises(ValueError, match="identical four-dimensional shapes"):
        action_gap_summary(np.zeros((2, 2, 3, 2)), np.zeros((1, 2, 3, 2)))
