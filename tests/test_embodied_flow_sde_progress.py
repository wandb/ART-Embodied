from __future__ import annotations

import time

from art_embodied.backends.flow_sde_progress import (
    emit_flow_sde_alignment,
    emit_flow_sde_training_progress,
    emit_flow_sde_trust_region_stop,
    should_emit_flow_sde_training_progress,
)


def test_flow_sde_alignment_prints_without_creating_logging_state(capsys) -> None:
    emit_flow_sde_alignment(
        update=2,
        mean_abs_delta=0.001,
        max_abs_delta=0.003,
        tolerance=0.02,
        ratio_mean=1.001,
        ratio_tolerance=0.02,
        passed=True,
    )

    output = capsys.readouterr().out
    assert "update=2 phase=training status=alignment_passed" in output
    assert "mean_abs_delta=0.001" in output
    assert "max_abs_delta=0.003" in output
    assert "tolerance=0.02" in output
    assert "ratio_mean=1.001" in output
    assert "ratio_tolerance=0.02" in output


def test_flow_sde_progress_is_bounded_and_includes_endpoints() -> None:
    emitted = [
        completed
        for completed in range(1, 97)
        if should_emit_flow_sde_training_progress(
            completed=completed,
            total=96,
            max_events=12,
        )
    ]

    assert emitted == [1, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96]


def test_flow_sde_progress_prints_without_creating_logging_state(capsys) -> None:
    emit_flow_sde_training_progress(
        update=3,
        completed=1,
        total=24,
        metrics={
            "loss": 0.25,
            "ratio_mean": 1.01,
            "approximate_kl": 0.002,
            "clip_fraction": 0.125,
        },
        started_at=time.perf_counter(),
    )

    output = capsys.readouterr().out
    assert "update=3 phase=training status=progress" in output
    assert "subupdates=1/24" in output
    assert "loss=0.25" in output
    assert "ratio_mean=1.01" in output


def test_flow_sde_trust_region_stop_is_explicit(capsys) -> None:
    emit_flow_sde_trust_region_stop(
        update=1,
        applied_subupdates=1,
        planned_subupdates=24,
        approximate_kl=0.063,
        approximate_kl_per_primitive=0.0063,
        scope="primitive_action",
        threshold=0.05,
    )

    output = capsys.readouterr().out
    assert "status=trust_region_stopped" in output
    assert "applied_subupdates=1/24" in output
    assert "probe_approximate_kl=0.063" in output
    assert "probe_abs_approximate_kl=0.063" in output
    assert "probe_approximate_kl_per_primitive=0.0063" in output
    assert "guard_scope=primitive_action" in output
    assert "max_approximate_kl=0.05" in output
