"""Bounded console progress for long Flow-SDE optimizer updates."""

from __future__ import annotations

import math
import time
from typing import Mapping


def emit_flow_sde_alignment(
    *,
    update: int,
    mean_abs_delta: float,
    max_abs_delta: float,
    tolerance: float,
    ratio_mean: float = 1.0,
    ratio_tolerance: float | None = None,
    passed: bool,
) -> None:
    """Report the one pre-update alignment gate without touching W&B history."""

    print(
        " ".join(
            [
                "[ART-Embodied]",
                f"update={int(update)}",
                "phase=training",
                "status=alignment_passed" if passed else "status=alignment_failed",
                f"mean_abs_delta={float(mean_abs_delta):.6g}",
                f"max_abs_delta={float(max_abs_delta):.6g}",
                f"tolerance={float(tolerance):.6g}",
                f"ratio_mean={float(ratio_mean):.6g}",
                (
                    f"ratio_tolerance={float(ratio_tolerance):.6g}"
                    if ratio_tolerance is not None
                    else "ratio_tolerance=disabled"
                ),
            ]
        ),
        flush=True,
    )


def emit_flow_sde_training_progress(
    *,
    update: int,
    completed: int,
    total: int,
    metrics: Mapping[str, float],
    started_at: float,
    max_events: int = 12,
) -> None:
    """Print sparse progress without creating extra W&B history steps."""

    if not should_emit_flow_sde_training_progress(
        completed=completed,
        total=total,
        max_events=max_events,
    ):
        return
    fields = [
        "[ART-Embodied]",
        f"update={int(update)}",
        "phase=training",
        "status=progress",
        f"subupdates={int(completed)}/{int(total)}",
    ]
    for key in (
        "loss",
        "ratio_mean",
        "approximate_kl",
        "approximate_kl_per_primitive",
        "clip_fraction",
    ):
        value = metrics.get(key)
        if value is not None and math.isfinite(float(value)):
            fields.append(f"{key}={float(value):.6g}")
    fields.append(f"elapsed_seconds={max(0.0, time.perf_counter() - started_at):.1f}")
    print(" ".join(fields), flush=True)


def emit_flow_sde_trust_region_stop(
    *,
    update: int,
    applied_subupdates: int,
    planned_subupdates: int,
    approximate_kl: float,
    approximate_kl_per_primitive: float,
    scope: str,
    threshold: float,
) -> None:
    """Report a pre-optimizer KL stop without consuming a W&B history row."""

    print(
        " ".join(
            [
                "[ART-Embodied]",
                f"update={int(update)}",
                "phase=training",
                "status=trust_region_stopped",
                f"applied_subupdates={int(applied_subupdates)}/{int(planned_subupdates)}",
                f"probe_approximate_kl={float(approximate_kl):.6g}",
                f"probe_abs_approximate_kl={abs(float(approximate_kl)):.6g}",
                "probe_approximate_kl_per_primitive="
                f"{float(approximate_kl_per_primitive):.6g}",
                f"guard_scope={scope}",
                f"max_approximate_kl={float(threshold):.6g}",
            ]
        ),
        flush=True,
    )


def should_emit_flow_sde_training_progress(
    *,
    completed: int,
    total: int,
    max_events: int = 12,
) -> bool:
    """Bound progress output while always reporting the first and final step."""

    if total < 1 or completed < 1 or completed > total:
        return False
    if completed == 1 or completed == total:
        return True
    interval = max(1, math.ceil(total / max(1, max_events)))
    return completed % interval == 0
