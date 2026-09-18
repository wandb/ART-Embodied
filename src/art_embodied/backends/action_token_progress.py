"""Bounded local progress diagnostics for long action-token updates."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any


def _maybe_write_action_token_train_progress(
    *,
    progress_path: Path | None,
    progress_every_microbatches: int,
    progress_max_bytes: int,
    microbatch_count: int,
    microbatch_start: int,
    total_items: int,
    microbatch_size: int,
    logprob_forward_seconds: float,
    loss_backward_seconds: float,
    total_loss_value: float,
    streamed_trajectory_span_backward: bool,
) -> None:
    """Emit bounded, YAML-configured progress for long gradient jobs."""

    processed_items = min(
        int(total_items), int(microbatch_start) + int(microbatch_size)
    )
    if not _should_write_action_token_train_progress(
        progress_path=progress_path,
        progress_every_microbatches=progress_every_microbatches,
        microbatch_count=microbatch_count,
        processed_items=processed_items,
        total_items=total_items,
    ):
        return
    record = {
        "event": "action_token_grpo_microbatch_progress",
        "time_unix": time.time(),
        "microbatch_count": int(microbatch_count),
        "microbatch_start": int(microbatch_start),
        "microbatch_size": int(microbatch_size),
        "processed_items": int(processed_items),
        "total_items": int(total_items),
        "progress_fraction": float(processed_items) / float(max(1, int(total_items))),
        "logprob_forward_seconds": float(logprob_forward_seconds),
        "loss_backward_seconds": float(loss_backward_seconds),
        "total_loss_value": float(total_loss_value),
        "streamed_trajectory_span_backward": bool(streamed_trajectory_span_backward),
    }
    _write_action_token_train_progress_record(
        record,
        progress_path=progress_path,
        progress_max_bytes=progress_max_bytes,
    )


def _maybe_write_action_token_train_microbatch_start(
    *,
    progress_path: Path | None,
    progress_every_microbatches: int,
    progress_max_bytes: int,
    microbatch_count: int,
    microbatch_start: int,
    total_items: int,
    microbatch_size: int,
    streamed_trajectory_span_backward: bool,
) -> None:
    """Emit an opt-in progress event before a long logprob forward starts."""

    processed_items = min(int(total_items), int(microbatch_start))
    if not _should_write_action_token_train_progress(
        progress_path=progress_path,
        progress_every_microbatches=progress_every_microbatches,
        microbatch_count=microbatch_count,
        processed_items=processed_items,
        total_items=total_items,
    ):
        return
    record = {
        "event": "action_token_grpo_microbatch_start",
        "time_unix": time.time(),
        "microbatch_count": int(microbatch_count),
        "microbatch_start": int(microbatch_start),
        "microbatch_size": int(microbatch_size),
        "processed_items": int(processed_items),
        "total_items": int(total_items),
        "progress_fraction": float(processed_items) / float(max(1, int(total_items))),
        "streamed_trajectory_span_backward": bool(streamed_trajectory_span_backward),
    }
    _write_action_token_train_progress_record(
        record,
        progress_path=progress_path,
        progress_max_bytes=progress_max_bytes,
    )


def _should_write_action_token_train_progress(
    *,
    progress_path: Path | None,
    progress_every_microbatches: int,
    microbatch_count: int,
    processed_items: int,
    total_items: int,
) -> bool:
    if progress_path is None:
        return False
    every = max(1, int(progress_every_microbatches))
    if int(microbatch_count) <= 1:
        return True
    if int(processed_items) >= int(total_items):
        return True
    return int(microbatch_count) % every == 0


def _write_action_token_train_progress_record(
    record: dict[str, Any],
    *,
    progress_path: Path | None,
    progress_max_bytes: int,
) -> None:
    if progress_path is None:
        return
    try:
        path = Path(progress_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, sort_keys=True) + "\n"
        line_size = len(line.encode("utf-8"))
        existing_size = path.stat().st_size if path.exists() else 0
        if existing_size + line_size > int(progress_max_bytes):
            return
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except OSError:
        return


def _maybe_write_action_token_rescore_progress(
    *,
    progress_path: Path | None,
    progress_every_microbatches: int,
    progress_max_bytes: int,
    event: str,
    microbatch_count: int,
    microbatch_start: int,
    total_items: int,
    microbatch_size: int,
    training_unit: str,
    source: str,
    elapsed_seconds: float,
) -> None:
    """Emit opt-in progress for old-logprob recomputation.

    Full embodied action-token rollouts can contain thousands of action spans, so
    old-logprob recomputation is long enough to look like a hang without
    periodic progress records.
    """

    processed_items = min(
        int(total_items), int(microbatch_start) + int(microbatch_size)
    )
    if event.endswith("_started"):
        processed_items = min(int(total_items), int(microbatch_start))
    if not _should_write_action_token_train_progress(
        progress_path=progress_path,
        progress_every_microbatches=progress_every_microbatches,
        microbatch_count=microbatch_count,
        processed_items=processed_items,
        total_items=total_items,
    ):
        return
    _write_action_token_train_progress_record(
        {
            "event": event,
            "time_unix": time.time(),
            "microbatch_count": int(microbatch_count),
            "microbatch_start": int(microbatch_start),
            "microbatch_size": int(microbatch_size),
            "processed_items": int(processed_items),
            "total_items": int(total_items),
            "progress_fraction": float(processed_items)
            / float(max(1, int(total_items))),
            "training_unit": str(training_unit),
            "source": str(source),
            "elapsed_seconds": float(elapsed_seconds),
        },
        progress_path=progress_path,
        progress_max_bytes=progress_max_bytes,
    )
