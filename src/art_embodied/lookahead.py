"""Policy- and simulator-neutral contracts for action-chunk lookahead media."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class LookaheadFrame:
    """A rendered speculative state and the chunk action that produced it."""

    image: Any
    action_index: int


@runtime_checkable
class ActionChunkLookaheadPreview(Protocol):
    """Render an action chunk in an isolated copy of a live environment.

    Implementations own simulator-specific state synchronization. Calling this
    method must not mutate ``source_environment``: the returned frames are
    observability artifacts, never part of the rollout transition.
    """

    def preview_action_chunk(
        self,
        action_chunk: Any,
        *,
        source_environment: Any,
        execution_horizon: int,
        frame_stride: int,
        max_frames: int,
    ) -> Sequence[LookaheadFrame]: ...


def unused_chunk_length(action_chunk: Any, execution_horizon: int) -> int:
    """Return how many predicted actions remain after the executed prefix."""

    shape = getattr(action_chunk, "shape", None)
    if shape is None or len(shape) < 1:
        try:
            horizon = len(action_chunk)
        except TypeError:
            return 0
    else:
        horizon = int(shape[0])
    return max(0, horizon - int(execution_horizon))
