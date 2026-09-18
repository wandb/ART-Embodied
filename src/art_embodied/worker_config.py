"""Project experiment configuration onto an ephemeral worker process."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def discard_coordinator_resume(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a worker-safe config without the coordinator's resume command.

    ``storage.resume_from_checkpoint`` is an instruction to the coordinator to
    restore policy and optimizer state once at process startup. Ephemeral
    rollout and gradient workers receive current policy snapshots separately;
    making them revalidate the original resume directory couples every worker
    restart to a checkpoint that retention may legitimately remove.
    """

    projected = dict(config)
    storage = dict(projected["storage"])
    storage["resume_from_checkpoint"] = None
    projected["storage"] = storage
    return projected
