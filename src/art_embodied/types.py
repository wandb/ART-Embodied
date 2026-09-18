"""Lightweight embodied training result types.

The upstream ART ``art.types`` module also defines these dataclasses, but it
imports OpenAI chat schema types for text-agent workflows. Robotics containers
often do not install those LLM-only dependencies. This module aliases upstream
types when available and falls back to equivalent local dataclasses otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    from art.types import (  # type: ignore[assignment]
        LocalTrainResult,
        ServerlessTrainResult,
        TrainResult,
    )
except ImportError:

    @dataclass
    class TrainResult:
        """Base result returned from an embodied backend.train() call."""

        step: int
        metrics: dict[str, float] = field(default_factory=dict)

    @dataclass
    class LocalTrainResult(TrainResult):
        """Result from local embodied training."""

        checkpoint_path: str | None = None

    @dataclass
    class ServerlessTrainResult(TrainResult):
        """Result from serverless embodied training."""

        artifact_name: str | None = None
