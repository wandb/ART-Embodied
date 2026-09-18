"""Core data models for embodied agentic RL trajectories.

These models intentionally avoid depending on any one robotics stack. They are
small containers that can wrap LeRobot, OpenPI, OpenVLA, gymnasium, LIBERO,
ManiSkill, Isaac Lab, or real-robot rollouts without forcing users to port their
existing environment or policy implementation into ART.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import datetime
import time
import traceback
from typing import Any, Literal

import pydantic

from .utils import make_json_safe

try:
    from art.trajectories import PydanticException
except ImportError:

    class PydanticException(pydantic.BaseModel):
        type: str
        message: str
        traceback: str


Message = dict[str, Any]

ObservationKind = Literal[
    "image",
    "state",
    "depth",
    "segmentation",
    "text",
    "video",
    "custom",
]
ActionKind = Literal["token", "continuous", "chunk", "tool", "stop", "custom"]
RewardSource = Literal[
    "env",
    "human",
    "verifier",
    "reward_model",
    "llm_judge",
    "script",
]
TrainableTarget = Literal[
    "assistant_tokens",
    "action_tokens",
    "continuous_actions",
    "tool_decisions",
    "stop_decision",
    "custom",
]
MediaKind = Literal["image", "video", "audio", "array", "html", "custom"]


class _JsonSafeModel(pydantic.BaseModel):
    @pydantic.field_serializer("metadata", check_fields=False)
    def _serialize_metadata(self, value: Any) -> Any:
        return make_json_safe(value)


class MediaRef(_JsonSafeModel):
    """Reference to media associated with an embodied trajectory.

    Media should normally be stored as files, W&B artifacts, or Weave objects.
    Keeping references here prevents trajectories from becoming huge opaque
    blobs while still making trace rendering and debugging straightforward.
    """

    uri: str
    kind: MediaKind
    step: int | None = None
    mime_type: str | None = None
    caption: str | None = None
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)


class Observation(_JsonSafeModel):
    """A single observation emitted by an embodied environment or tool."""

    step: int
    kind: ObservationKind
    value: Any | None = None
    media: list[MediaRef] = pydantic.Field(default_factory=list)
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)

    @pydantic.field_serializer("value")
    def _serialize_value(self, value: Any) -> Any:
        return make_json_safe(value)


class Action(_JsonSafeModel):
    """A policy action, tool action, stop decision, or decoded robot action."""

    step: int
    kind: ActionKind
    raw: Any
    decoded: Any | None = None
    logprobs: Any | None = None
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)

    @pydantic.field_serializer("raw", "decoded", "logprobs")
    def _serialize_payloads(self, value: Any) -> Any:
        return make_json_safe(value)


class EmbodiedToolCall(_JsonSafeModel):
    """A tool call made during an embodied rollout."""

    step: int
    name: str
    arguments: dict[str, Any] = pydantic.Field(default_factory=dict)
    output: Any | None = None
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)

    @pydantic.field_serializer("arguments", "output")
    def _serialize_tool_payloads(self, value: Any) -> Any:
        return make_json_safe(value)


class RewardEvent(_JsonSafeModel):
    """A reward contribution with provenance."""

    name: str
    value: float
    source: RewardSource
    step: int | None = None
    explanation: str | None = None
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)


class TrainableSpan(_JsonSafeModel):
    """Marks the parts of a trajectory a backend is allowed to optimize."""

    target: TrainableTarget
    start_step: int | None = None
    end_step: int | None = None
    backend_hint: str | None = None
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)


class EmbodiedTrajectory(_JsonSafeModel):
    """A multimodal trajectory for Physical AI / VLA agentic RL.

    The field names mirror ART's text trajectory where possible (`reward`,
    `metrics`, `metadata`, `logs`) while adding embodied-specific observations,
    actions, tool calls, reward events, media references, and trainable spans.
    """

    task: str
    messages: list[Message] = pydantic.Field(default_factory=list)
    observations: list[Observation] = pydantic.Field(default_factory=list)
    actions: list[Action] = pydantic.Field(default_factory=list)
    tool_calls: list[EmbodiedToolCall] = pydantic.Field(default_factory=list)
    rewards: list[RewardEvent] = pydantic.Field(default_factory=list)
    reward: float = 0.0
    media: list[MediaRef] = pydantic.Field(default_factory=list)
    trainable_spans: list[TrainableSpan] = pydantic.Field(default_factory=list)
    initial_policy_version: int | None = None
    final_policy_version: int | None = None
    metrics: dict[str, float | int | bool] = pydantic.Field(default_factory=dict)
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)
    logs: list[str] = pydantic.Field(default_factory=list)
    start_time: datetime = pydantic.Field(default_factory=datetime.now, exclude=True)

    @property
    def final_reward(self) -> float:
        """Alias for `reward` used by embodied RL literature."""

        return self.reward

    def log(self, message: str) -> None:
        self.logs.append(message)

    def discard_observation_values(self) -> int:
        """Release raw observation payloads while preserving trajectory evidence.

        Media references, observation metadata, actions, rewards, and backend
        replay metadata remain available. Callers must first establish that the
        selected training backend does not consume ``Observation.value``.
        """

        discarded = 0
        for observation in self.observations:
            if observation.value is not None:
                observation.value = None
                discarded += 1
        return discarded

    def finish(self) -> "EmbodiedTrajectory":
        duration = (datetime.now() - self.start_time).total_seconds()
        self.metrics["duration"] = duration
        return self

    def add_reward(
        self,
        name: str,
        value: float,
        source: RewardSource,
        *,
        step: int | None = None,
        explanation: str | None = None,
        metadata: dict[str, Any] | None = None,
        update_total: bool = True,
    ) -> RewardEvent:
        event = RewardEvent(
            name=name,
            value=value,
            source=source,
            step=step,
            explanation=explanation,
            metadata=metadata or {},
        )
        self.rewards.append(event)
        if update_total:
            self.reward += value
        return event

    def recompute_reward(self, *, reducer: Literal["sum", "last"] = "sum") -> float:
        if not self.rewards:
            self.reward = 0.0
        elif reducer == "sum":
            self.reward = sum(event.value for event in self.rewards)
        elif reducer == "last":
            self.reward = self.rewards[-1].value
        else:
            raise ValueError(f"Unsupported reducer: {reducer}")
        return self.reward

    def for_logging(self) -> dict[str, Any]:
        """Return a compact dict intended for console, W&B, or Weave logging."""

        return {
            "task": self.task,
            "reward": self.reward,
            "initial_policy_version": self.initial_policy_version,
            "final_policy_version": self.final_policy_version,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "messages": self.messages,
            "observations": [obs.model_dump(mode="json") for obs in self.observations],
            "actions": [action.model_dump(mode="json") for action in self.actions],
            "tool_calls": [call.model_dump(mode="json") for call in self.tool_calls],
            "rewards": [reward.model_dump(mode="json") for reward in self.rewards],
            "media": [media.model_dump(mode="json") for media in self.media],
            "trainable_spans": [
                span.model_dump(mode="json") for span in self.trainable_spans
            ],
            "logs": self.logs,
        }

    async def __aenter__(self) -> "EmbodiedTrajectory":
        self._context_start_time = time.monotonic()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.metrics["duration"] = time.monotonic() - self._context_start_time


class EmbodiedTrajectoryGroup(_JsonSafeModel):
    """A group of embodied trajectories, usually multiple attempts at one task."""

    trajectories: list[EmbodiedTrajectory]
    exceptions: list[PydanticException] = pydantic.Field(default_factory=list)
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)
    metrics: dict[str, float | int | bool] = pydantic.Field(default_factory=dict)
    logs: list[str] = pydantic.Field(default_factory=list)

    def __init__(
        self,
        trajectories: Iterable[EmbodiedTrajectory | BaseException],
        *,
        exceptions: list[BaseException] | None = None,
        metadata: dict[str, Any] | None = None,
        metrics: dict[str, float | int | bool] | None = None,
        logs: list[str] | None = None,
    ) -> None:
        trajectory_items = list(trajectories)
        exception_items = [
            item for item in trajectory_items if isinstance(item, BaseException)
        ] + (exceptions or [])
        super().__init__(
            trajectories=[
                item
                for item in trajectory_items
                if isinstance(item, EmbodiedTrajectory)
            ],
            exceptions=[
                PydanticException(
                    type=str(type(exception)),
                    message=str(exception),
                    traceback="".join(
                        traceback.format_exception(
                            type(exception), exception, exception.__traceback__
                        )
                    ),
                )
                for exception in exception_items
            ],
            metadata=metadata or {},
            metrics=metrics or {},
            logs=logs or [],
        )

    def log(self, message: str) -> None:
        self.logs.append(message)

    def __iter__(self) -> Iterator[EmbodiedTrajectory]:  # type: ignore[override]
        return iter(self.trajectories)

    def __len__(self) -> int:
        return len(self.trajectories)

    def rewards(self) -> list[float]:
        return [trajectory.reward for trajectory in self.trajectories]

    def for_logging(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "metrics": self.metrics,
            "logs": self.logs,
            "exceptions": [exc.model_dump(mode="json") for exc in self.exceptions],
            "trajectories": [
                trajectory.for_logging() for trajectory in self.trajectories
            ],
        }
