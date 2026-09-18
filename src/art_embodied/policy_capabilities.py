"""Explicit probability and action contracts between policies and backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

ProbabilityModel = Literal[
    "categorical_tokens",
    "gaussian_flow_sde",
    "deterministic_continuous",
    "unknown",
]


@dataclass(frozen=True, slots=True)
class PolicyCapabilities:
    """Capabilities that materially determine whether an RL objective is valid."""

    policy_type: str
    action_kind: Literal["token", "continuous"]
    probability_model: ProbabilityModel
    action_shape: tuple[int, ...]
    chunk_horizon: int
    exact_logprobs: bool
    teacher_forced_rescore: bool
    rng_replay: bool
    batched_inference: bool
    checkpoint_delta: Literal["lora", "full", "either", "unknown"]
    observation_normalization_revision: str | None = None

    def __post_init__(self) -> None:
        if not self.policy_type:
            raise ValueError("policy_type cannot be empty")
        if self.chunk_horizon < 1:
            raise ValueError("chunk_horizon must be positive")
        if not self.action_shape or any(
            dimension < 1 for dimension in self.action_shape
        ):
            raise ValueError("action_shape must contain positive dimensions")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BackendRequirements:
    """Probability contract required by one registered objective backend."""

    action_kind: Literal["token", "continuous"]
    probability_models: frozenset[ProbabilityModel]
    exact_logprobs: bool = True
    teacher_forced_rescore: bool = True
    rng_replay: bool = False

    def validate(self, capabilities: PolicyCapabilities) -> None:
        errors: list[str] = []
        if capabilities.action_kind != self.action_kind:
            errors.append(
                f"action_kind={capabilities.action_kind!r}, expected {self.action_kind!r}"
            )
        if capabilities.probability_model not in self.probability_models:
            expected = ", ".join(sorted(self.probability_models))
            errors.append(
                f"probability_model={capabilities.probability_model!r}, expected one of {expected}"
            )
        for field in ("exact_logprobs", "teacher_forced_rescore", "rng_replay"):
            if getattr(self, field) and not getattr(capabilities, field):
                errors.append(f"{field}=false")
        if errors:
            raise ValueError(
                f"Policy {capabilities.policy_type!r} does not satisfy backend "
                f"requirements: {'; '.join(errors)}"
            )
