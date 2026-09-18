"""LeRobot-first process actor built from application-owned components."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import importlib
import inspect
from pathlib import Path
from typing import Any

from art_embodied.config import EmbodiedExperimentConfig
from art_embodied.experiment import EmbodiedScenario, RolloutContext
from art_embodied.rollout_process import RolloutActorProcessContext
from art_embodied.trajectories import EmbodiedTrajectory

from .lerobot_rollout import (
    EnvironmentFactory,
    LeRobotEpisodeRollout,
    ObservationRecorder,
    PolicyAdapterFactory,
    SuccessFunction,
    TransitionRecorder,
)

SnapshotLoader = Callable[..., Any | Awaitable[Any]]
PhasePreparer = Callable[[str], Any | Awaitable[Any]]
CloseFunction = Callable[[], Any | Awaitable[Any]]
GroupRolloutFunction = Callable[..., Any | Awaitable[Any]]
OffloadFunction = Callable[[], Any | Awaitable[Any]]
RestoreFunction = Callable[[], Any | Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class LeRobotProcessComponents:
    """Native LeRobot components owned by one isolated rollout process.

    An application supplies these components because it owns model loading,
    processors, and simulator construction. ART owns the complete episode loop,
    trajectory recording, bounded video capture, and train/eval separation.
    """

    environment_factory: EnvironmentFactory
    policy_adapter_factory: PolicyAdapterFactory
    load_policy_snapshot: SnapshotLoader
    prepare_phase: PhasePreparer | None = None
    success_fn: SuccessFunction | None = None
    observation_recorder: ObservationRecorder | None = None
    transition_recorder: TransitionRecorder | None = None
    group_rollout: GroupRolloutFunction | None = None
    offload: OffloadFunction | None = None
    restore: RestoreFunction | None = None
    close: CloseFunction | None = None


class LeRobotProcessRolloutActor:
    """Adapt LeRobot components to ART's process-isolated actor protocol."""

    def __init__(
        self,
        *,
        config: EmbodiedExperimentConfig,
        components: LeRobotProcessComponents,
    ) -> None:
        self.components = components
        shared = {
            "config": config,
            "environment_factory": components.environment_factory,
            "policy_adapter_factory": components.policy_adapter_factory,
            "training_policy": None,
            "success_fn": components.success_fn,
            "observation_recorder": components.observation_recorder,
            "transition_recorder": components.transition_recorder,
        }
        self.rollouts = {
            "train": LeRobotEpisodeRollout.from_config(phase="train", **shared),
            "eval": LeRobotEpisodeRollout.from_config(phase="eval", **shared),
        }
        self.prepared_update: int | None = None

    async def prepare_update(
        self,
        *,
        update: int,
        policy_snapshot: Path,
    ) -> None:
        """Load the exact ART checkpoint before serving an update."""

        await _await_if_needed(
            self.components.load_policy_snapshot(
                update=update,
                policy_snapshot=policy_snapshot,
            )
        )
        self.prepared_update = int(update)

    async def rollout(
        self,
        scenario: EmbodiedScenario,
        context: RolloutContext,
        *,
        phase: str,
        policy_client: Any | None = None,
    ) -> EmbodiedTrajectory:
        if self.prepared_update != context.update:
            raise RuntimeError(
                "LeRobot rollout policy is stale: "
                f"prepared_update={self.prepared_update}, "
                f"requested_update={context.update}"
            )
        if self.components.prepare_phase is not None:
            await _await_if_needed(self.components.prepare_phase(phase))
        if self.components.group_rollout is not None:
            result = list(
                await _await_if_needed(
                    self.components.group_rollout(
                        scenario=scenario,
                        contexts=(context,),
                        phase=phase,
                        policy_client=policy_client,
                    )
                )
            )
            if len(result) != 1 or not isinstance(result[0], EmbodiedTrajectory):
                raise TypeError(
                    "LeRobot singleton group_rollout must return one EmbodiedTrajectory"
                )
            return result[0]
        if policy_client is not None:
            raise RuntimeError(
                "LeRobot batched-server inference requires an "
                "application-owned group_rollout"
            )
        try:
            rollout = self.rollouts[phase]
        except KeyError as exc:
            raise ValueError(f"Unsupported rollout phase: {phase!r}") from exc
        return await rollout(scenario, context)

    async def rollout_group(
        self,
        scenario: EmbodiedScenario,
        contexts: tuple[RolloutContext, ...],
        *,
        phase: str,
        policy_client: Any | None = None,
    ) -> list[EmbodiedTrajectory]:
        if self.prepared_update is None or any(
            context.update != self.prepared_update for context in contexts
        ):
            raise RuntimeError("LeRobot rollout-group policy is stale")
        if self.components.prepare_phase is not None:
            await _await_if_needed(self.components.prepare_phase(phase))
        if self.components.group_rollout is not None:
            result = await _await_if_needed(
                self.components.group_rollout(
                    scenario=scenario,
                    contexts=contexts,
                    phase=phase,
                    policy_client=policy_client,
                )
            )
            return list(result)
        return [
            await self.rollout(
                scenario,
                context,
                phase=phase,
                policy_client=policy_client,
            )
            for context in contexts
        ]

    async def close(self) -> None:
        if self.components.close is not None:
            await _await_if_needed(self.components.close())

    async def offload(self) -> None:
        if self.components.offload is None:
            raise RuntimeError("Embedded LeRobot actor does not support CPU offload")
        await _await_if_needed(self.components.offload())

    async def restore(self) -> None:
        if self.components.restore is None:
            raise RuntimeError("Embedded LeRobot actor does not support GPU restore")
        await _await_if_needed(self.components.restore())


async def create_lerobot_process_actor(
    *,
    config: EmbodiedExperimentConfig,
    context: RolloutActorProcessContext,
) -> LeRobotProcessRolloutActor:
    """Create the standard process actor from one importable component factory."""

    execution = config.runtime.rollout_execution
    options = dict(execution.actor_kwargs)
    # The process pool consumes this scheduling option before actor creation.
    options.pop("max_concurrent_rollouts", None)
    reference = options.pop("components_factory", None)
    component_kwargs = options.pop("components_kwargs", None)
    if options:
        raise ValueError(
            "Unknown LeRobot process actor options: " + ", ".join(sorted(options))
        )
    if not isinstance(reference, str) or not reference:
        raise ValueError(
            "actor_kwargs.components_factory must use the 'module:callable' form"
        )
    if ":" not in reference:
        raise ValueError(
            "actor_kwargs.components_factory must use the 'module:callable' form"
        )
    if not isinstance(component_kwargs, dict):
        raise TypeError("actor_kwargs.components_kwargs must be a mapping")

    factory = _resolve_factory(reference)
    components = await _await_if_needed(
        factory(config=config, context=context, **component_kwargs)
    )
    if not isinstance(components, LeRobotProcessComponents):
        raise TypeError(
            "LeRobot components factory must return LeRobotProcessComponents, "
            f"got {type(components).__name__}"
        )
    if (
        execution.inference_mode == "batched_server"
        and components.group_rollout is None
    ):
        raise ValueError(
            "LeRobot batched-server inference requires an application-owned "
            "group_rollout that uses policy_client"
        )
    if execution.lifecycle == "cpu_offload" and execution.inference_mode == "embedded":
        if components.offload is None or components.restore is None:
            raise ValueError(
                "Embedded cpu_offload requires LeRobotProcessComponents.offload "
                "and .restore hooks"
            )
    return LeRobotProcessRolloutActor(config=config, components=components)


def _resolve_factory(reference: str) -> Callable[..., Any]:
    module_name, _, qualname = reference.partition(":")
    value: Any = importlib.import_module(module_name)
    for component in qualname.split("."):
        value = getattr(value, component)
    if not callable(value):
        raise TypeError(f"Component factory is not callable: {reference}")
    return value


async def _await_if_needed(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value
