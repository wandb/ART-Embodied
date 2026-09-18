"""LIBERO components for NVIDIA GR00T N1.5 Flow-SDE experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from art_embodied.experiment import EmbodiedScenario, RolloutContext
from art_embodied.integrations.gr00t_flow_sde import (
    GR00TN15FlowSDEPolicyAdapter,
    GR00TN17FlowSDEPolicyAdapter,
)
from art_embodied.integrations.lerobot_process import LeRobotProcessComponents
from art_embodied.policies.factory import make_policy

from .environment import LiberoTaskCatalog
from .records import record_libero_observation, record_libero_transition
from .rollout import rollout_libero_group
from .settings import LiberoSettings


def create_gr00t_n1d5_components(
    *,
    config: Any,
    context: Any,
    sampling_mode_override: Literal["train", "eval"] | None = None,
) -> LeRobotProcessComponents:
    """Build one isolated N1.5 policy and its LIBERO simulator factory."""

    if config.policy.type != "gr00t_n1d5":
        raise ValueError(
            "GR00T N1.5 LIBERO components require policy.type='gr00t_n1d5'"
        )
    if config.runtime.rollout_execution.inference_mode != "embedded":
        raise ValueError(
            "GR00T N1.5 starts with embedded inference until its shared-worker "
            "roundtrip conformance test passes"
        )
    settings = LiberoSettings.from_config(config)
    local_config = config.model_copy(
        update={
            "policy": config.policy.model_copy(update={"device": context.local_device})
        }
    )
    policy = make_policy(local_config)
    catalog = LiberoTaskCatalog(settings)
    phase_state = {"value": "train"}

    def load_policy_snapshot(*, update: int, policy_snapshot: Path) -> None:
        policy.load_checkpoint({"path": str(policy_snapshot)})
        policy.rollout_update = int(update)

    def prepare_phase(phase: str) -> None:
        if phase not in {"train", "eval"}:
            raise ValueError(f"Unsupported rollout phase: {phase!r}")
        policy.eval()
        phase_state["value"] = phase

    def offload() -> None:
        policy.to("cpu")

    def restore() -> None:
        policy.to(context.local_device)

    def policy_adapter_factory(
        _scenario: EmbodiedScenario,
        _rollout_context: RolloutContext,
    ) -> GR00TN15FlowSDEPolicyAdapter:
        return GR00TN15FlowSDEPolicyAdapter(
            policy=policy,
            sampling_mode=phase_state["value"],
        )

    async def group_rollout(
        *,
        scenario: EmbodiedScenario,
        contexts: tuple[RolloutContext, ...],
        phase: str,
        policy_client: Any | None,
    ):
        if policy_client is not None:
            raise RuntimeError("GR00T embedded rollout received a policy client")
        adapter = GR00TN15FlowSDEPolicyAdapter(
            policy=policy,
            sampling_mode=sampling_mode_override or phase,
        )
        return await rollout_libero_group(
            config=local_config,
            policy=policy,
            catalog=catalog,
            settings=settings,
            scenario=scenario,
            contexts=contexts,
            phase=phase,
            embedded_batch_predictor=adapter.predict_batch,
            embedded_batch_reset=lambda seed: adapter.reset(seed=seed),
        )

    return LeRobotProcessComponents(
        environment_factory=catalog.make_environment,
        policy_adapter_factory=policy_adapter_factory,
        load_policy_snapshot=load_policy_snapshot,
        prepare_phase=prepare_phase,
        observation_recorder=record_libero_observation,
        transition_recorder=record_libero_transition,
        group_rollout=group_rollout,
        offload=offload,
        restore=restore,
    )


def create_gr00t_n1d7_components(
    *,
    config: Any,
    context: Any,
    sampling_mode_override: Literal["train", "eval"] | None = None,
) -> LeRobotProcessComponents:
    """Build one isolated N1.7 policy and its LIBERO simulator factory."""

    if config.policy.type != "gr00t_n1d7":
        raise ValueError(
            "GR00T N1.7 LIBERO components require policy.type='gr00t_n1d7'"
        )
    if config.runtime.rollout_execution.inference_mode != "embedded":
        raise ValueError(
            "GR00T N1.7 starts with embedded inference until its shared-worker "
            "roundtrip conformance test passes"
        )
    settings = LiberoSettings.from_config(config)
    local_config = config.model_copy(
        update={
            "policy": config.policy.model_copy(update={"device": context.local_device})
        }
    )
    policy = make_policy(local_config)
    catalog = LiberoTaskCatalog(settings)
    phase_state = {"value": "train"}

    def load_policy_snapshot(*, update: int, policy_snapshot: Path) -> None:
        policy.load_checkpoint({"path": str(policy_snapshot)})
        policy.rollout_update = int(update)

    def prepare_phase(phase: str) -> None:
        if phase not in {"train", "eval"}:
            raise ValueError(f"Unsupported rollout phase: {phase!r}")
        policy.eval()
        phase_state["value"] = phase

    def offload() -> None:
        policy.to("cpu")

    def restore() -> None:
        policy.to(context.local_device)

    def policy_adapter_factory(
        _scenario: EmbodiedScenario,
        _rollout_context: RolloutContext,
    ) -> GR00TN17FlowSDEPolicyAdapter:
        return GR00TN17FlowSDEPolicyAdapter(
            policy=policy,
            sampling_mode=phase_state["value"],
        )

    async def group_rollout(
        *,
        scenario: EmbodiedScenario,
        contexts: tuple[RolloutContext, ...],
        phase: str,
        policy_client: Any | None,
    ):
        if policy_client is not None:
            raise RuntimeError("GR00T N1.7 embedded rollout received a policy client")
        adapter = GR00TN17FlowSDEPolicyAdapter(
            policy=policy,
            sampling_mode=sampling_mode_override or phase,
        )
        return await rollout_libero_group(
            config=local_config,
            policy=policy,
            catalog=catalog,
            settings=settings,
            scenario=scenario,
            contexts=contexts,
            phase=phase,
            embedded_batch_predictor=adapter.predict_batch,
            embedded_batch_reset=lambda seed: adapter.reset(seed=seed),
        )

    return LeRobotProcessComponents(
        environment_factory=catalog.make_environment,
        policy_adapter_factory=policy_adapter_factory,
        load_policy_snapshot=load_policy_snapshot,
        prepare_phase=prepare_phase,
        observation_recorder=record_libero_observation,
        transition_recorder=record_libero_transition,
        group_rollout=group_rollout,
        offload=offload,
        restore=restore,
    )
