"""LIBERO components for LeRobot SmolVLA Flow-SDE experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from art_embodied.experiment import EmbodiedScenario, RolloutContext
from art_embodied.integrations.lerobot_process import LeRobotProcessComponents
from art_embodied.integrations.smolvla_flow_sde import (
    SmolVLAFlowSDEPolicyAdapter,
)
from art_embodied.policies.factory import make_policy

from .environment import LiberoTaskCatalog
from .records import record_libero_observation, record_libero_transition
from .rollout import rollout_libero_group
from .settings import LiberoSettings


def create_smolvla_components(
    *,
    config: Any,
    context: Any,
    sampling_mode_override: Literal["train", "eval"] | None = None,
) -> LeRobotProcessComponents:
    """Build one process-isolated SmolVLA policy and LIBERO simulator factory."""

    if config.policy.type != "smolvla":
        raise ValueError("SmolVLA LIBERO components require policy.type='smolvla'")
    shared_inference = (
        config.runtime.rollout_execution.inference_mode == "batched_server"
    )
    if shared_inference:
        raise ValueError(
            "SmolVLA batched-server inference is not enabled until its worker "
            "roundtrip conformance test passes; use embedded process actors"
        )
    if sampling_mode_override is not None and shared_inference:
        raise ValueError(
            "sampling_mode_override is supported only by embedded inference"
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
        # Environment collection is inference-only in both phases. Training
        # mode is restored by the optimizer backend before rescoring.
        policy.eval()
        phase_state["value"] = phase

    def offload() -> None:
        policy.to("cpu")

    def restore() -> None:
        policy.to(context.local_device)

    def policy_adapter_factory(
        _scenario: EmbodiedScenario,
        _rollout_context: RolloutContext,
    ) -> SmolVLAFlowSDEPolicyAdapter:
        return SmolVLAFlowSDEPolicyAdapter(
            policy=policy,
            robot_type="panda",
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
            raise RuntimeError("SmolVLA embedded rollout received a policy client")
        sampling_mode = sampling_mode_override or phase
        adapter = SmolVLAFlowSDEPolicyAdapter(
            policy=policy,
            robot_type="panda",
            sampling_mode=sampling_mode,
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
