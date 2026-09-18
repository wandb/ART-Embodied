"""LIBERO components for LeRobot PI0/PI0.5 Flow-SDE experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from art_embodied.experiment import EmbodiedScenario, RolloutContext
from art_embodied.integrations.lerobot_process import LeRobotProcessComponents
from art_embodied.integrations.pi_flow_sde import PIFlowSDEPolicyAdapter
from art_embodied.policies.factory import make_policy

from .environment import LiberoTaskCatalog
from .records import record_libero_observation, record_libero_transition
from .rollout import rollout_libero_group
from .settings import LiberoSettings


def create_pi_components(
    *,
    config: Any,
    context: Any,
    sampling_mode_override: Literal["train", "eval"] | None = None,
) -> LeRobotProcessComponents:
    """Build one process-isolated PI policy and LIBERO simulator factory."""

    if config.policy.type not in {"pi0", "pi05"}:
        raise ValueError("PI LIBERO components require policy.type='pi0' or 'pi05'")
    shared_inference = (
        config.runtime.rollout_execution.inference_mode == "batched_server"
    )
    if sampling_mode_override is not None and shared_inference:
        raise ValueError(
            "PI sampling_mode_override is supported only by embedded inference"
        )
    settings = LiberoSettings.from_config(config)
    local_config = config.model_copy(
        update={
            "policy": config.policy.model_copy(update={"device": context.local_device})
        }
    )
    policy = None if shared_inference else make_policy(local_config)
    catalog = LiberoTaskCatalog(settings)
    phase_state = {"value": "train"}

    def load_policy_snapshot(*, update: int, policy_snapshot: Path) -> None:
        if policy is None:
            return
        policy.load_checkpoint({"path": str(policy_snapshot)})
        policy.rollout_update = int(update)

    def prepare_phase(phase: str) -> None:
        if phase not in {"train", "eval"}:
            raise ValueError(f"Unsupported rollout phase: {phase!r}")
        if policy is not None:
            policy.eval()
        phase_state["value"] = phase

    def offload() -> None:
        if policy is None:
            return
        policy.to("cpu")

    def restore() -> None:
        if policy is None:
            return
        policy.to(context.local_device)

    def policy_adapter_factory(
        _scenario: EmbodiedScenario,
        _rollout_context: RolloutContext,
    ) -> PIFlowSDEPolicyAdapter:
        if policy is None:
            raise RuntimeError("Shared PI inference does not construct actor policies")
        return PIFlowSDEPolicyAdapter(
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
        if shared_inference != (policy_client is not None):
            raise RuntimeError("PI inference mode and rollout policy client disagree")
        if shared_inference:
            return await rollout_libero_group(
                config=local_config,
                policy=None,
                policy_client=policy_client,
                catalog=catalog,
                settings=settings,
                scenario=scenario,
                contexts=contexts,
                phase=phase,
            )
        assert policy is not None
        sampling_mode = sampling_mode_override or phase
        adapter = PIFlowSDEPolicyAdapter(
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
        offload=None if shared_inference else offload,
        restore=None if shared_inference else restore,
    )
