"""NVIDIA GR00T N1.7 plugin over the policy-neutral RoboCasa adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from art_embodied.experiment import EmbodiedScenario, RolloutContext
from art_embodied.integrations.lerobot_process import LeRobotProcessComponents

from .environment import POLICY_STATE_ACTION_LAYOUT, RoboCasaTaskCatalog
from .records import record_robocasa_observation, record_robocasa_transition
from .rollout import rollout_robocasa_group
from .settings import RoboCasaSettings
from .simulator import RoboCasaSimulatorProcess


def create_gr00t_n1d7_components(
    *, config: Any, context: Any
) -> LeRobotProcessComponents:
    settings = RoboCasaSettings.from_config(config)
    _validate_n1d7_contract(config)
    from art_embodied.integrations.gr00t_flow_sde import GR00TN17FlowSDEPolicyAdapter
    from art_embodied.policies.factory import make_policy

    local_config = config.model_copy(
        update={
            "policy": config.policy.model_copy(update={"device": context.local_device})
        }
    )
    policy = make_policy(local_config)
    catalog = RoboCasaTaskCatalog(settings)
    simulator = RoboCasaSimulatorProcess(
        settings=settings,
        startup_timeout_seconds=config.runtime.rollout_execution.startup_timeout_seconds,
    )
    phase_state = {"value": "train"}

    def load_policy_snapshot(*, update: int, policy_snapshot: Path) -> None:
        policy.load_checkpoint({"path": str(policy_snapshot)})
        policy.rollout_update = int(update)

    def prepare_phase(phase: str) -> None:
        if phase not in {"train", "eval"}:
            raise ValueError(f"Unsupported RoboCasa phase: {phase!r}")
        policy.eval()
        phase_state["value"] = (
            "train" if phase == "train" else _evaluation_sampling_mode(local_config)
        )

    def policy_adapter_factory(
        _scenario: EmbodiedScenario, _context: RolloutContext
    ) -> Any:
        return GR00TN17FlowSDEPolicyAdapter(
            policy=policy,
            sampling_mode=phase_state["value"],
            runtime_profile="robocasa_gr1_tabletop",
        )

    async def group_rollout(
        *,
        scenario: EmbodiedScenario,
        contexts: tuple[RolloutContext, ...],
        phase: str,
        policy_client: Any | None,
    ):
        if policy_client is not None:
            raise RuntimeError("RoboCasa embedded policy received a policy client")
        task_id = scenario.payload.get("task_id")
        if not isinstance(task_id, str) or task_id not in catalog.tasks:
            raise ValueError(f"Scenario declares unknown RoboCasa task: {task_id!r}")
        adapter = GR00TN17FlowSDEPolicyAdapter(
            policy=policy,
            sampling_mode=phase_state["value"],
            runtime_profile="robocasa_gr1_tabletop",
        )
        return await rollout_robocasa_group(
            config=local_config,
            settings=settings,
            simulator=simulator,
            policy_adapter=adapter,
            scenario=scenario,
            contexts=contexts,
            phase=phase,
        )

    def offload() -> None:
        policy.to("cpu")

    def restore() -> None:
        policy.to(context.local_device)

    return LeRobotProcessComponents(
        environment_factory=_unsupported_single_environment,
        policy_adapter_factory=policy_adapter_factory,
        load_policy_snapshot=load_policy_snapshot,
        prepare_phase=prepare_phase,
        observation_recorder=record_robocasa_observation,
        transition_recorder=record_robocasa_transition,
        group_rollout=group_rollout,
        offload=offload,
        restore=restore,
        close=simulator.close,
    )


def _unsupported_single_environment(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("RoboCasa rollouts require the grouped simulator-process path")


def _validate_n1d7_contract(config: Any) -> None:
    if config.policy.type != "gr00t_n1d7":
        raise ValueError("RoboCasa requires policy.type='gr00t_n1d7'")
    load = config.policy.load_kwargs
    expected_components = [
        {
            "key": name,
            "size": size,
            "executed": True,
            "execution_order": index,
        }
        for index, (name, size) in enumerate(POLICY_STATE_ACTION_LAYOUT)
    ]
    expected = {
        "embodiment_tag": "ROBOCASA_GR1_TABLETOP",
        "execution_horizon": 8,
        "processor_action_horizon": 8,
        "action_dim": 29,
        "execution_action_dim": 29,
        "model_action_horizon": 40,
        "action_components": expected_components,
    }
    actual = {key: load.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            "RoboCasa requires the exact official N1.7 GR-1 action contract: "
            f"expected={expected}, actual={actual}"
        )
    _evaluation_sampling_mode(config)


def _evaluation_sampling_mode(config: Any) -> str:
    declared = config.evaluation.kwargs.get("action_sampling", "native_flow_ode")
    if declared == "native_flow_ode":
        if not config.evaluation.deterministic:
            raise ValueError(
                "RoboCasa native_flow_ode evaluation must declare deterministic=true"
            )
        if config.policy.evaluation_generation.do_sample:
            raise ValueError(
                "RoboCasa native_flow_ode evaluation requires "
                "policy.evaluation_generation.do_sample=false"
            )
        return "eval"
    if declared == "gaussian_flow_sde":
        if config.evaluation.data_role != "diagnostic":
            raise ValueError(
                "RoboCasa gaussian_flow_sde evaluation is diagnostic-only"
            )
        if config.evaluation.deterministic:
            raise ValueError(
                "RoboCasa gaussian_flow_sde evaluation must declare "
                "deterministic=false"
            )
        if not config.policy.evaluation_generation.do_sample:
            raise ValueError(
                "RoboCasa gaussian_flow_sde evaluation requires "
                "policy.evaluation_generation.do_sample=true"
            )
        return "stochastic_eval"
    raise ValueError(
        "evaluation.kwargs.action_sampling must be native_flow_ode or "
        "gaussian_flow_sde"
    )
