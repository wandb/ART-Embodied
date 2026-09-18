"""LIBERO components for LeRobot pi0-FAST action-token experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from art_embodied.experiment import EmbodiedScenario, RolloutContext
from art_embodied.integrations.lerobot_process import LeRobotProcessComponents
from art_embodied.integrations.pi0_fast import PI0FastPolicyAdapter
from art_embodied.policies.factory import make_policy

from .records import record_libero_observation, record_libero_transition
from .rollout import rollout_libero_group


def create_pi0_fast_components(
    *,
    config: Any,
    context: Any,
    sampling_mode_override: Literal["train", "eval"] | None = None,
) -> LeRobotProcessComponents:
    """Build one process-isolated pi0-FAST policy and LIBERO simulator."""

    if config.policy.type != "pi0_fast":
        raise ValueError("pi0-FAST LIBERO components require policy.type='pi0_fast'")
    if config.runtime.rollout_execution.inference_mode == "batched_server":
        raise ValueError(
            "pi0-FAST starts with embedded process actors until its serialized "
            "token/logprob worker roundtrip passes conformance"
        )
    if config.environment.type == "libero":
        from .environment import LiberoTaskCatalog
        from .settings import LiberoSettings

        settings = LiberoSettings.from_config(config)
        catalog = LiberoTaskCatalog(settings)
    elif config.environment.type == "libero_plus":
        from examples.embodied.libero_plus.environment import LiberoPlusTaskCatalog
        from examples.embodied.libero_plus.settings import LiberoPlusSettings

        settings = LiberoPlusSettings.from_config(config)
        catalog = LiberoPlusTaskCatalog(settings)
    else:
        raise ValueError(
            "pi0-FAST LIBERO components require environment.type='libero' "
            "or 'libero_plus'"
        )
    local_config = config.model_copy(
        update={
            "policy": config.policy.model_copy(update={"device": context.local_device})
        }
    )
    policy = make_policy(local_config)
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

    def make_adapter(phase: str) -> PI0FastPolicyAdapter:
        generation = (
            local_config.policy.evaluation_generation
            if phase == "eval"
            else local_config.policy.rollout_generation
        )
        return PI0FastPolicyAdapter(
            policy=policy,
            robot_type="panda",
            sampling_mode=phase,
            do_sample=bool(generation.do_sample),
            temperature=float(generation.temperature),
            compute_rollout_logprobs=phase == "train",
            action_decoder=local_config.policy.load_kwargs.get(
                "action_decoder", "strict"
            ),
            invalid_action_handling=local_config.policy.load_kwargs.get(
                "invalid_action_handling", "raise"
            ),
            model_batch_size=(
                int(local_config.runtime.rollout_execution.inference_max_batch_size)
                if phase == "train"
                else None
            ),
        )

    def policy_adapter_factory(
        _scenario: EmbodiedScenario,
        _rollout_context: RolloutContext,
    ) -> PI0FastPolicyAdapter:
        return make_adapter(phase_state["value"])

    async def group_rollout(
        *,
        scenario: EmbodiedScenario,
        contexts: tuple[RolloutContext, ...],
        phase: str,
        policy_client: Any | None,
    ):
        if policy_client is not None:
            raise RuntimeError("pi0-FAST embedded rollout received a policy client")
        sampling_mode = sampling_mode_override or phase
        adapter = make_adapter(sampling_mode)
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
