"""LeRobot-first embodied trajectory RL for OpenPipe ART.

The public surface is loaded lazily so policy and evaluation workers do not
initialize ART or unrelated ML stacks unless they use the ART lifecycle.
Accessing an ART lifecycle class still performs the normal fail-fast ART
compatibility check.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_GROUPS: dict[str, tuple[str, ...]] = {
    "art_compat": (
        "EmbodiedBackend",
        "EmbodiedTrainableModel",
        "gather_trajectory_groups",
        "trajectory_group",
    ),
    "backends.action_token": (
        "ActionTokenExample",
        "ActionTokenGRPOBackend",
        "ActionTokenGSPOBackend",
        "NoopActionTokenBackend",
        "extract_action_token_examples",
        "extract_trajectory_action_token_examples",
        "prepare_action_token_examples",
        "refresh_action_token_logprobs",
    ),
    "backends.factory": (
        "make_action_token_backend",
        "make_embodied_backend",
        "register_embodied_backend",
        "registered_embodied_backends",
    ),
    "backends.local_process": ("LocalProcessActionTokenBackend",),
    "compatibility": (
        "RuntimeCompatibilityReport",
        "require_compatible_runtime",
        "require_compatible_worker_runtime",
        "runtime_compatibility_report",
    ),
    "config": (
        "EmbodiedExperimentConfig",
        "FlowSDEConfig",
        "FullUpdateScheduleConfig",
        "LookaheadPreviewConfig",
        "PIFlowLoadConfig",
        "RlinfActorBatchScheduleConfig",
        "RolloutExecutionConfig",
        "SmolVLAFlowLoadConfig",
        "TrajectoryMinibatchScheduleConfig",
    ),
    "conformance.rlinf": (
        "RlinfPreparedActionTokenUpdate",
        "RlinfScheduledActionTokenBackend",
        "prepare_rlinf_action_token_update",
    ),
    "evaluation": (
        "EvaluationEpisode",
        "FixedScenarioEvaluator",
        "compare_paired_evaluation_reports",
        "create_paired_evaluation_candidate",
        "summarize_evaluation_report",
        "validate_paired_evaluation_configs",
    ),
    "experiment": (
        "EmbodiedExperiment",
        "EmbodiedScenario",
        "EvaluationResult",
        "ExperimentProgress",
        "ExperimentStepResult",
        "RolloutContext",
    ),
    "integrations.lerobot": (
        "LeRobotActionPrediction",
        "LeRobotPolicyAdapter",
        "LeRobotPolicyAdapterProtocol",
        "SharedLeRobotPolicyAdapterFactory",
        "TrainableActionTokenPolicy",
    ),
    "integrations.lerobot_process": (
        "LeRobotProcessComponents",
        "LeRobotProcessRolloutActor",
        "create_lerobot_process_actor",
    ),
    "integrations.lerobot_rollout": (
        "LeRobotEpisodeRollout",
        "summarize_lerobot_observation",
    ),
    "lookahead": (
        "ActionChunkLookaheadPreview",
        "LookaheadFrame",
        "unused_chunk_length",
    ),
    "media": (
        "LookaheadPreviewRecorder",
        "RolloutVideoRecorder",
        "compose_lookahead_filmstrip",
    ),
    "observability": ("WandbWeaveObserver",),
    "policies.factory": (
        "make_openvla_oft_policy",
        "make_pi_flow_policy",
        "make_policy",
        "make_smolvla_flow_policy",
        "policy_capabilities",
        "register_policy_factory",
        "registered_policy_types",
    ),
    "policy_capabilities": ("BackendRequirements", "PolicyCapabilities"),
    "policies.inference": (
        "OpenVLABatchedInferenceEngine",
        "create_openvla_batched_inference_engine",
    ),
    "policies.openvla": ("OpenVLAPolicy",),
    "rollout_process": (
        "LocalPolicySnapshotProvider",
        "LocalProcessRolloutPool",
        "PolicySnapshotProvider",
        "RolloutActorProcessContext",
    ),
    "runner": (
        "EmbodiedEvaluationRunResult",
        "EmbodiedRunResult",
        "run_embodied_evaluation",
        "run_embodied_experiment",
        "run_lerobot_evaluation",
        "run_lerobot_experiment",
        "validate_runtime_device_availability",
    ),
    "trajectories": (
        "Action",
        "EmbodiedToolCall",
        "EmbodiedTrajectory",
        "EmbodiedTrajectoryGroup",
        "MediaRef",
        "Observation",
        "RewardEvent",
        "TrainableSpan",
    ),
}

_EXPORTS = {
    name: f"{__name__}.{module}"
    for module, names in _EXPORT_GROUPS.items()
    for name in names
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load one public component without initializing unrelated ML stacks."""

    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
