"""Lazy public exports for optional robotics integrations.

Importing one integration must not initialize unrelated policy stacks. This
keeps the core add-on usable without heavyweight extras such as PyTorch.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_GROUPS: dict[str, tuple[str, ...]] = {
    "lerobot": (
        "LeRobotActionPrediction",
        "LeRobotPolicyAdapter",
        "LeRobotPolicyAdapterProtocol",
        "SharedLeRobotPolicyAdapterFactory",
        "TrainableActionTokenPolicy",
    ),
    "lerobot_envhub": (
        "LeRobotEnvHubSpec",
        "SingleLeRobotVectorEnv",
        "load_lerobot_envhub_environment",
    ),
    "lerobot_process": (
        "LeRobotProcessComponents",
        "LeRobotProcessRolloutActor",
        "create_lerobot_process_actor",
    ),
    "lerobot_rollout": (
        "LeRobotEpisodeRollout",
        "summarize_lerobot_observation",
    ),
    "gr00t_flow_sde": (
        "GR00TN15FlowSDEPolicyAdapter",
        "GR00TN17FlowSDEPolicyAdapter",
    ),
    "pi_flow_sde": ("PIFlowSDEPolicyAdapter",),
    "smolvla_flow_sde": ("SmolVLAFlowSDEPolicyAdapter",),
}

_EXPORTS = {
    name: f"{__name__}.{module}"
    for module, names in _EXPORT_GROUPS.items()
    for name in names
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load one integration without importing unrelated optional runtimes."""

    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
