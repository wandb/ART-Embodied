"""Lazy public exports for embodied training backends."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_GROUPS: dict[str, tuple[str, ...]] = {
    "action_token": (
        "ActionTokenExample",
        "ActionTokenGRPOBackend",
        "ActionTokenGSPOBackend",
        "NoopActionTokenBackend",
        "extract_action_token_examples",
        "extract_trajectory_action_token_examples",
        "prepare_action_token_examples",
        "refresh_action_token_logprobs",
    ),
    "factory": (
        "make_action_token_backend",
        "make_embodied_backend",
        "register_embodied_backend",
        "registered_embodied_backends",
    ),
    "local_process": ("LocalProcessActionTokenBackend",),
}

_EXPORTS = {
    name: f"{__name__}.{module}"
    for module, names in _EXPORT_GROUPS.items()
    for name in names
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load one backend without initializing unrelated policy runtimes."""

    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
