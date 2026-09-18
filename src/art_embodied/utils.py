"""Small utilities for embodied trajectory adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import sys
import tempfile
from typing import Any


def make_json_safe(value: Any) -> Any:
    """Best-effort conversion of common array/scalar objects to JSON-safe data.

    Robotics environments frequently return numpy arrays or framework tensors.
    ART-Embodied should not require a hard dependency on those libraries just to
    capture trajectories, so this helper uses duck typing for common methods.
    """

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return make_json_safe(asdict(value))
    if hasattr(value, "detach") and callable(value.detach):
        return make_json_safe(value.detach().cpu().tolist())
    if hasattr(value, "tolist") and callable(value.tolist):
        return make_json_safe(value.tolist())
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [make_json_safe(item) for item in value]
    return repr(value)


def worker_python_command(
    *,
    configured_executable: str | None,
    module: str,
    spec_path: Path,
) -> list[str]:
    """Build an argv-only command for a native policy worker.

    ART's control plane and a VLA policy can require different Python
    environments. The experiment contract therefore selects one interpreter
    for rollout, inference, and training workers while retaining the current
    interpreter as the zero-configuration default.
    """

    executable = configured_executable or sys.executable
    return [executable, "-m", module, "--serve-spec", str(spec_path)]


def write_json_atomic(
    path: Path,
    value: Any,
    *,
    indent: int | None = None,
    sort_keys: bool = False,
) -> None:
    """Publish a complete JSON IPC marker with one atomic rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(value, handle, indent=indent, sort_keys=sort_keys)
            handle.write("\n")
            handle.flush()
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
