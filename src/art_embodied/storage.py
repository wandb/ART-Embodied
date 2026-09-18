"""Machine-local storage checks for long-running embodied experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil

_GIB = 1024**3


@dataclass(frozen=True)
class StoragePreflightReport:
    """Resolved output placement and capacity checked before GPU work."""

    output_dir: str
    resolved_output_dir: str
    required_root: str | None
    capacity_probe: str
    total_bytes: int
    used_bytes: int
    free_bytes: int
    minimum_free_bytes: int

    def as_dict(self) -> dict[str, int | str | None]:
        return {"status": "passed", **asdict(self)}


def preflight_storage(
    output_dir: Path,
    *,
    required_root: Path | None = None,
    minimum_free_gib: float = 0.0,
) -> StoragePreflightReport:
    """Fail before allocation work when output placement or capacity is unsafe."""

    if minimum_free_gib < 0:
        raise ValueError("minimum_free_gib must be non-negative")

    output = output_dir.expanduser()
    resolved_output = output.resolve(strict=False)
    resolved_root = (
        required_root.expanduser().resolve(strict=False)
        if required_root is not None
        else None
    )
    if resolved_root is not None and not resolved_output.is_relative_to(resolved_root):
        raise ValueError(
            "Experiment output is outside the required storage root: "
            f"output={resolved_output}, required_root={resolved_root}. "
            "Move the worktree output directory to durable data storage or add a "
            "symlink before requesting GPUs."
        )

    probe = resolved_output
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            raise ValueError(
                f"Cannot find an existing parent for output directory: {resolved_output}"
            )
        probe = parent

    usage = shutil.disk_usage(probe)
    minimum_free_bytes = int(minimum_free_gib * _GIB)
    if usage.free < minimum_free_bytes:
        raise ValueError(
            "Insufficient free storage for experiment output: "
            f"output={resolved_output}, free={usage.free / _GIB:.1f} GiB, "
            f"required={minimum_free_gib:.1f} GiB."
        )

    return StoragePreflightReport(
        output_dir=str(output),
        resolved_output_dir=str(resolved_output),
        required_root=str(resolved_root) if resolved_root is not None else None,
        capacity_probe=str(probe),
        total_bytes=int(usage.total),
        used_bytes=int(usage.used),
        free_bytes=int(usage.free),
        minimum_free_bytes=minimum_free_bytes,
    )
