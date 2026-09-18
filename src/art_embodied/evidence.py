"""Reproducible source, runtime, and policy identity for evaluation evidence."""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

from .checkpointing import CHECKPOINT_COMPLETE_MARKER

_IDENTITY_PACKAGES = (
    "art-embodied",
    "openpipe-art",
    "lerobot",
    "torch",
    "transformers",
    "tokenizers",
    "peft",
    "numpy",
    "pydantic",
    "hf-libero",
    "timm",
    "wandb",
    "weave",
)
_CONTAINER_ENVIRONMENT_KEYS = (
    "ART_EMBODIED_CONTAINER_IMAGE",
    "ART_EMBODIED_CONTAINER_DIGEST",
    "APPTAINER_CONTAINER",
    "SINGULARITY_CONTAINER",
)


def evaluation_evidence_identity(
    *,
    policy_path: str,
    policy_revision: str | None,
    adapter_path: str | None,
    checkpoint_path: str | None,
    evaluation_manifest_path: str | None,
    step: int,
) -> dict[str, Any]:
    """Identify the code, dependencies, runtime, and evaluated policy state."""

    source = source_identity()
    configured_adapter = _optional_artifact_identity(adapter_path)
    checkpoint = _optional_artifact_identity(checkpoint_path)
    local_base = (
        _artifact_identity(Path(policy_path)) if Path(policy_path).exists() else None
    )
    if checkpoint is not None:
        policy_state = {"status": "checkpoint", "artifact": checkpoint}
    elif configured_adapter is not None:
        policy_state = {"status": "configured_adapter", "artifact": configured_adapter}
    elif step == 0 and (policy_revision or local_base is not None):
        policy_state = {
            "status": (
                "immutable_local_base"
                if local_base is not None
                else "immutable_base_revision"
            ),
            "artifact": local_base,
        }
    else:
        policy_state = {"status": "unresolved_in_memory_state", "artifact": None}

    return {
        "source": source,
        "dependencies": {
            name: _distribution_version(name) for name in _IDENTITY_PACKAGES
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "container": {
                key: os.environ[key]
                for key in _CONTAINER_ENVIRONMENT_KEYS
                if os.environ.get(key)
            },
        },
        "policy": {
            "base": {
                "path": policy_path,
                "revision": policy_revision,
                "local_artifact": local_base,
            },
            "configured_adapter": configured_adapter,
            "evaluated_state": policy_state,
        },
        "evaluation_manifest": _optional_artifact_identity(evaluation_manifest_path),
    }


def require_resolved_sealed_identity(identity: dict[str, Any]) -> None:
    """Reject claim-facing evidence with unresolved policy or manifest identity."""

    state = identity["policy"]["evaluated_state"]
    artifact = state.get("artifact")
    unresolved_reference = (
        isinstance(artifact, dict) and artifact.get("kind") == "unresolved_reference"
    )
    if state["status"] == "unresolved_in_memory_state" or unresolved_reference:
        raise ValueError(
            "sealed evaluation requires an immutable policy identity: evaluate "
            "step 0 from a pinned base revision, configure peft_adapter_path, or "
            "evaluate an update that emitted a transactional checkpoint"
        )
    manifest = identity.get("evaluation_manifest")
    if isinstance(manifest, dict) and manifest.get("kind") == "unresolved_reference":
        raise ValueError(
            "sealed evaluation requires the configured evaluation manifest to "
            f"exist and be content-addressable: {manifest.get('path')}"
        )


def source_identity() -> dict[str, Any]:
    """Identify the installed package and repository-local integration code."""

    package_root = Path(__file__).resolve().parent
    result: dict[str, Any] = {
        "package_tree": _artifact_identity(package_root, include_suffixes={".py"}),
    }
    repository = _find_git_repository(package_root)
    if repository is None:
        result["git"] = None
        return result
    commit = _run_git(repository, "rev-parse", "HEAD")
    status = _run_git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    result["git"] = {
        "root": str(repository),
        "commit": commit,
        "dirty": bool(status),
    }
    integration_root = repository / "examples" / "embodied"
    if integration_root.is_dir():
        result["integration_tree"] = _artifact_identity(
            integration_root,
            include_suffixes={".json", ".py", ".yaml", ".yml"},
        )
    return result


def _optional_artifact_identity(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.exists():
        return {"path": value, "kind": "unresolved_reference"}
    return _artifact_identity(path)


def _artifact_identity(
    path: Path,
    *,
    include_suffixes: set[str] | None = None,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        return {
            "path": str(resolved),
            "kind": "file",
            "size": resolved.stat().st_size,
            "sha256": _sha256_file(resolved),
        }
    if not resolved.is_dir():
        raise ValueError(
            f"Evidence artifact is not a regular file or directory: {path}"
        )

    marker = resolved / CHECKPOINT_COMPLETE_MARKER
    if marker.is_file() and include_suffixes is None:
        return _transaction_identity(resolved, marker)

    files: list[dict[str, Any]] = []
    for candidate in sorted(resolved.rglob("*")):
        if candidate.is_symlink():
            raise ValueError(f"Evidence artifact cannot contain symlinks: {candidate}")
        if not candidate.is_file():
            continue
        if include_suffixes is not None and candidate.suffix not in include_suffixes:
            continue
        files.append(
            {
                "path": candidate.relative_to(resolved).as_posix(),
                "size": candidate.stat().st_size,
                "sha256": _sha256_file(candidate),
            }
        )
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "path": str(resolved),
        "kind": "directory",
        "file_count": len(files),
        "total_bytes": sum(record["size"] for record in files),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _transaction_identity(root: Path, marker: Path) -> dict[str, Any]:
    if marker.stat().st_size > 16 * 1024 * 1024:
        raise ValueError(
            f"Checkpoint completion marker is unexpectedly large: {marker}"
        )
    raw = marker.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("complete") is not True:
        raise ValueError(f"Checkpoint completion marker is invalid: {marker}")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"Checkpoint completion marker has no file records: {marker}")
    canonical_files = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "path": str(root),
        "kind": "transactional_checkpoint",
        "marker_sha256": hashlib.sha256(raw).hexdigest(),
        "payload_sha256": hashlib.sha256(canonical_files).hexdigest(),
        "file_count": len(files),
        "config_fingerprint": payload.get("config_fingerprint"),
        "resume_contract_fingerprint": payload.get("resume_contract_fingerprint"),
    }


def _find_git_repository(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _run_git(repository: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
