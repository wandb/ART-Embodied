"""Immutable, model-independent LIBERO initial-state manifests."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

STATE_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class LiberoStateManifestEntry:
    id: str
    task_id: int
    state_key: str
    generation_seed: int
    state_sha256: str
    state_length: int
    bddl_sha256: str


@dataclass(frozen=True, slots=True)
class LiberoStateManifest:
    path: Path
    suite_name: str
    simulator_compatibility: str
    archive_path: Path
    entries: tuple[LiberoStateManifestEntry, ...]
    states: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]


def validate_manifest_bddl_files(
    manifest: LiberoStateManifest,
    bddl_files: Mapping[int, str | Path],
) -> None:
    """Verify that manifest states are evaluated against their source tasks."""

    actual_hashes = {
        int(task_id): file_sha256(path) for task_id, path in bddl_files.items()
    }
    for entry in manifest.entries:
        actual = actual_hashes.get(entry.task_id)
        if actual is None:
            raise ValueError(
                f"LIBERO state manifest task_id={entry.task_id} has no BDDL file"
            )
        if actual != entry.bddl_sha256:
            raise ValueError(
                "LIBERO state manifest BDDL hash mismatch for "
                f"task_id={entry.task_id}: expected={entry.bddl_sha256}, "
                f"found={actual}"
            )


def state_sha256(state: Any) -> str:
    """Hash one flattened float64 MuJoCo state using a stable byte contract."""

    value = np.ascontiguousarray(np.asarray(state, dtype=np.float64).reshape(-1))
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_state_manifest(
    path: str | Path,
    *,
    expected_suite_name: str | None = None,
    expected_simulator_compatibility: str | None = None,
) -> LiberoStateManifest:
    manifest_path = Path(path).expanduser().resolve()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("LIBERO state manifest must contain a JSON object")
    if raw.get("schema_version") != STATE_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported LIBERO state manifest schema_version: "
            f"{raw.get('schema_version')!r}"
        )
    suite_name = _required_string(raw, "suite_name")
    simulator_compatibility = _required_string(raw, "simulator_compatibility")
    if expected_suite_name is not None and suite_name != expected_suite_name:
        raise ValueError(
            "LIBERO state manifest suite mismatch: "
            f"expected={expected_suite_name!r}, found={suite_name!r}"
        )
    if (
        expected_simulator_compatibility is not None
        and simulator_compatibility != expected_simulator_compatibility
    ):
        raise ValueError(
            "LIBERO state manifest simulator mismatch: "
            f"expected={expected_simulator_compatibility!r}, "
            f"found={simulator_compatibility!r}"
        )

    archive_name = _required_string(raw, "state_archive")
    archive_path = (manifest_path.parent / archive_name).resolve()
    if archive_path.parent != manifest_path.parent:
        raise ValueError("state_archive must be a file beside the manifest")
    expected_archive_hash = _required_string(raw, "state_archive_sha256")
    actual_archive_hash = file_sha256(archive_path)
    if actual_archive_hash != expected_archive_hash:
        raise ValueError(
            "LIBERO state archive hash mismatch: "
            f"expected={expected_archive_hash}, found={actual_archive_hash}"
        )

    entry_rows = raw.get("entries")
    if not isinstance(entry_rows, list) or not entry_rows:
        raise ValueError("LIBERO state manifest entries must be a non-empty list")
    entries = tuple(_parse_entry(row) for row in entry_rows)
    ids = [entry.id for entry in entries]
    keys = [entry.state_key for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("LIBERO state manifest entry ids must be unique")
    if len(keys) != len(set(keys)):
        raise ValueError("LIBERO state manifest state keys must be unique")

    states: dict[str, np.ndarray] = {}
    with np.load(archive_path, allow_pickle=False) as archive:
        archive_keys = set(archive.files)
        expected_keys = set(keys)
        if archive_keys != expected_keys:
            raise ValueError(
                "LIBERO state archive keys do not match manifest entries: "
                f"missing={sorted(expected_keys - archive_keys)}, "
                f"extra={sorted(archive_keys - expected_keys)}"
            )
        for entry in entries:
            state = np.asarray(archive[entry.state_key], dtype=np.float64).reshape(-1)
            if state.size != entry.state_length:
                raise ValueError(
                    f"State length mismatch for {entry.state_key}: "
                    f"expected={entry.state_length}, found={state.size}"
                )
            if not np.isfinite(state).all():
                raise ValueError(f"State {entry.state_key} contains NaN or infinity")
            actual_hash = state_sha256(state)
            if actual_hash != entry.state_sha256:
                raise ValueError(
                    f"State hash mismatch for {entry.state_key}: "
                    f"expected={entry.state_sha256}, found={actual_hash}"
                )
            states[entry.state_key] = np.array(state, copy=True)

    return LiberoStateManifest(
        path=manifest_path,
        suite_name=suite_name,
        simulator_compatibility=simulator_compatibility,
        archive_path=archive_path,
        entries=entries,
        states=states,
        metadata=raw,
    )


def write_state_manifest(
    output_dir: str | Path,
    *,
    suite_name: str,
    simulator_compatibility: str,
    states: Mapping[str, np.ndarray],
    entries: Sequence[Mapping[str, Any]],
    generator: Mapping[str, Any],
) -> Path:
    """Atomically write a compressed state archive followed by its manifest."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "states.npz"
    archive_tmp = destination / "states.npz.tmp"
    with archive_tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            **{
                str(key): np.asarray(value, dtype=np.float64).reshape(-1)
                for key, value in states.items()
            },
        )
    archive_tmp.replace(archive_path)
    payload = {
        "schema_version": STATE_MANIFEST_SCHEMA_VERSION,
        "suite_name": str(suite_name),
        "simulator_compatibility": str(simulator_compatibility),
        "state_archive": archive_path.name,
        "state_archive_sha256": file_sha256(archive_path),
        "generator": dict(generator),
        "entries": [dict(entry) for entry in entries],
    }
    manifest_path = destination / "manifest.json"
    manifest_tmp = destination / "manifest.json.tmp"
    manifest_tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_tmp.replace(manifest_path)
    load_state_manifest(
        manifest_path,
        expected_suite_name=suite_name,
        expected_simulator_compatibility=simulator_compatibility,
    )
    return manifest_path


def _parse_entry(value: Any) -> LiberoStateManifestEntry:
    if not isinstance(value, dict):
        raise TypeError("Each LIBERO state manifest entry must be an object")
    entry = LiberoStateManifestEntry(
        id=_required_string(value, "id"),
        task_id=int(value["task_id"]),
        state_key=_required_string(value, "state_key"),
        generation_seed=int(value["generation_seed"]),
        state_sha256=_required_string(value, "state_sha256"),
        state_length=int(value["state_length"]),
        bddl_sha256=_required_string(value, "bddl_sha256"),
    )
    if entry.task_id < 0:
        raise ValueError("LIBERO state manifest task_id must be non-negative")
    if entry.state_length < 1:
        raise ValueError("LIBERO state manifest state_length must be positive")
    for field_name, digest in (
        ("state_sha256", entry.state_sha256),
        ("bddl_sha256", entry.bddl_sha256),
    ):
        if len(digest) != 64 or any(
            value not in "0123456789abcdef" for value in digest
        ):
            raise ValueError(
                f"LIBERO state manifest {field_name} must be a lowercase SHA-256"
            )
    return entry


def _required_string(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"LIBERO state manifest {key} must be a non-empty string")
    return result
