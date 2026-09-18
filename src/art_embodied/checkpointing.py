"""Transactional checkpoint publication and integrity validation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any
import uuid

CHECKPOINT_COMPLETE_MARKER = "art_embodied_checkpoint_complete.json"
_CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class CheckpointValidation:
    """Validated checkpoint metadata returned before policy state is mutated."""

    path: Path
    manifest: dict[str, Any] | None
    legacy: bool


class CheckpointManager:
    """Publish immutable checkpoint directories and reject partial resumes."""

    def publish(
        self,
        path: str | Path,
        *,
        writer: Callable[[Path], None],
        config_fingerprint: str,
        resume_contract_fingerprint: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing checkpoint: {destination}"
            )
        staging = destination.parent / (
            f".{destination.name}.incomplete-{os.getpid()}-{uuid.uuid4().hex}"
        )
        staging.mkdir(mode=0o700)
        try:
            writer(staging)
            files = _checkpoint_file_records(staging)
            if not files:
                raise RuntimeError("Checkpoint writer produced no files")
            manifest = {
                "schema_version": _CHECKPOINT_SCHEMA_VERSION,
                "complete": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "config_fingerprint": str(config_fingerprint),
                "resume_contract_fingerprint": str(resume_contract_fingerprint),
                "metadata": dict(metadata or {}),
                "files": files,
            }
            marker = staging / CHECKPOINT_COMPLETE_MARKER
            marker.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _sync_checkpoint_tree(staging)
            staging.replace(destination)
            _sync_directory(destination.parent)
            return destination
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def validate(
        self,
        path: str | Path,
        *,
        expected_resume_contract_fingerprint: str,
        allow_legacy: bool = False,
        require_training_state: bool = True,
    ) -> CheckpointValidation:
        checkpoint = Path(path)
        if not checkpoint.is_dir():
            raise ValueError(f"Checkpoint directory does not exist: {checkpoint}")
        marker = checkpoint / CHECKPOINT_COMPLETE_MARKER
        if not marker.is_file():
            if not allow_legacy:
                raise ValueError(
                    "Checkpoint has no completion marker and may be partial: "
                    f"{marker}. Set storage.allow_legacy_checkpoint_resume=true "
                    "only for a trusted pre-transactional checkpoint."
                )
            _validate_legacy_checkpoint(
                checkpoint,
                require_training_state=require_training_state,
            )
            return CheckpointValidation(
                path=checkpoint,
                manifest=None,
                legacy=True,
            )

        manifest = _read_manifest(marker)
        contract = manifest.get("resume_contract_fingerprint")
        if contract != expected_resume_contract_fingerprint:
            raise ValueError(
                "Checkpoint resume contract does not match the current experiment: "
                f"saved={contract!r}, current={expected_resume_contract_fingerprint!r}"
            )
        _validate_file_records(checkpoint, manifest)
        if require_training_state:
            _require_regular_file(
                checkpoint / "art_embodied_training_state.pt",
                purpose="training state",
            )
        return CheckpointValidation(
            path=checkpoint,
            manifest=manifest,
            legacy=False,
        )

    def validate_payload(self, path: str | Path) -> CheckpointValidation:
        """Validate an immutable policy payload without treating it as a resume.

        Warm starts intentionally do not import optimizer or scheduler state, so
        their training contract need not match the new experiment.  They must
        still pass the same complete-marker and per-file integrity checks.
        """

        checkpoint = Path(path)
        if not checkpoint.is_dir():
            raise ValueError(f"Checkpoint directory does not exist: {checkpoint}")
        marker = checkpoint / CHECKPOINT_COMPLETE_MARKER
        if not marker.is_file():
            raise ValueError(
                "Warm-start checkpoint has no completion marker and may be partial: "
                f"{marker}"
            )
        manifest = _read_manifest(marker)
        _validate_file_records(checkpoint, manifest)
        return CheckpointValidation(
            path=checkpoint,
            manifest=manifest,
            legacy=False,
        )


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid checkpoint completion marker: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Checkpoint completion marker must be an object: {path}")
    if raw.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported checkpoint completion schema: {raw.get('schema_version')!r}"
        )
    if raw.get("complete") is not True:
        raise ValueError(f"Checkpoint is not marked complete: {path}")
    if not isinstance(raw.get("files"), list) or not raw["files"]:
        raise ValueError(f"Checkpoint completion marker has no file records: {path}")
    return raw


def _checkpoint_file_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Checkpoint payload cannot contain symlinks: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == CHECKPOINT_COMPLETE_MARKER:
            continue
        records.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return records


def _validate_file_records(root: Path, manifest: Mapping[str, Any]) -> None:
    expected: dict[str, tuple[int, str]] = {}
    for record in manifest["files"]:
        if not isinstance(record, dict):
            raise ValueError("Checkpoint file record must be an object")
        relative = record.get("path")
        size = record.get("size")
        digest = record.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or relative in expected
        ):
            raise ValueError(f"Invalid checkpoint file record: {record!r}")
        expected[relative] = (size, digest)

    actual = {
        record["path"]: (record["size"], record["sha256"])
        for record in _checkpoint_file_records(root)
    }
    if actual != expected:
        missing = sorted(set(expected).difference(actual))
        extra = sorted(set(actual).difference(expected))
        changed = sorted(
            path
            for path in set(actual).intersection(expected)
            if actual[path] != expected[path]
        )
        raise ValueError(
            "Checkpoint payload integrity check failed: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )


def _validate_legacy_checkpoint(
    path: Path,
    *,
    require_training_state: bool,
) -> None:
    _require_regular_file(
        path / "art_embodied_checkpoint.json",
        purpose="policy manifest",
    )
    if require_training_state:
        _require_regular_file(
            path / "art_embodied_training_state.pt",
            purpose="training state",
        )
        _require_regular_file(
            path / "art_embodied_training_state.json",
            purpose="training state summary",
        )


def _require_regular_file(path: Path, *, purpose: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Checkpoint is missing its {purpose}: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_checkpoint_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _sync_directory(path)
    _sync_directory(root)


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
