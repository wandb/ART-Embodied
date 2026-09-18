from __future__ import annotations

import json
from pathlib import Path

import pytest

from art_embodied.checkpointing import (
    CHECKPOINT_COMPLETE_MARKER,
    CheckpointManager,
)


def _write_checkpoint(path: Path) -> None:
    (path / "art_embodied_checkpoint.json").write_text("{}\n", encoding="utf-8")
    (path / "weights.bin").write_bytes(b"weights")
    (path / "art_embodied_training_state.pt").write_bytes(b"state")


def test_checkpoint_publish_is_complete_and_validated(tmp_path: Path) -> None:
    manager = CheckpointManager()
    checkpoint = manager.publish(
        tmp_path / "step_000001",
        writer=_write_checkpoint,
        config_fingerprint="config",
        resume_contract_fingerprint="contract",
        metadata={"update_step": 1},
    )

    validation = manager.validate(
        checkpoint,
        expected_resume_contract_fingerprint="contract",
    )

    assert validation.legacy is False
    assert validation.manifest is not None
    assert validation.manifest["metadata"]["update_step"] == 1
    assert (checkpoint / CHECKPOINT_COMPLETE_MARKER).is_file()
    assert not list(tmp_path.glob("*.incomplete-*"))


def test_checkpoint_publish_removes_partial_staging_on_failure(tmp_path: Path) -> None:
    manager = CheckpointManager()
    destination = tmp_path / "step_000001"

    def fail(path: Path) -> None:
        (path / "partial.bin").write_bytes(b"partial")
        raise RuntimeError("injected save failure")

    with pytest.raises(RuntimeError, match="injected save failure"):
        manager.publish(
            destination,
            writer=fail,
            config_fingerprint="config",
            resume_contract_fingerprint="contract",
        )

    assert not destination.exists()
    assert not list(tmp_path.iterdir())


def test_checkpoint_validation_rejects_corruption_and_contract_mismatch(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager()
    checkpoint = manager.publish(
        tmp_path / "step_000001",
        writer=_write_checkpoint,
        config_fingerprint="config",
        resume_contract_fingerprint="contract",
    )
    (checkpoint / "weights.bin").write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="integrity check failed"):
        manager.validate(
            checkpoint,
            expected_resume_contract_fingerprint="contract",
        )
    with pytest.raises(ValueError, match="resume contract does not match"):
        manager.validate(
            checkpoint,
            expected_resume_contract_fingerprint="different",
        )


def test_checkpoint_validation_requires_explicit_legacy_opt_in(tmp_path: Path) -> None:
    checkpoint = tmp_path / "legacy"
    checkpoint.mkdir()
    (checkpoint / "art_embodied_checkpoint.json").write_text("{}")
    (checkpoint / "art_embodied_training_state.pt").write_bytes(b"state")
    (checkpoint / "art_embodied_training_state.json").write_text("{}")
    manager = CheckpointManager()

    with pytest.raises(ValueError, match="no completion marker"):
        manager.validate(
            checkpoint,
            expected_resume_contract_fingerprint="contract",
        )

    validation = manager.validate(
        checkpoint,
        expected_resume_contract_fingerprint="contract",
        allow_legacy=True,
    )
    assert validation.legacy is True
    assert validation.manifest is None


def test_checkpoint_manifest_cannot_be_forged_with_path_traversal(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "art_embodied_training_state.pt").write_bytes(b"state")
    (checkpoint / CHECKPOINT_COMPLETE_MARKER).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "complete": True,
                "resume_contract_fingerprint": "contract",
                "files": [{"path": "../outside", "size": 0, "sha256": "0" * 64}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid checkpoint file record"):
        CheckpointManager().validate(
            checkpoint,
            expected_resume_contract_fingerprint="contract",
        )
