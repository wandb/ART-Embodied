from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from art_embodied.checkpointing import CheckpointManager
from art_embodied.evidence import (
    evaluation_evidence_identity,
    require_resolved_sealed_identity,
)


def test_evidence_identity_hashes_adapter_and_evaluation_manifest(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-weights")
    manifest = tmp_path / "states.npz"
    manifest.write_bytes(b"fixed-state-manifest")

    identity = evaluation_evidence_identity(
        policy_path="organization/base-policy",
        policy_revision="immutable-revision",
        adapter_path=str(adapter),
        checkpoint_path=None,
        evaluation_manifest_path=str(manifest),
        step=10,
    )

    state = identity["policy"]["evaluated_state"]
    assert state["status"] == "configured_adapter"
    assert state["artifact"]["kind"] == "directory"
    assert state["artifact"]["file_count"] == 1
    assert (
        identity["evaluation_manifest"]["sha256"]
        == hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    assert identity["source"]["package_tree"]["file_count"] > 0
    assert identity["source"]["integration_tree"]["file_count"] > 0
    assert "openpipe-art" in identity["dependencies"]


def test_evidence_identity_uses_transaction_manifest_for_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    CheckpointManager().publish(
        checkpoint,
        writer=lambda staging: (staging / "adapter.bin").write_bytes(b"weights"),
        config_fingerprint="config",
        resume_contract_fingerprint="resume",
    )

    identity = evaluation_evidence_identity(
        policy_path="organization/base-policy",
        policy_revision="immutable-revision",
        adapter_path=None,
        checkpoint_path=str(checkpoint),
        evaluation_manifest_path=None,
        step=20,
    )

    artifact = identity["policy"]["evaluated_state"]["artifact"]
    assert artifact["kind"] == "transactional_checkpoint"
    assert artifact["file_count"] == 1
    marker = checkpoint / "art_embodied_checkpoint_complete.json"
    assert artifact["marker_sha256"] == hashlib.sha256(marker.read_bytes()).hexdigest()
    assert json.loads(marker.read_text(encoding="utf-8"))["complete"] is True


def test_sealed_evidence_rejects_unidentified_in_memory_weights() -> None:
    identity = evaluation_evidence_identity(
        policy_path="organization/base-policy",
        policy_revision="immutable-revision",
        adapter_path=None,
        checkpoint_path=None,
        evaluation_manifest_path=None,
        step=10,
    )

    with pytest.raises(ValueError, match="immutable policy identity"):
        require_resolved_sealed_identity(identity)


def test_sealed_evidence_accepts_pinned_step_zero_base() -> None:
    identity = evaluation_evidence_identity(
        policy_path="organization/base-policy",
        policy_revision="immutable-revision",
        adapter_path=None,
        checkpoint_path=None,
        evaluation_manifest_path=None,
        step=0,
    )

    require_resolved_sealed_identity(identity)
    assert identity["policy"]["evaluated_state"]["status"] == (
        "immutable_base_revision"
    )


def test_sealed_evidence_rejects_missing_adapter_path(tmp_path: Path) -> None:
    identity = evaluation_evidence_identity(
        policy_path="organization/base-policy",
        policy_revision="immutable-revision",
        adapter_path=str(tmp_path / "missing-adapter"),
        checkpoint_path=None,
        evaluation_manifest_path=None,
        step=10,
    )

    with pytest.raises(ValueError, match="immutable policy identity"):
        require_resolved_sealed_identity(identity)


def test_sealed_evidence_rejects_missing_manifest_path(tmp_path: Path) -> None:
    identity = evaluation_evidence_identity(
        policy_path="organization/base-policy",
        policy_revision="immutable-revision",
        adapter_path=None,
        checkpoint_path=None,
        evaluation_manifest_path=str(tmp_path / "missing-manifest.json"),
        step=0,
    )

    with pytest.raises(ValueError, match="manifest.*exist"):
        require_resolved_sealed_identity(identity)
