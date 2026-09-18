from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from art_embodied import EmbodiedExperimentConfig
from examples.embodied.libero import records as records_module
from examples.embodied.libero.records import build_evaluation_scenarios
from examples.embodied.libero.state_manifest import (
    file_sha256,
    load_state_manifest,
    state_sha256,
    validate_manifest_bddl_files,
    write_state_manifest,
)

ROOT = Path(__file__).parents[1]


def _write_manifest(tmp_path: Path) -> Path:
    states = {
        "task_00_state_00": np.array([1.0, 2.0, 3.0]),
        "task_01_state_00": np.array([4.0, 5.0, 6.0]),
    }
    entries = [
        {
            "id": "libero_object/generated-held-out/task-00/state-00",
            "task_id": 0,
            "state_key": "task_00_state_00",
            "generation_seed": 100,
            "state_sha256": state_sha256(states["task_00_state_00"]),
            "state_length": 3,
            "bddl_sha256": "a" * 64,
        },
        {
            "id": "libero_object/generated-held-out/task-01/state-00",
            "task_id": 1,
            "state_key": "task_01_state_00",
            "generation_seed": 101,
            "state_sha256": state_sha256(states["task_01_state_00"]),
            "state_length": 3,
            "bddl_sha256": "b" * 64,
        },
    ]
    return write_state_manifest(
        tmp_path / "generated",
        suite_name="libero_object",
        simulator_compatibility="rlinf_v01",
        states=states,
        entries=entries,
        generator={
            "policy_loaded": False,
            "selection_uses_model_outcomes": False,
        },
    )


def test_state_manifest_round_trip_is_hash_verified(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path)

    manifest = load_state_manifest(
        path,
        expected_suite_name="libero_object",
        expected_simulator_compatibility="rlinf_v01",
    )

    assert len(manifest.entries) == 2
    np.testing.assert_array_equal(
        manifest.states["task_00_state_00"],
        np.array([1.0, 2.0, 3.0]),
    )
    assert manifest.metadata["generator"]["policy_loaded"] is False
    assert manifest.metadata["generator"]["selection_uses_model_outcomes"] is False
    assert manifest.metadata["state_archive_sha256"] == file_sha256(
        manifest.archive_path
    )


@pytest.mark.parametrize(
    ("relative_path", "suite_name"),
    [
        (
            "examples/embodied/libero/state_manifests/"
            "pi0_spatial_dev_v1/manifest.json",
            "libero_spatial",
        ),
        (
            "examples/embodied/libero/state_manifests/"
            "pi05_long_dev_v1/manifest.json",
            "libero_10",
        ),
    ],
)
def test_bundled_pi_development_manifests_are_hash_verified(
    relative_path: str,
    suite_name: str,
) -> None:
    manifest = load_state_manifest(
        ROOT / relative_path,
        expected_suite_name=suite_name,
        expected_simulator_compatibility="rlinf_v01",
    )

    assert len(manifest.entries) == 100
    assert len(manifest.states) == 100
    assert manifest.metadata["generator"]["policy_loaded"] is False
    assert manifest.metadata["generator"]["selection_uses_model_outcomes"] is False


def test_state_manifest_rejects_archive_tampering(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    archive = path.parent / manifest["state_archive"]
    archive.write_bytes(archive.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="archive hash mismatch"):
        load_state_manifest(path)


def test_state_manifest_verifies_bddl_task_definitions(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path)
    task_zero = tmp_path / "task-zero.bddl"
    task_one = tmp_path / "task-one.bddl"
    task_zero.write_text("task zero")
    task_one.write_text("task one")
    raw = json.loads(path.read_text())
    raw["entries"][0]["bddl_sha256"] = file_sha256(task_zero)
    raw["entries"][1]["bddl_sha256"] = file_sha256(task_one)
    path.write_text(json.dumps(raw))
    manifest = load_state_manifest(path)

    validate_manifest_bddl_files(manifest, {0: task_zero, 1: task_one})

    task_one.write_text("changed task one")
    with pytest.raises(ValueError, match="BDDL hash mismatch"):
        validate_manifest_bddl_files(manifest, {0: task_zero, 1: task_one})


def test_generated_manifest_drives_explicit_evaluation_scenarios(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    config = EmbodiedExperimentConfig.from_yaml(
        ROOT / "examples/embodied/openvla_oft_libero_object_sft_baseline_eval.yaml"
    )
    kwargs = {
        **config.environment.kwargs,
        "evaluation_state_manifest": str(manifest_path),
    }
    config = config.model_copy(
        update={"environment": config.environment.model_copy(update={"kwargs": kwargs})}
    )
    monkeypatch.setattr(
        records_module,
        "_task_languages",
        lambda _settings: {task_id: f"task {task_id}" for task_id in range(10)},
    )

    scenarios = build_evaluation_scenarios(config)

    assert [scenario.id for scenario in scenarios] == [
        "libero_object/generated-held-out/task-00/state-00",
        "libero_object/generated-held-out/task-01/state-00",
    ]
    assert scenarios[0].payload == {
        "task_id": 0,
        "reset_options": {"manifest_state_key": "task_00_state_00"},
        "state_sha256": state_sha256(np.array([1.0, 2.0, 3.0])),
        "generation_seed": 100,
    }


def test_generated_manifest_filters_to_configured_task_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    config = EmbodiedExperimentConfig.from_yaml(
        ROOT / "examples/embodied/openvla_oft_libero_object_sft_baseline_eval.yaml"
    )
    kwargs = {
        **config.environment.kwargs,
        "task_ids": [1],
        "training_task_ids": [1],
        "training_trial_ids": list(range(10, 40)),
        "evaluation_state_manifest": str(manifest_path),
        "init_state_selection": "partitioned_random_reset",
    }
    config = config.model_copy(
        update={"environment": config.environment.model_copy(update={"kwargs": kwargs})}
    )
    monkeypatch.setattr(
        records_module,
        "_task_languages",
        lambda _settings: {1: "task 1"},
    )

    scenarios = build_evaluation_scenarios(config)

    assert [scenario.id for scenario in scenarios] == [
        "libero_object/generated-held-out/task-01/state-00"
    ]
