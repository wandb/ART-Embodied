from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from art_embodied import Action, EmbodiedTrajectory, Observation
from examples.embodied.libero.build_rehearsal_dataset import (
    discover_task_episode_ids,
)
from examples.embodied.libero.collect_teacher_dataset import _rollout_contexts
from examples.embodied.libero.dataset_export import (
    LiberoDatasetExportSpec,
    LiberoTrajectoryDatasetWriter,
    create_normalization_view,
    libero_lerobot_features,
    trajectory_to_lerobot_frames,
)


def _trajectory(*, success: bool = True, primitives: int = 1) -> EmbodiedTrajectory:
    trajectory = EmbodiedTrajectory(
        task="put the object in the basket",
        metadata={"scenario_id": "libero_10/train/task-0/trial-10"},
    )
    trajectory.observations.append(
        Observation(
            step=0,
            kind="image",
            value={
                "image": np.zeros((12, 16, 3), dtype=np.uint8),
                "wrist_image": np.ones((12, 16, 3), dtype=np.uint8),
                "proprio_state": np.arange(8, dtype=np.float32),
            },
        )
    )
    trajectory.actions.append(
        Action(
            step=0,
            kind="continuous",
            raw=np.zeros((primitives, 7), dtype=np.float32),
            metadata={
                "observation_index": 0,
                "executed_action_vectors": np.zeros(
                    (primitives, 7), dtype=np.float32
                ).tolist(),
            },
        )
    )
    trajectory.metrics["success"] = success
    return trajectory


def _spec(tmp_path) -> LiberoDatasetExportSpec:
    return LiberoDatasetExportSpec(
        repo_id="local/pi05-teacher",
        root=tmp_path / "dataset",
        image_height=12,
        image_width=16,
    )


def test_trajectory_to_lerobot_frames_matches_standard_libero_schema(tmp_path) -> None:
    spec = _spec(tmp_path)

    frames = trajectory_to_lerobot_frames(_trajectory(), spec)

    assert len(frames) == 1
    assert set(frames[0]) == {
        "observation.images.image",
        "observation.images.image2",
        "observation.state",
        "action",
        "task",
    }
    assert frames[0]["observation.images.image"].shape == (12, 16, 3)
    assert frames[0]["observation.state"].shape == (8,)
    assert frames[0]["action"].shape == (7,)
    assert libero_lerobot_features(spec)["action"]["names"] == ["actions"]


def test_trajectory_to_lerobot_frames_rejects_failed_rollout(tmp_path) -> None:
    with pytest.raises(ValueError, match="Only successful trajectories"):
        trajectory_to_lerobot_frames(_trajectory(success=False), _spec(tmp_path))


def test_trajectory_to_lerobot_frames_rejects_unobserved_action_chunk(tmp_path) -> None:
    with pytest.raises(ValueError, match="exact observations before each action"):
        trajectory_to_lerobot_frames(_trajectory(primitives=5), _spec(tmp_path))


def test_trajectory_to_lerobot_frames_expands_observed_action_chunk(tmp_path) -> None:
    trajectory = _trajectory(primitives=3)
    trajectory.actions[0].metadata["primitive_observations_before"] = [
        {
            "image": np.full((12, 16, 3), index, dtype=np.uint8),
            "wrist_image": np.full((12, 16, 3), index + 1, dtype=np.uint8),
            "proprio_state": np.full(8, index, dtype=np.float32),
        }
        for index in range(3)
    ]

    frames = trajectory_to_lerobot_frames(trajectory, _spec(tmp_path))

    assert len(frames) == 3
    assert [float(frame["observation.state"][0]) for frame in frames] == [0, 1, 2]


def test_trajectory_to_lerobot_frames_resizes_to_standard_dataset_schema(
    tmp_path,
) -> None:
    spec = LiberoDatasetExportSpec(
        repo_id="local/pi05-teacher",
        root=tmp_path / "dataset",
        image_height=8,
        image_width=10,
        source_image_height=12,
        source_image_width=16,
    )

    frame = trajectory_to_lerobot_frames(_trajectory(), spec)[0]

    assert frame["observation.images.image"].shape == (8, 10, 3)
    assert frame["observation.images.image"].dtype == np.uint8


def test_streaming_writer_creates_lerobot_dataset_and_provenance(tmp_path) -> None:
    spec = LiberoDatasetExportSpec(
        repo_id="local/pi05-teacher",
        root=tmp_path / "dataset",
        image_height=12,
        image_width=16,
        use_videos=False,
    )
    writer = LiberoTrajectoryDatasetWriter(
        spec,
        provenance={"checkpoint": "step-000240"},
    )

    assert writer.add_successful_trajectory(_trajectory()) == 1
    report = writer.finalize()

    assert report.exported_episodes == 1
    assert report.exported_frames == 1
    sidecar = json.loads(
        (spec.root / "art_embodied_export.json").read_text(encoding="utf-8")
    )
    assert sidecar["provenance"]["checkpoint"] == "step-000240"
    assert sidecar["report"]["scenario_ids"] == ["libero_10/train/task-0/trial-10"]
    assert (spec.root / "meta" / "info.json").is_file()


def test_streaming_writer_accepts_existing_lerobot_episode(tmp_path) -> None:
    spec = LiberoDatasetExportSpec(
        repo_id="local/rehearsal",
        root=tmp_path / "dataset",
        image_height=12,
        image_width=16,
        use_videos=False,
    )
    frames = trajectory_to_lerobot_frames(_trajectory(), spec)
    writer = LiberoTrajectoryDatasetWriter(spec)

    assert writer.add_episode_frames(frames, scenario_id="base/episode-0") == 1
    report = writer.finalize()

    assert report.scenario_ids == ("base/episode-0",)


def test_discover_task_episode_ids_is_deterministic_and_balanced(tmp_path) -> None:
    data_root = tmp_path / "source"
    parquet_root = data_root / "data" / "chunk-000"
    parquet_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "episode_index": [9, 4, 4, 8, 2, 6, 6],
            "task_index": [0, 1, 1, 0, 0, 1, 1],
        }
    ).to_parquet(parquet_root / "file-000.parquet")

    selected = discover_task_episode_ids(
        data_root,
        task_ids=[0, 1],
        episodes_per_task=2,
    )

    assert selected == {0: (2, 8), 1: (4, 6)}


def test_native_teacher_collection_uses_one_attempt_per_state() -> None:
    config = SimpleNamespace(
        experiment=SimpleNamespace(seed=17),
        fingerprint="config-fingerprint",
    )

    contexts = _rollout_contexts(
        config=config,
        group_index=3,
        scenario_id="libero_10/train/task-4/trial-20",
        attempts=1,
    )

    assert len(contexts) == 1
    assert contexts[0].attempt_index == 0
    assert contexts[0].config_fingerprint == "config-fingerprint"


def test_normalization_view_links_frames_and_preserves_actual_stats(tmp_path) -> None:
    source = tmp_path / "source"
    (source / "meta").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "videos").mkdir()
    (source / "meta" / "info.json").write_text("{}\n", encoding="utf-8")
    actual = {
        "observation.state": {"mean": [1.0] * 8, "std": [2.0] * 8},
        "action": {"mean": [3.0] * 7, "std": [4.0] * 7},
    }
    reference = {
        "observation.state": {"mean": [5.0] * 8, "std": [6.0] * 8},
        "action": {"mean": [7.0] * 7, "std": [8.0] * 7},
    }
    (source / "meta" / "stats.json").write_text(json.dumps(actual), encoding="utf-8")
    reference_path = tmp_path / "reference-stats.json"
    reference_path.write_text(json.dumps(reference), encoding="utf-8")

    output = create_normalization_view(
        source_root=source,
        output_root=tmp_path / "view",
        normalization_stats_path=reference_path,
    )

    assert (output / "data").is_symlink()
    assert json.loads((output / "meta" / "stats.json").read_text()) == reference
    sidecar = json.loads((output / "art_embodied_normalization_view.json").read_text())
    assert sidecar["source_root"] == str(source.resolve())
    assert sidecar["source_stats_sha256"] != sidecar["normalization_stats_sha256"]
