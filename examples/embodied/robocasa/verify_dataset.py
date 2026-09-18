"""Verify the pinned 24-task GR-1 dataset before allocating training GPUs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from art_embodied.utils import write_json_atomic

from .environment import DATASET_STATE_ACTION_LAYOUT
from .settings import RoboCasaTaskManifest

_ROBOT = "GR1ArmsAndWaistFourierHands"
_DATASET_REVISION = "09c6de8af50168090e7e9cc01e1ec3bce788de24"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _expected_slices() -> dict[str, dict[str, Any]]:
    result = {}
    start = 0
    for name, size in DATASET_STATE_ACTION_LAYOUT:
        result[name] = {
            "original_key": "action",
            "start": start,
            "end": start + size,
        }
        start += size
    if start != 44:
        raise AssertionError(f"Internal GR-1 action layout totals {start}, not 44")
    return result


def verify_dataset(*, root: Path, manifest_path: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if root.name != _DATASET_REVISION:
        raise ValueError(
            "RoboCasa dataset root must identify the pinned revision: "
            f"expected={_DATASET_REVISION}, actual={root.name}"
        )
    status_path = root / "download.status"
    if not status_path.is_file() or status_path.read_text(encoding="utf-8").strip() != (
        "complete"
    ):
        raise RuntimeError(f"Dataset download is not complete: {status_path}")
    manifest = RoboCasaTaskManifest.load(manifest_path)
    expected_action = _expected_slices()
    expected_state = {
        name: {**row, "original_key": "observation.state"}
        for name, row in expected_action.items()
    }
    task_reports = []
    total_episodes = 0
    total_frames = 0
    for task in manifest.tasks:
        task_root = root / "LeRobot" / task.dataset_id
        info = _read_json(task_root / "meta" / "info.json")
        modality = _read_json(task_root / "meta" / "modality.json")
        _read_json(task_root / "meta" / "stats.json")
        if info.get("robot_type") != _ROBOT:
            raise ValueError(f"Unexpected robot_type in {task.dataset_id}")
        if info.get("fps") != 20.0:
            raise ValueError(f"Unexpected fps in {task.dataset_id}: {info.get('fps')}")
        features = info.get("features", {})
        for key in ("observation.state", "action"):
            if features.get(key, {}).get("shape") != [44]:
                raise ValueError(f"{task.dataset_id} {key} is not 44-dimensional")
        if modality.get("action") != expected_action:
            raise ValueError(f"Action slices differ in {task.dataset_id}")
        if modality.get("state") != expected_state:
            raise ValueError(f"State slices differ in {task.dataset_id}")
        episodes = int(info["total_episodes"])
        videos = int(info["total_videos"])
        parquet_count = sum(1 for _ in (task_root / "data").rglob("*.parquet"))
        video_count = sum(1 for _ in (task_root / "videos").rglob("*.mp4"))
        if episodes != 1000 or videos != 1000:
            raise ValueError(
                f"{task.dataset_id} expected 1000 episodes/videos, "
                f"got episodes={episodes}, videos={videos}"
            )
        if parquet_count != episodes or video_count != videos:
            raise ValueError(
                f"{task.dataset_id} file count mismatch: parquet={parquet_count}, "
                f"video={video_count}, expected={episodes}"
            )
        total_episodes += episodes
        total_frames += int(info["total_frames"])
        task_reports.append(
            {
                "task_id": task.id,
                "dataset_id": task.dataset_id,
                "episodes": episodes,
                "frames": int(info["total_frames"]),
                "parquet_files": parquet_count,
                "video_files": video_count,
            }
        )
    return {
        "schema_version": 1,
        "status": "passed",
        "root": str(root),
        "dataset_revision": _DATASET_REVISION,
        "simulator_source_revision": manifest.source_revision,
        "task_count": len(task_reports),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "action_dim": 44,
        "state_dim": 44,
        "fps": 20.0,
        "tasks": task_reports,
    }


def main() -> None:
    args = _parse_args()
    report = verify_dataset(root=args.root, manifest_path=args.manifest)
    write_json_atomic(args.output, report, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
