"""Build a small anti-forgetting SFT dataset from base and teacher episodes."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator, Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np

from examples.embodied.libero.dataset_export import (
    LiberoDatasetExportSpec,
    LiberoTrajectoryDatasetWriter,
    create_normalization_view,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-repo-id", default="lerobot/libero")
    parser.add_argument("--base-revision")
    parser.add_argument("--base-task-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--base-episodes-per-task", type=int, default=2)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--teacher-repo-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--normalization-view-root", type=Path, required=True)
    parser.add_argument("--output-repo-id", required=True)
    parser.add_argument("--video-backend", default="pyav")
    return parser.parse_args()


def discover_task_episode_ids(
    data_root: Path,
    *,
    task_ids: Sequence[int],
    episodes_per_task: int,
) -> dict[int, tuple[int, ...]]:
    """Select the first episode IDs for each task from local parquet metadata."""

    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive")
    requested = tuple(dict.fromkeys(int(task_id) for task_id in task_ids))
    if not requested:
        raise ValueError("At least one base task ID is required")
    parquet_files = sorted(data_root.glob("data/**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No source parquet files found under {data_root}")

    import pandas as pd

    found: dict[int, set[int]] = {task_id: set() for task_id in requested}
    for parquet_path in parquet_files:
        frame = pd.read_parquet(
            parquet_path,
            columns=["episode_index", "task_index"],
        ).drop_duplicates()
        for episode_index, task_index in frame.itertuples(index=False, name=None):
            task_index = int(task_index)
            if task_index in found:
                found[task_index].add(int(episode_index))
    selected = {
        task_id: tuple(sorted(episodes)[:episodes_per_task])
        for task_id, episodes in found.items()
    }
    incomplete = {
        task_id: episodes
        for task_id, episodes in selected.items()
        if len(episodes) != episodes_per_task
    }
    if incomplete:
        raise ValueError(
            f"Insufficient base demonstrations for requested tasks: {incomplete}"
        )
    return selected


def iter_lerobot_episodes(dataset: Any) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    """Yield source episodes as frames accepted by ``LeRobotDataset.add_frame``."""

    episode_index: int | None = None
    frames: list[dict[str, Any]] = []
    for item in dataset:
        current_index = int(_to_numpy(item["episode_index"]).item())
        if episode_index is not None and current_index != episode_index:
            yield episode_index, frames
            frames = []
        episode_index = current_index
        frames.append(_source_item_to_frame(item))
    if episode_index is not None:
        yield episode_index, frames


def build_rehearsal_dataset(
    *,
    base_dataset: Any,
    teacher_dataset: Any,
    selected_base_episodes: Mapping[int, Sequence[int]],
    output_root: Path,
    normalization_view_root: Path,
    output_repo_id: str,
    normalization_stats_path: Path,
) -> dict[str, Any]:
    """Materialize base rehearsal and successful teacher episodes together."""

    if output_root.exists() or normalization_view_root.exists():
        raise FileExistsError(
            "Refusing to overwrite rehearsal output or normalization view"
        )
    flattened_base_ids = {
        int(episode_id)
        for episode_ids in selected_base_episodes.values()
        for episode_id in episode_ids
    }
    writer = LiberoTrajectoryDatasetWriter(
        LiberoDatasetExportSpec(
            repo_id=output_repo_id,
            root=output_root,
            use_videos=True,
        ),
        provenance={
            "kind": "base_rehearsal_plus_successful_teacher",
            "base_repo_id": str(base_dataset.repo_id),
            "base_revision": str(base_dataset.revision),
            "base_root": str(Path(base_dataset.root).resolve()),
            "selected_base_episodes": {
                str(task_id): list(episode_ids)
                for task_id, episode_ids in selected_base_episodes.items()
            },
            "teacher_repo_id": str(teacher_dataset.repo_id),
            "teacher_revision": str(teacher_dataset.revision),
            "teacher_root": str(Path(teacher_dataset.root).resolve()),
        },
    )
    base_count = 0
    teacher_count = 0
    try:
        for episode_id, frames in iter_lerobot_episodes(base_dataset):
            if episode_id not in flattened_base_ids:
                continue
            writer.add_episode_frames(
                frames,
                scenario_id=f"base-rehearsal/episode-{episode_id:06d}",
            )
            base_count += 1
        for episode_id, frames in iter_lerobot_episodes(teacher_dataset):
            writer.add_episode_frames(
                frames,
                scenario_id=f"successful-teacher/episode-{episode_id:06d}",
            )
            teacher_count += 1
        report = writer.finalize()
    except BaseException:
        writer.abort()
        raise
    if base_count != len(flattened_base_ids):
        raise RuntimeError(
            f"Expected {len(flattened_base_ids)} base episodes, wrote {base_count}"
        )
    create_normalization_view(
        source_root=output_root,
        output_root=normalization_view_root,
        normalization_stats_path=normalization_stats_path,
    )
    summary = {
        "schema_version": 1,
        "base_episodes": base_count,
        "teacher_episodes": teacher_count,
        "total_episodes": report.exported_episodes,
        "total_frames": report.exported_frames,
        "output_root": str(output_root.resolve()),
        "normalization_view_root": str(normalization_view_root.resolve()),
    }
    (normalization_view_root / "art_embodied_rehearsal.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _source_item_to_frame(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "observation.images.image": _chw_to_hwc_uint8(item["observation.images.image"]),
        "observation.images.image2": _chw_to_hwc_uint8(
            item["observation.images.image2"]
        ),
        "observation.state": _to_numpy(item["observation.state"]).astype(
            np.float32, copy=False
        ),
        "action": _to_numpy(item["action"]).astype(np.float32, copy=False),
        "task": str(item["task"]),
    }


def _chw_to_hwc_uint8(value: Any) -> np.ndarray:
    image = _to_numpy(value)
    if image.shape == (3, 256, 256):
        image = np.moveaxis(image, 0, -1)
    if image.shape != (256, 256, 3) or image.dtype != np.uint8:
        raise ValueError(
            "Expected a uint8 LIBERO image with shape [3, 256, 256] or "
            f"[256, 256, 3], got shape={image.shape}, dtype={image.dtype}"
        )
    return np.ascontiguousarray(image)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def main() -> None:
    args = _parse_args()
    task_ids = tuple(int(value) for value in args.base_task_ids.split(","))

    from huggingface_hub import snapshot_download
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    metadata_snapshot = Path(
        snapshot_download(
            args.base_repo_id,
            repo_type="dataset",
            revision=args.base_revision,
            allow_patterns=["data/**/*.parquet", "meta/**"],
        )
    )
    selected = discover_task_episode_ids(
        metadata_snapshot,
        task_ids=task_ids,
        episodes_per_task=args.base_episodes_per_task,
    )
    selected_ids = sorted(
        episode_id for episode_ids in selected.values() for episode_id in episode_ids
    )
    base_dataset = LeRobotDataset(
        args.base_repo_id,
        episodes=selected_ids,
        revision=args.base_revision,
        video_backend=args.video_backend,
        return_uint8=True,
    )
    teacher_dataset = LeRobotDataset(
        args.teacher_repo_id,
        root=args.teacher_root,
        video_backend=args.video_backend,
        return_uint8=True,
    )
    summary = build_rehearsal_dataset(
        base_dataset=base_dataset,
        teacher_dataset=teacher_dataset,
        selected_base_episodes=selected,
        output_root=args.output_root,
        normalization_view_root=args.normalization_view_root,
        output_repo_id=args.output_repo_id,
        normalization_stats_path=Path(base_dataset.meta.root) / "meta" / "stats.json",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
