"""Deterministic, task-balanced native-SFT samples for pi0-FAST workers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class PI0FastSFTAnchorSample:
    task_index: int
    dataset_index: int
    seed: int
    loss: Any


def sft_anchor_sample_seed(
    *, seed: int, update_index: int, subupdate_index: int, task_index: int
) -> int:
    payload = f"{seed}:{update_index}:{subupdate_index}:{task_index}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


class PI0FastSFTAnchorProvider:
    """Own task-local LeRobot views and produce one native CE loss per task."""

    def __init__(
        self,
        *,
        policy: Any,
        dataset_repo_id: str,
        dataset_revision: str,
        task_indices: Sequence[int],
        seed: int,
    ) -> None:
        if not task_indices:
            raise ValueError("pi0-FAST SFT anchor worker has no assigned tasks")
        self.policy = policy
        self.seed = int(seed)
        self.datasets: dict[int, Any] = {}

        snapshot = _dataset_snapshot(dataset_repo_id, dataset_revision)
        for task_index in task_indices:
            task = int(task_index)
            episodes = _discover_task_episodes(snapshot, task)
            self.datasets[task] = _make_dataset(
                repo_id=dataset_repo_id,
                revision=dataset_revision,
                episodes=episodes,
                policy=policy,
            )

    def losses(
        self, *, update_index: int, subupdate_index: int
    ) -> list[PI0FastSFTAnchorSample]:
        import numpy as np

        samples = []
        for task_index, dataset in sorted(self.datasets.items()):
            sample_seed = sft_anchor_sample_seed(
                seed=self.seed,
                update_index=update_index,
                subupdate_index=subupdate_index,
                task_index=task_index,
            )
            rng = np.random.default_rng(sample_seed)
            dataset_index = int(rng.integers(0, len(dataset)))
            batch = _collate_one(dataset[dataset_index], dataset)
            prepared = _prepare_batch(
                batch, self.policy, list(dataset.meta.camera_keys)
            )
            loss, _details = self.policy.policy(prepared)
            samples.append(
                PI0FastSFTAnchorSample(
                    task_index=task_index,
                    dataset_index=dataset_index,
                    seed=sample_seed,
                    loss=loss,
                )
            )
        return samples


def _dataset_snapshot(repo_id: str, revision: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=["meta/**", "data/**"],
        )
    )


def _discover_task_episodes(root: Path, task_index: int) -> tuple[int, ...]:
    import pandas as pd

    episodes: set[int] = set()
    for parquet_path in sorted(root.glob("data/**/*.parquet")):
        frame = pd.read_parquet(parquet_path, columns=["episode_index", "task_index"])
        selected = frame.loc[frame["task_index"] == task_index, "episode_index"]
        episodes.update(int(value) for value in selected.unique())
    if not episodes:
        raise ValueError(f"No SFT anchor episodes for task_index={task_index}")
    return tuple(sorted(episodes))


def _make_dataset(
    *, repo_id: str, revision: str, episodes: tuple[int, ...], policy: Any
) -> Any:
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import (
        LeRobotDataset,
        LeRobotDatasetMetadata,
    )

    metadata = LeRobotDatasetMetadata(repo_id, revision=revision)
    return LeRobotDataset(
        repo_id,
        episodes=list(episodes),
        revision=revision,
        delta_timestamps=resolve_delta_timestamps(policy.config, metadata),
        return_uint8=True,
        video_backend="pyav",
    )


def _collate_one(sample: dict[str, Any], dataset: Any) -> dict[str, Any]:
    if not dataset.meta.has_language_columns:
        from torch.utils.data._utils.collate import default_collate

        return default_collate([sample])
    from lerobot.utils.collate import lerobot_collate_fn

    return lerobot_collate_fn([sample])


def _prepare_batch(batch: dict[str, Any], policy: Any, camera_keys: list[str]) -> Any:
    import torch

    for camera_key in camera_keys:
        value = batch.get(camera_key)
        if value is not None and value.dtype == torch.uint8:
            batch[camera_key] = value.float().div_(255.0)
    return policy.preprocessor(batch)
