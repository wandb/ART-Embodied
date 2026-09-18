"""Deterministic training-data replay for GR00T N1.7 gradient workers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GR00TN17SFTReplayExample:
    step_data: Any
    task_id: str
    episode_index: int
    step_index: int
    seed: int
    source_info_sha256: str


def sft_replay_sample_seed(
    *,
    seed: int,
    update_index: int,
    subupdate_index: int,
    worker_index: int,
) -> int:
    """Derive a stable 63-bit seed without depending on Python hash randomization."""

    payload = f"{seed}:{update_index}:{subupdate_index}:{worker_index}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


class GR00TN17SFTReplayProvider:
    """Load one deterministic native-SFT example for one task-owned worker."""

    def __init__(
        self,
        *,
        policy: Any,
        dataset_root: str | Path,
        dataset_prefix: str,
        task_ids: list[str],
        seed: int,
        worker_index: int,
    ) -> None:
        if not task_ids:
            raise ValueError("SFT replay requires at least one task")
        if not 0 <= worker_index < len(task_ids):
            raise ValueError(
                f"SFT replay worker index {worker_index} has no task in {task_ids}"
            )
        native_policy = getattr(policy, "native_policy", None)
        if native_policy is None:
            raise RuntimeError("SFT replay requires a loaded native GR00T policy")
        processor = native_policy.processor
        if bool(getattr(processor, "training", False)):
            raise RuntimeError(
                "SFT replay processor must remain in deterministic eval mode"
            )

        self.task_id = str(task_ids[worker_index])
        self.seed = int(seed)
        self.worker_index = int(worker_index)
        self.dataset_path = (
            Path(dataset_root).expanduser() / f"{dataset_prefix}.{self.task_id}"
        )
        marker_path = self.dataset_path / "art_embodied_preparation.json"
        if not marker_path.is_file():
            raise FileNotFoundError(
                f"SFT replay dataset lacks preparation marker: {marker_path}"
            )
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("schema") != "art-embodied.gr00t-n1d7-legacy-lerobot-overlay.v1":
            raise ValueError(f"Unsupported SFT replay dataset marker: {marker_path}")
        source_info_sha256 = marker.get("source_info_sha256")
        if not isinstance(source_info_sha256, str) or len(source_info_sha256) != 64:
            raise ValueError(f"Invalid source_info_sha256 in {marker_path}")
        self.source_info_sha256 = source_info_sha256

        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

        self.loader = LeRobotEpisodeLoader(
            dataset_path=self.dataset_path,
            modality_configs=native_policy.modality_configs,
        )
        action_indices = native_policy.modality_configs["action"].delta_indices
        self.action_horizon = max(action_indices) - min(action_indices) + 1
        self.valid_episode_indices = [
            index
            for index, length in enumerate(self.loader.episode_lengths)
            if int(length) >= self.action_horizon
        ]
        if not self.valid_episode_indices:
            raise ValueError(
                f"SFT replay dataset has no episode of horizon {self.action_horizon}: "
                f"{self.dataset_path}"
            )

    def sample(
        self,
        *,
        update_index: int,
        subupdate_index: int,
    ) -> GR00TN17SFTReplayExample:
        sample_seed = sft_replay_sample_seed(
            seed=self.seed,
            update_index=update_index,
            subupdate_index=subupdate_index,
            worker_index=self.worker_index,
        )
        rng = np.random.default_rng(sample_seed)
        episode_index = self.valid_episode_indices[
            int(rng.integers(0, len(self.valid_episode_indices)))
        ]
        effective_length = (
            int(self.loader.get_episode_length(episode_index)) - self.action_horizon + 1
        )
        step_index = int(rng.integers(0, effective_length))
        episode_data = self.loader[episode_index]

        from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
        from gr00t.data.embodiment_tags import EmbodimentTag

        step_data = extract_step_data(
            episode_data,
            step_index,
            self.loader.modality_configs,
            EmbodimentTag.ROBOCASA_GR1_TABLETOP,
            allow_padding=False,
        )
        return GR00TN17SFTReplayExample(
            step_data=step_data,
            task_id=self.task_id,
            episode_index=episode_index,
            step_index=step_index,
            seed=sample_seed,
            source_info_sha256=self.source_info_sha256,
        )
