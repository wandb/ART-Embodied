"""Export successful ART-Embodied LIBERO rollouts as a LeRobotDataset."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from art_embodied.trajectories import EmbodiedTrajectory


@dataclass(frozen=True, slots=True)
class LiberoDatasetExportSpec:
    """The explicit observation/action contract of a LIBERO SFT dataset."""

    repo_id: str
    root: Path
    fps: int = 10
    image_height: int = 256
    image_width: int = 256
    source_image_height: int | None = None
    source_image_width: int | None = None
    robot_type: str = "panda"
    primary_image_key: str = "image"
    wrist_image_key: str = "wrist_image"
    state_key: str = "proprio_state"
    use_videos: bool = True


@dataclass(frozen=True, slots=True)
class LiberoDatasetExportReport:
    """Auditable accounting for one successful-trajectory export."""

    source_trajectories: int
    successful_trajectories: int
    exported_episodes: int
    exported_frames: int
    scenario_ids: tuple[str, ...]


def create_normalization_view(
    *,
    source_root: Path,
    output_root: Path,
    normalization_stats_path: Path,
) -> Path:
    """Create a zero-copy dataset view with explicit reference normalization.

    Continued SFT on a narrow task slice should not silently replace a policy's
    global state/action normalization with slice-local statistics. The returned
    LeRobotDataset view links immutable frame data, copies metadata, replaces
    only ``meta/stats.json``, and records both hashes in a provenance sidecar.
    """

    source_root = source_root.resolve()
    output_root = output_root.resolve()
    normalization_stats_path = normalization_stats_path.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite dataset view: {output_root}")
    source_stats = source_root / "meta" / "stats.json"
    source_info = source_root / "meta" / "info.json"
    for required in (source_stats, source_info, normalization_stats_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    reference_stats = json.loads(normalization_stats_path.read_text(encoding="utf-8"))
    candidate_stats = json.loads(source_stats.read_text(encoding="utf-8"))
    _validate_normalization_contract(candidate_stats, reference_stats)

    output_root.mkdir(parents=True)
    shutil.copytree(source_root / "meta", output_root / "meta")
    for directory in ("data", "videos"):
        source_directory = source_root / directory
        if source_directory.exists():
            (output_root / directory).symlink_to(
                source_directory,
                target_is_directory=True,
            )
    (output_root / "meta" / "stats.json").write_text(
        json.dumps(reference_stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_sidecar = source_root / "art_embodied_export.json"
    if source_sidecar.is_file():
        shutil.copy2(source_sidecar, output_root / source_sidecar.name)
    provenance = {
        "schema_version": 1,
        "source_root": str(source_root),
        "source_stats_sha256": _file_sha256(source_stats),
        "normalization_stats_path": str(normalization_stats_path),
        "normalization_stats_sha256": _file_sha256(normalization_stats_path),
        "linked_directories": [
            directory
            for directory in ("data", "videos")
            if (output_root / directory).exists()
        ],
    }
    (output_root / "art_embodied_normalization_view.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_root


def libero_lerobot_features(spec: LiberoDatasetExportSpec) -> dict[str, dict[str, Any]]:
    """Return the standard LeRobot LIBERO SFT feature contract."""

    image_feature = {
        "dtype": "video" if spec.use_videos else "image",
        "shape": (spec.image_height, spec.image_width, 3),
        "names": ["height", "width", "channel"],
    }
    return {
        "observation.images.image": dict(image_feature),
        "observation.images.image2": dict(image_feature),
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["actions"],
        },
    }


def trajectory_to_lerobot_frames(
    trajectory: EmbodiedTrajectory,
    spec: LiberoDatasetExportSpec,
) -> list[dict[str, Any]]:
    """Convert a successful trajectory into primitive supervised frames.

    Horizon-1 rollouts use the policy observation directly. Multi-primitive
    chunks are accepted only when the environment captured the exact observation
    immediately before every executed action. This preserves the teacher's
    deployment horizon without introducing stale observation/action pairs.
    """

    if not bool(trajectory.metrics.get("success", False)):
        raise ValueError("Only successful trajectories may enter the SFT dataset")
    observations = {
        observation.step: observation for observation in trajectory.observations
    }
    frames: list[dict[str, Any]] = []
    for action in trajectory.actions:
        observation_index = int(action.metadata.get("observation_index", action.step))
        observation = observations.get(observation_index)
        if observation is None or not isinstance(observation.value, Mapping):
            raise ValueError(
                f"Action step {action.step} has no retained mapping observation at "
                f"index {observation_index}"
            )
        executed = np.asarray(
            action.metadata.get("executed_action_vectors", []), dtype=np.float32
        )
        if executed.ndim != 2 or executed.shape[1] != 7:
            raise ValueError(
                "LeRobot SFT export requires executed actions with shape [N, 7], "
                f"got {executed.shape} at action step {action.step}"
            )
        primitive_observations = action.metadata.get("primitive_observations_before")
        if primitive_observations is None:
            if len(executed) != 1:
                raise ValueError(
                    "Multi-primitive SFT export requires exact observations before "
                    f"each action; got {len(executed)} unobserved primitives at "
                    f"action step {action.step}"
                )
            frame_observations = [observation.value]
        elif not isinstance(primitive_observations, list) or len(
            primitive_observations
        ) != len(executed):
            raise ValueError(
                "Captured primitive observations must align with executed actions"
            )
        else:
            frame_observations = primitive_observations
        for primitive_index, (value, action_vector) in enumerate(
            zip(frame_observations, executed, strict=True)
        ):
            if not isinstance(value, Mapping):
                raise ValueError(
                    "Primitive observation must be a mapping at action step "
                    f"{action.step}, primitive {primitive_index}"
                )
            primary = _validated_image(
                value.get(spec.primary_image_key),
                name=spec.primary_image_key,
                spec=spec,
            )
            wrist = _validated_image(
                value.get(spec.wrist_image_key),
                name=spec.wrist_image_key,
                spec=spec,
            )
            state = np.asarray(value.get(spec.state_key), dtype=np.float32)
            if state.shape != (8,):
                raise ValueError(
                    f"LIBERO state must have shape (8,), got {state.shape} at "
                    f"action step {action.step}, primitive {primitive_index}"
                )
            frames.append(
                {
                    "observation.images.image": primary,
                    "observation.images.image2": wrist,
                    "observation.state": state,
                    "action": action_vector,
                    "task": trajectory.task,
                }
            )
    if not frames:
        raise ValueError("Successful trajectory contains no exportable actions")
    return frames


def export_successful_trajectories(
    trajectories: Iterable[EmbodiedTrajectory],
    spec: LiberoDatasetExportSpec,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> LiberoDatasetExportReport:
    """Create a finalized LeRobotDataset and an ART provenance sidecar."""

    source = tuple(trajectories)
    successful = tuple(
        trajectory
        for trajectory in source
        if bool(trajectory.metrics.get("success", False))
    )
    if not successful:
        raise ValueError("No successful trajectories were available for SFT export")
    writer = LiberoTrajectoryDatasetWriter(spec, provenance=provenance)
    try:
        for trajectory in successful:
            writer.add_successful_trajectory(trajectory)
        streamed_report = writer.finalize()
    except BaseException:
        writer.abort()
        raise
    return LiberoDatasetExportReport(
        source_trajectories=len(source),
        successful_trajectories=len(successful),
        exported_episodes=streamed_report.exported_episodes,
        exported_frames=streamed_report.exported_frames,
        scenario_ids=streamed_report.scenario_ids,
    )


class LiberoTrajectoryDatasetWriter:
    """Stream validated successful trajectories into one LeRobotDataset."""

    def __init__(
        self,
        spec: LiberoDatasetExportSpec,
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.spec = spec
        self.provenance = dict(provenance or {})
        self.dataset = LeRobotDataset.create(
            repo_id=spec.repo_id,
            fps=spec.fps,
            features=libero_lerobot_features(spec),
            root=spec.root,
            robot_type=spec.robot_type,
            use_videos=spec.use_videos,
            image_writer_processes=0,
            image_writer_threads=4,
        )
        self._scenario_ids: list[str] = []
        self._frame_count = 0
        self._finalized = False

    def add_successful_trajectory(self, trajectory: EmbodiedTrajectory) -> int:
        """Validate and persist one episode, returning its frame count."""

        frames = trajectory_to_lerobot_frames(trajectory, self.spec)
        return self.add_episode_frames(
            frames,
            scenario_id=str(trajectory.metadata.get("scenario_id", "unknown")),
        )

    def add_episode_frames(
        self,
        frames: Iterable[Mapping[str, Any]],
        *,
        scenario_id: str,
    ) -> int:
        """Persist one already-normalized LeRobot episode.

        This lower-level entry point lets rehearsal builders combine existing
        LeRobot demonstrations with newly collected teacher trajectories while
        retaining the same writer, schema, and provenance contract.
        """

        if self._finalized:
            raise RuntimeError("Cannot append to a finalized LeRobotDataset")
        episode = tuple(frames)
        if not episode:
            raise ValueError("Cannot append an empty LeRobot episode")
        for frame in episode:
            self.dataset.add_frame(dict(frame))
        self.dataset.save_episode()
        self._frame_count += len(episode)
        self._scenario_ids.append(scenario_id)
        return len(episode)

    def finalize(self) -> LiberoDatasetExportReport:
        """Finalize videos/metadata and write the auditable ART sidecar."""

        if self._finalized:
            raise RuntimeError("LeRobotDataset writer is already finalized")
        if not self._scenario_ids:
            raise ValueError("Cannot finalize a dataset without successful episodes")
        self.dataset.finalize()
        self._finalized = True
        report = LiberoDatasetExportReport(
            source_trajectories=len(self._scenario_ids),
            successful_trajectories=len(self._scenario_ids),
            exported_episodes=len(self._scenario_ids),
            exported_frames=self._frame_count,
            scenario_ids=tuple(self._scenario_ids),
        )
        sidecar = {
            "schema_version": 1,
            "export_spec": {**asdict(self.spec), "root": str(self.spec.root)},
            "report": asdict(report),
            "provenance": self.provenance,
        }
        self.spec.root.mkdir(parents=True, exist_ok=True)
        (self.spec.root / "art_embodied_export.json").write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report

    def abort(self) -> None:
        """Stop asynchronous image writing after an interrupted collection."""

        if self._finalized:
            return
        dataset_writer = getattr(self.dataset, "writer", None)
        image_writer = getattr(dataset_writer, "image_writer", None)
        if image_writer is not None:
            image_writer.stop()


def _validated_image(
    value: Any,
    *,
    name: str,
    spec: LiberoDatasetExportSpec,
) -> np.ndarray:
    image = np.asarray(value)
    source_height = spec.source_image_height or spec.image_height
    source_width = spec.source_image_width or spec.image_width
    expected_source = (source_height, source_width, 3)
    if image.shape != expected_source:
        raise ValueError(
            f"LIBERO source image {name!r} must have shape {expected_source}, "
            f"got {image.shape}"
        )
    if image.dtype != np.uint8:
        raise ValueError(
            f"LIBERO image {name!r} must retain uint8 pixels, got {image.dtype}"
        )
    if image.shape[:2] == (spec.image_height, spec.image_width):
        return image
    from PIL import Image

    resized = Image.fromarray(image).resize(
        (spec.image_width, spec.image_height),
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(resized, dtype=np.uint8)


def _validate_normalization_contract(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> None:
    required = {"observation.state", "action"}
    for name, payload in (("candidate", candidate), ("reference", reference)):
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(f"{name} normalization stats are missing: {missing}")
    for key in required:
        for statistic in ("mean", "std"):
            candidate_values = np.asarray(candidate[key][statistic], dtype=np.float64)
            reference_values = np.asarray(reference[key][statistic], dtype=np.float64)
            if candidate_values.shape != reference_values.shape:
                raise ValueError(
                    f"Normalization shape mismatch for {key}.{statistic}: "
                    f"candidate={candidate_values.shape}, "
                    f"reference={reference_values.shape}"
                )
            if not np.isfinite(reference_values).all():
                raise ValueError(
                    f"Reference normalization is non-finite: {key}.{statistic}"
                )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
