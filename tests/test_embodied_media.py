from __future__ import annotations

import numpy as np

from art_embodied.lookahead import LookaheadFrame
from art_embodied.media import (
    LookaheadPreviewRecorder,
    RolloutVideoRecorder,
    compose_lookahead_filmstrip,
    wandb_video_payload,
)
from art_embodied.trajectories import EmbodiedTrajectory, MediaRef


class _RenderEnvironment:
    def render(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)


def test_rollout_video_caption_identifies_replayable_outcome(tmp_path) -> None:
    recorder = RolloutVideoRecorder(tmp_path, fps=4, max_frames=2)
    recorder.capture(_RenderEnvironment(), step=0)
    trajectory = EmbodiedTrajectory(
        task="place the block",
        metrics={"success": True},
        metadata={
            "scenario_id": "libero_spatial/3",
            "environment_seed": 17,
            "policy_seed": 23,
        },
    )

    media = recorder.finalize(trajectory)

    assert len(media) == 1
    assert media[0].caption == (
        "task=place the block | outcome=success | "
        "scenario_id=libero_spatial/3 | environment_seed=17 | policy_seed=23"
    )
    assert media[0].metadata["success"] is True
    assert media[0].metadata["scenario_id"] == "libero_spatial/3"


def test_lookahead_preview_preserves_temporal_order_in_separate_panels() -> None:
    current = np.zeros((128, 128, 3), dtype=np.uint8)
    near = current.copy()
    far = current.copy()
    near[:, :] = (255, 0, 0)
    far[:, :] = (0, 255, 0)

    filmstrip = np.asarray(
        compose_lookahead_filmstrip(
            current,
            [
                LookaheadFrame(image=near, action_index=2),
                LookaheadFrame(image=far, action_index=7),
            ],
            execution_horizon=2,
            model_horizon=8,
            max_panels=2,
        )
    )

    assert filmstrip.shape == (128, 256, 3)
    assert filmstrip[100, 160, 0] > filmstrip[100, 160, 1]
    assert filmstrip[100, 224, 1] > filmstrip[100, 224, 0]


def test_lookahead_recorder_labels_media_role(tmp_path) -> None:
    recorder = LookaheadPreviewRecorder(tmp_path, max_frames=1)
    recorder.capture_lookahead(
        np.zeros((8, 8, 3), dtype=np.uint8),
        [
            LookaheadFrame(
                image=np.full((8, 8, 3), 255, dtype=np.uint8),
                action_index=2,
            )
        ],
        step=0,
        execution_horizon=2,
        model_horizon=8,
    )

    media = recorder.finalize()

    assert media[0].metadata["role"] == "lookahead_preview"
    assert media[0].metadata["experimental"] is True
    assert media[0].metadata["display"] == "current_plus_future_filmstrip"


def test_wandb_video_roles_use_separate_stable_namespaces(tmp_path) -> None:
    simulation = tmp_path / "simulation.gif"
    lookahead = tmp_path / "lookahead.gif"
    simulation.write_bytes(b"GIF89a")
    lookahead.write_bytes(b"GIF89a")
    trajectory = EmbodiedTrajectory(
        task="pick",
        media=[
            MediaRef(
                uri=simulation.resolve().as_uri(),
                kind="video",
                metadata={"role": "simulation", "format": "gif"},
            ),
            MediaRef(
                uri=lookahead.resolve().as_uri(),
                kind="video",
                metadata={"role": "lookahead_preview", "format": "gif"},
            ),
        ],
    )

    class _Wandb:
        class Video:
            def __init__(self, path, **_kwargs):
                self.path = path

    simulation_payload = wandb_video_payload(
        trajectory,
        prefix="media/simulation/train",
        media_role="simulation",
        wandb_module=_Wandb,
    )
    lookahead_payload = wandb_video_payload(
        trajectory,
        prefix="media/lookahead/train",
        media_role="lookahead_preview",
        wandb_module=_Wandb,
    )

    assert set(simulation_payload) == {"media/simulation/train/0"}
    assert set(lookahead_payload) == {"media/lookahead/train/0"}
    assert simulation_payload["media/simulation/train/0"].path == str(simulation)
    assert lookahead_payload["media/lookahead/train/0"].path == str(lookahead)
