"""OpenVLA action processing and LeRobot policy adapter for LIBERO."""

from __future__ import annotations

import random
from typing import Any, Mapping

import numpy as np

from art_embodied.integrations.lerobot import LeRobotActionPrediction
from art_embodied.trajectories import Action
from art_embodied.utils import make_json_safe

from .records import record_libero_observation
from .settings import LiberoSettings


class OpenVLAAdapter:
    """Adapt ART's native OpenVLA action to LeRobot's episode contract."""

    def __init__(self, policy: Any, settings: LiberoSettings) -> None:
        self.policy = policy
        self.settings = settings

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            _seed_policy(seed)

    def predict(
        self,
        observation: Mapping[str, Any],
        *,
        task: str | None,
        step: int,
        seed: int | None = None,
    ) -> LeRobotActionPrediction:
        # The episode runner owns deterministic episode initialization through
        # reset(). Re-seeding at every policy decision changes the stochastic
        # process and does not match RLinf's continuous episode-local RNG stream.
        del seed
        recorded = self.policy.act(
            record_libero_observation(observation=observation, step=step),
            {"scenario": {"task": task}, "step": step},
        )
        native_action = prepare_recorded_openvla_action(
            recorded,
            settings=self.settings,
            step=step,
        )
        return LeRobotActionPrediction(
            native_action=native_action,
            action=recorded,
        )

    def stateful_component_ids(self) -> tuple[int, ...]:
        return (id(self.policy),)


def prepare_recorded_openvla_action(
    recorded: Action,
    *,
    settings: LiberoSettings,
    step: int,
) -> np.ndarray:
    native_action = process_openvla_action_chunk(
        recorded.decoded,
        chunk_size=settings.action_chunk_size,
        normalize_gripper=settings.normalize_gripper,
        binarize_gripper=settings.binarize_gripper,
        invert_gripper=settings.invert_gripper,
    )
    recorded.metadata.update(
        {
            "framework": "art-openvla-oft-libero",
            "policy_step": step,
            "observation_index": step,
            "processed_action_chunk": make_json_safe(native_action),
            "action_postprocessing": {
                "normalize_gripper": settings.normalize_gripper,
                "binarize_gripper": settings.binarize_gripper,
                "invert_gripper": settings.invert_gripper,
            },
        }
    )
    return native_action


def process_openvla_action_chunk(
    decoded: Any,
    *,
    chunk_size: int,
    normalize_gripper: bool,
    binarize_gripper: bool,
    invert_gripper: bool,
) -> np.ndarray:
    # RLinf forwards OpenVLA-OFT's unnormalized NumPy actions without a
    # float32 round-trip. Preserve that dtype: even sub-ULP action differences
    # can change the next observation in a closed-loop physics rollout.
    actions = np.asarray(decoded)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim == 1:
        actions = actions.reshape(-1, 7)
    elif actions.ndim == 2 and actions.shape == (1, chunk_size * 7):
        actions = actions.reshape(-1, 7)
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise ValueError(f"Invalid OpenVLA action chunk shape: {actions.shape}")
    if actions.shape[0] < chunk_size:
        raise ValueError(
            "OpenVLA action chunk is shorter than the configured fixed geometry: "
            f"rows={actions.shape[0]}, required={chunk_size}"
        )
    if not np.issubdtype(actions.dtype, np.floating):
        actions = actions.astype(np.float64)
    else:
        actions = actions.copy()
    actions = actions[:chunk_size, :7]
    if normalize_gripper:
        actions[:, -1] = 2.0 * actions[:, -1] - 1.0
    if binarize_gripper:
        actions[:, -1] = np.sign(actions[:, -1])
    if invert_gripper:
        actions[:, -1] *= -1.0
    return actions


def _seed_policy(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    try:
        import torch
    except ModuleNotFoundError:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
