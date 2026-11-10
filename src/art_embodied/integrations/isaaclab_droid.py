"""Observation and action boundary for Arena's DROID/OpenPI example.

This uses Arena's absolute-joint-position embodiment, not LIBERO's end-effector
delta actions. OpenPI retains ownership of model normalization and sampling.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def pack_droid_observation(
    observation: Mapping[str, Any], *, prompt: str
) -> dict[str, Any]:
    """Map one unbatched Arena observation to the upstream OpenPI DROID API.

    Images are uint8 HWC RGB. Joint positions are seven absolute radians in
    panda_joint1..7 order. Gripper state is already normalized by the Arena
    embodiment (0=open, 1=closed); do not apply LIBERO's gripper conversion.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("DROID requires a nonempty language instruction")
    cameras = observation["camera_obs"]
    state = observation["policy"]
    return {
        "observation/exterior_image_1_left": _image(cameras["external_camera_rgb"]),
        "observation/wrist_image_left": _image(cameras["wrist_camera_rgb"]),
        "observation/joint_position": _state(state["joint_pos"], size=7),
        "observation/gripper_position": _state(state["gripper_pos"], size=1),
        "prompt": prompt,
    }


def droid_action_chunk(actions: Any, *, execution_horizon: int) -> np.ndarray:
    """Validate an OpenPI chunk and copy its executed prefix without rescaling.

    The eight columns are absolute arm joint positions followed by a gripper
    command. Arena applies its own >0.5 close threshold. This function neither
    clips predictions nor fabricates likelihoods for the evaluation-only API.
    """
    if type(execution_horizon) is not int or execution_horizon <= 0:
        raise ValueError("execution_horizon must be a positive integer")
    value = _numpy(actions)
    if (
        value.ndim != 2
        or value.shape[0] < execution_horizon
        or value.shape[1] != 8
        or not np.issubdtype(value.dtype, np.floating)
        or not np.isfinite(value).all()
    ):
        raise ValueError("Expected a finite floating-point DROID [horizon, 8] chunk")
    result = value[:execution_horizon].astype(np.float32, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("DROID actions overflow float32")
    return result


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _state(value: Any, *, size: int) -> np.ndarray:
    value = _numpy(value)
    if (
        value.shape != (size,)
        or not np.issubdtype(value.dtype, np.floating)
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"Expected a finite floating-point DROID state ({size},)")
    result = value.astype(np.float32, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("DROID state overflows float32")
    return result


def _image(value: Any) -> np.ndarray:
    from PIL import Image

    value = _numpy(value)
    if (
        value.dtype != np.uint8
        or value.ndim != 3
        or value.shape[2] != 3
        or min(value.shape[:2]) <= 0
    ):
        raise ValueError("Expected an unbatched uint8 HWC RGB camera image")
    height, width = value.shape[:2]
    if (height, width) == (224, 224):
        return value.copy()
    ratio = max(height / 224, width / 224)
    resized_height, resized_width = int(height / ratio), int(width / ratio)
    if min(resized_height, resized_width) == 0:
        raise ValueError("Camera aspect ratio produces an empty resized image")
    # Match openpi_client.image_tools (PIL bilinear), not the JAX model resize.
    resized = Image.fromarray(value).resize(
        (resized_width, resized_height), resample=Image.Resampling.BILINEAR
    )
    result = np.zeros((224, 224, 3), dtype=np.uint8)
    top, left = (224 - resized_height) // 2, (224 - resized_width) // 2
    result[top : top + resized_height, left : left + resized_width] = resized
    return result
