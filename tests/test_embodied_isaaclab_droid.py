from __future__ import annotations

import hashlib

import numpy as np
import pytest

from art_embodied.integrations.isaaclab_droid import (
    droid_action_chunk,
    pack_droid_observation,
)


def observation():
    return {
        "camera_obs": {
            "external_camera_rgb": np.full((100, 200, 3), 127, dtype=np.uint8),
            "wrist_camera_rgb": np.full((224, 224, 3), 31, dtype=np.uint8),
        },
        "policy": {
            "joint_pos": np.arange(7, dtype=np.float32),
            "gripper_pos": np.array([0.25], dtype=np.float32),
        },
    }


def test_pack_droid_preserves_joint_order_gripper_and_rgb_padding():
    pytest.importorskip("PIL")
    obs = observation()
    packed = pack_droid_observation(obs, prompt="Pick up the cube.")
    exterior = packed["observation/exterior_image_1_left"]
    assert exterior.shape == (224, 224, 3)
    assert exterior.dtype == np.uint8
    assert (exterior[:56] == 0).all()
    assert (exterior[56:168] == 127).all()
    assert (exterior[168:] == 0).all()
    np.testing.assert_array_equal(packed["observation/joint_position"], np.arange(7))
    assert packed["observation/gripper_position"].item() == 0.25
    assert packed["prompt"] == "Pick up the cube."
    packed["observation/wrist_image_left"][:] = 0
    packed["observation/joint_position"][:] = -1
    assert (obs["camera_obs"]["wrist_camera_rgb"] == 31).all()
    assert obs["policy"]["joint_pos"][0] == 0


@pytest.mark.parametrize(
    "shape,expected",
    [
        (
            (720, 1280),
            "85977fc1545fff6835a2e26bacefa4d573dc72ef4db8070d2ecf98c5e236bfb7",
        ),
        (
            (1280, 720),
            "d019043390e5c385baa135351871dca60cdbba0d1142acb521f8278d63c97985",
        ),
        (
            (127, 221),
            "e9a007b3e316a26beeb5bd2bef5223175baa41c8aeb416dd1d18adf7835ee7b6",
        ),
        (
            (224, 224),
            "8b59e2ec94f1e993515e3ae1effe5720cc8d05f59e84265bc802552a3d4c4e22",
        ),
        ((3, 5), "03380d3116ea0a6dc23314ca722a8e3a067ee7b2f8dea382da626dc6f249e3c3"),
    ],
)
def test_image_matches_pinned_openpi_client(shape, expected):
    # openpi_client.image_tools at c23745b5ad24e98f66967ea795a07b2588ed6c79.
    pytest.importorskip("PIL")
    height, width = shape
    image = (np.arange(height * width * 3, dtype=np.uint32) % 256).astype(np.uint8)
    obs = observation()
    obs["camera_obs"]["external_camera_rgb"] = image.reshape(height, width, 3)
    result = pack_droid_observation(obs, prompt="Pick up the cube.")
    assert (
        hashlib.sha256(
            result["observation/exterior_image_1_left"].tobytes()
        ).hexdigest()
        == expected
    )


@pytest.mark.parametrize(
    "shape,dtype",
    [((3, 224, 224), np.uint8), ((224, 224, 4), np.uint8), ((224, 224, 3), np.float32)],
)
def test_pack_rejects_implicit_camera_conversions(shape, dtype):
    pytest.importorskip("PIL")
    obs = observation()
    obs["camera_obs"]["external_camera_rgb"] = np.zeros(shape, dtype=dtype)
    with pytest.raises(ValueError, match="HWC RGB"):
        pack_droid_observation(obs, prompt="Pick up the cube.")


@pytest.mark.parametrize(
    "key,value",
    [
        ("joint_pos", np.zeros(8)),
        ("joint_pos", np.full(7, np.nan)),
        ("gripper_pos", np.array(0.5)),
    ],
)
def test_pack_rejects_wrong_or_nonfinite_state(key, value):
    pytest.importorskip("PIL")
    obs = observation()
    obs["policy"][key] = value
    with pytest.raises(ValueError, match="DROID state"):
        pack_droid_observation(obs, prompt="Pick up the cube.")


def test_pack_rejects_empty_prompt():
    with pytest.raises(ValueError, match="instruction"):
        pack_droid_observation(observation(), prompt=" ")


def test_action_prefix_does_not_rescale_clip_or_alias():
    actions = np.linspace(-2.0, 3.0, 20 * 8, dtype=np.float32).reshape(20, 8)
    actual = droid_action_chunk(actions, execution_horizon=15)
    np.testing.assert_array_equal(actual, actions[:15])
    actual[:] = 0
    assert actions[0, 0] == -2.0


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros((14, 8)),
        np.zeros((15, 7)),
        np.zeros((1, 15, 8)),
        np.full((15, 8), np.nan),
        np.zeros((15, 8), dtype=np.int32),
    ],
)
def test_action_rejects_invalid_predictions(actions):
    with pytest.raises(ValueError, match="DROID"):
        droid_action_chunk(actions, execution_horizon=15)
