from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from art_embodied.integrations.lerobot_envhub import (
    LeRobotEnvHubSpec,
    load_lerobot_envhub_environment,
)


class _VectorEnvironment:
    num_envs = 1
    single_observation_space = SimpleNamespace(shape=None)
    single_action_space = SimpleNamespace(shape=(2,))

    def __init__(self) -> None:
        self.actions: list[np.ndarray] = []
        self.closed = False

    def reset(self, *, seed, options):
        assert seed == 17
        assert options == {"variant": 3}
        return (
            {
                "observation.state": np.array([[1.0, 2.0]], dtype=np.float32),
                "observation.images.front": np.zeros(
                    (1, 4, 4, 3), dtype=np.uint8
                ),
            },
            {
                "seed": np.array([17]),
                "label": np.array(["pick-cube"]),
            },
        )

    def step(self, action):
        assert action.shape == (1, 2)
        self.actions.append(action)
        return (
            {"observation.state": np.array([[3.0, 4.0]], dtype=np.float32)},
            np.array([1.0], dtype=np.float32),
            np.array([True]),
            np.array([False]),
            {"success": np.array([True])},
        )

    def render(self):
        return np.full((1, 4, 4, 3), 127, dtype=np.uint8)

    def close(self):
        self.closed = True


def _spec(**overrides) -> LeRobotEnvHubSpec:
    values = {
        "hub_path": "organization/envs@0123456789abcdef:env.py",
        "suite": "maniskill",
        "task_id": 4,
        "trust_remote_code": True,
    }
    values.update(overrides)
    return LeRobotEnvHubSpec(**values)


def test_envhub_loader_selects_and_unbatches_one_task() -> None:
    vector_env = _VectorEnvironment()
    calls = []

    def make_env(*args, **kwargs):
        calls.append((args, kwargs))
        return {"maniskill": {4: vector_env}}

    env = load_lerobot_envhub_environment(_spec(), make_env_fn=make_env)

    observation, info = env.reset(seed=17, options={"variant": 3})
    assert observation["observation.state"].shape == (2,)
    assert observation["observation.images.front"].shape == (4, 4, 3)
    assert info["seed"] == 17
    assert info["label"] == "pick-cube"

    observation, reward, terminated, truncated, info = env.step(
        np.array([0.25, -0.5], dtype=np.float32)
    )
    assert observation["observation.state"].tolist() == [3.0, 4.0]
    assert reward == 1.0
    assert terminated is True
    assert truncated is False
    assert bool(info["success"]) is True
    assert env.render().shape == (4, 4, 3)
    env.close()
    assert vector_env.closed is True
    assert calls == [
        (
            ("organization/envs@0123456789abcdef:env.py",),
            {
                "n_envs": 1,
                "use_async_envs": False,
                "hub_cache_dir": None,
                "trust_remote_code": True,
            },
        )
    ]


def test_envhub_loader_requires_explicit_remote_code_consent() -> None:
    with pytest.raises(ValueError, match="trust_remote_code=True"):
        load_lerobot_envhub_environment(
            _spec(trust_remote_code=False),
            make_env_fn=lambda *_args, **_kwargs: {},
        )


def test_envhub_loader_requires_revision_pin_by_default() -> None:
    with pytest.raises(ValueError, match="pin a revision"):
        load_lerobot_envhub_environment(
            _spec(hub_path="organization/envs"),
            make_env_fn=lambda *_args, **_kwargs: {},
        )


def test_envhub_loader_requires_suite_when_package_has_multiple() -> None:
    with pytest.raises(ValueError, match="multiple suites"):
        load_lerobot_envhub_environment(
            _spec(suite=None),
            make_env_fn=lambda *_args, **_kwargs: {
                "maniskill": {0: _VectorEnvironment()},
                "isaaclab": {0: _VectorEnvironment()},
            },
        )


def test_envhub_loader_reports_available_tasks() -> None:
    with pytest.raises(ValueError, match="available task ids: 1, 2"):
        load_lerobot_envhub_environment(
            _spec(task_id=5),
            make_env_fn=lambda *_args, **_kwargs: {
                "maniskill": {1: _VectorEnvironment(), 2: _VectorEnvironment()}
            },
        )


def test_envhub_adapter_rejects_action_shape_mismatch() -> None:
    env = load_lerobot_envhub_environment(
        _spec(),
        make_env_fn=lambda *_args, **_kwargs: {
            "maniskill": {4: _VectorEnvironment()}
        },
    )
    with pytest.raises(ValueError, match="action shape"):
        env.step(np.zeros(3, dtype=np.float32))
