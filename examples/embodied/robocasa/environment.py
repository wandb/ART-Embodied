"""Gymnasium action-chunk adapter for official RoboCasa GR-1 tasks."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np

from .settings import RoboCasaSettings, RoboCasaTask

DATASET_STATE_ACTION_LAYOUT = (
    ("left_arm", 7),
    ("left_hand", 6),
    ("left_leg", 6),
    ("neck", 3),
    ("right_arm", 7),
    ("right_hand", 6),
    ("right_leg", 6),
    ("waist", 3),
)

# The dataset pads fixed lower-body components into a generic 44D GR-1 schema.
# N1.7's official RoboCasa processor and live simulator consume only these 29D.
POLICY_STATE_ACTION_LAYOUT = (
    ("left_arm", 7),
    ("right_arm", 7),
    ("left_hand", 6),
    ("right_hand", 6),
    ("waist", 3),
)
POLICY_ACTION_DIM = sum(size for _name, size in POLICY_STATE_ACTION_LAYOUT)


def reset_robocasa_environment(
    environment: Any, *, seed: int
) -> tuple[Mapping[str, Any], Any]:
    """Reset RoboCasa's Gym and simulator RNGs to the same replayable seed."""

    raw_environment = environment.unwrapped.env
    raw_environment.seed = int(seed)
    raw_environment.rng = np.random.default_rng(int(seed))
    return environment.reset(seed=int(seed))


def robocasa_state_hash(environment: Any, observation: Mapping[str, Any]) -> str:
    """Hash simulator state plus every policy-facing state component."""

    digest = hashlib.sha256()
    raw_environment = environment.unwrapped.env
    state = np.asarray(raw_environment.sim.get_state().flatten())
    digest.update(str(state.shape).encode("ascii"))
    digest.update(str(state.dtype).encode("ascii"))
    digest.update(state.tobytes())
    for key in sorted(key for key in observation if key.startswith("state.")):
        value = np.asarray(observation[key])
        digest.update(key.encode("utf-8"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def robocasa_observation_hash(observation: Mapping[str, Any]) -> str:
    """Hash every field presented to the policy at a control boundary."""

    digest = hashlib.sha256()
    for key in sorted(observation):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        value = observation[key]
        if isinstance(value, str):
            digest.update(b"str\0")
            digest.update(value.encode("utf-8"))
        else:
            array = np.asarray(value)
            digest.update(str(array.shape).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(array.tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


class RoboCasaTaskCatalog:
    """Construct only tasks declared by the immutable benchmark manifest."""

    def __init__(self, settings: RoboCasaSettings) -> None:
        self.settings = settings
        self.tasks = {task.id: task for task in settings.tasks}

    def make_environment(
        self, scenario: Any, _context: Any
    ) -> "RoboCasaChunkEnvironment":
        task_id = scenario.payload.get("task_id")
        if not isinstance(task_id, str) or task_id not in self.tasks:
            raise ValueError(f"Scenario declares unknown RoboCasa task: {task_id!r}")
        return RoboCasaChunkEnvironment(self.settings, self.tasks[task_id])


class RoboCasaChunkEnvironment:
    """Execute the first eight rows of one native 29D N1.7 action chunk."""

    def __init__(self, settings: RoboCasaSettings, task: RoboCasaTask) -> None:
        import gymnasium as gym
        import robocasa  # noqa: F401
        import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401

        self.settings = settings
        self.task = task
        self.env = gym.make(task.environment_id, enable_render=True)
        self.latest_observation: Mapping[str, Any] | None = None
        self.env_steps = 0
        self.success = False

    def reset(
        self,
        *,
        seed: int,
        options: Mapping[str, Any] | None,
    ) -> tuple[Mapping[str, Any], dict[str, Any]]:
        if options:
            raise ValueError("RoboCasa reset options are not supported")
        observation, info = reset_robocasa_environment(self.env, seed=seed)
        self.latest_observation = observation
        self.env_steps = 0
        self.success = bool(info.get("success", False))
        return observation, {
            "simulator": "robocasa",
            "task_id": self.task.id,
            "environment_id": self.task.environment_id,
            "environment_seed": int(seed),
            "source_revision": self.settings.manifest.source_revision,
            "success": self.success,
        }

    def step(
        self, action_chunk: Any
    ) -> tuple[Mapping[str, Any], float, bool, bool, dict[str, Any]]:
        actions = np.asarray(action_chunk, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.ndim != 2 or actions.shape[1] != POLICY_ACTION_DIM:
            raise ValueError(
                "RoboCasa expects an "
                f"[action_chunk, {POLICY_ACTION_DIM}] array, got {actions.shape}"
            )
        actions = actions[: self.settings.execution_horizon]
        primitive_rewards: list[float] = []
        primitive_mask: list[bool] = []
        executed_actions: list[list[float]] = []
        terminated = False
        truncated = False
        raw_environment_reward = 0.0
        for action in actions:
            if self.env_steps >= self.settings.max_environment_steps:
                truncated = True
                break
            observation, reward, done, env_truncated, info = self.env.step(
                _action_dict(action)
            )
            self.latest_observation = observation
            self.env_steps += 1
            raw_environment_reward += float(reward)
            self.success = self.success or bool(info.get("success", reward > 0))
            terminated = terminated or bool(done)
            truncated = truncated or bool(env_truncated)
            primitive_rewards.append(1.0 if self.success else 0.0)
            primitive_mask.append(True)
            executed_actions.append(action.tolist())
            if (terminated or truncated) and self.settings.stop_on_done:
                break
        missing = self.settings.execution_horizon - len(primitive_rewards)
        primitive_rewards.extend([0.0] * missing)
        primitive_mask.extend([False] * missing)
        if self.env_steps >= self.settings.max_environment_steps and not truncated:
            terminated = True
        if self.latest_observation is None:
            raise RuntimeError("RoboCasa produced no observation")
        return (
            self.latest_observation,
            float(sum(primitive_rewards)),
            bool(terminated or self.success),
            bool(truncated),
            {
                "success": self.success,
                "task_id": self.task.id,
                "env_steps": self.env_steps,
                "primitive_rewards": primitive_rewards,
                "primitive_loss_mask": primitive_mask,
                "executed_actions": executed_actions,
                "raw_environment_reward": raw_environment_reward,
            },
        )

    def render(self) -> np.ndarray:
        if self.latest_observation is None:
            raise RuntimeError("RoboCasa render requested before reset")
        return np.asarray(
            self.latest_observation["video.ego_view_pad_res256_freq20"],
            dtype=np.uint8,
        )

    def close(self) -> None:
        self.env.close()


def _action_dict(action: np.ndarray) -> dict[str, np.ndarray]:
    if action.shape != (POLICY_ACTION_DIM,):
        raise ValueError(
            "RoboCasa primitive action must have shape "
            f"({POLICY_ACTION_DIM},), got {action.shape}"
        )
    result = {}
    offset = 0
    for name, size in POLICY_STATE_ACTION_LAYOUT:
        result[f"action.{name}"] = action[offset : offset + size].copy()
        offset += size
    if offset != POLICY_ACTION_DIM:
        raise AssertionError(
            "Internal RoboCasa policy action layout has an inconsistent size"
        )
    return result
