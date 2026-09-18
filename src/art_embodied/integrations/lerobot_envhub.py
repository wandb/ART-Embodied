"""LeRobot EnvHub loading for ART's single-episode rollout contract.

LeRobot owns remote environment discovery and vector-environment creation.
ART-Embodied selects one suite/task and exposes it as an ordinary Gymnasium
environment so existing trajectory grouping remains the only parallelism layer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MakeEnv = Callable[..., Mapping[str, Mapping[int, Any]]]


@dataclass(frozen=True, slots=True)
class LeRobotEnvHubSpec:
    """Pinned selection of one environment from a LeRobot EnvHub package."""

    hub_path: str
    suite: str | None = None
    task_id: int = 0
    trust_remote_code: bool = False
    cache_dir: str | Path | None = None
    require_revision: bool = True

    def __post_init__(self) -> None:
        if not self.hub_path.strip():
            raise ValueError("hub_path must be non-empty")
        if self.task_id < 0:
            raise ValueError("task_id cannot be negative")
        if self.suite is not None and not self.suite.strip():
            raise ValueError("suite must be non-empty when provided")


def load_lerobot_envhub_environment(
    spec: LeRobotEnvHubSpec,
    *,
    make_env_fn: MakeEnv | None = None,
) -> "SingleLeRobotVectorEnv":
    """Load and select one revision-pinned EnvHub task.

    EnvHub executes Python supplied by the environment repository. Callers must
    opt in explicitly, and claim-facing runs require an ``@revision`` in the
    Hub reference rather than silently following a moving branch.
    """

    if not spec.trust_remote_code:
        raise ValueError(
            "EnvHub executes remote environment code; set trust_remote_code=True "
            "only for a reviewed repository"
        )
    if spec.require_revision and not _hub_path_has_revision(spec.hub_path):
        raise ValueError(
            "EnvHub hub_path must pin a revision with '@revision' for "
            "reproducible ART-Embodied rollouts"
        )

    make_env = make_env_fn or _load_lerobot_make_env()
    environments = make_env(
        spec.hub_path,
        n_envs=1,
        use_async_envs=False,
        hub_cache_dir=(str(spec.cache_dir) if spec.cache_dir is not None else None),
        trust_remote_code=True,
    )
    suite_name, tasks = _select_suite(environments, spec.suite)
    try:
        vector_env = tasks[spec.task_id]
    except KeyError as exc:
        available = ", ".join(str(task_id) for task_id in sorted(tasks)) or "<none>"
        raise ValueError(
            f"EnvHub suite {suite_name!r} has no task_id={spec.task_id}; "
            f"available task ids: {available}"
        ) from exc
    return SingleLeRobotVectorEnv(
        vector_env,
        hub_path=spec.hub_path,
        suite=suite_name,
        task_id=spec.task_id,
    )


class SingleLeRobotVectorEnv:
    """Expose a one-element LeRobot vector env as a normal Gymnasium env."""

    def __init__(
        self,
        vector_env: Any,
        *,
        hub_path: str,
        suite: str,
        task_id: int,
    ) -> None:
        num_envs = getattr(vector_env, "num_envs", 1)
        if int(num_envs) != 1:
            raise ValueError(
                "ART-Embodied EnvHub rollouts require exactly one environment; "
                f"received num_envs={num_envs}"
            )
        self.vector_env = vector_env
        self.hub_path = hub_path
        self.suite = suite
        self.task_id = int(task_id)
        self.observation_space = getattr(
            vector_env,
            "single_observation_space",
            getattr(vector_env, "observation_space", None),
        )
        self.action_space = getattr(
            vector_env,
            "single_action_space",
            getattr(vector_env, "action_space", None),
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        observation, info = self.vector_env.reset(seed=seed, options=options)
        return _unbatch(observation), _unbatch_info(info)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        result = self.vector_env.step(_batch_action(action, self.action_space))
        if not isinstance(result, tuple) or len(result) != 5:
            raise TypeError("EnvHub vector environment step() must return five values")
        observation, reward, terminated, truncated, info = result
        return (
            _unbatch(observation),
            float(_single_scalar(reward, name="reward")),
            bool(_single_scalar(terminated, name="terminated")),
            bool(_single_scalar(truncated, name="truncated")),
            _unbatch_info(info),
        )

    def render(self) -> Any:
        render = getattr(self.vector_env, "render", None)
        if callable(render):
            frame = render()
        else:
            call = getattr(self.vector_env, "call", None)
            if not callable(call):
                raise AttributeError("EnvHub vector environment has no render method")
            frame = call("render")
        return _unbatch(frame)

    def close(self) -> None:
        close = getattr(self.vector_env, "close", None)
        if callable(close):
            close()


def _load_lerobot_make_env() -> MakeEnv:
    try:
        from lerobot.envs.factory import make_env
    except ImportError as exc:  # pragma: no cover - optional runtime dependency.
        raise RuntimeError(
            "LeRobot EnvHub support requires a LeRobot release with "
            "lerobot.envs.factory.make_env"
        ) from exc
    return make_env


def _hub_path_has_revision(hub_path: str) -> bool:
    repository = hub_path.split(":", 1)[0]
    return "@" in repository and bool(repository.rsplit("@", 1)[1])


def _select_suite(
    environments: Mapping[str, Mapping[int, Any]],
    requested_suite: str | None,
) -> tuple[str, Mapping[int, Any]]:
    if not isinstance(environments, Mapping) or not environments:
        raise TypeError("LeRobot make_env() returned no EnvHub suites")
    if requested_suite is None:
        if len(environments) != 1:
            available = ", ".join(sorted(str(name) for name in environments))
            raise ValueError(
                "EnvHub package exposes multiple suites; select one explicitly. "
                f"Available suites: {available}"
            )
        suite_name = next(iter(environments))
    else:
        suite_name = requested_suite
    try:
        tasks = environments[suite_name]
    except KeyError as exc:
        available = ", ".join(sorted(str(name) for name in environments))
        raise ValueError(
            f"EnvHub package has no suite {suite_name!r}; available suites: "
            f"{available}"
        ) from exc
    if not isinstance(tasks, Mapping):
        raise TypeError(f"EnvHub suite {suite_name!r} must map task ids to envs")
    return str(suite_name), tasks


def _batch_action(action: Any, action_space: Any) -> Any:
    shape = tuple(getattr(action, "shape", ()))
    expected_shape = tuple(getattr(action_space, "shape", ()))
    if shape and shape == (1, *expected_shape):
        return action
    if shape and shape == expected_shape and callable(getattr(action, "unsqueeze", None)):
        return action.unsqueeze(0)

    array = np.asarray(action)
    if expected_shape and array.shape == (1, *expected_shape):
        return array
    if expected_shape and array.shape != expected_shape:
        raise ValueError(
            f"EnvHub action shape {array.shape} does not match {expected_shape}"
        )
    return np.expand_dims(array, axis=0)


def _unbatch(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _unbatch(item) for key, item in value.items()}
    if isinstance(value, (str, bytes)):
        return value
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) == 0 or int(shape[0]) != 1:
            raise ValueError(
                "EnvHub single-environment value must have a leading batch "
                f"dimension of 1; received shape={tuple(shape)}"
            )
        return value[0]
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(
                "EnvHub single-environment sequence must contain exactly one item"
            )
        return value[0]
    return value


def _unbatch_info(info: Any) -> dict[str, Any]:
    unbatched = _unbatch(info)
    if not isinstance(unbatched, dict):
        raise TypeError("EnvHub reset/step info must be a mapping")
    return unbatched


def _single_scalar(value: Any, *, name: str) -> Any:
    unbatched = _unbatch(value)
    shape = getattr(unbatched, "shape", None)
    if shape is not None and len(shape) != 0:
        raise ValueError(f"EnvHub {name} must contain one scalar")
    item = getattr(unbatched, "item", None)
    return item() if callable(item) else unbatched
