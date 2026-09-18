"""LeRobot EnvHub entry point for this task-specific simulation."""

from __future__ import annotations


def make_env(n_envs: int = 1, use_async_envs: bool = False):
    """Build seeded Gymnasium environments for this Use Case Pack."""

    raise RuntimeError(
        "TODO: compose the verified robot and scene MJCF, implement TaskSpec "
        "predicates, and return a Gymnasium Env or VectorEnv"
    )
