"""Batched action chunks for Isaac Lab manager-based environments.

The caller creates the simulator in its own runtime. No Isaac Lab or Torch
imports run until the adapter is constructed. Isaac Lab 3.0's same-step reset
requires ``cfg.compute_final_obs=True`` so terminal states survive resets.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
from typing import Any


def repeat_isaaclab_scene_state(
    state: Mapping[str, Any], *, count: int, source_index: int = 0
) -> dict[str, Any]:
    """Repeat one row of ``scene.get_state(is_relative=True)``.

    Pass the result to ``env.reset_to(..., is_relative=True)``. This copies
    physical state only: commands, randomized materials and task state need
    task-specific restoration and verification as well.
    """
    import torch

    if count <= 0 or source_index < 0:
        raise ValueError("count must be positive and source_index nonnegative")

    def repeat(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: repeat(item) for key, item in value.items()}
        if not isinstance(value, torch.Tensor) or value.ndim == 0:
            raise TypeError("Isaac Lab scene state leaves must be batched tensors")
        if source_index >= value.shape[0]:
            raise ValueError("source_index is outside the scene state batch")
        return value[source_index : source_index + 1].repeat(
            count, *([1] * (value.ndim - 1))
        )

    return repeat(state)


def isaaclab_reset_fingerprints(
    state: Mapping[str, Any], *, num_envs: int
) -> tuple[str, ...]:
    """Hash caller-supplied, environment-relative comparison state per row.

    Include all conditions shared within a GRPO group, not just scene poses.
    Tensor leaves must include the environment dimension. No tolerance is used:
    a failed equality check must be investigated rather than relabeled equal.
    """
    import torch

    digests = [hashlib.sha256() for _ in range(num_envs)]
    leaves = 0

    def visit(value: Any, path: tuple[str, ...]) -> None:
        nonlocal leaves
        if isinstance(value, Mapping):
            for key in sorted(value):
                if not isinstance(key, str):
                    raise TypeError("Reset state keys must be strings")
                visit(value[key], (*path, key))
            return
        if not isinstance(value, torch.Tensor) or value.ndim == 0:
            raise TypeError("Reset state leaves must be batched tensors")
        if value.shape[0] != num_envs:
            raise ValueError("Reset state batch size does not match num_envs")
        if not torch.isfinite(value).all():
            raise ValueError("Reset state contains nonfinite values")
        leaves += 1
        cpu = value.detach().cpu().contiguous()
        for row, digest in enumerate(digests):
            digest.update(repr((path, str(cpu.dtype), tuple(cpu.shape[1:]))).encode())
            digest.update(cpu[row].reshape(-1).view(torch.uint8).numpy().tobytes())

    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    visit(state, ())
    if not leaves:
        raise ValueError("Reset comparison state cannot be empty")
    return tuple(digest.hexdigest() for digest in digests)


class IsaacLabChunkEnvironment:
    """Execute ``[num_envs, horizon, action_dim]`` without crossing episodes.

    Finished rows are masked while the remaining rows continue in one vector
    simulator. Returned observations retain each row's last episode state.
    ``success_fn(env, observation, info)`` must return a boolean tensor using
    task success, not the shaped reward or the union of termination conditions.
    For manager-based tasks it can read a named success termination term.

    Use isolated cloned environments (no cross-environment collisions). The
    adapter does not change action units, quaternion order or camera layout.
    Finished rows receive zero by default. For absolute-position controls,
    supply ``inactive_action_fn(env)`` returning a safe full-batch command,
    such as the current joint positions. These idle commands are excluded
    from the returned episode actions and loss masks.
    """

    def __init__(
        self,
        env: Any,
        *,
        execution_horizon: int,
        max_environment_steps: int,
        success_fn: Callable[[Any, Mapping[str, Any], Mapping[str, Any]], Any],
        inactive_action_fn: Callable[[Any], Any] | None = None,
    ) -> None:
        import torch

        if execution_horizon <= 0 or max_environment_steps <= 0:
            raise ValueError("Horizons must be positive")
        if inactive_action_fn is not None and not callable(inactive_action_fn):
            raise TypeError("inactive_action_fn must be callable")
        self.env = env.unwrapped
        self.num_envs = int(self.env.num_envs)
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if not getattr(self.env.cfg, "compute_final_obs", False):
            raise ValueError("Isaac Lab requires cfg.compute_final_obs=True")
        sensors = getattr(getattr(self.env, "scene", None), "sensors", {})
        for name, sensor in sensors.items():
            camera_cfg = getattr(sensor, "cfg", None)
            renderer = getattr(camera_cfg, "renderer_cfg", None)
            if getattr(
                renderer, "renderer_type", None
            ) == "newton_warp" and not getattr(
                camera_cfg, "update_latest_camera_pose", False
            ):
                raise ValueError(
                    f"Warp camera {name!r} requires update_latest_camera_pose=True "
                    "before environment creation to refresh moving and reset poses"
                )
        shape = tuple(self.env.single_action_space.shape)
        if len(shape) != 1 or shape[0] <= 0:
            raise ValueError("Isaac Lab action space must be a flat vector")
        self.action_dim = shape[0]
        self.device = torch.device(self.env.device)
        self.execution_horizon = execution_horizon
        self.max_environment_steps = max_environment_steps
        self.success_fn = success_fn
        self.inactive_action_fn = inactive_action_fn
        self._observation: dict[str, Any] | None = None
        self._needs_reset = True
        self._closed = False

    def reset(
        self,
        *,
        seed: int,
        reset_fn: Callable[[Any, int], tuple[Mapping[str, Any], Mapping[str, Any]]]
        | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset independent rows, or use an explicit task-aware reset function."""
        import torch

        if self._closed:
            raise RuntimeError("Isaac Lab environment is closed")
        self._needs_reset = True
        observation, info = (
            self.env.reset(seed=seed) if reset_fn is None else reset_fn(self.env, seed)
        )
        self._observation = self._copy_observation(observation)
        self._validate_observation(self._observation)
        self._steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._terminated = torch.zeros_like(self._steps, dtype=torch.bool)
        self._truncated = torch.zeros_like(self._terminated)
        self._success = torch.zeros_like(self._terminated)
        self._needs_reset = False
        return self._copy_observation(self._observation), {
            "simulator": "isaaclab",
            "environment_seed": seed,
            "num_envs": self.num_envs,
        }

    def step(
        self, actions: Any
    ) -> tuple[dict[str, Any], Any, Any, Any, dict[str, Any]]:
        import torch

        if self._closed or self._needs_reset or self._observation is None:
            raise RuntimeError("Reset Isaac Lab before stepping")
        if (self._terminated | self._truncated).all():
            raise RuntimeError("All Isaac Lab episodes finished; reset before stepping")
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        if (
            actions.ndim != 3
            or actions.shape[0] != self.num_envs
            or actions.shape[1] < self.execution_horizon
            or actions.shape[2] != self.action_dim
        ):
            raise ValueError(
                "Expected [num_envs, horizon >= execution_horizon, action_dim] actions"
            )
        if not torch.isfinite(actions).all():
            raise ValueError("Actions contain nonfinite values")
        shape = (self.num_envs, self.execution_horizon)
        mask = torch.zeros(shape, dtype=torch.bool, device=self.device)
        rewards = torch.zeros(shape, device=self.device)
        executed = torch.zeros((*shape, self.action_dim), device=self.device)

        # A failure after physics stepping invalidates the whole batch. Never
        # continue a partial chunk as if it were an unexecuted policy action.
        self._needs_reset = True
        for index in range(self.execution_horizon):
            active = ~(self._terminated | self._truncated)
            if not active.any():
                break
            idle = torch.zeros_like(actions[:, index])
            if self.inactive_action_fn is not None and not active.all():
                idle = torch.as_tensor(
                    self.inactive_action_fn(self.env),
                    device=self.device,
                    dtype=torch.float32,
                )
                if idle.shape != (self.num_envs, self.action_dim):
                    raise ValueError(
                        "inactive_action_fn must return [num_envs, action_dim]"
                    )
                if not torch.isfinite(idle).all():
                    raise ValueError("inactive_action_fn returned nonfinite actions")
            command = torch.where(active[:, None], actions[:, index], idle)
            observation, reward, terminated, truncated, info = self.env.step(command)
            observation = self._copy_observation(observation)
            terminated = self._vector(terminated, "terminated", boolean=True)
            truncated = self._vector(truncated, "truncated", boolean=True)
            reward = self._vector(reward, "reward", boolean=False)
            ended = (terminated | truncated) & active
            if ended.any():
                if "final_obs" not in info:
                    raise RuntimeError(
                        "Isaac Lab omitted final_obs for terminated rows"
                    )
                final = self._copy_observation(info["final_obs"])
                observation = self._select(ended, final, observation)
            success = (
                self._vector(
                    self.success_fn(self.env, observation, info),
                    "success",
                    boolean=True,
                )
                & active
            )
            self._observation = self._select(active, observation, self._observation)
            self._validate_observation(self._observation)
            mask[:, index] = active
            rewards[:, index] = torch.where(active, reward, 0.0)
            executed[:, index] = torch.where(active[:, None], command, 0.0)
            self._steps += active
            self._success |= success
            self._terminated |= (terminated & active) | success
            self._truncated |= (truncated & active) | (
                (self._steps >= self.max_environment_steps) & ~self._terminated
            )
        self._needs_reset = False
        return (
            self._copy_observation(self._observation),
            rewards.sum(dim=1),
            self._terminated.clone(),
            self._truncated.clone(),
            {
                "success": self._success.clone(),
                "env_steps": self._steps.clone(),
                "primitive_rewards": rewards,
                "primitive_loss_mask": mask,
                "executed_actions": executed,
            },
        )

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.env.close()

    def _vector(self, value: Any, name: str, *, boolean: bool) -> Any:
        import torch

        if not isinstance(value, torch.Tensor) or value.shape != (self.num_envs,):
            raise ValueError(f"{name} must be a tensor with shape [num_envs]")
        if value.device != self.device:
            raise ValueError(f"{name} must be on the simulator device")
        if boolean and value.dtype != torch.bool:
            raise TypeError(f"{name} must be boolean, not a score or reward")
        if not boolean and not torch.isfinite(value).all():
            raise ValueError(f"{name} contains nonfinite values")
        return value.clone()

    def _copy_observation(self, observation: Any) -> dict[str, Any]:
        import torch

        if not isinstance(observation, Mapping) or not observation:
            raise TypeError("Isaac Lab observations must be nonempty tensor mappings")
        result = {}
        for key, value in observation.items():
            if isinstance(value, Mapping):
                result[key] = self._copy_observation(value)
            elif isinstance(value, torch.Tensor) and value.ndim > 0:
                if value.shape[0] != self.num_envs or value.device != self.device:
                    raise ValueError(
                        "Observation batch/device does not match simulator"
                    )
                result[key] = value.clone()
            else:
                raise TypeError("Isaac Lab observation leaves must be batched tensors")
        return result

    def _validate_observation(self, observation: Mapping[str, Any]) -> None:
        import torch

        checks = []

        def visit(values: Mapping[str, Any]) -> None:
            for value in values.values():
                if isinstance(value, Mapping):
                    visit(value)
                elif value.is_floating_point() or value.is_complex():
                    checks.append(torch.isfinite(value).all())

        visit(observation)
        # One host synchronization for all floating-point leaves. Validate after
        # terminal selection so unused, auto-reset rows cannot taint a trajectory.
        if checks and not torch.stack(checks).all():
            raise ValueError("Isaac Lab observation contains nonfinite values")

    def _select(self, mask: Any, selected: Mapping, other: Mapping) -> dict[str, Any]:
        import torch

        if selected.keys() != other.keys():
            raise ValueError("Isaac Lab observation keys changed during the episode")
        result = {}
        for key in selected:
            left, right = selected[key], other[key]
            if isinstance(left, Mapping) and isinstance(right, Mapping):
                result[key] = self._select(mask, left, right)
            elif isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
                if left.shape != right.shape or left.dtype != right.dtype:
                    raise ValueError("Isaac Lab observation shape/dtype changed")
                result[key] = torch.where(
                    mask.reshape(self.num_envs, *([1] * (left.ndim - 1))), left, right
                )
            else:
                raise TypeError("Isaac Lab observation structure changed")
        return result
