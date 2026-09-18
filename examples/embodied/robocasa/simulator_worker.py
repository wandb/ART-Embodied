"""Serve grouped RoboCasa environments from their pinned Python runtime."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import socket
import traceback
from typing import Any

import numpy as np

from art_embodied.inference_transport import read_message_sync, write_message_sync
from art_embodied.utils import write_json_atomic

from .environment import (
    POLICY_ACTION_DIM,
    _action_dict,
    reset_robocasa_environment,
    robocasa_observation_hash,
    robocasa_state_hash,
)
from .oracle_parity import OracleParitySession, run_oracle_chunk_parity
from .settings import ENVIRONMENT_PREFIX, ENVIRONMENT_SUFFIX


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve-spec", type=Path, required=True)
    return parser.parse_args()


class _RoboCasaRuntime:
    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        self.environments: list[Any] = []
        self.task_id: str | None = None
        self.active: set[int] = set()
        self.environment_steps: list[int] = []
        self.successes: list[bool] = []
        self.progress_rewards: list[float] = []
        self.progress_stages: list[str] = []
        self.oracle_parity_session: OracleParitySession | None = None

    def describe(self) -> dict[str, Any]:
        """Return the runtime identity consumed by conformance evidence."""

        return {
            "source_revision": str(self.spec["source_revision"]),
            "action_dim": POLICY_ACTION_DIM,
            "allowed_tasks": list(self.spec["allowed_tasks"]),
            "reward_contract": {
                "mode": str(self.spec["reward_mode"]),
                "grasp_reward": float(self.spec["progress_grasp_reward"]),
                "placement_reward": float(self.spec["progress_placement_reward"]),
                "success_reward": 1.0,
            },
            "versions": {
                package: version(package)
                for package in ("gymnasium", "mujoco", "robocasa", "robosuite")
            },
        }

    def _ensure_environments(self, *, task_id: str, num_envs: int) -> None:
        allowed = set(self.spec["allowed_tasks"])
        if task_id not in allowed:
            raise ValueError(f"RoboCasa task is not allowed by the run: {task_id!r}")
        if self.task_id == task_id and len(self.environments) == num_envs:
            return
        self.close()
        import gymnasium as gym
        import robocasa  # noqa: F401
        import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401

        environment_id = f"{ENVIRONMENT_PREFIX}{task_id}{ENVIRONMENT_SUFFIX}"
        self.environments = [
            gym.make(environment_id, enable_render=True) for _ in range(num_envs)
        ]
        self.task_id = task_id

    def reset(self, request: dict[str, Any]) -> dict[str, Any]:
        num_envs = int(request["num_envs"])
        task_id = str(request["task_id"])
        seed = int(request["seed"])
        self._ensure_environments(task_id=task_id, num_envs=num_envs)
        observations = {}
        state_hashes = []
        observation_hashes = []
        for env_id, environment in enumerate(self.environments):
            observation, _info = _reset_environment(environment, seed=seed)
            observations[env_id] = observation
            state_hashes.append(robocasa_state_hash(environment, observation))
            observation_hashes.append(robocasa_observation_hash(observation))
        self.active = set(range(num_envs))
        self.environment_steps = [0] * num_envs
        self.successes = [False] * num_envs
        self.progress_rewards = [0.0] * num_envs
        self.progress_stages = ["none"] * num_envs
        return {
            "observations": observations,
            "state_hashes": state_hashes,
            "observation_hashes": observation_hashes,
            "action_dim": POLICY_ACTION_DIM,
            "max_episode_steps": int(self.spec["max_environment_steps"]),
            "task_id": task_id,
        }

    def step(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self.environments:
            raise RuntimeError("RoboCasa step called before reset")
        actions = np.asarray(request["actions"], dtype=np.float32)
        expected = (len(self.environments), POLICY_ACTION_DIM)
        if actions.shape != expected:
            raise ValueError(
                f"RoboCasa action batch expected {expected}, got {actions.shape}"
            )
        observations = {}
        results = []
        max_steps = int(self.spec["max_environment_steps"])
        end_of_chunk = bool(request.get("end_of_chunk", True))
        for env_id in sorted(self.active):
            environment = self.environments[env_id]
            observation, reward, terminated, truncated, info = environment.step(
                _action_dict(actions[env_id])
            )
            observations[env_id] = observation
            self.environment_steps[env_id] += 1
            success = bool(info.get("success", reward > 0))
            self.successes[env_id] = self.successes[env_id] or success
            if str(self.spec["reward_mode"]) == "task_progress":
                progress = _task_progress(
                    environment,
                    task_id=str(self.task_id),
                    success=self.successes[env_id],
                    grasp_reward=float(self.spec["progress_grasp_reward"]),
                    placement_reward=float(self.spec["progress_placement_reward"]),
                )
            else:
                progress = _progress_payload(
                    grasped=False,
                    placed=False,
                    success=self.successes[env_id],
                    grasp_reward=float(self.spec["progress_grasp_reward"]),
                    placement_reward=float(self.spec["progress_placement_reward"]),
                )
            previous_progress = self.progress_rewards[env_id]
            if progress["reward"] > previous_progress:
                self.progress_rewards[env_id] = float(progress["reward"])
                self.progress_stages[env_id] = str(progress["stage"])
            progress_delta = self.progress_rewards[env_id] - previous_progress
            exhausted = self.environment_steps[env_id] >= max_steps
            done = (
                (self.successes[env_id] and end_of_chunk)
                or bool(terminated)
                or bool(truncated)
                or exhausted
            )
            if done:
                self.active.discard(env_id)
            results.append(
                {
                    "env_id": env_id,
                    "success": self.successes[env_id],
                    "terminated": bool(terminated or (exhausted and not truncated)),
                    "truncated": bool(truncated),
                    "reward": float(reward),
                    "progress_reward": self.progress_rewards[env_id],
                    "progress_reward_delta": progress_delta,
                    "progress_stage": self.progress_stages[env_id],
                    "progress_predicates": progress["predicates"],
                    "environment_steps": self.environment_steps[env_id],
                }
            )
        response = {
            "observations": observations,
            "active_env_ids": sorted(self.active),
            "results": results,
        }
        include_branch_hashes = bool(request.get("include_branch_hashes", False))
        prefix_completed = bool(request.get("shared_prefix_step", False)) and not (
            self.active
        )
        if include_branch_hashes or prefix_completed:
            if len(observations) != len(self.environments):
                raise RuntimeError(
                    "RoboCasa branch hashing requires every group replica to step"
                )
            response["state_hashes"] = [
                robocasa_state_hash(environment, observations[env_id])
                for env_id, environment in enumerate(self.environments)
            ]
            response["observation_hashes"] = [
                robocasa_observation_hash(observations[env_id])
                for env_id in range(len(self.environments))
            ]
        return response

    def close(self) -> None:
        if self.oracle_parity_session is not None:
            self.oracle_parity_session.close()
            self.oracle_parity_session = None
        for environment in self.environments:
            environment.close()
        self.environments = []
        self.task_id = None
        self.active = set()


def _task_progress(
    environment: Any,
    *,
    task_id: str,
    success: bool,
    grasp_reward: float,
    placement_reward: float,
) -> dict[str, Any]:
    """Score named task predicates without changing RoboCasa success semantics."""

    if str(environment.unwrapped.env.__class__.__name__) != task_id:
        raise RuntimeError(
            "RoboCasa progress task identity mismatch: "
            f"requested={task_id!r}, runtime="
            f"{environment.unwrapped.env.__class__.__name__!r}"
        )
    raw_environment = environment.unwrapped.env
    signals = raw_environment.get_subtask_term_signals()
    grasped = bool(signals.get("grasp_object", False))

    if task_id == "PnPCupToDrawerClose":
        from robocasa.utils.object_utils import obj_inside_of

        placed = bool(
            obj_inside_of(
                env=raw_environment,
                obj_name=raw_environment.objects["obj"].name,
                fixture_id=raw_environment.drawer,
                partial_check=True,
            )
        )
    elif task_id in {
        "PnPMilkToMicrowaveClose",
        "PnPPotatoToMicrowaveClose",
    }:
        from robocasa.utils.object_utils import obj_inside_of

        placed = bool(
            obj_inside_of(
                env=raw_environment,
                obj_name=raw_environment.objects["obj"].name,
                fixture_id=raw_environment.microwave,
                partial_check=True,
            )
        )
    else:
        from robocasa.utils import object_utils as OU

        highest_spawn_region = None
        if raw_environment.target_container in {"tiered_basket", "tiered_shelf"}:
            highest_spawn_region = OU.get_highest_spawn_region(
                raw_environment, raw_environment.objects["container"]
            )
        placed = bool(
            OU.check_obj_in_receptacle(
                raw_environment,
                "obj",
                "container",
                spawn_regions=[highest_spawn_region],
            )
        )

    return _progress_payload(
        grasped=grasped,
        placed=placed,
        success=success,
        grasp_reward=grasp_reward,
        placement_reward=placement_reward,
    )


def _progress_payload(
    *,
    grasped: bool,
    placed: bool,
    success: bool,
    grasp_reward: float,
    placement_reward: float,
) -> dict[str, Any]:
    if success:
        stage, reward = "success", 1.0
    elif placed:
        stage, reward = "placed", float(placement_reward)
    elif grasped:
        stage, reward = "grasped", float(grasp_reward)
    else:
        stage, reward = "none", 0.0
    return {
        "stage": stage,
        "reward": reward,
        "predicates": {
            "grasped_target_object": bool(grasped),
            "placed_in_target": bool(placed),
            "official_success": bool(success),
        },
    }


def _reset_environment(environment: Any, *, seed: int) -> tuple[dict[str, Any], Any]:
    """Compatibility wrapper around the shared training/evaluation reset."""

    observation, info = reset_robocasa_environment(environment, seed=seed)
    return dict(observation), info


def _serve(spec: dict[str, Any]) -> None:
    ready_path = Path(spec["ready_path"])
    socket_path = Path(spec["socket_path"])
    socket_path.unlink(missing_ok=True)
    runtime: _RoboCasaRuntime | None = None
    server: socket.socket | None = None
    try:
        import mujoco
        import robosuite

        if mujoco.__version__ != "3.2.6" or robosuite.__version__ != "1.5.1":
            raise RuntimeError(
                "RoboCasa runtime version mismatch: "
                f"mujoco={mujoco.__version__}, robosuite={robosuite.__version__}"
            )
        runtime = _RoboCasaRuntime(spec)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(socket_path))
        socket_path.chmod(0o600)
        server.listen(1)
    except Exception as exc:
        write_json_atomic(
            ready_path,
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            indent=2,
            sort_keys=True,
        )
        return
    write_json_atomic(ready_path, {"ok": True, "socket_path": str(socket_path)})
    try:
        shutdown = False
        while not shutdown:
            connection, _ = server.accept()
            with connection:
                while not shutdown:
                    try:
                        request = read_message_sync(connection)
                    except EOFError:
                        break
                    try:
                        operation = request.get("op")
                        if operation == "describe":
                            value = runtime.describe()
                        elif operation == "reset":
                            value = runtime.reset(request)
                        elif operation == "step":
                            value = runtime.step(request)
                        elif operation == "oracle_chunk_parity":
                            value = run_oracle_chunk_parity(request)
                        elif operation == "oracle_episode_reset":
                            if runtime.oracle_parity_session is not None:
                                runtime.oracle_parity_session.close()
                            runtime.oracle_parity_session = OracleParitySession(request)
                            value = runtime.oracle_parity_session.reset_response()
                        elif operation == "oracle_episode_step":
                            if runtime.oracle_parity_session is None:
                                raise RuntimeError(
                                    "Oracle parity step called before episode reset"
                                )
                            value = runtime.oracle_parity_session.step(
                                request["actions"]
                            )
                        elif operation == "shutdown":
                            value = {"closed": True}
                            shutdown = True
                        else:
                            raise ValueError(
                                f"Unsupported RoboCasa operation: {operation!r}"
                            )
                        response = {"ok": True, "value": value}
                    except Exception as exc:
                        response = {
                            "ok": False,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    write_message_sync(connection, response)
    finally:
        assert server is not None and runtime is not None
        server.close()
        runtime.close()
        socket_path.unlink(missing_ok=True)


def main() -> None:
    spec = json.loads(_parse_args().serve_spec.read_text(encoding="utf-8"))
    _serve(spec)


if __name__ == "__main__":
    main()
