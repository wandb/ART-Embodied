"""Compare one ART RoboCasa action chunk with NVIDIA's pinned evaluator."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import gymnasium as gym
import numpy as np

from .environment import (
    POLICY_ACTION_DIM,
    POLICY_STATE_ACTION_LAYOUT,
    _action_dict,
    reset_robocasa_environment,
    robocasa_observation_hash,
    robocasa_state_hash,
)


class _RecordingEnvironment(gym.Wrapper):
    def __init__(self, environment: gym.Env) -> None:
        super().__init__(environment)
        self.steps: list[dict[str, Any]] = []

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.steps.append(
            _primitive_record(
                self,
                observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
        )
        return observation, reward, terminated, truncated, info


class OracleParitySession:
    """Keep official and ART environments paired for a complete episode."""

    def __init__(self, request: Mapping[str, Any]) -> None:
        self.task_id = str(request["task_id"])
        self.environment_id = str(request["environment_id"])
        self.seed = int(request["seed"])
        self.max_environment_steps = int(request.get("max_environment_steps", 720))
        source = Path(str(request["official_multistep_wrapper"])).resolve()
        expected_sha256 = str(request["official_multistep_wrapper_sha256"])
        actual_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "Pinned NVIDIA MultiStepWrapper source hash changed: "
                f"expected={expected_sha256}, actual={actual_sha256}"
            )
        wrapper_type = _load_official_wrapper(source)

        import robocasa  # noqa: F401
        import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401

        official_raw = gym.make(self.environment_id, enable_render=True)
        self.art_environment = gym.make(self.environment_id, enable_render=True)
        self.recorder = _RecordingEnvironment(official_raw)
        contract = SimpleNamespace(
            n_action_steps=8,
            video_delta_indices_array=np.asarray([0]),
            state_delta_indices_array=np.asarray([0]),
        )
        self.official = wrapper_type(
            self.recorder,
            contract=contract,
            max_episode_steps=self.max_environment_steps,
            terminate_on_success=True,
        )
        official_observation, _ = reset_robocasa_environment(
            self.official, seed=self.seed
        )
        self.art_observation, _ = reset_robocasa_environment(
            self.art_environment, seed=self.seed
        )
        self.reset_record = _compare_boundary(
            self.official,
            _collapse_single_observation_horizon(official_observation),
            self.art_environment,
            self.art_observation,
        )
        self.oracle = {
            "implementation": "NVIDIA MultiStepWrapper loaded from pinned source",
            "source": str(source),
            "sha256": actual_sha256,
            "deterministic_fixture_delta": (
                "The raw RoboCasa internal Generator is explicitly reseeded before "
                "both paths; NVIDIA's outer reset(seed) only seeds numpy's legacy RNG."
            ),
        }
        self.chunk_index = 0
        self.environment_steps = 0
        self.art_rewards: list[float] = []
        self.art_success = False
        self.done = False

    def reset_response(self) -> dict[str, Any]:
        return {
            "status": "passed" if self.reset_record["exact_match"] else "failed",
            "task_id": self.task_id,
            "environment_id": self.environment_id,
            "seed": self.seed,
            "observation": self.art_observation,
            "reset": self.reset_record,
            "oracle": self.oracle,
            "mismatches": (
                []
                if self.reset_record["exact_match"]
                else [{"kind": "reset_boundary", **self.reset_record}]
            ),
        }

    def step(self, actions: Any) -> dict[str, Any]:
        if self.done:
            raise RuntimeError("Oracle parity episode is already complete")
        action_array = np.asarray(actions, dtype=np.float32)
        expected_shape = (8, POLICY_ACTION_DIM)
        if action_array.shape != expected_shape:
            raise ValueError(
                f"Oracle parity expected actions {expected_shape}, got {action_array.shape}"
            )

        self.recorder.steps.clear()
        (
            official_observation,
            official_reward,
            official_done,
            official_truncated,
            official_info,
        ) = self.official.step(_action_chunk_dict(action_array))
        remaining_steps = self.max_environment_steps - self.environment_steps
        art_steps = _run_art_chunk(self.art_environment, action_array[:remaining_steps])
        if not art_steps:
            raise RuntimeError("ART oracle path executed no primitive actions")
        self.environment_steps += len(art_steps)
        self.art_observation = art_steps[-1]["observation"]
        self.art_rewards.extend(float(row["reward"]) for row in art_steps)
        self.art_success = self.art_success or any(
            bool(row["success"]) for row in art_steps
        )
        exhausted = self.environment_steps >= self.max_environment_steps

        final_record = _compare_boundary(
            self.official,
            _collapse_single_observation_horizon(official_observation),
            self.art_environment,
            self.art_observation,
        )
        official_steps = self.recorder.steps
        mismatches = _compare_primitive_records(official_steps, art_steps)
        if not final_record["exact_match"]:
            mismatches.append({"kind": "final_boundary", **final_record})

        official_success = any(bool(value) for value in official_info["success"])
        art_terminated = bool(
            art_steps[-1]["terminated"] or self.art_success or exhausted
        )
        macro = {
            "official_reward": float(official_reward),
            "art_reward": float(max(self.art_rewards)),
            "official_success": official_success,
            "art_success": self.art_success,
            "official_terminated": bool(official_done),
            "art_terminated": art_terminated,
            "official_truncated": bool(official_truncated),
            "art_truncated": bool(art_steps[-1]["truncated"]),
        }
        macro["exact_match"] = (
            macro["official_reward"] == macro["art_reward"]
            and macro["official_success"] == macro["art_success"]
            and macro["official_terminated"] == macro["art_terminated"]
            and macro["official_truncated"] == macro["art_truncated"]
        )
        if not macro["exact_match"]:
            mismatches.append({"kind": "macro_result", **macro})
        self.done = bool(official_done or official_truncated)
        result = {
            "status": "passed" if not mismatches else "failed",
            "chunk_index": self.chunk_index,
            "primitive_steps": len(official_steps),
            "environment_steps": self.environment_steps,
            "observation": self.art_observation,
            "final": final_record,
            "macro": macro,
            "done": self.done,
            "mismatches": mismatches,
        }
        self.chunk_index += 1
        return result

    def close(self) -> None:
        self.official.close()
        self.art_environment.close()


def run_oracle_chunk_parity(request: Mapping[str, Any]) -> dict[str, Any]:
    """Run one action chunk through official and ART control paths."""

    session = OracleParitySession(request)
    try:
        result = session.step(request["actions"])
        mismatches = session.reset_response()["mismatches"] + result["mismatches"]
        return {
            "status": "passed" if not mismatches else "failed",
            "task_id": session.task_id,
            "environment_id": session.environment_id,
            "seed": session.seed,
            "reset": session.reset_record,
            "primitive_steps": result["primitive_steps"],
            "final": result["final"],
            "macro": result["macro"],
            "mismatches": mismatches,
            "oracle": session.oracle,
        }
    finally:
        session.close()


def _load_official_wrapper(source: Path) -> type:
    spec = importlib.util.spec_from_file_location(
        "_art_embodied_pinned_nvidia_multistep_wrapper", source
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load pinned NVIDIA wrapper: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MultiStepWrapper


def _action_chunk_dict(actions: np.ndarray) -> dict[str, np.ndarray]:
    result = {}
    offset = 0
    for name, size in POLICY_STATE_ACTION_LAYOUT:
        result[f"action.{name}"] = actions[:, offset : offset + size].copy()
        offset += size
    return result


def _collapse_single_observation_horizon(
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    result = {}
    for key, value in observation.items():
        if key.startswith(("video.", "state.")):
            array = np.asarray(value)
            if array.shape[0] != 1:
                raise RuntimeError(
                    f"Oracle parity expected one observation frame for {key}, got {array.shape}"
                )
            result[key] = array[0]
        else:
            result[key] = value
    return result


def _run_art_chunk(environment: gym.Env, actions: np.ndarray) -> list[dict[str, Any]]:
    records = []
    for action in actions:
        observation, reward, terminated, truncated, info = environment.step(
            _action_dict(action)
        )
        records.append(
            {
                **_primitive_record(
                    environment,
                    observation,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                ),
                "observation": observation,
            }
        )
        if terminated or truncated:
            break
    return records


def _primitive_record(
    environment: gym.Env,
    observation: Mapping[str, Any],
    *,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "state_sha256": robocasa_state_hash(environment, observation),
        "observation_sha256": robocasa_observation_hash(observation),
        "reward": float(reward),
        "success": bool(info.get("success", reward > 0)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def _compare_boundary(
    official_environment: gym.Env,
    official_observation: Mapping[str, Any],
    art_environment: gym.Env,
    art_observation: Mapping[str, Any],
) -> dict[str, Any]:
    record = {
        "official_state_sha256": robocasa_state_hash(
            official_environment, official_observation
        ),
        "art_state_sha256": robocasa_state_hash(art_environment, art_observation),
        "official_observation_sha256": robocasa_observation_hash(official_observation),
        "art_observation_sha256": robocasa_observation_hash(art_observation),
    }
    record["exact_match"] = (
        record["official_state_sha256"] == record["art_state_sha256"]
        and record["official_observation_sha256"] == record["art_observation_sha256"]
    )
    return record


def _compare_primitive_records(
    official: list[dict[str, Any]], art: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    mismatches = []
    if len(official) != len(art):
        mismatches.append(
            {"kind": "primitive_count", "official": len(official), "art": len(art)}
        )
        return mismatches
    for index, (expected, actual) in enumerate(zip(official, art, strict=True)):
        comparable_actual = {key: actual[key] for key in expected}
        if expected != comparable_actual:
            mismatches.append(
                {
                    "kind": "primitive_result",
                    "primitive_index": index,
                    "official": expected,
                    "art": comparable_actual,
                }
            )
    return mismatches
