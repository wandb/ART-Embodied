"""Exercise the pinned RoboCasa process boundary before policy training."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import traceback
from typing import Any

import numpy as np

from art_embodied import EmbodiedExperimentConfig
from art_embodied.utils import write_json_atomic

from .environment import POLICY_ACTION_DIM, POLICY_STATE_ACTION_LAYOUT
from .settings import RoboCasaSettings
from .simulator import RoboCasaSimulatorProcess

_EXPECTED_VERSIONS = {
    "gymnasium": "0.29.1",
    "mujoco": "3.2.6",
    "robosuite": "1.5.1",
}
_EXPECTED_STATE_SIZES = dict(POLICY_STATE_ACTION_LAYOUT)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id")
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args()


def _validate_observation(observation: dict[str, Any]) -> dict[str, Any]:
    image_keys = (
        "video.ego_view_pad_res256_freq20",
        "video.ego_view_bg_crop_pad_res256_freq20",
    )
    image_shapes = {}
    for key in image_keys:
        image = np.asarray(observation[key])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"RoboCasa image {key!r} has invalid shape {image.shape}")
        image_shapes[key] = list(image.shape)
    state_sizes = {}
    for component, expected_size in _EXPECTED_STATE_SIZES.items():
        key = f"state.{component}"
        value = np.asarray(observation[key])
        if value.shape != (expected_size,):
            raise ValueError(
                f"RoboCasa state {key!r} expected {(expected_size,)}, got {value.shape}"
            )
        state_sizes[key] = int(value.size)
    language = observation["annotation.human.coarse_action"]
    if not isinstance(language, str) or not language.strip():
        raise ValueError("RoboCasa coarse-action annotation must be non-empty text")
    return {
        "image_shapes": image_shapes,
        "state_sizes": state_sizes,
        "language": language,
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    config = EmbodiedExperimentConfig.from_yaml(args.config)
    settings = RoboCasaSettings.from_config(config)
    task_id = args.task_id or settings.tasks[0].id
    if task_id not in {task.id for task in settings.tasks}:
        raise ValueError(f"Conformance task is not enabled by YAML: {task_id!r}")
    simulator = RoboCasaSimulatorProcess(
        settings=settings,
        startup_timeout_seconds=config.runtime.rollout_execution.startup_timeout_seconds,
    )
    try:
        identity = await simulator.request({"op": "describe"})
        if {key: identity["versions"][key] for key in _EXPECTED_VERSIONS} != (
            _EXPECTED_VERSIONS
        ):
            raise RuntimeError(
                "RoboCasa runtime version mismatch: "
                f"expected={_EXPECTED_VERSIONS}, actual={identity['versions']}"
            )
        first = await simulator.request(
            {"op": "reset", "task_id": task_id, "num_envs": 2, "seed": args.seed}
        )
        if len(set(first["state_hashes"])) != 1:
            raise RuntimeError("Same-seed group replicas have different initial states")
        if len(set(first["observation_hashes"])) != 1:
            raise RuntimeError(
                "Same-seed group replicas have different initial policy inputs"
            )
        schemas = [
            _validate_observation(first["observations"][index]) for index in range(2)
        ]
        step = await simulator.request(
            {
                "op": "step",
                "actions": np.zeros((2, POLICY_ACTION_DIM), dtype=np.float32),
                "include_branch_hashes": True,
            }
        )
        if len(set(step["state_hashes"])) != 1:
            raise RuntimeError("Same-action replicas have different branch states")
        if len(set(step["observation_hashes"])) != 1:
            raise RuntimeError(
                "Same-action replicas have different branch policy inputs"
            )
        second = await simulator.request(
            {"op": "reset", "task_id": task_id, "num_envs": 2, "seed": args.seed}
        )
        if second["state_hashes"] != first["state_hashes"]:
            raise RuntimeError("Repeated same-seed reset changed the simulator state")
        if second["observation_hashes"] != first["observation_hashes"]:
            raise RuntimeError(
                "Repeated same-seed reset changed the policy-facing observation"
            )
        return {
            "schema_version": 1,
            "status": "passed",
            "config": str(args.config.resolve()),
            "config_fingerprint": config.fingerprint,
            "task_id": task_id,
            "seed": args.seed,
            "runtime": identity,
            "initial_state_sha256": first["state_hashes"][0],
            "initial_observation_sha256": first["observation_hashes"][0],
            "branch_state_sha256": step["state_hashes"][0],
            "branch_observation_sha256": step["observation_hashes"][0],
            "observation_schemas": schemas,
            "step_result_count": len(step["results"]),
            "active_after_zero_action": step["active_env_ids"],
        }
    finally:
        await simulator.close()


def main() -> None:
    args = _parse_args()
    try:
        report = asyncio.run(_run(args))
    except Exception as exc:
        report = {
            "schema_version": 1,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json_atomic(args.output, report, indent=2, sort_keys=True)
        raise
    write_json_atomic(args.output, report, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
