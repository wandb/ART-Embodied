"""Generate model-independent LIBERO evaluation initial states.

The generator samples the simulator placement distribution before loading any
policy.  It rejects exact duplicates of the official initial-state bank,
non-finite states, and states that already satisfy the task after the normal
reset settling sequence.  The resulting manifest can therefore be frozen
before evaluating any candidate checkpoint.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from art_embodied import EmbodiedExperimentConfig
from examples.embodied.libero.environment import LiberoTaskCatalog
from examples.embodied.libero.settings import LiberoSettings
from examples.embodied.libero.state_manifest import (
    file_sha256,
    state_sha256,
    write_state_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--states-per-task", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=20260718)
    parser.add_argument("--max-attempts-per-task", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def generate(
    config_path: Path,
    output_dir: Path,
    *,
    states_per_task: int,
    base_seed: int,
    max_attempts_per_task: int,
    overwrite: bool,
) -> Path:
    if states_per_task < 1:
        raise ValueError("states_per_task must be positive")
    if max_attempts_per_task < states_per_task:
        raise ValueError("max_attempts_per_task must cover states_per_task")
    destination = output_dir.expanduser().resolve()
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing state manifest directory: {destination}"
            )
        shutil.rmtree(destination)

    config = EmbodiedExperimentConfig.from_yaml(config_path)
    settings = LiberoSettings.from_config(config)
    if settings.evaluation_state_manifest is not None:
        raise ValueError(
            "Generation config must not already set evaluation_state_manifest"
        )
    catalog = LiberoTaskCatalog(settings)
    states: dict[str, np.ndarray] = {}
    entries: list[dict[str, Any]] = []
    rejection_counts = {
        "non_finite": 0,
        "official_duplicate": 0,
        "generated_duplicate": 0,
        "initial_success": 0,
        "settle_error": 0,
    }

    for task_id in settings.task_ids:
        task_entries = 0
        official = [
            np.asarray(value, dtype=np.float64).reshape(-1)
            for value in catalog.init_states[task_id]
        ]
        official_hashes = {state_sha256(value) for value in official}
        accepted_hashes: set[str] = set()
        env = _make_raw_environment(
            bddl_file=catalog.bddl_files[task_id],
            settings=settings,
        )
        try:
            for attempt in range(max_attempts_per_task):
                if task_entries >= states_per_task:
                    break
                seed = int(base_seed) + int(task_id) * 1_000_000 + attempt
                env.seed(seed)
                env.reset()
                state = np.asarray(env.get_sim_state(), dtype=np.float64).reshape(-1)
                if not np.isfinite(state).all():
                    rejection_counts["non_finite"] += 1
                    continue
                digest = state_sha256(state)
                if digest in official_hashes:
                    rejection_counts["official_duplicate"] += 1
                    continue
                if digest in accepted_hashes:
                    rejection_counts["generated_duplicate"] += 1
                    continue
                try:
                    validation = _validate_reset_state(
                        env,
                        state,
                        settings=settings,
                    )
                except Exception:
                    rejection_counts["settle_error"] += 1
                    continue
                if validation["initial_success"] or validation["success_after_settle"]:
                    rejection_counts["initial_success"] += 1
                    continue

                nearest_l2, nearest_max_abs = _nearest_official_distance(
                    state,
                    official,
                )
                state_key = f"task_{task_id:02d}_state_{task_entries:02d}"
                scenario_id = (
                    f"{settings.suite_name}/generated-held-out/"
                    f"task-{task_id:02d}/state-{task_entries:02d}"
                )
                states[state_key] = np.array(state, copy=True)
                entries.append(
                    {
                        "id": scenario_id,
                        "task_id": int(task_id),
                        "state_key": state_key,
                        "generation_seed": seed,
                        "state_sha256": digest,
                        "state_length": int(state.size),
                        "bddl_sha256": file_sha256(catalog.bddl_files[task_id]),
                        "nearest_official_l2": nearest_l2,
                        "nearest_official_max_abs": nearest_max_abs,
                        "validation": validation,
                    }
                )
                accepted_hashes.add(digest)
                task_entries += 1
        finally:
            env.close()
        if task_entries != states_per_task:
            raise RuntimeError(
                f"Generated only {task_entries}/{states_per_task} states for "
                f"task_id={task_id}; rejections={rejection_counts}"
            )
        print(
            json.dumps(
                {
                    "event": "state_manifest_task_completed",
                    "task_id": int(task_id),
                    "accepted": task_entries,
                    "accepted_total": len(entries),
                    "rejections": dict(rejection_counts),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    return write_state_manifest(
        destination,
        suite_name=settings.suite_name,
        simulator_compatibility=settings.simulator_compatibility,
        states=states,
        entries=entries,
        generator={
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "base_seed": int(base_seed),
            "states_per_task": int(states_per_task),
            "max_attempts_per_task": int(max_attempts_per_task),
            "policy_loaded": False,
            "selection_uses_model_outcomes": False,
            "official_state_count_per_task": 50,
            "wait_steps_after_reset": settings.wait_steps_after_reset,
            "reset_gripper_open": settings.reset_gripper_open,
            "rejection_counts": rejection_counts,
            "simulator_runtime": catalog.runtime_info,
        },
    )


def _make_raw_environment(*, bddl_file: str, settings: LiberoSettings) -> Any:
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=settings.observation_height,
        camera_widths=settings.observation_width,
    )
    if settings.control_mode not in {"unchanged", "default"}:
        use_delta = settings.control_mode == "relative"
        for robot in env.robots:
            robot.controller.use_delta = use_delta
    return env


def _validate_reset_state(
    env: Any,
    state: np.ndarray,
    *,
    settings: LiberoSettings,
) -> dict[str, Any]:
    observation = env.set_init_state(state)
    initial_success = bool(env.check_success())
    dummy_action = np.zeros(7, dtype=np.float64)
    if settings.reset_gripper_open:
        dummy_action[-1] = -1.0
    for _ in range(settings.wait_steps_after_reset):
        observation, _reward, _done, _info = env.step(dummy_action)
    settled_state = np.asarray(env.get_sim_state(), dtype=np.float64).reshape(-1)
    observations_finite = all(
        np.isfinite(np.asarray(value)).all() for value in observation.values()
    )
    if not np.isfinite(settled_state).all() or not observations_finite:
        raise ValueError("Generated state became non-finite during reset settling")
    return {
        "initial_success": initial_success,
        "success_after_settle": bool(env.check_success()),
        "settled_state_finite": True,
        "observations_finite": True,
        "settle_state_l2_delta": float(np.linalg.norm(settled_state - state)),
        "settle_state_max_abs_delta": float(np.max(np.abs(settled_state - state))),
    }


def _nearest_official_distance(
    state: np.ndarray,
    official_states: list[np.ndarray],
) -> tuple[float, float]:
    comparable = [value for value in official_states if value.shape == state.shape]
    if not comparable:
        raise ValueError("Generated and official LIBERO states have different shapes")
    deltas = [np.abs(state - value) for value in comparable]
    return (
        float(min(np.linalg.norm(delta) for delta in deltas)),
        float(min(np.max(delta) for delta in deltas)),
    )


if __name__ == "__main__":
    args = parse_args()
    path = generate(
        args.config,
        args.output_dir,
        states_per_task=args.states_per_task,
        base_seed=args.base_seed,
        max_attempts_per_task=args.max_attempts_per_task,
        overwrite=args.overwrite,
    )
    print(path)
