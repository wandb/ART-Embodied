"""Compare ART and RLinf LIBERO reset observations on one real scenario."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import numpy as np

from art_embodied import EmbodiedExperimentConfig, make_policy
from art_embodied.experiment import RolloutContext
from examples.embodied.libero.components import (
    LiberoSettings,
    LiberoTaskCatalog,
    build_evaluation_scenarios,
    record_libero_observation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scenario-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--compare-policy",
        action="store_true",
        help="Also compare converted and native env_obs through an RLinf-loader policy.",
    )
    return parser.parse_args()


def _seed_all(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _rlinf_image(value: Any) -> np.ndarray:
    image = _array(value)
    while image.ndim > 3:
        image = image[0]
    if image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    return image


def _delta(left: Any, right: Any) -> dict[str, Any]:
    a = _array(left)
    b = _array(right)
    result: dict[str, Any] = {
        "left_shape": list(a.shape),
        "right_shape": list(b.shape),
        "shape_equal": a.shape == b.shape,
        "left_dtype": str(a.dtype),
        "right_dtype": str(b.dtype),
    }
    if a.shape != b.shape:
        return result
    difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
    result.update(
        {
            "exact_equal": bool(np.array_equal(a, b)),
            "max_abs": float(difference.max(initial=0.0)),
            "mean_abs": float(difference.mean()) if difference.size else 0.0,
        }
    )
    return result


def _rlinf_env(config: EmbodiedExperimentConfig, *, task_id: int, trial_id: int):
    from omegaconf import OmegaConf
    from rlinf.envs.libero.libero_env import LiberoEnv
    from rlinf.envs.libero.utils import get_benchmark_overridden

    settings = LiberoSettings.from_config(config)
    suite = get_benchmark_overridden(settings.suite_name)()
    reset_id = (
        sum(len(suite.get_task_init_states(index)) for index in range(task_id))
        + trial_id
    )
    cfg = OmegaConf.create(
        {
            "env_type": "libero",
            "task_suite_name": settings.suite_name,
            "total_num_envs": 1,
            "auto_reset": False,
            "ignore_terminations": False,
            "max_steps_per_rollout_epoch": settings.max_episode_steps,
            "max_episode_steps": settings.max_episode_steps,
            "use_rel_reward": True,
            "use_step_penalty": False,
            "reward_coef": settings.reward_coefficient,
            "reset_gripper_open": settings.reset_gripper_open,
            "is_eval": True,
            "seed": 0,
            "group_size": 1,
            "use_fixed_reset_state_ids": True,
            "use_ordered_reset_state_ids": False,
            "specific_reset_id": int(reset_id),
            "init_params": {
                "camera_heights": settings.observation_height,
                "camera_widths": settings.observation_width,
            },
            "video_cfg": {
                "save_video": False,
                "info_on_video": False,
                "video_base_dir": "",
            },
        }
    )
    return LiberoEnv(cfg, num_envs=1, seed_offset=0, total_num_processes=1), reset_id


def _action_summary(action: Any) -> dict[str, Any]:
    return {
        "tokens": [int(value) for value in action.raw.get("tokens", [])],
        "decoded": _array(action.decoded).astype(float).tolist(),
    }


def main() -> None:
    args = parse_args()
    config = EmbodiedExperimentConfig.from_yaml(args.config)
    scenario = build_evaluation_scenarios(config)[args.scenario_index]
    task_id = int(scenario.payload["task_id"])
    trial_id = int(scenario.payload["reset_options"]["trial_id"])
    context = RolloutContext(
        update=0,
        group_index=args.scenario_index,
        attempt_index=0,
        environment_seed=0,
        policy_seed=args.seed,
        config_fingerprint="environment-conformance",
    )

    catalog = LiberoTaskCatalog(LiberoSettings.from_config(config))
    art_env = catalog.make_environment(scenario, context)
    rlinf_env, reset_id = _rlinf_env(config, task_id=task_id, trial_id=trial_id)
    try:
        art_observation, art_reset_info = art_env.reset(
            seed=0,
            options=scenario.payload["reset_options"],
        )
        rlinf_observation, _ = rlinf_env.reset()
    finally:
        art_env.close()
        rlinf_env.close()

    comparisons = {
        "primary_image": _delta(
            art_observation["image"],
            _rlinf_image(rlinf_observation["images"]),
        ),
        "wrist_image": _delta(
            art_observation["wrist_image"],
            _rlinf_image(rlinf_observation["wrist_images"]),
        ),
        "proprio_state": _delta(
            art_observation["proprio_state"],
            _array(rlinf_observation["states"])[0],
        ),
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "config": str(args.config.resolve()),
        "scenario": scenario.model_dump(mode="json"),
        "rlinf_reset_id": reset_id,
        "art_reset_info": art_reset_info,
        "observation_comparisons": comparisons,
    }

    if args.compare_policy:
        load_config = config.policy.load_kwargs
        if load_config.get("model_loader") != "rlinf":
            raise ValueError(
                "--compare-policy requires policy.load_kwargs.model_loader=rlinf"
            )
        policy = make_policy(config)
        policy.set_generation(do_sample=False, temperature=1.0)
        recorded = record_libero_observation(observation=art_observation, step=0)
        base_context = {
            "scenario": {"task": scenario.task},
            "task": scenario.task,
            "step": 0,
        }
        _seed_all(args.seed)
        converted = policy.act(recorded, dict(base_context))
        _seed_all(args.seed)
        native = policy.act(
            recorded,
            {**base_context, "rlinf_env_obs": rlinf_observation},
        )
        converted_summary = _action_summary(converted)
        native_summary = _action_summary(native)
        report["policy_comparison"] = {
            "tokens_equal": converted_summary["tokens"] == native_summary["tokens"],
            "decoded": _delta(converted_summary["decoded"], native_summary["decoded"]),
            "converted_art_observation": converted_summary,
            "native_rlinf_observation": native_summary,
        }

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    observation_ok = all(
        item.get("exact_equal", False) for item in comparisons.values()
    )
    policy_ok = report.get("policy_comparison", {}).get("tokens_equal", True)
    if not observation_ok or not policy_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
