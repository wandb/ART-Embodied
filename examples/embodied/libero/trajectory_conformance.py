"""Compare ART and RLinf LIBERO transitions under one shared policy stream."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from art_embodied import EmbodiedExperimentConfig, make_policy
from art_embodied.experiment import RolloutContext
from examples.embodied.libero.components import (
    LiberoSettings,
    LiberoTaskCatalog,
    build_evaluation_scenarios,
    process_openvla_action_chunk,
    record_libero_observation,
)
from examples.embodied.libero.environment_conformance import (
    _array,
    _delta,
    _rlinf_env,
    _rlinf_image,
    _seed_all,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scenario-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-policy-steps", type=int, default=64)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _observation_comparisons(
    art_observation: dict[str, Any],
    rlinf_observation: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
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


def _comparisons_match(comparisons: dict[str, dict[str, Any]]) -> bool:
    return all(item.get("exact_equal", False) for item in comparisons.values())


def _first_batch(value: Any) -> np.ndarray:
    array = _array(value)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or array.shape[0] != 1:
        raise ValueError(f"Expected one action batch [1,C,D], got {array.shape}")
    return array


def main() -> None:
    args = parse_args()
    if args.max_policy_steps < 1:
        raise ValueError("--max-policy-steps must be positive")
    config = EmbodiedExperimentConfig.from_yaml(args.config)
    if config.policy.load_kwargs.get("model_loader") != "rlinf":
        raise ValueError(
            "trajectory conformance requires policy.load_kwargs.model_loader=rlinf"
        )

    settings = LiberoSettings.from_config(config)
    scenario = build_evaluation_scenarios(config)[args.scenario_index]
    task_id = int(scenario.payload["task_id"])
    trial_id = int(scenario.payload["reset_options"]["trial_id"])
    context = RolloutContext(
        update=0,
        group_index=args.scenario_index,
        attempt_index=0,
        environment_seed=0,
        policy_seed=args.seed,
        config_fingerprint="trajectory-conformance",
    )
    catalog = LiberoTaskCatalog(settings)
    art_env = catalog.make_environment(scenario, context)
    rlinf_env, reset_id = _rlinf_env(config, task_id=task_id, trial_id=trial_id)
    policy = make_policy(config)
    generation = config.policy.evaluation_generation
    policy.set_generation(
        do_sample=generation.do_sample,
        temperature=generation.temperature,
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "config": str(args.config.resolve()),
        "scenario": scenario.model_dump(mode="json"),
        "seed": int(args.seed),
        "rlinf_reset_id": int(reset_id),
        "policy_steps": [],
        "first_mismatch": None,
        "art_success": False,
        "rlinf_success": False,
    }
    try:
        art_observation, art_reset_info = art_env.reset(
            seed=0,
            options=scenario.payload["reset_options"],
        )
        rlinf_observation, _ = rlinf_env.reset()
        report["art_reset_info"] = art_reset_info
        initial = _observation_comparisons(art_observation, rlinf_observation)
        report["initial_observation"] = initial
        if not _comparisons_match(initial):
            report["first_mismatch"] = {
                "kind": "initial_observation",
                "comparisons": initial,
            }
        _seed_all(args.seed)

        for policy_step in range(args.max_policy_steps):
            if report["first_mismatch"] is not None:
                break
            recorded_observation = record_libero_observation(
                observation=art_observation,
                step=policy_step,
            )
            action = policy.act(
                recorded_observation,
                {
                    "scenario": {"task": scenario.task},
                    "task": scenario.task,
                    "step": policy_step,
                    "rlinf_env_obs": rlinf_observation,
                },
            )
            # Preserve the policy's decoded dtype for the RLinf branch. ART's
            # product adapter intentionally casts to float32, while RLinf v0.1
            # forwards the unnormalized NumPy dtype. The conformance report
            # must expose that difference rather than erase it before stepping.
            raw_actions = _first_batch(action.decoded).copy()
            art_actions = process_openvla_action_chunk(
                raw_actions,
                chunk_size=settings.action_chunk_size,
                normalize_gripper=settings.normalize_gripper,
                binarize_gripper=settings.binarize_gripper,
                invert_gripper=settings.invert_gripper,
            )
            from rlinf.envs.action_utils import prepare_actions

            rlinf_actions = _array(
                prepare_actions(
                    raw_chunk_actions=raw_actions.copy(),
                    simulator_type="libero",
                    model_type="openvla_oft",
                    num_action_chunks=settings.action_chunk_size,
                    action_dim=7,
                    policy="libero",
                )
            )
            action_delta = _delta(art_actions, rlinf_actions[0])
            action_processing_close = bool(
                action_delta.get("shape_equal", False)
                and float(action_delta.get("max_abs", float("inf"))) <= 1.0e-6
            )
            step_report: dict[str, Any] = {
                "policy_step": int(policy_step),
                "tokens": [int(value) for value in action.raw.get("tokens", [])],
                "action_processing": action_delta,
                "action_processing_close_at_1e-6": action_processing_close,
                "primitives": [],
            }
            report["policy_steps"].append(step_report)
            if not action_processing_close:
                report["first_mismatch"] = {
                    "kind": "action_processing",
                    "policy_step": int(policy_step),
                    "delta": action_delta,
                }
                break

            for primitive_step in range(settings.action_chunk_size):
                (
                    art_observation,
                    art_reward,
                    art_terminated,
                    art_truncated,
                    art_info,
                ) = art_env.step(art_actions[primitive_step : primitive_step + 1])
                (
                    rlinf_observation,
                    rlinf_reward,
                    rlinf_terminated,
                    rlinf_truncated,
                    _rlinf_info,
                ) = rlinf_env.step(rlinf_actions[:, primitive_step], auto_reset=False)
                observations = _observation_comparisons(
                    art_observation,
                    rlinf_observation,
                )
                art_success = bool(art_info.get("success", False))
                rlinf_success = bool(np.asarray(rlinf_env.success_once)[0])
                transition_report = {
                    "primitive_step": int(primitive_step),
                    "observation": observations,
                    "art_reward": float(art_reward),
                    "rlinf_reward": float(_array(rlinf_reward).reshape(-1)[0]),
                    "art_terminated": bool(art_terminated),
                    "rlinf_terminated": bool(_array(rlinf_terminated).reshape(-1)[0]),
                    "art_truncated": bool(art_truncated),
                    "rlinf_truncated": bool(_array(rlinf_truncated).reshape(-1)[0]),
                    "art_success": art_success,
                    "rlinf_success": rlinf_success,
                }
                step_report["primitives"].append(transition_report)
                report["art_success"] = art_success
                report["rlinf_success"] = rlinf_success
                if not _comparisons_match(observations) or art_success != rlinf_success:
                    report["first_mismatch"] = {
                        "kind": "transition",
                        "policy_step": int(policy_step),
                        "primitive_step": int(primitive_step),
                        "transition": transition_report,
                    }
                    break
                if (
                    art_success
                    or rlinf_success
                    or art_truncated
                    or bool(_array(rlinf_truncated).reshape(-1)[0])
                ):
                    break
            if report["first_mismatch"] is not None or report["art_success"]:
                break
    finally:
        art_env.close()
        rlinf_env.close()

    report["ok"] = report["first_mismatch"] is None
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
