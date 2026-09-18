"""Compare native and RLinf OpenVLA-OFT loaders on one real LIBERO reset."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

from art_embodied import EmbodiedExperimentConfig, make_policy
from art_embodied.experiment import RolloutContext
from examples.embodied.libero.components import (
    LiberoSettings,
    LiberoTaskCatalog,
    build_evaluation_scenarios,
    record_libero_observation,
)
from examples.embodied.libero.environment_conformance import (
    _action_summary,
    _delta,
    _rlinf_env,
    _seed_all,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scenario-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _with_loader(
    config: EmbodiedExperimentConfig,
    loader: str,
) -> EmbodiedExperimentConfig:
    load_kwargs = {**config.policy.load_kwargs, "model_loader": loader}
    policy = config.policy.model_copy(update={"load_kwargs": load_kwargs})
    return config.model_copy(update={"policy": policy})


def _predict(
    policy: Any,
    *,
    observation: Any,
    task: str,
    seed: int,
    do_sample: bool,
    temperature: float,
    rlinf_observation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy.set_generation(do_sample=do_sample, temperature=temperature)
    _seed_all(seed)
    context: dict[str, Any] = {
        "scenario": {"task": task},
        "task": task,
        "step": 0,
    }
    if rlinf_observation is not None:
        context["rlinf_env_obs"] = rlinf_observation
    return _action_summary(policy.act(observation, context))


def _release_cuda_cache() -> None:
    gc.collect()
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
        config_fingerprint="loader-conformance",
    )
    catalog = LiberoTaskCatalog(LiberoSettings.from_config(config))
    art_env = catalog.make_environment(scenario, context)
    rlinf_env, reset_id = _rlinf_env(config, task_id=task_id, trial_id=trial_id)
    try:
        art_observation, _ = art_env.reset(
            seed=0,
            options=scenario.payload["reset_options"],
        )
        rlinf_observation, _ = rlinf_env.reset()
    finally:
        art_env.close()
        rlinf_env.close()
    recorded = record_libero_observation(observation=art_observation, step=0)

    modes = {
        "greedy": {"do_sample": False, "temperature": 1.0},
        "sampled": {
            "do_sample": True,
            "temperature": config.policy.evaluation_generation.temperature,
        },
    }
    rlinf_policy = make_policy(_with_loader(config, "rlinf"))
    rlinf_predictions: dict[str, dict[str, Any]] = {}
    for name, mode in modes.items():
        converted = _predict(
            rlinf_policy,
            observation=recorded,
            task=scenario.task,
            seed=args.seed,
            **mode,
        )
        native_observation = _predict(
            rlinf_policy,
            observation=recorded,
            task=scenario.task,
            seed=args.seed,
            rlinf_observation=rlinf_observation,
            **mode,
        )
        rlinf_predictions[name] = {
            "converted": converted,
            "native_observation": native_observation,
            "conversion_tokens_equal": converted["tokens"]
            == native_observation["tokens"],
            "conversion_decoded": _delta(
                converted["decoded"], native_observation["decoded"]
            ),
        }
    del rlinf_policy
    _release_cuda_cache()

    native_policy = make_policy(_with_loader(config, "native"))
    native_predictions = {
        name: _predict(
            native_policy,
            observation=recorded,
            task=scenario.task,
            seed=args.seed,
            **mode,
        )
        for name, mode in modes.items()
    }
    del native_policy
    _release_cuda_cache()

    comparisons: dict[str, Any] = {}
    for name in modes:
        oracle = rlinf_predictions[name]["native_observation"]
        native = native_predictions[name]
        comparisons[name] = {
            "tokens_equal": oracle["tokens"] == native["tokens"],
            "decoded": _delta(oracle["decoded"], native["decoded"]),
            "rlinf": oracle,
            "native": native,
        }
    report = {
        "schema_version": 1,
        "config": str(args.config.resolve()),
        "scenario": scenario.model_dump(mode="json"),
        "rlinf_reset_id": int(reset_id),
        "seed": int(args.seed),
        "rlinf_observation_conversion": rlinf_predictions,
        "loader_comparisons": comparisons,
    }
    report["ok"] = all(
        item["tokens_equal"] and item["decoded"].get("exact_equal", False)
        for item in comparisons.values()
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
