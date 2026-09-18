"""Verify that an OpenVLA-OFT rollout checkpoint preserves native actions.

This diagnostic exercises the same PEFT save/reload boundary used by process
rollout actors.  It intentionally uses one real LIBERO reset instead of a
synthetic image so model, observation, and checkpoint contracts are checked
together.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
import tempfile
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
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _seed_all(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _predict(policy: Any, observation: Any, task: str, *, seed: int) -> dict[str, Any]:
    policy.set_generation(do_sample=False, temperature=1.0)
    _seed_all(seed)
    action = policy.act(
        observation,
        {"scenario": {"task": task}, "task": task, "step": 0},
    )
    return {
        "tokens": [int(value) for value in action.raw.get("tokens", [])],
        "decoded": np.asarray(action.decoded, dtype=np.float64).tolist(),
        "logprobs": action.logprobs,
        "metadata": {
            key: action.metadata.get(key)
            for key in (
                "model_loader",
                "peft_adapter_path",
                "action_logit_extraction",
                "rlinf_native_predict_action_batch",
            )
        },
    }


def _predict_batch(
    policy: Any,
    observation: Any,
    task: str,
    *,
    seed: int,
) -> dict[str, Any]:
    policy.set_generation(do_sample=False, temperature=1.0)
    _seed_all(seed)
    action = policy.act_batch(
        [observation],
        [{"scenario": {"task": task}, "task": task, "step": 0}],
    )[0]
    return {
        "tokens": [int(value) for value in action.raw.get("tokens", [])],
        "decoded": np.asarray(action.decoded, dtype=np.float64).tolist(),
        "logprobs": action.logprobs,
        "metadata": {
            key: action.metadata.get(key)
            for key in (
                "model_loader",
                "peft_adapter_path",
                "action_logit_extraction",
                "rlinf_native_predict_action_batch",
                "rlinf_native_batched_policy_call",
            )
        },
    }


def _adapter_disabled(model: Any):
    disable = getattr(model, "disable_adapter", None)
    return disable() if callable(disable) else nullcontext()


def _snapshot_config(
    config: EmbodiedExperimentConfig,
    snapshot: Path,
) -> EmbodiedExperimentConfig:
    raw = config.model_dump(mode="python")
    policy = dict(raw["policy"])
    load_kwargs = dict(policy["load_kwargs"])
    load_kwargs["peft_adapter_path"] = str(snapshot)
    policy["load_kwargs"] = load_kwargs
    raw["policy"] = policy
    return EmbodiedExperimentConfig.model_validate(raw)


def _numeric_delta(left: Any, right: Any) -> dict[str, float]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        return {"shape_equal": 0.0, "max_abs": float("inf"), "mean_abs": float("inf")}
    delta = np.abs(a - b)
    return {
        "shape_equal": 1.0,
        "max_abs": float(delta.max(initial=0.0)),
        "mean_abs": float(delta.mean()) if delta.size else 0.0,
    }


def _comparison_matches(comparison: dict[str, Any]) -> bool:
    return bool(
        comparison["tokens_equal"] and comparison["decoded"]["max_abs"] <= 1.0e-6
    )


def main() -> None:
    args = parse_args()
    config = EmbodiedExperimentConfig.from_yaml(args.config)
    scenarios = build_evaluation_scenarios(config)
    scenario = scenarios[args.scenario_index]
    context = RolloutContext(
        update=0,
        group_index=args.scenario_index,
        attempt_index=0,
        environment_seed=0,
        policy_seed=args.seed,
        config_fingerprint="checkpoint-roundtrip",
    )
    catalog = LiberoTaskCatalog(LiberoSettings.from_config(config))
    environment = catalog.make_environment(scenario, context)
    try:
        raw_observation, reset_info = environment.reset(
            seed=context.environment_seed,
            options=scenario.payload.get("reset_options"),
        )
        observation = record_libero_observation(observation=raw_observation, step=0)
    finally:
        environment.close()

    policy = make_policy(config)
    before = _predict(policy, observation, scenario.task, seed=args.seed)
    before_batch = _predict_batch(policy, observation, scenario.task, seed=args.seed)
    with _adapter_disabled(policy.model):
        base = _predict(policy, observation, scenario.task, seed=args.seed)

    with tempfile.TemporaryDirectory(prefix="art-openvla-roundtrip-") as directory:
        snapshot = Path(directory) / "snapshot"
        checkpoint_ref = policy.save_checkpoint(str(snapshot))
        after_save = _predict(policy, observation, scenario.task, seed=args.seed)
        reloaded = make_policy(_snapshot_config(config, snapshot))
        after_reload = _predict(reloaded, observation, scenario.task, seed=args.seed)
        comparisons = {
            "base_vs_adapter": {
                "tokens_equal": base["tokens"] == before["tokens"],
                "decoded": _numeric_delta(base["decoded"], before["decoded"]),
            },
            "single_vs_batch_before_save": {
                "tokens_equal": before["tokens"] == before_batch["tokens"],
                "decoded": _numeric_delta(before["decoded"], before_batch["decoded"]),
            },
            "before_vs_after_save": {
                "tokens_equal": before["tokens"] == after_save["tokens"],
                "decoded": _numeric_delta(before["decoded"], after_save["decoded"]),
            },
            "before_vs_after_reload": {
                "tokens_equal": before["tokens"] == after_reload["tokens"],
                "decoded": _numeric_delta(before["decoded"], after_reload["decoded"]),
            },
        }
        required_roundtrip_checks = {
            name: _comparison_matches(comparisons[name])
            for name in (
                "single_vs_batch_before_save",
                "before_vs_after_save",
                "before_vs_after_reload",
            )
        }
        report = {
            "schema_version": 1,
            "config": str(args.config.resolve()),
            "scenario": scenario.model_dump(mode="json"),
            "reset_info": reset_info,
            "seed": args.seed,
            "checkpoint_ref": checkpoint_ref,
            "comparisons": comparisons,
            "checks": {
                "adapter_effect_observed": not _comparison_matches(
                    comparisons["base_vs_adapter"]
                ),
                **required_roundtrip_checks,
                "checkpoint_roundtrip_passed": all(required_roundtrip_checks.values()),
            },
            "predictions": {
                "base_adapter_disabled": base,
                "adapter_before_save": before,
                "adapter_batch_before_save": before_batch,
                "adapter_after_save": after_save,
                "adapter_after_reload": after_reload,
            },
        }

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not report["checks"]["checkpoint_roundtrip_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
