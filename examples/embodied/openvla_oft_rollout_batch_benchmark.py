"""Benchmark sequential and native-batched OpenVLA-OFT rollout inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import yaml

from art_embodied import EmbodiedExperimentConfig, Observation, make_policy


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load_config(path: Path, adapter_path: Path) -> EmbodiedExperimentConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    policy = dict(raw["policy"])
    load_kwargs = dict(policy["load_kwargs"])
    load_kwargs["peft_adapter_path"] = str(adapter_path.expanduser().resolve())
    policy["load_kwargs"] = load_kwargs
    policy["device"] = "cuda:0"
    raw["policy"] = policy
    return EmbodiedExperimentConfig.model_validate(raw)


def _observations(count: int) -> list[Observation]:
    rows = []
    for index in range(count):
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        image[..., 0] = 40 + index
        image[48:208, 80:176, 1] = 192
        rows.append(Observation(step=0, kind="image", value=image))
    return rows


def _contexts(count: int) -> list[dict]:
    return [
        {
            "step": 0,
            "policy_seed": 10_000 + index,
            "scenario": {"task": "pick up the red object and place it in the basket"},
        }
        for index in range(count)
    ]


def _synchronize() -> None:
    import torch

    torch.cuda.synchronize()


def _validate_actions(actions, expected: int) -> None:
    if len(actions) != expected:
        raise RuntimeError(f"Expected {expected} actions, received {len(actions)}")
    for action in actions:
        tokens = action.raw.get("tokens", [])
        logprobs = (action.logprobs or {}).get("token_logprobs", [])
        if not tokens or len(tokens) != len(logprobs):
            raise RuntimeError("Batched action token/logprob payload is incomplete")


def _measure(policy, *, batch_size: int, repeats: int, batched: bool) -> dict:
    observations = _observations(batch_size)
    contexts = _contexts(batch_size)
    _synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        if batched:
            actions = policy.act_batch(observations, contexts)
        else:
            actions = [
                policy.act(observation, context)
                for observation, context in zip(
                    observations,
                    contexts,
                    strict=True,
                )
            ]
        _validate_actions(actions, batch_size)
    _synchronize()
    elapsed = time.perf_counter() - started
    action_count = batch_size * repeats
    return {
        "mode": "batched" if batched else "sequential",
        "batch_size": batch_size,
        "repeats": repeats,
        "actions": action_count,
        "elapsed_seconds": elapsed,
        "actions_per_second": action_count / elapsed,
        "native_batch_fraction": sum(
            bool(action.metadata.get("rlinf_native_batched_policy_call"))
            for action in actions
        )
        / len(actions),
    }


def main() -> None:
    args = _arguments()
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    batch_sizes = [int(item) for item in args.batch_sizes.split(",")]
    if not batch_sizes or any(item <= 0 for item in batch_sizes):
        raise ValueError("batch sizes must be positive integers")

    import torch

    config = _load_config(args.config, args.adapter_path)
    policy = make_policy(config)
    for _ in range(args.warmup):
        policy.act_batch(_observations(max(batch_sizes)), _contexts(max(batch_sizes)))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    measurements = []
    for batch_size in batch_sizes:
        sequential = _measure(
            policy,
            batch_size=batch_size,
            repeats=args.repeats,
            batched=False,
        )
        batched = _measure(
            policy,
            batch_size=batch_size,
            repeats=args.repeats,
            batched=True,
        )
        batched["speedup_vs_sequential"] = (
            batched["actions_per_second"] / sequential["actions_per_second"]
        )
        measurements.extend([sequential, batched])

    report = {
        "config_fingerprint": config.fingerprint,
        "adapter_path": str(args.adapter_path.expanduser().resolve()),
        "device": torch.cuda.get_device_name(0),
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
        "measurements": measurements,
    }
    output = json.dumps(report, indent=2, sort_keys=True)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
