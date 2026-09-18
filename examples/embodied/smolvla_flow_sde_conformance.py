"""Run the real SmolVLA checkpoint through ART's sampler conformance gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import torch

from art_embodied.backends.flow_sde import FlowSDEExample, _flow_sde_microbatches
from art_embodied.config import EmbodiedExperimentConfig
from art_embodied.policies.factory import make_smolvla_flow_policy
from art_embodied.policies.flow_policy import FlowSDERollout


def _processed_probe(policy, *, task: str) -> dict[str, object]:
    from lerobot.policies import prepare_observation_for_inference

    observation = {
        "observation.images.image": np.zeros((256, 256, 3), dtype=np.uint8),
        "observation.images.image2": np.zeros((256, 256, 3), dtype=np.uint8),
        "observation.state": np.zeros(8, dtype=np.float32),
    }
    prepared = prepare_observation_for_inference(
        observation,
        torch.device(policy.device),
        task,
        "panda",
    )
    return policy.preprocessor(prepared)


def run(config_path: Path, *, output: Path | None = None) -> dict[str, object]:
    """Validate native ODE parity, SDE rescore parity, and gradient reachability."""

    config = EmbodiedExperimentConfig.from_yaml(config_path)
    if config.policy.type != "smolvla":
        raise ValueError("SmolVLA conformance requires policy.type='smolvla'")
    policy = make_smolvla_flow_policy(config)
    tasks = (
        "pick up the alphabet soup and place it in the basket",
        "pick up the milk and place it in the basket",
    )
    processed_rows = [_processed_probe(policy, task=task) for task in tasks]
    processed = processed_rows[0]
    model = policy.model
    initial_noise = torch.randn(
        1,
        int(policy.config.chunk_size),
        int(policy.config.max_action_dim),
        device=policy.device,
        dtype=torch.float32,
        generator=torch.Generator(device=policy.device).manual_seed(20260806),
    )

    policy.eval()
    native = policy.predict_native_action_chunk(
        processed,
        initial_noise=initial_noise.clone(),
    )
    bridge_ode = policy.predict_bridge_ode_action_chunk(
        processed,
        initial_noise=initial_noise.clone(),
    )
    ode_delta = (native - bridge_ode).abs()

    rollout_rows = [
        policy.sample_flow_sde(
            processed_row,
            selected_index=torch.tensor([index], device=policy.device),
            initial_noise=torch.randn(
                1,
                int(policy.config.chunk_size),
                int(policy.config.max_action_dim),
                device=policy.device,
                dtype=torch.float32,
                generator=torch.Generator(device=policy.device).manual_seed(
                    20260807 + index
                ),
            ),
            generator=torch.Generator(device=policy.device).manual_seed(
                20260817 + index
            ),
        )
        for index, processed_row in enumerate(processed_rows)
    ]
    examples = [
        FlowSDEExample(
            rollout=rollout,
            group_index=0,
            trajectory_index=index,
            action_index=0,
            reward=float(index),
            loss_mask=True,
            trajectory_primitive_steps=1,
        )
        for index, rollout in enumerate(rollout_rows)
    ]
    microbatches = _flow_sde_microbatches(
        examples,
        [0.0] * len(examples),
        microbatch_size=len(examples),
    )
    replay_pickle_bytes = sum(
        len(pickle.dumps(row.cpu(), protocol=pickle.HIGHEST_PROTOCOL))
        for row in rollout_rows
    )
    policy.train()
    for parameter in policy.parameters():
        parameter.grad = None
    rescore_deltas = []
    for batch, _advantages in microbatches:
        rollout = FlowSDERollout.concatenate([example.rollout for example in batch])
        rescored = policy.flow_sde_logprobs(rollout)
        old = rollout.transition.old_logprobs[
            :, : policy.execution_horizon, : policy.action_dim
        ]
        rescore_deltas.append((rescored.detach() - old).abs())
        (rescored.mean() / len(microbatches)).backward()
    rescore_delta = torch.cat(
        [value.reshape(value.shape[0], -1) for value in rescore_deltas], dim=0
    )
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    finite_gradients = [
        gradient for gradient in gradients if torch.isfinite(gradient).all()
    ]
    nonzero_gradients = [
        gradient for gradient in finite_gradients if gradient.abs().sum() > 0
    ]

    report: dict[str, object] = {
        "schema_version": 1,
        "policy_type": config.policy.type,
        "model_id": config.policy.path,
        "revision": config.policy.revision,
        "device": policy.device,
        "deterministic_sampler": policy.schedule.deterministic_sampler,
        "native_ode_max_abs_delta": float(ode_delta.max().cpu()),
        "native_ode_mean_abs_delta": float(ode_delta.mean().cpu()),
        "rescore_max_abs_delta": float(rescore_delta.max().cpu()),
        "rescore_mean_abs_delta": float(rescore_delta.mean().cpu()),
        "replay_input_type": type(rollout.inputs).__name__,
        "mixed_prompt_examples": len(examples),
        "replay_shape_buckets": len(microbatches),
        "largest_replay_microbatch": max(len(batch) for batch, _ in microbatches),
        "replay_pickle_bytes": replay_pickle_bytes,
        "replay_pickle_bytes_per_policy_step": (replay_pickle_bytes // len(examples)),
        "trainable_parameter_tensors": sum(
            1 for parameter in model.parameters() if parameter.requires_grad
        ),
        "gradient_tensors": len(gradients),
        "finite_gradient_tensors": len(finite_gradients),
        "nonzero_gradient_tensors": len(nonzero_gradients),
        "native_ode_pass": bool(ode_delta.max() <= 1e-5),
        "rescore_pass": bool(rescore_delta.max() <= 1e-5),
        "gradient_pass": bool(nonzero_gradients),
    }
    report["passed"] = bool(
        report["native_ode_pass"] and report["rescore_pass"] and report["gradient_pass"]
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.config, output=args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
