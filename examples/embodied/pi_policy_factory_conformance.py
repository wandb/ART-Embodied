"""GPU conformance gate for the YAML-defined PI0/PI0.5 LoRA policy surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from art_embodied.config import EmbodiedExperimentConfig
from art_embodied.integrations.pi_flow_sde import _prepare_pi_observation
from art_embodied.policies.factory import make_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EmbodiedExperimentConfig.from_yaml(args.config)
    policy = make_policy(config)
    trainable = {
        name: parameter
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    }
    if not trainable or not all("lora_" in name for name in trainable):
        raise AssertionError(
            "PI factory must expose only the YAML-declared LoRA surface"
        )

    prepared = _prepare_pi_observation(
        {
            "image": np.zeros((256, 256, 3), dtype=np.uint8),
            "wrist_image": np.full((256, 256, 3), 127, dtype=np.uint8),
            "proprio_state": np.zeros(8, dtype=np.float32),
        },
        policy=policy,
        task="pick up the red block",
        robot_type=None,
    )
    processed = policy.preprocessor(prepared)
    rollout = policy.sample_flow_sde(
        processed,
        selected_index=torch.tensor(
            [min(2, policy.schedule.num_steps - 1)],
            device=policy.device,
        ),
        generator=torch.Generator(device=policy.device).manual_seed(20260720),
    )
    old = rollout.transition.old_logprobs[
        :, : policy.execution_horizon, : policy.action_dim
    ]
    current = policy.flow_sde_logprobs(rollout)
    delta = (current.detach() - old).abs()
    if delta.max().item() > 2.0e-4:
        raise AssertionError(f"rollout/rescore mismatch: {delta.max().item():.6g}")

    policy.model.zero_grad(set_to_none=True)
    (-current.mean()).backward()
    gradients = [
        parameter.grad.detach().float().norm()
        for parameter in trainable.values()
        if parameter.grad is not None
    ]
    gradient_norm = torch.stack(gradients).norm() if gradients else torch.tensor(0.0)
    if not torch.isfinite(gradient_norm) or gradient_norm.item() == 0.0:
        raise AssertionError("PI LoRA surface produced no finite gradient")

    with tempfile.TemporaryDirectory(prefix="art-embodied-pi-gate-") as directory:
        policy.save_checkpoint(directory)
        checkpoint_bytes = sum(
            path.stat().st_size for path in Path(directory).rglob("*") if path.is_file()
        )
        policy.load_checkpoint({"path": directory})

    result = {
        "status": "ok",
        "config_fingerprint": config.fingerprint,
        "policy_type": config.policy.type,
        "model_horizon": config.policy.load_kwargs["model_chunk_size"],
        "execution_horizon": policy.execution_horizon,
        "denoise_steps": policy.schedule.num_steps,
        "noise_level": policy.schedule.noise_level,
        "trainable_parameters": sum(value.numel() for value in trainable.values()),
        "trainable_tensors": len(trainable),
        "trainable_name_examples": list(trainable)[:8],
        "rollout_rescore_max_abs_delta": delta.max().item(),
        "gradient_norm": gradient_norm.item(),
        "adapter_checkpoint_bytes": checkpoint_bytes,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
