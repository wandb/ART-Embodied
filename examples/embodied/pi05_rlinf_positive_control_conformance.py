"""End-to-end GPU gate for the RLinf PI0.5 SFT positive-control contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from art_embodied.integrations.pi_flow_sde import _prepare_pi_observation
from art_embodied.policies.flow_sde import FlowSDESchedule
from art_embodied.policies.pi import PIFlowPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="RLinf/RLinf-Pi05-SFT")
    parser.add_argument(
        "--revision", default="45ccfcc4e28634f1576ebf78cab0fbe2fd82432d"
    )
    parser.add_argument(
        "--architecture",
        default="lerobot/pi05_libero_finetuned_v044",
    )
    parser.add_argument(
        "--architecture-revision",
        default="dbf8a3f794a9c4297b44f40b752712f50073d945",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--execution-horizon", type=int, default=5)
    parser.add_argument("--model-chunk-size", type=int, default=10)
    parser.add_argument("--noise-level", type=float, default=0.3)
    parser.add_argument("--num-denoise-steps", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("PI0.5 positive-control conformance requires CUDA")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    policy = PIFlowPolicy(
        family="pi05",
        model_id=args.model,
        revision=args.revision,
        device=args.device,
        dtype="bfloat16",
        model_format="rlinf_openpi_safetensors",
        execution_horizon=args.execution_horizon,
        action_dim=7,
        processor_path=args.architecture,
        processor_revision=args.architecture_revision,
        model_chunk_size=args.model_chunk_size,
        normalization_stats_file="physical-intelligence/libero/norm_stats.json",
        discrete_state_input=False,
        extra_delta_transform=False,
        observation_key_map={
            "image": "observation.images.image",
            "wrist_image": "observation.images.image2",
            "proprio_state": "observation.state",
        },
        schedule=FlowSDESchedule(
            num_steps=args.num_denoise_steps,
            noise_level=args.noise_level,
        ),
        strict_weights=True,
        compile_model=False,
        gradient_checkpointing=False,
        train_expert_only=True,
    )
    policy.load()
    prepared = _prepare_pi_observation(
        {
            "image": np.zeros((256, 256, 3), dtype=np.uint8),
            "wrist_image": np.full((256, 256, 3), 127, dtype=np.uint8),
            "proprio_state": np.array(
                [0.0, 0.0, 0.75, 3.0, 0.0, 0.0, 0.02, -0.02],
                dtype=np.float32,
            ),
        },
        policy=policy,
        task="pick up the red block",
        robot_type=None,
    )
    processed = policy.preprocessor(prepared)
    rollout = policy.sample_flow_sde(
        processed,
        selected_index=torch.tensor(
            [min(2, args.num_denoise_steps - 1)], device=args.device
        ),
        generator=torch.Generator(device=args.device).manual_seed(args.seed + 1),
    )
    old = rollout.transition.old_logprobs[
        :, : policy.execution_horizon, : policy.action_dim
    ]
    current = policy.flow_sde_logprobs(rollout)
    delta = (current.detach() - old).abs()
    if delta.max().item() > 2.0e-4:
        raise AssertionError(
            f"rollout/rescore mismatch: max_abs_delta={delta.max().item():.6g}"
        )

    policy.model.zero_grad(set_to_none=True)
    (-current.mean()).backward()
    gradients = [
        parameter.grad.detach().float().norm()
        for parameter in policy.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    gradient_norm = torch.stack(gradients).norm() if gradients else torch.tensor(0.0)
    if not torch.isfinite(gradient_norm) or gradient_norm.item() == 0.0:
        raise AssertionError(
            "RLinf PI0.5 rescore produced no finite trainable gradient"
        )

    native = policy.postprocessor(rollout.actions)
    result = {
        "status": "ok",
        "model": args.model,
        "revision": args.revision,
        "architecture": args.architecture,
        "architecture_revision": args.architecture_revision,
        "model_format": policy.model_format,
        "chunk_size": policy.config.chunk_size,
        "execution_horizon": policy.execution_horizon,
        "noise_level": policy.schedule.noise_level,
        "num_denoise_steps": policy.schedule.num_steps,
        "action_dim": policy.action_dim,
        "processed_state_shape": list(processed["observation.state"].shape),
        "language_token_shape": list(processed["observation.language.tokens"].shape),
        "native_action_shape": list(native.shape),
        "rollout_rescore_max_abs_delta": delta.max().item(),
        "rollout_rescore_mean_abs_delta": delta.mean().item(),
        "gradient_norm": gradient_norm.item(),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in policy.parameters()
            if parameter.requires_grad
        ),
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
