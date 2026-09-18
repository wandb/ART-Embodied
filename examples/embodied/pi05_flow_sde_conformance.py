"""GPU conformance gate for LeRobot PI0.5 and ART-Embodied Flow-SDE.

The gate proves three contracts before a simulator training run is allowed:

1. ART's deterministic chain is the native LeRobot PI0.5 Euler sampler.
2. A sampled Flow-SDE transition has identical rollout and rescore logprobs.
3. The rescored transition differentiates through the native action expert.

It intentionally uses processed synthetic observations. Environment processor
parity is a separate gate; mixing it into the probability test would make a
failure ambiguous.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from art_embodied.policies.flow_sde import (
    FlowSDESchedule,
    flow_timesteps,
    rescore_flow_sde_chain,
    sample_flow_ode_chain,
    sample_flow_sde_chain,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="lerobot/pi05_libero_finetuned_v044",
        help="LeRobot PI0.5 model ID or local snapshot.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--noise-level", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("PI0.5 Flow-SDE conformance requires a CUDA device")

    try:
        from lerobot.configs import PreTrainedConfig
    except ImportError:  # LeRobot 0.4.x exports it from the policies module.
        from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    config = PreTrainedConfig.from_pretrained(args.model)
    config.device = str(device)
    config.compile_model = False
    config.gradient_checkpointing = False
    policy = PI05Policy.from_pretrained(args.model, config=config, strict=True)
    policy.eval()
    model = policy.model

    batch_size = args.batch_size
    image_count = sum(
        feature.type.name == "VISUAL"
        for feature in policy.config.input_features.values()
    )
    images = [
        torch.rand(batch_size, 3, 224, 224, device=device, dtype=torch.float32)
        for _ in range(image_count)
    ]
    image_masks = [
        torch.ones(batch_size, device=device, dtype=torch.bool)
        for _ in range(image_count)
    ]
    tokens = torch.ones(
        batch_size,
        policy.config.tokenizer_max_length,
        device=device,
        dtype=torch.long,
    )
    token_masks = torch.ones_like(tokens, dtype=torch.bool)
    initial_noise = torch.randn(
        batch_size,
        policy.config.chunk_size,
        policy.config.max_action_dim,
        device=device,
        dtype=torch.float32,
    )
    schedule = FlowSDESchedule(
        num_steps=args.num_steps,
        noise_level=args.noise_level,
    )

    with torch.no_grad():
        native_actions = model.sample_actions(
            images,
            image_masks,
            tokens,
            token_masks,
            noise=initial_noise.clone(),
            num_steps=args.num_steps,
        )
        prefix_masks, past_key_values = _prefix_cache(
            model,
            images=images,
            image_masks=image_masks,
            tokens=tokens,
            token_masks=token_masks,
            make_att_2d_masks=make_att_2d_masks,
        )

    def velocity(state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return model.denoise_step(
            prefix_pad_masks=prefix_masks,
            past_key_values=past_key_values,
            x_t=state,
            timestep=timestep,
        )

    ode_chain = sample_flow_ode_chain(
        initial_noise.clone(),
        schedule=schedule,
        velocity_fn=velocity,
    )
    native_delta = (ode_chain[:, -1] - native_actions).abs()
    if native_delta.max().item() > 1.0e-4:
        raise AssertionError(
            "ART Flow-SDE deterministic path diverges from native PI0.5: "
            f"max_abs_delta={native_delta.max().item():.6g}"
        )

    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    selected_indices = torch.arange(batch_size, device=device) % args.num_steps
    chain = sample_flow_sde_chain(
        initial_noise.clone(),
        selected_indices=selected_indices,
        schedule=schedule,
        velocity_fn=velocity,
        generator=generator,
    )
    rescored = rescore_flow_sde_chain(
        chain,
        schedule=schedule,
        velocity_fn=velocity,
    )
    rescore_delta = (rescored - chain.selected_logprobs).abs()
    if rescore_delta.max().item() > 2.0e-4:
        raise AssertionError(
            "PI0.5 Flow-SDE rollout/rescore mismatch: "
            f"max_abs_delta={rescore_delta.max().item():.6g}"
        )

    model.zero_grad(set_to_none=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.action_out_proj.weight.requires_grad_(True)
    gradient_logprobs = rescore_flow_sde_chain(
        chain,
        schedule=schedule,
        velocity_fn=velocity,
    )
    (-gradient_logprobs.mean()).backward()
    gradient = model.action_out_proj.weight.grad
    if gradient is None or not torch.isfinite(gradient).all() or gradient.norm() == 0:
        raise AssertionError(
            "Flow-SDE rescore did not produce a finite action-expert gradient"
        )

    result = {
        "status": "ok",
        "model": args.model,
        "device": str(device),
        "batch_size": batch_size,
        "image_count": image_count,
        "chunk_size": policy.config.chunk_size,
        "action_dim": policy.config.max_action_dim,
        "num_steps": args.num_steps,
        "noise_level": args.noise_level,
        "selected_indices": selected_indices.tolist(),
        "native_ode_max_abs_delta": native_delta.max().item(),
        "rollout_rescore_max_abs_delta": rescore_delta.max().item(),
        "rollout_rescore_mean_abs_delta": rescore_delta.mean().item(),
        "gradient_norm": gradient.norm().item(),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "timesteps": flow_timesteps(
            schedule,
            device="cpu",
        ).tolist(),
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")


def _prefix_cache(
    model,
    *,
    images,
    image_masks,
    tokens,
    token_masks,
    make_att_2d_masks,
):
    prefix_embeddings, prefix_masks, prefix_attention = model.embed_prefix(
        images,
        image_masks,
        tokens,
        token_masks,
    )
    attention = make_att_2d_masks(prefix_masks, prefix_attention)
    attention = model._prepare_attention_masks_4d(attention)  # noqa: SLF001
    positions = torch.cumsum(prefix_masks, dim=1) - 1
    model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
    _, cache = model.paligemma_with_expert.forward(
        attention_mask=attention,
        position_ids=positions,
        past_key_values=None,
        inputs_embeds=[prefix_embeddings, None],
        use_cache=True,
    )
    return prefix_masks, cache


if __name__ == "__main__":
    main()
