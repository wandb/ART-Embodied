"""Dump matching PI tensors from ART/LeRobot or RLinf/OpenPI runtimes.

The two runtimes cannot coexist in one Python environment.  Run this script
once in each supported environment, then compare the resulting tensor bundles
with ``pi_cross_runtime_compare.py``.  Inputs and initial flow noise are
generated from NumPy so PyTorch-version-specific RNG is outside the contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

Runtime = Literal["art", "rlinf"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", choices=("art", "rlinf"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config", type=Path, help="ART experiment YAML")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--num-steps", type=int, default=4)
    return parser.parse_args()


def synthetic_inputs(*, batch_size: int, seed: int) -> dict[str, Any]:
    """Create runtime-neutral LIBERO observations and initial flow noise."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    generator = np.random.default_rng(seed)
    images = generator.integers(
        0, 256, size=(batch_size, 256, 256, 3), dtype=np.uint8
    )
    wrist_images = generator.integers(
        0, 256, size=(batch_size, 256, 256, 3), dtype=np.uint8
    )
    states = generator.normal(size=(batch_size, 8)).astype(np.float32)
    initial_noise = generator.normal(size=(batch_size, 50, 32)).astype(np.float32)
    prompts = [
        "pick up the black bowl between the plate and the ramekin and place it on the plate"
    ] * batch_size
    return {
        "images": images,
        "wrist_images": wrist_images,
        "states": states,
        "initial_noise": initial_noise,
        "prompts": prompts,
    }


def postprocess_art_actions(
    policy: Any,
    normalized_actions: torch.Tensor,
    *,
    prepared_rows: list[dict[str, Any]],
) -> Any:
    """Mirror PIFlowSDEBridge's model-width to environment-width boundary."""

    from art_embodied.integrations.pi_flow_sde import _postprocess_pi_actions

    return _postprocess_pi_actions(
        policy,
        normalized_actions[:, :, : policy.action_dim],
        prepared_rows=prepared_rows,
    )


@torch.no_grad()
def model_intermediates(
    model: Any,
    *,
    images: list[torch.Tensor] | tuple[torch.Tensor, ...],
    image_masks: list[torch.Tensor] | tuple[torch.Tensor, ...],
    language_tokens: torch.Tensor,
    language_masks: torch.Tensor,
    state: torch.Tensor,
    initial_noise: torch.Tensor,
) -> dict[str, Any]:
    """Capture the model-boundary tensors preceding the first velocity."""

    image_embeddings = [
        model.paligemma_with_expert.embed_image(image) for image in images
    ]
    language_embeddings = model.paligemma_with_expert.embed_language_tokens(
        language_tokens
    )
    prefix_embeddings, prefix_masks, prefix_attention = model.embed_prefix(
        list(images),
        list(image_masks),
        language_tokens,
        language_masks,
    )
    timestep = torch.ones(
        initial_noise.shape[0],
        dtype=torch.float32,
        device=initial_noise.device,
    )
    suffix_embeddings, suffix_masks, suffix_attention, _ = model.embed_suffix(
        state,
        initial_noise,
        timestep,
    )
    return {
        "image_embeddings": image_embeddings,
        "language_embeddings": language_embeddings,
        "prefix_embeddings": prefix_embeddings,
        "prefix_masks": prefix_masks,
        "prefix_attention": prefix_attention,
        "suffix_embeddings": suffix_embeddings,
        "suffix_masks": suffix_masks,
        "suffix_attention": suffix_attention,
    }


def dump_art(
    values: dict[str, Any],
    *,
    config_path: Path,
    num_steps: int,
) -> dict[str, Any]:
    """Run the LeRobot 0.6 policy boundary used by ART-Embodied."""

    from art_embodied.config import EmbodiedExperimentConfig
    from art_embodied.integrations.pi_flow_sde import (
        _concatenate_processed_batches,
        _prepare_pi_observation,
    )
    from art_embodied.policies.factory import make_policy

    config = EmbodiedExperimentConfig.from_yaml(config_path)
    if config.policy.type != "pi0":
        raise ValueError("cross-runtime parity currently targets PI0")
    if config.algorithm.flow_sde.num_denoise_steps != num_steps:
        raise ValueError(
            "--num-steps must match algorithm.flow_sde.num_denoise_steps"
        )
    policy = make_policy(config)
    policy.eval()
    prepared_rows = []
    processed_rows = []
    for index, prompt in enumerate(values["prompts"]):
        prepared = _prepare_pi_observation(
            {
                "image": values["images"][index],
                "wrist_image": values["wrist_images"][index],
                "proprio_state": values["states"][index],
            },
            policy=policy,
            task=prompt,
            robot_type="panda",
        )
        prepared_rows.append(prepared)
        processed_rows.append(policy.preprocessor(prepared))
    processed = _concatenate_processed_batches(processed_rows)
    inputs = policy.bridge.prepare_inputs(processed)
    initial_noise = torch.from_numpy(values["initial_noise"]).to(policy.device)
    intermediates = model_intermediates(
        policy.model,
        images=inputs.images,
        image_masks=inputs.image_masks,
        language_tokens=inputs.language_tokens,
        language_masks=inputs.language_masks,
        state=inputs.state,
        initial_noise=initial_noise,
    )
    prefix_masks, cache = policy.bridge._prefix_cache(inputs)  # noqa: SLF001
    state = initial_noise
    velocities = []
    states = [state]
    for index in range(num_steps):
        time = torch.full(
            (state.shape[0],),
            1.0 - index / num_steps,
            dtype=torch.float32,
            device=state.device,
        )
        velocity = policy.bridge._velocity(  # noqa: SLF001
            inputs=inputs,
            prefix_masks=prefix_masks,
            cache=cache,
            state=state,
            timestep=time,
        )
        velocities.append(velocity)
        state = state + (-1.0 / num_steps) * velocity
        states.append(state)
    native = postprocess_art_actions(
        policy,
        state,
        prepared_rows=prepared_rows,
    )
    return _bundle(
        runtime="art",
        images=inputs.images,
        image_masks=inputs.image_masks,
        language_tokens=inputs.language_tokens,
        language_masks=inputs.language_masks,
        model_state=inputs.state,
        initial_noise=torch.from_numpy(values["initial_noise"]),
        flow_states=torch.stack(states, dim=1),
        velocities=torch.stack(velocities, dim=1),
        normalized_actions=state,
        native_actions=native,
        intermediates=intermediates,
    )


def dump_rlinf(
    values: dict[str, Any],
    *,
    model_path: str,
    num_steps: int,
) -> dict[str, Any]:
    """Run RLinf v0.1's OpenPI model and transform boundary."""

    from omegaconf import OmegaConf
    from openpi.models import model as openpi_model
    from rlinf.models import get_model
    from rlinf.models.embodiment.openpi_action_model import make_att_2d_masks

    model_config = OmegaConf.create(
        {
            "model_type": "openpi",
            "model_path": model_path,
            "precision": None,
            "num_action_chunks": 5,
            "action_dim": 7,
            "is_lora": False,
            "lora_rank": 32,
            "use_proprio": True,
            "num_steps": num_steps,
            "add_value_head": False,
            "openpi": {
                "config_name": "pi0_libero",
                "num_images_in_input": 2,
                "noise_level": 0.5,
                "action_chunk": 5,
                "num_steps": num_steps,
                "train_expert_only": True,
                "action_env_dim": 7,
                "noise_method": "flow_sde",
                "add_value_head": False,
                "detach_critic_input": False,
            },
        }
    )
    model = get_model(model_config).to("cuda:0").eval()
    env_obs = {
        "images": torch.from_numpy(values["images"]).permute(0, 3, 1, 2),
        "wrist_images": torch.from_numpy(values["wrist_images"]).permute(0, 3, 1, 2),
        "states": torch.from_numpy(values["states"]),
        "task_descriptions": values["prompts"],
    }
    transformed = model.precision_processor(
        model.input_transform(model.obs_processor(env_obs))
    )
    observation = openpi_model.Observation.from_dict(transformed)
    images, image_masks, tokens, token_masks, state_input = (
        model._preprocess_observation(observation, train=False)  # noqa: SLF001
    )
    initial_noise = torch.from_numpy(values["initial_noise"]).to("cuda:0")
    intermediates = model_intermediates(
        model,
        images=images,
        image_masks=image_masks,
        language_tokens=tokens,
        language_masks=token_masks,
        state=state_input,
        initial_noise=initial_noise,
    )
    prefix, prefix_masks, prefix_attention = model.embed_prefix(
        images, image_masks, tokens, token_masks
    )
    attention = make_att_2d_masks(prefix_masks, prefix_attention)
    attention = model._prepare_attention_masks_4d(attention)  # noqa: SLF001
    positions = torch.cumsum(prefix_masks, dim=1) - 1
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
        "eager"
    )
    _, cache = model.paligemma_with_expert.forward(
        attention_mask=attention,
        position_ids=positions,
        past_key_values=None,
        inputs_embeds=[prefix, None],
        use_cache=True,
    )
    flow_state = initial_noise
    velocities = []
    flow_states = [flow_state]
    for index in range(num_steps):
        time = torch.full(
            (flow_state.shape[0],),
            1.0 - index / num_steps,
            dtype=torch.float32,
            device=flow_state.device,
        )
        suffix = model.get_suffix_out(
            state_input,
            prefix_masks,
            cache,
            flow_state,
            time,
        )
        velocity = model.action_out_proj(suffix)
        velocities.append(velocity)
        flow_state = flow_state + (-1.0 / num_steps) * velocity
        flow_states.append(flow_state)
    native = model.output_transform(
        {"actions": flow_state, "state": observation.state}
    )["actions"]
    return _bundle(
        runtime="rlinf",
        images=tuple(images),
        image_masks=tuple(image_masks),
        language_tokens=tokens,
        language_masks=token_masks,
        model_state=state_input,
        initial_noise=torch.from_numpy(values["initial_noise"]),
        flow_states=torch.stack(flow_states, dim=1),
        velocities=torch.stack(velocities, dim=1),
        normalized_actions=flow_state,
        native_actions=torch.as_tensor(native),
        intermediates=intermediates,
    )


def _bundle(
    *,
    runtime: Runtime,
    images: tuple[torch.Tensor, ...] | list[torch.Tensor],
    image_masks: tuple[torch.Tensor, ...] | list[torch.Tensor],
    language_tokens: torch.Tensor,
    language_masks: torch.Tensor,
    model_state: torch.Tensor | None,
    initial_noise: torch.Tensor,
    flow_states: torch.Tensor,
    velocities: torch.Tensor,
    normalized_actions: torch.Tensor,
    native_actions: torch.Tensor,
    intermediates: dict[str, Any],
) -> dict[str, Any]:
    def cpu(value: torch.Tensor) -> torch.Tensor:
        return value.detach().cpu().contiguous()

    return {
        "schema_version": 1,
        "runtime": runtime,
        "images": [cpu(value) for value in images],
        "image_masks": [cpu(value) for value in image_masks],
        "language_tokens": cpu(language_tokens),
        "language_masks": cpu(language_masks),
        "model_state": None if model_state is None else cpu(model_state),
        "initial_noise": cpu(initial_noise),
        "flow_states": cpu(flow_states),
        "velocities": cpu(velocities),
        "normalized_actions": cpu(normalized_actions),
        "native_actions": cpu(native_actions),
        "intermediates": {
            key: (
                [cpu(item) for item in value]
                if isinstance(value, list)
                else cpu(value)
            )
            for key, value in intermediates.items()
        },
    }


def main() -> None:
    args = parse_args()
    values = synthetic_inputs(batch_size=args.batch_size, seed=args.seed)
    if args.runtime == "art":
        if args.config is None:
            raise ValueError("--config is required for --runtime art")
        bundle = dump_art(
            values,
            config_path=args.config,
            num_steps=args.num_steps,
        )
    else:
        bundle = dump_rlinf(
            values,
            model_path=args.model_path,
            num_steps=args.num_steps,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.output)
    summary = {
        "runtime": bundle["runtime"],
        "output": str(args.output.resolve()),
        "batch_size": args.batch_size,
        "num_steps": args.num_steps,
        "tensor_bytes": sum(
            value.numel() * value.element_size()
            for item in bundle.values()
            for value in (
                item if isinstance(item, list) else [item]
            )
            if isinstance(value, torch.Tensor)
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
