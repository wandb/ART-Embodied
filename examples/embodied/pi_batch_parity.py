"""Compare real PI policy outputs between batched and singleton inference.

The training path evaluates counterfactual siblings in one model batch, while
fixed evaluation normally uses singleton inference. This probe holds processed
observations and initial flow noise fixed, then measures whether batching alone
changes the native ODE action or Flow-SDE transition score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from art_embodied import EmbodiedExperimentConfig, make_policy
from art_embodied.compatibility import (
    require_compatible_runtime,
    runtime_profile_for_policy,
)
from art_embodied.experiment import RolloutContext
from art_embodied.integrations.pi_flow_sde import (
    _concatenate_processed_batches,
    _postprocess_pi_actions,
    _prepare_pi_observation,
)
from art_embodied.policies.pi_flow_sde import PIFlowModelInputs
from examples.embodied.libero.components import (
    LiberoSettings,
    LiberoTaskCatalog,
    build_evaluation_scenarios,
    prepare_libero_runtime_paths,
    validate_libero_runtime_imports,
    validate_libero_task_assets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.blake2b(
        "\x1f".join(str(part) for part in parts).encode(),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


def _model_actions(
    policy: Any,
    inputs: PIFlowModelInputs,
    noise: torch.Tensor,
    *,
    denoise_steps: int,
) -> torch.Tensor:
    model = policy.bridge.model
    kwargs: dict[str, Any] = {
        "images": list(inputs.images),
        "img_masks": list(inputs.image_masks),
        "noise": noise,
        "num_steps": denoise_steps,
    }
    if policy.family == "pi0":
        if inputs.state is None:
            raise RuntimeError("PI0 batch-parity input is missing proprioceptive state")
        kwargs.update(
            {
                "lang_tokens": inputs.language_tokens,
                "lang_masks": inputs.language_masks,
                "state": inputs.state,
            }
        )
    else:
        kwargs.update(
            {
                "tokens": inputs.language_tokens,
                "masks": inputs.language_masks,
            }
        )
    return model.sample_actions(**kwargs)


def _error_summary(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    delta = (candidate.detach().float() - reference.detach().float()).abs()
    return {
        "max_abs_delta": float(delta.max().cpu()),
        "mean_abs_delta": float(delta.mean().cpu()),
        "rmse": float(torch.sqrt(torch.mean(delta.square())).cpu()),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2")

    config = EmbodiedExperimentConfig.from_yaml(args.config)
    if config.policy.type not in {"pi0", "pi05"}:
        raise ValueError("PI batch parity requires policy.type pi0 or pi05")
    require_compatible_runtime(profile=runtime_profile_for_policy(config.policy.type))
    prepare_libero_runtime_paths()
    validate_libero_runtime_imports()
    settings = LiberoSettings.from_config(config)
    validate_libero_task_assets(settings)

    policy = make_policy(config)
    policy.eval()
    if policy.bridge is None or policy.preprocessor is None or policy.postprocessor is None:
        raise RuntimeError("PI policy did not initialize its bridge and processors")
    denoise_steps = int(policy.schedule.num_steps)
    scenarios = build_evaluation_scenarios(config)[: args.batch_size]
    if len(scenarios) != args.batch_size:
        raise ValueError("evaluation plan has fewer scenarios than --batch-size")

    catalog = LiberoTaskCatalog(settings)
    input_rows: list[PIFlowModelInputs] = []
    prepared_rows: list[dict[str, Any]] = []
    processed_rows: list[dict[str, Any]] = []
    scenario_ids: list[str] = []
    for index, scenario in enumerate(scenarios):
        environment_seed = _stable_seed(scenario.id, "pi-batch-parity")
        context = RolloutContext(
            update=0,
            group_index=index,
            attempt_index=0,
            environment_seed=environment_seed,
            policy_seed=args.seed,
            config_fingerprint="pi-batch-parity",
        )
        environment = catalog.make_environment(scenario, context)
        try:
            observation, _ = environment.reset(
                seed=environment_seed,
                options=scenario.payload.get("reset_options"),
            )
        finally:
            environment.close()
        prepared = _prepare_pi_observation(
            observation,
            policy=policy,
            task=scenario.task,
            robot_type="panda",
        )
        processed = policy.preprocessor(prepared)
        prepared_rows.append(prepared)
        processed_rows.append(processed)
        input_rows.append(policy.bridge.prepare_inputs(processed))
        scenario_ids.append(scenario.id)

    inputs = PIFlowModelInputs.concatenate(input_rows)
    processed_batch = _concatenate_processed_batches(processed_rows)
    generator = torch.Generator(device=policy.device).manual_seed(args.seed)
    noise = torch.randn(
        (
            args.batch_size,
            int(policy.policy.config.chunk_size),
            int(policy.policy.config.max_action_dim),
        ),
        generator=generator,
        device=policy.device,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        batched = _model_actions(
            policy,
            inputs,
            noise,
            denoise_steps=denoise_steps,
        )
        singleton = torch.cat(
            [
                _model_actions(
                    policy,
                    inputs.select(index),
                    noise[index : index + 1],
                    denoise_steps=denoise_steps,
                )
                for index in range(args.batch_size)
            ],
            dim=0,
        )
        # PI models predict max_action_dim internally. The production bridge
        # removes padded dimensions before LeRobot's action postprocessor.
        batched = batched[:, :, : policy.action_dim]
        singleton = singleton[:, :, : policy.action_dim]
        batched_native = _postprocess_pi_actions(
            policy,
            batched,
            prepared_rows=prepared_rows,
        )
        singleton_native = _postprocess_pi_actions(
            policy,
            singleton,
            prepared_rows=prepared_rows,
        )

    selected = torch.full(
        (args.batch_size,),
        min(denoise_steps // 2, denoise_steps - 1),
        dtype=torch.long,
        device=policy.device,
    )
    rollout = policy.sample_flow_sde(
        processed_batch,
        selected_index=selected,
        initial_noise=noise.clone(),
    )
    batched_scores = policy.flow_sde_logprobs(rollout)
    singleton_scores = torch.cat(
        [
            policy.flow_sde_logprobs(rollout.select(index))
            for index in range(args.batch_size)
        ],
        dim=0,
    )

    report = {
        "schema_version": 1,
        "config": str(args.config.expanduser().resolve()),
        "policy_type": config.policy.type,
        "policy_path": config.policy.path,
        "batch_size": args.batch_size,
        "denoise_steps": denoise_steps,
        "scenario_ids": scenario_ids,
        "normalized_ode_batch_vs_singleton": _error_summary(batched, singleton),
        "native_ode_batch_vs_singleton": _error_summary(
            batched_native,
            singleton_native,
        ),
        "flow_sde_rescore_batch_vs_singleton": _error_summary(
            batched_scores,
            singleton_scores,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
