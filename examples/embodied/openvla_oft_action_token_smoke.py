"""Implementation smoke for the real OpenVLA-OFT action-token update path."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import shutil

import numpy as np

from art_embodied import (
    EmbodiedBackend,
    EmbodiedExperimentConfig,
    EmbodiedTrainableModel,
    EmbodiedTrajectory,
    EmbodiedTrajectoryGroup,
    Observation,
    make_embodied_backend,
    make_policy,
)
from art_embodied.utils import make_json_safe


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_suffix(".yaml"),
    )
    parser.add_argument(
        "--keep-checkpoint",
        action="store_true",
        help="Retain the temporary LoRA checkpoint after the smoke succeeds.",
    )
    return parser.parse_args()


def _synthetic_observation() -> Observation:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[..., 0] = 48
    image[48:208, 80:176, 1] = 192
    return Observation(step=0, kind="image", value=image)


def _snapshot_trainable_parameters(policy):
    snapshots = {
        name: parameter.detach().cpu().clone()
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    }
    if not snapshots:
        raise RuntimeError("OpenVLA-OFT smoke found no trainable parameter")
    return snapshots


async def _run(config: EmbodiedExperimentConfig, *, keep_checkpoint: bool) -> None:
    import torch

    policy = make_policy(config)
    parameters_before = _snapshot_trainable_parameters(policy)
    observation = _synthetic_observation()
    task = "pick up the red object and place it in the basket"
    sampled_token_signatures: list[list[int]] = []
    native_backend = make_embodied_backend(config, policy=policy)
    backend = EmbodiedBackend(native_backend, config=config)
    model = EmbodiedTrainableModel(policy=policy, config=config)
    results = []
    art_model_step = 0
    try:
        await model.register(backend)
        for update_index in range(config.training.updates):
            groups = []
            group_count = (
                config.rollout.groups_per_update * config.rollout.epochs_per_update
            )
            for group_index in range(group_count):
                trajectories = []
                for attempt in range(config.algorithm.group_size):
                    reward = float(attempt < config.algorithm.group_size // 2)
                    policy_seed = (
                        config.experiment.seed
                        + update_index * group_count * config.algorithm.group_size
                        + group_index * config.algorithm.group_size
                        + attempt
                    )
                    torch.manual_seed(policy_seed)
                    action = policy.act(
                        observation,
                        {
                            "step": update_index,
                            "scenario": {"task": task},
                            "policy_seed": policy_seed,
                        },
                    )
                    action.metadata.setdefault("observation_index", 0)
                    sampled_token_signatures.append(
                        [int(token) for token in action.raw.get("tokens", [])]
                    )
                    trajectory = EmbodiedTrajectory(task=task, reward=reward)
                    trajectory.observations.append(observation.model_copy(deep=True))
                    trajectory.actions.append(action)
                    trajectories.append(trajectory.finish())
                groups.append(EmbodiedTrajectoryGroup(trajectories))
            results.append(
                await backend.train(
                    model,
                    groups,
                    learning_rate=config.training.optimizer.learning_rate,
                )
            )
        art_model_step = await model.get_step()
    finally:
        await model.close()

    result = results[-1]

    changed_parameters = [
        name
        for name, parameter in policy.named_parameters()
        if name in parameters_before
        and not torch.equal(parameters_before[name], parameter.detach().cpu())
    ]
    checkpoint_path = getattr(result, "checkpoint_path", None)
    if not checkpoint_path or not Path(checkpoint_path).is_dir():
        diagnostic = {
            "changed_parameter_count": len(changed_parameters),
            "checkpoint_path": checkpoint_path,
            "metrics": make_json_safe(result.metrics),
            "sampled_action_count": len(sampled_token_signatures),
            "unique_sampled_token_signatures": len(
                {tuple(tokens) for tokens in sampled_token_signatures}
            ),
        }
        raise RuntimeError(
            "OpenVLA-OFT smoke did not write a checkpoint:\n"
            + json.dumps(diagnostic, indent=2, sort_keys=True)
        )
    policy.load_checkpoint({"path": checkpoint_path})
    checkpoint_reloaded = (
        Path(policy.peft_adapter_path).resolve() == Path(checkpoint_path).resolve()
        if policy.peft_adapter_path is not None
        else False
    )
    if not checkpoint_reloaded:
        raise RuntimeError(
            "OpenVLA-OFT smoke did not reload its adapter-only checkpoint: "
            f"saved={checkpoint_path!r}, loaded={policy.peft_adapter_path!r}"
        )

    report = {
        "ok": True,
        "config_fingerprint": config.fingerprint,
        "native_runtime": make_json_safe(policy.native_runtime_report),
        "policy_type": config.policy.type,
        "trainable_parameter_count": len(parameters_before),
        "changed_parameter_count": len(changed_parameters),
        "changed_parameter_example": (
            changed_parameters[0] if changed_parameters else None
        ),
        "checkpoint_reloaded": checkpoint_reloaded,
        "parameter_changed": bool(changed_parameters),
        "sampled_action_count": len(sampled_token_signatures),
        "unique_sampled_token_signatures": len(
            {tuple(tokens) for tokens in sampled_token_signatures}
        ),
        "checkpoint_path": checkpoint_path,
        "metrics": make_json_safe(result.metrics),
        "art_model_step": art_model_step,
        "updates": [
            {
                "step": update_result.step,
                "checkpoint_path": update_result.checkpoint_path,
                "metrics": make_json_safe(update_result.metrics),
            }
            for update_result in results
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not changed_parameters:
        raise RuntimeError(
            "OpenVLA-OFT smoke completed without changing any trainable parameter; "
            "inspect the emitted metrics and sampled-token diversity"
        )
    if not keep_checkpoint:
        shutil.rmtree(checkpoint_path)


def main() -> None:
    args = _arguments()
    config = EmbodiedExperimentConfig.from_yaml(args.config)
    asyncio.run(_run(config, keep_checkpoint=args.keep_checkpoint))


if __name__ == "__main__":
    main()
