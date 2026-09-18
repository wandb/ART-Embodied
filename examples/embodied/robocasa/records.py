"""Task-balanced scenarios and trajectory records for RoboCasa."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from art_embodied.experiment import EmbodiedScenario
from art_embodied.trajectories import Action, EmbodiedTrajectory, Observation
from art_embodied.utils import make_json_safe

from .settings import RoboCasaSettings, RoboCasaTask


def build_train_scenarios(config: Any) -> list[EmbodiedScenario]:
    """Round-robin tasks while keeping every GRPO group task-homogeneous."""

    settings = RoboCasaSettings.from_config(config)
    groups = (
        int(config.training.updates)
        * int(config.rollout.groups_per_update)
        * int(config.rollout.epochs_per_update)
    )
    return [
        _scenario(
            settings.tasks[index % len(settings.tasks)], phase="train", index=index
        )
        for index in range(groups)
    ]


def build_evaluation_scenarios(config: Any) -> list[EmbodiedScenario]:
    settings = RoboCasaSettings.from_config(config)
    return [
        _scenario(task, phase="eval", index=index)
        for index, task in enumerate(settings.tasks)
    ]


def _scenario(task: RoboCasaTask, *, phase: str, index: int) -> EmbodiedScenario:
    return EmbodiedScenario(
        id=f"robocasa-gr1/{task.id}/{phase}",
        task=task.id,
        payload={
            "task_id": task.id,
            "environment_id": task.environment_id,
            "dataset_id": task.dataset_id,
            "schedule_index": int(index),
        },
    )


def record_robocasa_observation(
    *, observation: Mapping[str, Any], step: int
) -> Observation:
    value = {
        str(key): (item if isinstance(item, str) else np.asarray(item))
        for key, item in observation.items()
    }
    fields = {}
    for key, item in value.items():
        if isinstance(item, str):
            fields[key] = {"type": "text"}
        else:
            fields[key] = {"shape": list(item.shape), "dtype": str(item.dtype)}
    return Observation(
        step=int(step),
        kind="image",
        value=value,
        metadata={"framework": "robocasa", "fields": fields},
    )


def record_robocasa_transition(
    *,
    trajectory: EmbodiedTrajectory,
    action: Action,
    reward: float,
    info: Mapping[str, Any],
    policy_step: int,
    **_kwargs: Any,
) -> None:
    rewards = [float(value) for value in info["primitive_rewards"]]
    loss_mask = [bool(value) for value in info["primitive_loss_mask"]]
    if len(rewards) != len(loss_mask):
        raise ValueError("RoboCasa primitive rewards and loss mask must align")
    action.metadata.update(
        {
            "primitive_rewards": rewards,
            "primitive_loss_mask": loss_mask,
            "primitive_loss_mask_sum": int(sum(loss_mask)),
            "executed_action_vectors": make_json_safe(info["executed_actions"]),
            "environment_chunk_reward": float(reward),
            "policy_step": int(policy_step),
            "observation_index": int(policy_step),
            "task_id": str(info["task_id"]),
        }
    )
    executed_count = int(sum(loss_mask))
    first_executed_step = max(0, int(info["env_steps"]) - executed_count)
    for chunk_index, (value, keep) in enumerate(zip(rewards, loss_mask, strict=True)):
        trajectory.add_reward(
            "robocasa_primitive_reward",
            value,
            "env",
            step=first_executed_step + min(chunk_index + 1, executed_count),
            metadata={
                "policy_step": int(policy_step),
                "chunk_index": chunk_index,
                "primitive_loss_mask": keep,
                "task_id": str(info["task_id"]),
            },
            update_total=False,
        )
