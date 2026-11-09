"""Convert recorded DROID batches to the shared Flow-SDE training contract."""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

from art_embodied.backends.flow_sde import TRANSIENT_FLOW_SDE_ROLLOUT_KEY
from art_embodied.integrations.isaaclab_droid import droid_action_chunk
from art_embodied.integrations.isaaclab_reset_bank import IsaacLabResetBank
from art_embodied.policies.flow_policy import FlowSDERollout
from art_embodied.policies.flow_replay_io import compact_masked_images, load_flow_replay
from art_embodied.trajectories import (
    Action,
    EmbodiedTrajectory,
    EmbodiedTrajectoryGroup,
    MediaRef,
    Observation,
)
from art_embodied.utils import make_json_safe
from examples.embodied.isaaclab.droid_environment import DroidSceneConfig


def read_droid_groups(
    directory,
    records,
    *,
    bank: IsaacLabResetBank,
    policy,
    policy_version: int,
    group_size: int,
    group_offset: int = 0,
):
    """Reject incomplete, stale, or differently initialized behavior data."""
    directory, records = Path(directory), Path(records)
    report = json.loads((directory / "report.json").read_text())
    settings = DroidSceneConfig(**report["settings"])
    if bank.split != "train" or bank.contract != settings.reset_contract():
        raise ValueError("Training rollout and reset bank contracts differ")
    if (
        report["model_server"]["mode"] != "sde"
        or report["model_server"]["training_updates"] != policy_version
        or report["optimizer_updates"] != policy_version
        or report["split"] != "train"
        or report["group_size"] != group_size
        or report["model_server"]["batch_size"] != policy.batch_size
        or report["model_server"]["noise_level"] != policy.bridge.schedule.noise_level
        or report["model_server"]["denoising_steps"] != policy.bridge.schedule.num_steps
        or not report["initial_state_and_camera_replay_exact"]
    ):
        raise ValueError("Behavior rollout does not match the training policy contract")
    episodes = json.loads((directory / "episodes.json").read_text())
    if len(episodes) != report["episodes"] or not episodes:
        raise ValueError("Incomplete behavior episodes")
    by_row, episode_data = {}, {}
    for episode in episodes:
        key = (episode["batch"], episode["row"])
        if key in by_row:
            raise ValueError("Duplicate behavior episode")
        index = episode["bank_index"]
        if (
            type(index) is not int
            or not 0 <= index < len(bank)
            or episode["reset_sha256"] != bank.hashes[index]
        ):
            raise ValueError("Behavior episode reset differs from the training bank")
        if not (episode["terminated"] or episode["truncated"]):
            raise ValueError("Behavior episode is not finished")
        trajectory = EmbodiedTrajectory(
            task=settings.instruction,
            reward=float(episode["success"]),
            initial_policy_version=policy_version,
            final_policy_version=policy_version,
            metrics={
                "success": float(episode["success"]),
                "environment_steps": episode["env_steps"],
            },
            metadata={
                "simulator": "isaaclab",
                "phase": "train",
                "scenario_id": f"train-{index}",
                "policy_seed": episode["policy_seed_base"],
                "environment_seed": episode["environment_seed"],
                "reset_state": make_json_safe(bank.select([index])),
                "reset_info": {
                    "comparison_state_sha256": episode["reset_sha256"],
                    "observation_sha256": episode["observation_sha256"],
                },
                "termination_reason": episode["termination_reason"],
                "pre_update_policy_version": policy_version,
            },
            media=[MediaRef.model_validate(m) for m in episode["media"]],
        )
        trajectory.add_reward(
            "success",
            trajectory.reward,
            "env",
            step=episode["env_steps"],
            update_total=False,
        )
        by_row[key], episode_data[key] = trajectory, episode
    seen_records = set()
    for chunk in json.loads((directory / "actions.json").read_text()):
        record_id = chunk["record_id"]
        if (
            not isinstance(record_id, str)
            or not re.fullmatch(r"chunk-\d{6}", record_id)
            or record_id in seen_records
        ):
            raise ValueError("Invalid or repeated replay record ID")
        seen_records.add(record_id)
        rollout, valid = load_flow_replay(records / f"{record_id}.pt")
        if (
            valid != len(chunk["active_rows"])
            or rollout.inputs.batch_size != policy.batch_size
        ):
            raise ValueError("Behavior replay batch size differs")
        if (
            len(chunk["actions"]) != valid
            or len(chunk["execution_mask"]) != valid
            or len(set(chunk["active_rows"])) != valid
        ):
            raise ValueError("Invalid behavior action row mapping")
        for offset, row in enumerate(chunk["active_rows"]):
            trajectory = by_row[(chunk["batch"], row)]
            if chunk["chunk"] != len(trajectory.actions):
                raise ValueError("Behavior chunks are missing or duplicated")
            # Copy each row before handoff; retaining a view would serialize the
            # whole behavior batch once per trajectory in worker payloads.
            one = compact_masked_images(
                FlowSDERollout.concatenate([rollout.select(offset)])
            )
            physical = droid_action_chunk(
                policy.bridge.postprocess(one.inputs, one.actions),
                execution_horizon=policy.bridge.execution_horizon,
            )
            if not np.array_equal(physical, np.asarray(chunk["actions"][offset])):
                raise ValueError("Replayed action differs from the executed command")
            mask = chunk["execution_mask"][offset]
            if (
                len(mask) != len(physical)
                or any(type(value) is not bool for value in mask)
                or not any(mask)
            ):
                raise ValueError("Invalid execution loss mask")
            trajectory.observations.append(
                Observation(
                    step=chunk["chunk"],
                    kind="state",
                    value=one.inputs.state[0].tolist(),
                    metadata={"stage": "normalized_policy_input"},
                )
            )
            trajectory.actions.append(
                Action(
                    step=chunk["chunk"],
                    kind="continuous",
                    raw=physical.tolist(),
                    logprobs=one.transition.old_logprobs[
                        0, : len(physical), : physical.shape[1]
                    ].tolist(),
                    metadata={
                        TRANSIENT_FLOW_SDE_ROLLOUT_KEY: one,
                        "primitive_loss_mask": mask,
                        "primitive_loss_mask_sum": sum(mask),
                        "likelihood_event": "selected_flow_sde_transition",
                        "isaaclab_execution": {
                            "primitive_loss_mask": mask,
                            "executed_actions": physical[np.asarray(mask)].tolist(),
                        },
                    },
                )
            )
    grouped = {}
    for key, trajectory in by_row.items():
        episode = episode_data[key]
        if (
            sum(a.metadata["primitive_loss_mask_sum"] for a in trajectory.actions)
            != episode["env_steps"]
        ):
            raise ValueError("Replay omits executed environment steps")
        trajectory.metrics["episode_steps"] = len(trajectory.actions)
        grouped.setdefault(episode["group"], []).append(trajectory)
    groups = []
    for _, members in sorted(grouped.items()):
        if len(members) != group_size:
            raise ValueError("Incomplete GRPO group")
        if (
            len({t.metadata["scenario_id"] for t in members}) != 1
            or len({t.metadata["reset_info"]["observation_sha256"] for t in members})
            != 1
        ):
            raise ValueError("GRPO group has different initial conditions")
        if len({t.metadata["policy_seed"] for t in members}) != group_size:
            raise ValueError("GRPO attempts reused a policy seed")
        groups.append(
            EmbodiedTrajectoryGroup(
                members,
                metadata={
                    "group_index": group_offset + len(groups),
                    "scenario_id": members[0].metadata["scenario_id"],
                    "update": policy_version,
                },
            )
        )
    return groups


def logging_groups(groups):
    """Keep actions, rewards and media, without embedding training image tensors."""
    return [
        group.model_copy(
            update={
                "trajectories": [
                    trajectory.model_copy(
                        update={
                            "actions": [
                                action.model_copy(
                                    update={
                                        "metadata": {
                                            key: value
                                            for key, value in action.metadata.items()
                                            if key != TRANSIENT_FLOW_SDE_ROLLOUT_KEY
                                        }
                                    }
                                )
                                for action in trajectory.actions
                            ]
                        }
                    )
                    for trajectory in group.trajectories
                ]
            }
        )
        for group in groups
    ]
