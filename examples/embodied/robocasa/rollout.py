"""Batched same-task, same-reset rollout over process-isolated RoboCasa."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from art_embodied.backends.flow_sde import (
    FLOW_SDE_REPLAY_SELECTED_KEY,
    TRANSIENT_FLOW_SDE_ROLLOUT_KEY,
)
from art_embodied.experiment import EmbodiedScenario, RolloutContext
from art_embodied.integrations.lerobot import LeRobotActionPrediction
from art_embodied.media import RolloutVideoRecorder
from art_embodied.trajectories import EmbodiedTrajectory

from .environment import POLICY_ACTION_DIM
from .records import record_robocasa_observation, record_robocasa_transition
from .settings import RoboCasaSettings
from .simulator import RoboCasaSimulatorProcess


async def rollout_robocasa_group(
    *,
    config: Any,
    settings: RoboCasaSettings,
    simulator: RoboCasaSimulatorProcess,
    policy_adapter: Any,
    scenario: EmbodiedScenario,
    contexts: tuple[RolloutContext, ...],
    phase: str,
) -> list[EmbodiedTrajectory]:
    """Collect one comparison group without crossing task or reset boundaries."""

    if phase not in {"train", "eval"}:
        raise ValueError(f"Unsupported RoboCasa phase: {phase!r}")
    expected = int(config.algorithm.group_size) if phase == "train" else 1
    if len(contexts) != expected:
        raise ValueError(
            f"RoboCasa {phase} rollout expected {expected} contexts, got {len(contexts)}"
        )
    if not contexts or len({context.environment_seed for context in contexts}) != 1:
        raise ValueError("RoboCasa group contexts must share one environment_seed")
    task_id = scenario.payload.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("RoboCasa scenario requires a task_id")

    reset = await simulator.request(
        {
            "op": "reset",
            "task_id": task_id,
            "num_envs": len(contexts),
            "seed": int(contexts[0].environment_seed),
        }
    )
    if int(reset["action_dim"]) != POLICY_ACTION_DIM:
        raise RuntimeError("RoboCasa runtime does not expose the official 29D action")
    if int(reset["max_episode_steps"]) != settings.max_environment_steps:
        raise RuntimeError(
            "RoboCasa episode horizon differs from YAML: "
            f"runtime={reset['max_episode_steps']}, "
            f"declared={settings.max_environment_steps}"
        )
    if str(reset["task_id"]) != task_id:
        raise RuntimeError("RoboCasa runtime task differs from the scenario")
    state_hashes = list(reset["state_hashes"])
    if phase == "train" and len(set(state_hashes)) != 1:
        raise RuntimeError(
            "RoboCasa group replicas do not share the same initial simulator state"
        )
    shared_prefix_chunks = (
        int(config.rollout.shared_prefix_action_chunks) if phase == "train" else 0
    )
    if shared_prefix_chunks:
        observation_hashes = list(reset.get("observation_hashes", []))
        if len(observation_hashes) != len(contexts):
            raise RuntimeError(
                "RoboCasa shared prefix requires initial policy observation hashes"
            )
        if len(set(observation_hashes)) != 1:
            raise RuntimeError(
                "RoboCasa group replicas do not share the same initial policy input"
            )

    observations = {int(key): value for key, value in reset["observations"].items()}
    trajectories = [
        EmbodiedTrajectory(
            task=scenario.task,
            metadata={
                "framework": "robocasa",
                "simulator": "mujoco",
                "phase": phase,
                "scenario_id": scenario.id,
                "task_id": task_id,
                "environment_seed": context.environment_seed,
                "policy_seed": context.policy_seed,
                "config_fingerprint": context.config_fingerprint,
                "vectorized_group_rollout": len(contexts) > 1,
                "vectorized_group_size": len(contexts),
                "initial_state_sha256": state_hashes[index],
                "simulator_revision": settings.manifest.source_revision,
            },
        )
        for index, context in enumerate(contexts)
    ]
    for env_id, trajectory in enumerate(trajectories):
        trajectory.observations.append(
            record_robocasa_observation(observation=observations[env_id], step=0)
        )

    recorders = _video_recorders(
        config=config,
        contexts=contexts,
        trajectories=trajectories,
        observations=observations,
        phase=phase,
    )
    group_seed = _group_policy_seed(contexts)
    policy_adapter.reset(seed=group_seed)
    active = set(range(len(contexts)))
    successes = [False] * len(contexts)
    truncated = [False] * len(contexts)
    completed_policy_steps = [0] * len(contexts)
    completed_environment_steps = [0] * len(contexts)
    progress_rewards = [0.0] * len(contexts)
    progress_stages = ["none"] * len(contexts)
    progress_events: list[list[dict[str, Any]]] = [[] for _ in contexts]
    branch_state_hash: str | None = None
    branch_observation_hash: str | None = None
    prefix_chunks_executed = 0
    prefix_completed_episode = False

    for policy_step in range(settings.max_policy_steps):
        active_ids = sorted(active)
        if not active_ids:
            break
        in_shared_prefix = policy_step < shared_prefix_chunks
        if in_shared_prefix and active_ids != list(range(len(contexts))):
            raise RuntimeError(
                "RoboCasa shared-prefix replicas diverged before the branch"
            )
        if in_shared_prefix:
            leader_predictions = policy_adapter.predict_batch(
                [observations[active_ids[0]]],
                tasks=[scenario.task],
                step=policy_step,
            )
            if len(leader_predictions) != 1:
                raise RuntimeError("RoboCasa shared-prefix leader returned wrong batch")
            predictions = [
                _shared_prefix_prediction(
                    leader_predictions[0],
                    leader_env_id=active_ids[0],
                    prefix_chunk_index=policy_step,
                )
                for _env_id in active_ids
            ]
        else:
            predictions = policy_adapter.predict_batch(
                [observations[env_id] for env_id in active_ids],
                tasks=[scenario.task] * len(active_ids),
                step=policy_step,
            )
        if len(predictions) != len(active_ids):
            raise RuntimeError("RoboCasa policy returned the wrong batch size")
        by_env = dict(zip(active_ids, predictions, strict=True))
        executed: dict[int, list[list[float]]] = {env_id: [] for env_id in active_ids}
        masks: dict[int, list[bool]] = {env_id: [] for env_id in active_ids}
        primitive_rewards: dict[int, list[float]] = {
            env_id: [] for env_id in active_ids
        }

        for primitive_index in range(settings.execution_horizon):
            active_before = sorted(active)
            if not active_before:
                break
            action_batch = np.zeros(
                (len(contexts), POLICY_ACTION_DIM), dtype=np.float32
            )
            for env_id in active_before:
                chunk = np.asarray(by_env[env_id].native_action, dtype=np.float32)
                if chunk.shape != (settings.execution_horizon, POLICY_ACTION_DIM):
                    raise RuntimeError(
                        "Policy-native RoboCasa action chunk differs from the official "
                        f"(8, {POLICY_ACTION_DIM}) contract: got={chunk.shape}"
                    )
                action_batch[env_id] = chunk[primitive_index]
                executed[env_id].append(action_batch[env_id].tolist())
                masks[env_id].append(True)
                completed_environment_steps[env_id] += 1
            include_branch_hashes = bool(
                in_shared_prefix
                and policy_step == shared_prefix_chunks - 1
                and primitive_index == settings.execution_horizon - 1
            )
            step_result = await simulator.request(
                {
                    "op": "step",
                    "actions": action_batch,
                    "end_of_chunk": primitive_index == settings.execution_horizon - 1,
                    "shared_prefix_step": in_shared_prefix,
                    "include_branch_hashes": include_branch_hashes,
                }
            )
            observations.update(
                {int(key): value for key, value in step_result["observations"].items()}
            )
            primitive_step = policy_step * settings.execution_horizon + primitive_index
            if primitive_step % 4 == 0:
                for env_id, recorder in recorders.items():
                    observation = observations.get(env_id)
                    if observation is not None:
                        recorder.capture_frame(
                            observation["video.ego_view_pad_res256_freq20"],
                            step=primitive_step + 1,
                        )
            active = {int(value) for value in step_result["active_env_ids"]}
            branch_hashes_returned = "state_hashes" in step_result
            if in_shared_prefix and (include_branch_hashes or branch_hashes_returned):
                branch_states = list(step_result.get("state_hashes", []))
                branch_observations = list(step_result.get("observation_hashes", []))
                if len(branch_states) != len(contexts) or len(set(branch_states)) != 1:
                    raise RuntimeError(
                        "RoboCasa simulator states differ at shared-prefix branch"
                    )
                if (
                    len(branch_observations) != len(contexts)
                    or len(set(branch_observations)) != 1
                ):
                    raise RuntimeError(
                        "RoboCasa policy inputs differ at shared-prefix branch"
                    )
                if active and active != set(range(len(contexts))):
                    raise RuntimeError(
                        "RoboCasa shared-prefix replicas diverged at the branch"
                    )
                if not active:
                    if not all(
                        bool(result["success"]) for result in step_result["results"]
                    ):
                        raise RuntimeError(
                            "RoboCasa shared prefix ended every replica without "
                            "official success"
                        )
                    prefix_completed_episode = True
                branch_state_hash = branch_states[0]
                branch_observation_hash = branch_observations[0]
            for result in step_result["results"]:
                env_id = int(result["env_id"])
                successes[env_id] = successes[env_id] or bool(result["success"])
                truncated[env_id] = truncated[env_id] or bool(result["truncated"])
                progress_rewards[env_id] = float(
                    result.get("progress_reward", float(successes[env_id]))
                )
                progress_stages[env_id] = str(
                    result.get(
                        "progress_stage", "success" if successes[env_id] else "none"
                    )
                )
                progress_delta = float(
                    result.get(
                        "progress_reward_delta",
                        1.0 if successes[env_id] else 0.0,
                    )
                )
                primitive_rewards[env_id].append(
                    progress_delta
                    if phase == "train" and settings.reward_mode == "task_progress"
                    else (1.0 if successes[env_id] else 0.0)
                )
                if progress_delta > 0.0:
                    progress_events[env_id].append(
                        {
                            "environment_step": completed_environment_steps[env_id],
                            "stage": progress_stages[env_id],
                            "reward": progress_rewards[env_id],
                            "delta": progress_delta,
                            "predicates": dict(result.get("progress_predicates", {})),
                        }
                    )

        for env_id in active_ids:
            action = by_env[env_id].action
            mask = masks[env_id] + [False] * (
                settings.execution_horizon - len(masks[env_id])
            )
            rewards = primitive_rewards[env_id] + [0.0] * (
                settings.execution_horizon - len(primitive_rewards[env_id])
            )
            trajectories[env_id].actions.append(action)
            record_robocasa_transition(
                trajectory=trajectories[env_id],
                action=action,
                reward=float(sum(rewards)),
                info={
                    "primitive_rewards": rewards,
                    "primitive_loss_mask": mask,
                    "executed_actions": executed[env_id],
                    "env_steps": completed_environment_steps[env_id],
                    "task_id": task_id,
                },
                policy_step=policy_step,
            )
            completed_policy_steps[env_id] = policy_step + 1
            observation = observations.get(env_id)
            if observation is not None:
                trajectories[env_id].observations.append(
                    record_robocasa_observation(
                        observation=observation,
                        step=policy_step + 1,
                    )
                )

        if in_shared_prefix:
            prefix_chunks_executed = policy_step + 1

    if shared_prefix_chunks and branch_state_hash is None:
        raise RuntimeError(
            "RoboCasa shared prefix ended before a trainable suffix could branch"
        )

    for env_id, trajectory in enumerate(trajectories):
        training_reward = (
            progress_rewards[env_id]
            if phase == "train" and settings.reward_mode == "task_progress"
            else float(successes[env_id])
        )
        trajectory.add_reward(
            config.reward.name,
            float(config.reward.scale) * training_reward,
            "env",
            step=completed_policy_steps[env_id],
            metadata={
                "success": successes[env_id],
                "task_id": task_id,
                "reward_mode": settings.reward_mode,
                "progress_stage": progress_stages[env_id],
                "progress_reward": progress_rewards[env_id],
            },
        )
        trajectory.metrics.update(
            {
                "success": successes[env_id],
                "terminated": successes[env_id],
                "truncated": truncated[env_id],
                "episode_steps": completed_policy_steps[env_id],
                "environment_steps": completed_environment_steps[env_id],
                "environment_return": float(successes[env_id]),
                "training_reward": training_reward,
                "progress_reward": progress_rewards[env_id],
                "progress_grasp_reached": progress_rewards[env_id]
                >= settings.progress_grasp_reward,
                "progress_placement_reached": progress_rewards[env_id]
                >= settings.progress_placement_reward,
                "progress_success_reached": successes[env_id],
            }
        )
        if shared_prefix_chunks:
            trajectory.metrics.update(
                {
                    "shared_prefix_action_chunks": shared_prefix_chunks,
                    "shared_prefix_environment_steps": min(
                        completed_environment_steps[env_id],
                        shared_prefix_chunks * settings.execution_horizon,
                    ),
                    "trainable_suffix_action_chunks": max(
                        0, completed_policy_steps[env_id] - shared_prefix_chunks
                    ),
                    "shared_prefix_branch_verified": branch_state_hash is not None,
                    "shared_prefix_completed_episode": prefix_completed_episode,
                }
            )
        trajectory.metadata["progress_reward_contract"] = {
            "mode": settings.reward_mode,
            "grasp_reward": settings.progress_grasp_reward,
            "placement_reward": settings.progress_placement_reward,
            "success_reward": 1.0,
            "official_success_unchanged": True,
        }
        trajectory.metadata["progress_events"] = progress_events[env_id]
        trajectory.metadata["group_rollout_seed"] = group_seed
        trajectory.metadata["shared_inference"] = bool(shared_prefix_chunks)
        trajectory.metadata["shared_prefix"] = {
            "configured_action_chunks": shared_prefix_chunks,
            "executed_action_chunks": prefix_chunks_executed,
            "leader_attempt_index": 0 if shared_prefix_chunks else None,
            "prefix_excluded_from_training": True,
            "episode_completed_during_prefix": prefix_completed_episode,
            "independent_suffix_available": bool(
                shared_prefix_chunks and not prefix_completed_episode
            ),
            "branch_state_sha256": branch_state_hash,
            "branch_observation_sha256": branch_observation_hash,
        }
        recorder = recorders.get(env_id)
        if recorder is not None:
            trajectory.media.extend(recorder.finalize(trajectory))
        trajectory.finish()
    return trajectories


def _shared_prefix_prediction(
    prediction: LeRobotActionPrediction,
    *,
    leader_env_id: int,
    prefix_chunk_index: int,
) -> LeRobotActionPrediction:
    """Clone one leader chunk while making its non-trainable status explicit."""

    metadata = {
        key: value
        for key, value in prediction.action.metadata.items()
        if key != TRANSIENT_FLOW_SDE_ROLLOUT_KEY
    }
    metadata.update(
        {
            FLOW_SDE_REPLAY_SELECTED_KEY: False,
            "shared_prefix": True,
            "shared_prefix_leader_env_id": int(leader_env_id),
            "shared_prefix_chunk_index": int(prefix_chunk_index),
        }
    )
    action = prediction.action.model_copy(update={"metadata": metadata}, deep=False)
    predicted_chunk = (
        None
        if prediction.predicted_action_chunk is None
        else np.asarray(prediction.predicted_action_chunk).copy()
    )
    return LeRobotActionPrediction(
        native_action=np.asarray(prediction.native_action).copy(),
        action=action,
        predicted_action_chunk=predicted_chunk,
        execution_horizon=prediction.execution_horizon,
    )


def _group_policy_seed(contexts: tuple[RolloutContext, ...]) -> int:
    digest = hashlib.sha256()
    for context in contexts:
        digest.update(str(int(context.policy_seed)).encode("ascii"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest()[:8], "big") % (2**31 - 1)


def _video_recorders(
    *,
    config: Any,
    contexts: tuple[RolloutContext, ...],
    trajectories: list[EmbodiedTrajectory],
    observations: dict[int, Any],
    phase: str,
) -> dict[int, RolloutVideoRecorder]:
    limit = (
        int(config.observability.videos_per_update)
        if phase == "train"
        else int(config.observability.videos_per_evaluation)
    )
    recorders = {}
    for env_id, (context, trajectory) in enumerate(
        zip(contexts, trajectories, strict=True)
    ):
        slot = (
            context.group_index * int(config.algorithm.group_size)
            + context.attempt_index
            if phase == "train"
            else context.group_index
        )
        selected = slot < limit
        trajectory.metadata["video_capture_selected"] = selected
        if not selected:
            continue
        recorder = RolloutVideoRecorder(
            Path(config.storage.output_dir) / "media" / phase,
            filename_prefix=(
                f"{phase}-u{context.update:04d}-g{context.group_index:04d}"
                f"-a{context.attempt_index:03d}-{task_safe(trajectory.task)}"
            ),
            fps=5,
            max_frames=180,
        )
        recorder.capture_frame(
            observations[env_id]["video.ego_view_pad_res256_freq20"], step=0
        )
        recorders[env_id] = recorder
    return recorders


def task_safe(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value)
