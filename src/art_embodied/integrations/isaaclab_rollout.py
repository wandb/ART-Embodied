"""Isaac Lab group collection using ART's existing trajectory/media contract."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import inspect
from pathlib import Path
import threading
from typing import Any

from art_embodied.experiment import EmbodiedScenario, RolloutContext
from art_embodied.media import RolloutVideoRecorder
from art_embodied.trajectories import EmbodiedTrajectory
from art_embodied.utils import make_json_safe

from .isaaclab import IsaacLabChunkEnvironment, isaaclab_reset_fingerprints
from .lerobot import LeRobotActionPrediction
from .lerobot_rollout import _episode_step_seed, summarize_lerobot_observation


class IsaacLabGroupRollout:
    """Collect one GRPO group in a persistent vector simulator.

    ``policy`` supplies ``reset(seeds=...)``, ``predict_batch(observations,
    task=..., step=..., seeds=...)`` and ``prepare_update(update=...)``. Predictions
    use the existing LeRobotActionPrediction records, including native likelihoods.
    Only active rows are passed to prediction; seeds remain tied to attempt IDs.
    The batched policy must be stateless between chunks (no hidden action queue).

    ``reset_fn(env, scenario, seed)`` restores the task's shared initial conditions.
    ``reset_state_fn(env)`` returns batched tensors covering scene state relative
    to environment origins, commands and any other randomized task conditions.
    These are task-specific: scene state alone cannot certify a GRPO reset.

    The simulator stays on the creating thread. Run this collector in an isolated
    rollout worker when using an event loop shared with training or logging.
    Logging remains in EmbodiedExperiment; this class never calls wandb.log.
    """

    supports_group_rollout = True

    def __init__(
        self,
        *,
        environment: IsaacLabChunkEnvironment,
        policy: Any,
        reset_fn: Callable,
        reset_state_fn: Callable,
        observation_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        output_dir: str | Path,
        expected_action_kind: str,
        videos_per_update: int,
        video_fps: int = 10,
        frame_fn: Callable[[Mapping[str, Any]], Any] | None = None,
        observation_recorder: Callable = summarize_lerobot_observation,
        transition_recorder: Callable | None = None,
    ) -> None:
        if videos_per_update < 0 or video_fps <= 0:
            raise ValueError("Invalid video budget or fps")
        if videos_per_update and frame_fn is None:
            raise ValueError("Video capture requires frame_fn on policy observations")
        if not expected_action_kind:
            raise ValueError("expected_action_kind cannot be empty")
        for method in ("reset", "predict_batch", "prepare_update"):
            if not callable(getattr(policy, method, None)):
                raise TypeError(f"Isaac Lab batch policy must implement {method}")
        self.environment = environment
        self.policy = policy
        self.reset_fn = reset_fn
        self.reset_state_fn = reset_state_fn
        self.observation_fn = observation_fn
        self.output_dir = Path(output_dir)
        self.expected_action_kind = expected_action_kind
        self.videos_per_update = videos_per_update
        self.video_fps = video_fps
        self.frame_fn = frame_fn
        self.observation_recorder = observation_recorder
        self.transition_recorder = transition_recorder
        self._thread_id = threading.get_ident()
        self._prepared_update: int | None = None

    async def prepare_update(self, *, update: int) -> None:
        self._check_thread()
        self._prepared_update = None
        result = self.policy.prepare_update(update=update)
        if inspect.isawaitable(result):
            await result
        self._prepared_update = update

    async def __call__(self, scenario: EmbodiedScenario, context: RolloutContext):
        if self.environment.num_envs != 1:
            raise ValueError(
                "Use rollout_group for a multi-environment Isaac Lab worker"
            )
        return (await self.rollout_group(scenario, [context]))[0]

    async def rollout_group(
        self, scenario: EmbodiedScenario, contexts: Sequence[RolloutContext]
    ) -> Sequence[EmbodiedTrajectory]:
        import torch

        self._check_thread()
        if len(contexts) != self.environment.num_envs:
            raise ValueError("Group size must match Isaac Lab num_envs")
        first = contexts[0]
        shared = (
            first.update,
            first.group_index,
            first.environment_seed,
            first.config_fingerprint,
        )
        if any(
            (c.update, c.group_index, c.environment_seed, c.config_fingerprint)
            != shared
            for c in contexts
        ):
            raise ValueError(
                "Isaac Lab group contexts must share update, group and reset"
            )
        if {c.attempt_index for c in contexts} != set(range(len(contexts))):
            raise ValueError("Group attempt indices must cover each row exactly once")
        if len({c.policy_seed for c in contexts}) != len(contexts):
            raise ValueError("Group policy seeds must be distinct")
        if self._prepared_update != first.update:
            raise RuntimeError(
                "Synchronize the policy with prepare_update before collection"
            )

        observation, reset_info = self.environment.reset(
            seed=first.environment_seed,
            reset_fn=lambda env, seed: self.reset_fn(env, scenario, seed),
        )
        comparison_state = self.reset_state_fn(self.environment.env)
        fingerprints = isaaclab_reset_fingerprints(
            comparison_state, num_envs=len(contexts)
        )
        if len(set(fingerprints)) != 1:
            raise RuntimeError(
                "Isaac Lab group initial conditions differ across attempts"
            )
        self.policy.reset(seeds=[c.policy_seed for c in contexts])
        trajectories = [
            EmbodiedTrajectory(
                task=scenario.task,
                initial_policy_version=first.update,
                final_policy_version=first.update,
                metadata={
                    "framework": "lerobot",
                    "simulator": "isaaclab",
                    "phase": "train",
                    "scenario_id": scenario.id,
                    "environment_seed": c.environment_seed,
                    "policy_seed": c.policy_seed,
                    "config_fingerprint": c.config_fingerprint,
                    "reset_info": {
                        **reset_info,
                        "comparison_state_sha256": fingerprint,
                    },
                },
            )
            for c, fingerprint in zip(contexts, fingerprints, strict=True)
        ]
        for row, trajectory in enumerate(trajectories):
            trajectory.metadata["reset_state"] = make_json_safe(
                _row(comparison_state, row)
            )
        recorders = []
        for trajectory, context in zip(trajectories, contexts, strict=True):
            selected = (
                context.group_index * len(contexts) + context.attempt_index
                < self.videos_per_update
            )
            trajectory.metadata["video_capture_selected"] = selected
            recorders.append(
                RolloutVideoRecorder(
                    self.output_dir,
                    filename_prefix=(
                        f"train-u{context.update:04d}-g{context.group_index:04d}"
                        f"-a{context.attempt_index:03d}"
                    ),
                    fps=self.video_fps,
                    max_frames=self.environment.max_environment_steps + 1,
                )
                if selected
                else None
            )

        def record(row: int, policy_step: int) -> Mapping[str, Any]:
            mapped = self.observation_fn(_row(observation, row))
            if not isinstance(mapped, Mapping):
                raise TypeError(
                    "observation_fn must return a policy observation mapping"
                )
            trajectories[row].observations.append(
                self.observation_recorder(observation=mapped, step=policy_step)
            )
            recorder = recorders[row]
            if recorder is not None:
                recorder.capture_frame(self.frame_fn(mapped), step=policy_step)
            return mapped

        mapped_observations = [record(row, 0) for row in range(len(contexts))]
        active = list(range(len(contexts)))
        returns = [0.0] * len(contexts)
        policy_step = 0
        while active:
            predictions = list(
                self.policy.predict_batch(
                    [mapped_observations[row] for row in active],
                    task=scenario.task,
                    step=policy_step,
                    seeds=[
                        _episode_step_seed(contexts[row].policy_seed, policy_step)
                        for row in active
                    ],
                )
            )
            if len(predictions) != len(active):
                raise ValueError("Batch policy returned the wrong prediction count")
            commands = torch.zeros(
                (
                    len(contexts),
                    self.environment.execution_horizon,
                    self.environment.action_dim,
                ),
                device=self.environment.device,
            )
            for row, prediction in zip(active, predictions, strict=True):
                if not isinstance(prediction, LeRobotActionPrediction):
                    raise TypeError(
                        "Batch policy must return LeRobotActionPrediction records"
                    )
                if prediction.action.kind != self.expected_action_kind:
                    raise ValueError(
                        "Policy action kind differs from the training contract"
                    )
                if prediction.action.step != policy_step:
                    raise ValueError(
                        "Policy action step differs from the current chunk"
                    )
                if prediction.execution_horizon not in (
                    None,
                    self.environment.execution_horizon,
                ):
                    raise ValueError(
                        "Policy execution horizon differs from simulator horizon"
                    )
                native = torch.as_tensor(
                    prediction.native_action, device=commands.device
                )
                if native.shape != commands[row].shape:
                    raise ValueError(
                        "Policy native_action must be the exact execution prefix"
                    )
                commands[row] = native
            observation, rewards, terminated, truncated, info = self.environment.step(
                commands
            )
            policy_step += 1
            next_active = []
            for row, prediction in zip(active, predictions, strict=True):
                trajectory = trajectories[row]
                trajectory.actions.append(prediction.action)
                row_info = _row(info, row)
                # Keep simulator execution evidence separate from sampled-policy
                # likelihoods, especially when termination cuts a chunk short.
                prediction.action.metadata["isaaclab_execution"] = make_json_safe(
                    row_info
                )
                returns[row] += float(rewards[row])
                mapped_observations[row] = record(row, policy_step)
                if self.transition_recorder is not None:
                    self.transition_recorder(
                        trajectory=trajectory,
                        action=prediction.action,
                        observation=mapped_observations[row],
                        reward=float(rewards[row]),
                        terminated=bool(terminated[row]),
                        truncated=bool(truncated[row]),
                        info=row_info,
                        policy_step=policy_step - 1,
                    )
                if bool(terminated[row] | truncated[row]):
                    success = bool(info["success"][row])
                    trajectory.add_reward(
                        "success", float(success), "env", step=policy_step
                    )
                    trajectory.metrics.update(
                        {
                            "success": success,
                            "terminated": bool(terminated[row]),
                            "truncated": bool(truncated[row]),
                            "episode_steps": policy_step,
                            "environment_steps": int(info["env_steps"][row]),
                            "environment_return": returns[row],
                        }
                    )
                    if recorders[row] is not None:
                        trajectory.media.extend(recorders[row].finalize(trajectory))
                    trajectory.finish()
                else:
                    next_active.append(row)
            active = next_active
        return trajectories

    def close(self) -> None:
        self._check_thread()
        self.environment.close()

    def _check_thread(self) -> None:
        if threading.get_ident() != self._thread_id:
            raise RuntimeError("Isaac Lab must run on its creating thread")


def _row(value: Any, index: int) -> Any:
    if isinstance(value, Mapping):
        return {key: _row(item, index) for key, item in value.items()}
    return value[index]
