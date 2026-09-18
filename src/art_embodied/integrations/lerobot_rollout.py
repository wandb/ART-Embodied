"""LeRobot-first episode rollout with trajectory and video recording."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
import inspect
from pathlib import Path
import threading
from typing import Any, Literal

from art_embodied.config import EmbodiedExperimentConfig, LookaheadPreviewConfig
from art_embodied.experiment import EmbodiedScenario, RolloutContext
from art_embodied.lookahead import ActionChunkLookaheadPreview, unused_chunk_length
from art_embodied.media import LookaheadPreviewRecorder, RolloutVideoRecorder
from art_embodied.trajectories import EmbodiedTrajectory, Observation
from art_embodied.utils import make_json_safe

from .lerobot import LeRobotPolicyAdapterProtocol

EnvironmentFactory = Callable[[EmbodiedScenario, RolloutContext], Any]
PolicyAdapterFactory = Callable[
    [EmbodiedScenario, RolloutContext],
    LeRobotPolicyAdapterProtocol,
]
SuccessFunction = Callable[..., bool]
ObservationRecorder = Callable[..., Observation]
TransitionRecorder = Callable[..., None]


class LeRobotEpisodeRollout:
    """Turn a native LeRobot/Gymnasium episode into an ART trajectory.

    Environment and adapter factories are explicit because action-chunk queues,
    simulator state, and some policy preprocessors are episode-local. Returning
    isolated instances lets ``EmbodiedExperiment`` safely collect rollouts in
    parallel without ART reimplementing LeRobot's vector-environment contract.
    """

    def __init__(
        self,
        *,
        environment_factory: EnvironmentFactory,
        policy_adapter_factory: PolicyAdapterFactory,
        training_policy: Any | None,
        output_dir: str | Path,
        max_policy_steps: int,
        reward_name: str,
        reward_scale: float,
        terminal_reward_only: bool,
        success_key: str,
        expected_action_kind: str,
        video_limit_per_update: int,
        group_size: int,
        video_fps: int,
        video_max_frames: int,
        lookahead_config: LookaheadPreviewConfig | None = None,
        phase: Literal["train", "eval"] = "train",
        success_fn: SuccessFunction | None = None,
        observation_recorder: ObservationRecorder | None = None,
        transition_recorder: TransitionRecorder | None = None,
        close_environment: bool = True,
    ) -> None:
        if max_policy_steps <= 0:
            raise ValueError("max_policy_steps must be positive")
        if video_max_frames <= 0:
            raise ValueError("video_max_frames must be positive")
        if not success_key:
            raise ValueError("success_key must be non-empty")
        if not expected_action_kind:
            raise ValueError("expected_action_kind must be non-empty")
        if video_limit_per_update < 0:
            raise ValueError("video_limit_per_update cannot be negative")
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        self.environment_factory = environment_factory
        self.policy_adapter_factory = policy_adapter_factory
        self.training_policy = training_policy
        self.output_dir = Path(output_dir)
        self.max_policy_steps = int(max_policy_steps)
        self.reward_name = reward_name
        self.reward_scale = float(reward_scale)
        self.terminal_reward_only = bool(terminal_reward_only)
        self.success_key = success_key
        self.expected_action_kind = expected_action_kind
        self.video_limit_per_update = int(video_limit_per_update)
        self.group_size = int(group_size)
        self.video_fps = int(video_fps)
        self.video_max_frames = int(video_max_frames)
        self.lookahead_config = lookahead_config or LookaheadPreviewConfig()
        self.phase = phase
        self.success_fn = success_fn
        self.observation_recorder = (
            observation_recorder or summarize_lerobot_observation
        )
        self.transition_recorder = transition_recorder
        self.close_environment = bool(close_environment)
        self._policy_call_lock = threading.RLock()
        self._active_components_lock = threading.Lock()
        self._active_component_ids: set[int] = set()

    @classmethod
    def from_config(
        cls,
        config: EmbodiedExperimentConfig,
        *,
        environment_factory: EnvironmentFactory,
        policy_adapter_factory: PolicyAdapterFactory,
        training_policy: Any | None = None,
        phase: Literal["train", "eval"] = "train",
        success_fn: SuccessFunction | None = None,
        observation_recorder: ObservationRecorder | None = None,
        transition_recorder: TransitionRecorder | None = None,
    ) -> "LeRobotEpisodeRollout":
        """Build a rollout solely from visible YAML settings and factories."""

        success_key = config.environment.kwargs.get("success_key")
        if not isinstance(success_key, str) or not success_key:
            raise ValueError(
                "environment.kwargs.success_key must name the info field "
                "used for episode success"
            )
        video_limit = (
            config.observability.videos_per_update
            if phase == "train"
            else config.observability.videos_per_evaluation
        )
        return cls(
            environment_factory=environment_factory,
            policy_adapter_factory=policy_adapter_factory,
            training_policy=training_policy,
            output_dir=config.storage.output_dir / "videos" / phase,
            max_policy_steps=config.rollout.max_policy_steps,
            reward_name=config.reward.name,
            reward_scale=config.reward.scale,
            terminal_reward_only=config.reward.terminal_only,
            success_key=success_key,
            expected_action_kind=config.rollout.action_payload.kind,
            video_limit_per_update=video_limit,
            group_size=config.algorithm.group_size,
            video_fps=config.observability.video_fps,
            video_max_frames=config.rollout.max_episode_steps,
            lookahead_config=config.observability.lookahead_preview,
            phase=phase,
            success_fn=success_fn,
            observation_recorder=observation_recorder,
            transition_recorder=transition_recorder,
        )

    async def prepare_update(self, *, update: int) -> None:
        """Synchronize rollout replicas with the policy trained by ART."""

        if self.training_policy is None:
            return
        prepare = getattr(self.policy_adapter_factory, "prepare_update", None)
        if callable(prepare):
            result = prepare(policy=self.training_policy, update=update)
            if inspect.isawaitable(result):
                await result
            return
        if (
            getattr(self.policy_adapter_factory, "training_policy", None)
            is not self.training_policy
        ):
            raise RuntimeError(
                "LeRobot policy_adapter_factory does not prove rollout-policy "
                "synchronization. Provide prepare_update(policy=..., update=...) "
                "or bind training_policy to the policy trained by ART."
            )

    async def __call__(
        self,
        scenario: EmbodiedScenario,
        context: RolloutContext,
    ) -> EmbodiedTrajectory:
        return await asyncio.to_thread(self.run, scenario, context)

    def run(
        self,
        scenario: EmbodiedScenario,
        context: RolloutContext,
    ) -> EmbodiedTrajectory:
        env = self.environment_factory(scenario, context)
        leased_component_ids: tuple[int, ...] = ()
        trajectory = EmbodiedTrajectory(
            task=scenario.task,
            metadata={
                "framework": "lerobot",
                "phase": self.phase,
                "scenario_id": scenario.id,
                "environment_seed": context.environment_seed,
                "policy_seed": context.policy_seed,
                "config_fingerprint": context.config_fingerprint,
            },
        )
        recorder = None
        lookahead_recorder = None
        preview_env = None
        video_capture_selected = self._should_record_video(context)
        trajectory.metadata["video_capture_selected"] = video_capture_selected
        if video_capture_selected:
            recorder = RolloutVideoRecorder(
                self.output_dir,
                filename_prefix=(
                    f"{self.phase}-u{context.update:04d}-g{context.group_index:04d}"
                    f"-a{context.attempt_index:03d}-{scenario.id}"
                ),
                fps=self.video_fps,
                max_frames=self.video_max_frames,
            )
        lookahead_capture_selected = self._should_record_lookahead(context)
        trajectory.metadata["lookahead_preview_selected"] = lookahead_capture_selected
        if lookahead_capture_selected:
            settings = self.lookahead_config
            lookahead_recorder = LookaheadPreviewRecorder(
                self.output_dir.parent / "lookahead" / self.phase,
                filename_prefix=(
                    f"lookahead-{self.phase}-u{context.update:04d}"
                    f"-g{context.group_index:04d}-a{context.attempt_index:03d}"
                    f"-{scenario.id}"
                ),
                fps=self.video_fps,
                max_frames=self.max_policy_steps,
                max_panels=settings.max_panels,
            )
        try:
            adapter = self.policy_adapter_factory(scenario, context)
            leased_component_ids = self._acquire_adapter(adapter)
            reset_options = scenario.payload.get("reset_options")
            if reset_options is not None and not isinstance(reset_options, dict):
                raise TypeError("scenario.payload.reset_options must be a mapping")
            observation, reset_info = env.reset(
                seed=context.environment_seed,
                options=reset_options,
            )
            if lookahead_recorder is not None:
                preview_env = self.environment_factory(scenario, context)
                if not isinstance(preview_env, ActionChunkLookaheadPreview):
                    raise TypeError(
                        "lookahead_preview requires the environment adapter to "
                        "implement ActionChunkLookaheadPreview"
                    )
                preview_env.reset(
                    seed=context.environment_seed,
                    options=reset_options,
                )
            if not isinstance(observation, Mapping):
                raise TypeError(
                    "LeRobot environment reset observation must be a mapping"
                )
            with self._policy_access():
                adapter.reset(seed=context.policy_seed)
            trajectory.metadata["reset_info"] = make_json_safe(reset_info)
            trajectory.observations.append(
                self.observation_recorder(observation=observation, step=0)
            )
            if recorder is not None:
                recorder.capture(env, step=0)

            environment_return = 0.0
            success = False
            terminated = False
            truncated = False
            completed_steps = 0
            for step in range(self.max_policy_steps):
                with self._policy_access():
                    prediction = adapter.predict(
                        observation,
                        task=scenario.task,
                        step=step,
                        seed=_episode_step_seed(context.policy_seed, step),
                    )
                if prediction.action.kind != self.expected_action_kind:
                    raise RuntimeError(
                        "LeRobot adapter recorded an action representation that "
                        "does not match rollout.action_payload.kind: "
                        f"recorded={prediction.action.kind!r}, "
                        f"expected={self.expected_action_kind!r}"
                    )
                if (
                    lookahead_recorder is not None
                    and preview_env is not None
                    and prediction.predicted_action_chunk is not None
                    and prediction.execution_horizon is not None
                    and unused_chunk_length(
                        prediction.predicted_action_chunk,
                        prediction.execution_horizon,
                    )
                    > 0
                ):
                    try:
                        settings = self.lookahead_config
                        future_frames = preview_env.preview_action_chunk(
                            prediction.predicted_action_chunk,
                            source_environment=env,
                            execution_horizon=prediction.execution_horizon,
                            frame_stride=settings.future_stride,
                            max_frames=settings.max_future_frames,
                        )
                        lookahead_recorder.capture_lookahead(
                            env.render(),
                            list(future_frames),
                            step=step,
                            execution_horizon=prediction.execution_horizon,
                            model_horizon=len(prediction.predicted_action_chunk),
                        )
                    except Exception as exc:  # Media must not corrupt the episode.
                        lookahead_recorder.capture_error(exc, step=step)
                trajectory.actions.append(prediction.action)
                transition = env.step(prediction.native_action)
                if not isinstance(transition, tuple) or len(transition) != 5:
                    raise TypeError(
                        "LeRobot environment step must return the Gymnasium "
                        "(observation, reward, terminated, truncated, info) tuple"
                    )
                observation, reward, terminated, truncated, info = transition
                if not isinstance(observation, Mapping):
                    raise TypeError("LeRobot environment observation must be a mapping")
                if not isinstance(info, Mapping):
                    raise TypeError("LeRobot environment info must be a mapping")
                scalar_reward = float(reward)
                environment_return += scalar_reward
                completed_steps = step + 1
                if self.transition_recorder is not None:
                    self.transition_recorder(
                        trajectory=trajectory,
                        action=prediction.action,
                        observation=observation,
                        reward=scalar_reward,
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        info=info,
                        policy_step=step,
                    )
                success = success or self._is_success(
                    info=info,
                    reward=scalar_reward,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
                if not self.terminal_reward_only:
                    trajectory.add_reward(
                        self.reward_name,
                        scalar_reward * self.reward_scale,
                        "env",
                        step=step,
                        metadata={"environment_reward": scalar_reward},
                    )
                trajectory.observations.append(
                    self.observation_recorder(observation=observation, step=step + 1)
                )
                if recorder is not None:
                    recorder.capture(env, step=step + 1)
                if terminated or truncated:
                    break

            if self.terminal_reward_only:
                trajectory.add_reward(
                    self.reward_name,
                    self.reward_scale if success else 0.0,
                    "env",
                    step=completed_steps,
                    metadata={"success": success},
                )
            trajectory.metrics.update(
                {
                    "success": success,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "episode_steps": completed_steps,
                    "environment_return": environment_return,
                }
            )
            if recorder is not None:
                trajectory.media.extend(recorder.finalize(trajectory))
            if lookahead_recorder is not None:
                trajectory.media.extend(lookahead_recorder.finalize(trajectory))
            return trajectory.finish()
        finally:
            if leased_component_ids:
                self._release_adapter(leased_component_ids)
            if self.close_environment:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
                if preview_env is not None:
                    preview_close = getattr(preview_env, "close", None)
                    if callable(preview_close):
                        preview_close()

    def _should_record_video(self, context: RolloutContext) -> bool:
        """Select a bounded, deterministic subset before rendering starts."""

        if self.video_limit_per_update == 0:
            return False
        if self.phase == "eval":
            slot = context.group_index
        else:
            slot = context.group_index * self.group_size + context.attempt_index
        return slot < self.video_limit_per_update

    def _should_record_lookahead(self, context: RolloutContext) -> bool:
        settings = self.lookahead_config
        if not settings.enabled:
            return False
        limit = (
            settings.videos_per_update
            if self.phase == "train"
            else settings.videos_per_evaluation
        )
        if self.phase == "eval":
            slot = context.group_index
        else:
            slot = context.group_index * self.group_size + context.attempt_index
        return slot < limit

    def _policy_access(self):
        # LeRobot policies and processor pipelines can contain action queues and
        # use process-global RNG state. Threaded calls must therefore remain
        # serialized; scalable rollout workers should isolate process/device
        # state rather than disabling this guard.
        return self._policy_call_lock

    def _acquire_adapter(
        self,
        adapter: LeRobotPolicyAdapterProtocol,
    ) -> tuple[int, ...]:
        component_ids = adapter.stateful_component_ids()
        with self._active_components_lock:
            shared = self._active_component_ids.intersection(component_ids)
            if shared:
                raise RuntimeError(
                    "Concurrent LeRobot episodes cannot share a policy, "
                    "preprocessor, or postprocessor instance. Return "
                    "episode-isolated state from policy_adapter_factory, or "
                    "run rollouts in isolated worker processes."
                )
            self._active_component_ids.update(component_ids)
        return component_ids

    def _release_adapter(self, component_ids: tuple[int, ...]) -> None:
        with self._active_components_lock:
            self._active_component_ids.difference_update(component_ids)

    def _is_success(
        self,
        *,
        info: Mapping[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> bool:
        if self.success_fn is not None:
            return bool(
                self.success_fn(
                    info=info,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                )
            )
        return bool(info.get(self.success_key, False))


def summarize_lerobot_observation(
    *,
    observation: Mapping[str, Any],
    step: int,
) -> Observation:
    """Record shape/dtype metadata without embedding image arrays in traces."""

    fields: dict[str, Any] = {}
    for key, value in observation.items():
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        summary: dict[str, Any] = {"type": type(value).__name__}
        if shape is not None:
            summary["shape"] = [int(dimension) for dimension in shape]
        if dtype is not None:
            summary["dtype"] = str(dtype)
        if shape is None and isinstance(value, str | int | float | bool):
            summary["value"] = value
        fields[str(key)] = summary
    return Observation(
        step=step,
        kind="custom",
        metadata={"framework": "lerobot", "fields": fields},
    )


def _episode_step_seed(policy_seed: int, step: int) -> int:
    return (int(policy_seed) + int(step) + 1) % (2**31 - 1)
