"""Group-native LIBERO rollout and process-actor component factory."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import sys
from typing import Any, Mapping

from art_embodied.experiment import EmbodiedScenario, RolloutContext
from art_embodied.integrations.lerobot_process import LeRobotProcessComponents
from art_embodied.lookahead import unused_chunk_length
from art_embodied.media import LookaheadPreviewRecorder, RolloutVideoRecorder
from art_embodied.policies.factory import make_policy
from art_embodied.trajectories import Action, EmbodiedTrajectory
from art_embodied.utils import make_json_safe

from .environment import LiberoChunkEnvironment, LiberoTaskCatalog
from .policy import OpenVLAAdapter, _seed_policy, prepare_recorded_openvla_action
from .records import record_libero_observation, record_libero_transition
from .settings import LiberoSettings


def create_components(*, config: Any, context: Any) -> LeRobotProcessComponents:
    """Create LIBERO components for embedded or shared-model inference."""

    settings = LiberoSettings.from_config(config)
    local_config = config.model_copy(
        update={
            "policy": config.policy.model_copy(update={"device": context.local_device})
        }
    )
    shared_inference = (
        config.runtime.rollout_execution.inference_mode == "batched_server"
    )
    policy = None if shared_inference else make_policy(local_config)
    catalog = LiberoTaskCatalog(settings)

    def load_policy_snapshot(*, update: int, policy_snapshot: Path) -> None:
        if policy is None:
            return
        policy.load_checkpoint({"path": str(policy_snapshot)})
        policy.rollout_update = int(update)

    def prepare_phase(phase: str) -> None:
        if policy is None:
            return
        if phase == "train":
            generation = config.policy.rollout_generation
        elif phase == "eval":
            generation = config.policy.evaluation_generation
        else:
            raise ValueError(f"Unsupported rollout phase: {phase!r}")
        policy.set_generation(
            do_sample=generation.do_sample,
            temperature=generation.temperature,
        )

    def policy_adapter_factory(
        _scenario: EmbodiedScenario,
        _rollout_context: RolloutContext,
    ) -> OpenVLAAdapter:
        if policy is None:
            raise RuntimeError(
                "Shared-model LIBERO inference is available only through "
                "the group rollout path"
            )
        return OpenVLAAdapter(policy, settings)

    async def group_rollout(
        *,
        scenario: EmbodiedScenario,
        contexts: tuple[RolloutContext, ...],
        phase: str,
        policy_client: Any | None,
    ) -> list[EmbodiedTrajectory]:
        if shared_inference != (policy_client is not None):
            raise RuntimeError(
                "LIBERO inference mode and rollout policy client do not agree"
            )
        return await rollout_libero_group(
            config=local_config,
            policy=policy,
            policy_client=policy_client,
            catalog=catalog,
            settings=settings,
            scenario=scenario,
            contexts=contexts,
            phase=phase,
        )

    return LeRobotProcessComponents(
        environment_factory=catalog.make_environment,
        policy_adapter_factory=policy_adapter_factory,
        load_policy_snapshot=load_policy_snapshot,
        prepare_phase=prepare_phase,
        observation_recorder=record_libero_observation,
        transition_recorder=record_libero_transition,
        group_rollout=group_rollout,
    )


async def rollout_libero_group(
    *,
    config: Any,
    policy: Any | None,
    policy_client: Any | None = None,
    catalog: LiberoTaskCatalog,
    settings: LiberoSettings,
    scenario: EmbodiedScenario,
    contexts: tuple[RolloutContext, ...],
    phase: str,
    embedded_batch_predictor: Callable[..., Any] | None = None,
    embedded_batch_reset: Callable[[int], None] | None = None,
) -> list[EmbodiedTrajectory]:
    """Collect one counterfactual group with batched policy inference."""

    if not contexts:
        raise ValueError("LIBERO group rollout requires at least one context")
    if phase not in {"train", "eval"}:
        raise ValueError(f"Unsupported rollout phase: {phase!r}")
    expected_contexts = int(config.algorithm.group_size) if phase == "train" else 1
    if len(contexts) != expected_contexts:
        raise ValueError(
            "LIBERO rollout received the wrong context count: "
            f"phase={phase!r}, received={len(contexts)}, expected={expected_contexts}"
        )
    group_indices = {context.group_index for context in contexts}
    environment_seeds = {context.environment_seed for context in contexts}
    if len(group_indices) != 1 or len(environment_seeds) != 1:
        raise ValueError(
            "Counterfactual group attempts must share group index and environment seed"
        )

    environments: list[LiberoChunkEnvironment] = []
    trajectories = [
        EmbodiedTrajectory(
            task=scenario.task,
            metadata={
                "framework": "lerobot",
                "phase": phase,
                "scenario_id": scenario.id,
                "environment_seed": context.environment_seed,
                "policy_seed": context.policy_seed,
                "config_fingerprint": context.config_fingerprint,
                "vectorized_group_rollout": len(contexts) > 1,
                "vectorized_group_size": len(contexts),
            },
        )
        for context in contexts
    ]
    recorders = [
        _group_video_recorder(
            config=config,
            scenario=scenario,
            context=context,
            phase=phase,
        )
        for context in contexts
    ]
    lookahead_recorders = [
        _group_lookahead_recorder(
            config=config,
            scenario=scenario,
            context=context,
            phase=phase,
        )
        for context in contexts
    ]
    preview_environments: list[LiberoChunkEnvironment | None] = [None] * len(contexts)
    observations: list[Mapping[str, Any] | None] = [None] * len(contexts)
    active = [True] * len(contexts)
    successes = [False] * len(contexts)
    terminated = [False] * len(contexts)
    truncated = [False] * len(contexts)
    environment_returns = [0.0] * len(contexts)
    completed_steps = [0] * len(contexts)
    group_seed = (
        int(contexts[0].policy_seed)
        if len(contexts) == 1
        else _group_rollout_seed(contexts)
    )
    rng_stream_id = _group_rng_stream_id(
        scenario=scenario,
        contexts=contexts,
        phase=phase,
    )
    rng_stream_started = False
    try:
        for context in contexts:
            environments.append(catalog.make_environment(scenario, context))
        for index, (env, context, trajectory, recorder) in enumerate(
            zip(environments, contexts, trajectories, recorders, strict=True)
        ):
            reset_options = scenario.payload.get("reset_options")
            observation, reset_info = env.reset(
                seed=context.environment_seed,
                options=reset_options,
            )
            observations[index] = observation
            trajectory.metadata["reset_info"] = make_json_safe(reset_info)
            trajectory.observations.append(
                record_libero_observation(observation=observation, step=0)
            )
            trajectory.metadata["video_capture_selected"] = recorder is not None
            if recorder is not None:
                recorder.capture(env, step=0)
            if lookahead_recorders[index] is not None:
                preview = catalog.make_environment(scenario, context)
                preview.reset(
                    seed=context.environment_seed,
                    options=reset_options,
                )
                preview_environments[index] = preview
                trajectory.metadata["lookahead_preview_selected"] = True

        if policy_client is None:
            if embedded_batch_predictor is None:
                if policy is None:
                    raise RuntimeError("Embedded LIBERO rollout requires a policy")
                _seed_policy(group_seed)
            elif embedded_batch_reset is None:
                raise RuntimeError(
                    "Embedded batch prediction requires an explicit seeded reset"
                )
            else:
                embedded_batch_reset(group_seed)
        for trajectory in trajectories:
            trajectory.metadata["group_rollout_seed"] = group_seed
            trajectory.metadata["shared_inference"] = policy_client is not None
        for policy_step in range(int(config.rollout.max_policy_steps)):
            active_indices = [index for index, keep in enumerate(active) if keep]
            if not active_indices:
                break
            raw_batch_observations = [observations[index] for index in active_indices]
            if not all(
                isinstance(observation, Mapping)
                for observation in raw_batch_observations
            ):
                raise TypeError("Active LIBERO observations must be mappings")
            batch_observations = [
                record_libero_observation(
                    observation=observations[index],
                    step=policy_step,
                )
                for index in active_indices
            ]
            batch_contexts = [
                {
                    "scenario": {"task": scenario.task},
                    "task": scenario.task,
                    "step": policy_step,
                }
                for _index in active_indices
            ]
            if policy_client is None:
                native_actions = None
                predicted_action_chunks = None
                prediction_execution_horizons = None
                if embedded_batch_predictor is None:
                    assert policy is not None
                    recorded_actions = policy.act_batch(
                        batch_observations,
                        batch_contexts,
                    )
                else:
                    predictions = list(
                        embedded_batch_predictor(
                            raw_batch_observations,
                            tasks=[scenario.task] * len(batch_observations),
                            step=policy_step,
                        )
                    )
                    recorded_actions = [item.action for item in predictions]
                    native_actions = [item.native_action for item in predictions]
                    predicted_action_chunks = [
                        getattr(item, "predicted_action_chunk", None)
                        for item in predictions
                    ]
                    prediction_execution_horizons = [
                        getattr(item, "execution_horizon", None) for item in predictions
                    ]
                shared_inference_metadata: dict[str, Any] = {}
            else:
                response = await policy_client.predict(
                    {
                        "op": "predict_group",
                        "observations": batch_observations,
                        "raw_observations": raw_batch_observations,
                        "contexts": batch_contexts,
                        "phase": phase,
                        "rng_stream_id": rng_stream_id,
                        "rng_seed": group_seed,
                        "reset_rng_stream": policy_step == 0,
                    }
                )
                rng_stream_started = True
                if not isinstance(response, dict) or not isinstance(
                    response.get("actions"), list
                ):
                    raise RuntimeError(
                        "Shared policy inference returned an invalid group response"
                    )
                recorded_actions = [
                    Action.model_validate(item) for item in response["actions"]
                ]
                response_native_actions = response.get("native_actions")
                if response_native_actions is not None and not isinstance(
                    response_native_actions, list
                ):
                    raise RuntimeError(
                        "Shared policy inference returned invalid native actions"
                    )
                native_actions = response_native_actions
                predicted_action_chunks = response.get("predicted_action_chunks")
                prediction_execution_horizons = response.get(
                    "prediction_execution_horizons"
                )
                shared_inference_metadata = {
                    "shared_inference_model_batch_size": int(
                        response.get("model_batch_size", len(recorded_actions))
                    ),
                    "shared_inference_server_request_batch_size": int(
                        response.get("server_request_batch_size", 1)
                    ),
                }
            if len(recorded_actions) != len(active_indices):
                raise RuntimeError("Policy batch returned the wrong number of actions")
            if native_actions is not None and len(native_actions) != len(
                active_indices
            ):
                raise RuntimeError(
                    "Policy batch returned the wrong number of native actions"
                )
            for label, values in (
                ("predicted_action_chunks", predicted_action_chunks),
                ("prediction_execution_horizons", prediction_execution_horizons),
            ):
                if values is not None and (
                    not isinstance(values, list) or len(values) != len(active_indices)
                ):
                    raise RuntimeError(
                        f"Policy batch returned invalid {label}: expected one per action"
                    )

            for batch_index, env_index in enumerate(active_indices):
                recorded = recorded_actions[batch_index]
                if recorded.kind != config.rollout.action_payload.kind:
                    raise RuntimeError(
                        "OpenVLA group action kind does not match YAML action payload"
                    )
                recorded.metadata["group_rollout_seed"] = group_seed
                recorded.metadata.update(shared_inference_metadata)
                if recorded.metadata.get("terminate_episode") is True:
                    # A rejected policy decision is a terminal outcome, not an
                    # environment step. Keep it trainable with its sampled tokens.
                    trajectory = trajectories[env_index]
                    trajectory.actions.append(recorded)
                    trajectory.observations.append(
                        record_libero_observation(
                            observation=observations[env_index],
                            step=policy_step + 1,
                        )
                    )
                    trajectory.metadata["policy_termination_reason"] = (
                        recorded.metadata.get(
                            "termination_reason", "policy_rejected_action"
                        )
                    )
                    active[env_index] = False
                    terminated[env_index] = True
                    continue
                if native_actions is None:
                    native_action = prepare_recorded_openvla_action(
                        recorded,
                        settings=settings,
                        step=policy_step,
                    )
                else:
                    native_action = native_actions[batch_index]
                    recorded.metadata.update(
                        {
                            "policy_step": policy_step,
                            "observation_index": policy_step,
                            "processed_action_chunk": make_json_safe(native_action),
                        }
                    )
                trajectory = trajectories[env_index]
                lookahead_recorder = lookahead_recorders[env_index]
                preview_environment = preview_environments[env_index]
                predicted_action_chunk = (
                    predicted_action_chunks[batch_index]
                    if predicted_action_chunks is not None
                    else None
                )
                prediction_execution_horizon = (
                    prediction_execution_horizons[batch_index]
                    if prediction_execution_horizons is not None
                    else None
                )
                if (
                    lookahead_recorder is not None
                    and preview_environment is not None
                    and predicted_action_chunk is not None
                    and prediction_execution_horizon is not None
                    and unused_chunk_length(
                        predicted_action_chunk,
                        int(prediction_execution_horizon),
                    )
                    > 0
                ):
                    try:
                        lookahead_config = config.observability.lookahead_preview
                        future_frames = preview_environment.preview_action_chunk(
                            predicted_action_chunk,
                            source_environment=environments[env_index],
                            execution_horizon=int(prediction_execution_horizon),
                            frame_stride=int(lookahead_config.future_stride),
                            max_frames=int(lookahead_config.max_future_frames),
                        )
                        lookahead_recorder.capture_lookahead(
                            environments[env_index].render(),
                            list(future_frames),
                            step=policy_step,
                            execution_horizon=int(prediction_execution_horizon),
                            model_horizon=len(predicted_action_chunk),
                        )
                    except Exception as exc:  # Media must not corrupt RL evidence.
                        lookahead_recorder.capture_error(exc, step=policy_step)
                trajectory.actions.append(recorded)
                (
                    observation,
                    reward,
                    terminated_value,
                    truncated_value,
                    info,
                ) = environments[env_index].step(native_action)
                observations[env_index] = observation
                scalar_reward = float(reward)
                environment_returns[env_index] += scalar_reward
                completed_steps[env_index] = policy_step + 1
                record_libero_transition(
                    trajectory=trajectory,
                    action=recorded,
                    reward=scalar_reward,
                    info=info,
                    policy_step=policy_step,
                )
                success = bool(info.get("success", False))
                successes[env_index] = successes[env_index] or success
                terminated[env_index] = bool(terminated_value)
                truncated[env_index] = bool(truncated_value)
                if not config.reward.terminal_only:
                    trajectory.add_reward(
                        config.reward.name,
                        scalar_reward * float(config.reward.scale),
                        "env",
                        step=policy_step,
                        metadata={"environment_reward": scalar_reward},
                    )
                trajectory.observations.append(
                    record_libero_observation(
                        observation=observation,
                        step=policy_step + 1,
                    )
                )
                recorder = recorders[env_index]
                if recorder is not None:
                    recorder.capture(environments[env_index], step=policy_step + 1)
                if terminated_value or truncated_value:
                    active[env_index] = False

        for index, trajectory in enumerate(trajectories):
            if config.reward.terminal_only:
                trajectory.add_reward(
                    config.reward.name,
                    float(config.reward.scale) if successes[index] else 0.0,
                    "env",
                    step=completed_steps[index],
                    metadata={"success": successes[index]},
                )
            trajectory.metrics.update(
                {
                    "success": successes[index],
                    "terminated": terminated[index],
                    "truncated": truncated[index],
                    "episode_steps": completed_steps[index],
                    "environment_return": environment_returns[index],
                }
            )
            recorder = recorders[index]
            if recorder is not None:
                trajectory.media.extend(recorder.finalize(trajectory))
            lookahead_recorder = lookahead_recorders[index]
            if lookahead_recorder is not None:
                trajectory.media.extend(lookahead_recorder.finalize(trajectory))
            trajectory.finish()
        return trajectories
    finally:
        rollout_failed = sys.exc_info()[0] is not None
        first_close_error: Exception | None = None
        if policy_client is not None and rng_stream_started:
            try:
                await policy_client.predict(
                    {
                        "op": "release_rng_stream",
                        "rng_stream_id": rng_stream_id,
                    }
                )
            except Exception as exc:
                if first_close_error is None:
                    first_close_error = exc
        for env in environments:
            try:
                env.close()
            except Exception as exc:
                if first_close_error is None:
                    first_close_error = exc
        for env in preview_environments:
            if env is None:
                continue
            try:
                env.close()
            except Exception as exc:
                if first_close_error is None:
                    first_close_error = exc
        if first_close_error is not None and not rollout_failed:
            raise first_close_error


def _group_video_recorder(
    *,
    config: Any,
    scenario: EmbodiedScenario,
    context: RolloutContext,
    phase: str,
) -> RolloutVideoRecorder | None:
    limit = (
        int(config.observability.videos_per_update)
        if phase == "train"
        else int(config.observability.videos_per_evaluation)
    )
    slot = (
        context.group_index * int(config.algorithm.group_size) + context.attempt_index
        if phase == "train"
        else context.group_index
    )
    if slot >= limit:
        return None
    return RolloutVideoRecorder(
        Path(config.storage.output_dir) / "videos" / phase,
        filename_prefix=(
            f"{phase}-u{context.update:04d}-g{context.group_index:04d}"
            f"-a{context.attempt_index:03d}-{scenario.id}"
        ),
        fps=int(config.observability.video_fps),
        max_frames=int(config.rollout.max_episode_steps),
    )


def _group_lookahead_recorder(
    *,
    config: Any,
    scenario: EmbodiedScenario,
    context: RolloutContext,
    phase: str,
) -> LookaheadPreviewRecorder | None:
    settings = config.observability.lookahead_preview
    if not settings.enabled:
        return None
    limit = (
        int(settings.videos_per_update)
        if phase == "train"
        else int(settings.videos_per_evaluation)
    )
    slot = (
        context.group_index * int(config.algorithm.group_size) + context.attempt_index
        if phase == "train"
        else context.group_index
    )
    if slot >= limit:
        return None
    return LookaheadPreviewRecorder(
        Path(config.storage.output_dir) / "videos" / "lookahead" / phase,
        filename_prefix=(
            f"lookahead-{phase}-u{context.update:04d}-g{context.group_index:04d}"
            f"-a{context.attempt_index:03d}-{scenario.id}"
        ),
        fps=int(config.observability.video_fps),
        max_frames=int(config.rollout.max_policy_steps),
        max_panels=int(settings.max_panels),
    )


def _group_rollout_seed(contexts: tuple[RolloutContext, ...]) -> int:
    return (
        sum(
            (index + 1) * int(context.policy_seed)
            for index, context in enumerate(contexts)
        )
        + 1
    ) % (2**31 - 1)


def _group_rng_stream_id(
    *,
    scenario: EmbodiedScenario,
    contexts: tuple[RolloutContext, ...],
    phase: str,
) -> str:
    first = contexts[0]
    return (
        f"{phase}:u{first.update}:g{first.group_index}:"
        f"e{first.environment_seed}:{scenario.id}"
    )
