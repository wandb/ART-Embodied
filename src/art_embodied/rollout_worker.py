"""Child process serving one native embodied rollout actor."""

from __future__ import annotations

import argparse
import asyncio
import faulthandler
import importlib
import inspect
import json
from pathlib import Path
import pickle
import sys
import traceback
from typing import Any

from .backends.flow_sde import (
    FLOW_SDE_REPLAY_SELECTED_KEY,
    TRANSIENT_FLOW_SDE_ROLLOUT_KEY,
)
from .compatibility import require_compatible_worker_runtime
from .config import EmbodiedExperimentConfig
from .experiment import EmbodiedScenario, RolloutContext
from .inference_transport import BatchedPolicyClient
from .policies.flow_policy import FlowSDERollout
from .rollout_process import RolloutActorProcessContext
from .trajectories import EmbodiedTrajectory
from .utils import write_json_atomic
from .worker_config import discard_coordinator_resume

TRAINABLE_ACTION_SELECTED_KEY = "_art_embodied_trainable_action_selected"


def _worker_config(spec: dict[str, Any]) -> EmbodiedExperimentConfig:
    """Validate the experiment contract after removing parent-only state."""

    raw = discard_coordinator_resume(spec["config"])
    return EmbodiedExperimentConfig.model_validate(raw)


def _discard_unrequired_training_observation_values(
    trajectory: EmbodiedTrajectory,
    *,
    phase: str,
    config: EmbodiedExperimentConfig,
) -> None:
    """Drop raw observations after rollout when training cannot consume them.

    Action metadata carries the replay evidence required by the built-in VLA
    backends. Keeping every raw camera observation as well can substantially
    increase coordinator memory for large rollout batches. Evaluation and
    observation-dependent custom backends retain the complete trajectory.
    """

    if (
        phase != "train"
        or config.storage.retain_rollout_payloads
        or config.rollout.action_payload.require_observation
    ):
        return
    trajectory.discard_observation_values()


def _select_training_action_payloads(
    trajectory: EmbodiedTrajectory,
    *,
    phase: str,
    config: EmbodiedExperimentConfig,
) -> None:
    """Bound replay evidence without changing the executed trajectory.

    Flow policies may replan after every primitive action, while autoregressive
    policies may emit hundreds of tokens per decision. ``uniform_grid`` keeps
    deterministic coverage over the complete trajectory while preserving all
    lightweight action records for tracing. The backend ignores explicitly
    unselected rows. When raw payload retention is disabled, transient Flow-SDE
    tensors are removed before process handoff.
    """

    payload = config.rollout.action_payload
    limit = payload.max_trainable_actions_per_trajectory
    if phase != "train" or payload.trainable_action_selection == "all" or limit is None:
        return
    invalid_token_actions = [
        index
        for index, action in enumerate(trajectory.actions)
        if action.kind == "token"
        and action.metadata.get(
            "action_decode_valid", action.metadata.get("action_grammar_valid")
        )
        is False
        and not action.metadata.get("terminate_episode", False)
    ]
    for index in invalid_token_actions:
        action = trajectory.actions[index]
        action.metadata[TRAINABLE_ACTION_SELECTED_KEY] = False
        action.metadata["primitive_loss_mask_sum"] = 0

    eligible = [
        index
        for index, action in enumerate(trajectory.actions)
        if TRANSIENT_FLOW_SDE_ROLLOUT_KEY in action.metadata
        or (
            action.kind == "token"
            and (
                action.metadata.get(
                    "action_decode_valid", action.metadata.get("action_grammar_valid")
                )
                is not False
                or action.metadata.get("terminate_episode", False)
            )
        )
    ]
    selected_positions = _uniform_grid_positions(len(eligible), limit)
    selected = {eligible[position] for position in selected_positions}
    for index in eligible:
        action = trajectory.actions[index]
        keep = index in selected
        if action.kind == "token":
            action.metadata[TRAINABLE_ACTION_SELECTED_KEY] = keep
            if not keep:
                action.metadata["primitive_loss_mask_sum"] = 0
        else:
            action.metadata[FLOW_SDE_REPLAY_SELECTED_KEY] = keep
        if (
            not keep
            and action.kind != "token"
            and not config.storage.retain_rollout_payloads
        ):
            rollout = action.metadata[TRANSIENT_FLOW_SDE_ROLLOUT_KEY]
            if "primitive_loss_mask_sum" not in action.metadata and isinstance(
                rollout, FlowSDERollout
            ):
                action.metadata["primitive_loss_mask_sum"] = int(
                    rollout.transition.old_logprobs.shape[1]
                )
            action.metadata.pop(TRANSIENT_FLOW_SDE_ROLLOUT_KEY, None)
    if (
        config.rollout.action_payload.require_observation
        and not config.storage.retain_rollout_payloads
    ):
        selected_steps = {int(trajectory.actions[index].step) for index in selected}
        for observation in trajectory.observations:
            if int(observation.step) not in selected_steps:
                observation.value = None
    trajectory.metadata["training_action_selection"] = {
        "strategy": payload.trainable_action_selection,
        "eligible": len(eligible),
        "selected": len(selected),
        "selected_action_indices": sorted(selected),
    }


def _uniform_grid_positions(length: int, limit: int) -> list[int]:
    """Choose stable, endpoint-inclusive positions across a sequence."""

    if length <= 0 or limit <= 0:
        return []
    if length <= limit:
        return list(range(length))
    if limit == 1:
        return [length // 2]
    last = length - 1
    return [round(position * last / (limit - 1)) for position in range(limit)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve-spec", type=Path, required=True)
    return parser.parse_args()


def _resolve_factory(reference: str) -> Any:
    module_name, separator, qualname = reference.partition(":")
    if not separator or not module_name or not qualname:
        raise ValueError("actor_factory must use the 'module:callable' form")
    value: Any = importlib.import_module(module_name)
    for component in qualname.split("."):
        value = getattr(value, component)
    if not callable(value):
        raise TypeError(f"Rollout actor factory is not callable: {reference}")
    return value


async def _await_if_needed(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _serve(spec: dict[str, Any]) -> None:
    config = _worker_config(spec)
    process_context = RolloutActorProcessContext(
        worker_index=int(spec["worker_index"]),
        configured_device=str(spec["configured_device"]),
        local_device=str(spec["local_device"]),
    )
    ready_path = Path(spec["ready_path"])
    policy_client = (
        BatchedPolicyClient(str(spec["inference_socket"]))
        if spec.get("inference_socket")
        else None
    )
    try:
        factory = _resolve_factory(str(spec["actor_factory"]))
        actor = await _await_if_needed(factory(config=config, context=process_context))
        prepare = getattr(actor, "prepare_update", None)
        offload = getattr(actor, "offload", None)
        restore = getattr(actor, "restore", None)
        rollout = getattr(actor, "rollout", None)
        rollout_group = getattr(actor, "rollout_group", None)
        if not callable(rollout):
            raise TypeError(
                "Rollout actor must expose rollout(scenario, context, phase=...)"
            )
        if policy_client is None and not callable(prepare):
            raise TypeError(
                "An embedded-policy rollout actor must expose "
                "prepare_update(update=..., policy_snapshot=...)"
            )
    except Exception as exc:
        write_json_atomic(
            ready_path,
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            indent=2,
            sort_keys=True,
        )
        return

    write_json_atomic(
        ready_path,
        {"ok": True, "worker_index": process_context.worker_index},
    )
    try:
        while line := await asyncio.to_thread(sys.stdin.readline):
            command = json.loads(line)
            if command.get("op") == "shutdown":
                break
            result_path = Path(command["result_path"])
            try:
                if command.get("op") == "prepare":
                    if callable(prepare):
                        await _await_if_needed(
                            prepare(
                                update=int(command["update"]),
                                policy_snapshot=Path(command["policy_snapshot"]),
                            )
                        )
                elif command.get("op") == "offload":
                    if not callable(offload):
                        raise RuntimeError("Rollout actor does not support CPU offload")
                    await _await_if_needed(offload())
                elif command.get("op") == "restore":
                    if not callable(restore):
                        raise RuntimeError("Rollout actor does not support GPU restore")
                    await _await_if_needed(restore())
                elif command.get("op") == "rollout":
                    with Path(command["request_path"]).open("rb") as handle:
                        request = pickle.load(handle)
                    scenario = EmbodiedScenario.model_validate(request["scenario"])
                    context = RolloutContext(**request["context"])
                    rollout_kwargs = {"phase": str(request["phase"])}
                    if policy_client is not None:
                        rollout_kwargs["policy_client"] = policy_client
                    trajectory = await _await_if_needed(
                        rollout(scenario, context, **rollout_kwargs)
                    )
                    if not isinstance(trajectory, EmbodiedTrajectory):
                        raise TypeError(
                            "Rollout actor returned "
                            f"{type(trajectory).__name__}, expected EmbodiedTrajectory"
                        )
                    _discard_unrequired_training_observation_values(
                        trajectory,
                        phase=str(request["phase"]),
                        config=config,
                    )
                    _select_training_action_payloads(
                        trajectory,
                        phase=str(request["phase"]),
                        config=config,
                    )
                    with Path(command["trajectory_path"]).open("wb") as handle:
                        pickle.dump(
                            trajectory,
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                elif command.get("op") == "rollout_group":
                    with Path(command["request_path"]).open("rb") as handle:
                        request = pickle.load(handle)
                    scenario = EmbodiedScenario.model_validate(request["scenario"])
                    contexts = tuple(
                        RolloutContext(**item) for item in request["contexts"]
                    )
                    rollout_kwargs = {"phase": str(request["phase"])}
                    if policy_client is not None:
                        rollout_kwargs["policy_client"] = policy_client
                    if callable(rollout_group):
                        trajectories = await _await_if_needed(
                            rollout_group(scenario, contexts, **rollout_kwargs)
                        )
                    else:
                        trajectories = [
                            await _await_if_needed(
                                rollout(scenario, context, **rollout_kwargs)
                            )
                            for context in contexts
                        ]
                    trajectories = list(trajectories)
                    if len(trajectories) != len(contexts) or any(
                        not isinstance(item, EmbodiedTrajectory)
                        for item in trajectories
                    ):
                        raise TypeError(
                            "Rollout actor returned an invalid trajectory group"
                        )
                    for trajectory in trajectories:
                        _discard_unrequired_training_observation_values(
                            trajectory,
                            phase=str(request["phase"]),
                            config=config,
                        )
                        _select_training_action_payloads(
                            trajectory,
                            phase=str(request["phase"]),
                            config=config,
                        )
                    with Path(command["trajectories_path"]).open("wb") as handle:
                        pickle.dump(
                            trajectories,
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                else:
                    raise ValueError(
                        f"Unknown rollout worker command: {command.get('op')!r}"
                    )
                result = {"ok": True, "worker_index": process_context.worker_index}
            except Exception as exc:
                result = {
                    "ok": False,
                    "worker_index": process_context.worker_index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            write_json_atomic(result_path, result, indent=2, sort_keys=True)
    finally:
        if policy_client is not None:
            await policy_client.close()
        close = getattr(actor, "close", None)
        if callable(close):
            await _await_if_needed(close())


def main() -> None:
    """Run a rollout actor process from a trusted serialized specification."""

    # Native simulator, EGL, and CUDA failures may terminate the process
    # without raising Python exceptions. Preserve every thread's Python stack
    # in the actor stderr that the coordinator archives on failure.
    faulthandler.enable(all_threads=True)
    require_compatible_worker_runtime()
    args = _parse_args()
    spec = json.loads(args.serve_spec.read_text(encoding="utf-8"))
    asyncio.run(_serve(spec))


if __name__ == "__main__":
    main()
