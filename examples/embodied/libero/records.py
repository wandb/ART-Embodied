"""Scenario schedules and trajectory records for the LIBERO example."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from art_embodied.experiment import EmbodiedScenario
from art_embodied.trajectories import Action, EmbodiedTrajectory, Observation
from art_embodied.utils import make_json_safe

from .settings import LiberoSettings
from .state_manifest import load_state_manifest


def build_train_scenarios(config: Any) -> list[EmbodiedScenario]:
    """Materialize the configured deterministic RLinf v0.1 reset stream."""

    settings = LiberoSettings.from_config(config)
    task_languages = _task_languages(settings)
    groups_per_update = int(config.rollout.groups_per_update) * int(
        config.rollout.epochs_per_update
    )
    scenarios = []
    for sequence_index in range(int(config.training.updates) * groups_per_update):
        if settings.init_state_selection == "rlinf_v01_ordered_reset":
            task_id, trial_id = rlinf_v01_task_and_trial(
                sequence_index,
                settings.reset_schedule,
            )
        elif settings.init_state_selection == "rlinf_v01_random_reset":
            task_id, trial_id = rlinf_v01_random_task_and_trial(
                sequence_index,
                settings.reset_schedule,
            )
        elif settings.init_state_selection == "partitioned_random_reset":
            assert settings.training_trial_ids is not None
            task_id, trial_id = partitioned_random_task_and_trial(
                sequence_index,
                settings.reset_schedule,
                task_ids=settings.training_task_ids or settings.task_ids,
                trial_ids=settings.training_trial_ids,
            )
        elif settings.init_state_selection == "balanced_task_random_reset":
            assert settings.training_trial_ids is not None
            task_id, trial_id = balanced_task_random_trial(
                sequence_index,
                settings.reset_schedule,
                task_ids=settings.training_task_ids or settings.task_ids,
                trial_ids=settings.training_trial_ids,
            )
        else:  # LiberoSettings validates this before scenario construction.
            raise ValueError(
                "Unsupported LIBERO init_state_selection: "
                f"{settings.init_state_selection!r}"
            )
        scenarios.append(
            EmbodiedScenario(
                id=(
                    f"{settings.suite_name}/train/{sequence_index:06d}/"
                    f"task-{task_id}/trial-{trial_id}"
                ),
                task=task_languages[task_id],
                payload={
                    "task_id": task_id,
                    "reset_options": {"trial_id": trial_id},
                },
            )
        )
    return scenarios


def build_evaluation_scenarios(config: Any) -> list[EmbodiedScenario]:
    """Materialize the task-major matrix selected by the evaluation protocol."""

    settings = LiberoSettings.from_config(config)
    task_languages = _task_languages(settings)
    if settings.evaluation_state_manifest is not None:
        manifest = load_state_manifest(
            settings.evaluation_state_manifest,
            expected_suite_name=settings.suite_name,
            expected_simulator_compatibility=settings.simulator_compatibility,
        )
        scenarios = []
        for entry in manifest.entries:
            if entry.task_id not in task_languages:
                continue
            scenarios.append(
                EmbodiedScenario(
                    id=entry.id,
                    task=task_languages[entry.task_id],
                    payload={
                        "task_id": entry.task_id,
                        "reset_options": {
                            "manifest_state_key": entry.state_key,
                        },
                        "state_sha256": entry.state_sha256,
                        "generation_seed": entry.generation_seed,
                    },
                )
            )
        if not scenarios:
            raise ValueError(
                "Evaluation state manifest has no entries for configured task_ids: "
                f"{list(settings.task_ids)}"
            )
        return scenarios
    return [
        EmbodiedScenario(
            id=(f"{settings.suite_name}/eval/task-{task_id:02d}/trial-{trial_id:02d}"),
            task=task_languages[task_id],
            payload={
                "task_id": task_id,
                "reset_options": {"trial_id": trial_id},
            },
        )
        for task_id in settings.task_ids
        for trial_id in settings.evaluation_trial_ids
    ]


def record_libero_observation(
    *,
    observation: Mapping[str, Any],
    step: int,
) -> Observation:
    value = {str(key): np.asarray(item) for key, item in observation.items()}
    return Observation(
        step=int(step),
        kind="image",
        value=value,
        metadata={
            "framework": "libero",
            "fields": {
                key: {"shape": list(item.shape), "dtype": str(item.dtype)}
                for key, item in value.items()
            },
        },
    )


def record_libero_transition(
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
        raise ValueError("LIBERO primitive reward and loss masks must align")
    executed_count = int(sum(loss_mask))
    # Execution is not eligibility: an explicit fallback can move the robot
    # without being a sampled policy action that should receive task credit.
    if action.metadata.get("primitive_loss_mask_sum") == 0:
        loss_mask = [False] * len(loss_mask)
    action.metadata.update(
        {
            "primitive_rewards": rewards,
            "primitive_loss_mask": loss_mask,
            "primitive_loss_mask_sum": int(sum(loss_mask)),
            "executed_action_vectors": make_json_safe(info["executed_actions"]),
            "environment_chunk_reward": float(reward),
        }
    )
    primitive_observations = info.get("primitive_observations_before")
    if primitive_observations is not None:
        if (
            not isinstance(primitive_observations, list)
            or len(primitive_observations) != executed_count
        ):
            raise ValueError(
                "Captured primitive observations must align with executed actions"
            )
        # Keep NumPy arrays intact. Converting images to JSON lists here would
        # multiply memory use before the streaming dataset writer consumes them.
        action.metadata["primitive_observations_before"] = primitive_observations
    first_executed_step = max(0, int(info["env_steps"]) - executed_count)
    for chunk_index, (value, keep) in enumerate(zip(rewards, loss_mask, strict=True)):
        trajectory.add_reward(
            "libero_primitive_reward",
            value,
            "env",
            step=first_executed_step + min(chunk_index + 1, executed_count),
            metadata={
                "policy_step": int(policy_step),
                "chunk_index": chunk_index,
                "primitive_loss_mask": keep,
            },
            update_total=False,
        )


def rlinf_v01_task_and_trial(
    sequence_index: int,
    schedule: Mapping[str, int],
) -> tuple[int, int]:
    """Reproduce RLinf v0.1's ordered 500-state, 8-process stream."""

    total_states = int(schedule["total_reset_states"])
    process_count = int(schedule["total_num_processes"])
    groups_per_process = int(schedule["groups_per_process_per_rollout_epoch"])
    seed = int(schedule["numpy_seed"])
    valid_size = total_states - total_states % process_count
    row_size = valid_size // process_count
    groups_per_epoch = process_count * groups_per_process
    global_epoch, group_in_epoch = divmod(int(sequence_index), groups_per_epoch)
    process_index, process_offset = divmod(group_in_epoch, groups_per_process)
    epochs_per_shuffle = row_size // groups_per_process
    shuffle_cycle, epoch_in_shuffle = divmod(global_epoch, epochs_per_shuffle)
    generator = np.random.default_rng(seed=seed)
    matrix = None
    for _ in range(shuffle_cycle + 1):
        reset_ids = np.arange(total_states)
        generator.shuffle(reset_ids)
        matrix = reset_ids[:valid_size].reshape(process_count, -1)
    assert matrix is not None
    row_offset = epoch_in_shuffle * groups_per_process + process_offset
    reset_id = int(matrix[process_index, row_offset])
    return _reset_id_to_task_trial(reset_id, total_states=total_states)


def rlinf_v01_random_task_and_trial(
    sequence_index: int,
    schedule: Mapping[str, int],
) -> tuple[int, int]:
    """Reproduce RLinf v0.1's rank-local random PI reset stream.

    Each environment rank owns a NumPy generator seeded by ``base_seed + rank``
    and draws one reset ID per local group at the start of each rollout epoch.
    RLinf calls ``update_reset_state_ids`` at the end of every epoch. Sampling
    is with replacement, exactly as ``LiberoEnv._get_random_reset_state_ids``.
    """

    total_states = int(schedule["total_reset_states"])
    process_count = int(schedule["total_num_processes"])
    groups_per_process = int(schedule["groups_per_process_per_rollout_epoch"])
    base_seed = int(schedule["numpy_seed"])
    groups_per_epoch = process_count * groups_per_process
    global_epoch, group_in_epoch = divmod(int(sequence_index), groups_per_epoch)
    process_index, process_offset = divmod(group_in_epoch, groups_per_process)
    generator = np.random.default_rng(seed=base_seed + process_index)
    draws = generator.integers(
        low=0,
        high=total_states,
        size=(global_epoch + 1) * groups_per_process,
    )
    reset_id = int(draws[global_epoch * groups_per_process + process_offset])
    return _reset_id_to_task_trial(reset_id, total_states=total_states)


def partitioned_random_task_and_trial(
    sequence_index: int,
    schedule: Mapping[str, int],
    *,
    task_ids: tuple[int, ...],
    trial_ids: tuple[int, ...],
) -> tuple[int, int]:
    """Sample reproducibly from an explicit task/trial Cartesian product.

    The rank-local RNG geometry matches the RLinf random reset stream, while
    the candidate table makes train/evaluation state separation enforceable.
    """

    process_count = int(schedule["total_num_processes"])
    groups_per_process = int(schedule["groups_per_process_per_rollout_epoch"])
    base_seed = int(schedule["numpy_seed"])
    groups_per_epoch = process_count * groups_per_process
    global_epoch, group_in_epoch = divmod(int(sequence_index), groups_per_epoch)
    process_index, process_offset = divmod(group_in_epoch, groups_per_process)
    candidates = tuple(
        (int(task_id), int(trial_id)) for task_id in task_ids for trial_id in trial_ids
    )
    if not candidates:
        raise ValueError("partitioned random reset candidate table cannot be empty")
    generator = np.random.default_rng(seed=base_seed + process_index)
    draws = generator.integers(
        low=0,
        high=len(candidates),
        size=(global_epoch + 1) * groups_per_process,
    )
    candidate_index = int(draws[global_epoch * groups_per_process + process_offset])
    return candidates[candidate_index]


def balanced_task_random_trial(
    sequence_index: int,
    schedule: Mapping[str, int],
    *,
    task_ids: tuple[int, ...],
    trial_ids: tuple[int, ...],
) -> tuple[int, int]:
    """Visit every task once per panel while sampling task-local train trials."""

    if not task_ids or not trial_ids:
        raise ValueError("Balanced LIBERO reset requires tasks and trials")
    panel_index, task_offset = divmod(int(sequence_index), len(task_ids))
    task_id = int(task_ids[task_offset])
    generator = np.random.default_rng(seed=int(schedule["numpy_seed"]) + task_id)
    draws = generator.integers(0, len(trial_ids), size=panel_index + 1)
    return task_id, int(trial_ids[int(draws[-1])])


def _reset_id_to_task_trial(reset_id: int, *, total_states: int) -> tuple[int, int]:
    trials_per_task, remainder = divmod(total_states, 10)
    if remainder:
        raise ValueError("RLinf reset states must divide evenly across ten tasks")
    return reset_id // trials_per_task, reset_id % trials_per_task


def _task_languages(settings: LiberoSettings) -> dict[int, str]:
    # hf-libero bundles its task metadata, but upstream LIBERO otherwise asks
    # an interactive first-run question. Public scenario builders must remain
    # deterministic and non-interactive in clean jobs and test environments.
    from .environment import prepare_libero_runtime_paths

    prepare_libero_runtime_paths()
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[settings.suite_name]()
    return {
        task_id: str(suite.get_task(task_id).language) for task_id in settings.task_ids
    }
