"""Validated LIBERO experiment settings owned by the example integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

_LIBERO_TASKS_PER_SUITE = 10


@dataclass(frozen=True, slots=True)
class LiberoSettings:
    suite_name: str
    task_ids: tuple[int, ...]
    training_task_ids: tuple[int, ...] | None
    training_trial_ids: tuple[int, ...] | None
    evaluation_trial_ids: tuple[int, ...]
    evaluation_state_manifest: Path | None
    simulator_compatibility: str
    evaluation_protocol: str
    observation_height: int
    observation_width: int
    primary_image_key: str
    wrist_image_key: str
    rotate_images_180: bool
    wait_steps_after_reset: int
    action_chunk_size: int
    capture_primitive_observations: bool
    max_episode_steps: int
    stop_on_success: bool
    stop_on_done: bool
    normalize_gripper: bool
    binarize_gripper: bool
    invert_gripper: bool
    control_mode: str
    init_state_selection: str
    reset_gripper_open: bool
    reward_coefficient: float
    reset_schedule: Mapping[str, int]

    @classmethod
    def from_config(cls, config: Any) -> "LiberoSettings":
        if config.environment.type != "libero":
            raise ValueError("LIBERO components require environment.type='libero'")
        if config.reward.type != "environment":
            raise ValueError(
                "LIBERO positive-control components require reward.type='environment'"
            )
        if not config.reward.terminal_only:
            raise ValueError(
                "LIBERO positive-control components require reward.terminal_only=true"
            )
        if config.reward.kwargs:
            raise ValueError(
                "LIBERO positive-control components do not consume reward.kwargs; "
                "remove them or provide a custom components factory"
            )
        if config.environment.observation_processor:
            raise ValueError(
                "LIBERO positive-control components do not consume "
                "environment.observation_processor; remove it or provide a "
                "custom components factory"
            )
        if config.environment.action_processor:
            raise ValueError(
                "LIBERO positive-control components do not consume "
                "environment.action_processor; remove it or provide a custom "
                "components factory"
            )
        values = config.environment.kwargs
        required = {
            "task_ids",
            "evaluation_trial_ids",
            "simulator_compatibility",
            "observation_height",
            "observation_width",
            "primary_image_key",
            "wrist_image_key",
            "rotate_images_180",
            "wait_steps_after_reset",
            "action_chunk_size",
            "stop_on_success",
            "stop_on_done",
            "normalize_gripper",
            "binarize_gripper",
            "invert_gripper",
            "control_mode",
            "init_state_selection",
            "reset_schedule",
            "success_key",
        }
        missing = sorted(required.difference(values))
        if missing:
            raise ValueError(
                "LIBERO YAML must explicitly define environment.kwargs: "
                + ", ".join(missing)
            )
        if "reset_gripper_open" not in config.environment.reset:
            raise ValueError(
                "LIBERO YAML must explicitly define "
                "environment.reset.reset_gripper_open"
            )
        task_ids = tuple(int(value) for value in values["task_ids"])
        if not task_ids or len(set(task_ids)) != len(task_ids):
            raise ValueError("environment.kwargs.task_ids must be unique and non-empty")
        invalid_task_ids = sorted(
            task_id
            for task_id in task_ids
            if not 0 <= task_id < _LIBERO_TASKS_PER_SUITE
        )
        if invalid_task_ids:
            raise ValueError(
                "environment.kwargs.task_ids contains out-of-range tasks: "
                f"{invalid_task_ids}"
            )
        init_state_selection = str(values["init_state_selection"])
        if (
            init_state_selection
            in {"rlinf_v01_ordered_reset", "rlinf_v01_random_reset"}
            and task_ids != tuple(range(_LIBERO_TASKS_PER_SUITE))
        ):
            raise ValueError(
                "The RLinf v0.1 reset stream requires task_ids=[0, ..., 9]; "
                "use partitioned_random_reset or balanced_task_random_reset "
                "for a task subset"
            )
        evaluation_trial_ids = tuple(
            int(value) for value in values["evaluation_trial_ids"]
        )
        if not evaluation_trial_ids or len(set(evaluation_trial_ids)) != len(
            evaluation_trial_ids
        ):
            raise ValueError(
                "environment.kwargs.evaluation_trial_ids must be unique and non-empty"
            )
        training_trial_ids = (
            tuple(int(value) for value in values["training_trial_ids"])
            if values.get("training_trial_ids") is not None
            else None
        )
        if training_trial_ids is not None and (
            not training_trial_ids
            or len(set(training_trial_ids)) != len(training_trial_ids)
        ):
            raise ValueError(
                "environment.kwargs.training_trial_ids must be unique and non-empty"
            )
        training_task_ids = (
            tuple(int(value) for value in values["training_task_ids"])
            if values.get("training_task_ids") is not None
            else None
        )
        if training_task_ids is not None and (
            not training_task_ids
            or len(set(training_task_ids)) != len(training_task_ids)
        ):
            raise ValueError(
                "environment.kwargs.training_task_ids must be unique and non-empty"
            )
        simulator_compatibility = str(values["simulator_compatibility"])
        if simulator_compatibility != "rlinf_v01":
            raise ValueError(
                "The validated LIBERO runtime requires "
                "simulator_compatibility='rlinf_v01'"
            )
        # Existing OpenVLA recipes predate this field and are intentionally
        # pinned to their validated RLinf contract. New policy families must
        # name a different protocol explicitly rather than inheriting it.
        evaluation_protocol = str(values.get("evaluation_protocol", "rlinf_v01"))
        if evaluation_protocol == "rlinf_v01":
            expected_trials = tuple(range(40, 50))
        elif evaluation_protocol in {
            "lerobot_v060",
            "lerobot_v060_chunked",
            "lerobot_pi0_fast_v060",
        }:
            expected_trials = tuple(range(10))
        elif evaluation_protocol == "lerobot_pi0_fast_v060_all_official_states":
            expected_trials = tuple(range(50))
        else:
            raise ValueError(
                f"Unsupported LIBERO evaluation_protocol: {evaluation_protocol!r}"
            )
        if evaluation_trial_ids != expected_trials:
            raise ValueError(
                f"The {evaluation_protocol} evaluation requires "
                f"evaluation_trial_ids={list(expected_trials)}"
            )
        if str(values["success_key"]) != "success":
            raise ValueError(
                "LIBERO positive-control components require success_key='success'"
            )
        settings = cls(
            suite_name=str(config.environment.task),
            task_ids=task_ids,
            training_task_ids=training_task_ids,
            training_trial_ids=training_trial_ids,
            evaluation_trial_ids=evaluation_trial_ids,
            evaluation_state_manifest=(
                Path(str(values["evaluation_state_manifest"])).expanduser().resolve()
                if values.get("evaluation_state_manifest")
                else None
            ),
            simulator_compatibility=simulator_compatibility,
            evaluation_protocol=evaluation_protocol,
            observation_height=int(values["observation_height"]),
            observation_width=int(values["observation_width"]),
            primary_image_key=str(values["primary_image_key"]),
            wrist_image_key=str(values["wrist_image_key"]),
            rotate_images_180=bool(values["rotate_images_180"]),
            wait_steps_after_reset=int(values["wait_steps_after_reset"]),
            action_chunk_size=int(values["action_chunk_size"]),
            capture_primitive_observations=bool(
                values.get("capture_primitive_observations", False)
            ),
            max_episode_steps=int(config.rollout.max_episode_steps),
            stop_on_success=bool(values["stop_on_success"]),
            stop_on_done=bool(values["stop_on_done"]),
            normalize_gripper=bool(values["normalize_gripper"]),
            binarize_gripper=bool(values["binarize_gripper"]),
            invert_gripper=bool(values["invert_gripper"]),
            control_mode=str(values["control_mode"]),
            init_state_selection=init_state_selection,
            reset_gripper_open=bool(config.environment.reset["reset_gripper_open"]),
            reward_coefficient=float(config.reward.scale),
            reset_schedule={
                str(key): int(value)
                for key, value in dict(values["reset_schedule"]).items()
            },
        )
        if (
            min(
                settings.observation_height,
                settings.observation_width,
                settings.action_chunk_size,
                settings.max_episode_steps,
            )
            <= 0
        ):
            raise ValueError("LIBERO dimensions and horizons must be positive")
        if settings.wait_steps_after_reset < 0:
            raise ValueError("wait_steps_after_reset cannot be negative")
        if settings.evaluation_protocol in {
            "lerobot_v060",
            "lerobot_v060_chunked",
        }:
            _validate_lerobot_v060_contract(
                settings,
                require_single_action=(settings.evaluation_protocol == "lerobot_v060"),
            )
        elif settings.evaluation_protocol in {
            "lerobot_pi0_fast_v060",
            "lerobot_pi0_fast_v060_all_official_states",
        }:
            _validate_lerobot_pi0_fast_v060_contract(settings)
        if settings.init_state_selection not in {
            "rlinf_v01_ordered_reset",
            "rlinf_v01_random_reset",
            "partitioned_random_reset",
            "balanced_task_random_reset",
        }:
            raise ValueError(
                "LIBERO positive control requires init_state_selection to be "
                "'rlinf_v01_ordered_reset', 'rlinf_v01_random_reset', or "
                "'partitioned_random_reset'/'balanced_task_random_reset'"
            )
        required_schedule = {
            "total_reset_states",
            "total_num_processes",
            "groups_per_process_per_rollout_epoch",
            "rollout_epochs_per_update",
            "numpy_seed",
        }
        missing_schedule = sorted(required_schedule.difference(settings.reset_schedule))
        if missing_schedule:
            raise ValueError(
                "LIBERO reset_schedule is missing: " + ", ".join(missing_schedule)
            )
        if settings.init_state_selection in {
            "partitioned_random_reset",
            "balanced_task_random_reset",
        }:
            _validate_partitioned_reset_contract(settings)
        elif settings.training_trial_ids is not None:
            raise ValueError(
                "environment.kwargs.training_trial_ids is consumed only when "
                "init_state_selection='partitioned_random_reset' or "
                "'balanced_task_random_reset'"
            )
        elif settings.training_task_ids is not None:
            raise ValueError(
                "environment.kwargs.training_task_ids is consumed only when "
                "init_state_selection='partitioned_random_reset' or "
                "'balanced_task_random_reset'"
            )
        expected_workers = (
            settings.reset_schedule["total_num_processes"]
            * settings.reset_schedule["groups_per_process_per_rollout_epoch"]
        )
        if expected_workers != int(config.rollout.groups_per_update):
            raise ValueError(
                "RLinf reset_schedule process geometry must equal "
                "rollout.groups_per_update"
            )
        if settings.reset_schedule["rollout_epochs_per_update"] != int(
            config.rollout.epochs_per_update
        ):
            raise ValueError(
                "RLinf reset_schedule rollout_epochs_per_update must equal "
                "rollout.epochs_per_update"
            )
        schedule_action_chunk_size = getattr(
            config.training.schedule, "action_chunk_size", None
        )
        if schedule_action_chunk_size is not None and settings.action_chunk_size != int(
            schedule_action_chunk_size
        ):
            raise ValueError(
                "LIBERO action_chunk_size must equal training schedule action_chunk_size"
            )
        if settings.control_mode not in {
            "unchanged",
            "default",
            "relative",
            "absolute",
        }:
            raise ValueError(
                f"Unsupported LIBERO control_mode: {settings.control_mode}"
            )
        return settings


def _validate_partitioned_reset_contract(settings: LiberoSettings) -> None:
    """Ensure training and fixed evaluation cannot select the same reset state."""

    if settings.training_trial_ids is None:
        raise ValueError(
            "partitioned_random_reset requires environment.kwargs.training_trial_ids"
        )
    unknown_tasks = sorted(
        set(settings.training_task_ids or settings.task_ids).difference(
            settings.task_ids
        )
    )
    if unknown_tasks:
        raise ValueError(
            "environment.kwargs.training_task_ids contains tasks outside task_ids: "
            f"{unknown_tasks}"
        )
    trials_per_task, remainder = divmod(
        int(settings.reset_schedule["total_reset_states"]),
        _LIBERO_TASKS_PER_SUITE,
    )
    if remainder:
        raise ValueError(
            "partitioned_random_reset requires reset states to divide evenly "
            "across configured tasks"
        )
    invalid = sorted(
        trial_id
        for trial_id in settings.training_trial_ids
        if not 0 <= trial_id < trials_per_task
    )
    if invalid:
        raise ValueError(
            "environment.kwargs.training_trial_ids contains out-of-range trials: "
            f"{invalid}; valid range is [0, {trials_per_task - 1}]"
        )
    overlap = sorted(
        set(settings.training_trial_ids).intersection(settings.evaluation_trial_ids)
    )
    if overlap:
        raise ValueError(
            f"LIBERO training and evaluation trial partitions overlap: {overlap}"
        )


def _validate_lerobot_v060_contract(
    settings: LiberoSettings,
    *,
    require_single_action: bool,
) -> None:
    """Validate LeRobot 0.6 preprocessing and episode geometry.

    The exact protocol replans after every primitive action. The chunked
    protocol intentionally changes only that feedback interval while retaining
    LeRobot's observations, reset settling, controller, and primitive horizon.
    """

    suite_horizons = {
        "libero_spatial": 280,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    expected_horizon = suite_horizons.get(settings.suite_name)
    if expected_horizon is None:
        raise ValueError(
            f"lerobot_v060 has no validated horizon for suite {settings.suite_name!r}"
        )
    expected = {
        "observation_height": 360,
        "observation_width": 360,
        "rotate_images_180": True,
        "wait_steps_after_reset": 10,
        "max_episode_steps": expected_horizon,
        "control_mode": "relative",
        "reset_gripper_open": True,
    }
    if require_single_action:
        expected["action_chunk_size"] = 1
    actual = {name: getattr(settings, name) for name in expected}
    mismatches = [
        f"{name}={actual[name]!r} (expected {value!r})"
        for name, value in expected.items()
        if actual[name] != value
    ]
    if mismatches:
        raise ValueError(
            "lerobot_v060 evaluation contract mismatch: " + "; ".join(mismatches)
        )


def _validate_lerobot_pi0_fast_v060_contract(settings: LiberoSettings) -> None:
    """Validate the official LeRobot 0.6 pi0-FAST LIBERO geometry."""

    _validate_lerobot_v060_contract(settings, require_single_action=False)
    expected = {
        "action_chunk_size": 10,
        "normalize_gripper": False,
        "binarize_gripper": False,
        "invert_gripper": False,
    }
    actual = {name: getattr(settings, name) for name in expected}
    mismatches = [
        f"{name}={actual[name]!r} (expected {value!r})"
        for name, value in expected.items()
        if actual[name] != value
    ]
    if mismatches:
        raise ValueError(
            "lerobot_pi0_fast_v060 evaluation contract mismatch: "
            + "; ".join(mismatches)
        )
