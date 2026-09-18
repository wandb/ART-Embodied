"""Strict task and runtime contract for RoboCasa GR-1 Tabletop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROBOT_NAME = "GR1ArmsAndWaistFourierHands"
ENVIRONMENT_SUFFIX = f"_{ROBOT_NAME}_Env"
DATASET_PREFIX = "gr1_unified."
ENVIRONMENT_PREFIX = "gr1_unified/"
TASK_PROGRESS_SUPPORTED_TASKS = frozenset(
    {
        "PnPCupToDrawerClose",
        "PnPMilkToMicrowaveClose",
        "PnPPotatoToMicrowaveClose",
        "PosttrainPnPNovelFromCuttingboardToPanSplitA",
        "PosttrainPnPNovelFromPlacematToBasketSplitA",
        "PosttrainPnPNovelFromPlateToBowlSplitA",
        "PosttrainPnPNovelFromTrayToPotSplitA",
        "PosttrainPnPNovelFromTrayToTieredbasketSplitA",
    }
)


@dataclass(frozen=True, slots=True)
class RoboCasaTask:
    id: str
    official_n1d7_success_rate: float
    official_trials: int

    @property
    def environment_id(self) -> str:
        return f"{ENVIRONMENT_PREFIX}{self.id}{ENVIRONMENT_SUFFIX}"

    @property
    def dataset_id(self) -> str:
        return f"{DATASET_PREFIX}{self.id}"


@dataclass(frozen=True, slots=True)
class RoboCasaTaskManifest:
    source: Path
    source_revision: str
    tasks: tuple[RoboCasaTask, ...]

    @classmethod
    def load(cls, path: str | Path) -> "RoboCasaTaskManifest":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"RoboCasa task manifest is missing: {source}")
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("RoboCasa task manifest requires schema_version: 1")
        expected_top_level = {
            "schema_version",
            "benchmark",
            "source_revision",
            "robot",
            "tasks",
        }
        unknown = sorted(set(payload).difference(expected_top_level))
        if unknown:
            raise ValueError(
                "Unknown RoboCasa task manifest fields: " + ", ".join(unknown)
            )
        if payload.get("benchmark") != "robocasa_gr1_tabletop":
            raise ValueError(
                "RoboCasa task manifest benchmark must be robocasa_gr1_tabletop"
            )
        if payload.get("robot") != ROBOT_NAME:
            raise ValueError(f"RoboCasa task manifest robot must be {ROBOT_NAME}")
        rows = payload.get("tasks")
        if not isinstance(rows, list) or not rows:
            raise ValueError("RoboCasa task manifest requires a non-empty tasks list")
        tasks = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise TypeError(f"RoboCasa task row {index} must be a mapping")
            expected = {"id", "official_n1d7_success_rate", "official_trials"}
            row_unknown = sorted(set(row).difference(expected))
            missing = sorted(expected.difference(row))
            if missing or row_unknown:
                raise ValueError(
                    f"Invalid RoboCasa task row {index}: missing={missing}, "
                    f"unknown={row_unknown}"
                )
            task = RoboCasaTask(
                id=str(row["id"]),
                official_n1d7_success_rate=float(row["official_n1d7_success_rate"]),
                official_trials=int(row["official_trials"]),
            )
            if not task.id or not 0.0 <= task.official_n1d7_success_rate <= 1.0:
                raise ValueError(f"Invalid RoboCasa task values at row {index}")
            if task.official_trials <= 0:
                raise ValueError(
                    f"RoboCasa task trials must be positive at row {index}"
                )
            tasks.append(task)
        task_ids = [task.id for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("RoboCasa task manifest ids must be unique")
        return cls(
            source=source,
            source_revision=str(payload["source_revision"]),
            tasks=tuple(tasks),
        )

    def select(self, task_ids: tuple[str, ...]) -> tuple[RoboCasaTask, ...]:
        if not task_ids:
            return self.tasks
        by_id = {task.id: task for task in self.tasks}
        unknown = sorted(set(task_ids).difference(by_id))
        if unknown:
            raise ValueError("Unknown RoboCasa task ids: " + ", ".join(unknown))
        return tuple(by_id[task_id] for task_id in task_ids)


@dataclass(frozen=True, slots=True)
class RoboCasaSettings:
    manifest: RoboCasaTaskManifest
    tasks: tuple[RoboCasaTask, ...]
    simulator_python_executable: Path
    execution_horizon: int
    stop_on_success: bool
    stop_on_done: bool
    max_environment_steps: int
    max_policy_steps: int
    reward_mode: str
    progress_grasp_reward: float
    progress_placement_reward: float

    @classmethod
    def from_config(cls, config: Any) -> "RoboCasaSettings":
        if config.environment.type != "robocasa_gr1_tabletop":
            raise ValueError(
                "RoboCasa components require environment.type='robocasa_gr1_tabletop'"
            )
        if config.reward.type != "environment" or not config.reward.terminal_only:
            raise ValueError("RoboCasa GRPO requires terminal environment reward")
        reward_values = dict(config.reward.kwargs)
        reward_mode = str(reward_values.pop("mode", "success"))
        progress_grasp_reward = float(reward_values.pop("grasp_reward", 0.25))
        progress_placement_reward = float(reward_values.pop("placement_reward", 0.75))
        if reward_values:
            raise ValueError(
                f"Invalid RoboCasa reward.kwargs: unconsumed={sorted(reward_values)}"
            )
        if (
            config.environment.observation_processor
            or config.environment.action_processor
        ):
            raise ValueError("RoboCasa owns observation and action conversion")
        values = config.environment.kwargs
        required = {
            "task_manifest",
            "task_ids",
            "execution_horizon",
            "simulator_python_executable",
            "stop_on_success",
            "stop_on_done",
            "success_key",
        }
        missing = sorted(required.difference(values))
        unknown = sorted(set(values).difference(required))
        if missing or unknown:
            raise ValueError(
                "Invalid RoboCasa environment.kwargs: "
                f"missing={missing}; unconsumed={unknown}"
            )
        raw_task_ids = values["task_ids"]
        if not isinstance(raw_task_ids, list) or any(
            not isinstance(item, str) or not item for item in raw_task_ids
        ):
            raise TypeError("RoboCasa task_ids must be a list of non-empty strings")
        manifest = RoboCasaTaskManifest.load(values["task_manifest"])
        settings = cls(
            manifest=manifest,
            tasks=manifest.select(tuple(raw_task_ids)),
            simulator_python_executable=Path(
                str(values["simulator_python_executable"])
            ).expanduser(),
            execution_horizon=int(values["execution_horizon"]),
            stop_on_success=bool(values["stop_on_success"]),
            stop_on_done=bool(values["stop_on_done"]),
            max_environment_steps=int(config.rollout.max_episode_steps),
            max_policy_steps=int(config.rollout.max_policy_steps),
            reward_mode=reward_mode,
            progress_grasp_reward=progress_grasp_reward,
            progress_placement_reward=progress_placement_reward,
        )
        settings._validate(config)
        return settings

    def _validate(self, config: Any) -> None:
        if self.execution_horizon != 8:
            raise ValueError("RoboCasa GR-1 requires execution_horizon=8")
        if not str(self.simulator_python_executable):
            raise ValueError("RoboCasa simulator_python_executable cannot be empty")
        expected_policy_steps = (
            self.max_environment_steps + self.execution_horizon - 1
        ) // self.execution_horizon
        if self.max_policy_steps != expected_policy_steps:
            raise ValueError(
                "rollout.max_policy_steps must equal ceil(max_episode_steps / 8): "
                f"expected {expected_policy_steps}"
            )
        if str(config.environment.kwargs["success_key"]) != "success":
            raise ValueError("RoboCasa publishes success under info.success")
        if self.reward_mode not in {"success", "task_progress"}:
            raise ValueError(
                "RoboCasa reward.kwargs.mode must be 'success' or 'task_progress'"
            )
        if not (
            0.0 < self.progress_grasp_reward < self.progress_placement_reward < 1.0
        ):
            raise ValueError(
                "RoboCasa progress rewards must satisfy "
                "0 < grasp_reward < placement_reward < 1"
            )
        if self.reward_mode == "task_progress":
            unsupported = sorted(
                {task.id for task in self.tasks}.difference(
                    TASK_PROGRESS_SUPPORTED_TASKS
                )
            )
            if unsupported:
                raise ValueError(
                    "RoboCasa task_progress reward has no audited predicates for: "
                    + ", ".join(unsupported)
                )
