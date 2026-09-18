"""Validated settings for the revision-pinned LIBERO-Plus panel."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from examples.embodied.libero.settings import LiberoSettings

from .runtime import (
    LIBERO_PLUS_ASSET_ARCHIVE_SHA256,
    LIBERO_PLUS_CLASSIFICATION_SHA256,
    LIBERO_PLUS_SOURCE_REVISION,
)


@dataclass(frozen=True, slots=True)
class LiberoPlusPanelEntry:
    catalog_index: int
    official_id: int
    name: str
    base_task: str
    category: str
    difficulty_level: int
    split: str


@dataclass(frozen=True, slots=True)
class LiberoPlusSettings(LiberoSettings):
    evaluation_task_ids: tuple[int, ...]
    panel_entries: Mapping[int, LiberoPlusPanelEntry]
    panel_split: str
    panel_manifest: Path
    panel_manifest_sha256: str
    source_revision: str
    classification_sha256: str
    asset_archive_sha256: str
    training_groups_per_base_task: int

    @classmethod
    def from_config(cls, config: Any) -> "LiberoPlusSettings":
        if config.environment.type != "libero_plus":
            raise ValueError(
                "LIBERO-Plus components require environment.type='libero_plus'"
            )
        if str(config.environment.task) != "libero_10":
            raise ValueError("The first LIBERO-Plus panel requires task='libero_10'")
        if config.environment.observation_processor:
            raise ValueError("LIBERO-Plus does not consume observation_processor")
        if config.environment.action_processor:
            raise ValueError("LIBERO-Plus does not consume action_processor")
        values = config.environment.kwargs
        required = {
            "panel_manifest",
            "panel_manifest_sha256",
            "evaluation_panel_split",
            "sft_anchor_task_ids",
            "source_revision",
            "classification_sha256",
            "asset_archive_sha256",
            "simulator_compatibility",
            "evaluation_protocol",
            "observation_height",
            "observation_width",
            "primary_image_key",
            "wrist_image_key",
            "rotate_images_180",
            "wait_steps_after_reset",
            "action_chunk_size",
            "capture_primitive_observations",
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
                "LIBERO-Plus YAML must explicitly define environment.kwargs: "
                + ", ".join(missing)
            )
        expected_pins = {
            "source_revision": LIBERO_PLUS_SOURCE_REVISION,
            "classification_sha256": LIBERO_PLUS_CLASSIFICATION_SHA256,
            "asset_archive_sha256": LIBERO_PLUS_ASSET_ARCHIVE_SHA256,
        }
        mismatches = [
            f"{key}={values[key]!r} (expected {expected!r})"
            for key, expected in expected_pins.items()
            if str(values[key]) != expected
        ]
        if mismatches:
            raise ValueError(
                "LIBERO-Plus provenance mismatch: " + "; ".join(mismatches)
            )
        sft_anchor_task_ids = tuple(
            int(value) for value in values["sft_anchor_task_ids"]
        )
        if (
            len(set(sft_anchor_task_ids)) != len(sft_anchor_task_ids)
            or any(not 0 <= task_id < 10 for task_id in sft_anchor_task_ids)
        ):
            raise ValueError(
                "LIBERO-Plus sft_anchor_task_ids must be unique and in [0, 9]"
            )
        if str(values["simulator_compatibility"]) != "libero_plus_cvpr2026":
            raise ValueError(
                "LIBERO-Plus requires simulator_compatibility='libero_plus_cvpr2026'"
            )
        if str(values["evaluation_protocol"]) != "libero_plus_pi0_fast_v1":
            raise ValueError(
                "LIBERO-Plus pi0-FAST requires evaluation_protocol="
                "'libero_plus_pi0_fast_v1'"
            )
        manifest_path = Path(str(values["panel_manifest"])).expanduser().resolve()
        manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        expected_manifest_digest = str(values["panel_manifest_sha256"])
        if manifest_digest != expected_manifest_digest:
            raise ValueError(
                "LIBERO-Plus panel manifest digest mismatch: "
                f"expected {expected_manifest_digest}, found {manifest_digest}"
            )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_manifest_header(payload)
        entries: dict[int, LiberoPlusPanelEntry] = {}
        official_ids: set[int] = set()
        names: set[str] = set()
        for split_name, rows in dict(payload["splits"]).items():
            for raw in rows:
                entry = LiberoPlusPanelEntry(
                    catalog_index=int(raw["catalog_index"]),
                    official_id=int(raw["official_id"]),
                    name=str(raw["name"]),
                    base_task=str(raw["base_task"]),
                    category=str(raw["category"]),
                    difficulty_level=int(raw["difficulty_level"]),
                    split=str(split_name),
                )
                if entry.catalog_index in entries:
                    raise ValueError(
                        f"Duplicate LIBERO-Plus catalog index: {entry.catalog_index}"
                    )
                if entry.official_id in official_ids or entry.name in names:
                    raise ValueError("LIBERO-Plus panel entries must be unique")
                if entry.catalog_index != entry.official_id - 1:
                    raise ValueError(
                        "LIBERO-Plus official IDs must map to zero-based catalog "
                        f"indices: {entry.official_id}"
                    )
                entries[entry.catalog_index] = entry
                official_ids.add(entry.official_id)
                names.add(entry.name)
        split = str(values["evaluation_panel_split"])
        if split not in {"development", "sealed"}:
            raise ValueError("evaluation_panel_split must be 'development' or 'sealed'")
        training_task_ids = tuple(
            entry.catalog_index for entry in entries.values() if entry.split == "train"
        )
        evaluation_task_ids = tuple(
            entry.catalog_index for entry in entries.values() if entry.split == split
        )
        task_ids = tuple(sorted(set(training_task_ids + evaluation_task_ids)))
        if not training_task_ids or not evaluation_task_ids:
            raise ValueError("LIBERO-Plus panel splits must be non-empty")
        if set(training_task_ids).intersection(evaluation_task_ids):
            raise ValueError("LIBERO-Plus train and evaluation task IDs overlap")
        selection = payload.get("selection")
        if not isinstance(selection, dict):
            raise ValueError("LIBERO-Plus manifest selection must be a mapping")
        declared_split_sizes = selection.get(
            "split_sizes",
            {"train": 100, "development": 50, "sealed": 50},
        )
        if not isinstance(declared_split_sizes, dict):
            raise ValueError("LIBERO-Plus selection.split_sizes must be a mapping")
        expected_split_sizes = {
            name: int(declared_split_sizes[name])
            for name in ("train", "development", "sealed")
        }
        actual_split_sizes = {
            name: sum(entry.split == name for entry in entries.values())
            for name in expected_split_sizes
        }
        if actual_split_sizes != expected_split_sizes:
            raise ValueError(
                "LIBERO-Plus panel split sizes changed: "
                f"expected {expected_split_sizes}, found {actual_split_sizes}"
            )
        groups_per_base_task = int(values.get("training_groups_per_base_task", 0))
        if groups_per_base_task < 0:
            raise ValueError("training_groups_per_base_task must be non-negative")
        if groups_per_base_task:
            training_base_tasks = {
                entries[task_id].base_task for task_id in training_task_ids
            }
            expected_groups = len(training_base_tasks) * groups_per_base_task
        else:
            expected_groups = len(training_task_ids)
        if int(config.rollout.groups_per_update) != expected_groups:
            mode = (
                f"{groups_per_base_task} groups per base task"
                if groups_per_base_task
                else "one group per frozen training variant"
            )
            raise ValueError(
                f"LIBERO-Plus requires {mode}: groups_per_update="
                f"{expected_groups}"
            )
        if int(config.rollout.epochs_per_update) != 1:
            raise ValueError("LIBERO-Plus panel requires rollout.epochs_per_update=1")
        if int(config.evaluation.episodes) != len(evaluation_task_ids):
            raise ValueError(
                "LIBERO-Plus evaluation.episodes must equal the frozen panel size "
                f"({len(evaluation_task_ids)})"
            )
        geometry = {
            "observation_height": 360,
            "observation_width": 360,
            "rotate_images_180": True,
            "wait_steps_after_reset": 10,
            "action_chunk_size": 10,
            "max_episode_steps": 520,
            "control_mode": "relative",
            "normalize_gripper": False,
            "binarize_gripper": False,
            "invert_gripper": False,
        }
        actual_geometry = {
            **{key: values[key] for key in geometry if key != "max_episode_steps"},
            "max_episode_steps": int(config.rollout.max_episode_steps),
        }
        bad_geometry = [
            f"{key}={actual_geometry[key]!r} (expected {expected!r})"
            for key, expected in geometry.items()
            if actual_geometry[key] != expected
        ]
        if bad_geometry:
            raise ValueError(
                "LIBERO-Plus pi0-FAST geometry mismatch: " + "; ".join(bad_geometry)
            )
        if str(values["init_state_selection"]) != "balanced_task_random_reset":
            raise ValueError(
                "LIBERO-Plus requires init_state_selection='balanced_task_random_reset'"
            )
        if str(values["success_key"]) != "success":
            raise ValueError("LIBERO-Plus requires success_key='success'")
        if not bool(values["stop_on_success"]) or not bool(values["stop_on_done"]):
            raise ValueError("LIBERO-Plus requires stop_on_success/stop_on_done=true")
        if bool(values["capture_primitive_observations"]):
            raise ValueError(
                "The baseline LIBERO-Plus recipe disables primitive observations"
            )
        if config.reward.type != "environment" or not config.reward.terminal_only:
            raise ValueError("LIBERO-Plus requires terminal-only environment reward")
        if config.reward.kwargs:
            raise ValueError("LIBERO-Plus does not consume reward.kwargs")
        if "reset_gripper_open" not in config.environment.reset:
            raise ValueError("LIBERO-Plus requires reset.reset_gripper_open")
        schedule = {str(k): int(v) for k, v in dict(values["reset_schedule"]).items()}
        return cls(
            suite_name=str(config.environment.task),
            task_ids=task_ids,
            training_task_ids=training_task_ids,
            training_trial_ids=(0,),
            evaluation_trial_ids=(0,),
            evaluation_state_manifest=None,
            simulator_compatibility="libero_plus_cvpr2026",
            evaluation_protocol="libero_plus_pi0_fast_v1",
            observation_height=int(values["observation_height"]),
            observation_width=int(values["observation_width"]),
            primary_image_key=str(values["primary_image_key"]),
            wrist_image_key=str(values["wrist_image_key"]),
            rotate_images_180=bool(values["rotate_images_180"]),
            wait_steps_after_reset=int(values["wait_steps_after_reset"]),
            action_chunk_size=int(values["action_chunk_size"]),
            capture_primitive_observations=bool(
                values["capture_primitive_observations"]
            ),
            max_episode_steps=int(config.rollout.max_episode_steps),
            stop_on_success=bool(values["stop_on_success"]),
            stop_on_done=bool(values["stop_on_done"]),
            normalize_gripper=bool(values["normalize_gripper"]),
            binarize_gripper=bool(values["binarize_gripper"]),
            invert_gripper=bool(values["invert_gripper"]),
            control_mode=str(values["control_mode"]),
            init_state_selection="balanced_task_random_reset",
            reset_gripper_open=bool(config.environment.reset["reset_gripper_open"]),
            reward_coefficient=float(config.reward.scale),
            reset_schedule=schedule,
            evaluation_task_ids=evaluation_task_ids,
            panel_entries=entries,
            panel_split=split,
            panel_manifest=manifest_path,
            panel_manifest_sha256=manifest_digest,
            source_revision=LIBERO_PLUS_SOURCE_REVISION,
            classification_sha256=LIBERO_PLUS_CLASSIFICATION_SHA256,
            asset_archive_sha256=LIBERO_PLUS_ASSET_ARCHIVE_SHA256,
            training_groups_per_base_task=groups_per_base_task,
        )


def _validate_manifest_header(payload: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": 1,
        "suite": "libero_10",
        "source_revision": LIBERO_PLUS_SOURCE_REVISION,
        "classification_sha256": LIBERO_PLUS_CLASSIFICATION_SHA256,
    }
    mismatches = [
        f"{key}={payload.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if payload.get(key) != value
    ]
    if mismatches:
        raise ValueError("Invalid LIBERO-Plus panel manifest: " + "; ".join(mismatches))
    category = payload.get("category")
    if not isinstance(category, str) or not category.strip():
        raise ValueError("LIBERO-Plus panel category must be a non-empty string")
