"""Frozen training and evaluation schedules for LIBERO-Plus."""

from __future__ import annotations

from typing import Any

from art_embodied.experiment import EmbodiedScenario

from .settings import LiberoPlusPanelEntry, LiberoPlusSettings


def build_train_scenarios(config: Any) -> list[EmbodiedScenario]:
    """Repeat one task-stratified perturbation panel for every policy update."""

    settings = LiberoPlusSettings.from_config(config)
    languages = _task_languages(settings)
    scenarios: list[EmbodiedScenario] = []
    training_entries = [
        settings.panel_entries[task_id] for task_id in settings.training_task_ids
    ]
    entries_by_base_task: dict[str, list[LiberoPlusPanelEntry]] = {}
    for entry in training_entries:
        entries_by_base_task.setdefault(entry.base_task, []).append(entry)
    for entries in entries_by_base_task.values():
        entries.sort(key=lambda entry: entry.catalog_index)

    for update in range(int(config.training.updates)):
        if settings.training_groups_per_base_task:
            base_tasks = sorted(entries_by_base_task)
            offset = update % len(base_tasks)
            base_tasks = base_tasks[offset:] + base_tasks[:offset]
            task_ids = [
                entries_by_base_task[base_task][
                    (update + repetition) % len(entries_by_base_task[base_task])
                ].catalog_index
                for repetition in range(settings.training_groups_per_base_task)
                for base_task in base_tasks
            ]
        else:
            # Preserve the original one-group-per-variant panel schedule.
            offset = update % len(settings.training_task_ids)
            task_ids = list(
                settings.training_task_ids[offset:]
                + settings.training_task_ids[:offset]
            )
        for group_index, task_id in enumerate(task_ids):
            scenarios.append(
                _scenario(
                    settings.panel_entries[task_id],
                    language=languages[task_id],
                    sequence=f"update-{update:04d}/group-{group_index:03d}",
                )
            )
    return scenarios


def build_evaluation_scenarios(config: Any) -> list[EmbodiedScenario]:
    """Return the untouched development or sealed panel exactly once."""

    settings = LiberoPlusSettings.from_config(config)
    languages = _task_languages(settings)
    return [
        _scenario(
            settings.panel_entries[task_id],
            language=languages[task_id],
            sequence=f"eval-{index:03d}",
        )
        for index, task_id in enumerate(settings.evaluation_task_ids)
    ]


def _scenario(
    entry: LiberoPlusPanelEntry,
    *,
    language: str,
    sequence: str,
) -> EmbodiedScenario:
    return EmbodiedScenario(
        id=(f"libero_plus/{entry.split}/{sequence}/task-{entry.catalog_index:04d}"),
        task=language,
        payload={
            "task_id": entry.catalog_index,
            "reset_options": {"trial_id": 0},
            "base_task": entry.base_task,
            "perturbation_category": entry.category,
            "perturbation_difficulty": entry.difficulty_level,
            "panel_split": entry.split,
            "official_task_id": entry.official_id,
        },
    )


def _task_languages(settings: LiberoPlusSettings) -> dict[int, str]:
    from examples.embodied.libero.environment import _ensure_legacy_gym_import

    _ensure_legacy_gym_import()
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[settings.suite_name]()
    return {
        task_id: str(suite.get_task(task_id).language) for task_id in settings.task_ids
    }


def _base_task_language(base_task: str) -> str:
    scene = base_task.find("SCENE")
    if scene < 0:
        return base_task.replace("_", " ")
    offset = 8 if "SCENE10" in base_task else 7
    return base_task[scene + offset :].replace("_", " ")
