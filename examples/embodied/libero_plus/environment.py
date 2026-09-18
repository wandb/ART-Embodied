"""LIBERO-Plus task catalog backed by an external official checkout."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from art_embodied.experiment import EmbodiedScenario, RolloutContext
from examples.embodied.libero.environment import (
    LiberoChunkEnvironment,
    _ensure_legacy_gym_import,
)

from .runtime import prepare_libero_plus_runtime_paths
from .settings import LiberoPlusPanelEntry, LiberoPlusSettings


class LiberoPlusTaskCatalog:
    """Load only the frozen task variants selected by the panel manifest."""

    def __init__(self, settings: LiberoPlusSettings) -> None:
        runtime_info = prepare_libero_plus_runtime_paths()
        _ensure_legacy_gym_import()
        from libero.libero import benchmark

        self.settings = settings
        self.runtime_info = runtime_info
        self.suite = benchmark.get_benchmark_dict()[settings.suite_name]()
        self.tasks: dict[int, Any] = {}
        self.init_states: dict[int, Any] = {}
        self.bddl_files: dict[int, str] = {}
        cached_init_states: dict[Path, Any] = {}
        for task_id in settings.task_ids:
            task = self.suite.get_task(task_id)
            entry = settings.panel_entries[task_id]
            if str(task.name) != entry.name:
                raise RuntimeError(
                    "LIBERO-Plus manifest/catalog mismatch for index "
                    f"{task_id}: expected {entry.name!r}, found {task.name!r}"
                )
            self.tasks[task_id] = task
            self.bddl_files[task_id] = str(self.suite.get_task_bddl_file_path(task_id))
            init_path = libero_plus_init_states_path(
                task,
                Path(runtime_info["benchmark_root"]),
            )
            if init_path not in cached_init_states:
                cached_init_states[init_path] = load_libero_plus_init_states(init_path)
            self.init_states[task_id] = cached_init_states[init_path]

    def scenarios(self) -> list[EmbodiedScenario]:
        return [self._scenario(task_id) for task_id in self.settings.task_ids]

    def make_environment(
        self,
        scenario: EmbodiedScenario,
        context: RolloutContext,
    ) -> "LiberoPlusChunkEnvironment":
        task_id = int(scenario.payload["task_id"])
        return LiberoPlusChunkEnvironment(
            settings=self.settings,
            task=self.tasks[task_id],
            task_id=task_id,
            bddl_file=self.bddl_files[task_id],
            init_states=self.init_states[task_id],
            runtime_info=self.runtime_info,
            manifest_states=None,
            context=context,
            panel_entry=self.settings.panel_entries[task_id],
        )

    def _scenario(self, task_id: int) -> EmbodiedScenario:
        entry = self.settings.panel_entries[task_id]
        return EmbodiedScenario(
            id=f"libero_plus/{entry.split}/task-{task_id:04d}",
            task=_base_task_language(entry.base_task),
            payload={"task_id": task_id, "reset_options": {"trial_id": 0}},
        )


class LiberoPlusChunkEnvironment(LiberoChunkEnvironment):
    """Standard LIBERO chunk execution with perturbation provenance in info."""

    def __init__(self, *, panel_entry: LiberoPlusPanelEntry, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.panel_entry = panel_entry

    def reset(
        self,
        *,
        seed: int,
        options: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        observation, info = super().reset(seed=seed, options=options)
        info.update(
            {
                "benchmark": "LIBERO-Plus",
                "perturbation_category": self.panel_entry.category,
                "perturbation_difficulty": self.panel_entry.difficulty_level,
                "base_task": self.panel_entry.base_task,
                "panel_split": self.panel_entry.split,
                "official_task_id": self.panel_entry.official_id,
            }
        )
        return observation, info


def _base_task_language(base_task: str) -> str:
    scene = base_task.find("SCENE")
    if scene < 0:
        return base_task.replace("_", " ")
    offset = 8 if "SCENE10" in base_task else 7
    return base_task[scene + offset :].replace("_", " ")


def libero_plus_init_states_path(task: Any, benchmark_root: Path) -> Path:
    """Resolve an official variant to its revision-pinned initial states."""

    filename = str(task.init_states_file)
    folder = str(task.problem_folder)
    extension = filename.rsplit(".", maxsplit=1)[-1]
    root = (benchmark_root / "init_files").resolve()
    if "_language_" in filename:
        filename = f"{filename.split('_language_', maxsplit=1)[0]}.{extension}"
    elif "_view_" in filename:
        filename = f"{filename.split('_view_', maxsplit=1)[0]}.{extension}"
    elif "_table_" in filename:
        filename = re.sub(r"_table_\d+", "", filename)
    elif "_tb_" in filename:
        filename = re.sub(r"_tb_\d+", "", filename)
    elif "_light_" in filename:
        filename = f"{filename.split('_light_', maxsplit=1)[0]}.{extension}"
    elif "_add_" in filename or "_level" in filename:
        folder = str(Path("libero_newobj") / folder)
    path = (root / folder / filename).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise RuntimeError(f"Invalid LIBERO-Plus init-state path: {path}")
    return path


def load_libero_plus_init_states(path: Path) -> Any:
    """Load trusted init states from the verified official source revision."""

    import torch

    # PyTorch 2.6 changed this default. These pickle files are accepted only
    # after runtime.py verifies the pinned Git revision and path containment.
    states = torch.load(path, weights_only=False)
    if "_add_" in path.name or "_level" in path.name:
        states = states.reshape(1, -1)
    return states
