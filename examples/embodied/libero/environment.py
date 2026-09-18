"""LIBERO task catalog and action-chunk environment adapter."""

from __future__ import annotations

from importlib import metadata
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import numpy as np
from packaging.version import Version

from art_embodied.experiment import EmbodiedScenario, RolloutContext
from art_embodied.lookahead import ActionChunkLookaheadPreview, LookaheadFrame

from .reset_evidence import reset_fingerprints
from .settings import LiberoSettings
from .state_manifest import (
    LiberoStateManifest,
    load_state_manifest,
    validate_manifest_bddl_files,
)


class LiberoTaskCatalog:
    """Load immutable suite metadata once per rollout process."""

    def __init__(self, settings: LiberoSettings) -> None:
        import torch

        _ensure_legacy_gym_import()
        self.runtime_info = _validate_libero_runtime(settings.simulator_compatibility)
        self.runtime_info.update(prepare_libero_runtime_paths())
        from libero.libero import benchmark, get_libero_path

        suite_type = benchmark.get_benchmark_dict()[settings.suite_name]
        self.suite = suite_type()
        self.settings = settings
        self.tasks: dict[int, Any] = {}
        self.init_states: dict[int, Any] = {}
        self.bddl_files: dict[int, str] = {}
        self.evaluation_state_manifest: LiberoStateManifest | None = None
        for task_id in settings.task_ids:
            task = self.suite.get_task(task_id)
            self.tasks[task_id] = task
            self.bddl_files[task_id] = os.path.join(
                get_libero_path("bddl_files"),
                task.problem_folder,
                task.bddl_file,
            )
            init_states_path = (
                Path(get_libero_path("init_states"))
                / task.problem_folder
                / task.init_states_file
            )
            self.init_states[task_id] = torch.load(
                init_states_path,
                weights_only=False,
            )
        if settings.evaluation_state_manifest is not None:
            self.evaluation_state_manifest = load_state_manifest(
                settings.evaluation_state_manifest,
                expected_suite_name=settings.suite_name,
                expected_simulator_compatibility=settings.simulator_compatibility,
            )
            validate_manifest_bddl_files(
                self.evaluation_state_manifest,
                self.bddl_files,
            )

    def scenarios(self) -> list[EmbodiedScenario]:
        return [
            EmbodiedScenario(
                id=f"{self.settings.suite_name}/{task_id}",
                task=str(self.tasks[task_id].language),
                payload={"task_id": task_id},
            )
            for task_id in self.settings.task_ids
        ]

    def make_environment(
        self,
        scenario: EmbodiedScenario,
        context: RolloutContext,
    ) -> "LiberoChunkEnvironment":
        task_id = int(scenario.payload["task_id"])
        return LiberoChunkEnvironment(
            settings=self.settings,
            task=self.tasks[task_id],
            task_id=task_id,
            bddl_file=self.bddl_files[task_id],
            init_states=self.init_states[task_id],
            runtime_info=self.runtime_info,
            manifest_states=(
                self.evaluation_state_manifest.states
                if self.evaluation_state_manifest is not None
                else None
            ),
            context=context,
        )


class LiberoChunkEnvironment(ActionChunkLookaheadPreview):
    """Gymnasium-shaped wrapper that executes one OpenVLA action chunk."""

    def __init__(
        self,
        *,
        settings: LiberoSettings,
        task: Any,
        task_id: int,
        bddl_file: str,
        init_states: Any,
        runtime_info: Mapping[str, str],
        manifest_states: Mapping[str, np.ndarray] | None,
        context: RolloutContext,
    ) -> None:
        from libero.libero.envs import OffScreenRenderEnv

        self.settings = settings
        self.task = task
        self.task_id = int(task_id)
        self.init_states = init_states
        self.runtime_info = dict(runtime_info)
        self.manifest_states = manifest_states
        self.context = context
        self.env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_heights=settings.observation_height,
            camera_widths=settings.observation_width,
        )
        self._configure_controller()
        self.raw_observation: Mapping[str, Any] | None = None
        self.env_steps = 0
        self.success = False

    def _configure_controller(self) -> None:
        if self.settings.control_mode in {"unchanged", "default"}:
            return
        use_delta = self.settings.control_mode == "relative"
        for robot in self.env.robots:
            robot.controller.use_delta = use_delta

    def reset(
        self,
        *,
        seed: int,
        options: Mapping[str, Any] | None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        options = dict(options or {})
        unknown_options = sorted(
            set(options).difference({"trial_id", "manifest_state_key"})
        )
        if unknown_options:
            raise ValueError(
                "Unsupported LIBERO scenario reset options: "
                + ", ".join(unknown_options)
            )
        self.env.seed(int(seed))
        self.env.reset()
        has_trial = "trial_id" in options
        has_manifest_state = "manifest_state_key" in options
        if has_trial == has_manifest_state:
            raise ValueError(
                "LIBERO scenarios must define exactly one of "
                "reset_options.trial_id or reset_options.manifest_state_key"
            )
        trial_id: int | None = None
        init_state_index: int | None = None
        manifest_state_key: str | None = None
        if has_trial:
            trial_id = int(options["trial_id"])
            init_state_index = trial_id % len(self.init_states)
            init_state = self.init_states[init_state_index]
        else:
            manifest_state_key = str(options["manifest_state_key"])
            if self.manifest_states is None:
                raise ValueError(
                    "manifest_state_key requires environment.kwargs."
                    "evaluation_state_manifest"
                )
            if manifest_state_key not in self.manifest_states:
                raise KeyError(
                    f"Unknown LIBERO manifest state key: {manifest_state_key!r}"
                )
            init_state = self.manifest_states[manifest_state_key]
        observation = self.env.set_init_state(init_state)
        # Match RLinf v0.1's default NumPy action dtype exactly.
        dummy_action = np.zeros(7, dtype=np.float64)
        if self.settings.reset_gripper_open:
            dummy_action[-1] = -1.0
        for _ in range(self.settings.wait_steps_after_reset):
            observation, _reward, _done, _info = self.env.step(dummy_action)
        self.raw_observation = observation
        self.env_steps = 0
        self.success = bool(self.env.check_success())
        policy_observation = self._policy_observation()
        return policy_observation, {
            "suite": self.settings.suite_name,
            "task_id": self.task_id,
            "task_name": str(self.task.name),
            "trial_id": trial_id,
            "init_state_index": init_state_index,
            "manifest_state_key": manifest_state_key,
            "state_source": "manifest"
            if manifest_state_key is not None
            else "official",
            "environment_seed": int(seed),
            "wait_steps_after_reset": self.settings.wait_steps_after_reset,
            "reset_gripper_open": self.settings.reset_gripper_open,
            "control_mode": self.settings.control_mode,
            "simulator_compatibility": self.settings.simulator_compatibility,
            "simulator_runtime": self.runtime_info,
            "reset_fingerprints": reset_fingerprints(policy_observation, self.env.sim),
        }

    def step(
        self,
        action_chunk: Any,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        # Action dtype is part of the simulator contract. Do not silently
        # round OpenVLA-OFT's float64 decoded actions before stepping LIBERO.
        actions = np.asarray(action_chunk)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(
                "LIBERO expects a [chunk, 7] processed action array, "
                f"got {actions.shape}"
            )
        actions = actions[: self.settings.action_chunk_size]
        primitive_rewards: list[float] = []
        primitive_mask: list[bool] = []
        executed_actions: list[list[float]] = []
        primitive_observations_before: list[dict[str, np.ndarray]] = []
        terminated = False
        truncated = False
        last_info: Mapping[str, Any] = {}
        raw_environment_reward = 0.0
        for action in actions:
            if self.env_steps >= self.settings.max_episode_steps:
                truncated = True
                break
            observation_before = (
                _copy_observation(self._policy_observation())
                if self.settings.capture_primitive_observations
                else None
            )
            try:
                observation, reward, done, info = self.env.step(action)
            except ValueError as exc:
                if "terminated episode" not in str(exc):
                    raise
                terminated = True
                break
            self.raw_observation = observation
            self.env_steps += 1
            raw_reward = float(reward)
            raw_environment_reward += raw_reward
            last_info = info if isinstance(info, Mapping) else {}
            step_success = bool(self.env.check_success())
            self.success = self.success or step_success
            terminated = terminated or bool(done)
            reward_value = self.settings.reward_coefficient if step_success else 0.0
            primitive_rewards.append(reward_value)
            primitive_mask.append(True)
            executed_actions.append(action.tolist())
            if observation_before is not None:
                primitive_observations_before.append(observation_before)
            if self.success and self.settings.stop_on_success:
                break
            if terminated and self.settings.stop_on_done:
                break
        missing = self.settings.action_chunk_size - len(primitive_rewards)
        primitive_rewards.extend([0.0] * missing)
        primitive_mask.extend([False] * missing)
        if self.env_steps >= self.settings.max_episode_steps and not terminated:
            truncated = True
        info_payload = {
            **dict(last_info),
            "success": bool(self.success),
            "task_id": self.task_id,
            "env_steps": self.env_steps,
            "primitive_rewards": primitive_rewards,
            "primitive_loss_mask": primitive_mask,
            "executed_actions": executed_actions,
            "raw_environment_reward": raw_environment_reward,
        }
        if self.settings.capture_primitive_observations:
            info_payload["primitive_observations_before"] = (
                primitive_observations_before
            )
        return (
            self._policy_observation(),
            float(sum(primitive_rewards)),
            bool(terminated or self.success),
            bool(truncated),
            info_payload,
        )

    def _policy_observation(self) -> dict[str, np.ndarray]:
        if self.raw_observation is None:
            raise RuntimeError("LIBERO observation requested before reset")
        primary = _image(
            self.raw_observation[self.settings.primary_image_key],
            rotate_180=self.settings.rotate_images_180,
        )
        observation = {
            "image": primary,
            "proprio_state": _proprio_state(self.raw_observation),
        }
        if self.settings.wrist_image_key in self.raw_observation:
            observation["wrist_image"] = _image(
                self.raw_observation[self.settings.wrist_image_key],
                rotate_180=self.settings.rotate_images_180,
            )
        return observation

    def render(self) -> np.ndarray:
        return self._policy_observation()["image"]

    def preview_action_chunk(
        self,
        action_chunk: Any,
        *,
        source_environment: Any,
        execution_horizon: int,
        frame_stride: int,
        max_frames: int,
    ) -> list[LookaheadFrame]:
        """Render an unused chunk tail after synchronizing from a live clone.

        This method is called on a dedicated shadow environment. Simulator
        state is copied from ``source_environment`` before every preview, so
        speculative stepping cannot alter rewards, observations, or episode
        termination in the real rollout.
        """

        if not isinstance(source_environment, LiberoChunkEnvironment):
            raise TypeError("LIBERO lookahead requires a LIBERO source environment")
        if execution_horizon < 1 or frame_stride < 1 or max_frames < 1:
            raise ValueError("Lookahead horizons and frame bounds must be positive")
        actions = np.asarray(action_chunk)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(
                "LIBERO lookahead expects a [chunk, 7] action array, "
                f"got {actions.shape}"
            )
        if len(actions) <= execution_horizon:
            return []

        self.env.sim.set_state(source_environment.env.sim.get_state())
        self.env.sim.forward()
        _copy_runtime_scalars(source_environment.env, self.env)
        self.env_steps = int(source_environment.env_steps)
        self.success = bool(source_environment.success)
        self.raw_observation = _copy_raw_observation(source_environment.raw_observation)

        frames: list[LookaheadFrame] = []
        for action_index, action in enumerate(actions):
            try:
                observation, _reward, done, _info = self.env.step(action)
            except ValueError as exc:
                if "terminated episode" not in str(exc):
                    raise
                break
            self.raw_observation = observation
            self.env_steps += 1
            tail_index = action_index - execution_horizon
            if tail_index >= 0 and tail_index % frame_stride == 0:
                frames.append(
                    LookaheadFrame(
                        image=self.render().copy(),
                        action_index=action_index,
                    )
                )
                if len(frames) >= max_frames:
                    break
            if bool(done):
                break
        return frames

    def close(self) -> None:
        self.env.close()


def _copy_observation(
    observation: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Detach a primitive state from the mutable simulator observation."""

    return {key: np.asarray(value).copy() for key, value in observation.items()}


def _copy_raw_observation(
    observation: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if observation is None:
        return None
    return {
        key: np.asarray(value).copy() if hasattr(value, "shape") else value
        for key, value in observation.items()
    }


def _copy_runtime_scalars(source: Any, destination: Any) -> None:
    """Synchronize common episode counters without depending on LIBERO internals."""

    for name in ("timestep", "_elapsed_steps", "done"):
        if hasattr(source, name) and hasattr(destination, name):
            try:
                setattr(destination, name, getattr(source, name))
            except (AttributeError, TypeError):
                continue


def _validate_libero_runtime(compatibility: str) -> dict[str, str]:
    """Reject a simulator package that is not the validated parity runtime."""

    if compatibility != "rlinf_v01":
        raise ValueError(f"Unsupported LIBERO compatibility contract: {compatibility}")
    from importlib import metadata

    import libero

    module_path = Path(libero.__file__).resolve()
    try:
        distribution_version = metadata.version("libero")
    except metadata.PackageNotFoundError:
        distribution_version = None
    try:
        hf_version = metadata.version("hf-libero")
    except metadata.PackageNotFoundError:
        hf_version = None

    source_checkout = "RLinf-LIBERO" in module_path.parts
    hf_root = _distribution_root(metadata, "hf-libero")
    libero_root = _distribution_root(metadata, "libero")
    hf_owns_module = hf_root is not None and module_path.is_relative_to(hf_root)
    libero_owns_module = libero_root is not None and module_path.is_relative_to(
        libero_root
    )
    if source_checkout:
        provider = "rlinf-source-checkout"
        provider_version = "source-checkout"
    elif hf_version == "0.1.4" and hf_owns_module:
        provider = "hf-libero"
        provider_version = hf_version
    elif distribution_version == "0.1.0" and libero_owns_module:
        provider = "libero"
        provider_version = distribution_version
    else:
        raise RuntimeError(
            "simulator_compatibility='rlinf_v01' requires the validated product "
            "runtime (hf-libero==0.1.4 or libero==0.1.0); loaded "
            f"module={module_path}, libero={distribution_version or 'not-installed'}, "
            f"hf-libero={hf_version or 'not-installed'}. Install "
            "art-embodied[libero]. The RLinf/LIBERO source checkout is "
            "accepted only for conformance diagnostics."
        )
    return {
        "compatibility": compatibility,
        "module_path": str(module_path),
        "provider": provider,
        "distribution_version": provider_version,
    }


def prepare_libero_runtime_paths() -> dict[str, str]:
    """Validate LIBERO paths or derive them from the installed distribution.

    ``hf-libero`` bundles benchmark definitions and initial states inside the
    wheel, but LIBERO still discovers them through a mutable ``config.yaml``.
    Prefer an explicitly configured valid layout. If that layout is absent or
    stale, create a process-local config pointing at the installed package so
    clean installations do not depend on a machine-specific cache layout.
    """

    configured_dir = Path(
        os.environ.get("LIBERO_CONFIG_PATH", "~/.libero")
    ).expanduser()
    configured_file = configured_dir / "config.yaml"
    configured = _read_libero_path_config(configured_file)
    if configured is not None and _libero_path_config_is_valid(configured):
        return {
            "path_source": "configured",
            "config_path": str(configured_file.resolve()),
            "benchmark_root": str(Path(configured["benchmark_root"]).resolve()),
        }

    import libero

    module_dir = Path(libero.__file__).resolve().parent
    benchmark_root = next(
        (
            candidate
            for candidate in (module_dir / "libero", module_dir)
            if (candidate / "bddl_files").is_dir()
            and (candidate / "init_files").is_dir()
        ),
        None,
    )
    if benchmark_root is None:
        configured_detail = (
            f"configured file {configured_file} is missing or invalid; "
            if configured_file.exists()
            else f"configured file {configured_file} does not exist; "
        )
        raise RuntimeError(
            "Unable to locate LIBERO bddl_files and init_files: "
            f"{configured_detail}installed module root is {module_dir}. "
            "Install art-embodied[libero] or set LIBERO_CONFIG_PATH "
            "to a directory containing a valid config.yaml."
        )

    datasets = benchmark_root.parent / "datasets"
    paths = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(datasets),
        "assets": str(benchmark_root / "assets"),
    }
    runtime_dir = Path(tempfile.mkdtemp(prefix="art-embodied-libero-"))
    runtime_config = runtime_dir / "config.yaml"
    runtime_config.write_text(
        "".join(f"{key}: {value}\n" for key, value in paths.items()),
        encoding="utf-8",
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(runtime_dir)
    return {
        "path_source": "installed-distribution",
        "config_path": str(runtime_config),
        "benchmark_root": str(benchmark_root),
    }


def validate_libero_task_assets(settings: LiberoSettings) -> dict[str, int]:
    """Fail before model loading when a configured task asset is incomplete."""

    prepare_libero_runtime_paths()
    from libero.libero import benchmark, get_libero_path

    suite_type = benchmark.get_benchmark_dict()[settings.suite_name]
    suite = suite_type()
    missing: list[Path] = []
    for task_id in settings.task_ids:
        task = suite.get_task(task_id)
        candidates = (
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file,
            Path(get_libero_path("init_states"))
            / task.problem_folder
            / task.init_states_file,
        )
        missing.extend(path for path in candidates if not path.is_file())
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise RuntimeError(
            "LIBERO task assets are incomplete; refusing to load the policy before "
            f"these files are available:\n{formatted}"
        )
    return {"tasks": len(settings.task_ids), "files": 2 * len(settings.task_ids)}


def validate_libero_runtime_imports() -> None:
    """Import the simulator stack before policy weights consume GPU memory."""

    # LIBERO's upstream import asks an interactive first-run question unless a
    # path config already exists. Runtime validation is also a public boundary,
    # so make it safe in fresh non-interactive Slurm jobs on its own.
    prepare_libero_runtime_paths()
    _validate_libero_mujoco_abi()
    _ensure_legacy_gym_import()
    try:
        from libero.libero.envs import OffScreenRenderEnv  # noqa: F401
    except Exception as exc:
        import ctypes.util

        missing = getattr(exc, "name", None) or str(exc)
        egl_hint = ""
        if ctypes.util.find_library("EGL") is None:
            egl_hint = (
                " No GLVND EGL loader (libEGL.so.1) is visible. On Ubuntu, "
                "install the libegl1 system package or expose an equivalent "
                "loader through LD_LIBRARY_PATH."
            )
        raise RuntimeError(
            "LIBERO simulator runtime could not initialize its EGL renderer; "
            f"failed while importing {missing!r}. Run the training command on "
            "a GPU allocation with a working NVIDIA/EGL runtime and install "
            "the complete art-embodied PI/LIBERO profile. The CPU-only "
            f"--preflight command does not require EGL.{egl_hint}"
        ) from exc


def _validate_libero_mujoco_abi() -> None:
    """Reject MuJoCo releases that changed robosuite 1.4's controller ABI."""

    expected = Version("3.8.1")
    try:
        installed = Version(metadata.version("mujoco"))
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "LIBERO simulator runtime is incomplete; mujoco is not installed. "
            "Install the simulator-specific ART-Embodied extra before rollout."
        ) from exc
    if installed != expected:
        raise RuntimeError(
            "LIBERO simulator runtime requires mujoco==3.8.1 with "
            "robosuite==1.4.0; found "
            f"mujoco=={installed}. MuJoCo 3.10 changed mj_fullM and is not ABI "
            "compatible with this validated controller stack."
        )


def _read_libero_path_config(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    try:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    return {str(key): str(value) for key, value in payload.items()}


def _libero_path_config_is_valid(paths: Mapping[str, str]) -> bool:
    required_directories = ("benchmark_root", "bddl_files", "init_states", "assets")
    return all(
        key in paths and Path(paths[key]).expanduser().is_dir()
        for key in required_directories
    )


def _distribution_root(metadata: Any, distribution_name: str) -> Path | None:
    try:
        distribution = metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError:
        return None
    return Path(distribution.locate_file("")).resolve()


def _ensure_legacy_gym_import() -> None:
    """Expose Gymnasium as ``gym`` when the pinned LIBERO fork requests it."""

    try:
        import gym  # noqa: F401
    except ModuleNotFoundError:
        import gymnasium

        sys.modules["gym"] = gymnasium


def _image(value: Any, *, rotate_180: bool) -> np.ndarray:
    image = np.asarray(value)
    if rotate_180:
        image = image[::-1, ::-1]
    return image.astype(np.uint8, copy=False)


def _proprio_state(observation: Mapping[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(observation["robot0_eef_pos"]),
            _quat_xyzw_to_axisangle(observation["robot0_eef_quat"]),
            np.asarray(observation["robot0_gripper_qpos"]),
        ]
    ).astype(np.float32, copy=False)


def _quat_xyzw_to_axisangle(value: Any) -> np.ndarray:
    """Match LeRobot 0.6's LIBERO quaternion processor without importing EGL."""

    quat = np.asarray(value, dtype=np.float32).reshape(4)
    w = float(np.clip(quat[3], -1.0, 1.0))
    denominator = float(np.sqrt(max(0.0, 1.0 - w * w)))
    if denominator <= 1.0e-10:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arccos(w)
    return (quat[:3] * angle / denominator).astype(np.float32, copy=False)
