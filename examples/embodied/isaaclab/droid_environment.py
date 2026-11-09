"""DROID Maple-table pick-and-place on Newton/Warp (no RTX renderer).

Uses Arena's task, placement sampler, assets, and absolute-joint embodiment.
The explicit Panda armature is a modified-physics condition, not an unmodified
Arena benchmark. MuJoCo Menagerie's Panda supplies the 0.1 reference value.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path


def load_droid_recipe(path: Path | None = None) -> dict:
    import yaml

    path = path or Path(__file__).with_name("pi05_droid_soup_grpo.yaml")
    raw = yaml.safe_load(path.read_text())
    if (
        not isinstance(raw, dict)
        or raw.get("policy", {}).get("type") != "openpi_pi05_droid"
        or raw.get("environment", {}).get("type") != "isaaclab"
    ):
        raise ValueError("Use an Isaac Lab OpenPI DROID recipe")
    return raw


@dataclass(frozen=True)
class DroidSceneConfig:
    pick_up_object: str
    destination_location: str
    instruction: str
    num_envs: int = 32
    seed: int = 42
    episode_length_s: float = 70.0
    panda_armature: float = 0.1
    device: str = "cuda:0"

    @classmethod
    def from_recipe(cls, raw: dict, **runtime):
        settings = raw["environment"]["kwargs"]
        if settings.get("physics_condition") != (
            "native_mujoco_warp_fp32_panda_armature_0.1"
        ):
            raise ValueError(
                "The DROID recipe requires its qualified physics condition"
            )
        return cls(
            **{
                name: settings[name]
                for name in ("pick_up_object", "destination_location", "instruction")
            },
            **runtime,
        )

    def __post_init__(self):
        for value in (self.pick_up_object, self.destination_location, self.instruction):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    "Object, destination, and instruction must be explicit"
                )
        if type(self.num_envs) is not int or self.num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if not math.isfinite(self.episode_length_s) or self.episode_length_s <= 0:
            raise ValueError("episode_length_s must be finite and positive")
        if not math.isfinite(self.panda_armature) or self.panda_armature <= 0:
            raise ValueError("panda_armature must be finite and positive")

    def reset_contract(self):
        return {
            "scene": {
                key: value
                for key, value in asdict(self).items()
                if key not in {"num_envs", "seed", "device"}
            },
            "scene_version": 1,
            "renderer": "newton_warp_textured",
            "physics": "newton_mujoco_warp_native",
        }


def build_droid_scene(settings: DroidSceneConfig, *, output_dir: Path):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_environments.pick_and_place_maple_table_environment import (
        PickAndPlaceMapleTableEnvironment,
        PickAndPlaceMapleTableEnvironmentCfg,
    )
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    arena = PickAndPlaceMapleTableEnvironment().build(
        PickAndPlaceMapleTableEnvironmentCfg(
            enable_cameras=True,
            pick_up_object=settings.pick_up_object,
            destination_location=settings.destination_location,
            episode_length_s=settings.episode_length_s,
        )
    )
    camera_names = tuple(arena.embodiment.camera_config.camera_names())
    for name in camera_names:
        camera = getattr(arena.embodiment.camera_config, name)
        camera.renderer_cfg = NewtonWarpRendererCfg(enable_textures=True)
        camera.update_latest_camera_pose = True
    builder = ArenaEnvBuilder(
        arena,
        ArenaEnvBuilderCfg(
            num_envs=settings.num_envs,
            env_spacing=0.0,
            seed=settings.seed,
            placement_seed=settings.seed,
            presets="newton",
            recorder_dataset_export_dir_path=str(output_dir / "arena-recordings"),
            recorder_dataset_filename="episodes",
            language_instruction=settings.instruction,
        ),
    )
    cfg, kwargs = builder.build_env_cfg()
    cfg.sim.device = settings.device
    cfg.compute_final_obs = True
    cfg.seed = settings.seed
    solver = cfg.sim.physics.solver_cfg
    solver.ls_iterations = 100
    solver.iterations = 100
    solver.cone = "elliptic"
    solver.nconmax = 32768
    solver.njmax = 32768
    solver.impratio = 10.0
    cfg.sim.physics.num_substeps = 2
    actuators = cfg.scene.robot.actuators
    names = {name for name in actuators if name.startswith("panda_")}
    if names != {"panda_shoulder", "panda_forearm"}:
        raise ValueError("Unexpected DROID arm actuator layout")
    for name in names:
        actuators[name].armature = settings.panda_armature
    return cfg, kwargs, camera_names
