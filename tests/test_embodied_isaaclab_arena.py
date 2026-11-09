from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from art_embodied.integrations.isaaclab_arena import (
    arena_newton_compatibility,
    resolve_arena_source_prim,
)
from examples.embodied.isaaclab.droid_environment import DroidSceneConfig


class Prim:
    def __init__(self, name, valid=True, dynamic=True):
        self.name, self.valid, self.dynamic = name, valid, dynamic

    def IsValid(self):
        return self.valid


class Stage:
    def __init__(self, prims):
        self.prims = prims

    def GetPrimAtPath(self, name):
        return self.prims.get(name, Prim(name, valid=False))


class Paths:
    @staticmethod
    def get_world_prototype_asset_templates(plan):
        return plan.templates, plan.starts

    @staticmethod
    def get_asset_prototype_paths(plan):
        return plan.sources

    @staticmethod
    def relative_to(child, parent):
        if child == parent or child.startswith(parent + "/"):
            return child[len(parent) :]
        return None


def plan():
    return SimpleNamespace(
        topology=SimpleNamespace(
            world_prototype_layout=np.array([0, 1, 0]),
            world_prototypes=np.array([0, 1, 2, 3]),
        ),
        starts=np.array([0, 1, 3, 4]),
        templates=(
            "/shared",
            "/env_{}/object",
            "/env_{}/object/child",
            "/env_{}/object",
        ),
        sources=("/shared", "/prototype/a", "/prototype/child", "/prototype/b"),
    )


def test_source_resolution_uses_world_prototype_and_longest_asset_prefix():
    stage = Stage(
        {
            name: Prim(name)
            for name in (
                "/prototype/a/link",
                "/prototype/child/link",
                "/prototype/b/link",
            )
        }
    )
    assert (
        resolve_arena_source_prim(
            stage, plan(), "/env_2/object/link", 2, path_api=Paths
        ).name
        == "/prototype/a/link"
    )
    assert (
        resolve_arena_source_prim(
            stage, plan(), "/env_1/object/link", 1, path_api=Paths
        ).name
        == "/prototype/b/link"
    )
    assert (
        resolve_arena_source_prim(
            stage, plan(), "/env_2/object/child/link", 2, path_api=Paths
        ).name
        == "/prototype/child/link"
    )


def test_existing_prim_needs_no_clone_plan():
    prim = Prim("/existing")
    assert (
        resolve_arena_source_prim(
            Stage({"/existing": prim}), None, "/existing", 0, path_api=Paths
        )
        is prim
    )


@pytest.mark.parametrize(
    "runtime,world",
    [
        ("/env_2/objectish/link", 2),
        ("/env_2/missing", 2),
        ("/env_3/object", 3),
        ("/env_2/object", -1),
    ],
)
def test_missing_metadata_is_an_error_not_a_skipped_reset(runtime, world):
    with pytest.raises(ValueError):
        resolve_arena_source_prim(Stage({}), plan(), runtime, world, path_api=Paths)


def test_missing_authored_source_rejected():
    with pytest.raises(ValueError, match="Missing authored"):
        resolve_arena_source_prim(
            Stage({}), plan(), "/env_2/object/link", 2, path_api=Paths
        )


def test_context_filters_dynamic_rows_preserves_order_and_restores(monkeypatch):
    torch = pytest.importorskip("torch")
    names = (
        "warp",
        "isaaclab",
        "isaaclab.cloner",
        "isaaclab_arena",
        "isaaclab_arena.environments",
        "isaaclab_arena.terms",
        "isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg",
        "isaaclab_arena.terms.events",
    )
    modules = {name: ModuleType(name) for name in names}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, attr = name.rpartition(".")
        if parent:
            setattr(modules[parent], attr, module)
    modules["warp"].to_torch = lambda value: value
    modules["isaaclab.cloner"].path = Paths
    events = modules["isaaclab_arena.terms.events"]
    arena_cfg = modules[
        "isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg"
    ]
    settings_calls, fallback_calls, writes = [], [], []
    original_settings = lambda: settings_calls.append(True)
    original_velocity = lambda *args: fallback_calls.append(args)
    arena_cfg.apply_arena_global_settings = original_settings
    events.write_root_velocity_to_sim = original_velocity

    class Rigid:
        __backend_name__ = "newton"

        def write_root_velocity_to_sim(self, value, *, env_ids):
            writes.append((value.clone(), env_ids.clone()))

    events.BaseRigidObject = Rigid
    events.is_rigid_body = lambda prim: True
    events.is_enabled_dynamic_rigid_body = lambda prim: prim.dynamic
    asset = Rigid()
    asset.root_view = SimpleNamespace(
        articulation_ids=torch.tensor([0, 1, 2]),
        model=SimpleNamespace(
            articulation_label=[f"/env_{i}/object/link" for i in range(3)]
        ),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            stage=Stage(
                {
                    "/prototype/a/link": Prim("a"),
                    "/prototype/b/link": Prim("b", dynamic=False),
                }
            ),
            clone_plan=plan(),
        )
    )
    cfg = SimpleNamespace(
        scene=SimpleNamespace(
            camera=SimpleNamespace(
                renderer_cfg=SimpleNamespace(renderer_type="newton_warp")
            )
        )
    )
    velocity = torch.arange(18).reshape(3, 6)
    ids = torch.tensor([2, 1, 0])
    with pytest.raises(RuntimeError, match="test teardown"):
        with arena_newton_compatibility(cfg, camera_names=("camera",)):
            arena_cfg.apply_arena_global_settings()
            assert not settings_calls
            events.write_root_velocity_to_sim(env, asset, velocity, ids)
            torch.testing.assert_close(writes[0][0], velocity[[0, 2]])
            torch.testing.assert_close(writes[0][1], torch.tensor([2, 0]))
            asset.__backend_name__ = "physx"
            events.write_root_velocity_to_sim(env, asset, velocity, ids)
            assert len(fallback_calls) == 1
            raise RuntimeError("test teardown")
    assert events.write_root_velocity_to_sim is original_velocity
    assert arena_cfg.apply_arena_global_settings is original_settings
    cfg.scene.camera.renderer_cfg.renderer_type = "rtx"
    with pytest.raises(ValueError, match="Newton/Warp"):
        with arena_newton_compatibility(cfg, camera_names=("camera",)):
            pass
    assert events.write_root_velocity_to_sim is original_velocity


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_envs": 0},
        {"instruction": ""},
        {"panda_armature": 0},
        {"episode_length_s": float("nan")},
    ],
)
def test_scene_configuration_rejects_invalid_contract(kwargs):
    values = dict(
        pick_up_object="cube",
        destination_location="bowl",
        instruction="Put the cube in the bowl.",
    )
    values.update(kwargs)
    with pytest.raises(ValueError):
        DroidSceneConfig(**values)
