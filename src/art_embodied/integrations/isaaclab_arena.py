"""Scoped compatibility for Arena 0.3 with Isaac Lab's Newton/Warp backend.

Arena's renderer initialization still assumes RTX, and rigid-body reset metadata
is looked up at instance paths that Newton may not author in USD. This adapter
resolves clone sources without modifying simulator code or physical state.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import numpy as np


def resolve_arena_source_prim(
    stage: Any, plan: Any, runtime_path: str, world: int, *, path_api: Any
) -> Any:
    """Resolve clone metadata at its authored source, preserving world identity."""
    direct = stage.GetPrimAtPath(runtime_path)
    if direct.IsValid():
        return direct
    if plan is None:
        raise ValueError(f"Missing clone plan for {runtime_path}")
    topology = plan.topology
    layout = topology.world_prototype_layout
    if not isinstance(layout, np.ndarray) or not 0 <= world < len(layout):
        raise ValueError("Invalid world in Arena clone metadata lookup")
    prototype = int(layout[world])
    templates, starts = path_api.get_world_prototype_asset_templates(plan)
    sources = path_api.get_asset_prototype_paths(plan)
    if not 0 <= prototype < len(starts) - 2:
        raise ValueError("Invalid world prototype in Arena clone plan")
    candidates = []
    # starts[0:2] belongs to the shared world (-1), followed by world prototypes.
    for index in range(int(starts[prototype + 1]), int(starts[prototype + 2])):
        destination = templates[index].format(world)
        suffix = path_api.relative_to(runtime_path, destination)
        if suffix is not None:
            source = sources[int(topology.world_prototypes[index])]
            if source is not None:
                candidates.append((len(destination), source + suffix))
    if not candidates:
        raise ValueError(f"No authored clone source for {runtime_path}")
    longest = max(length for length, _ in candidates)
    selected = {source for length, source in candidates if length == longest}
    if len(selected) != 1:
        raise ValueError(f"Ambiguous clone source for {runtime_path}")
    prim = stage.GetPrimAtPath(selected.pop())
    if not prim.IsValid():
        raise ValueError(f"Missing authored clone source for {runtime_path}")
    return prim


@contextmanager
def arena_newton_compatibility(cfg: Any, *, camera_names: tuple[str, ...]):
    """Use around construction, resets, and teardown of one Warp-only Arena env.

    Arena uses process-global functions, so this context must run in a dedicated
    simulator process, without concurrent environment construction. Originals are
    restored on exit, including exceptions. No solver tolerances are changed.
    """
    from isaaclab.cloner import path
    import isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg as arena_cfg
    import isaaclab_arena.terms.events as events
    import torch
    import warp as wp

    if not camera_names or any(
        getattr(cfg.scene, name).renderer_cfg.renderer_type != "newton_warp"
        for name in camera_names
    ):
        raise ValueError("Arena compatibility requires explicit Newton/Warp cameras")
    original_settings = arena_cfg.apply_arena_global_settings
    original_velocity = events.write_root_velocity_to_sim

    def write_velocity(env, asset, velocity, env_ids):
        if (
            not isinstance(asset, events.BaseRigidObject)
            or asset.__backend_name__ != "newton"
        ):
            return original_velocity(env, asset, velocity, env_ids)
        if velocity.shape != (len(env_ids), 6) or env_ids.ndim != 1:
            raise ValueError("Velocity rows must correspond to env_ids")
        view = asset.root_view
        indices = wp.to_torch(view.articulation_ids).flatten()[env_ids].tolist()
        dynamic = []
        for row, (index, world) in enumerate(
            zip(indices, env_ids.tolist(), strict=True)
        ):
            prim = resolve_arena_source_prim(
                env.scene.stage,
                env.scene.clone_plan,
                view.model.articulation_label[index],
                world,
                path_api=path,
            )
            if not events.is_rigid_body(prim):
                raise ValueError("Arena reset source is not a rigid body")
            if events.is_enabled_dynamic_rigid_body(prim):
                dynamic.append(row)
        if dynamic:
            selected = torch.tensor(dynamic, dtype=torch.long, device=env_ids.device)
            asset.write_root_velocity_to_sim(
                velocity[selected], env_ids=env_ids[selected]
            )

    def warp_settings():
        # Arena's global settings only configure RTX/PhysX, neither used here.
        return None

    arena_cfg.apply_arena_global_settings = warp_settings
    events.write_root_velocity_to_sim = write_velocity
    try:
        yield
    finally:
        events.write_root_velocity_to_sim = original_velocity
        arena_cfg.apply_arena_global_settings = original_settings
