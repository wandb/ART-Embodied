"""Check the Isaac Lab 3.0 vector/reset/media boundary with official Cartpole.

Run from an isolated Isaac Lab 3.0 environment, with this checkout on PYTHONPATH.
This exercises the simulator adapter, not VLA inference or learning performance.
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera", action="store_true")

    import warp as wp

    wp.config.enable_backward = False
    from isaaclab.app import add_launcher_args, launch_simulation

    add_launcher_args(parser)
    parser.set_defaults(visualizer=[], device="cuda:0")
    args = parser.parse_args()
    if args.num_envs < 2 or args.max_steps < 4:
        parser.error("Use at least two environments and four primitive steps")

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    import torch

    from art_embodied.integrations.isaaclab import (
        IsaacLabChunkEnvironment,
        isaaclab_reset_fingerprints,
        repeat_isaaclab_scene_state,
    )
    from art_embodied.media import RolloutVideoRecorder

    task = "Isaac-Cartpole-Camera" if args.camera else "Isaac-Cartpole"
    overrides = ["physics=newton_mjwarp"]
    if args.camera:
        overrides.extend(["renderer=newton_renderer", "presets=rgb"])
    cfg = parse_env_cfg(
        task, device=args.device, num_envs=args.num_envs, overrides=overrides
    )
    cfg.compute_final_obs = True
    cfg.seed = 17
    cfg.episode_length_s = args.max_steps * cfg.sim.dt * cfg.decimation
    if args.camera:
        cfg.scene.tiled_camera.width = 128
        cfg.scene.tiled_camera.height = 128
        cfg.scene.tiled_camera.update_latest_camera_pose = True
        # This check records RGB, not the mean-centered training input of the
        # default Cartpole policy. Preserve raw pixels for media and replay.
        cfg.observations.policy.image.params["normalize"] = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with launch_simulation(cfg, args):
        raw = gym.make(task, cfg=cfg).unwrapped
        env = IsaacLabChunkEnvironment(
            raw,
            execution_horizon=4,
            max_environment_steps=args.max_steps + 4,
            # Cartpole is only a simulator conformance probe. No task-success
            # claim or learning signal is derived from its shaped reward.
            success_fn=lambda *_: torch.zeros(
                args.num_envs, dtype=torch.bool, device=raw.device
            ),
        )
        try:
            raw.reset(seed=17)
            state = repeat_isaaclab_scene_state(
                raw.scene.get_state(is_relative=True), count=args.num_envs
            )
            fingerprints = []
            observation_fingerprints = []
            terminal_steps = []
            total_transitions = 0
            image_ranges = []
            for replay in range(2):
                observation, _ = env.reset(
                    seed=17,
                    reset_fn=lambda raw, seed: raw.reset_to(
                        state, seed=seed, is_relative=True
                    ),
                )
                hashes = isaaclab_reset_fingerprints(
                    raw.scene.get_state(is_relative=True), num_envs=args.num_envs
                )
                if len(set(hashes)) != 1:
                    raise RuntimeError("Cloned Cartpole reset states differ")
                fingerprints.append(hashes[0])
                observation_fingerprints.append(
                    isaaclab_reset_fingerprints(observation, num_envs=args.num_envs)
                )
                recorder = (
                    RolloutVideoRecorder(
                        args.output_dir,
                        filename_prefix=f"cartpole-replay-{replay}",
                        fps=10,
                        max_frames=args.max_steps + 1,
                    )
                    if args.camera
                    else None
                )

                def capture(observation):
                    if recorder is not None:
                        # The official RGB task stacks two CHW frames. Take the
                        # newest three channels from the returned observation,
                        # including final_obs on the terminal transition.
                        image = observation["policy"][0, -3:].permute(1, 2, 0)
                        if (
                            not torch.isfinite(image).all()
                            or image.max() == image.min()
                        ):
                            raise RuntimeError(
                                "Camera returned a nonfinite or constant image"
                            )
                        image_ranges.append([float(image.min()), float(image.max())])
                        recorder.capture_frame(image.cpu().numpy())

                capture(observation)
                commands = torch.zeros(
                    args.num_envs, 4, env.action_dim, device=raw.device
                )
                for _ in range(args.max_steps + 4):
                    observation, _, terminated, truncated, info = env.step(commands)
                    capture(observation)
                    total_transitions += int(info["primitive_loss_mask"].sum())
                    if (terminated | truncated).all():
                        terminal_steps.append(info["env_steps"].cpu().tolist())
                        break
                else:
                    raise RuntimeError("Conformance episodes did not terminate")
                if recorder is not None:
                    media = recorder.finalize()
                    if not media or any(item.kind != "video" for item in media):
                        raise RuntimeError("Conformance video was not produced")
            if fingerprints[0] != fingerprints[1]:
                raise RuntimeError("Reset replay changed the initial physical state")
            if observation_fingerprints[0] != observation_fingerprints[1]:
                raise RuntimeError("Reset replay changed initial observations")
            report = {
                "task": task,
                "num_envs": args.num_envs,
                "physics": "newton_mjwarp",
                "camera": args.camera,
                "reset_fingerprints": fingerprints,
                "observation_replay_identical": True,
                "terminal_steps": terminal_steps,
                "primitive_transitions": total_transitions,
                "image_ranges": image_ranges,
                "wall_seconds": time.monotonic() - started,
                "runtime_versions": {
                    name: version(name)
                    for name in ("isaaclab", "torch", "warp-lang", "newton")
                },
                "scope": "simulator conformance only; no VLA learning qualification",
            }
            (args.output_dir / "conformance.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(json.dumps(report))
        finally:
            env.close()


if __name__ == "__main__":
    main()
