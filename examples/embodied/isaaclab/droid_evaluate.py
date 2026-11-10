"""Evaluate DROID on distinct Arena placements using an isolated OpenPI server."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time


def bank_indices(*, count, start=0, selected=None, bank_size=None):
    indices = list(range(start, start + count)) if selected is None else selected
    if (
        len(indices) != count
        or any(type(i) is not int or i < 0 for i in indices)
        or len(set(indices)) != count
        or (bank_size is not None and any(i >= bank_size for i in indices))
    ):
        raise ValueError(
            "Reset indices must be distinct, in range, and match the episode count"
        )
    return indices


def policy_seed_base(seed, bank_index, attempt, group_size):
    # Stable across actor counts and partitioning of the same reset bank.
    return (seed * 1000 + bank_index * group_size + attempt) * 10000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--pick-up-object", required=True)
    parser.add_argument("--destination-location", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--panda-armature", type=float, default=0.1)
    parser.add_argument("--mode", choices=("ode", "sde"), default="ode")
    parser.add_argument("--reset-bank", type=Path)
    parser.add_argument("--split", choices=("train", "development", "sealed"))
    parser.add_argument("--bank-start", type=int, default=0)
    parser.add_argument("--bank-indices", type=Path)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--policy-version", type=int, default=0)

    import warp as wp

    wp.config.enable_backward = False
    from isaaclab.app import add_launcher_args, launch_simulation

    add_launcher_args(parser)
    parser.set_defaults(visualizer=[], device="cuda:0")
    args = parser.parse_args()
    if args.num_envs < 1 or args.episodes < 1 or args.episodes % args.num_envs:
        parser.error("episodes must be a positive multiple of num-envs")
    if args.group_size < 1 or args.num_envs % args.group_size or args.bank_start < 0:
        parser.error(
            "group-size must divide num-envs and bank-start must be nonnegative"
        )
    if bool(args.reset_bank) != bool(args.split):
        parser.error("reset-bank and split must be specified together")
    if args.group_size > 1 and args.reset_bank is None:
        parser.error("Grouped attempts require an explicit reset bank")
    if args.bank_indices and (args.reset_bank is None or args.bank_start != 0):
        parser.error(
            "bank-indices requires a reset bank and cannot be combined with bank-start"
        )
    if args.mode == "sde" and args.split != "train":
        parser.error("SDE collection requires a training bank")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    from isaaclab_arena.embodiments.droid.observations import (
        arm_joint_pos,
        newton_gripper_pos,
    )
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import (
        IsaacLabArenaManagerBasedRLEnv,
    )
    import numpy as np
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    import torch

    from art_embodied.integrations.isaaclab import (
        IsaacLabChunkEnvironment,
        isaaclab_reset_fingerprints,
    )
    from art_embodied.integrations.isaaclab_arena import arena_newton_compatibility
    from art_embodied.integrations.isaaclab_droid import (
        droid_action_chunk,
        pack_droid_observation,
    )
    from art_embodied.integrations.isaaclab_reset_bank import IsaacLabResetBank
    from art_embodied.media import RolloutVideoRecorder
    from art_embodied.utils import write_json_atomic
    from examples.embodied.isaaclab.droid_environment import (
        DroidSceneConfig,
        build_droid_scene,
    )

    settings = DroidSceneConfig(
        pick_up_object=args.pick_up_object,
        destination_location=args.destination_location,
        instruction=args.instruction,
        num_envs=args.num_envs,
        seed=args.seed,
        panda_armature=args.panda_armature,
        device=args.device,
    )
    cfg, kwargs, camera_names = build_droid_scene(settings, output_dir=args.output_dir)
    client = WebsocketClientPolicy(host="127.0.0.1", port=args.port)
    metadata = client.get_server_metadata()
    if metadata["mode"] != args.mode or metadata["batch_size"] != args.num_envs:
        raise ValueError("Policy mode or batch size differs from the request")
    if metadata["training_updates"] != args.policy_version:
        raise ValueError("Policy version differs from the request")
    bank = (
        IsaacLabResetBank.load(
            args.reset_bank, contract=settings.reset_contract(), split=args.split
        )
        if args.reset_bank
        else None
    )
    selected_indices = bank_indices(
        count=args.episodes // args.group_size,
        start=args.bank_start,
        selected=json.loads(args.bank_indices.read_text())
        if args.bank_indices
        else None,
        bank_size=len(bank) if bank else None,
    )
    started = time.monotonic()
    episodes, timings, reset_states, actions_log = [], [], [], []

    def cpu_tree(value):
        if isinstance(value, dict):
            return {key: cpu_tree(child) for key, child in value.items()}
        return value.detach().cpu().clone()

    with (
        arena_newton_compatibility(cfg, camera_names=camera_names),
        launch_simulation(cfg, args),
    ):
        raw = IsaacLabArenaManagerBasedRLEnv(cfg, **kwargs)
        env = IsaacLabChunkEnvironment(
            raw,
            execution_horizon=15,
            max_environment_steps=1050,
            success_fn=lambda e, *_: e.termination_manager.get_term("success"),
            inactive_action_fn=lambda e: torch.cat(
                [arm_joint_pos(e), newton_gripper_pos(e)], dim=-1
            ),
        )
        try:
            for batch in range(args.episodes // args.num_envs):
                seed = args.seed + batch
                indices = [
                    selected_indices[(batch * args.num_envs + row) // args.group_size]
                    for row in range(args.num_envs)
                ]
                if bank is None:
                    observation, _ = env.reset(seed=seed)
                else:
                    state = bank.select(indices, device=args.device)
                    observation, _ = env.reset(
                        seed=seed,
                        reset_fn=lambda e, s: e.reset_to(
                            state, seed=s, is_relative=True
                        ),
                    )
                state = raw.scene.get_state(is_relative=True)
                hashes = isaaclab_reset_fingerprints(state, num_envs=args.num_envs)
                if bank and hashes != tuple(bank.hashes[i] for i in indices):
                    raise RuntimeError("Restored scene differs from the reset bank")
                image_hashes = isaaclab_reset_fingerprints(
                    observation, num_envs=args.num_envs
                )
                for start in range(0, args.num_envs, args.group_size):
                    if len(set(image_hashes[start : start + args.group_size])) != 1:
                        raise RuntimeError(
                            "Grouped attempts have different initial observations"
                        )
                reset_states.append(cpu_tree(state))
                # Verify replay before using this layout for a policy evaluation.
                observation, _ = env.reset(
                    seed=seed,
                    reset_fn=lambda e, s: e.reset_to(state, seed=s, is_relative=True),
                )
                if hashes != isaaclab_reset_fingerprints(
                    raw.scene.get_state(is_relative=True), num_envs=args.num_envs
                ):
                    raise RuntimeError("Physical state changed on reset replay")
                if image_hashes != isaaclab_reset_fingerprints(
                    observation, num_envs=args.num_envs
                ):
                    raise RuntimeError("Initial camera images changed on reset replay")
                active = list(range(args.num_envs))
                recorders = [
                    RolloutVideoRecorder(
                        args.output_dir,
                        filename_prefix=f"episode-{batch * args.num_envs + row}",
                        fps=1,
                        max_frames=72,
                    )
                    for row in active
                ]

                def pack(row):
                    request = pack_droid_observation(
                        {
                            group: {key: value[row] for key, value in values.items()}
                            for group, values in observation.items()
                        },
                        prompt=settings.instruction,
                    )
                    recorders[row].capture_frame(
                        np.concatenate(
                            [
                                request["observation/exterior_image_1_left"],
                                request["observation/wrist_image_left"],
                            ],
                            axis=1,
                        )
                    )
                    return request

                for chunk in range(70):
                    requests = []
                    for row in active:
                        request = pack(row)
                        request["sampling_seed"] = (
                            policy_seed_base(
                                args.seed,
                                indices[row],
                                row % args.group_size,
                                args.group_size,
                            )
                            + chunk
                        )
                        requests.append(request)
                    tick = time.monotonic()
                    result = client.infer({"batch": requests})
                    inference_seconds = time.monotonic() - tick
                    if result["mode"] != args.mode or len(result["actions"]) != len(
                        active
                    ):
                        raise RuntimeError(
                            "Policy response differs from requested batch"
                        )
                    commands = np.zeros((args.num_envs, 15, 8), dtype=np.float32)
                    for index, row in enumerate(active):
                        commands[row] = droid_action_chunk(
                            result["actions"][index], execution_horizon=15
                        )
                    tick = time.monotonic()
                    observation, _, terminated, truncated, info = env.step(commands)
                    timings.append(
                        [
                            inference_seconds,
                            time.monotonic() - tick,
                            int(info["primitive_loss_mask"].sum()),
                        ]
                    )
                    actions_log.append(
                        {
                            "batch": batch,
                            "chunk": chunk,
                            "active_rows": active.copy(),
                            "actions": commands[active].tolist(),
                            "execution_mask": info["primitive_loss_mask"][active]
                            .cpu()
                            .tolist(),
                            "record_id": result.get("record_id"),
                            "old_chunk_logprobs": result.get(
                                "old_chunk_logprobs", np.array([])
                            ).tolist(),
                            "logprob_max_error": result.get("logprob_max_error"),
                        }
                    )
                    next_active = []
                    for row in active:
                        if bool((terminated | truncated)[row]):
                            pack(row)
                            episodes.append(
                                {
                                    "episode": batch * args.num_envs + row,
                                    "batch": batch,
                                    "row": row,
                                    "bank_index": indices[row] if bank else None,
                                    "group": (batch * args.num_envs + row)
                                    // args.group_size,
                                    "success": bool(info["success"][row]),
                                    "terminated": bool(terminated[row]),
                                    "truncated": bool(truncated[row]),
                                    "termination_reason": "success"
                                    if bool(info["success"][row])
                                    else (
                                        "timeout"
                                        if bool(truncated[row])
                                        else "terminated"
                                    ),
                                    "env_steps": int(info["env_steps"][row]),
                                    "reset_sha256": hashes[row],
                                    "observation_sha256": image_hashes[row],
                                    "environment_seed": seed,
                                    "policy_seed_base": policy_seed_base(
                                        args.seed,
                                        indices[row],
                                        row % args.group_size,
                                        args.group_size,
                                    ),
                                    "media": [
                                        item.model_dump(mode="json")
                                        for item in recorders[row].finalize()
                                    ],
                                }
                            )
                        else:
                            next_active.append(row)
                    active = next_active
                    write_json_atomic(args.output_dir / "episodes.json", episodes)
                    print(
                        json.dumps(
                            {
                                "batch": batch,
                                "chunk": chunk,
                                "active": len(active),
                                "episodes": len(episodes),
                                "successes": sum(e["success"] for e in episodes),
                            }
                        ),
                        flush=True,
                    )
                    if not active:
                        break
                if active:
                    raise RuntimeError("Evaluation exceeded its episode horizon")
            distinct = len({episode["reset_sha256"] for episode in episodes})
            if distinct != args.episodes // args.group_size:
                raise RuntimeError(
                    "Evaluation placement sampler repeated an initial state"
                )
            torch.save(reset_states, args.output_dir / "initial-states.pt")
            write_json_atomic(args.output_dir / "actions.json", actions_log)
            write_json_atomic(
                args.output_dir / "report.json",
                {
                    "settings": asdict(settings),
                    "model_server": metadata,
                    "episodes": len(episodes),
                    "distinct_initial_states": distinct,
                    "group_size": args.group_size,
                    "split": args.split,
                    "successes": sum(e["success"] for e in episodes),
                    "optimizer_updates": args.policy_version,
                    "elapsed_seconds": time.monotonic() - started,
                    "inference_seconds": sum(t[0] for t in timings),
                    "simulation_seconds": sum(t[1] for t in timings),
                    "primitive_steps": sum(t[2] for t in timings),
                    "initial_state_and_camera_replay_exact": True,
                    "scope": "DROID pick-and-place with modified Panda armature",
                },
            )
        except Exception as error:
            write_json_atomic(
                args.output_dir / "failure.json",
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "completed_episodes": len(episodes),
                },
            )
            raise
        finally:
            env.close()


if __name__ == "__main__":
    main()
