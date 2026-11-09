"""Generate disjoint DROID scene banks without executing a policy."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--pick-up-object")
    parser.add_argument("--destination-location")
    parser.add_argument("--instruction")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--train-states", type=int, default=512)
    parser.add_argument("--development-states", type=int, default=128)
    parser.add_argument("--sealed-states", type=int, default=256)
    parser.add_argument("--panda-armature", type=float, default=0.1)

    import warp as wp

    wp.config.enable_backward = False
    from isaaclab.app import add_launcher_args, launch_simulation

    add_launcher_args(parser)
    parser.set_defaults(visualizer=[], device="cuda:0")
    args = parser.parse_args()
    task = (args.pick_up_object, args.destination_location, args.instruction)
    if args.recipe is not None and any(value is not None for value in task):
        parser.error("Use --recipe or explicit task arguments, not both")
    if args.recipe is None and not all(task):
        parser.error("Supply --recipe or all three task arguments")
    if args.recipe is not None and args.panda_armature != 0.1:
        parser.error("A recipe uses its qualified Panda armature of 0.1")
    sizes = {
        "train": args.train_states,
        "development": args.development_states,
        "sealed": args.sealed_states,
    }
    if args.num_envs < 1 or any(n < 1 or n % args.num_envs for n in sizes.values()):
        parser.error("Each bank size must be a positive multiple of num-envs")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import (
        IsaacLabArenaManagerBasedRLEnv,
    )

    from art_embodied.integrations.isaaclab import isaaclab_reset_fingerprints
    from art_embodied.integrations.isaaclab_arena import arena_newton_compatibility
    from art_embodied.integrations.isaaclab_reset_bank import IsaacLabResetBank
    from art_embodied.utils import write_json_atomic
    from examples.embodied.isaaclab.droid_environment import (
        DroidSceneConfig,
        build_droid_scene,
        load_droid_recipe,
    )

    runtime = dict(
        num_envs=args.num_envs,
        seed=args.seed,
        panda_armature=args.panda_armature,
        device=args.device,
    )
    settings = (
        DroidSceneConfig.from_recipe(load_droid_recipe(args.recipe), **runtime)
        if args.recipe is not None
        else DroidSceneConfig(*task, **runtime)
    )
    cfg, kwargs, cameras = build_droid_scene(settings, output_dir=args.output_dir)
    banks, next_seed = {}, args.seed
    with (
        arena_newton_compatibility(cfg, camera_names=cameras),
        launch_simulation(cfg, args),
    ):
        env = IsaacLabArenaManagerBasedRLEnv(cfg, **kwargs)
        try:
            for split, count in sizes.items():
                batches = []
                for _ in range(count // args.num_envs):
                    observation, _ = env.reset(seed=next_seed)
                    state = copy.deepcopy(env.scene.get_state(is_relative=True))
                    images = isaaclab_reset_fingerprints(
                        observation, num_envs=args.num_envs
                    )
                    replay, _ = env.reset_to(state, seed=next_seed, is_relative=True)
                    if images != isaaclab_reset_fingerprints(
                        replay, num_envs=args.num_envs
                    ):
                        raise RuntimeError(
                            "Initial observations changed on scene replay"
                        )
                    if isaaclab_reset_fingerprints(
                        state, num_envs=args.num_envs
                    ) != isaaclab_reset_fingerprints(
                        env.scene.get_state(is_relative=True), num_envs=args.num_envs
                    ):
                        raise RuntimeError("Physical state changed on scene replay")
                    batch = IsaacLabResetBank.from_batches(
                        [state], contract=settings.reset_contract(), split=split
                    )
                    batches.append(batch.select(list(range(args.num_envs))))
                    next_seed += 1
                bank = IsaacLabResetBank.from_batches(
                    batches, contract=settings.reset_contract(), split=split
                )
                for other in banks.values():
                    bank.assert_disjoint(other)
                bank.save(args.output_dir / split)
                banks[split] = bank
                print(
                    f"Created {split}: {len(bank)} distinct initial states", flush=True
                )
            write_json_atomic(
                args.output_dir / "report.json",
                {
                    "states": sizes,
                    "seed": args.seed,
                    "policy_evaluations": 0,
                    "disjoint": True,
                    "scene_and_observation_replay_exact": True,
                    "contract": settings.reset_contract(),
                },
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
