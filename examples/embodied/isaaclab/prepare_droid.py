"""Bind the DROID recipe to local model files and disjoint scene banks."""

from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import os
from pathlib import Path

from art_embodied.config import EmbodiedExperimentConfig
from art_embodied.integrations.isaaclab_reset_bank import IsaacLabResetBank
from examples.embodied.isaaclab.droid_environment import (
    DroidSceneConfig,
    load_droid_recipe,
)


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_config(
    *, checkpoint, banks, output_dir, policy_python, entity, num_gpus=8, recipe=None
):
    if num_gpus not in (2, 4, 8):
        raise ValueError("Use 2, 4 or 8 GPUs; the 512-trajectory geometry stays fixed")
    if not entity or entity == "WANDB_ENTITY":
        raise ValueError("Set the W&B entity explicitly")
    raw = load_droid_recipe(recipe)
    scene = DroidSceneConfig.from_recipe(raw)
    loaded = {
        split: IsaacLabResetBank.load(
            banks / split, contract=scene.reset_contract(), split=split
        )
        for split in ("train", "development", "sealed")
    }
    for a, b in combinations(loaded.values(), 2):
        a.assert_disjoint(b)
    for split, count in (("train", 512), ("development", 128), ("sealed", 256)):
        if len(loaded[split]) != count:
            raise ValueError(f"The reference recipe requires {count} {split} states")
    checkpoint, output_dir = checkpoint.resolve(), output_dir.resolve()
    raw["policy"]["path"] = str(checkpoint)
    raw["policy"]["load_kwargs"].update(
        checkpoint_sha256=file_sha256(checkpoint / "model.safetensors"),
        normalization_sha256=file_sha256(checkpoint / "assets/droid/norm_stats.json"),
    )
    raw["environment"]["kwargs"]["train_reset_bank_sha256"] = loaded[
        "train"
    ].fingerprint
    raw["evaluation"]["kwargs"]["reset_bank_sha256"] = loaded["development"].fingerprint
    raw["evaluation"]["fixed_scenarios"] = [f"development-{i}" for i in range(128)]
    raw["storage"]["output_dir"] = str(output_dir)
    devices = [f"cuda:{i}" for i in range(num_gpus)]
    raw["runtime"].update(
        rollout_devices=devices,
        training_devices=devices,
        # Resolving the executable symlink would bypass the virtual environment.
        worker_python_executable=str(policy_python.expanduser().absolute()),
        worker_handoff_dir=str(output_dir / "handoffs"),
    )
    raw["rollout"]["workers"] = num_gpus
    raw["training"]["schedule"].update(
        actor_world_size=num_gpus,
        groups_per_process_per_rollout_epoch=64 // num_gpus,
    )
    raw["observability"]["wandb"]["entity"] = entity
    return EmbodiedExperimentConfig.model_validate(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-python", type=Path, required=True)
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--num-gpus", type=int, choices=(2, 4, 8), default=8)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--recipe", type=Path, help="YAML recipe; defaults to soup-to-bowl"
    )
    args = parser.parse_args()
    config = build_config(
        checkpoint=args.checkpoint,
        banks=args.banks,
        output_dir=args.output_dir,
        policy_python=args.policy_python,
        entity=args.entity,
        num_gpus=args.num_gpus,
        recipe=args.recipe,
    )
    with args.config.open("x") as stream:
        stream.write(config.model_dump_json(indent=2) + "\n")
    print(args.config)


if __name__ == "__main__":
    main()
