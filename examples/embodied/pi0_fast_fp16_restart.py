"""Fresh FAST residual-FP16 SFT/GRPO, retaining the measured parallel workload."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import yaml

from art_embodied.config import EmbodiedExperimentConfig
from art_embodied.evaluation import validate_paired_evaluation_configs
from examples.embodied.libero.state_manifest import file_sha256
from examples.embodied.pi0_fast_native_precision_restart import configure


def plan(root, source):
    if root.exists():
        raise ValueError("Require a new experiment directory; no resume or SFT reuse")
    configs, provenance = {}, {}
    for phase in ("sft", "grpo", "sealed-baseline", "sealed-candidate"):
        path = source / (phase + ".yaml")
        raw = configure(yaml.safe_load(path.read_text()), root=root, phase=phase)
        raw["policy"]["load_kwargs"].update(
            model_compute_dtype="fp16_residual", training_loss_scale=128.0
        )
        raw["experiment"]["tags"] = [
            tag for tag in raw["experiment"]["tags"] if tag != "checkpoint-precision"
        ] + ["fp16-residual", "static-loss-scale-128"]
        if phase == "grpo":
            raw["training"]["microbatch_size"] = 8
            raw["algorithm"]["train_logprob_microbatch_size"] = 8
            raw["algorithm"]["logprob_microbatch_size"] = 8
        configs[phase] = raw
        EmbodiedExperimentConfig.model_validate(raw)
        provenance[phase] = {"recipe": str(path), "sha256": file_sha256(path)}
    validate_paired_evaluation_configs(
        EmbodiedExperimentConfig.model_validate(configs["sealed-baseline"]),
        EmbodiedExperimentConfig.model_validate(configs["sealed-candidate"]),
    )
    root.mkdir(parents=True)
    for phase, raw in configs.items():
        (root / (phase + ".yaml")).write_text(yaml.safe_dump(raw, sort_keys=False))
    # A small same-implementation first-update qualification precedes the real
    # 960-trajectory workload. It never substitutes for its throughput timing.
    qualification = deepcopy(configs["grpo"])
    qualification["policy"]["load_kwargs"]["warm_start_checkpoint"] = None
    (root / "precision-qualification.yaml").write_text(
        yaml.safe_dump(qualification, sort_keys=False)
    )
    (root / "plan.json").write_text(
        json.dumps(
            {
                "restart": "fresh base SFT and GRPO, no prior adapters or logprobs",
                "precision": "FP16 frozen operands; FP32 down outputs/residual/norm weights/vision/LoRA",
                "training_loss_scale": 128,
                "sft_updates": 400,
                "sft_tasks": list(range(10)),
                "sft_global_frames_per_update": 80,
                "grpo_tasks": [8],
                "grpo_updates": 100,
                "trajectories_per_update": 960,
                "group_size": 8,
                "validation_episodes": 100,
                "validation_interval": 5,
                "sealed_policy_evaluated": False,
                "source_recipes_only_not_checkpoints": provenance,
                "performance_continuity": {
                    "retained": [
                        "one node, eight GPUs",
                        "32 rollout actors",
                        "KV decode, per-row EOS termination",
                        "rollout batch8",
                        "full-sequence teacher-forcing batch8",
                        "all-GPU gradient training",
                        "no all-model FP32",
                        "all tokens and 960 trajectories",
                        "standard W&B axes/media/artifacts",
                    ],
                    "changed": [
                        "distinct FP16-residual numerical policy and fresh SFT",
                        "loss scale128 before backward, unscale before handoff",
                    ],
                    "unverified": [
                        "new profile full-update throughput",
                        "new profile peak rollout VRAM",
                        "learning lift",
                        "App rendering",
                    ],
                },
                "continuation": "GPU production conformance -> SFT first-update live W&B gate -> SFT400 -> GRPO first-update live gate -> continue in same allocation",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    plan(args.root.resolve(), args.source.resolve())
