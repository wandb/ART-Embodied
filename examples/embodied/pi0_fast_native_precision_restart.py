"""Fresh checkpoint-precision SFT; never adopt or resume an FP32 experiment."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import yaml

from art_embodied.config import EmbodiedExperimentConfig
from examples.embodied.libero.state_manifest import file_sha256, load_state_manifest


def configure_rollout_parallelism(raw, *, actors_per_device=4, inference_batch_size=8):
    """Pair actor capacity with admission limits; leave learning scope unchanged."""
    raw = deepcopy(raw)
    if raw["policy"]["type"] != "pi0_fast":
        raise ValueError("This measured runtime profile is specific to pi0-FAST")
    if raw["policy"]["load_kwargs"]["model_compute_dtype"] != "checkpoint":
        raise ValueError("The measured actor capacity requires checkpoint precision")
    if actors_per_device not in (2, 4) or inference_batch_size != 8:
        raise ValueError("Use the measured multi-actor, batch8 diagnostic profiles")
    execution = raw["runtime"]["rollout_execution"]
    if (
        execution["mode"] != "local_process"
        or execution["inference_mode"] != "embedded"
    ):
        raise ValueError("This profile requires process-isolated embedded inference")
    devices = raw["runtime"]["rollout_devices"]
    if not 1 <= len(devices) <= 8 or len(set(devices)) != len(devices):
        raise ValueError("Expected one node with one to eight distinct rollout devices")
    workers = len(devices) * actors_per_device
    raw["rollout"]["workers"] = workers
    execution["actors_per_device"] = actors_per_device
    execution["inference_max_batch_size"] = inference_batch_size
    if "max_concurrent_rollouts" in execution["actor_kwargs"]:
        execution["actor_kwargs"]["max_concurrent_rollouts"] = workers
    EmbodiedExperimentConfig.model_validate(raw)
    return raw


def configure(raw, *, root, phase):
    raw = deepcopy(raw)
    raw["experiment"]["run"] = root.name + "-" + phase
    raw["experiment"]["tags"] += ["checkpoint-precision", "fresh-sft-grpo"]
    raw["policy"]["load_kwargs"].update(
        model_compute_dtype="checkpoint",
        warm_start_checkpoint=None if phase == "sft" else str(root / "sft-warm-start"),
    )
    raw["storage"].update(
        output_dir=str(root / phase / "runtime")
        if phase == "sft"
        else str(root / phase),
        resume_from_checkpoint=None,
        allow_legacy_checkpoint_resume=False,
    )
    raw["runtime"]["worker_handoff_dir"] = str(root / (phase + "-handoffs"))
    raw["observability"]["wandb"].update(group=root.name, run_id=None, resume=None)
    if phase == "sealed-candidate":
        raw["evaluation"]["baseline_outcomes_path"] = str(
            root / "sealed-baseline/evaluation/update_000000_episode_outcomes.json"
        )
    if phase == "grpo":
        raw = configure_rollout_parallelism(raw)
    # No performance result qualifies on-policy likelihoods. GRPO is gated
    # separately; retain the existing objective and workload in this plan.
    EmbodiedExperimentConfig.model_validate(raw)
    return raw


def plan(root, source):
    if root.exists():
        raise ValueError("Fresh restart requires a new output directory")
    sft_source = Path(json.loads((source / "source.json").read_text())["root"])
    configs = {}
    provenance = {}
    for phase in ("sft", "grpo", "sealed-baseline", "sealed-candidate"):
        path = (sft_source if phase == "sft" else source) / (phase + ".yaml")
        configs[phase] = configure(
            yaml.safe_load(path.read_text()), root=root, phase=phase
        )
        provenance[phase] = {"recipe": str(path), "sha256": file_sha256(path)}
    from art_embodied.evaluation import validate_paired_evaluation_configs

    validate_paired_evaluation_configs(
        EmbodiedExperimentConfig.model_validate(configs["sealed-baseline"]),
        EmbodiedExperimentConfig.model_validate(configs["sealed-candidate"]),
    )
    states = {}
    for phase, raw in configs.items():
        path = Path(raw["environment"]["kwargs"]["evaluation_state_manifest"])
        manifest = load_state_manifest(path)
        states[phase] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "states": len(manifest.entries),
        }
    root.mkdir(parents=True)
    for phase, raw in configs.items():
        (root / (phase + ".yaml")).write_text(yaml.safe_dump(raw, sort_keys=False))
    (root / "plan.json").write_text(
        json.dumps(
            {
                "restart": "fresh SFT and GRPO; no previous adapter, optimizer, RNG, or run ID",
                "sft_updates": 400,
                "sft_tasks": list(range(10)),
                "sft_global_frames_per_update": 80,
                "grpo_tasks": [8],
                "grpo_updates": 100,
                "trajectories_per_update": 960,
                "precision": "checkpoint-native mixed BF16/FP32; FP32 trainable LoRA",
                "grpo_launch": "disabled pending native sampling/training likelihood qualification",
                "sealed_policy_evaluated": False,
                "source_recipes_only_not_checkpoints": provenance,
                "evaluation_manifests": states,
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
