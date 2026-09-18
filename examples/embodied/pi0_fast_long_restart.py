"""Isolated shared-LoRA Long SFT/GRPO plan; no global simulator overrides."""

import argparse
import asyncio
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from importlib import metadata
import json
from pathlib import Path
import shutil
import sys

import yaml

from art_embodied.config import EmbodiedExperimentConfig
from examples.embodied.pi0_fast_spatial_teacher_control import (
    diagnostic_cuda_mapping,
    write,
)


def check_runtime():
    import mujoco

    from art_embodied.compatibility import (
        VALIDATED_PI0_FAST_PACKAGES,
        require_compatible_worker_runtime,
    )

    require_compatible_worker_runtime()
    expected = dict(VALIDATED_PI0_FAST_PACKAGES, mujoco="3.3.0")
    actual = {p: metadata.version(p).split("+", 1)[0] for p in expected}
    if (
        actual != expected
        or mujoco.mj_versionString() != "3.3.0"
        or metadata.version("lerobot") != "0.6.0"
    ):
        raise ValueError(
            f"Isolated FAST runtime mismatch: {actual}; expected {expected}"
        )
    return actual


@contextmanager
def isolated_art_runtime():
    """Check this experimental profile without changing published policy profiles."""
    import art_embodied.art_compat as lifecycle

    original = lifecycle.require_compatible_runtime

    def checked(*, require_lerobot=False, profile="control"):
        if profile != "pi0_fast":
            return original(require_lerobot=require_lerobot, profile=profile)
        original(profile="control")
        check_runtime()

    lifecycle.require_compatible_runtime = checked
    try:
        yield
    finally:
        lifecycle.require_compatible_runtime = original


def adopt_sft(root, source):
    from art_embodied.checkpointing import CheckpointManager

    source = source.resolve()
    completed = json.loads((source / "sft/complete.json").read_text())
    verified = json.loads((source / "sft/verified-0401.json").read_text())
    if completed["updates"] != 400 or not (
        verified["history_verified"] and verified["artifact_verified"]
    ):
        raise ValueError("Require completed, verified multitask SFT400")
    old = EmbodiedExperimentConfig.from_yaml(source / "sft.yaml")
    new = EmbodiedExperimentConfig.from_yaml(root / "sft.yaml")
    for name in ("policy", "environment", "evaluation"):
        if getattr(old, name) != getattr(new, name):
            raise ValueError(f"SFT reuse changes {name}")
    if old.experiment.seed != new.experiment.seed:
        raise ValueError("SFT reuse changes seed")
    config = EmbodiedExperimentConfig.from_yaml(root / "grpo.yaml")
    CheckpointManager().publish(
        root / "sft-warm-start",
        writer=lambda staging: shutil.copytree(
            source / "sft/checkpoint-0400", staging / "policy"
        ),
        config_fingerprint=config.fingerprint,
        resume_contract_fingerprint=config.resume_contract_fingerprint,
        metadata={
            "backend": "pi0_fast_language_sft",
            "step": 400,
            "source": str(source),
            "shared_multitask": True,
        },
    )
    (root / "sft").symlink_to(source / "sft", target_is_directory=True)
    write(root / "sft-source.json", {"root": str(source), "verified": verified})


def plan(root, *, sealed_manifest=None):
    from examples.embodied.libero.components import (
        build_evaluation_scenarios,
        build_train_scenarios,
    )
    from examples.embodied.libero.state_manifest import file_sha256, load_state_manifest

    root.mkdir(parents=True, exist_ok=True)
    source = Path(
        "examples/embodied/pi0_fast_spatial_native_score_sum_minibatch_development_20260906.yaml"
    )
    old_long = yaml.safe_load(
        Path(
            "examples/embodied/pi0_fast_libero_long_task_balanced_grpo_development_h100.yaml"
        ).read_text()
    )
    raw = yaml.safe_load(source.read_text())
    raw["experiment"].update(
        project="art-embodied-pi0-fast-long",
        run=root.name + "-grpo",
        seed=20260909,
        tags=[
            "pi0-fast",
            "libero-long",
            "shared-lora-r32",
            "mujoco-3.3.0",
            "sft-then-grpo",
            "one-node",
            "native-decoder",
            "full-token-logprob",
        ],
    )
    raw["environment"] = deepcopy(old_long["environment"])
    env = raw["environment"]["kwargs"]
    env.update(
        wait_steps_after_reset=15,
        control_mode="unchanged",
        training_trial_ids=list(range(40)),
        evaluation_protocol="rlinf_v01",
        evaluation_trial_ids=list(range(40, 50)),
    )
    env.pop("sft_anchor_task_ids", None)
    raw["policy"]["load_kwargs"].update(
        sft_anchor=None, warm_start_checkpoint=str(root / "sft-warm-start")
    )
    raw["policy"]["lora"].update(rank=32, alpha=32, rank_partition=None)
    raw["rollout"].update(
        groups_per_update=120, max_episode_steps=520, max_policy_steps=52
    )
    raw["training"].update(
        updates=100, checkpoint_every_updates=1, optimizer_steps_per_update=8
    )
    raw["training"]["schedule"]["minibatch_trajectories"] = 120
    dev = load_state_manifest(
        "examples/embodied/libero/state_manifests/pi05_long_dev_v1/manifest.json"
    )
    sealed = load_state_manifest(
        sealed_manifest or "outputs/sealed/manifests/pi0_fast_long_v1/manifest.json"
    )
    if Counter(e.task_id for e in dev.entries) != Counter({i: 10 for i in range(10)}):
        raise ValueError("Require fixed 100-episode balanced Long development panel")
    if {e.state_sha256 for e in dev.entries} & {e.state_sha256 for e in sealed.entries}:
        raise ValueError("Development and sealed states overlap")
    env["evaluation_state_manifest"] = str(dev.path.resolve())
    raw["evaluation"].update(
        pre_training_success_gate=None,
        evaluate_before_training=True,
        evaluate_after_first_update=False,
        every_updates=5,
        episodes=100,
        fixed_scenarios=[e.id for e in dev.entries],
        checkpoint_selection="last",
    )
    raw["evaluation"]["kwargs"]["seed_contract"].update(
        environment_mode="fixed", fixed_environment_seed=0
    )
    raw["runtime"].update(
        worker_python_executable=sys.executable,
        worker_handoff_dir=str(root / "grpo-handoffs"),
        keep_worker_handoffs=False,
    )
    raw["runtime"]["rollout_execution"]["actor_python_executable"] = sys.executable
    raw["storage"].update(
        output_dir=str(root / "grpo"),
        resume_from_checkpoint=None,
        retain_checkpoint_updates=[1, 5, 20, 40, 60, 80, 100],
    )
    raw["observability"]["wandb"].update(
        group=root.name, run_id=None, resume=None, project="art-embodied-pi0-fast-long"
    )
    raw["observability"]["weave"]["project"] = "art-embodied-pi0-fast-long"
    configs = {"grpo": raw}
    sft = deepcopy(raw)
    sft["policy"]["load_kwargs"]["warm_start_checkpoint"] = None
    sft["experiment"]["run"] = root.name + "-sft"
    sft["storage"]["output_dir"] = str(root / "sft/runtime")
    sft["runtime"]["worker_handoff_dir"] = str(root / "sft/handoffs")
    sft["observability"]["wandb"]["enabled"] = False
    sft["observability"]["weave"]["enabled"] = False
    sft["observability"].update(
        require_train_video=False, require_evaluation_video=False
    )
    configs["sft"] = sft
    for arm in ("baseline", "candidate"):
        item = deepcopy(raw)
        item["experiment"]["run"] = root.name + "-sealed-" + arm
        item["environment"]["kwargs"]["evaluation_state_manifest"] = str(
            sealed.path.resolve()
        )
        item["evaluation"].update(
            data_role="sealed_test",
            episodes=len(sealed.entries),
            fixed_scenarios=[e.id for e in sealed.entries],
        )
        item["evaluation"]["kwargs"]["require_policy_checkpoint"] = True
        if arm == "candidate":
            item["evaluation"]["baseline_outcomes_path"] = str(
                root / "sealed-baseline/evaluation/update_000000_episode_outcomes.json"
            )
        item["storage"]["output_dir"] = str(root / ("sealed-" + arm))
        item["runtime"]["worker_handoff_dir"] = str(
            root / ("sealed-" + arm + "-handoffs")
        )
        configs["sealed-" + arm] = item
    for name, value in configs.items():
        EmbodiedExperimentConfig.model_validate(value)
        (root / f"{name}.yaml").write_text(yaml.safe_dump(value, sort_keys=False))
    config = EmbodiedExperimentConfig.model_validate(raw)
    scenarios = build_train_scenarios(config)
    for start in range(0, len(scenarios), 120):
        counts = Counter(s.payload["task_id"] for s in scenarios[start : start + 120])
        if sorted(counts.values()) != [12] * 10:
            raise ValueError(f"Unbalanced GRPO task schedule: {counts}")
    assert len(build_evaluation_scenarios(config)) == 100
    write(
        root / "plan.json",
        {
            "sft_updates": 400,
            "sft_global_frames_per_update": 80,
            "grpo_updates": 100,
            "trajectories_per_update": 960,
            "groups_per_task": 12,
            "sft_gradient": "global token mean; SUM allreduce before clipping/AdamW",
            "reset": "official states; 15 neutral steps; unchanged controller; MuJoCo 3.3.0",
            "source_sha256": file_sha256(source),
            "runtime": check_runtime(),
            "dev_manifest_sha256": file_sha256(dev.path),
            "sealed_manifest_sha256": file_sha256(sealed.path),
            "sealed_episodes": len(sealed.entries),
            "sealed_policy_evaluated": False,
            "sealed_selection": "SFT400 baseline vs last completed GRPO checkpoint, same sealed panel; no sealed outcomes during development",
            "no_headroom_gate": "Measure all tasks first; do not select or filter states by policy success.",
        },
    )


async def grpo(root):
    from art_embodied import make_policy
    from art_embodied.runner import run_lerobot_experiment
    from examples.embodied.libero.components import (
        build_evaluation_scenarios,
        build_train_scenarios,
    )
    from examples.embodied.pi0_fast_reset_restart_run import VerifiedObserver

    check_runtime()
    if not (root / "sft/complete.json").exists():
        raise ValueError("SFT must be complete and remotely verified")
    config = EmbodiedExperimentConfig.from_yaml(root / "grpo.yaml")
    observer = VerifiedObserver.start(config)
    error = None
    try:
        policy = make_policy(config)
        with isolated_art_runtime(), diagnostic_cuda_mapping():
            if (root / "REQUIRE_FULL_GROUP_QUALIFICATION").exists():
                from examples.embodied.pi0_fast_group_qualification import qualify

                report = await qualify(root, config, policy)
                observer.wandb_run.summary.update(
                    {
                        "qualification/full_group_passed": True,
                        "qualification/logprob_abs_mean": report["logprob_abs_mean"],
                        "qualification/optimizer_updates": 0,
                    }
                )
            await run_lerobot_experiment(
                config=config,
                policy=policy,
                train_scenarios=build_train_scenarios(config),
                evaluation_scenarios=build_evaluation_scenarios(config),
                observer=observer,
            )
    except BaseException as exc:
        error = exc
        raise
    finally:
        observer.close(exit_code=1 if error else 0)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--phase", choices=("plan", "adopt-sft", "grpo"), required=True)
    p.add_argument("--sft-source-root", type=Path)
    p.add_argument("--sealed-manifest", type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    if args.phase == "plan":
        plan(root, sealed_manifest=args.sealed_manifest)
    elif args.phase == "adopt-sft":
        if args.sft_source_root is None:
            p.error("adopt-sft requires --sft-source-root")
        check_runtime()
        adopt_sft(root, args.sft_source_root)
    else:
        asyncio.run(grpo(root))
