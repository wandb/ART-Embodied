"""Single-task control from verified shared SFT400; unchanged FAST learning rule."""

import argparse
import asyncio
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil

import yaml

from art_embodied.checkpointing import CheckpointManager
from art_embodied.config import EmbodiedExperimentConfig
from examples.embodied.libero.state_manifest import (
    file_sha256,
    load_state_manifest,
    write_state_manifest,
)
from examples.embodied.pi0_fast_long_restart import check_runtime, isolated_art_runtime
from examples.embodied.pi0_fast_reset_restart_run import VerifiedObserver
from examples.embodied.pi0_fast_spatial_teacher_control import (
    diagnostic_cuda_mapping,
    write,
)

TASK_ID = 8


def configure(source, root):
    raw = deepcopy(source)
    raw["experiment"]["run"] = root.name + "-grpo"
    raw["experiment"]["tags"] += ["single-task-8", "shared-sft400-control"]
    raw["environment"]["kwargs"]["task_ids"] = [TASK_ID]
    raw["environment"]["kwargs"]["training_task_ids"] = [TASK_ID]
    raw["environment"]["kwargs"]["evaluation_state_manifest"] = str(
        root / "manifests/development/manifest.json"
    )
    raw["policy"]["load_kwargs"]["warm_start_checkpoint"] = str(root / "sft-warm-start")
    raw["evaluation"]["fixed_scenarios"] = [
        f"libero_10/generated-held-out/task-08/state-{i:02d}" for i in range(100)
    ]
    raw["evaluation"]["episodes"] = 100
    raw["storage"]["output_dir"] = str(root / "grpo")
    raw["runtime"]["worker_handoff_dir"] = str(root / "grpo-handoffs")
    raw["observability"]["wandb"].update(group=root.name, run_id=None, resume=None)
    return raw


def combine_development(original, additional, destination):
    old = load_state_manifest(original)
    new = load_state_manifest(additional)
    entries = [deepcopy(e) for e in old.metadata["entries"] if e["task_id"] == TASK_ID]
    if len(entries) != 10 or len(new.entries) != 90:
        raise ValueError("Need the original ten task8 states and ninety new states")
    states = {e["state_key"]: old.states[e["state_key"]] for e in entries}
    for index, item in enumerate(new.metadata["entries"], start=10):
        if item["task_id"] != TASK_ID:
            raise ValueError("Unexpected task in development extension")
        entry = deepcopy(item)
        key = f"task_08_state_{index:02d}"
        entry.update(
            id=f"libero_10/generated-held-out/task-08/state-{index:02d}", state_key=key
        )
        states[key] = new.states[item["state_key"]]
        entries.append(entry)
    if len({e["state_sha256"] for e in entries}) != 100:
        raise ValueError("Duplicate development states")
    return write_state_manifest(
        destination,
        suite_name=old.suite_name,
        simulator_compatibility=old.simulator_compatibility,
        states=states,
        entries=entries,
        generator={
            "policy_loaded": False,
            "selection_uses_model_outcomes": False,
            "original_manifest": str(old.path),
            "original_manifest_sha256": file_sha256(old.path),
            "additional_manifest": str(new.path),
            "additional_manifest_sha256": file_sha256(new.path),
            "original_states_retained": 10,
            "additional_states": 90,
        },
    )


def generation_config(raw):
    generation = deepcopy(raw)
    generation["environment"]["kwargs"].pop("evaluation_state_manifest")
    generation["evaluation"].update(
        enabled=False, evaluate_before_training=False, fixed_scenarios=[]
    )
    generation["observability"].update(
        require_train_video=False, require_evaluation_video=False
    )
    generation["observability"]["wandb"]["enabled"] = False
    generation["observability"]["weave"]["enabled"] = False
    EmbodiedExperimentConfig.model_validate(generation)
    return generation


def check_transfer(root, source):
    def outcomes(path):
        rows = json.loads(path.read_text())["episodes"]
        return {e["scenario_id"]: e for e in rows if "/task-08/" in e["scenario_id"]}

    old = outcomes(source / "grpo/evaluation/update_000000_episode_outcomes.json")
    new = outcomes(root / "grpo/evaluation/update_000000_episode_outcomes.json")
    if len(old) != 10 or len(new) != 100:
        raise ValueError("Unexpected initial evaluation coverage")
    for key, previous in old.items():
        for field in ("success", "environment_seed", "action_sequence_sha256"):
            if previous[field] != new[key][field]:
                raise ValueError(f"SFT transfer differs at {key}: {field}")
    proof = {
        "passed": True,
        "episodes_compared": 10,
        "successes": sum(e["success"] for e in old.values()),
        "full_panel_success_rate": sum(e["success"] for e in new.values()) / 100,
        "policy_seed_comparison": "Not required: episode indexes change; decoding remains greedy",
    }
    write(root / "grpo/acceptance/multitask-sft-transfer.json", proof)
    return proof


class SingleTaskObserver(VerifiedObserver):
    async def log_initial_evaluation(self, evaluation, config):
        await super().log_initial_evaluation(evaluation, config)
        root = config.storage.output_dir.parent
        source = Path(json.loads((root / "source.json").read_text())["root"])
        print(
            "SFT task8 transfer verified: " + json.dumps(check_transfer(root, source)),
            flush=True,
        )


def prepare(root, source):
    from art_embodied.evaluation import validate_paired_evaluation_configs
    from examples.embodied.libero.components import (
        build_evaluation_scenarios,
        build_train_scenarios,
    )
    from examples.embodied.libero.generate_state_manifest import generate

    check_runtime()
    root.mkdir(parents=True, exist_ok=True)
    complete = json.loads((source / "sft/complete.json").read_text())
    verified = json.loads((source / "sft/verified-0401.json").read_text())
    if complete["updates"] != 400 or not (
        verified["history_verified"] and verified["artifact_verified"]
    ):
        raise ValueError("Need verified shared SFT400")
    source_raw = yaml.safe_load((source / "grpo.yaml").read_text())
    raw = configure(source_raw, root)
    generation = generation_config(raw)
    generation_path = root / "generation.yaml"
    generation_path.write_text(yaml.safe_dump(generation, sort_keys=False))
    generated = generate(
        generation_path,
        root / "manifests/development-extension",
        states_per_task=90,
        base_seed=20260910,
        max_attempts_per_task=2000,
        overwrite=False,
    )
    dev_path = combine_development(
        source_raw["environment"]["kwargs"]["evaluation_state_manifest"],
        generated,
        root / "manifests/development",
    )
    sealed_path = generate(
        generation_path,
        root / "manifests/sealed",
        states_per_task=100,
        base_seed=20270910,
        max_attempts_per_task=2000,
        overwrite=False,
    )
    dev, sealed = load_state_manifest(dev_path), load_state_manifest(sealed_path)
    if {e.state_sha256 for e in dev.entries} & {e.state_sha256 for e in sealed.entries}:
        raise ValueError("Development and sealed overlap")
    configs = {"grpo": raw}
    for arm in ("baseline", "candidate"):
        item = deepcopy(raw)
        item["experiment"]["run"] = root.name + "-sealed-" + arm
        item["environment"]["kwargs"]["evaluation_state_manifest"] = str(sealed_path)
        item["evaluation"].update(
            data_role="sealed_test", fixed_scenarios=[e.id for e in sealed.entries]
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
    validate_paired_evaluation_configs(
        *[
            EmbodiedExperimentConfig.model_validate(configs["sealed-" + arm])
            for arm in ("baseline", "candidate")
        ]
    )
    config = EmbodiedExperimentConfig.model_validate(raw)
    train = build_train_scenarios(config)
    counts = Counter(s.payload["task_id"] for s in train)
    assert set(counts) == {TASK_ID} and len(train) == 12000, counts
    assert len(build_evaluation_scenarios(config)) == 100
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
            "source": str(source / "sft/checkpoint-0400"),
            "shared_multitask": True,
        },
    )
    key = "adapter_model.safetensors"
    assert file_sha256(root / "sft-warm-start/policy" / key) == file_sha256(
        source / "sft/checkpoint-0400" / key
    )
    (root / "sft").symlink_to((source / "sft").resolve(), target_is_directory=True)
    write(
        root / "source.json",
        {
            "root": str(source),
            "sft_sha256": file_sha256(source / "sft/checkpoint-0400" / key),
        },
    )
    write(
        root / "plan.json",
        {
            "task_id": TASK_ID,
            "sft_retrained": False,
            "trajectories_per_update": 960,
            "source_task_trajectories_per_update": 96,
            "dev_states": 100,
            "retained_dev_states": 10,
            "sealed_states": 100,
            "sealed_policy_evaluated": False,
            "runtime": check_runtime(),
        },
    )


async def run(root):
    from art_embodied import make_policy
    from art_embodied.runner import run_lerobot_experiment
    from examples.embodied.libero.components import (
        build_evaluation_scenarios,
        build_train_scenarios,
    )

    check_runtime()
    config = EmbodiedExperimentConfig.from_yaml(root / "grpo.yaml")
    observer = SingleTaskObserver.start(config)
    error = None
    try:
        policy = make_policy(config)
        with isolated_art_runtime(), diagnostic_cuda_mapping():
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--phase", choices=("prepare", "run"), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase == "prepare":
        if args.source is None:
            parser.error("prepare requires --source")
        prepare(root, args.source.resolve())
    else:
        asyncio.run(run(root))
