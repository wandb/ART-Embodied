from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from examples.embodied.pi0_fast_native_precision_restart import (
    configure,
    configure_rollout_parallelism,
    plan,
)

SOURCE = Path("outputs/multitask/pi0fast-long-task8-1893")
REFERENCE = Path(
    "examples/embodied/pi0_fast_libero_long_task_balanced_grpo_development_h100.yaml"
)


@pytest.mark.parametrize("actors", [2, 4])
@pytest.mark.parametrize("device_count", [1, 8])
def test_parallelism_contract_without_local_experiment_outputs(actors, device_count):
    raw = yaml.safe_load(REFERENCE.read_text())
    raw["policy"]["load_kwargs"]["model_compute_dtype"] = "checkpoint"
    raw["runtime"]["rollout_devices"] = [f"cuda:{i}" for i in range(device_count)]
    before = deepcopy(raw)
    result = configure_rollout_parallelism(raw, actors_per_device=actors)
    assert result["rollout"]["workers"] == actors * device_count
    assert result["runtime"]["rollout_execution"]["inference_max_batch_size"] == 8
    assert result["runtime"]["rollout_execution"]["actors_per_device"] == actors
    assert raw == before
    for key in ("algorithm", "training", "environment", "evaluation", "observability"):
        assert result[key] == before[key]
    for key in ("groups_per_update", "epochs_per_update", "max_episode_steps"):
        assert result["rollout"][key] == before["rollout"][key]


@pytest.mark.parametrize(
    "devices", [[], ["cuda:0", "cuda:0"], [f"cuda:{i}" for i in range(9)]]
)
def test_parallelism_rejects_invalid_device_geometry(devices):
    raw = yaml.safe_load(REFERENCE.read_text())
    raw["policy"]["load_kwargs"]["model_compute_dtype"] = "checkpoint"
    raw["runtime"]["rollout_devices"] = devices
    with pytest.raises(ValueError, match="one node"):
        configure_rollout_parallelism(raw)


@pytest.mark.skipif(not (SOURCE / "grpo.yaml").exists(), reason="Local measured recipe")
@pytest.mark.parametrize(
    "phase", ["sft", "grpo", "sealed-baseline", "sealed-candidate"]
)
def test_restart_never_resumes_fp32(tmp_path, phase):
    original = yaml.safe_load((SOURCE / "grpo.yaml").read_text())
    saved = deepcopy(original)
    raw = configure(original, root=tmp_path, phase=phase)
    assert original == saved
    assert raw["policy"]["load_kwargs"]["model_compute_dtype"] == "checkpoint"
    assert raw["storage"]["resume_from_checkpoint"] is None
    assert raw["observability"]["wandb"]["run_id"] is None
    assert raw["observability"]["wandb"]["resume"] is None
    warm = raw["policy"]["load_kwargs"]["warm_start_checkpoint"]
    assert warm == (None if phase == "sft" else str(tmp_path / "sft-warm-start"))
    for key in ("algorithm", "training", "reward"):
        assert raw[key] == original[key]
    rollout = deepcopy(raw["rollout"])
    if phase == "grpo":
        assert rollout["workers"] == 4 * len(raw["runtime"]["rollout_devices"])
        rollout["workers"] = original["rollout"]["workers"]
    assert rollout == original["rollout"]


@pytest.mark.skipif(not (SOURCE / "grpo.yaml").exists(), reason="Local measured recipe")
@pytest.mark.parametrize("actors", [2, 4])
def test_parallelism_matches_actor_slots_without_cutting_work(tmp_path, actors):
    raw = configure(
        yaml.safe_load((SOURCE / "grpo.yaml").read_text()), root=tmp_path, phase="grpo"
    )
    raw["runtime"]["rollout_execution"]["actor_kwargs"]["max_concurrent_rollouts"] = 8
    before = deepcopy(raw)
    result = configure_rollout_parallelism(raw, actors_per_device=actors)
    assert raw == before
    execution = result["runtime"]["rollout_execution"]
    assert result["rollout"]["workers"] == actors * 8
    assert execution["actor_kwargs"]["max_concurrent_rollouts"] == actors * 8
    assert execution["inference_max_batch_size"] == 8
    assert execution["actors_per_device"] == actors
    for field in (
        "policy",
        "algorithm",
        "training",
        "reward",
        "evaluation",
        "observability",
        "environment",
    ):
        assert result[field] == before[field]
    assert (
        result["rollout"]["groups_per_update"] * result["algorithm"]["group_size"]
        == 960
    )
    raw["policy"]["load_kwargs"]["model_compute_dtype"] = "float32"
    with pytest.raises(ValueError, match="checkpoint precision"):
        configure_rollout_parallelism(raw)


def test_existing_directory_cannot_be_reused(tmp_path):
    with pytest.raises(ValueError, match="new output directory"):
        plan(tmp_path, SOURCE)


@pytest.mark.skipif(
    not (SOURCE / "source.json").exists(), reason="Local measured recipe"
)
def test_fresh_plan_preserves_scope_and_sealed_panel(tmp_path):
    root = tmp_path / "fresh"
    plan(root, SOURCE.resolve())
    sft = yaml.safe_load((root / "sft.yaml").read_text())
    grpo = yaml.safe_load((root / "grpo.yaml").read_text())
    assert sft["environment"]["kwargs"]["task_ids"] == list(range(10))
    assert grpo["environment"]["kwargs"]["task_ids"] == [8]
    assert sft["policy"]["lora"]["rank_partition"] is None
    assert not (root / "sft-warm-start").exists()
    assert grpo["rollout"]["groups_per_update"] * grpo["algorithm"]["group_size"] == 960
    assert grpo["training"]["updates"] == 100
    assert grpo["evaluation"]["every_updates"] == 5
    assert grpo["evaluation"]["episodes"] == 100
    report = json.loads((root / "plan.json").read_text())
    assert report["sealed_policy_evaluated"] is False
    assert report["evaluation_manifests"]["sealed-baseline"]["states"] == 100
