from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from art_embodied.integrations.isaaclab_reset_bank import IsaacLabResetBank
from examples.embodied.isaaclab.droid_environment import (
    DroidSceneConfig,
    load_droid_recipe,
)
from examples.embodied.isaaclab.prepare_droid import build_config, file_sha256


@pytest.fixture
def inputs(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "assets/droid").mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"model")
    (checkpoint / "assets/droid/norm_stats.json").write_text("{}")
    scene = DroidSceneConfig(
        "alphabet_soup_can_hope_robolab",
        "bowl_ycb_robolab",
        "Pick up the soup can and place it in the bowl.",
    )
    banks = tmp_path / "banks"
    offset = 0
    for split, count in (("train", 512), ("development", 128), ("sealed", 256)):
        state = {
            "robot": {"joint": torch.arange(offset, offset + count)[:, None].float()}
        }
        IsaacLabResetBank.from_batches(
            [state], contract=scene.reset_contract(), split=split
        ).save(banks / split)
        offset += count
    return dict(
        checkpoint=checkpoint,
        banks=banks,
        output_dir=tmp_path / "output",
        policy_python=Path("/test/policy/bin/python"),
        entity="test-entity",
    )


@pytest.mark.parametrize("num_gpus", [2, 4, 8])
def test_reference_recipe_keeps_work_and_eval_geometry(inputs, num_gpus):
    config = build_config(**inputs, num_gpus=num_gpus)
    assert config.algorithm.group_size * config.rollout.groups_per_update == 512
    assert config.rollout.max_policy_steps == 70
    assert config.training.optimizer_steps_per_update == 28
    assert config.training.schedule.global_batch_size == 2560
    assert config.training.schedule.actor_world_size == num_gpus
    assert (
        config.training.schedule.groups_per_process_per_rollout_epoch * num_gpus == 64
    )
    assert len(config.runtime.training_devices) == num_gpus
    assert config.policy.dtype == "bfloat16"
    assert config.policy.lora.rank == 32 and config.policy.lora.dropout == 0
    assert config.algorithm.flow_sde.num_denoise_steps == 10
    assert config.evaluation.runtime == "native_openpi"
    assert config.evaluation.episodes == 128 and config.evaluation.every_updates == 5
    assert config.evaluation.fixed_scenarios == [f"development-{i}" for i in range(128)]
    assert config.evaluation.evaluate_before_training
    assert not config.evaluation.evaluate_after_first_update
    assert config.observability.wandb.native_update_steps
    assert config.observability.videos_per_update == 512
    assert config.observability.videos_per_evaluation == 128
    assert config.policy.load_kwargs["checkpoint_sha256"] == file_sha256(
        inputs["checkpoint"] / "model.safetensors"
    )
    assert "sealed" not in config.model_dump_json()


def test_recipe_rejects_invalid_resource_geometry(inputs):
    with pytest.raises(ValueError, match="2, 4 or 8"):
        build_config(**inputs, num_gpus=3)


def test_recipe_requires_explicit_entity(inputs):
    inputs["entity"] = None
    with pytest.raises(ValueError, match="entity"):
        build_config(**inputs)


def test_recipe_keeps_virtual_environment_python_path(inputs, tmp_path):
    import sys

    python = tmp_path / "environment/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    inputs["policy_python"] = python
    config = build_config(**inputs)
    assert str(config.runtime.worker_python_executable) == str(python)


def test_recipe_detects_bank_tampering(inputs):
    (inputs["banks"] / "development/states.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        build_config(**inputs)


def test_explicit_default_recipe_preserves_qualified_configuration(inputs):
    recipe = (
        Path(__file__).parents[1]
        / "examples/embodied/isaaclab/pi05_droid_soup_grpo.yaml"
    )
    assert build_config(**inputs, recipe=recipe) == build_config(**inputs)


def test_custom_task_is_bound_to_its_own_scene_banks(inputs, tmp_path):
    import yaml

    raw = load_droid_recipe()
    raw["environment"]["task"] = "cube_to_bowl"
    raw["environment"]["kwargs"].update(
        pick_up_object="rubiks_cube_hot3d_robolab",
        instruction="Pick up the cube and place it in the bowl.",
    )
    recipe = tmp_path / "cube.yaml"
    recipe.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="contract"):
        build_config(**inputs, recipe=recipe)

    scene = DroidSceneConfig.from_recipe(load_droid_recipe(recipe))
    banks = tmp_path / "cube-banks"
    offset = 0
    for split, count in (("train", 512), ("development", 128), ("sealed", 256)):
        state = {"robot": {"joint": torch.arange(offset, offset + count)[:, None]}}
        IsaacLabResetBank.from_batches(
            [state], contract=scene.reset_contract(), split=split
        ).save(banks / split)
        offset += count
    inputs["banks"] = banks
    config = build_config(**inputs, recipe=recipe)
    assert config.environment.task == "cube_to_bowl"
    assert config.environment.kwargs["instruction"] == scene.instruction
    assert config.environment.kwargs["pick_up_object"] == scene.pick_up_object
    assert config.algorithm.group_size * config.rollout.groups_per_update == 512
    assert load_droid_recipe()["environment"]["task"] == "soup_to_bowl"


def test_custom_recipe_rejects_unrecognized_physics():
    raw = load_droid_recipe()
    raw["environment"]["kwargs"]["physics_condition"] = "another-simulator"
    with pytest.raises(ValueError, match="physics condition"):
        DroidSceneConfig.from_recipe(raw)


def test_custom_recipe_rejects_wrong_policy(tmp_path):
    recipe = tmp_path / "wrong.yaml"
    recipe.write_text("policy: {type: openvla}\nenvironment: {type: isaaclab}\n")
    with pytest.raises(ValueError, match="OpenPI DROID"):
        load_droid_recipe(recipe)
