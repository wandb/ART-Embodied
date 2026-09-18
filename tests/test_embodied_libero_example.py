from __future__ import annotations

import asyncio
from importlib import metadata
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from art_embodied import (
    Action,
    EmbodiedExperimentConfig,
    EmbodiedTrajectory,
    LeRobotPolicyAdapterProtocol,
    validate_paired_evaluation_configs,
)
from art_embodied.experiment import EmbodiedScenario, RolloutContext
from examples.embodied.libero import environment as libero_environment_module
from examples.embodied.libero import evaluate_policy_series as policy_series_module
from examples.embodied.libero import policy as libero_policy_module
from examples.embodied.libero import records as libero_records_module
from examples.embodied.libero import rollout as libero_rollout_module
from examples.embodied.libero import train_openvla_oft as train_openvla_oft_module
from examples.embodied.libero.checkpoint_roundtrip import _comparison_matches
from examples.embodied.libero.components import (
    LiberoSettings,
    OpenVLAAdapter,
    balanced_task_random_trial,
    partitioned_random_task_and_trial,
    process_openvla_action_chunk,
    record_libero_observation,
    record_libero_transition,
    rlinf_v01_random_task_and_trial,
    rlinf_v01_task_and_trial,
    rollout_libero_group,
)
from examples.embodied.libero.records import (
    build_evaluation_scenarios,
    build_train_scenarios,
)


def _config() -> EmbodiedExperimentConfig:
    return EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml"
    )


def _baseline_config() -> EmbodiedExperimentConfig:
    return EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_libero_object_sft_baseline_eval.yaml"
    )


def _public_grpo_config() -> EmbodiedExperimentConfig:
    return EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_libero_object_public_grpo_eval.yaml"
    )


def _spatial_config(name: str) -> EmbodiedExperimentConfig:
    return EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1] / f"examples/embodied/{name}"
    )


def _shared_rollout_config() -> EmbodiedExperimentConfig:
    return EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1] / "examples/embodied/"
        "openvla_oft_libero_object_grpo_lora_lr2e4_shared_rollout.yaml"
    )


def _h100_rollout_config() -> EmbodiedExperimentConfig:
    return EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_libero_object_grpo_lora_lr2e4_h100.yaml"
    )


def _h100_gspo_config() -> EmbodiedExperimentConfig:
    return EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_libero_object_gspo_lora_lr2e4_h100.yaml"
    )


@pytest.mark.parametrize(
    ("manifest_name", "expected_steps"),
    [
        ("smolvla_libero_10_teacher_sft_policy_series.yaml", [0, 100, 300, 500]),
        (
            "smolvla_libero_10_teacher_sft_low_dose_policy_series.yaml",
            [0, 10, 30, 50],
        ),
        (
            "smolvla_libero_10_rehearsal_sft_policy_series.yaml",
            [0, 25, 50, 100],
        ),
        (
            "smolvla_libero_10_rehearsal_lora_sft_policy_series.yaml",
            [0, 25, 50, 100],
        ),
    ],
)
def test_smolvla_teacher_sft_policy_series_uses_full_weight_profile(
    manifest_name: str,
    expected_steps: list[int],
) -> None:
    campaign = Path(__file__).parents[1] / "examples/embodied/libero" / manifest_name

    manifest = policy_series_module.PolicySeriesManifest.from_yaml(campaign)
    config_path = campaign.parent / manifest.base_config
    candidates = manifest.candidates

    assert manifest.schema_version == 1
    assert config_path.name.endswith("policy_series_eval_h100.yaml")
    assert [candidate.step for candidate in candidates] == expected_steps
    assert candidates[0].local is False
    assert all(candidate.local for candidate in candidates[1:])
    config = EmbodiedExperimentConfig.from_yaml(config_path)
    assert config.policy.lora.enabled is False
    assert config.policy.trainable_parameter_strategy == "smolvla_action_expert"
    assert config.policy.force_trainable_float32 is False
    assert config.observability.wandb.enabled is True


def test_policy_series_requires_one_step_zero_baseline() -> None:
    with pytest.raises(ValueError, match="Step 0 baseline"):
        policy_series_module.PolicySeriesManifest.model_validate(
            {
                "schema_version": 1,
                "workspace_root": ".",
                "base_config": "config.yaml",
                "candidates": [
                    {
                        "step": 25,
                        "name": "candidate",
                        "checkpoint_role": "sft_candidate",
                        "path": "checkpoint",
                        "local": True,
                    }
                ],
            }
        )


def _smolvla_positive_control_config() -> EmbodiedExperimentConfig:
    return EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1] / "examples/embodied/"
        "smolvla_libero_object_flow_sde_grpo_positive_control_h100.yaml"
    )


def _smolvla_distilled_config() -> EmbodiedExperimentConfig:
    return EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1]
        / "examples/embodied/smolvla_libero_10_distilled_flow_sde_grpo_h100.yaml"
    )


def test_checkpoint_roundtrip_equality_is_independent_of_adapter_effect() -> None:
    equal = {
        "tokens_equal": True,
        "decoded": {"shape_equal": 1.0, "max_abs": 0.0, "mean_abs": 0.0},
    }
    changed = {
        "tokens_equal": False,
        "decoded": {"shape_equal": 1.0, "max_abs": 0.5, "mean_abs": 0.1},
    }

    assert _comparison_matches(equal) is True
    assert _comparison_matches(changed) is False


def test_evaluation_checkpoint_loader_uses_policy_snapshot_contract(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "policy"
    snapshot.mkdir()
    loaded = []
    policy = SimpleNamespace(load_checkpoint=loaded.append)

    train_openvla_oft_module._load_policy_checkpoint(policy, snapshot)

    assert loaded == [{"path": str(snapshot.resolve())}]


def test_evaluation_checkpoint_loader_rejects_missing_snapshot(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint directory is missing"):
        train_openvla_oft_module._load_policy_checkpoint(
            SimpleNamespace(load_checkpoint=lambda _value: None),
            tmp_path / "missing",
        )


def test_candidate_evaluation_requires_explicit_policy_checkpoint() -> None:
    raw = _config().model_dump(mode="python")
    raw["evaluation"]["kwargs"]["require_policy_checkpoint"] = True
    config = EmbodiedExperimentConfig.model_validate(raw)

    with pytest.raises(ValueError, match="requires --policy-checkpoint"):
        train_openvla_oft_module._validate_required_policy_checkpoint(
            config,
            policy_checkpoint=None,
        )

    train_openvla_oft_module._validate_required_policy_checkpoint(
        config,
        policy_checkpoint=Path("candidate"),
    )


def test_training_entrypoint_rejects_checkpoint_without_evaluate_only(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires --evaluate-only"):
        asyncio.run(
            train_openvla_oft_module.run(
                tmp_path / "unused.yaml",
                policy_checkpoint=tmp_path,
            )
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("observation_processor", {"resize": 224}, "observation_processor"),
        ("action_processor", {"clip": True}, "action_processor"),
    ],
)
def test_builtin_libero_rejects_unconsumed_processor_config(
    field: str,
    value: dict[str, object],
    message: str,
) -> None:
    config = _config()
    environment = config.environment.model_copy(update={field: value})
    changed = config.model_copy(update={"environment": environment})

    with pytest.raises(ValueError, match=message):
        LiberoSettings.from_config(changed)


def test_builtin_libero_rejects_unconsumed_reward_kwargs() -> None:
    config = _config()
    reward = config.reward.model_copy(update={"kwargs": {"hidden_weight": 1.0}})
    changed = config.model_copy(update={"reward": reward})

    with pytest.raises(ValueError, match="reward.kwargs"):
        LiberoSettings.from_config(changed)


def test_object_gspo_profile_matches_grpo_rollout_and_evaluation_contract() -> None:
    grpo = _h100_rollout_config()
    gspo = _h100_gspo_config()

    assert gspo.policy == grpo.policy
    assert gspo.environment == grpo.environment
    assert gspo.reward == grpo.reward
    assert gspo.rollout == grpo.rollout
    assert gspo.runtime == grpo.runtime
    assert gspo.evaluation == grpo.evaluation
    assert gspo.training.optimizer == grpo.training.optimizer
    assert gspo.trajectories_per_update == grpo.trajectories_per_update == 1024
    assert gspo.algorithm.type == "gspo"
    assert gspo.algorithm.importance_sampling_level == "sequence"
    assert gspo.algorithm.training_unit == "trajectory"
    assert gspo.algorithm.action_advantage_mode == "example"
    assert gspo.algorithm.score_source == "trajectory_reward"
    assert gspo.algorithm.loss_aggregation == "trajectory_mean"
    assert gspo.training.schedule.type == "trajectory_minibatch"
    assert gspo.training.schedule.minibatch_trajectories == 256
    assert gspo.training.optimizer_steps_per_update == 4
    assert gspo.algorithm.clip_epsilon_low == pytest.approx(3.0e-4)
    assert gspo.algorithm.clip_epsilon_high == pytest.approx(4.0e-4)


def test_object_training_and_fixed_baseline_share_evaluation_matrix() -> None:
    baseline = _baseline_config()
    candidate = _config()

    assert baseline.policy.path == candidate.policy.path
    assert baseline.policy.revision == candidate.policy.revision
    assert baseline.policy.unnorm_key == candidate.policy.unnorm_key
    assert (
        baseline.policy.evaluation_generation == candidate.policy.evaluation_generation
    )
    assert baseline.environment == candidate.environment
    assert baseline.evaluation.fixed_scenarios == candidate.evaluation.fixed_scenarios
    assert baseline.evaluation.seeds == candidate.evaluation.seeds
    assert baseline.evaluation.episodes == candidate.evaluation.episodes
    assert candidate.evaluation.baseline_outcomes_path is None
    assert baseline.observability.wandb.group == candidate.observability.wandb.group
    assert baseline.observability.wandb.job_type == "fixed-evaluation"
    assert candidate.observability.wandb.job_type == "trajectory-rl"
    assert baseline.experiment.project == "art-embodied-release-validation"
    assert baseline.observability.wandb.project == baseline.experiment.project
    assert baseline.observability.weave.project == baseline.experiment.project
    assert candidate.experiment.project == baseline.experiment.project
    assert candidate.observability.wandb.project == baseline.experiment.project
    assert candidate.observability.weave.project == baseline.experiment.project
    assert baseline.observability.require_train_video is False
    assert baseline.observability.require_evaluation_video is True


def test_public_grpo_control_changes_only_model_and_run_identity() -> None:
    baseline = _baseline_config()
    public = _public_grpo_config()

    assert public.policy.path == "RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-object"
    assert public.policy.revision == "5c4a3495cb2ce9c84e9af7e75be6752147443d9f"
    assert public.policy.path != baseline.policy.path
    assert public.policy.revision != baseline.policy.revision
    assert public.environment == baseline.environment
    assert public.evaluation.fixed_scenarios == baseline.evaluation.fixed_scenarios
    assert public.evaluation.seeds == baseline.evaluation.seeds
    assert public.evaluation.episodes == baseline.evaluation.episodes
    assert public.evaluation.baseline_outcomes_path == (
        baseline.storage.output_dir / "evaluation/update_000000_episode_outcomes.json"
    )
    assert public.policy.evaluation_generation == baseline.policy.evaluation_generation
    assert public.storage.output_dir != baseline.storage.output_dir
    assert public.observability.wandb.group == baseline.observability.wandb.group
    assert public.observability.wandb.job_type == "fixed-evaluation"

    summary = validate_paired_evaluation_configs(baseline, public)
    assert summary["baseline_policy_path"] == baseline.policy.path
    assert summary["candidate_policy_path"] == public.policy.path


def test_fixed_object_controls_use_validated_default_attention_contract() -> None:
    baseline = _baseline_config()
    public = _public_grpo_config()
    training = _config()

    assert baseline.policy.load_kwargs["attn_implementation"] is None
    assert public.policy.load_kwargs["attn_implementation"] is None
    # RLinf v0.1 did not pass an attention override. Explicit eager and
    # FlashAttention 2 both change the bidirectional action-token logits.
    assert training.policy.load_kwargs["attn_implementation"] is None


def test_spatial_baseline_and_candidate_have_paired_evaluation_contract() -> None:
    baseline = _spatial_config("openvla_oft_libero_spatial_sft_baseline_eval.yaml")
    candidate = _spatial_config(
        "openvla_oft_libero_spatial_grpo_rlinf_positive_control.yaml"
    )

    summary = validate_paired_evaluation_configs(baseline, candidate)

    assert summary["episodes"] == 100
    assert summary["baseline_report_path"].endswith(
        "openvla-oft-libero-spatial-sft-baseline-eval/"
        "evaluation/update_000000_episode_outcomes.json"
    )
    assert baseline.observability.wandb.job_type == "fixed-evaluation"
    assert candidate.observability.wandb.job_type == "trajectory-rl"
    assert baseline.experiment.project == "art-embodied-release-validation"
    assert baseline.observability.wandb.project == baseline.experiment.project
    assert baseline.observability.weave.project == baseline.experiment.project
    assert candidate.experiment.project == baseline.experiment.project
    assert candidate.observability.wandb.project == baseline.experiment.project
    assert candidate.observability.weave.project == baseline.experiment.project


def test_libero_settings_are_complete_and_match_action_geometry() -> None:
    config = _config()
    settings = LiberoSettings.from_config(config)

    assert settings.suite_name == "libero_object"
    assert settings.task_ids == tuple(range(10))
    assert settings.evaluation_trial_ids == tuple(range(40, 50))
    assert settings.simulator_compatibility == "rlinf_v01"
    assert settings.action_chunk_size == config.training.schedule.action_chunk_size
    assert settings.max_episode_steps == config.rollout.max_episode_steps
    assert settings.reset_gripper_open is False
    assert settings.reward_coefficient == 5.0


def test_libero_policy_observation_matches_lerobot_v060_processor() -> None:
    torch = pytest.importorskip("torch")
    from lerobot.processor.env_processor import LiberoProcessorStep

    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    eef_pos = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    eef_quat = np.array([0.0, 0.0, np.sin(0.2), np.cos(0.2)])
    gripper = np.array([0.03, -0.03], dtype=np.float64)
    processed = LiberoProcessorStep().observation(
        {
            "observation.images.image": torch.from_numpy(image)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .div(255),
            "observation.robot_state": {
                "eef": {
                    "pos": torch.from_numpy(eef_pos).unsqueeze(0),
                    "quat": torch.from_numpy(eef_quat).unsqueeze(0),
                },
                "gripper": {"qpos": torch.from_numpy(gripper).unsqueeze(0)},
            },
        }
    )
    raw = {
        "robot0_eef_pos": eef_pos,
        "robot0_eef_quat": eef_quat,
        "robot0_gripper_qpos": gripper,
    }

    expected_image = (
        processed["observation.images.image"][0]
        .mul(255)
        .round()
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )
    np.testing.assert_array_equal(
        libero_environment_module._image(image, rotate_180=True),  # noqa: SLF001
        expected_image,
    )
    np.testing.assert_allclose(
        libero_environment_module._proprio_state(raw),  # noqa: SLF001
        processed["observation.state"][0].numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_smolvla_positive_control_uses_lerobot_v060_evaluation_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _smolvla_positive_control_config()
    settings = LiberoSettings.from_config(config)
    monkeypatch.setattr(
        libero_records_module,
        "_task_languages",
        lambda current: {task_id: f"task {task_id}" for task_id in current.task_ids},
    )

    assert settings.evaluation_protocol == "lerobot_v060"
    assert settings.evaluation_trial_ids == tuple(range(10))
    assert settings.observation_height == 360
    assert settings.observation_width == 360
    assert settings.wait_steps_after_reset == 10
    assert settings.max_episode_steps == 280
    assert settings.control_mode == "relative"
    assert settings.reset_gripper_open is True
    assert build_evaluation_scenarios(config)[-1].id.endswith("task-09/trial-09")


def test_smolvla_distilled_recipe_is_disjoint_and_uses_1024_trajectories() -> None:
    config = _smolvla_distilled_config()
    settings = LiberoSettings.from_config(config)

    assert settings.training_task_ids == tuple(range(10))
    assert settings.training_trial_ids == tuple(range(10, 50))
    assert settings.evaluation_trial_ids == tuple(range(10))
    assert config.rollout.groups_per_update == 8
    assert config.rollout.epochs_per_update == 16
    assert (
        config.rollout.groups_per_update
        * config.rollout.epochs_per_update
        * config.algorithm.group_size
        == 1024
    )
    assert config.policy.lora.enabled is True
    assert config.policy.trainable_parameter_strategy == "smolvla_action_expert_lora"


def test_lerobot_v060_evaluation_contract_rejects_short_horizon() -> None:
    config = _smolvla_positive_control_config()
    changed = config.model_copy(
        update={
            "rollout": config.rollout.model_copy(
                update={"max_episode_steps": 128, "max_policy_steps": 128}
            )
        }
    )

    with pytest.raises(
        ValueError,
        match=r"max_episode_steps=128 \(expected 280\)",
    ):
        LiberoSettings.from_config(changed)


def test_lerobot_v060_chunked_contract_preserves_primitive_horizon() -> None:
    config = EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1]
        / "examples/embodied/smolvla_libero_10_distilled_flow_sde_grpo_h10_h100.yaml"
    )

    settings = LiberoSettings.from_config(config)

    assert settings.evaluation_protocol == "lerobot_v060_chunked"
    assert settings.action_chunk_size == 10
    assert settings.max_episode_steps == 520
    assert config.rollout.max_policy_steps == 52
    assert config.policy.load_kwargs["execution_horizon"] == 10
    assert config.training.schedule.action_chunk_size == 10


def test_pi0_fast_uses_dedicated_lerobot_v060_evaluation_contract() -> None:
    config = EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1]
        / "examples/embodied/pi0_fast_libero_spatial_grpo_development_h100.yaml"
    )

    settings = LiberoSettings.from_config(config)

    assert settings.evaluation_protocol == "lerobot_pi0_fast_v060"
    assert settings.evaluation_trial_ids == tuple(range(10))
    assert settings.observation_height == 360
    assert settings.observation_width == 360
    assert settings.wait_steps_after_reset == 10
    assert settings.action_chunk_size == 10
    assert settings.max_episode_steps == 280
    assert settings.control_mode == "relative"
    assert settings.normalize_gripper is False
    assert settings.binarize_gripper is False
    assert settings.invert_gripper is False
    assert settings.reset_gripper_open is True


def test_pi0_fast_partitioned_reset_allows_single_task() -> None:
    config = EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1]
        / "examples/embodied/pi0_fast_libero_spatial_grpo_development_h100.yaml"
    )
    values = dict(config.environment.kwargs)
    values.update(
        {
            "task_ids": [8],
            "training_task_ids": [8],
            "training_trial_ids": list(range(10, 50)),
            "init_state_selection": "partitioned_random_reset",
        }
    )
    config = config.model_copy(
        update={"environment": config.environment.model_copy(update={"kwargs": values})}
    )

    settings = LiberoSettings.from_config(config)

    assert settings.task_ids == (8,)
    assert settings.training_task_ids == (8,)
    assert settings.training_trial_ids == tuple(range(10, 50))


def test_pi0_fast_all_official_states_qualification_protocol() -> None:
    config = EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1]
        / "examples/embodied/pi0_fast_libero_spatial_grpo_development_h100.yaml"
    )
    values = dict(config.environment.kwargs)
    values.update(
        {
            "evaluation_protocol": "lerobot_pi0_fast_v060_all_official_states",
            "evaluation_trial_ids": list(range(50)),
        }
    )
    config = config.model_copy(
        update={"environment": config.environment.model_copy(update={"kwargs": values})}
    )

    settings = LiberoSettings.from_config(config)

    assert settings.evaluation_trial_ids == tuple(range(50))


def test_pi0_fast_evaluation_contract_rejects_gripper_transforms() -> None:
    config = EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1]
        / "examples/embodied/pi0_fast_libero_spatial_grpo_development_h100.yaml"
    )
    changed = config.model_copy(
        update={
            "environment": config.environment.model_copy(
                update={
                    "kwargs": {
                        **config.environment.kwargs,
                        "binarize_gripper": True,
                    }
                }
            )
        }
    )

    with pytest.raises(
        ValueError,
        match=r"binarize_gripper=True \(expected False\)",
    ):
        LiberoSettings.from_config(changed)


def test_libero_settings_allow_schedule_without_action_chunk_geometry() -> None:
    config = EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_libero_object_gspo_lora_smoke_h100.yaml"
    )

    settings = LiberoSettings.from_config(config)

    assert settings.action_chunk_size == config.environment.kwargs["action_chunk_size"]
    assert config.training.schedule.type == "trajectory_minibatch"


def _mock_libero_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    module_path: str,
    libero_version: str | None = None,
    hf_libero_version: str | None = None,
    libero_root: str = "/venv/site-packages",
    hf_libero_root: str = "/venv/site-packages",
) -> None:
    monkeypatch.setitem(sys.modules, "libero", SimpleNamespace(__file__=module_path))
    versions = {
        "libero": libero_version,
        "hf-libero": hf_libero_version,
    }
    roots = {
        "libero": Path(libero_root) if libero_version else None,
        "hf-libero": Path(hf_libero_root) if hf_libero_version else None,
    }

    def version(distribution_name: str) -> str:
        value = versions[distribution_name]
        if value is None:
            raise metadata.PackageNotFoundError(distribution_name)
        return value

    monkeypatch.setattr(metadata, "version", version)

    def distribution(distribution_name: str) -> SimpleNamespace:
        root = roots[distribution_name]
        if root is None:
            raise metadata.PackageNotFoundError(distribution_name)
        return SimpleNamespace(locate_file=lambda _path: root)

    monkeypatch.setattr(metadata, "distribution", distribution)


def test_libero_runtime_accepts_pinned_product_distribution(monkeypatch) -> None:
    _mock_libero_runtime(
        monkeypatch,
        module_path="/venv/site-packages/libero/__init__.py",
        hf_libero_version="0.1.4",
    )

    runtime = libero_environment_module._validate_libero_runtime("rlinf_v01")

    assert runtime["provider"] == "hf-libero"
    assert runtime["distribution_version"] == "0.1.4"


def test_libero_runtime_accepts_legacy_conformance_checkout(monkeypatch) -> None:
    _mock_libero_runtime(
        monkeypatch,
        module_path="/oracle/RLinf-LIBERO/libero/__init__.py",
        hf_libero_version="0.1.4",
    )

    runtime = libero_environment_module._validate_libero_runtime("rlinf_v01")

    assert runtime["provider"] == "rlinf-source-checkout"
    assert runtime["distribution_version"] == "source-checkout"


def test_libero_runtime_attributes_module_to_owning_distribution(monkeypatch) -> None:
    _mock_libero_runtime(
        monkeypatch,
        module_path="/overlay/libero/__init__.py",
        libero_version="0.1.0",
        hf_libero_version="0.1.4",
        libero_root="/opt/venv/site-packages",
        hf_libero_root="/overlay",
    )

    runtime = libero_environment_module._validate_libero_runtime("rlinf_v01")

    assert runtime["provider"] == "hf-libero"
    assert runtime["distribution_version"] == "0.1.4"


def test_libero_runtime_rejects_unvalidated_distribution(monkeypatch) -> None:
    _mock_libero_runtime(
        monkeypatch,
        module_path="/venv/site-packages/libero/__init__.py",
        hf_libero_version="0.1.5",
    )

    with pytest.raises(RuntimeError, match="hf-libero==0.1.4"):
        libero_environment_module._validate_libero_runtime("rlinf_v01")


def test_libero_runtime_paths_preserve_valid_explicit_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "custom-libero"
    for name in ("bddl_files", "init_files", "assets"):
        (root / name).mkdir(parents=True)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                f"benchmark_root: {root}",
                f"bddl_files: {root / 'bddl_files'}",
                f"init_states: {root / 'init_files'}",
                f"assets: {root / 'assets'}",
                f"datasets: {root / 'datasets'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(config_dir))

    runtime = libero_environment_module.prepare_libero_runtime_paths()

    assert runtime["path_source"] == "configured"
    assert runtime["config_path"] == str(config_file)
    assert os.environ["LIBERO_CONFIG_PATH"] == str(config_dir)


def test_libero_runtime_paths_recover_from_stale_config_using_wheel_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_dir = tmp_path / "site-packages" / "libero"
    benchmark_root = module_dir / "libero"
    for name in ("bddl_files", "init_files", "assets"):
        (benchmark_root / name).mkdir(parents=True)
    monkeypatch.setitem(
        sys.modules,
        "libero",
        SimpleNamespace(__file__=str(module_dir / "__init__.py")),
    )
    stale_config_dir = tmp_path / "stale"
    stale_config_dir.mkdir()
    (stale_config_dir / "config.yaml").write_text(
        "benchmark_root: /missing\n"
        "bddl_files: /missing/bddl_files\n"
        "init_states: /missing/init_files\n"
        "assets: /missing/assets\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(stale_config_dir))
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    monkeypatch.setattr(
        libero_environment_module.tempfile,
        "mkdtemp",
        lambda **_kwargs: str(runtime_dir),
    )

    runtime = libero_environment_module.prepare_libero_runtime_paths()

    assert runtime["path_source"] == "installed-distribution"
    assert runtime["benchmark_root"] == str(benchmark_root)
    assert os.environ["LIBERO_CONFIG_PATH"] == str(runtime_dir)
    generated = (runtime_dir / "config.yaml").read_text(encoding="utf-8")
    assert f"init_states: {benchmark_root / 'init_files'}" in generated


def test_libero_task_asset_preflight_fails_before_policy_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "libero"
    for name in ("bddl_files", "init_files", "assets"):
        (root / name).mkdir(parents=True)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        f"benchmark_root: {root}\n"
        f"bddl_files: {root / 'bddl_files'}\n"
        f"init_states: {root / 'init_files'}\n"
        f"assets: {root / 'assets'}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(config_dir))
    task = SimpleNamespace(
        problem_folder="libero_object",
        bddl_file="task.bddl",
        init_states_file="task.pruned_init",
    )
    suite = SimpleNamespace(get_task=lambda _task_id: task)
    benchmark_module = SimpleNamespace(
        get_benchmark_dict=lambda: {"libero_object": lambda: suite}
    )
    libero_module = SimpleNamespace(
        benchmark=benchmark_module,
        get_libero_path=lambda key: {
            "bddl_files": str(root / "bddl_files"),
            "init_states": str(root / "init_files"),
        }[key],
    )
    monkeypatch.setitem(sys.modules, "libero.libero", libero_module)
    settings = LiberoSettings.from_config(_config())

    with pytest.raises(RuntimeError, match="refusing to load the policy"):
        libero_environment_module.validate_libero_task_assets(settings)


def test_libero_runtime_import_preflight_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        libero_environment_module,
        "prepare_libero_runtime_paths",
        lambda: {},
    )
    monkeypatch.setattr(
        libero_environment_module,
        "_ensure_legacy_gym_import",
        lambda: None,
    )
    monkeypatch.setattr(
        libero_environment_module.metadata,
        "version",
        lambda _name: "3.8.1",
    )
    monkeypatch.setitem(sys.modules, "libero.libero.envs", None)

    with pytest.raises(RuntimeError, match="could not initialize its EGL renderer"):
        libero_environment_module.validate_libero_runtime_imports()


def test_libero_runtime_preflight_rejects_mujoco_abi_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        libero_environment_module,
        "prepare_libero_runtime_paths",
        lambda: {},
    )
    monkeypatch.setattr(
        libero_environment_module.metadata,
        "version",
        lambda _name: "3.10.0",
    )

    with pytest.raises(RuntimeError, match="mj_fullM"):
        libero_environment_module.validate_libero_runtime_imports()


def test_training_entrypoint_preflight_does_not_load_policy_or_check_gpu(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = (
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml"
    )

    def fail(*_args, **_kwargs):
        raise AssertionError("preflight must stop before GPU or policy initialization")

    monkeypatch.setattr(
        train_openvla_oft_module,
        "prepare_libero_runtime_paths",
        lambda: {},
    )
    monkeypatch.setattr(
        train_openvla_oft_module,
        "require_compatible_runtime",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        train_openvla_oft_module,
        "validate_libero_task_assets",
        lambda _settings: {"tasks": 10, "bddl_files": 10, "init_state_files": 10},
    )
    monkeypatch.setattr(
        train_openvla_oft_module,
        "validate_libero_runtime_imports",
        fail,
    )
    monkeypatch.setattr(
        libero_records_module,
        "_task_languages",
        lambda settings: {task_id: f"task {task_id}" for task_id in settings.task_ids},
    )

    monkeypatch.setattr(
        train_openvla_oft_module,
        "validate_runtime_device_availability",
        fail,
    )
    monkeypatch.setattr(train_openvla_oft_module, "make_policy", fail)

    asyncio.run(train_openvla_oft_module.run(config_path, preflight=True))

    output = capsys.readouterr().out
    assert '"status": "ok"' in output
    assert '"train_scenarios": 128' in output
    assert '"evaluation_scenarios": 100' in output


@pytest.mark.parametrize("policy_load_fails", [False, True])
def test_training_entrypoint_starts_observability_before_policy_load(
    monkeypatch: pytest.MonkeyPatch,
    policy_load_fails: bool,
) -> None:
    config_path = (
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml"
    )
    order: list[str] = []
    events = []
    close_codes: list[int] = []

    class FakeObserver:
        async def log_progress(self, event, _config) -> None:
            events.append(event)

        def close(self, *, exit_code: int = 0) -> None:
            close_codes.append(exit_code)

    observer = FakeObserver()

    def start_observer(_config):
        order.append("observer")
        return observer

    def load_policy(_config):
        order.append("policy")
        if policy_load_fails:
            raise RuntimeError("load failed")
        return object()

    async def run_experiment(**kwargs) -> None:
        order.append("experiment")
        assert kwargs["observer"] is observer

    monkeypatch.setattr(
        train_openvla_oft_module, "require_compatible_runtime", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        train_openvla_oft_module, "prepare_libero_runtime_paths", lambda: {}
    )
    monkeypatch.setattr(
        train_openvla_oft_module,
        "validate_libero_runtime_imports",
        lambda: order.append("runtime"),
    )
    monkeypatch.setattr(
        train_openvla_oft_module,
        "validate_libero_task_assets",
        lambda _settings: {},
    )
    monkeypatch.setattr(
        train_openvla_oft_module, "build_evaluation_scenarios", lambda _config: []
    )
    monkeypatch.setattr(
        train_openvla_oft_module, "build_train_scenarios", lambda _config: []
    )
    monkeypatch.setattr(
        train_openvla_oft_module,
        "validate_runtime_device_availability",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        train_openvla_oft_module.WandbWeaveObserver,
        "start",
        start_observer,
    )
    monkeypatch.setattr(train_openvla_oft_module, "make_policy", load_policy)
    monkeypatch.setattr(
        train_openvla_oft_module, "run_lerobot_experiment", run_experiment
    )

    if policy_load_fails:
        with pytest.raises(RuntimeError, match="load failed"):
            asyncio.run(train_openvla_oft_module.run(config_path))
        assert order == ["observer", "runtime", "policy"]
        assert [event.status for event in events] == ["started", "failed"]
        assert close_codes == [1]
    else:
        asyncio.run(train_openvla_oft_module.run(config_path))
        assert order == ["observer", "runtime", "policy", "experiment"]
        assert [event.status for event in events] == ["started", "completed"]
        assert close_codes == [0]
    assert all(event.phase == "initialization" for event in events)
    assert all(event.update == 0 for event in events)


def test_training_initialization_uses_resume_checkpoint_step(tmp_path: Path) -> None:
    config = EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml"
    )
    checkpoint = tmp_path / "step-000100"
    checkpoint.mkdir()
    (checkpoint / "art_embodied_training_state.json").write_text(
        '{"step": 100}\n', encoding="utf-8"
    )
    config = config.model_copy(
        update={
            "storage": config.storage.model_copy(
                update={"resume_from_checkpoint": checkpoint}
            )
        }
    )

    assert (
        train_openvla_oft_module._initialization_update(
            config,
            evaluate_only=False,
            evaluation_step=0,
        )
        == 100
    )


def test_training_initialization_uses_v2_resume_checkpoint_update_step(
    tmp_path: Path,
) -> None:
    config = EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml"
    )
    checkpoint = tmp_path / "step-000001"
    checkpoint.mkdir()
    (checkpoint / "art_embodied_training_state.json").write_text(
        '{"schema_version": 2, "update_step": 1, "backend_step": 1}\n',
        encoding="utf-8",
    )
    config = config.model_copy(
        update={
            "storage": config.storage.model_copy(
                update={"resume_from_checkpoint": checkpoint}
            )
        }
    )

    assert (
        train_openvla_oft_module._initialization_update(
            config,
            evaluate_only=False,
            evaluation_step=0,
        )
        == 1
    )


def test_openvla_action_processing_preserves_decoded_float_dtype() -> None:
    decoded = np.zeros((8, 7), dtype=np.float64)
    decoded[:, 0] = np.nextafter(0.25, 1.0)

    processed = process_openvla_action_chunk(
        decoded,
        chunk_size=8,
        normalize_gripper=True,
        binarize_gripper=True,
        invert_gripper=True,
    )

    assert processed.dtype == np.float64
    np.testing.assert_array_equal(processed[:, 0], decoded[:, 0])


def test_shared_rollout_profile_uses_one_model_for_multiple_actors() -> None:
    config = _shared_rollout_config()
    summary = config.execution_summary()

    assert config.runtime.rollout_execution.inference_mode == "batched_server"
    assert summary["rollout_actor_count"] == 8
    assert summary["rollout_model_replicas"] == 4
    assert summary["rollout_environment_slots"] == 64
    assert summary["trajectories_per_update"] == 1024


def test_h100_profile_uses_measured_replica_and_actor_geometry() -> None:
    config = _h100_rollout_config()
    summary = config.execution_summary()

    assert config.training.optimizer.learning_rate == 2.0e-4
    assert config.rollout.workers == 48
    assert config.runtime.rollout_execution.actors_per_device == 6
    assert config.runtime.rollout_execution.inference_replicas_per_device == 3
    assert config.runtime.rollout_execution.inference_max_batch_size == 2
    assert config.runtime.rollout_execution.lifecycle == "cpu_offload"
    assert config.runtime.training_worker_lifecycle == "cpu_offload"
    assert summary["rollout_actor_count"] == 48
    assert summary["rollout_model_replicas"] == 24
    assert summary["trajectories_per_update"] == 1024


def test_shared_rollout_components_do_not_load_policy_in_actor(
    monkeypatch,
) -> None:
    config = _shared_rollout_config()

    def fail_policy_load(_config):
        raise AssertionError("shared simulator actor must not load OpenVLA")

    monkeypatch.setattr(libero_rollout_module, "make_policy", fail_policy_load)
    monkeypatch.setattr(
        libero_rollout_module,
        "LiberoTaskCatalog",
        lambda _settings: SimpleNamespace(make_environment=lambda *_args: None),
    )
    components = libero_rollout_module.create_components(
        config=config,
        context=SimpleNamespace(local_device="cpu"),
    )

    assert components.group_rollout is not None
    components.load_policy_snapshot(update=0, policy_snapshot=Path("unused"))


def test_libero_evaluation_scenarios_are_explicit_task_major_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        libero_records_module,
        "_task_languages",
        lambda settings: {task_id: f"task {task_id}" for task_id in settings.task_ids},
    )
    scenarios = build_evaluation_scenarios(_baseline_config())

    assert len(scenarios) == 100
    assert scenarios[0].id == "libero_object/eval/task-00/trial-40"
    assert scenarios[0].payload == {
        "task_id": 0,
        "reset_options": {"trial_id": 40},
    }
    assert scenarios[9].id == "libero_object/eval/task-00/trial-49"
    assert scenarios[10].id == "libero_object/eval/task-01/trial-40"
    assert scenarios[-1].id == "libero_object/eval/task-09/trial-49"


def test_openvla_adapter_seeds_once_at_episode_reset(monkeypatch) -> None:
    seed_calls: list[int] = []
    monkeypatch.setattr(libero_policy_module, "_seed_policy", seed_calls.append)

    class FakePolicy:
        def act(self, _observation, _context):
            return Action(
                step=0,
                kind="token",
                raw={"tokens": list(range(56))},
                decoded=np.zeros((8, 7), dtype=np.float32).tolist(),
                logprobs={"token_logprobs": [-0.1] * 56},
            )

    adapter = OpenVLAAdapter(FakePolicy(), LiberoSettings.from_config(_config()))
    adapter.reset(seed=11)
    adapter.predict(
        {
            "image": np.zeros((4, 4, 3), dtype=np.uint8),
            "proprio_state": np.zeros(8, dtype=np.float32),
        },
        task="pick",
        step=0,
        seed=12,
    )
    adapter.predict(
        {
            "image": np.zeros((4, 4, 3), dtype=np.uint8),
            "proprio_state": np.zeros(8, dtype=np.float32),
        },
        task="pick",
        step=1,
        seed=13,
    )

    assert seed_calls == [11]


def test_openvla_adapter_satisfies_public_lerobot_rollout_protocol() -> None:
    adapter = OpenVLAAdapter(
        policy=object(), settings=LiberoSettings.from_config(_config())
    )

    assert isinstance(adapter, LeRobotPolicyAdapterProtocol)


def test_libero_settings_reject_nonterminal_reward_contract() -> None:
    config = _config()
    config = config.model_copy(
        update={"reward": config.reward.model_copy(update={"terminal_only": False})}
    )

    with pytest.raises(ValueError, match="reward.terminal_only=true"):
        LiberoSettings.from_config(config)


def test_libero_settings_reject_incomplete_rlinf_task_geometry() -> None:
    config = _config()
    config = config.model_copy(
        update={
            "environment": config.environment.model_copy(
                update={
                    "kwargs": {
                        **config.environment.kwargs,
                        "task_ids": list(range(9)),
                    }
                }
            )
        }
    )

    with pytest.raises(ValueError, match=r"task_ids=\[0, \.\.\., 9\]"):
        LiberoSettings.from_config(config)


def test_rlinf_ordered_reset_schedule_matches_epoch_scoped_rank_stream() -> None:
    settings = LiberoSettings.from_config(_config())
    schedule = settings.reset_schedule
    process_count = schedule["total_num_processes"]
    groups_per_process = schedule["groups_per_process_per_rollout_epoch"]
    groups_per_epoch = process_count * groups_per_process
    total_states = schedule["total_reset_states"]
    valid_size = total_states - total_states % process_count
    row_size = valid_size // process_count
    epochs_per_shuffle = row_size // groups_per_process

    for sequence_index in (0, groups_per_epoch, 3153, 3154, 3155):
        epoch, group_in_epoch = divmod(sequence_index, groups_per_epoch)
        process, offset = divmod(group_in_epoch, groups_per_process)
        shuffle_cycle, epoch_in_shuffle = divmod(epoch, epochs_per_shuffle)
        generator = np.random.default_rng(schedule["numpy_seed"])
        matrix = None
        for _ in range(shuffle_cycle + 1):
            reset_ids = np.arange(total_states)
            generator.shuffle(reset_ids)
            matrix = reset_ids[:valid_size].reshape(process_count, -1)
        assert matrix is not None
        reset_id = int(matrix[process, epoch_in_shuffle * groups_per_process + offset])
        assert rlinf_v01_task_and_trial(sequence_index, schedule) == (
            reset_id // 50,
            reset_id % 50,
        )


def test_rlinf_random_reset_schedule_advances_rank_state_each_rollout_epoch() -> None:
    schedule = LiberoSettings.from_config(_config()).reset_schedule
    process_count = schedule["total_num_processes"]
    groups_per_process = schedule["groups_per_process_per_rollout_epoch"]
    groups_per_epoch = process_count * groups_per_process
    for epoch in range(12):
        for rank in range(process_count):
            generator = np.random.default_rng(schedule["numpy_seed"] + rank)
            draws = generator.integers(
                0,
                schedule["total_reset_states"],
                size=(epoch + 1) * groups_per_process,
            )
            for offset in range(groups_per_process):
                reset_id = int(draws[epoch * groups_per_process + offset])
                expected = (reset_id // 50, reset_id % 50)
                sequence_index = (
                    epoch * groups_per_epoch + rank * groups_per_process + offset
                )
                assert (
                    rlinf_v01_random_task_and_trial(sequence_index, schedule)
                    == expected
                )


def test_partitioned_random_reset_is_deterministic_and_disjoint() -> None:
    config = _smolvla_positive_control_config()
    training_trials = tuple(range(10, 50))
    kwargs = {
        **config.environment.kwargs,
        "training_task_ids": [0, 4],
        "training_trial_ids": list(training_trials),
        "init_state_selection": "partitioned_random_reset",
    }
    changed = config.model_copy(
        update={"environment": config.environment.model_copy(update={"kwargs": kwargs})}
    )
    settings = LiberoSettings.from_config(changed)
    pairs = [
        partitioned_random_task_and_trial(
            sequence_index,
            settings.reset_schedule,
            task_ids=settings.training_task_ids or settings.task_ids,
            trial_ids=settings.training_trial_ids or (),
        )
        for sequence_index in range(2048)
    ]

    assert pairs == [
        partitioned_random_task_and_trial(
            sequence_index,
            settings.reset_schedule,
            task_ids=settings.training_task_ids or settings.task_ids,
            trial_ids=settings.training_trial_ids or (),
        )
        for sequence_index in range(2048)
    ]
    assert {task_id for task_id, _ in pairs} == {0, 4}
    assert all(trial_id in training_trials for _, trial_id in pairs)
    assert not {trial_id for _, trial_id in pairs}.intersection(
        settings.evaluation_trial_ids
    )


def test_balanced_task_random_reset_covers_every_task_once_per_panel() -> None:
    schedule = LiberoSettings.from_config(_config()).reset_schedule
    task_ids = tuple(range(10))
    trial_ids = tuple(range(10, 50))

    first = [
        balanced_task_random_trial(
            index,
            schedule,
            task_ids=task_ids,
            trial_ids=trial_ids,
        )
        for index in range(20)
    ]

    assert [task for task, _trial in first[:10]] == list(task_ids)
    assert [task for task, _trial in first[10:]] == list(task_ids)
    assert all(trial in trial_ids for _task, trial in first)


def test_partitioned_random_reset_rejects_evaluation_overlap() -> None:
    config = _smolvla_positive_control_config()
    kwargs = {
        **config.environment.kwargs,
        "training_task_ids": [0, 4],
        "training_trial_ids": list(range(50)),
        "init_state_selection": "partitioned_random_reset",
    }
    changed = config.model_copy(
        update={"environment": config.environment.model_copy(update={"kwargs": kwargs})}
    )

    with pytest.raises(ValueError, match="partitions overlap"):
        LiberoSettings.from_config(changed)


def test_partitioned_random_reset_rejects_nonexistent_trial_for_task_subset() -> None:
    config = _smolvla_positive_control_config()
    kwargs = {
        **config.environment.kwargs,
        "training_task_ids": [0, 4],
        "training_trial_ids": [50],
        "init_state_selection": "partitioned_random_reset",
    }
    changed = config.model_copy(
        update={"environment": config.environment.model_copy(update={"kwargs": kwargs})}
    )

    with pytest.raises(ValueError, match=r"valid range is \[0, 49\]"):
        LiberoSettings.from_config(changed)


def test_partitioned_training_scenarios_never_use_fixed_evaluation_trials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        libero_records_module,
        "_task_languages",
        lambda settings: {task_id: f"task-{task_id}" for task_id in settings.task_ids},
    )
    config = _smolvla_positive_control_config()
    kwargs = {
        **config.environment.kwargs,
        "training_task_ids": [0, 4],
        "training_trial_ids": list(range(10, 50)),
        "init_state_selection": "partitioned_random_reset",
    }
    changed = config.model_copy(
        update={
            "environment": config.environment.model_copy(update={"kwargs": kwargs}),
            "training": config.training.model_copy(update={"updates": 32}),
        }
    )

    scenarios = build_train_scenarios(changed)
    trial_ids = {
        int(scenario.payload["reset_options"]["trial_id"]) for scenario in scenarios
    }

    assert trial_ids.issubset(set(range(10, 50)))
    assert trial_ids.isdisjoint(set(range(10)))
    assert {int(scenario.payload["task_id"]) for scenario in scenarios} == {0, 4}


def test_pi0_training_scenarios_have_rlinf_epoch_scoped_reset_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        libero_records_module,
        "_task_languages",
        lambda settings: {task_id: f"task-{task_id}" for task_id in settings.task_ids},
    )
    config = _spatial_config(
        "pi0_libero_spatial_flow_sde_grpo_rlinf_positive_control.yaml"
    )
    groups_per_epoch = config.training.schedule.actor_world_size
    groups_per_update = (
        config.rollout.groups_per_update * config.rollout.epochs_per_update
    )
    scenarios = build_train_scenarios(config)[:groups_per_update]
    pairs = [
        (
            int(scenario.payload["task_id"]),
            int(scenario.payload["reset_options"]["trial_id"]),
        )
        for scenario in scenarios
    ]

    expected = []
    schedule = LiberoSettings.from_config(config).reset_schedule
    for sequence_index in range(groups_per_update):
        expected.append(rlinf_v01_random_task_and_trial(sequence_index, schedule))
    assert pairs == expected
    assert len(set(pairs)) > groups_per_epoch


@pytest.mark.parametrize(
    ("control_name", "large_name"),
    [
        (
            "pi0_libero_spatial_flow_sde_grpo_rlinf_positive_control.yaml",
            "pi0_libero_spatial_flow_sde_grpo_1024.yaml",
        ),
        (
            "pi05_libero_long_flow_sde_grpo_rlinf_positive_control.yaml",
            "pi05_libero_long_flow_sde_grpo_1024.yaml",
        ),
    ],
)
def test_pi_flow_1024_profiles_extend_the_same_reset_stream(
    control_name: str,
    large_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        libero_records_module,
        "_task_languages",
        lambda settings: {task_id: f"task-{task_id}" for task_id in settings.task_ids},
    )

    def reset_pairs(config: EmbodiedExperimentConfig) -> list[tuple[int, int]]:
        group_count = (
            config.rollout.groups_per_update * config.rollout.epochs_per_update
        )
        return [
            (
                int(scenario.payload["task_id"]),
                int(scenario.payload["reset_options"]["trial_id"]),
            )
            for scenario in build_train_scenarios(config)[:group_count]
        ]

    control_pairs = reset_pairs(_spatial_config(control_name))
    large_pairs = reset_pairs(_spatial_config(large_name))

    assert len(control_pairs) == 64
    assert len(large_pairs) == 128
    assert large_pairs[: len(control_pairs)] == control_pairs
    assert len(set(control_pairs)) == 62
    assert len(set(large_pairs)) == 112


def test_rlinf_ordered_reset_schedule_advances_state_each_rollout_epoch() -> None:
    schedule = LiberoSettings.from_config(_config()).reset_schedule
    process_count = schedule["total_num_processes"]
    groups_per_process = schedule["groups_per_process_per_rollout_epoch"]
    groups_per_epoch = process_count * groups_per_process
    first_epoch = [
        rlinf_v01_task_and_trial(group, schedule) for group in range(groups_per_epoch)
    ]
    second_epoch = [
        rlinf_v01_task_and_trial(groups_per_epoch + group, schedule)
        for group in range(groups_per_epoch)
    ]
    assert second_epoch != first_epoch


def test_openvla_action_chunk_matches_libero_gripper_convention() -> None:
    decoded = np.zeros((8, 7), dtype=np.float32)
    decoded[:, -1] = np.array([0.0, 0.25, 0.49, 0.5, 0.51, 0.75, 1.0, 1.5])

    processed = process_openvla_action_chunk(
        decoded,
        chunk_size=8,
        normalize_gripper=True,
        binarize_gripper=True,
        invert_gripper=True,
    )

    np.testing.assert_array_equal(
        processed[:, -1],
        np.array([1.0, 1.0, 1.0, 0.0, -1.0, -1.0, -1.0, -1.0]),
    )


def test_openvla_action_chunk_rejects_short_fixed_horizon() -> None:
    with pytest.raises(ValueError, match="shorter than the configured fixed geometry"):
        process_openvla_action_chunk(
            np.zeros((7, 7), dtype=np.float32),
            chunk_size=8,
            normalize_gripper=True,
            binarize_gripper=True,
            invert_gripper=True,
        )


def test_libero_observation_retains_training_inputs() -> None:
    observation = {
        "image": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
        "proprio_state": np.arange(8, dtype=np.float32),
    }

    recorded = record_libero_observation(observation=observation, step=3)

    assert recorded.kind == "image"
    np.testing.assert_array_equal(recorded.value["image"], observation["image"])
    np.testing.assert_array_equal(
        recorded.value["proprio_state"], observation["proprio_state"]
    )
    assert recorded.metadata["fields"]["image"]["shape"] == [2, 2, 3]


def test_libero_transition_records_rlinf_action_level_contract() -> None:
    action = Action(
        step=2,
        kind="token",
        raw={"tokens": list(range(56))},
        decoded=np.zeros((8, 7)).tolist(),
        logprobs={"token_logprobs": [-0.1] * 56},
        metadata={"policy_step": 2},
    )
    trajectory = EmbodiedTrajectory(task="pick")
    primitive_observations = [
        {"image": np.full((2, 2, 3), value, dtype=np.uint8)} for value in (1, 2)
    ]

    record_libero_transition(
        trajectory=trajectory,
        action=action,
        observation={},
        reward=1.0,
        terminated=True,
        truncated=False,
        info={
            "primitive_rewards": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "primitive_loss_mask": [
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            "executed_actions": [[0.0] * 7, [1.0] * 7],
            "primitive_observations_before": primitive_observations,
            "env_steps": 2,
        },
        policy_step=2,
    )

    assert action.metadata["primitive_loss_mask_sum"] == 2
    assert action.metadata["primitive_observations_before"] is primitive_observations
    assert len(action.metadata["primitive_rewards"]) == 8
    assert len(trajectory.rewards) == 8
    assert trajectory.reward == 0.0
    assert [event.metadata["primitive_loss_mask"] for event in trajectory.rewards] == [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert [event.step for event in trajectory.rewards] == [1, 2, 2, 2, 2, 2, 2, 2]


def test_libero_transition_does_not_unmask_non_policy_fallback():
    from art_embodied.backends.action_token import _is_fully_masked_action

    action = Action(
        step=0,
        kind="token",
        raw={"tokens": [1]},
        metadata={"primitive_loss_mask_sum": 0},
    )
    trajectory = EmbodiedTrajectory(task="pick")
    observations = [{"state": [1]}, {"state": [2]}]
    record_libero_transition(
        trajectory=trajectory,
        action=action,
        reward=1.0,
        policy_step=0,
        info={
            "primitive_rewards": [0.0, 1.0, 0.0],
            "primitive_loss_mask": [True, True, False],
            "executed_actions": [[0.0] * 7, [0.0] * 7],
            "env_steps": 9,
            "primitive_observations_before": observations,
        },
    )
    assert _is_fully_masked_action(action)
    assert action.metadata["primitive_loss_mask"] == [False] * 3
    assert action.metadata["primitive_observations_before"] is observations
    assert [e.step for e in trajectory.rewards] == [8, 9, 9]
    assert not any(e.metadata["primitive_loss_mask"] for e in trajectory.rewards)


def test_libero_group_rollout_batches_policy_and_keeps_trajectory_outcomes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config().model_copy(
        update={
            "storage": _config().storage.model_copy(
                update={"output_dir": tmp_path / "output"}
            ),
            "observability": _config().observability.model_copy(
                update={
                    "videos_per_update": 0,
                    "videos_per_evaluation": 0,
                    "require_train_video": False,
                    "require_evaluation_video": False,
                }
            ),
        }
    )
    settings = LiberoSettings.from_config(config)

    class FakeEnvironment:
        def __init__(self, attempt_index: int) -> None:
            self.attempt_index = attempt_index
            self.closed = False
            self.policy_steps = 0

        def reset(self, *, seed: int, options):
            del seed, options
            return {
                "image": np.zeros((4, 4, 3), dtype=np.uint8),
                "proprio_state": np.zeros(8, dtype=np.float32),
            }, {"attempt_index": self.attempt_index}

        def step(self, _action):
            self.policy_steps += 1
            terminated = self.policy_steps >= 2
            success = terminated and self.attempt_index % 2 == 0
            rewards = [5.0 if success else 0.0] + [0.0] * 7
            return (
                {
                    "image": np.zeros((4, 4, 3), dtype=np.uint8),
                    "proprio_state": np.zeros(8, dtype=np.float32),
                },
                float(sum(rewards)),
                terminated,
                False,
                {
                    "success": success,
                    "primitive_rewards": rewards,
                    "primitive_loss_mask": [True] + [False] * 7,
                    "executed_actions": [[0.0] * 7],
                    "env_steps": 1,
                },
            )

        def close(self) -> None:
            self.closed = True

    class FakeCatalog:
        def __init__(self) -> None:
            self.environments = []

        def make_environment(self, _scenario, context):
            env = FakeEnvironment(context.attempt_index)
            self.environments.append(env)
            return env

    class FakePolicy:
        def __init__(self) -> None:
            self.batch_sizes = []

        def act_batch(self, observations, contexts):
            assert len(observations) == len(contexts)
            self.batch_sizes.append(len(observations))
            return [
                Action(
                    step=0,
                    kind="token",
                    raw={"tokens": list(range(56))},
                    decoded=np.zeros((8, 7), dtype=np.float32).tolist(),
                    logprobs={"token_logprobs": [-0.1] * 56},
                )
                for _ in observations
            ]

    contexts = tuple(
        RolloutContext(
            update=0,
            group_index=0,
            attempt_index=index,
            environment_seed=10,
            policy_seed=20 + index,
            config_fingerprint=config.fingerprint,
        )
        for index in range(config.algorithm.group_size)
    )
    catalog = FakeCatalog()
    policy = FakePolicy()
    seed_calls: list[int] = []
    monkeypatch.setattr(libero_rollout_module, "_seed_policy", seed_calls.append)

    trajectories = asyncio.run(
        rollout_libero_group(
            config=config,
            policy=policy,
            catalog=catalog,
            settings=settings,
            scenario=EmbodiedScenario(
                id="libero_object/train/0",
                task="pick",
                payload={"task_id": 0, "reset_options": {"trial_id": 0}},
            ),
            contexts=contexts,
            phase="train",
        )
    )

    assert policy.batch_sizes == [
        config.algorithm.group_size,
        config.algorithm.group_size,
    ]
    assert len(seed_calls) == 1
    assert [trajectory.metrics["success"] for trajectory in trajectories] == [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    assert [trajectory.reward for trajectory in trajectories] == [
        5.0,
        0.0,
        5.0,
        0.0,
        5.0,
        0.0,
        5.0,
        0.0,
    ]
    assert all(environment.closed for environment in catalog.environments)

    embedded_catalog = FakeCatalog()
    embedded_batch_sizes: list[int] = []
    embedded_reset_seeds: list[int] = []

    def embedded_predictor(observations, *, tasks, step):
        assert all(isinstance(observation, dict) for observation in observations)
        assert all("image" in observation for observation in observations)
        assert tasks == ["pick"] * len(observations)
        embedded_batch_sizes.append(len(observations))
        return [
            SimpleNamespace(
                action=Action(
                    step=step,
                    kind="token",
                    raw={"tokens": list(range(56))},
                    decoded=np.zeros((8, 7), dtype=np.float32).tolist(),
                    logprobs={"token_logprobs": [-0.1] * 56},
                ),
                native_action=np.zeros((8, 7), dtype=np.float32),
            )
            for _ in observations
        ]

    embedded_trajectories = asyncio.run(
        rollout_libero_group(
            config=config,
            policy=object(),
            catalog=embedded_catalog,
            settings=settings,
            scenario=EmbodiedScenario(
                id="libero_object/train/0",
                task="pick",
                payload={"task_id": 0, "reset_options": {"trial_id": 0}},
            ),
            contexts=contexts,
            phase="train",
            embedded_batch_predictor=embedded_predictor,
            embedded_batch_reset=embedded_reset_seeds.append,
        )
    )

    assert embedded_batch_sizes == [8, 8]
    assert len(embedded_reset_seeds) == 1
    assert [item.metrics["success"] for item in embedded_trajectories] == [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    assert all(environment.closed for environment in embedded_catalog.environments)

    class FakePolicyClient:
        def __init__(self) -> None:
            self.requests = []

        async def predict(self, request):
            self.requests.append(request)
            if request["op"] == "release_rng_stream":
                return {"released": True}
            return {
                "actions": [
                    Action(
                        step=context["step"],
                        kind="token",
                        raw={"tokens": list(range(56))},
                        decoded=np.zeros((8, 7), dtype=np.float32).tolist(),
                        logprobs={"token_logprobs": [-0.1] * 56},
                    )
                    for context in request["contexts"]
                ]
            }

    shared_catalog = FakeCatalog()
    client = FakePolicyClient()
    shared_trajectories = asyncio.run(
        rollout_libero_group(
            config=config,
            policy=None,
            policy_client=client,
            catalog=shared_catalog,
            settings=settings,
            scenario=EmbodiedScenario(
                id="libero_object/train/0",
                task="pick",
                payload={"task_id": 0, "reset_options": {"trial_id": 0}},
            ),
            contexts=contexts,
            phase="train",
        )
    )

    prediction_requests = [
        request for request in client.requests if request["op"] == "predict_group"
    ]
    assert [len(request["observations"]) for request in prediction_requests] == [8, 8]
    assert prediction_requests[0]["reset_rng_stream"] is True
    assert prediction_requests[1]["reset_rng_stream"] is False
    assert client.requests[-1]["op"] == "release_rng_stream"
    assert all(item.metadata["shared_inference"] for item in shared_trajectories)
    assert all(
        action.metadata["shared_inference_model_batch_size"] == 8
        and action.metadata["shared_inference_server_request_batch_size"] == 1
        for item in shared_trajectories
        for action in item.actions
    )
    assert [item.metrics["success"] for item in shared_trajectories] == [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]

    eval_catalog = FakeCatalog()
    eval_client = FakePolicyClient()
    eval_trajectory = asyncio.run(
        rollout_libero_group(
            config=config,
            policy=None,
            policy_client=eval_client,
            catalog=eval_catalog,
            settings=settings,
            scenario=EmbodiedScenario(
                id="libero_object/eval/0",
                task="pick",
                payload={"task_id": 0, "reset_options": {"trial_id": 40}},
            ),
            contexts=(contexts[0],),
            phase="eval",
        )
    )[0]
    eval_prediction = eval_client.requests[0]
    assert eval_prediction["phase"] == "eval"
    assert eval_prediction["rng_seed"] == contexts[0].policy_seed
    assert eval_trajectory.metadata["vectorized_group_rollout"] is False


def test_libero_group_rollout_closes_partially_created_environments(
    tmp_path: Path,
) -> None:
    config = _config().model_copy(
        update={
            "storage": _config().storage.model_copy(
                update={"output_dir": tmp_path / "output"}
            )
        }
    )
    settings = LiberoSettings.from_config(config)

    class FakeEnvironment:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FailingCatalog:
        def __init__(self) -> None:
            self.environments: list[FakeEnvironment] = []

        def make_environment(self, _scenario, context):
            if context.attempt_index == 3:
                raise RuntimeError("environment construction failed")
            environment = FakeEnvironment()
            self.environments.append(environment)
            return environment

    contexts = tuple(
        RolloutContext(
            update=0,
            group_index=0,
            attempt_index=index,
            environment_seed=10,
            policy_seed=20 + index,
            config_fingerprint=config.fingerprint,
        )
        for index in range(config.algorithm.group_size)
    )
    catalog = FailingCatalog()

    with pytest.raises(RuntimeError, match="environment construction failed"):
        asyncio.run(
            rollout_libero_group(
                config=config,
                policy=object(),
                catalog=catalog,
                settings=settings,
                scenario=EmbodiedScenario(
                    id="libero_object/train/0",
                    task="pick",
                    payload={"task_id": 0},
                ),
                contexts=contexts,
                phase="train",
            )
        )

    assert len(catalog.environments) == 3
    assert all(environment.closed for environment in catalog.environments)


def test_rejected_decision_terminates_only_its_episode_and_remains_trainable(
    monkeypatch,
):
    from art_embodied.backends.action_token import extract_action_token_examples
    from art_embodied.trajectories import EmbodiedTrajectoryGroup

    config = _config()
    monkeypatch.setattr(
        libero_rollout_module, "_group_video_recorder", lambda **kw: None
    )
    monkeypatch.setattr(
        libero_rollout_module, "_group_lookahead_recorder", lambda **kw: None
    )
    monkeypatch.setattr(libero_rollout_module, "_seed_policy", lambda seed: None)

    class Env:
        def __init__(self, index):
            self.index, self.steps, self.closed = index, 0, False

        def reset(self, **kw):
            return {
                "image": np.zeros((4, 4, 3), dtype=np.uint8),
                "proprio_state": np.zeros(8),
            }, {}

        def step(self, action):
            assert self.index > 0, "Rejected action must never reach env.step"
            self.steps += 1
            return (
                self.reset()[0],
                1.0,
                True,
                False,
                {
                    "success": True,
                    "primitive_rewards": [1.0],
                    "primitive_loss_mask": [True],
                    "executed_actions": [[0.0] * 7],
                    "env_steps": 1,
                },
            )

        def close(self):
            self.closed = True

    envs = []

    def make_environment(scenario, context):
        env = Env(context.attempt_index)
        envs.append(env)
        return env

    class Policy:
        def act_batch(self, observations, contexts):
            return [
                Action(
                    step=0,
                    kind="token",
                    raw={"tokens": [1, 2], "prompt": "pick"},
                    logprobs=[-0.1, -0.2],
                    decoded=[] if i == 0 else [[0.0] * 7] * 8,
                    metadata={
                        "token_loss_mask": [True, True],
                        **(
                            {
                                "terminate_episode": True,
                                "termination_reason": "invalid_action_tokens",
                                "executed_primitive_count": 0,
                                "execution_horizon": 0,
                                "action_grammar_valid": False,
                            }
                            if i == 0
                            else {}
                        ),
                    },
                )
                for i in range(len(observations))
            ]

    contexts = tuple(
        RolloutContext(
            update=0,
            group_index=0,
            attempt_index=i,
            environment_seed=10,
            policy_seed=20 + i,
            config_fingerprint=config.fingerprint,
        )
        for i in range(config.algorithm.group_size)
    )
    trajectories = asyncio.run(
        rollout_libero_group(
            config=config,
            policy=Policy(),
            catalog=SimpleNamespace(make_environment=make_environment),
            settings=LiberoSettings.from_config(config),
            scenario=EmbodiedScenario(id="test", task="pick", payload={"task_id": 0}),
            contexts=contexts,
            phase="train",
        )
    )
    assert envs[0].steps == 0
    assert all(e.steps == 1 and e.closed for e in envs[1:])
    assert envs[0].closed
    assert trajectories[0].reward == 0
    assert all(t.reward > 0 for t in trajectories[1:])
    assert trajectories[0].metrics["terminated"] is True
    examples = extract_action_token_examples(
        [EmbodiedTrajectoryGroup(trajectories)], require_logprobs=True
    )
    rejected = [e for e in examples if e.metadata["trajectory_index_in_group"] == 0]
    assert len(rejected) == 1
    assert rejected[0].metadata["group_advantage"] < 0
