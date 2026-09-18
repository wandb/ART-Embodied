from __future__ import annotations

from pathlib import Path

import pydantic
import pytest
import yaml

from art_embodied.config import EmbodiedExperimentConfig, PI0FastLoadConfig

POSITIVE_CONTROL_CONFIG = (
    Path(__file__).parents[1]
    / "examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml"
)
SPATIAL_POSITIVE_CONTROL_CONFIG = (
    Path(__file__).parents[1]
    / "examples/embodied/openvla_oft_libero_spatial_grpo_rlinf_positive_control.yaml"
)
HIGH_LR_OBJECT_CONFIG = (
    Path(__file__).parents[1]
    / "examples/embodied/openvla_oft_libero_object_grpo_lora_lr2e4.yaml"
)
H100_OBJECT_CONFIG = (
    Path(__file__).parents[1]
    / "examples/embodied/openvla_oft_libero_object_grpo_lora_lr2e4_h100.yaml"
)
H100_SPATIAL_CONFIG = (
    Path(__file__).parents[1]
    / "examples/embodied/openvla_oft_libero_spatial_grpo_lora_lr2e4_h100.yaml"
)
PI0_POSITIVE_CONTROL_CONFIG = (
    Path(__file__).parents[1]
    / "examples/embodied/pi0_libero_object_flow_sde_grpo_rlinf_positive_control.yaml"
)
PI0_SPATIAL_POSITIVE_CONTROL_CONFIG = (
    Path(__file__).parents[1]
    / "examples/embodied/pi0_libero_spatial_flow_sde_grpo_rlinf_positive_control.yaml"
)
PI0_LONG_POSITIVE_CONTROL_CONFIG = (
    Path(__file__).parents[1]
    / "examples/embodied/pi0_libero_long_flow_sde_grpo_rlinf_positive_control.yaml"
)
PI05_POSITIVE_CONTROL_CONFIG = (
    Path(__file__).parents[1]
    / "examples/embodied/pi05_libero_object_flow_sde_grpo_rlinf_positive_control.yaml"
)
PI05_SPATIAL_POSITIVE_CONTROL_CONFIG = (
    Path(__file__).parents[1]
    / "examples/embodied/pi05_libero_spatial_flow_sde_grpo_rlinf_positive_control.yaml"
)
PI05_LONG_POSITIVE_CONTROL_CONFIG = (
    Path(__file__).parents[1]
    / "examples/embodied/pi05_libero_long_flow_sde_grpo_rlinf_positive_control.yaml"
)
PI0_SPATIAL_1024_CONFIG = (
    Path(__file__).parents[1]
    / "examples/embodied/pi0_libero_spatial_flow_sde_grpo_1024.yaml"
)
PI05_LONG_1024_CONFIG = (
    Path(__file__).parents[1]
    / "examples/embodied/pi05_libero_long_flow_sde_grpo_1024.yaml"
)
EMBODIED_EXAMPLE_CONFIGS = tuple(
    sorted((Path(__file__).parents[1] / "examples/embodied").glob("*.yaml"))
)


def _assert_model_fields_explicit(
    model: pydantic.BaseModel,
    raw: dict,
    *,
    path: str,
) -> None:
    for name in type(model).model_fields:
        qualified = f"{path}.{name}" if path else name
        if qualified == "evaluation.evaluate_after_first_update" and name not in raw:
            # Historical recipes keep their cadence and original file bytes.
            assert model.evaluate_after_first_update is False
            continue
        if (
            qualified == "policy.lora.rank_partition.forward_routing"
            and name not in raw
        ):
            # Historical recipes must never acquire task-ID routing implicitly.
            assert model.forward_routing == "all_active"
            continue
        if qualified == "rollout.shared_prefix_action_chunks" and name not in raw:
            # Preserve byte-identical historical preregistered recipes. New
            # shared-prefix recipes declare this field explicitly.
            continue
        if qualified == "observability.wandb.log_system_metrics" and name not in raw:
            # Preserve byte-identical historical and sealed recipes. New recipes
            # declare this explicitly when changing the default.
            continue
        if qualified == "observability.wandb.native_update_steps" and name not in raw:
            # Preserve historical recipe bytes and fingerprints. New native-step
            # recipes explicitly opt in and have separate resume/media tests.
            assert model.native_update_steps is False
            continue
        if qualified == "observability.wandb.save_code" and name not in raw:
            # Preserve historical recipe bytes. New claim-facing recipes opt in
            # explicitly when W&B source provenance is part of their contract.
            continue
        if qualified == "training.schedule.update_epochs" and name not in raw:
            # Historical full-update recipes predate repeat epochs and retain
            # their byte-identical default of one. New recipes declare it.
            continue
        assert name in raw, f"complete experiment YAML omitted {qualified}"
        value = getattr(model, name)
        if isinstance(value, pydantic.BaseModel):
            child = raw[name]
            assert isinstance(child, dict), f"{qualified} must be a mapping"
            _assert_model_fields_explicit(value, child, path=qualified)


def test_pi0_fast_load_config_requires_explicit_supported_rl_token_scope() -> None:
    base = {
        "runtime_contract": "lerobot_v060",
        "execution_horizon": 10,
        "action_dim": 7,
        "max_decoding_steps": 256,
        "action_tokenizer_revision": "a" * 40,
        "observation_key_map": {"image": "observation.images.image"},
        "strict_weights": True,
        "compile_model": False,
        "gradient_checkpointing": True,
        "use_kv_cache": True,
    }

    assert PI0FastLoadConfig.model_validate(base).rl_token_scope == (
        "generated_sequence"
    )
    assert PI0FastLoadConfig.model_validate(base).model_compute_dtype == "checkpoint"
    assert PI0FastLoadConfig.model_validate(base).invalid_action_handling == "raise"
    assert PI0FastLoadConfig.model_validate(
        {**base, "invalid_action_handling": "terminate_episode"}
    )
    with pytest.raises(ValueError, match="generated_sequence"):
        PI0FastLoadConfig.model_validate(
            {
                **base,
                "invalid_action_handling": "terminate_episode",
                "rl_token_scope": "fast_payload",
            }
        )
    assert (
        PI0FastLoadConfig.model_validate(
            {**base, "model_compute_dtype": "float32"}
        ).model_compute_dtype
        == "float32"
    )
    assert (
        PI0FastLoadConfig.model_validate(
            {**base, "rl_token_scope": "fast_payload"}
        ).rl_token_scope
        == "fast_payload"
    )
    with pytest.raises(pydantic.ValidationError, match="rl_token_scope"):
        PI0FastLoadConfig.model_validate({**base, "rl_token_scope": "payloadish"})


@pytest.mark.parametrize(
    "path",
    EMBODIED_EXAMPLE_CONFIGS,
)
def test_example_yaml_explicitly_declares_every_config_field(
    path: Path,
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw["storage"].get("resume_from_checkpoint"):
        # Test schema completeness without depending on retained experiment data.
        # Actual checkpoint contents are validated by the checkpointing tests.
        checkpoint = tmp_path / "schema-only-checkpoint"
        checkpoint.mkdir()
        (checkpoint / "art_embodied_training_state.pt").touch()
        raw["storage"]["resume_from_checkpoint"] = str(checkpoint)
    config = EmbodiedExperimentConfig.model_validate(raw)

    _assert_model_fields_explicit(config, raw, path="")


def test_resume_contract_allows_operational_changes_but_not_optimizer_math(
    tmp_path: Path,
) -> None:
    config = EmbodiedExperimentConfig.from_yaml(EMBODIED_EXAMPLE_CONFIGS[3])
    operational = config.model_copy(
        update={
            "experiment": config.experiment.model_copy(update={"run": "resumed"}),
            "training": config.training.model_copy(
                update={"updates": config.training.updates + 100}
            ),
            "storage": config.storage.model_copy(
                update={"output_dir": tmp_path, "resume_from_checkpoint": tmp_path}
            ),
            "runtime": config.runtime.model_copy(
                update={"worker_timeout_seconds": 9999}
            ),
        }
    )
    changed_optimizer = config.model_copy(
        update={
            "training": config.training.model_copy(
                update={
                    "optimizer": config.training.optimizer.model_copy(
                        update={"learning_rate": 9.0e-4}
                    )
                }
            )
        }
    )

    assert operational.fingerprint != config.fingerprint
    assert operational.resume_contract_fingerprint == config.resume_contract_fingerprint
    assert (
        changed_optimizer.resume_contract_fingerprint
        != config.resume_contract_fingerprint
    )


@pytest.mark.parametrize(
    ("connection", "run_id", "resume"),
    [
        ("primary", None, None),
        ("resume", "interrupted-run", "allow"),
    ],
)
def test_online_checkpoint_resume_requires_same_wandb_run(
    tmp_path: Path,
    connection: str,
    run_id: str | None,
    resume: str | None,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "art_embodied_training_state.pt").write_bytes(b"state")
    raw = yaml.safe_load(H100_OBJECT_CONFIG.read_text(encoding="utf-8"))
    raw["evaluation"]["evaluate_before_training"] = False
    raw["storage"]["resume_from_checkpoint"] = str(checkpoint)
    raw["observability"]["wandb"].update(
        connection=connection,
        run_id=run_id,
        resume=resume,
    )

    with pytest.raises(ValueError, match="same run"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_online_checkpoint_resume_accepts_explicit_wandb_recovery(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "art_embodied_training_state.pt").write_bytes(b"state")
    raw = yaml.safe_load(H100_OBJECT_CONFIG.read_text(encoding="utf-8"))
    raw["evaluation"]["evaluate_before_training"] = False
    raw["storage"]["resume_from_checkpoint"] = str(checkpoint)
    raw["observability"]["wandb"].update(
        connection="resume",
        run_id="interrupted-run",
        resume="must",
    )

    config = EmbodiedExperimentConfig.model_validate(raw)

    assert config.storage.resume_from_checkpoint == checkpoint
    assert config.observability.wandb.run_id == "interrupted-run"


def test_pre_training_success_gate_requires_step_zero_evaluation(
    tmp_path: Path,
) -> None:
    raw = _complete_config(tmp_path)
    raw["evaluation"]["evaluate_before_training"] = False
    raw["evaluation"]["pre_training_success_gate"] = {
        "minimum_success_rate": 0.05,
        "maximum_success_rate": 0.90,
    }

    with pytest.raises(
        ValueError,
        match="pre_training_success_gate requires.*evaluate_before_training=true",
    ):
        EmbodiedExperimentConfig.model_validate(raw)


def test_pre_training_success_gate_requires_an_ordered_interval(
    tmp_path: Path,
) -> None:
    raw = _complete_config(tmp_path)
    raw["evaluation"]["evaluate_before_training"] = True
    raw["evaluation"]["pre_training_success_gate"] = {
        "minimum_success_rate": 0.50,
        "maximum_success_rate": 0.50,
    }

    with pytest.raises(ValueError, match="minimum_success_rate must be smaller"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_checkpoint_resume_requires_existing_step_zero_baseline_reference(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "art_embodied_training_state.pt").write_bytes(b"state")
    output_dir = tmp_path / "output"
    baseline = output_dir / "evaluation/update_000000_episode_outcomes.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_text("[]", encoding="utf-8")
    raw = yaml.safe_load(H100_OBJECT_CONFIG.read_text(encoding="utf-8"))
    raw["evaluation"]["evaluate_before_training"] = False
    raw["storage"]["output_dir"] = str(output_dir)
    raw["storage"]["resume_from_checkpoint"] = str(checkpoint)
    raw["observability"]["wandb"].update(
        connection="resume",
        run_id="interrupted-run",
        resume="must",
    )

    with pytest.raises(
        ValueError,
        match="evaluation.baseline_outcomes_path is unset",
    ):
        EmbodiedExperimentConfig.model_validate(raw)

    raw["evaluation"]["baseline_outcomes_path"] = str(baseline)
    resumed = EmbodiedExperimentConfig.model_validate(raw)

    assert resumed.evaluation.baseline_outcomes_path == baseline


def test_resume_contract_ignores_evaluation_state_manifest(tmp_path: Path) -> None:
    config = EmbodiedExperimentConfig.model_validate(_complete_config(tmp_path))
    raw = config.model_dump(mode="python")
    raw["environment"]["kwargs"]["evaluation_state_manifest"] = (
        tmp_path / "sealed-manifest.json"
    )

    modified = EmbodiedExperimentConfig.model_validate(raw)

    assert modified.fingerprint != config.fingerprint
    assert modified.resume_contract_fingerprint == config.resume_contract_fingerprint


def _complete_config(tmp_path: Path) -> dict:
    return {
        "schema_version": 1,
        "experiment": {
            "project": "art-embodied",
            "run": "openvla-libero-object",
            "seed": 7,
            "tags": ["openvla", "grpo"],
        },
        "policy": {
            "type": "openvla_oft",
            "path": "Haozhan72/Openvla-oft-SFT-libero-object-traj1",
            "revision": None,
            "device": "cuda",
            "dtype": "bfloat16",
            "trust_remote_code": True,
            "load_kwargs": {
                "attn_implementation": None,
                "peft_adapter_path": None,
                "dataset_statistics_path": None,
                "logprob_batch_size": 32,
                "strict_batched_logprobs": True,
                "max_prompt_length": 128,
                "prompt_template": "In: {instruction}\nOut:",
                "lowercase_instruction": True,
                "num_images_in_input": 1,
                "use_proprio": False,
                "model_loader": "rlinf",
            },
            "lora": {
                "enabled": True,
                "rank": 32,
                "alpha": 32,
                "dropout": 0.0,
                "init": "gaussian",
                "target_modules": ["q_proj", "v_proj"],
            },
            "unnorm_key": "libero_object_no_noops",
            "rollout_generation": {
                "do_sample": True,
                "temperature": 1.6,
                "top_p": 1.0,
            },
            "train_generation": {
                "do_sample": True,
                "temperature": 1.6,
                "top_p": 1.0,
            },
            "evaluation_generation": {
                "do_sample": True,
                "temperature": 1.6,
                "top_p": 1.0,
            },
            "trainable_parameter_strategy": "openvla_oft_lora",
            "force_trainable_float32": True,
        },
        "environment": {
            "type": "libero",
            "task": "libero_object",
            "robot_type": None,
            "kwargs": {},
            "reset": {"reset_gripper_open": False},
            "observation_processor": {},
            "action_processor": {},
        },
        "reward": {
            "type": "environment",
            "name": "success",
            "terminal_only": True,
            "scale": 5.0,
            "kwargs": {},
        },
        "algorithm": {
            "type": "grpo",
            "flow_sde": None,
            "group_size": 8,
            "clip_epsilon_low": 0.2,
            "clip_epsilon_high": 0.2,
            "kl_coefficient": 0.0,
            "clip_ratio_c": None,
            "importance_sampling_level": "token",
            "training_unit": "action",
            "action_advantage_mode": "rlinf_action_level_cumulative",
            "score_source": "chunk_rewards",
            "pad_fixed_horizon_examples": True,
            "filter_rewards": True,
            "reward_filter_mode": "loss_mask",
            "rewards_lower_bound": 0.5,
            "rewards_upper_bound": 4.5,
            "loss_aggregation": "rlinf_masked_mean_ratio",
            "advantage": {
                "scope": "group",
                "normalize": True,
                "extra_global_normalization": False,
                "mask_zero_variance_groups": False,
                "std_unbiased": False,
                "epsilon": 1e-6,
            },
            "precalculate_logprobs": False,
            "rollout_logprob_source": "rollout_action",
            "logprob_eval_mode": False,
            "logprob_microbatch_size": 32,
            "train_logprob_microbatch_size": 8,
            "pre_update_logprob_kl_tolerance": 0.01,
            "pre_update_ratio_tolerance": 0.02,
            "skip_optimizer_step_without_policy_gradient_signal": False,
        },
        "rollout": {
            "groups_per_update": 16,
            "epochs_per_update": 1,
            "workers": 8,
            "failure_policy": "fail_update",
            "minimum_completed_attempts_per_group": 8,
            "max_episode_steps": 512,
            "max_policy_steps": 64,
            "temperature": 1.6,
            "deterministic": False,
            "action_payload": {
                "kind": "token",
                "require_old_logprobs": True,
                "require_prompt": True,
                "require_observation": True,
            },
        },
        "training": {
            "updates": 100,
            "optimizer_steps_per_update": 1,
            "microbatch_size": 8,
            "log_action_token_progress": True,
            "action_token_progress_every_microbatches": 128,
            "schedule": {
                "type": "rlinf_actor_global_batch",
                "global_batch_size": 8192,
                "actor_seed": 1234,
                "actor_world_size": 1,
                "rank_local_shuffle": True,
                "groups_per_process_per_rollout_epoch": 16,
                "action_chunk_size": 8,
                "strict_geometry": True,
                "pre_update_alignment_guard": "disabled",
            },
            "checkpoint_every_updates": 20,
            "optimizer": {
                "type": "adamw",
                "learning_rate": 2e-5,
                "weight_decay": 0.0,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8,
                "max_grad_norm": 1.0,
            },
        },
        "evaluation": {
            "enabled": True,
            "split": "held_out",
            "data_role": "development",
            "runtime": "native_lerobot",
            "baseline_outcomes_path": None,
            "every_updates": 20,
            "episodes": 100,
            "seeds": [40, 41, 42],
            "deterministic": False,
            "temperature": 1.6,
            "fixed_scenarios": ["libero_object/0", "libero_object/1"],
            "checkpoint_selection": "evaluation_success",
            "kwargs": {},
        },
        "observability": {
            "wandb": {
                "enabled": True,
                "entity": None,
                "project": "art-embodied",
                "group": "test-group",
                "job_type": "trajectory-rl",
                "mode": "online",
                "log_model_artifacts": True,
                "log_evaluation_artifacts": True,
                "log_evaluation_table": True,
                "max_evaluation_table_rows": 100,
                "log_rollout_progress": True,
                "rollout_progress_every_groups": 1,
                "log_evaluation_progress": True,
                "evaluation_progress_every_episodes": 5,
            },
            "weave": {
                "enabled": True,
                "project": "art-embodied",
                "trace_trajectories": True,
                "max_groups_per_update": 4,
                "max_trajectories_per_group": 4,
                "max_evaluation_trajectories": 8,
                "use_server_cache": False,
                "server_cache_dir": None,
                "server_cache_size_mb": 0,
            },
            "videos_per_update": 2,
            "videos_per_evaluation": 8,
            "require_train_video": False,
            "require_evaluation_video": False,
            "video_fps": 20,
        },
        "storage": {
            "output_dir": str(tmp_path / "run"),
            "keep_last_checkpoints": 2,
            "max_log_file_mb": 64,
            "retain_rollout_payloads": False,
        },
        "runtime": {
            "backend": "local",
            "rollout_devices": ["cuda:0"],
            "training_devices": ["cuda:0"],
            "rollout_execution": {
                "mode": "in_process",
                "actor_factory": None,
                "actor_kwargs": {},
                "actors_per_device": 1,
                "group_batching": False,
                "lifecycle": "per_update",
                "policy_sync": "shared",
                "inference_mode": "embedded",
                "inference_factory": None,
                "inference_replicas_per_device": 1,
                "inference_max_batch_size": 1,
                "inference_max_wait_ms": 0.0,
                "startup_timeout_seconds": 7200,
                "request_timeout_seconds": 7200,
            },
            "distributed_training": False,
            "training_worker_lifecycle": "per_update",
            "worker_handoff_dir": str(tmp_path / "handoff"),
            "worker_timeout_seconds": 7200,
            "max_worker_handoff_mb": 8192,
            "keep_worker_handoffs": False,
        },
    }


def test_complete_config_is_stable_and_counts_trajectories(tmp_path: Path) -> None:
    config = EmbodiedExperimentConfig.model_validate(_complete_config(tmp_path))

    assert config.trajectories_per_update == 128
    assert len(config.fingerprint) == 16
    assert config.environment.reset == {"reset_gripper_open": False}
    consumption = config.consumption_report()
    assert consumption["field_count"] > 100
    assert consumption["unowned_fields"] == []
    by_path = {row["path"]: row for row in consumption["fields"]}
    assert by_path["training.microbatch_size"]["status"] == "executed"
    assert by_path["algorithm.train_logprob_microbatch_size"]["status"] == "assertion"
    assert by_path["evaluation.split"]["status"] == "metadata"
    assert by_path["evaluation.data_role"]["status"] == "assertion"
    assert by_path["rollout.groups_per_update"]["status"] == "owned"
    assert config.execution_summary() == {
        "config_fingerprint": config.fingerprint,
        "policy_type": "openvla_oft",
        "algorithm": "grpo",
        "flow_sde_contract": None,
        "pi0_fast_rl_token_scope": None,
        "training_unit": "action",
        "precalculate_logprobs": False,
        "rollout_logprob_source": "rollout_action",
        "schedule": "rlinf_actor_global_batch",
        "updates": 100,
        "groups_per_update": 16,
        "trajectories_per_update": 128,
        "total_trajectories": 12800,
        "optimizer_steps_per_update": 1,
        "sft_replay": None,
        "sft_anchor": None,
        "optimizer_update_epochs": 1,
        "action_token_progress_enabled": True,
        "action_token_progress_every_microbatches": 128,
        "max_log_file_mb": 64,
        "fixed_horizon_rows": 8192,
        "optimizer_rows": 8192,
        "evaluation_enabled": True,
        "evaluation_before_training": False,
        "evaluation_after_first_update": False,
        "evaluation_data_role": "development",
        "paired_evaluation_enabled": False,
        "paired_baseline_source": None,
        "baseline_outcomes_path": None,
        "evaluation_every_updates": 20,
        "evaluation_episodes": 100,
        "wandb_enabled": True,
        "wandb_train_video_required": False,
        "wandb_evaluation_video_required": False,
        "videos_per_update": 2,
        "videos_per_evaluation": 8,
        "rollout_workers": 8,
        "rollout_max_environment_steps": 512,
        "rollout_max_policy_steps": 64,
        "rollout_shared_prefix_action_chunks": 0,
        "rollout_execution_mode": "in_process",
        "rollout_actor_factory": None,
        "rollout_actor_kwargs": {},
        "rollout_actors_per_device": 1,
        "rollout_group_batching": False,
        "rollout_actor_lifecycle": "per_update",
        "rollout_policy_sync": "shared",
        "rollout_inference_mode": "embedded",
        "rollout_inference_factory": None,
        "rollout_inference_replicas_per_device": 1,
        "rollout_inference_max_batch_size": 1,
        "rollout_inference_max_wait_ms": 0.0,
        "rollout_actor_python_executable": None,
        "rollout_inference_python_executable": None,
        "rollout_devices": ["cuda:0"],
        "rollout_device_count": 1,
        "rollout_actor_count": 1,
        "rollout_active_actor_count": 1,
        "rollout_model_replicas": 1,
        "rollout_environment_slots": 8,
        "rollout_active_environment_slots": 8,
        "training_devices": ["cuda:0"],
        "worker_python_executable": None,
        "training_device_count": 1,
        "training_model_replicas": 1,
        "shared_rollout_training_devices": ["cuda:0"],
        "rollout_training_time_shared": True,
        "serial_phase_device_reuse": True,
        "serial_phase_order": [
            "train_rollout",
            "training",
            "periodic_evaluation",
        ],
        "distributed_training": False,
        "training_worker_lifecycle": "per_update",
        "worker_handoff_dir": str(tmp_path / "handoff"),
        "worker_timeout_seconds": 7200,
        "max_worker_handoff_mb": 8192,
        "keep_worker_handoffs": False,
    }


def test_runtime_worker_python_executable_is_normalized(tmp_path: Path) -> None:
    raw = _complete_config(tmp_path)
    raw["runtime"]["worker_python_executable"] = "  /opt/vla/bin/python  "

    config = EmbodiedExperimentConfig.model_validate(raw)

    assert config.runtime.worker_python_executable == "/opt/vla/bin/python"


def test_runtime_worker_python_executable_rejects_blank_value(tmp_path: Path) -> None:
    raw = _complete_config(tmp_path)
    raw["runtime"]["worker_python_executable"] = "   "

    with pytest.raises(ValueError, match="worker_python_executable cannot be empty"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_rollout_actor_and_inference_python_can_be_isolated(tmp_path: Path) -> None:
    raw = _complete_config(tmp_path)
    raw["rollout"]["workers"] = 1
    execution = raw["runtime"]["rollout_execution"]
    execution.update(
        {
            "mode": "local_process",
            "actor_factory": "tests.fake:create_actor",
            "actor_kwargs": {},
            "group_batching": True,
            "lifecycle": "per_update",
            "policy_sync": "checkpoint",
            "inference_mode": "batched_server",
            "inference_factory": "tests.fake:create_inference",
            "actor_python_executable": "  /opt/isaac/bin/python  ",
            "inference_python_executable": " /opt/gr00t/bin/python ",
        }
    )

    config = EmbodiedExperimentConfig.model_validate(raw)

    assert execution is not config.runtime.rollout_execution
    assert (
        config.runtime.rollout_execution.actor_python_executable
        == "/opt/isaac/bin/python"
    )
    assert (
        config.runtime.rollout_execution.inference_python_executable
        == "/opt/gr00t/bin/python"
    )


def test_embedded_inference_rejects_separate_inference_python(tmp_path: Path) -> None:
    raw = _complete_config(tmp_path)
    raw["runtime"]["rollout_execution"]["inference_python_executable"] = (
        "/opt/gr00t/bin/python"
    )

    with pytest.raises(ValueError, match="in-process.*Python executables"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_builtin_optimizer_rejects_silently_ignored_sgd(tmp_path: Path) -> None:
    raw = _complete_config(tmp_path)
    raw["training"]["optimizer"]["type"] = "sgd"

    with pytest.raises(ValueError, match="adamw"):
        EmbodiedExperimentConfig.model_validate(raw)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_config_rejects_non_finite_declared_floats(
    tmp_path: Path,
    value: float,
) -> None:
    raw = _complete_config(tmp_path)
    raw["reward"]["scale"] = value

    with pytest.raises(ValueError, match="finite number"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_training_microbatch_is_canonical_and_must_match_compatibility_field(
    tmp_path: Path,
) -> None:
    raw = _complete_config(tmp_path)
    raw["training"]["microbatch_size"] = 4

    with pytest.raises(ValueError, match="microbatch_size.*must match"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_action_token_objective_requires_old_logprobs(tmp_path: Path) -> None:
    raw = _complete_config(tmp_path)
    raw["rollout"]["action_payload"]["require_old_logprobs"] = False

    with pytest.raises(ValueError, match="require_old_logprobs=true"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_fixed_horizon_flag_must_match_executed_schedule(tmp_path: Path) -> None:
    raw = _complete_config(tmp_path)
    raw["algorithm"]["pad_fixed_horizon_examples"] = False

    with pytest.raises(ValueError, match="pad_fixed_horizon_examples must be true"):
        EmbodiedExperimentConfig.model_validate(raw)

    raw = _complete_config(tmp_path)
    raw["training"]["schedule"] = {"type": "full_update"}
    raw["training"]["optimizer_steps_per_update"] = 1
    with pytest.raises(ValueError, match="pad_fixed_horizon_examples must be false"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_required_wandb_train_video_needs_logging_and_capture_limit(
    tmp_path: Path,
) -> None:
    raw = _complete_config(tmp_path)
    raw["observability"]["require_train_video"] = True
    raw["observability"]["wandb"]["enabled"] = False

    with pytest.raises(ValueError, match="requires W&B logging"):
        EmbodiedExperimentConfig.model_validate(raw)

    raw["observability"]["wandb"]["enabled"] = True
    raw["observability"]["videos_per_update"] = 0
    with pytest.raises(ValueError, match="videos_per_update >= 1"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_lookahead_preview_rejects_excessive_panel_count() -> None:
    raw = yaml.safe_load(
        (
            Path(__file__).parents[1]
            / "examples/embodied/lerobot_action_token_grpo.template.yaml"
        ).read_text(encoding="utf-8")
    )
    raw["observability"]["lookahead_preview"]["max_panels"] = 7

    with pytest.raises(ValueError, match="max_panels"):
        EmbodiedExperimentConfig.model_validate(raw)


@pytest.mark.parametrize("connection", ["shared_primary", "shared_worker"])
def test_evaluation_rejects_wandb_shared_mode(
    tmp_path: Path,
    connection: str,
) -> None:
    raw = _complete_config(tmp_path)
    raw["observability"]["wandb"].update(
        {
            "connection": connection,
            "run_id": "existing-run" if connection == "shared_worker" else None,
        }
    )

    with pytest.raises(
        pydantic.ValidationError,
        match="shared mode cannot satisfy ART-Embodied's native-step evaluation contract",
    ):
        EmbodiedExperimentConfig.model_validate(raw)


def test_weave_server_cache_requires_explicit_bounded_directory(
    tmp_path: Path,
) -> None:
    raw = _complete_config(tmp_path)
    raw["observability"]["weave"].update(
        {
            "use_server_cache": True,
            "server_cache_dir": None,
            "server_cache_size_mb": 0,
        }
    )

    with pytest.raises(ValueError, match="server_cache_dir is required"):
        EmbodiedExperimentConfig.model_validate(raw)

    raw["observability"]["weave"]["server_cache_dir"] = str(tmp_path / "weave-cache")
    with pytest.raises(ValueError, match="server_cache_size_mb must be positive"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_positive_control_records_demonstrated_lora_profile() -> None:
    config = EmbodiedExperimentConfig.from_yaml(POSITIVE_CONTROL_CONFIG)

    assert config.policy.lora.rank == 32
    assert config.policy.lora.alpha == 32
    assert config.training.optimizer.learning_rate == 5e-5
    assert config.environment.reset == {"reset_gripper_open": False}
    assert config.trajectories_per_update == 1024
    assert config.training.optimizer_steps_per_update == 4
    assert config.training.schedule.global_batch_size == 16384
    assert config.execution_summary()["fixed_horizon_rows"] == 65536
    assert config.execution_summary()["optimizer_rows"] == 65536


def test_pi05_long_positive_control_matches_rlinf_v01_geometry() -> None:
    config = EmbodiedExperimentConfig.from_yaml(PI05_LONG_POSITIVE_CONTROL_CONFIG)

    assert config.policy.type == "pi05"
    assert config.environment.task == "libero_10"
    assert config.environment.reset == {"reset_gripper_open": True}
    assert config.policy.load_kwargs["execution_horizon"] == 10
    assert config.policy.load_kwargs["model_chunk_size"] == 10
    assert config.environment.kwargs["action_chunk_size"] == 10
    assert config.algorithm.flow_sde is not None
    assert config.algorithm.flow_sde.noise_level == 0.5
    assert config.algorithm.flow_sde.num_denoise_steps == 4
    assert config.execution_summary()["flow_sde_contract"] == {
        "noise_level": 0.5,
        "num_denoise_steps": 4,
        "stochastic_transitions_per_sample": 1,
        "selected_step_sampling": "uniform",
        "joint_logprob": False,
    }
    assert config.rollout.max_episode_steps == 480
    assert config.rollout.max_policy_steps == 48
    assert config.training.schedule.update_epochs == 4
    assert config.training.schedule.action_chunk_size == 10
    assert config.training.optimizer_steps_per_update == 48
    assert config.trajectories_per_update == 512
    assert config.runtime.distributed_training is True
    assert config.runtime.training_devices == [f"cuda:{index}" for index in range(8)]
    assert config.runtime.training_worker_lifecycle == "cpu_offload"
    assert config.execution_summary()["fixed_horizon_rows"] == 24576
    assert config.execution_summary()["optimizer_rows"] == 98304


@pytest.mark.parametrize(
    ("path", "family", "model_horizon", "denoise_steps", "update_epochs"),
    [
        (PI0_SPATIAL_POSITIVE_CONTROL_CONFIG, "pi0", 50, 4, 2),
        (PI05_SPATIAL_POSITIVE_CONTROL_CONFIG, "pi05", 10, 3, 1),
    ],
)
def test_pi_spatial_positive_controls_match_rlinf_v01_geometry(
    path: Path,
    family: str,
    model_horizon: int,
    denoise_steps: int,
    update_epochs: int,
) -> None:
    config = EmbodiedExperimentConfig.from_yaml(path)

    assert config.policy.type == family
    assert config.environment.task == "libero_spatial"
    assert config.policy.load_kwargs["execution_horizon"] == 5
    assert config.policy.load_kwargs["model_chunk_size"] == model_horizon
    assert config.environment.kwargs["action_chunk_size"] == 5
    assert config.algorithm.flow_sde is not None
    assert config.algorithm.flow_sde.noise_level == 0.5
    assert config.algorithm.flow_sde.num_denoise_steps == denoise_steps
    assert config.rollout.max_episode_steps == 240
    assert config.rollout.max_policy_steps == 48
    assert config.training.schedule.update_epochs == update_epochs
    assert config.training.schedule.action_chunk_size == 5
    assert config.training.optimizer_steps_per_update == 12 * update_epochs
    assert config.trajectories_per_update == 512
    assert config.runtime.rollout_execution.group_batching is True
    assert config.runtime.rollout_execution.lifecycle == "cpu_offload"
    assert config.runtime.distributed_training is True
    assert config.runtime.training_devices == [f"cuda:{index}" for index in range(8)]
    assert config.runtime.training_worker_lifecycle == "cpu_offload"
    assert config.execution_summary()["fixed_horizon_rows"] == 24576
    assert config.execution_summary()["optimizer_rows"] == 24576 * update_epochs


def test_strict_pi_flow_sde_requires_group_native_rollout() -> None:
    raw = yaml.safe_load(
        PI05_SPATIAL_POSITIVE_CONTROL_CONFIG.read_text(encoding="utf-8")
    )
    raw["runtime"]["rollout_execution"]["group_batching"] = False

    with pytest.raises(ValueError, match="one sampled denoise index is shared"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_pi0_long_positive_control_matches_rlinf_v01_geometry() -> None:
    config = EmbodiedExperimentConfig.from_yaml(PI0_LONG_POSITIVE_CONTROL_CONFIG)

    assert config.policy.type == "pi0"
    assert config.policy.path == "RLinf/RLinf-Pi0-LIBERO-Long-SFT"
    assert config.policy.revision == "cc392caf163f715c55f21d9e60a1ea0c88e2e443"
    assert config.policy.load_kwargs["extra_delta_transform"] is True
    assert config.environment.task == "libero_10"
    assert config.environment.reset == {"reset_gripper_open": True}
    assert config.policy.load_kwargs["execution_horizon"] == 10
    assert config.policy.load_kwargs["model_chunk_size"] == 50
    assert config.environment.kwargs["action_chunk_size"] == 10
    assert config.algorithm.flow_sde is not None
    assert config.algorithm.flow_sde.noise_level == 0.5
    assert config.algorithm.flow_sde.num_denoise_steps == 4
    assert config.rollout.max_episode_steps == 480
    assert config.rollout.max_policy_steps == 48
    assert config.training.schedule.update_epochs == 2
    assert config.training.schedule.action_chunk_size == 10
    assert config.training.optimizer_steps_per_update == 24
    assert config.trajectories_per_update == 512
    assert config.execution_summary()["fixed_horizon_rows"] == 24576
    assert config.execution_summary()["optimizer_rows"] == 49152


@pytest.mark.parametrize(
    ("control_path", "large_path"),
    [
        (PI0_SPATIAL_POSITIVE_CONTROL_CONFIG, PI0_SPATIAL_1024_CONFIG),
        (PI05_LONG_POSITIVE_CONTROL_CONFIG, PI05_LONG_1024_CONFIG),
    ],
)
def test_pi_flow_1024_profiles_reduce_sampling_variance_without_larger_update(
    control_path: Path,
    large_path: Path,
) -> None:
    control = EmbodiedExperimentConfig.from_yaml(control_path)
    large = EmbodiedExperimentConfig.from_yaml(large_path)

    assert control.trajectories_per_update == 512
    assert large.trajectories_per_update == 1024
    assert large.algorithm.group_size == control.algorithm.group_size == 8
    assert large.training.optimizer_steps_per_update == (
        control.training.optimizer_steps_per_update
    )
    assert (
        large.execution_summary()["optimizer_rows"]
        == (control.execution_summary()["optimizer_rows"])
    )
    assert large.training.schedule.update_epochs * 2 == (
        control.training.schedule.update_epochs
    )


def test_high_lr_profile_changes_only_experiment_identity_lr_and_output() -> None:
    control = yaml.safe_load(POSITIVE_CONTROL_CONFIG.read_text(encoding="utf-8"))
    high_lr = yaml.safe_load(HIGH_LR_OBJECT_CONFIG.read_text(encoding="utf-8"))

    assert high_lr["training"]["optimizer"]["learning_rate"] == 2e-4
    assert control["training"]["optimizer"]["learning_rate"] == 5e-5
    for payload in (control, high_lr):
        payload["experiment"]["run"] = "normalized"
        payload["experiment"]["tags"] = ["normalized"]
        payload["training"]["optimizer"]["learning_rate"] = 0.0
        payload["storage"]["output_dir"] = "normalized"
    assert high_lr == control


def test_positive_control_freezes_demonstrated_rlinf_parity_contract() -> None:
    config = EmbodiedExperimentConfig.from_yaml(POSITIVE_CONTROL_CONFIG)

    assert {
        "model": {
            "path": config.policy.path,
            "revision": config.policy.revision,
            "loader": config.policy.load_kwargs["model_loader"],
            "unnorm_key": config.policy.unnorm_key,
            "lora": config.policy.lora.model_dump(mode="python"),
        },
        "sampling": {
            "do_sample": config.policy.rollout_generation.do_sample,
            "temperature": config.policy.rollout_generation.temperature,
            "top_p": config.policy.rollout_generation.top_p,
        },
        "environment": {
            "task": config.environment.task,
            "reset_gripper_open": config.environment.reset["reset_gripper_open"],
            "task_ids": config.environment.kwargs["task_ids"],
            "action_chunk_size": config.environment.kwargs["action_chunk_size"],
            "wait_steps_after_reset": config.environment.kwargs[
                "wait_steps_after_reset"
            ],
            "rotate_images_180": config.environment.kwargs["rotate_images_180"],
        },
        "grpo": {
            "group_size": config.algorithm.group_size,
            "clip_low": config.algorithm.clip_epsilon_low,
            "clip_high": config.algorithm.clip_epsilon_high,
            "advantage_std_unbiased": config.algorithm.advantage.std_unbiased,
            "loss_aggregation": config.algorithm.loss_aggregation,
            "groups_per_update": config.rollout.groups_per_update,
            "epochs_per_update": config.rollout.epochs_per_update,
            "reward_scale": config.reward.scale,
        },
        "optimizer": config.training.optimizer.model_dump(mode="python"),
        "schedule": {
            "optimizer_steps": config.training.optimizer_steps_per_update,
            "global_batch_size": config.training.schedule.global_batch_size,
            "actor_world_size": config.training.schedule.actor_world_size,
            "action_chunk_size": config.training.schedule.action_chunk_size,
        },
    } == {
        "model": {
            "path": "Haozhan72/Openvla-oft-SFT-libero-object-traj1",
            "revision": "62e5a8daba3f619c993f23b248ae04b0d9677bb5",
            "loader": "native",
            "unnorm_key": "libero_object_no_noops",
            "lora": {
                "enabled": True,
                "rank": 32,
                "alpha": 32,
                "dropout": 0.0,
                "init": "gaussian",
                "target_modules": [
                    "gate_proj",
                    "k_proj",
                    "up_proj",
                    "qkv",
                    "fc1",
                    "lm_head",
                    "down_proj",
                    "v_proj",
                    "q",
                    "proj",
                    "kv",
                    "fc2",
                    "fc3",
                    "q_proj",
                    "o_proj",
                ],
                "rank_partition": None,
            },
        },
        "sampling": {"do_sample": True, "temperature": 1.6, "top_p": 1.0},
        "environment": {
            "task": "libero_object",
            "reset_gripper_open": False,
            "task_ids": list(range(10)),
            "action_chunk_size": 8,
            "wait_steps_after_reset": 15,
            "rotate_images_180": True,
        },
        "grpo": {
            "group_size": 8,
            "clip_low": 0.2,
            "clip_high": 0.28,
            "advantage_std_unbiased": True,
            "loss_aggregation": "rlinf_masked_mean_ratio",
            "groups_per_update": 8,
            "epochs_per_update": 16,
            "reward_scale": 5.0,
        },
        "optimizer": {
            "type": "adamw",
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-5,
            "max_grad_norm": 1.0,
        },
        "schedule": {
            "optimizer_steps": 4,
            "global_batch_size": 16384,
            "actor_world_size": 8,
            "action_chunk_size": 8,
        },
    }


def test_spatial_positive_control_changes_only_the_suite_contract() -> None:
    object_config = EmbodiedExperimentConfig.from_yaml(POSITIVE_CONTROL_CONFIG)
    spatial_config = EmbodiedExperimentConfig.from_yaml(SPATIAL_POSITIVE_CONTROL_CONFIG)

    assert spatial_config.policy.path == (
        "Haozhan72/Openvla-oft-SFT-libero-spatial-traj1"
    )
    assert spatial_config.policy.revision == (
        "39e5240e879c80b6cda6b3a83763dad717f5c05d"
    )
    assert spatial_config.policy.unnorm_key == "libero_spatial_no_noops"
    assert spatial_config.environment.task == "libero_spatial"
    assert spatial_config.algorithm == object_config.algorithm
    assert spatial_config.rollout == object_config.rollout
    assert spatial_config.training == object_config.training
    assert spatial_config.runtime == object_config.runtime


def test_spatial_h100_regression_reuses_object_learning_and_runtime_contract() -> None:
    object_config = EmbodiedExperimentConfig.from_yaml(H100_OBJECT_CONFIG)
    spatial_config = EmbodiedExperimentConfig.from_yaml(H100_SPATIAL_CONFIG)

    assert spatial_config.algorithm == object_config.algorithm
    assert spatial_config.reward == object_config.reward
    assert spatial_config.rollout == object_config.rollout
    assert spatial_config.training == object_config.training
    assert spatial_config.runtime == object_config.runtime
    assert spatial_config.policy.lora == object_config.policy.lora
    assert spatial_config.policy.rollout_generation == (
        object_config.policy.rollout_generation
    )
    assert spatial_config.policy.train_generation == (
        object_config.policy.train_generation
    )
    assert spatial_config.policy.evaluation_generation == (
        object_config.policy.evaluation_generation
    )
    assert spatial_config.environment.kwargs == object_config.environment.kwargs
    assert spatial_config.environment.task == "libero_spatial"
    assert object_config.environment.task == "libero_object"


def test_strict_rlinf_schedule_requires_action_training_unit(tmp_path: Path) -> None:
    raw = _complete_config(tmp_path)
    raw["algorithm"]["training_unit"] = "trajectory"
    raw["algorithm"]["action_advantage_mode"] = "example"
    raw["algorithm"]["score_source"] = "trajectory_reward"

    with pytest.raises(
        ValueError,
        match="strict RLinf actor-batch geometry requires.*training_unit='action'",
    ):
        EmbodiedExperimentConfig.model_validate(raw)


def test_strict_rlinf_schedule_rejects_policy_environment_horizon_mismatch(
    tmp_path: Path,
) -> None:
    raw = _complete_config(tmp_path)
    raw["rollout"]["max_policy_steps"] = 512

    with pytest.raises(
        ValueError,
        match="max_policy_steps.*max_episode_steps",
    ):
        EmbodiedExperimentConfig.model_validate(raw)


def test_yaml_round_trip_keeps_full_contract(tmp_path: Path) -> None:
    config = EmbodiedExperimentConfig.model_validate(_complete_config(tmp_path))
    path = config.to_yaml(tmp_path / "experiment.yaml")

    loaded = EmbodiedExperimentConfig.from_yaml(path)

    assert loaded == config
    assert loaded.fingerprint == config.fingerprint


def test_sealed_test_rejects_checkpoint_selection_leakage(tmp_path: Path) -> None:
    raw = _complete_config(tmp_path)
    raw["evaluation"].update(
        {"data_role": "sealed_test", "checkpoint_selection": "evaluation_success"}
    )

    with pytest.raises(ValueError, match="sealed evaluation cannot select"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_train_matched_evaluation_must_be_diagnostic(tmp_path: Path) -> None:
    raw = _complete_config(tmp_path)
    raw["evaluation"].update({"split": "train_matched", "data_role": "development"})

    with pytest.raises(ValueError, match="train_matched.*diagnostic"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_distributed_runtime_requires_multiple_explicit_devices(tmp_path: Path) -> None:
    raw = _complete_config(tmp_path)
    raw["runtime"]["distributed_training"] = True

    with pytest.raises(ValueError, match="at least two"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_single_device_runtime_rejects_ignored_extra_devices(tmp_path: Path) -> None:
    raw = _complete_config(tmp_path)
    raw["runtime"]["training_devices"] = ["cuda:0", "cuda:1"]

    with pytest.raises(ValueError, match="extra devices would be ignored"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_local_process_rollout_requires_explicit_actor_geometry(
    tmp_path: Path,
) -> None:
    raw = _complete_config(tmp_path)
    raw["runtime"]["rollout_devices"] = ["cuda:0", "cuda:1"]
    raw["runtime"]["rollout_execution"] = {
        "mode": "local_process",
        "actor_factory": "company.rollout:create_actor",
        "actor_kwargs": {},
        "actors_per_device": 2,
        "group_batching": False,
        "lifecycle": "per_update",
        "policy_sync": "checkpoint",
        "inference_mode": "embedded",
        "inference_factory": None,
        "inference_replicas_per_device": 1,
        "inference_max_batch_size": 1,
        "inference_max_wait_ms": 0.0,
        "startup_timeout_seconds": 600,
        "request_timeout_seconds": 300,
    }
    raw["rollout"]["workers"] = 3

    with pytest.raises(ValueError, match="rollout.workers to equal"):
        EmbodiedExperimentConfig.model_validate(raw)

    raw["rollout"]["workers"] = 4
    config = EmbodiedExperimentConfig.model_validate(raw)
    assert config.runtime.rollout_execution.mode == "local_process"


def test_group_batching_requires_atomic_failure_policy(tmp_path: Path) -> None:
    raw = _complete_config(tmp_path)
    raw["runtime"]["rollout_execution"].update(
        {
            "mode": "local_process",
            "actor_factory": "company.rollout:create_actor",
            "actors_per_device": 1,
            "group_batching": True,
            "policy_sync": "checkpoint",
        }
    )
    raw["rollout"]["failure_policy"] = "keep_partial"

    with pytest.raises(ValueError, match="group-native actor request is atomic"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_batched_rollout_requires_an_inference_factory(tmp_path: Path) -> None:
    raw = _complete_config(tmp_path)
    raw["rollout"]["workers"] = 1
    raw["runtime"]["rollout_devices"] = ["cpu"]
    raw["runtime"]["training_devices"] = ["cuda:0"]
    raw["runtime"]["rollout_execution"] = {
        "mode": "local_process",
        "actor_factory": "package.module:create_actor",
        "actor_kwargs": {},
        "actors_per_device": 1,
        "group_batching": False,
        "lifecycle": "persistent",
        "policy_sync": "checkpoint",
        "inference_mode": "batched_server",
        "inference_factory": None,
        "inference_replicas_per_device": 1,
        "inference_max_batch_size": 8,
        "inference_max_wait_ms": 2.0,
        "startup_timeout_seconds": 30,
        "request_timeout_seconds": 30,
    }

    with pytest.raises(ValueError, match="inference_factory"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_persistent_rollout_rejects_training_device_overlap(tmp_path: Path) -> None:
    raw = _complete_config(tmp_path)
    raw["runtime"]["rollout_execution"] = {
        "mode": "local_process",
        "actor_factory": "company.rollout:create_actor",
        "actor_kwargs": {},
        "actors_per_device": 8,
        "group_batching": False,
        "lifecycle": "persistent",
        "policy_sync": "checkpoint",
        "inference_mode": "embedded",
        "inference_factory": None,
        "inference_replicas_per_device": 1,
        "inference_max_batch_size": 1,
        "inference_max_wait_ms": 0.0,
        "startup_timeout_seconds": 600,
        "request_timeout_seconds": 300,
    }

    with pytest.raises(ValueError, match="devices disjoint"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_cpu_offload_rollout_allows_training_device_overlap(tmp_path: Path) -> None:
    raw = _complete_config(tmp_path)
    raw["runtime"]["rollout_execution"].update(
        {
            "mode": "local_process",
            "actor_factory": "company.rollout:create_actor",
            "actors_per_device": 8,
            "lifecycle": "cpu_offload",
            "policy_sync": "checkpoint",
            "inference_mode": "batched_server",
            "inference_factory": "company.rollout:create_inference",
            "inference_replicas_per_device": 1,
            "inference_max_batch_size": 8,
            "inference_max_wait_ms": 2.0,
        }
    )

    config = EmbodiedExperimentConfig.model_validate(raw)

    assert config.runtime.rollout_execution.lifecycle == "cpu_offload"
    assert config.execution_summary()["rollout_training_time_shared"] is True


def test_cpu_offload_rollout_allows_embedded_inference(tmp_path: Path) -> None:
    raw = _complete_config(tmp_path)
    raw["runtime"]["rollout_execution"].update(
        {
            "mode": "local_process",
            "actor_factory": "company.rollout:create_actor",
            "actors_per_device": 8,
            "lifecycle": "cpu_offload",
            "policy_sync": "checkpoint",
            "inference_mode": "embedded",
            "inference_factory": None,
        }
    )

    config = EmbodiedExperimentConfig.model_validate(raw)

    assert config.runtime.rollout_execution.lifecycle == "cpu_offload"
    assert config.runtime.rollout_execution.inference_mode == "embedded"


def test_batched_inference_replicas_cannot_exceed_actors(tmp_path: Path) -> None:
    raw = _complete_config(tmp_path)
    raw["runtime"]["rollout_execution"].update(
        {
            "mode": "local_process",
            "actor_factory": "company.rollout:create_actor",
            "actors_per_device": 2,
            "policy_sync": "checkpoint",
            "inference_mode": "batched_server",
            "inference_factory": "company.rollout:create_inference",
            "inference_replicas_per_device": 3,
        }
    )
    raw["rollout"]["workers"] = 2

    with pytest.raises(ValueError, match="cannot exceed actors_per_device"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_unknown_or_missing_conditions_fail_closed(tmp_path: Path) -> None:
    raw = _complete_config(tmp_path)
    raw["environment"]["hidden_reset_mode"] = True
    with pytest.raises(ValueError, match="hidden_reset_mode"):
        EmbodiedExperimentConfig.model_validate(raw)

    raw = _complete_config(tmp_path)
    del raw["environment"]["reset"]
    with pytest.raises(ValueError, match="reset"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_gspo_requires_sequence_importance_sampling(tmp_path: Path) -> None:
    raw = _complete_config(tmp_path)
    raw["algorithm"]["type"] = "gspo"
    raw["algorithm"]["precalculate_logprobs"] = True
    raw["algorithm"]["rollout_logprob_source"] = "recomputed_current_policy"
    raw["algorithm"]["pre_update_logprob_kl_tolerance"] = 1e-4
    raw["algorithm"]["pre_update_ratio_tolerance"] = 1e-4
    with pytest.raises(ValueError, match="importance_sampling_level='sequence'"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_grpo_accepts_seq_mean_token_sum_loss(tmp_path: Path) -> None:
    raw = _complete_config(tmp_path)
    raw["algorithm"]["loss_aggregation"] = "seq_mean_token_sum"

    config = EmbodiedExperimentConfig.model_validate(raw)

    assert config.algorithm.loss_aggregation == "seq_mean_token_sum"


def test_gspo_requires_training_scorer_old_logprobs(tmp_path: Path) -> None:
    raw = _complete_config(tmp_path)
    raw["algorithm"].update(
        {
            "type": "gspo",
            "importance_sampling_level": "sequence",
            "training_unit": "trajectory",
            "action_advantage_mode": "example",
            "score_source": "trajectory_reward",
            "loss_aggregation": "trajectory_mean",
            "filter_rewards": False,
        }
    )

    with pytest.raises(ValueError, match="precalculate_logprobs=true"):
        EmbodiedExperimentConfig.model_validate(raw)

    raw["algorithm"]["precalculate_logprobs"] = True
    with pytest.raises(ValueError, match="recomputed_current_policy"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_gspo_alignment_guard_must_be_stricter_than_sequence_clip(
    tmp_path: Path,
) -> None:
    raw = _complete_config(tmp_path)
    raw["algorithm"].update(
        {
            "type": "gspo",
            "importance_sampling_level": "sequence",
            "training_unit": "trajectory",
            "action_advantage_mode": "example",
            "score_source": "trajectory_reward",
            "loss_aggregation": "trajectory_mean",
            "filter_rewards": False,
            "precalculate_logprobs": True,
            "rollout_logprob_source": "recomputed_current_policy",
            "clip_epsilon_low": 3e-4,
            "clip_epsilon_high": 4e-4,
            "pre_update_logprob_kl_tolerance": 1e-4,
            "pre_update_ratio_tolerance": 1e-3,
        }
    )

    with pytest.raises(ValueError, match="stricter than sequence clipping"):
        EmbodiedExperimentConfig.model_validate(raw)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"training_unit": "action"}, "training_unit='trajectory'"),
        (
            {"action_advantage_mode": "rlinf_action_level_cumulative"},
            "action_advantage_mode='example'",
        ),
        ({"score_source": "chunk_rewards"}, "score_source='trajectory_reward'"),
        ({"loss_aggregation": "token_mean"}, "loss_aggregation='trajectory_mean'"),
        ({"filter_rewards": True}, "does not yet support reward filtering"),
    ],
)
def test_gspo_rejects_ambiguous_non_trajectory_contract(
    tmp_path: Path,
    updates: dict[str, object],
    message: str,
) -> None:
    raw = _complete_config(tmp_path)
    raw["algorithm"].update(
        {
            "type": "gspo",
            "importance_sampling_level": "sequence",
            "training_unit": "trajectory",
            "action_advantage_mode": "example",
            "score_source": "trajectory_reward",
            "loss_aggregation": "trajectory_mean",
            "filter_rewards": False,
            "precalculate_logprobs": True,
            "rollout_logprob_source": "recomputed_current_policy",
            "pre_update_logprob_kl_tolerance": 1e-4,
            "pre_update_ratio_tolerance": 1e-4,
        }
    )
    raw["algorithm"].update(updates)

    with pytest.raises(ValueError, match=message):
        EmbodiedExperimentConfig.model_validate(raw)


def test_gspo_rejects_rlinf_action_subupdate_schedule(tmp_path: Path) -> None:
    raw = _complete_config(tmp_path)
    raw["algorithm"].update(
        {
            "type": "gspo",
            "importance_sampling_level": "sequence",
            "training_unit": "trajectory",
            "action_advantage_mode": "example",
            "score_source": "trajectory_reward",
            "loss_aggregation": "trajectory_mean",
            "filter_rewards": False,
            "precalculate_logprobs": True,
            "rollout_logprob_source": "recomputed_current_policy",
            "pre_update_logprob_kl_tolerance": 1e-4,
            "pre_update_ratio_tolerance": 1e-4,
        }
    )
    raw["training"]["schedule"]["strict_geometry"] = False

    with pytest.raises(ValueError, match="RLinf actor-batch subupdates"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_strict_rlinf_schedule_rejects_bounded_action_replay(tmp_path: Path) -> None:
    raw = _complete_config(tmp_path)
    raw["rollout"]["action_payload"].update(
        trainable_action_selection="uniform_grid",
        max_trainable_actions_per_trajectory=8,
    )

    with pytest.raises(ValueError, match="requires every action row"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_gspo_trajectory_minibatch_requires_exact_distributed_geometry(
    tmp_path: Path,
) -> None:
    raw = _complete_config(tmp_path)
    raw["algorithm"].update(
        {
            "type": "gspo",
            "importance_sampling_level": "sequence",
            "training_unit": "trajectory",
            "action_advantage_mode": "example",
            "score_source": "trajectory_reward",
            "loss_aggregation": "trajectory_mean",
            "filter_rewards": False,
            "precalculate_logprobs": True,
            "rollout_logprob_source": "recomputed_current_policy",
            "pre_update_logprob_kl_tolerance": 1e-4,
            "pre_update_ratio_tolerance": 1e-4,
        }
    )
    raw["training"]["optimizer_steps_per_update"] = 4
    raw["training"]["schedule"] = {
        "type": "trajectory_minibatch",
        "minibatch_trajectories": 32,
        "shuffle_seed": 1234,
    }
    raw["algorithm"]["pad_fixed_horizon_examples"] = False
    raw["runtime"]["distributed_training"] = True
    raw["runtime"]["training_devices"] = ["cuda:0", "cuda:1"]

    config = EmbodiedExperimentConfig.model_validate(raw)
    assert config.execution_summary()["optimizer_rows"] == 128

    raw["training"]["schedule"]["minibatch_trajectories"] = 31
    with pytest.raises(ValueError, match="consume every rollout trajectory"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_strict_rlinf_geometry_rejects_unexecuted_batch_contract(
    tmp_path: Path,
) -> None:
    raw = _complete_config(tmp_path)
    raw["training"]["schedule"]["global_batch_size"] = 4096

    with pytest.raises(ValueError, match="fixed-horizon rows"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_rlinf_schedule_requires_explicit_geometry(tmp_path: Path) -> None:
    raw = _complete_config(tmp_path)
    del raw["training"]["schedule"]["global_batch_size"]

    with pytest.raises(ValueError, match="global_batch_size"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_full_update_requires_optimizer_steps_to_match_update_epochs(
    tmp_path: Path,
) -> None:
    raw = _complete_config(tmp_path)
    raw["training"]["schedule"] = {"type": "full_update", "update_epochs": 1}
    raw["training"]["optimizer_steps_per_update"] = 2
    raw["algorithm"]["pad_fixed_horizon_examples"] = False

    with pytest.raises(ValueError, match="optimizer_steps_per_update"):
        EmbodiedExperimentConfig.model_validate(raw)

    raw["training"]["schedule"]["update_epochs"] = 2
    with pytest.raises(ValueError, match="distributed_training"):
        EmbodiedExperimentConfig.model_validate(raw)

    raw["runtime"]["distributed_training"] = True
    raw["runtime"]["training_devices"] = ["cuda:0", "cuda:1"]
    config = EmbodiedExperimentConfig.model_validate(raw)
    assert config.training.schedule.update_epochs == 2


def test_action_token_generation_contract_rejects_silent_drift(
    tmp_path: Path,
) -> None:
    raw = _complete_config(tmp_path)
    raw["policy"]["train_generation"]["temperature"] = 1.2
    with pytest.raises(ValueError, match="identical"):
        EmbodiedExperimentConfig.model_validate(raw)

    raw = _complete_config(tmp_path)
    raw["rollout"]["temperature"] = 1.2
    with pytest.raises(ValueError, match="rollout.temperature"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_continuous_policy_does_not_claim_token_sampling_controls() -> None:
    raw = yaml.safe_load(PI05_POSITIVE_CONTROL_CONFIG.read_text(encoding="utf-8"))
    raw["policy"]["train_generation"]["temperature"] = 9.0
    raw["policy"]["evaluation_generation"]["top_p"] = 0.5
    raw["rollout"]["temperature"] = 3.0

    config = EmbodiedExperimentConfig.model_validate(raw)

    assert config.rollout.action_payload.kind == "continuous"
    assert config.policy.train_generation.temperature == 9.0
    consumption = {row["path"]: row for row in config.consumption_report()["fields"]}
    assert (
        consumption["policy.rollout_generation.temperature"]["status"]
        == "not_applicable"
    )
    assert consumption["rollout.temperature"]["status"] == "not_applicable"


@pytest.mark.parametrize(
    "path",
    [PI0_POSITIVE_CONTROL_CONFIG, PI05_POSITIVE_CONTROL_CONFIG],
)
def test_pi_rlinf_positive_controls_reset_with_open_gripper(path: Path) -> None:
    config = EmbodiedExperimentConfig.from_yaml(path)

    assert config.environment.reset == {"reset_gripper_open": True}


def test_pi0_object_control_pins_canonical_public_checkpoint() -> None:
    config = EmbodiedExperimentConfig.from_yaml(PI0_POSITIVE_CONTROL_CONFIG)

    assert config.policy.path == "RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT"
    assert config.policy.revision == "4ec11623fba96e1432e535e2c76a634a5e0bdb96"
    assert config.policy.load_kwargs["extra_delta_transform"] is True
    assert config.policy.load_kwargs["execution_horizon"] == 5
    assert config.policy.load_kwargs["model_chunk_size"] == 50


@pytest.mark.parametrize(
    ("policy_type", "model_chunk_size", "expected"),
    [("pi0", 10, 50), ("pi05", 50, 10)],
)
def test_rlinf_pi_model_horizon_fails_closed(
    policy_type: str,
    model_chunk_size: int,
    expected: int,
) -> None:
    source = (
        PI0_POSITIVE_CONTROL_CONFIG
        if policy_type == "pi0"
        else PI05_POSITIVE_CONTROL_CONFIG
    )
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["policy"]["load_kwargs"]["model_chunk_size"] = model_chunk_size

    with pytest.raises(ValueError, match=rf"requires model_chunk_size={expected}"):
        EmbodiedExperimentConfig.model_validate(raw)


@pytest.mark.parametrize(
    "path",
    [
        ("environment", "kwargs", "action_chunk_size"),
        ("training", "schedule", "action_chunk_size"),
    ],
)
def test_pi_execution_horizon_must_match_executed_chunk(
    path: tuple[str, str, str],
) -> None:
    raw = yaml.safe_load(PI0_POSITIVE_CONTROL_CONFIG.read_text(encoding="utf-8"))
    target = raw
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = raw["policy"]["load_kwargs"]["execution_horizon"] + 1

    with pytest.raises(ValueError, match="execution_horizon must equal"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_generic_policy_config_does_not_require_openvla_load_fields(
    tmp_path: Path,
) -> None:
    raw = _complete_config(tmp_path)
    raw["policy"]["type"] = "company_custom_policy"
    raw["policy"]["load_kwargs"] = {"vendor_option": "kept-for-user-factory"}
    raw["policy"]["rollout_generation"]["top_p"] = 0.9
    raw["policy"]["train_generation"]["top_p"] = 0.9
    raw["policy"]["evaluation_generation"]["top_p"] = 0.8

    config = EmbodiedExperimentConfig.model_validate(raw)

    assert config.policy.type == "company_custom_policy"
    assert config.policy.load_kwargs == {"vendor_option": "kept-for-user-factory"}
    assert config.policy.rollout_generation.top_p == 0.9


def test_generic_lerobot_template_has_no_rlinf_schedule_contract() -> None:
    path = (
        Path(__file__).parents[1]
        / "examples/embodied/lerobot_action_token_grpo.template.yaml"
    )

    config = EmbodiedExperimentConfig.from_yaml(path)

    assert config.policy.type == "my_lerobot_action_token_policy"
    assert config.training.schedule.type == "full_update"
    assert "rlinf" not in config.training.model_dump_json().lower()
    assert config.evaluation.runtime == "native_lerobot"
