"""Validate the trained GR00T policy against one real RoboCasa process."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import traceback
from typing import Any

import numpy as np
import torch

from art_embodied import EmbodiedExperimentConfig, make_policy
from art_embodied.backends.flow_sde import TRANSIENT_FLOW_SDE_ROLLOUT_KEY
from art_embodied.backends.flow_sde_grpo import flow_sde_reference_kl_loss
from art_embodied.integrations.gr00t_flow_sde import (
    GR00TN17FlowSDEPolicyAdapter,
    _robocasa_gr1_action_batch,
    _seed_policy,
)
from art_embodied.lora_rank_partition import adapter_from_parameter_name
from art_embodied.policies.flow_policy import FlowSDERollout
from art_embodied.utils import write_json_atomic

from .environment import POLICY_ACTION_DIM
from .settings import RoboCasaSettings
from .simulator import RoboCasaSimulatorProcess


def _official_sim_observation_batch(
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Batch raw Gym observations exactly as NVIDIA's MultiStepWrapper does."""

    result: dict[str, Any] = {}
    for key in (
        "video.ego_view_bg_crop_pad_res256_freq20",
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.waist",
    ):
        result[key] = np.stack([np.asarray(row[key]) for row in observations])[:, None]
    result["annotation.human.coarse_action"] = np.asarray(
        [str(row["annotation.human.coarse_action"]) for row in observations]
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--oracle-task-id")
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument(
        "--official-gr00t-source",
        type=Path,
        default=Path(
            ".runtime-sources/isaac-gr00t-376ba890cff8c9de64d71d982772a9c36185fdd7"
        ),
    )
    parser.add_argument(
        "--oracle-manifest",
        type=Path,
        default=Path("examples/embodied/robocasa/oracle_parity_manifest.json"),
    )
    oracle_mode = parser.add_mutually_exclusive_group()
    oracle_mode.add_argument("--oracle-only", action="store_true")
    oracle_mode.add_argument("--oracle-full-episode", action="store_true")
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    config = EmbodiedExperimentConfig.from_yaml(args.config)
    settings = RoboCasaSettings.from_config(config)
    oracle_manifest = json.loads(args.oracle_manifest.read_text(encoding="utf-8"))
    if oracle_manifest.get("schema_version") != 1:
        raise ValueError("RoboCasa oracle manifest requires schema_version 1")
    oracle_task_id = str(args.oracle_task_id or oracle_manifest["task_id"])
    expansion_task_ids = {
        str(task_id) for task_id in oracle_manifest["expansion_task_ids"]
    }
    if oracle_task_id not in expansion_task_ids:
        raise ValueError(
            f"Task {oracle_task_id!r} is outside the oracle expansion manifest"
        )
    enabled_tasks = {task.id: task for task in settings.tasks}
    if oracle_task_id not in enabled_tasks:
        raise ValueError(
            f"Oracle task {oracle_task_id!r} is not enabled by the conformance config"
        )
    if args.oracle_task_id is None and (
        enabled_tasks[oracle_task_id].environment_id
        != oracle_manifest["environment_id"]
    ):
        raise ValueError("Oracle manifest environment_id differs from the task catalog")
    contract = oracle_manifest["contract"]
    if (
        settings.manifest.source_revision
        != oracle_manifest["official_robocasa"]["revision"]
        or settings.execution_horizon != int(contract["execution_horizon"])
        or POLICY_ACTION_DIM != int(contract["action_dimension"])
        or settings.max_environment_steps != int(contract["max_environment_steps"])
    ):
        raise RuntimeError("RoboCasa runtime settings differ from the oracle manifest")
    checkpoint_contract = oracle_manifest["sft_checkpoint"]
    configured_checkpoint = Path(str(config.policy.path)).resolve()
    expected_checkpoint = Path(str(checkpoint_contract["path"])).resolve()
    if configured_checkpoint != expected_checkpoint:
        raise RuntimeError(
            "Policy base checkpoint differs from the oracle manifest: "
            f"expected={expected_checkpoint}, actual={configured_checkpoint}"
        )
    completion_marker = configured_checkpoint / "art_embodied_sft_complete.json"
    completion_marker_sha256 = hashlib.sha256(
        completion_marker.read_bytes()
    ).hexdigest()
    if completion_marker_sha256 != checkpoint_contract["completion_marker_sha256"]:
        raise RuntimeError("SFT completion marker differs from the oracle manifest")
    policy = make_policy(config)
    if args.policy_checkpoint is not None:
        policy.load_checkpoint(args.policy_checkpoint)
    policy.train()
    simulator = RoboCasaSimulatorProcess(
        settings=settings,
        startup_timeout_seconds=config.runtime.rollout_execution.startup_timeout_seconds,
    )
    task = enabled_tasks[oracle_task_id]
    try:
        # Match the rollout and gradient-worker microbatch contract. N1.7 runs
        # its action head in BF16, where changing the GEMM batch shape can move
        # an element log-prob by one representable value even with identical
        # weights and inputs. A singleton rescore is useful as a diagnostic, but
        # it is not the probability path used by the optimizer.
        batch_size = (
            1
            if args.oracle_only or args.oracle_full_episode
            else int(config.algorithm.logprob_microbatch_size)
        )
        reset = await simulator.request(
            {
                "op": "reset",
                "task_id": task.id,
                "num_envs": batch_size,
                "seed": args.seed,
            }
        )
        state_hashes = list(reset["state_hashes"])
        if len(set(state_hashes)) != 1:
            raise RuntimeError("Same-seed policy replicas received different states")
        observations = [reset["observations"][index] for index in range(batch_size)]

        # Evaluation must be byte-for-byte equivalent to NVIDIA's documented
        # Gr00tSimPolicyWrapper path. This also proves that a freshly attached
        # LoRA adapter is an identity function at update zero, not merely that
        # its B matrices happen to contain zeros.
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

        policy.eval()
        official_wrapper = Gr00tSimPolicyWrapper(policy.native_policy)
        official_input = _official_sim_observation_batch(observations)
        _seed_policy(args.seed)
        official_components, _ = official_wrapper.get_action(official_input)
        official_chunks, official_mask = _robocasa_gr1_action_batch(
            official_components,
            action_components_layout=policy.action_components,
        )

        eval_adapter = GR00TN17FlowSDEPolicyAdapter(
            policy=policy,
            sampling_mode="eval",
            runtime_profile="robocasa_gr1_tabletop",
        )
        eval_adapter.reset(seed=args.seed)
        eval_predictions = eval_adapter.predict_batch(
            observations,
            tasks=[task.id] * batch_size,
            step=0,
        )
        art_eval_chunks = np.stack(
            [
                np.asarray(prediction.predicted_action_chunk)
                for prediction in eval_predictions
            ]
        )

        disable_adapter = getattr(policy.model, "disable_adapter", None)
        if not callable(disable_adapter):
            raise RuntimeError("Fresh-LoRA parity requires PEFT disable_adapter()")
        with disable_adapter():
            _seed_policy(args.seed)
            sft_components, _ = official_wrapper.get_action(official_input)
        sft_chunks, sft_mask = _robocasa_gr1_action_batch(
            sft_components,
            action_components_layout=policy.action_components,
        )

        official_art_delta = np.abs(official_chunks - art_eval_chunks)
        fresh_lora_delta = np.abs(official_chunks - sft_chunks)
        if not np.array_equal(official_chunks, art_eval_chunks):
            raise RuntimeError(
                "ART eval actions differ from NVIDIA Gr00tSimPolicyWrapper: "
                f"max_abs_delta={official_art_delta.max():.8g}"
            )
        if not np.array_equal(official_chunks, sft_chunks):
            raise RuntimeError(
                "Fresh LoRA changes native update-zero actions: "
                f"max_abs_delta={fresh_lora_delta.max():.8g}"
            )
        if official_mask != sft_mask or not all(official_mask):
            raise RuntimeError(
                "Official RoboCasa action-dimension mask is not all-active"
            )

        official_contract = oracle_manifest["official_gr00t"]
        official_multistep_contract = official_contract["multistep_wrapper"]
        official_multistep_source = (
            args.official_gr00t_source / official_multistep_contract["path"]
        ).resolve()
        if not official_multistep_source.is_file():
            raise FileNotFoundError(
                f"Pinned NVIDIA MultiStepWrapper is missing: {official_multistep_source}"
            )
        actual_multistep_sha256 = hashlib.sha256(
            official_multistep_source.read_bytes()
        ).hexdigest()
        official_multistep_sha256 = str(official_multistep_contract["sha256"])
        if actual_multistep_sha256 != official_multistep_sha256:
            raise RuntimeError(
                "Pinned NVIDIA MultiStepWrapper differs from the oracle manifest: "
                f"expected={official_multistep_sha256}, "
                f"actual={actual_multistep_sha256}"
            )
        oracle_request = {
            "task_id": task.id,
            "environment_id": task.environment_id,
            "seed": args.seed,
            "official_multistep_wrapper": str(official_multistep_source),
            "official_multistep_wrapper_sha256": official_multistep_sha256,
            "max_environment_steps": settings.max_environment_steps,
        }
        if args.oracle_full_episode:
            episode_reset = await simulator.request(
                {"op": "oracle_episode_reset", **oracle_request}
            )
            if episode_reset["status"] != "passed":
                raise RuntimeError(
                    "ART RoboCasa reset differs from NVIDIA's pinned evaluator: "
                    f"{episode_reset['mismatches']}"
                )
            if episode_reset["reset"]["art_state_sha256"] != state_hashes[0]:
                raise RuntimeError(
                    "Policy input reset and oracle episode reset produced different states"
                )
            current_observation = episode_reset["observation"]
            current_actions = official_chunks[0]
            episode_chunks = []
            for policy_step in range(settings.max_policy_steps):
                episode_step = await simulator.request(
                    {"op": "oracle_episode_step", "actions": current_actions}
                )
                if episode_step["status"] != "passed":
                    raise RuntimeError(
                        "ART RoboCasa episode differs from NVIDIA's pinned evaluator: "
                        f"chunk={policy_step}, mismatches={episode_step['mismatches']}"
                    )
                current_observation = episode_step.pop("observation")
                episode_chunks.append(episode_step)
                if episode_step["done"]:
                    break
                next_predictions = eval_adapter.predict_batch(
                    [current_observation], tasks=[task.id], step=policy_step + 1
                )
                current_actions = np.asarray(
                    next_predictions[0].predicted_action_chunk, dtype=np.float32
                )
            else:
                raise RuntimeError(
                    "Oracle episode exceeded the configured policy horizon"
                )
            oracle_chunk_parity = {
                "status": "passed",
                "reset": {
                    key: value
                    for key, value in episode_reset.items()
                    if key != "observation"
                },
                "chunks": episode_chunks,
                "policy_steps": len(episode_chunks),
                "environment_steps": episode_chunks[-1]["environment_steps"],
                "success": episode_chunks[-1]["macro"]["official_success"],
                "terminal_reason": (
                    "success"
                    if episode_chunks[-1]["macro"]["official_success"]
                    else "max_environment_steps"
                ),
            }
        else:
            oracle_chunk_parity = await simulator.request(
                {
                    "op": "oracle_chunk_parity",
                    "actions": official_chunks[0],
                    **oracle_request,
                }
            )
        if oracle_chunk_parity["status"] != "passed":
            raise RuntimeError(
                "ART RoboCasa control differs from NVIDIA's pinned evaluator: "
                f"{oracle_chunk_parity.get('mismatches', [])}"
            )
        if args.oracle_only or args.oracle_full_episode:
            return {
                "schema_version": 1,
                "kind": (
                    "gr00t_n1d7_robocasa_one_task_full_episode_oracle_parity"
                    if args.oracle_full_episode
                    else "gr00t_n1d7_robocasa_one_task_oracle_parity"
                ),
                "status": "passed",
                "config": str(args.config.resolve()),
                "config_fingerprint": config.fingerprint,
                "oracle_manifest": str(args.oracle_manifest.resolve()),
                "task_id": task.id,
                "seed": args.seed,
                "initial_state_sha256": state_hashes[0],
                "official_native_action_shape": list(official_chunks.shape),
                "official_art_eval_exact_match": True,
                "official_art_eval_max_abs_delta": float(official_art_delta.max()),
                "fresh_lora_native_exact_match": True,
                "fresh_lora_native_max_abs_delta": float(fresh_lora_delta.max()),
                "official_action_dimension_mask": list(official_mask),
                "official_oracle_chunk_parity": oracle_chunk_parity,
            }

        policy.train()
        adapter = GR00TN17FlowSDEPolicyAdapter(
            policy=policy,
            sampling_mode="train",
            runtime_profile="robocasa_gr1_tabletop",
        )
        adapter.reset(seed=args.seed)
        selected_indices = torch.arange(batch_size, device=policy.device) % int(
            policy.schedule.num_steps
        )
        predictions = adapter.predict_batch(
            observations,
            tasks=[task.id] * batch_size,
            step=0,
            selected_indices=selected_indices,
        )
        native_actions = np.stack(
            [np.asarray(prediction.native_action) for prediction in predictions]
        )
        if native_actions.shape != (batch_size, 8, POLICY_ACTION_DIM):
            raise RuntimeError(
                "GR00T/RoboCasa policy must produce "
                f"(batch, 8, {POLICY_ACTION_DIM}), got "
                f"{native_actions.shape}"
            )

        retained_rows: list[FlowSDERollout] = []
        for prediction in predictions:
            retained = prediction.action.metadata[TRANSIENT_FLOW_SDE_ROLLOUT_KEY]
            if not isinstance(retained, FlowSDERollout):
                raise TypeError("N1.7 prediction did not retain a Flow-SDE rollout")
            retained_rows.append(retained)

        retained_batch = FlowSDERollout.concatenate(retained_rows).to(policy.device)
        rescored_batch = policy.flow_sde_logprobs(retained_batch)
        active_adapters_before = tuple(policy.model.active_adapters)
        reference_batch = policy.flow_sde_reference_logprobs(retained_batch)
        active_adapters_after = tuple(policy.model.active_adapters)
        if active_adapters_after != active_adapters_before:
            raise RuntimeError(
                "SFT-reference scoring did not restore active adapters: "
                f"before={active_adapters_before}, after={active_adapters_after}"
            )
        if reference_batch.requires_grad or reference_batch.grad_fn is not None:
            raise RuntimeError("SFT-reference scores retained an autograd graph")
        adapter_reference_delta = (
            rescored_batch.detach().float() - reference_batch.float()
        ).abs()
        reference_kl_loss, reference_kl_metrics = flow_sde_reference_kl_loss(
            rescored_batch,
            reference_batch,
        )
        if not bool(torch.isfinite(reference_kl_loss)):
            raise RuntimeError("SFT-reference KL loss is non-finite")
        if args.policy_checkpoint is not None and not bool(
            torch.any(adapter_reference_delta > 0.0)
        ):
            raise RuntimeError(
                "Loaded policy checkpoint is indistinguishable from the SFT reference"
            )
        old_batch = retained_batch.transition.old_logprobs[
            :, : policy.execution_horizon, : policy.action_dim
        ].to(rescored_batch.device)
        batch_delta = (rescored_batch.detach().float() - old_batch.float()).abs()

        # Record, but do not gate on, the numerically different singleton shape.
        # This makes BF16 shape sensitivity visible without pretending that it is
        # the distributed optimizer's batch-preserving rescore contract.
        singleton_deltas = []
        with torch.no_grad():
            for retained in retained_rows:
                singleton = policy.flow_sde_logprobs(retained.to(policy.device))
                singleton_old = retained.transition.old_logprobs[
                    :, : policy.execution_horizon, : policy.action_dim
                ].to(singleton.device)
                singleton_deltas.append(
                    float(
                        (singleton.float() - singleton_old.float()).abs().max().item()
                    )
                )

        (-rescored_batch.mean() + 0.01 * reference_kl_loss).backward()
        gradients = [
            parameter.grad
            for parameter in policy.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients:
            raise RuntimeError("RoboCasa policy conformance produced no gradients")
        gradients_finite = all(
            bool(torch.isfinite(gradient).all()) for gradient in gradients
        )
        if not gradients_finite:
            raise RuntimeError(
                "RoboCasa policy conformance produced non-finite gradients"
            )

        adapter_round_trip: dict[str, Any] = {"enabled": False}
        if config.policy.lora.rank_partition is not None:
            expected_adapters = list(policy.model.peft_config)
            representative_parameters: dict[str, Any] = {}
            for name, parameter in policy.named_parameters():
                adapter_name = adapter_from_parameter_name(name)
                if (
                    parameter.requires_grad
                    and adapter_name in expected_adapters
                    and adapter_name not in representative_parameters
                ):
                    representative_parameters[adapter_name] = parameter
            if list(representative_parameters) != expected_adapters:
                raise RuntimeError(
                    "N1.7 conformance could not find every partitioned adapter: "
                    f"found={list(representative_parameters)}, "
                    f"expected={expected_adapters}"
                )
            baseline_values = {
                adapter_name: parameter.detach().view(-1)[0].clone()
                for adapter_name, parameter in representative_parameters.items()
            }
            with tempfile.TemporaryDirectory(prefix="n17-partition-round-trip-") as tmp:
                snapshot = Path(tmp) / "policy"
                policy.save_checkpoint(str(snapshot))
                missing_directories = [
                    adapter_name
                    for adapter_name in expected_adapters
                    if not (snapshot / adapter_name).is_dir()
                ]
                if missing_directories:
                    raise RuntimeError(
                        "N1.7 partitioned snapshot omitted adapter directories: "
                        f"{missing_directories}"
                    )
                with torch.no_grad():
                    for parameter in representative_parameters.values():
                        parameter.view(-1)[0].add_(1.0)
                policy.load_checkpoint(snapshot)
            restored = all(
                torch.equal(parameter.detach().view(-1)[0], baseline_values[name])
                for name, parameter in representative_parameters.items()
            )
            active_adapters = list(policy.model.active_adapters)
            if not restored or active_adapters != expected_adapters:
                raise RuntimeError(
                    "N1.7 partitioned adapter checkpoint round-trip failed: "
                    f"restored={restored}, active={active_adapters}, "
                    f"expected={expected_adapters}"
                )
            adapter_round_trip = {
                "enabled": True,
                "adapter_names": expected_adapters,
                "representative_parameters_restored": restored,
                "active_adapters": active_adapters,
            }

        step = await simulator.request(
            {"op": "step", "actions": native_actions[:, 0, :]}
        )
        if len(step["results"]) != batch_size:
            raise RuntimeError("RoboCasa did not execute every policy action row")
        maximum_delta = float(batch_delta.max().item())
        if maximum_delta > 1.0e-5:
            raise RuntimeError(
                "RoboCasa rollout/rescore alignment exceeded tolerance: "
                f"{maximum_delta:.8g} > 1e-5"
            )
        return {
            "schema_version": 1,
            "status": "passed",
            "config": str(args.config.resolve()),
            "config_fingerprint": config.fingerprint,
            "oracle_manifest": str(args.oracle_manifest.resolve()),
            "task_id": task.id,
            "seed": args.seed,
            "initial_state_sha256": state_hashes[0],
            "native_action_shape": list(native_actions.shape),
            "official_native_action_shape": list(official_chunks.shape),
            "official_art_eval_exact_match": True,
            "official_art_eval_max_abs_delta": float(official_art_delta.max()),
            "fresh_lora_native_exact_match": True,
            "fresh_lora_native_max_abs_delta": float(fresh_lora_delta.max()),
            "official_action_dimension_mask": list(official_mask),
            "official_oracle_chunk_parity": oracle_chunk_parity,
            "rescore_batch_size": batch_size,
            "rollout_rescore_max_abs_delta": maximum_delta,
            "rollout_rescore_mean_abs_delta": float(batch_delta.mean().item()),
            "policy_checkpoint": (
                str(args.policy_checkpoint.resolve())
                if args.policy_checkpoint is not None
                else None
            ),
            "sft_reference_adapter_state_restored": True,
            "sft_reference_requires_grad": reference_batch.requires_grad,
            "adapter_reference_max_abs_delta": float(
                adapter_reference_delta.max().item()
            ),
            "adapter_reference_mean_abs_delta": float(
                adapter_reference_delta.mean().item()
            ),
            "sft_reference_kl_loss": float(reference_kl_loss.detach().item()),
            "sft_reference_kl_mean_per_primitive": float(
                reference_kl_metrics.mean_per_primitive.item()
            ),
            "singleton_rescore_max_abs_delta": max(singleton_deltas),
            "gradient_tensor_count": len(gradients),
            "all_gradients_finite": gradients_finite,
            "adapter_checkpoint_round_trip": adapter_round_trip,
            "simulator_step_result_count": len(step["results"]),
        }
    finally:
        await simulator.close()


def main() -> None:
    args = _parse_args()
    try:
        report = asyncio.run(_run(args))
    except Exception as exc:
        report = {
            "schema_version": 1,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json_atomic(args.output, report, indent=2, sort_keys=True)
        raise
    write_json_atomic(args.output, report, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
