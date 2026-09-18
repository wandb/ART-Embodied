"""Worker entry point for local-process Flow-SDE gradient computation."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import pickle
import sys
import time
import traceback
from typing import Any

import torch

from art_embodied.backends.action_token_gradients import (
    _save_gradient_payload,
    _trainable_gradient_payload,
)
from art_embodied.backends.flow_sde import (
    FlowSDEExample,
    _flow_sde_action_mask_tensor,
    _flow_sde_microbatches,
)
from art_embodied.backends.flow_sde_grpo import (
    chunk_logprobs,
    flow_sde_grpo_loss,
    flow_sde_reference_kl_loss,
    primitive_normalized_chunk_abs_delta,
)
from art_embodied.compatibility import require_compatible_worker_runtime
from art_embodied.config import EmbodiedExperimentConfig, GR00TN17FlowLoadConfig
from art_embodied.lora_rank_partition import (
    rank_blocks_from_lora_config,
    require_partitioned_trainable_parameters,
    task_adapter_map,
)
from art_embodied.policies.factory import make_policy
from art_embodied.policies.flow_policy import FlowSDERollout
from art_embodied.utils import make_json_safe, write_json_atomic
from art_embodied.worker_config import discard_coordinator_resume


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve-spec", type=Path, required=True)
    return parser.parse_args()


def _worker_config(spec: dict[str, Any]) -> EmbodiedExperimentConfig:
    raw = discard_coordinator_resume(spec["config"])
    policy = dict(raw["policy"])
    policy["device"] = "cuda:0"
    raw["policy"] = policy
    runtime = dict(raw["runtime"])
    runtime["training_devices"] = ["cuda:0"]
    runtime["distributed_training"] = False
    raw["runtime"] = runtime
    # This local projection is one child of a coordinator-owned distributed
    # update. Validation context distinguishes that internal role without
    # weakening the user-facing requirement for distributed training.
    return EmbodiedExperimentConfig.model_validate(
        raw,
        context={"art_embodied_process_role": "distributed_gradient_worker"},
    )


def _build_policy(spec: dict[str, Any]) -> tuple[Any, EmbodiedExperimentConfig]:
    config = _worker_config(spec)
    policy = make_policy(config)
    policy.load_checkpoint(spec["policy_snapshot"])
    return policy, config


def _refresh_policy(policy: Any, snapshot: str) -> None:
    load = getattr(policy, "load_checkpoint", None)
    if not callable(load):
        raise TypeError("Flow-SDE worker policy must implement load_checkpoint")
    load(snapshot)


def _move_policy(policy: Any, device: str) -> None:
    move = getattr(policy, "to", None)
    if not callable(move):
        raise TypeError("Flow-SDE worker policy must implement to(device)")
    move(device)
    if device == "cpu":
        torch.cuda.empty_cache()


def _compute_gradient_job(
    command: dict[str, Any],
    *,
    policy: Any,
    config: EmbodiedExperimentConfig | None = None,
    sft_replay_provider: Any | None = None,
) -> dict[str, Any]:
    examples, advantages, payload_cache_hit = _load_gradient_payload(command)
    if len(examples) != len(advantages):
        raise ValueError("Flow-SDE worker examples and advantages do not align")
    microbatch_size = int(command["microbatch_size"])
    denominator = int(command["loss_denominator"])
    length_normalized = bool(command["length_normalized"])
    max_episode_steps = int(command["max_episode_steps"])
    reference_kl_coefficient = float(command.get("reference_kl_coefficient", 0.0))
    sft_replay_coefficient = float(command.get("sft_replay_coefficient", 0.0))
    if reference_kl_coefficient > 0.0 and not callable(
        getattr(policy, "flow_sde_reference_logprobs", None)
    ):
        raise TypeError(
            "positive reference_kl_coefficient requires policy."
            "flow_sde_reference_logprobs"
        )
    if sft_replay_coefficient > 0.0:
        if sft_replay_provider is None:
            raise TypeError(
                "positive sft_replay_coefficient requires a replay provider"
            )
        if not callable(getattr(policy, "sft_replay_loss", None)):
            raise TypeError(
                "positive sft_replay_coefficient requires policy.sft_replay_loss"
            )
        if config is None or config.policy.type != "gr00t_n1d7":
            raise ValueError("SFT replay worker requires a serialized replay config")
        replay_config = GR00TN17FlowLoadConfig.model_validate(
            config.policy.load_kwargs
        ).sft_replay
        if replay_config is None:
            raise ValueError("SFT replay worker requires a serialized replay config")
        configured = float(replay_config.coefficient)
        if configured != sft_replay_coefficient:
            raise ValueError(
                "SFT replay command/config coefficient mismatch: "
                f"command={sft_replay_coefficient}, config={configured}"
            )
    parameters = [
        parameter for parameter in policy.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("Flow-SDE gradient worker has no trainable parameters")
    device = parameters[0].device
    for parameter in parameters:
        parameter.grad = None
    policy.train()
    blocks = rank_blocks_from_lora_config(config.policy.lora) if config else ()
    adapter_by_task = task_adapter_map(blocks)
    parameters_by_adapter: dict[str, list[Any]] = {}
    if blocks:
        parameter_adapters = require_partitioned_trainable_parameters(policy, blocks)
        for name, parameter in policy.named_parameters():
            adapter = parameter_adapters.get(str(name))
            if adapter is not None:
                parameters_by_adapter.setdefault(adapter, []).append(parameter)
    active_adapters: set[str] = set()

    started = time.perf_counter()
    totals = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "reference_kl": 0.0,
        "reference_kl_loss": 0.0,
        "ratio": 0.0,
        "kl": 0.0,
        "kl_per_primitive": 0.0,
        "clip": 0.0,
    }
    valid_total = sum(example.loss_mask for example in examples)
    alignment_sum = 0.0
    alignment_count = 0
    alignment_max = 0.0
    alignment_ratio_sum = 0.0
    active_alignment_sum = 0.0
    active_alignment_count = 0
    active_alignment_max = 0.0
    active_alignment_ratio_sum = 0.0
    primitive_alignment_sum = 0.0
    primitive_alignment_count = 0
    primitive_alignment_max = 0.0
    active_primitive_alignment_sum = 0.0
    active_primitive_alignment_count = 0
    active_primitive_alignment_max = 0.0
    microbatch_count = 0
    audit_rows: list[dict[str, torch.Tensor]] | None = (
        [] if command.get("audit_path") else None
    )
    audit_microbatch_losses: list[float] = []
    for batch, batch_advantages in _flow_sde_microbatches(
        examples,
        advantages,
        microbatch_size=microbatch_size,
        separate_task_keys=bool(blocks),
    ):
        rollout = FlowSDERollout.concatenate([example.rollout for example in batch]).to(
            device
        )
        current = policy.flow_sde_logprobs(rollout).float()
        old = rollout.transition.old_logprobs[
            :, : current.shape[1], : current.shape[2]
        ].to(device=current.device, dtype=torch.float32)
        mask = torch.tensor(
            [example.loss_mask for example in batch],
            device=current.device,
            dtype=torch.bool,
        )
        action_mask = _flow_sde_action_mask_tensor(
            batch,
            horizon=current.shape[1],
            action_dim=current.shape[2],
            device=current.device,
        )
        row_weights = None
        if length_normalized:
            row_weights = torch.tensor(
                [
                    max_episode_steps / example.trajectory_primitive_steps
                    for example in batch
                ],
                device=current.device,
                dtype=torch.float32,
            )
        policy_loss, metrics = flow_sde_grpo_loss(
            current,
            old,
            torch.tensor(batch_advantages, device=current.device, dtype=torch.float32),
            loss_mask=mask,
            action_mask=action_mask,
            row_weights=row_weights,
            loss_denominator=denominator,
            clip_epsilon_low=float(command["clip_epsilon_low"]),
            clip_epsilon_high=float(command["clip_epsilon_high"]),
            clip_ratio_c=command.get("clip_ratio_c"),
        )
        reference_kl = torch.zeros((), device=current.device)
        reference_kl_metric = torch.zeros((), device=current.device)
        if reference_kl_coefficient > 0.0:
            with torch.no_grad():
                reference = policy.flow_sde_reference_logprobs(rollout).float()
            reference_kl, reference_metrics = flow_sde_reference_kl_loss(
                current,
                reference,
                loss_mask=mask,
                action_mask=action_mask,
                row_weights=row_weights,
                loss_denominator=denominator,
            )
            reference_kl_metric = reference_metrics.mean_per_primitive
        loss = policy_loss + reference_kl_coefficient * reference_kl
        current_chunks = chunk_logprobs(current.detach(), action_mask=action_mask)
        old_chunks = chunk_logprobs(old, action_mask=action_mask)
        chunk_log_ratio = current_chunks - old_chunks
        chunk_delta = chunk_log_ratio.abs()
        primitive_delta = primitive_normalized_chunk_abs_delta(
            current.detach(), old, action_mask=action_mask
        )
        alignment_sum += float(chunk_delta.sum().cpu())
        alignment_count += int(chunk_delta.numel())
        alignment_max = max(alignment_max, float(chunk_delta.max().cpu()))
        chunk_ratio = torch.exp(chunk_log_ratio)
        alignment_ratio_sum += float(chunk_ratio.sum().cpu())
        primitive_alignment_sum += float(primitive_delta.sum().cpu())
        primitive_alignment_count += int(primitive_delta.numel())
        primitive_alignment_max = max(
            primitive_alignment_max, float(primitive_delta.max().cpu())
        )
        active_delta = chunk_delta[mask]
        active_ratio = chunk_ratio[mask]
        active_primitive_delta = primitive_delta[mask]
        if active_delta.numel():
            active_alignment_sum += float(active_delta.sum().cpu())
            active_alignment_count += int(active_delta.numel())
            active_alignment_max = max(
                active_alignment_max, float(active_delta.max().cpu())
            )
            active_alignment_ratio_sum += float(active_ratio.sum().cpu())
            active_primitive_alignment_sum += float(active_primitive_delta.sum().cpu())
            active_primitive_alignment_count += int(active_primitive_delta.numel())
            active_primitive_alignment_max = max(
                active_primitive_alignment_max,
                float(active_primitive_delta.max().cpu()),
            )
        if blocks:
            task_keys = {example.task_key for example in batch}
            if len(task_keys) != 1:
                raise RuntimeError(
                    "Task-partitioned LoRA requires task-homogeneous microbatches"
                )
            task_key = next(iter(task_keys))
            try:
                active_adapter = adapter_by_task[task_key]
            except KeyError as exc:
                raise ValueError(
                    f"No LoRA rank block is configured for task {task_key!r}"
                ) from exc
            active_adapters.add(active_adapter)
            torch.autograd.backward(
                loss,
                inputs=parameters_by_adapter[active_adapter],
            )
        else:
            loss.backward()
        microbatch_count += 1
        batch_valid = int(mask.count_nonzero())
        weight = batch_valid / max(valid_total, 1)
        totals["loss"] += float(loss.detach().cpu())
        totals["policy_loss"] += float(policy_loss.detach().cpu())
        totals["reference_kl_loss"] += float(reference_kl.detach().cpu())
        totals["reference_kl"] += float(reference_kl_metric.detach().cpu()) * weight
        totals["ratio"] += float(metrics.ratio_mean.detach().cpu()) * weight
        totals["kl"] += float(metrics.approximate_kl.detach().cpu()) * weight
        totals["kl_per_primitive"] += (
            float(metrics.approximate_kl_per_primitive.detach().cpu()) * weight
        )
        totals["clip"] += float(metrics.clip_fraction.detach().cpu()) * weight
        if audit_rows is not None:
            audit_microbatch_losses.append(float(loss.detach().cpu()))
            audit_rows.append(
                {
                    "current_chunk_logprobs": current_chunks.cpu(),
                    "old_chunk_logprobs": old_chunks.cpu(),
                    "advantages": torch.tensor(batch_advantages, dtype=torch.float32),
                    "loss_mask": mask.detach().cpu(),
                    "action_mask": action_mask.detach().cpu(),
                    "trajectory_primitive_steps": torch.tensor(
                        [example.trajectory_primitive_steps for example in batch],
                        dtype=torch.float32,
                    ),
                    "row_weights": (
                        row_weights.detach().cpu()
                        if row_weights is not None
                        else torch.ones(len(batch), dtype=torch.float32)
                    ),
                }
            )

    replay_example = None
    sft_replay_loss = torch.zeros((), device=device, dtype=torch.float32)
    if sft_replay_coefficient > 0.0:
        worker_count = int(command["worker_count"])
        if worker_count < 1:
            raise ValueError("SFT replay worker_count must be positive")
        replay_example = sft_replay_provider.sample(
            update_index=int(command["update_index"]),
            subupdate_index=int(command["subupdate_index"]),
        )
        sft_replay_loss = policy.sft_replay_loss(
            [replay_example.step_data],
            seed=int(replay_example.seed),
        ).float()
        (sft_replay_coefficient * sft_replay_loss / worker_count).backward()

    gradient_payload = _trainable_gradient_payload(policy)
    if blocks:
        gradient_payload["active_adapters"] = sorted(active_adapters)
        gradient_payload["rank_partition_mode"] = "task_all_active"
    _save_gradient_payload(gradient_payload, Path(command["gradient_path"]))
    if audit_rows is not None:
        keys = (
            "current_chunk_logprobs",
            "old_chunk_logprobs",
            "advantages",
            "loss_mask",
            "action_mask",
            "trajectory_primitive_steps",
            "row_weights",
        )
        torch.save(
            {
                **{key: torch.cat([row[key] for row in audit_rows]) for key in keys},
                "microbatch_losses": torch.tensor(
                    audit_microbatch_losses, dtype=torch.float32
                ),
                "loss_denominator": denominator,
                "max_episode_steps": max_episode_steps,
            },
            Path(command["audit_path"]),
        )
    return {
        "ok": True,
        "worker_index": int(command["worker_index"]),
        "elapsed_seconds": time.perf_counter() - started,
        "payload_cache_hit": payload_cache_hit,
        "metrics": make_json_safe(
            {
                "loss": totals["loss"],
                "policy_loss": totals["policy_loss"],
                "reference_kl": totals["reference_kl"],
                "reference_kl_loss": totals["reference_kl_loss"],
                "reference_kl_coefficient": reference_kl_coefficient,
                "sft_replay_loss": float(sft_replay_loss.detach().cpu()),
                "sft_replay_weighted_loss": (
                    sft_replay_coefficient * float(sft_replay_loss.detach().cpu())
                ),
                "sft_replay_coefficient": sft_replay_coefficient,
                "sft_replay_examples": float(replay_example is not None),
                "ratio_mean": totals["ratio"],
                "approximate_kl": totals["kl"],
                "approximate_kl_per_primitive": totals["kl_per_primitive"],
                "clip_fraction": totals["clip"],
                "rows": float(len(examples)),
                "valid_rows": float(valid_total),
                "microbatches": float(microbatch_count),
                "previous_abs_delta_mean": (
                    alignment_sum / alignment_count if alignment_count else 0.0
                ),
                "previous_abs_delta_max": alignment_max,
                "previous_ratio_mean": (
                    alignment_ratio_sum / alignment_count if alignment_count else 1.0
                ),
                "active_previous_abs_delta_mean": (
                    active_alignment_sum / active_alignment_count
                    if active_alignment_count
                    else 0.0
                ),
                "active_previous_abs_delta_max": active_alignment_max,
                "active_previous_ratio_mean": (
                    active_alignment_ratio_sum / active_alignment_count
                    if active_alignment_count
                    else 1.0
                ),
                "previous_abs_delta_per_primitive_mean": (
                    primitive_alignment_sum / primitive_alignment_count
                    if primitive_alignment_count
                    else 0.0
                ),
                "previous_abs_delta_per_primitive_max": primitive_alignment_max,
                "active_previous_abs_delta_per_primitive_mean": (
                    active_primitive_alignment_sum / active_primitive_alignment_count
                    if active_primitive_alignment_count
                    else 0.0
                ),
                "active_previous_abs_delta_per_primitive_max": (
                    active_primitive_alignment_max
                ),
            }
        ),
        "sft_replay_sample": (
            {
                "task_id": replay_example.task_id,
                "episode_index": replay_example.episode_index,
                "step_index": replay_example.step_index,
                "seed": replay_example.seed,
                "source_info_sha256": replay_example.source_info_sha256,
            }
            if replay_example is not None
            else None
        ),
    }


def _load_gradient_payload(
    command: dict[str, Any],
) -> tuple[list[FlowSDEExample], list[float], bool]:
    examples_path = command.get("examples_path")
    if examples_path is None:
        raise ValueError("Flow-SDE gradient payload path is required")
    with Path(examples_path).open("rb") as handle:
        payload = pickle.load(handle)
    examples = list(payload["examples"])
    advantages = [float(value) for value in payload["advantages"]]
    return examples, advantages, bool(command.get("payload_cache_hit", False))


async def _serve(spec: dict[str, Any]) -> None:
    policy, config = _build_policy(spec)
    sft_replay_provider = None
    replay = (
        GR00TN17FlowLoadConfig.model_validate(config.policy.load_kwargs).sft_replay
        if config.policy.type == "gr00t_n1d7"
        else None
    )
    if replay is not None and replay.coefficient > 0.0:
        from art_embodied.policies.gr00t_n1d7_sft_replay import (
            GR00TN17SFTReplayProvider,
        )

        sft_replay_provider = GR00TN17SFTReplayProvider(
            policy=policy,
            dataset_root=replay.dataset_root,
            dataset_prefix=replay.dataset_prefix,
            task_ids=replay.task_ids,
            seed=replay.seed,
            worker_index=int(spec["worker_index"]),
        )
    current_snapshot = str(spec["policy_snapshot"])
    write_json_atomic(
        Path(spec["ready_path"]),
        {"ok": True, "worker_index": int(spec["worker_index"])},
        sort_keys=True,
    )
    while line := await asyncio.to_thread(sys.stdin.readline):
        command = json.loads(line)
        if command.get("op") == "shutdown":
            break
        result_path = Path(command["result_path"])
        try:
            if command.get("op") == "offload":
                started = time.perf_counter()
                _move_policy(policy, "cpu")
                result = {
                    "ok": True,
                    "worker_index": int(spec["worker_index"]),
                    "offloaded": True,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                write_json_atomic(result_path, result, indent=2, sort_keys=True)
                continue
            if command.get("op") == "restore":
                started = time.perf_counter()
                requested = str(command["policy_snapshot"])
                _move_policy(policy, "cuda:0")
                _refresh_policy(policy, requested)
                current_snapshot = requested
                result = {
                    "ok": True,
                    "worker_index": int(spec["worker_index"]),
                    "restored": True,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                write_json_atomic(result_path, result, indent=2, sort_keys=True)
                continue
            if command.get("op") != "gradient":
                raise ValueError(
                    f"Unknown Flow-SDE worker command: {command.get('op')!r}"
                )
            requested = str(command["policy_snapshot"])
            refreshed = requested != current_snapshot
            if refreshed:
                _refresh_policy(policy, requested)
                current_snapshot = requested
            result = _compute_gradient_job(
                command,
                policy=policy,
                config=config,
                sft_replay_provider=sft_replay_provider,
            )
            result["adapter_refreshed"] = refreshed
        except Exception as exc:
            result = {
                "ok": False,
                "worker_index": int(spec["worker_index"]),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        write_json_atomic(result_path, result, indent=2, sort_keys=True)


def main() -> None:
    """Run the persistent Flow-SDE worker described by ``--serve-spec``."""

    require_compatible_worker_runtime()
    args = _parse_args()
    spec = json.loads(args.serve_spec.read_text(encoding="utf-8"))
    asyncio.run(_serve(spec))


if __name__ == "__main__":
    main()
