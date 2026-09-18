"""Worker entry point for local-process action-token gradient computation."""

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

from art_embodied.backends.action_token import rescore_action_token_examples
from art_embodied.backends.action_token_gradients import (
    _save_gradient_payload,
    _trainable_gradient_payload,
    load_action_token_gradient_payload,
)
from art_embodied.backends.factory import make_action_token_backend
from art_embodied.compatibility import require_compatible_worker_runtime
from art_embodied.config import EmbodiedExperimentConfig, PI0FastLoadConfig
from art_embodied.conformance.rlinf import RlinfScheduledActionTokenBackend
from art_embodied.lora_rank_partition import (
    TaskAdapterGradientGate,
    adapter_from_parameter_name,
    rank_blocks_from_lora_config,
)
from art_embodied.policies.factory import make_policy
from art_embodied.policies.openvla import refresh_openvla_peft_adapter
from art_embodied.utils import make_json_safe, write_json_atomic
from art_embodied.worker_config import discard_coordinator_resume


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--spec", type=Path)
    group.add_argument("--serve-spec", type=Path)
    return parser.parse_args()


def _worker_config(spec: dict[str, Any]) -> EmbodiedExperimentConfig:
    raw = discard_coordinator_resume(spec["config"])
    training_raw = dict(raw["training"])
    if training_raw["schedule"]["type"] == "trajectory_minibatch":
        # The coordinator has already selected one trajectory minibatch. A
        # worker computes exactly one shard gradient and must not recursively
        # schedule the parent update.
        training_raw["schedule"] = {"type": "full_update"}
        training_raw["optimizer_steps_per_update"] = 1
    raw["training"] = training_raw
    policy_raw = dict(raw["policy"])
    if policy_raw["type"] == "openvla_oft":
        load_kwargs = dict(policy_raw["load_kwargs"])
        load_kwargs["peft_adapter_path"] = str(spec["policy_snapshot"])
        policy_raw["load_kwargs"] = load_kwargs
    policy_raw["device"] = "cuda:0"
    raw["policy"] = policy_raw
    runtime = dict(raw["runtime"])
    # The worker does not construct rollout actors, but rollout geometry is a
    # validated part of the experiment contract. Preserve it while projecting
    # only this worker's training device into its CUDA-visible namespace.
    runtime["training_devices"] = ["cuda:0"]
    runtime["distributed_training"] = False
    raw["runtime"] = runtime
    return EmbodiedExperimentConfig.model_validate(
        raw,
        context={"art_embodied_process_role": "distributed_gradient_worker"},
    )


def _gradient_job_config(
    spec: dict[str, Any],
    persistent_config: EmbodiedExperimentConfig | None,
) -> EmbodiedExperimentConfig:
    """Resolve config for one-shot and persistent worker protocols."""

    if persistent_config is not None:
        return persistent_config
    return _worker_config(spec)


def _build_runtime(
    spec: dict[str, Any],
) -> tuple[EmbodiedExperimentConfig, Any, Any]:
    config = _worker_config(spec)
    policy = make_policy(config)
    if config.policy.type != "openvla_oft":
        _refresh_policy_snapshot(policy, str(spec["policy_snapshot"]))
    constructed = make_action_token_backend(config, policy=policy)
    backend = (
        constructed.backend
        if isinstance(constructed, RlinfScheduledActionTokenBackend)
        else constructed
    )
    backend.checkpoint_dir = None
    return config, policy, backend


async def _run_gradient_job(
    spec: dict[str, Any],
    *,
    policy: Any | None = None,
    backend: Any | None = None,
    config: EmbodiedExperimentConfig | None = None,
    current_snapshot: str | None = None,
    sft_anchor_provider: Any | None = None,
) -> tuple[dict[str, Any], str]:
    owns_runtime = policy is None or backend is None
    if owns_runtime:
        config, policy, backend = _build_runtime(spec)
        current_snapshot = str(spec["policy_snapshot"])
    assert policy is not None
    assert backend is not None
    config = _gradient_job_config(spec, config)

    requested_snapshot = str(spec["policy_snapshot"])
    refresh_report: dict[str, Any] | None = None
    if current_snapshot != requested_snapshot:
        refresh_report = _refresh_policy_snapshot(policy, requested_snapshot)

    with Path(spec["examples_path"]).open("rb") as handle:
        examples = pickle.load(handle)
    blocks = rank_blocks_from_lora_config(config.policy.lora)
    gradient_gate = None
    active_adapter = None
    if blocks:
        task_keys = {str(example.task) for example in examples}
        declared_task_key = str(spec.get("task_key") or "")
        if task_keys != {declared_task_key}:
            raise ValueError(
                "Task-partitioned action-token jobs must be task homogeneous: "
                f"declared={declared_task_key!r}, observed={sorted(task_keys)}"
            )
        gradient_gate = TaskAdapterGradientGate(
            policy,
            blocks,
            forward_routing=config.policy.lora.rank_partition.forward_routing,
        )
        active_adapter = gradient_gate.activate(declared_task_key)
    started = time.perf_counter()
    original_kl_tolerance = backend.pre_update_logprob_kl_tolerance
    original_ratio_tolerance = backend.pre_update_ratio_tolerance
    if not bool(spec.get("enforce_pre_update_alignment", True)):
        backend.pre_update_logprob_kl_tolerance = None
        backend.pre_update_ratio_tolerance = None
    try:
        result = await backend.train(
            [],
            _action_token_grpo_precomputed_examples=examples,
            _action_token_grpo_precomputed_examples_prepared=True,
            _action_token_grpo_precomputed_reward_filter_report=spec.get(
                "reward_filter_report"
            ),
            _action_token_grpo_global_example_count=spec["global_example_count"],
            _action_token_grpo_global_token_count=spec["global_token_count"],
            _action_token_grpo_global_positive_token_count=spec[
                "global_positive_token_count"
            ],
            _action_token_grpo_global_negative_token_count=spec[
                "global_negative_token_count"
            ],
            _action_token_grpo_return_gradients=True,
            _action_token_grpo_gradient_output_path=spec["gradient_path"],
        )
        anchor_metrics = _add_pi0_fast_sft_anchor_gradients(
            policy=policy,
            config=config,
            provider=sft_anchor_provider,
            update_index=int(spec.get("update_index", 0)),
            subupdate_index=int(spec.get("subupdate_index", 0)),
            gradient_path=Path(spec["gradient_path"]),
        )
        result.metrics.update(anchor_metrics)
    finally:
        if gradient_gate is not None:
            gradient_gate.close()
        backend.pre_update_logprob_kl_tolerance = original_kl_tolerance
        backend.pre_update_ratio_tolerance = original_ratio_tolerance
        if owns_runtime:
            await backend.close()
    if active_adapter is not None:
        gradient_path = Path(spec["gradient_path"])
        payload = load_action_token_gradient_payload(gradient_path)
        payload["gradients"] = {
            name: gradient
            for name, gradient in payload["gradients"].items()
            if adapter_from_parameter_name(name) == active_adapter
        }
        payload["shapes"] = {
            name: shape
            for name, shape in payload["shapes"].items()
            if adapter_from_parameter_name(name) == active_adapter
        }
        payload["missing_gradients"] = [
            name
            for name in payload["missing_gradients"]
            if adapter_from_parameter_name(name) == active_adapter
        ]
        payload["rank_partition_mode"] = "task_all_active"
        payload["active_adapters"] = [active_adapter]
        payload["task_key"] = str(spec["task_key"])
        _save_gradient_payload(payload, gradient_path)
    return (
        {
            "ok": True,
            "worker_index": int(spec["worker_index"]),
            "elapsed_seconds": time.perf_counter() - started,
            "metrics": make_json_safe(result.metrics),
            "adapter_refresh": make_json_safe(refresh_report),
            "rank_partition_active_adapter": active_adapter,
        },
        requested_snapshot,
    )


def _add_pi0_fast_sft_anchor_gradients(
    *,
    policy: Any,
    config: EmbodiedExperimentConfig,
    provider: Any | None,
    update_index: int,
    subupdate_index: int,
    gradient_path: Path,
) -> dict[str, float]:
    if config.policy.type != "pi0_fast":
        return {}
    anchor = PI0FastLoadConfig.model_validate(config.policy.load_kwargs).sft_anchor
    if anchor is None:
        return {}
    if provider is None:
        # A task-balanced anchor may have fewer tasks than distributed workers.
        # Those workers retain the policy-gradient payload written by backend.train.
        return {
            "embodied_action_token_grpo/sft_anchor/loss": 0.0,
            "embodied_action_token_grpo/sft_anchor/weighted_loss": 0.0,
            "embodied_action_token_grpo/sft_anchor/examples": 0.0,
            "embodied_action_token_grpo/sft_anchor/coefficient": float(
                anchor.coefficient
            ),
        }

    import torch

    native_policy = getattr(policy, "policy", None)
    if native_policy is None:
        raise RuntimeError("pi0-FAST SFT anchor requires the native LeRobot policy")
    was_training = bool(getattr(native_policy, "training", False))
    losses: list[float] = []
    samples = []
    try:
        native_policy.train()
        samples = provider.losses(
            update_index=update_index,
            subupdate_index=subupdate_index,
        )
        task_count = len(anchor.task_indices)
        for sample in samples:
            loss = sample.loss.float()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Non-finite pi0-FAST SFT anchor loss for task {sample.task_index}"
                )
            (float(anchor.coefficient) * loss / float(task_count)).backward()
            losses.append(float(loss.detach().cpu()))
    finally:
        native_policy.train(was_training)

    payload = _trainable_gradient_payload(policy)
    payload["sft_anchor_tasks"] = [sample.task_index for sample in samples]
    payload["sft_anchor_coefficient"] = float(anchor.coefficient)
    _save_gradient_payload(payload, gradient_path)
    contribution = sum(losses) / float(len(anchor.task_indices))
    return {
        "embodied_action_token_grpo/sft_anchor/loss": contribution,
        "embodied_action_token_grpo/sft_anchor/weighted_loss": (
            float(anchor.coefficient) * contribution
        ),
        "embodied_action_token_grpo/sft_anchor/examples": float(len(samples)),
        "embodied_action_token_grpo/sft_anchor/coefficient": float(
            anchor.coefficient
        ),
    }


def _run_rescore_job(
    spec: dict[str, Any],
    *,
    policy: Any,
    backend: Any,
    current_snapshot: str,
) -> tuple[dict[str, Any], str]:
    requested_snapshot = str(spec["policy_snapshot"])
    refresh_report: dict[str, Any] | None = None
    if current_snapshot != requested_snapshot:
        refresh_report = _refresh_policy_snapshot(policy, requested_snapshot)
    with Path(spec["examples_path"]).open("rb") as handle:
        examples = pickle.load(handle)

    started = time.perf_counter()
    model = getattr(policy, "model", None)
    previous_training = bool(getattr(model, "training", False))
    if model is None:
        raise RuntimeError("Action-token rescore requires a loaded policy model")
    import torch

    try:
        model.eval() if backend.logprob_eval_mode else model.train()
        microbatch_size = int(
            backend.train_logprob_microbatch_size
            or backend.logprob_microbatch_size
            or 1
        )
        # Score through the same autograd-enabled execution path used by the
        # policy-gradient objective. Some autoregressive policies, including
        # pi0-FAST, produce different teacher-forced logits under no_grad.
        # The rows are detached before serialization, so no graph is retained.
        with torch.enable_grad():
            report = rescore_action_token_examples(
                policy,
                examples,
                device=str(getattr(policy, "device", "cuda:0")),
                training_unit=backend.training_unit,
                microbatch_size=microbatch_size,
                pad_to_batch_size=microbatch_size,
                source=str(spec["source"]),
            )
    finally:
        model.train() if previous_training else model.eval()

    with Path(spec["output_examples_path"]).open("wb") as handle:
        pickle.dump(examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return (
        {
            "ok": True,
            "worker_index": int(spec["worker_index"]),
            "elapsed_seconds": time.perf_counter() - started,
            "report": make_json_safe(report),
            "adapter_refresh": make_json_safe(refresh_report),
        },
        requested_snapshot,
    )


async def _run(spec: dict[str, Any]) -> dict[str, Any]:
    result, _snapshot = await _run_gradient_job(spec)
    return result


def _offload_runtime(policy: Any, backend: Any) -> dict[str, Any]:
    """Release worker GPU memory while preserving the loaded base model on CPU."""

    started = time.perf_counter()
    optimizer = getattr(backend, "optimizer", None)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    model = getattr(policy, "model", None)
    if model is not None and callable(getattr(model, "to", None)):
        model.to("cpu")
    import torch

    torch.cuda.empty_cache()
    return {
        "ok": True,
        "offloaded": True,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _refresh_policy_snapshot(policy: Any, path: str) -> dict[str, Any]:
    """Refresh a worker policy through its model-family checkpoint contract."""

    if getattr(policy, "family", None) == "openvla_oft":
        return refresh_openvla_peft_adapter(policy, path)
    load = getattr(policy, "load_checkpoint", None)
    if not callable(load):
        raise TypeError(
            "Distributed action-token workers require policy.load_checkpoint(path)"
        )
    load({"path": path})
    model = getattr(policy, "model", None)
    if model is not None and callable(getattr(model, "to", None)):
        model.to(str(getattr(policy, "device", "cuda:0")))
    if model is not None and callable(getattr(model, "eval", None)):
        model.eval()
    return {
        "adapter_path": path,
        "base_model_reloaded": False,
        "missing_keys": 0,
        "unexpected_keys": 0,
    }


async def _serve(spec: dict[str, Any]) -> None:
    config, policy, backend = _build_runtime(spec)
    sft_anchor_provider = None
    anchor = (
        PI0FastLoadConfig.model_validate(config.policy.load_kwargs).sft_anchor
        if config.policy.type == "pi0_fast"
        else None
    )
    assigned_tasks = [int(value) for value in spec.get("sft_anchor_task_indices", [])]
    if anchor is not None and assigned_tasks:
        from art_embodied.policies.pi0_fast_sft_anchor import (
            PI0FastSFTAnchorProvider,
        )

        sft_anchor_provider = PI0FastSFTAnchorProvider(
            policy=policy,
            dataset_repo_id=anchor.dataset_repo_id,
            dataset_revision=anchor.dataset_revision,
            task_indices=assigned_tasks,
            seed=anchor.seed,
        )
    current_snapshot = str(spec["policy_snapshot"])
    ready_path = Path(spec["ready_path"])
    write_json_atomic(
        ready_path,
        {
            "ok": True,
            "worker_index": int(spec["worker_index"]),
            "policy_loads": 1,
            "policy_snapshot": current_snapshot,
        },
        sort_keys=True,
    )
    try:
        while line := await asyncio.to_thread(sys.stdin.readline):
            command = json.loads(line)
            if command.get("op") == "shutdown":
                break
            if command.get("op") == "offload":
                result_path = Path(command["result_path"])
                try:
                    result = _offload_runtime(policy, backend)
                except Exception as exc:
                    result = {
                        "ok": False,
                        "worker_index": int(spec["worker_index"]),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                write_json_atomic(result_path, result, indent=2, sort_keys=True)
                continue
            if command.get("op") == "rescore":
                result_path = Path(command["result_path"])
                try:
                    result, current_snapshot = _run_rescore_job(
                        command,
                        policy=policy,
                        backend=backend,
                        current_snapshot=current_snapshot,
                    )
                except Exception as exc:
                    result = {
                        "ok": False,
                        "worker_index": int(spec["worker_index"]),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                write_json_atomic(result_path, result, indent=2, sort_keys=True)
                continue
            if command.get("op") != "gradient":
                raise ValueError(f"Unknown worker command: {command.get('op')!r}")
            result_path = Path(command["result_path"])
            try:
                result, current_snapshot = await _run_gradient_job(
                    command,
                    policy=policy,
                    backend=backend,
                    config=config,
                    current_snapshot=current_snapshot,
                    sft_anchor_provider=sft_anchor_provider,
                )
            except Exception as exc:
                result = {
                    "ok": False,
                    "worker_index": int(spec["worker_index"]),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            write_json_atomic(result_path, result, indent=2, sort_keys=True)
    finally:
        await backend.close()


def main() -> None:
    """Run one action-token worker from its serialized coordinator specification."""

    require_compatible_worker_runtime()
    args = _parse_args()
    source = args.serve_spec or args.spec
    assert source is not None
    spec = json.loads(source.read_text(encoding="utf-8"))
    if args.serve_spec is not None:
        asyncio.run(_serve(spec))
    else:
        result = asyncio.run(_run(spec))
        write_json_atomic(Path(spec["result_path"]), result, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
