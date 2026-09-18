"""Gradient handoff and optimizer-application protocol for action-token RL."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _clear_trainable_gradients(policy: Any) -> None:
    """Clear stale gradients from all trainable policy parameters."""

    for parameter in policy.parameters() if hasattr(policy, "parameters") else []:
        if bool(getattr(parameter, "requires_grad", False)):
            parameter.grad = None


def _named_trainable_parameter_iter(policy: Any) -> list[tuple[str, Any]]:
    if hasattr(policy, "named_parameters"):
        return [
            (str(name), parameter)
            for name, parameter in policy.named_parameters()
            if bool(getattr(parameter, "requires_grad", False))
        ]
    if hasattr(policy, "parameters"):
        return [
            (f"parameter_{index}", parameter)
            for index, parameter in enumerate(policy.parameters())
            if bool(getattr(parameter, "requires_grad", False))
        ]
    return []


def _parameter_metric_role(name: str) -> str:
    if "lora_A" in name:
        return "lora_a"
    if "lora_B" in name:
        return "lora_b"
    if "lora_" in name.lower():
        return "lora_other"
    return "non_lora"


def _gradient_metrics(
    policy: Any,
    *,
    prefix: str = "embodied_action_token_grpo",
) -> dict[str, float]:
    import torch

    trainable_tensors = 0
    tensors_with_grad = 0
    nonzero_grad_tensors = 0
    grad_norm_sq = 0.0
    grad_abs_max = 0.0
    role_stats: dict[str, dict[str, float]] = {}
    for name, parameter in _named_trainable_parameter_iter(policy):
        if not bool(getattr(parameter, "requires_grad", False)):
            continue
        role = _parameter_metric_role(name)
        stats = role_stats.setdefault(
            role,
            {
                "trainable_tensors": 0.0,
                "tensors_with_grad": 0.0,
                "nonzero_grad_tensors": 0.0,
                "grad_norm_sq": 0.0,
                "grad_abs_max": 0.0,
            },
        )
        trainable_tensors += 1
        stats["trainable_tensors"] += 1.0
        grad = getattr(parameter, "grad", None)
        if grad is None:
            continue
        tensors_with_grad += 1
        stats["tensors_with_grad"] += 1.0
        values = grad.detach().float()
        if values.numel() == 0:
            continue
        norm = float(torch.linalg.vector_norm(values).cpu().item())
        max_abs = float(values.abs().max().cpu().item())
        grad_norm_sq += norm * norm
        grad_abs_max = max(grad_abs_max, max_abs)
        stats["grad_norm_sq"] += norm * norm
        stats["grad_abs_max"] = max(float(stats["grad_abs_max"]), max_abs)
        if max_abs > 0.0:
            nonzero_grad_tensors += 1
            stats["nonzero_grad_tensors"] += 1.0
    metrics = {
        f"{prefix}/trainable_tensors": float(trainable_tensors),
        f"{prefix}/tensors_with_grad": float(tensors_with_grad),
        f"{prefix}/nonzero_grad_tensors": float(nonzero_grad_tensors),
        f"{prefix}/grad_norm": float(grad_norm_sq**0.5),
        f"{prefix}/grad_abs_max": float(grad_abs_max),
    }
    for role, stats in role_stats.items():
        metrics[f"{prefix}/{role}_trainable_tensors"] = float(
            stats["trainable_tensors"]
        )
        metrics[f"{prefix}/{role}_tensors_with_grad"] = float(
            stats["tensors_with_grad"]
        )
        metrics[f"{prefix}/{role}_nonzero_grad_tensors"] = float(
            stats["nonzero_grad_tensors"]
        )
        metrics[f"{prefix}/{role}_grad_norm"] = float(stats["grad_norm_sq"] ** 0.5)
        metrics[f"{prefix}/{role}_grad_abs_max"] = float(stats["grad_abs_max"])
    return metrics


def _trainable_gradient_payload(policy: Any) -> dict[str, Any]:
    """Return CPU gradients for every trainable parameter.

    Distributed embodied GRPO uses this as a narrow gradient handoff surface:
    workers compute exactly the same objective as ``ActionTokenGRPOBackend`` but
    do not step an optimizer.  The parent sums these gradients onto its own
    policy and applies the real optimizer step once.
    """

    import torch

    gradients: dict[str, Any] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    missing: list[str] = []
    for name, parameter in (
        policy.named_parameters() if hasattr(policy, "named_parameters") else []
    ):
        if not bool(getattr(parameter, "requires_grad", False)):
            continue
        shapes[name] = tuple(parameter.shape)
        grad = getattr(parameter, "grad", None)
        if grad is None:
            missing.append(name)
            gradients[name] = torch.zeros(tuple(parameter.shape), dtype=torch.float32)
        else:
            gradients[name] = grad.detach().float().cpu().clone()
    return {
        "format": "art_embodied_action_token_grpo_gradients_v1",
        "gradients": gradients,
        "shapes": shapes,
        "missing_gradients": missing,
    }


def _save_gradient_payload(payload: dict[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_action_token_gradient_payload(path: str | Path) -> dict[str, Any]:
    """Load one trusted worker-gradient payload onto CPU memory."""

    import torch

    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location="cpu")


def _optimizer_step_state_summary(optimizer: Any) -> dict[str, float]:
    """Summarize Adam-style per-parameter step counters without assuming AdamW."""

    steps: list[float] = []
    state = getattr(optimizer, "state", {})
    for parameter_state in state.values() if hasattr(state, "values") else []:
        if not isinstance(parameter_state, Mapping):
            continue
        value = parameter_state.get("step")
        if value is None:
            continue
        if hasattr(value, "numel"):
            if int(value.numel()) != 1:
                continue
            value = value.detach().item()
        try:
            steps.append(float(value))
        except (TypeError, ValueError):
            continue
    return {
        "entries": float(len(steps)),
        "step_min": float(min(steps)) if steps else 0.0,
        "step_max": float(max(steps)) if steps else 0.0,
        "step_mean": float(sum(steps) / len(steps)) if steps else 0.0,
    }


def _gradient_payload_coherence_metrics(
    payloads: Sequence[dict[str, Any]],
    *,
    prefix: str,
) -> dict[str, float]:
    """Measure whether independently sampled worker gradients agree.

    A large aggregate gradient is not sufficient evidence of a useful policy
    update: high-variance score-function estimates can also sum to a large
    vector. These metrics compare worker gradients before summation so a run
    can distinguish coherent reward-conditioned movement from cancellation.
    """

    import torch

    gradient_maps: list[dict[str, Any]] = []
    for payload in payloads:
        gradients = payload.get("gradients") or {}
        if not isinstance(gradients, dict):
            raise ValueError("Gradient payload has no gradients mapping")
        gradient_maps.append(gradients)

    worker_count = len(gradient_maps)
    norm_squares = [0.0] * worker_count
    pairwise_dots = {
        (left, right): 0.0
        for left in range(worker_count)
        for right in range(left + 1, worker_count)
    }
    parameter_names = sorted(
        {name for gradients in gradient_maps for name in gradients}
    )
    for name in parameter_names:
        tensors: list[Any | None] = []
        expected_shape: tuple[int, ...] | None = None
        for gradients in gradient_maps:
            value = gradients.get(name)
            if value is None:
                tensors.append(None)
                continue
            tensor = value.detach().float().cpu()
            shape = tuple(tensor.shape)
            if expected_shape is None:
                expected_shape = shape
            elif shape != expected_shape:
                raise ValueError(
                    f"Gradient payload shape mismatch for {name!r}: "
                    f"expected {expected_shape}, got {shape}"
                )
            tensors.append(tensor)
        for index, tensor in enumerate(tensors):
            if tensor is not None:
                norm_squares[index] += float(torch.sum(tensor * tensor).item())
        for left in range(worker_count):
            if tensors[left] is None:
                continue
            for right in range(left + 1, worker_count):
                if tensors[right] is not None:
                    pairwise_dots[(left, right)] += float(
                        torch.sum(tensors[left] * tensors[right]).item()
                    )

    norms = [value**0.5 for value in norm_squares]
    nonzero = sum(norm > 0.0 for norm in norms)
    cosine_values = [
        dot / (norms[left] * norms[right])
        for (left, right), dot in pairwise_dots.items()
        if norms[left] > 0.0 and norms[right] > 0.0
    ]
    summed_norm_square = sum(norm_squares) + 2.0 * sum(pairwise_dots.values())
    summed_norm = max(summed_norm_square, 0.0) ** 0.5
    norm_sum = sum(norms)
    rms_norm = (sum(norm_squares) / worker_count) ** 0.5 if worker_count else 0.0
    mean_gradient_norm = summed_norm / worker_count if worker_count else 0.0
    residual_rms = max(rms_norm * rms_norm - mean_gradient_norm**2, 0.0) ** 0.5
    noise_to_signal = (
        residual_rms / mean_gradient_norm
        if mean_gradient_norm > 0.0
        else (1.0e12 if residual_rms > 0.0 else 0.0)
    )
    resultant_ratio = summed_norm / norm_sum if norm_sum > 0.0 else 0.0

    return {
        f"{prefix}/worker_gradient_payloads": float(worker_count),
        f"{prefix}/worker_gradient_nonzero_payloads": float(nonzero),
        f"{prefix}/worker_gradient_norm_mean": (
            sum(norms) / worker_count if worker_count else 0.0
        ),
        f"{prefix}/worker_gradient_norm_rms": float(rms_norm),
        f"{prefix}/worker_gradient_norm_min": min(norms) if norms else 0.0,
        f"{prefix}/worker_gradient_norm_max": max(norms) if norms else 0.0,
        f"{prefix}/worker_gradient_pairwise_cosine_mean": (
            sum(cosine_values) / len(cosine_values) if cosine_values else 0.0
        ),
        f"{prefix}/worker_gradient_pairwise_cosine_min": (
            min(cosine_values) if cosine_values else 0.0
        ),
        f"{prefix}/worker_gradient_pairwise_cosine_max": (
            max(cosine_values) if cosine_values else 0.0
        ),
        f"{prefix}/worker_gradient_resultant_ratio": float(resultant_ratio),
        f"{prefix}/worker_gradient_signal_to_rms_ratio": (
            mean_gradient_norm / rms_norm if rms_norm > 0.0 else 0.0
        ),
        f"{prefix}/worker_gradient_noise_to_signal_ratio": float(noise_to_signal),
        f"{prefix}/worker_gradient_effective_aligned_workers": (
            float(worker_count) * resultant_ratio**2
        ),
    }


def _gradient_map_dot(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    import torch

    total = 0.0
    for name in sorted(set(left).intersection(right)):
        left_tensor = left[name].detach().float().cpu()
        right_tensor = right[name].detach().float().cpu()
        if tuple(left_tensor.shape) != tuple(right_tensor.shape):
            raise ValueError(f"Gradient payload shape mismatch for {name!r}")
        total += float(torch.sum(left_tensor * right_tensor).item())
    return total


def _gradient_payload_pairwise_summary(
    payloads: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """Summarize pairwise task-gradient conflict without flattening tensors."""

    gradient_maps: list[Mapping[str, Any]] = []
    norm_squares: list[float] = []
    for payload in payloads:
        gradients = payload.get("gradients") or {}
        if not isinstance(gradients, dict):
            raise ValueError("Gradient payload has no gradients mapping")
        gradient_maps.append(gradients)
        norm_squares.append(_gradient_map_dot(gradients, gradients))

    dots = [
        _gradient_map_dot(gradient_maps[left], gradient_maps[right])
        for left in range(len(gradient_maps))
        for right in range(left + 1, len(gradient_maps))
    ]
    cosines: list[float] = []
    pair_index = 0
    for left in range(len(gradient_maps)):
        for right in range(left + 1, len(gradient_maps)):
            denominator = (
                max(norm_squares[left], 0.0) ** 0.5
                * max(norm_squares[right], 0.0) ** 0.5
            )
            if denominator > 0.0:
                cosines.append(dots[pair_index] / denominator)
            pair_index += 1
    conflicting_pairs = sum(dot < 0.0 for dot in dots)
    return {
        "pairs": float(len(dots)),
        "conflicting_pairs": float(conflicting_pairs),
        "conflict_fraction": (float(conflicting_pairs / len(dots)) if dots else 0.0),
        "cosine_mean": sum(cosines) / len(cosines) if cosines else 0.0,
        "cosine_min": min(cosines) if cosines else 0.0,
        "cosine_max": max(cosines) if cosines else 0.0,
    }


def _gradient_payload_sum_norm(payloads: Sequence[dict[str, Any]]) -> float:
    import torch

    totals: dict[str, Any] = {}
    for payload in payloads:
        gradients = payload.get("gradients") or {}
        if not isinstance(gradients, dict):
            raise ValueError("Gradient payload has no gradients mapping")
        for name, value in gradients.items():
            tensor = value.detach().float().cpu()
            if name in totals:
                totals[name] = totals[name] + tensor
            else:
                totals[name] = tensor.clone()
    norm_square = sum(
        float(torch.sum(value * value).item()) for value in totals.values()
    )
    return norm_square**0.5


def _task_pcgrad_payloads(
    payloads: Sequence[dict[str, Any]],
    *,
    prefix: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Deterministically project negative task-gradient components."""

    task_keys = [str(payload.get("task_key") or "") for payload in payloads]
    if any(not task_key for task_key in task_keys):
        raise ValueError("task_pcgrad requires a task_key on every gradient payload")
    if len(set(task_keys)) != len(task_keys):
        raise ValueError("task_pcgrad requires exactly one payload per task_key")

    ordered = sorted(payloads, key=lambda payload: str(payload["task_key"]))
    original: list[dict[str, Any]] = []
    for payload in ordered:
        gradients = payload.get("gradients") or {}
        if not isinstance(gradients, dict):
            raise ValueError("Gradient payload has no gradients mapping")
        original.append(
            {
                name: tensor.detach().float().cpu().clone()
                for name, tensor in gradients.items()
            }
        )

    raw_pairwise = _gradient_payload_pairwise_summary(ordered)
    pair_count = int(raw_pairwise["pairs"])
    conflicting_pairs = int(raw_pairwise["conflicting_pairs"])
    raw_norm = _gradient_payload_sum_norm(ordered)
    if conflicting_pairs == 0:
        return list(payloads), {
            f"{prefix}/task_pcgrad_enabled": 1.0,
            f"{prefix}/task_gradient_pairs": float(pair_count),
            f"{prefix}/task_gradient_conflicting_pairs": 0.0,
            f"{prefix}/task_gradient_conflict_fraction": 0.0,
            f"{prefix}/task_gradient_raw_pairwise_cosine_mean": raw_pairwise[
                "cosine_mean"
            ],
            f"{prefix}/task_gradient_raw_pairwise_cosine_min": raw_pairwise[
                "cosine_min"
            ],
            f"{prefix}/task_gradient_projected_conflicting_pairs": 0.0,
            f"{prefix}/task_gradient_projected_conflict_fraction": 0.0,
            f"{prefix}/task_gradient_projected_pairwise_cosine_mean": raw_pairwise[
                "cosine_mean"
            ],
            f"{prefix}/task_gradient_projected_pairwise_cosine_min": raw_pairwise[
                "cosine_min"
            ],
            f"{prefix}/task_pcgrad_projections": 0.0,
            f"{prefix}/task_gradient_raw_aggregate_norm": float(raw_norm),
            f"{prefix}/task_gradient_projected_aggregate_norm": float(raw_norm),
            f"{prefix}/task_gradient_projected_to_raw_norm_ratio": 1.0,
        }

    projected = [
        {name: tensor.clone() for name, tensor in gradients.items()}
        for gradients in original
    ]
    projections = 0
    for left in range(len(projected)):
        for offset in range(1, len(original)):
            right = (left + offset) % len(original)
            denominator = _gradient_map_dot(original[right], original[right])
            if denominator <= 0.0:
                continue
            dot = _gradient_map_dot(projected[left], original[right])
            if dot >= 0.0:
                continue
            scale = dot / denominator
            for name, reference in original[right].items():
                if name not in projected[left]:
                    projected[left][name] = -scale * reference
                else:
                    projected[left][name].add_(reference, alpha=-scale)
            projections += 1

    projected_payloads = []
    for payload, gradients in zip(ordered, projected, strict=True):
        projected_payloads.append({**payload, "gradients": gradients})
    projected_pairwise = _gradient_payload_pairwise_summary(projected_payloads)
    projected_norm = _gradient_payload_sum_norm(projected_payloads)
    ratio = projected_norm / raw_norm if raw_norm > 0.0 else 1.0e12
    return projected_payloads, {
        f"{prefix}/task_pcgrad_enabled": 1.0,
        f"{prefix}/task_gradient_pairs": float(pair_count),
        f"{prefix}/task_gradient_conflicting_pairs": float(conflicting_pairs),
        f"{prefix}/task_gradient_conflict_fraction": (
            float(conflicting_pairs / pair_count) if pair_count else 0.0
        ),
        f"{prefix}/task_gradient_raw_pairwise_cosine_mean": raw_pairwise["cosine_mean"],
        f"{prefix}/task_gradient_raw_pairwise_cosine_min": raw_pairwise["cosine_min"],
        f"{prefix}/task_gradient_projected_conflicting_pairs": projected_pairwise[
            "conflicting_pairs"
        ],
        f"{prefix}/task_gradient_projected_conflict_fraction": projected_pairwise[
            "conflict_fraction"
        ],
        f"{prefix}/task_gradient_projected_pairwise_cosine_mean": projected_pairwise[
            "cosine_mean"
        ],
        f"{prefix}/task_gradient_projected_pairwise_cosine_min": projected_pairwise[
            "cosine_min"
        ],
        f"{prefix}/task_pcgrad_projections": float(projections),
        f"{prefix}/task_gradient_raw_aggregate_norm": float(raw_norm),
        f"{prefix}/task_gradient_projected_aggregate_norm": float(projected_norm),
        f"{prefix}/task_gradient_projected_to_raw_norm_ratio": float(ratio),
    }


def apply_action_token_gradient_payloads(
    policy: Any,
    optimizer: Any,
    payloads: Sequence[dict[str, Any]],
    *,
    max_grad_norm: float | None = None,
    optimizer_lr_scale: float | None = None,
    skip_optimizer_step_without_policy_gradient_signal: bool = True,
    prefix: str = "embodied_action_token_grpo",
    gradient_aggregation: str = "sum",
) -> dict[str, float]:
    """Aggregate worker gradients onto ``policy`` and step ``optimizer`` once."""

    import torch

    if optimizer is None:
        raise ValueError("apply_action_token_gradient_payloads requires an optimizer")
    if not payloads:
        raise ValueError("No gradient payloads were provided")
    partition_modes = {
        str(payload["rank_partition_mode"])
        for payload in payloads
        if payload.get("rank_partition_mode") is not None
    }
    if len(partition_modes) > 1:
        raise ValueError(
            f"Gradient payloads disagree on rank-partition mode: {partition_modes}"
        )
    partition_mode = next(iter(partition_modes), None)
    if partition_mode is not None and partition_mode != "task_all_active":
        raise ValueError(
            f"Unsupported gradient rank-partition mode: {partition_mode!r}"
        )
    if partition_mode is not None and len(partition_modes) != 1:
        raise ValueError("Invalid gradient rank-partition payload metadata")
    named_parameters = {
        name: parameter
        for name, parameter in policy.named_parameters()
        if bool(getattr(parameter, "requires_grad", False))
    }
    if not named_parameters:
        raise ValueError("The parent policy has no trainable parameters")
    for payload_index, payload in enumerate(payloads):
        gradients = payload.get("gradients") or {}
        if not isinstance(gradients, dict):
            raise ValueError("Gradient payload has no gradients mapping")
        expected = set(named_parameters)
        if partition_mode == "task_all_active":
            from art_embodied.lora_rank_partition import adapter_from_parameter_name

            adapters = payload.get("active_adapters")
            if (
                payload.get("rank_partition_mode") != partition_mode
                or not isinstance(adapters, list)
                or not adapters
                or any(not isinstance(name, str) for name in adapters)
            ):
                raise ValueError("Invalid task_all_active gradient payload metadata")
            known = {adapter_from_parameter_name(name) for name in expected}
            if None in known or not set(adapters) <= known:
                raise ValueError("Unknown adapter in gradient payload")
            expected = {
                name
                for name in expected
                if adapter_from_parameter_name(name) in adapters
            }
        if set(gradients) != expected:
            raise ValueError(
                f"Gradient payload keys mismatch: payload={payload_index}, "
                f"missing={sorted(expected - set(gradients))[:20]}, "
                f"unexpected={sorted(set(gradients) - expected)[:20]}"
            )
        for name, gradient in gradients.items():
            if not torch.is_tensor(gradient):
                raise TypeError(
                    "Gradient payload values must be tensors: "
                    f"payload={payload_index}, parameter={name!r}"
                )
            if gradient.shape != named_parameters[name].shape:
                raise ValueError(f"Gradient payload shape mismatches: {name}")
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError(
                    "Refusing to apply a non-finite distributed gradient: "
                    f"payload={payload_index}, parameter={name!r}"
                )
    coherence_metrics = _gradient_payload_coherence_metrics(
        payloads,
        prefix=prefix,
    )
    if gradient_aggregation == "sum":
        aggregation_payloads = list(payloads)
        aggregation_metrics = {f"{prefix}/task_pcgrad_enabled": 0.0}
    elif gradient_aggregation == "task_pcgrad":
        if partition_mode is not None:
            raise ValueError("task_pcgrad cannot be combined with rank partitioning")
        aggregation_payloads, aggregation_metrics = _task_pcgrad_payloads(
            payloads,
            prefix=prefix,
        )
    else:
        raise ValueError(f"Unsupported gradient aggregation: {gradient_aggregation!r}")
    for parameter in named_parameters.values():
        parameter.grad = None

    payload_count = 0
    tensors_applied = 0
    missing_payload_tensors = 0
    shape_mismatches = 0
    for payload in aggregation_payloads:
        gradients = payload.get("gradients") or {}
        if not isinstance(gradients, dict):
            raise ValueError("Gradient payload has no gradients mapping")
        payload_count += 1
        for name, parameter in named_parameters.items():
            grad = gradients.get(name)
            if grad is None:
                missing_payload_tensors += 1
                continue
            grad = grad.detach().to(device=parameter.device, dtype=parameter.dtype)
            if tuple(grad.shape) != tuple(parameter.shape):
                shape_mismatches += 1
                continue
            if parameter.grad is None:
                parameter.grad = grad.clone()
            else:
                parameter.grad = parameter.grad + grad
            tensors_applied += 1

    if shape_mismatches:
        raise ValueError(f"Gradient payload shape mismatches: {shape_mismatches}")

    active_adapters: set[str] = set()
    if partition_mode == "task_all_active":
        from art_embodied.lora_rank_partition import adapter_from_parameter_name

        for payload in payloads:
            if payload.get("rank_partition_mode") != partition_mode:
                raise ValueError(
                    "Every worker payload must declare task_all_active when one does"
                )
            active_adapters.update(str(value) for value in payload["active_adapters"])
        for name, parameter in named_parameters.items():
            adapter = adapter_from_parameter_name(name)
            if adapter is None:
                raise ValueError(
                    "Task-partitioned optimizer received a non-adapter trainable "
                    f"parameter: {name}"
                )
            if adapter not in active_adapters:
                # AdamW skips parameters with grad=None, preserving their value,
                # moments, per-parameter step, and avoiding decoupled weight decay.
                parameter.grad = None

    pre_clip_metrics = _gradient_metrics(policy, prefix=prefix)
    has_nonzero_gradient = float(pre_clip_metrics.get(f"{prefix}/grad_norm", 0.0)) > 0.0
    if not has_nonzero_gradient and skip_optimizer_step_without_policy_gradient_signal:
        metrics = {
            f"{prefix}/distributed_gradient_handoff": 1.0,
            f"{prefix}/distributed_gradient_payloads": float(payload_count),
            f"{prefix}/distributed_gradient_tensors_applied": float(tensors_applied),
            f"{prefix}/distributed_gradient_missing_payload_tensors": float(
                missing_payload_tensors
            ),
            f"{prefix}/optimizer_step_completed": 0.0,
            f"{prefix}/optimizer_step_skipped_no_group_signal": 1.0,
            f"{prefix}/optimizer_step_skipped_logprob_misalignment": 0.0,
            f"{prefix}/policy_parameters_updated": 0.0,
        }
        metrics.update(pre_clip_metrics)
        metrics.update(coherence_metrics)
        metrics.update(aggregation_metrics)
        return metrics
    if max_grad_norm is not None:
        _clip_grad_norm(policy, max_grad_norm)
    post_clip_metrics = _gradient_metrics(policy, prefix=prefix)
    parameter_snapshot = _trainable_parameter_snapshot(policy)
    lr_scale = 1.0 if optimizer_lr_scale is None else float(optimizer_lr_scale)
    if lr_scale <= 0.0:
        raise ValueError("optimizer_lr_scale must be positive when provided")
    original_lrs: list[float] = []
    for group in optimizer.param_groups:
        original_lrs.append(float(group.get("lr", 0.0)))
        if optimizer_lr_scale is not None:
            group["lr"] = float(group.get("lr", 0.0)) * lr_scale
    optimizer_state_before = _optimizer_step_state_summary(optimizer)
    try:
        optimizer.step()
    finally:
        if optimizer_lr_scale is not None:
            for group, lr in zip(optimizer.param_groups, original_lrs, strict=True):
                group["lr"] = lr
    optimizer_state_after = _optimizer_step_state_summary(optimizer)
    parameter_update_metrics = _parameter_update_metrics(
        policy,
        parameter_snapshot,
        prefix=prefix,
    )
    metrics = {
        f"{prefix}/distributed_gradient_handoff": 1.0,
        f"{prefix}/distributed_gradient_payloads": float(payload_count),
        f"{prefix}/distributed_gradient_tensors_applied": float(tensors_applied),
        f"{prefix}/distributed_gradient_missing_payload_tensors": float(
            missing_payload_tensors
        ),
        f"{prefix}/optimizer_step_completed": 1.0,
        f"{prefix}/optimizer_step_skipped_no_group_signal": 0.0,
        f"{prefix}/optimizer_step_skipped_logprob_misalignment": 0.0,
        f"{prefix}/optimizer_lr_scale": float(lr_scale),
        f"{prefix}/rank_partition_enabled": float(partition_mode is not None),
        f"{prefix}/rank_partition_active_adapters": float(len(active_adapters)),
        f"{prefix}/optimizer_state_entries_before": optimizer_state_before["entries"],
        f"{prefix}/optimizer_state_step_min_before": optimizer_state_before["step_min"],
        f"{prefix}/optimizer_state_step_max_before": optimizer_state_before["step_max"],
        f"{prefix}/optimizer_state_entries_after": optimizer_state_after["entries"],
        f"{prefix}/optimizer_state_step_min_after": optimizer_state_after["step_min"],
        f"{prefix}/optimizer_state_step_max_after": optimizer_state_after["step_max"],
        f"{prefix}/optimizer_state_step_mean_after": optimizer_state_after["step_mean"],
    }
    metrics.update(pre_clip_metrics)
    metrics.update(coherence_metrics)
    metrics.update(aggregation_metrics)
    if max_grad_norm is not None:
        metrics[f"{prefix}/grad_norm_before_clip"] = pre_clip_metrics.get(
            f"{prefix}/grad_norm",
            0.0,
        )
        metrics[f"{prefix}/grad_abs_max_before_clip"] = pre_clip_metrics.get(
            f"{prefix}/grad_abs_max",
            0.0,
        )
        metrics.update(post_clip_metrics)
    metrics.update(parameter_update_metrics)
    return metrics


def _optimizer_parameter_lrs(optimizer: Any, *, default_lr: float) -> dict[int, float]:
    lrs: dict[int, float] = {}
    if optimizer is None:
        return lrs
    for group in getattr(optimizer, "param_groups", []) or []:
        lr = float(group.get("lr", default_lr))
        for parameter in group.get("params", []) or []:
            lrs[id(parameter)] = lr
    return lrs


def _trainable_parameter_snapshot(policy: Any) -> list[Any]:
    import torch

    snapshots = []
    for parameter in policy.parameters() if hasattr(policy, "parameters") else []:
        if not bool(getattr(parameter, "requires_grad", False)):
            continue
        snapshots.append(parameter.detach().float().cpu().clone())
    return snapshots


def _restore_trainable_parameter_snapshot(policy: Any, before: list[Any]) -> bool:
    if not before or not hasattr(policy, "parameters"):
        return False
    before_iter = iter(before)
    restored = 0
    import torch

    with torch.no_grad():
        for parameter in policy.parameters():
            if not bool(getattr(parameter, "requires_grad", False)):
                continue
            try:
                old_values = next(before_iter)
            except StopIteration:
                break
            parameter.copy_(
                old_values.to(device=parameter.device, dtype=parameter.dtype)
            )
            restored += 1
    return restored > 0


def _parameter_update_metrics(
    policy: Any,
    before: list[Any],
    *,
    prefix: str,
) -> dict[str, float]:
    import torch

    changed_tensors = 0
    compared_tensors = 0
    delta_norm_sq = 0.0
    delta_abs_max = 0.0
    before_iter = iter(before)
    role_stats: dict[str, dict[str, float]] = {}
    for name, parameter in _named_trainable_parameter_iter(policy):
        if not bool(getattr(parameter, "requires_grad", False)):
            continue
        role = _parameter_metric_role(name)
        stats = role_stats.setdefault(
            role,
            {
                "compared_tensors": 0.0,
                "changed_tensors": 0.0,
                "delta_norm_sq": 0.0,
                "delta_abs_max": 0.0,
            },
        )
        try:
            old_values = next(before_iter)
        except StopIteration:
            break
        compared_tensors += 1
        stats["compared_tensors"] += 1.0
        new_values = parameter.detach().float().cpu()
        if tuple(old_values.shape) != tuple(new_values.shape):
            changed_tensors += 1
            stats["changed_tensors"] += 1.0
            continue
        delta = new_values - old_values
        if delta.numel() == 0:
            continue
        norm = float(torch.linalg.vector_norm(delta).item())
        max_abs = float(delta.abs().max().item())
        delta_norm_sq += norm * norm
        delta_abs_max = max(delta_abs_max, max_abs)
        stats["delta_norm_sq"] += norm * norm
        stats["delta_abs_max"] = max(float(stats["delta_abs_max"]), max_abs)
        if max_abs > 0.0:
            changed_tensors += 1
            stats["changed_tensors"] += 1.0
    metrics = {
        f"{prefix}/parameter_tensors_compared": float(compared_tensors),
        f"{prefix}/updated_parameter_tensors": float(changed_tensors),
        f"{prefix}/parameter_delta_norm": float(delta_norm_sq**0.5),
        f"{prefix}/parameter_delta_abs_max": float(delta_abs_max),
        f"{prefix}/policy_parameters_updated": float(changed_tensors > 0),
    }
    for role, stats in role_stats.items():
        metrics[f"{prefix}/{role}_parameter_tensors_compared"] = float(
            stats["compared_tensors"]
        )
        metrics[f"{prefix}/{role}_updated_parameter_tensors"] = float(
            stats["changed_tensors"]
        )
        metrics[f"{prefix}/{role}_parameter_delta_norm"] = float(
            stats["delta_norm_sq"] ** 0.5
        )
        metrics[f"{prefix}/{role}_parameter_delta_abs_max"] = float(
            stats["delta_abs_max"]
        )
    return metrics


def _clip_grad_norm(policy: Any, max_norm: float) -> None:
    if not hasattr(policy, "parameters"):
        return
    import torch

    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm)
