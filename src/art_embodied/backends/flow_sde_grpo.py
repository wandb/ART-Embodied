"""Pure GRPO objective math for Flow-SDE action chunks.

The policy adapter owns the sampler and teacher-forced Gaussian rescore. This
module owns only the objective geometry used by RLinf's PI0/PI0.5 positive
control: element logprobs are summed over one executed action chunk before the
importance ratio is formed, and one group-relative trajectory advantage is
broadcast to each valid policy decision.

The conformance target is RLinf release/v0.1 at commit
9df6dc80dc729a6caccab92dd676004e51b1d3a2 (Apache-2.0, Copyright 2025 The
RLinf Authors). This implementation adds ART-Embodied-specific APIs,
validation, masking, diagnostics, distributed weighting, and reference-KL
support.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class FlowSDEGRPOMetrics:
    """Detached diagnostics describing one clipped Flow-SDE policy update."""

    ratio_mean: torch.Tensor
    approximate_kl: torch.Tensor
    approximate_kl_per_primitive: torch.Tensor
    clip_fraction: torch.Tensor
    valid_chunks: torch.Tensor


@dataclass(frozen=True, slots=True)
class FlowSDEReferenceKLMetrics:
    """Detached diagnostics for the immutable SFT-reference penalty."""

    mean_per_primitive: torch.Tensor
    weighted_loss: torch.Tensor
    valid_chunks: torch.Tensor


def group_relative_advantages(
    rewards: torch.Tensor,
    *,
    epsilon: float = 1.0e-6,
    std_unbiased: bool = True,
) -> torch.Tensor:
    """Normalize rewards independently inside each same-reset group."""

    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("rewards must have shape [groups, group_size >= 2]")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    centered = rewards - rewards.mean(dim=-1, keepdim=True)
    scale = rewards.std(dim=-1, keepdim=True, unbiased=std_unbiased)
    return centered / (scale + epsilon)


def chunk_logprobs(
    element_logprobs: torch.Tensor,
    *,
    action_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce ``[batch, horizon, action_dim]`` Gaussian scores to chunks."""

    if element_logprobs.ndim != 3:
        raise ValueError(
            "element_logprobs must have shape [batch, horizon, action_dim]"
        )
    if action_mask is None:
        return element_logprobs.sum(dim=(1, 2))
    mask = _expanded_action_mask(action_mask, element_logprobs)
    return (element_logprobs * mask).sum(dim=(1, 2))


def primitive_normalized_chunk_abs_delta(
    current_element_logprobs: torch.Tensor,
    old_element_logprobs: torch.Tensor,
    *,
    action_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return rollout/rescore drift per executed primitive action.

    Flow-SDE optimizes a joint action-chunk likelihood, so its importance ratio
    must continue to use the summed chunk log-probability.  An alignment guard,
    however, must not become stricter merely because a policy executes a longer
    horizon.  Normalize only the diagnostic delta by the number of executed
    primitive actions; action dimensions remain part of each primitive action.
    """

    if current_element_logprobs.shape != old_element_logprobs.shape:
        raise ValueError("current and old element logprobs must have identical shapes")
    delta = (
        chunk_logprobs(current_element_logprobs, action_mask=action_mask).float()
        - chunk_logprobs(old_element_logprobs, action_mask=action_mask).float()
    ).abs()
    return delta / _primitive_action_counts(action_mask, current_element_logprobs)


def flow_sde_grpo_loss(
    current_element_logprobs: torch.Tensor,
    old_element_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    loss_mask: torch.Tensor | None = None,
    action_mask: torch.Tensor | None = None,
    row_weights: torch.Tensor | None = None,
    loss_denominator: int | torch.Tensor | None = None,
    clip_epsilon_low: float = 0.2,
    clip_epsilon_high: float = 0.2,
    clip_ratio_c: float | None = 3.0,
) -> tuple[torch.Tensor, FlowSDEGRPOMetrics]:
    """Compute RLinf-compatible clipped policy loss per action chunk."""

    if current_element_logprobs.shape != old_element_logprobs.shape:
        raise ValueError("current and old element logprobs must have identical shapes")
    primitive_counts = _primitive_action_counts(action_mask, current_element_logprobs)
    current = chunk_logprobs(current_element_logprobs, action_mask=action_mask)
    old = chunk_logprobs(old_element_logprobs, action_mask=action_mask)
    advantage = advantages.to(device=current.device, dtype=torch.float32).reshape(-1)
    if advantage.shape != current.shape:
        raise ValueError(
            f"advantages must have shape {tuple(current.shape)}, got {tuple(advantage.shape)}"
        )
    if clip_epsilon_low < 0.0 or clip_epsilon_high < 0.0:
        raise ValueError("clip epsilons cannot be negative")
    if clip_ratio_c is not None and clip_ratio_c <= 1.0:
        raise ValueError("clip_ratio_c must be greater than 1")

    mask = _loss_mask(loss_mask, current)
    log_ratio = current.float() - old.float()
    ratio = torch.where(mask, torch.exp(log_ratio), torch.zeros_like(log_ratio))
    clipped_ratio = torch.clamp(
        ratio,
        1.0 - clip_epsilon_low,
        1.0 + clip_epsilon_high,
    )
    loss_unclipped = -advantage * ratio
    loss_clipped = -advantage * clipped_ratio
    policy_loss = torch.maximum(loss_unclipped, loss_clipped)
    if clip_ratio_c is not None:
        dual_bound = torch.sign(advantage) * clip_ratio_c * advantage
        policy_loss = torch.minimum(policy_loss, dual_bound)

    valid = mask.count_nonzero().clamp_min(1)
    if row_weights is None:
        weights = torch.ones_like(policy_loss)
    else:
        weights = row_weights.to(
            device=policy_loss.device, dtype=policy_loss.dtype
        ).reshape(-1)
        if weights.shape != policy_loss.shape:
            raise ValueError(
                f"row_weights must have shape {tuple(policy_loss.shape)}, "
                f"got {tuple(weights.shape)}"
            )
        if not torch.isfinite(weights).all() or torch.any(weights <= 0.0):
            raise ValueError("row_weights must be finite and positive")
    denominator = (
        valid
        if loss_denominator is None
        else torch.as_tensor(
            loss_denominator, device=policy_loss.device, dtype=policy_loss.dtype
        )
    )
    if denominator.numel() != 1 or denominator.item() <= 0:
        raise ValueError("loss_denominator must be a positive scalar")
    loss = (
        torch.where(mask, policy_loss * weights, torch.zeros_like(policy_loss)).sum()
        / denominator
    )
    metrics = FlowSDEGRPOMetrics(
        ratio_mean=torch.where(mask, ratio, torch.zeros_like(ratio)).sum() / valid,
        approximate_kl=-torch.where(
            mask,
            log_ratio.detach(),
            torch.zeros_like(log_ratio),
        ).sum()
        / valid,
        approximate_kl_per_primitive=-torch.where(
            mask,
            log_ratio.detach() / primitive_counts,
            torch.zeros_like(log_ratio),
        ).sum()
        / valid,
        clip_fraction=(
            ((loss_unclipped.detach() < loss_clipped.detach()) & mask)
            .count_nonzero()
            .float()
            / valid
        ),
        valid_chunks=valid,
    )
    return loss, metrics


def flow_sde_reference_kl_loss(
    current_element_logprobs: torch.Tensor,
    reference_element_logprobs: torch.Tensor,
    *,
    loss_mask: torch.Tensor | None = None,
    action_mask: torch.Tensor | None = None,
    row_weights: torch.Tensor | None = None,
    loss_denominator: int | torch.Tensor | None = None,
) -> tuple[torch.Tensor, FlowSDEReferenceKLMetrics]:
    """Estimate ``KL(current || SFT)`` with the non-negative K3 estimator.

    The sampled transition is scored by both policies. The reference score is
    detached here as a fail-safe, and the per-element K3 terms are summed over
    action dimensions then averaged over executed primitive actions. Loss-row
    weighting exactly matches the GRPO objective's aggregation contract.
    """

    if current_element_logprobs.shape != reference_element_logprobs.shape:
        raise ValueError(
            "current and reference element logprobs must have identical shapes"
        )
    if current_element_logprobs.ndim != 3:
        raise ValueError(
            "current and reference element logprobs must have shape "
            "[batch, horizon, action_dim]"
        )
    current = current_element_logprobs.float()
    reference = reference_element_logprobs.detach().to(
        device=current.device, dtype=torch.float32
    )
    element_mask = (
        torch.ones_like(current)
        if action_mask is None
        else _expanded_action_mask(action_mask, current)
    )
    log_ratio = current - reference
    # Schulman's K3 estimator: exp(-r) - 1 + r, where r=log(pi/ref).
    element_kl = (torch.expm1(-log_ratio) + log_ratio).clamp_min(0.0)
    if not torch.isfinite(element_kl).all():
        raise FloatingPointError("non-finite Flow-SDE SFT-reference KL estimate")
    row_kl = (element_kl * element_mask).sum(dim=(1, 2)) / (
        _primitive_action_counts(action_mask, current)
    )
    mask = _loss_mask(loss_mask, row_kl)
    valid = mask.count_nonzero().clamp_min(1)
    if row_weights is None:
        weights = torch.ones_like(row_kl)
    else:
        weights = row_weights.to(device=row_kl.device, dtype=row_kl.dtype).reshape(-1)
        if weights.shape != row_kl.shape:
            raise ValueError(
                f"row_weights must have shape {tuple(row_kl.shape)}, "
                f"got {tuple(weights.shape)}"
            )
        if not torch.isfinite(weights).all() or torch.any(weights <= 0.0):
            raise ValueError("row_weights must be finite and positive")
    denominator = (
        valid
        if loss_denominator is None
        else torch.as_tensor(
            loss_denominator, device=row_kl.device, dtype=row_kl.dtype
        )
    )
    if denominator.numel() != 1 or denominator.item() <= 0:
        raise ValueError("loss_denominator must be a positive scalar")
    weighted_loss = (
        torch.where(mask, row_kl * weights, torch.zeros_like(row_kl)).sum()
        / denominator
    )
    metrics = FlowSDEReferenceKLMetrics(
        mean_per_primitive=(
            torch.where(mask, row_kl.detach(), torch.zeros_like(row_kl)).sum()
            / valid
        ),
        weighted_loss=weighted_loss.detach(),
        valid_chunks=valid,
    )
    return weighted_loss, metrics


def _loss_mask(value: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor:
    if value is None:
        return torch.ones_like(target, dtype=torch.bool)
    mask = value.to(device=target.device, dtype=torch.bool).reshape(-1)
    if mask.shape != target.shape:
        raise ValueError(
            f"loss_mask must have shape {tuple(target.shape)}, got {tuple(mask.shape)}"
        )
    return mask


def _expanded_action_mask(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = value.to(device=target.device, dtype=target.dtype)
    if mask.ndim == 2:
        mask = mask.unsqueeze(-1)
    if mask.ndim != 3 or mask.shape[:2] != target.shape[:2]:
        raise ValueError(
            "action_mask must have shape [batch, horizon] or "
            "[batch, horizon, action_dim]"
        )
    if mask.shape[-1] == 1:
        mask = mask.expand_as(target)
    elif mask.shape != target.shape:
        raise ValueError("action_mask action dimension does not match logprobs")
    return mask


def _primitive_action_counts(
    action_mask: torch.Tensor | None,
    target: torch.Tensor,
) -> torch.Tensor:
    """Count executed primitive actions in each chunk, excluding action dims."""

    if action_mask is None:
        return torch.full(
            (target.shape[0],),
            target.shape[1],
            device=target.device,
            dtype=torch.float32,
        )
    mask = action_mask.to(device=target.device, dtype=torch.bool)
    if mask.ndim == 3:
        if mask.shape != target.shape:
            raise ValueError("action_mask action dimension does not match logprobs")
        mask = mask.any(dim=-1)
    if mask.ndim != 2 or mask.shape != target.shape[:2]:
        raise ValueError(
            "action_mask must have shape [batch, horizon] or "
            "[batch, horizon, action_dim]"
        )
    counts = mask.count_nonzero(dim=1).to(dtype=torch.float32)
    if torch.any(counts <= 0):
        raise ValueError(
            "each Flow-SDE chunk must execute at least one primitive action"
        )
    return counts
