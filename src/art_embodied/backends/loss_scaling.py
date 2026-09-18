"""Explicit FAST-only static loss scaling, before gradient handoff or clipping."""

import math


def policy_loss_scale(policy):
    if (
        getattr(policy, "family", None) != "pi0_fast"
        or getattr(policy, "model_compute_dtype", None) != "fp16_residual"
    ):
        return 1.0
    scale = float(policy.training_loss_scale)
    if not math.isfinite(scale) or scale < 1 or not math.log2(scale).is_integer():
        raise ValueError("FAST loss scale must be a finite power of two >= 1")
    return scale


def backward_policy_loss(policy, loss):
    scale = policy_loss_scale(policy)
    if scale == 1:
        loss.backward()
    else:
        (loss * scale).backward()


def unscale_policy_gradients(policy):
    """Call once after all microbatches, before metrics/all-reduce/handoff/clip.

    Nonfinite gradients abort the update; never silently skip an optimizer step
    or change the scale during a run.
    """
    scale = policy_loss_scale(policy)
    if scale == 1:
        return {}
    import torch

    gradients = [p.grad for p in policy.parameters() if p.grad is not None]
    with torch.no_grad():
        for grad in gradients:
            grad.div_(scale)
        if gradients and not bool(
            torch.stack([torch.isfinite(g).all() for g in gradients]).all()
        ):
            raise FloatingPointError("Nonfinite FAST gradients after loss unscaling")
    return {"optimization/loss_scale": scale}
