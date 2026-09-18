"""Likelihood ratios for a complete sampled action-token chunk.

This is an opt-in joint latent-action PPO surrogate with trajectory-relative
advantages, not GSPO's length-normalized ratio and not a trajectory-wide ratio.
For FAST, retain the entire sampled BPE/DCT sequence: token positions are not
independent primitive-action times that can be truncated to the executed prefix.
"""

from typing import Any


def action_chunk_log_ratio(current: Any, old: Any, example: Any) -> Any:
    if current.ndim != 1 or current.shape != old.shape or current.numel() == 0:
        raise ValueError("Action-chunk ratios require matching nonempty token vectors")
    if len(example.tokens) != current.numel():
        raise ValueError("Action-chunk ratio must score the complete sampled sequence")
    metadata = example.metadata
    if metadata.get("action_spans") is not None:
        raise ValueError("Action-chunk ratios cannot combine multiple action decisions")
    if metadata.get("token_advantages") is not None:
        raise ValueError("Action-chunk ratios require one scalar trajectory advantage")
    mask = metadata.get("token_loss_mask")
    if mask is not None and (
        not isinstance(mask, list | tuple)
        or len(mask) != current.numel()
        or not all(value == 1 for value in mask)
    ):
        raise ValueError("Action-chunk ratios do not support partial token masks")
    # Reduce before exponentiating/clipping. No division by sequence length.
    return (current.float() - old.detach().float()).sum()
