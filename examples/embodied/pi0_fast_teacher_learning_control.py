"""Actual optimizer controls on fixed teacher data, not GRPO qualification."""

import torch

from examples.embodied.pi0_fast_teacher_path_control import (
    teacher_rows,
    teacher_score_loss,
)

ARMS = ("native", "aligned", "sampler")


def objective(policy, batch, arm):
    if arm not in ARMS:
        raise ValueError(f"Unknown teacher arm: {arm}")
    with torch.enable_grad():
        if arm == "native":
            return policy.policy(batch)[0]
        return teacher_score_loss(
            policy.policy, batch, mode="sft" if arm == "aligned" else "sampler"
        )


def token_counts(policy, batches):
    if not batches:
        raise ValueError("Empty teacher batch")
    return [sum(map(len, teacher_rows(policy.policy, b))) for b in batches]


def evaluate(policy, batches, arm):
    counts = token_counts(policy, batches)
    # no_grad changes the actual FAST attention path; detach immediately instead.
    values = [float(objective(policy, b, arm).detach()) for b in batches]
    if not all(torch.isfinite(torch.tensor(values))):
        raise ValueError("Nonfinite teacher evaluation")
    return sum(n * v for n, v in zip(counts, values, strict=True)) / sum(counts)


def update(policy, batches, arm, optimizer, clip_norm):
    counts = token_counts(policy, batches)
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for batch, count in zip(batches, counts, strict=True):
        loss = objective(policy, batch, arm) * count / sum(counts)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite teacher training loss")
        loss.backward()
        total += float(loss.detach())
    trainable = [p for p in policy.parameters() if p.requires_grad]
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable):
        raise ValueError("Incomplete or nonfinite trainable gradient")
    norm = torch.nn.utils.clip_grad_norm_(trainable, clip_norm, error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in trainable):
        raise ValueError("Nonfinite updated weights")
    return total, float(norm)
