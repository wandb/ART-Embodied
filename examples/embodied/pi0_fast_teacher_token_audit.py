"""Read-only loss decomposition through actual native SFT and ART scoring."""

import math


def split_token_losses(tokens, losses, prefix, suffix):
    if not prefix or not suffix or len(tokens) <= len(prefix) + len(suffix):
        raise ValueError("Empty protocol or payload")
    if tokens[: len(prefix)] != prefix or tokens[-len(suffix) :] != suffix:
        raise ValueError("Teacher token protocol mismatch")
    if len(losses) != len(tokens) or not all(math.isfinite(v) for v in losses):
        raise ValueError("Invalid per-token losses")
    sections = {
        "prefix": losses[: len(prefix)],
        "payload": losses[len(prefix) : -len(suffix)],
        "suffix": losses[-len(suffix) :],
    }
    return {
        key: {
            "count": len(values),
            "nll_sum": sum(values),
            "ce": sum(values) / len(values),
        }
        for key, values in sections.items()
    }


def native_token_nll(native, batch):
    from lerobot.utils.constants import ACTION_TOKEN_MASK, ACTION_TOKENS
    import torch

    captures = []
    targets = batch[ACTION_TOKENS][:, 1:]
    mask = batch[ACTION_TOKEN_MASK][:, 1:].bool()

    def capture(_module, _inputs, logits):
        # Hook the actual native forward; do not rebuild its attention masks.
        with torch.no_grad():
            values = -torch.log_softmax(logits[:, :-1].detach().float(), dim=-1)
            captures.append(values.gather(-1, targets.unsqueeze(-1)).squeeze(-1))

    handle = native.model.paligemma_with_expert.paligemma.lm_head.register_forward_hook(
        capture
    )
    try:
        with torch.enable_grad():
            loss = float(native(batch)[0].detach())
    finally:
        handle.remove()
    if len(captures) != 1 or targets.shape[0] != 1:
        raise ValueError("Expected one native head call for one teacher frame")
    values = captures[0][mask]
    if not math.isclose(float(values.mean()), loss, abs_tol=2e-5, rel_tol=2e-5):
        raise ValueError("Per-token reconstruction differs from actual native loss")
    return values.cpu().tolist(), loss


def measure(policy, batch):
    import torch

    from art_embodied.policies.pi0_fast import _pi0_fast_token_logprobs
    from examples.embodied.pi0_fast_teacher_path_control import teacher_rows

    native = policy.policy
    rows = teacher_rows(native, batch)
    if len(rows) != 1:
        raise ValueError("Expected one teacher frame")
    prefix = native._paligemma_tokenizer.encode("Action: ", add_special_tokens=False)
    # Native ActionTokenizerProcessorStep uses default special tokens here,
    # unlike its Action prefix: the teacher suffix includes tokenizer EOS.
    suffix = native._paligemma_tokenizer.encode("|")
    values, native_loss = native_token_nll(native, batch)
    with torch.enable_grad():
        scores = _pi0_fast_token_logprobs(
            native_policy=native,
            processed_batch=batch,
            token_rows=rows,
            temperature=1.0,
        )
    sampler = (-scores[0].detach()).cpu().tolist()
    return {
        "tokens": rows[0],
        "native_nll": values,
        "sampler_nll": sampler,
        "native_loss": native_loss,
        "native": split_token_losses(rows[0], values, prefix, suffix),
        "sampler": split_token_losses(rows[0], sampler, prefix, suffix),
    }


def aggregate_details(rows):
    result = {}
    for mode in ("native", "sampler"):
        for section in ("prefix", "payload", "suffix"):
            parts = [row["token_detail"][mode][section] for row in rows]
            result[f"{mode}_{section}_ce"] = sum(p["nll_sum"] for p in parts) / sum(
                p["count"] for p in parts
            )
    return result
