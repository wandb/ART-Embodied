"""Teacher-likelihood controls, not GRPO training or a production mask change.

Call the native LeRobot SFT forward and ART's actual rollout scorer on identical
prepared teacher tokens. The optional BOS boundary intervention changes only
whether conditioning tokens can attend to the appended action BOS.
"""

from contextlib import contextmanager


def teacher_rows(native, batch):
    from lerobot.utils.constants import ACTION_TOKEN_MASK, ACTION_TOKENS
    import torch

    tokens, mask = batch[ACTION_TOKENS], batch[ACTION_TOKEN_MASK].bool()
    if tokens.ndim != 2 or tokens.shape != mask.shape or tokens.shape[1] < 2:
        raise ValueError("Expected padded teacher token rows with BOS and targets")
    if (
        not mask[:, 0].all()
        or not (tokens[:, 0] == native._paligemma_tokenizer.bos_token_id).all()
    ):
        raise ValueError("Teacher sequence must start with valid action BOS")
    lengths = mask.sum(1)
    expected = (
        torch.arange(tokens.shape[1], device=tokens.device)[None] < lengths[:, None]
    )
    if not torch.equal(mask, expected) or (lengths < 2).any():
        raise ValueError("Teacher mask must be nonempty, contiguous right-padding")
    return [row[1 : int(n)].tolist() for row, n in zip(tokens, lengths, strict=True)]


@contextmanager
def bos_boundary(model, mode):
    if mode not in ("sampler", "sft"):
        raise ValueError(f"Unknown BOS attention mode: {mode}")
    if mode == "sampler":
        yield
        return
    original = model.embed_prefix_fast
    had_override = "embed_prefix_fast" in model.__dict__
    previous_override = model.__dict__.get("embed_prefix_fast")

    def changed(*args, **kwargs):
        if kwargs.get("fast_action_tokens") is not None:
            raise ValueError("Intervention applies only to the RL conditioning prefix")
        embeddings, pads, attention, images, actions = original(*args, **kwargs)
        attention = attention.clone()
        # The appended BOS still sees the prefix; earlier queries cannot see BOS.
        attention[:, :-1, -1] = False
        return embeddings, pads, attention, images, actions

    model.embed_prefix_fast = changed
    try:
        yield
    finally:
        if had_override:
            model.embed_prefix_fast = previous_override
        else:
            del model.embed_prefix_fast


def teacher_score_loss(native, batch, *, mode="sampler", temperature=1.0):
    import torch

    from art_embodied.policies.pi0_fast import _pi0_fast_token_logprobs

    if temperature <= 0:
        raise ValueError("Teacher scoring temperature must be positive")
    rows = teacher_rows(native, batch)
    with bos_boundary(native.model, mode), torch.enable_grad():
        scores = _pi0_fast_token_logprobs(
            native_policy=native,
            processed_batch=batch,
            token_rows=rows,
            temperature=temperature,
        )
    return -sum(row.sum() for row in scores) / sum(map(len, rows))


def measure_gradients(policy, batch, save_gradient):
    """Use the ART wrapper's parameter API, not torch.Module-only methods."""
    import time

    import torch

    from art_embodied.backends.action_token_gradients import _trainable_gradient_payload
    from examples.embodied.pi0_fast_repeat_gradient_audit import vector

    names = sorted(n for n, p in policy.named_parameters() if p.requires_grad)
    gradients, metrics = {}, {}
    for case in (
        "native_eval",
        "rl_sft_bos_t1",
        "rl_sampler_t1",
        "rl_sampler_t02",
        "native_train",
    ):
        for parameter in policy.parameters():
            parameter.grad = None
        policy.train(case == "native_train")
        tick = time.monotonic()
        with torch.enable_grad():
            if case.startswith("native"):
                loss, _ = policy.policy(batch)
            else:
                loss = teacher_score_loss(
                    policy.policy,
                    batch,
                    mode="sft" if case == "rl_sft_bos_t1" else "sampler",
                    temperature=0.2 if case == "rl_sampler_t02" else 1.0,
                )
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite teacher loss: {case}")
            loss.backward()
        payload = _trainable_gradient_payload(policy)
        if payload["missing_gradients"] or sorted(payload["gradients"]) != names:
            raise ValueError("Incomplete trainable gradient surface")
        gradients[case] = vector(payload)
        metrics[f"{case}_loss"] = float(loss.detach())
        metrics[f"{case}_seconds"] = time.monotonic() - tick
        save_gradient(case, payload)
    policy.eval()
    return gradients, metrics
