"""Process-local FAST performance experiments; never enabled by importing."""

from contextlib import contextmanager


@contextmanager
def split_bf16_frozen_linears(policy):
    """Experimental three-product approximation with FP32 outputs/residuals.

    Write X=Xh+Xl and W=Wh+Wl with BF16 high/low parts. Compute XhWh,
    XhWl, XlWh, accumulating FP32 and omitting only XlWl. Trainable LoRA
    matrices are untouched. Neither equivalence nor speed is assumed.
    """
    import torch

    if policy.family != "pi0_fast":
        raise ValueError("This experiment is restricted to FAST")

    def product(x, high, low):
        xh = x.to(torch.bfloat16)
        xl = (x - xh.float()).to(torch.bfloat16)
        value = torch.mm(xh, high, out_dtype=torch.float32)
        value.add_(torch.mm(xh, low, out_dtype=torch.float32))
        value.add_(torch.mm(xl, high, out_dtype=torch.float32))
        return value

    class FrozenLinear(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, high, low, bias):
            ctx.save_for_backward(high, low)
            ctx.shape = x.shape
            value = product(x.reshape(-1, x.shape[-1]), high.t(), low.t())
            if bias is not None:
                value.add_(bias)
            return value.reshape(*x.shape[:-1], high.shape[0])

        @staticmethod
        def backward(ctx, grad):
            high, low = ctx.saved_tensors
            dx = product(grad.reshape(-1, grad.shape[-1]), high, low)
            return dx.reshape(ctx.shape), None, None, None

    originals = []
    try:
        for module in policy.model.modules():
            if not isinstance(module, torch.nn.Linear) or module.weight.requires_grad:
                continue
            if module.weight.dtype != torch.float32 or (
                module.bias is not None and module.bias.requires_grad
            ):
                continue
            high = module.weight.detach().to(torch.bfloat16)
            low = (module.weight.detach() - high.float()).to(torch.bfloat16)
            original = module.forward

            def forward(x, high=high, low=low, bias=module.bias, original=original):
                if x.dtype != torch.float32:
                    return original(x)
                return FrozenLinear.apply(x, high, low, bias)

            originals.append((module, original))
            module.forward = forward
        yield len(originals)
    finally:
        for module, original in originals:
            module.forward = original


@contextmanager
def compact_decode_rows(policy):
    """Skip completed rows in transformer decoding, preserving full-batch RNG.

    Prefix processing, sampler output layout and categorical draw shape stay
    unchanged. Completed rows get dummy hidden states; their post-end tokens
    are already discarded by the native action/likelihood contract. This is
    experimental until real-device output and likelihood comparisons pass.
    """
    import torch

    if policy.family != "pi0_fast" or not policy.use_kv_cache:
        raise ValueError("Compaction requires the native FAST KV sampler")
    transformer = policy.model.paligemma_with_expert
    original_forward = transformer.forward
    original_embed = transformer.embed_language_tokens
    end = int(policy.policy._paligemma_tokenizer.convert_tokens_to_ids("|"))
    state = {}
    counters = {"decode_rows_requested": 0, "decode_rows_computed": 0}

    def embed(tokens):
        if state and tokens.ndim == 2 and tokens.shape[1] == 1:
            state["alive"] &= tokens[:, 0] != end
        return original_embed(tokens)

    def forward(**kwargs):
        cache = kwargs.get("past_key_values")
        if cache is None:
            result = original_forward(**kwargs)
            prefix = kwargs["inputs_embeds"][0]
            size = prefix.shape[0]
            state.update(
                alive=torch.ones(size, dtype=torch.bool, device=prefix.device),
                indices=torch.arange(size, device=prefix.device),
            )
            return result
        alive = state["alive"]
        previous = state["indices"]
        keep = torch.nonzero(alive[previous], as_tuple=False).flatten()
        indices = previous[keep]
        if not indices.numel():
            raise RuntimeError("Sampler must stop before forwarding an empty batch")
        if keep.numel() != previous.numel():
            if not hasattr(cache, "batch_select_indices"):
                raise TypeError("Unsupported KV cache for row compaction")
            cache.batch_select_indices(keep)
        state["indices"] = indices
        counters["decode_rows_requested"] += int(alive.numel())
        counters["decode_rows_computed"] += int(indices.numel())
        selected = dict(kwargs)
        for name in ("attention_mask", "position_ids"):
            selected[name] = kwargs[name].index_select(0, indices)
        selected["inputs_embeds"] = [
            kwargs["inputs_embeds"][0].index_select(0, indices),
            None,
        ]
        (hidden, other), cache = original_forward(**selected)
        expanded = hidden.new_zeros((alive.numel(), *hidden.shape[1:]))
        expanded.index_copy_(0, indices, hidden)
        return (expanded, other), cache

    transformer.forward = forward
    transformer.embed_language_tokens = embed
    try:
        yield counters
    finally:
        transformer.forward = original_forward
        transformer.embed_language_tokens = original_embed
