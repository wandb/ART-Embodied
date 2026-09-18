"""Opt-in FAST FP16 operands with FP32 MLP outputs and residual streams.

This is a distinct numerical policy, not a transparent optimization of BF16.
Install after freezing the backbone and attaching FP32 LoRA, on both actors
and learners. Checkpoints must retain the precision profile.
"""

from types import MethodType

import torch


class _FrozenProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(weight)
        ctx.input_shape = x.shape
        return torch.mm(
            x.reshape(-1, x.shape[-1]), weight.T, out_dtype=torch.float32
        ).reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        (weight,) = ctx.saved_tensors
        # Frozen W: dX = dY W. The pinned torch.mm(out_dtype=...) has no
        # native backward. Keep low-precision operands and FP32 accumulation.
        grad_input = torch.mm(
            grad_output.reshape(-1, grad_output.shape[-1]).to(weight.dtype),
            weight,
            out_dtype=torch.float32,
        )
        return grad_input.to(weight.dtype).reshape(ctx.input_shape), None


def _projection_forward(module, x):
    if x.dtype != torch.float16 or module.weight.dtype != torch.float16:
        raise ValueError("FAST FP16 residual projection operand dtype changed")
    if module.weight.requires_grad:
        raise ValueError("FAST FP16 residual projection must remain frozen")
    return _FrozenProjection.apply(x, module.weight)


def _fp16_product_with_scale(left, right):
    if left.dtype != torch.float16 or right.dtype != torch.float16:
        raise ValueError("FAST gated-product operands must be FP16")
    product = left.float() * right.float()
    # For the usual scale=1 path this rounds exactly as an FP16 product.
    # Detach the power-of-two range selection: (x/s)W*s has derivative W.
    with torch.no_grad():
        bound = product.abs().amax(dim=-1, keepdim=True)
        scale = torch.exp2(torch.ceil(torch.log2(bound.clamp_min(65504.0) / 65504.0)))
    return (product / scale).to(torch.float16), scale


def _mlp_forward(module, x):
    product, scale = _fp16_product_with_scale(
        module.act_fn(module.gate_proj(x)), module.up_proj(x)
    )
    return module.down_proj(product) * scale


def _norm_forward(module, *args, **kwargs):
    result = module._art_original_forward(*args, **kwargs)
    if isinstance(result, tuple):
        return (result[0].to(torch.float16), *result[1:])
    return result.to(torch.float16)


def _language_forward(module, *args, **kwargs):
    # LeRobot's native boundary casts embeddings only for BF16 weights.
    kwargs["inputs_embeds"] = kwargs["inputs_embeds"].to(torch.float16)
    mask = kwargs.get("attention_mask")
    if isinstance(mask, torch.Tensor) and mask.is_floating_point():
        # Native FAST constructs zero/negative-sentinel additive masks. Casting
        # retains their support (-inf for blocked entries), without host sync.
        kwargs["attention_mask"] = mask.to(torch.float16)
    return module._art_original_forward(*args, **kwargs)


def install_fp16_residual(model, language):
    """Preserve parameter identities, module paths, and state-dict keys."""
    if getattr(language, "_art_fp16_residual", False):
        raise ValueError("FAST FP16 residual profile is already installed")
    mlps = [layer.mlp for layer in language.layers]
    projections = [mlp.down_proj for mlp in mlps]
    norms = [
        module
        for name, module in language.named_modules()
        if name == "norm"
        or name.endswith(("input_layernorm", "post_attention_layernorm"))
    ]
    if not projections or not norms:
        raise ValueError("Unsupported FAST language model structure")
    if any(
        not all(hasattr(mlp, key) for key in ("gate_proj", "up_proj", "act_fn"))
        for mlp in mlps
    ):
        raise ValueError("Require native FAST gated MLPs")
    for module in projections:
        if (
            not isinstance(module, torch.nn.Linear)
            or module.bias is not None
            or module.weight.requires_grad
            or module.weight.dtype != torch.bfloat16
        ):
            raise ValueError("Require frozen unbiased BF16 native down projections")
    converted = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            raise ValueError(f"FP32 trainable parameters required: {name}")
        if parameter.dtype == torch.bfloat16:
            if not bool(torch.isfinite(parameter).all()) or bool(
                (parameter.float().abs() > torch.finfo(torch.float16).max).any()
            ):
                raise ValueError(f"FAST FP16 parameter range overflow: {name}")
            converted.append(parameter)
    # Complete structure/range validation before changing any module.
    for parameter in converted:
        parameter.data = parameter.data.to(torch.float16)
    for module in projections:
        module.forward = MethodType(_projection_forward, module)
    for module in mlps:
        module.forward = MethodType(_mlp_forward, module)
    for module in norms:
        module._art_original_forward = module.forward
        module.forward = MethodType(_norm_forward, module)
    language._art_original_forward = language.forward
    language.forward = MethodType(_language_forward, language)
    language._art_fp16_residual = True
    return {
        "profile": "fp16_residual",
        "frozen_converted_elements": sum(p.numel() for p in converted),
        "fp32_down_projections": len(projections),
        "fp16_normalized_outputs": len(norms),
        "range_scaled_gated_products": len(mlps),
    }
