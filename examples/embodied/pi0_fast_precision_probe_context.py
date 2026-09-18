"""Reversible low-precision interventions for isolated FAST diagnostics only."""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch


@contextmanager
def frozen_bf16_as_fp16(model, language, *, mask_mode="finite"):
    """Keep FP32 vision, norms, trainable LoRA, and tied Parameter identities intact."""
    import torch

    if mask_mode not in ("finite", "inf"):
        raise ValueError("Unknown diagnostic attention mask mode")
    originals = []
    report = {"converted_elements": 0, "changed_elements": 0, "max_cast_error": 0.0}
    forward = language.forward

    def cast_language_input(*args, **kwargs):
        # Upstream FAST explicitly casts only for BF16, not for FP16.
        kwargs["inputs_embeds"] = kwargs["inputs_embeds"].to(torch.float16)
        mask = kwargs.get("attention_mask")
        if isinstance(mask, torch.Tensor) and mask.is_floating_point():
            if not bool(((mask == 0) | (mask <= torch.finfo(torch.float16).min)).all()):
                raise ValueError("Expected a pure allowed/blocked attention mask")
            # Preserve a finite negative sentinel, including fully masked pads.
            converted_mask = mask.to(torch.float16)
            if mask_mode == "finite":
                converted_mask = converted_mask.clamp_min(
                    torch.finfo(torch.float16).min
                )
            kwargs["attention_mask"] = converted_mask
        return forward(*args, **kwargs)

    try:
        for name, parameter in model.named_parameters():
            if parameter.dtype != torch.bfloat16:
                continue
            if parameter.requires_grad:
                raise ValueError(f"Expected frozen BF16 weights, got trainable {name}")
            converted = parameter.detach().to(torch.float16)
            if not bool(torch.isfinite(converted).all()):
                raise ValueError(f"FP16 parameter range overflow: {name}")
            delta = (converted.float() - parameter.detach().float()).abs()
            report["converted_elements"] += parameter.numel()
            report["changed_elements"] += int((delta != 0).sum())
            report["max_cast_error"] = max(report["max_cast_error"], delta.max().item())
            originals.append((parameter, parameter.data))
            parameter.data = converted
        if not originals:
            raise ValueError("No frozen BF16 weights to test")
        with patch.object(language, "forward", cast_language_input):
            yield report
    finally:
        for parameter, original in originals:
            parameter.data = original


@contextmanager
def low_precision_mlp_output_fp32(language, *, dtype):
    """Diagnostic: low-precision GEMM operands, FP32 MLP output and residual."""
    import torch

    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Expected a supported low-precision operand dtype")

    class FrozenProjection(torch.autograd.Function):
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
            # dX = dY W for frozen W. Match low-precision operands, retain
            # FP32 GEMM output before returning X's low-precision gradient.
            grad_input = torch.mm(
                grad_output.reshape(-1, grad_output.shape[-1]).to(weight.dtype),
                weight,
                out_dtype=torch.float32,
            )
            return grad_input.to(weight.dtype).reshape(ctx.input_shape), None

    def projection(module):
        if (
            not isinstance(module, torch.nn.Linear)
            or module.weight.requires_grad
            or module.bias is not None
        ):
            raise ValueError(
                "Only frozen, unbiased native MLP down projections are supported"
            )
        if module.weight.dtype != dtype:
            raise ValueError("Unexpected down-projection operand dtype")

        def forward(x):
            if x.dtype != dtype:
                raise ValueError("Unexpected MLP activation dtype")
            return FrozenProjection.apply(x, module.weight)

        return forward

    def normalized_output(forward):
        def apply(*args, **kwargs):
            result = forward(*args, **kwargs)
            if isinstance(result, tuple):
                return (result[0].to(dtype), *result[1:])
            return result.to(dtype)

        return apply

    with ExitStack() as stack:
        for layer in language.layers:
            module = layer.mlp.down_proj
            stack.enter_context(patch.object(module, "forward", projection(module)))
        for name, module in language.named_modules():
            if name == "norm" or name.endswith(
                ("input_layernorm", "post_attention_layernorm")
            ):
                stack.enter_context(
                    patch.object(module, "forward", normalized_output(module.forward))
                )
        yield


@contextmanager
def fp16_mlp_output_fp32(language):
    """FP16-specific entry point retained for existing diagnostic scripts."""
    import torch

    with low_precision_mlp_output_fp32(language, dtype=torch.float16):
        yield
