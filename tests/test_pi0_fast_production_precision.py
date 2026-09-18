from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")

from art_embodied.backends.loss_scaling import (
    backward_policy_loss,
    unscale_policy_gradients,
)
from art_embodied.policies.pi0_fast_precision import (
    _fp16_product_with_scale,
    install_fp16_residual,
)
from examples.embodied.pi0_fast_precision_probe_context import (
    fp16_mlp_output_fp32,
    frozen_bf16_as_fp16,
)


class ToyPolicy(torch.nn.Linear):
    family = "pi0_fast"
    model_compute_dtype = "fp16_residual"
    training_loss_scale = 128.0


@pytest.mark.parametrize("microbatches", [1, 4])
def test_scale_unscale_matches_unscaled_update_and_gradient_payload(microbatches):
    from art_embodied.backends.action_token_gradients import _trainable_gradient_payload

    torch.manual_seed(614)
    policy = ToyPolicy(3, 2)
    reference = deepcopy(policy)
    for _ in range(microbatches):
        x = torch.randn(2, 3)
        backward_policy_loss(policy, policy(x).square().sum() / microbatches)
        (reference(x).square().sum() / microbatches).backward()
    metrics = unscale_policy_gradients(policy)
    assert metrics == {"optimization/loss_scale": 128.0}
    for a, b in zip(policy.parameters(), reference.parameters(), strict=True):
        assert torch.equal(a.grad, b.grad)
    actual = _trainable_gradient_payload(policy)
    expected = _trainable_gradient_payload(reference)
    for key in actual["gradients"]:
        assert torch.equal(actual["gradients"][key], expected["gradients"][key])
    for model in (policy, reference):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    for a, b in zip(policy.parameters(), reference.parameters(), strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("family", ["openvla_oft", "pi0", "pi05", "gr00t_n1d7"])
def test_other_policy_families_are_bit_identical(family):
    policy = ToyPolicy(3, 2)
    policy.family = family
    policy.training_loss_scale = float("nan")  # must not even inspect this
    reference = deepcopy(policy)
    x = torch.ones(2, 3)
    backward_policy_loss(policy, policy(x).sum())
    reference(x).sum().backward()
    assert unscale_policy_gradients(policy) == {}
    for a, b in zip(policy.parameters(), reference.parameters(), strict=True):
        assert torch.equal(a.grad, b.grad)


def test_nonfinite_gradient_aborts_before_optimizer_update():
    policy = ToyPolicy(3, 2)
    original = deepcopy(policy.state_dict())
    backward_policy_loss(policy, policy(torch.ones(2, 3)).sum() * float("inf"))
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        unscale_policy_gradients(policy)
    for key, value in policy.state_dict().items():
        assert torch.equal(value, original[key])


class TupleNorm(torch.nn.Module):
    def forward(self, x):
        return x / 2, None


class Language(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.down_proj = torch.nn.Linear(
            4, 4, bias=False, device=device, dtype=torch.bfloat16
        ).requires_grad_(False)
        layer.mlp.gate_proj = torch.nn.Identity()
        layer.mlp.up_proj = torch.nn.Identity()
        layer.mlp.act_fn = torch.nn.Identity()
        layer.input_layernorm = TupleNorm()
        layer.post_attention_layernorm = TupleNorm()
        self.layers = torch.nn.ModuleList([layer])
        self.norm = TupleNorm()
        self.adapter = torch.nn.Parameter(torch.ones(4, device=device))

    def forward(self, *, inputs_embeds, attention_mask=None):
        self.last_mask = attention_mask
        x, _ = self.layers[0].input_layernorm(inputs_embeds.float())
        return self.layers[0].mlp.down_proj(x) * self.adapter


def test_range_scaling_is_bit_identical_for_normal_products_and_gradients():
    torch.manual_seed(912)
    left = torch.randn(8, 32, dtype=torch.float16).requires_grad_()
    right = torch.randn_like(left).requires_grad_()
    reference_left = left.detach().clone().requires_grad_()
    reference_right = right.detach().clone().requires_grad_()
    product, scale = _fp16_product_with_scale(left, right)
    reference = reference_left * reference_right
    assert torch.equal(scale, torch.ones_like(scale))
    assert torch.equal(product, reference)
    grad = torch.randn_like(product)
    (product.float() * scale).backward(grad.float())
    reference.backward(grad)
    assert torch.equal(left.grad, reference_left.grad)
    assert torch.equal(right.grad, reference_right.grad)


def test_range_scaling_preserves_large_products_without_clipping():
    left = torch.tensor(
        [[300.0, -300.0, 0.0], [65504.0, -65504.0, 1.0]], dtype=torch.float16
    )
    right = torch.tensor(
        [[300.0, 300.0, 1.0], [65504.0, 65504.0, 1.0]], dtype=torch.float16
    )
    product, scale = _fp16_product_with_scale(left, right)
    assert torch.isfinite(product).all()
    assert torch.equal(torch.log2(scale), torch.log2(scale).round())
    assert torch.all(scale >= 1)
    expected = left.double() * right.double()
    actual = product.double() * scale.double()
    torch.testing.assert_close(actual, expected, rtol=5e-4, atol=0.01)


def test_range_scaling_does_not_hide_nonfinite_inputs():
    product, _ = _fp16_product_with_scale(
        torch.tensor([[float("nan"), float("inf")]], dtype=torch.float16),
        torch.ones(1, 2, dtype=torch.float16),
    )
    assert not torch.isfinite(product).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA projection required")
def test_scaled_mlp_forward_and_backward_match_fp32_reference_on_cuda():
    model = Language("cuda")
    install_fp16_residual(model, model)
    x = torch.tensor(
        [[300.0, -300.0, 250.0, -250.0]],
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )
    actual = model.layers[0].mlp(x)
    reference_x = x.detach().float().requires_grad_()
    reference = (reference_x.square()) @ model.layers[0].mlp.down_proj.weight.float().T
    grad = torch.full_like(actual, 0.0001)
    actual.backward(grad)
    reference.backward(grad)
    assert torch.isfinite(actual).all() and torch.isfinite(x.grad).all()
    torch.testing.assert_close(actual, reference, rtol=0.002, atol=10.0)
    torch.testing.assert_close(x.grad.float(), reference_x.grad, rtol=0.005, atol=0.001)


def test_install_preserves_keys_ties_and_fp32_trainables():
    model = Language()
    model.tied = model.layers[0].mlp.down_proj
    parameters = dict(model.named_parameters())
    keys = tuple(model.state_dict())
    report = install_fp16_residual(model, model)
    assert tuple(model.state_dict()) == keys
    assert report["fp32_down_projections"] == 1
    assert model.tied.weight.dtype == torch.float16
    assert model.adapter.dtype == torch.float32
    for key, value in model.named_parameters():
        assert parameters[key] is value
    with pytest.raises(ValueError, match="already installed"):
        install_fp16_residual(model, model)


@pytest.mark.parametrize("violation", ["trainable", "overflow"])
def test_install_rejects_invalid_weights_before_any_mutation(violation):
    model = Language()
    if violation == "trainable":
        model.layers[0].mlp.down_proj.weight.requires_grad_(True)
    else:
        model.bad = torch.nn.Parameter(
            torch.tensor([1e10], dtype=torch.bfloat16), requires_grad=False
        )
    before = {n: p.data_ptr() for n, p in model.named_parameters()}
    with pytest.raises(ValueError):
        install_fp16_residual(model, model)
    assert before == {n: p.data_ptr() for n, p in model.named_parameters()}
    assert not getattr(model, "_art_fp16_residual", False)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Actual CUDA out_dtype kernel"
)
def test_production_matches_diagnostic_forward_backward_and_state_reload():
    torch.manual_seed(614)
    model = Language("cuda")
    reference = deepcopy(model)
    install_fp16_residual(model, model)
    model.to("cpu").to("cuda")
    restored = Language("cuda")
    install_fp16_residual(restored, restored)
    restored.load_state_dict(model.state_dict(), strict=True)
    x = torch.randn(2, 3, 4, device="cuda", requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)
    mask = torch.tensor([0.0, -2.38e38], device="cuda")
    y = model(inputs_embeds=x, attention_mask=mask)
    (y.sum() * 128).backward()
    with frozen_bf16_as_fp16(reference, reference, mask_mode="inf"):
        with fp16_mlp_output_fp32(reference):
            ref_y = reference(inputs_embeds=ref_x, attention_mask=mask)
            (ref_y.sum() * 128).backward()
    assert torch.equal(y, ref_y)
    assert torch.equal(x.grad, ref_x.grad)
    assert torch.equal(model.adapter.grad, reference.adapter.grad)
    assert torch.isneginf(model.last_mask[1])
    assert torch.equal(y, restored(inputs_embeds=x, attention_mask=mask))
