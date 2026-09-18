from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from art_embodied.backends.flow_sde_grpo import (  # noqa: E402
    chunk_logprobs,
    flow_sde_grpo_loss,
    flow_sde_reference_kl_loss,
    group_relative_advantages,
    primitive_normalized_chunk_abs_delta,
)


def test_group_relative_advantages_match_rlinf_grpo_equation() -> None:
    rewards = torch.tensor([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]])

    actual = group_relative_advantages(rewards, std_unbiased=True)
    expected = (rewards - rewards.mean(dim=-1, keepdim=True)) / (
        rewards.std(dim=-1, keepdim=True) + 1.0e-6
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_chunk_logprobs_sum_only_executed_action_surface() -> None:
    values = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    mask = torch.tensor([[True, True, False], [True, False, False]])

    actual = chunk_logprobs(values, action_mask=mask)

    torch.testing.assert_close(actual, torch.tensor([6.0, 13.0]))


def test_alignment_delta_is_normalized_by_executed_primitive_horizon() -> None:
    old = torch.zeros(2, 4, 3)
    current = torch.full_like(old, 0.1)
    mask = torch.tensor([[True, True, False, False], [True, False, False, False]])

    actual = primitive_normalized_chunk_abs_delta(current, old, action_mask=mask)

    # Each primitive is the complete three-dimensional action vector. The
    # result therefore remains 0.3 for either a one- or two-step prefix.
    torch.testing.assert_close(actual, torch.tensor([0.3, 0.3]))


def test_flow_sde_loss_matches_rlinf_v01_chunk_level_reference() -> None:
    old = torch.tensor(
        [
            [[-1.0, -0.4], [-0.5, -0.2]],
            [[-0.8, -0.3], [-0.6, -0.1]],
            [[-0.7, -0.2], [-0.4, -0.3]],
        ],
        dtype=torch.float32,
    )
    current = old.clone().requires_grad_(True)
    current.data[0] += 0.03
    current.data[1] -= 0.04
    advantages = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float32)
    loss_mask = torch.tensor([True, True, False])

    actual, metrics = flow_sde_grpo_loss(
        current,
        old,
        advantages,
        loss_mask=loss_mask,
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.2,
        clip_ratio_c=3.0,
    )

    current_chunk = current.sum(dim=(1, 2))
    old_chunk = old.sum(dim=(1, 2))
    ratio = torch.where(
        loss_mask,
        torch.exp(current_chunk - old_chunk),
        torch.zeros_like(current_chunk),
    )
    clipped = torch.clamp(ratio, 0.8, 1.2)
    loss1 = -advantages * ratio
    loss2 = -advantages * clipped
    expected_rows = torch.minimum(
        torch.maximum(loss1, loss2),
        torch.sign(advantages) * 3.0 * advantages,
    )
    expected = expected_rows[loss_mask].mean()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(metrics.ratio_mean, ratio[loss_mask].mean())
    actual.backward()
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()


def test_flow_sde_loss_excludes_unexecuted_chunk_tail() -> None:
    old = torch.zeros(1, 3, 1)
    current = torch.tensor([[[0.1], [50.0], [-50.0]]], requires_grad=True)
    action_mask = torch.tensor([[True, False, False]])

    loss, metrics = flow_sde_grpo_loss(
        current,
        old,
        torch.ones(1),
        action_mask=action_mask,
        clip_epsilon_low=100.0,
        clip_epsilon_high=100.0,
        clip_ratio_c=None,
    )

    torch.testing.assert_close(metrics.approximate_kl, torch.tensor(-0.1))
    torch.testing.assert_close(metrics.approximate_kl_per_primitive, torch.tensor(-0.1))
    loss.backward()
    assert current.grad is not None
    assert current.grad[0, 0, 0] != 0.0
    torch.testing.assert_close(current.grad[0, 1:], torch.zeros(2, 1))


def test_flow_sde_reports_horizon_normalized_kl_without_changing_joint_ratio() -> None:
    old = torch.zeros(1, 2, 1)
    current = torch.full((1, 2, 1), -0.1, requires_grad=True)

    _loss, metrics = flow_sde_grpo_loss(current, old, torch.ones(1))

    torch.testing.assert_close(metrics.approximate_kl, torch.tensor(0.2))
    torch.testing.assert_close(metrics.approximate_kl_per_primitive, torch.tensor(0.1))


def test_flow_sde_loss_matches_rlinf_masked_mean_ratio() -> None:
    old = torch.zeros(3, 1, 1)
    current = old.clone().requires_grad_(True)
    advantages = torch.tensor([1.0, -1.0, 0.5])
    loss_mask = torch.tensor([True, True, False])
    # RLinf computes (loss / (trajectory_steps / max_steps) * mask).mean()
    # over the fixed, padding-inclusive actor global batch.
    row_weights = torch.tensor([240.0 / 120.0, 240.0 / 60.0, 1.0])

    actual, _metrics = flow_sde_grpo_loss(
        current,
        old,
        advantages,
        loss_mask=loss_mask,
        row_weights=row_weights,
        loss_denominator=8,
        clip_ratio_c=3.0,
    )

    expected = ((-1.0 * 2.0) + (1.0 * 4.0)) / 8.0
    torch.testing.assert_close(actual, torch.tensor(expected), rtol=0, atol=0)
    actual.backward()
    torch.testing.assert_close(
        current.grad[:, 0, 0],
        torch.tensor([-2.0 / 8.0, 4.0 / 8.0, 0.0]),
        rtol=0,
        atol=0,
    )


def test_flow_sde_loss_rejects_elementwise_shape_drift() -> None:
    current = torch.zeros(2, 3, 4)
    old = torch.zeros(2, 3, 5)

    with pytest.raises(ValueError, match="identical shapes"):
        flow_sde_grpo_loss(current, old, torch.ones(2))


def test_reference_kl_is_nonnegative_and_pulls_current_toward_sft() -> None:
    current = torch.full((1, 2, 1), 0.5, requires_grad=True)
    reference = torch.zeros_like(current, requires_grad=True)

    loss, metrics = flow_sde_reference_kl_loss(current, reference)

    assert loss.item() > 0.0
    torch.testing.assert_close(loss, metrics.mean_per_primitive)
    loss.backward()
    assert current.grad is not None
    assert torch.all(current.grad > 0.0)
    assert reference.grad is None


def test_reference_kl_matches_grpo_mask_weight_and_denominator_contract() -> None:
    current = torch.tensor([[[0.5], [50.0]], [[-0.5], [-50.0]]])
    reference = torch.zeros_like(current)
    action_mask = torch.tensor([[True, False], [True, False]])
    loss_mask = torch.tensor([True, False])
    row_weights = torch.tensor([4.0, 8.0])

    loss, metrics = flow_sde_reference_kl_loss(
        current,
        reference,
        loss_mask=loss_mask,
        action_mask=action_mask,
        row_weights=row_weights,
        loss_denominator=8,
    )

    expected_row = torch.expm1(torch.tensor(-0.5)) + 0.5
    torch.testing.assert_close(loss, expected_row * 4.0 / 8.0)
    torch.testing.assert_close(metrics.mean_per_primitive, expected_row)


def test_reference_kl_is_zero_at_the_sft_policy() -> None:
    current = torch.randn(3, 2, 4, requires_grad=True)

    loss, metrics = flow_sde_reference_kl_loss(current, current.detach())

    torch.testing.assert_close(loss, torch.tensor(0.0))
    torch.testing.assert_close(metrics.mean_per_primitive, torch.tensor(0.0))
