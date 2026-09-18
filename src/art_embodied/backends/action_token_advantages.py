"""Pure advantage and loss-weight helpers for action-token objectives.

This module deliberately has no policy, optimizer, or environment dependencies.
Keeping these transforms isolated makes their tensor semantics independently
testable and prevents the action-token backend from becoming the only place
where GRPO/GSPO contracts can be audited.

RLinf-compatible action-level behavior is reimplemented against RLinf
release/v0.1 at commit 9df6dc80dc729a6caccab92dd676004e51b1d3a2
(Apache-2.0, Copyright 2025 The RLinf Authors). The helper APIs, validation, and
weighting extensions are specific to ART-Embodied.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class ActionTokenExampleLike(Protocol):
    """Structural fields required by pure advantage and weighting transforms."""

    tokens: Sequence[int | str]
    logprobs: Sequence[float] | None
    reward: float
    metadata: dict[str, Any]


def _coerce_float_sequence(value: Any, *, field: str) -> list[float]:
    if value is None:
        raise ValueError(f"RLinf action-level advantages require {field}")
    if not isinstance(value, list | tuple):
        raise ValueError(f"{field} must be a list or tuple")
    return [float(item) for item in value]


def _coerce_bool_sequence(value: Any, *, field: str) -> list[bool]:
    if value is None:
        raise ValueError(f"RLinf action-level advantages require {field}")
    if not isinstance(value, list | tuple):
        raise ValueError(f"{field} must be a list or tuple")
    return [bool(item) for item in value]


def _example_has_token_advantages(example: ActionTokenExampleLike) -> bool:
    metadata = example.metadata if isinstance(example.metadata, dict) else {}
    values = metadata.get("token_advantages")
    return isinstance(values, list | tuple)


def _examples_have_token_advantages(examples: Sequence[ActionTokenExampleLike]) -> bool:
    return any(_example_has_token_advantages(example) for example in examples)


def _examples_have_prepared_scalar_advantages(
    examples: Sequence[ActionTokenExampleLike],
) -> bool:
    """Return whether globally normalized scalar advantages are already frozen."""

    return bool(examples) and all(
        bool(example.metadata.get("group_advantage_prepared", False))
        for example in examples
    )


def _example_token_advantage_tensor(
    example: ActionTokenExampleLike,
    *,
    fallback: Any,
    token_count: int,
    device: str,
    dtype: Any,
) -> Any:
    metadata = example.metadata if isinstance(example.metadata, dict) else {}
    values = metadata.get("token_advantages")
    if values is None:
        return fallback
    if not isinstance(values, list | tuple):
        raise ValueError("token_advantages must be a list or tuple")
    if len(values) != token_count:
        raise ValueError(
            f"token_advantages length {len(values)} does not match token logprob count {token_count}"
        )
    import torch

    return torch.as_tensor(
        [float(value) for value in values], dtype=dtype, device=device
    )


def _example_token_loss_mask_tensor(
    example: ActionTokenExampleLike,
    *,
    token_count: int,
    device: str,
    dtype: Any,
) -> Any:
    metadata = example.metadata if isinstance(example.metadata, dict) else {}
    values = metadata.get("token_loss_mask")
    if values is None:
        import torch

        return torch.ones(token_count, dtype=dtype, device=device)
    if not isinstance(values, list | tuple):
        raise ValueError("token_loss_mask must be a list or tuple")
    if len(values) != token_count:
        raise ValueError(
            f"token_loss_mask length {len(values)} does not match token logprob count {token_count}"
        )
    import torch

    return torch.as_tensor(
        [1.0 if bool(value) else 0.0 for value in values], dtype=dtype, device=device
    )


def _example_objective_token_count(example: ActionTokenExampleLike) -> int:
    metadata = example.metadata if isinstance(example.metadata, dict) else {}
    mask = metadata.get("token_loss_mask")
    if isinstance(mask, list | tuple):
        return int(sum(1 for value in mask if bool(value)))
    if example.logprobs is not None:
        return len(example.logprobs)
    return len(example.tokens)


def _example_loss_denominator_count(
    example: ActionTokenExampleLike, *, loss_aggregation: str
) -> int:
    """Return the denominator unit count for the selected loss aggregation.

    ``rlinf_token_mean`` keeps ART's historical interpretation of RLinf token
    mean: after any primitive-step mask has been expanded to action-token
    granularity, divide by valid action-token count.

    ``rlinf_chunk_mean`` mirrors RLinf's OpenVLA-OFT ``loss_agg_func:
    token-mean`` more literally.  RLinf passes ``policy_loss`` as
    ``[batch, chunks, action_dim]`` and ``loss_mask`` as ``[batch, chunks, 1]``;
    PyTorch broadcasts the mask in the numerator, but ``masked_mean`` divides by
    ``loss_mask.sum()``, i.e. valid primitive chunks rather than valid action
    tokens.  This is the parity path for action-token OpenVLA-OFT.

    ``rlinf_masked_mean_ratio`` follows RLinf's fixed-horizon
    ``masked_mean_ratio`` more closely: the numerator is reweighted by
    ``max_episode_steps / valid_steps`` and the denominator includes invalid
    tail slots.  ART stores chunks as separate examples, so the action-level
    attachment step allocates any missing invalid tail to the final observed
    example in the trajectory.
    """

    if loss_aggregation == "rlinf_masked_mean_ratio":
        return _example_rlinf_masked_mean_ratio_denominator_count(example)
    if loss_aggregation == "rlinf_chunk_mean":
        return _example_rlinf_chunk_mean_denominator_count(example)
    return _example_objective_token_count(example)


def _example_rlinf_masked_mean_ratio_scale(example: ActionTokenExampleLike) -> float:
    metadata = example.metadata if isinstance(example.metadata, dict) else {}
    valid_count = _positive_int_or_none(
        metadata.get("rlinf_action_level_trajectory_loss_mask_sum")
    )
    primitive_slots = _positive_int_or_none(
        metadata.get("rlinf_action_level_trajectory_primitive_slots")
    )
    if valid_count is not None and primitive_slots is not None:
        return float(primitive_slots) / float(valid_count)

    primitive_mask = metadata.get("rlinf_action_level_primitive_loss_mask")
    if isinstance(primitive_mask, list | tuple):
        valid_local = sum(1 for value in primitive_mask if bool(value))
        if valid_local > 0:
            return float(len(primitive_mask)) / float(valid_local)
    action_metadata = metadata.get("action_metadata")
    if isinstance(action_metadata, dict):
        chunk_mask = action_metadata.get("primitive_loss_mask")
        if isinstance(chunk_mask, list | tuple):
            valid_local = sum(1 for value in chunk_mask if bool(value))
            if valid_local > 0:
                return float(len(chunk_mask)) / float(valid_local)
    return 1.0


def _example_rlinf_chunk_mean_denominator_count(example: ActionTokenExampleLike) -> int:
    metadata = example.metadata if isinstance(example.metadata, dict) else {}
    primitive_mask = metadata.get("rlinf_action_level_primitive_loss_mask")
    if isinstance(primitive_mask, list | tuple):
        valid_local = sum(1 for value in primitive_mask if bool(value))
        if valid_local > 0:
            return int(valid_local)

    action_metadata = metadata.get("action_metadata")
    if isinstance(action_metadata, dict):
        chunk_mask = action_metadata.get("primitive_loss_mask")
        if isinstance(chunk_mask, list | tuple):
            valid_local = sum(1 for value in chunk_mask if bool(value))
            if valid_local > 0:
                return int(valid_local)
    valid_count = _positive_int_or_none(
        metadata.get("rlinf_action_level_trajectory_loss_mask_sum")
    )
    if valid_count is not None:
        return max(1, valid_count)
    return _example_objective_token_count(example)


def _example_rlinf_masked_mean_ratio_denominator_count(
    example: ActionTokenExampleLike,
) -> int:
    metadata = example.metadata if isinstance(example.metadata, dict) else {}
    token_count = (
        len(example.logprobs) if example.logprobs is not None else len(example.tokens)
    )
    action_dim = _positive_int_or_none(metadata.get("rlinf_action_level_action_dim"))
    extra_primitive_slots = int(
        metadata.get("rlinf_action_level_possible_extra_primitive_slots") or 0
    )
    if action_dim is not None and extra_primitive_slots > 0:
        return max(1, token_count + extra_primitive_slots * action_dim)

    mask = metadata.get("token_loss_mask")
    if isinstance(mask, list | tuple):
        return max(1, len(mask))
    return max(1, token_count)


def _example_advantage_sign_token_counts(
    example: ActionTokenExampleLike,
    *,
    fallback_advantage: float | None = None,
) -> dict[str, int]:
    metadata = example.metadata if isinstance(example.metadata, dict) else {}
    token_count = (
        len(example.logprobs) if example.logprobs is not None else len(example.tokens)
    )
    values = metadata.get("token_advantages")
    if isinstance(values, list | tuple):
        if len(values) != token_count:
            raise ValueError(
                f"token_advantages length {len(values)} does not match token count {token_count}"
            )
        advantages = [float(value) for value in values]
    else:
        value = float(
            fallback_advantage if fallback_advantage is not None else example.reward
        )
        advantages = [value for _ in range(token_count)]

    mask = metadata.get("token_loss_mask")
    if isinstance(mask, list | tuple):
        if len(mask) != token_count:
            raise ValueError(
                f"token_loss_mask length {len(mask)} does not match token count {token_count}"
            )
        keep = [bool(value) for value in mask]
    else:
        keep = [True for _ in range(token_count)]

    positive = sum(
        1
        for advantage, enabled in zip(advantages, keep, strict=True)
        if enabled and advantage > 1.0e-8
    )
    negative = sum(
        1
        for advantage, enabled in zip(advantages, keep, strict=True)
        if enabled and advantage < -1.0e-8
    )
    zero = sum(
        1
        for advantage, enabled in zip(advantages, keep, strict=True)
        if enabled and abs(advantage) <= 1.0e-8
    )
    return {"positive": int(positive), "negative": int(negative), "zero": int(zero)}


def _examples_advantage_sign_token_counts(
    examples: Sequence[ActionTokenExampleLike],
    *,
    advantages: Any,
) -> dict[str, int]:
    counts = {"positive": 0, "negative": 0, "zero": 0}
    raw_advantages = [
        float(value) for value in advantages.detach().cpu().reshape(-1).tolist()
    ]
    for example, fallback in zip(examples, raw_advantages, strict=True):
        item = _example_advantage_sign_token_counts(
            example, fallback_advantage=fallback
        )
        counts["positive"] += item["positive"]
        counts["negative"] += item["negative"]
        counts["zero"] += item["zero"]
    return counts


def _advantage_sign_balance_scale_tensor(
    token_advantage: Any,
    token_loss_mask: Any,
    *,
    positive_token_count: int | None,
    negative_token_count: int | None,
    denominator_token_count: int,
) -> Any:
    """Balance positive and negative advantage token buckets for diagnostics.

    This is intentionally not the RLinf parity objective. It tests whether a
    token-mean objective is drowning sparse positive-advantage action tokens
    under a much larger negative-advantage bucket.
    """

    import torch

    advantage = torch.as_tensor(token_advantage).detach().reshape(-1)
    mask = torch.as_tensor(token_loss_mask).detach().reshape(-1)
    if advantage.numel() == 1 and mask.numel() > 1:
        advantage = advantage.expand_as(mask)
    if advantage.shape != mask.shape:
        raise ValueError(
            "advantage_sign_balanced_token_mean requires token advantage and "
            f"loss mask shapes to match: advantage={tuple(advantage.shape)}, "
            f"mask={tuple(mask.shape)}"
        )
    positive_count = int(positive_token_count or 0)
    negative_count = int(negative_token_count or 0)
    if positive_count <= 0 or negative_count <= 0:
        return torch.ones_like(mask)

    denominator = float(max(1, int(denominator_token_count)))
    pos_scale = denominator / (2.0 * float(positive_count))
    neg_scale = denominator / (2.0 * float(negative_count))
    scale = torch.ones_like(mask)
    scale = torch.where(
        (advantage > 1.0e-8) & (mask > 0), torch.full_like(scale, pos_scale), scale
    )
    scale = torch.where(
        (advantage < -1.0e-8) & (mask > 0), torch.full_like(scale, neg_scale), scale
    )
    return scale


def _positive_int_or_none(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _has_policy_gradient_signal(
    examples: Sequence[ActionTokenExampleLike],
    advantages: Any,
    *,
    device: str,
) -> bool:
    if _examples_have_token_advantages(examples):
        import torch

        values: list[float] = []
        for example in examples:
            metadata = example.metadata if isinstance(example.metadata, dict) else {}
            token_advantages = metadata.get("token_advantages")
            token_loss_mask = metadata.get("token_loss_mask")
            if not isinstance(token_advantages, list | tuple):
                continue
            if token_loss_mask is None:
                token_loss_mask = [True for _ in token_advantages]
            if not isinstance(token_loss_mask, list | tuple):
                raise ValueError("token_loss_mask must be a list or tuple")
            if len(token_advantages) != len(token_loss_mask):
                raise ValueError(
                    "token_advantages and token_loss_mask must have the same length"
                )
            values.extend(
                float(advantage)
                for advantage, keep in zip(
                    token_advantages, token_loss_mask, strict=True
                )
                if bool(keep)
            )
        if not values:
            return False
        tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
        return bool(torch.any(torch.abs(tensor) > 1.0e-8).item())
    return bool((abs(advantages) > 1e-8).any().item())


def _example_advantages(
    examples: list[ActionTokenExampleLike],
    *,
    normalize: bool,
    scope: str,
    std_unbiased: bool,
    eps: float,
    device: str,
) -> Any:
    import torch

    values = [
        float(example.metadata.get("group_advantage", example.reward))
        for example in examples
    ]
    advantages = torch.as_tensor(values, dtype=torch.float32, device=device)
    if not normalize or advantages.numel() <= 1:
        return advantages
    trajectory_units = _trajectory_advantage_units(examples)
    if scope == "group":
        normalized = advantages.clone()
        group_indices = [example.metadata.get("group_index") for example in examples]
        for group_index in sorted(
            {index for index in group_indices if index is not None}
        ):
            positions = [
                pos for pos, index in enumerate(group_indices) if index == group_index
            ]
            units = _units_for_positions(trajectory_units, positions)
            representative_positions = [unit[0] for unit in units]
            if len(representative_positions) <= 1:
                normalized[positions] = 0.0
                continue
            group_values = advantages[representative_positions]
            std = group_values.std(
                unbiased=std_unbiased and len(representative_positions) > 1
            )
            if float(std.detach().cpu().item()) > eps:
                unit_values = (group_values - group_values.mean()) / (std + eps)
                for unit, unit_value in zip(units, unit_values, strict=True):
                    normalized[unit] = unit_value
            else:
                normalized[positions] = 0.0
        return normalized
    if scope != "global":
        raise ValueError("advantage normalization scope must be 'global' or 'group'")
    representative_positions = [unit[0] for unit in trajectory_units]
    representative_values = advantages[representative_positions]
    if representative_values.numel() > 1:
        std = representative_values.std(unbiased=std_unbiased)
        if float(std.detach().cpu().item()) > eps:
            normalized_values = (
                representative_values - representative_values.mean()
            ) / (std + eps)
            advantages = advantages.clone()
            for unit, unit_value in zip(
                trajectory_units, normalized_values, strict=True
            ):
                advantages[unit] = unit_value
    return advantages


def _trajectory_advantage_units(
    examples: Sequence[ActionTokenExampleLike],
) -> list[list[int]]:
    """Return equal-weight trajectory units for terminal-reward action rows."""

    units: dict[tuple[Any, Any], list[int]] = {}
    for position, example in enumerate(examples):
        metadata = example.metadata if isinstance(example.metadata, dict) else {}
        if metadata.get("action_reward_source") != "trajectory_reward":
            return [[index] for index in range(len(examples))]
        group_index = metadata.get("group_index")
        trajectory_index = metadata.get("trajectory_index_in_group")
        if group_index is None or trajectory_index is None:
            return [[index] for index in range(len(examples))]
        units.setdefault((group_index, trajectory_index), []).append(position)

    for positions in units.values():
        unit_values = {
            float(
                examples[position].metadata.get(
                    "group_advantage", examples[position].reward
                )
            )
            for position in positions
        }
        if len(unit_values) != 1:
            raise ValueError(
                "Trajectory-reward action rows must share one group advantage"
            )
    return list(units.values())


def _units_for_positions(
    units: Sequence[Sequence[int]], positions: Sequence[int]
) -> list[list[int]]:
    selected = set(positions)
    return [list(unit) for unit in units if unit and unit[0] in selected]
