"""Replayable model inputs and rollout evidence for Flow-SDE policies.

The probability backend operates on this model-family-neutral record. Policy
bridges remain responsible for preparing native model inputs and evaluating the
velocity field, while the backend can batch PI, SmolVLA, and future flow-policy
examples without model-specific branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch

from .flow_sde import FlowSDETransitionRecord


@dataclass(frozen=True, slots=True)
class FlowModelInputs:
    """Replayable, already-processed conditioning for one flow model call."""

    images: tuple[torch.Tensor, ...]
    image_masks: tuple[torch.Tensor, ...]
    language_tokens: torch.Tensor
    language_masks: torch.Tensor
    state: torch.Tensor | None

    @property
    def batch_size(self) -> int:
        return int(self.language_tokens.shape[0])

    def select(self, index: int) -> "FlowModelInputs":
        row = slice(index, index + 1)
        return FlowModelInputs(
            images=tuple(value[row] for value in self.images),
            image_masks=tuple(value[row] for value in self.image_masks),
            language_tokens=self.language_tokens[row],
            language_masks=self.language_masks[row],
            state=self.state[row] if self.state is not None else None,
        )

    def to(self, device: torch.device | str) -> "FlowModelInputs":
        return FlowModelInputs(
            images=tuple(value.to(device) for value in self.images),
            image_masks=tuple(value.to(device) for value in self.image_masks),
            language_tokens=self.language_tokens.to(device),
            language_masks=self.language_masks.to(device),
            state=self.state.to(device) if self.state is not None else None,
        )

    def cpu(self) -> "FlowModelInputs":
        return self.to("cpu")

    @classmethod
    def concatenate(cls, rows: list["FlowModelInputs"]) -> "FlowModelInputs":
        if not rows:
            raise ValueError("Cannot concatenate an empty flow input batch")
        image_count = len(rows[0].images)
        mask_count = len(rows[0].image_masks)
        if any(len(row.images) != image_count for row in rows):
            raise ValueError("Flow input batches have inconsistent image counts")
        if any(len(row.image_masks) != mask_count for row in rows):
            raise ValueError("Flow input batches have inconsistent image-mask counts")
        states = [row.state for row in rows]
        if any(state is None for state in states) and not all(
            state is None for state in states
        ):
            raise ValueError("Flow input batches mix present and absent state tensors")
        return cls(
            images=tuple(
                torch.cat([row.images[index] for row in rows], dim=0)
                for index in range(image_count)
            ),
            image_masks=tuple(
                torch.cat([row.image_masks[index] for row in rows], dim=0)
                for index in range(mask_count)
            ),
            language_tokens=torch.cat([row.language_tokens for row in rows], dim=0),
            language_masks=torch.cat([row.language_masks for row in rows], dim=0),
            state=(
                None
                if states[0] is None
                else torch.cat([state for state in states if state is not None], dim=0)
            ),
        )


@dataclass(frozen=True, slots=True)
class FrozenPrefixFlowModelInputs:
    """Compact replay inputs when the expensive prefix encoder is frozen.

    Flow policies such as SmolVLA re-encode high-resolution camera tensors at
    every control step. Retaining those tensors for policy-gradient replay can
    make one same-reset group many gigabytes large. This record stores the exact
    frozen image/language embeddings instead, while keeping state unembedded so
    gradients can still reach a trainable state projection during rescoring.

    The policy bridge must reject this representation whenever any omitted
    prefix encoder parameter is trainable.
    """

    frozen_prefix_embeddings: torch.Tensor
    prefix_pad_masks: torch.Tensor
    prefix_attention_masks: torch.Tensor
    state: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.frozen_prefix_embeddings.shape[0])

    def select(self, index: int) -> "FrozenPrefixFlowModelInputs":
        row = slice(index, index + 1)
        return FrozenPrefixFlowModelInputs(
            frozen_prefix_embeddings=self.frozen_prefix_embeddings[row],
            prefix_pad_masks=self.prefix_pad_masks[row],
            prefix_attention_masks=self.prefix_attention_masks[row],
            state=self.state[row],
        )

    def to(self, device: torch.device | str) -> "FrozenPrefixFlowModelInputs":
        return FrozenPrefixFlowModelInputs(
            frozen_prefix_embeddings=self.frozen_prefix_embeddings.to(device),
            prefix_pad_masks=self.prefix_pad_masks.to(device),
            prefix_attention_masks=self.prefix_attention_masks.to(device),
            state=self.state.to(device),
        )

    def cpu(self) -> "FrozenPrefixFlowModelInputs":
        return self.to("cpu")

    @classmethod
    def concatenate(
        cls,
        rows: list["FrozenPrefixFlowModelInputs"],
    ) -> "FrozenPrefixFlowModelInputs":
        if not rows:
            raise ValueError("Cannot concatenate an empty frozen-prefix batch")
        reference = rows[0]
        for row in rows[1:]:
            if (
                row.frozen_prefix_embeddings.shape[1:]
                != (reference.frozen_prefix_embeddings.shape[1:])
            ):
                raise ValueError("Frozen-prefix embedding shapes are inconsistent")
            if row.prefix_pad_masks.shape[1:] != reference.prefix_pad_masks.shape[1:]:
                raise ValueError("Frozen-prefix padding-mask shapes are inconsistent")
            if (
                row.prefix_attention_masks.shape[1:]
                != (reference.prefix_attention_masks.shape[1:])
            ):
                raise ValueError("Frozen-prefix attention-mask shapes are inconsistent")
            if row.state.shape[1:] != reference.state.shape[1:]:
                raise ValueError("Frozen-prefix state shapes are inconsistent")
        return cls(
            frozen_prefix_embeddings=torch.cat(
                [row.frozen_prefix_embeddings for row in rows], dim=0
            ),
            prefix_pad_masks=torch.cat([row.prefix_pad_masks for row in rows], dim=0),
            prefix_attention_masks=torch.cat(
                [row.prefix_attention_masks for row in rows], dim=0
            ),
            state=torch.cat([row.state for row in rows], dim=0),
        )


@dataclass(frozen=True, slots=True)
class GR00TFlowModelInputs:
    """Replayable GR00T conditioning after the frozen vision-language path.

    The action head's state/action encoders and DiT remain in the differentiable
    rescore path. Only conditioning modules proven frozen by the policy plugin
    may contribute to these cached tensors.
    """

    vision_language_features: torch.Tensor
    vision_language_attention_mask: torch.Tensor
    state: torch.Tensor
    embodiment_id: torch.Tensor
    image_mask: torch.Tensor | None = None
    model_family: str = "gr00t_n1d5"

    @property
    def batch_size(self) -> int:
        return int(self.vision_language_features.shape[0])

    def select(self, index: int) -> "GR00TFlowModelInputs":
        row = slice(index, index + 1)
        return GR00TFlowModelInputs(
            vision_language_features=self.vision_language_features[row],
            vision_language_attention_mask=self.vision_language_attention_mask[row],
            state=self.state[row],
            embodiment_id=self.embodiment_id[row],
            image_mask=(None if self.image_mask is None else self.image_mask[row]),
            model_family=self.model_family,
        )

    def to(self, device: torch.device | str) -> "GR00TFlowModelInputs":
        return GR00TFlowModelInputs(
            vision_language_features=self.vision_language_features.to(device),
            vision_language_attention_mask=self.vision_language_attention_mask.to(
                device
            ),
            state=self.state.to(device),
            embodiment_id=self.embodiment_id.to(device),
            image_mask=(
                None if self.image_mask is None else self.image_mask.to(device)
            ),
            model_family=self.model_family,
        )

    def cpu(self) -> "GR00TFlowModelInputs":
        return self.to("cpu")

    @classmethod
    def concatenate(
        cls,
        rows: list["GR00TFlowModelInputs"],
    ) -> "GR00TFlowModelInputs":
        if not rows:
            raise ValueError("Cannot concatenate an empty GR00T flow input batch")
        reference = rows[0]
        for row in rows[1:]:
            if row.model_family != reference.model_family:
                raise ValueError("GR00T model families are inconsistent")
            if (
                row.vision_language_features.shape[1:]
                != (reference.vision_language_features.shape[1:])
            ):
                raise ValueError("GR00T backbone feature shapes are inconsistent")
            if (
                row.vision_language_attention_mask.shape[1:]
                != (reference.vision_language_attention_mask.shape[1:])
            ):
                raise ValueError("GR00T backbone mask shapes are inconsistent")
            if row.state.shape[1:] != reference.state.shape[1:]:
                raise ValueError("GR00T state shapes are inconsistent")
            if row.embodiment_id.shape[1:] != reference.embodiment_id.shape[1:]:
                raise ValueError("GR00T embodiment ID shapes are inconsistent")
            if (row.image_mask is None) != (reference.image_mask is None):
                raise ValueError("GR00T image-mask availability is inconsistent")
            if (
                row.image_mask is not None
                and reference.image_mask is not None
                and row.image_mask.shape[1:] != reference.image_mask.shape[1:]
            ):
                raise ValueError("GR00T image-mask shapes are inconsistent")
        return cls(
            vision_language_features=torch.cat(
                [row.vision_language_features for row in rows], dim=0
            ),
            vision_language_attention_mask=torch.cat(
                [row.vision_language_attention_mask for row in rows], dim=0
            ),
            state=torch.cat([row.state for row in rows], dim=0),
            embodiment_id=torch.cat([row.embodiment_id for row in rows], dim=0),
            image_mask=(
                None
                if reference.image_mask is None
                else torch.cat([row.image_mask for row in rows], dim=0)
            ),
            model_family=reference.model_family,
        )


FlowReplayInputs: TypeAlias = (
    FlowModelInputs | FrozenPrefixFlowModelInputs | GR00TFlowModelInputs
)


@dataclass(frozen=True, slots=True)
class FlowSDERollout:
    """Executed action chunk and sufficient evidence for policy rescoring."""

    actions: torch.Tensor
    transition: FlowSDETransitionRecord
    inputs: FlowReplayInputs

    def batch_signature(self) -> tuple[object, ...]:
        """Return tensor geometry that can be concatenated without padding.

        Inserting masked tokens into an already encoded transformer prefix is
        not score-preserving for all native policy implementations. Training
        backends therefore bucket this signature and accumulate gradients over
        separate microbatches instead of changing replay semantics.
        """

        if isinstance(self.inputs, FrozenPrefixFlowModelInputs):
            input_signature: tuple[object, ...] = (
                "frozen_prefix",
                tuple(self.inputs.frozen_prefix_embeddings.shape[1:]),
                tuple(self.inputs.prefix_pad_masks.shape[1:]),
                tuple(self.inputs.prefix_attention_masks.shape[1:]),
                tuple(self.inputs.state.shape[1:]),
            )
        elif isinstance(self.inputs, GR00TFlowModelInputs):
            input_signature = (
                f"{self.inputs.model_family}_backbone",
                tuple(self.inputs.vision_language_features.shape[1:]),
                tuple(self.inputs.vision_language_attention_mask.shape[1:]),
                tuple(self.inputs.state.shape[1:]),
                tuple(self.inputs.embodiment_id.shape[1:]),
                (
                    None
                    if self.inputs.image_mask is None
                    else tuple(self.inputs.image_mask.shape[1:])
                ),
            )
        else:
            input_signature = (
                "raw_flow",
                tuple(tuple(value.shape[1:]) for value in self.inputs.images),
                tuple(tuple(value.shape[1:]) for value in self.inputs.image_masks),
                tuple(self.inputs.language_tokens.shape[1:]),
                tuple(self.inputs.language_masks.shape[1:]),
                (
                    None
                    if self.inputs.state is None
                    else tuple(self.inputs.state.shape[1:])
                ),
            )
        return (
            input_signature,
            tuple(self.actions.shape[1:]),
            tuple(self.transition.previous_states.shape[1:]),
            tuple(self.transition.next_states.shape[1:]),
            tuple(self.transition.old_logprobs.shape[1:]),
        )

    def with_old_logprobs(self, old_logprobs: torch.Tensor) -> "FlowSDERollout":
        """Return the same replay evidence with a newly frozen behavior score."""

        if old_logprobs.shape != self.transition.old_logprobs.shape:
            raise ValueError(
                "Refreshed Flow-SDE logprobs must preserve transition shape: "
                f"new={tuple(old_logprobs.shape)}, "
                f"old={tuple(self.transition.old_logprobs.shape)}"
            )
        transition = self.transition
        return FlowSDERollout(
            actions=self.actions,
            transition=FlowSDETransitionRecord(
                previous_states=transition.previous_states,
                next_states=transition.next_states,
                selected_indices=transition.selected_indices,
                old_logprobs=old_logprobs,
            ),
            inputs=self.inputs,
        )

    def select(self, index: int) -> "FlowSDERollout":
        if index < 0 or index >= self.inputs.batch_size:
            raise IndexError(index)
        row = slice(index, index + 1)
        record = self.transition
        return FlowSDERollout(
            actions=self.actions[row],
            transition=FlowSDETransitionRecord(
                previous_states=record.previous_states[row],
                next_states=record.next_states[row],
                selected_indices=record.selected_indices[row],
                old_logprobs=record.old_logprobs[row],
            ),
            inputs=self.inputs.select(index),
        )

    def to(self, device: torch.device | str) -> "FlowSDERollout":
        return FlowSDERollout(
            actions=self.actions.to(device),
            transition=self.transition.to(device),
            inputs=self.inputs.to(device),
        )

    def cpu(self) -> "FlowSDERollout":
        return self.to("cpu")

    @classmethod
    def concatenate(cls, rows: list["FlowSDERollout"]) -> "FlowSDERollout":
        if not rows:
            raise ValueError("Cannot concatenate an empty Flow-SDE batch")
        input_type = type(rows[0].inputs)
        if any(type(row.inputs) is not input_type for row in rows):
            raise ValueError("Flow-SDE rows mix incompatible replay input types")
        return cls(
            actions=torch.cat([row.actions for row in rows], dim=0),
            transition=FlowSDETransitionRecord(
                previous_states=torch.cat(
                    [row.transition.previous_states for row in rows], dim=0
                ),
                next_states=torch.cat(
                    [row.transition.next_states for row in rows], dim=0
                ),
                selected_indices=torch.cat(
                    [row.transition.selected_indices for row in rows], dim=0
                ),
                old_logprobs=torch.cat(
                    [row.transition.old_logprobs for row in rows], dim=0
                ),
            ),
            inputs=input_type.concatenate([row.inputs for row in rows]),
        )
