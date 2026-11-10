"""Tensor-only storage for raw-observation Flow-SDE replay records."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import torch

from .flow_policy import FlowModelInputs, FlowSDERollout
from .flow_sde import FlowSDETransitionRecord


def compact_masked_images(rollout: FlowSDERollout) -> FlowSDERollout:
    """Store a constant masked image as a scalar view without changing its values."""
    if not isinstance(rollout.inputs, FlowModelInputs):
        raise TypeError("Raw-observation replay requires FlowModelInputs")
    images = []
    for image, mask in zip(
        rollout.inputs.images, rollout.inputs.image_masks, strict=True
    ):
        if image.numel() and not bool(mask.any()):
            expanded = image.reshape(-1)[0].clone().expand_as(image)
            if torch.equal(image, expanded):
                image = expanded
        images.append(image)
    return replace(rollout, inputs=replace(rollout.inputs, images=tuple(images)))


def save_flow_replay(path, rollout: FlowSDERollout, *, valid_count: int):
    """Write only tensors and primitive containers, never custom Python objects."""
    if not isinstance(rollout.inputs, FlowModelInputs):
        raise TypeError("Raw-observation replay requires FlowModelInputs")
    if type(valid_count) is not int or not 0 < valid_count <= rollout.inputs.batch_size:
        raise ValueError("Invalid replay valid_count")
    path = Path(path)
    if path.exists() or path.with_suffix(".tmp").exists():
        raise FileExistsError(path)
    rollout = compact_masked_images(rollout.cpu())
    payload = {
        "version": 1,
        "valid_count": valid_count,
        "actions": rollout.actions,
        "inputs": {
            f.name: getattr(rollout.inputs, f.name) for f in fields(FlowModelInputs)
        },
        "transition": {
            f.name: getattr(rollout.transition, f.name)
            for f in fields(FlowSDETransitionRecord)
        },
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_flow_replay(path) -> tuple[FlowSDERollout, int]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["version"] != 1:
        raise ValueError("Unknown replay version")
    rollout = FlowSDERollout(
        actions=payload["actions"],
        inputs=FlowModelInputs(**payload["inputs"]),
        transition=FlowSDETransitionRecord(**payload["transition"]),
    )
    batch = rollout.inputs.batch_size
    valid = payload["valid_count"]
    if type(valid) is not int or not 0 < valid <= batch:
        raise ValueError("Invalid replay valid_count")
    tensors = [
        rollout.actions,
        *rollout.inputs.images,
        *rollout.inputs.image_masks,
        rollout.inputs.language_tokens,
        rollout.inputs.language_masks,
        *(getattr(rollout.transition, f.name) for f in fields(FlowSDETransitionRecord)),
    ]
    if rollout.inputs.state is not None:
        tensors.append(rollout.inputs.state)
    if any(
        not isinstance(t, torch.Tensor) or t.ndim == 0 or t.shape[0] != batch
        for t in tensors
    ):
        raise ValueError("Inconsistent replay batch dimensions")
    if any(not torch.isfinite(t).all() for t in tensors):
        raise ValueError("Nonfinite replay tensor")
    if len(rollout.inputs.images) != len(rollout.inputs.image_masks):
        raise ValueError("Replay image masks do not match images")
    return rollout, valid
