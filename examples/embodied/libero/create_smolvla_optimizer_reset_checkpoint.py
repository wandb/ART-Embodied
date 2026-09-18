"""Create a numbered SmolVLA checkpoint after expanding its LoRA surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import yaml

from art_embodied import EmbodiedExperimentConfig, make_policy
from art_embodied.checkpointing import CheckpointManager


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def create_optimizer_reset_checkpoint(
    *,
    config_path: Path,
    output_path: Path,
    step: int,
    device: str,
) -> Path:
    """Publish a new-rank policy at an existing global update number.

    A LoRA rank or target-surface change cannot restore the old Adam moments:
    both parameter and optimizer-state shapes differ. This checkpoint carries
    the expanded policy exactly, records the previous global update cursor,
    and intentionally starts Adam with empty moments. The ordinary resume path
    can then continue update numbering without pretending optimizer continuity.
    """

    if step < 1:
        raise ValueError("Bootstrap step must be positive")
    config_path = config_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Experiment config must be a mapping: {config_path}")

    # The final YAML points at the checkpoint this function is about to create.
    # Remove only that operational reference while resolving the mathematical
    # resume contract used to publish the checkpoint.
    storage = raw.get("storage")
    if not isinstance(storage, dict):
        raise TypeError("Experiment config storage section must be a mapping")
    storage["resume_from_checkpoint"] = None
    config = EmbodiedExperimentConfig.model_validate(raw)
    if config.policy.type != "smolvla":
        raise ValueError("Optimizer-reset bootstrap currently requires SmolVLA")
    if config.training.optimizer.type != "adamw":
        raise ValueError("SmolVLA Flow-SDE bootstrap currently requires AdamW")
    if step >= config.training.updates:
        raise ValueError(
            "Bootstrap step must be smaller than training.updates: "
            f"step={step}, updates={config.training.updates}"
        )

    random.seed(config.experiment.seed)
    np.random.seed(config.experiment.seed % (2**32))
    torch.manual_seed(config.experiment.seed)
    load_config = config.model_copy(
        update={"policy": config.policy.model_copy(update={"device": str(device)})}
    )
    policy = make_policy(load_config)
    parameters = [
        parameter for parameter in policy.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("Expanded SmolVLA policy has no trainable parameters")
    optimizer_config = config.training.optimizer
    optimizer = torch.optim.AdamW(
        parameters,
        lr=optimizer_config.learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
        eps=optimizer_config.epsilon,
        weight_decay=optimizer_config.weight_decay,
    )

    def write(staging: Path) -> None:
        policy.save_checkpoint(str(staging / "policy"))
        torch.save(
            {"step": int(step), "optimizer": optimizer.state_dict()},
            staging / "art_embodied_training_state.pt",
        )
        (staging / "art_embodied_training_state.json").write_text(
            json.dumps({"step": int(step)}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "kind": "smolvla_optimizer_reset_resume_bootstrap",
            "step": int(step),
            "optimizer_state_carried_forward": False,
            "optimizer_state_entries": len(optimizer.state_dict()["state"]),
            "trainable_parameters": sum(parameter.numel() for parameter in parameters),
            "source_config": str(config_path),
            "policy_path": str(config.policy.path),
        }
        (staging / "art_embodied_optimizer_reset_resume.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return CheckpointManager().publish(
        output_path,
        writer=write,
        config_fingerprint=config.fingerprint,
        resume_contract_fingerprint=config.resume_contract_fingerprint,
        metadata={
            "backend": "flow_sde_grpo",
            "step": int(step),
            "optimizer_reset": True,
        },
    )


def main() -> None:
    args = _parse_args()
    path = create_optimizer_reset_checkpoint(
        config_path=args.config,
        output_path=args.output,
        step=args.step,
        device=args.device,
    )
    print(path)


if __name__ == "__main__":
    main()
