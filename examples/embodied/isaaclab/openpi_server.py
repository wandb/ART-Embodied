"""Serve native OpenPI ODE/Flow-SDE inference on localhost in its own runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def inference_configuration(path: Path, *, device: str):
    from art_embodied.config import EmbodiedExperimentConfig

    raw = json.loads(path.read_text())
    # Inference loads --adapter-checkpoint, not the coordinator's optimizer
    # resume source, which may already have been removed by checkpoint retention.
    raw["storage"]["resume_from_checkpoint"] = None
    config = EmbodiedExperimentConfig.model_validate(raw)
    return config.model_copy(
        update={"policy": config.policy.model_copy(update={"device": device})}
    )


def validate_adapter_checkpoint(path: Path, *, config, policy_version: int):
    from art_embodied.checkpointing import CheckpointManager

    if path.name != "policy":
        raise ValueError(
            "adapter-checkpoint must be a completed update's policy directory"
        )
    checked = CheckpointManager().validate(
        path.parent,
        expected_resume_contract_fingerprint=config.resume_contract_fingerprint,
    )
    if checked.manifest["metadata"].get("step") != policy_version:
        raise ValueError("Checkpoint update differs from the requested policy version")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-name", default="pi05_droid_jointpos_polaris")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mode", choices=("ode", "sde"), default="ode")
    parser.add_argument("--noise-level", type=float, default=0.5)
    parser.add_argument("--denoising-steps", type=int, default=10)
    parser.add_argument("--record-dir", type=Path)
    parser.add_argument(
        "--config", type=Path, help="ART-Embodied training configuration"
    )
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--policy-version", type=int, default=0)
    args = parser.parse_args()
    if args.adapter_checkpoint and not args.config:
        parser.error("adapter-checkpoint requires config")
    if args.policy_version < 0 or (args.policy_version > 0) != bool(
        args.adapter_checkpoint
    ):
        parser.error("A trained policy version requires an adapter checkpoint")

    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    import torch

    from art_embodied.policies.flow_replay_io import save_flow_replay
    from art_embodied.policies.flow_sde import FlowSDESchedule
    from art_embodied.policies.openpi_flow_sde import (
        OpenPIBatchedSampler,
        OpenPIFlowSDEBridge,
        load_openpi_pytorch,
    )

    if args.config:
        import random

        import numpy as np

        from art_embodied.policies.openpi import make_openpi_flow_policy

        config = inference_configuration(args.config, device=args.device)
        if (
            Path(config.policy.path).resolve() != args.checkpoint.resolve()
            or config.policy.load_kwargs["config_name"] != args.config_name
            or config.training.microbatch_size != args.batch_size
            or config.algorithm.flow_sde.num_denoise_steps != args.denoising_steps
            or config.algorithm.flow_sde.noise_level != args.noise_level
        ):
            parser.error("Inference arguments differ from the training configuration")
        torch.manual_seed(config.experiment.seed)
        random.seed(config.experiment.seed)
        np.random.seed(config.experiment.seed)
        trainable = make_openpi_flow_policy(config)
        if args.adapter_checkpoint:
            validate_adapter_checkpoint(
                args.adapter_checkpoint,
                config=config,
                policy_version=args.policy_version,
            )
            trainable.restore_checkpoint(args.adapter_checkpoint, restore_rng=False)
        trainable.requires_grad_(False)
        trainable.eval()
        bridge = trainable.bridge
    else:
        native = load_openpi_pytorch(
            args.checkpoint, config_name=args.config_name, device=args.device
        )
        native._model.requires_grad_(False)
        native._model.eval()
        bridge = OpenPIFlowSDEBridge(
            native,
            schedule=FlowSDESchedule(
                args.denoising_steps,
                args.noise_level,
                deterministic_sampler="native_euler",
            ),
            execution_horizon=15,
            action_dim=8,
        )
    policy = OpenPIBatchedSampler(bridge, batch_size=args.batch_size)
    if args.record_dir:
        if args.mode != "sde":
            parser.error("--record-dir requires --mode sde")
        args.record_dir.mkdir(parents=True, exist_ok=False)

    class Server:
        counter = 0

        def infer(self, request):
            requests = request["batch"]
            start = time.monotonic()
            actions, record = policy.infer(requests, mode=args.mode)
            result = {
                "actions": actions,
                "mode": args.mode,
                "model_seconds": time.monotonic() - start,
            }
            if record is not None:
                if self.counter < 16:
                    with torch.no_grad():
                        scores = bridge.rescore(record)
                    old = record.transition.old_logprobs[:, :15, :8]
                    torch.testing.assert_close(scores, old, rtol=0, atol=2e-5)
                    result["logprob_max_error"] = float((scores - old).abs().max())
                result["selected_indices"] = (
                    record.transition.selected_indices[: len(requests)].cpu().numpy()
                )
                result["old_chunk_logprobs"] = (
                    record.transition.old_logprobs[: len(requests), :15, :8]
                    .sum(dim=(1, 2))
                    .cpu()
                    .numpy()
                )
                if args.record_dir:
                    record_id = f"chunk-{self.counter:06d}"
                    destination = args.record_dir / f"{record_id}.pt"
                    save_flow_replay(destination, record, valid_count=len(requests))
                    result["record_id"] = record_id
            self.counter += 1
            return result

    WebsocketPolicyServer(
        Server(),
        host="127.0.0.1",
        port=args.port,
        metadata={
            "runtime": "pytorch",
            "model": args.config_name,
            "mode": args.mode,
            "batch_size": args.batch_size,
            "noise_level": args.noise_level,
            "denoising_steps": args.denoising_steps,
            "training_updates": args.policy_version,
        },
    ).serve_forever()


if __name__ == "__main__":
    main()
