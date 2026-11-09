"""OpenPI PyTorch boundary for the shared PI Flow-SDE implementation.

OpenPI owns tokenization, image preprocessing, and action normalization. Install
it in an isolated policy runtime; importing this module does not import OpenPI.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from .flow_policy import FlowModelInputs, FlowSDERollout
from .flow_sde import FlowSDESchedule, sample_flow_sde_chain
from .pi_flow_sde import PIFlowSDEBridge


def load_openpi_pytorch(
    checkpoint: str | Path,
    *,
    config_name: str,
    device: str = "cuda:0",
) -> Any:
    """Load converted weights and their normalization assets through OpenPI."""
    if not (Path(checkpoint) / "model.safetensors").is_file():
        raise ValueError("OpenPI requires a local converted model.safetensors")
    from openpi.policies.policy_config import create_trained_policy
    from openpi.training.config import get_config

    config = get_config(config_name)
    config = dataclasses.replace(
        config,
        model=dataclasses.replace(config.model, pytorch_compile_mode=None),
    )
    # OpenPI applies its native mixed BF16/FP32 policy, not whole-model FP32.
    return create_trained_policy(config, checkpoint, pytorch_device=device)


class OpenPIFlowSDEBridge(PIFlowSDEBridge):
    """Reuse PI transition probabilities with OpenPI's native processors."""

    def __init__(
        self,
        native: Any,
        *,
        schedule: FlowSDESchedule,
        execution_horizon: int,
        action_dim: int,
    ) -> None:
        model = native._model
        if not isinstance(model, torch.nn.Module):
            raise TypeError("OpenPI Flow-SDE requires the PyTorch model")
        if schedule.deterministic_sampler != "native_euler":
            raise ValueError("OpenPI requires the native_euler schedule")
        self.native = native
        super().__init__(
            SimpleNamespace(
                model=model,
                config=SimpleNamespace(
                    chunk_size=model.config.action_horizon,
                    max_action_dim=model.config.action_dim,
                ),
            ),
            schedule=schedule,
            execution_horizon=execution_horizon,
            action_dim=action_dim,
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def transform(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Transform one observation without letting native transforms mutate it."""
        import jax

        inputs = self.native._input_transform(copy.deepcopy(dict(request)))
        return jax.tree.map(
            lambda value: torch.from_numpy(np.array(value)).to(self.device)[None],
            inputs,
        )

    def prepare_inputs(self, processed_batch: Any) -> FlowModelInputs:
        if isinstance(processed_batch, FlowModelInputs):
            return processed_batch
        from openpi.models.model import Observation

        images, masks, tokens, token_masks, state = self.model._preprocess_observation(
            Observation.from_dict(processed_batch), train=False
        )
        return FlowModelInputs(tuple(images), tuple(masks), tokens, token_masks, state)

    def postprocess(self, inputs: FlowModelInputs, actions: torch.Tensor) -> np.ndarray:
        """Unnormalize one full predicted chunk before taking its executed prefix."""
        width = int(self.policy.config.max_action_dim)
        horizon = int(self.policy.config.chunk_size)
        if (
            inputs.batch_size != 1
            or inputs.state is None
            or actions.ndim != 3
            or actions.shape[:2] != (1, horizon)
            or actions.shape[-1] not in (self.action_dim, width)
            or not torch.isfinite(actions).all()
        ):
            raise ValueError("Expected one finite, full-horizon OpenPI action chunk")
        if actions.shape[-1] != width:
            # Native quantile unnormalization consumes padded coordinates before
            # the DROID output transform drops them. Never unnormalize twice.
            actions = torch.nn.functional.pad(actions, (0, width - self.action_dim))
        output = self.native._output_transform(
            {
                "state": inputs.state[0].detach().cpu().numpy().copy(),
                "actions": actions[0].detach().cpu().numpy().copy(),
            }
        )
        physical = np.asarray(output["actions"])
        if (
            physical.shape != (horizon, self.action_dim)
            or not np.isfinite(physical).all()
        ):
            raise ValueError("OpenPI output transform returned invalid actions")
        return physical.copy()


class OpenPIBatchedSampler:
    """Fixed-shape inference with per-trajectory RNG and replayable transitions.

    Seeds belong to requests, not batch positions. Padding preserves GEMM shapes
    as episodes finish; only real requests are returned to the environment.
    """

    def __init__(self, bridge: OpenPIFlowSDEBridge, *, batch_size: int) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if bridge.schedule.num_steps < 2:
            raise ValueError("Flow-SDE requires at least two denoising steps")
        self.bridge = bridge
        self.batch_size = batch_size

    def prepare(self, requests: Sequence[Mapping[str, Any]]):
        import jax

        if not 0 < len(requests) <= self.batch_size:
            raise ValueError("Request count must be in [1, batch_size]")
        seeds = [request["sampling_seed"] for request in requests]
        if any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds):
            raise ValueError("sampling_seed must be an integer in [0, 2**63)")
        observations = [
            {key: value for key, value in request.items() if key != "sampling_seed"}
            for request in requests
        ]
        observations += [observations[-1]] * (self.batch_size - len(requests))
        seeds += [seeds[-1]] * (self.batch_size - len(requests))
        transformed = [
            self.bridge.transform(observation) for observation in observations
        ]
        batch = jax.tree.map(lambda *values: torch.cat(values, dim=0), *transformed)
        shape = (
            self.bridge.policy.config.chunk_size,
            self.bridge.policy.config.max_action_dim,
        )
        noise = torch.from_numpy(
            np.stack(
                [
                    np.random.default_rng(seed).normal(size=shape).astype(np.float32)
                    for seed in seeds
                ]
            )
        ).to(self.bridge.device)
        return batch, noise, seeds

    @torch.no_grad()
    def infer(self, requests: Sequence[Mapping[str, Any]], *, mode: str):
        if mode not in ("ode", "sde"):
            raise ValueError("mode must be ode or sde")
        batch, noise, seeds = self.prepare(requests)
        inputs = self.bridge.prepare_inputs(batch)
        record = None
        if mode == "ode":
            from openpi.models.model import Observation

            normalized = self.bridge.model.sample_actions(
                self.bridge.device,
                Observation.from_dict(batch),
                noise=noise,
                num_steps=self.bridge.schedule.num_steps,
            )
        else:
            indices = torch.tensor(
                [
                    np.random.default_rng(np.random.SeedSequence([seed, 1])).integers(
                        0, self.bridge.schedule.num_steps - 1
                    )
                    for seed in seeds
                ],
                device=self.bridge.device,
            )
            generators = [
                torch.Generator(device=self.bridge.device).manual_seed(seed)
                for seed in seeds
            ]
            prefix_masks, cache = self.bridge._prefix_cache(inputs)

            def velocity(state, timestep):
                return self.bridge._velocity(
                    inputs=inputs,
                    prefix_masks=prefix_masks,
                    cache=cache,
                    state=state,
                    timestep=timestep,
                )

            def per_row_noise(shape, device, dtype):
                return torch.stack(
                    [
                        torch.randn(
                            shape[1:], device=device, dtype=dtype, generator=generator
                        )
                        for generator in generators
                    ]
                )

            chain = sample_flow_sde_chain(
                noise,
                selected_indices=indices,
                schedule=self.bridge.schedule,
                velocity_fn=velocity,
                noise_fn=per_row_noise,
            )
            record = FlowSDERollout(
                actions=chain.actions[:, :, : self.bridge.action_dim],
                transition=chain.selected_transitions(),
                inputs=inputs,
            )
            normalized = record.actions
        actions = np.stack(
            [
                self.bridge.postprocess(inputs.select(row), normalized[row : row + 1])
                for row in range(len(requests))
            ]
        )
        return actions, record
