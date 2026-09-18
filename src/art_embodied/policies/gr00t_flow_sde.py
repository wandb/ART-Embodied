"""NVIDIA GR00T N1.5 bridge for sampler-aligned Flow-SDE policy RL.

The bridge deliberately consumes the native action head instead of copying its
weights or model layers. It reproduces RLinf's N1.5 velocity evaluation while
leaving advantage construction and optimization to ART-Embodied's critic-free
GRPO backend.
"""

from __future__ import annotations

from typing import Any

import torch

from .flow_policy import FlowSDERollout, GR00TFlowModelInputs
from .flow_sde import (
    FlowSDESchedule,
    rescore_flow_sde_transitions,
    sample_flow_sde_chain,
)


class GR00TN15FlowSDEBridge:
    """Expose a native N1.5 action head through ART's Flow-SDE contract."""

    def __init__(
        self,
        action_head: Any,
        *,
        schedule: FlowSDESchedule,
        execution_horizon: int,
        action_dim: int,
    ) -> None:
        if schedule.noise_time != "zero":
            raise ValueError("GR00T N1.5 requires Flow-SDE noise_time='zero'")
        if execution_horizon < 1 or action_dim < 1:
            raise ValueError("execution_horizon and action_dim must be positive")
        config = getattr(action_head, "config", None)
        if config is None:
            raise TypeError("GR00T action head does not expose config")
        if execution_horizon > int(config.action_horizon):
            raise ValueError("execution_horizon exceeds GR00T action_horizon")
        if action_dim > int(config.action_dim):
            raise ValueError("action_dim exceeds GR00T action_dim")
        required = (
            "state_encoder",
            "action_encoder",
            "future_tokens",
            "model",
            "action_decoder",
            "num_timestep_buckets",
        )
        missing = [name for name in required if not hasattr(action_head, name)]
        if missing:
            raise TypeError(
                "GR00T N1.5 action head is missing required members: "
                + ", ".join(missing)
            )
        self.action_head = action_head
        self.schedule = schedule
        self.execution_horizon = int(execution_horizon)
        self.action_dim = int(action_dim)

    @torch.no_grad()
    def sample(
        self,
        inputs: GR00TFlowModelInputs,
        *,
        selected_index: int | torch.Tensor,
        generator: torch.Generator | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> FlowSDERollout:
        """Sample an N1.5 action chunk and retain one stochastic transition."""

        if initial_noise is None:
            initial_noise = self._sample_model_noise(
                torch.Size(
                    (
                        inputs.batch_size,
                        int(self.action_head.config.action_horizon),
                        int(self.action_head.config.action_dim),
                    )
                ),
                inputs.state.device,
                inputs.vision_language_features.dtype,
            )

        chain = sample_flow_sde_chain(
            initial_noise,
            selected_indices=selected_index,
            schedule=self.schedule,
            velocity_fn=lambda state, time: self.velocity(inputs, state, time),
            generator=generator,
            noise_fn=(None if generator is not None else self._sample_model_noise),
        )
        return FlowSDERollout(
            actions=chain.actions[:, :, : self.action_dim],
            transition=chain.selected_transitions(),
            inputs=inputs,
        )

    def rescore(self, rollout: FlowSDERollout) -> torch.Tensor:
        """Recompute selected-transition logprobs with policy gradients."""

        if not isinstance(rollout.inputs, GR00TFlowModelInputs):
            raise TypeError("GR00T rescore requires GR00TFlowModelInputs")
        scores = rescore_flow_sde_transitions(
            rollout.transition,
            schedule=self.schedule,
            velocity_fn=lambda state, time: self.velocity(rollout.inputs, state, time),
        )
        return scores[:, : self.execution_horizon, : self.action_dim]

    def velocity(
        self,
        inputs: GR00TFlowModelInputs,
        state: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the exact N1.5 velocity field used by RLinf Flow-SDE."""

        if state.shape[0] != inputs.batch_size:
            raise ValueError("GR00T state batch does not match conditioning batch")
        timestep_buckets = (timestep * int(self.action_head.num_timestep_buckets)).to(
            dtype=torch.int64, device=state.device
        )
        embodiment_id = inputs.embodiment_id.to(device=state.device)
        state_features = self.action_head.state_encoder(inputs.state, embodiment_id)
        action_features = self.action_head.action_encoder(
            state, timestep_buckets, embodiment_id
        )
        if bool(getattr(self.action_head.config, "add_pos_embed", False)):
            positions = torch.arange(
                action_features.shape[1], dtype=torch.long, device=state.device
            )
            action_features = action_features + self.action_head.position_embedding(
                positions
            ).unsqueeze(0)
        future_tokens = self.action_head.future_tokens.weight.unsqueeze(0).expand(
            inputs.batch_size, -1, -1
        )
        state_action = torch.cat(
            (state_features, future_tokens, action_features), dim=1
        )
        model_output = self.action_head.model(
            hidden_states=state_action,
            encoder_hidden_states=inputs.vision_language_features,
            timestep=timestep_buckets,
        )
        model_output = model_output[:, -int(self.action_head.config.action_horizon) :]
        return self.action_head.action_decoder(model_output, embodiment_id)

    def _sample_model_noise(
        self,
        shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        sampler = getattr(self.action_head, "sample_noise", None)
        if callable(sampler):
            return sampler(shape, device).to(dtype=dtype)
        return torch.randn(shape, device=device, dtype=dtype)
