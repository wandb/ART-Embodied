"""Sampler-aligned Flow-SDE bridge for NVIDIA GR00T N1.7."""

from __future__ import annotations

from typing import Any

import torch

from .flow_policy import FlowSDERollout, GR00TFlowModelInputs
from .flow_sde import (
    FlowSDESchedule,
    rescore_flow_sde_transitions,
    sample_flow_sde_chain,
)


class GR00TN17FlowSDEBridge:
    """Expose the native N1.7 action head without changing its velocity field."""

    def __init__(
        self,
        action_head: Any,
        *,
        schedule: FlowSDESchedule,
        execution_horizon: int,
        action_dim: int,
    ) -> None:
        if schedule.noise_time != "zero":
            raise ValueError("GR00T N1.7 requires Flow-SDE noise_time='zero'")
        if execution_horizon < 1 or action_dim < 1:
            raise ValueError("execution_horizon and action_dim must be positive")
        config = getattr(action_head, "config", None)
        if config is None:
            raise TypeError("GR00T N1.7 action head does not expose config")
        if execution_horizon > int(config.action_horizon):
            raise ValueError("execution_horizon exceeds N1.7 action_horizon")
        if not hasattr(action_head, "action_dim"):
            raise TypeError("GR00T N1.7 action head does not expose action_dim")
        if action_dim > int(action_head.action_dim):
            raise ValueError("action_dim exceeds N1.7 action_dim")
        required = (
            "state_encoder",
            "action_encoder",
            "model",
            "action_decoder",
            "num_timestep_buckets",
        )
        missing = [name for name in required if not hasattr(action_head, name)]
        if missing:
            raise TypeError(
                "GR00T N1.7 action head is missing required members: "
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
        """Generate the full native horizon and retain one SDE transition."""

        self._validate_inputs(inputs)
        if initial_noise is None:
            initial_noise = torch.randn(
                (
                    inputs.batch_size,
                    int(self.action_head.config.action_horizon),
                    int(self.action_head.action_dim),
                ),
                device=inputs.state.device,
                dtype=inputs.vision_language_features.dtype,
            )
        chain = sample_flow_sde_chain(
            initial_noise,
            selected_indices=selected_index,
            schedule=self.schedule,
            velocity_fn=lambda state, time: self.velocity(inputs, state, time),
            generator=generator,
        )
        return FlowSDERollout(
            actions=chain.actions[:, :, : self.action_dim],
            transition=chain.selected_transitions(),
            inputs=inputs,
        )

    def rescore(self, rollout: FlowSDERollout) -> torch.Tensor:
        """Score only the action prefix that was applied to the environment."""

        if not isinstance(rollout.inputs, GR00TFlowModelInputs):
            raise TypeError("GR00T N1.7 rescore requires GR00TFlowModelInputs")
        self._validate_inputs(rollout.inputs)
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
        """Evaluate the N1.7 velocity field from the pinned NVIDIA source."""

        self._validate_inputs(inputs)
        if state.shape[0] != inputs.batch_size:
            raise ValueError("N1.7 state batch does not match conditioning batch")
        timestep_buckets = (timestep * int(self.action_head.num_timestep_buckets)).to(
            dtype=torch.int64, device=state.device
        )
        embodiment_id = inputs.embodiment_id.to(device=state.device)
        robot_state = inputs.state.to(device=state.device)
        robot_state = robot_state.view(robot_state.shape[0], 1, -1)
        state_features = self.action_head.state_encoder(robot_state, embodiment_id)
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
        state_action = torch.cat((state_features, action_features), dim=1)
        model_kwargs = {
            "hidden_states": state_action,
            "encoder_hidden_states": inputs.vision_language_features,
            "timestep": timestep_buckets,
        }
        if bool(getattr(self.action_head.config, "use_alternate_vl_dit", False)):
            if inputs.image_mask is None:
                raise ValueError("Alternate N1.7 VL-DiT requires image_mask")
            model_kwargs.update(
                image_mask=inputs.image_mask,
                backbone_attention_mask=inputs.vision_language_attention_mask,
            )
        model_output = self.action_head.model(**model_kwargs)
        prediction = self.action_head.action_decoder(model_output, embodiment_id)
        return prediction[:, -int(self.action_head.config.action_horizon) :]

    @staticmethod
    def _validate_inputs(inputs: GR00TFlowModelInputs) -> None:
        if inputs.model_family != "gr00t_n1d7":
            raise ValueError(
                f"GR00T N1.7 bridge received conditioning for {inputs.model_family!r}"
            )
