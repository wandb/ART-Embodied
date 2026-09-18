"""LeRobot PI0/PI0.5 bridge for sampler-aligned Flow-SDE policy RL.

This is the model boundary, not an environment adapter. Callers keep LeRobot's
native processor pipeline, pass the processed batch here, and receive the
executed action chunk plus the compact transition needed for later rescore.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import torch

from .flow_policy import FlowModelInputs, FlowSDERollout
from .flow_sde import (
    FlowSDESchedule,
    FlowSDETransitionRecord,
    rescore_flow_sde_transitions,
    sample_flow_sde_chain,
)

# Backward-compatible names for downstream PI diagnostics. The backend and new
# model integrations use the model-family-neutral names directly.
PIFlowModelInputs = FlowModelInputs
PIFlowSDERollout = FlowSDERollout


class PIFlowSDEBridge:
    """Apply ART's Flow-SDE probability contract to a native LeRobot PI model."""

    def __init__(
        self,
        policy: Any,
        *,
        schedule: FlowSDESchedule,
        execution_horizon: int,
        action_dim: int,
    ) -> None:
        if execution_horizon < 1 or action_dim < 1:
            raise ValueError("execution_horizon and action_dim must be positive")
        model = getattr(policy, "model", None)
        config = getattr(policy, "config", None)
        if model is None or config is None:
            raise TypeError("PIFlowSDEBridge requires a loaded LeRobot PI policy")
        if execution_horizon > int(config.chunk_size):
            raise ValueError("execution_horizon exceeds policy.config.chunk_size")
        if action_dim > int(config.max_action_dim):
            raise ValueError("action_dim exceeds policy.config.max_action_dim")
        if not callable(getattr(model, "denoise_step", None)):
            raise TypeError("PI policy model does not expose denoise_step")
        self.policy = policy
        self.model = model
        self.schedule = schedule
        self.execution_horizon = int(execution_horizon)
        self.action_dim = int(action_dim)

    @torch.no_grad()
    def sample(
        self,
        processed_batch: dict[str, torch.Tensor],
        *,
        selected_index: int | torch.Tensor,
        generator: torch.Generator | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> PIFlowSDERollout:
        """Sample native actions and retain one stochastic transition."""

        inputs = self.prepare_inputs(processed_batch)
        if initial_noise is None:
            initial_noise = self.model.sample_noise(
                (
                    inputs.batch_size,
                    int(self.policy.config.chunk_size),
                    int(self.policy.config.max_action_dim),
                ),
                inputs.language_tokens.device,
            )
        prefix_masks, cache = self._prefix_cache(inputs)

        def velocity(state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            return self._velocity(
                inputs=inputs,
                prefix_masks=prefix_masks,
                cache=cache,
                state=state,
                timestep=timestep,
            )

        chain = sample_flow_sde_chain(
            initial_noise,
            selected_indices=selected_index,
            schedule=self.schedule,
            velocity_fn=velocity,
            generator=generator,
            noise_fn=(None if generator is not None else self._sample_model_noise),
        )
        return PIFlowSDERollout(
            actions=chain.actions[:, :, : self.action_dim],
            transition=chain.selected_transitions(),
            inputs=inputs,
        )

    def rescore(self, rollout: PIFlowSDERollout) -> torch.Tensor:
        """Recompute element logprobs while preserving policy gradients."""

        prefix_masks, cache = self._prefix_cache(rollout.inputs)

        def velocity(state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            return self._velocity(
                inputs=rollout.inputs,
                prefix_masks=prefix_masks,
                cache=cache,
                state=state,
                timestep=timestep,
            )

        scores = rescore_flow_sde_transitions(
            rollout.transition,
            schedule=self.schedule,
            velocity_fn=velocity,
        )
        return scores[:, : self.execution_horizon, : self.action_dim]

    def _sample_model_noise(
        self,
        shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Use the PI model's sampler for RLinf-compatible RNG consumption."""

        return self.model.sample_noise(shape, device).to(dtype=dtype)

    def prepare_inputs(
        self,
        processed_batch: dict[str, torch.Tensor],
    ) -> PIFlowModelInputs:
        from lerobot.utils.constants import (
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
        )

        images, image_masks = self.policy._preprocess_images(processed_batch)  # noqa: SLF001
        state = None
        prepare_state = getattr(self.policy, "prepare_state", None)
        if callable(prepare_state):
            state = prepare_state(processed_batch)
        return PIFlowModelInputs(
            images=tuple(images),
            image_masks=tuple(image_masks),
            language_tokens=processed_batch[OBS_LANGUAGE_TOKENS],
            language_masks=processed_batch[OBS_LANGUAGE_ATTENTION_MASK],
            state=state,
        )

    def _prefix_cache(
        self,
        inputs: PIFlowModelInputs,
    ) -> tuple[torch.Tensor, Any]:
        modeling_module = import_module(type(self.model).__module__)
        make_att_2d_masks = modeling_module.make_att_2d_masks

        prefix_embeddings, prefix_masks, prefix_attention = self.model.embed_prefix(
            list(inputs.images),
            list(inputs.image_masks),
            inputs.language_tokens,
            inputs.language_masks,
        )
        attention = make_att_2d_masks(prefix_masks, prefix_attention)
        attention = self.model._prepare_attention_masks_4d(attention)  # noqa: SLF001
        positions = torch.cumsum(prefix_masks, dim=1) - 1
        language_model = self.model.paligemma_with_expert.paligemma.model.language_model
        language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, cache = self.model.paligemma_with_expert.forward(
            attention_mask=attention,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[prefix_embeddings, None],
            use_cache=True,
        )
        return prefix_masks, cache

    def _velocity(
        self,
        *,
        inputs: PIFlowModelInputs,
        prefix_masks: torch.Tensor,
        cache: Any,
        state: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        kwargs: dict[str, Any] = {
            "prefix_pad_masks": prefix_masks,
            "past_key_values": cache,
            "x_t": state,
            "timestep": timestep,
        }
        if inputs.state is not None:
            kwargs["state"] = inputs.state
        return self.model.denoise_step(**kwargs)
