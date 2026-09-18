"""LeRobot SmolVLA bridge for sampler-aligned Flow-SDE policy RL."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import torch

from .flow_policy import (
    FlowModelInputs,
    FlowSDERollout,
    FrozenPrefixFlowModelInputs,
)
from .flow_sde import (
    FlowSDESchedule,
    rescore_flow_sde_transitions,
    sample_flow_sde_chain,
)


class SmolVLAFlowSDEBridge:
    """Expose SmolVLA's native velocity field through ART's Flow-SDE contract."""

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
        if getattr(policy, "name", None) != "smolvla":
            raise TypeError("SmolVLAFlowSDEBridge requires a native SmolVLA policy")
        model = getattr(policy, "model", None)
        config = getattr(policy, "config", None)
        if model is None or config is None:
            raise TypeError("SmolVLA policy does not expose its model and config")
        if schedule.num_steps != int(config.num_steps):
            raise ValueError(
                "Flow-SDE schedule must match SmolVLA config.num_steps: "
                f"{schedule.num_steps} != {config.num_steps}"
            )
        if execution_horizon > int(config.chunk_size):
            raise ValueError("execution_horizon exceeds SmolVLA chunk_size")
        if action_dim > int(config.max_action_dim):
            raise ValueError("action_dim exceeds SmolVLA max_action_dim")
        if not callable(getattr(model, "denoise_step", None)):
            raise TypeError("SmolVLA model does not expose denoise_step")
        self.policy = policy
        self.model = model
        self.schedule = schedule
        self.execution_horizon = int(execution_horizon)
        self.action_dim = int(action_dim)
        self._frozen_prefix_surface_validated = False

    @torch.no_grad()
    def sample(
        self,
        processed_batch: dict[str, torch.Tensor],
        *,
        selected_index: int | torch.Tensor,
        generator: torch.Generator | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> FlowSDERollout:
        """Sample one native SmolVLA chunk and retain its stochastic transition."""

        inputs = self.prepare_inputs(processed_batch)
        if initial_noise is None:
            initial_noise = self.model.sample_noise(
                (
                    inputs.batch_size,
                    int(self.policy.config.chunk_size),
                    int(self.policy.config.max_action_dim),
                ),
                inputs.state.device,
            )
        prefix_masks, cache = self._prefix_cache(inputs)

        def velocity(state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            return self.model.denoise_step(
                x_t=state,
                prefix_pad_masks=prefix_masks,
                past_key_values=cache,
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
        actions = chain.actions[:, :, : self.action_dim]
        if bool(getattr(self.policy.config, "adapt_to_pi_aloha", False)):
            actions = self.policy._pi_aloha_encode_actions(actions)  # noqa: SLF001
        return FlowSDERollout(
            actions=actions,
            transition=chain.selected_transitions(),
            inputs=inputs,
        )

    def rescore(self, rollout: FlowSDERollout) -> torch.Tensor:
        """Recompute rollout transition logprobs with gradients enabled."""

        prefix_masks, cache = self._prefix_cache(rollout.inputs)

        def velocity(state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            return self.model.denoise_step(
                x_t=state,
                prefix_pad_masks=prefix_masks,
                past_key_values=cache,
                timestep=timestep,
            )

        scores = rescore_flow_sde_transitions(
            rollout.transition,
            schedule=self.schedule,
            velocity_fn=velocity,
        )
        return scores[:, : self.execution_horizon, : self.action_dim]

    @torch.no_grad()
    def sample_native_ode(
        self,
        processed_batch: dict[str, torch.Tensor],
        *,
        initial_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Reproduce SmolVLA's deterministic Euler sampler for conformance checks."""

        inputs = self.prepare_inputs(processed_batch)
        prefix_masks, cache = self._prefix_cache(inputs)

        def velocity(state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            return self.model.denoise_step(
                x_t=state,
                prefix_pad_masks=prefix_masks,
                past_key_values=cache,
                timestep=timestep,
            )

        # Keep the native Python-float timestep construction and Euler update
        # order. On the real SmolVLA checkpoint, replacing this loop with an
        # algebraically equivalent linspace/interpolation path measurably changes
        # the final action because the nonlinear velocity field amplifies ULPs.
        state = initial_noise
        delta = -1.0 / self.schedule.num_steps
        for step in range(self.schedule.num_steps):
            timestep = torch.tensor(
                1.0 + step * delta,
                dtype=torch.float32,
                device=state.device,
            ).expand(state.shape[0])
            state = state + delta * velocity(state, timestep)
        actions = state[:, :, : self.action_dim]
        if bool(getattr(self.policy.config, "adapt_to_pi_aloha", False)):
            actions = self.policy._pi_aloha_encode_actions(actions)  # noqa: SLF001
        return actions

    @torch.no_grad()
    def prepare_inputs(
        self,
        processed_batch: dict[str, torch.Tensor],
    ) -> FrozenPrefixFlowModelInputs:
        """Retain exact frozen prefix features instead of camera tensors.

        The SmolVLA checkpoint executes one action per inference call. Keeping
        its resized float camera tensors at every environment step would make a
        single eight-trajectory LIBERO group roughly 12 GiB. The image and
        language encoders are frozen by the supported LoRA contract, so their
        exact output can be cached without changing the gradient. State remains
        unembedded because ``state_proj`` is part of the trainable surface.
        """

        self._validate_frozen_prefix_trainable_surface()
        from lerobot.utils.constants import (
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
            OBS_STATE,
        )

        prepared = dict(processed_batch)
        if bool(getattr(self.policy.config, "adapt_to_pi_aloha", False)):
            prepared[OBS_STATE] = prepared[OBS_STATE].clone()
        prepared = self.policy._prepare_batch(prepared)  # noqa: SLF001
        images, image_masks = self.policy.prepare_images(prepared)
        state = self.policy.prepare_state(prepared)
        raw_inputs = FlowModelInputs(
            images=tuple(images),
            image_masks=tuple(image_masks),
            language_tokens=prepared[OBS_LANGUAGE_TOKENS],
            language_masks=prepared[OBS_LANGUAGE_ATTENTION_MASK],
            state=state,
        )
        prefix_embeddings, prefix_masks, prefix_attention = self.model.embed_prefix(
            list(raw_inputs.images),
            list(raw_inputs.image_masks),
            raw_inputs.language_tokens,
            raw_inputs.language_masks,
            state=raw_inputs.state,
        )
        if int(getattr(self.model, "prefix_length", 0)) != 0:
            raise RuntimeError(
                "SmolVLA frozen-prefix replay currently requires prefix_length=0"
            )
        state_embeddings = self.model.state_proj(raw_inputs.state)
        state_sequence_length = 1 if state_embeddings.ndim == 2 else int(
            state_embeddings.shape[1]
        )
        if state_sequence_length < 1 or state_sequence_length >= prefix_embeddings.shape[1]:
            raise RuntimeError("SmolVLA produced an invalid state prefix length")
        return FrozenPrefixFlowModelInputs(
            frozen_prefix_embeddings=prefix_embeddings[
                :, :-state_sequence_length
            ].detach(),
            prefix_pad_masks=prefix_masks.detach(),
            prefix_attention_masks=prefix_attention.detach(),
            state=raw_inputs.state.detach(),
        )

    def _validate_frozen_prefix_trainable_surface(self) -> None:
        """Reject cached replay if omitted prefix encoders can receive gradients."""

        if self._frozen_prefix_surface_validated:
            return
        allowed_markers = (
            "vlm_with_expert.lm_expert",
            "state_proj",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        )
        unsupported = [
            name
            for name, parameter in self.policy.named_parameters()
            if parameter.requires_grad
            and not any(marker in name for marker in allowed_markers)
        ]
        if unsupported:
            raise RuntimeError(
                "SmolVLA compact Flow-SDE replay cannot omit trainable prefix "
                "encoders; unsupported trainable parameters include "
                f"{unsupported[:8]}"
            )
        self._frozen_prefix_surface_validated = True

    def _prefix_cache(
        self,
        inputs: FlowModelInputs | FrozenPrefixFlowModelInputs,
    ) -> tuple[torch.Tensor, Any]:
        modeling_module = import_module(type(self.model).__module__)
        make_att_2d_masks = modeling_module.make_att_2d_masks
        if isinstance(inputs, FrozenPrefixFlowModelInputs):
            state_embeddings = self.model.state_proj(inputs.state)
            if state_embeddings.ndim == 2:
                state_embeddings = state_embeddings[:, None, :]
            prefix_embeddings = torch.cat(
                [inputs.frozen_prefix_embeddings, state_embeddings], dim=1
            )
            prefix_masks = inputs.prefix_pad_masks
            prefix_attention = inputs.prefix_attention_masks
            if prefix_embeddings.shape[:2] != prefix_masks.shape:
                raise RuntimeError(
                    "SmolVLA frozen-prefix replay has inconsistent sequence lengths"
                )
        else:
            prefix_embeddings, prefix_masks, prefix_attention = (
                self.model.embed_prefix(
                    list(inputs.images),
                    list(inputs.image_masks),
                    inputs.language_tokens,
                    inputs.language_masks,
                    state=inputs.state,
                )
            )
        attention = make_att_2d_masks(prefix_masks, prefix_attention)
        positions = torch.cumsum(prefix_masks, dim=1) - 1
        _, cache = self.model.vlm_with_expert.forward(
            attention_mask=attention,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[prefix_embeddings, None],
            use_cache=self.model.config.use_cache,
            fill_kv_cache=True,
        )
        return prefix_masks, cache

    def _sample_model_noise(
        self,
        shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.model.sample_noise(shape, device).to(dtype=dtype)
