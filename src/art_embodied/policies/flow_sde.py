"""Sampler-aligned Flow-SDE transition math for flow-matching policies.

This module does not know about PI0, SmolVLA, or a simulator.  A policy plugin
supplies the velocity field; this module owns the stochastic transition used
both during rollout and teacher-forced rescoring.  Keeping those two paths on
one implementation prevents optimizing a score that is unrelated to the
native action sampler.

The transition formulas and conformance-sensitive operation ordering are
reimplemented from RLinf release/v0.1 at commit
9df6dc80dc729a6caccab92dd676004e51b1d3a2 (Apache-2.0, Copyright 2025 The
RLinf Authors). The framework-neutral API, transition records, validation, and
rollout/rescore separation are ART-Embodied modifications and extensions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True, slots=True)
class FlowSDESchedule:
    """Discrete ODE-to-SDE schedule used by the PI-RL Flow-SDE method."""

    num_steps: int
    noise_level: float
    noise_time: Literal["zero", "one"] = "one"
    deterministic_sampler: Literal["rlinf_interpolation", "native_euler"] = (
        "rlinf_interpolation"
    )

    def __post_init__(self) -> None:
        if self.num_steps < 1:
            raise ValueError("num_steps must be positive")
        if self.noise_level <= 0.0:
            raise ValueError("noise_level must be positive")


@dataclass(frozen=True, slots=True)
class GaussianTransition:
    """Mean and standard deviation of one reverse-flow transition."""

    mean: torch.Tensor
    std: torch.Tensor


@dataclass(frozen=True, slots=True)
class FlowSDEChain:
    """A sampled denoising chain and the transition selected for policy RL."""

    states: torch.Tensor
    selected_indices: torch.Tensor
    selected_logprobs: torch.Tensor

    @property
    def actions(self) -> torch.Tensor:
        return self.states[:, -1]

    def selected_transitions(self) -> "FlowSDETransitionRecord":
        """Discard deterministic chain states not needed by non-joint rescore."""

        batch = torch.arange(self.states.shape[0], device=self.states.device)
        return FlowSDETransitionRecord(
            previous_states=self.states[batch, self.selected_indices],
            next_states=self.states[batch, self.selected_indices + 1],
            selected_indices=self.selected_indices,
            old_logprobs=self.selected_logprobs,
        )


@dataclass(frozen=True, slots=True)
class FlowSDETransitionRecord:
    """Minimal sufficient record for one selected transition per sample."""

    previous_states: torch.Tensor
    next_states: torch.Tensor
    selected_indices: torch.Tensor
    old_logprobs: torch.Tensor

    def to(self, device: torch.device | str) -> "FlowSDETransitionRecord":
        return FlowSDETransitionRecord(
            previous_states=self.previous_states.to(device),
            next_states=self.next_states.to(device),
            selected_indices=self.selected_indices.to(device),
            old_logprobs=self.old_logprobs.to(device),
        )

    def cpu(self) -> "FlowSDETransitionRecord":
        return self.to("cpu")


VelocityFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
NoiseFunction = Callable[[torch.Size, torch.device, torch.dtype], torch.Tensor]


def flow_timesteps(
    schedule: FlowSDESchedule,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the model-native time coordinate for each denoising state.

    PI and SmolVLA place noise at ``t=1`` and integrate toward zero. GR00T
    places noise at ``t=0`` and integrates toward one. The generated actions
    have the same semantic direction, but their velocity fields receive
    opposite time coordinates.
    """

    if schedule.noise_time == "zero":
        return torch.linspace(
            0.0,
            1.0,
            schedule.num_steps + 1,
            device=device,
            dtype=dtype,
        )
    if schedule.deterministic_sampler == "native_euler":
        # SmolVLA constructs each timestep from a Python float. Building the
        # same values explicitly avoids a measurable action drift from
        # torch.linspace after repeated nonlinear denoising calls.
        delta = -1.0 / schedule.num_steps
        steps = torch.tensor(
            [1.0 + step * delta for step in range(schedule.num_steps)],
            device=device,
            dtype=dtype,
        )
    else:
        steps = torch.linspace(
            1.0,
            1.0 / schedule.num_steps,
            schedule.num_steps,
            device=device,
            dtype=dtype,
        )
    return torch.cat((steps, torch.zeros(1, device=device, dtype=dtype)))


def flow_sde_transition(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    step_indices: int | torch.Tensor,
    schedule: FlowSDESchedule,
    *,
    stochastic: bool | torch.Tensor,
) -> GaussianTransition:
    """Construct the exact transition used by rollout and rescoring.

    ``stochastic=False`` is the native Euler ODE step.  ``stochastic=True``
    applies the ODE-to-SDE conversion from PI-RL.  Tensor-valued flags support
    selecting a different denoising transition for each batch item.
    """

    if x_t.shape != velocity.shape:
        raise ValueError(
            f"x_t and velocity must have the same shape, got {x_t.shape} and "
            f"{velocity.shape}"
        )
    if x_t.ndim < 1:
        raise ValueError("x_t must include a batch dimension")

    indices = _batch_indices(step_indices, batch_size=x_t.shape[0], device=x_t.device)
    if torch.any(indices < 0) or torch.any(indices >= schedule.num_steps):
        raise ValueError("step_indices are outside the denoising schedule")

    timesteps = flow_timesteps(schedule, device=x_t.device, dtype=x_t.dtype)
    t = _expand_batch_scalar(timesteps[indices], x_t)
    if schedule.noise_time == "zero":
        delta = _expand_batch_scalar(timesteps[indices + 1] - timesteps[indices], x_t)
        t_next = t + delta
    else:
        # Preserve the established PI/SmolVLA operation order exactly. Tiny
        # floating-point changes here invalidate rollout/rescore tensor parity.
        delta = _expand_batch_scalar(timesteps[indices] - timesteps[indices + 1], x_t)
        t_next = t - delta

    x0_prediction = x_t - velocity * t
    x1_prediction = x_t + velocity * (1.0 - t)

    # This is deliberately written in RLinf's interpolation order. Although it
    # is algebraically equivalent to an Euler step, the operation order is part
    # of the discrete Flow-SDE rollout probability model and must be reproduced
    # by teacher-forced rescoring.
    deterministic_x0_weight = 1.0 - t_next
    deterministic_x1_weight = t_next
    euler_delta = 1.0 / schedule.num_steps
    if schedule.noise_time == "one":
        euler_delta = -euler_delta
    if schedule.deterministic_sampler == "native_euler":
        deterministic_mean = x_t + euler_delta * velocity
    else:
        deterministic_mean = (
            x0_prediction * deterministic_x0_weight
            + x1_prediction * deterministic_x1_weight
        )
    deterministic_std = torch.zeros_like(x_t)

    if schedule.noise_time == "zero":
        # GR00T parameterizes noise at t=0 and data at t=1. Preserve RLinf's
        # zero-denominator substitution and interpolation order exactly.
        sigma_denominator_time = torch.where(
            timesteps[:-1] == 0.0,
            timesteps[1],
            timesteps[:-1],
        )
        sigmas = schedule.noise_level * torch.sqrt(
            (1.0 - timesteps[:-1]) / sigma_denominator_time
        )
        sigma = _expand_batch_scalar(sigmas[indices], x_t)
        stochastic_x0_weight = 1.0 - t_next - sigma.square() * delta / (2.0 * (1.0 - t))
        stochastic_mean = x0_prediction * stochastic_x0_weight + x1_prediction * t_next
    else:
        # At t=1, PI-RL uses the second schedule point in the denominator
        # instead of dividing by zero.
        sigma_denominator_time = torch.where(
            timesteps[:-1] == 1.0,
            timesteps[1],
            timesteps[:-1],
        )
        sigmas = schedule.noise_level * torch.sqrt(
            timesteps[:-1] / (1.0 - sigma_denominator_time)
        )
        sigma = _expand_batch_scalar(sigmas[indices], x_t)
        stochastic_x1_weight = t_next - sigma.square() * delta / (2.0 * t)
        stochastic_mean = (
            x0_prediction * (1.0 - t_next) + x1_prediction * stochastic_x1_weight
        )
    stochastic_std = torch.sqrt(delta) * sigma

    stochastic_mask = _batch_mask(
        stochastic,
        batch_size=x_t.shape[0],
        device=x_t.device,
        target=x_t,
    )
    return GaussianTransition(
        mean=torch.where(stochastic_mask, stochastic_mean, deterministic_mean),
        std=torch.where(stochastic_mask, stochastic_std, deterministic_std),
    )


def gaussian_logprobs(
    sample: torch.Tensor,
    transition: GaussianTransition,
) -> torch.Tensor:
    """Elementwise Gaussian log probability for a stochastic transition."""

    if sample.shape != transition.mean.shape or sample.shape != transition.std.shape:
        raise ValueError("sample, mean, and std must have identical shapes")
    if torch.any(transition.std <= 0.0):
        raise ValueError("Gaussian logprobs require strictly positive std")
    variance_term = ((sample - transition.mean) / transition.std).square()
    return (
        -torch.log(transition.std)
        - 0.5 * torch.log(torch.full_like(sample, 2.0 * torch.pi))
        - 0.5 * variance_term
    )


@torch.no_grad()
def sample_flow_ode_chain(
    initial_noise: torch.Tensor,
    *,
    schedule: FlowSDESchedule,
    velocity_fn: VelocityFunction,
) -> torch.Tensor:
    """Run LeRobot's native Euler sampler and retain every denoising state.

    Native deployment uses one Euler step in the direction declared by
    ``noise_time``. Keep that operation
    order separate from :func:`flow_sde_transition`, whose deterministic steps
    intentionally reproduce RLinf's algebraically equivalent interpolation.
    """

    timesteps = flow_timesteps(
        schedule,
        device=initial_noise.device,
        dtype=initial_noise.dtype,
    )
    state = initial_noise
    states = [state]
    for step in range(schedule.num_steps):
        time = timesteps[step].expand(initial_noise.shape[0])
        velocity = velocity_fn(state, time)
        direction = 1.0 if schedule.noise_time == "zero" else -1.0
        state = state + (direction / schedule.num_steps) * velocity
        states.append(state)
    return torch.stack(states, dim=1)


@torch.no_grad()
def sample_flow_sde_chain(
    initial_noise: torch.Tensor,
    *,
    selected_indices: int | torch.Tensor,
    schedule: FlowSDESchedule,
    velocity_fn: VelocityFunction,
    generator: torch.Generator | None = None,
    noise_fn: NoiseFunction | None = None,
) -> FlowSDEChain:
    """Sample one chain with one stochastic transition per batch item."""

    indices = _batch_indices(
        selected_indices,
        batch_size=initial_noise.shape[0],
        device=initial_noise.device,
    )
    timesteps = flow_timesteps(
        schedule,
        device=initial_noise.device,
        dtype=initial_noise.dtype,
    )
    state = initial_noise
    states = [state]
    selected_logprobs = torch.empty_like(initial_noise)

    for step in range(schedule.num_steps):
        time = timesteps[step].expand(initial_noise.shape[0])
        velocity = velocity_fn(state, time)
        selected = indices == step
        transition = flow_sde_transition(
            state,
            velocity,
            step,
            schedule,
            stochastic=selected,
        )
        if noise_fn is None:
            noise = torch.randn(
                state.shape,
                dtype=state.dtype,
                device=state.device,
                generator=generator,
            )
        else:
            if generator is not None:
                raise ValueError("generator and noise_fn are mutually exclusive")
            noise = noise_fn(state.shape, state.device, state.dtype)
            if noise.shape != state.shape:
                raise ValueError(
                    "noise_fn returned the wrong shape: "
                    f"expected={tuple(state.shape)}, actual={tuple(noise.shape)}"
                )
            noise = noise.to(device=state.device, dtype=state.dtype)
        next_state = transition.mean + noise * transition.std
        if torch.any(selected):
            stochastic_transition = flow_sde_transition(
                state[selected],
                velocity[selected],
                step,
                schedule,
                stochastic=True,
            )
            selected_logprobs[selected] = gaussian_logprobs(
                next_state[selected],
                stochastic_transition,
            )
        state = next_state
        states.append(state)

    return FlowSDEChain(
        states=torch.stack(states, dim=1),
        selected_indices=indices,
        selected_logprobs=selected_logprobs,
    )


def rescore_flow_sde_chain(
    chain: FlowSDEChain,
    *,
    schedule: FlowSDESchedule,
    velocity_fn: VelocityFunction,
) -> torch.Tensor:
    """Recompute selected transition logprobs under the current policy."""

    if chain.states.ndim < 3:
        raise ValueError("chain.states must have shape [batch, steps + 1, ...]")
    if chain.states.shape[1] != schedule.num_steps + 1:
        raise ValueError("chain length does not match schedule.num_steps")

    return rescore_flow_sde_transitions(
        chain.selected_transitions(),
        schedule=schedule,
        velocity_fn=velocity_fn,
    )


def rescore_flow_sde_transitions(
    record: FlowSDETransitionRecord,
    *,
    schedule: FlowSDESchedule,
    velocity_fn: VelocityFunction,
) -> torch.Tensor:
    """Teacher-force compact selected transitions under the current policy."""

    previous = record.previous_states
    following = record.next_states
    if previous.shape != following.shape or previous.shape != record.old_logprobs.shape:
        raise ValueError(
            "previous_states, next_states, and old_logprobs must have identical shapes"
        )
    if record.selected_indices.shape != (previous.shape[0],):
        raise ValueError("selected_indices must have shape [batch]")
    timesteps = flow_timesteps(
        schedule,
        device=previous.device,
        dtype=previous.dtype,
    )
    time = timesteps[record.selected_indices]
    velocity = velocity_fn(previous, time)
    transition = flow_sde_transition(
        previous,
        velocity,
        record.selected_indices,
        schedule,
        stochastic=True,
    )
    return gaussian_logprobs(following, transition)


def _batch_indices(
    indices: int | torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(indices, int):
        return torch.full((batch_size,), indices, dtype=torch.long, device=device)
    result = indices.to(device=device, dtype=torch.long)
    if result.ndim == 0:
        result = result.expand(batch_size)
    if result.shape != (batch_size,):
        raise ValueError(
            f"step_indices must have shape [{batch_size}], got {tuple(result.shape)}"
        )
    return result


def _batch_mask(
    value: bool | torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    target: torch.Tensor,
) -> torch.Tensor:
    if isinstance(value, bool):
        mask = torch.full((batch_size,), value, dtype=torch.bool, device=device)
    else:
        mask = value.to(device=device, dtype=torch.bool)
        if mask.ndim == 0:
            mask = mask.expand(batch_size)
        if mask.shape != (batch_size,):
            raise ValueError(
                f"stochastic mask must have shape [{batch_size}], got {tuple(mask.shape)}"
            )
    return _expand_batch_scalar(mask, target)


def _expand_batch_scalar(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return value.reshape((target.shape[0],) + (1,) * (target.ndim - 1))
