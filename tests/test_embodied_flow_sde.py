from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from art_embodied.policies.flow_sde import (  # noqa: E402
    FlowSDESchedule,
    flow_sde_transition,
    gaussian_logprobs,
    rescore_flow_sde_chain,
    rescore_flow_sde_transitions,
    sample_flow_ode_chain,
    sample_flow_sde_chain,
)


def test_deterministic_transition_matches_rlinf_v01_interpolation_order() -> None:
    schedule = FlowSDESchedule(num_steps=5, noise_level=0.3)
    state = torch.randn(3, 4, 2)
    velocity = torch.randn_like(state)

    transition = flow_sde_transition(
        state,
        velocity,
        2,
        schedule,
        stochastic=False,
    )

    timesteps = torch.linspace(1, 1 / schedule.num_steps, schedule.num_steps)
    timesteps = torch.cat([timesteps, torch.tensor([0.0])])
    time = timesteps[2]
    delta = timesteps[2] - timesteps[3]
    x0 = state - velocity * time
    x1 = state + velocity * (1 - time)
    expected = x0 * (1 - (time - delta)) + x1 * (time - delta)
    assert torch.equal(transition.mean, expected)
    assert torch.count_nonzero(transition.std) == 0


def test_rollout_and_rescore_share_exact_selected_logprob() -> None:
    schedule = FlowSDESchedule(num_steps=5, noise_level=0.3)
    generator = torch.Generator().manual_seed(7)
    initial = torch.randn(4, 3, 2, generator=generator)
    weight = torch.tensor(0.25)

    def velocity(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return state * weight + time[:, None, None]

    chain = sample_flow_sde_chain(
        initial,
        selected_indices=torch.tensor([0, 1, 2, 4]),
        schedule=schedule,
        velocity_fn=velocity,
        generator=generator,
    )
    rescored = rescore_flow_sde_chain(
        chain,
        schedule=schedule,
        velocity_fn=velocity,
    )

    torch.testing.assert_close(rescored, chain.selected_logprobs, rtol=1e-6, atol=1e-6)
    compact_rescore = rescore_flow_sde_transitions(
        chain.selected_transitions(),
        schedule=schedule,
        velocity_fn=velocity,
    )
    torch.testing.assert_close(compact_rescore, rescored, rtol=0, atol=0)
    assert (
        chain.selected_transitions().previous_states.numel() * 2 < chain.states.numel()
    )


def test_flow_sde_consumes_model_noise_at_every_denoise_step() -> None:
    schedule = FlowSDESchedule(num_steps=4, noise_level=0.5)
    initial = torch.zeros(2, 3, 1)
    calls = []

    def noise_fn(shape, device, dtype):
        calls.append((shape, device, dtype))
        return torch.full(shape, 0.25, device=device, dtype=dtype)

    chain = sample_flow_sde_chain(
        initial,
        selected_indices=2,
        schedule=schedule,
        velocity_fn=lambda state, time: torch.zeros_like(state),
        noise_fn=noise_fn,
    )

    assert len(calls) == schedule.num_steps
    assert all(shape == initial.shape for shape, _, _ in calls)
    assert torch.count_nonzero(chain.states[:, 1]) == 0
    assert torch.count_nonzero(chain.states[:, 2]) == 0
    assert torch.count_nonzero(chain.states[:, 3]) > 0


def test_flow_sde_rejects_two_rng_owners() -> None:
    schedule = FlowSDESchedule(num_steps=2, noise_level=0.5)
    initial = torch.zeros(1, 1, 1)

    with pytest.raises(ValueError, match="mutually exclusive"):
        sample_flow_sde_chain(
            initial,
            selected_indices=0,
            schedule=schedule,
            velocity_fn=lambda state, time: torch.zeros_like(state),
            generator=torch.Generator().manual_seed(1),
            noise_fn=lambda shape, device, dtype: torch.zeros(
                shape, device=device, dtype=dtype
            ),
        )


def test_stochastic_transition_matches_rlinf_v01_reference_equations() -> None:
    schedule = FlowSDESchedule(num_steps=5, noise_level=0.3)
    state = torch.randn(3, 4, 2)
    velocity = torch.randn_like(state)
    indices = torch.tensor([0, 2, 4])

    transition = flow_sde_transition(
        state,
        velocity,
        indices,
        schedule,
        stochastic=True,
    )

    timesteps = torch.linspace(1, 1 / schedule.num_steps, schedule.num_steps)
    timesteps = torch.cat([timesteps, torch.tensor([0.0])])
    time = timesteps[indices, None, None]
    delta = (timesteps[indices] - timesteps[indices + 1])[:, None, None]
    x0 = state - velocity * time
    x1 = state + velocity * (1 - time)
    sigmas = (
        schedule.noise_level
        * torch.sqrt(
            timesteps / (1 - torch.where(timesteps == 1, timesteps[1], timesteps))
        )[:-1]
    )
    sigma = sigmas[indices, None, None]
    expected_mean = x0 * (1 - (time - delta)) + x1 * (
        time - delta - sigma.square() * delta / (2 * time)
    )
    expected_std = (torch.sqrt(delta) * sigma).expand_as(state)

    torch.testing.assert_close(transition.mean, expected_mean, rtol=0, atol=0)
    torch.testing.assert_close(transition.std, expected_std, rtol=0, atol=0)


def test_gr00t_transition_matches_rlinf_n1d5_reference_equations() -> None:
    schedule = FlowSDESchedule(
        num_steps=4,
        noise_level=0.5,
        noise_time="zero",
    )
    state = torch.randn(3, 4, 2)
    velocity = torch.randn_like(state)
    indices = torch.tensor([0, 1, 3])

    transition = flow_sde_transition(
        state,
        velocity,
        indices,
        schedule,
        stochastic=True,
    )

    timesteps = torch.linspace(0, 1, schedule.num_steps + 1)
    time = timesteps[indices, None, None]
    delta = (timesteps[indices + 1] - timesteps[indices])[:, None, None]
    x0 = state - velocity * time
    x1 = state + velocity * (1 - time)
    sigmas = (
        schedule.noise_level
        * torch.sqrt(
            (1 - timesteps) / torch.where(timesteps == 0, timesteps[1], timesteps)
        )[:-1]
    )
    sigma = sigmas[indices, None, None]
    expected_mean = x0 * (
        1 - (time + delta) - sigma.square() * delta / (2 * (1 - time))
    ) + x1 * (time + delta)
    expected_std = (torch.sqrt(delta) * sigma).expand_as(state)

    torch.testing.assert_close(transition.mean, expected_mean, rtol=0, atol=0)
    torch.testing.assert_close(transition.std, expected_std, rtol=0, atol=0)


def test_gr00t_ode_chain_integrates_from_zero_to_one() -> None:
    schedule = FlowSDESchedule(
        num_steps=4,
        noise_level=0.5,
        noise_time="zero",
        deterministic_sampler="native_euler",
    )
    initial = torch.randn(2, 3, 1)

    def velocity(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return state.square() * 0.01 + time[:, None, None]

    states = sample_flow_ode_chain(
        initial,
        schedule=schedule,
        velocity_fn=velocity,
    )
    expected = initial
    for step in range(schedule.num_steps):
        time = torch.full((initial.shape[0],), step / schedule.num_steps)
        expected = expected + velocity(expected, time) / schedule.num_steps

    torch.testing.assert_close(states[:, -1], expected, rtol=1e-6, atol=1e-6)


def test_ode_chain_is_repeated_native_euler_sampling() -> None:
    schedule = FlowSDESchedule(num_steps=5, noise_level=0.3)
    initial = torch.randn(2, 3, 1)

    def velocity(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return state.square() * 0.01 + time[:, None, None]

    states = sample_flow_ode_chain(
        initial,
        schedule=schedule,
        velocity_fn=velocity,
    )
    expected = initial
    for step in range(schedule.num_steps):
        time = torch.full((initial.shape[0],), 1.0 - step / schedule.num_steps)
        expected = expected - velocity(expected, time) / schedule.num_steps

    torch.testing.assert_close(states[:, -1], expected, rtol=1e-6, atol=1e-6)


def test_rescore_has_policy_gradient_and_correct_direction() -> None:
    schedule = FlowSDESchedule(num_steps=4, noise_level=0.5)
    generator = torch.Generator().manual_seed(11)
    initial = torch.randn(2, 2, 1, generator=generator)
    rollout_weight = torch.tensor(0.1)

    def rollout_velocity(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return state * rollout_weight + time[:, None, None]

    chain = sample_flow_sde_chain(
        initial,
        selected_indices=1,
        schedule=schedule,
        velocity_fn=rollout_velocity,
        generator=generator,
    )
    train_weight = torch.tensor(0.1, requires_grad=True)

    def train_velocity(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return state * train_weight + time[:, None, None]

    logprobs = rescore_flow_sde_chain(
        chain,
        schedule=schedule,
        velocity_fn=train_velocity,
    )
    loss = -logprobs.mean()
    loss.backward()

    assert train_weight.grad is not None
    assert torch.isfinite(train_weight.grad)
    assert train_weight.grad.abs() > 0


def test_gaussian_logprobs_reject_deterministic_transition() -> None:
    schedule = FlowSDESchedule(num_steps=2, noise_level=0.3)
    sample = torch.zeros(1, 1, 1)
    transition = flow_sde_transition(
        sample,
        torch.ones_like(sample),
        0,
        schedule,
        stochastic=False,
    )

    with pytest.raises(ValueError, match="strictly positive"):
        gaussian_logprobs(sample, transition)
