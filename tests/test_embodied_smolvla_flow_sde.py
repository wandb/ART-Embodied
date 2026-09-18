from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from art_embodied.policies.flow_policy import (
    FlowModelInputs,
    FrozenPrefixFlowModelInputs,
)
from art_embodied.policies.flow_sde import FlowSDESchedule, sample_flow_ode_chain
from art_embodied.policies.smolvla_flow_sde import SmolVLAFlowSDEBridge


def make_att_2d_masks(pad_masks, attention_masks):
    """Minimal stand-in for LeRobot's prefix mask builder."""

    del attention_masks
    batch, length = pad_masks.shape
    return torch.ones(batch, length, length, dtype=torch.bool)


class _FakeVLM:
    def forward(self, **kwargs):
        embeddings = kwargs["inputs_embeds"][0]
        return (embeddings, None), embeddings.mean(dim=1)


class _FakeSmolModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.velocity_scale = torch.nn.Parameter(torch.tensor(0.25))
        self.config = SimpleNamespace(use_cache=True)
        self.vlm_with_expert = _FakeVLM()
        self.state_proj = torch.nn.Linear(2, 2, bias=False)
        self.prefix_length = 0

    def sample_noise(self, shape, device):
        return torch.randn(shape, device=device)

    def embed_prefix(self, images, image_masks, tokens, token_masks, *, state):
        del images, image_masks, token_masks
        batch = tokens.shape[0]
        embeddings = torch.stack((tokens.float(), tokens.float()), dim=-1)
        masks = torch.ones(batch, tokens.shape[1], dtype=torch.bool)
        attention = torch.zeros_like(masks)
        if state is not None:
            state_embedding = self.state_proj(state)[:, None, :]
            embeddings = torch.cat((embeddings, state_embedding), dim=1)
            masks = torch.cat((masks, torch.ones(batch, 1, dtype=torch.bool)), dim=1)
            attention = torch.cat(
                (attention, torch.ones(batch, 1, dtype=torch.bool)), dim=1
            )
        return embeddings, masks, attention

    def denoise_step(self, *, x_t, prefix_pad_masks, past_key_values, timestep):
        del prefix_pad_masks, past_key_values
        return self.velocity_scale * x_t + timestep[:, None, None]


class _FakeSmolPolicy:
    name = "smolvla"

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            num_steps=4,
            chunk_size=3,
            max_action_dim=4,
            adapt_to_pi_aloha=False,
        )
        self.model = _FakeSmolModel()


def _inputs(batch_size: int = 2) -> FlowModelInputs:
    return FlowModelInputs(
        images=(torch.zeros(batch_size, 3, 4, 4),),
        image_masks=(torch.ones(batch_size, dtype=torch.bool),),
        language_tokens=torch.arange(batch_size * 3).reshape(batch_size, 3),
        language_masks=torch.ones(batch_size, 3, dtype=torch.bool),
        state=torch.zeros(batch_size, 2),
    )


def test_smolvla_rollout_and_rescore_use_the_same_transition_model(monkeypatch) -> None:
    policy = _FakeSmolPolicy()
    schedule = FlowSDESchedule(
        num_steps=4,
        noise_level=0.2,
        deterministic_sampler="native_euler",
    )
    bridge = SmolVLAFlowSDEBridge(
        policy,
        schedule=schedule,
        execution_horizon=2,
        action_dim=3,
    )
    inputs = _inputs()
    monkeypatch.setattr(bridge, "prepare_inputs", lambda _batch: inputs)
    initial_noise = torch.linspace(-1.0, 1.0, 24).reshape(2, 3, 4)
    generator = torch.Generator().manual_seed(7)

    rollout = bridge.sample(
        {},
        selected_index=torch.tensor([0, 3]),
        initial_noise=initial_noise,
        generator=generator,
    )
    rescored = bridge.rescore(rollout)
    old = rollout.transition.old_logprobs[:, :2, :3]

    torch.testing.assert_close(rescored, old, rtol=1e-6, atol=1e-6)
    assert rollout.actions.shape == (2, 3, 3)


def test_smolvla_rescore_backpropagates_through_native_velocity(monkeypatch) -> None:
    policy = _FakeSmolPolicy()
    bridge = SmolVLAFlowSDEBridge(
        policy,
        schedule=FlowSDESchedule(
            num_steps=4,
            noise_level=0.2,
            deterministic_sampler="native_euler",
        ),
        execution_horizon=2,
        action_dim=3,
    )
    monkeypatch.setattr(bridge, "prepare_inputs", lambda _batch: _inputs())
    rollout = bridge.sample(
        {},
        selected_index=torch.tensor([1, 2]),
        initial_noise=torch.randn(2, 3, 4),
        generator=torch.Generator().manual_seed(11),
    )

    bridge.rescore(rollout).sum().backward()

    gradient = policy.model.velocity_scale.grad
    assert gradient is not None
    assert torch.isfinite(gradient)
    assert gradient.abs() > 0


def test_smolvla_rollout_samples_initial_noise_on_compact_input_device(
    monkeypatch,
) -> None:
    policy = _FakeSmolPolicy()
    bridge = SmolVLAFlowSDEBridge(
        policy,
        schedule=FlowSDESchedule(
            num_steps=4,
            noise_level=0.2,
            deterministic_sampler="native_euler",
        ),
        execution_horizon=2,
        action_dim=3,
    )
    raw = _inputs()
    full_embeddings, masks, attention = policy.model.embed_prefix(
        list(raw.images),
        list(raw.image_masks),
        raw.language_tokens,
        raw.language_masks,
        state=raw.state,
    )
    compact = FrozenPrefixFlowModelInputs(
        frozen_prefix_embeddings=full_embeddings[:, :-1].detach(),
        prefix_pad_masks=masks,
        prefix_attention_masks=attention,
        state=raw.state,
    )
    monkeypatch.setattr(bridge, "prepare_inputs", lambda _batch: compact)

    rollout = bridge.sample(
        {},
        selected_index=torch.tensor([0, 3]),
        generator=torch.Generator().manual_seed(13),
    )

    assert rollout.actions.shape == (2, 3, 3)
    assert rollout.inputs.state.device.type == "cpu"


def test_smolvla_deterministic_chain_matches_native_euler_contract() -> None:
    policy = _FakeSmolPolicy()
    schedule = FlowSDESchedule(
        num_steps=4,
        noise_level=0.2,
        deterministic_sampler="native_euler",
    )
    bridge = SmolVLAFlowSDEBridge(
        policy,
        schedule=schedule,
        execution_horizon=2,
        action_dim=3,
    )
    inputs = _inputs()
    prefix_masks, cache = bridge._prefix_cache(inputs)  # noqa: SLF001
    initial_noise = torch.randn(2, 3, 4)

    def velocity(state, timestep):
        return policy.model.denoise_step(
            x_t=state,
            prefix_pad_masks=prefix_masks,
            past_key_values=cache,
            timestep=timestep,
        )

    generic = sample_flow_ode_chain(
        initial_noise,
        schedule=schedule,
        velocity_fn=velocity,
    )[:, -1]
    native = initial_noise
    for step in range(schedule.num_steps):
        timestep = torch.full((2,), 1.0 - step / schedule.num_steps)
        native = native - velocity(native, timestep) / schedule.num_steps

    torch.testing.assert_close(generic, native, rtol=0.0, atol=0.0)


def test_smolvla_bridge_ode_uses_full_native_chunk_before_action_truncation(
    monkeypatch,
) -> None:
    policy = _FakeSmolPolicy()
    bridge = SmolVLAFlowSDEBridge(
        policy,
        schedule=FlowSDESchedule(
            num_steps=4,
            noise_level=0.2,
            deterministic_sampler="native_euler",
        ),
        execution_horizon=2,
        action_dim=3,
    )
    monkeypatch.setattr(bridge, "prepare_inputs", lambda _batch: _inputs())
    initial_noise = torch.randn(2, 3, 4)

    actions = bridge.sample_native_ode({}, initial_noise=initial_noise)

    assert actions.shape == (2, 3, 3)


def test_smolvla_frozen_prefix_replay_recomputes_trainable_state_projection() -> None:
    policy = _FakeSmolPolicy()
    bridge = SmolVLAFlowSDEBridge(
        policy,
        schedule=FlowSDESchedule(
            num_steps=4,
            noise_level=0.2,
            deterministic_sampler="native_euler",
        ),
        execution_horizon=2,
        action_dim=3,
    )
    raw = _inputs()
    full_embeddings, masks, attention = policy.model.embed_prefix(
        list(raw.images),
        list(raw.image_masks),
        raw.language_tokens,
        raw.language_masks,
        state=raw.state,
    )
    compact = FrozenPrefixFlowModelInputs(
        frozen_prefix_embeddings=full_embeddings[:, :-1].detach(),
        prefix_pad_masks=masks,
        prefix_attention_masks=attention,
        state=raw.state,
    )

    _raw_masks, raw_cache = bridge._prefix_cache(raw)  # noqa: SLF001
    compact_masks, compact_cache = bridge._prefix_cache(compact)  # noqa: SLF001

    torch.testing.assert_close(compact_masks, masks)
    torch.testing.assert_close(compact_cache, raw_cache)
    compact_cache.sum().backward()
    assert policy.model.state_proj.weight.grad is not None


def test_frozen_prefix_replay_selects_and_concatenates_without_camera_tensors() -> None:
    inputs = FrozenPrefixFlowModelInputs(
        frozen_prefix_embeddings=torch.randn(2, 5, 8),
        prefix_pad_masks=torch.ones(2, 6, dtype=torch.bool),
        prefix_attention_masks=torch.zeros(2, 6, dtype=torch.bool),
        state=torch.randn(2, 4),
    )

    rows = [inputs.select(0), inputs.select(1)]
    combined = FrozenPrefixFlowModelInputs.concatenate(rows)

    torch.testing.assert_close(
        combined.frozen_prefix_embeddings, inputs.frozen_prefix_embeddings
    )
    torch.testing.assert_close(combined.state, inputs.state)
    assert combined.batch_size == 2


def test_frozen_prefix_replay_rejects_variable_language_geometry() -> None:
    short = FrozenPrefixFlowModelInputs(
        frozen_prefix_embeddings=torch.arange(24).reshape(1, 3, 8).float(),
        prefix_pad_masks=torch.tensor([[True, True, True, True]]),
        prefix_attention_masks=torch.tensor([[False, False, False, True]]),
        state=torch.randn(1, 4),
    )
    long = FrozenPrefixFlowModelInputs(
        frozen_prefix_embeddings=torch.arange(40).reshape(1, 5, 8).float(),
        prefix_pad_masks=torch.tensor([[True, True, True, True, True, True]]),
        prefix_attention_masks=torch.tensor(
            [[False, False, False, False, False, True]]
        ),
        state=torch.randn(1, 4),
    )

    with pytest.raises(ValueError, match="embedding shapes"):
        FrozenPrefixFlowModelInputs.concatenate([short, long])
