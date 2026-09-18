from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from art_embodied.policies.flow_policy import (  # noqa: E402
    FlowSDERollout,
    GR00TFlowModelInputs,
)
from art_embodied.policies.flow_sde import FlowSDESchedule  # noqa: E402
from art_embodied.policies.gr00t_flow_sde import (  # noqa: E402
    GR00TN15FlowSDEBridge,
)


class _StateEncoder(torch.nn.Module):
    def forward(self, state, embodiment_id):
        return state + embodiment_id[:, None, None].to(state.dtype) * 0.01


class _ActionEncoder(torch.nn.Module):
    def __init__(self, weight: torch.nn.Parameter) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, actions, timesteps, embodiment_id):
        del embodiment_id
        return actions * self.weight + timesteps[:, None, None] * 0.001


class _DiT(torch.nn.Module):
    def forward(self, *, hidden_states, encoder_hidden_states, timestep):
        del timestep
        context = encoder_hidden_states.mean(dim=1, keepdim=True)
        return hidden_states + context


class _ActionDecoder(torch.nn.Module):
    def forward(self, model_output, embodiment_id):
        del embodiment_id
        return model_output


class _FakeN15ActionHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.2))
        self.config = SimpleNamespace(
            action_horizon=3,
            action_dim=2,
            add_pos_embed=False,
        )
        self.num_timestep_buckets = 1000
        self.state_encoder = _StateEncoder()
        self.action_encoder = _ActionEncoder(self.weight)
        self.future_tokens = torch.nn.Embedding(1, 2)
        torch.nn.init.zeros_(self.future_tokens.weight)
        self.model = _DiT()
        self.action_decoder = _ActionDecoder()

    def sample_noise(self, shape, device):
        return torch.zeros(shape, device=device)


def _inputs(batch_size: int = 2) -> GR00TFlowModelInputs:
    return GR00TFlowModelInputs(
        vision_language_features=torch.full((batch_size, 2, 2), 0.1),
        vision_language_attention_mask=torch.ones(batch_size, 2, dtype=torch.bool),
        state=torch.full((batch_size, 1, 2), 0.3),
        embodiment_id=torch.arange(batch_size),
    )


def test_gr00t_bridge_rejects_pi_time_direction() -> None:
    with pytest.raises(ValueError, match="noise_time='zero'"):
        GR00TN15FlowSDEBridge(
            _FakeN15ActionHead(),
            schedule=FlowSDESchedule(num_steps=4, noise_level=0.5),
            execution_horizon=2,
            action_dim=2,
        )


def test_gr00t_bridge_samples_and_rescores_selected_transition_exactly() -> None:
    head = _FakeN15ActionHead()
    bridge = GR00TN15FlowSDEBridge(
        head,
        schedule=FlowSDESchedule(
            num_steps=4,
            noise_level=0.5,
            noise_time="zero",
        ),
        execution_horizon=2,
        action_dim=2,
    )

    rollout = bridge.sample(
        _inputs(),
        selected_index=torch.tensor([1, 3]),
        generator=torch.Generator().manual_seed(7),
    )
    rescored = bridge.rescore(rollout)

    assert isinstance(rollout, FlowSDERollout)
    assert rollout.actions.shape == (2, 3, 2)
    assert rollout.batch_signature()[0][0] == "gr00t_n1d5_backbone"
    torch.testing.assert_close(
        rescored,
        rollout.transition.old_logprobs[:, :2, :2],
        rtol=0,
        atol=0,
    )
    (-rescored.mean()).backward()
    assert head.weight.grad is not None
    assert torch.isfinite(head.weight.grad)


def test_gr00t_inputs_round_trip_through_generic_flow_batch() -> None:
    inputs = _inputs()
    combined = GR00TFlowModelInputs.concatenate([inputs.select(0), inputs.select(1)])

    torch.testing.assert_close(
        combined.vision_language_features,
        inputs.vision_language_features,
    )
    torch.testing.assert_close(combined.state, inputs.state)
    torch.testing.assert_close(combined.embodiment_id, inputs.embodiment_id)
