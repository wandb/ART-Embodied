"""Native execution acceptance is distinct from strict grammar diagnostics."""

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from art_embodied.integrations.pi0_fast import PI0FastPolicyAdapter
from art_embodied.policies.pi0_fast import PI0FastPolicy, _objective_action_token_rows
from tests.test_embodied_pi0_fast import _FakeRolloutPolicy


class NativeDecoder:
    config = SimpleNamespace(chunk_size=2)

    def decode_actions_with_fast(self, tokens):
        if tokens[0, 0] == 2:
            logging.warning("Error decoding tokens: synthetic native fallback")
        if tokens[0, 0] == 3:
            raise AssertionError("bad prefix")
        if tokens[0, 0] == 4:
            return torch.full((1, 2, 7), float("nan"))
        if tokens[0, 0] == 5:
            raise RuntimeError("unexpected runtime error")
        return torch.zeros(1, 2, 7)

    def detokenize_actions(self, tokens, **kwargs):
        return self.decode_actions_with_fast(tokens)


def make_policy():
    policy = PI0FastPolicy.__new__(PI0FastPolicy)
    policy.policy = NativeDecoder()
    policy.action_dim = 7
    policy.postprocessor = lambda normalized: normalized + 1
    return policy


def test_native_rejects_error_fallback_not_legitimate_zero():
    policy = make_policy()
    handlers = list(logging.getLogger().handlers)
    chunks, errors = policy.decode_action_tokens_native(
        torch.tensor([[1], [2], [3], [4]])
    )
    assert errors[0] is None
    torch.testing.assert_close(chunks[0], torch.ones(2, 7))
    assert all(errors[i] is not None and chunks[i].numel() == 0 for i in (1, 2, 3))
    assert logging.getLogger().handlers == handlers


def test_native_runtime_failure_propagates_and_removes_handler():
    handlers = list(logging.getLogger().handlers)
    with pytest.raises(RuntimeError, match="unexpected runtime"):
        make_policy().decode_action_tokens_native(torch.tensor([[5]]))
    assert logging.getLogger().handlers == handlers


def test_native_postprocessor_must_be_finite():
    policy = make_policy()
    policy.postprocessor = lambda normalized: normalized * float("nan")
    chunks, errors = policy.decode_action_tokens_native(torch.tensor([[1]]))
    assert chunks[0].numel() == 0
    assert "postprocessor" in errors[0]


def test_native_retains_generated_sequence_even_with_noncanonical_prefix():
    rows, valid, discarded, _ = _objective_action_token_rows(
        [[99, 40, 41, 12, 98]],
        prefix_ids=[10, 11],
        end_token_id=12,
        action_token_min_id=40,
        action_token_max_id=49,
        trim_invalid_prefix=False,
    )
    assert rows == [[99, 40, 41, 12]]
    assert valid == [False]
    assert discarded == [1]


@pytest.mark.parametrize("decode_failed", [False, True])
def test_native_adapter_uses_decode_status_not_grammar(decode_failed):
    policy = _FakeRolloutPolicy()
    policy.prepare_generated_action_tokens = lambda tokens, **kwargs: (
        tokens.tolist(),
        [False, False],
        [0, 0],
        [[False, False]] * 2,
    )
    policy.decode_action_tokens_native = lambda tokens: (
        [torch.empty(0, 7) if decode_failed else torch.ones(3, 7), torch.ones(3, 7)],
        ["decode error" if decode_failed else None, None],
    )
    predictions = PI0FastPolicyAdapter(
        policy=policy,
        temperature=1.5,
        action_decoder="native",
        invalid_action_handling="terminate_episode",
    ).predict_batch([{}, {}], tasks=["pick"] * 2, step=0)
    for i, pred in enumerate(predictions):
        rejected = i == 0 and decode_failed
        assert pred.action.metadata["action_decode_valid"] is not rejected
        assert pred.action.metadata["action_grammar_valid"] is False
        assert pred.action.metadata.get("terminate_episode", False) is rejected
        assert "primitive_loss_mask_sum" not in pred.action.metadata
        assert pred.action.metadata["token_loss_mask"] == [True, True]
        assert pred.native_action.shape == ((0, 7) if rejected else (2, 7))
    assert predictions[0].action.logprobs == pytest.approx([-0.2, -0.3])


def test_native_cannot_use_payload_only_scope():
    policy = _FakeRolloutPolicy()
    policy.rl_token_scope = "fast_payload"
    with pytest.raises(ValueError, match="generated_sequence"):
        PI0FastPolicyAdapter(policy=policy, action_decoder="native")


def test_native_metrics_distinguish_grammar_from_execution():
    from art_embodied.observability import _trajectory_metric_payload
    from art_embodied.trajectories import Action, EmbodiedTrajectory

    trajectories = [
        EmbodiedTrajectory(
            task="pick",
            actions=[
                Action(
                    step=0,
                    kind="token",
                    raw={"tokens": [1]},
                    metadata={
                        "action_grammar_valid": False,
                        "action_decode_valid": valid,
                    },
                )
            ],
        )
        for valid in (True, False)
    ]
    metrics = _trajectory_metric_payload("train_details", trajectories)
    assert metrics["train_details/action_grammar_valid_rate"] == 0
    assert metrics["train_details/action_decode_valid_rate"] == 0.5
    assert metrics["train_details/episodes_with_invalid_action_rate"] == 0.5


def test_native_evidence_is_not_pruned_by_legacy_grammar_flag():
    from art_embodied.config import EmbodiedExperimentConfig
    from art_embodied.rollout_worker import (
        TRAINABLE_ACTION_SELECTED_KEY,
        _select_training_action_payloads,
    )
    from art_embodied.trajectories import Action, EmbodiedTrajectory, Observation

    config = EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1]
        / "examples/embodied/pi0_fast_libero_spatial_grpo_development_h100.yaml"
    )
    trajectory = EmbodiedTrajectory(
        task="pick",
        observations=[
            Observation(step=i, kind="image", value={"image": i}) for i in range(2)
        ],
        actions=[
            Action(
                step=i,
                kind="token",
                raw={"tokens": [1]},
                metadata={
                    "action_grammar_valid": False,
                    "action_decode_valid": i == 0,
                    "terminate_episode": i == 1,
                },
            )
            for i in range(2)
        ],
    )
    _select_training_action_payloads(trajectory, phase="train", config=config)
    assert all(a.metadata[TRAINABLE_ACTION_SELECTED_KEY] for a in trajectory.actions)
    assert all("primitive_loss_mask_sum" not in a.metadata for a in trajectory.actions)
    assert all(o.value is not None for o in trajectory.observations)
