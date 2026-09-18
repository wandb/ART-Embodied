"""Joint action-token ratios against the qualified Flow-SDE loss geometry."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

torch = pytest.importorskip("torch")

from art_embodied import (
    Action,
    ActionTokenGRPOBackend,
    EmbodiedTrajectory,
    EmbodiedTrajectoryGroup,
)
from art_embodied.backends.action_token_ratios import action_chunk_log_ratio
from art_embodied.backends.flow_sde_grpo import flow_sde_grpo_loss
from art_embodied.config import EmbodiedExperimentConfig


class Policy(torch.nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.logprobs = torch.nn.Parameter(torch.tensor(delta).float() - 2.0)

    def action_token_logprobs(self, examples):
        return [self.logprobs[torch.tensor(e.tokens)] for e in examples]


def groups():
    trajectories = []
    # Unequal chunk and trajectory lengths catch accidental length normalization.
    for reward, chunks in ((1.0, [[0, 1], [2]]), (-1.0, [[3, 4, 5]])):
        trajectory = EmbodiedTrajectory(task="test", reward=reward)
        for step, tokens in enumerate(chunks):
            trajectory.actions.append(
                Action(
                    step=step,
                    kind="token",
                    raw={"tokens": tokens, "prompt": "test"},
                    logprobs={"token_logprobs": [-2.0] * len(tokens)},
                )
            )
        trajectories.append(trajectory)
    return [EmbodiedTrajectoryGroup(trajectories)]


def backend(policy, **overrides):
    kwargs = dict(
        optimizer=torch.optim.SGD(policy.parameters(), lr=0.01),
        importance_sampling_level="action_chunk",
        training_unit="action",
        action_advantage_mode="example",
        rlinf_action_level_score_source="trajectory_reward",
        loss_aggregation="seq_mean_token_sum",
        normalize_advantages=False,
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.28,
        logprob_microbatch_size=1,
    )
    kwargs.update(overrides)
    return ActionTokenGRPOBackend(policy, **kwargs)


@pytest.mark.parametrize("microbatch", [1, 2, 10])
@pytest.mark.parametrize("dual_clip", [None, 3.0])
@pytest.mark.parametrize("delta", [[0.0] * 6, [0.15, 0.15, -0.1, -0.1, -0.1, -0.1]])
def test_backend_loss_gradient_and_probe_match_flow_oracle(
    delta, microbatch, dual_clip
):
    policy = Policy(delta)
    trainer = backend(
        policy, logprob_microbatch_size=microbatch, clip_ratio_c=dual_clip
    )
    expected_parameters = policy.logprobs.detach().clone().requires_grad_(True)
    # Each padded row is one complete action chunk, not a whole trajectory.
    padded = torch.stack(
        [
            torch.stack(
                [
                    expected_parameters[0],
                    expected_parameters[1],
                    expected_parameters[0] * 0,
                ]
            ),
            torch.stack(
                [
                    expected_parameters[2],
                    expected_parameters[0] * 0,
                    expected_parameters[0] * 0,
                ]
            ),
            expected_parameters[3:6],
        ]
    ).unsqueeze(-1)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=torch.bool)
    expected, _ = flow_sde_grpo_loss(
        padded,
        torch.full_like(padded, -2.0),
        torch.tensor([1.0, 1.0, -1.0]),
        action_mask=mask,
        loss_denominator=2,
        clip_epsilon_high=0.28,
        clip_ratio_c=dual_clip,
    )
    expected.backward()
    probe = trainer.probe_logprob_metrics(groups())
    result = asyncio.run(
        trainer.train(groups(), _action_token_grpo_return_gradients=True)
    )
    prefix = "embodied_action_token_grpo/"
    assert result.metrics[prefix + "loss"] == pytest.approx(expected.item(), abs=1e-6)
    assert probe[prefix + "probe_surrogate_loss"] == pytest.approx(
        expected.item(), abs=1e-6
    )
    assert result.metrics[prefix + "action_chunk_importance_sampling"] == 1
    torch.testing.assert_close(policy.logprobs.grad, expected_parameters.grad)


def test_on_policy_joint_and_token_score_sum_gradients_identical():
    grads = []
    for mode in ("token", "action_chunk"):
        policy = Policy([0.0] * 6)
        asyncio.run(
            backend(policy, importance_sampling_level=mode).train(
                groups(), _action_token_grpo_return_gradients=True
            )
        )
        grads.append(policy.logprobs.grad.clone())
    torch.testing.assert_close(*grads, rtol=0, atol=0)


def test_distributed_shards_preserve_shared_trajectory_denominator():
    from art_embodied.backends.action_token import prepare_action_token_examples
    from art_embodied.backends.local_process import _attach_full_update_advantages

    policy = Policy([0.08, 0.08, -0.1, -0.03, -0.03, -0.03])
    trainer = backend(
        policy, normalize_advantages=True, advantage_normalization_scope="group"
    )
    examples, _ = prepare_action_token_examples(groups(), backend=trainer)
    _attach_full_update_advantages(examples, backend=trainer)
    assert all(e.metadata["group_advantage_prepared"] for e in examples)
    assert all("token_advantages" not in e.metadata for e in examples)
    asyncio.run(trainer.train(groups(), _action_token_grpo_return_gradients=True))
    expected = policy.logprobs.grad.clone()
    total = torch.zeros_like(expected)
    # Split one trajectory across workers; dividing by local trajectory/chunk
    # counts would change its weight. Production workers use the shared count.
    for subset in (examples[:1], examples[1:]):
        asyncio.run(
            trainer.train(
                [],
                _action_token_grpo_precomputed_examples=subset,
                _action_token_grpo_precomputed_examples_prepared=True,
                _action_token_grpo_global_example_count=2,
                _action_token_grpo_return_gradients=True,
            )
        )
        total += policy.logprobs.grad
    torch.testing.assert_close(total, expected)


def test_clip_catches_joint_movement_when_no_individual_token_exceeds_bounds():
    grads = []
    for mode in ("token", "action_chunk"):
        policy = Policy([0.15, 0.15, 0.0, -0.1, -0.1, -0.1])
        asyncio.run(
            backend(policy, importance_sampling_level=mode).train(
                groups(), _action_token_grpo_return_gradients=True
            )
        )
        grads.append(policy.logprobs.grad.clone())
    assert torch.count_nonzero(grads[0]) == 6
    torch.testing.assert_close(grads[1], torch.tensor([0.0, 0.0, -0.5, 0.0, 0.0, 0.0]))


@pytest.mark.parametrize("mask", [[True, False], [True], [0, 0], [1, 2], "bad"])
def test_reject_partial_or_invalid_masks(mask):
    example = SimpleNamespace(tokens=[0, 1], metadata={"token_loss_mask": mask})
    with pytest.raises(ValueError, match="partial token masks"):
        action_chunk_log_ratio(torch.zeros(2), torch.zeros(2), example)


def test_joint_ratio_detaches_old_and_does_not_length_normalize():
    current = torch.tensor([-1.9, -1.8], requires_grad=True)
    old = torch.tensor([-2.0, -2.0], requires_grad=True)
    ratio = action_chunk_log_ratio(
        current, old, SimpleNamespace(tokens=[0, 1], metadata={})
    )
    assert ratio.item() == pytest.approx(0.3)
    ratio.backward()
    assert old.grad is None
    torch.testing.assert_close(current.grad, torch.ones(2))


@pytest.mark.parametrize(
    "metadata", [{"action_spans": []}, {"token_advantages": [1.0, -1.0]}]
)
def test_reject_multi_action_or_per_token_advantages(metadata):
    with pytest.raises(ValueError):
        action_chunk_log_ratio(
            torch.zeros(2),
            torch.zeros(2),
            SimpleNamespace(tokens=[0, 1], metadata=metadata),
        )


def test_candidate_recipe_changes_only_ratio_geometry_and_has_sealed_pair():
    from art_embodied.evaluation import validate_paired_evaluation_configs

    root = Path(__file__).parents[1] / "examples/embodied"
    old = EmbodiedExperimentConfig.from_yaml(
        root / "pi0_fast_spatial_native_score_sum_minibatch_development_20260906.yaml"
    )
    new = EmbodiedExperimentConfig.from_yaml(
        root / "pi0_fast_spatial_joint_chunk_development_20260908.yaml"
    )
    for field in ("policy", "environment", "rollout", "training", "evaluation"):
        assert getattr(old, field) == getattr(new, field)
    assert new.algorithm.model_dump(
        exclude={"importance_sampling_level"}
    ) == old.algorithm.model_dump(exclude={"importance_sampling_level"})
    assert new.training.updates == 100
    assert new.storage.resume_from_checkpoint is None
    assert new.observability.wandb.run_id is None
    assert new.storage.output_dir != old.storage.output_dir
    pair = [
        EmbodiedExperimentConfig.from_yaml(
            root / f"pi0_fast_spatial_joint_chunk_sealed_{role}_20260908.yaml"
        )
        for role in ("baseline", "candidate")
    ]
    validate_paired_evaluation_configs(*pair)
    assert pair[0].evaluation.episodes == 192
    assert (
        pair[0].environment.kwargs["evaluation_state_manifest"]
        != new.environment.kwargs["evaluation_state_manifest"]
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"training_unit": "trajectory"},
        {"kl_coef": 0.1},
        {"reference_logprob_l2_coef": 0.1},
        {"policy_step_loss_weights": {0: 0.5}},
        {"action_dim_loss_weights": {0: 0.5}},
        {"loss_aggregation": "trajectory_mean"},
        {"rlinf_action_level_score_source": "chunk_rewards"},
    ],
)
def test_backend_rejects_unsupported_objective_combinations(overrides):
    with pytest.raises(ValueError):
        backend(Policy([0.0] * 6), **overrides)


def test_full_config_accepts_action_chunk_candidate_only_with_explicit_contract():
    path = (
        Path(__file__).parents[1]
        / "examples/embodied/pi0_fast_spatial_native_score_sum_minibatch_gate_20260906.yaml"
    )
    raw = yaml.safe_load(path.read_text())
    raw["algorithm"]["importance_sampling_level"] = "action_chunk"
    config = EmbodiedExperimentConfig.model_validate(raw)
    assert config.algorithm.importance_sampling_level == "action_chunk"
    raw["algorithm"]["training_unit"] = "trajectory"
    with pytest.raises(ValueError, match="action_chunk requires"):
        EmbodiedExperimentConfig.model_validate(raw)
