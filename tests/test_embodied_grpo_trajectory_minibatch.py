from __future__ import annotations

import asyncio
import math
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

from art_embodied.backends.action_token import ActionTokenGRPOBackend
from art_embodied.backends.local_process import (
    _PersistentGradientWorkerPool,
    _prepare_gradient_batches,
)
from art_embodied.config import EmbodiedExperimentConfig
from examples.embodied.pi0_fast_score_function_audit import (
    INITIAL,
    TreePolicy,
    backend_gradient,
    finite_difference,
    make_group,
    reward,
    token_probabilities,
)

LEAVES = ((2,), (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2,))


def config_payload():
    source = Path(__file__).parents[1] / (
        "examples/embodied/pi0_fast_spatial_native_score_sum_development_20260906.yaml"
    )
    raw = yaml.safe_load(source.read_text())
    raw["rollout"]["groups_per_update"] = 1
    raw["training"]["optimizer_steps_per_update"] = 4
    raw["training"]["schedule"] = {
        "type": "trajectory_minibatch",
        "minibatch_trajectories": 2,
        "shuffle_seed": 41,
    }
    return raw


def prepared():
    config = EmbodiedExperimentConfig.model_validate(config_payload())
    policy = TreePolicy(INITIAL, 0.2)
    backend = ActionTokenGRPOBackend(
        policy,
        optimizer=torch.optim.SGD(policy.parameters(), lr=0),
        normalize_advantages=True,
        advantage_normalization_scope="group",
        advantage_std_unbiased=True,
        advantage_epsilon=1e-6,
        rlinf_action_level_score_source="trajectory_reward",
        training_unit="action",
        loss_aggregation="seq_mean_token_sum",
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.28,
        logprob_microbatch_size=1,
        skip_optimizer_step_without_policy_gradient_signal=False,
    )
    batches, _ = _prepare_gradient_batches(
        [make_group(LEAVES, 0.2, "chunks", pad=True)],
        config=config,
        backend=backend,
        update_step=3,
    )
    return config, policy, backend, batches


async def gradient(policy, backend, batch, shards=3):
    result = torch.zeros_like(policy.weights)
    for rank in range(shards):
        examples = list(batch.examples[rank::shards])
        if not examples:
            continue
        await backend.train(
            [],
            _action_token_grpo_return_gradients=True,
            _action_token_grpo_precomputed_examples=examples,
            _action_token_grpo_precomputed_examples_prepared=True,
            _action_token_grpo_global_example_count=batch.denominator_examples,
            _action_token_grpo_global_token_count=batch.denominator_tokens,
        )
        result += policy.weights.grad.detach()
    return result


def test_complete_trajectories_keep_global_advantages_and_masks():
    _, _, _, batches = prepared()
    assert len(batches) == 4
    all_ids = []
    rewards = [reward(leaf) for leaf in LEAVES]
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / (len(rewards) - 1))
    for batch in batches:
        ids = {e.trajectory_index for e in batch.examples}
        assert len(ids) == batch.denominator_examples == 2
        all_ids.extend(ids)
        for trajectory in ids:
            examples = [e for e in batch.examples if e.trajectory_index == trajectory]
            assert len(examples) == len(LEAVES[trajectory])
            for example in examples:
                advantage = (rewards[trajectory] - mean) / (std + 1e-6)
                assert example.metadata["token_advantages"] == pytest.approx(
                    [advantage] * len(example.tokens)
                )
                assert example.metadata["token_loss_mask"] == [True, False]
    assert sorted(all_ids) == list(range(8))
    _, _, _, repeated = prepared()
    assert [[e.trajectory_index for e in b.examples] for b in batches] == [
        [e.trajectory_index for e in b.examples] for b in repeated
    ]


def test_frozen_model_minibatch_gradients_average_to_full_batch_gradient():
    _, policy, backend, batches = prepared()
    actual = sum(asyncio.run(gradient(policy, backend, b)) for b in batches) / 4
    reference = asyncio.run(backend_gradient(LEAVES, normalize=True))
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-7)


def scalar_minibatch_loss(weights, ids):
    rewards = [reward(leaf) for leaf in LEAVES]
    mean = sum(rewards) / 8
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / 7)
    total = 0.0
    for index in ids:
        advantage = (rewards[index] - mean) / (std + 1e-6)
        old = token_probabilities(INITIAL, LEAVES[index], 0.2)
        new = token_probabilities(weights, LEAVES[index], 0.2)
        for before, after in zip(old, new, strict=True):
            ratio = after / before
            total += max(-advantage * ratio, -advantage * min(1.28, max(0.8, ratio)))
    return total / len(ids)


def test_sequential_updates_match_independent_clipped_finite_difference():
    _, policy, backend, batches = prepared()
    old_logs = [[tuple(e.logprobs) for e in b.examples] for b in batches]
    off_policy = False
    for batch in batches:
        ids = sorted({e.trajectory_index for e in batch.examples})
        weights = tuple(policy.weights.detach().tolist())
        reference = finite_difference(lambda w: scalar_minibatch_loss(w, ids), weights)
        actual = asyncio.run(gradient(policy, backend, batch))
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
        off_policy |= any(abs(a - b) > 1e-4 for a, b in zip(weights, INITIAL))
        with torch.no_grad():
            policy.weights.add_(actual, alpha=-0.1)
    assert off_policy
    assert old_logs == [[tuple(e.logprobs) for e in b.examples] for b in batches]


def test_guard_only_requires_behavior_alignment_before_first_minibatch():
    config, _, _, _ = prepared()
    pool = object.__new__(_PersistentGradientWorkerPool)
    pool.config = config
    pool.job_index = 0
    assert pool._guard_alignment_for_job()
    pool.job_index = 1
    assert not pool._guard_alignment_for_job()


@pytest.mark.parametrize(
    "field,value",
    [
        ("score_source", "chunk_rewards"),
        ("loss_aggregation", "trajectory_mean"),
        ("filter_rewards", True),
    ],
)
def test_unsupported_objectives_fail_closed(field, value):
    raw = config_payload()
    raw["algorithm"][field] = value
    if field == "filter_rewards":
        raw["algorithm"].update(rewards_lower_bound=0.0, rewards_upper_bound=1.0)
    with pytest.raises(ValueError, match="GRPO trajectory_minibatch"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_missing_trajectory_fails_before_any_optimizer_update():
    config, _, backend, _ = prepared()
    with pytest.raises(ValueError, match="complete rollout batch"):
        _prepare_gradient_batches(
            [make_group(LEAVES[:-1], 0.2, "chunks")], config=config, backend=backend
        )


def test_coordinator_updates_four_times_and_resumes_outer_update_one(
    tmp_path, monkeypatch
):
    from art_embodied.backends import local_process
    from art_embodied.backends.action_token_gradients import _trainable_gradient_payload

    raw = config_payload()
    raw["storage"]["output_dir"] = str(tmp_path / "run")
    raw["runtime"]["worker_handoff_dir"] = str(tmp_path / "handoff")
    config = EmbodiedExperimentConfig.model_validate(raw)
    _, policy, backend, _ = prepared()
    backend.optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01, weight_decay=0)
    snapshots = []

    class CpuPool:
        reused_workers = False
        last_startup_seconds = 0.0
        last_offload_seconds = 0.0

        def __init__(self, **kwargs):
            pass

        def begin_update(self, **kwargs):
            pass

        def compute(self, batch, **kwargs):
            snapshots.append(policy.weights.detach().clone())
            policy.weights.grad = asyncio.run(gradient(policy, backend, batch))
            return [
                local_process._WorkerResult(
                    index=0,
                    device="cpu",
                    metrics={},
                    gradient_payload=_trainable_gradient_payload(policy),
                    elapsed_seconds=0.0,
                    policy_loads=0,
                    adapter_refreshes=1,
                )
            ]

        def finish_update(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(local_process, "_PersistentGradientWorkerPool", CpuPool)
    policy.save_checkpoint = lambda path: torch.save(
        policy.state_dict(), Path(path) / "toy-policy.pt"
    )
    coordinator = local_process.LocalProcessActionTokenBackend(
        config=config, policy=policy, backend=backend
    )
    result = coordinator._train_sync([make_group(LEAVES, 0.2, "chunks", pad=True)])
    assert result.step == coordinator.update_step == 1
    assert backend.step == 4
    assert result.metrics["embodied_action_token_grpo/optimizer_step_completed"] == 4
    assert result.metrics["embodied_action_token_schedule/subupdates"] == 4
    assert len(snapshots) == 4
    assert all(not torch.equal(a, b) for a, b in zip(snapshots, snapshots[1:]))
    assert {int(s["step"].item()) for s in backend.optimizer.state.values()} == {4}

    raw["storage"]["resume_from_checkpoint"] = result.checkpoint_path
    raw["evaluation"]["evaluate_before_training"] = False
    raw["evaluation"]["pre_training_success_gate"] = None
    raw["observability"]["wandb"].update(
        connection="resume", run_id="test0001", resume="must"
    )
    resumed_config = EmbodiedExperimentConfig.model_validate(raw)
    _, restored_policy, restored_backend, _ = prepared()
    restored_policy.load_checkpoint = lambda checkpoint: (
        restored_policy.load_state_dict(
            torch.load(Path(checkpoint["path"]) / "toy-policy.pt", weights_only=True)
        )
    )
    restored_backend.optimizer = torch.optim.AdamW(
        restored_policy.parameters(), lr=0.01
    )
    restored = local_process.LocalProcessActionTokenBackend(
        config=resumed_config, policy=restored_policy, backend=restored_backend
    )
    assert restored.update_step == 1
    assert restored.backend.step == 4
    assert {
        int(s["step"].item()) for s in restored_backend.optimizer.state.values()
    } == {4}
    torch.testing.assert_close(restored_policy.weights, policy.weights, rtol=0, atol=0)
