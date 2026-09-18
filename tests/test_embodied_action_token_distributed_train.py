from __future__ import annotations

import asyncio
from collections import UserDict
import copy
import gc
import math
from pathlib import Path
import pickle
from types import SimpleNamespace
import weakref

import pytest

torch = pytest.importorskip("torch")

from art_embodied import (
    Action,
    ActionTokenGRPOBackend,
    ActionTokenGSPOBackend,
    EmbodiedTrajectory,
    EmbodiedTrajectoryGroup,
)
from art_embodied.backends.action_token import (
    _example_bucket_loss_weight_tensor,
    _grpo_metrics,
    _maybe_guard_pre_update_logprob_alignment,
    _trainable_gradient_payload,
    apply_action_token_gradient_payloads,
    extract_action_token_examples,
    extract_trajectory_action_token_examples,
    load_action_token_gradient_payload,
    prepare_action_token_examples,
    rescore_action_token_examples,
)
from art_embodied.backends.action_token_worker import (
    _add_pi0_fast_sft_anchor_gradients,
    _gradient_job_config,
    _run_rescore_job,
    _worker_config,
)
from art_embodied.backends.local_process import (
    _attach_full_update_advantages,
    _attach_task_balance_weights,
    _partition_examples,
    _pi0_fast_sft_anchor_tasks_for_worker,
)
from art_embodied.config import EmbodiedExperimentConfig


class TinyTokenPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(
            torch.tensor([0.05, -0.05], dtype=torch.float32)
        )

    def action_token_logprobs(self, examples):
        logprobs = torch.log_softmax(self.logits, dim=-1)
        return [
            logprobs[torch.as_tensor([int(token) for token in example.tokens])]
            for example in examples
        ]


class IndependentTokenLogprobPolicy(torch.nn.Module):
    """Expose independent scalar logprobs for objective-mask direction tests."""

    def __init__(self, token_count: int) -> None:
        super().__init__()
        self.logprobs = torch.nn.Parameter(torch.zeros(token_count))

    def action_token_logprobs(self, examples):
        return [
            self.logprobs[torch.as_tensor(example.tokens, dtype=torch.long)]
            for example in examples
        ]


class BatchSensitiveTokenPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))
        self.calls = 0
        self.grad_enabled_calls: list[bool] = []

    def action_token_logprobs(self, examples):
        self.calls += 1
        self.grad_enabled_calls.append(torch.is_grad_enabled())
        value = self.bias - float(len(examples))
        return [value.expand(len(example.tokens)) for example in examples]


class WrappedBatchSensitiveTokenPolicy:
    def __init__(self) -> None:
        self.model = BatchSensitiveTokenPolicy()
        self.device = "cpu"

    def action_token_logprobs(self, examples):
        return self.model.action_token_logprobs(examples)


def test_action_rescore_can_materialize_deferred_rollout_logprobs() -> None:
    groups = [_group(1.0, 0.0, "pick red cube")]
    for trajectory in groups[0]:
        for action in trajectory.actions:
            action.logprobs = None
    policy = TinyTokenPolicy()
    backend = ActionTokenGRPOBackend(
        policy,
        training_unit="action",
        require_prompts=True,
        precalculate_logprobs=True,
        rollout_logprob_source="recomputed_current_policy",
    )

    examples, _ = prepare_action_token_examples(groups, backend=backend)

    assert all(example.logprobs is None for example in examples)
    report = rescore_action_token_examples(
        policy,
        examples,
        device="cpu",
        training_unit="action",
        microbatch_size=1,
        pad_to_batch_size=1,
        source="recomputed_current_policy",
    )
    assert report["tokens_updated"] == 2
    assert report["previous_abs_delta_mean"] == 0.0
    assert all(example.logprobs is not None for example in examples)


def test_trajectory_rescore_uses_training_microbatch_geometry() -> None:
    examples = extract_trajectory_action_token_examples(
        [_gspo_group(1.0, 0.0, "pick red cube")],
        require_logprobs=True,
        require_prompts=True,
    )
    for example in examples:
        example.logprobs = [-1.0] * len(example.tokens)

    report = rescore_action_token_examples(
        BatchSensitiveTokenPolicy(),
        examples,
        device="cpu",
        training_unit="trajectory",
        microbatch_size=2,
        pad_to_batch_size=2,
        source="recomputed_current_policy",
    )

    assert report["previous_abs_delta_mean"] == pytest.approx(1.0)
    assert report["policy_logprob_calls"] == 2
    assert all(
        value == -2.0 for example in examples for value in (example.logprobs or [])
    )
    assert all(
        example.metadata["old_logprob_source"] == "recomputed_current_policy"
        for example in examples
    )


def test_single_process_gspo_rescores_old_logprobs_with_training_geometry() -> None:
    groups = [_gspo_group(1.0, 0.0, "pick red cube")]
    for trajectory in groups[0]:
        for action in trajectory.actions:
            action.logprobs = {"token_logprobs": [-1.0] * len(action.raw["tokens"])}
    policy = BatchSensitiveTokenPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)
    backend = ActionTokenGSPOBackend(
        policy,
        optimizer=optimizer,
        require_prompts=True,
        normalize_advantages=True,
        advantage_normalization_scope="group",
        advantage_std_unbiased=False,
        loss_aggregation="trajectory_mean",
        precalculate_logprobs=True,
        rollout_logprob_source="recomputed_current_policy",
        logprob_eval_mode=True,
        logprob_microbatch_size=8,
        train_logprob_microbatch_size=2,
    )

    result = asyncio.run(backend.train(groups))

    assert result.metrics[
        "embodied_action_token_gspo/old_logprob_rescore_previous_abs_delta_mean"
    ] == pytest.approx(1.0)
    assert result.metrics["embodied_action_token_gspo/ratio_mean"] == pytest.approx(1.0)
    assert result.metrics["embodied_action_token_gspo/clip_fraction"] == pytest.approx(
        0.0
    )
    assert result.metrics[
        "embodied_action_token_gspo/optimizer_step_completed"
    ] == pytest.approx(1.0)


def test_single_process_gspo_rejects_singleton_before_rescore() -> None:
    policy = BatchSensitiveTokenPolicy()
    backend = ActionTokenGSPOBackend(
        policy,
        precalculate_logprobs=True,
        rollout_logprob_source="recomputed_current_policy",
    )
    singleton = EmbodiedTrajectoryGroup(
        [_multi_action_trajectory(1, 1.0, "pick red cube")]
    )

    with pytest.raises(ValueError, match="singleton"):
        asyncio.run(backend.train([singleton]))

    assert policy.calls == 0


def test_worker_rescore_roundtrip_preserves_prepared_examples(tmp_path: Path) -> None:
    examples = extract_trajectory_action_token_examples(
        [_gspo_group(1.0, 0.0, "pick red cube")],
        require_logprobs=True,
        require_prompts=True,
    )
    for example in examples:
        example.logprobs = [-1.0] * len(example.tokens)
    input_path = tmp_path / "examples.pkl"
    output_path = tmp_path / "rescored.pkl"
    with input_path.open("wb") as handle:
        pickle.dump(examples, handle)

    policy = WrappedBatchSensitiveTokenPolicy()
    result, snapshot = _run_rescore_job(
        {
            "policy_snapshot": str(tmp_path / "snapshot"),
            "examples_path": str(input_path),
            "output_examples_path": str(output_path),
            "source": "recomputed_current_policy",
            "worker_index": 0,
        },
        policy=policy,
        backend=SimpleNamespace(
            logprob_eval_mode=False,
            train_logprob_microbatch_size=2,
            logprob_microbatch_size=8,
            training_unit="trajectory",
        ),
        current_snapshot=str(tmp_path / "snapshot"),
    )

    with output_path.open("rb") as handle:
        rescored = pickle.load(handle)
    assert snapshot == str(tmp_path / "snapshot")
    assert result["report"]["previous_abs_delta_mean"] == pytest.approx(1.0)
    assert policy.model.grad_enabled_calls == [True, True]
    assert all(
        value == -2.0 for example in rescored for value in (example.logprobs or [])
    )


def test_alignment_guard_releases_autograd_graphs_after_each_check() -> None:
    examples = extract_trajectory_action_token_examples(
        [_gspo_group(1.0, 0.0, "pick red cube")],
        require_logprobs=True,
        require_prompts=True,
    )
    saved_tensor_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    class SaveActivation(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            activation = torch.ones(1024, dtype=value.dtype, device=value.device)
            ctx.save_for_backward(activation)
            saved_tensor_refs.append(weakref.ref(activation))
            return value.clone()

        @staticmethod
        def backward(ctx, gradient):
            return gradient

    class SavedActivationPolicy(TinyTokenPolicy):
        def action_token_logprobs(self, batch):
            logits = SaveActivation.apply(self.logits)
            logprobs = torch.log_softmax(logits, dim=-1)
            return [
                logprobs[torch.as_tensor([int(token) for token in example.tokens])]
                for example in batch
            ]

    policy = SavedActivationPolicy()
    for example, row in zip(
        examples,
        policy.action_token_logprobs(examples),
        strict=True,
    ):
        example.logprobs = row.detach().tolist()

    # Ignore activations created while preparing the exact old logprobs.
    saved_tensor_refs.clear()
    metrics = _maybe_guard_pre_update_logprob_alignment(
        policy=policy,
        examples=examples,
        device="cpu",
        importance_sampling_level="token",
        training_unit="action",
        loss_aggregation="trajectory_mean",
        microbatch_size=1,
        kl_tolerance=0.02,
        ratio_tolerance=0.02,
    )

    gc.collect()
    assert metrics is not None
    assert metrics["old_new_logprobs_aligned"] is True
    assert saved_tensor_refs
    assert not any(reference() is not None for reference in saved_tensor_refs)


def test_gspo_clip_metrics_use_asymmetric_sequence_thresholds() -> None:
    metrics = _grpo_metrics(
        groups=[],
        examples=[],
        loss=0.0,
        advantages=[],
        ratios=[torch.tensor([0.9995, 1.0005])],
        approx_kls=[torch.zeros(2)],
        reference_l2_penalties=[],
        clip_hits=[torch.ones(2)],
        token_counts=[1.0, 1.0],
        normalize_advantages=True,
        advantage_normalization_scope="group",
        advantage_std_unbiased=False,
        reward_filter_report=None,
        importance_sampling_level="sequence",
        training_unit="trajectory",
        loss_aggregation="trajectory_mean",
        kl_coef=0.0,
        clip_epsilon=0.2,
        clip_epsilon_low=3e-4,
        clip_epsilon_high=4e-4,
        reference_logprob_l2_coef=0.0,
    )

    assert metrics["embodied_action_token_grpo/clip_low_fraction"] == 0.5
    assert metrics["embodied_action_token_grpo/clip_high_fraction"] == 0.5


def test_distributed_worker_projection_preserves_rollout_geometry(
    tmp_path: Path,
) -> None:
    config = EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_libero_object_grpo_lora_lr1e4_shared_rollout.yaml"
    )

    worker = _worker_config(
        {
            "config": config.model_dump(mode="json"),
            "policy_snapshot": str(tmp_path / "adapter"),
        }
    )

    assert worker.runtime.rollout_devices == config.runtime.rollout_devices
    assert worker.rollout.workers == config.rollout.workers
    assert worker.runtime.training_devices == ["cuda:0"]
    assert worker.runtime.distributed_training is False
    assert worker.policy.device == "cuda:0"


def test_pi0_fast_worker_projection_uses_native_snapshot_loading(
    tmp_path: Path,
) -> None:
    config = EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1]
        / "examples/embodied/pi0_fast_libero_spatial_grpo_development_h100.yaml"
    )

    worker = _worker_config(
        {
            "config": config.model_dump(mode="json"),
            "policy_snapshot": str(tmp_path / "adapter"),
        }
    )

    assert worker.policy.type == "pi0_fast"
    assert "peft_adapter_path" not in worker.policy.load_kwargs
    assert worker.policy.device == "cuda:0"
    assert worker.runtime.training_devices == ["cuda:0"]
    assert worker.runtime.distributed_training is False


def test_pi0_fast_worker_projection_accepts_repeated_full_update_epochs(
    tmp_path: Path,
) -> None:
    config = EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1] / "examples/embodied/"
        "pi0_fast_libero_spatial_task5_full_trajectory_grpo_positive_control_h100.yaml"
    )

    worker = _worker_config(
        {
            "config": config.model_dump(mode="json"),
            "policy_snapshot": str(tmp_path / "adapter"),
        }
    )

    assert worker.training.schedule.type == "full_update"
    assert worker.training.schedule.update_epochs == 2
    assert worker.training.optimizer_steps_per_update == 2
    assert worker.runtime.training_devices == ["cuda:0"]
    assert worker.runtime.distributed_training is False


def test_distributed_worker_projection_drops_parent_resume_checkpoint(
    tmp_path: Path,
) -> None:
    config = EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_libero_object_grpo_lora_lr1e4_shared_rollout.yaml"
    )
    raw = config.model_dump(mode="json")
    raw["storage"]["resume_from_checkpoint"] = str(tmp_path / "step_000010")

    worker = _worker_config(
        {
            "config": raw,
            "policy_snapshot": str(tmp_path / "adapter"),
        }
    )

    assert worker.storage.resume_from_checkpoint is None
    assert worker.runtime.distributed_training is False
    assert worker.policy.load_kwargs["peft_adapter_path"] == str(tmp_path / "adapter")


def test_persistent_gradient_job_reuses_bootstrap_config_without_job_config(
    tmp_path: Path,
) -> None:
    config = EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_action_token_distributed_smoke.yaml"
    )

    assert _gradient_job_config({"op": "gradient"}, config) is config


def test_distributed_worker_projects_gspo_minibatch_to_one_gradient_job(
    tmp_path: Path,
) -> None:
    config = EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_libero_object_gspo_lora_lr2e4_h100.yaml"
    )

    worker = _worker_config(
        {
            "config": config.model_dump(mode="json"),
            "policy_snapshot": str(tmp_path / "adapter"),
        }
    )

    assert worker.algorithm.type == "gspo"
    assert worker.training.schedule.type == "full_update"
    assert worker.training.optimizer_steps_per_update == 1
    assert worker.runtime.distributed_training is False


def _trajectory(token: int, reward: float, task: str) -> EmbodiedTrajectory:
    item = EmbodiedTrajectory(task=task, reward=reward)
    item.actions.append(
        Action(
            step=0,
            kind="token",
            raw={"tokens": [token], "prompt": task},
            logprobs={"token_logprobs": [math.log(0.5)]},
        )
    )
    return item


def _trajectory_with_policy_step(
    token: int, reward: float, task: str, policy_step: int
) -> EmbodiedTrajectory:
    item = EmbodiedTrajectory(task=task, reward=reward)
    item.actions.append(
        Action(
            step=0,
            kind="token",
            raw={"tokens": [token], "prompt": task},
            logprobs={"token_logprobs": [math.log(0.5)]},
            metadata={"policy_step": int(policy_step)},
        )
    )
    return item


def _group(good_reward: float, bad_reward: float, task: str) -> EmbodiedTrajectoryGroup:
    return EmbodiedTrajectoryGroup(
        [
            _trajectory(1, good_reward, task),
            _trajectory(0, bad_reward, task),
        ]
    )


def _multi_action_trajectory(
    token: int, reward: float, task: str
) -> EmbodiedTrajectory:
    item = EmbodiedTrajectory(task=task, reward=reward)
    for step in range(2):
        item.actions.append(
            Action(
                step=step,
                kind="token",
                raw={"tokens": [token, token], "prompt": task},
                logprobs={"token_logprobs": [math.log(0.5), math.log(0.5)]},
            )
        )
    return item


def _gspo_group(
    good_reward: float, bad_reward: float, task: str
) -> EmbodiedTrajectoryGroup:
    return EmbodiedTrajectoryGroup(
        [
            _multi_action_trajectory(1, good_reward, task),
            _multi_action_trajectory(0, bad_reward, task),
        ]
    )


def test_action_examples_propagate_token_loss_mask_from_action_metadata() -> None:
    trajectory = EmbodiedTrajectory(task="pick object", reward=1.0)
    trajectory.actions.append(
        Action(
            step=0,
            kind="token",
            raw={"tokens": [10, 11, 40, 41, 12], "prompt": "pick object"},
            logprobs={"token_logprobs": [-0.1, -0.2, -0.3, -0.4, -0.5]},
            metadata={
                "rl_token_scope": "fast_payload",
                "token_loss_mask": [False, False, True, True, False],
            },
        )
    )

    example = extract_action_token_examples(
        [EmbodiedTrajectoryGroup([trajectory])],
        score_source="trajectory_reward",
    )[0]

    assert example.metadata["token_loss_mask"] == [
        False,
        False,
        True,
        True,
        False,
    ]


def test_action_examples_reject_misaligned_token_loss_mask() -> None:
    trajectory = EmbodiedTrajectory(task="pick object", reward=1.0)
    trajectory.actions.append(
        Action(
            step=0,
            kind="token",
            raw={"tokens": [10, 11, 40], "prompt": "pick object"},
            metadata={"token_loss_mask": [False, True]},
        )
    )

    with pytest.raises(ValueError, match="length must match action tokens"):
        extract_action_token_examples(
            [EmbodiedTrajectoryGroup([trajectory])],
            score_source="trajectory_reward",
        )


def test_payload_mask_updates_only_positive_and_negative_action_tokens() -> None:
    def trajectory(*, tokens: list[int], reward: float) -> EmbodiedTrajectory:
        item = EmbodiedTrajectory(task="pick object", reward=reward)
        item.actions.append(
            Action(
                step=0,
                kind="token",
                raw={"tokens": tokens, "prompt": "pick object"},
                logprobs={"token_logprobs": [0.0] * len(tokens)},
                metadata={
                    "rl_token_scope": "fast_payload",
                    "token_loss_mask": [False, True, False],
                },
            )
        )
        return item

    policy = IndependentTokenLogprobPolicy(token_count=6)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    backend = ActionTokenGRPOBackend(
        policy,
        optimizer=optimizer,
        require_prompts=True,
        normalize_advantages=True,
        advantage_normalization_scope="group",
        advantage_std_unbiased=False,
        training_unit="action",
        rlinf_action_level_score_source="trajectory_reward",
        loss_aggregation="seq_mean_token_sum",
    )

    asyncio.run(
        backend.train(
            [
                EmbodiedTrajectoryGroup(
                    [
                        trajectory(tokens=[0, 1, 2], reward=1.0),
                        trajectory(tokens=[3, 4, 5], reward=0.0),
                    ]
                )
            ]
        )
    )

    assert policy.logprobs.detach().tolist() == pytest.approx(
        [0.0, 0.05, 0.0, 0.0, -0.05, 0.0]
    )


def test_action_examples_honor_trajectory_reward_score_source() -> None:
    from art_embodied import extract_action_token_examples

    def trajectory(*, reward: float, token: int) -> EmbodiedTrajectory:
        item = EmbodiedTrajectory(task="pick object", reward=reward)
        for policy_step, chunk_reward in enumerate((0.0, reward)):
            item.actions.append(
                Action(
                    step=policy_step,
                    kind="token",
                    raw={"tokens": [token], "prompt": "pick object"},
                    logprobs={"token_logprobs": [math.log(0.5)]},
                    metadata={"policy_step": policy_step},
                )
            )
            item.add_reward(
                "success",
                chunk_reward,
                "env",
                metadata={"policy_step": policy_step},
                update_total=False,
            )
        return item

    group = EmbodiedTrajectoryGroup(
        [trajectory(reward=1.0, token=1), trajectory(reward=0.0, token=0)]
    )

    chunk_examples = extract_action_token_examples(
        [group], score_source="chunk_rewards"
    )
    trajectory_examples = extract_action_token_examples(
        [group], score_source="trajectory_reward"
    )

    assert [example.reward for example in chunk_examples] == [0.0, 1.0, 0.0, 0.0]
    assert [example.reward for example in trajectory_examples] == [1.0, 1.0, 0.0, 0.0]
    assert all(
        example.metadata["action_reward_source"] == "trajectory_reward"
        for example in trajectory_examples
    )
    assert [example.metadata["group_advantage"] for example in trajectory_examples] == [
        0.5,
        0.5,
        -0.5,
        -0.5,
    ]


def test_trajectory_reward_advantages_ignore_unequal_action_counts() -> None:
    def trajectory(*, reward: float, token: int, actions: int) -> EmbodiedTrajectory:
        item = EmbodiedTrajectory(task="pick object", reward=reward)
        for step in range(actions):
            item.actions.append(
                Action(
                    step=step,
                    kind="token",
                    raw={"tokens": [token], "prompt": "pick object"},
                    logprobs={"token_logprobs": [math.log(0.5)]},
                )
            )
        return item

    group = EmbodiedTrajectoryGroup(
        [
            trajectory(reward=1.0, token=1, actions=1),
            trajectory(reward=0.0, token=0, actions=3),
        ]
    )
    examples = extract_action_token_examples([group], score_source="trajectory_reward")
    backend = ActionTokenGRPOBackend(
        TinyTokenPolicy(),
        normalize_advantages=True,
        advantage_normalization_scope="group",
        advantage_std_unbiased=False,
        training_unit="action",
        loss_aggregation="trajectory_mean",
    )

    _attach_full_update_advantages(examples, backend=backend)

    assert [example.metadata["group_advantage"] for example in examples] == [
        0.5,
        -0.5,
        -0.5,
        -0.5,
    ]
    assert [example.metadata["token_advantages"] for example in examples] == [
        [pytest.approx(1.0)],
        [pytest.approx(-1.0)],
        [pytest.approx(-1.0)],
        [pytest.approx(-1.0)],
    ]
    assert [example.metadata["trajectory_action_count"] for example in examples] == [
        1,
        3,
        3,
        3,
    ]
    trajectory_weights = [
        float(
            _example_bucket_loss_weight_tensor(
                example,
                token_count=1,
                device="cpu",
                dtype=torch.float32,
                policy_step_loss_weights={},
                action_dim_loss_weights={},
                loss_aggregation="trajectory_mean",
            ).item()
        )
        for example in examples
    ]
    assert trajectory_weights == pytest.approx([1.0, 1 / 3, 1 / 3, 1 / 3])


def test_trajectory_mean_gradient_is_invariant_to_trajectory_action_count() -> None:
    def group(*, failed_actions: int) -> EmbodiedTrajectoryGroup:
        trajectories = []
        for reward, token, action_count in ((1.0, 1, 1), (0.0, 0, failed_actions)):
            trajectory = EmbodiedTrajectory(task="pick object", reward=reward)
            for step in range(action_count):
                trajectory.actions.append(
                    Action(
                        step=step,
                        kind="token",
                        raw={"tokens": [token], "prompt": "pick object"},
                        logprobs={"token_logprobs": [math.log(0.5)]},
                    )
                )
            trajectories.append(trajectory)
        return EmbodiedTrajectoryGroup(trajectories)

    async def update(*, failed_actions: int) -> torch.Tensor:
        policy = TinyTokenPolicy()
        backend = ActionTokenGRPOBackend(
            policy,
            optimizer=torch.optim.SGD(policy.parameters(), lr=0.1),
            normalize_advantages=True,
            advantage_normalization_scope="group",
            advantage_std_unbiased=False,
            training_unit="action",
            rlinf_action_level_score_source="trajectory_reward",
            loss_aggregation="trajectory_mean",
            logprob_microbatch_size=1,
        )
        await backend.train([group(failed_actions=failed_actions)])
        return policy.logits.detach()

    equal_length = asyncio.run(update(failed_actions=1))
    unequal_length = asyncio.run(update(failed_actions=3))

    torch.testing.assert_close(unequal_length, equal_length)


def test_seq_mean_token_sum_matches_direct_trajectory_formula() -> None:
    def trajectory(*, reward: float, token: int, actions: int) -> EmbodiedTrajectory:
        item = EmbodiedTrajectory(task="pick object", reward=reward)
        for step in range(actions):
            item.actions.append(
                Action(
                    step=step,
                    kind="token",
                    raw={"tokens": [token], "prompt": "pick object"},
                    logprobs={"token_logprobs": [math.log(0.5)]},
                )
            )
        return item

    group = EmbodiedTrajectoryGroup(
        [
            trajectory(reward=1.0, token=1, actions=1),
            trajectory(reward=0.0, token=0, actions=3),
        ]
    )
    policy = TinyTokenPolicy()
    initial_probabilities = torch.softmax(policy.logits.detach(), dim=-1)
    expected_loss = (
        -(initial_probabilities[1] / 0.5) + 3.0 * (initial_probabilities[0] / 0.5)
    ) / 2.0
    backend = ActionTokenGRPOBackend(
        policy,
        optimizer=torch.optim.SGD(policy.parameters(), lr=0.1),
        normalize_advantages=True,
        advantage_normalization_scope="group",
        advantage_std_unbiased=False,
        training_unit="action",
        rlinf_action_level_score_source="trajectory_reward",
        loss_aggregation="seq_mean_token_sum",
        logprob_microbatch_size=1,
    )

    result = asyncio.run(backend.train([group]))

    assert result.metrics["embodied_action_token_grpo/loss"] == pytest.approx(
        float(expected_loss), rel=1e-6
    )
    assert result.metrics[
        "embodied_action_token_grpo/loss_aggregation_seq_mean_token_sum"
    ] == pytest.approx(1.0)


def test_seq_mean_token_sum_preserves_trajectory_length_weight() -> None:
    async def update(*, failed_actions: int) -> torch.Tensor:
        trajectories = []
        for reward, token, action_count in ((1.0, 1, 1), (0.0, 0, failed_actions)):
            item = EmbodiedTrajectory(task="pick object", reward=reward)
            for step in range(action_count):
                item.actions.append(
                    Action(
                        step=step,
                        kind="token",
                        raw={"tokens": [token], "prompt": "pick object"},
                        logprobs={"token_logprobs": [math.log(0.5)]},
                    )
                )
            trajectories.append(item)
        policy = TinyTokenPolicy()
        backend = ActionTokenGRPOBackend(
            policy,
            optimizer=torch.optim.SGD(policy.parameters(), lr=0.1),
            normalize_advantages=True,
            advantage_normalization_scope="group",
            advantage_std_unbiased=False,
            training_unit="action",
            rlinf_action_level_score_source="trajectory_reward",
            loss_aggregation="seq_mean_token_sum",
            logprob_microbatch_size=1,
        )
        await backend.train([EmbodiedTrajectoryGroup(trajectories)])
        return policy.logits.detach()

    equal_length = asyncio.run(update(failed_actions=1))
    unequal_length = asyncio.run(update(failed_actions=3))

    assert not torch.allclose(unequal_length, equal_length)


def test_signal_metrics_recover_trajectory_rewards_from_flat_examples() -> None:
    from art_embodied import extract_action_token_examples
    from art_embodied.backends.action_token import _group_signal_metrics

    group = _gspo_group(1.0, 0.0, "pick object")
    examples = extract_action_token_examples(
        [group],
        score_source="trajectory_reward",
    )

    metrics = _group_signal_metrics(
        groups=[],
        examples=examples,
        advantages=[0.5, 0.5, -0.5, -0.5],
        prefix="test",
    )

    assert metrics["test/trajectory_reward_positive_fraction"] == pytest.approx(0.5)


def test_trajectory_group_metrics_use_complete_coordinator_groups() -> None:
    from art_embodied.backends.action_token import _trajectory_group_signal_metrics

    groups = [
        _gspo_group(1.0, 0.0, "pick object"),
        _gspo_group(1.0, 1.0, "place object"),
    ]

    metrics = _trajectory_group_signal_metrics(groups, prefix="test")

    assert metrics["test/groups_nonempty"] == 2.0
    assert metrics["test/groups_with_reward_variance"] == 1.0
    assert metrics["test/groups_with_zero_reward_variance"] == 1.0
    assert metrics["test/effective_group_fraction"] == pytest.approx(0.5)
    assert metrics["test/trajectory_reward_positive_fraction"] == pytest.approx(0.75)


def _gspo_backend(
    policy: TinyTokenPolicy,
    *,
    optimizer=None,
    kl_coef: float = 0.0,
    reference_logprob_l2_coef: float = 0.0,
    logprob_microbatch_size: int = 1,
) -> ActionTokenGSPOBackend:
    return ActionTokenGSPOBackend(
        policy,
        optimizer=optimizer,
        lr=0.1,
        clip_epsilon=0.2,
        kl_coef=kl_coef,
        require_prompts=True,
        reference_logprob_l2_coef=reference_logprob_l2_coef,
        normalize_advantages=True,
        advantage_normalization_scope="group",
        advantage_std_unbiased=False,
        loss_aggregation="trajectory_mean",
        logprob_microbatch_size=logprob_microbatch_size,
    )


def test_gspo_updates_complete_trajectory_sequence_toward_rewarded_actions() -> None:
    policy = TinyTokenPolicy()
    initial = policy.logits.detach().clone()
    groups = [_gspo_group(1.0, 0.0, "pick red cube")]

    result = asyncio.run(_gspo_backend(policy).train(groups))

    assert (
        result.metrics["embodied_action_token_gspo/sequence_importance_sampling"] == 1.0
    )
    assert result.metrics["embodied_action_token_gspo/trajectory_training_unit"] == 1.0
    assert result.metrics["embodied_action_token_gspo/optimizer_step_completed"] == 1.0
    assert policy.logits[1] > initial[1]
    assert policy.logits[0] < initial[0]
    assert (
        result.metrics["embodied_action_token_gspo/streamed_sequence_two_pass"] == 1.0
    )
    assert not any(
        key.startswith("embodied_action_token_grpo/") for key in result.metrics
    )


def test_gspo_no_signal_metrics_hide_internal_grpo_namespace() -> None:
    result = asyncio.run(
        _gspo_backend(TinyTokenPolicy()).train([_gspo_group(1.0, 1.0, "pick red cube")])
    )

    assert any(key.startswith("embodied_action_token_gspo/") for key in result.metrics)
    assert not any(
        key.startswith("embodied_action_token_grpo/") for key in result.metrics
    )


@pytest.mark.parametrize(
    ("force_sequence_clipping", "mask_alternating_tokens"),
    [(False, False), (True, False), (True, True)],
)
def test_streamed_gspo_gradient_matches_full_trajectory_objective(
    force_sequence_clipping: bool,
    mask_alternating_tokens: bool,
) -> None:
    groups = [
        _gspo_group(1.0, 0.0, "pick red cube"),
        _gspo_group(0.8, 0.2, "pick blue cube"),
    ]
    prepared = extract_trajectory_action_token_examples(
        groups,
        require_logprobs=True,
        require_prompts=True,
    )
    preparation_backend = _gspo_backend(TinyTokenPolicy())
    _attach_full_update_advantages(prepared, backend=preparation_backend)
    if force_sequence_clipping:
        current_rows = preparation_backend.policy.action_token_logprobs(prepared)
        target_sequence_ratios = (1.5, 0.5, 1.1, 0.9)
        for example, current, target_ratio in zip(
            prepared, current_rows, target_sequence_ratios, strict=True
        ):
            example.logprobs = [
                float(value) - math.log(target_ratio) for value in current.detach()
            ]
    if mask_alternating_tokens:
        for example in prepared:
            example.metadata["token_loss_mask"] = [True, False, True, False]
    for example in prepared:
        example.metadata["reference_logprobs"] = [
            value - 0.03 for value in example.logprobs or []
        ]

    async def worker_gradients(examples):
        policy = TinyTokenPolicy()
        backend = _gspo_backend(
            policy,
            kl_coef=0.07,
            reference_logprob_l2_coef=0.11,
            logprob_microbatch_size=2,
        )
        result = await backend.train(
            [],
            _action_token_grpo_return_gradients=True,
            _action_token_grpo_precomputed_examples=examples,
            _action_token_grpo_precomputed_examples_prepared=True,
            _action_token_grpo_precomputed_reward_filter_report={"enabled": False},
            _action_token_grpo_global_example_count=len(prepared),
            _action_token_grpo_global_token_count=sum(
                len(example.logprobs or []) for example in prepared
            ),
        )
        return _trainable_gradient_payload(policy), result

    streamed_payload, streamed_result = asyncio.run(
        worker_gradients(copy.deepcopy(prepared))
    )
    full_examples = copy.deepcopy(prepared)
    for example in full_examples:
        example.metadata.pop("action_spans")
    full_payload, full_result = asyncio.run(worker_gradients(full_examples))

    assert (
        streamed_result.metrics["embodied_action_token_gspo/streamed_sequence_two_pass"]
        == 1.0
    )
    assert (
        full_result.metrics["embodied_action_token_gspo/streamed_sequence_two_pass"]
        == 0.0
    )
    assert streamed_payload["gradients"].keys() == full_payload["gradients"].keys()
    for name, streamed_gradient in streamed_payload["gradients"].items():
        torch.testing.assert_close(
            streamed_gradient,
            full_payload["gradients"][name],
            rtol=1e-5,
            atol=1e-6,
        )
    assert streamed_result.metrics["embodied_action_token_gspo/loss"] == pytest.approx(
        full_result.metrics["embodied_action_token_gspo/loss"], rel=1e-6, abs=1e-7
    )
    if force_sequence_clipping:
        assert streamed_result.metrics["embodied_action_token_gspo/clip_fraction"] > 0


def test_gspo_distributed_gradient_handoff_matches_single_process_update() -> None:
    groups = [
        _gspo_group(1.0, 0.0, "pick red cube"),
        _gspo_group(0.8, 0.2, "pick blue cube"),
    ]
    base_policy = TinyTokenPolicy()

    async def run_single():
        policy = copy.deepcopy(base_policy)
        result = await _gspo_backend(policy).train(groups)
        return policy, result

    single_policy, single_result = asyncio.run(run_single())
    assert (
        single_result.metrics["embodied_action_token_gspo/optimizer_step_completed"]
        == 1.0
    )

    parent_policy = copy.deepcopy(base_policy)
    parent_optimizer = torch.optim.AdamW(
        parent_policy.parameters(), lr=0.1, weight_decay=0.0
    )
    payloads = []

    prepared = extract_trajectory_action_token_examples(
        groups,
        require_logprobs=True,
        require_prompts=True,
    )
    preparation_backend = _gspo_backend(copy.deepcopy(base_policy))
    _attach_full_update_advantages(prepared, backend=preparation_backend)
    assert all(
        example.metadata.get("group_advantage_prepared") is True for example in prepared
    )
    assert all("token_advantages" not in example.metadata for example in prepared)

    async def run_worker(shard):
        worker_policy = copy.deepcopy(base_policy)
        backend = _gspo_backend(worker_policy)
        await backend.train(
            [],
            _action_token_grpo_return_gradients=True,
            _action_token_grpo_precomputed_examples=shard,
            _action_token_grpo_precomputed_examples_prepared=True,
            _action_token_grpo_precomputed_reward_filter_report={"enabled": False},
            _action_token_grpo_global_example_count=4,
            _action_token_grpo_global_token_count=16,
        )
        return _trainable_gradient_payload(worker_policy)

    for shard in _partition_examples(prepared, workers=2):
        payloads.append(asyncio.run(run_worker(shard)))

    apply_action_token_gradient_payloads(parent_policy, parent_optimizer, payloads)

    torch.testing.assert_close(
        parent_policy.logits.detach(), single_policy.logits.detach()
    )


def _backend(policy: TinyTokenPolicy, *, optimizer=None) -> ActionTokenGRPOBackend:
    return ActionTokenGRPOBackend(
        policy,
        optimizer=optimizer,
        lr=0.1,
        clip_epsilon=0.2,
        require_prompts=True,
        loss_aggregation="token_mean",
        advantage_normalization_scope="group",
        logprob_microbatch_size=1,
    )


def test_action_token_gradient_handoff_matches_single_process_update() -> None:
    groups = [
        _group(1.0, 0.0, "pick red cube"),
        _group(0.8, 0.2, "pick blue cube"),
    ]
    base_policy = TinyTokenPolicy()

    async def run_single():
        policy = copy.deepcopy(base_policy)
        backend = _backend(policy)
        result = await backend.train(groups)
        return policy, result

    single_policy, single_result = asyncio.run(run_single())
    assert (
        single_result.metrics["embodied_action_token_grpo/optimizer_step_completed"]
        == 1.0
    )

    parent_policy = copy.deepcopy(base_policy)
    parent_optimizer = torch.optim.AdamW(
        parent_policy.parameters(), lr=0.1, weight_decay=0.0
    )
    global_example_count = 4
    global_token_count = 4
    payloads = []

    async def run_worker(shard: list[EmbodiedTrajectoryGroup]):
        worker_policy = copy.deepcopy(base_policy)
        backend = _backend(worker_policy)
        result = await backend.train(
            shard,
            _action_token_grpo_return_gradients=True,
            _action_token_grpo_global_example_count=global_example_count,
            _action_token_grpo_global_token_count=global_token_count,
        )
        assert (
            result.metrics["embodied_action_token_grpo/gradient_handoff_worker"] == 1.0
        )
        return _trainable_gradient_payload(worker_policy)

    for group in groups:
        payloads.append(asyncio.run(run_worker([group])))

    apply_metrics = apply_action_token_gradient_payloads(
        parent_policy, parent_optimizer, payloads
    )

    assert (
        apply_metrics["embodied_action_token_grpo/distributed_gradient_handoff"] == 1.0
    )
    torch.testing.assert_close(
        parent_policy.logits.detach(), single_policy.logits.detach()
    )


@pytest.mark.parametrize("loss_aggregation", ["trajectory_mean", "seq_mean_token_sum"])
def test_prepared_distributed_trajectory_objective_matches_single_process_update(
    loss_aggregation: str,
) -> None:
    def trajectory(
        *, token: int, reward: float, task: str, action_count: int
    ) -> EmbodiedTrajectory:
        item = EmbodiedTrajectory(task=task, reward=reward)
        for step in range(action_count):
            item.actions.append(
                Action(
                    step=step,
                    kind="token",
                    raw={"tokens": [token], "prompt": task},
                    logprobs={"token_logprobs": [math.log(0.5)]},
                )
            )
        return item

    groups = [
        EmbodiedTrajectoryGroup(
            [
                trajectory(
                    token=1,
                    reward=1.0,
                    task="pick red cube",
                    action_count=1,
                ),
                trajectory(
                    token=0,
                    reward=0.0,
                    task="pick red cube",
                    action_count=3,
                ),
            ]
        ),
        EmbodiedTrajectoryGroup(
            [
                trajectory(
                    token=1,
                    reward=0.8,
                    task="pick blue cube",
                    action_count=2,
                ),
                trajectory(
                    token=0,
                    reward=0.2,
                    task="pick blue cube",
                    action_count=4,
                ),
            ]
        ),
    ]
    base_policy = TinyTokenPolicy()

    def backend(policy: TinyTokenPolicy, *, optimizer=None) -> ActionTokenGRPOBackend:
        return ActionTokenGRPOBackend(
            policy,
            optimizer=optimizer,
            lr=0.1,
            require_prompts=True,
            normalize_advantages=True,
            advantage_normalization_scope="group",
            advantage_std_unbiased=False,
            training_unit="action",
            rlinf_action_level_score_source="trajectory_reward",
            loss_aggregation=loss_aggregation,
            logprob_microbatch_size=1,
        )

    single_policy = copy.deepcopy(base_policy)
    asyncio.run(backend(single_policy).train(copy.deepcopy(groups)))

    preparation_backend = backend(copy.deepcopy(base_policy))
    prepared, report = prepare_action_token_examples(
        copy.deepcopy(groups), backend=preparation_backend
    )
    _attach_full_update_advantages(prepared, backend=preparation_backend)
    parent_policy = copy.deepcopy(base_policy)
    parent_optimizer = torch.optim.AdamW(
        parent_policy.parameters(), lr=0.1, weight_decay=0.0
    )
    payloads = []

    async def worker_gradient(shard):
        worker_policy = copy.deepcopy(base_policy)
        result = await backend(worker_policy).train(
            [],
            _action_token_grpo_return_gradients=True,
            _action_token_grpo_precomputed_examples=shard,
            _action_token_grpo_precomputed_examples_prepared=True,
            _action_token_grpo_precomputed_reward_filter_report=report,
            _action_token_grpo_global_example_count=4,
            _action_token_grpo_global_token_count=sum(
                len(example.logprobs or []) for example in prepared
            ),
        )
        assert result.metrics[
            "embodied_action_token_grpo/gradient_handoff_worker"
        ] == pytest.approx(1.0)
        return _trainable_gradient_payload(worker_policy)

    for shard in _partition_examples(prepared, workers=2):
        payloads.append(asyncio.run(worker_gradient(shard)))
    apply_action_token_gradient_payloads(parent_policy, parent_optimizer, payloads)

    torch.testing.assert_close(
        parent_policy.logits.detach(), single_policy.logits.detach()
    )


def test_task_balanced_trajectory_mean_matches_equal_task_panel() -> None:
    base_policy = TinyTokenPolicy()
    task_a = _group(1.0, 0.0, "task-a")
    task_b = _group(0.0, 1.0, "task-b")

    async def gradient(groups, *, task_balanced: bool):
        policy = copy.deepcopy(base_policy)
        backend = ActionTokenGRPOBackend(
            policy,
            require_prompts=True,
            normalize_advantages=True,
            advantage_normalization_scope="group",
            advantage_std_unbiased=False,
            loss_aggregation=(
                "task_balanced_trajectory_mean" if task_balanced else "trajectory_mean"
            ),
            logprob_microbatch_size=1,
        )
        examples = extract_action_token_examples(
            groups, require_logprobs=True, require_prompts=True
        )
        _attach_full_update_advantages(examples, backend=backend)
        if task_balanced:
            _attach_task_balance_weights(examples)
        await backend.train(
            [],
            _action_token_grpo_precomputed_examples=examples,
            _action_token_grpo_precomputed_examples_prepared=True,
            _action_token_grpo_precomputed_reward_filter_report={"enabled": False},
            _action_token_grpo_global_example_count=len(examples),
            _action_token_grpo_global_token_count=sum(
                len(example.tokens) for example in examples
            ),
            _action_token_grpo_return_gradients=True,
        )
        return _trainable_gradient_payload(policy)

    balanced = asyncio.run(gradient([task_a, task_b], task_balanced=False))
    imbalanced = asyncio.run(
        gradient([task_a, task_b, copy.deepcopy(task_b)], task_balanced=True)
    )

    torch.testing.assert_close(
        imbalanced["gradients"]["logits"], balanced["gradients"]["logits"]
    )


def test_task_balance_weights_equalize_each_task_total() -> None:
    examples = extract_action_token_examples(
        [
            _group(1.0, 0.0, "task-a"),
            _group(1.0, 0.0, "task-b"),
            _group(1.0, 0.0, "task-b"),
        ],
        require_logprobs=True,
        require_prompts=True,
    )

    _attach_task_balance_weights(examples)

    totals: dict[str, float] = {}
    for example in examples:
        totals[example.task] = totals.get(example.task, 0.0) + float(
            example.metadata["task_balance_weight"]
        )
    assert totals == pytest.approx({"task-a": 3.0, "task-b": 3.0})


def test_pi0_fast_sft_anchor_assigns_every_task_once() -> None:
    config = EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1]
        / "examples/embodied/pi0_fast_libero_long_task_balanced_grpo_development_h100.yaml"
    )

    assignments = [
        _pi0_fast_sft_anchor_tasks_for_worker(
            config, worker_index=index, worker_count=8
        )
        for index in range(8)
    ]

    assert assignments[:2] == [[0, 8], [1, 9]]
    assert assignments[2:] == [[2], [3], [4], [5], [6], [7]]
    assert sorted(task for tasks in assignments for task in tasks) == list(range(10))


def test_pi0_fast_sft_anchor_allows_fewer_tasks_than_workers(tmp_path: Path) -> None:
    config = EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1] / "examples/embodied/"
        "pi0_fast_libero_plus_language_kitchen4_single_v1_grpo_development_h100.yaml"
    )
    config = config.model_copy(
        update={
            "policy": config.policy.model_copy(
                update={
                    "load_kwargs": {
                        **config.policy.load_kwargs,
                        "sft_anchor": {
                            "coefficient": 0.05,
                            "dataset_repo_id": "lerobot/libero",
                            "dataset_revision": (
                                "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
                            ),
                            "task_indices": [1],
                            "seed": 20260830,
                            "samples_per_task": 1,
                        },
                    }
                }
            )
        }
    )
    assignments = [
        _pi0_fast_sft_anchor_tasks_for_worker(
            config, worker_index=index, worker_count=8
        )
        for index in range(8)
    ]
    assert assignments == [[1], [], [], [], [], [], [], []]

    metrics = _add_pi0_fast_sft_anchor_gradients(
        policy=object(),
        config=config,
        provider=None,
        update_index=0,
        subupdate_index=0,
        gradient_path=tmp_path / "existing-gradient.pt",
    )
    assert metrics["embodied_action_token_grpo/sft_anchor/examples"] == 0.0
    assert metrics["embodied_action_token_grpo/sft_anchor/coefficient"] == 0.05


def test_pi0_fast_task_balanced_recipe_uses_twelve_groups_per_task() -> None:
    config = EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1]
        / "examples/embodied/pi0_fast_libero_long_task_balanced_grpo_development_h100.yaml"
    )

    assert config.algorithm.group_size == 8
    assert config.rollout.groups_per_update == 120
    assert config.rollout.epochs_per_update == 1
    assert config.trajectories_per_update == 960
    assert config.rollout.groups_per_update // 10 == 12


def test_pi0_fast_sft_anchor_adds_globally_task_normalized_gradient(
    tmp_path: Path,
) -> None:
    config = EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1]
        / "examples/embodied/pi0_fast_libero_long_task_balanced_grpo_development_h100.yaml"
    )

    class Native(torch.nn.Module):
        pass

    class Policy:
        def __init__(self) -> None:
            self.value = torch.nn.Parameter(torch.tensor(2.0))
            self.policy = Native()

        def named_parameters(self):
            return [("value", self.value)]

    class Provider:
        def __init__(self, policy) -> None:
            self.policy = policy

        def losses(self, **_kwargs):
            return [
                SimpleNamespace(task_index=0, loss=self.policy.value.square()),
                SimpleNamespace(task_index=8, loss=self.policy.value.square()),
            ]

    policy = Policy()
    gradient_path = tmp_path / "gradient.pt"

    metrics = _add_pi0_fast_sft_anchor_gradients(
        policy=policy,
        config=config,
        provider=Provider(policy),
        update_index=3,
        subupdate_index=0,
        gradient_path=gradient_path,
    )
    payload = load_action_token_gradient_payload(gradient_path)

    # Two worker-owned tasks each contribute beta / all_task_count * d(value^2).
    assert payload["gradients"]["value"].item() == pytest.approx(0.04)
    assert payload["sft_anchor_tasks"] == [0, 8]
    assert metrics["embodied_action_token_grpo/sft_anchor/loss"] == pytest.approx(0.8)


def test_action_token_gradient_handoff_matches_rlinf_masked_mean_ratio_update() -> None:
    initial_logprobs = torch.log_softmax(TinyTokenPolicy().logits.detach(), dim=-1)

    def trajectory(token: int, rewards: list[float], task: str) -> EmbodiedTrajectory:
        item = EmbodiedTrajectory(task=task, reward=sum(rewards))
        item.actions.append(
            Action(
                step=0,
                kind="token",
                raw={"tokens": [token for _ in rewards], "prompt": task},
                logprobs={
                    "token_logprobs": [float(initial_logprobs[token]) for _ in rewards]
                },
                metadata={
                    "primitive_rewards": rewards,
                    "primitive_loss_mask": [True for _ in rewards],
                    "primitive_loss_mask_sum": len(rewards),
                },
            )
        )
        return item

    groups = [
        EmbodiedTrajectoryGroup(
            [
                trajectory(1, [1.0], "choose red"),
                trajectory(0, [0.0, 0.0, 0.0, 0.0], "choose red"),
            ]
        ),
        EmbodiedTrajectoryGroup(
            [
                trajectory(1, [0.8, 0.0], "choose blue"),
                trajectory(0, [0.2, 0.0, 0.0], "choose blue"),
            ]
        ),
    ]
    base_policy = TinyTokenPolicy()

    def backend(policy: TinyTokenPolicy, *, optimizer=None) -> ActionTokenGRPOBackend:
        return ActionTokenGRPOBackend(
            policy,
            optimizer=optimizer,
            lr=0.1,
            clip_epsilon=0.2,
            require_prompts=True,
            action_advantage_mode="rlinf_action_level_cumulative",
            normalize_advantages=True,
            advantage_std_unbiased=False,
            loss_aggregation="rlinf_masked_mean_ratio",
            importance_sampling_level="token",
            logprob_microbatch_size=1,
        )

    async def run_single():
        policy = copy.deepcopy(base_policy)
        result = await backend(policy).train(groups)
        return policy, result

    single_policy, single_result = asyncio.run(run_single())
    assert (
        single_result.metrics["embodied_action_token_grpo/optimizer_step_completed"]
        == 1.0
    )
    global_example_count = int(
        single_result.metrics["embodied_action_token_grpo/examples"]
    )
    global_token_count = int(
        single_result.metrics[
            "embodied_action_token_grpo/global_loss_denominator_tokens"
        ]
    )
    assert global_example_count == 4
    assert global_token_count > int(
        single_result.metrics["embodied_action_token_grpo/tokens_total"]
    )

    parent_policy = copy.deepcopy(base_policy)
    parent_optimizer = torch.optim.AdamW(
        parent_policy.parameters(), lr=0.1, weight_decay=0.0
    )
    payloads = []

    async def run_worker(shard: list[EmbodiedTrajectoryGroup]):
        worker_policy = copy.deepcopy(base_policy)
        result = await backend(worker_policy).train(
            shard,
            _action_token_grpo_return_gradients=True,
            _action_token_grpo_global_example_count=global_example_count,
            _action_token_grpo_global_token_count=global_token_count,
        )
        assert (
            result.metrics["embodied_action_token_grpo/gradient_handoff_worker"] == 1.0
        )
        assert (
            result.metrics[
                "embodied_action_token_grpo/loss_aggregation_rlinf_masked_mean_ratio"
            ]
            == 1.0
        )
        return _trainable_gradient_payload(worker_policy)

    for group in groups:
        payloads.append(asyncio.run(run_worker([group])))

    apply_metrics = apply_action_token_gradient_payloads(
        parent_policy, parent_optimizer, payloads
    )

    assert (
        apply_metrics["embodied_action_token_grpo/distributed_gradient_handoff"] == 1.0
    )
    torch.testing.assert_close(
        parent_policy.logits.detach(), single_policy.logits.detach()
    )
    updated_logprobs = torch.log_softmax(single_policy.logits.detach(), dim=-1)
    assert updated_logprobs[1] > initial_logprobs[1]
    assert updated_logprobs[0] < initial_logprobs[0]


def test_action_token_policy_step_loss_weight_can_mask_update() -> None:
    groups = [
        EmbodiedTrajectoryGroup(
            [
                _trajectory_with_policy_step(1, 1.0, "pick red cube", 0),
                _trajectory_with_policy_step(0, 0.0, "pick red cube", 0),
            ]
        )
    ]
    base_policy = TinyTokenPolicy()

    async def run_backend(policy_step_loss_weights=None):
        policy = copy.deepcopy(base_policy)
        backend = ActionTokenGRPOBackend(
            policy,
            lr=0.1,
            clip_epsilon=0.2,
            require_prompts=True,
            loss_aggregation="token_mean",
            advantage_normalization_scope="group",
            policy_step_loss_weights=policy_step_loss_weights,
        )
        result = await backend.train(groups)
        return policy, result

    unmasked_policy, unmasked_result = asyncio.run(run_backend())
    masked_policy, masked_result = asyncio.run(run_backend({"policy_step_00": 0.0}))

    assert (
        unmasked_result.metrics["embodied_action_token_grpo/policy_parameters_updated"]
        == 1.0
    )
    assert (
        masked_result.metrics["embodied_action_token_grpo/optimizer_step_completed"]
        == 1.0
    )
    assert (
        masked_result.metrics["embodied_action_token_grpo/bucket_loss_weights_enabled"]
        == 1.0
    )
    assert (
        masked_result.metrics[
            "embodied_action_token_grpo/bucket_loss_zero_weight_token_fraction"
        ]
        == 1.0
    )
    assert not torch.allclose(
        unmasked_policy.logits.detach(), base_policy.logits.detach()
    )
    torch.testing.assert_close(
        masked_policy.logits.detach(), base_policy.logits.detach()
    )


def test_action_token_bucket_loss_weights_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="policy_step_loss_weights"):
        ActionTokenGRPOBackend(
            TinyTokenPolicy(),
            policy_step_loss_weights={"policy_step_00": -1.0},
        )


def test_action_token_gradient_direction_probe_reports_raw_gradient_improvement() -> (
    None
):
    policy = TinyTokenPolicy()
    old_logprobs = torch.log_softmax(policy.logits.detach(), dim=-1)

    def trajectory(token: int, reward: float, task: str) -> EmbodiedTrajectory:
        item = EmbodiedTrajectory(task=task, reward=reward)
        item.actions.append(
            Action(
                step=0,
                kind="token",
                raw={"tokens": [token], "prompt": task},
                logprobs={"token_logprobs": [float(old_logprobs[token])]},
            )
        )
        return item

    def group(
        good_reward: float, bad_reward: float, task: str
    ) -> EmbodiedTrajectoryGroup:
        return EmbodiedTrajectoryGroup(
            [
                trajectory(1, good_reward, task),
                trajectory(0, bad_reward, task),
            ]
        )

    groups = [
        group(1.0, 0.0, "pick red cube"),
        group(0.8, 0.2, "pick blue cube"),
    ]
    backend = _backend(policy)

    result = asyncio.run(
        backend.train(
            groups,
            _action_token_grpo_gradient_direction_probe=UserDict(
                {
                    "enabled": True,
                    "max_groups": 2,
                    "scales": [0.1, 1.0],
                }
            ),
        )
    )

    metrics = result.metrics
    assert (
        metrics["embodied_action_token_grpo/gradient_direction_probe_config_seen"]
        == 1.0
    )
    assert (
        metrics["embodied_action_token_grpo/gradient_direction_probe_config_is_mapping"]
        == 1.0
    )
    assert (
        metrics["embodied_action_token_grpo/gradient_direction_probe_config_enabled"]
        == 1.0
    )
    assert (
        metrics[
            "embodied_action_token_grpo/gradient_direction_probe_config_scale_count"
        ]
        == 2.0
    )
    assert metrics["embodied_action_token_grpo/gradient_direction_probe_enabled"] == 1.0
    assert (
        metrics["embodied_action_token_grpo/gradient_direction_probe_available"] == 1.0
    )
    assert (
        metrics["embodied_action_token_grpo/gradient_direction_probe_improved_scales"]
        > 0.0
    )
    assert (
        metrics[
            "embodied_action_token_grpo/gradient_direction_probe_best_advantage_logprob_delta_product_mean"
        ]
        > 0.0
    )
    assert (
        metrics[
            "embodied_action_token_grpo/gradient_direction_probe_scale_00/surrogate_weighted_objective_direction_advantage_logprob_delta_product_mean"
        ]
        > 0.0
    )
    assert (
        metrics[
            "embodied_action_token_grpo/gradient_direction_probe_scale_00/probe_unclipped_surrogate_loss_delta"
        ]
        < 0.0
    )
    assert (
        metrics[
            "embodied_action_token_grpo/gradient_direction_probe_best_surrogate_loss_delta"
        ]
        < 0.0
    )


def test_action_token_probe_reports_surrogate_weighted_direction_metrics() -> None:
    policy = TinyTokenPolicy()
    old_logprobs = torch.log_softmax(policy.logits.detach(), dim=-1)

    def trajectory(token: int, reward: float, task: str) -> EmbodiedTrajectory:
        item = EmbodiedTrajectory(task=task, reward=reward)
        item.actions.append(
            Action(
                step=0,
                kind="token",
                raw={"tokens": [token], "prompt": task},
                logprobs={"token_logprobs": [float(old_logprobs[token])]},
            )
        )
        return item

    groups = [
        EmbodiedTrajectoryGroup(
            [
                trajectory(1, 1.0, "weighted probe"),
                trajectory(0, 0.0, "weighted probe"),
            ]
        )
    ]
    backend = _backend(policy)
    result = asyncio.run(backend.train(groups))
    assert result.metrics["embodied_action_token_grpo/optimizer_step_completed"] == 1.0

    probe = backend.probe_logprob_metrics(groups)
    assert (
        probe[
            "embodied_action_token_grpo/surrogate_weighted/objective_direction_advantage_logprob_delta_product_mean"
        ]
        > 0.0
    )
    assert (
        probe[
            "embodied_action_token_grpo/surrogate_weighted/objective_direction_positive_ratio_mean"
        ]
        > 1.0
    )
    assert (
        probe[
            "embodied_action_token_grpo/surrogate_weighted/objective_direction_negative_ratio_mean"
        ]
        < 1.0
    )


def test_rlinf_chunk_mean_trajectory_reward_moves_sparse_success_tokens_up() -> None:
    base_policy = TinyTokenPolicy()
    old_logprobs = torch.log_softmax(base_policy.logits.detach(), dim=-1)

    def trajectory(token: int, reward: float, task: str) -> EmbodiedTrajectory:
        item = EmbodiedTrajectory(task=task, reward=reward)
        item.actions.append(
            Action(
                step=0,
                kind="token",
                raw={"tokens": [token, token], "prompt": task},
                logprobs={"token_logprobs": [float(old_logprobs[token])] * 2},
                metadata={
                    # Native LIBERO sparse success can live only on the
                    # trajectory reward; per-step env rewards may stay zero.
                    "primitive_rewards": [0.0, 0.0],
                    "primitive_loss_mask": [True, True],
                    "primitive_loss_mask_sum": 2,
                },
            )
        )
        return item

    groups = [
        EmbodiedTrajectoryGroup(
            [
                trajectory(1, 5.0, "native sparse success"),
                trajectory(0, 0.0, "native sparse success"),
            ]
        )
    ]
    policy = copy.deepcopy(base_policy)
    backend = ActionTokenGRPOBackend(
        policy,
        optimizer=torch.optim.SGD(policy.parameters(), lr=0.2),
        clip_epsilon=0.2,
        require_prompts=True,
        action_advantage_mode="rlinf_action_level_cumulative",
        rlinf_action_level_score_source="trajectory_reward",
        rlinf_action_level_mask_zero_variance_groups=True,
        rlinf_action_level_extra_global_normalization=True,
        normalize_advantages=True,
        advantage_normalization_scope="group",
        loss_aggregation="rlinf_chunk_mean",
        importance_sampling_level="token",
        logprob_microbatch_size=1,
    )

    result = asyncio.run(backend.train(groups))
    assert result.metrics["embodied_action_token_grpo/optimizer_step_completed"] == 1.0
    assert result.metrics["embodied_action_token_grpo/policy_parameters_updated"] == 1.0
    assert result.metrics[
        "embodied_action_token_grpo/trajectory_reward_positive_fraction"
    ] == pytest.approx(0.5)

    new_logprobs = torch.log_softmax(policy.logits.detach(), dim=-1)
    assert new_logprobs[1] > old_logprobs[1]
    assert new_logprobs[0] < old_logprobs[0]

    probe = backend.probe_logprob_metrics(groups)
    assert (
        probe[
            "embodied_action_token_grpo/objective_direction_advantage_logprob_delta_product_mean"
        ]
        > 0.0
    )
    assert (
        probe["embodied_action_token_grpo/objective_direction_positive_ratio_mean"]
        > 1.0
    )
    assert (
        probe["embodied_action_token_grpo/objective_direction_negative_ratio_mean"]
        < 1.0
    )


def test_rlinf_action_level_advantage_preserves_negative_trajectory_scores() -> None:
    from art_embodied import extract_action_token_examples
    from art_embodied.backends.action_token import (
        _attach_rlinf_action_level_token_advantages,
        _mask_rlinf_action_level_zero_variance_groups,
    )

    def trajectory(token: int, score: float) -> EmbodiedTrajectory:
        item = EmbodiedTrajectory(task="signed safety reward", reward=score)
        item.actions.append(
            Action(
                step=0,
                kind="token",
                raw={"tokens": [token, token], "prompt": "avoid collision"},
                logprobs={"token_logprobs": [-0.5, -0.5]},
                metadata={
                    "primitive_rewards": [score, 0.0],
                    "primitive_loss_mask": [True, False],
                    "primitive_loss_mask_sum": 1,
                },
            )
        )
        return item

    examples = extract_action_token_examples(
        [EmbodiedTrajectoryGroup([trajectory(0, -2.0), trajectory(1, -1.0)])],
        require_logprobs=True,
        require_prompts=True,
    )
    _attach_rlinf_action_level_token_advantages(
        examples,
        normalize=True,
        group_std_unbiased=False,
        eps=1.0e-6,
    )

    assert [
        example.metadata["rlinf_action_level_trajectory_score"] for example in examples
    ] == [-2.0, -1.0]
    assert examples[0].metadata["group_advantage"] < 0.0
    assert examples[1].metadata["group_advantage"] > 0.0
    report = _mask_rlinf_action_level_zero_variance_groups(examples, eps=1.0e-6)
    assert report["groups_masked"] == 0


def test_action_token_gradient_handoff_matches_single_process_rlinf_action_level_global_norm() -> (
    None
):
    old_logprob = math.log(0.5)

    def trajectory(token: int, rewards: list[float], task: str) -> EmbodiedTrajectory:
        item = EmbodiedTrajectory(task=task, reward=sum(rewards))
        item.actions.append(
            Action(
                step=0,
                kind="token",
                raw={"tokens": [token for _ in rewards], "prompt": task},
                logprobs={"token_logprobs": [old_logprob for _ in rewards]},
                metadata={
                    "primitive_rewards": rewards,
                    "primitive_loss_mask": [True for _ in rewards],
                    "primitive_loss_mask_sum": len(rewards),
                },
            )
        )
        return item

    groups = [
        EmbodiedTrajectoryGroup(
            [
                trajectory(1, [1.0, 0.0, 0.0, 0.0], "pick red cube"),
                trajectory(0, [0.0], "pick red cube"),
            ]
        ),
        EmbodiedTrajectoryGroup(
            [
                trajectory(1, [0.8, 0.2], "pick blue cube"),
                trajectory(0, [0.1, 0.0, 0.0], "pick blue cube"),
            ]
        ),
    ]
    global_example_count = 4
    global_token_count = 10

    from art_embodied import extract_action_token_examples
    from art_embodied.backends.action_token import (
        _apply_rlinf_action_level_extra_global_normalization,
        _attach_rlinf_action_level_token_advantages,
    )

    norm_examples = extract_action_token_examples(
        groups, require_logprobs=True, require_prompts=True
    )
    _attach_rlinf_action_level_token_advantages(
        norm_examples,
        normalize=True,
        group_std_unbiased=True,
    )
    global_norm = _apply_rlinf_action_level_extra_global_normalization(norm_examples)
    assert global_norm["applied"] is True

    def backend(
        policy: TinyTokenPolicy,
        *,
        optimizer=None,
        global_mean: float | None = None,
        global_scale: float | None = None,
    ) -> ActionTokenGRPOBackend:
        return ActionTokenGRPOBackend(
            policy,
            optimizer=optimizer,
            lr=0.05,
            clip_epsilon=0.2,
            require_prompts=True,
            action_advantage_mode="rlinf_action_level_cumulative",
            rlinf_action_level_extra_global_normalization=True,
            rlinf_action_level_global_advantage_mean=global_mean,
            rlinf_action_level_global_advantage_scale=global_scale,
            normalize_advantages=True,
            advantage_normalization_scope="group",
            advantage_std_unbiased=True,
            loss_aggregation="rlinf_token_mean",
            importance_sampling_level="token",
            logprob_microbatch_size=1,
        )

    base_policy = TinyTokenPolicy()

    async def run_single():
        policy = copy.deepcopy(base_policy)
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.05)
        result = await backend(policy, optimizer=optimizer).train(groups)
        return policy, result

    single_policy, single_result = asyncio.run(run_single())
    assert (
        single_result.metrics["embodied_action_token_grpo/optimizer_step_completed"]
        == 1.0
    )
    assert (
        single_result.metrics[
            "embodied_action_token_grpo/rlinf_action_level_extra_global_normalization"
        ]
        == 1.0
    )

    parent_policy = copy.deepcopy(base_policy)
    parent_optimizer = torch.optim.SGD(parent_policy.parameters(), lr=0.05)
    payloads = []

    async def run_worker(shard: list[EmbodiedTrajectoryGroup]):
        worker_policy = copy.deepcopy(base_policy)
        result = await backend(
            worker_policy,
            global_mean=float(global_norm["mean"]),
            global_scale=float(global_norm["scale"]),
        ).train(
            shard,
            _action_token_grpo_return_gradients=True,
            _action_token_grpo_global_example_count=global_example_count,
            _action_token_grpo_global_token_count=global_token_count,
        )
        assert (
            result.metrics["embodied_action_token_grpo/gradient_handoff_worker"] == 1.0
        )
        assert (
            result.metrics[
                "embodied_action_token_grpo/rlinf_action_level_extra_global_normalization"
            ]
            == 1.0
        )
        assert result.metrics[
            "embodied_action_token_grpo/rlinf_action_level_global_advantage_mean"
        ] == pytest.approx(global_norm["mean"])
        assert result.metrics[
            "embodied_action_token_grpo/rlinf_action_level_global_advantage_scale"
        ] == pytest.approx(global_norm["scale"])
        return _trainable_gradient_payload(worker_policy)

    for group in groups:
        payloads.append(asyncio.run(run_worker([group])))

    apply_metrics = apply_action_token_gradient_payloads(
        parent_policy, parent_optimizer, payloads
    )

    assert (
        apply_metrics["embodied_action_token_grpo/distributed_gradient_handoff"] == 1.0
    )
    torch.testing.assert_close(
        parent_policy.logits.detach(), single_policy.logits.detach()
    )


def test_action_token_gradient_handoff_reports_passing_logprob_alignment() -> None:
    policy = TinyTokenPolicy()
    backend = ActionTokenGRPOBackend(
        policy,
        lr=0.1,
        clip_epsilon=0.2,
        require_prompts=True,
        loss_aggregation="token_mean",
        advantage_normalization_scope="group",
        logprob_microbatch_size=1,
        pre_update_logprob_kl_tolerance=0.2,
        pre_update_ratio_tolerance=0.2,
    )

    async def run_worker():
        return await backend.train(
            [_group(1.0, 0.0, "alignment telemetry")],
            _action_token_grpo_return_gradients=True,
            _action_token_grpo_global_example_count=2,
            _action_token_grpo_global_token_count=2,
        )

    result = asyncio.run(run_worker())

    assert result.metrics["embodied_action_token_grpo/gradient_handoff_worker"] == 1.0
    assert result.metrics["embodied_action_token_grpo/optimizer_step_completed"] == 0.0
    assert (
        result.metrics["embodied_action_token_grpo/old_new_logprobs_evaluated"] == 1.0
    )
    assert result.metrics["embodied_action_token_grpo/old_new_logprobs_aligned"] == 1.0
    assert (
        result.metrics["embodied_action_token_grpo/alignment_old_new_logprobs_aligned"]
        == 1.0
    )
    assert (
        result.metrics["embodied_action_token_grpo/alignment_approx_kl_abs_mean"] <= 0.2
    )


def test_action_token_probe_logprob_debug_samples_are_bounded() -> None:
    policy = TinyTokenPolicy()
    backend = _backend(policy)

    report = backend.probe_logprob_debug_samples(
        [_group(1.0, 0.0, "debug sample")],
        max_examples=1,
        max_tokens=1,
    )

    assert report["examples_before_filter"] == 2
    assert report["examples_after_filter"] == 2
    assert report["max_examples"] == 1
    assert report["max_tokens"] == 1
    assert len(report["samples"]) == 1
    sample = report["samples"][0]
    assert sample["token_count"] == 1
    assert sample["tokens_head"] == [1]
    assert sample["old_logprobs_head"] == pytest.approx([math.log(0.5)])
    expected_current = torch.log_softmax(policy.logits, dim=-1)[1].item()
    assert sample["current_logprobs_head"] == pytest.approx([expected_current])
    assert sample["delta_abs_mean"] == pytest.approx(
        abs(expected_current - math.log(0.5))
    )


def test_action_token_gradient_handoff_writes_zero_payload_for_zero_signal_shard(
    tmp_path: Path,
) -> None:
    group = _group(0.5, 0.5, "ambiguous task")
    policy = TinyTokenPolicy()
    backend = _backend(policy)
    gradient_path = tmp_path / "gradients.pt"

    async def run_worker():
        return await backend.train(
            [group],
            _action_token_grpo_return_gradients=True,
            _action_token_grpo_gradient_output_path=str(gradient_path),
            _action_token_grpo_global_example_count=2,
            _action_token_grpo_global_token_count=2,
        )

    result = asyncio.run(run_worker())
    payload = load_action_token_gradient_payload(gradient_path)

    assert result.metrics["embodied_action_token_grpo/gradient_handoff_worker"] == 1.0
    assert result.metrics["embodied_action_token_grpo/optimizer_step_completed"] == 0.0
    assert (
        result.metrics[
            "embodied_action_token_grpo/optimizer_step_skipped_no_group_signal"
        ]
        == 1.0
    )
    assert set(payload["gradients"]) == {"logits"}
    assert torch.count_nonzero(payload["gradients"]["logits"]).item() == 0


def test_action_token_gradient_handoff_skips_parent_step_when_all_gradients_zero() -> (
    None
):
    policy = TinyTokenPolicy()
    before = policy.logits.detach().clone()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.1, weight_decay=0.5)
    zero_payload = {
        "format": "art_embodied_action_token_grpo_gradients_v1",
        "gradients": {"logits": torch.zeros_like(policy.logits.detach())},
        "shapes": {"logits": tuple(policy.logits.shape)},
        "missing_gradients": [],
    }

    metrics = apply_action_token_gradient_payloads(policy, optimizer, [zero_payload])

    assert metrics["embodied_action_token_grpo/optimizer_step_completed"] == 0.0
    assert (
        metrics["embodied_action_token_grpo/optimizer_step_skipped_no_group_signal"]
        == 1.0
    )
    torch.testing.assert_close(policy.logits.detach(), before)


def test_action_token_gradient_handoff_rejects_nonfinite_payload_before_update() -> (
    None
):
    policy = TinyTokenPolicy()
    before = policy.logits.detach().clone()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.1, weight_decay=0.5)
    payload = {
        "format": "art_embodied_action_token_grpo_gradients_v1",
        "gradients": {"logits": torch.tensor([float("inf"), 0.0], dtype=torch.float32)},
        "shapes": {"logits": tuple(policy.logits.shape)},
        "missing_gradients": [],
    }

    with pytest.raises(FloatingPointError, match="non-finite distributed gradient"):
        apply_action_token_gradient_payloads(policy, optimizer, [payload])

    torch.testing.assert_close(policy.logits.detach(), before)
    assert policy.logits.grad is None
    assert not optimizer.state


def test_action_token_gradient_handoff_fails_when_parent_has_no_trainable_tensors() -> (
    None
):
    policy = TinyTokenPolicy()
    policy.logits.requires_grad_(False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.1, weight_decay=0.0)
    payload = {
        "format": "art_embodied_action_token_grpo_gradients_v1",
        "gradients": {"logits": torch.ones_like(policy.logits.detach())},
        "shapes": {"logits": tuple(policy.logits.shape)},
        "missing_gradients": [],
    }

    with pytest.raises(ValueError, match="parent policy has no trainable parameters"):
        apply_action_token_gradient_payloads(policy, optimizer, [payload])


def test_action_token_gradient_handoff_can_scale_parent_optimizer_lr() -> None:
    policy = TinyTokenPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    before = policy.logits.detach().clone()
    payload = {
        "format": "art_embodied_action_token_grpo_gradients_v1",
        "gradients": {"logits": torch.tensor([2.0, -4.0], dtype=torch.float32)},
        "shapes": {"logits": tuple(policy.logits.shape)},
        "missing_gradients": [],
    }

    metrics = apply_action_token_gradient_payloads(
        policy,
        optimizer,
        [payload],
        optimizer_lr_scale=0.25,
    )

    assert metrics["embodied_action_token_grpo/optimizer_step_completed"] == 1.0
    assert metrics["embodied_action_token_grpo/optimizer_lr_scale"] == pytest.approx(
        0.25
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    expected = before - 0.1 * 0.25 * payload["gradients"]["logits"]
    torch.testing.assert_close(policy.logits.detach(), expected)


def test_gradient_handoff_uses_caller_metric_prefix_for_signal_and_clipping() -> None:
    policy = TinyTokenPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    payload = {
        "format": "art_embodied_action_token_grpo_gradients_v1",
        "gradients": {"logits": torch.tensor([2.0, -4.0], dtype=torch.float32)},
        "shapes": {"logits": tuple(policy.logits.shape)},
        "missing_gradients": [],
    }
    prefix = "embodied_flow_sde_grpo"

    metrics = apply_action_token_gradient_payloads(
        policy,
        optimizer,
        [payload],
        max_grad_norm=1.0,
        prefix=prefix,
    )

    assert metrics[f"{prefix}/optimizer_step_completed"] == 1.0
    assert metrics[f"{prefix}/grad_norm_before_clip"] == pytest.approx(20**0.5)
    assert metrics[f"{prefix}/grad_abs_max_before_clip"] == pytest.approx(4.0)
    assert metrics[f"{prefix}/grad_norm"] == pytest.approx(1.0)
    assert not any(key.startswith("embodied_action_token_grpo/") for key in metrics)


def test_gradient_handoff_reports_worker_gradient_coherence() -> None:
    policy = TinyTokenPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    prefix = "embodied_flow_sde_grpo"

    def payload(values: list[float]) -> dict[str, object]:
        tensor = torch.tensor(values, dtype=torch.float32)
        return {
            "format": "art_embodied_action_token_grpo_gradients_v1",
            "gradients": {"logits": tensor},
            "shapes": {"logits": tuple(tensor.shape)},
            "missing_gradients": [],
        }

    aligned = apply_action_token_gradient_payloads(
        policy,
        optimizer,
        [payload([1.0, 0.0]), payload([1.0, 0.0])],
        prefix=prefix,
    )
    assert aligned[f"{prefix}/worker_gradient_pairwise_cosine_mean"] == pytest.approx(
        1.0
    )
    assert aligned[f"{prefix}/worker_gradient_resultant_ratio"] == pytest.approx(1.0)
    assert aligned[f"{prefix}/worker_gradient_noise_to_signal_ratio"] == pytest.approx(
        0.0
    )
    assert aligned[
        f"{prefix}/worker_gradient_effective_aligned_workers"
    ] == pytest.approx(2.0)

    opposed = apply_action_token_gradient_payloads(
        policy,
        optimizer,
        [payload([1.0, 0.0]), payload([-1.0, 0.0])],
        prefix=prefix,
        skip_optimizer_step_without_policy_gradient_signal=True,
    )
    assert opposed[f"{prefix}/worker_gradient_pairwise_cosine_mean"] == pytest.approx(
        -1.0
    )
    assert opposed[f"{prefix}/worker_gradient_resultant_ratio"] == pytest.approx(0.0)
    assert opposed[f"{prefix}/worker_gradient_signal_to_rms_ratio"] == pytest.approx(
        0.0
    )
    assert opposed[f"{prefix}/optimizer_step_completed"] == 0.0


def test_task_pcgrad_is_noop_for_nonconflicting_task_gradients() -> None:
    policy = TinyTokenPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    before = policy.logits.detach().clone()

    def payload(task_key: str, values: list[float]) -> dict[str, object]:
        tensor = torch.tensor(values, dtype=torch.float32)
        return {
            "format": "art_embodied_action_token_grpo_gradients_v1",
            "task_key": task_key,
            "gradients": {"logits": tensor},
            "shapes": {"logits": tuple(tensor.shape)},
            "missing_gradients": [],
        }

    metrics = apply_action_token_gradient_payloads(
        policy,
        optimizer,
        [payload("task-b", [0.0, 2.0]), payload("task-a", [1.0, 0.0])],
        gradient_aggregation="task_pcgrad",
    )

    torch.testing.assert_close(
        policy.logits.detach(), before - torch.tensor([0.1, 0.2])
    )
    assert metrics["embodied_action_token_grpo/task_gradient_conflicting_pairs"] == 0.0
    assert (
        metrics["embodied_action_token_grpo/task_gradient_projected_conflicting_pairs"]
        == 0.0
    )
    assert metrics["embodied_action_token_grpo/task_pcgrad_projections"] == 0.0
    assert metrics[
        "embodied_action_token_grpo/task_gradient_projected_to_raw_norm_ratio"
    ] == pytest.approx(1.0)


def test_task_pcgrad_deterministically_prevents_exact_cancellation() -> None:
    def run(payloads: list[dict[str, object]]) -> tuple[torch.Tensor, dict[str, float]]:
        policy = TinyTokenPolicy()
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
        before = policy.logits.detach().clone()
        metrics = apply_action_token_gradient_payloads(
            policy,
            optimizer,
            payloads,
            gradient_aggregation="task_pcgrad",
        )
        return policy.logits.detach() - before, metrics

    def payload(task_key: str, values: list[float]) -> dict[str, object]:
        tensor = torch.tensor(values, dtype=torch.float32)
        return {
            "format": "art_embodied_action_token_grpo_gradients_v1",
            "task_key": task_key,
            "gradients": {"logits": tensor},
            "shapes": {"logits": tuple(tensor.shape)},
            "missing_gradients": [],
        }

    rows = [
        payload("task-a", [1.0, 0.0]),
        payload("task-b", [-1.0, 1.0]),
    ]
    forward_delta, forward_metrics = run(rows)
    reverse_delta, reverse_metrics = run(list(reversed(rows)))

    torch.testing.assert_close(forward_delta, reverse_delta)
    assert torch.linalg.vector_norm(forward_delta).item() > 0.0
    assert (
        forward_metrics["embodied_action_token_grpo/task_gradient_conflicting_pairs"]
        == 1.0
    )
    assert (
        forward_metrics[
            "embodied_action_token_grpo/task_gradient_projected_conflicting_pairs"
        ]
        == 0.0
    )
    assert (
        forward_metrics[
            "embodied_action_token_grpo/task_gradient_projected_pairwise_cosine_min"
        ]
        >= 0.0
    )
    assert forward_metrics["embodied_action_token_grpo/task_pcgrad_projections"] > 0.0
    assert forward_metrics == reverse_metrics


def test_action_token_gradient_handoff_reports_continuous_adam_state_steps() -> None:
    policy = TinyTokenPolicy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01, weight_decay=0.0)
    payload = {
        "format": "art_embodied_action_token_grpo_gradients_v1",
        "gradients": {"logits": torch.tensor([1.0, -1.0], dtype=torch.float32)},
        "shapes": {"logits": tuple(policy.logits.shape)},
        "missing_gradients": [],
    }

    first = apply_action_token_gradient_payloads(policy, optimizer, [payload])
    second = apply_action_token_gradient_payloads(policy, optimizer, [payload])

    prefix = "embodied_action_token_grpo"
    assert first[f"{prefix}/optimizer_state_entries_before"] == 0.0
    assert first[f"{prefix}/optimizer_state_step_min_after"] == 1.0
    assert first[f"{prefix}/optimizer_state_step_max_after"] == 1.0
    assert second[f"{prefix}/optimizer_state_step_min_before"] == 1.0
    assert second[f"{prefix}/optimizer_state_step_max_before"] == 1.0
    assert second[f"{prefix}/optimizer_state_step_min_after"] == 2.0
    assert second[f"{prefix}/optimizer_state_step_max_after"] == 2.0


def test_action_token_grpo_skips_cleanly_when_reward_filter_removes_everything() -> (
    None
):
    policy = TinyTokenPolicy()
    backend = ActionTokenGRPOBackend(
        policy,
        lr=0.1,
        clip_epsilon=0.2,
        require_prompts=True,
        filter_rewards=True,
        rewards_lower_bound=10.0,
        rewards_upper_bound=20.0,
    )
    before = policy.logits.detach().clone()

    async def run_train():
        return await backend.train([_group(1.0, 0.0, "filtered task")])

    result = asyncio.run(run_train())

    assert result.metrics["embodied_action_token_grpo/examples"] == 0.0
    assert result.metrics["embodied_action_token_grpo/optimizer_step_completed"] == 0.0
    assert (
        result.metrics[
            "embodied_action_token_grpo/optimizer_step_skipped_no_group_signal"
        ]
        == 1.0
    )
    assert result.metrics["embodied_action_token_grpo/gradient_handoff_worker"] == 0.0
    torch.testing.assert_close(policy.logits.detach(), before)


def test_action_token_grpo_filtered_empty_clears_stale_worker_gradients(
    tmp_path: Path,
) -> None:
    policy = TinyTokenPolicy()
    backend = _backend(policy)

    async def run_signal_train():
        return await backend.train([_group(1.0, 0.0, "warm signal")])

    signal_result = asyncio.run(run_signal_train())
    assert (
        signal_result.metrics["embodied_action_token_grpo/optimizer_step_completed"]
        == 1.0
    )
    assert policy.logits.grad is not None
    assert torch.count_nonzero(policy.logits.grad.detach()).item() > 0

    backend.filter_rewards = True
    backend.rewards_lower_bound = 10.0
    backend.rewards_upper_bound = 20.0
    gradient_path = tmp_path / "filtered_gradients.pt"

    async def run_filtered_worker():
        return await backend.train(
            [_group(1.0, 0.0, "filtered after signal")],
            _action_token_grpo_return_gradients=True,
            _action_token_grpo_gradient_output_path=str(gradient_path),
            _action_token_grpo_global_example_count=0,
            _action_token_grpo_global_token_count=0,
        )

    result = asyncio.run(run_filtered_worker())
    payload = load_action_token_gradient_payload(gradient_path)

    assert result.metrics["embodied_action_token_grpo/examples"] == 0.0
    assert result.metrics["embodied_action_token_grpo/grad_norm"] == 0.0
    assert torch.count_nonzero(payload["gradients"]["logits"]).item() == 0


def test_action_token_probe_reports_policy_step_and_action_dim_direction_buckets() -> (
    None
):
    policy = TinyTokenPolicy()
    group = EmbodiedTrajectoryGroup(
        [
            _trajectory_with_bucket_metadata(1, 1.0, "bucket probe"),
            _trajectory_with_bucket_metadata(0, 0.0, "bucket probe"),
        ]
    )
    backend = _backend(policy)

    metrics = backend.probe_logprob_metrics([group])

    assert metrics["embodied_action_token_grpo/by_policy_step/bucket_count"] == 1.0
    assert metrics["embodied_action_token_grpo/by_action_dim/bucket_count"] == 2.0
    assert (
        "embodied_action_token_grpo/by_policy_step/policy_step_03/"
        "objective_direction_advantage_logprob_delta_product_mean"
    ) in metrics
    assert (
        "embodied_action_token_grpo/by_action_dim/action_dim_00/"
        "objective_direction_advantage_logprob_delta_product_mean"
    ) in metrics
    assert (
        "embodied_action_token_grpo/by_action_dim/action_dim_01/"
        "objective_direction_advantage_logprob_delta_product_mean"
    ) in metrics


def _trajectory_with_bucket_metadata(
    token: int, reward: float, task: str
) -> EmbodiedTrajectory:
    item = EmbodiedTrajectory(task=task, reward=reward)
    tokens = [token, 1 - token, token, 1 - token]
    item.actions.append(
        Action(
            step=3,
            kind="token",
            raw={"tokens": tokens, "prompt": task},
            logprobs={"token_logprobs": [math.log(0.5)] * len(tokens)},
            metadata={
                "policy_step": 3,
                "primitive_loss_mask": [True, True],
            },
        )
    )
    return item


@pytest.mark.parametrize("corruption", ["missing", "extra", "shape"])
def test_incomplete_worker_gradient_rejected_before_mutation(corruption):
    policy = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01)
    for parameter in policy.parameters():
        parameter.grad = torch.ones_like(parameter)
    before = {name: p.detach().clone() for name, p in policy.named_parameters()}
    gradients_before = {name: p.grad.clone() for name, p in policy.named_parameters()}
    good = _trainable_gradient_payload(policy)
    bad = copy.deepcopy(good)
    if corruption == "missing":
        bad["gradients"].pop("bias")
    elif corruption == "extra":
        bad["gradients"]["unknown"] = torch.zeros(1)
    else:
        bad["gradients"]["bias"] = torch.zeros(9)
    with pytest.raises(ValueError, match="Gradient payload"):
        apply_action_token_gradient_payloads(policy, optimizer, [good, bad])
    assert not optimizer.state
    for name, parameter in policy.named_parameters():
        torch.testing.assert_close(parameter, before[name])
        torch.testing.assert_close(parameter.grad, gradients_before[name])
