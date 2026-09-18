from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from art_embodied.backends.flow_sde import (  # noqa: E402
    FLOW_SDE_ACTION_DIMENSION_MASK_KEY,
    FLOW_SDE_REPLAY_SELECTED_KEY,
    TRANSIENT_FLOW_SDE_ROLLOUT_KEY,
    FlowSDEExample,
    FlowSDEGRPOBackend,
    _flow_sde_microbatches,
    _flow_sde_subupdates,
    _should_stop_flow_sde_before_optimizer,
    flow_sde_replay_selection_counts,
    precalculate_flow_sde_logprobs,
    prepare_flow_sde_examples,
)
from art_embodied.checkpointing import CHECKPOINT_COMPLETE_MARKER  # noqa: E402
from art_embodied.config import (  # noqa: E402
    FullUpdateScheduleConfig,
    RlinfActorBatchScheduleConfig,
)
from art_embodied.policies.flow_sde import FlowSDETransitionRecord  # noqa: E402
from art_embodied.policies.pi_flow_sde import (  # noqa: E402
    PIFlowModelInputs,
    PIFlowSDERollout,
)
from art_embodied.trajectories import (  # noqa: E402
    Action,
    EmbodiedTrajectory,
    EmbodiedTrajectoryGroup,
)


class _Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def flow_sde_logprobs(self, rollout: PIFlowSDERollout) -> torch.Tensor:
        feature = rollout.inputs.language_tokens[:, :1].float().view(1, 1, 1)
        return rollout.transition.old_logprobs + self.weight * feature

    def save_checkpoint(self, path: str) -> None:
        destination = Path(path)
        destination.mkdir(parents=True)
        torch.save(self.state_dict(), destination / "adapter.pt")


class _ReferencePolicy(_Policy):
    def __init__(self, *, fail_reference: bool = False) -> None:
        super().__init__()
        self.fail_reference = fail_reference
        self.reference_calls = 0

    def flow_sde_logprobs(self, rollout: PIFlowSDERollout) -> torch.Tensor:
        feature = rollout.inputs.language_tokens[:, :1].float().view(-1, 1, 1)
        return rollout.transition.old_logprobs + self.weight * feature

    def flow_sde_reference_logprobs(
        self,
        rollout: PIFlowSDERollout,
    ) -> torch.Tensor:
        self.reference_calls += 1
        if self.fail_reference:
            raise AssertionError("reference scoring must not run")
        return torch.zeros_like(rollout.transition.old_logprobs)


def _rollout(feature: int, *, language_length: int = 1) -> PIFlowSDERollout:
    old = torch.zeros(1, 2, 1)
    inputs = PIFlowModelInputs(
        images=(torch.zeros(1, 3, 2, 2),),
        image_masks=(torch.ones(1, dtype=torch.bool),),
        language_tokens=torch.full((1, language_length), feature),
        language_masks=torch.ones(1, language_length, dtype=torch.bool),
        state=None,
    )
    return PIFlowSDERollout(
        actions=torch.zeros(1, 2, 1),
        transition=FlowSDETransitionRecord(
            previous_states=torch.zeros(1, 2, 1),
            next_states=torch.zeros(1, 2, 1),
            selected_indices=torch.zeros(1, dtype=torch.long),
            old_logprobs=old,
        ),
        inputs=inputs,
    )


def _trajectory(*, reward: float, feature: int) -> EmbodiedTrajectory:
    return EmbodiedTrajectory(
        task="same reset",
        reward=reward,
        actions=[
            Action(
                step=0,
                kind="continuous",
                raw=[[0.0], [0.0]],
                metadata={TRANSIENT_FLOW_SDE_ROLLOUT_KEY: _rollout(feature)},
            )
        ],
    )


def _group() -> EmbodiedTrajectoryGroup:
    return EmbodiedTrajectoryGroup(
        [
            _trajectory(reward=0.0, feature=-1),
            _trajectory(reward=1.0, feature=1),
        ],
        metadata={"environment_seed": 7},
    )


def test_prepare_flow_sde_examples_broadcasts_group_advantage_to_chunks() -> None:
    examples, advantages, kept_groups = prepare_flow_sde_examples(
        [_group()],
        group_size=2,
        advantage_epsilon=1.0e-6,
        advantage_std_unbiased=True,
        filter_rewards=True,
        rewards_lower_bound=0.1,
        rewards_upper_bound=0.9,
    )

    assert len(examples) == 2
    assert kept_groups == 1
    assert advantages[0] == pytest.approx(-advantages[1])
    assert advantages[1] > 0.0
    assert all(example.loss_mask for example in examples)


def test_prepare_flow_sde_examples_retains_executed_action_prefix() -> None:
    group = _group()
    for trajectory in group.trajectories:
        trajectory.actions[0].metadata["primitive_loss_mask"] = [True, False]
        trajectory.actions[0].metadata["primitive_loss_mask_sum"] = 1

    examples, _advantages, _kept_groups = prepare_flow_sde_examples(
        [group],
        group_size=2,
        advantage_epsilon=1.0e-6,
        advantage_std_unbiased=True,
        filter_rewards=False,
        rewards_lower_bound=None,
        rewards_upper_bound=None,
    )

    assert [example.action_mask for example in examples] == [
        (True, False),
        (True, False),
    ]


def test_prepare_flow_sde_examples_masks_non_executed_action_dimensions() -> None:
    group = _group()
    for trajectory in group.trajectories:
        metadata = trajectory.actions[0].metadata
        metadata["primitive_loss_mask"] = [True, False]
        metadata["primitive_loss_mask_sum"] = 1
        metadata[FLOW_SDE_ACTION_DIMENSION_MASK_KEY] = [False, True]

    examples, _advantages, _kept_groups = prepare_flow_sde_examples(
        [group],
        group_size=2,
        advantage_epsilon=1.0e-6,
        advantage_std_unbiased=True,
        filter_rewards=False,
        rewards_lower_bound=None,
        rewards_upper_bound=None,
    )

    assert [example.action_mask for example in examples] == [
        ((False, True), (False, False)),
        ((False, True), (False, False)),
    ]


def test_flow_sde_microbatches_separate_incompatible_prompt_lengths() -> None:
    examples = [
        FlowSDEExample(
            rollout=_rollout(index + 1, language_length=length),
            group_index=0,
            trajectory_index=index,
            action_index=0,
            reward=float(index),
            loss_mask=True,
            trajectory_primitive_steps=2,
        )
        for index, length in enumerate((2, 3, 2, 3, 2))
    ]
    batches = _flow_sde_microbatches(
        examples,
        [float(index) for index in range(len(examples))],
        microbatch_size=2,
    )

    assert [len(batch) for batch, _advantages in batches] == [2, 1, 2]
    assert [
        batch[0].rollout.inputs.language_tokens.shape[1]
        for batch, _advantages in batches
    ] == [2, 2, 3]
    for batch, _advantages in batches:
        PIFlowSDERollout.concatenate([example.rollout for example in batch])


def test_flow_sde_precalculate_freezes_exact_training_batch_score() -> None:
    class BatchGeometryPolicy(_Policy):
        def flow_sde_logprobs(self, rollout: PIFlowSDERollout) -> torch.Tensor:
            return (
                torch.full_like(
                    rollout.transition.old_logprobs,
                    float(rollout.inputs.batch_size),
                )
                + self.weight
            )

    examples = [
        FlowSDEExample(
            rollout=_rollout(index + 1),
            group_index=0,
            trajectory_index=index,
            action_index=0,
            reward=float(index),
            loss_mask=True,
            trajectory_primitive_steps=2,
        )
        for index in range(2)
    ]
    policy = BatchGeometryPolicy()

    refreshed, report = precalculate_flow_sde_logprobs(
        policy,
        examples,
        [-1.0, 1.0],
        microbatch_size=2,
        device="cpu",
    )

    combined = PIFlowSDERollout.concatenate([example.rollout for example in refreshed])
    torch.testing.assert_close(
        combined.transition.old_logprobs,
        policy.flow_sde_logprobs(combined),
    )
    assert report["old_logprob_rescore_rows"] == 2.0
    assert report["old_logprob_rescore_microbatches"] == 1.0
    assert report["old_logprob_rescore_previous_abs_delta_mean"] == 2.0


def test_flow_sde_precalculate_preserves_unexecuted_native_chunk_scores() -> None:
    class ExecutedPrefixPolicy(_Policy):
        def flow_sde_logprobs(self, rollout: PIFlowSDERollout) -> torch.Tensor:
            batch = rollout.inputs.batch_size
            return torch.full((batch, 1, 1), 3.0) + self.weight

    rollout = _rollout(1)
    full_old = torch.full((1, 2, 3), 7.0)
    rollout = PIFlowSDERollout(
        actions=torch.zeros(1, 2, 3),
        transition=FlowSDETransitionRecord(
            previous_states=torch.zeros(1, 2, 3),
            next_states=torch.zeros(1, 2, 3),
            selected_indices=torch.zeros(1, dtype=torch.long),
            old_logprobs=full_old,
        ),
        inputs=rollout.inputs,
    )
    example = FlowSDEExample(
        rollout=rollout,
        group_index=0,
        trajectory_index=0,
        action_index=0,
        reward=1.0,
        loss_mask=True,
        trajectory_primitive_steps=2,
    )

    refreshed, _report = precalculate_flow_sde_logprobs(
        ExecutedPrefixPolicy(),
        [example],
        [1.0],
        microbatch_size=1,
        device="cpu",
    )

    old = refreshed[0].rollout.transition.old_logprobs
    assert old[0, 0, 0].item() == 3.0
    torch.testing.assert_close(old[0, 0, 1:], torch.full((2,), 7.0))
    torch.testing.assert_close(old[0, 1], torch.full((3,), 7.0))


def test_prepare_flow_sde_examples_skips_explicitly_unselected_replay_rows() -> None:
    group = _group()
    for trajectory in group.trajectories:
        trajectory.actions[0].metadata["primitive_loss_mask_sum"] = 2
        trajectory.actions.append(
            Action(
                step=1,
                kind="continuous",
                raw=[[0.0], [0.0]],
                metadata={
                    FLOW_SDE_REPLAY_SELECTED_KEY: False,
                    "primitive_loss_mask_sum": 2,
                },
            )
        )

    examples, advantages, _kept_groups = prepare_flow_sde_examples(
        [group],
        group_size=2,
        advantage_epsilon=1.0e-6,
        advantage_std_unbiased=True,
        filter_rewards=False,
        rewards_lower_bound=None,
        rewards_upper_bound=None,
    )

    assert len(examples) == 2
    assert len(advantages) == 2
    assert all(example.action_index == 0 for example in examples)
    assert all(example.trajectory_primitive_steps == 4 for example in examples)
    assert flow_sde_replay_selection_counts([group]) == (4, 2)


def test_flow_sde_backend_updates_advantage_direction_and_publishes_checkpoint(
    tmp_path: Path,
) -> None:
    policy = _Policy()
    backend = FlowSDEGRPOBackend(
        policy=policy,
        optimizer=None,
        device="cpu",
        learning_rate=0.1,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        epsilon=1.0e-8,
        max_grad_norm=1.0,
        microbatch_size=1,
        optimizer_steps_per_update=1,
        training_schedule=FullUpdateScheduleConfig(type="full_update"),
        max_episode_steps=2,
        group_size=2,
        advantage_epsilon=1.0e-6,
        advantage_std_unbiased=True,
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.2,
        clip_ratio_c=3.0,
        filter_rewards=True,
        rewards_lower_bound=0.1,
        rewards_upper_bound=0.9,
        checkpoint_dir=tmp_path / "checkpoints",
        config_fingerprint="config",
        resume_contract_fingerprint="resume",
    )

    result = asyncio.run(backend.train([_group()]))

    assert result.step == 1
    assert policy.weight.item() > 0.0
    checkpoint = Path(result.checkpoint_path)
    assert checkpoint.is_dir()
    assert (checkpoint / CHECKPOINT_COMPLETE_MARKER).is_file()
    assert (checkpoint / "policy/adapter.pt").is_file()
    assert result.metrics["embodied_flow_sde_grpo/groups_kept"] == 1.0
    assert result.metrics["embodied_flow_sde_grpo/gradient_norm"] > 0.0


def test_flow_sde_backend_reference_anchor_pulls_policy_toward_sft(
    tmp_path: Path,
) -> None:
    policy = _ReferencePolicy()
    policy.weight.data.fill_(0.5)
    backend = FlowSDEGRPOBackend(
        policy=policy,
        optimizer=None,
        device="cpu",
        learning_rate=0.1,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        epsilon=1.0e-8,
        max_grad_norm=None,
        microbatch_size=2,
        optimizer_steps_per_update=1,
        training_schedule=FullUpdateScheduleConfig(type="full_update"),
        max_episode_steps=2,
        group_size=2,
        advantage_epsilon=1.0e-6,
        advantage_std_unbiased=True,
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.2,
        clip_ratio_c=3.0,
        filter_rewards=False,
        rewards_lower_bound=None,
        rewards_upper_bound=None,
        checkpoint_dir=tmp_path / "checkpoints",
        config_fingerprint="config",
        resume_contract_fingerprint="resume",
        reference_kl_coefficient=1.0,
    )
    examples = [
        FlowSDEExample(
            rollout=_rollout(1),
            group_index=0,
            trajectory_index=index,
            action_index=0,
            reward=0.0,
            loss_mask=True,
            trajectory_primitive_steps=2,
        )
        for index in range(2)
    ]

    metrics = backend._train_subupdate(
        examples,
        [0.0, 0.0],
        loss_denominator=2,
        length_normalized=False,
    )

    assert policy.reference_calls == 1
    assert policy.weight.item() < 0.5
    assert metrics["policy_loss"] == pytest.approx(0.0)
    assert metrics["reference_kl"] > 0.0
    assert metrics["reference_kl_loss"] > 0.0


def test_flow_sde_backend_zero_reference_coefficient_skips_reference_forward(
    tmp_path: Path,
) -> None:
    policy = _ReferencePolicy(fail_reference=True)
    backend = FlowSDEGRPOBackend(
        policy=policy,
        optimizer=None,
        device="cpu",
        learning_rate=0.1,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        epsilon=1.0e-8,
        max_grad_norm=None,
        microbatch_size=2,
        optimizer_steps_per_update=1,
        training_schedule=FullUpdateScheduleConfig(type="full_update"),
        max_episode_steps=2,
        group_size=2,
        advantage_epsilon=1.0e-6,
        advantage_std_unbiased=True,
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.2,
        clip_ratio_c=3.0,
        filter_rewards=False,
        rewards_lower_bound=None,
        rewards_upper_bound=None,
        checkpoint_dir=tmp_path / "checkpoints",
        config_fingerprint="config",
        resume_contract_fingerprint="resume",
        reference_kl_coefficient=0.0,
    )

    backend._train_subupdate(
        [
            FlowSDEExample(
                rollout=_rollout(1),
                group_index=0,
                trajectory_index=index,
                action_index=0,
                reward=0.0,
                loss_mask=True,
                trajectory_primitive_steps=2,
            )
            for index in range(2)
        ],
        [0.0, 0.0],
        loss_denominator=2,
        length_normalized=False,
    )

    assert policy.reference_calls == 0


def test_flow_sde_backend_skips_gradient_that_exceeds_measured_kl(
    tmp_path: Path,
) -> None:
    policy = _Policy()
    schedule = RlinfActorBatchScheduleConfig(
        type="rlinf_actor_global_batch",
        global_batch_size=2,
        actor_seed=42,
        actor_world_size=2,
        rank_local_shuffle=False,
        groups_per_process_per_rollout_epoch=1,
        action_chunk_size=2,
        update_epochs=1,
        strict_geometry=True,
        pre_update_alignment_guard="first_subupdate",
        max_approximate_kl=1.0e-6,
        min_optimizer_steps_before_kl_stop=1,
    )
    backend = FlowSDEGRPOBackend(
        policy=policy,
        optimizer=None,
        device="cpu",
        learning_rate=0.1,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        epsilon=1.0e-8,
        max_grad_norm=1.0,
        microbatch_size=1,
        optimizer_steps_per_update=2,
        training_schedule=schedule,
        max_episode_steps=2,
        group_size=2,
        advantage_epsilon=1.0e-6,
        advantage_std_unbiased=True,
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.2,
        clip_ratio_c=3.0,
        filter_rewards=True,
        rewards_lower_bound=0.1,
        rewards_upper_bound=0.9,
        checkpoint_dir=tmp_path / "checkpoints",
        config_fingerprint="config",
        resume_contract_fingerprint="resume",
    )

    result = asyncio.run(backend.train([_group(), _group()]))

    assert result.metrics["embodied_flow_sde_grpo/optimizer_subupdates"] == 1.0
    assert result.metrics["embodied_flow_sde_grpo/optimizer_subupdates_planned"] == 2.0
    assert result.metrics["embodied_flow_sde_grpo/trust_region_early_stop"] == 1.0
    assert (
        result.metrics["embodied_flow_sde_grpo/trust_region_stop_abs_approximate_kl"]
        > 1.0e-6
    )


def test_flow_sde_backend_rejects_misalignment_before_optimizer_step(
    tmp_path: Path,
) -> None:
    policy = _Policy()
    policy.weight.data.fill_(1.0)
    backend = FlowSDEGRPOBackend(
        policy=policy,
        optimizer=None,
        device="cpu",
        learning_rate=0.1,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        epsilon=1.0e-8,
        max_grad_norm=1.0,
        pre_update_logprob_kl_tolerance=0.02,
        microbatch_size=1,
        optimizer_steps_per_update=1,
        training_schedule=FullUpdateScheduleConfig(type="full_update"),
        max_episode_steps=2,
        group_size=2,
        advantage_epsilon=1.0e-6,
        advantage_std_unbiased=True,
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.2,
        clip_ratio_c=3.0,
        filter_rewards=True,
        rewards_lower_bound=0.1,
        rewards_upper_bound=0.9,
        checkpoint_dir=tmp_path / "checkpoints",
        config_fingerprint="config",
        resume_contract_fingerprint="resume",
    )

    with pytest.raises(RuntimeError, match="before optimizer step"):
        asyncio.run(backend.train([_group()]))

    assert policy.weight.item() == 1.0
    assert backend.step == 0
    assert not backend.checkpoint_dir.exists()


def test_flow_sde_backend_rejects_ratio_drift_before_optimizer_step(
    tmp_path: Path,
) -> None:
    policy = _Policy()
    policy.weight.data.fill_(0.1)
    backend = FlowSDEGRPOBackend(
        policy=policy,
        optimizer=None,
        device="cpu",
        learning_rate=0.1,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        epsilon=1.0e-8,
        max_grad_norm=1.0,
        pre_update_logprob_kl_tolerance=1.0,
        pre_update_ratio_tolerance=0.01,
        microbatch_size=1,
        optimizer_steps_per_update=1,
        training_schedule=FullUpdateScheduleConfig(type="full_update"),
        max_episode_steps=2,
        group_size=2,
        advantage_epsilon=1.0e-6,
        advantage_std_unbiased=True,
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.2,
        clip_ratio_c=3.0,
        filter_rewards=True,
        rewards_lower_bound=0.1,
        rewards_upper_bound=0.9,
        checkpoint_dir=tmp_path / "checkpoints",
        config_fingerprint="config",
        resume_contract_fingerprint="resume",
    )

    with pytest.raises(RuntimeError, match="ratio_tolerance=0.01"):
        asyncio.run(backend.train([_group()]))

    assert policy.weight.item() == pytest.approx(0.1)
    assert backend.step == 0
    assert not backend.checkpoint_dir.exists()


def test_flow_sde_checkpoint_retention_preserves_milestones(
    tmp_path: Path,
) -> None:
    policy = _Policy()
    backend = FlowSDEGRPOBackend(
        policy=policy,
        optimizer=None,
        device="cpu",
        learning_rate=0.1,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        epsilon=1.0e-8,
        max_grad_norm=1.0,
        microbatch_size=1,
        optimizer_steps_per_update=1,
        training_schedule=FullUpdateScheduleConfig(type="full_update"),
        max_episode_steps=2,
        group_size=2,
        advantage_epsilon=1.0e-6,
        advantage_std_unbiased=True,
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.2,
        clip_ratio_c=3.0,
        filter_rewards=True,
        rewards_lower_bound=0.1,
        rewards_upper_bound=0.9,
        checkpoint_dir=tmp_path / "checkpoints",
        config_fingerprint="config",
        resume_contract_fingerprint="resume",
        keep_last_checkpoints=2,
        retain_checkpoint_updates=(2,),
    )

    for step in range(1, 5):
        backend.step = step
        backend._save_checkpoint()

    assert [path.name for path in sorted(backend.checkpoint_dir.iterdir())] == [
        "step-000002",
        "step-000003",
        "step-000004",
    ]


def test_flow_sde_backend_masks_groups_outside_reward_frontier(tmp_path: Path) -> None:
    group = EmbodiedTrajectoryGroup(
        [
            _trajectory(reward=1.0, feature=-1),
            _trajectory(reward=1.0, feature=1),
        ]
    )

    examples, _advantages, kept_groups = prepare_flow_sde_examples(
        [group],
        group_size=2,
        advantage_epsilon=1.0e-6,
        advantage_std_unbiased=True,
        filter_rewards=True,
        rewards_lower_bound=0.1,
        rewards_upper_bound=0.9,
    )

    assert kept_groups == 0
    assert examples
    assert not any(example.loss_mask for example in examples)


def test_flow_sde_rlinf_schedule_consumes_each_chunk_once() -> None:
    groups = [_group(), _group()]
    examples, advantages, _kept_groups = prepare_flow_sde_examples(
        groups,
        group_size=2,
        advantage_epsilon=1.0e-6,
        advantage_std_unbiased=True,
        filter_rewards=True,
        rewards_lower_bound=0.1,
        rewards_upper_bound=0.9,
    )
    schedule = RlinfActorBatchScheduleConfig(
        type="rlinf_actor_global_batch",
        global_batch_size=2,
        actor_seed=42,
        actor_world_size=2,
        rank_local_shuffle=True,
        groups_per_process_per_rollout_epoch=1,
        action_chunk_size=2,
        strict_geometry=True,
        pre_update_alignment_guard="first_subupdate",
    )

    subupdates = _flow_sde_subupdates(
        examples,
        advantages,
        group_count=2,
        group_size=2,
        optimizer_steps_per_update=2,
        schedule=schedule,
        max_episode_steps=2,
    )

    assert len(subupdates) == 2
    assert all(denominator == 2 for _, _, denominator, _ in subupdates)
    assert all(length_normalized for _, _, _, length_normalized in subupdates)
    consumed = [
        (example.group_index, example.trajectory_index, example.action_index)
        for batch, _, _, _ in subupdates
        for example in batch
    ]
    assert sorted(consumed) == [
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, 0),
    ]


def test_flow_sde_backend_executes_rlinf_optimizer_subupdates(tmp_path: Path) -> None:
    policy = _Policy()
    schedule = RlinfActorBatchScheduleConfig(
        type="rlinf_actor_global_batch",
        global_batch_size=2,
        actor_seed=42,
        actor_world_size=2,
        rank_local_shuffle=False,
        groups_per_process_per_rollout_epoch=1,
        action_chunk_size=2,
        strict_geometry=True,
        pre_update_alignment_guard="first_subupdate",
    )
    backend = FlowSDEGRPOBackend(
        policy=policy,
        optimizer=None,
        device="cpu",
        learning_rate=0.1,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        epsilon=1.0e-8,
        max_grad_norm=1.0,
        microbatch_size=1,
        optimizer_steps_per_update=2,
        training_schedule=schedule,
        max_episode_steps=2,
        group_size=2,
        advantage_epsilon=1.0e-6,
        advantage_std_unbiased=True,
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.2,
        clip_ratio_c=3.0,
        filter_rewards=True,
        rewards_lower_bound=0.1,
        rewards_upper_bound=0.9,
        checkpoint_dir=tmp_path / "checkpoints",
        config_fingerprint="config",
        resume_contract_fingerprint="resume",
    )

    result = asyncio.run(backend.train([_group(), _group()]))

    assert result.metrics["embodied_flow_sde_grpo/optimizer_subupdates"] == 2.0
    assert policy.weight.item() > 0.0


def test_flow_sde_backend_writes_bounded_same_tensor_audit(tmp_path: Path) -> None:
    policy = _Policy()
    diagnostics = tmp_path / "diagnostics"
    backend = FlowSDEGRPOBackend(
        policy=policy,
        optimizer=None,
        device="cpu",
        learning_rate=0.1,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        epsilon=1.0e-8,
        max_grad_norm=1.0,
        microbatch_size=1,
        optimizer_steps_per_update=1,
        training_schedule=FullUpdateScheduleConfig(type="full_update"),
        max_episode_steps=2,
        group_size=2,
        advantage_epsilon=1.0e-6,
        advantage_std_unbiased=True,
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.2,
        clip_ratio_c=3.0,
        filter_rewards=True,
        rewards_lower_bound=0.1,
        rewards_upper_bound=0.9,
        checkpoint_dir=tmp_path / "checkpoints",
        config_fingerprint="config",
        resume_contract_fingerprint="resume",
        diagnostics_dir=diagnostics,
    )

    asyncio.run(backend.train([_group()]))

    audit_path = diagnostics / "update-000001.pt"
    audit = torch.load(audit_path, map_location="cpu", weights_only=True)
    assert audit["schema_version"] == 1
    torch.testing.assert_close(audit["rewards"], torch.tensor([[0.0, 1.0]]))
    assert audit["kept_groups"] == 1
    assert len(audit["subupdates"]) == 1
    subupdate = audit["subupdates"][0]
    assert subupdate["loss_denominator"] == 2
    assert subupdate["current_chunk_logprobs"].shape == (2,)
    assert subupdate["old_chunk_logprobs"].shape == (2,)
    assert subupdate["loss_mask"].tolist() == [True, True]


def test_flow_sde_rlinf_update_epochs_reuse_same_partition_order() -> None:
    groups = [_group(), _group()]
    examples, advantages, _kept_groups = prepare_flow_sde_examples(
        groups,
        group_size=2,
        advantage_epsilon=1.0e-6,
        advantage_std_unbiased=True,
        filter_rewards=True,
        rewards_lower_bound=0.1,
        rewards_upper_bound=0.9,
    )
    schedule = RlinfActorBatchScheduleConfig(
        type="rlinf_actor_global_batch",
        global_batch_size=2,
        actor_seed=42,
        actor_world_size=2,
        rank_local_shuffle=True,
        groups_per_process_per_rollout_epoch=1,
        action_chunk_size=2,
        update_epochs=2,
        strict_geometry=True,
        pre_update_alignment_guard="first_subupdate",
    )

    subupdates = _flow_sde_subupdates(
        examples,
        advantages,
        group_count=2,
        group_size=2,
        optimizer_steps_per_update=4,
        schedule=schedule,
        max_episode_steps=2,
    )

    assert len(subupdates) == 4
    first_epoch = [
        [(item.group_index, item.trajectory_index) for item in batch]
        for batch, _, _, _ in subupdates[:2]
    ]
    second_epoch = [
        [(item.group_index, item.trajectory_index) for item in batch]
        for batch, _, _, _ in subupdates[2:]
    ]
    assert second_epoch == first_epoch


def test_flow_sde_trust_region_stops_before_unsafe_optimizer_step() -> None:
    schedule = RlinfActorBatchScheduleConfig(
        type="rlinf_actor_global_batch",
        global_batch_size=2,
        actor_seed=42,
        actor_world_size=2,
        rank_local_shuffle=True,
        groups_per_process_per_rollout_epoch=1,
        action_chunk_size=2,
        update_epochs=2,
        strict_geometry=True,
        pre_update_alignment_guard="first_subupdate",
        max_approximate_kl=0.05,
        min_optimizer_steps_before_kl_stop=1,
    )

    assert not _should_stop_flow_sde_before_optimizer(
        schedule, subupdate_index=0, approximate_kl=0.20
    )
    assert not _should_stop_flow_sde_before_optimizer(
        schedule, subupdate_index=1, approximate_kl=0.05
    )
    assert _should_stop_flow_sde_before_optimizer(
        schedule, subupdate_index=1, approximate_kl=0.051
    )


def test_flow_sde_primitive_kl_guard_is_horizon_invariant() -> None:
    schedule = RlinfActorBatchScheduleConfig(
        type="rlinf_actor_global_batch",
        global_batch_size=2,
        actor_seed=42,
        actor_world_size=2,
        rank_local_shuffle=True,
        groups_per_process_per_rollout_epoch=1,
        action_chunk_size=10,
        update_epochs=1,
        strict_geometry=True,
        pre_update_alignment_guard="first_subupdate",
        max_approximate_kl=0.05,
        approximate_kl_guard_scope="primitive_action",
        min_optimizer_steps_before_kl_stop=1,
    )

    assert not _should_stop_flow_sde_before_optimizer(
        schedule,
        subupdate_index=1,
        approximate_kl=0.20,
        approximate_kl_per_primitive=0.02,
    )
    assert _should_stop_flow_sde_before_optimizer(
        schedule,
        subupdate_index=1,
        approximate_kl=0.60,
        approximate_kl_per_primitive=0.06,
    )
