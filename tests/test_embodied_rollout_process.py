from __future__ import annotations

import asyncio
import os
from pathlib import Path
import stat

import pytest
import yaml

torch = pytest.importorskip("torch")

from art_embodied.backends.flow_sde import (
    FLOW_SDE_REPLAY_SELECTED_KEY,
    TRANSIENT_FLOW_SDE_ROLLOUT_KEY,
)
from art_embodied.config import EmbodiedExperimentConfig
from art_embodied.experiment import EmbodiedExperiment, EmbodiedScenario, RolloutContext
from art_embodied.policies.flow_policy import FlowModelInputs, FlowSDERollout
from art_embodied.policies.flow_sde import FlowSDETransitionRecord
from art_embodied.rollout_inference_worker import (
    _worker_config as _inference_worker_config,
)
from art_embodied.rollout_process import (
    LocalProcessRolloutPool,
)
from art_embodied.rollout_process import (
    _worker_environment as _rollout_worker_environment,
)
from art_embodied.rollout_worker import (
    TRAINABLE_ACTION_SELECTED_KEY,
    _discard_unrequired_training_observation_values,
    _select_training_action_payloads,
    _uniform_grid_positions,
)
from art_embodied.rollout_worker import (
    _worker_config as _rollout_worker_config,
)
from art_embodied.runner import run_embodied_experiment
from art_embodied.trajectories import Action, EmbodiedTrajectory, MediaRef, Observation
from art_embodied.types import LocalTrainResult


class _TrackingModel:
    def __init__(self) -> None:
        self.moves: list[str] = []

    def to(self, device: str) -> "_TrackingModel":
        self.moves.append(device)
        return self


class _SnapshotPolicy:
    def __init__(self) -> None:
        self.version = 0
        self.saves = 0
        self.model = _TrackingModel()

    def save_checkpoint(self, path: str) -> None:
        self.saves += 1
        (Path(path) / "policy_version.txt").write_text(str(self.version))


class _SnapshotProvider:
    def __init__(self) -> None:
        self.saves: list[tuple[Path, int]] = []
        self.offloads = 0
        self.restores: list[str] = []

    def save_snapshot(self, path: Path, *, update: int) -> None:
        self.saves.append((path, update))
        (path / "policy_version.txt").write_text(str(update))

    def offload(self) -> None:
        self.offloads += 1

    def restore(self, device: str) -> None:
        self.restores.append(device)


class _Backend:
    def __init__(self, policy: _SnapshotPolicy | None = None) -> None:
        self.policy = policy

    async def train(self, groups):
        del groups
        if self.policy is not None:
            self.policy.version += 1
        return LocalTrainResult(step=1, metrics={})

    async def close(self):
        return None


class _SnapshotBackend(_Backend):
    def __init__(self) -> None:
        super().__init__()
        self.saves: list[tuple[Path, int]] = []
        self.offloads = 0
        self.restores: list[str] = []

    def save_snapshot(self, path: Path, *, update: int) -> None:
        self.saves.append((path, update))
        (path / "policy_version.txt").write_text(str(update))

    def offload(self) -> None:
        self.offloads += 1

    def restore(self, device: str) -> None:
        self.restores.append(device)


def _config(tmp_path: Path) -> EmbodiedExperimentConfig:
    source = (
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_action_token_smoke.yaml"
    )
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["algorithm"].update(
        {
            "group_size": 4,
            "training_unit": "trajectory",
            "action_advantage_mode": "example",
            "score_source": "trajectory_reward",
        }
    )
    raw["rollout"].update(
        {
            "groups_per_update": 1,
            "epochs_per_update": 1,
            "workers": 2,
            "minimum_completed_attempts_per_group": 4,
        }
    )
    raw["training"].update(
        {
            "updates": 1,
            "optimizer_steps_per_update": 1,
            "schedule": {"type": "full_update"},
        }
    )
    raw["algorithm"]["pad_fixed_horizon_examples"] = False
    raw["evaluation"]["enabled"] = False
    raw["observability"]["wandb"]["enabled"] = False
    raw["observability"]["weave"]["enabled"] = False
    raw["storage"]["output_dir"] = str(tmp_path / "output")
    raw["runtime"].update(
        {
            "rollout_devices": ["cpu"],
            "training_devices": ["cpu"],
            "rollout_execution": {
                "mode": "local_process",
                "actor_factory": ("tests.support.embodied_rollout_actor:create_actor"),
                "actor_kwargs": {},
                "actors_per_device": 2,
                "group_batching": False,
                "lifecycle": "per_update",
                "policy_sync": "checkpoint",
                "inference_mode": "embedded",
                "inference_factory": None,
                "inference_replicas_per_device": 1,
                "inference_max_batch_size": 1,
                "inference_max_wait_ms": 0.0,
                "startup_timeout_seconds": 30,
                "request_timeout_seconds": 30,
            },
            "distributed_training": False,
            "worker_handoff_dir": str(tmp_path / "handoff"),
            "worker_timeout_seconds": 30,
            "max_worker_handoff_mb": 16,
            "keep_worker_handoffs": False,
        }
    )
    return EmbodiedExperimentConfig.model_validate(raw)


def test_rollout_worker_pins_cuda_and_egl_to_the_same_physical_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "0")

    env = _rollout_worker_environment("cuda:7")

    assert env["CUDA_VISIBLE_DEVICES"] == "7"
    assert env["MUJOCO_EGL_DEVICE_ID"] == "7"


@pytest.mark.parametrize(
    "worker_config",
    [_rollout_worker_config, _inference_worker_config],
)
def test_rollout_workers_do_not_revalidate_consumed_resume_checkpoint(
    tmp_path: Path,
    worker_config,
) -> None:
    checkpoint = tmp_path / "step-000010"
    checkpoint.mkdir()
    (checkpoint / "art_embodied_training_state.pt").write_bytes(b"state")
    raw = _config(tmp_path).model_dump(mode="python")
    raw["evaluation"]["evaluate_before_training"] = False
    raw["runtime"]["distributed_training"] = True
    raw["runtime"]["training_devices"] = ["cuda:0", "cuda:1"]
    raw["storage"]["resume_from_checkpoint"] = checkpoint
    config = EmbodiedExperimentConfig.model_validate(raw)

    checkpoint.rename(tmp_path / "retained-away")
    projected = worker_config({"config": config.model_dump(mode="json")})

    assert projected.storage.resume_from_checkpoint is None


@pytest.mark.parametrize(
    ("phase", "retain_payloads", "require_observation", "expect_value"),
    [
        ("train", False, False, False),
        ("eval", False, False, True),
        ("train", True, False, True),
        ("train", False, True, True),
    ],
)
def test_rollout_worker_discards_only_unrequired_training_observation_values(
    tmp_path: Path,
    phase: str,
    retain_payloads: bool,
    require_observation: bool,
    expect_value: bool,
) -> None:
    raw = _config(tmp_path).model_dump(mode="python")
    raw["storage"]["retain_rollout_payloads"] = retain_payloads
    raw["rollout"]["action_payload"]["require_observation"] = require_observation
    config = EmbodiedExperimentConfig.model_validate(raw)
    media = MediaRef(uri="video.mp4", kind="video")
    trajectory = EmbodiedTrajectory(
        task="pick",
        observations=[
            Observation(
                step=0,
                kind="image",
                value={"pixels": bytearray(1024)},
                media=[media],
                metadata={"camera": "agentview"},
            )
        ],
        metadata={"scenario_id": "scenario-0"},
    )

    _discard_unrequired_training_observation_values(
        trajectory,
        phase=phase,
        config=config,
    )

    assert (trajectory.observations[0].value is not None) is expect_value
    assert trajectory.observations[0].media == [media]
    assert trajectory.observations[0].metadata == {"camera": "agentview"}
    assert trajectory.metadata == {"scenario_id": "scenario-0"}


def test_rollout_worker_uniformly_bounds_transient_training_actions(
    tmp_path: Path,
) -> None:
    source = (
        Path(__file__).parents[1]
        / "examples/embodied/smolvla_libero_object_flow_sde_grpo_smoke_h100.yaml"
    )
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["storage"]["output_dir"] = str(tmp_path / "output")
    raw["rollout"]["action_payload"].update(
        trainable_action_selection="uniform_grid",
        max_trainable_actions_per_trajectory=4,
    )
    config = EmbodiedExperimentConfig.model_validate(raw)
    trajectory = EmbodiedTrajectory(
        task="pick",
        actions=[
            Action(
                step=index,
                kind="continuous",
                raw=[0.0],
                metadata={
                    TRANSIENT_FLOW_SDE_ROLLOUT_KEY: {"index": index},
                    "primitive_loss_mask_sum": 1,
                },
            )
            for index in range(10)
        ],
    )

    _select_training_action_payloads(trajectory, phase="train", config=config)

    assert _uniform_grid_positions(10, 4) == [0, 3, 6, 9]
    selected = [
        action.step
        for action in trajectory.actions
        if action.metadata[FLOW_SDE_REPLAY_SELECTED_KEY]
    ]
    retained = [
        action.step
        for action in trajectory.actions
        if TRANSIENT_FLOW_SDE_ROLLOUT_KEY in action.metadata
    ]
    assert selected == [0, 3, 6, 9]
    assert retained == selected
    assert trajectory.metadata["training_action_selection"] == {
        "strategy": "uniform_grid",
        "eligible": 10,
        "selected": 4,
        "selected_action_indices": [0, 3, 6, 9],
    }


def test_rollout_worker_uniformly_bounds_action_token_evidence() -> None:
    source = (
        Path(__file__).parents[1]
        / "examples/embodied/pi0_fast_libero_spatial_grpo_development_h100.yaml"
    )
    config = EmbodiedExperimentConfig.from_yaml(source)
    trajectory = EmbodiedTrajectory(
        task="pick",
        observations=[
            Observation(step=index, kind="image", value={"image": index})
            for index in range(11)
        ],
        actions=[
            Action(
                step=index,
                kind="token",
                raw={"tokens": [index], "prompt": "pick"},
                logprobs=[-0.1],
            )
            for index in range(10)
        ],
    )

    _select_training_action_payloads(trajectory, phase="train", config=config)

    selected = [
        action.step
        for action in trajectory.actions
        if action.metadata[TRAINABLE_ACTION_SELECTED_KEY]
    ]
    assert selected == [0, 3, 6, 9]
    assert [
        observation.step
        for observation in trajectory.observations
        if observation.value is not None
    ] == selected
    assert all(
        action.metadata.get("primitive_loss_mask_sum") == 0
        for action in trajectory.actions
        if action.step not in selected
    )


def test_rollout_worker_excludes_invalid_action_token_evidence() -> None:
    source = (
        Path(__file__).parents[1]
        / "examples/embodied/pi0_fast_libero_spatial_grpo_development_h100.yaml"
    )
    config = EmbodiedExperimentConfig.from_yaml(source)
    trajectory = EmbodiedTrajectory(
        task="pick",
        observations=[
            Observation(step=index, kind="image", value={"image": index})
            for index in range(6)
        ],
        actions=[
            Action(
                step=index,
                kind="token",
                raw={"tokens": [index], "prompt": "pick"},
                logprobs=[-0.1],
                metadata={"action_grammar_valid": index != 2},
            )
            for index in range(5)
        ],
    )

    _select_training_action_payloads(trajectory, phase="train", config=config)

    selected = [
        action.step
        for action in trajectory.actions
        if action.metadata[TRAINABLE_ACTION_SELECTED_KEY]
    ]
    assert selected == [0, 1, 3, 4]
    assert trajectory.actions[2].metadata[TRAINABLE_ACTION_SELECTED_KEY] is False
    assert trajectory.actions[2].metadata["primitive_loss_mask_sum"] == 0
    assert trajectory.observations[2].value is None


def test_rollout_worker_preserves_primitive_count_before_pruning(
    tmp_path: Path,
) -> None:
    source = (
        Path(__file__).parents[1]
        / "examples/embodied/smolvla_libero_object_flow_sde_grpo_smoke_h100.yaml"
    )
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["storage"]["output_dir"] = str(tmp_path / "output")
    raw["rollout"]["action_payload"].update(
        trainable_action_selection="uniform_grid",
        max_trainable_actions_per_trajectory=1,
    )
    config = EmbodiedExperimentConfig.model_validate(raw)

    def rollout() -> FlowSDERollout:
        return FlowSDERollout(
            actions=torch.zeros((1, 3, 2)),
            transition=FlowSDETransitionRecord(
                previous_states=torch.zeros((1, 2, 3, 2)),
                next_states=torch.zeros((1, 2, 3, 2)),
                selected_indices=torch.zeros((1, 2), dtype=torch.long),
                old_logprobs=torch.zeros((1, 2)),
            ),
            inputs=FlowModelInputs(
                images=(torch.zeros((1, 3, 2, 2)),),
                image_masks=(torch.ones((1,), dtype=torch.bool),),
                language_tokens=torch.zeros((1, 1), dtype=torch.long),
                language_masks=torch.ones((1, 1), dtype=torch.bool),
                state=torch.zeros((1, 2)),
            ),
        )

    trajectory = EmbodiedTrajectory(
        task="pick",
        actions=[
            Action(
                step=index,
                kind="continuous",
                raw=[0.0],
                metadata={TRANSIENT_FLOW_SDE_ROLLOUT_KEY: rollout()},
            )
            for index in range(3)
        ],
    )

    _select_training_action_payloads(trajectory, phase="train", config=config)

    assert trajectory.actions[0].metadata["primitive_loss_mask_sum"] == 2
    assert TRANSIENT_FLOW_SDE_ROLLOUT_KEY not in trajectory.actions[0].metadata
    assert TRANSIENT_FLOW_SDE_ROLLOUT_KEY in trajectory.actions[1].metadata
    assert trajectory.actions[2].metadata["primitive_loss_mask_sum"] == 2


def test_local_process_compacts_unrequired_observations_before_ipc(
    tmp_path: Path,
) -> None:
    raw = _config(tmp_path).model_dump(mode="python")
    raw["runtime"]["rollout_execution"]["actor_factory"] = (
        "tests.support.embodied_rollout_actor:create_observation_actor"
    )
    raw["rollout"]["action_payload"]["require_observation"] = False
    config = EmbodiedExperimentConfig.model_validate(raw)
    policy = _SnapshotPolicy()
    pool = LocalProcessRolloutPool(config=config, policy=policy)
    experiment = EmbodiedExperiment(
        config=config,
        scenarios=[EmbodiedScenario(id="task-0", task="pick", payload={})],
        rollout=pool.for_phase("train"),
        backend=_Backend(),
    )

    groups = asyncio.run(experiment.collect(update=0))

    observations = [
        observation
        for trajectory in groups[0].trajectories
        for observation in trajectory.observations
    ]
    assert len(observations) == config.algorithm.group_size
    assert all(observation.value is None for observation in observations)
    assert all(
        observation.metadata == {"camera": "agentview"} for observation in observations
    )
    lifecycle_metrics = pool.for_phase("train").lifecycle_metrics
    assert lifecycle_metrics["trajectory_handoff_files"] == config.algorithm.group_size
    assert lifecycle_metrics["trajectory_handoff_trajectories"] == (
        config.algorithm.group_size
    )
    assert lifecycle_metrics["trajectory_handoff_bytes"] > 0
    assert lifecycle_metrics["trajectory_handoff_bytes_mean"] > 0


def test_training_observation_compaction_reduces_worker_handoff_bytes(
    tmp_path: Path,
) -> None:
    handoff_bytes: dict[bool, float] = {}
    for retain_payloads in (False, True):
        profile = "retained" if retain_payloads else "compact"
        raw = _config(tmp_path / profile).model_dump(mode="python")
        raw["runtime"]["rollout_execution"]["actor_factory"] = (
            "tests.support.embodied_rollout_actor:create_observation_actor"
        )
        raw["storage"]["retain_rollout_payloads"] = retain_payloads
        raw["rollout"]["action_payload"]["require_observation"] = False
        config = EmbodiedExperimentConfig.model_validate(raw)
        pool = LocalProcessRolloutPool(config=config, policy=_SnapshotPolicy())
        experiment = EmbodiedExperiment(
            config=config,
            scenarios=[
                EmbodiedScenario(
                    id="task-0",
                    task="pick",
                    payload={"observation_bytes": 2 * 1024 * 1024},
                )
            ],
            rollout=pool.for_phase("train"),
            backend=_Backend(),
        )

        groups = asyncio.run(experiment.collect(update=0))

        handoff_bytes[retain_payloads] = pool.for_phase("train").lifecycle_metrics[
            "trajectory_handoff_bytes"
        ]
        observations = [
            observation
            for trajectory in groups[0].trajectories
            for observation in trajectory.observations
        ]
        assert all(
            (observation.value is not None) is retain_payloads
            for observation in observations
        )

    assert handoff_bytes[True] - handoff_bytes[False] >= 8 * 1024 * 1024


def test_local_process_rollout_uses_isolated_actors_and_cleans_handoffs(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    policy = _SnapshotPolicy()
    pool = LocalProcessRolloutPool(config=config, policy=policy)
    experiment = EmbodiedExperiment(
        config=config,
        scenarios=[EmbodiedScenario(id="task-0", task="pick", payload={})],
        rollout=pool.for_phase("train"),
        backend=_Backend(),
    )

    groups = asyncio.run(experiment.collect(update=0))

    trajectories = groups[0].trajectories
    assert policy.saves == 1
    assert {item.metadata["fixture_actor"] for item in trajectories} == {0, 1}
    assert {item.metadata["fixture_policy_version"] for item in trajectories} == {0}
    assert {item.metadata["fixture_phase"] for item in trajectories} == {"train"}
    assert all(
        item.metadata["rollout_execution"] == "local_process" for item in trajectories
    )
    assert policy.model.moves == ["cpu", "cpu"]
    lifecycle_metrics = pool.for_phase("train").lifecycle_metrics
    assert lifecycle_metrics["prepare_seconds"] >= 0.0
    assert lifecycle_metrics["worker_startup_seconds"] >= 0.0
    assert lifecycle_metrics["snapshot_seconds"] >= 0.0
    assert lifecycle_metrics["actor_prepare_seconds"] >= 0.0
    assert lifecycle_metrics["finish_seconds"] >= 0.0
    assert lifecycle_metrics["worker_shutdown_seconds"] >= 0.0
    assert pool.workers == []
    assert list((tmp_path / "handoff").iterdir()) == []


def test_local_process_rollout_accepts_external_snapshot_provider(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    provider = _SnapshotProvider()
    pool = LocalProcessRolloutPool(config=config, snapshot_provider=provider)
    experiment = EmbodiedExperiment(
        config=config,
        scenarios=[EmbodiedScenario(id="task-0", task="pick", payload={})],
        rollout=pool.for_phase("train"),
        backend=_Backend(),
    )

    groups = asyncio.run(experiment.collect(update=0))

    assert len(groups[0].trajectories) == 4
    assert [(path.name, update) for path, update in provider.saves] == [
        ("update-000000", 0)
    ]
    assert provider.offloads == 1
    assert provider.restores == ["cpu"]
    assert list((tmp_path / "handoff").iterdir()) == []


def test_local_process_rollout_can_bound_active_actor_concurrency(
    tmp_path: Path,
) -> None:
    raw = _config(tmp_path).model_dump(mode="python")
    raw["runtime"]["rollout_execution"]["actor_kwargs"] = {"max_concurrent_rollouts": 1}
    config = EmbodiedExperimentConfig.model_validate(raw)
    pool = LocalProcessRolloutPool(config=config, policy=_SnapshotPolicy())
    summary = config.execution_summary()

    async def exercise_pool() -> float:
        await pool.prepare_update(update=0)
        scenario = EmbodiedScenario(
            id="slow",
            task="pick",
            payload={"sleep_seconds": 0.2},
        )
        started = asyncio.get_running_loop().time()
        await asyncio.gather(
            *(
                pool.rollout(
                    scenario,
                    RolloutContext(
                        update=0,
                        group_index=0,
                        attempt_index=attempt,
                        environment_seed=attempt,
                        policy_seed=attempt,
                        config_fingerprint=config.fingerprint,
                    ),
                    phase="eval",
                )
                for attempt in range(2)
            )
        )
        elapsed = asyncio.get_running_loop().time() - started
        await pool.close()
        return elapsed

    assert pool.max_concurrent_rollouts == 1
    assert summary["rollout_actor_count"] == 2
    assert summary["rollout_active_actor_count"] == 1
    assert asyncio.run(exercise_pool()) >= 0.35


def test_dead_rollout_actor_is_not_requeued_and_pool_restarts(
    tmp_path: Path,
) -> None:
    raw = _config(tmp_path).model_dump(mode="python")
    raw["runtime"]["rollout_execution"].update(
        {
            "actor_factory": (
                "tests.support.embodied_rollout_actor:create_crashing_actor"
            ),
            "lifecycle": "cpu_offload",
        }
    )
    config = EmbodiedExperimentConfig.model_validate(raw)
    policy = _SnapshotPolicy()
    pool = LocalProcessRolloutPool(config=config, policy=policy)

    async def exercise_pool() -> None:
        await pool.prepare_update(update=0)
        assert pool.available is not None
        with pytest.raises(RuntimeError, match="exited with code 17"):
            await pool.rollout(
                EmbodiedScenario(
                    id="crash",
                    task="crash",
                    payload={"crash_worker": True},
                ),
                RolloutContext(
                    update=0,
                    group_index=0,
                    attempt_index=0,
                    environment_seed=0,
                    policy_seed=0,
                    config_fingerprint=config.fingerprint,
                ),
                phase="eval",
            )
        assert pool.available.qsize() == 1

        await pool.finish_update(update=0)
        assert pool.workers == []
        assert pool.available is None

        policy.version = 1
        await pool.prepare_update(update=1)
        assert len(pool.workers) == 2
        assert pool.available is not None
        assert pool.available.qsize() == 2
        await pool.close()

    asyncio.run(exercise_pool())

    failure_reports = list(
        (tmp_path / "output" / "worker-failures").glob(
            "rollout-actors-*/worker-*/exit.json"
        )
    )
    assert len(failure_reports) == 1
    assert '"return_code": 17' in failure_reports[0].read_text(encoding="utf-8")


def test_one_device_run_serializes_initial_eval_rollout_training_and_eval(
    tmp_path: Path,
) -> None:
    """A workstation profile must not require a dedicated evaluation GPU."""

    raw = _config(tmp_path).model_dump(mode="python")
    raw["evaluation"].update(
        {
            "enabled": True,
            "evaluate_before_training": True,
            "split": "train_matched",
            "data_role": "diagnostic",
            "every_updates": 1,
            "episodes": 1,
            "seeds": [0],
            "fixed_scenarios": ["eval-0"],
            "baseline_outcomes_path": None,
        }
    )
    raw["runtime"]["rollout_devices"] = ["cpu"]
    raw["runtime"]["training_devices"] = ["cpu"]
    raw["runtime"]["rollout_execution"]["lifecycle"] = "per_update"
    config = EmbodiedExperimentConfig.model_validate(raw)

    class _SerialBackend(_Backend):
        def __init__(self) -> None:
            super().__init__()
            self.events: list[str] = []

        def save_snapshot(self, path: Path, *, update: int) -> None:
            self.events.append(f"snapshot:{update}")
            (path / "policy_version.txt").write_text(str(update))

        def offload(self) -> None:
            self.events.append("coordinator:offload")

        def restore(self, device: str) -> None:
            self.events.append(f"coordinator:restore:{device}")

        async def train(self, groups):
            del groups
            self.events.append("training")
            return LocalTrainResult(step=1, metrics={})

    backend = _SerialBackend()
    result = asyncio.run(
        run_embodied_experiment(
            config=config,
            policy=_SnapshotPolicy(),
            train_scenarios=[EmbodiedScenario(id="train-0", task="pick", payload={})],
            evaluation_scenarios=[
                EmbodiedScenario(id="eval-0", task="pick", payload={})
            ],
            rollout=None,
            backend=backend,
        )
    )

    assert len(result.steps) == 1
    assert backend.events == [
        # Initial SFT evaluation (policy version zero).
        "coordinator:offload",
        "snapshot:0",
        "coordinator:restore:cpu",
        # Training rollout from the same policy version.
        "coordinator:offload",
        "snapshot:0",
        "coordinator:restore:cpu",
        "training",
        # Fixed evaluation of policy version one.
        "coordinator:offload",
        "snapshot:1",
        "coordinator:restore:cpu",
    ]
    assert config.execution_summary()["serial_phase_order"] == [
        "initial_evaluation",
        "train_rollout",
        "training",
        "periodic_evaluation",
    ]
    assert list((tmp_path / "handoff").iterdir()) == []


def test_local_process_rollout_rejects_ambiguous_snapshot_ownership(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="policy or snapshot_provider"):
        LocalProcessRolloutPool(
            config=_config(tmp_path),
            policy=_SnapshotPolicy(),
            snapshot_provider=_SnapshotProvider(),
        )


def test_runner_prefers_backend_owned_snapshot_provider(tmp_path: Path) -> None:
    config = _config(tmp_path)
    policy = _SnapshotPolicy()
    backend = _SnapshotBackend()

    result = asyncio.run(
        run_embodied_experiment(
            config=config,
            policy=policy,
            train_scenarios=[EmbodiedScenario(id="task-0", task="pick", payload={})],
            rollout=None,
            backend=backend,
        )
    )

    assert len(result.steps) == 1
    assert policy.saves == 0
    assert [(path.name, update) for path, update in backend.saves] == [
        ("update-000000", 0)
    ]
    assert backend.offloads == 1
    assert backend.restores == ["cpu"]


def test_local_process_group_batching_is_explicit_and_preserves_group_order(
    tmp_path: Path,
) -> None:
    raw = _config(tmp_path).model_dump(mode="python")
    raw["runtime"]["rollout_execution"]["group_batching"] = True
    config = EmbodiedExperimentConfig.model_validate(raw)
    policy = _SnapshotPolicy()
    pool = LocalProcessRolloutPool(config=config, policy=policy)
    experiment = EmbodiedExperiment(
        config=config,
        scenarios=[EmbodiedScenario(id="task-0", task="pick", payload={})],
        rollout=pool.for_phase("train"),
        backend=_Backend(),
    )

    groups = asyncio.run(experiment.collect(update=0))

    trajectories = groups[0].trajectories
    assert [item.metadata["attempt_index"] for item in trajectories] == list(range(4))
    assert {item.metadata["fixture_actor"] for item in trajectories} == {0}
    assert all(item.metadata["rollout_group_batch"] for item in trajectories)
    lifecycle_metrics = pool.for_phase("train").lifecycle_metrics
    assert lifecycle_metrics["trajectory_handoff_files"] == 1
    assert lifecycle_metrics["trajectory_handoff_trajectories"] == 4


def test_runner_resynchronizes_process_actors_before_fixed_evaluation(
    tmp_path: Path,
) -> None:
    raw = _config(tmp_path).model_dump(mode="python")
    raw["evaluation"].update(
        {
            "enabled": True,
            "split": "held_out",
            "runtime": "native_lerobot",
            "every_updates": 1,
            "episodes": 2,
            "seeds": [11],
            "fixed_scenarios": ["eval-0"],
        }
    )
    config = EmbodiedExperimentConfig.model_validate(raw)
    policy = _SnapshotPolicy()
    captured_evaluation = []

    class _Observer:
        async def log_step(
            self,
            step,
            groups,
            train_result,
            evaluation,
            logged_config,
        ):
            del step, groups, train_result, logged_config
            captured_evaluation.append(evaluation)

    result = asyncio.run(
        run_embodied_experiment(
            config=config,
            policy=policy,
            train_scenarios=[EmbodiedScenario(id="train-0", task="pick", payload={})],
            rollout=None,
            evaluation_scenarios=[
                EmbodiedScenario(id="eval-0", task="place", payload={})
            ],
            backend=_Backend(policy),
            observer=_Observer(),
        )
    )

    assert policy.saves == 2
    assert result.steps[0].evaluation is not None
    assert result.steps[0].evaluation.metrics["success_rate"] == 1.0
    assert {
        item.metadata["fixture_policy_version"]
        for item in captured_evaluation[0].trajectories
    } == {1}
    assert {
        item.metadata["fixture_phase"] for item in captured_evaluation[0].trajectories
    } == {"eval"}
    assert list((tmp_path / "handoff").iterdir()) == []


def test_cpu_offload_reuses_same_policy_snapshot_after_evaluation(
    tmp_path: Path,
) -> None:
    raw = _config(tmp_path).model_dump(mode="python")
    raw["runtime"]["rollout_execution"].update(
        {
            "lifecycle": "cpu_offload",
            "inference_mode": "batched_server",
            "inference_factory": (
                "tests.support.embodied_rollout_actor:create_inference_engine"
            ),
            "inference_max_batch_size": 2,
            "inference_max_wait_ms": 2.0,
        }
    )
    raw["training"]["updates"] = 2
    raw["evaluation"].update(
        {
            "enabled": True,
            "split": "held_out",
            "runtime": "native_lerobot",
            "every_updates": 1,
            "episodes": 2,
            "seeds": [11],
            "fixed_scenarios": ["eval-0"],
        }
    )
    config = EmbodiedExperimentConfig.model_validate(raw)
    policy = _SnapshotPolicy()

    class _Observer:
        async def log_step(self, *args, **kwargs):
            del args, kwargs

    result = asyncio.run(
        run_embodied_experiment(
            config=config,
            policy=policy,
            train_scenarios=[EmbodiedScenario(id="train-0", task="pick", payload={})],
            rollout=None,
            evaluation_scenarios=[
                EmbodiedScenario(id="eval-0", task="place", payload={})
            ],
            backend=_Backend(policy),
            observer=_Observer(),
        )
    )

    assert len(result.steps) == 2
    # Train update 1 and its evaluation publish policy versions 0 and 1. The
    # next rollout reuses version 1 instead of trying to recreate its directory.
    assert policy.saves == 3
    assert policy.version == 2
    assert list((tmp_path / "handoff").iterdir()) == []


def test_local_process_rollout_batches_inference_across_environment_actors(
    tmp_path: Path,
) -> None:
    raw = _config(tmp_path).model_dump(mode="python")
    raw["runtime"]["rollout_execution"].update(
        {
            "inference_mode": "batched_server",
            "inference_factory": (
                "tests.support.embodied_rollout_actor:create_inference_engine"
            ),
            "inference_max_batch_size": 2,
            "inference_max_wait_ms": 200.0,
        }
    )
    config = EmbodiedExperimentConfig.model_validate(raw)
    policy = _SnapshotPolicy()
    pool = LocalProcessRolloutPool(config=config, policy=policy)
    experiment = EmbodiedExperiment(
        config=config,
        scenarios=[EmbodiedScenario(id="task-0", task="pick", payload={})],
        rollout=pool.for_phase("train"),
        backend=_Backend(),
    )

    groups = asyncio.run(experiment.collect(update=0))

    inferences = [item.metadata["fixture_inference"] for item in groups[0].trajectories]
    assert all(item["batch_size"] == 2 for item in inferences)
    assert {item["policy_version"] for item in inferences} == {0}
    assert {item["server_index"] for item in inferences} == {0}
    assert list((tmp_path / "handoff").iterdir()) == []


def test_batched_inference_socket_is_private_with_permissive_umask(tmp_path):
    raw = _config(tmp_path).model_dump(mode="python")
    raw["runtime"]["rollout_execution"].update(
        {
            "lifecycle": "cpu_offload",
            "inference_mode": "batched_server",
            "inference_factory": (
                "tests.support.embodied_rollout_actor:create_inference_engine"
            ),
        }
    )
    config = EmbodiedExperimentConfig.model_validate(raw)
    pool = LocalProcessRolloutPool(config=config, policy=_SnapshotPolicy())
    socket_paths = []

    async def check():
        try:
            await pool.prepare_update(update=0)
            assert pool.inference_workers
            for worker in pool.inference_workers:
                path = worker.socket_path
                socket_paths.append(path)
                assert path.is_socket()
                assert stat.S_IMODE(path.stat().st_mode) == 0o600
                assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
                assert path.parent.stat().st_uid == os.getuid()
        finally:
            await pool.close()

    previous = os.umask(0)
    try:
        asyncio.run(check())
    finally:
        os.umask(previous)
    assert socket_paths
    assert all(not path.parent.exists() for path in socket_paths)


def test_local_process_rollout_distributes_actors_across_inference_replicas(
    tmp_path: Path,
) -> None:
    raw = _config(tmp_path).model_dump(mode="python")
    raw["runtime"]["rollout_execution"].update(
        {
            "inference_mode": "batched_server",
            "inference_factory": (
                "tests.support.embodied_rollout_actor:create_inference_engine"
            ),
            "inference_replicas_per_device": 2,
            "inference_max_batch_size": 1,
            "inference_max_wait_ms": 0.0,
        }
    )
    config = EmbodiedExperimentConfig.model_validate(raw)
    pool = LocalProcessRolloutPool(config=config, policy=_SnapshotPolicy())
    experiment = EmbodiedExperiment(
        config=config,
        scenarios=[EmbodiedScenario(id="task-0", task="pick", payload={})],
        rollout=pool.for_phase("train"),
        backend=_Backend(),
    )

    groups = asyncio.run(experiment.collect(update=0))

    server_indices = {
        item.metadata["fixture_inference"]["server_index"]
        for item in groups[0].trajectories
    }
    assert server_indices == {0, 1}
    assert config.execution_summary()["rollout_model_replicas"] == 2
    assert list((tmp_path / "handoff").iterdir()) == []


def test_cpu_offload_rollout_reuses_inference_worker_across_training(
    tmp_path: Path,
) -> None:
    raw = _config(tmp_path).model_dump(mode="python")
    raw["runtime"]["rollout_execution"].update(
        {
            "lifecycle": "cpu_offload",
            "inference_mode": "batched_server",
            "inference_factory": (
                "tests.support.embodied_rollout_actor:create_inference_engine"
            ),
            "inference_max_batch_size": 2,
            "inference_max_wait_ms": 2.0,
        }
    )
    raw["evaluation"].update(
        {
            "enabled": True,
            "split": "held_out",
            "runtime": "native_lerobot",
            "every_updates": 1,
            "episodes": 2,
            "seeds": [11],
            "fixed_scenarios": ["eval-0"],
        }
    )
    config = EmbodiedExperimentConfig.model_validate(raw)
    policy = _SnapshotPolicy()
    captured_evaluation = []

    class _Observer:
        async def log_step(
            self,
            step,
            groups,
            train_result,
            evaluation,
            logged_config,
        ):
            del step, groups, train_result, logged_config
            captured_evaluation.append(evaluation)

    result = asyncio.run(
        run_embodied_experiment(
            config=config,
            policy=policy,
            train_scenarios=[EmbodiedScenario(id="train-0", task="pick", payload={})],
            rollout=None,
            evaluation_scenarios=[
                EmbodiedScenario(id="eval-0", task="place", payload={})
            ],
            backend=_Backend(policy),
            observer=_Observer(),
        )
    )

    assert result.steps[0].evaluation is not None
    rollout_metrics = result.steps[0].training.metrics
    assert rollout_metrics["rollout/prepare_seconds"] >= 0.0
    assert rollout_metrics["rollout/inference_prepare_seconds"] >= 0.0
    assert rollout_metrics["rollout/collection_seconds"] >= 0.0
    assert rollout_metrics["rollout/inference_offload_seconds"] >= 0.0
    assert rollout_metrics["rollout/coordinator_restore_seconds"] >= 0.0
    assert rollout_metrics["rollout/finish_seconds"] >= 0.0
    inferences = [
        trajectory.metadata["fixture_inference"]
        for trajectory in captured_evaluation[0].trajectories
    ]
    assert {item["policy_version"] for item in inferences} == {1}
    assert {item["offload_count"] for item in inferences} == {1}
    assert list((tmp_path / "handoff").iterdir()) == []


def test_builtin_lerobot_group_actor_shares_batched_inference_server(
    tmp_path: Path,
) -> None:
    raw = _config(tmp_path).model_dump(mode="python")
    raw["runtime"]["rollout_execution"].update(
        {
            "actor_factory": (
                "art_embodied.integrations.lerobot_process:create_lerobot_process_actor"
            ),
            "actor_kwargs": {
                "components_factory": (
                    "tests.support.embodied_rollout_actor:"
                    "create_shared_lerobot_components"
                ),
                "components_kwargs": {"label": "shared-lerobot"},
                "max_concurrent_rollouts": 1,
            },
            "group_batching": True,
            "inference_mode": "batched_server",
            "inference_factory": (
                "tests.support.embodied_rollout_actor:create_inference_engine"
            ),
            "inference_max_batch_size": 2,
            "inference_max_wait_ms": 2.0,
        }
    )
    config = EmbodiedExperimentConfig.model_validate(raw)
    policy = _SnapshotPolicy()
    pool = LocalProcessRolloutPool(config=config, policy=policy)
    experiment = EmbodiedExperiment(
        config=config,
        scenarios=[EmbodiedScenario(id="task-0", task="pick", payload={})],
        rollout=pool.for_phase("train"),
        backend=_Backend(),
    )

    groups = asyncio.run(experiment.collect(update=0))

    trajectories = groups[0].trajectories
    assert [item.metadata["attempt_index"] for item in trajectories] == list(range(4))
    assert {item.metadata["fixture_label"] for item in trajectories} == {
        "shared-lerobot"
    }
    inferences = {
        item.metadata["fixture_inference"]["policy_version"] for item in trajectories
    }
    assert inferences == {0}
    assert all(
        item.metadata["fixture_inference"]["attempt_indices"] == list(range(4))
        for item in trajectories
    )
    assert list((tmp_path / "handoff").iterdir()) == []


def test_builtin_lerobot_process_actor_owns_episode_recording(
    tmp_path: Path,
) -> None:
    raw = _config(tmp_path).model_dump(mode="python")
    raw["environment"]["kwargs"]["success_key"] = "is_success"
    raw["reward"].update({"terminal_only": True, "scale": 1.0})
    raw["observability"].update(
        {
            "videos_per_update": 0,
            "videos_per_evaluation": 0,
            "require_train_video": False,
            "require_evaluation_video": False,
        }
    )
    raw["runtime"]["rollout_execution"].update(
        {
            "actor_factory": (
                "art_embodied.integrations.lerobot_process:create_lerobot_process_actor"
            ),
            "actor_kwargs": {
                "components_factory": (
                    "tests.support.embodied_rollout_actor:create_lerobot_components"
                ),
                "components_kwargs": {"label": "lerobot-first"},
            },
        }
    )
    config = EmbodiedExperimentConfig.model_validate(raw)
    policy = _SnapshotPolicy()
    pool = LocalProcessRolloutPool(config=config, policy=policy)
    experiment = EmbodiedExperiment(
        config=config,
        scenarios=[EmbodiedScenario(id="task-0", task="pick", payload={})],
        rollout=pool.for_phase("train"),
        backend=_Backend(),
    )

    groups = asyncio.run(experiment.collect(update=0))

    trajectories = groups[0].trajectories
    assert len(trajectories) == 4
    assert all(item.metadata["framework"] == "lerobot" for item in trajectories)
    assert all(item.metadata["phase"] == "train" for item in trajectories)
    assert all(item.metrics["success"] is True for item in trajectories)
    assert all(item.metrics["episode_steps"] == 1 for item in trajectories)
    assert all(item.reward == 1.0 for item in trajectories)
    assert all(item.actions[0].raw["tokens"] == [0] for item in trajectories)
    assert all(
        item.actions[0].metadata["fixture_label"] == "lerobot-first"
        for item in trajectories
    )
    assert all(
        item.metadata["video_capture_selected"] is False for item in trajectories
    )
    assert list((tmp_path / "handoff").iterdir()) == []


def test_embedded_cpu_offload_reuses_lerobot_actor_across_updates(
    tmp_path: Path,
) -> None:
    raw = _config(tmp_path).model_dump(mode="python")
    raw["environment"]["kwargs"]["success_key"] = "is_success"
    raw["training"]["updates"] = 2
    raw["runtime"]["rollout_execution"].update(
        {
            "actor_factory": (
                "art_embodied.integrations.lerobot_process:create_lerobot_process_actor"
            ),
            "actor_kwargs": {
                "components_factory": (
                    "tests.support.embodied_rollout_actor:create_lerobot_components"
                ),
                "components_kwargs": {"label": "embedded-offload"},
            },
            "lifecycle": "cpu_offload",
        }
    )
    config = EmbodiedExperimentConfig.model_validate(raw)
    policy = _SnapshotPolicy()

    class _CapturingObserver:
        def __init__(self) -> None:
            self.actions = []

        async def log_step(self, _step, groups, *_args) -> None:
            self.actions.append(groups[0].trajectories[0].actions[0])

    observer = _CapturingObserver()
    result = asyncio.run(
        run_embodied_experiment(
            config=config,
            policy=policy,
            train_scenarios=[EmbodiedScenario(id="train-0", task="pick", payload={})],
            rollout=None,
            backend=_Backend(policy),
            observer=observer,
        )
    )

    assert len(result.steps) == 2
    assert policy.saves == 2
    assert [
        action.metadata["fixture_offload_count"] for action in observer.actions
    ] == [
        0,
        1,
    ]
    assert [
        action.metadata["fixture_restore_count"] for action in observer.actions
    ] == [
        0,
        1,
    ]
    assert list((tmp_path / "handoff").iterdir()) == []


def test_builtin_lerobot_process_actor_refreshes_persistent_policy(
    tmp_path: Path,
) -> None:
    raw = _config(tmp_path).model_dump(mode="python")
    raw["environment"]["kwargs"]["success_key"] = "is_success"
    raw["observability"].update(
        {
            "videos_per_update": 0,
            "videos_per_evaluation": 0,
            "require_train_video": False,
            "require_evaluation_video": False,
        }
    )
    raw["runtime"].update(
        {
            "rollout_devices": ["cpu"],
            "training_devices": ["cuda:0"],
        }
    )
    raw["runtime"]["rollout_execution"].update(
        {
            "actor_factory": (
                "art_embodied.integrations.lerobot_process:create_lerobot_process_actor"
            ),
            "actor_kwargs": {
                "components_factory": (
                    "tests.support.embodied_rollout_actor:create_lerobot_components"
                ),
                "components_kwargs": {"label": "persistent"},
            },
            "lifecycle": "persistent",
        }
    )
    config = EmbodiedExperimentConfig.model_validate(raw)
    policy = _SnapshotPolicy()
    pool = LocalProcessRolloutPool(config=config, policy=policy)
    experiment = EmbodiedExperiment(
        config=config,
        scenarios=[EmbodiedScenario(id="task-0", task="pick", payload={})],
        rollout=pool.for_phase("train"),
        backend=_Backend(),
    )

    first = asyncio.run(experiment.collect(update=0))
    worker_pids = [worker.process.pid for worker in pool.workers]
    policy.version = 1
    second = asyncio.run(experiment.collect(update=1))

    try:
        assert [worker.process.pid for worker in pool.workers] == worker_pids
        assert all(
            item.actions[0].raw["tokens"] == [0] for item in first[0].trajectories
        )
        assert all(
            item.actions[0].raw["tokens"] == [1] for item in second[0].trajectories
        )
        assert {item.metadata["rollout_policy_update"] for item in first[0]} == {0}
        assert {item.metadata["rollout_policy_update"] for item in second[0]} == {1}
    finally:
        asyncio.run(pool.close())
    assert list((tmp_path / "handoff").iterdir()) == []
