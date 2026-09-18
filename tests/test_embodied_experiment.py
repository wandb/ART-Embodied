from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from art_embodied.config import EmbodiedExperimentConfig
from art_embodied.experiment import (
    EmbodiedExperiment,
    EmbodiedScenario,
    EvaluationResult,
    _discard_transient_rollout_payloads,
    _process_memory_metrics,
)
from art_embodied.trajectories import (
    Action,
    EmbodiedTrajectory,
    EmbodiedTrajectoryGroup,
    Observation,
)
from art_embodied.types import LocalTrainResult


class _Backend:
    def __init__(self) -> None:
        self.groups = []
        self.closed = False
        self.step = 0

    async def train(self, groups, **kwargs):
        self.groups.append(groups)
        self.step += 1
        return LocalTrainResult(
            step=self.step,
            metrics={"reward": 0.5},
            checkpoint_path=f"checkpoint-{self.step}",
        )

    async def close(self):
        self.closed = True


def test_process_memory_metrics_parse_linux_proc_status(tmp_path: Path) -> None:
    status = tmp_path / "status"
    status.write_text(
        "Name:\tpython\nVmHWM:\t2097152 kB\nVmRSS:\t1048576 kB\n",
        encoding="utf-8",
    )

    assert _process_memory_metrics(status) == {
        "peak_resident_memory_mb": 2048.0,
        "resident_memory_mb": 1024.0,
    }


def test_process_memory_metrics_are_optional(tmp_path: Path) -> None:
    assert _process_memory_metrics(tmp_path / "missing") == {}


def test_discard_transient_rollout_payloads_preserves_public_metadata() -> None:
    trajectory = EmbodiedTrajectory(
        task="move object",
        metadata={
            "scenario_id": "scenario-1",
            "_art_embodied_transient_root": object(),
        },
        actions=[
            Action(
                step=0,
                kind="continuous",
                raw={},
                decoded=[0.1],
                metadata={
                    "policy_step": 0,
                    "_art_embodied_transient_flow_sde_rollout": object(),
                },
            )
        ],
    )

    removed = _discard_transient_rollout_payloads(
        [EmbodiedTrajectoryGroup([trajectory])]
    )

    assert removed == 2
    assert trajectory.metadata == {"scenario_id": "scenario-1"}
    assert trajectory.actions[0].metadata == {"policy_step": 0}


def _config(tmp_path: Path) -> EmbodiedExperimentConfig:
    source = (
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml"
    )
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["storage"]["output_dir"] = str(tmp_path / "run")
    raw["algorithm"]["group_size"] = 2
    raw["rollout"]["groups_per_update"] = 2
    raw["rollout"]["epochs_per_update"] = 1
    raw["rollout"]["workers"] = 2
    raw["rollout"]["minimum_completed_attempts_per_group"] = 2
    rollout_execution = raw["runtime"]["rollout_execution"]
    rollout_execution["mode"] = "in_process"
    rollout_execution["group_batching"] = False
    rollout_execution["policy_sync"] = "shared"
    rollout_execution["actor_factory"] = None
    rollout_execution["actor_kwargs"] = {}
    raw["training"]["updates"] = 2
    raw["training"]["optimizer_steps_per_update"] = 1
    raw["training"]["schedule"] = {"type": "full_update"}
    raw["algorithm"]["pad_fixed_horizon_examples"] = False
    raw["evaluation"]["every_updates"] = 2
    raw["evaluation"]["enabled"] = True
    raw["evaluation"]["split"] = "held_out"
    raw["evaluation"]["data_role"] = "development"
    return EmbodiedExperimentConfig.model_validate(raw)


@pytest.mark.parametrize("stop_before_first", [False, True])
def test_operator_stop_returns_normally_after_logging(tmp_path, stop_before_first):
    config = _config(tmp_path)
    config = config.model_copy(
        update={"evaluation": config.evaluation.model_copy(update={"every_updates": 1})}
    )
    marker = Path(config.storage.output_dir) / "STOP_REQUESTED"
    marker.parent.mkdir(parents=True, exist_ok=True)
    backend = _Backend()
    events = []
    if stop_before_first:
        marker.write_text("User requested investigation, not continuation.\n")

    async def rollout(scenario, context):
        return EmbodiedTrajectory(
            task=scenario.task, reward=float(context.attempt_index)
        )

    async def evaluate(step, train_result, experiment_config):
        events.append("evaluation")
        marker.write_text("Finish this evaluation and stop.\n")
        return EvaluationResult(step=step, metrics={"success_rate": 0.5}, artifacts={})

    async def log_step(*args):
        events.append("logging")

    experiment = EmbodiedExperiment(
        config=config,
        scenarios=[EmbodiedScenario(id="task", task="pick", payload={})],
        rollout=rollout,
        backend=backend,
        evaluate=evaluate,
        log_step=log_step,
    )
    results = asyncio.run(experiment.run())
    assert backend.closed
    assert events == ([] if stop_before_first else ["evaluation", "logging"])
    assert [r.update for r in results] == ([] if stop_before_first else [1])
    assert marker.exists(), "Restart must not silently ignore an operator stop"


def test_experiment_groups_comparable_resets_and_distinct_policy_seeds(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    backend = _Backend()
    seen = []
    evaluations = []
    logs = []
    progress = []

    async def rollout(scenario, context):
        seen.append((scenario.id, context))
        return EmbodiedTrajectory(
            task=scenario.task, reward=float(context.attempt_index)
        )

    async def evaluate(step, train_result, experiment_config):
        evaluations.append((step, train_result.step, experiment_config.fingerprint))
        return EvaluationResult(
            step=step,
            metrics={"success": 0.75},
            artifacts={"video": "rollout.mp4"},
            trajectories=(EmbodiedTrajectory(task="held-out", reward=1.0),),
        )

    async def log_step(step, groups, train_result, evaluation, experiment_config):
        logs.append((step, len(groups), train_result.step, evaluation))

    async def log_progress(event, experiment_config):
        assert experiment_config.fingerprint == config.fingerprint
        progress.append(event)

    experiment = EmbodiedExperiment(
        config=config,
        scenarios=[
            EmbodiedScenario(id="task-0", task="pick cube", payload={}),
            EmbodiedScenario(id="task-1", task="place cube", payload={}),
        ],
        rollout=rollout,
        backend=backend,
        evaluate=evaluate,
        log_step=log_step,
        log_progress=log_progress,
    )

    results = asyncio.run(experiment.run())

    assert [result.update for result in results] == [1, 2]
    assert backend.closed is True
    assert len(backend.groups) == 2
    assert [len(groups) for groups in backend.groups] == [2, 2]
    assert all(
        len(group.trajectories) == 2 for groups in backend.groups for group in groups
    )
    assert len(evaluations) == 1
    assert evaluations[0][:2] == (2, 2)
    assert logs[0][3] is None
    assert logs[1][3].metrics == {"success": 0.75}
    assert len(logs[1][3].trajectories) == 1
    assert results[1].evaluation is not None
    assert results[1].evaluation.trajectories == ()
    assert [(event.update, event.phase, event.status) for event in progress] == [
        (1, "update", "started"),
        (1, "rollout", "started"),
        (1, "rollout", "progress"),
        (1, "rollout", "progress"),
        (1, "rollout", "completed"),
        (1, "training", "started"),
        (1, "training", "completed"),
        (1, "logging", "started"),
        (1, "logging", "completed"),
        (1, "update", "completed"),
        (2, "update", "started"),
        (2, "rollout", "started"),
        (2, "rollout", "progress"),
        (2, "rollout", "progress"),
        (2, "rollout", "completed"),
        (2, "training", "started"),
        (2, "training", "completed"),
        (2, "evaluation", "started"),
        (2, "evaluation", "completed"),
        (2, "logging", "started"),
        (2, "logging", "completed"),
        (2, "update", "completed"),
    ]
    rollout_progress = [
        event
        for event in progress
        if event.update == 1 and event.phase == "rollout" and event.status == "progress"
    ]
    assert rollout_progress[-1].metrics["trajectories_completed"] == 4
    assert rollout_progress[-1].metrics["elapsed_seconds"] >= 0.0
    assert rollout_progress[-1].metrics["groups_per_second"] > 0.0
    assert rollout_progress[-1].metrics["trajectories_per_second"] > 0.0
    rollout_completed = [
        event
        for event in progress
        if event.update == 1
        and event.phase == "rollout"
        and event.status == "completed"
    ][0]
    assert rollout_completed.metrics["elapsed_seconds"] >= 0.0
    assert rollout_completed.metrics["groups_per_second"] > 0.0
    assert rollout_completed.metrics["trajectories_per_second"] > 0.0
    assert results[0].training.metrics["rollout/elapsed_seconds"] >= 0.0
    assert results[0].training.metrics["rollout/groups_per_second"] > 0.0
    assert results[0].training.metrics["rollout/trajectories_per_second"] > 0.0
    assert results[0].training.metrics["rollout/collection_seconds"] >= 0.0
    assert results[0].training.metrics["training/elapsed_seconds"] >= 0.0

    contexts_by_group = {}
    for _, context in seen:
        contexts_by_group.setdefault((context.update, context.group_index), []).append(
            context
        )
    for contexts in contexts_by_group.values():
        assert len({context.environment_seed for context in contexts}) == 1
        assert len({context.policy_seed for context in contexts}) == 2


def test_in_process_collection_discards_unrequired_observation_values(
    tmp_path: Path,
) -> None:
    raw = _config(tmp_path).model_dump(mode="python")
    raw["storage"]["retain_rollout_payloads"] = False
    raw["rollout"]["action_payload"]["require_observation"] = False
    config = EmbodiedExperimentConfig.model_validate(raw)

    async def rollout(scenario, context):
        del context
        return EmbodiedTrajectory(
            task=scenario.task,
            observations=[
                Observation(
                    step=0,
                    kind="image",
                    value={"pixels": bytearray(1024)},
                    metadata={"camera": "agentview"},
                )
            ],
        )

    experiment = EmbodiedExperiment(
        config=config,
        scenarios=[EmbodiedScenario(id="task-0", task="pick", payload={})],
        rollout=rollout,
        backend=_Backend(),
        evaluate=lambda *_args: None,
    )

    groups = asyncio.run(experiment.collect(update=0))

    for trajectory in groups[0].trajectories:
        assert trajectory.observations[0].value is None
        assert trajectory.observations[0].metadata == {"camera": "agentview"}


def test_experiment_prepares_rollout_before_each_update(tmp_path: Path) -> None:
    config = _config(tmp_path)
    prepared = []

    class PreparedRollout:
        async def prepare_update(self, *, update):
            prepared.append(update)

        async def __call__(self, scenario, context):
            return EmbodiedTrajectory(
                task=scenario.task,
                reward=float(context.attempt_index),
            )

    experiment = EmbodiedExperiment(
        config=config,
        scenarios=[EmbodiedScenario(id="task", task="pick", payload={})],
        rollout=PreparedRollout(),
        backend=_Backend(),
        evaluate=lambda step, train_result, config: asyncio.sleep(
            0,
            result=EvaluationResult(step=step, metrics={}, artifacts={}),
        ),
    )

    asyncio.run(experiment.run())

    assert prepared == [0, 1]


def test_experiment_resumes_from_backend_update_step(tmp_path: Path) -> None:
    config = _config(tmp_path)
    backend = _Backend()
    backend.update_step = 1
    backend.step = 1
    seen_updates = []

    async def rollout(scenario, context):
        seen_updates.append(context.update)
        return EmbodiedTrajectory(task=scenario.task, reward=1.0)

    experiment = EmbodiedExperiment(
        config=config,
        scenarios=[EmbodiedScenario(id="task", task="pick", payload={})],
        rollout=rollout,
        backend=backend,
        evaluate=lambda step, train_result, config: asyncio.sleep(
            0,
            result=EvaluationResult(step=step, metrics={}, artifacts={}),
        ),
    )

    results = asyncio.run(experiment.run())

    assert [result.update for result in results] == [2]
    assert seen_updates == [1, 1, 1, 1]


def test_experiment_reports_progress_during_long_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "art_embodied.experiment._TRAINING_PROGRESS_INTERVAL_SECONDS",
        0.01,
    )
    config = _config(tmp_path)
    progress = []

    class SlowBackend(_Backend):
        async def train(self, groups, **kwargs):
            await asyncio.sleep(0.025)
            return await super().train(groups, **kwargs)

    async def rollout(scenario, context):
        return EmbodiedTrajectory(
            task=scenario.task,
            reward=float(context.attempt_index),
        )

    async def log_progress(event, experiment_config):
        assert experiment_config.fingerprint == config.fingerprint
        progress.append(event)

    experiment = EmbodiedExperiment(
        config=config,
        scenarios=[EmbodiedScenario(id="task", task="pick", payload={})],
        rollout=rollout,
        backend=SlowBackend(),
        evaluate=lambda step, train_result, experiment_config: asyncio.sleep(
            0,
            result=EvaluationResult(step=step, metrics={}, artifacts={}),
        ),
        log_progress=log_progress,
    )

    asyncio.run(experiment.train_step(update=0))

    training_progress = [
        event
        for event in progress
        if event.phase == "training" and event.status == "progress"
    ]
    assert len(training_progress) >= 2
    assert all(event.update == 1 for event in training_progress)
    assert all(
        event.metrics is not None and event.metrics["training/elapsed_seconds"] >= 0.01
        for event in training_progress
    )
    assert all(
        event.message == "Training backend is active" for event in training_progress
    )


def test_enabled_evaluation_requires_explicit_evaluator(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicit evaluation function"):
        EmbodiedExperiment(
            config=_config(tmp_path),
            scenarios=[EmbodiedScenario(id="task", task="task", payload={})],
            rollout=lambda scenario, context: None,
            backend=_Backend(),
        )


def test_all_failed_group_stops_before_training(tmp_path: Path) -> None:
    backend = _Backend()

    async def rollout(scenario, context):
        raise RuntimeError("simulator failed")

    experiment = EmbodiedExperiment(
        config=_config(tmp_path),
        scenarios=[EmbodiedScenario(id="task", task="task", payload={})],
        rollout=rollout,
        backend=backend,
        evaluate=lambda step, train_result, config: None,
    )

    with pytest.raises(RuntimeError, match="All rollouts failed"):
        asyncio.run(experiment.train_step(update=0))
    assert backend.groups == []


def test_step_zero_observability_gate_stops_before_training(tmp_path: Path) -> None:
    backend = _Backend()

    async def rollout(scenario, context):
        del context
        return EmbodiedTrajectory(task=scenario.task, reward=1.0)

    async def reject_unobservable_rollout(policy_version, groups, config):
        del policy_version, groups, config
        raise RuntimeError("W&B Step-0 history commit failed")

    experiment = EmbodiedExperiment(
        config=_config(tmp_path),
        scenarios=[EmbodiedScenario(id="task", task="task", payload={})],
        rollout=rollout,
        backend=backend,
        evaluate=lambda step, train_result, config: None,
        log_rollout=reject_unobservable_rollout,
    )

    with pytest.raises(RuntimeError, match="W&B Step-0 history commit failed"):
        asyncio.run(experiment.train_step(update=0))
    assert backend.groups == []


def test_partial_group_fails_closed_before_grpo(tmp_path: Path) -> None:
    backend = _Backend()

    async def rollout(scenario, context):
        if context.attempt_index == 1:
            raise RuntimeError("one simulator failed")
        return EmbodiedTrajectory(task=scenario.task, reward=1.0)

    experiment = EmbodiedExperiment(
        config=_config(tmp_path),
        scenarios=[EmbodiedScenario(id="task", task="task", payload={})],
        rollout=rollout,
        backend=backend,
        evaluate=lambda step, train_result, config: None,
    )

    with pytest.raises(RuntimeError, match="failure_policy='fail_update'"):
        asyncio.run(experiment.train_step(update=0))
    assert backend.groups == []


def test_rollout_cancellation_is_not_converted_to_failed_trajectory(
    tmp_path: Path,
) -> None:
    backend = _Backend()

    async def cancelled_rollout(scenario, context):
        del scenario, context
        raise asyncio.CancelledError

    experiment = EmbodiedExperiment(
        config=_config(tmp_path),
        scenarios=[EmbodiedScenario(id="task", task="task", payload={})],
        rollout=cancelled_rollout,
        backend=backend,
        evaluate=lambda step, train_result, config: None,
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(experiment.train_step(update=0))
    assert backend.groups == []


@pytest.mark.parametrize("first_eval, expected", [(False, [5]), (True, [1, 5])])
def test_first_update_evaluation_is_independent_of_periodic_cadence(
    tmp_path, first_eval, expected
):
    config = _config(tmp_path)
    config = config.model_copy(
        update={
            "training": config.training.model_copy(update={"updates": 6}),
            "evaluation": config.evaluation.model_copy(
                update={
                    "every_updates": 5,
                    "evaluate_after_first_update": first_eval,
                    "evaluate_before_training": False,
                }
            ),
        }
    )
    steps = []

    async def rollout(scenario, context):
        return EmbodiedTrajectory(
            task=scenario.task, reward=float(context.attempt_index)
        )

    async def evaluate(step, result, config):
        steps.append(step)
        return EvaluationResult(step=step, metrics={"success_rate": 0.5}, artifacts={})

    experiment = EmbodiedExperiment(
        config=config,
        scenarios=[EmbodiedScenario(id="task", task="pick cube", payload={})],
        rollout=rollout,
        backend=_Backend(),
        evaluate=evaluate,
    )
    asyncio.run(experiment.run())
    assert steps == expected
