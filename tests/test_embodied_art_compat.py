from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import art
import pytest

from art_embodied import (
    EmbodiedBackend,
    EmbodiedExperimentConfig,
    EmbodiedTrainableModel,
    EmbodiedTrajectory,
    gather_trajectory_groups,
    trajectory_group,
)
from art_embodied.types import LocalTrainResult


def _config(tmp_path: Path) -> EmbodiedExperimentConfig:
    source = (
        Path(__file__).parents[1] / "examples/embodied/"
        "openvla_oft_libero_object_grpo_lora_lr1e4_shared_rollout.yaml"
    )
    raw = EmbodiedExperimentConfig.from_yaml(source).model_dump(mode="python")
    raw["storage"]["output_dir"] = tmp_path / "output"
    raw["runtime"]["worker_handoff_dir"] = tmp_path / "handoff"
    raw["observability"]["wandb"]["enabled"] = False
    raw["observability"]["weave"]["enabled"] = False
    raw["observability"]["require_train_video"] = False
    raw["observability"]["require_evaluation_video"] = False
    return EmbodiedExperimentConfig.model_validate(raw)


class _NativeBackend:
    def __init__(self, policy) -> None:
        self.policy = policy
        self.update_step = 0
        self.received = None
        self.closed = False

    async def train(self, groups):
        self.received = groups
        self.update_step += 1
        return LocalTrainResult(
            step=self.update_step,
            metrics={"loss": 0.25},
            checkpoint_path=None,
        )

    async def close(self):
        self.closed = True


class _Observer:
    def __init__(self) -> None:
        self.steps = []

    async def log_step(self, *args):
        self.steps.append(args)


def test_embodied_types_use_real_upstream_classes() -> None:
    from art.trajectories import PydanticException
    from art.types import LocalTrainResult as ARTLocalTrainResult

    from art_embodied.trajectories import PydanticException as EmbodiedException

    assert LocalTrainResult is ARTLocalTrainResult
    assert EmbodiedException is PydanticException
    result = LocalTrainResult(step=7, checkpoint_path="checkpoint-7")
    assert result.step == 7
    assert result.checkpoint_path == "checkpoint-7"
    if hasattr(result, "checkpoint_ready"):
        assert result.checkpoint_ready is None


def test_embodied_model_is_an_art_trainable_model(tmp_path: Path) -> None:
    policy = object()
    config = _config(tmp_path)
    model = EmbodiedTrainableModel(policy=policy, config=config)

    assert isinstance(model, art.TrainableModel)
    assert model.policy is policy
    assert model.name == config.experiment.run
    assert model.project == config.experiment.project
    assert model.base_model == config.policy.path


def test_embodied_model_maps_legacy_run_name_to_art_model_name(tmp_path: Path) -> None:
    model = EmbodiedTrainableModel(
        policy=object(),
        config=_config(tmp_path),
        name="lower-priority-name",
        run_name="canonical-run-name",
    )

    assert model.name == "canonical-run-name"


@pytest.mark.parametrize(
    ("name", "run_name", "expected"),
    [
        (None, None, None),
        ("named-run", None, "named-run"),
        ("serving-name", "durable-run", "durable-run"),
    ],
)
def test_embodied_identity_matches_art_checkpoint_identity(
    tmp_path: Path,
    name: str | None,
    run_name: str | None,
    expected: str | None,
) -> None:
    config = _config(tmp_path)
    model = EmbodiedTrainableModel(
        policy=object(),
        config=config,
        name=name,
        run_name=run_name,
    )
    identity = expected or config.experiment.run

    assert model.name == identity
    assert model.model_dump()["name"] == identity
    assert model.report_metrics == []
    if "run_name" in art.TrainableModel.model_fields:
        assert model.run_name == identity
        assert model.model_dump()["run_name"] == identity
        assert model._storage_name() == identity


def test_art_train_model_groups_log_lifecycle(tmp_path: Path) -> None:
    policy = object()
    config = _config(tmp_path)
    observer = _Observer()
    model = EmbodiedTrainableModel(
        policy=policy,
        config=config,
        observer=observer,
    )
    native = _NativeBackend(policy)
    backend = EmbodiedBackend(native, config=config)
    group = asyncio.run(
        trajectory_group(
            [_rollout("task", 1.0), _rollout("task", 0.0)],
            metadata={"scenario_id": "scenario-0"},
        )
    )

    async def run():
        await model.register(backend)
        result = await backend.train(
            model,
            [group],
            learning_rate=config.training.optimizer.learning_rate,
        )
        await model.log(
            [group],
            split="train",
            metrics=result.metrics,
            step=result.step,
        )
        assert await model.get_step() == 1
        await model.close()
        return result

    result = asyncio.run(run())

    assert result.step == 1
    assert native.received == [group]
    assert native.closed is True
    assert len(observer.steps) == 1
    assert observer.steps[0][0] == 1
    assert observer.steps[0][1] == [group]
    assert observer.steps[0][2] is result


def test_registered_backend_accepts_bound_experiment_call_shape(tmp_path: Path) -> None:
    policy = object()
    config = _config(tmp_path)
    model = EmbodiedTrainableModel(policy=policy, config=config)
    native = _NativeBackend(policy)
    backend = EmbodiedBackend(native, config=config)
    group = asyncio.run(trajectory_group([_rollout("task", 1.0)]))

    async def run():
        await model.register(backend)
        result = await backend.train([group])
        return result, await model.get_step(), backend.update_step

    result, model_step, backend_step = asyncio.run(run())

    assert result.step == 1
    assert model_step == 1
    assert backend_step == 1


def test_learning_rate_must_match_yaml(tmp_path: Path) -> None:
    policy = object()
    config = _config(tmp_path)
    model = EmbodiedTrainableModel(policy=policy, config=config)
    backend = EmbodiedBackend(_NativeBackend(policy), config=config)

    async def run():
        await model.register(backend)
        with pytest.raises(ValueError, match="must match"):
            await backend.train(model, [], learning_rate=123.0)

    asyncio.run(run())


def test_art_lifecycle_resume_uses_completed_backend_step(tmp_path: Path) -> None:
    config = _config(tmp_path)
    policy = object()
    native = _NativeBackend(policy)
    native.update_step = 7
    observer = _Observer()
    model = EmbodiedTrainableModel(policy=policy, config=config, observer=observer)
    backend = EmbodiedBackend(native, config=config)

    async def run():
        await model.register(backend)
        assert await model.get_step() == 7
        assert not observer.steps
        result = await backend.train(model, [])
        await model.log([], split="train")
        assert result.step == await model.get_step() == 8
        await model.close()

    asyncio.run(run())
    assert [step[0] for step in observer.steps] == [8]


def test_embodied_gather_matches_art_grouping_shape() -> None:
    async def run():
        return await gather_trajectory_groups(
            [
                trajectory_group(
                    [_rollout("one", 1.0), _rollout("one", 0.0)],
                    return_exceptions=True,
                ),
                trajectory_group(
                    [_rollout("two", 0.5), _rollout("two", 0.25)],
                    return_exceptions=True,
                ),
            ]
        )

    groups = asyncio.run(run())

    assert [len(group) for group in groups] == [2, 2]
    assert [group.rewards() for group in groups] == [[1.0, 0.0], [0.5, 0.25]]


async def _rollout(task: str, reward: float) -> EmbodiedTrajectory:
    return EmbodiedTrajectory(
        task=task,
        reward=reward,
        metrics={"success": reward == 1.0},
    )
