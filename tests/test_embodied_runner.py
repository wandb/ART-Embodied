from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

from art_embodied.config import (
    EmbodiedExperimentConfig,
    PreTrainingSuccessGateConfig,
)
from art_embodied.experiment import EmbodiedScenario, EvaluationResult
from art_embodied.integrations.lerobot import LeRobotPolicyAdapter
from art_embodied.runner import (
    _apply_pre_training_success_gate,
    run_embodied_evaluation,
    run_embodied_experiment,
    run_lerobot_experiment,
    validate_runtime_device_availability,
)
from art_embodied.trajectories import EmbodiedTrajectory
from art_embodied.types import LocalTrainResult


class _Backend:
    def __init__(self) -> None:
        self.step = 0
        self.closed = False

    async def train(self, groups):
        self.step += 1
        return LocalTrainResult(step=self.step, metrics={"groups": len(groups)})

    async def close(self):
        self.closed = True


class _Observer:
    def __init__(self) -> None:
        self.steps = []
        self.progress = []
        self.closed = False

    async def log_step(self, step, groups, train_result, evaluation, config):
        self.steps.append((step, len(groups), train_result.step, evaluation, config))

    async def log_evaluation(self, step, evaluation, config):
        self.steps.append((step, evaluation, config))

    async def log_progress(self, event, config):
        self.progress.append((event, config))

    def close(self):
        self.closed = True


def _config(tmp_path: Path) -> EmbodiedExperimentConfig:
    source = (
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml"
    )
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["storage"]["output_dir"] = str(tmp_path / "run")
    raw["algorithm"]["group_size"] = 2
    raw["rollout"]["groups_per_update"] = 1
    raw["rollout"]["epochs_per_update"] = 1
    raw["rollout"]["workers"] = 1
    raw["rollout"]["minimum_completed_attempts_per_group"] = 2
    raw["training"]["updates"] = 1
    raw["training"]["optimizer_steps_per_update"] = 1
    raw["training"]["schedule"] = {"type": "full_update"}
    raw["algorithm"]["pad_fixed_horizon_examples"] = False
    raw["evaluation"]["enabled"] = False
    raw["evaluation"]["evaluate_before_training"] = False
    raw["evaluation"]["baseline_outcomes_path"] = None
    raw["observability"]["wandb"]["enabled"] = False
    raw["observability"]["weave"]["enabled"] = False
    raw["observability"]["require_train_video"] = False
    raw["observability"]["require_evaluation_video"] = False
    raw["runtime"]["rollout_execution"].update(
        {
            "mode": "in_process",
            "group_batching": False,
            "actor_factory": None,
            "actor_kwargs": {},
            "policy_sync": "shared",
        }
    )
    return EmbodiedExperimentConfig.model_validate(raw)


def test_high_level_runner_uses_injected_resources_without_owning_them(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    backend = _Backend()
    observer = _Observer()

    async def rollout(scenario, context):
        return EmbodiedTrajectory(
            task=scenario.task,
            reward=float(context.attempt_index),
        )

    result = asyncio.run(
        run_embodied_experiment(
            config=config,
            policy=object(),
            train_scenarios=[EmbodiedScenario(id="train-0", task="pick", payload={})],
            rollout=rollout,
            backend=backend,
            observer=observer,
        )
    )

    assert result.config_fingerprint == config.fingerprint
    assert len(result.steps) == 1
    assert observer.steps[0][0:3] == (1, 1, 1)
    assert backend.closed is False
    assert observer.closed is False


def test_runtime_device_validation_fails_before_work_for_invisible_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2)),
    )

    with pytest.raises(RuntimeError, match=r"cuda:2, cuda:3.*visible.*=2"):
        validate_runtime_device_availability(config)


def test_runtime_device_validation_can_ignore_training_devices_for_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _config(tmp_path)
    config = source.model_copy(
        update={
            "runtime": source.runtime.model_copy(
                update={"rollout_devices": ["cuda:0"]}
            ),
        }
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1)),
    )

    validate_runtime_device_availability(
        config,
        include_training_devices=False,
    )


def _config_with_pre_training_gate(
    tmp_path: Path,
) -> EmbodiedExperimentConfig:
    config = _config(tmp_path)
    return config.model_copy(
        update={
            "evaluation": config.evaluation.model_copy(
                update={
                    "enabled": True,
                    "evaluate_before_training": True,
                    "pre_training_success_gate": PreTrainingSuccessGateConfig(
                        minimum_success_rate=0.05,
                        maximum_success_rate=0.90,
                    ),
                }
            )
        }
    )


def test_pre_training_success_gate_accepts_informative_step_zero(
    tmp_path: Path,
) -> None:
    config = _config_with_pre_training_gate(tmp_path)
    observer = _Observer()

    asyncio.run(
        _apply_pre_training_success_gate(
            config=config,
            evaluation=EvaluationResult(
                step=0,
                metrics={"success_rate": 0.40},
                artifacts={},
            ),
            observer=observer,
        )
    )

    event = observer.progress[-1][0]
    assert event.phase == "training_gate"
    assert event.status == "completed"
    assert event.metrics["success_rate"] == 0.40


@pytest.mark.parametrize(
    "success_rate",
    [0.0, 0.95],
)
def test_pre_training_success_gate_rejects_no_signal_or_no_headroom(
    tmp_path: Path,
    success_rate: float,
) -> None:
    config = _config_with_pre_training_gate(tmp_path)
    observer = _Observer()

    with pytest.raises(RuntimeError, match="rejected optimizer start"):
        asyncio.run(
            _apply_pre_training_success_gate(
                config=config,
                evaluation=EvaluationResult(
                    step=0,
                    metrics={"success_rate": success_rate},
                    artifacts={},
                ),
                observer=observer,
            )
        )

    assert observer.progress[-1][0].status == "failed"


def test_evaluation_only_runner_skips_backend_and_writes_fixed_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _config(tmp_path).model_dump(mode="python")
    raw["evaluation"].update(
        {
            "enabled": True,
            "episodes": 2,
            "fixed_scenarios": ["eval-0"],
            "seeds": [10],
        }
    )
    config = EmbodiedExperimentConfig.model_validate(raw)
    observer = _Observer()

    def fail_backend(*args, **kwargs):
        del args, kwargs
        raise AssertionError("evaluation-only must not construct a backend")

    monkeypatch.setattr("art_embodied.runner.make_embodied_backend", fail_backend)

    async def rollout(scenario, context):
        return EmbodiedTrajectory(
            task=scenario.task,
            reward=1.0,
            metrics={"success": context.group_index == 0},
        )

    result = asyncio.run(
        run_embodied_evaluation(
            config=config,
            policy=object(),
            evaluation_scenarios=[
                EmbodiedScenario(id="eval-0", task="place", payload={})
            ],
            rollout=rollout,
            step=7,
            observer=observer,
        )
    )

    assert result.config_fingerprint == config.fingerprint
    assert result.evaluation.step == 7
    assert result.evaluation.metrics["success_rate"] == 0.5
    assert Path(result.evaluation.artifacts["episode_outcomes_json"]).is_file()
    assert observer.steps[0][0] == 7


def test_evaluation_only_runner_propagates_loaded_checkpoint_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _config(tmp_path).model_dump(mode="python")
    raw["evaluation"].update(
        {
            "enabled": True,
            "episodes": 1,
            "fixed_scenarios": ["eval-0"],
            "seeds": [10],
        }
    )
    config = EmbodiedExperimentConfig.model_validate(raw)
    observer = _Observer()
    checkpoint = tmp_path / "checkpoint" / "policy"
    checkpoint.mkdir(parents=True)
    captured: list[str | None] = []

    class _Evaluator:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __call__(self, step, train_result, _config):
            captured.append(train_result.checkpoint_path)
            return EvaluationResult(
                step=step,
                metrics={"success_rate": 1.0},
                artifacts={},
            )

    monkeypatch.setattr("art_embodied.runner.FixedScenarioEvaluator", _Evaluator)

    asyncio.run(
        run_embodied_evaluation(
            config=config,
            policy=object(),
            evaluation_scenarios=[
                EmbodiedScenario(id="eval-0", task="place", payload={})
            ],
            rollout=lambda _scenario, _context: None,
            step=10,
            checkpoint_path=checkpoint,
            observer=observer,
        )
    )

    assert captured == [str(checkpoint.resolve())]


def test_evaluation_only_runner_refuses_a_new_standalone_wandb_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _config(tmp_path).model_dump(mode="python")
    raw["evaluation"].update(
        {
            "enabled": True,
            "episodes": 1,
            "fixed_scenarios": ["eval-0"],
            "seeds": [10],
        }
    )
    raw["observability"]["wandb"]["enabled"] = True
    config = EmbodiedExperimentConfig.model_validate(raw)
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    monkeypatch.delenv("WANDB_RESUME", raising=False)

    with pytest.raises(ValueError, match="cannot safely attach W&B metrics"):
        asyncio.run(
            run_embodied_evaluation(
                config=config,
                policy=object(),
                evaluation_scenarios=[
                    EmbodiedScenario(id="eval-0", task="place", payload={})
                ],
                rollout=lambda _scenario, _context: None,
                step=10,
            )
        )


def test_high_level_runner_closes_owned_backend_when_setup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    raw = config.model_dump(mode="python")
    raw["evaluation"]["enabled"] = True
    raw["evaluation"]["fixed_scenarios"] = ["held-out-0"]
    config = EmbodiedExperimentConfig.model_validate(raw)
    backend = _Backend()
    monkeypatch.setattr(
        "art_embodied.runner.make_embodied_backend",
        lambda config, policy: backend,
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 4)),
    )

    async def rollout(scenario, context):
        return EmbodiedTrajectory(task=scenario.task)

    with pytest.raises(ValueError, match="evaluation_scenarios"):
        asyncio.run(
            run_embodied_experiment(
                config=config,
                policy=object(),
                train_scenarios=[
                    EmbodiedScenario(id="train-0", task="pick", payload={})
                ],
                rollout=rollout,
            )
        )
    assert backend.closed is True


def test_lerobot_runner_builds_distinct_train_and_evaluation_phases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _config(tmp_path).model_dump(mode="python")
    raw["evaluation"].update(
        {
            "enabled": True,
            "every_updates": 1,
            "episodes": 1,
            "fixed_scenarios": ["eval-0"],
            "seeds": [10],
        }
    )
    config = EmbodiedExperimentConfig.model_validate(raw)
    phases = []

    def make_rollout(_config, **kwargs):
        phase = kwargs["phase"]
        phases.append(phase)

        async def rollout(scenario, context):
            del context
            return EmbodiedTrajectory(
                task=scenario.task,
                reward=1.0,
                metrics={"success": True},
                metadata={"phase": phase},
            )

        return rollout

    monkeypatch.setattr(
        "art_embodied.runner.LeRobotEpisodeRollout.from_config",
        make_rollout,
    )
    backend = _Backend()
    observer = _Observer()
    policy = object()
    policy_adapter = LeRobotPolicyAdapter(
        policy=policy,
        preprocessor=object(),
        postprocessor=object(),
        device="cpu",
    )
    result = asyncio.run(
        run_lerobot_experiment(
            config=config,
            policy=policy,
            train_scenarios=[EmbodiedScenario(id="train-0", task="pick", payload={})],
            evaluation_scenarios=[
                EmbodiedScenario(id="eval-0", task="place", payload={})
            ],
            environment_factory=lambda scenario, context: None,
            policy_adapter=policy_adapter,
            backend=backend,
            observer=observer,
        )
    )

    assert phases == ["train", "eval"]
    assert result.steps[0].evaluation is not None
    assert result.steps[0].evaluation.metrics["success_rate"] == 1.0


def test_lerobot_runner_rejects_unsynchronized_policy_factory(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    with pytest.raises(ValueError, match="prepare_update"):
        asyncio.run(
            run_lerobot_experiment(
                config=config,
                policy=object(),
                train_scenarios=[
                    EmbodiedScenario(id="train-0", task="pick", payload={})
                ],
                environment_factory=lambda scenario, context: None,
                policy_adapter_factory=lambda scenario, context: None,
                backend=_Backend(),
                observer=_Observer(),
            )
        )


def test_lerobot_runner_rejects_shared_adapter_for_parallel_rollout(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    raw = config.model_dump(mode="python")
    raw["rollout"]["workers"] = 2
    config = EmbodiedExperimentConfig.model_validate(raw)
    policy = object()
    adapter = LeRobotPolicyAdapter(
        policy=policy,
        preprocessor=object(),
        postprocessor=object(),
        device="cpu",
    )

    with pytest.raises(ValueError, match="rollout.workers=1"):
        asyncio.run(
            run_lerobot_experiment(
                config=config,
                policy=policy,
                train_scenarios=[
                    EmbodiedScenario(id="train-0", task="pick", payload={})
                ],
                environment_factory=lambda scenario, context: None,
                policy_adapter=adapter,
                backend=_Backend(),
                observer=_Observer(),
            )
        )
