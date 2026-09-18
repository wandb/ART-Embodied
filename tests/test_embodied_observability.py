from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from art_embodied.config import EmbodiedExperimentConfig
from art_embodied.experiment import EvaluationResult, ExperimentProgress
import art_embodied.observability as observability_module
from art_embodied.observability import (
    WandbWeaveObserver,
    _configure_weave_server_cache,
)
from art_embodied.trajectories import (
    Action,
    EmbodiedTrajectory,
    EmbodiedTrajectoryGroup,
    MediaRef,
)
from art_embodied.types import LocalTrainResult


def test_trajectory_metrics_expose_shared_prefix_execution() -> None:
    trajectory = EmbodiedTrajectory(
        task="test",
        metrics={
            "shared_prefix_action_chunks": 15,
            "shared_prefix_environment_steps": 120,
            "trainable_suffix_action_chunks": 75,
            "shared_prefix_branch_verified": True,
            "shared_prefix_completed_episode": False,
        },
    )

    payload = observability_module._trajectory_metric_payload(
        "train_details", [trajectory]
    )

    assert payload["train_details/shared_prefix_action_chunks_mean"] == 15
    assert payload["train_details/shared_prefix_environment_steps_mean"] == 120
    assert payload["train_details/trainable_suffix_action_chunks_mean"] == 75
    assert payload["train_details/shared_prefix_branch_verified_rate"] == 1
    assert payload["train_details/shared_prefix_completed_episode_rate"] == 0


def test_trajectory_metrics_expose_action_token_grammar_health() -> None:
    trajectories = [
        EmbodiedTrajectory(
            task="test",
            actions=[
                Action(
                    step=0,
                    kind="token",
                    raw={"tokens": [1]},
                    metadata={
                        "action_grammar_valid": valid,
                        "rollout_logprobs_computed": valid,
                        "generated_token_count": generated,
                        "post_termination_tokens_discarded": discarded,
                    },
                )
            ],
        )
        for valid, generated, discarded in ((True, 10, 2), (False, 20, 4))
    ]

    payload = observability_module._trajectory_metric_payload(
        "train_details", trajectories
    )

    assert payload["train_details/action_grammar_valid_rate"] == 0.5
    assert payload["train_details/episodes_with_invalid_action_rate"] == 0.5
    assert payload["train_details/rollout_logprobs_computed_rate"] == 0.5
    assert payload["train_details/generated_token_count_mean"] == 15
    assert payload["train_details/post_termination_tokens_discarded_mean"] == 3


def test_primary_evaluation_payload_exposes_perturbation_breakdown() -> None:
    evaluation = EvaluationResult(
        step=0,
        metrics={
            "success_rate": 0.5,
            "category/sensor-noise/success_rate": 0.7,
            "difficulty/3/success_rate": 0.6,
        },
        artifacts={},
    )

    payload = observability_module._wandb_primary_evaluation_payload(evaluation)

    assert payload["validation/success_rate"] == 0.5
    assert payload["validation/category/sensor-noise/success_rate"] == 0.7
    assert payload["validation/difficulty/3/success_rate"] == 0.6


class _Wandb:
    class Settings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Video:
        def __init__(self, path, caption=None, format=None):
            self.path = path
            self.caption = caption
            self.format = format

    class Table:
        def __init__(self, columns, data):
            self.columns = columns
            self.data = data

    class Artifact:
        def __init__(self, name, type, metadata):
            self.name = name
            self.type = type
            self.metadata = metadata
            self.paths = []

        def add_dir(self, path):
            self.paths.append(("dir", path))

        def add_file(self, path, name=None):
            if name is None:
                self.paths.append(("file", path))
            else:
                self.paths.append(("file", path, name))


class _Run:
    def __init__(self):
        self.logs = []
        self.log_commits = []
        self.metric_definitions = []
        self.summary = _Summary()
        self.models = []
        self.artifacts = []
        self.used_artifacts = []
        self.finished = False
        self.exit_code = None
        self.entity = "test-entity"
        self.project = "test-project"
        self.id = "run-id"
        self.url = "https://wandb.example/run-id"

    def log(self, payload, step=None, commit=None):
        self.logs.append((payload, step))
        self.log_commits.append(commit)

    def define_metric(self, name, **kwargs):
        metric = SimpleNamespace(name=name)
        self.metric_definitions.append((name, kwargs, metric))
        return metric

    def log_model(self, **kwargs):
        self.models.append(kwargs)

    def log_artifact(self, artifact, aliases):
        self.artifacts.append((artifact, aliases))

    def use_artifact(self, artifact, use_as=None):
        self.used_artifacts.append((artifact, use_as))
        return artifact

    def finish(self, exit_code=0):
        self.finished = True
        self.exit_code = exit_code


class _CommittedRun(_Run):
    """Minimal W&B native-Step row merger for commit contract tests."""

    def __init__(self):
        super().__init__()
        self.pending = {}
        self.history = []

    def log(self, payload, step=None, commit=None):
        super().log(payload, step=step, commit=commit)
        self.pending.update(payload)
        if commit is not False:
            self.history.append(dict(self.pending))
            self.pending.clear()


class _CommittedRunWithNativeStep(_CommittedRun):
    @property
    def step(self):
        return len(self.history)


class _FailOnceRun(_Run):
    def __init__(self):
        super().__init__()
        self.log_attempts = 0

    def log(self, payload, step=None, commit=None):
        self.log_attempts += 1
        if self.log_attempts == 1:
            raise ConnectionError("injected W&B outage")
        super().log(payload, step=step, commit=commit)


class _Summary(dict):
    def __init__(self):
        super().__init__()
        self.updates = []

    def update(self, payload):
        self.updates.append(dict(payload))
        super().update(payload)


class _Content:
    @staticmethod
    def from_path(path, mimetype=None, metadata=None):
        return {"path": str(path), "mimetype": mimetype, "metadata": metadata}


class _WeaveClient:
    def __init__(self):
        self.calls = []
        self.finished = []

    def create_call(self, op, inputs, parent=None, **kwargs):
        call = SimpleNamespace(op=op, inputs=inputs, parent=parent)
        self.calls.append(call)
        return call

    def finish_call(self, call, output=None, exception=None):
        self.finished.append((call, output, exception))


class _FailingWeaveClient(_WeaveClient):
    def create_call(self, op, inputs, parent=None, **kwargs):
        raise ConnectionError("injected Weave outage")


def _config(tmp_path: Path) -> EmbodiedExperimentConfig:
    source = (
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml"
    )
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["storage"]["output_dir"] = str(tmp_path / "output")
    raw["training"]["checkpoint_every_updates"] = 1
    raw["observability"]["require_train_video"] = False
    raw["observability"]["require_evaluation_video"] = False
    return EmbodiedExperimentConfig.model_validate(raw)


def _integration_config(
    tmp_path: Path,
    *,
    wandb_enabled: bool,
    weave_enabled: bool,
) -> EmbodiedExperimentConfig:
    raw = _config(tmp_path).model_dump(mode="python")
    raw["observability"]["wandb"]["enabled"] = wandb_enabled
    raw["observability"]["weave"]["enabled"] = weave_enabled
    return EmbodiedExperimentConfig.model_validate(raw)


@pytest.mark.parametrize(
    ("module_name", "integration", "wandb_enabled", "weave_enabled"),
    [
        ("wandb", "W&B logging", True, False),
        ("weave", "Weave tracing", False, True),
    ],
)
def test_observer_start_explains_missing_observability_extra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    integration: str,
    wandb_enabled: bool,
    weave_enabled: bool,
) -> None:
    config = _integration_config(
        tmp_path,
        wandb_enabled=wandb_enabled,
        weave_enabled=weave_enabled,
    )

    def missing_dependency(name: str):
        raise ModuleNotFoundError(
            f"No module named '{name}'",
            name=name,
        )

    monkeypatch.setattr(observability_module, "import_module", missing_dependency)

    with pytest.raises(RuntimeError, match=integration) as exc_info:
        WandbWeaveObserver.start(config)

    assert module_name in str(exc_info.value)
    assert "art-embodied[observability]" in str(exc_info.value)


def test_observer_start_does_not_mask_transitive_import_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _integration_config(
        tmp_path,
        wandb_enabled=True,
        weave_enabled=False,
    )

    def broken_dependency(name: str):
        raise ModuleNotFoundError(
            "No module named 'wandb_dependency'",
            name="wandb_dependency",
        )

    monkeypatch.setattr(observability_module, "import_module", broken_dependency)

    with pytest.raises(ModuleNotFoundError, match="wandb_dependency"):
        WandbWeaveObserver.start(config)


def test_observer_applies_weave_cache_contract_before_wandb_auto_init(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _integration_config(
        tmp_path,
        wandb_enabled=True,
        weave_enabled=True,
    )
    observed_cache_settings = []
    run = _Run()
    weave_client = _WeaveClient()

    class FakeWandb:
        Settings = _Wandb.Settings

        @staticmethod
        def init(**_kwargs):
            observed_cache_settings.append(
                (
                    os.environ.get("WEAVE_USE_SERVER_CACHE"),
                    os.environ.get("WEAVE_SERVER_CACHE_DIR"),
                )
            )
            return run

    class FakeWeave:
        @staticmethod
        def init(_project):
            return weave_client

    modules = {"wandb": FakeWandb, "weave": FakeWeave}
    monkeypatch.setattr(
        observability_module,
        "import_module",
        lambda name: modules[name],
    )
    monkeypatch.setenv("WEAVE_USE_SERVER_CACHE", "true")
    monkeypatch.setenv("WEAVE_SERVER_CACHE_DIR", "/tmp/shared-weave-cache")

    observer = WandbWeaveObserver.start(config)

    assert observer.wandb_run is run
    assert observer.weave_client is weave_client
    assert observed_cache_settings == [("false", None)]


def test_observer_rejects_non_primary_shared_evaluation_writer(
    tmp_path: Path,
) -> None:
    raw = _integration_config(
        tmp_path,
        wandb_enabled=True,
        weave_enabled=False,
    ).model_dump(mode="python")
    raw["observability"]["wandb"].update(
        {
            "connection": "shared_worker",
            "run_id": "source-run-id",
            "resume": None,
            "writer_label": "sealed-eval-update-10",
        }
    )
    with pytest.raises(
        ValueError,
        match="shared mode cannot satisfy ART-Embodied's native-step evaluation contract",
    ):
        EmbodiedExperimentConfig.model_validate(raw)


def test_observer_starts_coordinator_as_single_primary_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _integration_config(
        tmp_path,
        wandb_enabled=True,
        weave_enabled=False,
    )
    init_calls = []

    class FakeWandb:
        Settings = _Wandb.Settings

        @staticmethod
        def init(**kwargs):
            init_calls.append(kwargs)
            return _Run()

    monkeypatch.setattr(
        observability_module,
        "import_module",
        lambda name: FakeWandb if name == "wandb" else None,
    )

    observer = WandbWeaveObserver.start(config)

    settings = init_calls[0]["settings"].kwargs
    assert settings["mode"] == "online"
    assert "x_primary" not in settings
    assert "x_update_finish_state" not in settings
    assert "x_label" not in settings
    assert settings["console_multipart"] is True
    assert settings["console_chunk_max_bytes"] == 1_048_576
    assert settings["console_chunk_max_seconds"] == 60
    assert settings["x_disable_stats"] is False
    assert "resume" not in init_calls[0]
    assert init_calls[0]["save_code"] is False
    definitions = observer.wandb_run.metric_definitions
    assert [(name, kwargs) for name, kwargs, _ in definitions] == [
        ("experiment/update", {"hidden": True}),
        ("train/*", {"step_metric": definitions[0][2]}),
        ("validation/*", {"step_metric": definitions[0][2]}),
        ("test/*", {"step_metric": definitions[0][2]}),
        ("optimization/*", {"step_metric": definitions[0][2]}),
        ("performance/*", {"step_metric": definitions[0][2]}),
        ("signal/*", {"step_metric": definitions[0][2]}),
        ("train_details/*", {"step_metric": definitions[0][2]}),
        ("eval_details/*", {"step_metric": definitions[0][2]}),
        ("train_tasks/*", {"step_metric": definitions[0][2]}),
        ("eval_tasks/*", {"step_metric": definitions[0][2]}),
        ("telemetry/*", {"step_metric": definitions[0][2]}),
    ]


def test_observer_can_disable_wandb_system_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _integration_config(
        tmp_path,
        wandb_enabled=True,
        weave_enabled=False,
    ).model_dump(mode="python")
    raw["observability"]["wandb"]["log_system_metrics"] = False
    config = EmbodiedExperimentConfig.model_validate(raw)
    init_calls = []

    class FakeWandb:
        Settings = _Wandb.Settings

        @staticmethod
        def init(**kwargs):
            init_calls.append(kwargs)
            return _Run()

    monkeypatch.setattr(
        observability_module,
        "import_module",
        lambda name: FakeWandb if name == "wandb" else None,
    )

    WandbWeaveObserver.start(config)

    assert init_calls[0]["settings"].kwargs["x_disable_stats"] is True


def test_observer_declares_completed_warm_start_as_input_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "sft-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    marker = {
        "schema_version": 1,
        "status": "complete",
        "optimizer_steps": 60_000,
    }
    (checkpoint / "art_embodied_sft_complete.json").write_text(
        json.dumps(marker), encoding="utf-8"
    )
    raw = _integration_config(
        tmp_path,
        wandb_enabled=True,
        weave_enabled=False,
    ).model_dump(mode="python")
    raw["policy"]["path"] = str(checkpoint)
    raw["observability"]["wandb"]["log_input_model_artifact"] = True
    config = EmbodiedExperimentConfig.model_validate(raw)
    run = _Run()

    class FakeWandb(_Wandb):
        @staticmethod
        def init(**_kwargs):
            return run

    monkeypatch.setattr(
        observability_module,
        "import_module",
        lambda name: FakeWandb if name == "wandb" else None,
    )

    observer = WandbWeaveObserver.start(config)

    assert observer.wandb_run is run
    assert len(run.used_artifacts) == 1
    artifact, use_as = run.used_artifacts[0]
    assert use_as == "warm_start_policy"
    assert artifact.type == "model"
    assert artifact.paths == [("dir", str(checkpoint.resolve()))]
    assert artifact.metadata["role"] == "warm_start_policy"
    assert artifact.metadata["sft_completion"] == marker


def test_observer_uses_pre_registered_input_model_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "sft-checkpoint"
    checkpoint.mkdir()
    marker = {
        "schema_version": 1,
        "status": "complete",
        "optimizer_steps": 60_000,
    }
    (checkpoint / "art_embodied_sft_complete.json").write_text(
        json.dumps(marker), encoding="utf-8"
    )
    raw = _integration_config(
        tmp_path,
        wandb_enabled=True,
        weave_enabled=False,
    ).model_dump(mode="python")
    raw["policy"]["path"] = str(checkpoint)
    raw["observability"]["wandb"]["log_input_model_artifact"] = True
    artifact_ref = "entity/project/pre-registered-model:v7"
    raw["observability"]["wandb"]["input_model_artifact_ref"] = artifact_ref
    config = EmbodiedExperimentConfig.model_validate(raw)
    run = _Run()

    class FakeWandb(_Wandb):
        Artifact = None

        @staticmethod
        def init(**_kwargs):
            return run

    monkeypatch.setattr(
        observability_module,
        "import_module",
        lambda name: FakeWandb if name == "wandb" else None,
    )

    observer = WandbWeaveObserver.start(config)

    assert observer.wandb_run is run
    assert run.used_artifacts == [(artifact_ref, "warm_start_policy")]


def test_input_model_artifact_reference_requires_immutable_version(
    tmp_path: Path,
) -> None:
    raw = _integration_config(
        tmp_path,
        wandb_enabled=True,
        weave_enabled=False,
    ).model_dump(mode="python")
    raw["observability"]["wandb"]["log_input_model_artifact"] = True
    raw["observability"]["wandb"]["input_model_artifact_ref"] = (
        "entity/project/model:latest"
    )

    with pytest.raises(ValueError, match="immutable :vN"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_input_model_artifact_reference_does_not_change_behavior_fingerprint(
    tmp_path: Path,
) -> None:
    raw = _integration_config(
        tmp_path,
        wandb_enabled=True,
        weave_enabled=False,
    ).model_dump(mode="python")
    raw["observability"]["wandb"]["log_input_model_artifact"] = True
    without_reference = EmbodiedExperimentConfig.model_validate(raw)
    raw["observability"]["wandb"]["input_model_artifact_ref"] = (
        "entity/project/model:v3"
    )
    with_reference = EmbodiedExperimentConfig.model_validate(raw)

    assert with_reference.fingerprint == without_reference.fingerprint


def test_observer_rejects_unmarked_input_model_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "incomplete-sft-checkpoint"
    checkpoint.mkdir()
    raw = _integration_config(
        tmp_path,
        wandb_enabled=True,
        weave_enabled=False,
    ).model_dump(mode="python")
    raw["policy"]["path"] = str(checkpoint)
    raw["observability"]["wandb"]["log_input_model_artifact"] = True
    config = EmbodiedExperimentConfig.model_validate(raw)

    class FakeWandb(_Wandb):
        @staticmethod
        def init(**_kwargs):
            return _Run()

    monkeypatch.setattr(
        observability_module,
        "import_module",
        lambda name: FakeWandb if name == "wandb" else None,
    )

    with pytest.raises(FileNotFoundError, match="SFT completion marker"):
        WandbWeaveObserver.start(config)


def test_new_run_logs_measured_sft_evaluation_as_validation_step_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_path = tmp_path / "baseline-outcomes.json"
    baseline_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "step": 0,
                "split": "train_matched",
                "data_role": "diagnostic",
                "episodes": [
                    {
                        "scenario_id": f"scenario-{index}",
                        "task": "pick cube",
                        "success": index < 3,
                        "completed": True,
                    }
                    for index in range(10)
                ],
            }
        ),
        encoding="utf-8",
    )
    raw = _integration_config(
        tmp_path,
        wandb_enabled=True,
        weave_enabled=False,
    ).model_dump(mode="python")
    raw["evaluation"]["baseline_outcomes_path"] = baseline_path
    config = EmbodiedExperimentConfig.model_validate(raw)
    run = _Run()

    class FakeWandb:
        Settings = _Wandb.Settings

        @staticmethod
        def init(**kwargs):
            return run

    monkeypatch.setattr(
        observability_module,
        "import_module",
        lambda name: FakeWandb if name == "wandb" else None,
    )

    observer = WandbWeaveObserver.start(config)

    assert observer.wandb_run is run
    # A cached comparison report is evidence, not the current policy. The
    # runner must measure the initialized SFT policy before writing Step 0.
    assert run.logs == []
    initial_evaluation = EvaluationResult(
        step=0,
        metrics={
            "success_rate": 0.3,
            "episodes": 10.0,
            "task_macro_success_rate": 0.3,
        },
        artifacts={},
    )
    asyncio.run(observer.log_initial_evaluation(initial_evaluation, config))

    assert len(run.logs) == 1
    baseline_payload, logged_step = run.logs[0]
    assert logged_step is None
    assert baseline_payload == {
        "experiment/update": 0,
        "validation/success_rate": 0.3,
        "validation/checkpoint_role": "sft_baseline",
        "validation/episodes": 10.0,
        "validation/task_macro_success_rate": 0.3,
    }
    assert run.summary["validation/latest_success_rate"] == 0.3
    assert run.summary["validation/latest_update"] == 0
    assert run.log_commits == [True]

    trajectory = EmbodiedTrajectory(
        task="pick cube",
        reward=1.0,
        metrics={"success": True},
    )
    evaluation = EvaluationResult(
        step=5,
        metrics={
            "episodes": 10,
            "paired/baseline_success_rate": 0.3,
            "paired/candidate_success_rate": 0.6,
            "paired/success_rate_lift": 0.3,
        },
        artifacts={},
        trajectories=(trajectory,),
    )
    asyncio.run(
        observer.log_step(
            5,
            [EmbodiedTrajectoryGroup([trajectory])],
            LocalTrainResult(step=5, metrics={}),
            evaluation,
            config,
        )
    )

    assert [payload["experiment/update"] for payload, _ in run.logs] == [0, 5]
    assert [payload["validation/success_rate"] for payload, _ in run.logs] == [
        0.3,
        0.6,
    ]
    assert all(logged_step is None for _, logged_step in run.logs)


def test_evaluation_commits_before_next_rollout_with_policy_version_axis(
    tmp_path: Path,
) -> None:
    raw = _config(tmp_path).model_dump(mode="python")
    raw["training"]["updates"] = 2
    config = EmbodiedExperimentConfig.model_validate(raw)
    run = _CommittedRun()
    observer = WandbWeaveObserver(config, wandb_run=run, wandb_module=_Wandb)
    trajectory = EmbodiedTrajectory(
        task="pick cube",
        reward=1.0,
        metrics={"success": True},
    )
    groups = [EmbodiedTrajectoryGroup([trajectory])]

    asyncio.run(
        observer.log_initial_evaluation(
            EvaluationResult(step=0, metrics={"success_rate": 0.25}, artifacts={}),
            config,
        )
    )
    asyncio.run(observer.log_rollout(0, groups, config))
    asyncio.run(
        observer.log_step(
            1,
            groups,
            LocalTrainResult(step=1, metrics={"loss": 0.5}),
            EvaluationResult(step=1, metrics={"success_rate": 0.56}, artifacts={}),
            config,
        )
    )
    assert run.history[-1]["validation/success_rate"] == 0.56
    assert "train/success_rate" not in run.history[-1]
    asyncio.run(observer.log_rollout(1, groups, config))
    asyncio.run(
        observer.log_step(
            2,
            groups,
            LocalTrainResult(step=2, metrics={"loss": 0.25}),
            EvaluationResult(step=2, metrics={"success_rate": 0.75}, artifacts={}),
            config,
        )
    )

    assert [row["experiment/update"] for row in run.history] == [0, 0, 1, 1, 2]
    assert run.history[0]["validation/success_rate"] == 0.25
    assert run.history[1]["train/success_rate"] == 1.0
    assert run.history[2]["validation/success_rate"] == 0.56
    assert run.history[3]["train/success_rate"] == 1.0
    assert "validation/success_rate" not in run.history[3]
    assert run.history[4]["validation/success_rate"] == 0.75
    assert "train/success_rate" not in run.history[4]
    assert all(step is None for _, step in run.logs)


def test_completed_update_is_committed_even_if_later_work_crashes(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    run = _Run()
    observer = WandbWeaveObserver(config, wandb_run=run, wandb_module=_Wandb)
    group = EmbodiedTrajectoryGroup(
        [EmbodiedTrajectory(task="pick cube", reward=1.0, metrics={"success": True})]
    )

    asyncio.run(
        observer.log_step(
            1,
            [group],
            LocalTrainResult(step=1, metrics={"loss": 0.25}),
            None,
            config,
        )
    )
    observer.close(exit_code=1)

    pending_path = config.storage.output_dir / "wandb" / "pending-history.json"
    assert not pending_path.exists()
    assert run.log_commits == [True]
    assert run.logs == [
        (
            {
                "experiment/update": 1,
                "telemetry/degraded": 0,
                "telemetry/delivery_failures_total": 0,
                "telemetry/wandb_delivery_failures": 0,
                "telemetry/weave_delivery_failures": 0,
                "optimization/loss": 0.25,
            },
            None,
        )
    ]
    assert run.finished is True
    assert run.exit_code == 1


def test_initial_wandb_history_commit_is_gated_before_optimizer_work(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    run = _CommittedRunWithNativeStep()
    observer = WandbWeaveObserver(
        config,
        wandb_run=run,
        wandb_module=_Wandb,
        enforce_initial_history_commit=True,
    )
    trajectory = EmbodiedTrajectory(
        task="pick cube",
        reward=1.0,
        metrics={"success": True},
    )

    asyncio.run(
        observer.log_initial_evaluation(
            EvaluationResult(step=0, metrics={"success_rate": 0.25}, artifacts={}),
            config,
        )
    )
    asyncio.run(
        observer.log_rollout(0, [EmbodiedTrajectoryGroup([trajectory])], config)
    )

    audit = json.loads(
        (config.storage.output_dir / "wandb" / "initial-history-commit.json").read_text(
            encoding="utf-8"
        )
    )
    assert run.history[0]["validation/success_rate"] == 0.25
    assert run.history[1]["train/success_rate"] == 1.0
    assert audit["native_step_after_commit"] == 2
    assert audit["wandb"]["run_url"] == run.url


def test_initial_wandb_history_gate_rejects_uncommitted_native_step(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    run = _CommittedRunWithNativeStep()
    observer = WandbWeaveObserver(
        config,
        wandb_run=run,
        wandb_module=_Wandb,
        enforce_initial_history_commit=True,
    )
    run.log = lambda *_args, **_kwargs: None
    trajectory = EmbodiedTrajectory(task="pick cube", reward=1.0)

    with pytest.raises(RuntimeError, match="native Step must be 1"):
        asyncio.run(
            observer.log_rollout(
                0,
                [EmbodiedTrajectoryGroup([trajectory])],
                config,
            )
        )


def test_observer_limits_resume_to_explicit_writer_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _integration_config(
        tmp_path,
        wandb_enabled=True,
        weave_enabled=False,
    ).model_dump(mode="python")
    raw["observability"]["wandb"].update(
        {
            "connection": "resume",
            "run_id": "interrupted-run-id",
            "resume": "must",
        }
    )
    config = EmbodiedExperimentConfig.model_validate(raw)
    init_calls = []

    class FakeWandb:
        Settings = _Wandb.Settings

        @staticmethod
        def init(**kwargs):
            init_calls.append(kwargs)
            return _Run()

    monkeypatch.setattr(
        observability_module,
        "import_module",
        lambda name: FakeWandb if name == "wandb" else None,
    )

    WandbWeaveObserver.start(config)

    assert init_calls[0]["id"] == "interrupted-run-id"
    assert init_calls[0]["resume"] == "must"
    assert init_calls[0]["settings"].kwargs["mode"] == "online"


def test_pending_wandb_step_survives_process_recovery(tmp_path: Path) -> None:
    config = _integration_config(
        tmp_path,
        wandb_enabled=True,
        weave_enabled=False,
    )
    trajectory = EmbodiedTrajectory(
        task="pick cube",
        reward=1.0,
        metrics={"success": True},
    )
    group = EmbodiedTrajectoryGroup([trajectory])
    config = config.model_copy(
        update={
            "observability": config.observability.model_copy(
                update={"delivery_failure_policy": "best_effort"}
            )
        }
    )
    first_run = _FailOnceRun()
    first_observer = WandbWeaveObserver(
        config,
        wandb_run=first_run,
        wandb_module=_Wandb,
    )

    asyncio.run(
        first_observer.log_step(
            1,
            [group],
            LocalTrainResult(step=1, metrics={"loss": 0.25}),
            None,
            config,
        )
    )
    pending_path = config.storage.output_dir / "wandb" / "pending-history.json"
    assert pending_path.is_file()

    resume_config = config.model_copy(
        update={
            "observability": config.observability.model_copy(
                update={
                    "wandb": config.observability.wandb.model_copy(
                        update={
                            "connection": "resume",
                            "run_id": "run-id",
                            "resume": "must",
                        }
                    )
                }
            )
        }
    )
    resumed_run = _Run()
    resumed_observer = WandbWeaveObserver(
        resume_config,
        wandb_run=resumed_run,
        wandb_module=_Wandb,
    )

    asyncio.run(resumed_observer.log_rollout(1, [group], resume_config))

    resumed_payload = resumed_run.logs[0][0]
    assert resumed_payload["optimization/loss"] == 0.25
    assert resumed_payload["train/success_rate"] == 1.0
    assert resumed_payload["train/reward_mean"] == 1.0
    assert not pending_path.exists()


@pytest.mark.parametrize("connection", ["primary", "resume"])
def test_resume_config_preserves_history_after_restoration(tmp_path, connection):
    raw = _integration_config(
        tmp_path, wandb_enabled=True, weave_enabled=False
    ).model_dump(mode="python")
    raw["training"]["updates"] = 100
    if connection == "resume":
        raw["observability"]["wandb"].update(
            connection="resume", run_id="run-id", resume="must"
        )
    config = EmbodiedExperimentConfig.model_validate(raw)
    previous = {"training": {"updates": 1}, "original_marker": "preserve-me"}

    class RunConfig(dict):
        def update(self, value, *, allow_val_change=False):
            assert allow_val_change
            assert len(run.artifacts) == 1
            super().update(value)

    run = _Run()
    run.config = RunConfig(previous)
    observer = WandbWeaveObserver(config, wandb_run=run, wandb_module=_Wandb)
    assert dict(run.config) == previous
    for phase, update in (("initialization", 0), ("update", 2), ("update", 3)):
        asyncio.run(
            observer.log_progress(
                ExperimentProgress(update=update, phase=phase, status="started"),
                config,
            )
        )
        if phase == "initialization":
            assert dict(run.config) == previous
            assert not run.artifacts
    assert not run.logs  # Configuration changes must not consume metric steps.
    if connection == "primary":
        assert dict(run.config) == previous
        assert not run.artifacts
        return
    assert run.config["training"]["updates"] == 100
    assert len(run.artifacts) == 1
    artifact, aliases = run.artifacts[0]
    assert artifact.type == "experiment-config"
    assert aliases == ["latest", "resume-update-1"]
    record = json.loads(Path(artifact.paths[0][1]).read_text())
    assert record["previous"] == previous
    assert record["current"] == config.model_dump(mode="json")
    assert record["restored_policy_version"] == 1
    assert record["next_optimizer_update"] == 2


def test_resume_config_delivery_failure_stops_before_overwrite(tmp_path):
    raw = _integration_config(
        tmp_path, wandb_enabled=True, weave_enabled=False
    ).model_dump(mode="python")
    raw["observability"]["delivery_failure_policy"] = "fail_run"
    raw["observability"]["wandb"].update(
        connection="resume", run_id="run-id", resume="must"
    )
    config = EmbodiedExperimentConfig.model_validate(raw)

    class FailingRun(_Run):
        def log_artifact(self, artifact, aliases):
            raise ConnectionError("configuration artifact unavailable")

    run = FailingRun()
    run.config = {"training": {"updates": 1}}
    observer = WandbWeaveObserver(config, wandb_run=run, wandb_module=_Wandb)
    with pytest.raises(ConnectionError, match="configuration artifact unavailable"):
        asyncio.run(
            observer.log_progress(
                ExperimentProgress(update=1, phase="update", status="started"),
                config,
            )
        )
    assert run.config == {"training": {"updates": 1}}
    assert not observer._resume_config_recorded
    assert not observer._update_started_at
    assert (
        len(list((tmp_path / "output/wandb/configuration-history").glob("*.json"))) == 1
    )


def test_pending_wandb_recovery_filters_legacy_unbounded_diagnostics(
    tmp_path: Path,
) -> None:
    config = _integration_config(
        tmp_path,
        wandb_enabled=True,
        weave_enabled=False,
    )
    pending_path = config.storage.output_dir / "wandb" / "pending-history.json"
    pending_path.parent.mkdir(parents=True)
    legacy_payload = {
        "experiment/update": 1,
        "optimization/embodied_action_token_grpo/loss": 0.25,
        "validation/success_rate": 0.75,
        **{
            f"optimization/embodied_action_token_grpo/internal_percentile_{index}": (
                float(index)
            )
            for index in range(400)
        },
    }
    pending_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "policy_version": 1,
                "resume_contract_fingerprint": config.resume_contract_fingerprint,
                "payload": legacy_payload,
            }
        ),
        encoding="utf-8",
    )
    config = config.model_copy(
        update={
            "observability": config.observability.model_copy(
                update={
                    "wandb": config.observability.wandb.model_copy(
                        update={
                            "connection": "resume",
                            "run_id": "run-id",
                            "resume": "must",
                        }
                    )
                }
            )
        }
    )
    run = _Run()
    observer = WandbWeaveObserver(config, wandb_run=run, wandb_module=_Wandb)
    group = EmbodiedTrajectoryGroup(
        [EmbodiedTrajectory(task="pick cube", reward=1.0, metrics={"success": True})]
    )

    asyncio.run(observer.log_rollout(1, [group], config))

    payload = run.logs[0][0]
    assert payload["optimization/embodied_action_token_grpo/loss"] == 0.25
    assert payload["validation/success_rate"] == 0.75
    assert not any("internal_percentile" in key for key in payload)
    assert len(payload) < 32


def test_weave_server_cache_disabled_by_yaml_clears_stale_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setenv("WEAVE_USE_SERVER_CACHE", "true")
    monkeypatch.setenv("WEAVE_SERVER_CACHE_DIR", "/not/writable")
    monkeypatch.setenv("WEAVE_SERVER_CACHE_SIZE_LIMIT", "999")

    _configure_weave_server_cache(config)

    assert config.observability.weave.use_server_cache is False
    assert os.environ["WEAVE_USE_SERVER_CACHE"] == "false"
    assert "WEAVE_SERVER_CACHE_DIR" not in os.environ
    assert "WEAVE_SERVER_CACHE_SIZE_LIMIT" not in os.environ


def test_weave_server_cache_uses_yaml_directory_and_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    cache_dir = tmp_path / "weave-cache"
    config = config.model_copy(
        update={
            "observability": config.observability.model_copy(
                update={
                    "weave": config.observability.weave.model_copy(
                        update={
                            "use_server_cache": True,
                            "server_cache_dir": cache_dir,
                            "server_cache_size_mb": 16,
                        }
                    )
                }
            )
        }
    )
    monkeypatch.delenv("WEAVE_SERVER_CACHE_DIR", raising=False)

    _configure_weave_server_cache(config)

    assert cache_dir.is_dir()
    assert os.environ["WEAVE_USE_SERVER_CACHE"] == "true"
    assert os.environ["WEAVE_SERVER_CACHE_DIR"] == str(cache_dir.resolve())
    assert os.environ["WEAVE_SERVER_CACHE_SIZE_LIMIT"] == str(16 * 1024 * 1024)


def test_primary_train_namespace_rejects_ui_clutter() -> None:
    observability_module._validate_primary_training_namespace(
        {
            "train/success_rate": 0.5,
            "train/reward_mean": 0.25,
            "performance/rollout/trajectories_per_second": 1.0,
        }
    )

    with pytest.raises(ValueError, match=r"train/training/loss"):
        observability_module._validate_primary_training_namespace(
            {
                "train/success_rate": 0.5,
                "train/training/loss": 0.1,
            }
        )


def test_wandb_training_history_filters_unbounded_backend_diagnostics() -> None:
    metrics: dict[str, float] = {
        f"embodied_action_token_grpo/internal_percentile_{index}": float(index)
        for index in range(400)
    }
    metrics.update(
        {
            "embodied_action_token_grpo/loss": 0.25,
            "embodied_action_token_grpo/grad_norm": 0.5,
            "embodied_action_token_grpo/groups_with_reward_variance": 6.0,
            "embodied_action_token_grpo/optimizer_step_completed": 1.0,
            "embodied_action_token_grpo/policy_parameters_updated": 1.0,
            "embodied_action_token_grpo/ratio_mean": 1.0,
            "embodied_action_token_grpo/loss_first_subupdate": 0.0,
            "embodied_action_token_grpo/loss_last_subupdate": 0.01,
            "embodied_action_token_grpo/ratio_max_last_subupdate": 2.5,
            "embodied_action_token_grpo/worker_gradient_pairwise_cosine_mean": -0.02,
            "embodied_action_token_grpo/worker_gradient_resultant_ratio": 0.34,
            "embodied_action_token_grpo/worker_gradient_signal_to_rms_ratio": 0.32,
            "embodied_action_token_grpo/worker_gradient_noise_to_signal_ratio": 2.94,
            "embodied_action_token_grpo/worker_gradient_effective_aligned_workers": 0.9,
            "rollout/trajectories_per_second": 2.0,
        }
    )

    payload = observability_module._wandb_training_history_payload(metrics)

    assert payload == {
        "optimization/embodied_action_token_grpo/loss": 0.25,
        "optimization/embodied_action_token_grpo/grad_norm": 0.5,
        "signal/embodied_action_token_grpo/groups_with_reward_variance": 6.0,
        "optimization/embodied_action_token_grpo/optimizer_step_completed": 1.0,
        "optimization/embodied_action_token_grpo/policy_parameters_updated": 1.0,
        "optimization/embodied_action_token_grpo/ratio_mean": 1.0,
        "optimization/embodied_action_token_grpo/loss_first_subupdate": 0.0,
        "optimization/embodied_action_token_grpo/loss_last_subupdate": 0.01,
        "optimization/embodied_action_token_grpo/ratio_max_last_subupdate": 2.5,
        "signal/embodied_action_token_grpo/worker_gradient_pairwise_cosine_mean": -0.02,
        "signal/embodied_action_token_grpo/worker_gradient_resultant_ratio": 0.34,
        "signal/embodied_action_token_grpo/worker_gradient_signal_to_rms_ratio": 0.32,
        "signal/embodied_action_token_grpo/worker_gradient_noise_to_signal_ratio": 2.94,
        "signal/embodied_action_token_grpo/worker_gradient_effective_aligned_workers": 0.9,
        "performance/rollout/trajectories_per_second": 2.0,
    }
    assert len(payload) <= observability_module._MAX_WANDB_TRAINING_HISTORY_METRICS


def test_observer_logs_metrics_model_video_and_nested_weave_calls(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    checkpoint_manifest = {
        "schema_version": 1,
        "type": "openvla-peft-adapter",
        "base_model_id": config.policy.path,
        "base_model_revision": config.policy.revision,
        "model_loader": "native",
    }
    (checkpoint / "art_embodied_checkpoint.json").write_text(
        json.dumps(checkpoint_manifest),
        encoding="utf-8",
    )
    (checkpoint / "art_embodied_checkpoint_complete.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "complete": True,
                "config_fingerprint": config.fingerprint,
                "resume_contract_fingerprint": config.resume_contract_fingerprint,
                "metadata": {"update_step": 1},
                "files": [{"path": "adapter_model.safetensors"}],
            }
        ),
        encoding="utf-8",
    )
    video = tmp_path / "rollout.gif"
    video.write_bytes(b"GIF89a")
    trajectory = EmbodiedTrajectory(
        task="pick cube",
        reward=1.0,
        metrics={"success": True, "episode_steps": 1, "duration": 0.5},
        actions=[
            Action(
                step=0,
                kind="continuous",
                raw=[0.1],
                decoded=[0.1],
                metadata={"native_runtime": {"runtime": "art_native_openvla_oft"}},
            )
        ],
        media=[
            MediaRef(
                uri=video.resolve().as_uri(),
                kind="video",
                mime_type="image/gif",
                caption="successful rollout",
                metadata={"fps": 10, "format": "gif"},
            )
        ],
    )
    group = EmbodiedTrajectoryGroup(
        [trajectory],
        metadata={
            "scenario_id": "s0",
            "group_index": 0,
            "environment_seed": 123,
        },
    )
    train_result = LocalTrainResult(
        step=1,
        metrics={
            "loss": 0.25,
            "nonfinite_metric": float("inf"),
            "embodied_flow_sde_grpo/approximate_kl": 0.01,
            "embodied_flow_sde_grpo/groups_kept": 1.0,
            "training/elapsed_seconds": 2.0,
            "rollout/trajectories_per_second": 3.0,
            "rollout/coordinator_resident_memory_mb": 1024.0,
            # Defensive inputs: backend prefixes must not create nested train/*
            # keys that bury the two primary training curves in W&B.
            "train/training/optimizer_seconds": 1.5,
            "train/rollout/actor_wait_seconds": 0.5,
        },
        checkpoint_path=str(checkpoint),
    )
    run = _Run()
    weave_client = _WeaveClient()
    observer = WandbWeaveObserver(
        config,
        wandb_run=run,
        wandb_module=_Wandb,
        weave_client=weave_client,
        weave_module=SimpleNamespace(Content=_Content),
    )

    asyncio.run(observer.log_rollout(0, [group], config))
    asyncio.run(observer.log_step(1, [group], train_result, None, config))
    observer.close()

    rollout_payload, rollout_step = run.logs[0]
    payload, step = run.logs[1]
    assert rollout_step is None
    assert step is None
    assert rollout_payload["train/reward_mean"] == 1.0
    assert rollout_payload["train/success_rate"] == 1.0
    assert rollout_payload["signal/rollout_groups"] == 1
    assert rollout_payload["signal/rollout_trajectories"] == 1
    assert rollout_payload["signal/mixed_reward_groups"] == 0
    assert rollout_payload["signal/all_failure_groups"] == 0
    assert rollout_payload["signal/all_success_groups"] == 1
    assert rollout_payload["signal/mixed_reward_group_fraction"] == 0.0
    group_table = rollout_payload["train_details/group_outcomes"]
    assert group_table.columns[-1] == "signal_class"
    assert group_table.data[0][-1] == "all_success"
    assert rollout_payload["train_details/success_count"] == 1
    assert rollout_payload["train_details/success_denominator"] == 1
    assert rollout_payload["train_details/episode_steps_mean"] == 1.0
    assert rollout_payload["train_details/duration_mean"] == 0.5
    assert payload["optimization/loss"] == 0.25
    assert payload["optimization/embodied_flow_sde_grpo/approximate_kl"] == 0.01
    assert payload["signal/embodied_flow_sde_grpo/groups_kept"] == 1.0
    assert payload["performance/training/elapsed_seconds"] == 2.0
    assert payload["performance/rollout/trajectories_per_second"] == 3.0
    assert payload["performance/rollout/coordinator_resident_memory_mb"] == 1024.0
    assert payload["performance/training/optimizer_seconds"] == 1.5
    assert payload["performance/rollout/actor_wait_seconds"] == 0.5
    assert {key for key in rollout_payload if key.startswith("train/")} == {
        "train/reward_mean",
        "train/success_rate",
    }
    assert not any(key.startswith("train/") for key in payload)
    assert rollout_payload["media/simulation/train/0"].path == str(video)
    rollout_evidence = json.loads(
        (
            config.storage.output_dir / "rollout-evidence/policy-version-000000.json"
        ).read_text(encoding="utf-8")
    )
    assert rollout_evidence["summary"] == {
        "all_failure_groups": 0,
        "all_success_groups": 1,
        "groups": 1,
        "mixed_reward_group_fraction": 0.0,
        "mixed_reward_groups": 0,
        "trajectories": 1,
    }
    assert rollout_evidence["groups"][0]["environment_seed"] == 123
    assert rollout_evidence["groups"][0]["successes"] == [1]
    artifact, aliases = run.artifacts[0]
    assert aliases == ["latest", "update-1"]
    assert artifact.type == "model"
    assert artifact.metadata["update"] == 1
    assert artifact.metadata["train_metrics"]["nonfinite_metric"] is None
    assert artifact.metadata["config_fingerprint"] == config.fingerprint
    assert artifact.metadata["policy"]["revision"] == config.policy.revision
    assert artifact.metadata["checkpoint"] == {
        **checkpoint_manifest,
        "transaction": {
            "schema_version": 1,
            "complete": True,
            "config_fingerprint": config.fingerprint,
            "resume_contract_fingerprint": config.resume_contract_fingerprint,
            "metadata": {"update_step": 1},
            "file_count": 1,
        },
    }
    assert artifact.paths == [("dir", str(checkpoint))]
    assert run.finished is True
    assert run.exit_code == 0

    assert [call.op for call in weave_client.calls] == [
        "art_embodied.training_update",
        "art_embodied.trajectory_group",
        "art_embodied.trajectory",
    ]
    trajectory_call, trajectory_output, exception = weave_client.finished[0]
    assert trajectory_call.op == "art_embodied.trajectory"
    assert trajectory_output["media"]["video_0"]["path"] == str(video)
    assert trajectory_output["summary"]["success"] is True
    assert trajectory_output["steps"][0]["actions"][0]["decoded"] == [0.1]
    assert trajectory_output["actions"][0]["metadata"]["native_runtime"] == {
        "runtime": "art_native_openvla_oft"
    }
    assert exception is None
    update_call = weave_client.calls[0]
    assert update_call.inputs["policy"]["path"] == config.policy.path
    update_output = next(
        output
        for call, output, error in weave_client.finished
        if call.op == "art_embodied.training_update" and error is None
    )
    assert update_output["checkpoint"] == artifact.metadata["checkpoint"]


def test_rollout_group_evidence_is_local_even_without_wandb(tmp_path: Path) -> None:
    raw = _config(tmp_path).model_dump(mode="python")
    raw["observability"]["wandb"]["enabled"] = False
    config = EmbodiedExperimentConfig.model_validate(raw)
    groups = [
        EmbodiedTrajectoryGroup(
            [
                EmbodiedTrajectory(
                    task="frontier-task",
                    reward=float(success),
                    metrics={"success": success},
                )
                for success in (False, True)
            ],
            metadata={"scenario_id": "frontier", "group_index": 7},
        )
    ]
    observer = WandbWeaveObserver(config)

    asyncio.run(observer.log_rollout(3, groups, config))

    evidence = json.loads(
        (
            config.storage.output_dir / "rollout-evidence/policy-version-000003.json"
        ).read_text(encoding="utf-8")
    )
    assert evidence["summary"]["mixed_reward_groups"] == 1
    assert evidence["summary"]["mixed_reward_group_fraction"] == 1.0
    assert evidence["groups"][0]["group_index"] == 7
    assert evidence["groups"][0]["signal_class"] == "mixed"


def test_observer_keeps_simulation_and_lookahead_in_separate_panels(
    tmp_path: Path,
) -> None:
    raw = _config(tmp_path).model_dump(mode="python")
    raw["observability"]["lookahead_preview"]["enabled"] = True
    config = EmbodiedExperimentConfig.model_validate(raw)
    simulation = tmp_path / "simulation.gif"
    lookahead = tmp_path / "lookahead.gif"
    simulation.write_bytes(b"GIF89a")
    lookahead.write_bytes(b"GIF89a")
    trajectory = EmbodiedTrajectory(
        task="pick",
        reward=1.0,
        metrics={"success": True},
        media=[
            MediaRef(
                uri=simulation.resolve().as_uri(),
                kind="video",
                metadata={"role": "simulation", "format": "gif"},
            ),
            MediaRef(
                uri=lookahead.resolve().as_uri(),
                kind="video",
                metadata={"role": "lookahead_preview", "format": "gif"},
            ),
        ],
    )
    run = _Run()
    observer = WandbWeaveObserver(
        config,
        wandb_run=run,
        wandb_module=_Wandb,
    )

    asyncio.run(
        observer.log_rollout(
            0,
            [EmbodiedTrajectoryGroup([trajectory])],
            config,
        )
    )

    payload, _step = run.logs[0]
    assert payload["media/simulation/train/0"].path == str(simulation)
    assert payload["media/lookahead/train/0"].path == str(lookahead)


def test_observer_publishes_evaluation_evidence_bundle(tmp_path: Path) -> None:
    config = _config(tmp_path)
    outcomes = tmp_path / "outcomes.json"
    outcomes.write_text('{"schema_version": 1, "episodes": [{}]}\n')
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "art_embodied_evaluation_evidence",
                "identity": {"source": {"package_tree": {"sha256": "abc"}}},
                "outcomes": {"sha256": "outcomes-digest"},
            }
        )
    )
    run = _Run()
    observer = WandbWeaveObserver(config, wandb_run=run, wandb_module=_Wandb)
    evaluation = EvaluationResult(
        step=10,
        metrics={"success_rate": 0.75},
        artifacts={
            "episode_outcomes_json": str(outcomes),
            "evaluation_evidence_json": str(evidence),
        },
    )

    asyncio.run(observer.log_evaluation(10, evaluation, config))

    artifact, aliases = run.artifacts[0]
    assert artifact.name == f"{config.experiment.run}-evaluation"
    assert artifact.type == "evaluation"
    assert aliases == ["latest", "update-10"]
    assert artifact.metadata["data_role"] == "diagnostic"
    assert artifact.metadata["outcomes_sha256"] == "outcomes-digest"
    assert artifact.paths == [
        ("file", str(outcomes)),
        ("file", str(evidence)),
    ]


def test_observer_names_colliding_evaluation_artifact_files(tmp_path: Path) -> None:
    config = _config(tmp_path)
    candidate = tmp_path / "candidate" / "update_000010_episode_outcomes.json"
    baseline = tmp_path / "baseline" / "update_000010_episode_outcomes.json"
    candidate.parent.mkdir()
    baseline.parent.mkdir()
    candidate.write_text('{"schema_version": 1, "episodes": [{}]}\n')
    baseline.write_text('{"schema_version": 1, "episodes": [{}]}\n')
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "art_embodied_evaluation_evidence",
                "outcomes": {"sha256": "candidate-digest"},
            }
        )
    )
    run = _Run()
    observer = WandbWeaveObserver(config, wandb_run=run, wandb_module=_Wandb)
    evaluation = EvaluationResult(
        step=10,
        metrics={"success_rate": 0.75},
        artifacts={
            "episode_outcomes_json": str(candidate),
            "baseline_episode_outcomes_json": str(baseline),
            "evaluation_evidence_json": str(evidence),
        },
    )

    asyncio.run(observer.log_evaluation(10, evaluation, config))

    artifact, _aliases = run.artifacts[0]
    assert artifact.paths == [
        ("file", str(candidate)),
        (
            "file",
            str(baseline),
            "baseline_episode_outcomes_json/update_000010_episode_outcomes.json",
        ),
        ("file", str(evidence)),
    ]


def test_observer_logs_evaluation_only_without_train_payload(tmp_path: Path) -> None:
    config = _config(tmp_path)
    video = tmp_path / "evaluation.gif"
    video.write_bytes(b"GIF89a")
    outcome_path = tmp_path / "outcomes.json"
    outcome_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "episodes": [
                    {
                        "episode": 0,
                        "scenario_id": "eval-0",
                        "task": "pick cube",
                        "reward": 1.0,
                        "success": 1.0,
                        "episode_steps": 2,
                        "duration_seconds": 0.5,
                        "environment_seed": 1,
                        "policy_seed": 2,
                        "completed": True,
                        "error_type": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    trajectory = EmbodiedTrajectory(
        task="pick cube",
        reward=1.0,
        metrics={"success": True, "episode_steps": 2},
        media=[
            MediaRef(
                uri=video.resolve().as_uri(),
                kind="video",
                mime_type="image/gif",
            )
        ],
    )
    evaluation = EvaluationResult(
        step=0,
        metrics={
            "success_rate": 1.0,
            "scenario/eval-0/episodes": 1.0,
            "scenario/eval-0/success_rate": 1.0,
            "task/held-out-pick/episodes": 1.0,
            "task/held-out-pick/success_rate": 1.0,
            "paired/task/held-out-pick/success_rate_lift": 0.5,
        },
        artifacts={"episode_outcomes_json": str(outcome_path)},
        trajectories=(trajectory,),
    )
    run = _Run()
    weave_client = _WeaveClient()
    observer = WandbWeaveObserver(
        config,
        wandb_run=run,
        wandb_module=_Wandb,
        weave_client=weave_client,
        weave_module=SimpleNamespace(Content=_Content),
    )

    asyncio.run(observer.log_evaluation(0, evaluation, config))

    payload, logged_step = run.logs[0]
    assert logged_step is None
    assert payload["validation/success_rate"] == 1.0
    assert "eval_details/success_rate" not in payload
    assert "eval/success_rate" not in payload
    assert "eval_details/scenario/eval-0/episodes" not in payload
    assert "eval_details/scenario/eval-0/success_rate" not in payload
    assert "eval/scenario/eval-0/success_rate" not in payload
    assert "eval_tasks/task/held-out-pick/episodes" not in payload
    assert "eval_tasks/task/held-out-pick/success_rate" not in payload
    assert "eval_tasks/paired/task/held-out-pick/success_rate_lift" not in payload
    assert "eval/task/held-out-pick/success_rate" not in payload
    assert isinstance(payload["eval_details/episodes_table"], _Wandb.Table)
    assert isinstance(payload["eval_details/task_summary_table"], _Wandb.Table)
    assert payload["media/simulation/eval/0"].path == str(video)
    assert "train/groups" not in payload
    root = weave_client.calls[0]
    assert root.op == "art_embodied.evaluation_run"
    evaluation_call = next(
        call for call in weave_client.calls if call.op == "art_embodied.evaluation"
    )
    trajectory_call = next(
        call
        for call in weave_client.calls
        if call.op == "art_embodied.evaluation_trajectory"
    )
    assert evaluation_call.parent is root
    assert trajectory_call.parent is evaluation_call
    root_output = next(
        output for call, output, error in weave_client.finished if call is root
    )
    assert root_output["metrics"]["success_rate"] == 1.0


def test_observer_logs_evaluation_video_as_wandb_media_and_weave_content(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    video = tmp_path / "evaluation.gif"
    video.write_bytes(b"GIF89a")
    trajectory = EmbodiedTrajectory(
        task="held-out pick",
        reward=1.0,
        metrics={"success": True},
        media=[
            MediaRef(
                uri=video.resolve().as_uri(),
                kind="video",
                mime_type="image/gif",
            )
        ],
    )
    outcomes_path = tmp_path / "evaluation-outcomes.json"
    outcomes_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "episodes": [
                    {
                        "episode": 0,
                        "scenario_id": "held-out-0",
                        "task": "held-out pick",
                        "reward": 1.0,
                        "success": 1.0,
                        "episode_steps": 8,
                        "duration_seconds": 1.5,
                        "environment_seed": 11,
                        "policy_seed": 12,
                        "completed": True,
                        "error_type": None,
                    },
                    {
                        "episode": 1,
                        "scenario_id": "held-out-1",
                        "task": "held-out place",
                        "reward": 0.0,
                        "success": 0.0,
                        "episode_steps": 0,
                        "duration_seconds": None,
                        "environment_seed": 13,
                        "policy_seed": 14,
                        "completed": False,
                        "error_type": "RuntimeError",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    evaluation = EvaluationResult(
        step=1,
        metrics={"success_rate": 1.0},
        artifacts={"episode_outcomes_json": str(outcomes_path)},
        trajectories=(trajectory,),
    )
    run = _Run()
    weave_client = _WeaveClient()
    observer = WandbWeaveObserver(
        config,
        wandb_run=run,
        wandb_module=_Wandb,
        weave_client=weave_client,
        weave_module=SimpleNamespace(Content=_Content),
    )

    asyncio.run(
        observer.log_step(
            1,
            [EmbodiedTrajectoryGroup([trajectory])],
            LocalTrainResult(step=1, metrics={}),
            evaluation,
            config,
        )
    )

    payload, logged_step = run.logs[0]
    assert logged_step is None
    assert payload["validation/success_rate"] == 1.0
    assert "eval_details/success_rate" not in payload
    assert "eval/success_rate" not in payload
    assert payload["media/simulation/eval/0"].path == str(video)
    assert payload["eval_details/episodes_table"].data[0][3] == "held-out pick"
    assert payload["eval_details/episodes_table"].data[0][5] == 1.0
    assert payload["eval_details/episodes_table"].data[1][10] is False
    assert payload["eval_details/episodes_table"].data[1][11] == "RuntimeError"
    assert [call.op for call in weave_client.calls][-2:] == [
        "art_embodied.evaluation",
        "art_embodied.evaluation_trajectory",
    ]
    eval_output = next(
        output
        for call, output, exception in weave_client.finished
        if call.op == "art_embodied.evaluation_trajectory" and exception is None
    )
    assert eval_output["media"]["video_0"]["path"] == str(video)


def test_observer_prefers_success_and_failure_representative_videos(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config = config.model_copy(
        update={
            "observability": config.observability.model_copy(
                update={
                    "videos_per_update": 2,
                    "videos_per_evaluation": 2,
                    "weave": config.observability.weave.model_copy(
                        update={
                            "max_groups_per_update": 2,
                            "max_trajectories_per_group": 2,
                            "max_evaluation_trajectories": 2,
                        }
                    ),
                }
            )
        }
    )

    trajectories = []
    for index, success in enumerate((True, True, False)):
        video = tmp_path / f"trajectory-{index}.gif"
        video.write_bytes(b"GIF89a")
        trajectories.append(
            EmbodiedTrajectory(
                task=f"task-{index}",
                reward=float(success),
                metrics={"success": success},
                media=[
                    MediaRef(
                        uri=video.resolve().as_uri(),
                        kind="video",
                        mime_type="image/gif",
                    )
                ],
            )
        )

    run = _Run()
    weave_client = _WeaveClient()
    observer = WandbWeaveObserver(
        config,
        wandb_run=run,
        wandb_module=_Wandb,
        weave_client=weave_client,
        weave_module=SimpleNamespace(Content=_Content),
    )
    groups = [EmbodiedTrajectoryGroup([trajectory]) for trajectory in trajectories]
    evaluation = EvaluationResult(
        step=1,
        metrics={"success_rate": 2 / 3},
        artifacts={},
        trajectories=tuple(trajectories),
    )

    asyncio.run(observer.log_rollout(0, groups, config))
    asyncio.run(
        observer.log_step(
            1,
            groups,
            LocalTrainResult(step=1, metrics={}),
            evaluation,
            config,
        )
    )

    rollout_payload, rollout_step = run.logs[0]
    evaluation_payload, evaluation_step = run.logs[1]
    assert rollout_step is None
    assert evaluation_step is None
    train_video_keys = sorted(
        key for key in rollout_payload if key.startswith("media/simulation/train/")
    )
    evaluation_video_keys = sorted(
        key for key in evaluation_payload if key.startswith("media/simulation/eval/")
    )
    assert train_video_keys == [
        "media/simulation/train/0",
        "media/simulation/train/1",
    ]
    assert evaluation_video_keys == [
        "media/simulation/eval/0",
        "media/simulation/eval/1",
    ]
    assert rollout_payload["media/simulation/train/0"].path.endswith("trajectory-0.gif")
    assert rollout_payload["media/simulation/train/1"].path.endswith("trajectory-2.gif")
    assert evaluation_payload["media/simulation/eval/0"].path.endswith(
        "trajectory-0.gif"
    )
    assert evaluation_payload["media/simulation/eval/1"].path.endswith(
        "trajectory-2.gif"
    )
    traced_group_indices = [
        call.inputs["group_index"]
        for call in weave_client.calls
        if call.op == "art_embodied.trajectory_group"
    ]
    assert traced_group_indices == [0, 2]
    traced_evaluation_indices = [
        call.inputs["trajectory_index"]
        for call in weave_client.calls
        if call.op == "art_embodied.evaluation_trajectory"
    ]
    assert traced_evaluation_indices == [0, 2]


def test_observer_reuses_stable_media_keys_across_updates(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config = config.model_copy(
        update={
            "observability": config.observability.model_copy(
                update={
                    "videos_per_update": 1,
                    "videos_per_evaluation": 1,
                }
            )
        }
    )
    video = tmp_path / "trajectory.gif"
    video.write_bytes(b"GIF89a")
    trajectory = EmbodiedTrajectory(
        task="pick cube",
        reward=1.0,
        metrics={"success": True},
        media=[
            MediaRef(
                uri=video.resolve().as_uri(),
                kind="video",
                mime_type="image/gif",
                caption="task=pick cube | outcome=success",
            )
        ],
    )
    group = EmbodiedTrajectoryGroup([trajectory])
    evaluation = EvaluationResult(
        step=1,
        metrics={"success_rate": 1.0},
        artifacts={},
        trajectories=(trajectory,),
    )
    run = _Run()
    observer = WandbWeaveObserver(
        config,
        wandb_run=run,
        wandb_module=_Wandb,
    )

    for update in (1, 2):
        asyncio.run(observer.log_rollout(update - 1, [group], config))
        asyncio.run(
            observer.log_step(
                update,
                [group],
                LocalTrainResult(step=update, metrics={}),
                evaluation,
                config,
            )
        )

    rollout_payloads = [run.logs[0][0], run.logs[2][0]]
    evaluation_payloads = [run.logs[1][0], run.logs[3][0]]
    assert all(
        sorted(key for key in payload if key.startswith("media/"))
        == ["media/simulation/train/0"]
        for payload in rollout_payloads
    )
    assert all(
        sorted(key for key in payload if key.startswith("media/"))
        == ["media/simulation/eval/0"]
        for payload in evaluation_payloads
    )
    assert [payload["experiment/update"] for payload in rollout_payloads] == [0, 1]
    assert [payload["experiment/update"] for payload in evaluation_payloads] == [1, 2]
    assert all(logged_step is None for _, logged_step in run.logs)


def test_observer_logs_a_small_stable_fixed_evaluation_surface(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    run = _Run()
    observer = WandbWeaveObserver(config, wandb_run=run, wandb_module=_Wandb)
    trajectory = EmbodiedTrajectory(
        task="pick cube",
        reward=1.0,
        metrics={"success": True},
    )
    evaluation = EvaluationResult(
        step=5,
        metrics={
            "success_rate": 0.69,
            "episodes": 100.0,
            "scenario/noisy-detail/success_rate": 1.0,
            "task/pick-cube/candidate_success_rate": 0.7,
            "paired/baseline_success_rate": 0.62,
            "paired/candidate_success_rate": 0.69,
            "paired/success_rate_lift": 0.07,
            "paired/success_rate_lift_ci95_low": 0.0,
            "paired/success_rate_lift_ci95_high": 0.14,
            "paired/baseline_task_macro_success_rate": 0.61,
            "paired/candidate_task_macro_success_rate": 0.68,
            "paired/task_macro_success_rate_lift": 0.07,
            "paired/task_macro_success_rate_lift_ci95_low": 0.01,
            "paired/task_macro_success_rate_lift_ci95_high": 0.13,
            "paired/mcnemar_exact_p_value": 0.0125,
            "paired/improved_pairs": 14,
            "paired/regressed_pairs": 4,
            "paired/unchanged_success_pairs": 51,
            "paired/unchanged_failure_pairs": 31,
        },
        artifacts={},
        trajectories=(trajectory,),
    )

    asyncio.run(
        observer.log_step(
            5,
            [EmbodiedTrajectoryGroup([trajectory])],
            LocalTrainResult(step=5, metrics={}),
            evaluation,
            config,
        )
    )

    payload, logged_step = run.logs[0]
    assert logged_step is None
    assert payload["experiment/update"] == 5
    assert payload["validation/success_rate"] == 0.69
    assert payload["validation/checkpoint_role"] == "candidate"
    assert payload["validation/success_rate_lift"] == 0.07
    assert payload["validation/success_rate_lift_ci95_low"] == 0.0
    assert payload["validation/success_rate_lift_ci95_high"] == 0.14
    assert payload["validation/episodes"] == 100.0
    assert payload["validation/task_macro_success_rate"] == 0.68
    assert payload["validation/task_macro_success_rate_lift"] == 0.07
    assert payload["validation/task_macro_success_rate_lift_ci95_low"] == 0.01
    assert payload["validation/task_macro_success_rate_lift_ci95_high"] == 0.13
    assert payload["validation/mcnemar_exact_p_value"] == 0.0125
    assert payload["validation/improved_pairs"] == 14.0
    assert payload["validation/regressed_pairs"] == 4.0
    assert payload["validation/unchanged_success_pairs"] == 51.0
    assert payload["validation/unchanged_failure_pairs"] == 31.0
    assert "eval_details/scenario/noisy-detail/success_rate" not in payload
    assert "eval_tasks/task/pick-cube/candidate_success_rate" not in payload
    assert run.summary["validation/latest_update"] == 5
    assert run.summary["validation/split"] == "train_matched"
    assert run.summary["validation/data_role"] == "diagnostic"
    assert run.summary["validation/success_rate"] == 0.69
    assert run.summary["validation/latest_success_rate"] == 0.69
    assert run.summary["validation/success_rate_lift"] == 0.07


def test_policy_series_evaluation_uses_native_wandb_step(tmp_path: Path) -> None:
    config = _config(tmp_path)

    class NativeRun(_Run):
        step = 0

        def log(self, payload, step=None, commit=None):
            super().log(payload, step=step, commit=commit)
            if step is not None:
                self.step = step
            if commit is not False:
                self.step += 1

    run = NativeRun()
    observer = WandbWeaveObserver(config, wandb_run=run, wandb_module=_Wandb)
    evaluation = EvaluationResult(
        step=300,
        metrics={"success_rate": 0.72, "episodes": 100.0},
        artifacts={},
    )

    asyncio.run(
        observer.log_evaluation_checkpoint(
            300,
            evaluation,
            config,
            checkpoint_role="sft_candidate",
            use_native_wandb_step=True,
        )
    )

    assert run.logs[0] == ({}, 300)
    assert run.log_commits[0] is False
    payload, logged_step = run.logs[1]
    assert logged_step == 300
    assert payload["experiment/update"] == 300
    assert payload["validation/success_rate"] == 0.72
    assert payload["validation/checkpoint_role"] == "sft_candidate"
    assert run.summary["validation/latest_update"] == 300
    assert run.summary["validation/latest_success_rate"] == 0.72
    assert run.summary["validation/best_update"] == 300
    assert run.summary["validation/best_success_rate"] == 0.72

    asyncio.run(
        observer.log_evaluation_checkpoint(
            301,
            EvaluationResult(
                step=301,
                metrics={"success_rate": 0.65, "episodes": 100.0},
                artifacts={},
            ),
            config,
            checkpoint_role="sft_candidate",
            use_native_wandb_step=True,
        )
    )
    assert run.summary["validation/latest_update"] == 301
    assert run.summary["validation/latest_success_rate"] == 0.65
    assert run.summary["validation/best_update"] == 300
    assert run.summary["validation/best_success_rate"] == 0.72

    with pytest.raises(ValueError, match="cannot overwrite a native W&B step"):
        asyncio.run(
            observer.log_evaluation_checkpoint(
                300,
                evaluation,
                config,
                checkpoint_role="sft_candidate",
                use_native_wandb_step=True,
            )
        )
    assert run.step == 302


def test_observer_keeps_sealed_evaluation_on_a_distinct_primary_surface(
    tmp_path: Path,
) -> None:
    raw = _config(tmp_path).model_dump(mode="python")
    raw["evaluation"].update(
        {
            "split": "held_out",
            "data_role": "sealed_test",
            "checkpoint_selection": "last",
        }
    )
    config = EmbodiedExperimentConfig.model_validate(raw)
    run = _Run()
    observer = WandbWeaveObserver(config, wandb_run=run, wandb_module=_Wandb)
    trajectory = EmbodiedTrajectory(task="pick cube", reward=1.0)
    evaluation = EvaluationResult(
        step=10,
        metrics={
            "episodes": 100.0,
            "paired/baseline_success_rate": 0.70,
            "paired/candidate_success_rate": 0.75,
            "paired/success_rate_lift": 0.05,
        },
        artifacts={},
        trajectories=(trajectory,),
    )

    asyncio.run(
        observer.log_step(
            10,
            [EmbodiedTrajectoryGroup([trajectory])],
            LocalTrainResult(step=10, metrics={}),
            evaluation,
            config,
        )
    )

    payload, _ = run.logs[0]
    assert payload["test/success_rate"] == 0.75
    assert payload["test/success_rate_lift"] == 0.05
    assert payload["test/checkpoint_role"] == "candidate"
    assert "validation/success_rate" not in payload
    assert run.summary["test/latest_update"] == 10
    assert run.summary["test/data_role"] == "sealed_test"


def test_weave_trace_selection_keeps_a_video_backed_trajectory(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config = config.model_copy(
        update={
            "observability": config.observability.model_copy(
                update={
                    "videos_per_update": 1,
                    "videos_per_evaluation": 0,
                    "weave": config.observability.weave.model_copy(
                        update={
                            "max_groups_per_update": 1,
                            "max_trajectories_per_group": 1,
                        }
                    ),
                }
            ),
            "evaluation": config.evaluation.model_copy(update={"enabled": False}),
        }
    )
    video = tmp_path / "trace.gif"
    video.write_bytes(b"GIF89a")
    video_trajectory = EmbodiedTrajectory(
        task="video failure",
        metrics={"success": False},
        media=[
            MediaRef(
                uri=video.resolve().as_uri(),
                kind="video",
                mime_type="image/gif",
            )
        ],
    )
    no_video_trajectory = EmbodiedTrajectory(
        task="unrecorded success",
        metrics={"success": True},
    )
    groups = [
        EmbodiedTrajectoryGroup([no_video_trajectory]),
        EmbodiedTrajectoryGroup([video_trajectory]),
    ]
    weave_client = _WeaveClient()
    observer = WandbWeaveObserver(
        config,
        wandb_run=_Run(),
        wandb_module=_Wandb,
        weave_client=weave_client,
        weave_module=SimpleNamespace(Content=_Content),
    )

    asyncio.run(
        observer.log_step(
            1,
            groups,
            LocalTrainResult(step=1, metrics={}),
            None,
            config,
        )
    )

    traced_group = next(
        call
        for call in weave_client.calls
        if call.op == "art_embodied.trajectory_group"
    )
    assert traced_group.inputs["group_index"] == 1
    traced_output = next(
        output
        for call, output, exception in weave_client.finished
        if call.op == "art_embodied.trajectory" and exception is None
    )
    assert traced_output["media"]["video_0"]["path"] == str(video)


def test_weave_trace_excludes_transient_training_payloads(tmp_path: Path) -> None:
    observer = WandbWeaveObserver(_config(tmp_path))
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
                decoded=[0.1, 0.2],
                metadata={
                    "policy_step": 0,
                    "_art_embodied_transient_flow_sde_rollout": object(),
                },
            )
        ],
    )

    output = observer._weave_trajectory_output(trajectory)

    assert output["metadata"] == {"scenario_id": "scenario-1"}
    assert output["actions"][0]["metadata"] == {"policy_step": 0}
    assert output["steps"][0]["actions"][0]["metadata"] == {"policy_step": 0}


def test_observer_fails_acceptance_gate_when_required_train_video_is_missing(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config = config.model_copy(
        update={
            "observability": config.observability.model_copy(
                update={"require_train_video": True}
            )
        }
    )
    observer = WandbWeaveObserver(
        config,
        wandb_run=_Run(),
        wandb_module=_Wandb,
    )
    trajectory = EmbodiedTrajectory(
        task="missing render",
        reward=0.0,
        metrics={"success": False},
    )

    with pytest.raises(RuntimeError, match="Required W&B train video is missing"):
        asyncio.run(
            observer.log_rollout(
                0,
                [EmbodiedTrajectoryGroup([trajectory])],
                config,
            )
        )


def test_observer_aliases_only_improving_evaluated_models_as_best(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config = config.model_copy(
        update={
            "evaluation": config.evaluation.model_copy(
                update={"checkpoint_selection": "evaluation_success"}
            )
        }
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    run = _Run()
    observer = WandbWeaveObserver(config, wandb_run=run, wandb_module=_Wandb)
    group = EmbodiedTrajectoryGroup(
        [EmbodiedTrajectory(task="task", reward=0.0, metrics={"success": False})]
    )

    for step, success_rate in ((1, 0.4), (2, 0.3), (3, 0.6)):
        asyncio.run(
            observer.log_step(
                step,
                [group],
                LocalTrainResult(
                    step=step,
                    metrics={},
                    checkpoint_path=str(checkpoint),
                ),
                EvaluationResult(
                    step=step,
                    metrics={"success_rate": success_rate},
                    artifacts={},
                ),
                config,
            )
        )

    assert run.artifacts[0][1] == ["latest", "update-1", "best"]
    assert run.artifacts[1][1] == ["latest", "update-2"]
    assert run.artifacts[2][1] == ["latest", "update-3", "best"]


def test_observer_logs_live_rollout_progress_and_spans_update_trace(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    run = _Run()
    weave_client = _WeaveClient()
    observer = WandbWeaveObserver(
        config,
        wandb_run=run,
        wandb_module=_Wandb,
        weave_client=weave_client,
        weave_module=SimpleNamespace(Content=_Content),
    )

    asyncio.run(
        observer.log_progress(
            ExperimentProgress(update=1, phase="update", status="started"),
            config,
        )
    )
    asyncio.run(
        observer.log_progress(
            ExperimentProgress(
                update=1,
                phase="rollout",
                status="progress",
                completed=4,
                total=8,
                metrics={
                    "trajectories_completed": 32,
                    "success_count": 20,
                    "success_denominator": 32,
                    "success_rate": 0.625,
                },
            ),
            config,
        )
    )

    assert run.logs == []
    progress_payload = run.summary
    assert progress_payload["monitor/phase"] == "rollout"
    assert progress_payload["monitor/progress"] == 0.5
    assert progress_payload["monitor/success_rate"] == 0.625
    assert progress_payload["monitor/trajectories_completed"] == 32
    root = weave_client.calls[0]
    assert root.op == "art_embodied.training_update"
    assert root.inputs["wandb"]["run_id"] == "run-id"
    assert weave_client.finished == []

    trajectory = EmbodiedTrajectory(
        task="pick cube",
        reward=1.0,
        metrics={"success": True},
    )
    asyncio.run(
        observer.log_step(
            1,
            [EmbodiedTrajectoryGroup([trajectory])],
            LocalTrainResult(step=1, metrics={}),
            None,
            config,
        )
    )
    assert len(run.logs) == 1
    assert run.logs[0][0]["experiment/update"] == 1
    assert run.logs[0][1] is None
    root_finishes = [item for item in weave_client.finished if item[0] is root]
    assert len(root_finishes) == 1
    assert root_finishes[0][1]["wandb"]["run_url"].endswith("run-id")
    assert root_finishes[0][2] is None


def test_observer_keeps_live_training_summary_human_sized(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run = _Run()
    observer = WandbWeaveObserver(config, wandb_run=run, wandb_module=_Wandb)

    asyncio.run(
        observer.log_progress(
            ExperimentProgress(
                update=1,
                phase="training",
                status="completed",
                metrics={
                    "embodied_action_token_gspo/loss": 0.25,
                    "embodied_action_token_gspo/clip_fraction": 0.5,
                    "embodied_action_token_gspo/ratio_p99": 1.1,
                    "embodied_flow_sde_grpo/approximate_kl": 0.002,
                    "embodied_flow_sde_grpo/previous_abs_delta_mean": 0.001,
                    "embodied_flow_sde_grpo/pre_update_alignment_abs_delta_mean": 0.0002,
                    "embodied_flow_sde_grpo/pre_update_active_alignment_abs_delta_mean": 0.0001,
                    "embodied_flow_sde_grpo/optimization_old_policy_abs_delta_mean": 0.001,
                    "embodied_action_token_gspo/lora_a_parameter_delta_norm": 0.01,
                    "embodied_action_token_schedule/subupdates": 4,
                    "rollout/trajectories_per_second": 2.0,
                    "training/elapsed_seconds": 30.0,
                },
            ),
            config,
        )
    )

    assert run.summary["monitor/embodied_action_token_gspo/loss"] == 0.25
    assert run.summary["monitor/embodied_action_token_gspo/clip_fraction"] == 0.5
    assert run.summary["monitor/embodied_flow_sde_grpo/approximate_kl"] == 0.002
    assert (
        run.summary["monitor/embodied_flow_sde_grpo/previous_abs_delta_mean"] == 0.001
    )
    assert (
        run.summary[
            "monitor/embodied_flow_sde_grpo/pre_update_alignment_abs_delta_mean"
        ]
        == 0.0002
    )
    assert (
        run.summary[
            "monitor/embodied_flow_sde_grpo/pre_update_active_alignment_abs_delta_mean"
        ]
        == 0.0001
    )
    assert (
        run.summary["monitor/embodied_action_token_gspo/lora_a_parameter_delta_norm"]
        == 0.01
    )
    assert run.summary["monitor/embodied_action_token_schedule/subupdates"] == 4
    assert run.summary["monitor/rollout/trajectories_per_second"] == 2.0
    assert run.summary["monitor/training/elapsed_seconds"] == 30.0
    assert "monitor/embodied_action_token_gspo/ratio_p99" not in run.summary


def test_observer_excludes_per_scenario_metrics_from_live_summary(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    run = _Run()
    observer = WandbWeaveObserver(config, wandb_run=run, wandb_module=_Wandb)

    asyncio.run(
        observer.log_progress(
            ExperimentProgress(
                update=10,
                phase="evaluation",
                status="completed",
                completed=100,
                total=100,
                metrics={
                    "success_rate": 0.82,
                    "scenario_macro_success_rate": 0.82,
                    "task_macro_success_rate": 0.81,
                    "paired/success_rate_lift": 0.34,
                    "scenario/task-00-state-00/success_rate": 1.0,
                    "task/pick/success_rate": 0.9,
                    "paired/task/pick/success_rate_lift": 0.2,
                },
            ),
            config,
        )
    )

    assert run.summary["monitor/success_rate"] == 0.82
    assert run.summary["monitor/scenario_macro_success_rate"] == 0.82
    assert run.summary["monitor/task_macro_success_rate"] == 0.81
    assert run.summary["monitor/paired/success_rate_lift"] == 0.34
    assert "monitor/scenario/task-00-state-00/success_rate" not in run.summary
    assert "monitor/task/pick/success_rate" not in run.summary
    assert "monitor/paired/task/pick/success_rate_lift" not in run.summary


def test_observer_marks_failed_update_and_wandb_run(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run = _Run()
    weave_client = _WeaveClient()
    observer = WandbWeaveObserver(
        config,
        wandb_run=run,
        wandb_module=_Wandb,
        weave_client=weave_client,
        weave_module=SimpleNamespace(Content=_Content),
    )
    asyncio.run(
        observer.log_progress(
            ExperimentProgress(update=1, phase="update", status="started"),
            config,
        )
    )
    asyncio.run(
        observer.log_progress(
            ExperimentProgress(
                update=1,
                phase="update",
                status="failed",
                message="RuntimeError: rollout failed",
            ),
            config,
        )
    )
    observer.close(exit_code=1)

    assert run.finished is True
    assert run.exit_code == 1
    root, output, exception = weave_client.finished[-1]
    assert root.op == "art_embodied.training_update"
    assert output is None
    assert "rollout failed" in str(exception)


def test_observer_throttles_live_evaluation_progress(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run = _Run()
    observer = WandbWeaveObserver(config, wandb_run=run, wandb_module=_Wandb)

    for completed in (1, 5, 6, 10):
        asyncio.run(
            observer.log_progress(
                ExperimentProgress(
                    update=1,
                    phase="evaluation",
                    status="progress",
                    completed=completed,
                    total=10,
                    metrics={"evaluation_success_rate": completed / 10},
                ),
                config,
            )
        )

    assert [payload["monitor/completed"] for payload in run.summary.updates] == [5, 10]


def test_wandb_delivery_failure_does_not_abort_and_recovery_is_visible(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config = config.model_copy(
        update={
            "observability": config.observability.model_copy(
                update={"delivery_failure_policy": "best_effort"}
            )
        }
    )
    run = _FailOnceRun()
    observer = WandbWeaveObserver(config, wandb_run=run, wandb_module=_Wandb)
    group = EmbodiedTrajectoryGroup([EmbodiedTrajectory(task="pick cube", reward=1.0)])

    asyncio.run(
        observer.log_step(
            1,
            [group],
            LocalTrainResult(step=1, metrics={}),
            None,
            config,
        )
    )
    asyncio.run(
        observer.log_step(
            2,
            [group],
            LocalTrainResult(step=2, metrics={}),
            None,
            config,
        )
    )

    failures = [
        json.loads(line)
        for line in (config.storage.output_dir / "telemetry_failures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert failures[0]["integration"] == "wandb"
    assert failures[0]["operation"] == "log_training_update"
    assert failures[0]["step"] == 1
    assert run.logs[0][0]["telemetry/degraded"] == 1
    assert run.logs[0][0]["telemetry/wandb_delivery_failures"] == 1


def test_required_telemetry_delivery_failure_aborts_run(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config = config.model_copy(
        update={
            "observability": config.observability.model_copy(
                update={"delivery_failure_policy": "fail_run"}
            )
        }
    )
    run = _FailOnceRun()
    observer = WandbWeaveObserver(config, wandb_run=run, wandb_module=_Wandb)
    group = EmbodiedTrajectoryGroup([EmbodiedTrajectory(task="pick cube", reward=1.0)])

    with pytest.raises(ConnectionError, match="injected W&B outage"):
        asyncio.run(
            observer.log_step(
                1,
                [group],
                LocalTrainResult(step=1, metrics={}),
                None,
                config,
            )
        )

    failures = (config.storage.output_dir / "telemetry_failures.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"operation": "log_training_update"' in failures


def test_weave_delivery_failure_does_not_block_wandb_or_close(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config = config.model_copy(
        update={
            "observability": config.observability.model_copy(
                update={"delivery_failure_policy": "best_effort"}
            )
        }
    )
    run = _Run()
    observer = WandbWeaveObserver(
        config,
        wandb_run=run,
        wandb_module=_Wandb,
        weave_client=_FailingWeaveClient(),
        weave_module=SimpleNamespace(Content=_Content),
    )
    group = EmbodiedTrajectoryGroup([EmbodiedTrajectory(task="pick cube", reward=0.0)])

    asyncio.run(
        observer.log_step(
            3,
            [group],
            LocalTrainResult(step=3, metrics={}),
            None,
            config,
        )
    )
    observer.close()

    assert run.logs[0][1] is None
    failures = (config.storage.output_dir / "telemetry_failures.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"integration": "weave"' in failures
    assert run.finished is True


@pytest.mark.parametrize("success_rate", [0.5, 0.9, 0.95])
def test_resume_preserves_best_summary_and_model_alias(tmp_path, success_rate):
    config = _config(tmp_path)
    config = config.model_copy(
        update={
            "evaluation": config.evaluation.model_copy(
                update={"checkpoint_selection": "evaluation_success"}
            ),
            "observability": config.observability.model_copy(
                update={
                    "wandb": config.observability.wandb.model_copy(
                        update={
                            "connection": "resume",
                            "run_id": "run-id",
                            "resume": "must",
                        }
                    )
                }
            ),
        }
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    run = _Run()
    run.summary.update(
        {"validation/best_success_rate": 0.9, "validation/best_update": 5}
    )
    observer = WandbWeaveObserver(config, wandb_run=run, wandb_module=_Wandb)
    asyncio.run(
        observer.log_step(
            10,
            [],
            LocalTrainResult(step=10, metrics={}, checkpoint_path=str(checkpoint)),
            EvaluationResult(
                step=10, metrics={"success_rate": success_rate}, artifacts={}
            ),
            config,
        )
    )
    assert run.summary["validation/best_success_rate"] == max(0.9, success_rate)
    assert run.summary["validation/best_update"] == (10 if success_rate > 0.9 else 5)
    assert ("best" in run.artifacts[0][1]) == (success_rate > 0.9)
    assert run.log_commits == [True]


def test_training_phase_timings_are_visible_outside_train_namespace():
    names = [
        "alignment_guard_seconds",
        "alignment_guard_seconds_max",
        "train_logprob_forward_seconds",
        "train_logprob_forward_seconds_max",
        "train_loss_backward_seconds",
        "train_loss_backward_seconds_max",
        "gradient_apply_seconds",
        "distributed_workers",
        "distributed_worker_seconds_max",
        "distributed_worker_seconds_mean",
    ]
    payload = observability_module._wandb_training_history_payload(
        {
            f"embodied_action_token_grpo/{name}": float(index + 1)
            for index, name in enumerate(names)
        }
    )
    assert set(payload) == {
        f"performance/embodied_action_token_grpo/{name}" for name in names
    }
