import asyncio
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from art_embodied.checkpointing import CheckpointManager
from art_embodied.config import EmbodiedExperimentConfig
from examples.embodied import pi0_fast_long_sealed_eval as entry


@pytest.fixture
def setup(tmp_path, monkeypatch):
    raw = EmbodiedExperimentConfig.from_yaml(
        Path(
            "examples/embodied/pi0_fast_libero_long_task8_full_trajectory_sealed_baseline_v1.yaml"
        )
    ).model_dump(mode="json")
    raw["observability"]["wandb"].update(
        enabled=True,
        mode="online",
        connection="primary",
        run_id=None,
        resume=None,
        native_update_steps=False,
    )
    raw["policy"]["load_kwargs"].update(
        model_compute_dtype="fp16_residual", training_loss_scale=128.0
    )
    raw["evaluation"].update(
        enabled=True, data_role="sealed_test", checkpoint_selection="last"
    )
    raw["storage"]["resume_from_checkpoint"] = None
    config = EmbodiedExperimentConfig.model_validate(raw)
    snapshot = {
        "family": "pi0_fast",
        "model_id": config.policy.path,
        "revision": config.policy.revision,
        "model_compute_dtype": "fp16_residual",
        "training_loss_scale": 128.0,
    }

    def publish(role):
        container = tmp_path / role

        def write(staging):
            policy = staging / "policy" if role == "sft_baseline" else staging
            policy.mkdir(exist_ok=True)
            (policy / "art_embodied_pi0_fast_snapshot.json").write_text(
                json.dumps(snapshot)
            )
            (policy / "adapter_model.safetensors").write_bytes(
                b"fake policy; never loaded"
            )

        CheckpointManager().publish(
            container,
            writer=write,
            config_fingerprint="test",
            resume_contract_fingerprint="test",
            metadata={"update_step": 100},
        )
        return container / "policy" if role == "sft_baseline" else container

    args = SimpleNamespace(
        config=config, baseline=publish("sft_baseline"), candidate=publish("candidate")
    )
    monkeypatch.setattr(
        entry.EmbodiedExperimentConfig, "from_yaml", lambda _: args.config
    )
    monkeypatch.setattr(entry, "check_runtime", Mock(return_value={"mujoco": "3.3.0"}))
    monkeypatch.setattr(entry, "require_compatible_runtime", Mock())
    monkeypatch.setattr(entry, "validate_runtime_device_availability", Mock())
    monkeypatch.setattr(
        entry,
        "_integration",
        lambda _: (
            Mock(),
            SimpleNamespace(from_config=Mock()),
            Mock(return_value={}),
            Mock(return_value=["scenario"]),
            Mock(),
            Mock(),
        ),
    )
    observer = SimpleNamespace(log_progress=AsyncMock(), close=Mock())
    args.observer = observer
    monkeypatch.setattr(entry.WandbWeaveObserver, "start", Mock(return_value=observer))
    monkeypatch.setattr(entry, "make_policy", Mock(return_value=object()))
    monkeypatch.setattr(entry, "_load_policy_checkpoint", Mock())
    monkeypatch.setattr(entry, "run_lerobot_evaluation", AsyncMock())
    args.contexts = []

    @contextmanager
    def context():
        args.contexts.append("enter")
        try:
            yield
        finally:
            args.contexts.append("exit")

    monkeypatch.setattr(entry, "isolated_art_runtime", context)
    monkeypatch.setattr(entry, "diagnostic_cuda_mapping", context)
    return args


def test_preflight_has_no_model_gpu_observer_or_evaluation(setup, capsys):
    asyncio.run(
        entry.run(
            Path("unused"),
            checkpoint=setup.baseline,
            step=0,
            role="sft_baseline",
            preflight=True,
        )
    )
    report = json.loads(capsys.readouterr().out)
    assert report["preflight_only"] and not report["sealed_policy_evaluated"]
    assert report["runtime"]["mujoco"] == "3.3.0"
    entry.check_runtime.assert_called_once_with()
    entry.require_compatible_runtime.assert_called_once_with(profile="control")
    entry.make_policy.assert_not_called()
    entry.validate_runtime_device_availability.assert_not_called()
    entry.WandbWeaveObserver.start.assert_not_called()
    entry.run_lerobot_evaluation.assert_not_called()


@pytest.mark.parametrize("role,step", [("sft_baseline", 0), ("candidate", 100)])
def test_evaluation_loads_explicit_policy_and_native_step(setup, role, step):
    checkpoint = setup.baseline if step == 0 else setup.candidate
    asyncio.run(entry.run(Path("unused"), checkpoint=checkpoint, step=step, role=role))
    entry._load_policy_checkpoint.assert_called_once_with(
        entry.make_policy.return_value, checkpoint
    )
    kwargs = entry.run_lerobot_evaluation.await_args.kwargs
    assert kwargs["use_native_wandb_step"] is True
    assert kwargs["step"] == step and kwargs["checkpoint_role"] == role
    assert kwargs["checkpoint_path"] == checkpoint
    assert all(
        call.args[0].update == step
        for call in setup.observer.log_progress.await_args_list
    )
    assert setup.contexts == ["enter", "enter", "exit", "exit"]
    setup.observer.close.assert_called_once_with(exit_code=0)


@pytest.mark.parametrize(
    "step,role", [(1, "sft_baseline"), (0, "candidate"), (101, "candidate")]
)
def test_inconsistent_steps_rejected_before_runtime_or_run(setup, step, role):
    with pytest.raises(ValueError):
        asyncio.run(
            entry.run(Path("unused"), checkpoint=setup.candidate, step=step, role=role)
        )
    entry.check_runtime.assert_not_called()
    entry.WandbWeaveObserver.start.assert_not_called()


def test_modified_snapshot_is_rejected(setup):
    (setup.baseline / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        entry.validate_inputs(setup.config, setup.baseline, 0, "sft_baseline")


def test_precision_mismatch_is_rejected(setup):
    path = setup.baseline / "art_embodied_pi0_fast_snapshot.json"
    metadata = json.loads(path.read_text())
    metadata["model_compute_dtype"] = "float32"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="precision"):
        entry.validate_inputs(setup.config, setup.baseline, 0, "sft_baseline")


def test_runtime_mismatch_stops_before_any_evaluation(setup):
    entry.check_runtime.side_effect = ValueError("runtime mismatch")
    with pytest.raises(ValueError, match="runtime mismatch"):
        asyncio.run(
            entry.run(
                Path("unused"), checkpoint=setup.baseline, step=0, role="sft_baseline"
            )
        )
    entry.make_policy.assert_not_called()
    entry.WandbWeaveObserver.start.assert_not_called()


def test_evaluation_failure_restores_contexts_and_closes_failed_run(setup):
    entry.run_lerobot_evaluation.side_effect = RuntimeError("evaluation failed")
    with pytest.raises(RuntimeError, match="evaluation failed"):
        asyncio.run(
            entry.run(
                Path("unused"), checkpoint=setup.baseline, step=0, role="sft_baseline"
            )
        )
    assert setup.contexts == ["enter", "enter", "exit", "exit"]
    setup.observer.close.assert_called_once_with(exit_code=1)


@pytest.mark.parametrize("fail", [False, True])
def test_isolated_simulator_checks_exact_runtime_and_restores_public_validator(monkeypatch, fail):
    from examples.embodied.libero import environment

    original = environment._validate_libero_mujoco_abi
    checked = Mock()
    monkeypatch.setattr(entry, "check_runtime", checked)

    def imports():
        environment._validate_libero_mujoco_abi()
        if fail:
            raise RuntimeError("import failure")

    if fail:
        with pytest.raises(RuntimeError, match="import failure"):
            entry.validate_isolated_simulator(imports)
    else:
        entry.validate_isolated_simulator(imports)
    assert checked.call_count == 2
    assert environment._validate_libero_mujoco_abi is original


def test_simulator_import_failure_occurs_before_wandb_or_model(setup, monkeypatch):
    monkeypatch.setattr(entry, "validate_isolated_simulator", Mock(side_effect=RuntimeError("imports")))
    with pytest.raises(RuntimeError, match="imports"):
        asyncio.run(entry.run(Path("unused"), checkpoint=setup.baseline, step=0, role="sft_baseline"))
    entry.WandbWeaveObserver.start.assert_not_called()
    entry.make_policy.assert_not_called()
