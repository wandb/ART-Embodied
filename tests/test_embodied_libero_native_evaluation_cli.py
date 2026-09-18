from __future__ import annotations

import asyncio
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from art_embodied.config import EmbodiedExperimentConfig
from examples.embodied.libero import train_openvla_oft as entry


@pytest.fixture
def invocation(monkeypatch, tmp_path):
    config = EmbodiedExperimentConfig.from_yaml(
        Path("examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml")
    )
    raw = config.model_dump(mode="json")
    raw["observability"]["wandb"].update(
        connection="primary",
        resume=None,
        run_id=None,
        native_update_steps=False,
    )
    config = EmbodiedExperimentConfig.model_validate(raw)
    monkeypatch.setattr(entry, "require_compatible_runtime", Mock())
    monkeypatch.setattr(entry, "validate_runtime_device_availability", Mock())
    scenarios = [object()]
    monkeypatch.setattr(
        entry,
        "_integration",
        lambda _: (
            Mock(),
            SimpleNamespace(from_config=Mock(return_value=object())),
            Mock(return_value={}),
            Mock(return_value=scenarios),
            Mock(
                side_effect=AssertionError(
                    "Evaluation must not build training scenarios"
                )
            ),
            Mock(),
        ),
    )
    observer = SimpleNamespace(log_progress=AsyncMock(), close=Mock())
    start = Mock(return_value=observer)
    monkeypatch.setattr(entry.WandbWeaveObserver, "start", start)
    policy = SimpleNamespace(load_checkpoint=Mock())
    monkeypatch.setattr(entry, "make_policy", Mock(return_value=policy))
    evaluate = AsyncMock()
    monkeypatch.setattr(entry, "run_lerobot_evaluation", evaluate)
    checkpoint = tmp_path / "policy"
    checkpoint.mkdir()
    args = SimpleNamespace(
        config=config,
        observer=observer,
        start=start,
        policy=policy,
        evaluate=evaluate,
        checkpoint=checkpoint,
        scenarios=scenarios,
    )
    monkeypatch.setattr(
        entry.EmbodiedExperimentConfig, "from_yaml", lambda _: args.config
    )
    return args


@pytest.mark.parametrize(
    "step,role,native",
    [
        (0, "sft_baseline", True),
        (100, "candidate", True),
        (100, "candidate", False),
    ],
)
def test_evaluation_forwards_native_step_role_and_checkpoint(
    invocation, step, role, native
):
    args = invocation
    asyncio.run(
        entry.run(
            Path("unused.yaml"),
            evaluate_only=True,
            evaluation_step=step,
            policy_checkpoint=args.checkpoint,
            native_wandb_step=native,
            checkpoint_role=role,
        )
    )
    kwargs = args.evaluate.await_args.kwargs
    assert kwargs["step"] == step
    assert kwargs["use_native_wandb_step"] is native
    assert kwargs["checkpoint_role"] == role
    assert kwargs["checkpoint_path"] == args.checkpoint
    assert kwargs["evaluation_scenarios"] == args.scenarios
    args.policy.load_checkpoint.assert_called_once_with({"path": str(args.checkpoint)})
    args.observer.close.assert_called_once_with(exit_code=0)
    assert all(
        call.args[0].update == step
        for call in args.observer.log_progress.await_args_list
    )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"native_wandb_step": True}, "requires --evaluate-only"),
        ({"checkpoint_role": "sft_baseline"}, "requires --evaluate-only"),
        ({"checkpoint_role": "invalid"}, "Unsupported"),
        ({"evaluate_only": True, "evaluation_step": -1}, "cannot be negative"),
    ],
)
def test_invalid_evaluation_does_not_start_a_run(invocation, kwargs, match):
    with pytest.raises(ValueError, match=match):
        asyncio.run(entry.run(Path("unused.yaml"), **kwargs))
    invocation.start.assert_not_called()


@pytest.mark.parametrize(
    "connection,resume",
    [
        ("resume", "must"),
        ("resume", "allow"),
        ("resume", "auto"),
    ],
)
def test_native_evaluation_rejects_nonfresh_run(invocation, connection, resume):
    raw = invocation.config.model_dump(mode="json")
    raw["observability"]["wandb"].update(connection=connection, resume=resume)
    if connection != "primary":
        raw["observability"]["wandb"]["run_id"] = "existing"
    invocation.config = EmbodiedExperimentConfig.model_validate(raw)
    with pytest.raises(ValueError, match="fresh primary"):
        asyncio.run(
            entry.run(Path("unused.yaml"), evaluate_only=True, native_wandb_step=True)
        )
    invocation.start.assert_not_called()


def test_native_evaluation_rejects_contiguous_training_writer(invocation):
    raw = invocation.config.model_dump(mode="json")
    raw["observability"]["wandb"]["native_update_steps"] = True
    invocation.config = EmbodiedExperimentConfig.model_validate(raw)
    with pytest.raises(ValueError, match="contiguous training writer"):
        asyncio.run(
            entry.run(Path("unused.yaml"), evaluate_only=True, native_wandb_step=True)
        )
    invocation.start.assert_not_called()


def test_evaluation_error_closes_failed_run(invocation):
    invocation.evaluate.side_effect = RuntimeError("evaluation failed")
    with pytest.raises(RuntimeError, match="evaluation failed"):
        asyncio.run(
            entry.run(Path("unused.yaml"), evaluate_only=True, native_wandb_step=True)
        )
    invocation.observer.close.assert_called_once_with(exit_code=1)


def test_preflight_does_not_load_model_or_log(invocation):
    asyncio.run(
        entry.run(
            Path("unused.yaml"),
            evaluate_only=True,
            native_wandb_step=True,
            preflight=True,
        )
    )
    invocation.start.assert_not_called()
    entry.make_policy.assert_not_called()
    invocation.evaluate.assert_not_called()


def test_native_evaluation_cli_arguments(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--config",
            "recipe.yaml",
            "--evaluate-only",
            "--native-wandb-step",
            "--evaluation-step",
            "100",
            "--checkpoint-role",
            "candidate",
            "--policy-checkpoint",
            "checkpoint",
        ],
    )
    args = entry.parse_args()
    assert args.native_wandb_step is True
    assert args.evaluation_step == 100
    assert args.checkpoint_role == "candidate"
    assert args.policy_checkpoint == Path("checkpoint")


def test_legacy_cli_defaults_unchanged(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train", "--config", "recipe.yaml"])
    args = entry.parse_args()
    assert args.native_wandb_step is False
    assert args.checkpoint_role == "candidate"
    assert args.evaluate_only is False


def test_real_sdk_nonzero_evaluation_binds_media_to_same_native_step(tmp_path):
    if importlib.util.find_spec("wandb") is None:
        pytest.skip("Optional W&B SDK is not installed")
    import art_embodied.observability as observer_module

    script = r"""
import asyncio, importlib.util, json, sys
from pathlib import Path
from PIL import Image
import wandb
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb
from art_embodied.config import EmbodiedExperimentConfig
from art_embodied.experiment import EvaluationResult
from art_embodied.trajectories import EmbodiedTrajectory, MediaRef
source, output = map(Path, sys.argv[1:])
name = 'art_embodied._offline_native_eval_test'
spec = importlib.util.spec_from_file_location(name, source)
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
raw = EmbodiedExperimentConfig.from_yaml(Path(
    'examples/embodied/pi0_fast_spatial_native_score_sum_gate_20260906.yaml'
)).model_dump(mode='json')
raw['storage']['output_dir'] = str(output)
raw['observability']['wandb'].update(
    mode='offline', entity='offline-test', project='offline-test',
    connection='primary', run_id=None, resume=None, native_update_steps=False,
    log_evaluation_artifacts=False, log_evaluation_table=False,
    log_model_artifacts=False, save_code=False,
)
raw['observability']['weave']['enabled'] = False
raw['observability']['lookahead_preview']['enabled'] = False
raw['observability'].update(videos_per_evaluation=1, require_evaluation_video=True)
config = EmbodiedExperimentConfig.model_validate(raw)
video = output / 'fixture.gif'
frames = [Image.new('RGB', (32, 32), color) for color in ['black', 'white']]
frames[0].save(video, save_all=True, append_images=frames[1:], duration=100, loop=0)
trajectory = EmbodiedTrajectory(task='offline-fixture', reward=1.0,
    media=[MediaRef(uri=video.as_uri(), kind='video', mime_type='image/gif')])
observer = module.WandbWeaveObserver.start(config)
try:
    asyncio.run(observer.log_evaluation_checkpoint(100,
        EvaluationResult(step=100, metrics={'success_rate': 1.0}, artifacts={},
                         trajectories=(trajectory,)),
        config, checkpoint_role='candidate', use_native_wandb_step=True))
    assert observer.wandb_run.step == 101
finally:
    observer.close()
# Wait for the isolated SDK service to flush and close its transaction file.
wandb.teardown()
files = list(output.rglob('run-*.wandb'))
assert len(files) == 1
store = DataStore(); store.open_for_scan(str(files[0]))
rows = []
try:
    while (data := store.scan_data()) is not None:
        record = pb.Record(); record.ParseFromString(data)
        if record.HasField('history'):
            row = {}
            for item in record.history.item:
                keys = list(item.nested_key) or [item.key]
                target = row
                for key in keys[:-1]:
                    target = target.setdefault(key, {})
                target[keys[-1]] = json.loads(item.value_json)
            rows.append(row)
finally:
    store.close()
assert len(rows) == 1, rows
assert rows[0]['_step'] == rows[0]['experiment/update'] == 100
assert '_100_' in Path(rows[0]['media/simulation/eval/0']['path']).name, rows
"""
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("WANDB_", "WEAVE_"))
    }
    env.update(WANDB_MODE="offline", WANDB_SILENT="true")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(Path(observer_module.__file__).resolve()),
            str(tmp_path),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
