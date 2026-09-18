"""Exercise real W&B serialization offline, without publishing synthetic metrics."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_real_sdk_keeps_all_policy_versions_and_custom_axes(tmp_path):
    if importlib.util.find_spec("wandb") is None:
        pytest.skip("Optional W&B SDK is not installed")
    root = Path(__file__).resolve().parents[1]
    script = r"""
import asyncio
import json
from pathlib import Path
import sys

import wandb
from art_embodied import EmbodiedExperimentConfig
from art_embodied.experiment import EvaluationResult
from art_embodied.observability import WandbWeaveObserver
from art_embodied.trajectories import EmbodiedTrajectory, EmbodiedTrajectoryGroup
from art_embodied.types import LocalTrainResult
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

root, output = map(Path, sys.argv[1:])
config = EmbodiedExperimentConfig.from_yaml(
    root / 'examples/embodied/pi0_fast_spatial_native_score_sum_gate_20260906.yaml'
).model_dump(mode='json')
config['storage']['output_dir'] = str(output)
config['observability'].update(
    require_train_video=False, require_evaluation_video=False,
    videos_per_update=0, videos_per_evaluation=0,
)
config['observability']['weave']['enabled'] = False
config['observability']['wandb'].update(
    mode='offline', entity='offline-test', project='offline-test',
    log_evaluation_artifacts=False, log_evaluation_table=False,
    log_model_artifacts=False, save_code=False,
)
config = EmbodiedExperimentConfig.model_validate(config)
observer = WandbWeaveObserver.start(config)

async def emit():
    await observer.log_initial_evaluation(
        EvaluationResult(step=0, metrics={'success_rate':0.3}, artifacts={}), config,
    )
    for version in range(4):
        trajectories = [
            EmbodiedTrajectory(task='offline-test', reward=float(i < version),
                               metrics={'success': i < version})
            for i in range(4)
        ]
        groups = [EmbodiedTrajectoryGroup(trajectories)]
        await observer.log_rollout(version, groups, config)
        step = version + 1
        await observer.log_step(
            step, groups, LocalTrainResult(step=step, metrics={'loss':0.1}),
            EvaluationResult(step=step, metrics={'success_rate':0.4}, artifacts={})
            if step == 1 else None,
            config,
        )

try:
    asyncio.run(emit())
    # Exercise real SDK config/artifact serialization locally. Offline mode
    # cannot test server-side resume; the recovery lifecycle has separate tests.
    continuation = config.model_dump(mode='json')
    continuation['training']['updates'] = 100
    continuation = EmbodiedExperimentConfig.model_validate(continuation)
    recovered = WandbWeaveObserver(
        continuation, wandb_run=observer.wandb_run,
        wandb_module=observer._wandb_module,
    )
    recovered._record_resume_config(2)
    assert observer.wandb_run.config['training']['updates'] == 100
finally:
    observer.close()

# Wait for the isolated SDK service to flush and close its transaction file.
wandb.teardown()
files = list(output.rglob('run-*.wandb'))
assert len(files) == 1, files
store = DataStore()
store.open_for_scan(str(files[0]))
history, metrics, configurations, artifacts = [], [], [], []
try:
    while (data := store.scan_data()) is not None:
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if record.HasField('history'):
            history.append({item.key or '.'.join(item.nested_key):
                            json.loads(item.value_json) for item in record.history.item})
        if record.HasField('metric'):
            metrics.append({'name':record.metric.name,
                            'glob_name':record.metric.glob_name,
                            'step_metric':record.metric.step_metric})
        if record.HasField('config'):
            configurations.extend(
                {'key':item.key or '.'.join(item.nested_key),
                 'value':json.loads(item.value_json)} for item in record.config.update
            )
        if record.HasField('run'):
            configurations.extend(
                {'key':item.key or '.'.join(item.nested_key),
                 'value':json.loads(item.value_json)} for item in record.run.config.update
            )
        if record.HasField('artifact'):
            artifacts.append({'name':record.artifact.name, 'type':record.artifact.type})
finally:
    store.close()
(output / 'serialized.json').write_text(json.dumps(
    {'history':history, 'metrics':metrics, 'configurations':configurations,
     'artifacts':artifacts}
))
"""
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("WANDB_", "WEAVE_"))
    }
    env.update(WANDB_MODE="offline", WANDB_SILENT="true", PYTHONPATH=str(root / "src"))
    subprocess.run(
        [sys.executable, "-c", script, str(root), str(tmp_path)],
        env=env,
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    result = json.loads((tmp_path / "serialized.json").read_text())
    training_configs = [
        item["value"]["updates"]
        for item in result["configurations"]
        if item["key"] == "training"
    ]
    assert training_configs[0] == 1
    assert training_configs[-1] == 100
    assert any(a["type"] == "experiment-config" for a in result["artifacts"])
    records = list((tmp_path / "wandb/configuration-history").glob("*.json"))
    assert len(records) == 1
    configuration_record = json.loads(records[0].read_text())
    assert configuration_record["previous"]["training"]["updates"] == 1
    assert configuration_record["current"]["training"]["updates"] == 100
    rows = result["history"]
    train = [row for row in rows if "train/success_rate" in row]
    assert [row["experiment/update"] for row in train] == [0, 1, 2, 3]
    assert [row["train/success_rate"] for row in train] == [0, 0.25, 0.5, 0.75]
    assert [row["_step"] for row in train] == [1, 3, 5, 7]
    assert [
        row["experiment/update"] for row in rows if "validation/success_rate" in row
    ] == [0, 1]
    for name in ("train/*", "validation/*", "optimization/*", "performance/*"):
        definitions = [m for m in result["metrics"] if m["glob_name"] == name]
        assert definitions
        assert all(m["step_metric"] == "experiment/update" for m in definitions)
