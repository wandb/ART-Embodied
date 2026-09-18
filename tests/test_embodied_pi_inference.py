from __future__ import annotations

from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import yaml

torch = pytest.importorskip("torch")

from art_embodied.config import EmbodiedExperimentConfig  # noqa: E402
from art_embodied.policies.pi_inference import (  # noqa: E402
    PIBatchedInferenceEngine,
)
from art_embodied.rollout_process import RolloutActorProcessContext  # noqa: E402
from art_embodied.trajectories import Action  # noqa: E402


def _config(tmp_path: Path) -> EmbodiedExperimentConfig:
    source = (
        Path(__file__).parents[1]
        / "examples/embodied/pi05_libero_long_flow_sde_grpo_rlinf_positive_control.yaml"
    )
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    execution = raw["runtime"]["rollout_execution"]
    execution.update(
        {
            "actors_per_device": 2,
            "lifecycle": "cpu_offload",
            "inference_mode": "batched_server",
            "inference_factory": (
                "art_embodied.policies.pi_inference:create_pi_batched_inference_engine"
            ),
        }
    )
    raw["runtime"].update(
        {
            "rollout_devices": ["cpu"],
            "training_devices": ["cuda:0"],
            "distributed_training": False,
            "worker_handoff_dir": str(tmp_path / "handoff"),
        }
    )
    raw["rollout"]["workers"] = 2
    raw["storage"]["output_dir"] = str(tmp_path / "output")
    return EmbodiedExperimentConfig.model_validate(raw)


class _FakePolicy:
    family = "pi05"

    def __init__(self) -> None:
        self.loaded: list[str] = []
        self.devices: list[str] = []
        self.eval_calls = 0
        self.reset_calls = 0

    def load_checkpoint(self, checkpoint) -> None:
        self.loaded.append(str(checkpoint["path"]))

    def eval(self) -> None:
        self.eval_calls += 1

    def to(self, device: str) -> None:
        self.devices.append(str(device))

    def reset(self) -> None:
        self.reset_calls += 1


class _FakeAdapter:
    def __init__(self, *, policy, robot_type, sampling_mode) -> None:
        del policy, robot_type
        self.sampling_mode = sampling_mode

    def predict_batch(self, observations, *, tasks, step):
        del observations, tasks
        values = [random.random() for _ in range(2)]
        return [
            SimpleNamespace(
                native_action=torch.tensor([[value]], dtype=torch.float32),
                action=Action(
                    step=step,
                    kind="continuous",
                    raw=[value],
                    decoded=[value],
                    logprobs=None if self.sampling_mode == "eval" else {},
                ),
            )
            for value in values
        ]


def _request(stream: str, seed: int, step: int, *, phase: str = "train") -> dict:
    return {
        "op": "predict_group",
        "raw_observations": [{"state": step}, {"state": step}],
        "contexts": [
            {"task": "pick", "step": step},
            {"task": "pick", "step": step},
        ],
        "phase": phase,
        "rng_stream_id": stream,
        "rng_seed": seed,
        "reset_rng_stream": step == 0,
    }


def test_pi_batch_engine_reuses_policy_and_isolates_rng_streams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "art_embodied.policies.pi_inference.PIFlowSDEPolicyAdapter",
        _FakeAdapter,
    )
    policy = _FakePolicy()
    loaded_configs = []
    engine = PIBatchedInferenceEngine(
        config=_config(tmp_path),
        context=RolloutActorProcessContext(0, "cpu", "cpu"),
        policy_factory=lambda config: loaded_configs.append(config) or policy,
    )
    snapshot = tmp_path / "snapshot"
    engine.prepare_update(update=1, policy_snapshot=snapshot)

    a0 = engine.predict_batch([_request("a", 11, 0)])[0]
    engine.predict_batch([_request("b", 22, 0)])
    a1 = engine.predict_batch([_request("a", 11, 1)])[0]

    expected = random.Random(11)
    assert [item.decoded[0] for item in a0["actions"]] == [
        expected.random(),
        expected.random(),
    ]
    assert [item.decoded[0] for item in a1["actions"]] == [
        expected.random(),
        expected.random(),
    ]
    assert all(item.device.type == "cpu" for item in a0["native_actions"])
    assert a0["model_batch_size"] == 2
    assert a0["server_request_batch_size"] == 1
    assert len(loaded_configs) == 1
    assert policy.loaded == [str(snapshot)]
    assert policy.reset_calls == 2

    assert engine.offload() == {"offloaded": True, "policy_loaded": True}
    engine.prepare_update(update=2, policy_snapshot=snapshot)
    assert len(loaded_configs) == 1
    assert policy.loaded == [str(snapshot), str(snapshot)]
    assert policy.devices == ["cpu", "cpu"]
    assert policy.eval_calls == 2


def test_pi_batch_engine_releases_rng_stream(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "art_embodied.policies.pi_inference.PIFlowSDEPolicyAdapter",
        _FakeAdapter,
    )
    engine = PIBatchedInferenceEngine(
        config=_config(tmp_path),
        context=RolloutActorProcessContext(0, "cpu", "cpu"),
        policy_factory=lambda _config: _FakePolicy(),
    )
    engine.prepare_update(update=1, policy_snapshot=tmp_path / "snapshot")
    engine.predict_batch([_request("a", 11, 0)])

    result = engine.predict_batch([{"op": "release_rng_stream", "rng_stream_id": "a"}])[
        0
    ]

    assert result == {"released": True, "policy_update": 1}


def test_pi_batch_engine_isolates_native_eval_rng_streams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "art_embodied.policies.pi_inference.PIFlowSDEPolicyAdapter",
        _FakeAdapter,
    )
    policy = _FakePolicy()
    engine = PIBatchedInferenceEngine(
        config=_config(tmp_path),
        context=RolloutActorProcessContext(0, "cpu", "cpu"),
        policy_factory=lambda _config: policy,
    )
    engine.prepare_update(update=0, policy_snapshot=tmp_path / "snapshot")

    a0 = engine.predict_batch([_request("eval-a", 37, 0, phase="eval")])[0]
    engine.predict_batch([_request("eval-b", 91, 0, phase="eval")])
    a1 = engine.predict_batch([_request("eval-a", 37, 1, phase="eval")])[0]

    expected = random.Random(37)
    assert [item.decoded[0] for item in a0["actions"]] == [
        expected.random(),
        expected.random(),
    ]
    assert [item.decoded[0] for item in a1["actions"]] == [
        expected.random(),
        expected.random(),
    ]
    assert all(item.logprobs is None for item in a0["actions"])
    assert policy.reset_calls == 2
