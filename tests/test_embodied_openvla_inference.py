from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from art_embodied.config import EmbodiedExperimentConfig
from art_embodied.policies.inference import OpenVLABatchedInferenceEngine
from art_embodied.policies.openvla import OpenVLAPolicy
from art_embodied.rollout_process import RolloutActorProcessContext
from art_embodied.trajectories import Action, Observation


class _FakeOpenVLAPolicy:
    def __init__(self) -> None:
        self.generation = None

    def set_generation(self, *, do_sample, temperature):
        self.generation = (do_sample, temperature)

    def act_batch(self, observations, contexts):
        return [
            Action(
                step=int(context["step"]),
                kind="token",
                raw={"tokens": [index]},
                decoded=[float(index)],
                logprobs={"token_logprobs": [-0.1]},
                metadata={"forward_tensor": torch.tensor([index], device="cpu")},
            )
            for index, context in enumerate(contexts)
        ]


class _StochasticFakeOpenVLAPolicy(_FakeOpenVLAPolicy):
    def act_batch(self, observations, contexts):
        return [
            Action(
                step=int(context["step"]),
                kind="token",
                raw={"tokens": [0]},
                decoded=[
                    random.random(),
                    float(np.random.rand()),
                    float(torch.rand(())),
                ],
                logprobs={"token_logprobs": [-0.1]},
            )
            for context in contexts
        ]


def _config(tmp_path: Path) -> EmbodiedExperimentConfig:
    source = (
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_action_token_smoke.yaml"
    )
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["rollout"]["workers"] = 2
    raw["runtime"].update(
        {
            "rollout_devices": ["cpu"],
            "training_devices": ["cuda:0"],
            "rollout_execution": {
                "mode": "local_process",
                "actor_factory": "application.rollout:create_actor",
                "actor_kwargs": {},
                "actors_per_device": 2,
                "group_batching": False,
                "lifecycle": "persistent",
                "policy_sync": "checkpoint",
                "inference_mode": "batched_server",
                "inference_factory": (
                    "art_embodied.policies.inference:"
                    "create_openvla_batched_inference_engine"
                ),
                "inference_replicas_per_device": 1,
                "inference_max_batch_size": 8,
                "inference_max_wait_ms": 2.0,
                "startup_timeout_seconds": 30,
                "request_timeout_seconds": 30,
            },
            "worker_handoff_dir": str(tmp_path / "handoff"),
        }
    )
    raw["storage"]["output_dir"] = str(tmp_path / "output")
    return EmbodiedExperimentConfig.model_validate(raw)


def test_openvla_batch_engine_loads_snapshot_and_returns_cpu_actions(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    snapshot = tmp_path / "adapter"
    snapshot.mkdir()
    loaded_configs = []
    fake_policy = _FakeOpenVLAPolicy()

    def policy_factory(worker_config):
        loaded_configs.append(worker_config)
        return fake_policy

    engine = OpenVLABatchedInferenceEngine(
        config=config,
        context=RolloutActorProcessContext(
            worker_index=0,
            configured_device="cpu",
            local_device="cpu",
        ),
        policy_factory=policy_factory,
    )
    engine.prepare_update(update=3, policy_snapshot=snapshot)
    results = engine.predict_batch(
        [
            {
                "observation": Observation(step=index, kind="state", value=[index]),
                "context": {"step": index, "scenario": {"task": "pick"}},
            }
            for index in range(2)
        ]
    )

    assert len(loaded_configs) == 1
    assert loaded_configs[0].policy.device == "cpu"
    assert loaded_configs[0].policy.load_kwargs["peft_adapter_path"] == str(snapshot)
    assert [item["native_action"] for item in results] == [[0.0], [1.0]]
    assert all(item["policy_update"] == 3 for item in results)
    assert all(item["batch_size"] == 2 for item in results)
    assert all(
        item["action"].metadata["forward_tensor"].device.type == "cpu"
        for item in results
    )
    assert engine.offload() == {"offloaded": True, "policy_loaded": False}


def test_openvla_policy_keeps_training_interface() -> None:
    assert callable(OpenVLAPolicy.train)
    assert callable(OpenVLAPolicy.eval)
    assert callable(OpenVLAPolicy.state_dict)


def test_openvla_batch_engine_preserves_independent_group_rng_streams(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    snapshot = tmp_path / "adapter"
    snapshot.mkdir()
    fake_policy = _StochasticFakeOpenVLAPolicy()
    engine = OpenVLABatchedInferenceEngine(
        config=config,
        context=RolloutActorProcessContext(
            worker_index=0,
            configured_device="cpu",
            local_device="cpu",
        ),
        policy_factory=lambda _config: fake_policy,
    )
    engine.prepare_update(update=3, policy_snapshot=snapshot)

    def request(stream: str, seed: int, step: int) -> dict:
        return {
            "op": "predict_group",
            "observations": [Observation(step=step, kind="state", value=[step])],
            "contexts": [{"step": step, "scenario": {"task": "pick"}}],
            "phase": "train",
            "rng_stream_id": stream,
            "rng_seed": seed,
            "reset_rng_stream": step == 0,
        }

    a0 = engine.predict_batch([request("a", 11, 0)])[0]["actions"][0].decoded
    b0 = engine.predict_batch([request("b", 22, 0)])[0]["actions"][0].decoded
    a1 = engine.predict_batch([request("a", 11, 1)])[0]["actions"][0].decoded

    _seed_all(11)
    expected_a0 = _draw_rng_triplet()
    expected_a1 = _draw_rng_triplet()
    _seed_all(22)
    expected_b0 = _draw_rng_triplet()
    np.testing.assert_allclose(a0, expected_a0)
    np.testing.assert_allclose(a1, expected_a1)
    np.testing.assert_allclose(b0, expected_b0)
    assert fake_policy.generation == (True, 1.6)

    released = engine.predict_batch(
        [{"op": "release_rng_stream", "rng_stream_id": "a"}]
    )[0]
    assert released["released"] is True


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _draw_rng_triplet() -> list[float]:
    return [random.random(), float(np.random.rand()), float(torch.rand(()))]
