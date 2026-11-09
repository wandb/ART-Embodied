from __future__ import annotations

import copy
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from art_embodied.policies.flow_sde import FlowSDESchedule
from art_embodied.policies.openpi_flow_sde import (
    OpenPIBatchedSampler,
    OpenPIFlowSDEBridge,
    load_openpi_pytorch,
)


def tree_map(function, *values):
    if isinstance(values[0], dict):
        return {
            key: tree_map(function, *(value[key] for value in values))
            for key in values[0]
        }
    return function(*values)


@pytest.fixture(autouse=True)
def optional_processors(monkeypatch):
    # Exercise the boundary without installing OpenPI/JAX or downloading weights.
    jax = ModuleType("jax")
    jax.tree = SimpleNamespace(map=tree_map)
    monkeypatch.setitem(sys.modules, "jax", jax)
    observation = ModuleType("openpi.models.model")
    observation.Observation = SimpleNamespace(from_dict=lambda batch: batch)
    monkeypatch.setitem(sys.modules, "openpi.models.model", observation)


class NativeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))
        self.config = SimpleNamespace(action_horizon=3, action_dim=4)
        self.calls = []

    def denoise_step(self):
        raise AssertionError("Test bridge supplies its velocity field")

    def _preprocess_observation(self, batch, *, train):
        assert not train
        return (
            [batch["image"]],
            [batch["mask"]],
            batch["tokens"],
            batch["mask"],
            batch["state"],
        )

    def sample_actions(self, device, observation, *, noise, num_steps):
        self.calls.append((device, num_steps, noise.clone()))
        return noise + observation["state"][:, None, :]


class NativePolicy:
    def __init__(self):
        self._model = NativeModel()
        self.outputs = []

    def _input_transform(self, request):
        request["state"][0] += 1
        return {
            "state": request["state"],
            "tokens": np.array([1, 2]),
            "mask": np.array([True, True]),
            "image": np.zeros((2, 2, 3)),
        }

    def _output_transform(self, output):
        self.outputs.append(copy.deepcopy(output))
        assert output["actions"].shape == (3, 4)
        # Emulate per-coordinate unnormalization followed by DROID's crop.
        return {"actions": output["actions"][:, :2] * 2 + output["state"][:2]}


class ToyBridge(OpenPIFlowSDEBridge):
    def _prefix_cache(self, inputs):
        return None, None

    def _velocity(self, *, inputs, state, timestep, **_):
        return state * self.model.weight + inputs.state[:, None, :] * 0.01


def bridge():
    return ToyBridge(
        NativePolicy(),
        schedule=FlowSDESchedule(4, 0.5, deterministic_sampler="native_euler"),
        execution_horizon=2,
        action_dim=2,
    )


def requests():
    return [
        {"sampling_seed": seed, "state": np.full(4, seed / 100, dtype=np.float32)}
        for seed in (17, 23)
    ]


def test_transform_is_native_and_does_not_mutate_inputs():
    value = bridge()
    request = requests()[0]
    before = request["state"].copy()
    transformed = value.transform(request)
    np.testing.assert_array_equal(request["state"], before)
    assert transformed["state"].shape == (1, 4)
    assert transformed["state"][0, 0].item() == pytest.approx(before[0] + 1)


def test_sampler_preserves_fixed_shape_and_does_not_mutate_requests():
    sampler = OpenPIBatchedSampler(bridge(), batch_size=4)
    original = requests()
    batch, noise, seeds = sampler.prepare(original)
    assert batch["state"].shape == (4, 4)
    assert noise.shape == (4, 3, 4)
    assert seeds == [17, 23, 23, 23]
    assert [request["sampling_seed"] for request in original] == [17, 23]
    np.testing.assert_array_equal(
        noise[0].numpy(),
        np.random.default_rng(17).normal(size=(3, 4)).astype(np.float32),
    )
    torch.testing.assert_close(noise[1], noise[3], rtol=0, atol=0)


def test_ode_uses_native_sampler_and_explicit_schedule():
    value = bridge()
    sampler = OpenPIBatchedSampler(value, batch_size=4)
    actions, record = sampler.infer(requests(), mode="ode")
    assert actions.shape == (2, 3, 2)
    assert record is None
    assert value.model.calls[0][:2] == (torch.device("cpu"), 4)
    assert len(value.native.outputs) == 2


def test_sde_rescore_and_gradients_match_recorded_transition():
    value = bridge()
    sampler = OpenPIBatchedSampler(value, batch_size=4)
    actions, record = sampler.infer(requests(), mode="sde")
    assert actions.shape == (2, 3, 2)
    assert record.inputs.batch_size == 4
    assert record.transition.old_logprobs.shape == (4, 3, 4)
    assert (record.transition.selected_indices < 3).all()
    scores = value.rescore(record)
    torch.testing.assert_close(
        scores, record.transition.old_logprobs[:, :2, :2], rtol=0, atol=0
    )
    (-scores.mean()).backward()
    assert value.model.weight.grad is not None
    assert torch.isfinite(value.model.weight.grad)
    assert value.model.weight.grad != 0
    assert (value.native.outputs[0]["actions"][:, 2:] == 0).all()


@pytest.mark.parametrize("mode", ["ode", "sde"])
def test_rng_follows_request_not_batch_position_or_padding(mode):
    sampler = OpenPIBatchedSampler(bridge(), batch_size=4)
    original = requests()
    first, _ = sampler.infer(original, mode=mode)
    reordered, _ = sampler.infer(list(reversed(original)), mode=mode)
    single, _ = sampler.infer([original[0]], mode=mode)
    np.testing.assert_array_equal(first, reordered[::-1])
    np.testing.assert_array_equal(first[:1], single)
    repeated, _ = sampler.infer(original, mode=mode)
    np.testing.assert_array_equal(first, repeated)


@pytest.mark.parametrize("seed", [-1, 2**63, 1.5, True, "2"])
def test_seed_validation(seed):
    sampler = OpenPIBatchedSampler(bridge(), batch_size=4)
    request = requests()[0]
    request["sampling_seed"] = seed
    with pytest.raises(ValueError, match="sampling_seed"):
        sampler.prepare([request])


def test_reject_bad_geometry_mode_and_empty_batches():
    value = bridge()
    with pytest.raises(ValueError, match="native_euler"):
        OpenPIFlowSDEBridge(
            value.native,
            schedule=FlowSDESchedule(4, 0.5),
            execution_horizon=2,
            action_dim=2,
        )
    with pytest.raises(ValueError, match="batch_size"):
        OpenPIBatchedSampler(value, batch_size=0)
    sampler = OpenPIBatchedSampler(value, batch_size=1)
    with pytest.raises(ValueError, match="Request count"):
        sampler.prepare([])
    with pytest.raises(ValueError, match="Request count"):
        sampler.prepare(requests())
    with pytest.raises(ValueError, match="mode"):
        sampler.infer(requests()[:1], mode="unknown")


def test_postprocessing_rejects_batched_nonfinite_and_wrong_horizon():
    value = bridge()
    sampler = OpenPIBatchedSampler(value, batch_size=2)
    batch, noise, _ = sampler.prepare(requests())
    inputs = value.prepare_inputs(batch)
    for actions in (noise, noise[:1, :2], torch.full((1, 3, 2), float("nan"))):
        with pytest.raises(ValueError, match="full-horizon"):
            value.postprocess(inputs.select(0), actions)


def test_loader_fails_before_import_without_converted_weights(tmp_path):
    with pytest.raises(ValueError, match="model.safetensors"):
        load_openpi_pytorch(tmp_path, config_name="pi05_droid_jointpos_polaris")
