from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from art_embodied.policies.flow_sde import (  # noqa: E402
    FlowSDESchedule,
    FlowSDETransitionRecord,
)
from art_embodied.policies.pi import (  # noqa: E402
    PI_TOKENIZER_ID,
    PIFlowPolicy,
    _configure_rlinf_openpi_contract,
    _load_rlinf_openpi_normalization_stats,
    _load_rlinf_openpi_weights,
    _preload_pi_tokenizer,
)
from art_embodied.policies.pi_flow_sde import (  # noqa: E402
    PIFlowModelInputs,
    PIFlowSDEBridge,
    PIFlowSDERollout,
)


def test_pi_tokenizer_preflight_populates_shared_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformers import AutoTokenizer

    requested: list[str] = []
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda model_id: requested.append(model_id),
    )

    _preload_pi_tokenizer()

    assert requested == [PI_TOKENIZER_ID]


def test_pi_tokenizer_preflight_explains_gated_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformers import AutoTokenizer

    def deny(_model_id: str) -> None:
        raise OSError("access denied")

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", deny)

    with pytest.raises(RuntimeError, match="hf auth login") as error:
        _preload_pi_tokenizer()

    assert PI_TOKENIZER_ID in str(error.value)
    assert "before policy weights" in str(error.value)


def make_att_2d_masks(pad_masks, attention_masks):
    del attention_masks
    batch, length = pad_masks.shape
    return torch.ones(batch, length, length, dtype=torch.bool, device=pad_masks.device)


class _FakeJointModel:
    def __init__(self) -> None:
        language_model = SimpleNamespace(config=SimpleNamespace())
        self.paligemma = SimpleNamespace(
            model=SimpleNamespace(language_model=language_model)
        )

    def forward(self, **kwargs):
        del kwargs
        return (None, object())


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.2))
        self.paligemma_with_expert = _FakeJointModel()

    def sample_noise(self, shape, device):
        return torch.zeros(shape, device=device)

    def embed_prefix(self, images, image_masks, language_tokens, language_masks):
        del images, image_masks, language_masks
        batch, length = language_tokens.shape
        embeddings = torch.zeros(batch, length, 2, device=language_tokens.device)
        masks = torch.ones(
            batch, length, dtype=torch.bool, device=language_tokens.device
        )
        attention = torch.zeros_like(masks)
        return embeddings, masks, attention

    def _prepare_attention_masks_4d(self, masks):
        return masks

    def denoise_step(self, *, x_t, timestep, **kwargs):
        del kwargs
        return x_t * self.weight + timestep[:, None, None]


class _FakePolicy:
    def __init__(self) -> None:
        self.model = _FakeModel()
        self.config = SimpleNamespace(chunk_size=3, max_action_dim=2)

    def _preprocess_images(self, batch):
        image = batch["observation.images.front"]
        mask = torch.ones(image.shape[0], dtype=torch.bool, device=image.device)
        return [image], [mask]


def test_pi_flow_bridge_samples_compact_record_and_rescores_exactly() -> None:
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )

    policy = _FakePolicy()
    bridge = PIFlowSDEBridge(
        policy,
        schedule=FlowSDESchedule(num_steps=3, noise_level=0.3),
        execution_horizon=2,
        action_dim=2,
    )
    batch = {
        "observation.images.front": torch.zeros(2, 3, 8, 8),
        OBS_LANGUAGE_TOKENS: torch.ones(2, 4, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 4, dtype=torch.bool),
    }
    generator = torch.Generator().manual_seed(17)

    rollout = bridge.sample(
        batch,
        selected_index=1,
        generator=generator,
    )
    rescored = bridge.rescore(rollout)

    # The rollout retains the complete model horizon. The LeRobot adapter
    # postprocesses it before selecting the two actions executed by the env.
    assert rollout.actions.shape == (2, 3, 2)
    assert rollout.transition.previous_states.shape == (2, 3, 2)
    torch.testing.assert_close(
        rescored,
        rollout.transition.old_logprobs[:, :2, :2],
        rtol=0,
        atol=0,
    )
    (-rescored.mean()).backward()
    assert policy.model.weight.grad is not None
    assert torch.isfinite(policy.model.weight.grad)


def test_rlinf_raw_weight_loader_strictly_loads_model_surface(tmp_path) -> None:
    from safetensors.torch import save_file

    class NativePolicy:
        def __init__(self) -> None:
            self.model = torch.nn.Linear(2, 1)
            self.config = SimpleNamespace(device="cpu")

    checkpoint = tmp_path / "model.safetensors"
    expected = {
        "weight": torch.tensor([[1.0, -2.0]]),
        "bias": torch.tensor([0.5]),
    }
    save_file(expected, checkpoint)
    policy = NativePolicy()

    _load_rlinf_openpi_weights(
        policy,
        model_id=str(checkpoint),
        revision=None,
        strict=True,
    )

    torch.testing.assert_close(policy.model.weight, expected["weight"])
    torch.testing.assert_close(policy.model.bias, expected["bias"])


def test_rlinf_raw_weight_loader_supports_validated_sharded_checkpoint(
    tmp_path,
) -> None:
    import json

    from safetensors.torch import save_file

    class NativePolicy:
        def __init__(self) -> None:
            self.model = torch.nn.Linear(2, 1)
            self.config = SimpleNamespace(device="cpu")

    save_file({"weight": torch.tensor([[1.0, -2.0]])}, tmp_path / "part-1.safetensors")
    save_file({"bias": torch.tensor([0.5])}, tmp_path / "part-2.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "weight": "part-1.safetensors",
                    "bias": "part-2.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    policy = NativePolicy()

    _load_rlinf_openpi_weights(
        policy,
        model_id=str(tmp_path),
        revision=None,
        strict=True,
    )

    torch.testing.assert_close(policy.model.weight, torch.tensor([[1.0, -2.0]]))
    torch.testing.assert_close(policy.model.bias, torch.tensor([0.5]))


def test_rlinf_raw_weight_loader_rejects_incorrect_shard_index(tmp_path) -> None:
    import json

    from safetensors.torch import save_file

    save_file({"weight": torch.tensor([[1.0, -2.0]])}, tmp_path / "part.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "weight": "part.safetensors",
                    "missing": "part.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    policy = SimpleNamespace(
        model=torch.nn.Linear(2, 1),
        config=SimpleNamespace(device="cpu"),
    )

    with pytest.raises(RuntimeError, match="index does not match"):
        _load_rlinf_openpi_weights(
            policy,
            model_id=str(tmp_path),
            revision=None,
            strict=True,
        )


def test_rlinf_raw_loader_copies_lm_head_into_missing_embedding(
    tmp_path, monkeypatch
) -> None:
    embedding = torch.nn.Parameter(torch.zeros(2, 2))
    lm_head = torch.nn.Parameter(torch.full((2, 2), 3.0))
    paligemma = SimpleNamespace(
        model=SimpleNamespace(
            language_model=SimpleNamespace(
                embed_tokens=SimpleNamespace(weight=embedding)
            )
        ),
        lm_head=SimpleNamespace(weight=lm_head),
    )
    policy = SimpleNamespace(
        model=SimpleNamespace(
            paligemma_with_expert=SimpleNamespace(paligemma=paligemma)
        ),
        config=SimpleNamespace(device="cpu"),
    )
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    monkeypatch.setattr(
        "art_embodied.policies.pi._read_safetensors_dtypes",
        lambda _checkpoints: {},
    )
    monkeypatch.setattr(
        "safetensors.torch.load_model",
        lambda *_args, **_kwargs: (
            [
                "paligemma_with_expert.paligemma.model.language_model."
                "embed_tokens.weight"
            ],
            [],
        ),
    )

    _load_rlinf_openpi_weights(
        policy,
        model_id=str(checkpoint),
        revision=None,
        strict=True,
    )
    torch.testing.assert_close(embedding, lm_head)


def test_rlinf_raw_loader_accepts_policy_checkpoint_with_value_head(
    tmp_path, monkeypatch
) -> None:
    policy = SimpleNamespace(
        model=torch.nn.Linear(2, 1),
        config=SimpleNamespace(device="cpu"),
    )
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    monkeypatch.setattr(
        "art_embodied.policies.pi._read_safetensors_dtypes",
        lambda _checkpoints: {},
    )
    monkeypatch.setattr(
        "safetensors.torch.load_model",
        lambda *_args, **_kwargs: (
            [],
            ["value_head.mlp.0.weight", "value_head.mlp.0.bias"],
        ),
    )

    _load_rlinf_openpi_weights(
        policy,
        model_id=str(checkpoint),
        revision=None,
        strict=True,
    )


def test_rlinf_raw_loader_rejects_unknown_auxiliary_checkpoint_keys(
    tmp_path, monkeypatch
) -> None:
    policy = SimpleNamespace(
        model=torch.nn.Linear(2, 1),
        config=SimpleNamespace(device="cpu"),
    )
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    monkeypatch.setattr(
        "art_embodied.policies.pi._read_safetensors_dtypes",
        lambda _checkpoints: {},
    )
    monkeypatch.setattr(
        "safetensors.torch.load_model",
        lambda *_args, **_kwargs: ([], ["unknown_head.weight"]),
    )

    with pytest.raises(RuntimeError, match="unknown_head.weight"):
        _load_rlinf_openpi_weights(
            policy,
            model_id=str(checkpoint),
            revision=None,
            strict=True,
        )


def test_rlinf_pi05_contract_overrides_serialized_lerobot_defaults() -> None:
    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.utils.constants import ACTION, OBS_STATE

    config = SimpleNamespace(
        chunk_size=50,
        n_action_steps=50,
        num_inference_steps=10,
        compile_model=True,
        max_state_dim=32,
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )

    _configure_rlinf_openpi_contract(
        config,
        family="pi05",
        execution_horizon=5,
        action_dim=7,
        model_chunk_size=10,
        num_inference_steps=5,
        extra_delta_transform=False,
    )

    assert config.chunk_size == 10
    assert config.n_action_steps == 5
    assert config.num_inference_steps == 5
    assert config.compile_model is True
    assert config.input_features[OBS_STATE].shape == (32,)
    assert config.output_features[ACTION].shape == (7,)
    assert config.normalization_mapping["STATE"] == NormalizationMode.QUANTILES


def test_rlinf_pi0_contract_preserves_four_step_sampler() -> None:
    config = SimpleNamespace(
        chunk_size=50,
        n_action_steps=50,
        num_inference_steps=10,
        compile_model=True,
    )

    _configure_rlinf_openpi_contract(
        config,
        family="pi0",
        execution_horizon=5,
        action_dim=7,
        model_chunk_size=10,
        num_inference_steps=4,
        extra_delta_transform=True,
    )

    assert config.chunk_size == 10
    assert config.n_action_steps == 5
    assert config.num_inference_steps == 4
    assert config.compile_model is True
    assert config.use_relative_actions is True
    assert config.relative_exclude_joints == ["gripper"]
    assert config.action_feature_names[-1] == "gripper"
    assert len(config.action_feature_names) == 7


def test_rlinf_pi0_extra_delta_mask_matches_openpi_six_plus_gripper() -> None:
    from lerobot.processor.relative_action_processor import (
        RelativeActionsProcessorStep,
        to_absolute_actions,
        to_relative_actions,
    )

    config = SimpleNamespace(
        chunk_size=50,
        n_action_steps=50,
        num_inference_steps=10,
        compile_model=True,
    )
    _configure_rlinf_openpi_contract(
        config,
        family="pi0",
        execution_horizon=5,
        action_dim=7,
        model_chunk_size=10,
        num_inference_steps=4,
        extra_delta_transform=True,
    )
    step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=config.relative_exclude_joints,
        action_names=config.action_feature_names,
    )
    mask = step._build_mask(7)
    assert mask == [True, True, True, True, True, True, False]

    state = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.25, 99.0]])
    actions = torch.tensor(
        [[[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, -1.0]]]
    )
    relative = to_relative_actions(actions, state, mask)
    torch.testing.assert_close(
        relative,
        torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0]]]),
    )
    torch.testing.assert_close(to_absolute_actions(relative, state, mask), actions)


def test_rlinf_pi0_extra_delta_contract_rejects_non_libero_action_shape() -> None:
    config = SimpleNamespace(
        chunk_size=50,
        n_action_steps=50,
        num_inference_steps=10,
        compile_model=True,
    )

    with pytest.raises(ValueError, match="action_dim=7"):
        _configure_rlinf_openpi_contract(
            config,
            family="pi0",
            execution_horizon=5,
            action_dim=8,
            model_chunk_size=10,
            num_inference_steps=4,
            extra_delta_transform=True,
        )


def test_rlinf_norm_stats_map_to_lerobot_features_and_slice_action(tmp_path) -> None:
    import json

    stats = {
        "norm_stats": {
            "state": {key: list(range(32)) for key in ("mean", "std", "q01", "q99")},
            "actions": {key: list(range(32)) for key in ("mean", "std", "q01", "q99")},
        }
    }
    (tmp_path / "stats.json").write_text(json.dumps(stats), encoding="utf-8")

    converted = _load_rlinf_openpi_normalization_stats(
        model_id=str(tmp_path),
        revision=None,
        filename="stats.json",
        state_dim=32,
        action_dim=7,
    )

    assert converted["observation.state"]["q99"].shape == (32,)
    assert converted["action"]["q99"].shape == (7,)


def test_pi_full_checkpoint_roundtrip(tmp_path) -> None:
    class NativePolicy:
        def __init__(self) -> None:
            self.model = torch.nn.Linear(2, 1)

        def to(self, device):
            self.model.to(device)
            return self

    policy = object.__new__(PIFlowPolicy)
    policy.family = "pi05"
    policy.model_id = "base"
    policy.revision = "revision"
    policy.model_format = "lerobot"
    policy.device = "cpu"
    policy.policy = NativePolicy()
    original = policy.model.weight.detach().clone()

    policy.save_checkpoint(str(tmp_path))
    with torch.no_grad():
        policy.model.weight.add_(10.0)
    policy.load_checkpoint({"path": str(tmp_path)})

    torch.testing.assert_close(policy.model.weight, original)


def test_pi_cpu_offload_releases_cuda_allocator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NativePolicy:
        def __init__(self) -> None:
            self.model = torch.nn.Linear(2, 1)

        def to(self, device):
            self.model.to(device)
            return self

    policy = object.__new__(PIFlowPolicy)
    policy.policy = NativePolicy()
    policy.device = "cuda:0"
    calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(True))

    result = policy.to("cpu")

    assert result is policy
    assert policy.device == "cpu"
    assert calls == [True]


def test_pi_flow_rollout_concatenate_restores_native_batch() -> None:
    rows = []
    for index in range(2):
        rows.append(
            PIFlowSDERollout(
                actions=torch.full((1, 2, 1), float(index)),
                transition=FlowSDETransitionRecord(
                    previous_states=torch.full((1, 2, 1), float(index)),
                    next_states=torch.full((1, 2, 1), float(index + 1)),
                    selected_indices=torch.tensor([index]),
                    old_logprobs=torch.full((1, 2, 1), float(index)),
                ),
                inputs=PIFlowModelInputs(
                    images=(torch.full((1, 3, 2, 2), float(index)),),
                    image_masks=(torch.ones(1, dtype=torch.bool),),
                    language_tokens=torch.tensor([[index]]),
                    language_masks=torch.ones(1, 1, dtype=torch.bool),
                    state=None,
                ),
            )
        )

    combined = PIFlowSDERollout.concatenate(rows)

    assert combined.inputs.batch_size == 2
    for index, expected in enumerate(rows):
        actual = combined.select(index)
        torch.testing.assert_close(actual.actions, expected.actions)
        torch.testing.assert_close(
            actual.transition.old_logprobs,
            expected.transition.old_logprobs,
        )
        torch.testing.assert_close(
            actual.inputs.language_tokens,
            expected.inputs.language_tokens,
        )
