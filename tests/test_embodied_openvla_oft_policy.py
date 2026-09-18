from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip("torch")

from art_embodied import (
    EmbodiedTrajectory,
    EmbodiedTrajectoryGroup,
    Observation,
)
from art_embodied.backends.action_token import (
    TRANSIENT_RLINF_FORWARD_INPUTS_METADATA_KEY,
    ActionTokenExample,
)
import art_embodied.policies.openvla as openvla_module
from art_embodied.policies.openvla import (
    OpenVLAPolicy,
    _flatten_first_batch_values,
    _load_rlinf_openvla_oft_model,
    _merge_external_dataset_statistics,
    _merge_rlinf_env_obs_rows,
    _oft_action_token_logprobs,
    _oft_action_token_logprobs_batch,
    _openvla_oft_action_logit_range,
    _openvla_oft_action_model,
    _openvla_oft_action_token_range,
    _predict_action_with_tokens,
    _predict_oft_actions_with_tokens_batch,
    _prepare_openvla_inputs,
    _resolve_rlinf_openvla_oft_loaders,
    _rlinf_native_env_obs_batch_from_observations,
    _sample_rlinf_oft_action_tokens,
    _try_prepare_openvla_oft_inputs_with_proprio,
    openvla_oft_v01_runtime_issues,
)
from art_embodied.policies.openvla_oft_native import (
    configure_native_openvla_oft_model,
)


def _missing_module(name: str) -> ModuleNotFoundError:
    return ModuleNotFoundError(f"No module named {name!r}", name=name)


def test_native_prismatic_shim_does_not_execute_installed_parent_packages(
    tmp_path: Path,
) -> None:
    package = tmp_path / "prismatic"
    training = package / "training"
    training.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "raise RuntimeError('prismatic package imported')\n", encoding="utf-8"
    )
    (training / "__init__.py").write_text(
        "raise RuntimeError('training package imported')\n", encoding="utf-8"
    )
    (training / "train_utils.py").write_text("VALUE = 1\n", encoding="utf-8")
    source_root = Path(__file__).parents[1] / "src"
    code = "\n".join(
        [
            "from art_embodied.policies.openvla import "
            "_install_openvla_oft_prismatic_compat_if_needed",
            "assert _install_openvla_oft_prismatic_compat_if_needed("
            "hint='libero', robot_platform='libero')",
            "from prismatic.training.train_utils import get_current_action_mask",
            "assert callable(get_current_action_mask)",
        ]
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": f"{tmp_path}:{source_root}"},
    )

    assert result.returncode == 0, result.stderr


def test_openvla_oft_v01_runtime_contract_accepts_validated_versions(
    monkeypatch,
) -> None:
    versions = {
        "torch": "2.6.0+cu124",
        "transformers": "4.40.1",
        "tokenizers": "0.19.1",
        "timm": "0.9.10",
        "peft": "0.11.1",
    }
    monkeypatch.setattr(openvla_module.metadata, "version", versions.__getitem__)

    assert openvla_oft_v01_runtime_issues(require_peft=True) == []


def test_openvla_oft_v01_runtime_contract_reports_behavior_drift(
    monkeypatch,
) -> None:
    versions = {
        "torch": "2.9.0+cu128",
        "transformers": "4.57.1",
        "tokenizers": "0.22.2",
        "timm": "0.9.16",
        "peft": "0.19.1",
    }
    monkeypatch.setattr(openvla_module.metadata, "version", versions.__getitem__)

    assert openvla_oft_v01_runtime_issues(require_peft=True) == [
        "torch==2.6.0 (found 2.9.0+cu128)",
        "transformers==4.40.1 (found 4.57.1)",
        "tokenizers==0.19.1 (found 0.22.2)",
        "timm==0.9.10 (found 0.9.16)",
        "peft==0.11.1 (found 0.19.1)",
    ]


def test_flatten_first_batch_values_supports_bfloat16() -> None:
    values = torch.tensor([[0.25, -0.5]], dtype=torch.bfloat16)

    assert _flatten_first_batch_values(values, dtype=float) == [0.25, -0.5]


def test_rlinf_openvla_oft_loader_resolves_modern_api(monkeypatch) -> None:
    get_model = object()

    def fake_import(name: str):
        assert name == "rlinf.models.embodiment.openvla_oft"
        return SimpleNamespace(get_model=lambda: get_model)

    monkeypatch.setattr(openvla_module, "import_module", fake_import)

    resolved_get_model, processor_loader, api = _resolve_rlinf_openvla_oft_loaders()

    assert resolved_get_model() is get_model
    assert processor_loader is None
    assert api == "modern"


def test_rlinf_openvla_oft_loader_resolves_v01_api(monkeypatch) -> None:
    get_model = lambda _cfg: object()
    get_processor = lambda _cfg: (object(), object())

    def fake_import(name: str):
        if name == "rlinf.models.embodiment.openvla_oft":
            raise _missing_module(name)
        assert name == "rlinf.models"
        return SimpleNamespace(
            get_model=get_model,
            get_vla_model_config_and_processor=get_processor,
        )

    monkeypatch.setattr(openvla_module, "import_module", fake_import)

    resolved_get_model, resolved_get_processor, api = (
        _resolve_rlinf_openvla_oft_loaders()
    )

    assert resolved_get_model is get_model
    assert resolved_get_processor is get_processor
    assert api == "v0.1"


def test_rlinf_openvla_oft_loader_does_not_hide_modern_dependency_failure(
    monkeypatch,
) -> None:
    def fake_import(name: str):
        assert name == "rlinf.models.embodiment.openvla_oft"
        raise _missing_module("flash_attn")

    monkeypatch.setattr(openvla_module, "import_module", fake_import)

    with pytest.raises(RuntimeError, match="present RLinf OpenVLA-OFT module"):
        _resolve_rlinf_openvla_oft_loaders()


def test_load_rlinf_openvla_oft_model_reproduces_v01_worker_setup(
    monkeypatch,
    tmp_path,
) -> None:
    calls: dict[str, object] = {}
    call_order: list[str] = []

    def as_namespace(value):
        if isinstance(value, dict):
            return SimpleNamespace(**{key: as_namespace(item) for key, item in value.items()})
        if isinstance(value, list):
            return [as_namespace(item) for item in value]
        return value

    def as_container(value):
        if isinstance(value, SimpleNamespace):
            return {key: as_container(item) for key, item in vars(value).items()}
        if isinstance(value, list):
            return [as_container(item) for item in value]
        return value

    class FakeOmegaConf:
        create = staticmethod(as_namespace)
        to_container = staticmethod(lambda value, resolve=True: as_container(value))

    monkeypatch.setitem(
        sys.modules,
        "omegaconf",
        SimpleNamespace(OmegaConf=FakeOmegaConf),
    )

    class FakeModel:
        def setup_config_and_processor(self, model_config, cfg, input_processor):
            calls["setup"] = (model_config, cfg, input_processor)

        def to(self, device):
            calls["device"] = device
            return self

        def eval(self):
            calls["eval"] = True
            return self

    model = FakeModel()
    model_config = object()
    input_processor = object()

    def fake_get_model(cfg):
        call_order.append("get_model")
        calls["model_cfg"] = cfg
        return model

    def fake_get_processor(actor_cfg):
        call_order.append("get_processor")
        calls["actor_cfg"] = actor_cfg
        return model_config, input_processor

    monkeypatch.setattr(
        openvla_module,
        "_resolve_rlinf_openvla_oft_loaders",
        lambda: (fake_get_model, fake_get_processor, "v0.1"),
    )
    monkeypatch.setattr(
        openvla_module,
        "_resolve_rlinf_local_model_path",
        lambda _model_id: tmp_path,
    )
    monkeypatch.setattr(
        openvla_module,
        "_ensure_rlinf_openvla_oft_prismatic_shim",
        lambda **_kwargs: None,
    )

    loaded = _load_rlinf_openvla_oft_model(
        model_id="fake/openvla-oft",
        torch_dtype=torch.bfloat16,
        device="cuda:2",
        unnorm_key="libero_object_no_noops",
        max_prompt_length=128,
        num_images_in_input=2,
        attn_implementation="flash_attention_2",
        peft_adapter_path=None,
    )

    actor_cfg = calls["actor_cfg"]
    setup_model_config, runtime_cfg, setup_processor = calls["setup"]
    assert loaded is model
    assert loaded._art_rlinf_loader_api == "v0.1"
    assert call_order == ["get_model", "get_processor"]
    assert actor_cfg.model.model_path == str(tmp_path)
    assert actor_cfg.tokenizer.tokenizer_model == str(tmp_path)
    assert setup_model_config is model_config
    assert setup_processor is input_processor
    assert runtime_cfg.actor.model.num_images_in_input == 2
    assert runtime_cfg.runner.max_prompt_length == 128
    assert calls["device"] == "cuda:2"
    assert calls["eval"] is True


def test_merge_rlinf_env_obs_rows_preserves_v01_schema() -> None:
    rows = [
        {
            "images": torch.zeros((1, 3, 8, 9), dtype=torch.uint8),
            "wrist_images": torch.zeros((1, 3, 8, 9), dtype=torch.uint8),
            "states": torch.zeros((1, 8)),
            "task_descriptions": ["task a"],
        },
        {
            "images": torch.ones((1, 3, 8, 9), dtype=torch.uint8),
            "wrist_images": torch.ones((1, 3, 8, 9), dtype=torch.uint8),
            "states": torch.ones((1, 8)),
            "task_descriptions": ["task b"],
        },
    ]

    merged, reports = _merge_rlinf_env_obs_rows(
        rows,
        instructions=["fallback a", "fallback b"],
    )

    assert "main_images" not in merged
    assert merged["images"].shape == (2, 3, 8, 9)
    assert merged["wrist_images"].shape == (2, 3, 8, 9)
    assert merged["states"].shape == (2, 8)
    assert merged["task_descriptions"] == ["task a", "task b"]
    assert [report["primary_image_key"] for report in reports] == ["images", "images"]


def test_rlinf_native_observation_batch_preserves_v01_primary_image_key() -> None:
    model = SimpleNamespace(_art_rlinf_loader_api="v0.1")
    observations = [
        Observation(step=0, kind="image", value=np.zeros((8, 9, 3), dtype=np.uint8)),
        Observation(step=0, kind="image", value=np.ones((8, 9, 3), dtype=np.uint8)),
    ]

    env_obs, reports = _rlinf_native_env_obs_batch_from_observations(
        model,
        observations,
        instructions=["task a", "task b"],
        num_images_in_input=1,
        use_proprio=False,
    )

    assert "main_images" not in env_obs
    assert env_obs["images"].shape == (2, 3, 8, 9)
    assert env_obs["task_descriptions"] == ["task a", "task b"]
    assert [report["primary_image_key"] for report in reports] == ["images", "images"]


class FakeVisionBackbone:
    def __init__(self) -> None:
        self.num_images_in_input = 1

    def get_num_patches(self) -> int:
        return 0

    def get_num_images_in_input(self) -> int:
        return self.num_images_in_input

    def set_num_images_in_input(self, value: int) -> None:
        self.num_images_in_input = int(value)


class FakeLanguageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_logits = torch.nn.Parameter(torch.tensor([4.0, 5.0]))

    def forward(self, *, inputs_embeds, **_kwargs):
        batch, seq_len, _hidden = inputs_embeds.shape
        logits = torch.zeros(batch, seq_len, 20, device=inputs_embeds.device)
        # OpenVLA-OFT / RLinf samples action-bin logits from the range starting
        # at vocab_size - n_action_bins. These decoys make range conflation fail
        # loudly without overpowering the valid action-token slice.
        logits[:, 2, 12] = 99.0
        logits[:, 3, 13] = 99.0
        logits[:, 2, 17] = self.action_logits[0]
        logits[:, 3, 18] = self.action_logits[1]
        return SimpleNamespace(
            logits=logits,
            hidden_states=[inputs_embeds],
            loss=None,
            past_key_values=None,
            attentions=None,
        )


class FakeOpenVLAOFTModel(torch.nn.Module):
    action_dim = 2
    num_actions_chunk = 1
    ignore_index = -100
    stop_index = 2
    action_token_begin_idx = 15

    def __init__(self) -> None:
        super().__init__()
        self.language_model = FakeLanguageModel()
        self.vision_backbone = FakeVisionBackbone()
        self.register_buffer("runtime_buffer", torch.tensor([1.0], dtype=torch.float64))
        self.vocab_size = 20
        self.config = SimpleNamespace(n_action_bins=4)
        self.bin_centers = np.asarray([-0.75, -0.25, 0.25, 0.75])

    def get_action_dim(self, _unnorm_key=None) -> int:
        return self.action_dim

    def get_input_embeddings(self):
        def embed(input_ids):
            batch, seq_len = input_ids.shape
            return torch.ones(batch, seq_len, 4, device=input_ids.device)

        return embed

    def _prepare_input_for_action_prediction(self, input_ids, attention_mask):
        action_tokens = torch.ones(
            (input_ids.shape[0], self.action_dim * self.num_actions_chunk),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        stop = torch.full(
            (input_ids.shape[0], 1),
            self.stop_index,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        input_ids = torch.cat([input_ids, action_tokens, stop], dim=-1)
        mask_extension = torch.ones(
            (attention_mask.shape[0], action_tokens.shape[1] + 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        return input_ids, torch.cat([attention_mask, mask_extension], dim=-1)

    def _prepare_labels_for_action_prediction(self, labels, input_ids):
        extension = torch.full(
            (labels.shape[0], input_ids.shape[-1] - labels.shape[-1]),
            self.action_token_begin_idx + 1,
            dtype=labels.dtype,
            device=labels.device,
        )
        labels = torch.cat([labels, extension], dim=-1)
        labels[:, -1] = self.stop_index
        return labels

    def _process_action_masks(self, labels):
        return labels > self.action_token_begin_idx

    def _process_vision_features(
        self, pixel_values, language_embeddings, use_film=False
    ):
        hidden_dim = (
            language_embeddings.shape[-1] if language_embeddings is not None else 4
        )
        return torch.zeros(
            pixel_values.shape[0],
            0,
            hidden_dim,
            device=pixel_values.device,
        )

    def _build_multimodal_attention(
        self, input_embeddings, projected_patch_embeddings, attention_mask
    ):
        return input_embeddings, attention_mask

    def _unnormalize_actions(self, normalized_actions, _unnorm_key=None):
        return normalized_actions


def test_native_openvla_oft_runtime_binds_explicit_rollout_contract() -> None:
    model = FakeOpenVLAOFTModel()
    model.norm_stats = {"libero_object_no_noops": {"action": {}}}
    model.config.text_config = SimpleNamespace(vocab_size=24)
    model.config.pad_to_multiple_of = 4
    processor = SimpleNamespace(tokenizer=object())

    report = configure_native_openvla_oft_model(
        model,
        processor,
        action_dim=2,
        num_action_chunks=1,
        max_prompt_length=128,
        num_images_in_input=1,
        unnorm_key="libero_object_no_noops",
    )

    assert report == {
        "runtime": "art_native_openvla_oft",
        "action_dim": 2,
        "num_action_chunks": 1,
        "action_tokens_per_chunk": 2,
        "action_tokens_per_policy_call": 2,
        "max_prompt_length": 128,
        "num_images_in_input": 1,
        "unnorm_key": "libero_object_no_noops",
        "stop_token_id": 2,
        "model_dtype": "torch.float32",
    }
    assert model.input_processor is processor
    assert model.processor is processor
    assert model._art_native_openvla_oft is True
    assert model.vision_backbone.get_num_images_in_input() == 1
    assert model.runtime_buffer.dtype == torch.float32

    input_ids = torch.tensor([[1, 42, 29871, 1, 1, 2]])
    attention_mask = torch.ones_like(input_ids)
    embeddings, mask = model._build_embedding(
        input_ids,
        attention_mask,
        torch.ones((1, 3, 4, 4)),
    )

    assert embeddings.shape == (1, 5, 4)
    assert torch.count_nonzero(embeddings[:, -2:]) == 0
    assert torch.equal(mask, attention_mask[:, :-1])


def test_native_openvla_oft_preprocessing_requires_tensor_path(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeInputs(dict):
        def to(self, device, dtype):
            calls["to"] = (device, dtype)
            return self

    expected = FakeInputs(
        input_ids=torch.ones((1, 4), dtype=torch.long),
        attention_mask=torch.ones((1, 4), dtype=torch.long),
        pixel_values=torch.ones((1, 1, 6, 4, 4)),
    )

    def fake_tensor_inputs(processor, **kwargs):
        calls["processor"] = processor
        calls["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(
        openvla_module,
        "_try_prepare_rlinf_openvla_oft_tensor_inputs",
        fake_tensor_inputs,
    )

    class FakeProcessor:
        def __call__(self, *_args, **_kwargs):
            raise AssertionError(
                "native tensor preprocessing must not use the PIL fallback"
            )

    processor = FakeProcessor()
    observation = Observation(
        step=0,
        kind="image",
        value=np.zeros((8, 9, 3), dtype=np.uint8),
    )
    actual = _prepare_openvla_inputs(
        processor,
        observation=observation,
        prompt="In: task\nOut: ",
        device="cpu",
        dtype="float32",
        max_prompt_length=128,
        max_images=1,
        use_proprio=False,
        prefer_tensor_image_processing=True,
    )

    assert actual is expected
    assert calls["processor"] is processor
    assert calls["to"] == ("cpu", torch.float32)
    kwargs = calls["kwargs"]
    assert kwargs["prompts"] == ["In: task\nOut: "]
    assert len(kwargs["proprio_states"]) == 1
    assert kwargs["proprio_states"][0].shape == (8,)


class FakeRLinfNativeOpenVLAOFTModel(FakeOpenVLAOFTModel):
    def __init__(self) -> None:
        super().__init__()
        self.num_action_chunks = 1
        self.input_processor = object()
        self.native_weight = torch.nn.Parameter(torch.tensor(0.5))
        self.predict_calls = 0
        self.default_forward_calls = 0
        self.predict_batch_sizes: list[int] = []
        self.seen_task_descriptions: list[str] = []
        self.last_env_obs = None
        self.default_forward_action_tokens: torch.Tensor | None = None

    def _build_embedding(self, input_ids, attention_mask, pixel_values):
        batch = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        return torch.ones(batch, seq_len, 4, device=input_ids.device), attention_mask

    def predict_action_batch(
        self, *, env_obs, do_sample, temperature, top_k, calculate_values
    ):
        self.predict_calls += 1
        self.last_env_obs = env_obs
        self.seen_task_descriptions = list(env_obs["task_descriptions"])
        batch = len(self.seen_task_descriptions)
        self.predict_batch_sizes.append(batch)
        forward_inputs = {
            "input_ids": torch.tensor([[1, 42, 29871]] * batch, dtype=torch.long),
            "attention_mask": torch.ones(batch, 3, dtype=torch.bool),
            "pixel_values": torch.ones(batch, 3, 4, 4),
            # This placeholder must be overwritten by the rescore helper.
            "action_tokens": torch.full(
                (batch, self.num_action_chunks, self.action_dim), 16, dtype=torch.long
            ),
        }
        return torch.zeros(batch, self.num_action_chunks, self.action_dim), {
            "forward_inputs": forward_inputs,
            "prev_logprobs": torch.zeros(
                batch, self.action_dim * self.num_action_chunks
            ),
        }

    def default_forward(
        self,
        *,
        forward_inputs,
        compute_logprobs,
        compute_entropy,
        compute_values,
        temperature,
        top_k,
    ):
        self.default_forward_calls += 1
        assert compute_logprobs is True
        tokens = forward_inputs["action_tokens"].reshape(
            forward_inputs["action_tokens"].shape[0], -1
        )
        self.default_forward_action_tokens = tokens.detach().clone()
        return {"logprobs": self.native_weight * (tokens.float() - 16.0)}


class FakePeftWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.base_model = SimpleNamespace(model=model)

    def forward(self, *args, **kwargs):
        return self.base_model.model(*args, **kwargs)


class FakeRegisteredSubmodulePeftWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.adapter_shell = torch.nn.Module()
        self.adapter_shell.remote_model = model

    def forward(self, *args, **kwargs):
        return self.adapter_shell.remote_model(*args, **kwargs)


class FakeProxyPeftWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.base_model = SimpleNamespace(model=model)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            base_model = super().__getattribute__("base_model")
            return getattr(base_model.model, name)

    def forward(self, *args, **kwargs):
        return self.base_model.model(*args, **kwargs)


class FakeProcessor:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.call_count = 0

    def __call__(self, _prompt, _image):
        self.call_count += 1
        prompts = _prompt if isinstance(_prompt, list) else [_prompt]
        self.prompts.extend(prompts)
        batch = len(prompts)
        return {
            "input_ids": torch.tensor([[1, 7]] * batch, dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]] * batch, dtype=torch.long),
            "pixel_values": torch.ones(batch, 3, 8, 8),
        }


class FakeRLinfTokenizer:
    padding_side = "left"
    pad_token_id = 0
    bos_token_id = 1

    def __call__(
        self,
        prompts,
        *,
        return_tensors,
        padding,
        truncation=None,
        max_length=None,
    ):
        assert return_tensors == "pt"
        batch = len(prompts)
        width = int(max_length or 6)
        input_ids = torch.zeros(batch, width, dtype=torch.long)
        attention_mask = torch.zeros(batch, width, dtype=torch.long)
        input_ids[:, -3:] = torch.tensor([self.bos_token_id, 42, 29871])
        attention_mask[:, -3:] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class FakeRLinfTensorImageProcessor:
    input_sizes = [4]
    tvf_resize_params = [{"size": [4, 4]}]
    tvf_crop_params = [{"output_size": [4, 4]}]
    tvf_normalize_params = [{"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}]


class FakeRLinfTensorProcessor:
    def __init__(self) -> None:
        self.tokenizer = FakeRLinfTokenizer()
        self.image_processor = FakeRLinfTensorImageProcessor()

    def __call__(self, *_args, **_kwargs):
        raise AssertionError(
            "OpenVLA-OFT RLinf parity path should not use PIL fallback"
        )


def _inputs():
    return {
        "input_ids": torch.tensor([[1, 7]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        "pixel_values": torch.ones(1, 3, 8, 8),
    }


def test_openvla_oft_predict_action_with_tokens_uses_placeholder_logits() -> None:
    model = FakeOpenVLAOFTModel()

    action, tokens, logprobs = _predict_action_with_tokens(
        model,
        _inputs(),
        unnorm_key=None,
        do_sample=False,
        temperature=1.0,
    )

    assert tokens == [17, 18]
    assert len(logprobs) == 2
    assert np.asarray(action).shape == (1, 2)


def test_openvla_oft_predict_action_uses_unpadded_prompt_length_after_left_padding() -> (
    None
):
    model = FakeOpenVLAOFTModel()
    inputs = {
        "input_ids": torch.tensor([[0, 0, 1, 7, 29871]], dtype=torch.long),
        "attention_mask": torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.long),
        "pixel_values": torch.ones(1, 3, 8, 8),
    }

    action, tokens, logprobs = _predict_action_with_tokens(
        model,
        inputs,
        unnorm_key=None,
        do_sample=False,
        temperature=1.0,
    )

    assert tokens == [17, 18]
    assert len(logprobs) == 2
    assert np.asarray(action).shape == (1, 2)


def test_openvla_oft_predict_action_with_tokens_unwraps_peft_model() -> None:
    model = FakeOpenVLAOFTModel()
    model.num_actions_chunk = 2
    wrapped = FakePeftWrapper(model)

    action, tokens, logprobs = _predict_action_with_tokens(
        wrapped,
        _inputs(),
        unnorm_key=None,
        do_sample=False,
        temperature=1.0,
    )

    assert tokens == [17, 18, 16, 16]
    assert len(logprobs) == 4
    assert np.asarray(action).shape == (2, 2)


def test_openvla_oft_action_model_unwraps_registered_peft_submodule() -> None:
    model = FakeOpenVLAOFTModel()
    wrapped = FakeRegisteredSubmodulePeftWrapper(model)

    assert _openvla_oft_action_model(wrapped) is model
    action, tokens, logprobs = _predict_action_with_tokens(
        wrapped,
        _inputs(),
        unnorm_key=None,
        do_sample=False,
        temperature=1.0,
    )

    assert tokens == [17, 18]
    assert len(logprobs) == 2
    assert np.asarray(action).shape == (1, 2)


def test_openvla_oft_proxy_wrapper_uses_base_constants() -> None:
    model = FakeOpenVLAOFTModel()
    model.num_actions_chunk = 2
    wrapped = FakeProxyPeftWrapper(model)

    action, tokens, logprobs = _predict_action_with_tokens(
        wrapped,
        _inputs(),
        unnorm_key=None,
        do_sample=False,
        temperature=1.0,
    )

    assert tokens == [17, 18, 16, 16]
    assert len(logprobs) == 4
    assert np.asarray(action).shape == (2, 2)


def test_openvla_oft_external_dataset_statistics_merge(tmp_path) -> None:
    model = FakeOpenVLAOFTModel()
    model.norm_stats = {"bridge_orig": {"action": {}}}
    model.config = SimpleNamespace(norm_stats={"bridge_orig": {"action": {}}})
    stats_dir = tmp_path / "fake-openvla-oft"
    stats_dir.mkdir()
    (stats_dir / "dataset_statistics.json").write_text(
        '{"libero_object_no_noops": {"action": {"q01": [0], "q99": [1]}}}',
        encoding="utf-8",
    )

    report = _merge_external_dataset_statistics(model, str(stats_dir))

    assert report["loaded"] is True
    assert "libero_object_no_noops" in model.norm_stats
    assert "libero_object_no_noops" in model.config.norm_stats


def test_openvla_oft_external_dataset_statistics_explicit_path_wins(tmp_path) -> None:
    model = FakeOpenVLAOFTModel()
    model.norm_stats = {"bridge_orig": {"action": {}}}
    model.config = SimpleNamespace(norm_stats={"bridge_orig": {"action": {}}})
    model_dir = tmp_path / "model-with-wrong-stats"
    model_dir.mkdir()
    (model_dir / "dataset_statistics.json").write_text(
        '{"wrong_key": {"action": {"q01": [0], "q99": [1]}}}',
        encoding="utf-8",
    )
    explicit = tmp_path / "explicit_dataset_statistics.json"
    explicit.write_text(
        '{"libero_object_no_noops": {"action": {"q01": [0], "q99": [1]}}}',
        encoding="utf-8",
    )

    report = _merge_external_dataset_statistics(
        model, str(model_dir), explicit_path=explicit
    )

    assert report["loaded"] is True
    assert report["path"] == str(explicit)
    assert "libero_object_no_noops" in model.norm_stats
    assert "wrong_key" not in model.norm_stats


def test_openvla_oft_external_dataset_statistics_uses_model_revision(
    tmp_path,
    monkeypatch,
) -> None:
    model = FakeOpenVLAOFTModel()
    model.norm_stats = {}
    model.config = SimpleNamespace(norm_stats={})
    downloaded = tmp_path / "dataset_statistics.json"
    downloaded.write_text(
        '{"libero_object_no_noops": {"action": {"q01": [0], "q99": [1]}}}',
        encoding="utf-8",
    )
    request: dict[str, object] = {}

    def hf_hub_download(repo_id, filename, *, revision):
        request.update(repo_id=repo_id, filename=filename, revision=revision)
        return str(downloaded)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=hf_hub_download),
    )

    report = _merge_external_dataset_statistics(
        model,
        "example/openvla-oft",
        revision="0123456789abcdef",
    )

    assert report["loaded"] is True
    assert request == {
        "repo_id": "example/openvla-oft",
        "filename": "dataset_statistics.json",
        "revision": "0123456789abcdef",
    }


def test_openvla_oft_tensor_processor_path_matches_rlinf_image_contract() -> None:
    processor = FakeRLinfTensorProcessor()
    image_array = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)

    inputs = _try_prepare_openvla_oft_inputs_with_proprio(
        processor,
        prompts=["In: What action should the robot take to pick up the cube?\nOut: "],
        images=[Image.fromarray(image_array)],
        proprio_states=[np.zeros(8, dtype=np.float32)],
        processor_kwargs={"padding": "max_length", "max_length": 6},
    )

    assert inputs is not None
    expected_pixels = (
        torch.as_tensor(image_array).permute(2, 0, 1).unsqueeze(0).unsqueeze(1).float()
        / 255.0
    )
    assert tuple(inputs["pixel_values"].shape) == (1, 1, 3, 4, 4)
    assert torch.allclose(inputs["pixel_values"], expected_pixels)
    assert inputs["input_ids"].tolist() == [[1, 0, 0, 0, 42, 29871]]
    assert inputs["attention_mask"].tolist() == [[1, 0, 0, 0, 1, 1]]


def test_openvla_oft_tensor_processor_path_supports_multiple_image_streams() -> None:
    processor = FakeRLinfTensorProcessor()
    main = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    wrist = np.full((4, 4, 3), 255, dtype=np.uint8)

    inputs = _try_prepare_openvla_oft_inputs_with_proprio(
        processor,
        prompts=["In: What action should the robot take to pick up the cube?\nOut: "],
        images=[[Image.fromarray(main), Image.fromarray(wrist)]],
        proprio_states=[np.zeros(8, dtype=np.float32)],
        processor_kwargs={"padding": "max_length", "max_length": 6},
    )

    assert inputs is not None
    expected_pixels = (
        torch.stack(
            [
                torch.as_tensor(main).permute(2, 0, 1),
                torch.as_tensor(wrist).permute(2, 0, 1),
            ],
            dim=0,
        )
        .unsqueeze(0)
        .float()
        / 255.0
    )
    assert tuple(inputs["pixel_values"].shape) == (1, 2, 3, 4, 4)
    assert torch.allclose(inputs["pixel_values"], expected_pixels)


def test_openvla_oft_action_token_logprobs_are_differentiable() -> None:
    model = FakeOpenVLAOFTModel()

    row = _oft_action_token_logprobs(
        model,
        _inputs(),
        tokens=[17, 18],
        unnorm_key=None,
        temperature=1.0,
    )
    loss = -row.sum()
    loss.backward()

    assert tuple(row.shape) == (2,)
    assert model.language_model.action_logits.grad is not None
    assert float(model.language_model.action_logits.grad.abs().sum()) > 0.0


def test_openvla_oft_batched_action_token_logprobs_are_differentiable() -> None:
    model = FakeOpenVLAOFTModel()

    rows = _oft_action_token_logprobs_batch(
        model,
        {
            "input_ids": torch.tensor([[1, 7], [1, 7]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
            "pixel_values": torch.ones(2, 3, 8, 8),
        },
        token_rows=[[17, 18], [17, 18]],
        unnorm_key=None,
        temperature=1.0,
    )
    loss = -sum(row.sum() for row in rows)
    loss.backward()

    assert len(rows) == 2
    assert all(tuple(row.shape) == (2,) for row in rows)
    assert model.language_model.action_logits.grad is not None
    assert float(model.language_model.action_logits.grad.abs().sum()) > 0.0


def test_openvla_strict_batched_logprobs_fail_closed_before_fallback() -> None:
    policy = OpenVLAPolicy(strict_batched_logprobs=True)
    example = SimpleNamespace(
        task="pick object",
        prompt="In: What action should the robot take to pick object?\nOut:",
        observation=None,
        tokens=[17, 18],
    )

    with pytest.raises(ValueError, match="requires observations"):
        policy._batched_action_token_logprobs_for_examples([example])


def test_openvla_rlinf_batched_logprobs_accept_cached_forward_inputs_without_observations() -> (
    None
):
    model = FakeRLinfNativeOpenVLAOFTModel()
    policy = OpenVLAPolicy(
        model_id="fake/openvla-oft",
        device="cpu",
        dtype="float32",
        model_loader="rlinf",
        strict_batched_logprobs=True,
        capture_action_tokens=True,
    )
    policy.model = model
    policy.processor = object()
    policy._rlinf_native_loader = True
    forward_inputs = {
        "input_ids": torch.tensor([[1, 42, 29871]], dtype=torch.long),
        "attention_mask": torch.ones(1, 3, dtype=torch.bool),
        "pixel_values": torch.ones(1, 3, 4, 4),
    }
    example = ActionTokenExample(
        task="Move The Cube",
        trajectory_index=0,
        action_index=0,
        step=0,
        tokens=[17, 18],
        reward=1.0,
        prompt="cached forward inputs should define the scorer input",
        observation=None,
        logprobs=[-0.1, -0.2],
        metadata={TRANSIENT_RLINF_FORWARD_INPUTS_METADATA_KEY: forward_inputs},
    )

    rows = policy.action_token_logprobs([example])
    loss = -rows[0].sum()
    loss.backward()

    assert model.predict_calls == 0
    assert model.default_forward_calls == 0
    assert tuple(rows[0].shape) == (2,)
    assert model.language_model.action_logits.grad is not None
    assert float(model.language_model.action_logits.grad.abs().sum()) > 0.0


def test_openvla_oft_predict_actions_with_tokens_batch_uses_one_forward() -> None:
    model = FakeOpenVLAOFTModel()

    predictions = _predict_oft_actions_with_tokens_batch(
        model,
        {
            "input_ids": torch.tensor([[1, 7], [1, 7]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
            "pixel_values": torch.ones(2, 3, 8, 8),
        },
        unnorm_key=None,
        do_sample=False,
        temperature=1.0,
    )

    assert len(predictions) == 2
    assert [row[1] for row in predictions] == [[17, 18], [17, 18]]
    assert all(len(row[2]) == 2 for row in predictions)
    assert all(np.asarray(row[0]).shape == (1, 2) for row in predictions)


def test_openvla_oft_sampling_is_restricted_to_action_token_range() -> None:
    model = FakeOpenVLAOFTModel()
    start, end = _openvla_oft_action_token_range(model)
    logit_start, logit_end = _openvla_oft_action_logit_range(model)

    _action, tokens, logprobs = _predict_action_with_tokens(
        model,
        _inputs(),
        unnorm_key=None,
        do_sample=True,
        temperature=1.6,
    )

    assert start == 16
    assert end == 20
    assert (logit_start, logit_end) == (16, 20)
    assert all(start <= token < end for token in tokens)
    assert len(logprobs) == 2


def test_openvla_oft_sampling_matches_rlinf_full_vocab_multinomial() -> None:
    logits = torch.tensor(
        [
            [99.0, 98.0, -1.0, 0.0, 1.0, 2.0, 97.0, 96.0],
            [95.0, 94.0, 2.0, 1.0, 0.0, -1.0, 93.0, 92.0],
            [91.0, 90.0, 0.5, -0.5, 1.5, -1.5, 89.0, 88.0],
        ],
        dtype=torch.float32,
    )
    temperature = 1.6

    torch.manual_seed(1234)
    sampled, selected_logprobs = _sample_rlinf_oft_action_tokens(
        logits,
        action_logit_start=2,
        action_logit_end=6,
        do_sample=True,
        temperature=temperature,
    )

    rlinf_logits = logits.clone()
    rlinf_logits[..., :2] = -torch.inf
    rlinf_logits[..., 6:] = -torch.inf
    rlinf_logprobs = torch.log_softmax(rlinf_logits / temperature, dim=-1)
    torch.manual_seed(1234)
    expected = torch.multinomial(
        torch.exp(rlinf_logprobs).reshape(-1, logits.shape[-1]),
        num_samples=1,
        replacement=True,
    ).reshape(logits.shape[:-1])

    assert torch.equal(sampled, expected)
    assert torch.equal(
        selected_logprobs,
        rlinf_logprobs.gather(-1, expected.unsqueeze(-1)).squeeze(-1),
    )
    assert torch.all((sampled >= 2) & (sampled < 6))


def test_openvla_oft_logit_range_uses_padded_language_model_vocab() -> None:
    model = FakeOpenVLAOFTModel()
    model.config.pad_to_multiple_of = 2
    model.config.text_config = SimpleNamespace(vocab_size=22)

    # OpenVLA-OFT remote code slices logits[..., -n_bins-pad:-pad].  The
    # language-model logits include padded rows, while model.vocab_size does not.
    # The resulting sampled local bins are emitted as final action-token ids.
    assert _openvla_oft_action_token_range(model) == (16, 20)
    assert _openvla_oft_action_logit_range(model) == (16, 20)


def test_openvla_oft_logprobs_reject_non_action_tokens() -> None:
    model = FakeOpenVLAOFTModel()

    try:
        _oft_action_token_logprobs(
            model,
            _inputs(),
            tokens=[15, 18],
            unnorm_key=None,
            temperature=1.0,
        )
    except ValueError as exc:
        assert "action-token range" in str(exc)
    else:
        raise AssertionError("expected non-action token ids to be rejected")


def test_openvla_policy_recomputes_trajectory_action_token_logprobs() -> None:
    model = FakeOpenVLAOFTModel()
    policy = OpenVLAPolicy(
        model_id="fake/openvla-oft",
        device="cpu",
        dtype="float32",
        unnorm_key=None,
    )
    policy.model = model
    policy.processor = FakeProcessor()
    observation0 = Observation(
        step=0,
        kind="image",
        value=Image.new("RGB", (8, 8), color="red"),
    )
    observation1 = Observation(
        step=1,
        kind="image",
        value=Image.new("RGB", (8, 8), color="blue"),
    )
    example = ActionTokenExample(
        task="place the soup in the basket",
        trajectory_index=0,
        action_index=-1,
        step=0,
        tokens=[17, 18, 17, 18],
        reward=1.0,
        prompt="prompt-0",
        observation=observation0,
        logprobs=[-0.1, -0.2, -0.3, -0.4],
        metadata={
            "training_unit": "trajectory",
            "action_spans": [
                {"token_start": 0, "token_end": 2},
                {"token_start": 2, "token_end": 4},
            ],
            "action_observations": [observation0, observation1],
            "prompts": ["prompt-0", "prompt-1"],
        },
    )

    rows = policy.action_token_logprobs([example])
    loss = -rows[0].sum()
    loss.backward()

    assert len(rows) == 1
    assert tuple(rows[0].shape) == (4,)
    assert policy.processor.call_count == 1
    assert policy.processor.prompts == ["prompt-0", "prompt-1"]
    assert model.language_model.action_logits.grad is not None


def test_openvla_policy_recomputes_action_token_logprobs_in_batches() -> None:
    model = FakeOpenVLAOFTModel()
    policy = OpenVLAPolicy(
        model_id="fake/openvla-oft",
        device="cpu",
        dtype="float32",
        unnorm_key=None,
    )
    policy.model = model
    policy.processor = FakeProcessor()
    observation0 = Observation(
        step=0,
        kind="image",
        value=Image.new("RGB", (8, 8), color="red"),
    )
    observation1 = Observation(
        step=1,
        kind="image",
        value=Image.new("RGB", (8, 8), color="blue"),
    )
    examples = [
        ActionTokenExample(
            task="place the soup in the basket",
            trajectory_index=index,
            action_index=0,
            step=index,
            tokens=[17, 18],
            reward=1.0,
            prompt=f"prompt-{index}",
            observation=observation,
            logprobs=[-0.1, -0.2],
        )
        for index, observation in enumerate([observation0, observation1])
    ]

    rows = policy.action_token_logprobs(examples)
    loss = -sum(row.sum() for row in rows)
    loss.backward()

    assert len(rows) == 2
    assert policy.processor.prompts == ["prompt-0", "prompt-1"]
    assert model.language_model.action_logits.grad is not None


def test_openvla_policy_rlinf_loader_rescores_with_native_predict_slice() -> None:
    model = FakeRLinfNativeOpenVLAOFTModel()
    processor = FakeProcessor()
    policy = OpenVLAPolicy(
        model_id="fake/openvla-oft",
        device="cpu",
        dtype="float32",
        model_loader="rlinf",
        strict_batched_logprobs=True,
        capture_action_tokens=True,
    )
    policy.model = model
    policy.processor = processor
    policy._rlinf_native_loader = True
    observation = Observation(
        step=0,
        kind="image",
        value=Image.new("RGB", (8, 8), color="green"),
    )
    example = ActionTokenExample(
        task="Move The Cube",
        trajectory_index=0,
        action_index=0,
        step=0,
        tokens=[17, 18],
        reward=1.0,
        prompt="this prompt must not drive the RLinf native instruction path",
        observation=observation,
        logprobs=[-0.1, -0.2],
    )

    rows = policy.action_token_logprobs([example])
    loss = -rows[0].sum()
    loss.backward()

    assert model.predict_calls == 1
    assert model.default_forward_calls == 0
    assert processor.call_count == 0
    assert model.seen_task_descriptions == ["Move The Cube"]
    expected0 = torch.tensor(4.0) - torch.logsumexp(
        torch.tensor([0.0, 4.0, 0.0, 0.0]), dim=0
    )
    expected1 = torch.tensor(5.0) - torch.logsumexp(
        torch.tensor([0.0, 0.0, 5.0, 0.0]), dim=0
    )
    assert torch.allclose(rows[0].detach(), torch.stack([expected0, expected1]))
    assert model.language_model.action_logits.grad is not None
    assert model.native_weight.grad is None


def test_openvla_policy_rlinf_act_uses_context_env_obs_passthrough() -> None:
    model = FakeRLinfNativeOpenVLAOFTModel()
    policy = OpenVLAPolicy(
        model_id="fake/openvla-oft",
        device="cpu",
        dtype="float32",
        model_loader="rlinf",
        capture_action_tokens=True,
    )
    policy.model = model
    policy.processor = object()
    policy._rlinf_native_loader = True
    observation = Observation(
        step=0,
        kind="image",
        value=Image.new("RGB", (8, 8), color="red"),
    )
    passthrough_obs = {
        "main_images": torch.full((1, 8, 8, 3), 7, dtype=torch.uint8),
        "states": torch.ones((1, 8), dtype=torch.float32),
        "task_descriptions": ["passthrough task"],
    }

    action = policy.act(
        observation,
        {
            "scenario": {"task": "context task"},
            "rlinf_env_obs": passthrough_obs,
            "step": 4,
        },
    )

    assert action.step == 4
    assert model.predict_calls == 1
    assert model.seen_task_descriptions == ["passthrough task"]
    assert model.last_env_obs is not passthrough_obs
    assert torch.equal(
        model.last_env_obs["main_images"], passthrough_obs["main_images"]
    )
    assert torch.equal(model.last_env_obs["states"], passthrough_obs["states"])


def test_openvla_policy_rlinf_v01_converts_generic_observation_to_images_key() -> None:
    model = FakeRLinfNativeOpenVLAOFTModel()
    model._art_rlinf_loader_api = "v0.1"
    policy = OpenVLAPolicy(
        model_id="fake/openvla-oft",
        device="cpu",
        dtype="float32",
        model_loader="rlinf",
        capture_action_tokens=True,
    )
    policy.model = model
    policy.processor = object()
    policy._rlinf_native_loader = True

    policy.act(
        Observation(
            step=0,
            kind="image",
            value=Image.new("RGB", (8, 8), color="red"),
        ),
        {"scenario": {"task": "move the cube"}},
    )

    assert "images" in model.last_env_obs
    assert "main_images" not in model.last_env_obs
    assert model.last_env_obs["images"].shape == (1, 3, 8, 8)


def test_openvla_policy_rlinf_act_batch_respects_logprob_batch_size() -> None:
    model = FakeRLinfNativeOpenVLAOFTModel()
    processor = FakeProcessor()
    policy = OpenVLAPolicy(
        model_id="fake/openvla-oft",
        device="cpu",
        dtype="float32",
        model_loader="rlinf",
        strict_batched_logprobs=True,
        capture_action_tokens=True,
        logprob_batch_size=1,
    )
    policy.model = model
    policy.processor = processor
    policy._rlinf_native_loader = True
    observations = [
        Observation(
            step=index,
            kind="image",
            value=Image.new("RGB", (8, 8), color=color),
        )
        for index, color in enumerate(["red", "blue"])
    ]
    contexts = [
        {"scenario": {"task": "move to the red cube"}, "step": 0},
        {"scenario": {"task": "move to the blue cube"}, "step": 1},
    ]

    actions = policy.act_batch(observations, contexts)

    assert len(actions) == 2
    assert model.predict_calls == 2
    assert model.predict_batch_sizes == [1, 1]
    assert [action.raw["batch_size"] for action in actions] == [1, 1]
    assert [action.metadata["batch_size"] for action in actions] == [1, 1]
    assert all(
        action.metadata["rlinf_native_batched_policy_call"] is True
        for action in actions
    )


def test_openvla_policy_rlinf_act_batch_uses_context_env_obs_passthrough() -> None:
    model = FakeRLinfNativeOpenVLAOFTModel()
    policy = OpenVLAPolicy(
        model_id="fake/openvla-oft",
        device="cpu",
        dtype="float32",
        model_loader="rlinf",
        capture_action_tokens=True,
    )
    policy.model = model
    policy.processor = object()
    policy._rlinf_native_loader = True
    observations = [
        Observation(step=0, kind="image", value=Image.new("RGB", (8, 8), color="red")),
        Observation(step=1, kind="image", value=Image.new("RGB", (8, 8), color="blue")),
    ]
    contexts = [
        {
            "scenario": {"task": "ignored red"},
            "rlinf_env_obs": {
                "main_images": torch.full((1, 8, 8, 3), 3, dtype=torch.uint8),
                "states": torch.zeros((1, 8), dtype=torch.float32),
                "task_descriptions": ["passthrough red"],
            },
        },
        {
            "scenario": {"task": "ignored blue"},
            "rlinf_env_obs": {
                "main_images": torch.full((1, 8, 8, 3), 9, dtype=torch.uint8),
                "states": torch.ones((1, 8), dtype=torch.float32),
                "task_descriptions": ["passthrough blue"],
            },
        },
    ]

    actions = policy.act_batch(observations, contexts)

    assert len(actions) == 2
    assert model.predict_calls == 1
    assert model.predict_batch_sizes == [2]
    assert model.seen_task_descriptions == ["passthrough red", "passthrough blue"]
    assert model.last_env_obs["main_images"].shape == (2, 8, 8, 3)
    assert model.last_env_obs["states"].shape == (2, 8)
    assert torch.equal(
        model.last_env_obs["main_images"][0],
        contexts[0]["rlinf_env_obs"]["main_images"][0],
    )
    assert torch.equal(
        model.last_env_obs["main_images"][1],
        contexts[1]["rlinf_env_obs"]["main_images"][0],
    )


def test_openvla_policy_act_batch_uses_single_processor_batch() -> None:
    model = FakeOpenVLAOFTModel()
    processor = FakeProcessor()
    policy = OpenVLAPolicy(
        model_id="fake/openvla-oft",
        device="cpu",
        dtype="float32",
        unnorm_key=None,
    )
    policy.model = model
    policy.processor = processor
    observations = [
        Observation(
            step=index,
            kind="image",
            value=Image.new("RGB", (8, 8), color=color),
        )
        for index, color in enumerate(["red", "blue"])
    ]
    contexts = [
        {"scenario": {"task": "move to the red cube"}, "step": 0},
        {"scenario": {"task": "move to the blue cube"}, "step": 1},
    ]

    actions = policy.act_batch(observations, contexts)

    assert len(actions) == 2
    assert processor.call_count == 1
    assert processor.prompts == [
        "In: What action should the robot take to move to the red cube?\nOut: ",
        "In: What action should the robot take to move to the blue cube?\nOut: ",
    ]
    assert [action.raw["tokens"] for action in actions] == [[17, 18], [17, 18]]
    assert [action.metadata["batch_index"] for action in actions] == [0, 1]
    assert all(action.metadata["batched_policy_call"] is True for action in actions)
    assert actions[0].metadata["openvla_oft_action_token_range"] == [16, 20]
    assert actions[0].metadata["openvla_oft_action_logit_range"] == [16, 20]
    assert actions[0].metadata["openvla_oft_action_logit_to_token_offset"] == 0


def test_openvla_policy_act_batch_logprobs_match_recompute_for_peft_wrappers() -> None:
    model = FakeOpenVLAOFTModel()
    processor = FakeProcessor()
    policy = OpenVLAPolicy(
        model_id="fake/openvla-oft",
        device="cpu",
        dtype="float32",
        unnorm_key=None,
    )
    policy.model = FakePeftWrapper(model)
    policy.processor = processor
    observations = [
        Observation(
            step=index,
            kind="image",
            value=Image.new("RGB", (8, 8), color=color),
        )
        for index, color in enumerate(["red", "blue"])
    ]
    contexts = [
        {"scenario": {"task": "move to the red cube"}, "step": 0},
        {"scenario": {"task": "move to the blue cube"}, "step": 1},
    ]

    actions = policy.act_batch(observations, contexts)
    examples = [
        ActionTokenExample(
            task=str(context["scenario"]["task"]),
            trajectory_index=index,
            action_index=0,
            step=index,
            tokens=list(action.raw["tokens"]),
            reward=1.0,
            prompt=str(action.raw["prompt"]),
            observation=observations[index],
            logprobs=list(action.logprobs["token_logprobs"]),
        )
        for index, (action, context) in enumerate(zip(actions, contexts, strict=True))
    ]

    recomputed = policy.action_token_logprobs(examples)

    assert _openvla_oft_action_model(policy.model) is model
    for action, row in zip(actions, recomputed, strict=True):
        old = torch.as_tensor(action.logprobs["token_logprobs"], dtype=row.dtype)
        assert torch.allclose(row.detach().cpu(), old, atol=1e-6)


def test_openvla_policy_delegates_module_parameter_interfaces() -> None:
    model = FakeOpenVLAOFTModel()
    policy = OpenVLAPolicy(model_id="fake/openvla-oft", device="cpu", dtype="float32")
    policy.model = model

    named = dict(policy.named_parameters())

    assert "language_model.action_logits" in named
    assert list(policy.parameters())
    policy.train()
    assert model.training is True
    policy.eval()
    assert model.training is False
    assert "language_model.action_logits" in policy.state_dict()


def test_openvla_oft_policy_prompt_matches_rlinf_template() -> None:
    processor = FakeProcessor()
    policy = OpenVLAPolicy(
        model_id="fake/openvla-oft",
        revision="0123456789abcdef",
        device="cpu",
        dtype="float32",
        capture_action_tokens=True,
    )
    policy.model = FakeOpenVLAOFTModel()
    policy.processor = processor
    observation = Observation(
        step=0,
        kind="image",
        value=Image.new("RGB", (8, 8), color="red"),
    )

    action = policy.act(observation, {"scenario": {"task": "Pick Up The Tomato Sauce"}})

    assert processor.prompts == [
        "In: What action should the robot take to pick up the tomato sauce?\nOut: "
    ]
    assert action.raw["prompt"] == processor.prompts[0]
    assert action.raw["revision"] == "0123456789abcdef"
    assert action.metadata["revision"] == "0123456789abcdef"


def test_openvla_oft_trainable_selection_can_target_late_language_layers() -> None:
    from art_embodied.vla_trainable import configure_vla_trainable_parameters

    class FakeSelfAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(2, 2)
            self.v_proj = torch.nn.Linear(2, 2)

    class FakeLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = FakeSelfAttention()

    class FakeLM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList(
                [FakeLayer(), FakeLayer(), FakeLayer()]
            )

    class FakePolicy:
        def __init__(self) -> None:
            self.model = torch.nn.Module()
            self.model.language_model = FakeLM()

        def named_parameters(self, *args, **kwargs):
            return self.model.named_parameters(*args, **kwargs)

        def named_modules(self, *args, **kwargs):
            return self.model.named_modules(*args, **kwargs)

    policy = FakePolicy()
    policy, report = configure_vla_trainable_parameters(
        policy,
        algorithm_cfg={
            "trainable_parameter_strategy": "openvla_oft_lm_late",
            "trainable_last_n_layers": 1,
            "allow_embedding_training": False,
            "force_trainable_float32": False,
            "trainable_parameter_patterns": [],
            "peft": {"enabled": False},
        },
        policy_type="openvla_oft",
    )
    trainable_names = [name for name, p in policy.named_parameters() if p.requires_grad]

    assert report["ok"] is True
    assert report["layer_selection"] == {"last_n_layers": 1, "selected_layers": [2]}
    assert trainable_names
    assert all("layers.2." in name for name in trainable_names)
    assert any("q_proj" in name for name in trainable_names)


def test_openvla_oft_attached_peft_lora_trainables_can_be_forced_to_float32() -> None:
    from art_embodied.vla_trainable import configure_vla_trainable_parameters

    class FakeAttachedPeftModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.peft_config = {"default": object()}
            self.lora_A = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.bfloat16))
            self.base_weight = torch.nn.Parameter(
                torch.ones(2, 2, dtype=torch.bfloat16)
            )

    class FakePolicy:
        def __init__(self) -> None:
            self.model = FakeAttachedPeftModel()

        def named_parameters(self, *args, **kwargs):
            return self.model.named_parameters(*args, **kwargs)

        def named_modules(self, *args, **kwargs):
            return self.model.named_modules(*args, **kwargs)

    policy = FakePolicy()
    policy, report = configure_vla_trainable_parameters(
        policy,
        algorithm_cfg={
            "trainable_parameter_strategy": "openvla_oft_lora",
            "force_trainable_float32": True,
            "peft": {"enabled": True},
        },
        policy_type="openvla_oft",
    )
    named = dict(policy.named_parameters())

    assert report["ok"] is True
    assert report["method"] == "peft_lora_attached_adapter"
    assert report["force_trainable_float32"] is True
    assert named["lora_A"].requires_grad is True
    assert named["lora_A"].dtype == torch.float32
    assert named["base_weight"].requires_grad is False
    assert named["base_weight"].dtype == torch.bfloat16
    assert report["dtype_summary"]["torch.float32"]["trainable_parameters"] == 4
    assert report["dtype_summary"]["torch.bfloat16"]["trainable_parameters"] == 0


def test_vla_trainable_peft_lora_preserves_zero_dropout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from art_embodied.vla_trainable import configure_vla_trainable_parameters

    class FakeLoraConfig:
        def __init__(self, **kwargs) -> None:
            self.r = kwargs["r"]
            self.lora_alpha = kwargs["lora_alpha"]
            self.lora_dropout = kwargs["lora_dropout"]
            self.bias = kwargs["bias"]
            self.target_modules = kwargs["target_modules"]
            self.modules_to_save = kwargs["modules_to_save"]
            self.init_lora_weights = kwargs["init_lora_weights"]

    def fake_get_peft_model(
        model: torch.nn.Module, _config: FakeLoraConfig
    ) -> torch.nn.Module:
        model.lora_B = torch.nn.Parameter(torch.zeros(1))
        model.lora_B.requires_grad_(True)
        return model

    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(LoraConfig=FakeLoraConfig, get_peft_model=fake_get_peft_model),
    )

    class FakePolicy:
        def __init__(self) -> None:
            self.model = torch.nn.Module()
            self.model.q_proj = torch.nn.Linear(2, 2)

        def named_parameters(self, *args, **kwargs):
            return self.model.named_parameters(*args, **kwargs)

        def named_modules(self, *args, **kwargs):
            return self.model.named_modules(*args, **kwargs)

    _policy, report = configure_vla_trainable_parameters(
        FakePolicy(),
        algorithm_cfg={
            "trainable_parameter_strategy": "openvla_oft_lora",
            "force_trainable_float32": False,
            "peft": {
                "enabled": True,
                "r": 32,
                "lora_alpha": 32,
                "lora_dropout": 0.0,
                "target_modules": "q_proj",
                "modules_to_save": [],
                "init_lora_weights": "gaussian",
            },
        },
        policy_type="openvla_oft",
    )

    assert report["ok"] is True
    assert report["method"] == "peft_lora"
    assert report["peft"]["lora_dropout"] == 0.0
    assert report["peft"]["initialization"] == {
        "ok": True,
        "checked_tensors": 1,
        "max_abs": 0.0,
        "nonzero_name_sample": [],
        "error": None,
    }


def test_vla_trainable_rejects_nonzero_fresh_lora_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from art_embodied.vla_trainable import configure_vla_trainable_parameters

    class FakeLoraConfig:
        def __init__(self, **kwargs) -> None:
            for name, value in kwargs.items():
                setattr(self, name, value)

    def fake_get_peft_model(
        model: torch.nn.Module, _config: FakeLoraConfig
    ) -> torch.nn.Module:
        model.lora_B = torch.nn.Parameter(torch.ones(1))
        return model

    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(LoraConfig=FakeLoraConfig, get_peft_model=fake_get_peft_model),
    )

    class FakePolicy:
        def __init__(self) -> None:
            self.model = torch.nn.Module()
            self.model.q_proj = torch.nn.Linear(2, 2)

        def named_parameters(self, *args, **kwargs):
            return self.model.named_parameters(*args, **kwargs)

        def named_modules(self, *args, **kwargs):
            return self.model.named_modules(*args, **kwargs)

    _policy, report = configure_vla_trainable_parameters(
        FakePolicy(),
        algorithm_cfg={
            "trainable_parameter_strategy": "openvla_oft_lora",
            "force_trainable_float32": False,
            "peft": {
                "enabled": True,
                "r": 8,
                "lora_alpha": 8,
                "lora_dropout": 0.0,
                "target_modules": "q_proj",
                "modules_to_save": [],
                "init_lora_weights": "gaussian",
            },
        },
        policy_type="openvla_oft",
    )

    assert report["ok"] is False
    assert "changed the policy at attachment time" in report["error"]


def test_openvla_policy_checkpoint_ref_can_point_to_peft_adapter() -> None:
    policy = OpenVLAPolicy(model_id="fake/openvla-oft", device="cpu", dtype="float32")
    policy.model = FakeOpenVLAOFTModel()
    policy.processor = FakeProcessor()

    policy.load_checkpoint({"peft_adapter_path": "/tmp/fake-openvla-adapter"})

    assert policy.peft_adapter_path == "/tmp/fake-openvla-adapter"
    assert policy.model is not None
    assert policy.processor is not None


def test_openvla_policy_checkpoint_is_self_describing_peft_adapter(tmp_path) -> None:
    class FakeAdapterModel:
        peft_config = {"default": object()}

        def save_pretrained(self, path) -> None:
            (path / "adapter_config.json").write_text("{}", encoding="utf-8")
            (path / "adapter_model.safetensors").write_bytes(b"adapter")

    policy = OpenVLAPolicy(
        model_id="organization/openvla-oft",
        revision="0123456789abcdef",
        device="cpu",
        dtype="float32",
        model_loader="native",
    )
    policy.model = FakeAdapterModel()
    checkpoint = tmp_path / "checkpoint"

    reference = policy.save_checkpoint(str(checkpoint))
    manifest = json.loads(
        (checkpoint / "art_embodied_checkpoint.json").read_text(encoding="utf-8")
    )

    assert reference["type"] == "openvla-peft-adapter"
    assert reference["path"] == str(checkpoint)
    assert reference["peft_adapter_path"] == str(checkpoint)
    assert manifest == {
        "base_model_id": "organization/openvla-oft",
        "base_model_revision": "0123456789abcdef",
        "model_loader": "native",
        "schema_version": 1,
        "type": "openvla-peft-adapter",
    }


def test_openvla_policy_loads_adapter_only_path_against_original_base(
    tmp_path,
    monkeypatch,
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    policy = OpenVLAPolicy(
        model_id="organization/openvla-oft",
        revision="0123456789abcdef",
        device="cpu",
        dtype="float32",
        model_loader="native",
    )
    loads: list[bool] = []

    def load(self) -> None:
        loads.append(True)
        self.model = object()

    monkeypatch.setattr(OpenVLAPolicy, "load", load)

    policy.load_checkpoint({"path": str(adapter)})

    assert loads == [True]
    assert policy.model_id == "organization/openvla-oft"
    assert policy.revision == "0123456789abcdef"
    assert policy.peft_adapter_path == str(adapter)


def test_openvla_policy_adopts_manifest_revision_before_cold_adapter_load(
    tmp_path,
    monkeypatch,
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "art_embodied_checkpoint.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "type": "openvla-peft-adapter",
                "base_model_id": "organization/openvla-oft",
                "base_model_revision": "0123456789abcdef",
                "model_loader": "native",
            }
        ),
        encoding="utf-8",
    )
    policy = OpenVLAPolicy(
        model_id="organization/openvla-oft",
        revision=None,
        device="cpu",
        dtype="float32",
        model_loader="native",
    )

    monkeypatch.setattr(
        OpenVLAPolicy, "load", lambda self: setattr(self, "model", object())
    )

    policy.load_checkpoint(adapter)

    assert policy.revision == "0123456789abcdef"


def test_openvla_policy_rejects_adapter_for_different_base_model(tmp_path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "art_embodied_checkpoint.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "type": "openvla-peft-adapter",
                "base_model_id": "organization/different-base",
                "base_model_revision": "0123456789abcdef",
                "model_loader": "native",
            }
        ),
        encoding="utf-8",
    )
    policy = OpenVLAPolicy(
        model_id="organization/openvla-oft",
        revision="0123456789abcdef",
        device="cpu",
        dtype="float32",
        model_loader="native",
    )

    with pytest.raises(ValueError, match="base model mismatch"):
        policy.load_checkpoint(adapter)


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ("{not-json", "Invalid OpenVLA checkpoint manifest JSON"),
        (json.dumps([]), "manifest must be a JSON object"),
        (
            json.dumps(
                {
                    "schema_version": 2,
                    "type": "openvla-peft-adapter",
                    "base_model_id": "organization/openvla-oft",
                    "base_model_revision": "0123456789abcdef",
                    "model_loader": "native",
                }
            ),
            "Unsupported OpenVLA checkpoint manifest schema_version",
        ),
    ],
)
def test_openvla_policy_rejects_invalid_adapter_manifest(
    tmp_path,
    manifest,
    message,
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "art_embodied_checkpoint.json").write_text(
        manifest,
        encoding="utf-8",
    )
    policy = OpenVLAPolicy(
        model_id="organization/openvla-oft",
        revision="0123456789abcdef",
        device="cpu",
        dtype="float32",
        model_loader="native",
    )

    with pytest.raises(ValueError, match=message):
        policy.load_checkpoint(adapter)


def test_openvla_policy_rejects_oversized_adapter_manifest(tmp_path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "art_embodied_checkpoint.json").write_text(
        " " * (64 * 1024 + 1),
        encoding="utf-8",
    )
    policy = OpenVLAPolicy(
        model_id="organization/openvla-oft",
        revision="0123456789abcdef",
        device="cpu",
        dtype="float32",
        model_loader="native",
    )

    with pytest.raises(ValueError, match="64 KiB safety limit"):
        policy.load_checkpoint(adapter)


def test_openvla_policy_loads_full_checkpoint_path_as_new_base(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "full-model"
    checkpoint.mkdir()
    policy = OpenVLAPolicy(
        model_id="organization/openvla-oft",
        revision="0123456789abcdef",
        device="cpu",
        dtype="float32",
        model_loader="native",
    )
    loads: list[bool] = []

    def load(self) -> None:
        loads.append(True)
        self.model = object()

    monkeypatch.setattr(OpenVLAPolicy, "load", load)

    policy.load_checkpoint(checkpoint)

    assert loads == [True]
    assert policy.model_id == str(checkpoint)
    assert policy.revision is None
