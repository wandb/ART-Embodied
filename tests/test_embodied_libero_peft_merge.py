from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from examples.embodied.libero.merge_lerobot_peft_checkpoint import (
    merge_lerobot_peft_checkpoint,
)


def _install_fake_merge_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_save: bool = False,
) -> None:
    class FakePolicyConfig:
        type = "smolvla"
        device = "cpu"
        compile_model = True
        use_peft = True

        @classmethod
        def from_pretrained(cls, _path: str) -> "FakePolicyConfig":
            return cls()

    class FakePolicy:
        def __init__(self) -> None:
            self.config = FakePolicyConfig()

        @classmethod
        def from_pretrained(
            cls,
            model_id: str,
            *,
            config: FakePolicyConfig,
            strict: bool,
        ) -> "FakePolicy":
            assert model_id == "local/base-policy"
            assert config.use_peft is False
            assert strict is True
            return cls()

        def save_pretrained(self, path: Path) -> None:
            if fail_save:
                raise RuntimeError("injected save failure")
            (path / "model.safetensors").write_bytes(b"merged")
            (path / "config.json").write_text("{}\n", encoding="utf-8")

    class FakePeftConfig:
        base_model_name_or_path = "local/base-policy"

        @classmethod
        def from_pretrained(cls, _path: str) -> "FakePeftConfig":
            return cls()

    class FakePeftModel:
        @classmethod
        def from_pretrained(
            cls,
            policy: FakePolicy,
            _path: str,
            *,
            config: FakePeftConfig,
            is_trainable: bool,
        ) -> "FakePeftModel":
            assert isinstance(policy, FakePolicy)
            assert isinstance(config, FakePeftConfig)
            assert is_trainable is False
            instance = cls()
            instance.policy = policy
            return instance

        def merge_and_unload(self, *, safe_merge: bool) -> FakePolicy:
            assert safe_merge is True
            return self.policy

    modules = {
        "lerobot.configs": SimpleNamespace(PreTrainedConfig=FakePolicyConfig),
        "lerobot.policies.smolvla.modeling_smolvla": SimpleNamespace(
            SmolVLAPolicy=FakePolicy
        ),
        "peft": SimpleNamespace(
            PeftConfig=FakePeftConfig,
            PeftModel=FakePeftModel,
        ),
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(vars(attributes))
        monkeypatch.setitem(sys.modules, name, module)


def _adapter(path: Path) -> Path:
    path.mkdir()
    (path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"adapter")
    (path / "config.json").write_text("{}\n", encoding="utf-8")
    (path / "policy_preprocessor.json").write_text("{}\n", encoding="utf-8")
    (path / "policy_postprocessor.json").write_text("{}\n", encoding="utf-8")
    return path


def test_merge_lerobot_peft_checkpoint_is_atomic_and_records_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_merge_runtime(monkeypatch)
    adapter = _adapter(tmp_path / "adapter")
    output = tmp_path / "merged"

    manifest = merge_lerobot_peft_checkpoint(
        adapter_path=adapter,
        output_path=output,
        device="cpu",
    )

    assert (output / "model.safetensors").read_bytes() == b"merged"
    assert (output / "policy_preprocessor.json").is_file()
    assert manifest["base_model_id"] == "local/base-policy"
    assert len(manifest["adapter_sha256"]) == 64
    assert len(manifest["adapter_config_sha256"]) == 64
    assert len(manifest["source_policy_config_sha256"]) == 64
    assert list(manifest["merged_model_files"]) == ["model.safetensors"]
    assert len(manifest["merged_model_files"]["model.safetensors"]) == 64
    assert (
        json.loads(
            (output / "art_embodied_peft_merge.json").read_text(encoding="utf-8")
        )
        == manifest
    )
    assert not list(tmp_path.glob(".merged-merge-*"))


def test_merge_lerobot_peft_checkpoint_removes_partial_output_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_merge_runtime(monkeypatch, fail_save=True)
    adapter = _adapter(tmp_path / "adapter")
    output = tmp_path / "merged"

    with pytest.raises(RuntimeError, match="injected save failure"):
        merge_lerobot_peft_checkpoint(
            adapter_path=adapter,
            output_path=output,
            device="cpu",
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".merged-merge-*"))
