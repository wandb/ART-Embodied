from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from examples.embodied.isaaclab import convert_droid as conversion


@dataclass
class ModelConfig:
    dtype: str = "float32"
    pytorch_compile_mode: str | None = None


def fake_model(*, tied=True):
    model = torch.nn.Module()
    expert = model.paligemma_with_expert = torch.nn.Module()
    pg = expert.paligemma = torch.nn.Module()
    pg.model = torch.nn.Module()
    pg.model.language_model = torch.nn.Module()
    pg.model.language_model.embed_tokens = torch.nn.Embedding(4, 2)
    pg.lm_head = torch.nn.Linear(2, 4, bias=False)
    if tied:
        pg.lm_head.weight = pg.model.language_model.embed_tokens.weight
    expert.gemma_expert = torch.nn.Module()
    expert.gemma_expert.lm_head = torch.nn.Linear(2, 4, bias=False)
    model.action_in_proj = torch.nn.Linear(2, 2)
    return model


def setup_conversion(tmp_path, monkeypatch, *, defect=None):
    checkpoint = tmp_path / "source-checkpoint"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "assets/droid").mkdir(parents=True)
    (checkpoint / "assets/droid/norm_stats.json").write_text('{"test":"correct"}')
    converted = []

    def convert(path, precision, output, config):
        assert Path(path).resolve() == checkpoint
        assert Path(path).name == "pi05"
        assert precision == "float32" and config.dtype == "float32"
        assert config.pytorch_compile_mode is None
        model = fake_model(tied=defect != "untied")
        state = {
            key: torch.full_like(value, 1.0001)
            for key, value in model.state_dict().items()
            if key not in (conversion.TIED_HEAD, conversion.UNUSED_HEAD)
        }
        if defect == "missing":
            state.pop("action_in_proj.weight")
        if defect == "extra":
            state["unexpected"] = torch.zeros(1)
        if defect == "dtype":
            model.bfloat16()
        if defect == "empty":
            state = {}
        if defect != "skipped":
            model.load_state_dict(state, strict=False)
        converted.append(model)
        Path(output).mkdir()
        (Path(output) / "model.safetensors").write_bytes(b"test model")
        # Simulate upstream finding normalization in the wrong parent directory.
        (Path(output) / "assets/droid").mkdir(parents=True)
        (Path(output) / "assets/droid/norm_stats.json").write_text("wrong")

    monkeypatch.setattr(
        conversion,
        "load_upstream",
        lambda source: (SimpleNamespace(convert_pi0_checkpoint=convert), ModelConfig()),
    )
    return checkpoint, converted


def test_conversion_preserves_weights_own_assets_and_restores_loader(
    tmp_path, monkeypatch
):
    checkpoint, converted = setup_conversion(tmp_path, monkeypatch)
    original = torch.nn.Module.load_state_dict
    rng = torch.get_rng_state().clone()
    output = tmp_path / "converted"
    report = conversion.convert_checkpoint(
        source=tmp_path, checkpoint=checkpoint, output=output
    )
    assert torch.nn.Module.load_state_dict is original
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(converted[0].action_in_proj.weight, torch.full((2, 2), 1.0001))
    assert (output / "assets/droid/norm_stats.json").read_text() == '{"test":"correct"}'
    assert report["normalization_sha256"] == conversion.sha256(
        output / "assets/droid/norm_stats.json"
    )
    assert report["loads"][0]["missing_keys"] == sorted(
        [conversion.TIED_HEAD, conversion.UNUSED_HEAD]
    )
    assert (output / "conversion.json").is_file()


@pytest.mark.parametrize(
    "defect", ["missing", "extra", "untied", "dtype", "empty", "skipped"]
)
def test_conversion_fails_closed_and_restores_loader(tmp_path, monkeypatch, defect):
    checkpoint, _ = setup_conversion(tmp_path, monkeypatch, defect=defect)
    original = torch.nn.Module.load_state_dict
    rng = torch.get_rng_state().clone()
    output = tmp_path / "converted"
    with pytest.raises(ValueError):
        conversion.convert_checkpoint(
            source=tmp_path, checkpoint=checkpoint, output=output
        )
    assert torch.nn.Module.load_state_dict is original
    assert torch.equal(torch.get_rng_state(), rng)
    assert not output.exists()
    assert not list(tmp_path.glob(".droid-conversion-*"))


def test_conversion_refuses_existing_output_before_loading(tmp_path, monkeypatch):
    def unexpected(source):
        raise AssertionError("must not load dependencies")

    monkeypatch.setattr(conversion, "load_upstream", unexpected)
    with pytest.raises(FileExistsError):
        conversion.convert_checkpoint(
            source=tmp_path, checkpoint="unused", output=tmp_path
        )


def test_converter_source_hash_checked_before_importing_openpi(tmp_path):
    converter = tmp_path / "examples/convert_jax_model_to_pytorch.py"
    converter.parent.mkdir()
    converter.write_text("raise AssertionError('must not execute')")
    with pytest.raises(ValueError, match="OpenPI source"):
        conversion.load_upstream(tmp_path)
