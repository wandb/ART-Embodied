"""Convert the native DROID checkpoint with the pinned OpenPI converter."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

CONVERTER_SHA256 = "62980fda1a1b1a09167d413232a2a9295db8370e52b58a70cfba52ef5d88c440"
CONFIG_NAME = "pi05_droid_jointpos_polaris"
CHECKPOINT_URL = "gs://openpi-assets-simeval/pi05_droid_jointpos"
TIED_HEAD = "paligemma_with_expert.paligemma.lm_head.weight"
UNUSED_HEAD = "paligemma_with_expert.gemma_expert.lm_head.weight"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_upstream(source):
    converter_path = source / "examples/convert_jax_model_to_pytorch.py"
    if sha256(converter_path) != CONVERTER_SHA256:
        raise ValueError("Use the OpenPI source installed by install_runtime policy")
    from openpi.training import config

    if not Path(config.__file__).resolve().is_relative_to(source.resolve()):
        raise ValueError("The installed OpenPI package differs from --openpi-source")
    spec = importlib.util.spec_from_file_location("openpi_converter", converter_path)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    model_config = dataclasses.replace(
        config.get_config(CONFIG_NAME).model,
        dtype="float32",
        pytorch_compile_mode=None,
    )
    return converter, model_config


def convert_checkpoint(*, source, checkpoint, output):
    if output.exists():
        raise FileExistsError(output)
    converter, model_config = load_upstream(source)
    if str(checkpoint).startswith("gs://"):
        from openpi.shared.download import maybe_download

        local = Path(maybe_download(str(checkpoint)))
    else:
        local = Path(checkpoint).expanduser().resolve()
    norm = local / "assets/droid/norm_stats.json"
    if not (local / "params").is_dir() or not norm.is_file():
        raise ValueError("Expected native params/ and assets/droid/norm_stats.json")

    import torch

    loads = []
    original_load = torch.nn.Module.load_state_dict

    def audited_load(module, state, *args, **kwargs):
        result = original_load(module, state, *args, **kwargs)
        missing, unexpected = set(result.missing_keys), list(result.unexpected_keys)
        if not state or unexpected or missing - {TIED_HEAD, UNUSED_HEAD}:
            raise ValueError(
                f"Checkpoint key mismatch: missing={missing}, extra={unexpected}"
            )
        pg = module.paligemma_with_expert.paligemma
        if (
            pg.lm_head.weight.data_ptr()
            != pg.model.language_model.embed_tokens.weight.data_ptr()
        ):
            raise ValueError("PaliGemma output head must share the token embeddings")
        if any(
            p.is_floating_point() and p.dtype != torch.float32
            for p in module.parameters()
        ):
            raise ValueError("Conversion must preserve FP32 source weights on disk")
        loads.append({"loaded_keys": len(state), "missing_keys": sorted(missing)})
        return result

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".droid-conversion-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / "checkpoint"
        # The pinned converter selects some PI0.5 keys from the path string.
        # Keep that convention independent of the user's checkpoint directory.
        native = Path(temporary) / "pi05"
        native.symlink_to(local, target_is_directory=True)
        # Conversion runs on CPU. Runtime inference still uses OpenPI's native
        # BF16/FP32 casting. Seed only the unused expert vocabulary head.
        with (
            torch.random.fork_rng(devices=[]),
            patch.object(torch.nn.Module, "load_state_dict", audited_load),
        ):
            torch.manual_seed(0)
            converter.convert_pi0_checkpoint(
                str(native), "float32", str(staging), model_config
            )
        if len(loads) != 1 or not (staging / "model.safetensors").is_file():
            raise ValueError("The converter did not produce one audited model")
        # The upstream converter searches the parent directory for assets.
        # Bind normalization to this checkpoint, even if a sibling has assets.
        if (staging / "assets").exists():
            shutil.rmtree(staging / "assets")
        shutil.copytree(local / "assets", staging / "assets")
        report = {
            "config_name": CONFIG_NAME,
            "source_checkpoint": str(checkpoint),
            "converter_sha256": CONVERTER_SHA256,
            "model_sha256": sha256(staging / "model.safetensors"),
            "normalization_sha256": sha256(norm),
            "storage_precision": "float32",
            "runtime_precision": "native_openpi_bfloat16_float32",
            "unused_head_seed": 0,
            "loads": loads,
        }
        (staging / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-source", type=Path, required=True)
    parser.add_argument("--checkpoint", default=CHECKPOINT_URL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            convert_checkpoint(
                source=args.openpi_source.expanduser().resolve(),
                checkpoint=args.checkpoint,
                output=args.output.expanduser().absolute(),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
