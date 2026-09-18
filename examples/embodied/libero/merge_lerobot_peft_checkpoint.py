"""Merge a LeRobot PEFT policy into a standalone ART-compatible checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def merge_lerobot_peft_checkpoint(
    *,
    adapter_path: Path,
    output_path: Path,
    device: str,
) -> dict[str, Any]:
    """Merge one LeRobot adapter without changing its policy processors.

    LeRobot wraps the complete policy with PEFT, whereas ART-Embodied attaches
    online-RL adapters to the policy's inner flow model. Materializing a merged
    warm start is therefore the unambiguous boundary between offline PEFT SFT
    and a fresh ART adapter.
    """

    adapter_path = adapter_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite merged checkpoint: {output_path}")
    for filename in ("adapter_config.json", "adapter_model.safetensors", "config.json"):
        if not (adapter_path / filename).is_file():
            raise FileNotFoundError(adapter_path / filename)

    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from peft import PeftConfig, PeftModel

    peft_config = PeftConfig.from_pretrained(str(adapter_path))
    base_model_id = str(peft_config.base_model_name_or_path or "")
    if not base_model_id:
        raise ValueError("PEFT adapter does not declare a base policy")
    policy_config = PreTrainedConfig.from_pretrained(str(adapter_path))
    if getattr(policy_config, "type", None) != "smolvla":
        raise ValueError("Only SmolVLA PEFT checkpoints are supported")
    policy_config.device = device
    policy_config.compile_model = False
    policy_config.use_peft = False
    base_policy = SmolVLAPolicy.from_pretrained(
        base_model_id,
        config=policy_config,
        strict=True,
    )
    wrapped = PeftModel.from_pretrained(
        base_policy,
        str(adapter_path),
        config=peft_config,
        is_trainable=False,
    )
    merged = wrapped.merge_and_unload(safe_merge=True)
    merged.config.use_peft = False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}-merge-",
            dir=output_path.parent,
        )
    )
    try:
        merged.save_pretrained(temporary_path)
        copied_processors = _copy_processor_artifacts(adapter_path, temporary_path)
        merged_model_files = sorted(temporary_path.glob("model*.safetensors"))
        if not merged_model_files:
            raise FileNotFoundError(
                "Merged policy did not produce any model safetensors files"
            )
        manifest = {
            "schema_version": 1,
            "kind": "merged_lerobot_peft_warm_start",
            "base_model_id": base_model_id,
            "adapter_path": str(adapter_path),
            "adapter_sha256": _file_sha256(adapter_path / "adapter_model.safetensors"),
            "adapter_config_sha256": _file_sha256(
                adapter_path / "adapter_config.json"
            ),
            "source_policy_config_sha256": _file_sha256(
                adapter_path / "config.json"
            ),
            "merged_model_files": {
                path.name: _file_sha256(path) for path in merged_model_files
            },
            "copied_processor_artifacts": copied_processors,
        }
        (temporary_path / "art_embodied_peft_merge.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.rename(output_path)
    except BaseException:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise
    return manifest


def _copy_processor_artifacts(source: Path, destination: Path) -> list[str]:
    copied: list[str] = []
    for path in sorted(source.glob("policy_*")):
        if not path.is_file():
            continue
        shutil.copy2(path, destination / path.name)
        copied.append(path.name)
    if not copied:
        raise FileNotFoundError(f"No policy processor artifacts found in {source}")
    return copied


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = _parse_args()
    result = merge_lerobot_peft_checkpoint(
        adapter_path=args.adapter,
        output_path=args.output,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
