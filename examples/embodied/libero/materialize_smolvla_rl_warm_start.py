"""Materialize an ART SmolVLA LoRA snapshot as a standalone LeRobot policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

from art_embodied import EmbodiedExperimentConfig, make_policy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def materialize_smolvla_rl_warm_start(
    *,
    config_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    device: str,
) -> dict[str, Any]:
    """Merge one online-RL adapter while preserving its LeRobot processors.

    Changing LoRA rank or target modules is not an exact optimizer resume: both
    the trainable tensors and Adam state change shape. Materializing the current
    policy first preserves its behavior, after which a larger zero-delta LoRA
    can be attached without pretending that the optimizer state is compatible.
    """

    config_path = config_path.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite warm start: {output_path}")
    policy_snapshot = checkpoint_path / "policy"
    if not policy_snapshot.is_dir():
        raise FileNotFoundError(
            f"Checkpoint policy snapshot is missing: {policy_snapshot}"
        )

    config = EmbodiedExperimentConfig.from_yaml(config_path)
    if config.policy.type != "smolvla":
        raise ValueError(
            "SmolVLA warm-start materialization requires policy.type='smolvla'"
        )
    config = config.model_copy(
        update={"policy": config.policy.model_copy(update={"device": str(device)})}
    )
    policy = make_policy(config)
    policy.load_checkpoint(policy_snapshot)
    if not hasattr(policy.model, "merge_and_unload"):
        raise TypeError("SmolVLA checkpoint did not produce a mergeable PEFT model")

    merged_model = policy.model.merge_and_unload(safe_merge=True)
    policy.model = merged_model
    policy.policy.config.use_peft = False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}-materialize-",
            dir=output_path.parent,
        )
    )
    try:
        policy.policy.save_pretrained(temporary_path)
        processors = _copy_processor_artifacts(
            Path(config.policy.path).expanduser().resolve(), temporary_path
        )
        model_files = sorted(temporary_path.glob("model*.safetensors"))
        if not model_files:
            raise FileNotFoundError("Materialized SmolVLA policy has no model weights")
        manifest = {
            "schema_version": 1,
            "kind": "art_embodied_smolvla_rl_warm_start",
            "source_config": str(config_path),
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_manifest_sha256": _file_sha256(
                checkpoint_path / "art_embodied_checkpoint_complete.json"
            ),
            "source_adapter_sha256": _file_sha256(
                policy_snapshot / "adapter_model.safetensors"
            ),
            "source_adapter_config_sha256": _file_sha256(
                policy_snapshot / "adapter_config.json"
            ),
            "model_files": {path.name: _file_sha256(path) for path in model_files},
            "processor_files": processors,
            "optimizer_state_carried_forward": False,
        }
        (temporary_path / "art_embodied_smolvla_warm_start.json").write_text(
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
        if path.is_file():
            shutil.copy2(path, destination / path.name)
            copied.append(path.name)
    if not copied:
        raise FileNotFoundError(f"No LeRobot processor artifacts found in {source}")
    return copied


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = _parse_args()
    manifest = materialize_smolvla_rl_warm_start(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device=args.device,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
