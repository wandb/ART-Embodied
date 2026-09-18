"""Register a warm-start checkpoint with W&B before allocating GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from art_embodied import EmbodiedExperimentConfig
from art_embodied.utils import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=14_400)
    return parser.parse_args()


def _completion_marker(checkpoint: Path) -> tuple[Path, dict[str, Any], str]:
    marker_path = checkpoint / "art_embodied_sft_complete.json"
    if not marker_path.is_file() or marker_path.stat().st_size > 1024 * 1024:
        raise FileNotFoundError(f"Missing bounded SFT completion marker: {marker_path}")
    marker_bytes = marker_path.read_bytes()
    marker = json.loads(marker_bytes)
    if marker.get("schema_version") != 1 or marker.get("status") != "complete":
        raise ValueError("SFT completion marker is not complete")
    return marker_path, marker, hashlib.sha256(marker_bytes).hexdigest()


def main() -> None:
    args = parse_args()
    config = EmbodiedExperimentConfig.from_yaml(args.config)
    wandb_config = config.observability.wandb
    if not wandb_config.enabled or wandb_config.mode != "online":
        raise ValueError("Input-model registration requires online W&B logging")

    checkpoint = Path(config.policy.path).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint}")
    marker_path, marker, marker_sha256 = _completion_marker(checkpoint)

    import wandb

    run = wandb.init(
        entity=wandb_config.entity,
        project=wandb_config.project,
        name=f"register-{args.artifact_name}",
        job_type="model-artifact-registration",
        config={
            "source_config": str(args.config),
            "source_config_fingerprint": config.fingerprint,
            "checkpoint": str(checkpoint),
            "completion_marker_sha256": marker_sha256,
        },
    )
    if run is None:
        raise RuntimeError("wandb.init returned no run")

    artifact = wandb.Artifact(
        name=args.artifact_name,
        type="model",
        metadata={
            "role": "warm_start_policy",
            "policy_type": config.policy.type,
            "policy_revision": config.policy.revision,
            "completion_marker_path": marker_path.name,
            "completion_marker_sha256": marker_sha256,
            "sft_completion": marker,
        },
    )
    artifact.add_dir(str(checkpoint))
    logged = run.log_artifact(artifact, aliases=["latest"])
    committed = logged.wait(timeout=args.timeout_seconds)
    version = committed.version
    if version is None or not version.startswith("v"):
        raise RuntimeError(f"W&B returned an invalid artifact version: {version!r}")
    artifact_ref = f"{run.entity}/{run.project}/{args.artifact_name}:{version}"
    report = {
        "schema_version": 1,
        "status": "passed",
        "artifact_ref": artifact_ref,
        "artifact_digest": committed.digest,
        "artifact_size_bytes": committed.size,
        "checkpoint": str(checkpoint),
        "completion_marker_sha256": marker_sha256,
        "config": str(args.config),
        "config_fingerprint": config.fingerprint,
        "wandb_run_id": run.id,
        "wandb_run_url": run.url,
    }
    write_json_atomic(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    run.finish(exit_code=0)


if __name__ == "__main__":
    main()
