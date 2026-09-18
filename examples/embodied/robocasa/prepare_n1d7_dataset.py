"""Prepare legacy GR-1 LeRobot metadata for the GR00T N1.7 loader.

The public GR-1 tabletop parquet columns contain fixed-width numeric vectors,
but their legacy ``info.json`` files label those vectors as ``object``. GR00T
N1.7 only computes normalization statistics for features declared as floating
point. This module creates a metadata-only overlay with the truthful logical
dtype while symlinking the immutable data and video payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any
import zipfile

import yaml

NUMERIC_VECTOR_FEATURES = {
    "observation.state": 44,
    "action": 44,
}
PREPARATION_SCHEMA = "art-embodied.gr00t-n1d7-legacy-lerobot-overlay.v1"
LEGACY_INITIAL_ACTIONS_NAME = "initial_actions.legacy-unsafe.npz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _normalized_info(source_info: Path) -> tuple[dict[str, Any], dict[str, str]]:
    info = json.loads(source_info.read_text(encoding="utf-8"))
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"Dataset has no feature map: {source_info}")

    previous_dtypes: dict[str, str] = {}
    for key, width in NUMERIC_VECTOR_FEATURES.items():
        feature = features.get(key)
        if not isinstance(feature, dict):
            raise ValueError(
                f"Dataset is missing required feature {key!r}: {source_info}"
            )
        if feature.get("shape") != [width]:
            raise ValueError(
                f"Expected {key!r} shape [{width}], found {feature.get('shape')!r}: {source_info}"
            )
        dtype = str(feature.get("dtype"))
        if dtype not in {"object", "float32", "float64"}:
            raise ValueError(f"Unsupported {key!r} dtype {dtype!r}: {source_info}")
        previous_dtypes[key] = dtype
        feature["dtype"] = "float32"
    return info, previous_dtypes


def _quarantine_legacy_initial_actions(meta_dir: Path) -> str:
    """Keep legacy pickle bytes for audit without exposing them to ``np.load``."""

    initial_actions = meta_dir / "initial_actions.npz"
    quarantined = meta_dir / LEGACY_INITIAL_ACTIONS_NAME
    if not initial_actions.exists():
        return "quarantined" if quarantined.exists() else "absent"
    with zipfile.ZipFile(initial_actions) as archive:
        safe_schema_present = "__schema__.npy" in archive.namelist()
    if safe_schema_present:
        return "safe-keyed-npz"
    if quarantined.exists():
        raise FileExistsError(
            f"Legacy initial-actions quarantine already exists: {quarantined}"
        )
    initial_actions.rename(quarantined)
    return "quarantined"


def _validate_existing_overlay(source: Path, target: Path) -> dict[str, Any]:
    provenance_path = target / "art_embodied_preparation.json"
    if not provenance_path.is_file():
        raise FileExistsError(f"Unrecognized prepared dataset directory: {target}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected = {
        "schema": PREPARATION_SCHEMA,
        "source": str(source.resolve()),
        "source_info_sha256": _sha256(source / "meta" / "info.json"),
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(
                f"Prepared dataset provenance mismatch for {key}: "
                f"expected {value!r}, found {provenance.get(key)!r}"
            )
    normalized, _ = _normalized_info(target / "meta" / "info.json")
    actual = json.loads((target / "meta" / "info.json").read_text(encoding="utf-8"))
    if actual != normalized:
        raise ValueError(f"Prepared dataset metadata is not normalized: {target}")
    initial_actions_status = _quarantine_legacy_initial_actions(target / "meta")
    if provenance.get("initial_actions") != initial_actions_status:
        provenance["initial_actions"] = initial_actions_status
        _write_json_atomic(provenance_path, provenance)
    return provenance


def prepare_dataset_overlay(source: Path, target: Path) -> dict[str, Any]:
    """Create or validate one metadata-only N1.7 dataset overlay."""

    source = source.resolve()
    if target.exists():
        return _validate_existing_overlay(source, target)

    for relative in ("meta/info.json", "meta/modality.json", "data", "videos"):
        if not (source / relative).exists():
            raise FileNotFoundError(source / relative)

    target.mkdir(parents=True)
    shutil.copytree(source / "meta", target / "meta")
    for payload in ("data", "videos"):
        (target / payload).symlink_to(source / payload, target_is_directory=True)

    normalized_info, previous_dtypes = _normalized_info(source / "meta" / "info.json")
    _write_json_atomic(target / "meta" / "info.json", normalized_info)
    initial_actions_status = _quarantine_legacy_initial_actions(target / "meta")
    provenance = {
        "schema": PREPARATION_SCHEMA,
        "source": str(source),
        "source_info_sha256": _sha256(source / "meta" / "info.json"),
        "logical_dtype_migration": {
            key: {"from": previous_dtypes[key], "to": "float32"}
            for key in NUMERIC_VECTOR_FEATURES
        },
        "initial_actions": initial_actions_status,
        "payload_mode": "symlink",
    }
    _write_json_atomic(target / "art_embodied_preparation.json", provenance)
    return provenance


def prepare_manifest(
    root: Path, manifest_path: Path, output_root: Path
) -> dict[str, Any]:
    """Prepare every manifest dataset and return an auditable report."""

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    tasks = manifest.get("tasks", [])
    if len(tasks) != 24:
        raise ValueError(f"Expected 24 RoboCasa tasks, found {len(tasks)}")

    prepared = []
    for task in tasks:
        dataset_name = f"gr1_unified.{task['id']}"
        source = root / "LeRobot" / dataset_name
        target = output_root / dataset_name
        provenance = prepare_dataset_overlay(source, target)
        prepared.append(
            {
                "dataset": dataset_name,
                "path": str(target.resolve()),
                "source_info_sha256": provenance["source_info_sha256"],
            }
        )
    return {"schema": PREPARATION_SCHEMA, "datasets": prepared}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    report = prepare_manifest(args.root, args.manifest, args.output_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
