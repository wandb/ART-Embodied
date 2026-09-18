#!/usr/bin/env python3
"""Scaffold an auditable real-to-sim Use Case Pack for ART-Embodied.

The initializer captures user intent and evidence before Codex begins editing
simulation code. It does not pretend to compile a finished environment:
task-critical fields are emitted as explicit ``TODO`` values, and the validator
will reject the pack until geometry, robot, task, and runtime contracts have
been resolved. This fail-closed behavior is the distinction between a useful
authoring scaffold and an unverified generated demo.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import shutil
from typing import Any

import yaml

SKILL_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = SKILL_ROOT / "assets" / "use-case-pack"
SCHEMA_ROOT = SKILL_ROOT / "assets" / "schemas"


def _slug(value: str) -> str:
    """Return a stable filesystem-safe identifier for a user-facing name."""

    result = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not result:
        raise ValueError("name must contain at least one letter or digit")
    return result


def _sha256(path: Path) -> str:
    """Hash an input incrementally so large images do not need to fit in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_yaml(path: Path, payload: Any) -> None:
    """Write readable YAML while preserving non-ASCII task instructions."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _reference(
    path_or_url: str,
    *,
    external_upload_authorized: bool,
) -> dict[str, Any]:
    """Normalize one reference without moving or uploading its contents.

    Authorization is recorded as evidence, not acted upon. A later provider
    call must still be explicit, which keeps pack creation itself side-effect
    free.
    """

    value = Path(path_or_url).expanduser()
    if value.is_file():
        return {
            "kind": "local_file",
            "path": str(value.resolve()),
            "sha256": _sha256(value),
            "external_upload_authorized": external_upload_authorized,
        }
    if path_or_url.startswith(("http://", "https://")):
        return {
            "kind": "url",
            "url": path_or_url,
            "sha256": None,
            "external_upload_authorized": external_upload_authorized,
        }
    raise FileNotFoundError(f"reference image does not exist: {path_or_url}")


def build_pack(args: argparse.Namespace) -> Path:
    """Create a new Use Case Pack and return its absolute path.

    All user inputs are validated before the output directory is populated.
    Consequently, a bad scale annotation or missing image cannot leave behind a
    half-initialized directory that a later run mistakes for valid evidence.
    Existing non-empty directories are never overwritten.
    """

    # Resolve evidence first. Filesystem mutation begins only after every input
    # has passed the checks needed to make the manifest reproducible.
    if args.known_scale_m is not None and not args.known_scale_description:
        raise ValueError("--known-scale-description is required with --known-scale-m")
    references = [
        _reference(
            value,
            external_upload_authorized=args.authorize_reference_upload,
        )
        for value in args.reference_image
    ]
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TEMPLATE_ROOT, output, dirs_exist_ok=True)
    shutil.copytree(SCHEMA_ROOT, output / "schemas", dirs_exist_ok=True)

    use_case_id = _slug(args.name)
    created_at = datetime.now(timezone.utc).isoformat()

    # The manifest is the pack's index and provenance anchor. Derived artifacts
    # may change during repair, while these input identities remain stable.
    _write_yaml(
        output / "manifest.yaml",
        {
            "schema_version": 1,
            "id": use_case_id,
            "status": "draft",
            "task_instruction": args.task,
            "robot_profile": args.robot_profile,
            "created_at_utc": created_at,
            "reference_inputs": references,
            "known_scale": (
                {
                    "value_m": args.known_scale_m,
                    "description": args.known_scale_description,
                }
                if args.known_scale_m is not None
                else None
            ),
            "artifacts": {
                "scene_ir": "scene/scene_ir.yaml",
                "task_spec": "task/task_spec.yaml",
                "robot_profile": "robot/robot_profile.yaml",
                "mjcf": None,
                "lerobot_env": "lerobot/env.py",
                "art_embodied_config": "art_embodied/experiment.yaml",
                "validation_report": "validation/report.json",
            },
        },
    )
    _write_yaml(
        output / "scene" / "scene_ir.yaml",
        {
            "schema_version": 1,
            "world": {
                "coordinate_system": "right_handed_z_up",
                "length_unit": "m",
                "mass_unit": "kg",
                "gravity_m_s2": [0.0, 0.0, -9.81],
            },
            "objects": [],
        },
    )
    # TODO markers are part of the contract: they force the structure gate to
    # remain red until Codex has grounded task semantics and robot interfaces.
    _write_yaml(
        output / "task" / "task_spec.yaml",
        {
            "schema_version": 1,
            "id": use_case_id,
            "instruction": args.task,
            "manipulated_objects": ["TODO"],
            "target_regions": ["TODO"],
            "control": {"mode": "TODO", "frequency_hz": 20},
            "initial_state": {"distributions": {}},
            "success": {"predicates": [], "hold_seconds": 0.5},
            "failure": {
                "predicates": [{"op": "episode_timeout"}],
                "timeout_seconds": 10.0,
            },
            "reward": {
                "terminal_success": 1.0,
                "terminal_failure": 0.0,
                "shaping": [],
            },
            "safety": {"constraints": []},
        },
    )
    _write_yaml(
        output / "robot" / "robot_profile.yaml",
        {
            "schema_version": 1,
            "id": args.robot_profile,
            "simulator": "mujoco",
            "asset": {
                "source": "TODO: verified catalog or customer asset",
                "revision": "TODO",
                "path": "TODO",
                "sha256": None,
            },
            "control": {
                "mode": "TODO",
                "frequency_hz": 20,
                "action_order": ["TODO"],
                "units": ["TODO"],
                "limits": [[-1.0, 1.0]],
            },
            "observations": {"TODO": {}},
            "frames": {"base": "TODO", "end_effector": "TODO"},
            "safety": {"workspace_bounds_m": "TODO"},
        },
    )
    _write_yaml(
        output / "physics" / "priors.yaml",
        {
            "schema_version": 1,
            "parameters": {},
            "policy": (
                "Store estimate, range, unit, confidence, source, status, "
                "task_sensitivity, human_override, and calibrated_value."
            ),
        },
    )
    _write_yaml(
        output / "physics" / "overrides.yaml",
        {"schema_version": 1, "human_overrides": {}},
    )
    _write_yaml(
        output / "randomization" / "domain_randomization.yaml",
        {
            "schema_version": 1,
            "training_only": True,
            "parameters": {},
            "fixed_evaluation_manifest": None,
        },
    )
    _write_yaml(
        output / "evidence" / "ledger.yaml",
        {
            "schema_version": 1,
            "entries": [
                {
                    "id": f"user-reference-{index:02d}",
                    "source_type": reference["kind"],
                    "source": reference.get(
                        "path", reference.get("url", reference.get("value"))
                    ),
                    "sha256": reference.get("sha256"),
                    "retrieved_at_utc": created_at,
                    "license": "customer-provided",
                    "redistribution": "not_assessed",
                    "exact_match_confidence": 1.0,
                    "notes": "User-provided task reference.",
                }
                for index, reference in enumerate(references)
            ],
        },
    )
    (output / "inputs").mkdir(exist_ok=True)
    (output / "validation").mkdir(exist_ok=True)
    print(output)
    return output


def build_parser() -> argparse.ArgumentParser:
    """Define the narrow, reproducible input contract for pack creation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--robot-profile", required=True)
    parser.add_argument("--known-scale-m", type=float)
    parser.add_argument("--known-scale-description")
    parser.add_argument("--reference-image", action="append", default=[])
    parser.add_argument(
        "--authorize-reference-upload",
        action="store_true",
        help=(
            "record explicit authorization to send references to an external "
            "provider; this does not upload them"
        ),
    )
    return parser


def main() -> None:
    """Create a pack or exit without presenting a Python traceback to the user."""

    args = build_parser().parse_args()
    if args.known_scale_m is not None and args.known_scale_m <= 0:
        raise SystemExit("--known-scale-m must be positive")
    try:
        build_pack(args)
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
