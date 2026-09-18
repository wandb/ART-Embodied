#!/usr/bin/env python3
"""Validate a Use Case Pack through cumulative real-to-sim readiness gates.

The four levels answer deliberately different questions:

``structure``
    Are the evidence and typed contracts complete enough to inspect?
``sim``
    Does the selected MJCF compile and remain numerically finite at rest?
``lerobot``
    Does the generated environment satisfy the executable EnvHub boundary?
``art``
    Can ART-Embodied resolve the experiment into a fingerprinted configuration?

Each level includes all earlier levels. A passing report is therefore a precise
claim about the requested boundary, not a general assertion that simulation is
physically calibrated, the task is reachable, or reinforcement learning will
succeed. Task-specific probes remain mandatory before an ``RL-ready`` claim.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from typing import Any
import xml.etree.ElementTree as ET

import yaml

LEVELS = {"structure": 1, "sim": 2, "lerobot": 3, "art": 4}
REQUIRED_FILES = (
    "manifest.yaml",
    "scene/scene_ir.yaml",
    "task/task_spec.yaml",
    "robot/robot_profile.yaml",
    "physics/priors.yaml",
    "physics/overrides.yaml",
    "randomization/domain_randomization.yaml",
    "evidence/ledger.yaml",
    "lerobot/env.py",
    "art_embodied/experiment.yaml",
)
SCHEMA_FILES = {
    "manifest.yaml": "use-case-manifest.schema.json",
    "scene/scene_ir.yaml": "scene-ir.schema.json",
    "task/task_spec.yaml": "task-spec.schema.json",
    "robot/robot_profile.yaml": "robot-profile.schema.json",
}
NONFINITE_PATTERN = re.compile(
    r"(?<![A-Za-z])(?:[-+]?nan|[-+]?inf(?:inity)?)(?![A-Za-z])", re.IGNORECASE
)


@dataclass
class Check:
    """One machine-readable readiness assertion."""

    name: str
    level: str
    status: str
    detail: str


class Validator:
    """Accumulate deterministic checks for one immutable view of a pack.

    Warnings remain visible in the report but do not fail a gate. Only an
    explicit ``failed`` check at or below ``target_level`` makes ``passed``
    false; later-level checks are not run and cannot affect the result.
    """

    def __init__(self, root: Path, *, target_level: str, random_steps: int) -> None:
        """Prepare validation without importing or executing pack code."""

        self.root = root
        self.target_level = target_level
        self.random_steps = random_steps
        self.checks: list[Check] = []
        self.documents: dict[str, Any] = {}

    def add(self, name: str, level: str, status: str, detail: str) -> None:
        """Append a check in execution order for readable diagnostics."""

        self.checks.append(Check(name, level, status, detail))

    def run(self) -> dict[str, Any]:
        """Run the cumulative gate and return the serializable report."""

        self._validate_structure()
        if self._requires("sim"):
            self._validate_sim()
        if self._requires("lerobot"):
            self._validate_lerobot()
        if self._requires("art"):
            self._validate_art()
        passed = all(
            check.status != "failed"
            for check in self.checks
            if LEVELS[check.level] <= LEVELS[self.target_level]
        )
        return {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "pack": str(self.root),
            "target_level": self.target_level,
            "passed": passed,
            "checks": [asdict(check) for check in self.checks],
            "summary": {
                status: sum(check.status == status for check in self.checks)
                for status in ("passed", "failed", "skipped", "warning")
            },
        }

    def _requires(self, level: str) -> bool:
        """Return whether ``level`` contributes to the requested readiness claim."""

        return LEVELS[self.target_level] >= LEVELS[level]

    def _validate_structure(self) -> None:
        """Parse required contracts and reject unresolved or ill-typed values."""

        missing = [path for path in REQUIRED_FILES if not (self.root / path).is_file()]
        self.add(
            "required_files",
            "structure",
            "failed" if missing else "passed",
            "missing: " + ", ".join(missing) if missing else "all required files exist",
        )
        for relative in REQUIRED_FILES:
            path = self.root / relative
            if not path.is_file() or path.suffix != ".yaml":
                continue
            try:
                self.documents[relative] = yaml.safe_load(
                    path.read_text(encoding="utf-8")
                )
                self.add(f"yaml:{relative}", "structure", "passed", "parsed")
            except Exception as exc:
                self.add(f"yaml:{relative}", "structure", "failed", str(exc))

        placeholders: list[str] = []
        for relative, document in self.documents.items():
            placeholders.extend(
                f"{relative}:{path}" for path in _placeholder_paths(document)
            )
        self.add(
            "unresolved_placeholders",
            "structure",
            "failed" if placeholders else "passed",
            ", ".join(placeholders[:20]) if placeholders else "none",
        )
        self._validate_schemas()
        self._validate_physics_ranges()

    def _validate_schemas(self) -> None:
        """Apply the schemas shipped inside the pack for portable validation."""

        try:
            import jsonschema
        except ImportError:
            self.add(
                "json_schemas",
                "structure",
                "warning",
                "jsonschema is unavailable; semantic checks still ran",
            )
            return
        failures: list[str] = []
        for document_name, schema_name in SCHEMA_FILES.items():
            document = self.documents.get(document_name)
            schema_path = self.root / "schemas" / schema_name
            if document is None:
                failures.append(f"{document_name}: not parsed")
                continue
            if not schema_path.is_file():
                failures.append(f"{schema_name}: missing")
                continue
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            errors = sorted(
                jsonschema.Draft202012Validator(schema).iter_errors(document),
                key=lambda error: list(error.absolute_path),
            )
            failures.extend(
                f"{document_name}:{'.'.join(map(str, error.absolute_path)) or '<root>'}:"
                f" {error.message}"
                for error in errors
            )
        self.add(
            "json_schemas",
            "structure",
            "failed" if failures else "passed",
            " | ".join(failures[:20]) if failures else "all schemas pass",
        )

    def _validate_physics_ranges(self) -> None:
        """Reject malformed uncertainty bounds before they reach a simulator."""

        failures: list[str] = []
        for relative in (
            "scene/scene_ir.yaml",
            "physics/priors.yaml",
            "randomization/domain_randomization.yaml",
        ):
            for path, value in _walk(self.documents.get(relative)):
                if path.endswith(".range") and isinstance(value, list):
                    if len(value) != 2 or not all(
                        _finite_number(item) for item in value
                    ):
                        failures.append(
                            f"{relative}:{path} must contain two finite numbers"
                        )
                    elif value[0] > value[1]:
                        failures.append(f"{relative}:{path} is reversed")
                if path.endswith(".confidence") and _finite_number(value):
                    if not 0 <= float(value) <= 1:
                        failures.append(f"{relative}:{path} must be in [0,1]")
        self.add(
            "physics_ranges",
            "structure",
            "failed" if failures else "passed",
            " | ".join(failures[:20]) if failures else "finite and ordered",
        )

    def _mjcf_path(self) -> Path | None:
        """Resolve the canonical MJCF without guessing among multiple scenes."""

        manifest = self.documents.get("manifest.yaml")
        configured = None
        if isinstance(manifest, dict):
            artifacts = manifest.get("artifacts")
            if isinstance(artifacts, dict):
                configured = artifacts.get("mjcf")
        if configured:
            return (self.root / str(configured)).resolve()
        candidates = sorted((self.root / "scene").rglob("*.xml"))
        if len(candidates) == 1:
            return candidates[0]
        return None

    def _validate_sim(self) -> None:
        """Compile MJCF and run a zero-control numerical-stability probe.

        This catches malformed XML, invalid model parameters, initial
        penetrations that make the solver explode, and non-finite state. It does
        not establish reachability, grasp quality, parameter calibration, or
        agreement with the physical workcell.
        """

        path = self._mjcf_path()
        if path is None:
            self.add(
                "mjcf_selection",
                "sim",
                "failed",
                "set manifest.artifacts.mjcf or retain exactly one scene XML",
            )
            return
        if not path.is_file():
            self.add("mjcf_selection", "sim", "failed", f"missing: {path}")
            return
        try:
            root = ET.parse(path).getroot()
            if root.tag != "mujoco":
                raise ValueError(f"expected <mujoco>, got <{root.tag}>")
            text = path.read_text(encoding="utf-8")
            if NONFINITE_PATTERN.search(text):
                raise ValueError("MJCF contains NaN or infinity")
            self.add("mjcf_xml", "sim", "passed", str(path))
        except Exception as exc:
            self.add("mjcf_xml", "sim", "failed", str(exc))
            return
        try:
            import mujoco

            model = mujoco.MjModel.from_xml_path(str(path))
            data = mujoco.MjData(model)
            # Zero control isolates model and reset stability from controller
            # quality. Random-action and scripted-task probes are separate gates.
            for _ in range(max(1, self.random_steps)):
                mujoco.mj_step(model, data)
                if not all(math.isfinite(float(value)) for value in data.qpos):
                    raise ValueError("non-finite qpos during zero-control probe")
                if not all(math.isfinite(float(value)) for value in data.qvel):
                    raise ValueError("non-finite qvel during zero-control probe")
            self.add(
                "mujoco_probe",
                "sim",
                "passed",
                f"compiled and stepped {max(1, self.random_steps)} times",
            )
        except ImportError:
            self.add("mujoco_probe", "sim", "failed", "mujoco is not installed")
        except Exception as exc:
            self.add("mujoco_probe", "sim", "failed", str(exc))

    def _validate_lerobot(self) -> None:
        """Execute the generated EnvHub boundary and sample valid transitions.

        Importing ``lerobot/env.py`` executes code from the Use Case Pack. Run
        this gate only for a trusted pack or inside an appropriate sandbox.
        """

        path = self.root / "lerobot" / "env.py"
        try:
            spec = importlib.util.spec_from_file_location(
                "art_embodied_use_case_env", path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            factory = getattr(module, "make_env")
            environment = factory(n_envs=2, use_async_envs=False)
            environment = _select_environment(environment)
            observation, _info = environment.reset(seed=0)
            if observation is None:
                raise ValueError("reset returned no observation")
            success_seen = False
            for _ in range(max(1, self.random_steps)):
                action = environment.action_space.sample()
                result = environment.step(action)
                if not isinstance(result, tuple) or len(result) != 5:
                    raise ValueError(
                        "step must return (obs, reward, terminated, truncated, info)"
                    )
                observation, _reward, terminated, truncated, info = result
                if not isinstance(info, dict) or "is_success" not in info:
                    raise ValueError('every step must expose info["is_success"]')
                success_seen = True
                if _done(terminated) or _done(truncated):
                    observation, _info = environment.reset()
            environment.close()
            if not success_seen:
                raise ValueError("no transition exposed success status")
            self.add(
                "lerobot_env",
                "lerobot",
                "passed",
                f"reset and {max(1, self.random_steps)} steps passed",
            )
        except Exception as exc:
            self.add("lerobot_env", "lerobot", "failed", str(exc))

    def _validate_art(self) -> None:
        """Resolve ART-Embodied configuration and record its stable fingerprint."""

        path = self.root / "art_embodied" / "experiment.yaml"
        try:
            from art_embodied import EmbodiedExperimentConfig

            config = EmbodiedExperimentConfig.from_yaml(path)
            self.add(
                "art_embodied_config",
                "art",
                "passed",
                f"fingerprint={config.fingerprint()}",
            )
        except Exception as exc:
            self.add("art_embodied_config", "art", "failed", str(exc))


def _placeholder_paths(value: Any, path: str = "<root>") -> list[str]:
    """Return JSON-like paths whose string values still contain ``TODO``."""

    found: list[str] = []
    if isinstance(value, str) and "TODO" in value.upper():
        found.append(path)
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(_placeholder_paths(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_placeholder_paths(item, f"{path}[{index}]"))
    return found


def _walk(value: Any, path: str = "<root>"):
    """Yield every nested value together with its JSON-like path."""

    if isinstance(value, dict):
        for key, item in value.items():
            current = f"{path}.{key}"
            yield current, item
            yield from _walk(item, current)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            current = f"{path}[{index}]"
            yield current, item
            yield from _walk(item, current)


def _finite_number(value: Any) -> bool:
    """Recognize finite real scalars while excluding booleans."""

    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _select_environment(value: Any) -> Any:
    """Select one executable environment from supported EnvHub return shapes."""

    if not isinstance(value, dict):
        return value
    if not value:
        raise ValueError("make_env returned an empty mapping")
    suite = next(iter(value.values()))
    if isinstance(suite, dict):
        if not suite:
            raise ValueError("make_env returned an empty task mapping")
        suite = next(iter(suite.values()))
    if isinstance(suite, list | tuple):
        if not suite:
            raise ValueError("make_env returned an empty environment list")
        suite = suite[0]
    return suite


def _done(value: Any) -> bool:
    """Collapse scalar or vector termination flags to one control-flow boolean."""

    if hasattr(value, "any"):
        return bool(value.any())
    return bool(value)


def build_parser() -> argparse.ArgumentParser:
    """Define the progressive validation CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", type=Path)
    parser.add_argument("--level", choices=tuple(LEVELS), default="structure")
    parser.add_argument("--random-steps", type=int, default=1000)
    parser.add_argument("--report", type=Path)
    return parser


def main() -> None:
    """Validate a pack, atomically persist the report, and return gate status."""

    args = build_parser().parse_args()
    if args.random_steps < 1:
        raise SystemExit("--random-steps must be positive")
    root = args.pack.expanduser().resolve()
    report_path = args.report or root / "validation" / "report.json"
    validator = Validator(
        root,
        target_level=args.level,
        random_steps=args.random_steps,
    )
    report = validator.run()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(report_path)
    for check in validator.checks:
        print(f"[{check.status.upper():7}] {check.level}/{check.name}: {check.detail}")
    print(f"report: {report_path}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
