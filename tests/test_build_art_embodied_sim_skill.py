from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = (
    REPOSITORY_ROOT / ".agents" / "skills" / "build-art-embodied-sim"
)


def _load_script(name: str):
    path = SKILL_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"skill_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_yaml(path: Path, payload: object) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _complete_structure(pack: Path) -> None:
    manifest = yaml.safe_load((pack / "manifest.yaml").read_text())
    manifest["status"] = "task_ready"
    manifest["artifacts"]["mjcf"] = "scene/scene.xml"
    _write_yaml(pack / "manifest.yaml", manifest)
    _write_yaml(
        pack / "scene" / "scene_ir.yaml",
        {
            "schema_version": 1,
            "world": {
                "coordinate_system": "right_handed_z_up",
                "length_unit": "m",
                "mass_unit": "kg",
                "gravity_m_s2": [0, 0, -9.81],
            },
            "objects": [
                {
                    "id": "part",
                    "role": "manipulated_object",
                    "dynamic": True,
                    "geometry": {
                        "visual_source": "scene/scene.xml",
                        "collision_source": "primitive_box",
                    },
                }
            ],
        },
    )
    _write_yaml(
        pack / "task" / "task_spec.yaml",
        {
            "schema_version": 1,
            "id": "fixture-task",
            "instruction": "Place the part in the target.",
            "manipulated_objects": ["part"],
            "target_regions": ["target"],
            "control": {"mode": "joint_delta", "frequency_hz": 20},
            "initial_state": {},
            "success": {
                "predicates": [{"op": "inside", "args": ["part", "target"]}],
                "hold_seconds": 0.5,
            },
            "failure": {
                "predicates": [{"op": "episode_timeout"}],
                "timeout_seconds": 10,
            },
            "reward": {},
            "safety": {},
        },
    )
    _write_yaml(
        pack / "robot" / "robot_profile.yaml",
        {
            "schema_version": 1,
            "id": "test-robot",
            "simulator": "mujoco",
            "asset": {
                "source": "test",
                "revision": "1",
                "path": "scene/scene.xml",
                "sha256": None,
            },
            "control": {
                "mode": "joint_delta",
                "frequency_hz": 20,
                "action_order": ["joint"],
                "units": ["rad"],
                "limits": [[-1, 1]],
            },
            "observations": {"joint": {"unit": "rad"}},
            "frames": {"base": "world", "end_effector": "tool"},
            "safety": {"workspace_bounds_m": [[-1, 1], [-1, 1], [0, 2]]},
        },
    )
    _write_yaml(
        pack / "art_embodied" / "experiment.yaml",
        {"schema_version": 1, "use_case_pack": "fixture-task"},
    )
    (pack / "scene" / "scene.xml").write_text(
        """
<mujoco model="fixture">
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="part" pos="0 0 0.1">
      <freejoint/>
      <geom type="box" size="0.02 0.02 0.02" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
""".strip(),
        encoding="utf-8",
    )


def test_scaffold_fails_closed_until_codex_completes_contract(tmp_path: Path) -> None:
    init = _load_script("init_use_case.py")
    pack = tmp_path / "pack"
    init.build_pack(
        SimpleNamespace(
            output=pack,
            name="Fixture Task",
            task="Place the part in the target.",
            robot_profile="test-robot",
            known_scale_m=0.1,
            known_scale_description="part width",
            reference_image=[],
            authorize_reference_upload=False,
        )
    )
    validate = _load_script("validate_use_case.py")
    report = validate.Validator(
        pack, target_level="structure", random_steps=1
    ).run()
    assert report["passed"] is False
    assert any(
        check["name"] == "unresolved_placeholders"
        and check["status"] == "failed"
        for check in report["checks"]
    )


def test_completed_pack_passes_structure_and_mujoco_probe(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")

    init = _load_script("init_use_case.py")
    pack = tmp_path / "pack"
    init.build_pack(
        SimpleNamespace(
            output=pack,
            name="Fixture Task",
            task="Place the part in the target.",
            robot_profile="test-robot",
            known_scale_m=0.1,
            known_scale_description="part width",
            reference_image=[],
            authorize_reference_upload=False,
        )
    )
    _complete_structure(pack)
    validate = _load_script("validate_use_case.py")
    report = validate.Validator(pack, target_level="sim", random_steps=10).run()
    assert report["passed"] is True, json.dumps(report, indent=2)


def test_scaffold_rejects_missing_reference_and_requires_labeled_scale(
    tmp_path: Path,
) -> None:
    init = _load_script("init_use_case.py")
    common = {
        "output": tmp_path / "pack",
        "name": "Fixture Task",
        "task": "Place the part in the target.",
        "robot_profile": "test-robot",
        "authorize_reference_upload": False,
    }
    with pytest.raises(ValueError, match="known-scale-description"):
        init.build_pack(
            SimpleNamespace(
                **common,
                known_scale_m=0.1,
                known_scale_description=None,
                reference_image=[],
            )
        )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        init.build_pack(
            SimpleNamespace(
                **common,
                known_scale_m=None,
                known_scale_description=None,
                reference_image=[str(tmp_path / "missing.jpg")],
            )
        )


def test_skill_has_valid_metadata() -> None:
    skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    metadata = (SKILL_ROOT / "agents" / "openai.yaml").read_text(encoding="utf-8")
    assert "name: build-art-embodied-sim" in skill
    assert "$build-art-embodied-sim" in metadata


def test_scaffold_keeps_references_local_without_provider_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socket

    def reject_network(*args, **kwargs):
        raise AssertionError("scaffolding must not open network connections")

    monkeypatch.setattr(socket, "socket", reject_network)
    reference = tmp_path / "reference.jpg"
    reference.write_bytes(b"local-reference-fixture")
    pack = tmp_path / "pack"
    init = _load_script("init_use_case.py")
    init.build_pack(
        SimpleNamespace(
            output=pack,
            name="Local Task",
            task="Place the part in the target.",
            robot_profile="test-robot",
            known_scale_m=None,
            known_scale_description=None,
            reference_image=[str(reference)],
            authorize_reference_upload=False,
        )
    )
    manifest = yaml.safe_load((pack / "manifest.yaml").read_text())
    assert manifest["reference_inputs"] == [
        {
            "kind": "local_file",
            "path": str(reference.resolve()),
            "sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
            "external_upload_authorized": False,
        }
    ]
    assert not (pack / "provider").exists()
    assert (pack / "lerobot" / "env.py").is_file()
    assert (pack / "art_embodied" / "experiment.yaml").is_file()
