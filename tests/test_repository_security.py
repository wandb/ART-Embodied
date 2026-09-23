"""Maintain least-privilege, pinned CI configuration."""

from pathlib import Path
import re
import tomllib

from packaging.version import Version
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_actions_are_pinned_and_checkout_drops_credentials():
    paths = list((ROOT / ".github/workflows").glob("*.y*ml"))
    assert paths
    for path in paths:
        workflow = yaml.safe_load(path.read_text())
        assert workflow["permissions"] == {"contents": "read"}
        for job in workflow["jobs"].values():
            if "permissions" in job:
                assert job["permissions"] == {"contents": "read"}
            if "uses" in job:
                assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", job["uses"])
            for step in job.get("steps", []):
                if "uses" not in step:
                    continue
                assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", step["uses"])
                if step["uses"].startswith("actions/checkout@"):
                    assert step["with"]["persist-credentials"] is False


def test_codeowners_declares_maintaining_team():
    owners = (ROOT / ".github/CODEOWNERS").read_text().splitlines()
    assert "* @wandb/art-embodied" in owners


def test_anyio_security_floor_in_standard_install_profiles():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "anyio>=4.14.2" in project["tool"]["uv"]["constraint-dependencies"]
    assert "anyio>=4.14.2" in (ROOT / "constraints/security.txt").read_text().splitlines()
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    versions = [p["version"] for p in lock["package"] if p["name"] == "anyio"]
    assert versions
    assert all(Version(version) >= Version("4.14.2") for version in versions)
