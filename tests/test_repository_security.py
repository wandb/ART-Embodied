"""Maintain least-privilege, pinned CI configuration."""

from pathlib import Path
import re

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
