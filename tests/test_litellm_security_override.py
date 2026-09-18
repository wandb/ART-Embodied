import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

ROOT = Path(__file__).parents[1]


def test_litellm_override_is_scoped_to_the_tested_release() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    overrides = project["tool"]["uv"]["override-dependencies"]
    assert "litellm==1.101.0" in overrides
    assert {
        "package": {"name": "litellm", "version": "1.101.0"},
        "dependencies": ["tokenizers>=0.19.1,<1.0"],
    } in overrides
    # Other packages must retain their own Tokenizers requirements.
    assert not any(
        isinstance(item, str) and item.startswith("tokenizers") for item in overrides
    )
    assert not project["tool"]["uv"].get("exclude-dependencies")
    assert project["tool"]["uv"]["required-version"] == ">=0.12.0"


def test_lock_uses_patched_litellm_without_upgrading_policy_dependencies() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    versions = {}
    for package in lock["package"]:
        versions.setdefault(package["name"], set()).add(package["version"])
    expected = {
        "litellm": {"1.101.0"},
        "openpipe-art": {"0.5.18", "0.5.20"},
        "tokenizers": {"0.19.1", "0.22.2"},
        "transformers": {"4.40.1", "5.5.4"},
        "torch": {"2.6.0", "2.10.0"},
        "lerobot": {"0.4.4", "0.6.0"},
        "safetensors": {"0.7.0", "0.8.0"},
    }
    for name, required in expected.items():
        assert versions[name] == required, name


def test_ci_uses_uv_version_with_scoped_override_support() -> None:
    import yaml

    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/package-install.yml").read_text()
    )
    versions = [
        step["with"]["version"]
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if step.get("uses", "").startswith("astral-sh/setup-uv@")
    ]
    assert versions and all(version == "0.12.0" for version in versions)


def test_art_and_litellm_tokenizer_compatibility_without_network() -> None:
    if importlib.util.find_spec("litellm") is None:
        pytest.skip("Requires the LiteLLM checkout dependency")
    # Exercise imports in a fresh interpreter rather than relying on test caches.
    code = """
import socket
def blocked(*args, **kwargs):
    raise AssertionError("Unexpected network access")
socket.socket.connect = blocked
socket.socket.connect_ex = blocked
import art
import litellm
from tokenizers import Tokenizer, models, pre_tokenizers
tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "pick": 1, "cup": 2}, unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
custom = litellm.create_tokenizer(tokenizer.to_str())
assert litellm.encode(text="pick cup", custom_tokenizer=custom) == [1, 2]
assert litellm.decode(tokens=[1, 2], custom_tokenizer=custom) == "pick cup"
assert litellm.token_counter(text="pick cup", custom_tokenizer=custom) == 2
assert litellm.encode(text="", custom_tokenizer=custom) == []
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "HF_HUB_OFFLINE": "1",
        },
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
