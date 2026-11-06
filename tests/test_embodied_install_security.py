"""Check that orjson source builds fail before executing a build backend."""

from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_orjson_wheel_only_settings():
    settings = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["uv"]
    assert "orjson" in settings["no-build-package"]
    assert "orjson" in settings["pip"]["only-binary"]


@pytest.mark.parametrize("frontend", ["sync", "pip"])
def test_orjson_source_backend_is_not_executed(tmp_path, frontend):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("Requires uv (installed in CI)")
    settings = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["uv"]
    marker = tmp_path / "backend-executed"
    files = {
        "pyproject.toml": (
            '[project]\nname = "orjson"\nversion = "3.13.0"\n'
            '[build-system]\nrequires = []\nbuild-backend = "backend"\n'
            'backend-path = ["."]\n'
        ),
        "PKG-INFO": "Metadata-Version: 2.2\nName: orjson\nVersion: 3.13.0\n\n",
        "backend.py": (
            f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
            'raise RuntimeError("Source backend must not execute")\n'
        ),
    }
    sdist = tmp_path / "orjson-3.13.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        for name, text in files.items():
            data = text.encode()
            entry = tarfile.TarInfo(f"orjson-3.13.0/{name}")
            entry.size = len(data)
            archive.addfile(entry, BytesIO(data))

    requirement = f"orjson @ {sdist.as_uri()}"
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname = "install-guard-test"\nversion = "0.0.0"\n'
        f"dependencies = {json.dumps([requirement])}\n"
        '[tool.uv]\npackage = false\n'
        f"no-build-package = {json.dumps(settings['no-build-package'])}\n"
        '[tool.uv.pip]\n'
        f"only-binary = {json.dumps(settings['pip']['only-binary'])}\n"
    )
    # Do not let caller flags or caches supply the guard being tested.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("UV_", "PIP_"))}
    if frontend == "sync":
        args = ["sync", "--python", sys.executable, "--no-index"]
    else:
        args = [
            "pip", "install", "--python", sys.executable, "--no-index",
            "--no-deps", "--target", str(tmp_path / "installed"), requirement,
        ]
    result = subprocess.run(
        [uv, "--offline", "--no-cache", *args],
        cwd=project, env=env, capture_output=True, text=True, timeout=60,
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    expected = (
        "marked as `--no-build` but has no binary distribution"
        if frontend == "sync" else "Building source distributions is disabled"
    )
    assert expected in output, output
    assert not marker.exists(), output

    # The same fixture must execute its backend when the guard is removed.
    config = project / "pyproject.toml"
    config.write_text("\n".join(
        line for line in config.read_text().splitlines()
        if not line.startswith(("no-build-package =", "only-binary ="))
    ) + "\n")
    control = subprocess.run(
        [uv, "--offline", "--no-cache", *args],
        cwd=project, env=env, capture_output=True, text=True, timeout=60,
    )
    assert control.returncode != 0
    assert marker.exists(), control.stdout + control.stderr
