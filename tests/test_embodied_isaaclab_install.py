import hashlib
import io
import os
from pathlib import Path
import tarfile
import tomllib

import pytest

from examples.embodied.isaaclab.install_runtime import (
    CUROBO_REVISION,
    SOURCES,
    extract_source,
    install,
    patch_arena_for_newton,
    simulator_project,
    source_environment,
)


def test_simulator_resolves_native_physics_and_ik_together(tmp_path):
    lab, arena, openpi = [tmp_path / name for name in ("lab", "arena", "openpi")]
    for package in ("isaaclab", "isaaclab_newton", "isaaclab_teleop"):
        path = lab / "source" / package
        path.mkdir(parents=True)
        (path / "pyproject.toml").write_text(f'[project]\nname = "{package}"\n')
    project = tomllib.loads(simulator_project(lab, arena, openpi))
    dependencies = project["project"]["dependencies"]
    assert "isaaclab-dev" in dependencies
    assert "pin==3.9.0" in dependencies
    assert "dex-retargeting==0.5.0" in dependencies
    assert "coacd==1.0.14" in dependencies
    assert "setuptools==84.1.0" in dependencies
    assert all("isaacsim" not in name and "[" not in name for name in dependencies)
    uv = project["tool"]["uv"]
    assert uv["exclude-dependencies"] == ["argparse"]
    assert "numpy==2.3.1" in uv["override-dependencies"]
    assert "setuptools==84.1.0" in uv["override-dependencies"]
    assert uv["sources"]["isaaclab-newton"]["path"] == str(
        lab / "source/isaaclab_newton"
    )
    assert uv["sources"]["nvidia-curobo"]["rev"] == CUROBO_REVISION
    assert uv["sources"]["torch"]["index"] == "pytorch-cu130"


def test_lfs_settings_are_process_local_and_keep_prior_config(monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "http.version")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "HTTP/1.1")
    monkeypatch.setenv("PYTHONPATH", "/unrelated/runtime")
    monkeypatch.setenv("VIRTUAL_ENV", "/unrelated/environment")
    env = source_environment()
    assert env["GIT_CONFIG_COUNT"] == "5"
    assert env["GIT_CONFIG_KEY_0"] == "http.version"
    assert env["GIT_CONFIG_KEY_1"] == "filter.lfs.process"
    assert env["GIT_CONFIG_VALUE_1"] == ""
    assert env["UV_LINK_MODE"] == "copy"
    assert "PYTHONPATH" not in env and "VIRTUAL_ENV" not in env
    assert os.environ["GIT_CONFIG_COUNT"] == "1"
    assert os.environ["PYTHONPATH"] == "/unrelated/runtime"


def test_installer_refuses_existing_directory_before_download(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda tool: f"/bin/{tool}")
    marker = tmp_path / "existing-data"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        install("policy", tmp_path)
    assert marker.read_text() == "keep"


def test_controller_uses_project_lock_and_cpu_torch_without_openpi(
    tmp_path, monkeypatch
):
    from examples.embodied.isaaclab import install_runtime

    calls = []
    monkeypatch.setattr("shutil.which", lambda tool: f"/bin/{tool}")
    monkeypatch.setattr(
        install_runtime.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw))
    )
    monkeypatch.setattr(
        install_runtime.subprocess, "check_output", lambda *a, **kw: "[]"
    )
    monkeypatch.setattr(
        install_runtime,
        "fetch_source",
        lambda *a: pytest.fail("No OpenPI download for controller"),
    )
    destination = tmp_path / "controller"
    install("controller", destination)
    assert len(calls) == 3
    sync, torch_install, probe = [args for args, _ in calls]
    assert "--frozen" in sync and "observability" in sync
    assert "torch==2.13.0+cpu" in torch_install
    assert "safetensors==0.8.0" in torch_install
    assert "fsspec==2026.6.0" in torch_install
    assert "https://download.pytorch.org/whl/cpu/torch/" in torch_install
    assert "https://pypi.org/simple" in torch_install
    assert "setuptools==84.1.0" in torch_install
    assert "torch.version.cuda is None" in probe[-1]
    assert calls[0][1]["env"]["UV_PROJECT_ENVIRONMENT"] == str(
        destination / "environment"
    )
    import json

    report = json.loads((destination / "installation.json").read_text())
    assert not report["sources"] and not report["gpu_verified"]
    assert len(report["controller_lock_sha256"]) == 64


def test_policy_installer_applies_security_requirements_after_upstream_lock(
    tmp_path, monkeypatch
):
    import json

    from examples.embodied.isaaclab import install_runtime

    calls = []
    monkeypatch.setattr("shutil.which", lambda tool: f"/bin/{tool}")
    monkeypatch.setattr(
        install_runtime.subprocess, "run", lambda cmd, **kw: calls.append(cmd)
    )
    monkeypatch.setattr(
        install_runtime.subprocess,
        "check_output",
        lambda cmd, **kw: "[]" if "list" in cmd else str(tmp_path / "site"),
    )
    monkeypatch.setattr(install_runtime, "fetch_source", lambda name, root: root / name)
    monkeypatch.setattr(install_runtime.shutil, "copytree", lambda *a, **kw: None)
    destination = tmp_path / "policy"
    install("policy", destination)
    sync, overlay, probe = calls
    assert "--frozen" in sync
    assert "--requirement" in overlay
    requirements = Path(overlay[overlay.index("--requirement") + 1])
    assert requirements.name == "isaaclab-openpi-security.txt"
    project = tomllib.loads((requirements.parents[1] / "pyproject.toml").read_text())
    assert "constraints/isaaclab-openpi-security.txt" in project["tool"]["hatch"][
        "build"
    ]["targets"]["sdist"]["only-include"]
    assert "4.53.2" in probe[-1]
    lines = set(requirements.read_text().splitlines())
    for requirement in (
        "-c security.txt",
        "torch==2.7.1",
        "transformers==4.53.2",
        "numpy==1.26.4",
        "pillow==13.0.0",
        "diffusers==0.38.0",
        "aiohttp==3.14.4",
        "sentencepiece==0.2.1",
        "safetensors==0.8.0",
        "datasets==5.0.1",
        "fsspec==2026.6.0",
        "setuptools==84.1.0",
    ):
        assert requirement in lines
    report = json.loads((destination / "installation.json").read_text())
    assert report["security_requirements_sha256"][requirements.name] == hashlib.sha256(
        requirements.read_bytes()
    ).hexdigest()


def test_source_archives_are_revision_and_digest_pinned():
    for repository, revision, digest in SOURCES.values():
        assert "/" in repository
        assert len(revision) == 40 and int(revision, 16) > 0
        assert len(digest) == 64 and int(digest, 16) > 0


def test_arena_compatibility_checks_all_sources_before_writing(tmp_path, monkeypatch):
    from examples.embodied.isaaclab import install_runtime

    first, second = tmp_path / "first.py", tmp_path / "second.py"
    first.write_text("old\n")
    second.write_text("unexpected\n")
    expected = hashlib.sha256(b"old\n").hexdigest()
    monkeypatch.setattr(
        install_runtime,
        "ARENA_NEWTON_PATCHES",
        {name: (expected, (("old", "new"),)) for name in ("first.py", "second.py")},
    )
    with pytest.raises(ValueError, match="Unexpected Arena source"):
        patch_arena_for_newton(tmp_path)
    assert first.read_text() == "old\n"
    second.write_text("old\n")
    report = patch_arena_for_newton(tmp_path)
    assert first.read_text() == second.read_text() == "new\n"
    assert report["first.py"] == {
        "before": expected,
        "after": hashlib.sha256(b"new\n").hexdigest(),
    }
    with pytest.raises(ValueError, match="Unexpected Arena source"):
        patch_arena_for_newton(tmp_path)


def test_arena_patch_refuses_ambiguous_replacement(tmp_path, monkeypatch):
    from examples.embodied.isaaclab import install_runtime

    source = b"old old\n"
    (tmp_path / "module.py").write_bytes(source)
    monkeypatch.setattr(
        install_runtime,
        "ARENA_NEWTON_PATCHES",
        {"module.py": (hashlib.sha256(source).hexdigest(), (("old", "new"),))},
    )
    with pytest.raises(ValueError, match="Ambiguous Arena"):
        patch_arena_for_newton(tmp_path)
    assert (tmp_path / "module.py").read_bytes() == source


def test_archive_integrity_checked_before_extracting(tmp_path):
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        info = tarfile.TarInfo("source/file.py")
        info.size = 4
        stream.addfile(info, io.BytesIO(b"test"))
    destination = tmp_path / "installed"
    with pytest.raises(ValueError, match="checksum"):
        extract_source(archive, destination, sha256="0" * 64)
    assert not destination.exists()
    extract_source(
        archive, destination, sha256=hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    assert (destination / "file.py").read_bytes() == b"test"


def test_archive_cannot_escape_destination(tmp_path):
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        info = tarfile.TarInfo("../../escaped")
        info.size = 4
        stream.addfile(info, io.BytesIO(b"test"))
    with pytest.raises(tarfile.FilterError):
        extract_source(
            archive,
            tmp_path / "installed",
            sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        )
