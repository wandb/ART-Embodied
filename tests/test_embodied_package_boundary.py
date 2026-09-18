from __future__ import annotations

import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

from packaging.requirements import Requirement
import yaml

from art_embodied.compatibility import (
    VALIDATED_OPENVLA_OFT_V01_PACKAGES,
    art_version_for_python,
)

ROOT = Path(__file__).parents[1]
_EMBODIED_ROOT = ROOT / "src/art_embodied"
_INTEGRATION_OWNED_PARTS = {"conformance", "integrations", "policies"}
_FORBIDDEN_GENERIC_IMPORT_ROOTS = {
    "examples",
    "libero",
    "lerobot",
    "rlinf",
    "slurm",
}


def test_example_handoff_paths_do_not_depend_on_personal_mounts() -> None:
    for path in sorted((ROOT / "examples/embodied").glob("*.yaml")):
        config = yaml.safe_load(path.read_text())
        if not isinstance(config, dict):
            continue
        handoff = str(config.get("runtime", {}).get("worker_handoff_dir", ""))
        assert not handoff.startswith(("/mnt/" + "home/", "/mnt/" + "data/")), path.name


def test_dependency_guidance_matches_current_overrides() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    guide = (ROOT / "docs/experimental/dependency-security.md").read_text()
    for entry in project["tool"]["uv"]["override-dependencies"]:
        if not isinstance(entry, str):
            continue
        requirement = Requirement(entry)
        version = next(iter(requirement.specifier)).version
        assert version in guide, requirement.name
    assert "Plain pip" in guide
    assert "GR00T N1.7" in guide
    assert "existing RC2 release assets" in guide


def _absolute_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.add(node.module)
    return imports


def test_generic_runtime_does_not_import_integration_or_scheduler_packages() -> None:
    violations: list[str] = []
    for path in sorted(_EMBODIED_ROOT.rglob("*.py")):
        relative = path.relative_to(_EMBODIED_ROOT)
        if relative.parts[0] in _INTEGRATION_OWNED_PARTS:
            continue
        for module in sorted(_absolute_imports(path)):
            root = module.split(".", 1)[0]
            if root in _FORBIDDEN_GENERIC_IMPORT_ROOTS:
                violations.append(f"{relative}: {module}")

    assert violations == []


def test_product_runtime_never_imports_example_modules() -> None:
    violations = [
        f"{path.relative_to(_EMBODIED_ROOT)}: {module}"
        for path in sorted(_EMBODIED_ROOT.rglob("*.py"))
        for module in sorted(_absolute_imports(path))
        if module == "examples" or module.startswith("examples.")
    ]

    assert violations == []


def test_addon_has_no_static_rlinf_distribution_dependency() -> None:
    """RLinf is an optional conformance oracle, not a product dependency."""

    violations = [
        f"{path.relative_to(_EMBODIED_ROOT)}: {module}"
        for path in sorted(_EMBODIED_ROOT.rglob("*.py"))
        for module in sorted(_absolute_imports(path))
        if module == "rlinf" or module.startswith("rlinf.")
    ]

    assert violations == []


def test_addon_distribution_owns_only_art_embodied_namespace() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = [Requirement(value) for value in project["project"]["dependencies"]]
    art_dependencies = [
        dependency for dependency in dependencies if dependency.name == "openpipe-art"
    ]

    assert project["project"]["name"] == "art-embodied"
    assert project["project"]["license-files"] == ["LICENSE", "THIRD-PARTY-NOTICES"]
    for python_version in ("3.11", "3.12", "3.13"):
        matching = [
            dependency
            for dependency in art_dependencies
            if dependency.marker is None
            or dependency.marker.evaluate({"python_version": python_version})
        ]
        assert len(matching) == 1
        assert str(art_version_for_python(python_version)) in matching[0].specifier
        assert "0.5.18" in matching[0].specifier
        assert ("0.5.20" in matching[0].specifier) is (python_version != "3.11")
        for unqualified in ("0.5.17", "0.5.19", "0.5.21", "0.6.0"):
            assert unqualified not in matching[0].specifier
    assert not any(
        "rlinf" in dependency.lower()
        for dependency in project["project"]["dependencies"]
    )
    assert project["project"]["requires-python"] == ">=3.11"
    assert project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "src/art_embodied"
    ]
    assert (ROOT / "src/art_embodied/__init__.py").is_file()
    assert not (ROOT / "src/art").exists()
    assert not (ROOT / "src/mp_actors").exists()


def test_gr00t_n1d7_installer_retains_native_dependencies() -> None:
    installer = (ROOT / "scripts/install-gr00t-n1d7-runtime.sh").read_text()
    constraints = installer.split("printf '%s\\n'", 1)[1].split(
        '> "${constraints}"', 1
    )[0]
    for requirement in (
        "openpipe-art==0.5.18",
        "scipy==1.15.3",
        "weave==0.52.37",
        "wandb==0.24.2",
    ):
        assert f"'{requirement}'" in constraints
    assert '--constraint "${constraints}"' in installer


def test_diffusers_security_override_matches_lock_and_installer() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    override_path = "constraints/diffusers-security.txt"
    overrides = [
        line
        for line in (ROOT / override_path).read_text().splitlines()
        if line and not line.startswith("#")
    ]
    assert overrides == ["diffusers==0.38.0"]
    assert all(
        item in project["tool"]["uv"]["override-dependencies"] for item in overrides
    )
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    assert {p["version"] for p in lock["package"] if p["name"] == "diffusers"} == {
        "0.38.0"
    }
    assert (
        override_path
        in project["tool"]["hatch"]["build"]["targets"]["sdist"]["only-include"]
    )
    installer = ROOT / "scripts/install-gr00t-n1d7-runtime.sh"
    text = installer.read_text()
    assert '--overrides "${diffusers_override}"' in text
    assert '"diffusers==0.38.0" "safetensors==0.8.0"' in text
    assert "intentional metadata mismatch" in text
    assert 'command -v git-lfs' in text
    assert 'git clone --no-checkout' in text
    assert 'lfs install --local' in text
    assert 'for tool in cmake c++ make' in text
    install_commands = [
        line for line in text.splitlines() if line.startswith("uv pip install ")
    ]
    assert len(install_commands) == 3
    assert all("--no-config" in line for line in install_commands)
    subprocess.run(["bash", "-n", str(installer)], check=True)
    workflow = (ROOT / ".github/workflows/package-install.yml").read_text()
    assert (
        "--overrides constraints/diffusers-security.txt 'diffusers==0.38.0'"
        in workflow
    )


def test_gr00t_installer_rejects_missing_tools_before_cloning(tmp_path) -> None:
    installer = ROOT / "scripts/install-gr00t-n1d7-runtime.sh"
    for missing in ("git-lfs", "cmake", "c++", "make"):
        bin_dir = tmp_path / missing
        bin_dir.mkdir()
        (bin_dir / "dirname").symlink_to(shutil.which("dirname"))
        for tool in ("uv", "git-lfs", "cmake", "c++", "make"):
            if tool != missing:
                (bin_dir / tool).symlink_to(shutil.which("true"))
        result = subprocess.run(
            [shutil.which("bash"), str(installer)],
            env={**os.environ, "PATH": str(bin_dir)},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert ("Git LFS" if missing == "git-lfs" else missing) in result.stderr
        assert "not found" not in result.stderr


def test_readme_languages_share_release_and_art_install_profiles() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    release = project["project"]["version"]
    for name in (
        "README.md",
        "README.ja.md",
        "README.ko.md",
        "README.zh-CN.md",
        "README.zh-TW.md",
    ):
        assert name in project["tool"]["hatch"]["build"]["targets"]["sdist"][
            "only-include"
        ]
        readme = (ROOT / name).read_text()
        assert f"`{release}`" in readme, name
        assert "docs/experimental/upstream-art-compatibility.md" in readme, name
        assert (
            "python -m pip install -c constraints/security.txt 'openpipe-art==0.5.18'"
            in readme
        ), name
        assert (
            "python -m pip install -c constraints/security.txt "
            "'openpipe-art==0.5.20' '.[pi-libero]'"
            in readme
        ), name
        assert "uv sync --python 3.11 --extra libero" in readme, name
        assert "./scripts/install-gr00t-n1d7-runtime.sh" in readme, name
        assert "https://github.com/nejumi/ART-Embodied.git" not in readme, name
        assert "0.1.0rc2.dev0" not in readme, name


def test_security_constraints_match_uv_and_locked_versions() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    lines = {
        line.strip()
        for line in (ROOT / "constraints/security.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert lines == set(project["tool"]["uv"]["constraint-dependencies"])
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    for line in lines:
        requirement = Requirement(line)
        versions = [
            package["version"]
            for package in lock["package"]
            if package["name"] == requirement.name
        ]
        assert versions, requirement.name
        assert all(version in requirement.specifier for version in versions), line
    assert "constraints/security.txt" in project["tool"]["hatch"]["build"][
        "targets"
    ]["sdist"]["only-include"]


def test_lerobot_and_openvla_libero_extras_are_isolated_worker_profiles() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = project["project"]["optional-dependencies"]

    assert extras["lerobot"] == ["lerobot>=0.4.4,<0.5"]
    assert extras["pi"] == ["lerobot[peft,pi]==0.6.0; python_version >= '3.12'"]
    assert extras["pi-libero"] == [
        "lerobot[libero,peft,pi]==0.6.0; python_version >= '3.12'",
        "matplotlib>=3.10.3,<4; python_version >= '3.12'",
        "mujoco==3.8.1; python_version >= '3.12'",
        "numba==0.61.2; python_version >= '3.12'",
        "robosuite==1.4.0; python_version >= '3.12'",
    ]
    assert extras["libero"] == [
        "accelerate==1.14.0",
        "hf-libero==0.1.4",
        "h5py==3.14.0",
        "matplotlib>=3.8,<4",
        "mujoco==3.8.1",
        "numba==0.61.2",
        "numpy==1.26.4",
        "peft==0.11.1",
        "safetensors==0.7.0",
        "sentencepiece==0.2.1",
        "timm==0.9.10",
        "tokenizers==0.19.1",
        "torch==2.6.0",
        "torchvision==0.21.0",
        "transformers==4.40.1",
    ]
    assert [
        {"extra": "lerobot"},
        {"extra": "libero"},
    ] in project["tool"]["uv"]["conflicts"]
    assert project["tool"]["uv"]["extra-build-dependencies"] == {
        "egl-probe": ["cmake>=3.20,<4"],
        "hf-egl-probe": ["cmake>=3.20,<4"],
    }

    packaged_profile: dict[str, str] = {}
    for dependency in extras["libero"]:
        requirement = Requirement(dependency)
        if requirement.name == "matplotlib":
            assert str(requirement.specifier) == "<4,>=3.8"
            continue
        specifiers = list(requirement.specifier)
        assert len(specifiers) == 1
        assert specifiers[0].operator == "=="
        packaged_profile[requirement.name] = specifiers[0].version
    assert packaged_profile == VALIDATED_OPENVLA_OFT_V01_PACKAGES


def test_third_party_notice_describes_addon_distribution_only() -> None:
    notice = (ROOT / "THIRD-PARTY-NOTICES").read_text(encoding="utf-8")

    assert "src/art/" not in notice
    assert "do not vendor" in notice
    assert "required runtime dependency (`openpipe-art`), not vendored" in notice


def test_distribution_has_no_stale_copyleft_notice() -> None:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    notice = (ROOT / "THIRD-PARTY-NOTICES").read_text(encoding="utf-8")

    assert "GNU Lesser General Public License" not in license_text
    assert "licenses/ directory" not in license_text
    assert not (ROOT / "licenses").exists()
    assert (
        "do not contain project-owned or\nvendored GPL- or LGPL-licensed code" in notice
    )


def test_worker_module_paths_use_addon_namespace() -> None:
    rollout_process = (ROOT / "src/art_embodied/rollout_process.py").read_text(
        encoding="utf-8"
    )
    local_backend = (ROOT / "src/art_embodied/backends/local_process.py").read_text(
        encoding="utf-8"
    )

    assert '"art_embodied.rollout_worker"' in rollout_process
    assert '"art_embodied.rollout_inference_worker"' in rollout_process
    assert '"art_embodied.backends.action_token_worker"' in local_backend
    assert "art.embodied" not in rollout_process
    assert "art.embodied" not in local_backend


def test_evaluation_runner_import_does_not_require_art(tmp_path: Path) -> None:
    """Policy/evaluation workers must not initialize ART unless they use it."""

    shadow = tmp_path / "art.py"
    shadow.write_text("raise ImportError('ART control plane is unavailable')\n")
    source_root = Path(__file__).parents[1] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(source_root), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-c", "import art_embodied.runner"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr


def test_lerobot_protocol_import_does_not_require_torch(tmp_path: Path) -> None:
    """Core users must not pay for optional PI policy dependencies."""

    shadow = tmp_path / "torch.py"
    shadow.write_text("raise ImportError('PyTorch is unavailable')\n")
    source_root = ROOT / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(source_root), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from art_embodied.integrations import LeRobotPolicyAdapterProtocol",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
