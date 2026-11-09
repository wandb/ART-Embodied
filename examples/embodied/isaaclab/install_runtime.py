"""Install isolated controller, Newton/Arena or OpenPI DROID runtimes.

Requires uv and Git. Does not install Isaac Sim, download model weights, or
modify an existing environment. GPU conformance is a separate step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
from urllib.request import urlopen

SOURCES = {
    "isaaclab": (
        "isaac-sim/IsaacLab",
        "e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4",
        "3c85f8a2914bd0e8ef75cd7c4a739e5f6546a055f6c8b9e12e19bd4c9811a508",
    ),
    "arena": (
        "isaac-sim/IsaacLab-Arena",
        "668bb80b97e8721e47e90fa8e305b4cd97369d79",
        "3139d3e70acb30e16592dd6ae6506ca22f432b29f87ecee82f26af3477ef3202",
    ),
    "openpi": (
        "Physical-Intelligence/openpi",
        "c23745b5ad24e98f66967ea795a07b2588ed6c79",
        "ddb5e2f7e16ced84fb89a6104622ce1da5a4b9b1cb278e85eedf372a894963dd",
    ),
}
CUROBO_REVISION = "78fd485fa82d9b9a063fb4985e371814587e666a"

ARENA_NEWTON_PATCHES = {
    "isaaclab_arena/utils/phyx_utils.py": (
        "eb3b56bc4e8c2b3afabd32ac321bf37906cbe18a66a9882dcfc0b8e6710868c3",
        (
            ("from pxr import PhysxSchema, Usd", "from pxr import Usd"),
            (
                "    cr_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)",
                "    from pxr import PhysxSchema\n\n"
                "    cr_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)",
            ),
        ),
    ),
    "isaaclab_arena/utils/cameras.py": (
        "a949a8b4430c83cfa9873d64e8fbfcc6fc40910277b71b4148393ee574acc961",
        (
            (
                "        from isaaclab_arena.utils.isaaclab_utils.untiled_isaac_rtx_renderer import (",
                "        if self._use_tiled_camera and all(\n"
                '            getattr(self, name).renderer_cfg.renderer_type == "newton_warp"\n'
                "            for name in self.camera_names()\n"
                "        ):\n"
                "            return self.copy()\n"
                "        from isaaclab_arena.utils.isaaclab_utils.untiled_isaac_rtx_renderer import (",
            ),
        ),
    ),
    "isaaclab_arena/utils/usd/helpers.py": (
        "012671994276639cddcd08aae11d21520dd794a454984884b0a969cc482e5d81",
        (
            (
                "    stage = Usd.Stage.Open(path)",
                '    if str(path).startswith(("https://", "http://", "omniverse://")):\n'
                "        from isaaclab.utils.assets import retrieve_file_path\n"
                "        path = retrieve_file_path(path)\n"
                "    stage = Usd.Stage.Open(path)",
            ),
        ),
    ),
}


def patch_arena_for_newton(root):
    """Adapt the pinned Arena source to kit-less rendering and asset loading.

    Keep PhysX-only imports lazy, retain Warp camera configurations, and use
    Lab's downloader instead of requiring Omniverse's USD resolver plugin.
    Validate every input before writing; never patch a different Arena revision.
    """
    prepared, hashes = {}, {}
    for name, (expected, replacements) in ARENA_NEWTON_PATCHES.items():
        path = root / name
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != expected:
            raise ValueError(
                f"Unexpected Arena source for Newton compatibility: {name}"
            )
        text = source.decode()
        for before, after in replacements:
            if text.count(before) != 1:
                raise ValueError(f"Ambiguous Arena compatibility replacement: {name}")
            text = text.replace(before, after, 1)
        prepared[path] = text
        hashes[name] = {
            "before": expected,
            "after": hashlib.sha256(text.encode()).hexdigest(),
        }
    for path, text in prepared.items():
        path.write_text(text)
    return hashes


def extract_source(archive, destination, *, sha256):
    with Path(archive).open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != sha256:
        raise ValueError(f"Source archive checksum mismatch: {archive}")
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        with tarfile.open(archive) as source:
            source.extractall(temp, filter="data")
        roots = list(Path(temp).iterdir())
        if len(roots) != 1 or not roots[0].is_dir():
            raise ValueError("Expected one source directory in the archive")
        roots[0].rename(destination)


def fetch_source(name, root):
    repository, revision, sha256 = SOURCES[name]
    archive = root / f"{name}.tar.gz"
    with (
        urlopen(
            f"https://codeload.github.com/{repository}/tar.gz/{revision}", timeout=120
        ) as response,
        archive.open("xb") as output,
    ):
        shutil.copyfileobj(response, output)
    destination = root / name
    extract_source(archive, destination, sha256=sha256)
    return destination


def simulator_project(lab, arena, openpi):
    """Resolve Arena with the kit-less Lab sources in a single environment."""
    packages = {}
    for directory in sorted((lab / "source").iterdir()):
        metadata = directory / "pyproject.toml"
        if metadata.is_file():
            name = tomllib.loads(metadata.read_text())["project"]["name"]
            packages[name.replace("_", "-")] = directory
    if "isaaclab" not in packages or "isaaclab-newton" not in packages:
        raise ValueError("Isaac Lab source packages are missing")
    packages.update(
        {
            "isaaclab-dev": lab,
            "isaaclab-arena": arena,
            "openpi-client": openpi / "packages/openpi-client",
        }
    )
    dependencies = list(packages) + [
        "coacd==1.0.14",
        "pin==3.9.0",
        "dex-retargeting==0.5.0",
        "cmeel-urdfdom==4.0.1",
        "setuptools==84.1.0",
    ]
    # Arena uses the same NumPy override for openpi-client. Keep the IK binary
    # stack coherent instead of allowing a later install to downgrade pin.
    overrides = [
        "numpy==2.3.1",
        "torch==2.11.0",
        "torchvision==0.26.0",
        "newton[sim]==1.6.1",
        "mujoco==3.12.0",
        "mujoco-warp==3.12.0",
        "warp-lang==1.17.0",
        "usd-exchange==3.0.1",
        "websockets==16.1.1",
        # Torch's setuptools cap predates the patched packaging release.
        "setuptools==84.1.0",
    ]
    sources = [
        f"{json.dumps(name)} = {{path = {json.dumps(str(path))}, editable = true}}"
        for name, path in packages.items()
    ]
    sources += [
        'torch = {index = "pytorch-cu130"}',
        'torchvision = {index = "pytorch-cu130"}',
        'isaacteleop = {index = "pypi"}',
        'mujoco-usd-converter = {index = "pypi"}',
        'nvidia-curobo = {git = "https://github.com/NVlabs/curobo.git", '
        f'rev = "{CUROBO_REVISION}"}}',
    ]
    return (
        '[project]\nname = "art-embodied-droid-simulator"\nversion = "0.0.0"\n'
        'requires-python = ">=3.12,<3.13"\n'
        f"dependencies = {json.dumps(dependencies)}\n"
        '[tool.uv]\npackage = false\nexclude-dependencies = ["argparse"]\n'
        "environments = [\"sys_platform == 'linux' and platform_machine == 'x86_64'\"]\n"
        'prerelease = "if-necessary"\n'
        f"override-dependencies = {json.dumps(overrides)}\n"
        "[tool.uv.sources]\n" + "\n".join(sources) + "\n"
        '[[tool.uv.index]]\nurl = "https://pypi.nvidia.com"\n'
        '[[tool.uv.index]]\nname = "pypi"\nurl = "https://pypi.org/simple"\ndefault = true\n'
        '[[tool.uv.index]]\nname = "pytorch-cu130"\n'
        'url = "https://download.pytorch.org/whl/cu130"\nexplicit = true\n'
    )


def source_environment():
    env = dict(os.environ, UV_LINK_MODE="copy")
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONPATH", None)
    # OpenPI's pinned LeRobot dependency needs its Python source, not LFS data.
    # Limit these Git settings to this installation and its subprocesses.
    pairs = [
        ("filter.lfs.process", ""),
        ("filter.lfs.smudge", "cat"),
        ("filter.lfs.clean", "cat"),
        ("filter.lfs.required", "false"),
    ]
    count = int(env.get("GIT_CONFIG_COUNT", "0"))
    for index, (key, value) in enumerate(pairs, count):
        env[f"GIT_CONFIG_KEY_{index}"] = key
        env[f"GIT_CONFIG_VALUE_{index}"] = value
    env["GIT_CONFIG_COUNT"] = str(count + len(pairs))
    return env


def install(runtime, directory):
    if runtime not in ("controller", "simulator", "policy"):
        raise ValueError(f"Unknown runtime: {runtime}")
    for tool in ("uv", "git"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} is required")
    directory.mkdir(parents=True, exist_ok=False)
    sources = directory / "sources"
    sources.mkdir()
    env = source_environment()
    environment = directory / "environment"
    env["UV_PROJECT_ENVIRONMENT"] = str(environment)
    checkout = Path(__file__).resolve().parents[3]
    openpi = fetch_source("openpi", sources) if runtime != "controller" else None
    arena_patches = {}
    if runtime == "controller":
        subprocess.run(
            [
                "uv",
                "sync",
                "--project",
                str(checkout),
                "--frozen",
                "--extra",
                "observability",
                "--no-dev",
                "--python",
                "3.12",
            ],
            env=env,
            check=True,
        )
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(environment / "bin/python"),
                "--index-url",
                "https://pypi.org/simple",
                "--find-links",
                "https://download.pytorch.org/whl/cpu/torch/",
                "torch==2.13.0+cpu",
                "safetensors==0.8.0",
                "setuptools==84.1.0",
                "fsspec==2026.6.0",
            ],
            env=env,
            check=True,
        )
        probe = (
            "import torch,wandb,weave; from importlib.metadata import version; "
            "from safetensors.torch import load_file; "
            "from examples.embodied.isaaclab.run_droid import DroidRunner; "
            "assert version('openpipe-art')=='0.5.24'; "
            "assert torch.version.cuda is None; print('Controller imports passed')"
        )
    elif runtime == "simulator":
        lab, arena = fetch_source("isaaclab", sources), fetch_source("arena", sources)
        arena_patches = patch_arena_for_newton(arena)
        project = directory / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text(simulator_project(lab, arena, openpi))
        subprocess.run(
            ["uv", "sync", "--project", str(project), "--python", "3.12", "--no-dev"],
            env=env,
            check=True,
        )
        probe = (
            "import argparse,numpy,pinocchio,pink,newton; "
            "from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder; "
            "from isaaclab_arena.scene.scene import Scene; "
            "from art_embodied.integrations.isaaclab_reset_bank import IsaacLabResetBank; "
            "from art_embodied.media import RolloutVideoRecorder; "
            "from examples.embodied.isaaclab.droid_environment import build_droid_scene; "
            "from openpi_client.websocket_client_policy import WebsocketClientPolicy; "
            "assert numpy.__version__=='2.3.1'; "
            "assert pinocchio.__version__=='3.9.0'; "
            "assert 'site-packages' not in argparse.__file__; print('Simulator imports passed')"
        )
    else:
        subprocess.run(
            [
                "uv",
                "sync",
                "--project",
                str(openpi),
                "--frozen",
                "--no-dev",
                "--python",
                "3.11",
            ],
            env=env,
            check=True,
        )
        requirements = checkout / "constraints/isaaclab-openpi-security.txt"
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(environment / "bin/python"),
                "--requirement",
                str(requirements),
            ],
            env=env,
            check=True,
        )
        site = subprocess.check_output(
            [
                str(environment / "bin/python"),
                "-c",
                "import sysconfig; print(sysconfig.get_paths()['purelib'])",
            ],
            env=env,
            text=True,
        ).strip()
        shutil.copytree(
            openpi / "src/openpi/models_pytorch/transformers_replace",
            Path(site) / "transformers",
            dirs_exist_ok=True,
        )
        probe = (
            "import torch,transformers,peft; "
            "from openpi.training.config import get_config; "
            "from art_embodied.backends.factory import make_embodied_backend; "
            "from art_embodied.policies.openpi import make_openpi_flow_policy; "
            "from art_embodied.backends.flow_sde_worker import _build_policy; "
            "import examples.embodied.isaaclab.droid_training; "
            "assert torch.__version__.startswith('2.7.1'); "
            "assert transformers.__version__=='4.53.2'; "
            "get_config('pi05_droid_jointpos_polaris'); print('Policy imports passed')"
        )
    python = environment / "bin/python"
    probe_env = dict(env, PYTHONPATH=f"{checkout / 'src'}:{checkout}")
    subprocess.run([str(python), "-c", probe], env=probe_env, cwd=checkout, check=True)
    packages = subprocess.check_output(
        ["uv", "pip", "list", "--python", str(python), "--format", "json"],
        env=env,
        text=True,
    )
    (directory / "installed-packages.json").write_text(packages)
    (directory / "installation.json").write_text(
        json.dumps(
            {
                "runtime": runtime,
                "arena_newton_patches": arena_patches,
                "python": str(python),
                "sources": {
                    name: value
                    for name, value in SOURCES.items()
                    if runtime == "simulator"
                    or (runtime == "policy" and name == "openpi")
                },
                "controller_lock_sha256": (
                    hashlib.sha256((checkout / "uv.lock").read_bytes()).hexdigest()
                    if runtime == "controller"
                    else None
                ),
                "security_requirements_sha256": (
                    {
                        name: hashlib.sha256(
                            (checkout / "constraints" / name).read_bytes()
                        ).hexdigest()
                        for name in ("isaaclab-openpi-security.txt", "security.txt")
                    }
                    if runtime == "policy"
                    else {}
                ),
                "gpu_verified": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(python)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", choices=("controller", "simulator", "policy"))
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    install(args.runtime, args.directory.expanduser().resolve())


if __name__ == "__main__":
    main()
