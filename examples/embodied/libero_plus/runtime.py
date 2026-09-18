"""External runtime contract for the official LIBERO-Plus benchmark."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from typing import Any

from examples.embodied.libero.environment import (
    _ensure_legacy_gym_import,
    _validate_libero_mujoco_abi,
)

LIBERO_PLUS_SOURCE_REVISION = "4976dc30028e805ff8094b55501d532c48fec182"
LIBERO_PLUS_CLASSIFICATION_SHA256 = (
    "faa87cce3e3ba434da01df7c77523a391b5f2912e4774330b0aa1be5f6a999e6"
)
LIBERO_PLUS_ASSET_ARCHIVE_SHA256 = (
    "96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf"
)
_SOURCE_ENV = "ART_EMBODIED_LIBERO_PLUS_ROOT"
_ASSET_PROVENANCE = "art_embodied_libero_plus_assets.json"


def prepare_libero_plus_runtime_paths() -> dict[str, str]:
    """Activate a user-installed, revision-pinned LIBERO-Plus checkout."""

    source_root = _source_root()
    package_root = source_root / "libero"
    benchmark_root = source_root / "libero" / "libero"
    _validate_source(source_root, benchmark_root)
    _activate_package(package_root)

    paths = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(benchmark_root.parent / "datasets"),
        "assets": str(benchmark_root / "assets"),
    }
    runtime_dir = Path(tempfile.mkdtemp(prefix="art-embodied-libero-plus-"))
    config_path = runtime_dir / "config.yaml"
    config_path.write_text(
        "".join(f"{key}: {value}\n" for key, value in paths.items()),
        encoding="utf-8",
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(runtime_dir)
    return {
        "compatibility": "libero_plus_cvpr2026",
        "provider": "libero-plus-source-checkout",
        "source_root": str(source_root),
        "source_revision": LIBERO_PLUS_SOURCE_REVISION,
        "classification_sha256": LIBERO_PLUS_CLASSIFICATION_SHA256,
        "asset_archive_sha256": LIBERO_PLUS_ASSET_ARCHIVE_SHA256,
        "path_source": "external-revision-pinned-checkout",
        "config_path": str(config_path),
        "benchmark_root": str(benchmark_root),
    }


def validate_libero_plus_task_assets(settings: Any) -> dict[str, int | str]:
    """Validate every selected BDDL and official init-state mapping."""

    runtime = prepare_libero_plus_runtime_paths()
    benchmark_root = Path(runtime["benchmark_root"])
    missing_bddl: list[Path] = []
    missing_init: list[Path] = []
    for task_id in settings.task_ids:
        entry = settings.panel_entries[task_id]
        bddl_path = (
            benchmark_root
            / "bddl_files"
            / settings.suite_name
            / f"{entry.base_task}.bddl"
        )
        init_path = (
            benchmark_root
            / "init_files"
            / settings.suite_name
            / f"{entry.base_task}.pruned_init"
        )
        if not bddl_path.is_file():
            missing_bddl.append(bddl_path)
        if not init_path.is_file():
            missing_init.append(init_path)
    if missing_bddl or missing_init:
        formatted = "\n".join(f"- {path}" for path in [*missing_bddl, *missing_init])
        raise RuntimeError(
            "LIBERO-Plus task assets are incomplete; missing files:\n" + formatted
        )
    return {
        "tasks": len(settings.task_ids),
        "bddl_files": len(settings.task_ids),
        "init_state_files": len(
            {settings.panel_entries[task_id].base_task for task_id in settings.task_ids}
        ),
        "provider": runtime["provider"],
        "source_revision": runtime["source_revision"],
    }


def validate_libero_plus_runtime_imports() -> None:
    """Import the pinned simulator only after its external contract passes."""

    prepare_libero_plus_runtime_paths()
    _validate_libero_mujoco_abi()
    _ensure_legacy_gym_import()
    try:
        from skimage.filters import gaussian  # noqa: F401
        from wand.api import library as wand_library
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "LIBERO-Plus official image dependencies are unavailable. Install "
            "the pi0-fast-libero extra and source "
            "scripts/libero-plus-runtime-env.sh."
        ) from exc
    if getattr(wand_library, "MagickMotionBlurImage", None) is None:
        raise RuntimeError(
            "LIBERO-Plus ImageMagick runtime does not expose motion blur support."
        )
    try:
        from libero.libero.envs import OffScreenRenderEnv  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "LIBERO-Plus could not initialize its EGL renderer. Run on a GPU "
            "allocation with the ART-Embodied LIBERO runtime profile."
        ) from exc


def _source_root() -> Path:
    value = os.environ.get(_SOURCE_ENV)
    if not value:
        raise RuntimeError(
            f"{_SOURCE_ENV} is required. Run scripts/setup-libero-plus.sh and "
            "export the resulting checkout path. ART-Embodied does not vendor "
            "the third-party benchmark or its assets."
        )
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"{_SOURCE_ENV} does not name a directory: {root}")
    return root


def _validate_source(source_root: Path, benchmark_root: Path) -> None:
    required = (
        benchmark_root / "bddl_files",
        benchmark_root / "init_files",
        benchmark_root / "assets",
        benchmark_root / "benchmark" / "task_classification.json",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "LIBERO-Plus checkout/assets are incomplete:\n"
            + "\n".join(f"- {path}" for path in missing)
        )
    revision = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != LIBERO_PLUS_SOURCE_REVISION:
        raise RuntimeError(
            "LIBERO-Plus source revision mismatch: "
            f"expected {LIBERO_PLUS_SOURCE_REVISION}, found {revision}"
        )
    classification = benchmark_root / "benchmark" / "task_classification.json"
    digest = hashlib.sha256(classification.read_bytes()).hexdigest()
    if digest != LIBERO_PLUS_CLASSIFICATION_SHA256:
        raise RuntimeError(
            "LIBERO-Plus task classification digest mismatch: "
            f"expected {LIBERO_PLUS_CLASSIFICATION_SHA256}, found {digest}"
        )
    provenance_path = benchmark_root / "assets" / _ASSET_PROVENANCE
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Missing valid LIBERO-Plus asset provenance: {provenance_path}. "
            "Run scripts/setup-libero-plus.sh."
        ) from exc
    if provenance.get("asset_archive_sha256") != LIBERO_PLUS_ASSET_ARCHIVE_SHA256:
        raise RuntimeError("LIBERO-Plus asset archive digest is not the pinned value")


def _activate_package(package_root: Path) -> None:
    conflicting = []
    for name, module in sys.modules.items():
        if name != "libero" and not name.startswith("libero."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is not None and not Path(
            str(module_file)
        ).resolve().is_relative_to(package_root):
            conflicting.append(f"{name}={module_file}")
    if conflicting:
        raise RuntimeError(
            "A different LIBERO provider is already imported in this process: "
            + ", ".join(conflicting)
            + ". Start a fresh process for LIBERO-Plus."
        )
    loaded = sys.modules.get("libero")
    if loaded is None:
        # The upstream checkout uses ``libero`` as a namespace directory and
        # ``libero.libero`` as its package. An installed regular ``libero``
        # package would otherwise silently win namespace resolution.
        namespace = types.ModuleType("libero")
        namespace.__path__ = [str(package_root)]  # type: ignore[attr-defined]
        namespace.__package__ = "libero"
        sys.modules["libero"] = namespace
