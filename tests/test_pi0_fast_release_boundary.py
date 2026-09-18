"""Pruning research files must not strand the retained entrypoint imports."""

import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _local_module_exists(name):
    path = ROOT / name.replace(".", "/")
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


def test_retained_fast_entrypoints_have_their_local_imports():
    paths = list((ROOT / "examples/embodied").glob("pi0_fast*.py"))
    paths += list((ROOT / "examples/embodied/libero").glob("*.py"))
    paths += list((ROOT / "examples/embodied/libero_plus").glob("*.py"))
    missing = []
    for path in paths:
        package = ".".join(path.relative_to(ROOT).parent.parts)
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    module = importlib.util.resolve_name(
                        "." * node.level + module, package,
                    )
                modules = [module]
                # `from . import module` has no function/attribute ambiguity.
                if node.module is None:
                    modules = [module + "." + item.name for item in node.names]
            for module in modules:
                if module.startswith("examples.") and not _local_module_exists(module):
                    missing.append((str(path.relative_to(ROOT)), module))
    assert not missing, missing


def test_result_guides_and_readmes_have_no_archived_local_links():
    import re

    paths = list(ROOT.glob("README*.md")) + [
        ROOT / "docs/experimental/pi0-fast-long-result.md",
        ROOT / "docs/experimental/libero-reset-health.md",
    ]
    missing = []
    for path in paths:
        for target in re.findall(r"\]\(([^)\s]+)\)", path.read_text()):
            if ":" in target or target.startswith("#"):
                continue
            if not (path.parent / target.split("#", 1)[0]).exists():
                missing.append((str(path.relative_to(ROOT)), target))
    assert not missing, missing
