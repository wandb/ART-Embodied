"""Keep internal research archives out of the distributed documentation."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def test_no_internal_research_directories():
    for name in ("research", "proposals"):
        assert not (ROOT / "docs" / name).exists()


def test_contributor_scratch_guidance_keeps_internal_records_outside_git():
    guide = (ROOT / "CONTRIBUTING.md").read_text()
    section = guide.split("## Submitting a pull request", 1)[1]
    assert (
        "Keep internal notes, operational records, and private data outside Git."
        in section
    )
    assert "research worktree/branch" not in section
    assert "versioned archival" not in section


def test_contributor_validation_does_not_require_personal_gpu_access():
    guide = (ROOT / "CONTRIBUTING.md").read_text()
    template = (ROOT / ".github/pull_request_template.md").read_text()
    assert "You do not need access to a GPU" in guide
    assert "Maintainers can run" in guide
    assert "GPU access is not required" in template
    assert "H100" not in guide


def test_examples_and_launchers_do_not_embed_personal_mount_paths():
    mount = re.compile(r"/mnt/(?:home|data)/(?!\$)[^\s\"']+")
    for directory in ("examples", "scripts"):
        for path in (ROOT / directory).rglob("*"):
            if path.suffix in {".py", ".sh", ".yaml", ".json"}:
                assert not mount.search(path.read_text()), path.relative_to(ROOT)


def test_public_docs_do_not_link_internal_research():
    paths = list(ROOT.glob("README*.md"))
    paths += list((ROOT / "docs").rglob("*.md"))
    paths += list((ROOT / "docs").rglob("*.mdx"))
    for path in paths:
        for target in re.findall(r"\]\(([^)\s]+)\)", path.read_text()):
            assert not re.search(r"(?:docs|\.\.)/(?:research|proposals)/", target), (
                path,
                target,
            )
