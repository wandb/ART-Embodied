"""Exercise the publication guard against real temporary Git histories."""

from pathlib import Path
import subprocess
import sys

import pytest

GUARD = Path(__file__).resolve().parents[1] / "tools/check_public_content.py"


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args]).decode().strip()


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Guard Test")
    git(tmp_path, "config", "user.email", "guard@example.com")
    return tmp_path


def stage(repo, name, text):
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    git(repo, "add", "-f", name)


def run(repo, *args, stdin=None):
    return subprocess.run(
        [sys.executable, str(GUARD), *args],
        cwd=repo,
        input=stdin,
        text=True,
        capture_output=True,
    )


def test_clean_index_and_history(repo):
    stage(repo, "README.md", "Public usage guide\n")
    assert run(repo, "--staged").returncode == 0
    git(repo, "commit", "-qm", "Add guide")
    assert run(repo, "--history", "HEAD").returncode == 0


@pytest.mark.parametrize(
    "name",
    [
        "docs/research/notes.md",
        "docs/proposals/notes.md",
        ".research/notes.md",
        "nested/internal-notes/notes.md",
        "scratch/notes.md",
        ".private/note.txt",
    ],
)
def test_force_added_private_path_rejected(repo, name):
    stage(repo, ".gitignore", name + "\n")
    stage(repo, name, "A private note\n")
    result = run(repo, "--staged")
    assert result.returncode == 1
    assert "Private path" in result.stderr


def test_misplaced_marked_note_rejected(repo):
    stage(repo, "docs/guide.md", "ART-EMBODIED " + "INTERNAL ONLY\nA note\n")
    assert run(repo, "--staged").returncode == 1


def test_deleted_note_still_rejected_in_push_history(repo):
    stage(repo, "README.md", "Public guide\n")
    git(repo, "commit", "-qm", "Baseline")
    baseline = git(repo, "rev-parse", "HEAD")
    stage(repo, "docs/research/notes.md", "Internal note\n")
    git(repo, "commit", "-qm", "Add note")
    git(repo, "rm", "docs/research/notes.md")
    git(repo, "commit", "-qm", "Remove note")
    assert run(repo, "--staged").returncode == 0
    assert run(repo, "--history", "HEAD").returncode == 1
    head = git(repo, "rev-parse", "HEAD")
    result = run(
        repo, "--pre-push", stdin=f"refs/heads/main {head} refs/heads/main {baseline}\n"
    )
    assert result.returncode == 1


def test_deleted_marked_note_still_rejected(repo):
    stage(repo, "docs/guide.md", "ART-EMBODIED " + "INTERNAL ONLY\n")
    git(repo, "commit", "-qm", "Add note")
    git(repo, "rm", "docs/guide.md")
    git(repo, "commit", "-qm", "Remove note")
    assert run(repo, "--history", "HEAD").returncode == 1


def test_marker_at_stream_boundary(repo):
    stage(repo, "notes.txt", "x" * 65530 + "ART-EMBODIED " + "INTERNAL ONLY\n")
    assert run(repo, "--staged").returncode == 1
