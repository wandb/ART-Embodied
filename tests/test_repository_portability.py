import os
from pathlib import Path
import shutil
import subprocess
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_public_recipes_do_not_embed_user_home_paths() -> None:
    recipes = (REPOSITORY_ROOT / "examples" / "embodied").glob("*.yaml")
    offenders: list[str] = []
    for recipe in recipes:
        text = recipe.read_text(encoding="utf-8")
        if "/mnt/home/" in text or "/home/" in text:
            offenders.append(str(recipe.relative_to(REPOSITORY_ROOT)))
    assert offenders == []


def test_slurm_wrapper_has_portable_repository_roots() -> None:
    wrapper = REPOSITORY_ROOT / "scripts" / "slurm" / "run-in-repo.sh"
    text = wrapper.read_text(encoding="utf-8")
    assert "BASH_SOURCE[0]" in text
    assert "SLURM_SUBMIT_DIR" in text
    assert "ART_EMBODIED_REPO_ROOT" in text
    assert "ART_EMBODIED_ENV_FILE" in text
    assert "/mnt/home/" not in text
    assert "/home/" not in text


def test_slurm_wrapper_recovers_checkout_after_spool_copy(tmp_path: Path) -> None:
    wrapper = REPOSITORY_ROOT / "scripts" / "slurm" / "run-in-repo.sh"
    spooled = tmp_path / "slurm_script"
    shutil.copy2(wrapper, spooled)
    environment = os.environ.copy()
    environment.update(
        {
            "SLURM_SUBMIT_DIR": str(REPOSITORY_ROOT),
            "ART_EMBODIED_ENV_FILE": str(tmp_path / "missing.env"),
        }
    )

    result = subprocess.run(
        [
            str(spooled),
            sys.executable,
            "-c",
            "from pathlib import Path; print(Path.cwd())",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == REPOSITORY_ROOT
