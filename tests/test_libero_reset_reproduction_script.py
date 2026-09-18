import json
from pathlib import Path
import subprocess

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/reproduce-libero-reset.sh"


def test_help_requires_no_simulator():
    result = subprocess.run(
        ["bash", str(SCRIPT), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "NOT a passed training gate" in result.stdout


def test_missing_interpreter_does_not_create_output(tmp_path):
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", str(SCRIPT), "/no/such/python", "/no/such/python", str(output)],
        capture_output=True,
    )
    assert result.returncode == 2
    assert not output.exists()


def test_same_contract_both_arms_and_no_overwrite(tmp_path):
    log = tmp_path / "calls"
    python = tmp_path / "fake-python"
    python.write_text(f'#!/bin/bash\nprintf "%s\\n" "$*" >> "{log}"\n')
    python.chmod(0o755)
    output = tmp_path / "output"
    command = ["bash", str(SCRIPT), str(python), str(python), str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0
    calls = log.read_text().splitlines()
    assert len(calls) == 3
    assert calls[0].split(" --output")[0] == calls[1].split(" --output")[0]
    assert (
        "--all-states --seed 0 --wait-steps 10 --control-mode relative --report-only"
        in calls[0]
    )
    assert "compare_reset_audits" in calls[2]
    assert subprocess.run(command, capture_output=True).returncode == 2
    assert log.read_text().splitlines() == calls


def test_incomplete_audit_stops_comparison(tmp_path):
    python = tmp_path / "fake-python"
    python.write_text("#!/bin/bash\nexit 2\n")
    python.chmod(0o755)
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", str(SCRIPT), str(python), str(python), str(output)],
        capture_output=True,
    )
    assert result.returncode == 2
    assert not (output / "comparison.json").exists()


def test_grpo_hold_prevents_publish_and_training(tmp_path, monkeypatch):
    from examples.embodied import pi0_fast_reset_restart_run as runner

    def forbidden(*args, **kwargs):
        raise AssertionError("Must not publish or load a training config while held")

    monkeypatch.setattr(runner, "publish_sft", forbidden)
    monkeypatch.setattr(runner.EmbodiedExperimentConfig, "from_yaml", forbidden)
    (tmp_path / "HOLD_BEFORE_GRPO").write_text("diagnostic condition only")
    runner.main(tmp_path)
    assert json.loads((tmp_path / "grpo-held.json").read_text()) == {
        "status": "held_before_grpo",
        "reason": "diagnostic condition only",
    }
