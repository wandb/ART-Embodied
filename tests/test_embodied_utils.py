from __future__ import annotations

import json
from pathlib import Path
import sys

from art_embodied.utils import worker_python_command, write_json_atomic


def test_worker_python_command_defaults_to_current_interpreter() -> None:
    command = worker_python_command(
        configured_executable=None,
        module="art_embodied.rollout_worker",
        spec_path=Path("bootstrap.json"),
    )

    assert command == [
        sys.executable,
        "-m",
        "art_embodied.rollout_worker",
        "--serve-spec",
        "bootstrap.json",
    ]


def test_worker_python_command_uses_explicit_policy_runtime() -> None:
    command = worker_python_command(
        configured_executable="/opt/venv/openvla/bin/python",
        module="art_embodied.backends.action_token_worker",
        spec_path=Path("/tmp/worker spec.json"),
    )

    assert command == [
        "/opt/venv/openvla/bin/python",
        "-m",
        "art_embodied.backends.action_token_worker",
        "--serve-spec",
        "/tmp/worker spec.json",
    ]


def test_write_json_atomic_publishes_only_complete_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "result.json"
    real_replace = Path.replace
    observed: dict[str, object] = {}

    def inspect_before_publish(source: Path, destination: str | Path) -> Path:
        assert not Path(destination).exists()
        observed["payload"] = json.loads(source.read_text(encoding="utf-8"))
        return real_replace(source, destination)

    monkeypatch.setattr("art_embodied.utils.Path.replace", inspect_before_publish)

    write_json_atomic(target, {"ok": True, "worker": 3}, sort_keys=True)

    assert observed["payload"] == {"ok": True, "worker": 3}
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "ok": True,
        "worker": 3,
    }
    assert list(tmp_path.iterdir()) == [target]
