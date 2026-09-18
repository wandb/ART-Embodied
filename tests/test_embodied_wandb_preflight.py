from __future__ import annotations

from types import SimpleNamespace

import art_embodied.wandb_preflight as wandb_preflight
from art_embodied.wandb_preflight import _inspect_roundtrip, run_preflight


class _PublicRun:
    id = "preflight-token"
    config = {"art_embodied_preflight_token": "sentinel"}
    summary = {"art_embodied_preflight_summary_token": "sentinel"}

    def scan_history(self, *, keys):
        assert keys == [
            "art_embodied_preflight/history_value",
            "art_embodied_preflight/history_token",
            "experiment/update",
            "train/success_rate",
            "validation/success_rate",
        ]
        return iter(
            [
                {
                    "art_embodied_preflight/history_value": 1.0,
                    "art_embodied_preflight/history_token": "sentinel",
                    "experiment/update": 0,
                    "train/success_rate": 0.5,
                    "validation/success_rate": 0.5,
                }
            ]
        )


class _Api:
    def __init__(self, *, timeout):
        assert timeout == 30

    def run(self, path):
        assert path == "entity/project/preflight-token"
        return _PublicRun()

    def runs(self, path):
        assert path == "entity/project"
        return [SimpleNamespace(id="preflight-token")]


def test_roundtrip_requires_models_discovery_config_history_and_summary() -> None:
    checks = _inspect_roundtrip(
        SimpleNamespace(Api=_Api),
        run_path="entity/project/preflight-token",
        token="sentinel",
    )

    assert checks == {
        "direct_run_discovery": True,
        "project_list_discovery": True,
        "config_roundtrip": True,
        "history_roundtrip": True,
        "chart_metrics_roundtrip": True,
        "summary_roundtrip": True,
    }


def test_preflight_reads_back_after_isolated_upload(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(wandb_preflight, "_run_upload_process", lambda **_: 0)
    monkeypatch.setattr(
        wandb_preflight,
        "import_module",
        lambda name: SimpleNamespace(__version__="0.test"),
    )
    monkeypatch.setattr(
        wandb_preflight,
        "_inspect_roundtrip",
        lambda wandb, *, run_path, token: {
            "direct_run_discovery": True,
            "project_list_discovery": True,
            "config_roundtrip": True,
            "history_roundtrip": True,
            "chart_metrics_roundtrip": True,
            "summary_roundtrip": True,
        },
    )
    output = tmp_path / "preflight.json"

    report = run_preflight(
        entity="entity",
        project="project",
        output=output,
        timeout_seconds=1,
        poll_seconds=0.01,
    )

    assert report["status"] == "passed"
    assert report["upload_process_isolated"] is True
    assert report["upload_process_exit_code"] == 0
    assert report["sdk_version"] == "0.test"
    assert output.is_file()


def test_preflight_fails_closed_when_upload_process_fails(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(wandb_preflight, "_run_upload_process", lambda **_: 7)
    inspect_called = False

    def fail_if_inspected(*args, **kwargs):
        nonlocal inspect_called
        inspect_called = True
        raise AssertionError("readback must not run after a failed upload")

    monkeypatch.setattr(wandb_preflight, "_inspect_roundtrip", fail_if_inspected)
    output = tmp_path / "preflight.json"

    report = run_preflight(
        entity="entity",
        project="project",
        output=output,
        timeout_seconds=1,
        poll_seconds=0.01,
    )

    assert report["status"] == "failed"
    assert report["upload_process_exit_code"] == 7
    assert report["error_type"] == "RuntimeError"
    assert inspect_called is False
    assert output.is_file()
