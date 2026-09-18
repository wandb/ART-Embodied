import json
from types import SimpleNamespace

import pytest

from examples.embodied import wandb_verified_recovery as recovery

RUN = "team/project/run123"
EXPECTED = [
    {"train/loss": 2.0, "experiment/update": 0},
    {"train/loss": 1.0, "experiment/update": 1},
]


def transaction(path, *, closed=True, final=False, run_id="run123"):
    pytest.importorskip("wandb")
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal.datastore import DataStore

    store = DataStore()
    store.open_for_write(str(path))
    store.write(
        pb.Record(run=pb.RunRecord(entity="team", project="project", run_id=run_id))
    )
    for index, row in enumerate(EXPECTED):
        record = pb.Record()
        for key, value in (row | {"_step": index}).items():
            record.history.item.add(key=key, value_json=json.dumps(value))
        store.write(record)
    if closed:
        record = pb.Record()
        record.exit.exit_code = 0
        store.write(record)
    if final:
        record = pb.Record()
        record.final.SetInParent()
        store.write(record)
    store.close()


def test_history_checks_order_values_and_non_numeric_metadata():
    rows = [r | {"_step": i} for i, r in enumerate(EXPECTED)]
    recovery.assert_history(rows, EXPECTED)
    recovery.assert_history(
        [{"_step": 0, "label": "recovery"}], [{"label": "recovery"}]
    )
    with pytest.raises(ValueError, match="Expected"):
        recovery.assert_history(rows[:-1], EXPECTED)
    with pytest.raises(ValueError, match="discontinuity"):
        recovery.assert_history(list(reversed(rows)), EXPECTED)
    with pytest.raises(ValueError, match="mismatch"):
        recovery.assert_history([rows[0], rows[1] | {"train/loss": 0.5}], EXPECTED)
    with pytest.raises(ValueError):
        recovery.assert_history(
            [{"_step": 0, "x": float("nan")}], [{"x": float("nan")}]
        )


@pytest.mark.parametrize("final", [False, True])
def test_closed_core_and_legacy_transaction(tmp_path, final):
    path = tmp_path / "run.wandb"
    transaction(path, final=final)
    assert recovery.read_transaction(path, RUN, EXPECTED)["closed"]
    with pytest.raises(ValueError, match="identity"):
        recovery.read_transaction(path, "team/project/other", EXPECTED)


def test_unclosed_and_incomplete_transactions_rejected(tmp_path):
    path = tmp_path / "run.wandb"
    transaction(path, closed=False)
    with pytest.raises(ValueError, match="not closed"):
        recovery.read_transaction(path, RUN, EXPECTED)
    path.write_bytes(path.read_bytes()[:-3])
    with pytest.raises(AssertionError):
        recovery.read_transaction(path, RUN, EXPECTED)


@pytest.mark.parametrize("state,newer", [("running", False), ("finished", True)])
def test_no_sync_with_active_writer_or_stale_journal(
    tmp_path, monkeypatch, state, newer
):
    path = tmp_path / "run.wandb"
    transaction(path)
    monkeypatch.setattr(
        recovery,
        "capture",
        lambda *a: {
            "state": state,
            "error": "missing",
            "rows": [{"_step": 3}] if newer else [],
        },
    )
    monkeypatch.setattr(
        recovery.subprocess, "run", lambda *a, **k: pytest.fail("Unsafe sync")
    )
    with pytest.raises(ValueError, match="running|newer"):
        recovery.recover_closed(RUN, path, EXPECTED, tmp_path / "attempt")


def test_sync_is_bounded_verified_and_not_labeled_normal(tmp_path, monkeypatch):
    path = tmp_path / "run.wandb"
    transaction(path)
    before = path.read_bytes()
    monkeypatch.setattr(
        recovery,
        "capture",
        lambda *a: {"state": "finished", "error": "missing", "rows": []},
    )
    commands, checked = [], []

    def sync(command, **kwargs):
        commands.append(command)
        assert kwargs["timeout"] == 180
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(recovery.subprocess, "run", sync)
    monkeypatch.setattr(recovery, "verify", lambda *a, **k: checked.append(a))
    result = recovery.recover_closed(RUN, path, EXPECTED, tmp_path / "attempt")
    assert len(commands) == len(checked) == 1
    assert "--append" not in commands[0] and "--clean" not in commands[0]
    assert "--no-mark-synced" in commands[0]
    assert path.read_bytes() == before
    assert result["recovery_sync_used"] and result["recovered_history_verified"]
    assert not result["normal_live_delivery_accepted"]


def test_regression_after_first_good_read_uses_recovery(tmp_path, monkeypatch):
    path = tmp_path / "run.wandb"
    transaction(path)
    calls = []

    def capture(*args):
        calls.append(args)
        return {
            "state": "finished",
            "error": None if len(calls) == 1 else "missing",
            "rows": [],
        }

    checks = []

    def verify(*args, **kwargs):
        checks.append(args)
        if len(checks) == 1:
            raise TimeoutError("Regressed")

    monkeypatch.setattr(recovery, "capture", capture)
    monkeypatch.setattr(recovery, "verify", verify)
    monkeypatch.setattr(
        recovery.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0)
    )
    result = recovery.recover_closed(RUN, path, EXPECTED, tmp_path / "attempt")
    assert result["recovered_history_verified"]
    assert len(calls) == len(checks) == 2
