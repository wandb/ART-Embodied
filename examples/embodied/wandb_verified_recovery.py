"""Bounded, explicit recovery of a closed initial W&B writer, never silent success.

This experimental control does not support replaying a partial resumed segment.
Keep the writer closed throughout recovery; resuming is a separate verified step.
"""

import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time


def assert_history(rows, expected):
    if not expected or len(rows) != len(expected):
        raise ValueError(f"Expected {len(expected)} rows, observed {len(rows)}")
    for index, (row, reference) in enumerate(zip(rows, expected, strict=True)):
        if row.get("_step") != index:
            raise ValueError(f"Native step discontinuity at {index}")
        for key, value in reference.items():
            actual = row.get(key)
            json.dumps(value, allow_nan=False)
            matches = actual == value
            if isinstance(value, (float, int)):
                matches = isinstance(actual, (float, int)) and math.isclose(
                    actual, value, rel_tol=1e-10, abs_tol=1e-12
                )
            if key not in row or not matches:
                raise ValueError(f"Value mismatch at row {index}: {key}")


def read_transaction(path, run_path, expected):
    from wandb.proto.wandb_internal_pb2 import Record
    from wandb.sdk.internal.datastore import DataStore

    entity, project, run_id = run_path.split("/")
    store = DataStore()
    store.open_for_scan(str(path))
    rows, identities, endings = [], [], set()
    last_kind = None
    try:
        while (data := store.scan_data()) is not None:
            record = Record()
            record.ParseFromString(data)
            kind = record.WhichOneof("record_type")
            last_kind = kind
            if kind == "run":
                identities.append(
                    (record.run.entity, record.run.project, record.run.run_id)
                )
            elif kind == "history":
                row = {}
                for item in record.history.item:
                    key = item.key or ".".join(item.nested_key)
                    row[key] = json.loads(item.value_json)
                rows.append(row)
            elif kind in ("exit", "final"):
                endings.add(kind)
    finally:
        store.close()
    if not identities or any(
        identity != (entity, project, run_id) for identity in identities
    ):
        raise ValueError("Transaction identity does not match destination")
    # wandb-core 0.24.2 persists exit last; legacy service may also persist final.
    if "exit" not in endings or last_kind not in ("exit", "final"):
        raise ValueError("Transaction is not closed; refuse concurrent writer recovery")
    assert_history(rows, expected)
    return {"rows": len(rows), "identity": identities[0], "closed": True}


def capture(run_path, expected, path):
    import wandb

    run = wandb.Api(timeout=30).run(run_path)
    rows = list(
        run.scan_history(page_size=1000, max_step=max(1000, len(expected) + 100))
    )
    try:
        assert_history(rows, expected)
        error = None
    except ValueError as exc:
        error = str(exc)
    result = {"time": time.time(), "state": run.state, "rows": rows, "error": error}
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def verify(run_path, expected, output, *, timeout=90, stable_reads=3, interval=5):
    output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + timeout
    successes, attempt = 0, 0
    while True:
        result = capture(run_path, expected, output / f"read-{attempt:03d}.json")
        successes = successes + 1 if result["error"] is None else 0
        if successes >= stable_reads:
            return result
        if time.monotonic() >= deadline:
            raise TimeoutError(f"History did not stabilize: {result['error']}")
        attempt += 1
        time.sleep(interval)


def recover_closed(run_path, transaction, expected, output, *, timeout=90):
    """Verify or sync once, then verify; caller must not have an active writer."""
    output.mkdir(parents=True, exist_ok=False)
    source_hash = hashlib.sha256(transaction.read_bytes()).hexdigest()
    local = read_transaction(transaction, run_path, expected)
    snapshot = output / transaction.name
    shutil.copy2(transaction, snapshot)
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != source_hash:
        raise ValueError("Transaction backup mismatch")
    incident = {
        "run": run_path,
        "transaction": str(transaction.resolve()),
        "transaction_sha256": source_hash,
        "local": local,
        "recovery_sync_used": False,
        "normal_live_delivery_accepted": False,
        "recovered_history_verified": False,
        "rendering_verified": False,
    }

    def persist():
        (output / "recovery.json").write_text(json.dumps(incident, indent=2) + "\n")

    persist()
    try:
        before = capture(run_path, expected, output / "before.json")
        if before["state"] == "running":
            raise ValueError("Remote run is still running; close the writer first")
        if before["error"] is None:
            try:
                verify(
                    run_path, expected, output / "normal-verification", timeout=timeout
                )
            except TimeoutError:
                before = capture(run_path, expected, output / "regressed.json")
                if before["state"] == "running":
                    raise ValueError("Writer became active during verification")
            else:
                incident["normal_live_delivery_accepted"] = True
                persist()
                return incident
        # A stale transaction must never overwrite a later continuation.
        if any(row.get("_step", -1) >= len(expected) for row in before["rows"]):
            raise ValueError("Remote history contains a newer continuation")
        entity, project, run_id = run_path.split("/")
        command = [
            str(Path(sys.executable).with_name("wandb")),
            "sync",
            "--id",
            run_id,
            "--entity",
            entity,
            "--project",
            project,
            "--include-online",
            "--include-offline",
            "--include-synced",
            "--no-mark-synced",
            "--no-sync-tensorboard",
            str(transaction),
        ]
        incident["recovery_sync_used"] = True
        incident["command"] = command
        persist()
        with (output / "sync.log").open("w") as stream:
            completed = subprocess.run(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=180,
                check=False,
            )
        incident["sync_exit_code"] = completed.returncode
        if completed.returncode:
            raise RuntimeError(f"wandb sync failed: {completed.returncode}")
        if hashlib.sha256(transaction.read_bytes()).hexdigest() != source_hash:
            raise ValueError("Source transaction changed during recovery")
        verify(run_path, expected, output / "recovered-verification", timeout=timeout)
        incident["recovered_history_verified"] = True
        persist()
        return incident
    except Exception as exc:
        incident["error"] = repr(exc)
        persist()
        raise
