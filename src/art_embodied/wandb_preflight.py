"""Fail-closed W&B Models round-trip preflight for expensive experiments."""

from __future__ import annotations

import argparse
from importlib import import_module
import json
import multiprocessing
from pathlib import Path
import time
from typing import Any
import uuid

from .utils import write_json_atomic


def _upload_canary(
    *,
    entity: str,
    project: str,
    run_id: str,
    token: str,
) -> None:
    """Upload a canary in a disposable process.

    Some W&B SDK shutdown paths terminate the process after ``Run.finish()``.
    Keeping that behavior in a child guarantees that the readback report is still
    produced by the parent.
    """

    wandb = import_module("wandb")
    run = wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        name=run_id,
        job_type="observability-preflight",
        tags=["art-embodied", "observability-preflight", "preserve"],
        config={"art_embodied_preflight_token": token},
        settings=wandb.Settings(mode="online"),
    )
    if run is None:
        raise RuntimeError("wandb.init returned no Run")
    update_metric = run.define_metric("experiment/update", hidden=True)
    run.define_metric("train/*", step_metric=update_metric)
    run.define_metric("validation/*", step_metric=update_metric)
    run.log(
        {
            "experiment/update": 0,
            "art_embodied_preflight/history_value": 1.0,
            "art_embodied_preflight/history_token": token,
            "train/success_rate": 0.5,
            "validation/success_rate": 0.5,
        }
    )
    run.summary["art_embodied_preflight_summary_token"] = token
    run.finish()


def _run_upload_process(
    *,
    entity: str,
    project: str,
    run_id: str,
    token: str,
    timeout_seconds: float,
) -> int:
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_upload_canary,
        kwargs={
            "entity": entity,
            "project": project,
            "run_id": run_id,
            "token": token,
        },
        name=f"wandb-preflight-upload-{run_id}",
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError(
            f"W&B canary upload did not finish within {timeout_seconds:g} seconds"
        )
    if process.exitcode is None:
        raise RuntimeError("W&B canary upload process has no exit code")
    return process.exitcode


def run_preflight(
    *,
    entity: str,
    project: str,
    output: Path,
    timeout_seconds: float = 120.0,
    poll_seconds: float = 10.0,
) -> dict[str, Any]:
    """Create and verify one preserved canary Run through the Public API."""

    if not entity or not project:
        raise ValueError("W&B preflight requires non-empty entity and project")
    if timeout_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("W&B preflight timeouts must be positive")
    token = uuid.uuid4().hex
    run_id = f"preflight-{token[:16]}"
    run_path = f"{entity}/{project}/{run_id}"
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "failed",
        "entity": entity,
        "project": project,
        "run_id": run_id,
        "run_path": run_path,
        "run_url": f"https://wandb.ai/{entity}/{project}/runs/{run_id}",
        "token": token,
        "upload_process_isolated": True,
        "checks": {},
        "attempts": [],
    }
    try:
        upload_exit_code = _run_upload_process(
            entity=entity,
            project=project,
            run_id=run_id,
            token=token,
            timeout_seconds=timeout_seconds,
        )
        report["upload_process_exit_code"] = upload_exit_code
        if upload_exit_code != 0:
            raise RuntimeError(
                f"W&B canary upload process exited with code {upload_exit_code}"
            )
        wandb = import_module("wandb")
        report["sdk_version"] = getattr(wandb, "__version__", None)
    except BaseException as exc:
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["completed_at_unix_seconds"] = time.time()
        write_json_atomic(output, report, indent=2, sort_keys=True)
        return report

    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            checks = _inspect_roundtrip(wandb, run_path=run_path, token=token)
            report["checks"] = checks
            report["attempts"].append(
                {
                    "elapsed_seconds": timeout_seconds
                    - max(0.0, deadline - time.monotonic()),
                    "checks": checks,
                }
            )
            if all(checks.values()):
                report["status"] = "passed"
                break
        except BaseException as exc:
            report["attempts"].append(
                {
                    "elapsed_seconds": timeout_seconds
                    - max(0.0, deadline - time.monotonic()),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        if time.monotonic() >= deadline:
            report["error"] = (
                "W&B Models did not round-trip project discovery, config, history, "
                "and summary before the deadline"
            )
            break
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))

    report["completed_at_unix_seconds"] = time.time()
    write_json_atomic(output, report, indent=2, sort_keys=True)
    return report


def _inspect_roundtrip(
    wandb: Any,
    *,
    run_path: str,
    token: str,
) -> dict[str, bool]:
    entity, project, run_id = run_path.split("/", 2)
    api = wandb.Api(timeout=30)
    public_run = api.run(run_path)
    listed_ids = {str(candidate.id) for candidate in api.runs(f"{entity}/{project}")}
    config = dict(public_run.config or {})
    summary = dict(public_run.summary or {})
    rows = list(
        public_run.scan_history(
            keys=[
                "art_embodied_preflight/history_value",
                "art_embodied_preflight/history_token",
                "experiment/update",
                "train/success_rate",
                "validation/success_rate",
            ]
        )
    )
    history_ok = any(
        row.get("art_embodied_preflight/history_value") == 1.0
        and row.get("art_embodied_preflight/history_token") == token
        for row in rows
    )
    chart_metrics_ok = any(
        row.get("experiment/update") == 0
        and row.get("train/success_rate") == 0.5
        and row.get("validation/success_rate") == 0.5
        for row in rows
    )
    return {
        "direct_run_discovery": str(public_run.id) == run_id,
        "project_list_discovery": run_id in listed_ids,
        "config_roundtrip": config.get("art_embodied_preflight_token") == token,
        "history_roundtrip": history_ok,
        "chart_metrics_roundtrip": chart_metrics_ok,
        "summary_roundtrip": (
            summary.get("art_embodied_preflight_summary_token") == token
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_preflight(
        entity=args.entity,
        project=args.project,
        output=args.output,
        timeout_seconds=args.timeout_seconds,
        poll_seconds=args.poll_seconds,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
