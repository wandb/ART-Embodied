"""Aggregate task-level RoboCasa oracle checks into one gated evidence file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from art_embodied.utils import write_json_atomic


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def aggregate_oracle_evidence(
    *,
    manifest_path: Path,
    report_paths: Iterable[Path],
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_tasks = [str(task_id) for task_id in manifest["expansion_task_ids"]]
    one_chunk: dict[str, dict[str, Any]] = {}
    full_episodes = []
    failures = []
    inputs = []

    for report_path in report_paths:
        report_path = report_path.resolve()
        raw = report_path.read_bytes()
        report = json.loads(raw)
        task_id = str(report.get("task_id", ""))
        kind = str(report.get("kind", ""))
        inputs.append(
            {
                "path": str(report_path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "kind": kind,
                "task_id": task_id,
            }
        )
        if report.get("status") != "passed":
            failures.append({"kind": "failed_input", "path": str(report_path)})
            continue
        if Path(str(report.get("oracle_manifest", ""))).resolve() != manifest_path:
            failures.append({"kind": "manifest_mismatch", "path": str(report_path)})
            continue
        if kind == "gr00t_n1d7_robocasa_one_task_oracle_parity":
            if task_id in one_chunk:
                failures.append({"kind": "duplicate_task", "task_id": task_id})
            one_chunk[task_id] = report
        elif kind == "gr00t_n1d7_robocasa_one_task_full_episode_oracle_parity":
            full_episodes.append(report)
        else:
            failures.append({"kind": "unsupported_report", "path": str(report_path)})

    missing = sorted(set(expected_tasks).difference(one_chunk))
    extra = sorted(set(one_chunk).difference(expected_tasks))
    if missing:
        failures.append({"kind": "missing_tasks", "task_ids": missing})
    if extra:
        failures.append({"kind": "unexpected_tasks", "task_ids": extra})
    if not full_episodes:
        failures.append({"kind": "missing_full_episode"})

    task_results = []
    for task_id in expected_tasks:
        report = one_chunk.get(task_id)
        if report is None:
            continue
        parity = report["official_oracle_chunk_parity"]
        exact = bool(
            report["official_art_eval_exact_match"]
            and parity["status"] == "passed"
            and parity["reset"]["exact_match"]
            and parity["final"]["exact_match"]
            and parity["macro"]["exact_match"]
            and not parity["mismatches"]
        )
        if not exact:
            failures.append({"kind": "non_exact_task", "task_id": task_id})
        task_results.append({"task_id": task_id, "exact_match": exact})

    full_episode_results = []
    for report in full_episodes:
        parity = report["official_oracle_chunk_parity"]
        chunks = parity["chunks"]
        exact = bool(
            parity["status"] == "passed"
            and chunks
            and all(
                chunk["status"] == "passed"
                and chunk["final"]["exact_match"]
                and chunk["macro"]["exact_match"]
                and not chunk["mismatches"]
                for chunk in chunks
            )
        )
        if not exact:
            failures.append(
                {"kind": "non_exact_full_episode", "task_id": report["task_id"]}
            )
        full_episode_results.append(
            {
                "task_id": report["task_id"],
                "exact_match": exact,
                "policy_steps": parity["policy_steps"],
                "environment_steps": parity["environment_steps"],
                "success": parity["success"],
                "terminal_reason": parity["terminal_reason"],
            }
        )

    return {
        "schema_version": 1,
        "kind": "gr00t_n1d7_robocasa_frontier8_oracle_evidence",
        "status": "passed" if not failures else "failed",
        "manifest": {
            "path": str(manifest_path),
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
        "expected_task_ids": expected_tasks,
        "task_results": task_results,
        "full_episode_results": full_episode_results,
        "inputs": inputs,
        "failures": failures,
    }


def main() -> None:
    args = _parse_args()
    report = aggregate_oracle_evidence(
        manifest_path=args.manifest,
        report_paths=args.reports,
    )
    write_json_atomic(args.output, report, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
