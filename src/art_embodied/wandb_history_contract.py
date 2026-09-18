"""Read-only comparison of W&B history with measured experiment evidence.

Storage verification is necessary, not sufficient: this does not verify that
the App renders the target run's charts correctly.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

from .utils import write_json_atomic


def expected_training_history(
    output: Path, *, last_policy_version: int, completed_update_steps: bool = False
) -> dict:
    if last_policy_version < 0:
        raise ValueError("last_policy_version must be nonnegative")
    expected = {}
    for version in range(last_policy_version + 1):
        path = output / "rollout-evidence" / f"policy-version-{version:06d}.json"
        evidence = json.loads(path.read_text())
        if evidence["policy_version"] != version:
            raise ValueError(f"Policy version mismatch in {path}")
        groups = evidence["groups"]
        count = sum(row["completed_trajectories"] for row in groups)
        rewards = [value for row in groups for value in row["rewards"]]
        if count <= 0 or len(rewards) != count:
            raise ValueError(f"Incomplete trajectory counts in {path}")
        expected[version + int(completed_update_steps)] = {
            "train/success_rate": sum(row["success_count"] for row in groups) / count,
            "train/reward_mean": math.fsum(rewards) / count,
        }
    return expected


def inspect_metric_rows(
    rows: list[dict], expected: dict[int, float], *, metric: str
) -> dict:
    by_version = defaultdict(list)
    errors = []
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        version = row.get("experiment/update")
        if (
            isinstance(version, bool)
            or not isinstance(version, int | float)
            or not math.isfinite(version)
            or version < 0
            or int(version) != version
        ):
            errors.append(
                {"kind": "invalid_update_axis", "native_step": row.get("_step")}
            )
            continue
        by_version[int(version)].append(row)
    for version, reference in expected.items():
        observed = by_version.get(version, [])
        if len(observed) != 1:
            errors.append(
                {
                    "kind": "missing" if not observed else "duplicate",
                    "update": version,
                    "rows": len(observed),
                }
            )
            continue
        value = observed[0][metric]
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or not math.isclose(value, reference, rel_tol=1e-7, abs_tol=1e-9)
        ):
            errors.append(
                {
                    "kind": "value_mismatch",
                    "update": version,
                    "expected": reference,
                    "observed": value,
                }
            )
    return {
        "metric": metric,
        "expected_updates": sorted(expected),
        "observed_updates": sorted(by_version),
        "rows": rows,
        "errors": errors,
        "storage_verified": not errors,
    }


def verify_run_history(
    run: Any,
    expected: dict[int, dict[str, float]],
    *,
    require_native_steps: bool = False,
) -> dict:
    if not expected:
        raise ValueError("A history contract needs measured expected values")
    metrics = sorted({key for values in expected.values() for key in values})
    results = []
    native_errors = []
    all_rows = list(run.scan_history()) if require_native_steps else None
    if all_rows is not None:
        expected_steps = list(range(max(expected) + 1))
        observed_steps = [row.get("_step") for row in all_rows]
        if observed_steps != expected_steps:
            native_errors.append(
                {
                    "kind": "native_step_sequence",
                    "observed": observed_steps,
                    "expected": expected_steps,
                }
            )
        for row in all_rows:
            if row.get("_step") != row.get("experiment/update"):
                native_errors.append(
                    {
                        "kind": "native_step_mismatch",
                        "native_step": row.get("_step"),
                        "update": row.get("experiment/update"),
                    }
                )
    for metric in metrics:
        # Explicit per-metric scans avoid requiring train and evaluation values
        # to occupy the same native SDK history row.
        rows = (
            [row for row in all_rows if row.get(metric) is not None]
            if all_rows is not None
            else list(run.scan_history(keys=["_step", "experiment/update", metric]))
        )
        results.append(
            inspect_metric_rows(
                rows,
                {
                    v: values[metric]
                    for v, values in expected.items()
                    if metric in values
                },
                metric=metric,
            )
        )
    return {
        "run_url": run.url,
        "run_state": run.state,
        "storage_verified": not native_errors
        and all(result["storage_verified"] for result in results),
        "native_steps_verified": require_native_steps and not native_errors,
        "native_step_errors": native_errors,
        "app_rendering_verified": False,
        "experiment_accepted": False,
        "metrics": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="entity/project/run_id")
    parser.add_argument("--experiment-output", type=Path, required=True)
    parser.add_argument("--last-policy-version", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    expected = expected_training_history(
        args.experiment_output, last_policy_version=args.last_policy_version
    )
    import wandb

    report = verify_run_history(wandb.Api(timeout=30).run(args.run), expected)
    write_json_atomic(args.report, report, indent=2)
    print(json.dumps({key: value for key, value in report.items() if key != "metrics"}))
    if not report["storage_verified"]:
        raise SystemExit("W&B history differs from measured evidence; see report")


if __name__ == "__main__":
    main()
