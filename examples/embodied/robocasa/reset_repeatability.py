"""Gate RoboCasa evaluation resets across replicas and independent workers."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import traceback
from typing import Any

from art_embodied import EmbodiedExperimentConfig
from art_embodied.evaluation import FixedScenarioEvaluator
from art_embodied.evidence import source_identity
from art_embodied.utils import write_json_atomic

from .records import build_evaluation_scenarios
from .settings import RoboCasaSettings
from .simulator import RoboCasaSimulatorProcess


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id")
    parser.add_argument("--worker-repeats", type=int, default=2)
    return parser.parse_args()


def _evaluation_cases(
    config: EmbodiedExperimentConfig,
    *,
    task_id: str | None,
) -> list[dict[str, Any]]:
    evaluator = FixedScenarioEvaluator(
        config=config,
        scenarios=build_evaluation_scenarios(config),
        rollout=_unused_rollout,
    )
    cases = [
        {
            "episode": episode.index,
            "task_id": str(episode.scenario.payload["task_id"]),
            "scenario_id": episode.scenario.id,
            "environment_seed": episode.environment_seed,
            "policy_seed": episode.policy_seed,
        }
        for episode in evaluator.plan
        if task_id is None or episode.scenario.payload["task_id"] == task_id
    ]
    if not cases:
        raise ValueError(f"No evaluation episodes matched task_id={task_id!r}")
    return sorted(cases, key=lambda case: (case["task_id"], case["episode"]))


async def _unused_rollout(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("Reset repeatability never executes policy rollouts")


async def _capture_worker(
    *,
    config: EmbodiedExperimentConfig,
    settings: RoboCasaSettings,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    simulator = RoboCasaSimulatorProcess(
        settings=settings,
        startup_timeout_seconds=(
            config.runtime.rollout_execution.startup_timeout_seconds
        ),
    )
    try:
        identity = await simulator.request({"op": "describe"})
        records = []
        for case in cases:
            request = {
                "op": "reset",
                "task_id": case["task_id"],
                "num_envs": 2,
                "seed": case["environment_seed"],
            }
            first = await simulator.request(request)
            repeated = await simulator.request(request)
            _require_replica_equality(first, case=case, reset="first")
            _require_replica_equality(repeated, case=case, reset="repeated")
            if first["state_hashes"] != repeated["state_hashes"]:
                raise RuntimeError(f"Repeated reset changed simulator state for {case}")
            if first["observation_hashes"] != repeated["observation_hashes"]:
                raise RuntimeError(
                    f"Repeated reset changed policy observation for {case}"
                )
            records.append(
                {
                    **case,
                    "initial_state_sha256": first["state_hashes"][0],
                    "initial_observation_sha256": first["observation_hashes"][0],
                }
            )
        return {"runtime": identity, "records": records}
    finally:
        await simulator.close()


def _require_replica_equality(
    response: dict[str, Any],
    *,
    case: dict[str, Any],
    reset: str,
) -> None:
    if len(set(response["state_hashes"])) != 1:
        raise RuntimeError(
            f"Same-seed replicas changed simulator state at {reset} reset for {case}"
        )
    if len(set(response["observation_hashes"])) != 1:
        raise RuntimeError(
            f"Same-seed replicas changed policy observation at {reset} reset for {case}"
        )


def _compare_workers(captures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reference = captures[0]
    mismatches = []
    for worker_index, capture in enumerate(captures[1:], start=1):
        if capture["runtime"] != reference["runtime"]:
            mismatches.append(
                {
                    "worker": worker_index,
                    "kind": "runtime_identity",
                    "expected": reference["runtime"],
                    "actual": capture["runtime"],
                }
            )
        if len(capture["records"]) != len(reference["records"]):
            mismatches.append(
                {
                    "worker": worker_index,
                    "kind": "record_count",
                    "expected": len(reference["records"]),
                    "actual": len(capture["records"]),
                }
            )
            continue
        for expected, actual in zip(
            reference["records"], capture["records"], strict=True
        ):
            if actual != expected:
                mismatches.append(
                    {
                        "worker": worker_index,
                        "kind": "reset_identity",
                        "episode": expected["episode"],
                        "task_id": expected["task_id"],
                        "expected": expected,
                        "actual": actual,
                    }
                )
    return mismatches


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.worker_repeats < 2:
        raise ValueError("--worker-repeats must be at least 2")
    config = EmbodiedExperimentConfig.from_yaml(args.config)
    settings = RoboCasaSettings.from_config(config)
    cases = _evaluation_cases(config, task_id=args.task_id)
    captures = [
        await _capture_worker(config=config, settings=settings, cases=cases)
        for _ in range(args.worker_repeats)
    ]
    mismatches = _compare_workers(captures)
    return {
        "schema_version": 1,
        "kind": "robocasa_reset_repeatability",
        "status": "passed" if not mismatches else "failed",
        "config": str(args.config.resolve()),
        "config_fingerprint": config.fingerprint,
        "source": source_identity(),
        "worker_repeats": args.worker_repeats,
        "episodes": len(cases),
        "task_ids": sorted({case["task_id"] for case in cases}),
        "runtime": captures[0]["runtime"],
        "records": captures[0]["records"],
        "mismatches": mismatches,
    }


def main() -> None:
    args = _parse_args()
    try:
        report = asyncio.run(_run(args))
    except Exception as exc:
        report = {
            "schema_version": 1,
            "kind": "robocasa_reset_repeatability",
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json_atomic(args.output, report, indent=2, sort_keys=True)
        raise
    write_json_atomic(args.output, report, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
