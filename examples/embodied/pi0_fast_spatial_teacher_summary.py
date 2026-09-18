"""Read-only paired analysis of the Spatial teacher-path diagnostic."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def episode_map(episodes):
    mapped = {}
    for episode in episodes:
        key = episode["scenario_id"]
        if key in mapped:
            raise ValueError(f"Duplicate scenario: {key}")
        if episode["completed"] is not True or episode.get("error") is not None:
            raise ValueError(f"Incomplete evaluation: {key}")
        # Evaluation artifacts serialize binary success as 0.0/1.0.
        if not isinstance(episode["success"], (bool, int, float)) or episode[
            "success"
        ] not in (0, 1):
            raise ValueError(f"Nonbinary success: {key}")
        if not episode["action_chunk_sha256s"]:
            raise ValueError(f"Missing action evidence: {key}")
        mapped[key] = episode
    if not mapped:
        raise ValueError("Empty evaluation")
    return mapped


def compare(before, after):
    if before.keys() != after.keys():
        raise ValueError("Evaluation populations differ")
    counts = {
        "failure_to_success": 0,
        "success_to_failure": 0,
        "both_success": 0,
        "both_failure": 0,
        "identical_action_sequences": 0,
    }
    for key, old in before.items():
        new = after[key]
        for field in ("environment_seed", "policy_seed", "task", "scenario_payload"):
            if old[field] != new[field]:
                raise ValueError(f"Unpaired {field}: {key}")
        if old["success"] and new["success"]:
            counts["both_success"] += 1
        elif old["success"]:
            counts["success_to_failure"] += 1
        elif new["success"]:
            counts["failure_to_success"] += 1
        else:
            counts["both_failure"] += 1
        counts["identical_action_sequences"] += (
            old["action_chunk_sha256s"] == new["action_chunk_sha256s"]
        )
    n = len(before)
    return counts | {
        "episodes": n,
        "before_successes": sum(e["success"] for e in before.values()),
        "after_successes": sum(e["success"] for e in after.values()),
        "success_rate_change": (
            counts["failure_to_success"] - counts["success_to_failure"]
        )
        / n,
    }


def evaluation(path, expected_count):
    result = read(path)
    outcomes = read(Path(result["artifacts"]["episode_outcomes_json"]))
    if outcomes["data_role"] != "development":
        raise ValueError("This diagnostic must not access sealed outcomes")
    episodes = episode_map(outcomes["episodes"])
    if len(episodes) != expected_count:
        raise ValueError("Incomplete evaluation population")
    measured = sum(e["success"] for e in episodes.values()) / len(episodes)
    if abs(measured - result["metrics"]["success_rate"]) > 1e-12:
        raise ValueError("Evaluation summary differs from episode outcomes")
    return episodes


def summarize(root):
    arms, baselines, indices, plans = {}, {}, {}, {}
    for arm in ("native", "sampler"):
        directory = root / arm
        plans[arm] = plan = read(directory / "plan.json")
        records = read(directory / "records.json")
        updates = [row["experiment/update"] for row in records]
        if not updates or updates != list(range(len(updates))):
            raise ValueError(f"Nonconsecutive local history: {arm}")
        indices[arm] = read(directory / "batch-indices.json")
        baseline = evaluation(
            directory / "evaluation-0000.json", plan["evaluation_episodes"]
        )
        baselines[arm] = baseline
        results = {}
        for step in plan["evaluation_updates"]:
            path = directory / f"evaluation-{step:04d}.json"
            if path.exists():
                results[str(step)] = compare(baseline, evaluation(path, len(baseline)))
        failure = directory / "failure.json"
        arms[arm] = {
            "run": read(directory / "run.json"),
            "last_locally_logged_update": updates[-1],
            "completed_training_updates": len(indices[arm]),
            "first_update_verification": read(directory / "FIRST_UPDATE_VERIFIED.json"),
            "evaluations": results,
            "complete": (directory / "complete.json").exists(),
            "failure": read(failure) if failure.exists() else None,
        }
    for field in (
        "initial_sha256",
        "inventory",
        "seed",
        "mode",
        "microbatch",
        "accumulation",
        "teacher_temperature",
        "optimizer",
        "scheduler",
        "evaluation_updates",
        "evaluation_episodes",
        "updates",
    ):
        if plans["native"][field] != plans["sampler"][field]:
            raise ValueError(f"Unpaired experiment setting: {field}")
    n = min(map(len, indices.values()))
    if indices["native"][:n] != indices["sampler"][:n]:
        raise ValueError("Teacher data order differs")
    initial = compare(baselines["native"], baselines["sampler"])
    if initial["identical_action_sequences"] != initial["episodes"]:
        raise ValueError("Initial policy behavior differs between arms")
    if initial["failure_to_success"] or initial["success_to_failure"]:
        raise ValueError("Initial outcomes differ between arms")
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "SFT teacher-path diagnostic; not GRPO qualification",
        "matching_training_data_updates": n,
        "identical_initial_behavior": initial,
        "arms": arms,
        "live_remote_history_checked_by_this_summary": False,
        "rendering_verified": False,
        "root_cause_confirmed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    if args.output.exists():
        parser.error("Refusing to overwrite an existing analysis snapshot")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
