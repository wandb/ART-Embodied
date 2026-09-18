"""Compare completed reset audits, never interpret every BDDL predicate as a gate."""

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_reset import write_report


def compare(left, right):
    if not left.get("completed") or not right.get("completed"):
        raise ValueError("Cannot compare incomplete audits")
    if left["reset"] != right["reset"]:
        raise ValueError("Reset contracts differ")
    for package in set(left["versions"]) | set(right["versions"]):
        if package != "mujoco" and left["versions"].get(package) != right[
            "versions"
        ].get(package):
            raise ValueError(f"Additional package differs: {package}")

    def keyed(report):
        result = {(t["suite"], t["task_id"]): t for t in report["tasks"]}
        if len(result) != len(report["tasks"]):
            raise ValueError("Duplicate tasks")
        return result

    a, b = keyed(left), keyed(right)
    if a.keys() != b.keys():
        raise ValueError("Task membership differs")
    results = []
    for key in a:
        x, y = a[key], b[key]
        for field in ("bddl_sha256", "bank_sha256", "bank_size"):
            if x[field] != y[field]:
                raise ValueError(f"{key}: different {field}")
        if [e["index"] for e in x["episodes"]] != [e["index"] for e in y["episodes"]]:
            raise ValueError("State membership differs")
        changes, max_distance = [], 0.0
        errors = []
        for xe, ye in zip(x["episodes"], y["episodes"], strict=True):
            if xe["source_state_sha256"] != ye["source_state_sha256"]:
                raise ValueError("Source state differs")
            xp = {tuple(p["predicate"]): p for p in xe["predicates"]}
            yp = {tuple(p["predicate"]): p for p in ye["predicates"]}
            if xp.keys() != yp.keys() or xe["objects"].keys() != ye["objects"].keys():
                raise ValueError("Object/predicate membership differs")
            for predicate in xp:
                p, q = xp[predicate], yp[predicate]
                if p.get("error") or q.get("error"):
                    errors.append({"index": xe["index"], "predicate": list(predicate)})
                elif p["value"] != q["value"]:
                    changes.append(
                        {
                            "index": xe["index"],
                            "predicate": list(predicate),
                            "left": p["value"],
                            "right": q["value"],
                        }
                    )
            for name, body in xe["objects"].items():
                distance = float(
                    np.linalg.norm(
                        np.asarray(body["position"]) - ye["objects"][name]["position"]
                    )
                )
                if not np.isfinite(distance):
                    raise ValueError("Nonfinite position distance")
                max_distance = max(max_distance, distance)
        results.append(
            {
                "suite": key[0],
                "task_id": key[1],
                "task_name": x["task_name"],
                "episodes": len(x["episodes"]),
                "predicate_changes": changes,
                "predicate_errors": errors,
                "max_object_position_delta_m": max_distance,
                "review_needed": bool(changes or errors or max_distance > 0.005),
                "qualification": "screening_only_not_policy_or_all_state_validation",
            }
        )
    return {
        "left_mujoco": left["versions"]["mujoco"],
        "right_mujoco": right["versions"]["mujoco"],
        "same_bank_and_reset_contract": True,
        "tasks": results,
        "review_tasks": sum(t["review_needed"] for t in results),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("left", "right", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        json.loads(args.left.read_text()), json.loads(args.right.read_text())
    )
    write_report(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "tasks"}))
