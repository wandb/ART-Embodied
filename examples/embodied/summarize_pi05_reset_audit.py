"""Compare paired reset timelines and reference-run task outcomes, without GPU."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from examples.embodied.libero.audit_reset import write_report
from examples.embodied.libero.compare_reset_audits import compare
from examples.embodied.libero.state_manifest import file_sha256


def timeline_comparison(left, right):
    comparison = compare(left, right)
    tasks = []
    for a, b in zip(left["tasks"], right["tasks"], strict=True):
        if (a["suite"], a["task_id"]) != (b["suite"], b["task_id"]):
            raise ValueError("Task order differs")
        if a["controller_use_delta"] != b["controller_use_delta"]:
            raise ValueError("Actual controller mode differs")
        distances = defaultdict(list)
        worst = {}
        for x, y in zip(a["episodes"], b["episodes"], strict=True):
            xt, yt = x["reset_timeline"], y["reset_timeline"]
            if [f["control_step"] for f in xt] != [f["control_step"] for f in yt]:
                raise ValueError("Timeline checkpoints differ")
            for xf, yf in zip(xt, yt, strict=True):
                step = xf["control_step"]
                if xf["objects"].keys() != yf["objects"].keys():
                    raise ValueError("Timeline objects differ")
                ds = {
                    name: float(
                        np.linalg.norm(
                            np.asarray(body["position"])
                            - yf["objects"][name]["position"]
                        )
                    )
                    for name, body in xf["objects"].items()
                }
                name = max(ds, key=ds.get)
                if not all(np.isfinite(v) for v in ds.values()):
                    raise ValueError("Nonfinite timeline difference")
                distances[step].append(ds[name])
                if step not in worst or ds[name] > worst[step]["distance_m"]:
                    worst[step] = {
                        "index": x["index"],
                        "object": name,
                        "distance_m": ds[name],
                        "left_position": xf["objects"][name]["position"],
                        "right_position": yf["objects"][name]["position"],
                    }
        tasks.append(
            {
                "task_id": a["task_id"],
                "task_name": a["task_name"],
                "states": len(a["episodes"]),
                "timeline": [
                    {
                        "control_step": step,
                        "max_position_delta_m": max(ds),
                        "median_state_max_position_delta_m": float(np.median(ds)),
                        "states_with_delta_above_5mm": sum(d > 0.005 for d in ds),
                        "states_with_delta_above_5cm": sum(d > 0.05 for d in ds),
                        "worst": worst[step],
                    }
                    for step, ds in distances.items()
                ],
            }
        )
    return {
        "final_comparison": comparison,
        "tasks": tasks,
        "threshold_scope": "5mm and 5cm are descriptive thresholds, not task success criteria",
    }


def reference_outcomes(reference, manifest_path):
    manifest_hash = file_sha256(manifest_path)
    populations, by_update = {}, {}
    for update in (0, 250):
        prefix = reference / f"evaluation-{update}"
        evidence = json.loads(
            (prefix / f"update_{update:06d}_evidence.json").read_text()
        )
        path = prefix / f"update_{update:06d}_episode_outcomes.json"
        if evidence["outcomes"]["sha256"] != file_sha256(path):
            raise ValueError("Reference outcomes hash mismatch")
        if evidence["identity"]["evaluation_manifest"]["sha256"] != manifest_hash:
            raise ValueError("Historical evaluation used a different manifest")
        episodes = json.loads(path.read_text())["episodes"]
        tasks = defaultdict(lambda: {"episodes": 0, "successes": 0})
        population = []
        for e in episodes:
            r = e["reset_info"]
            if (
                r["environment_seed"],
                r["wait_steps_after_reset"],
                r["control_mode"],
                r["state_source"],
                r["reset_gripper_open"],
            ) != (0, 15, "unchanged", "manifest", True):
                raise ValueError("Historical episode has a different reset contract")
            if (
                r["simulator_runtime"]["provider"] != "hf-libero"
                or r["simulator_runtime"]["distribution_version"] != "0.1.4"
            ):
                raise ValueError("Different historical simulator provider")
            if not e["completed"] or e["error"]:
                raise ValueError("Reference contains incomplete episodes")
            population.append(e["scenario_id"])
            task = tasks[r["task_id"]]
            task["episodes"] += 1
            task["successes"] += int(e["success"])
        if len(population) != 100 or len(set(population)) != 100:
            raise ValueError("Invalid historical population")
        populations[update] = population
        by_update[update] = dict(tasks)
    if populations[0] != populations[250]:
        raise ValueError("Historical populations differ")
    return {
        "manifest_sha256": manifest_hash,
        "same_100_scenarios": True,
        "all_200_reset_contracts_verified": True,
        "task_outcomes": by_update,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "reference", "manifest"):
        p.add_argument(f"--{name}", type=Path, required=True)
    args = p.parse_args()
    result = {"reference": reference_outcomes(args.reference, args.manifest)}
    for population in ("dev", "official"):
        left, right = [
            json.loads((args.audit / f"{population}-{engine}/report.json").read_text())
            for engine in ("mj330", "mj381")
        ]
        result[population] = timeline_comparison(left, right)
    write_report(args.audit / "timeline-summary.json", result)
    print(json.dumps({"output": str(args.audit / "timeline-summary.json")}))
