"""Inspect LIBERO banks or manifests without loading a policy or repairing states.

Run from a source checkout in the simulator environment. Exit 2 on a known
invalid reset or incomplete audit, unless --report-only is explicitly requested.
Other tasks are screened, not certified: BDDL placement predicates are not all
valid post-settle invariants. See docs/experimental/libero-reset-health.md.
"""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import sys

import numpy as np

from .state_manifest import file_sha256, load_state_manifest, state_sha256

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TASK5_BDDL_SHA256 = "10239815a9d18ad53e09365d453a362ee54d29ca276e04eef411cf1db8d72940"
TASK5_RELATION = ("on", "akita_black_bowl_1", "glazed_rim_porcelain_ramekin_1")


def select_indices(total: int, count: int | None) -> list[int]:
    if total < 1 or (count is not None and not 1 <= count <= total):
        raise ValueError("Sample count must be positive and no larger than the bank")
    return (
        list(range(total))
        if count is None
        else np.linspace(0, total - 1, count, dtype=int).tolist()
    )


def classify(
    bddl_sha: str, predicates: list[dict], *, finite: bool, initial_success: bool
) -> dict:
    failures = []
    if not finite:
        failures.append("non_finite_state")
    if initial_success:
        failures.append("already_successful_at_reset")
    if any(p.get("error") for p in predicates):
        failures.append("predicate_check_incomplete")
    known = bddl_sha == TASK5_BDDL_SHA256
    if known:
        matching = [p for p in predicates if tuple(p["predicate"]) == TASK5_RELATION]
        if len(matching) != 1 or matching[0].get("value") is not True:
            failures.append("spatial_task5_bowl_not_on_ramekin")
    return {
        "failures": failures,
        "known_support_rule_checked": known,
        "semantic_qualification": "known_support_rule_only"
        if known
        else "not_qualified",
        "unvalidated_false_predicates": [
            p["predicate"]
            for p in predicates
            if p.get("value") is False
            and not (known and tuple(p["predicate"]) == TASK5_RELATION)
        ],
    }


def write_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def manifest_task_states(manifest, task_id, bddl_sha):
    entries = [e for e in manifest.entries if e.task_id == task_id]
    if not entries or any(e.bddl_sha256 != bddl_sha for e in entries):
        raise ValueError(f"Missing task or BDDL mismatch in manifest: {task_id}")
    return entries, [manifest.states[e.state_key] for e in entries]


def object_snapshot(env):
    domain, sim = env.env, env.sim
    return {
        name: {
            "position": sim.data.body_xpos[body].tolist(),
            "quaternion": sim.data.body_xquat[body].tolist(),
            "model_position": sim.model.body_pos[body].tolist(),
            "joint_count": int(sim.model.body_jntnum[body]),
        }
        for name, body in domain.obj_body_id.items()
    }


def audit(args):
    import mujoco
    import torch

    from .environment import _ensure_legacy_gym_import, prepare_libero_runtime_paths

    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    _ensure_legacy_gym_import()
    runtime_paths = prepare_libero_runtime_paths()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    manifest = None
    if args.state_manifest:
        if len(args.suites) != 1:
            raise ValueError("A manifest requires exactly one suite")
        manifest = load_state_manifest(
            args.state_manifest,
            expected_suite_name=args.suites[0],
            expected_simulator_compatibility="rlinf_v01",
        )
    versions = {
        p: metadata.version(p) for p in ("mujoco", "robosuite", "hf-libero", "numpy")
    }
    if (
        versions["mujoco"] != mujoco.__version__
        or mujoco.__version__ != mujoco.mj_versionString()
    ):
        raise RuntimeError("MuJoCo package and native engine disagree")
    report = {
        "schema_version": 1,
        "scope": __doc__,
        "versions": versions,
        "reset": {
            "seed": args.seed,
            "wait_steps": args.wait_steps,
            "gripper_open": True,
            "control_mode": args.control_mode,
        },
        "runtime_paths": runtime_paths,
        "state_source": "manifest" if manifest else "official",
        "manifest_sha256": file_sha256(manifest.path) if manifest else None,
        "sampling": "all" if args.all_states else "evenly_spaced_bank_indices",
        "policy_loaded": False,
        "states_modified": False,
        "sealed_accessed": None,
        "split_scope": "not_inferred; caller must authorize the input population",
        "completed": False,
        "tasks": [],
    }
    path = args.output / "report.json"
    write_report(path, report)
    try:
        for suite_name in args.suites:
            suite = benchmark.get_benchmark_dict()[suite_name]()
            tasks = (
                list(range(suite.n_tasks)) if args.task_ids is None else args.task_ids
            )
            if (
                not tasks
                or len(set(tasks)) != len(tasks)
                or any(not 0 <= i < suite.n_tasks for i in tasks)
            ):
                raise ValueError(f"Invalid task IDs for {suite_name}")
            for task_id in tasks:
                task = suite.get_task(task_id)
                bddl = (
                    Path(get_libero_path("bddl_files"))
                    / task.problem_folder
                    / task.bddl_file
                )
                bank = (
                    Path(get_libero_path("init_states"))
                    / task.problem_folder
                    / task.init_states_file
                )
                entries = None
                if manifest:
                    entries, states = manifest_task_states(
                        manifest, task_id, file_sha256(bddl)
                    )
                    bank = manifest.archive_path
                else:
                    states = torch.load(bank, weights_only=False)
                indices = select_indices(
                    len(states), None if args.all_states else args.states_per_task
                )
                row = {
                    "suite": suite_name,
                    "task_id": task_id,
                    "task_name": task.name,
                    "bddl_sha256": file_sha256(bddl),
                    "bank_sha256": file_sha256(bank),
                    "bank_size": len(states),
                    "episodes": [],
                }
                report["tasks"].append(row)
                env = ControlEnv(
                    bddl_file_name=str(bddl),
                    use_camera_obs=args.render,
                    has_offscreen_renderer=args.render,
                    has_renderer=False,
                    camera_heights=256,
                    camera_widths=256,
                )
                try:
                    if args.control_mode == "relative":
                        for robot in env.robots:
                            robot.controller.use_delta = True
                    row["controller_use_delta"] = [
                        bool(robot.controller.use_delta) for robot in env.robots
                    ]
                    for index in indices:
                        env.seed(args.seed)
                        env.reset()
                        obs = env.set_init_state(states[index])
                        dummy = np.zeros(7, dtype=np.float64)
                        dummy[-1] = -1
                        timeline = []

                        def capture(step):
                            if args.trace_reset:
                                timeline.append(
                                    {
                                        "control_step": step,
                                        "objects": object_snapshot(env),
                                        "max_abs_qvel": float(
                                            np.max(np.abs(env.sim.data.qvel))
                                        ),
                                        "contacts": int(env.sim.data.ncon),
                                    }
                                )

                        capture(0)
                        for step in range(1, args.wait_steps + 1):
                            obs, *_ = env.step(dummy)
                            if step in {1, 5, 10, args.wait_steps}:
                                capture(step)
                        domain, sim = env.env, env.sim
                        finite = bool(np.isfinite(sim.get_state().flatten()).all())
                        predicates = []
                        for p in domain.parsed_problem["initial_state"]:
                            item = {"predicate": list(p), "value": None, "error": None}
                            try:
                                item["value"] = bool(domain._eval_predicate(p))
                            except Exception as exc:
                                item["error"] = repr(exc)
                            predicates.append(item)
                        episode = {
                            "index": index,
                            "source_state_sha256": state_sha256(states[index]),
                            "predicates": predicates,
                            **classify(
                                row["bddl_sha256"],
                                predicates,
                                finite=finite,
                                initial_success=bool(env.check_success()),
                            ),
                        }
                        episode["objects"] = object_snapshot(env) if finite else {}
                        if entries:
                            episode["manifest_id"] = entries[index].id
                            episode["state_key"] = entries[index].state_key
                        if args.trace_reset:
                            episode["reset_timeline"] = timeline
                        row["episodes"].append(episode)
                        if args.render and index == indices[0]:
                            from PIL import Image

                            image = f"{suite_name}-{task_id:02d}.png"
                            Image.fromarray(
                                np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                            ).save(args.output / image)
                            row["image"] = image
                finally:
                    env.close()
                write_report(path, report)
                print(
                    json.dumps(
                        {
                            "suite": suite_name,
                            "task": task_id,
                            "episodes": len(row["episodes"]),
                            "failures": sum(
                                bool(e["failures"]) for e in row["episodes"]
                            ),
                        }
                    ),
                    flush=True,
                )
        report["completed"] = True
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        write_report(path, report)
    return report


def exit_code(report, report_only=False):
    if not report.get("completed"):
        return 2
    failed = any(e["failures"] for t in report["tasks"] for e in t["episodes"])
    return 2 if failed and not report_only else 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suites", nargs="+", choices=SUITES, default=["libero_spatial"])
    p.add_argument("--task-ids", type=int, nargs="+")
    p.add_argument("--states-per-task", type=int, default=3)
    p.add_argument("--all-states", action="store_true")
    p.add_argument("--wait-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--state-manifest", type=Path)
    p.add_argument(
        "--control-mode", choices=["relative", "unchanged"], default="relative"
    )
    p.add_argument("--trace-reset", action="store_true")
    p.add_argument(
        "--render",
        action="store_true",
        help="Save one reset image per task; requires a renderer",
    )
    p.add_argument(
        "--report-only",
        action="store_true",
        help="Record known failures without returning exit 2",
    )
    p.add_argument("--output", type=Path, required=True)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.wait_steps < 0 or args.states_per_task < 1:
        p.error("wait-steps must be nonnegative and states-per-task positive")
    if len(set(args.suites)) != len(args.suites):
        p.error("suites must be unique")
    try:
        report = audit(args)
    except Exception as exc:
        print(f"Reset audit failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "scope": "screening_only_not_full_semantic_qualification",
                "report": str(args.output / "report.json"),
                "failed_states": sum(
                    bool(e["failures"]) for t in report["tasks"] for e in t["episodes"]
                ),
            }
        )
    )
    return exit_code(report, args.report_only)


if __name__ == "__main__":
    raise SystemExit(main())
