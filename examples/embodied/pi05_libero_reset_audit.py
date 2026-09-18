"""Audit PI0.5 Long success-run reset conditions without loading a policy."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from examples.embodied.libero.audit_reset import write_report
from examples.embodied.libero.compare_reset_audits import compare
from examples.embodied.libero.state_manifest import file_sha256, load_state_manifest
from examples.embodied.pi0_fast_spatial_postfit_audit import (
    close_with_verification,
    verify,
)
from examples.embodied.pi0_fast_spatial_teacher_control import verify_wandb_identity


def phases():
    return [
        ("current-first-dev", False, True, True),
        ("dev-mj330", True, True, False),
        ("dev-mj381", False, True, False),
        ("official-mj330", True, False, False),
        ("official-mj381", False, False, False),
    ]


def validate_reference(config, manifest):
    env, evaluation = config["environment"], config["evaluation"]
    kw = env["kwargs"]
    expected = {
        "task_ids": list(range(10)),
        "wait_steps_after_reset": 15,
        "control_mode": "unchanged",
        "observation_height": 256,
        "observation_width": 256,
        "rotate_images_180": True,
    }
    if env["task"] != "libero_10" or not env["reset"]["reset_gripper_open"]:
        raise ValueError("Unexpected successful-run environment")
    if any(kw[k] != v for k, v in expected.items()):
        raise ValueError("Reference reset contract changed")
    if evaluation["seeds"] != [0] or evaluation["episodes"] != 100:
        raise ValueError("Unexpected evaluation population")
    contract = evaluation["kwargs"]["seed_contract"]
    if (
        contract["environment_mode"] != "fixed"
        or contract["fixed_environment_seed"] != 0
    ):
        raise ValueError("Unexpected environment seed contract")
    if [e.id for e in manifest.entries] != evaluation["fixed_scenarios"]:
        raise ValueError("Manifest differs from the successful run's dev membership")
    if str(manifest.path).endswith(kw["evaluation_state_manifest"]) is False:
        raise ValueError("Wrong manifest path")


def main(args):
    import wandb

    if os.environ.get("SLURM_JOB_NUM_NODES") != "1":
        raise ValueError("One node required")
    viewer = wandb.Api(timeout=30).viewer
    verify_wandb_identity(viewer)
    config = json.loads((args.reference / "config.json").read_text())
    manifest = load_state_manifest(args.manifest, expected_suite_name="libero_10")
    validate_reference(config, manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    plan = {
        "scope": __doc__,
        "reference_run": "t0a9mnd3",
        "reference_config_sha256": file_sha256(args.reference / "config.json"),
        "manifest_sha256": file_sha256(manifest.path),
        "archive_sha256": file_sha256(manifest.archive_path),
        "engines": ["3.3.0", "3.8.1"],
        "seed": 0,
        "wait_steps": 15,
        "control_mode": "unchanged",
        "gripper_open": True,
        "dev_states": 100,
        "official_states": 500,
        "policy_loaded": False,
        "sealed_accessed": False,
        "scope_limit": "Reset replay, not historical binary/assets provenance or policy performance qualification",
    }
    write_report(args.output / "plan.json", plan)
    run = wandb.init(
        entity="wandb-japan",
        project="art-embodied-libero-reset-health",
        name=f"pi05-long-success-recipe-reset-{os.environ['SLURM_JOB_ID']}",
        job_type="reset-compatibility-diagnostic",
        config=plan,
    )
    run.define_metric("diagnostics/stage")
    for key in ("diagnostics/*", "media/*"):
        run.define_metric(key, step_metric="diagnostics/stage")
    write_report(args.output / "run.json", {"id": run.id, "url": run.url})
    records, reports = [], {}
    try:
        for stage, (label, old, dev, first) in enumerate(phases()):
            output = args.output / label
            env = dict(os.environ)
            if old:
                env["PYTHONPATH"] = (
                    str(args.old_engine) + ":" + env.get("PYTHONPATH", "")
                )
            command = [
                sys.executable,
                "-u",
                "-m",
                "examples.embodied.libero.audit_reset",
                "--output",
                str(output),
                "--suites",
                "libero_10",
                "--wait-steps",
                "15",
                "--control-mode",
                "unchanged",
                "--trace-reset",
                "--render",
                "--report-only",
            ]
            if dev:
                command += ["--state-manifest", str(manifest.path)]
            command += (
                ["--task-ids", "0", "--states-per-task", "1"]
                if first
                else ["--all-states"]
            )
            with (args.output / f"{label}.log").open("w") as log:
                subprocess.run(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            report = json.loads((output / "report.json").read_text())
            if not report["completed"] or report["versions"]["mujoco"] != (
                "3.3.0" if old else "3.8.1"
            ):
                raise ValueError("Incomplete report or wrong engine")
            episodes = [e for t in report["tasks"] for e in t["episodes"]]
            if len(episodes) != (1 if first else 100 if dev else 500):
                raise ValueError("Wrong reset population size")
            reports[label] = report
            for population in ("dev", "official"):
                if label == f"{population}-mj381":
                    comparison = compare(reports[f"{population}-mj330"], report)
                    write_report(
                        args.output / f"{population}-comparison.json", comparison
                    )
                    reports[f"{population}-comparison"] = comparison
            result = args.output / "results.json"
            write_report(result, reports)
            record = {
                "diagnostics/stage": stage,
                "diagnostics/tasks": len(report["tasks"]),
                "diagnostics/states": len(episodes),
                "diagnostics/failed_state_checks": sum(
                    bool(e["failures"]) for e in episodes
                ),
                "diagnostics/unvalidated_false_predicates": sum(
                    len(e["unvalidated_false_predicates"]) for e in episodes
                ),
            }
            records.append(record)
            write_report(args.output / "records.json", records)
            run.log(
                record
                | {
                    "media/teacher_observation": wandb.Image(
                        str(output / report["tasks"][0]["image"]),
                        caption=f"{label}: environment reset only; no teacher or policy",
                    )
                }
            )
            table = wandb.Table(
                columns=["suite", "task", "reset_image", "states", "checked_failures"]
            )
            for task in report["tasks"]:
                table.add_data(
                    task["suite"],
                    task["task_name"],
                    wandb.Image(str(output / task["image"])),
                    len(task["episodes"]),
                    sum(bool(e["failures"]) for e in task["episodes"]),
                )
            artifact = wandb.Artifact(f"pi05-reset-{run.id}-{stage}", type="diagnostic")
            artifact.add_file(str(result))
            artifact.add_file(str(args.output / "plan.json"))
            artifact.add(table, "task_resets")
            name = run.log_artifact(artifact).wait().qualified_name
            verify(run, records, result, name, args.output)
            print(json.dumps(record | {"label": label}), flush=True)
        delivery = close_with_verification(run, records, result, name, args.output)
        write_report(args.output / "complete.json", delivery)
    except BaseException as exc:
        write_report(args.output / "failure.json", {"error": repr(exc)})
        run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "old-engine", "reference", "manifest"):
        p.add_argument(f"--{name}", type=Path, required=True)
    main(p.parse_args())
