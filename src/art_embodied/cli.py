"""Command-line validation and paired-evaluation tools for ART-Embodied."""

from __future__ import annotations

from enum import Enum
import json
import os
from pathlib import Path

import typer

app = typer.Typer(help="Validate and inspect ART-Embodied experiments.")


class DoctorProfile(str, Enum):
    """Dependency profiles that can be checked without loading a policy."""

    control = "control"
    lerobot = "lerobot"
    libero = "libero"
    pi = "pi"


@app.command("doctor")
def diagnose_runtime(
    profile: DoctorProfile = typer.Option(
        DoctorProfile.control,
        "--profile",
        help=(
            "Validate the control, generic LeRobot, pinned OpenVLA/LIBERO, "
            "or PI0/PI0.5 profile."
        ),
    ),
    require_lerobot: bool = typer.Option(
        False,
        "--require-lerobot",
        help="Require the validated LeRobot version range in this environment.",
    ),
    worker: bool = typer.Option(
        False,
        "--worker",
        help="Validate a policy worker environment without requiring ART.",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Emit a machine-readable compatibility report.",
    ),
) -> None:
    """Check package compatibility before loading a policy or simulator."""

    from art_embodied.compatibility import runtime_compatibility_report

    resolved_profile = profile.value
    if require_lerobot:
        if profile is DoctorProfile.libero:
            typer.echo(
                "--require-lerobot cannot be combined with --profile libero; "
                "the validated LIBERO worker is an isolated profile",
                err=True,
            )
            raise typer.Exit(2)
        resolved_profile = DoctorProfile.lerobot.value
    report = runtime_compatibility_report(
        require_art=not worker,
        require_lerobot=require_lerobot,
        profile=resolved_profile,
    )
    payload = {"mode": "worker" if worker else "control-plane", **report.as_dict()}
    if as_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo("ART-Embodied runtime compatibility")
        typer.echo(f"  mode: {payload['mode']}")
        typer.echo(f"  profile: {report.profile}")
        typer.echo(f"  Python: {report.python_version}")
        typer.echo(f"  OpenPipe ART: {report.art_version or 'not installed'}")
        typer.echo(f"  LeRobot: {report.lerobot_version or 'not installed'}")
        for package, installed in report.package_versions.items():
            typer.echo(f"  {package}: {installed or 'not installed'}")
        typer.echo(f"  status: {'compatible' if report.compatible else 'incompatible'}")
        for issue in report.issues:
            typer.echo(f"  issue: {issue}")
    if not report.compatible:
        raise typer.Exit(1)


@app.command("validate")
def validate_embodied_config(
    config: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to a complete ART-Embodied YAML experiment contract.",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Emit a machine-readable execution summary.",
    ),
) -> None:
    """Validate embodied YAML without loading a policy or simulator."""

    try:
        from art_embodied import EmbodiedExperimentConfig
    except ImportError as exc:
        typer.echo(
            "Embodied support is not installed. Install art-embodied.",
            err=True,
        )
        raise typer.Exit(2) from exc

    try:
        experiment = EmbodiedExperimentConfig.from_yaml(config)
    except Exception as exc:
        typer.echo(f"Invalid embodied experiment: {exc}", err=True)
        raise typer.Exit(1) from exc

    summary = experiment.execution_summary()
    consumption = experiment.consumption_report()
    if as_json:
        typer.echo(
            json.dumps(
                {**summary, "config_consumption": consumption},
                indent=2,
                sort_keys=True,
            )
        )
        return

    typer.echo("Valid ART-Embodied experiment")
    typer.echo(f"  fingerprint: {summary['config_fingerprint']}")
    typer.echo(f"  policy: {summary['policy_type']}")
    typer.echo(f"  objective: {summary['algorithm']} / {summary['training_unit']}")
    typer.echo(f"  schedule: {summary['schedule']}")
    typer.echo(
        "  rollout update: "
        f"{summary['groups_per_update']} groups, "
        f"{summary['trajectories_per_update']} trajectories"
    )
    if summary["fixed_horizon_rows"] is not None:
        typer.echo(
            "  optimizer geometry: "
            f"{summary['fixed_horizon_rows']} fixed-horizon rows = "
            f"{summary['optimizer_rows']} optimizer rows"
        )
    typer.echo(
        "  run total: "
        f"{summary['updates']} updates, "
        f"{summary['total_trajectories']} trajectories"
    )
    typer.echo(
        "  local progress log: "
        + (
            "every "
            f"{summary['action_token_progress_every_microbatches']} microbatches, "
            f"capped at {summary['max_log_file_mb']} MiB"
            if summary["action_token_progress_enabled"]
            else "disabled"
        )
    )
    typer.echo(
        "  evaluation: "
        + (
            f"every {summary['evaluation_every_updates']} updates, "
            f"{summary['evaluation_episodes']} episodes, "
            f"role={summary['evaluation_data_role']}, "
            + (
                f"paired against {summary['baseline_outcomes_path']}"
                if summary["paired_baseline_source"] == "external_outcomes"
                else (
                    "paired against measured Step 0 policy"
                    if summary["paired_baseline_source"] == "measured_step_zero"
                    else "unpaired"
                )
            )
            if summary["evaluation_enabled"]
            else "disabled"
        )
    )
    typer.echo(
        "  W&B media: "
        f"train={summary['videos_per_update']} "
        f"(required={summary['wandb_train_video_required']}), "
        f"eval={summary['videos_per_evaluation']} "
        f"(required={summary['wandb_evaluation_video_required']})"
    )
    typer.echo(
        "  rollout resources: "
        f"{summary['rollout_device_count']} devices, "
        f"{summary['rollout_actor_count']} actors "
        f"({summary['rollout_active_actor_count']} active max), "
        f"{summary['rollout_model_replicas']} model replicas, "
        f"{summary['rollout_environment_slots']} environment slots "
        f"({summary['rollout_active_environment_slots']} active max)"
    )
    typer.echo(
        "  training resources: "
        f"{summary['training_device_count']} devices, "
        f"{summary['training_model_replicas']} model replicas, "
        f"time-shared={summary['rollout_training_time_shared']}"
    )
    if summary["serial_phase_device_reuse"]:
        typer.echo(
            "  serial device reuse: "
            + " -> ".join(summary["serial_phase_order"])
            + " on "
            + ", ".join(summary["shared_rollout_training_devices"])
        )
    typer.echo(
        "  policy worker Python: "
        f"{summary['worker_python_executable'] or 'current interpreter'}"
    )
    typer.echo(
        "  config contract: "
        f"{consumption['field_count']} fields, "
        f"{len(consumption['unowned_fields'])} unowned"
    )


@app.command("storage-preflight")
def preflight_embodied_storage(
    config: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Experiment YAML whose output storage must be checked.",
    ),
    required_root: Path | None = typer.Option(
        None,
        "--required-root",
        help=(
            "Required durable-storage root. Defaults to "
            "ART_EMBODIED_STORAGE_ROOT when set."
        ),
    ),
    minimum_free_gib: float = typer.Option(
        100.0,
        "--min-free-gib",
        min=0.0,
        help="Minimum free GiB required on the output filesystem.",
    ),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Verify durable output placement and capacity before requesting GPUs."""

    from art_embodied import EmbodiedExperimentConfig
    from art_embodied.storage import preflight_storage

    environment_root = os.environ.get("ART_EMBODIED_STORAGE_ROOT")
    storage_root = required_root or (
        Path(environment_root) if environment_root else None
    )
    try:
        experiment = EmbodiedExperimentConfig.from_yaml(config)
        report = preflight_storage(
            experiment.storage.output_dir,
            required_root=storage_root,
            minimum_free_gib=minimum_free_gib,
        )
    except Exception as exc:
        typer.echo(f"Unsafe experiment storage: {exc}", err=True)
        raise typer.Exit(1) from exc

    payload = report.as_dict()
    if as_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo("Valid ART-Embodied experiment storage")
    typer.echo(f"  output: {payload['resolved_output_dir']}")
    typer.echo(f"  required root: {payload['required_root'] or 'not enforced'}")
    typer.echo(f"  free: {int(payload['free_bytes']) / (1024**3):.1f} GiB")
    typer.echo(
        "  minimum free: "
        f"{int(payload['minimum_free_bytes']) / (1024**3):.1f} GiB"
    )


@app.command("compare-evaluations")
def compare_embodied_evaluations(
    baseline: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Baseline fixed-evaluation episode outcome JSON.",
    ),
    candidate: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Candidate fixed-evaluation episode outcome JSON.",
    ),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Compare baseline and candidate on the identical fixed episode plan."""

    from art_embodied import compare_paired_evaluation_reports

    try:
        metrics = compare_paired_evaluation_reports(baseline, candidate)
    except Exception as exc:
        typer.echo(f"Invalid paired evaluation: {exc}", err=True)
        raise typer.Exit(1) from exc
    if as_json:
        typer.echo(json.dumps(metrics, indent=2, sort_keys=True))
        return
    typer.echo("Paired ART-Embodied evaluation")
    typer.echo(f"  episodes: {int(metrics['episodes'])}")
    typer.echo(
        "  success: "
        f"{metrics['baseline_success_rate']:.1%} -> "
        f"{metrics['candidate_success_rate']:.1%} "
        f"({metrics['success_rate_lift']:+.1%})"
    )
    typer.echo(
        "  paired lift 95% CI: "
        f"[{metrics['success_rate_lift_ci95_low']:+.1%}, "
        f"{metrics['success_rate_lift_ci95_high']:+.1%}]"
    )
    typer.echo(
        "  task macro: "
        f"{metrics['baseline_task_macro_success_rate']:.1%} -> "
        f"{metrics['candidate_task_macro_success_rate']:.1%} "
        f"({metrics['task_macro_success_rate_lift']:+.1%})"
    )
    typer.echo(
        "  discordant pairs: "
        f"improved={int(metrics['improved_pairs'])}, "
        f"regressed={int(metrics['regressed_pairs'])}, "
        f"McNemar p={metrics['mcnemar_exact_p_value']:.6g}"
    )


@app.command("validate-evaluation-pair")
def validate_embodied_evaluation_pair(
    baseline_config: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Complete YAML used to produce the immutable baseline report.",
    ),
    candidate_config: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Complete YAML for the trained candidate evaluation.",
    ),
    baseline_step: int = typer.Option(0, min=0),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Validate paired evaluation semantics before allocating accelerators."""

    from art_embodied import (
        EmbodiedExperimentConfig,
        validate_paired_evaluation_configs,
    )

    try:
        baseline = EmbodiedExperimentConfig.from_yaml(baseline_config)
        candidate = EmbodiedExperimentConfig.from_yaml(candidate_config)
        summary = validate_paired_evaluation_configs(
            baseline,
            candidate,
            baseline_step=baseline_step,
        )
    except Exception as exc:
        typer.echo(f"Invalid paired evaluation contract: {exc}", err=True)
        raise typer.Exit(1) from exc

    if as_json:
        typer.echo(json.dumps(summary, indent=2, sort_keys=True))
        return
    typer.echo("Valid paired ART-Embodied evaluation")
    typer.echo(
        "  treatment: "
        f"{summary['baseline_policy_path']}@{summary['baseline_policy_revision']} -> "
        f"{summary['candidate_policy_path']}@{summary['candidate_policy_revision']}"
    )
    typer.echo(f"  baseline report: {summary['baseline_report_path']}")
    typer.echo(
        f"  plan: {summary['episodes']} episodes, "
        f"{len(summary['fixed_scenarios'])} scenarios, "
        f"{len(summary['seeds'])} seeds"
    )
    typer.echo(f"  runtime: {summary['runtime']} / {summary['split']}")


@app.command("prepare-evaluation-candidate")
def prepare_embodied_evaluation_candidate(
    baseline_config: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Claim-facing baseline YAML whose evaluation controls are preserved.",
    ),
    output_config: Path = typer.Argument(
        ...,
        dir_okay=False,
        help="Destination for the generated candidate YAML.",
    ),
    run: str = typer.Option(..., help="Candidate run identity."),
    output_dir: Path = typer.Option(..., help="Candidate artifacts directory."),
    adapter_path: Path | None = typer.Option(
        None,
        exists=True,
        help="ART-trained PEFT adapter directory.",
    ),
    policy_path: str | None = typer.Option(
        None,
        help="Complete candidate model/checkpoint path instead of an adapter.",
    ),
    policy_revision: str | None = typer.Option(
        None,
        help="Pinned revision for --policy-path.",
    ),
    baseline_step: int = typer.Option(0, min=0),
    baseline_wait_timeout_seconds: int = typer.Option(
        0,
        min=0,
        help=(
            "Wait for a concurrently scheduled baseline outcome report before "
            "paired comparison."
        ),
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite output_config."),
) -> None:
    """Generate a candidate YAML without drifting native evaluation controls."""

    from art_embodied import (
        EmbodiedExperimentConfig,
        create_paired_evaluation_candidate,
    )

    if output_config.exists() and not force:
        typer.echo(
            f"Refusing to overwrite existing candidate config: {output_config}",
            err=True,
        )
        raise typer.Exit(1)
    try:
        baseline = EmbodiedExperimentConfig.from_yaml(baseline_config)
        candidate = create_paired_evaluation_candidate(
            baseline,
            run=run,
            output_dir=output_dir,
            peft_adapter_path=adapter_path,
            policy_path=policy_path,
            policy_revision=policy_revision,
            baseline_step=baseline_step,
            baseline_wait_timeout_seconds=baseline_wait_timeout_seconds,
        )
        destination = candidate.to_yaml(output_config)
    except Exception as exc:
        typer.echo(f"Cannot prepare paired evaluation candidate: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo("Prepared paired ART-Embodied evaluation candidate")
    typer.echo(f"  config: {destination}")
    typer.echo(f"  treatment: {candidate.policy.path}")
    typer.echo(f"  baseline report: {candidate.evaluation.baseline_outcomes_path}")


if __name__ == "__main__":
    app()
