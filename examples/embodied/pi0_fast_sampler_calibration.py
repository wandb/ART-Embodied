"""Compare greedy and sampled pi0-FAST behavior on identical LIBERO resets."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from art_embodied import (
    EmbodiedExperimentConfig,
    ExperimentProgress,
    WandbWeaveObserver,
    make_policy,
    run_lerobot_evaluation,
    validate_runtime_device_availability,
)
from art_embodied.compatibility import (
    require_compatible_runtime,
    runtime_profile_for_policy,
)
from art_embodied.evaluation import compare_paired_evaluation_reports
from art_embodied.experiment import EvaluationResult
from examples.embodied.libero.components import (
    LiberoSettings,
    build_evaluation_scenarios,
    prepare_libero_runtime_paths,
    validate_libero_runtime_imports,
    validate_libero_task_assets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--greedy-outcomes",
        type=Path,
        help=(
            "Reuse a completed greedy episode-outcome report and run only the "
            "sampled candidates. The report must use the YAML's fixed plan."
        ),
    )
    parser.add_argument(
        "--temperature",
        action="append",
        type=float,
        dest="temperatures",
        help=(
            "Sampled decoding temperature. Repeat for a bounded sweep. "
            "The rollout-generation temperature is used when omitted."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        help="Optional bounded prefix of the YAML's fixed evaluation plan.",
    )
    parser.add_argument(
        "--max-absolute-success-rate-gap",
        type=float,
        help=(
            "Fail after logging all evidence if any sampled-minus-greedy "
            "success-rate gap exceeds this fraction."
        ),
    )
    return parser.parse_args()


def _condition_config(
    base: EmbodiedExperimentConfig,
    *,
    output_dir: Path,
    label: str,
    do_sample: bool,
    temperature: float,
    episodes: int | None,
) -> EmbodiedExperimentConfig:
    """Change only the decoder treatment and evidence destinations."""

    evaluation_kwargs = dict(base.evaluation.kwargs)
    seed_contract = dict(evaluation_kwargs.get("seed_contract", {}))
    seed_contract["policy_mode"] = "derived"
    evaluation_kwargs["seed_contract"] = seed_contract
    evaluation = base.evaluation.model_copy(
        update={
            "evaluate_before_training": False,
            "pre_training_success_gate": None,
            "baseline_outcomes_path": None,
            "data_role": "diagnostic",
            "episodes": int(episodes or base.evaluation.episodes),
            "deterministic": not do_sample,
            "temperature": float(temperature),
            "kwargs": evaluation_kwargs,
        }
    )
    generation = base.policy.evaluation_generation.model_copy(
        update={
            "do_sample": do_sample,
            "temperature": float(temperature),
            "top_p": 1.0,
        }
    )
    policy = base.policy.model_copy(update={"evaluation_generation": generation})
    tags = list(base.experiment.tags)
    for tag in ("sampler-calibration", "same-reset", label):
        if tag not in tags:
            tags.append(tag)
    experiment = base.experiment.model_copy(
        update={
            "run": f"{base.experiment.run}-sampler-{label}",
            "tags": tags,
        }
    )
    wandb = base.observability.wandb.model_copy(
        update={
            "run_id": None,
            "resume": None,
            "group": f"{base.observability.wandb.group}-sampler-calibration",
            "job_type": "sampler-calibration",
        }
    )
    observability = base.observability.model_copy(update={"wandb": wandb})
    storage = base.storage.model_copy(
        update={
            "output_dir": output_dir / label,
            "resume_from_checkpoint": None,
        }
    )
    return base.model_copy(
        update={
            "experiment": experiment,
            "policy": policy,
            "evaluation": evaluation,
            "observability": observability,
            "storage": storage,
        }
    )


async def _evaluate_condition(
    config: EmbodiedExperimentConfig,
    *,
    measured_baseline_path: Path | None,
) -> tuple[Path, dict[str, float]]:
    """Use the normal observable evaluation path for one decoder condition."""

    observer = WandbWeaveObserver.start(config)
    run_error: BaseException | None = None
    try:
        await observer.log_progress(
            ExperimentProgress(
                update=0,
                phase="initialization",
                status="started",
                message="Loading pi0-FAST policy for sampler calibration",
            ),
            config,
        )
        policy = make_policy(config)
        await observer.log_progress(
            ExperimentProgress(
                update=0,
                phase="initialization",
                status="completed",
                message="Policy loaded",
            ),
            config,
        )
        result = await run_lerobot_evaluation(
            config=config,
            policy=policy,
            evaluation_scenarios=build_evaluation_scenarios(config),
            step=0,
            observer=observer,
            checkpoint_role="sampler_calibration",
            measured_baseline_path=measured_baseline_path,
        )
        diagnostics = _sampler_diagnostics(result.evaluation)
        diagnostics_path = config.storage.output_dir / "sampler_diagnostics.json"
        diagnostics_path.write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        await observer.log_progress(
            ExperimentProgress(
                update=0,
                phase="sampler_calibration",
                status="completed",
                completed=config.evaluation.episodes,
                total=config.evaluation.episodes,
                metrics={
                    "success_rate": float(result.evaluation.metrics["success_rate"]),
                    **diagnostics,
                },
            ),
            config,
        )
        return (
            Path(result.evaluation.artifacts["episode_outcomes_json"]),
            diagnostics,
        )
    except BaseException as exc:
        run_error = exc
        await observer.log_progress(
            ExperimentProgress(
                update=0,
                phase="sampler_calibration",
                status="failed",
                message=f"{type(exc).__name__}: {exc}",
            ),
            config,
        )
        raise
    finally:
        observer.close(exit_code=1 if run_error is not None else 0)


def _temperature_label(temperature: float) -> str:
    return f"sample-t{temperature:g}".replace(".", "p")


def _sampler_diagnostics(evaluation: EvaluationResult) -> dict[str, float]:
    """Summarize FAST grammar health without retaining rollout payloads."""

    grammar_valid: list[float] = []
    generated_token_counts: list[float] = []
    discarded_token_counts: list[float] = []
    episodes_with_invalid_action = 0
    for trajectory in evaluation.trajectories:
        invalid_in_episode = False
        for action in trajectory.actions:
            if action.kind != "token":
                continue
            metadata = action.metadata
            valid = bool(metadata.get("action_grammar_valid", False))
            grammar_valid.append(float(valid))
            invalid_in_episode |= not valid
            generated_token_counts.append(
                float(metadata.get("generated_token_count", 0))
            )
            discarded_token_counts.append(
                float(metadata.get("post_termination_tokens_discarded", 0))
            )
        episodes_with_invalid_action += int(invalid_in_episode)
    return {
        "action_chunks": float(len(grammar_valid)),
        "action_grammar_valid_rate": _mean_or_zero(grammar_valid),
        "episodes_with_invalid_action_rate": (
            episodes_with_invalid_action / len(evaluation.trajectories)
            if evaluation.trajectories
            else 0.0
        ),
        "generated_token_count_mean": _mean_or_zero(generated_token_counts),
        "post_termination_tokens_discarded_mean": _mean_or_zero(discarded_token_counts),
    }


def _mean_or_zero(values: Sequence[float]) -> float:
    return float(fmean(values)) if values else 0.0


def _validate_args(args: argparse.Namespace, base: EmbodiedExperimentConfig) -> None:
    if base.policy.type != "pi0_fast":
        raise ValueError("pi0-FAST sampler calibration requires policy.type='pi0_fast'")
    if not base.evaluation.enabled:
        raise ValueError(
            "pi0-FAST sampler calibration requires evaluation.enabled=true"
        )
    if args.episodes is not None and args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if args.temperatures and any(value <= 0.0 for value in args.temperatures):
        raise ValueError("--temperature values must be positive")
    gap = args.max_absolute_success_rate_gap
    if gap is not None and not 0.0 <= gap <= 1.0:
        raise ValueError("--max-absolute-success-rate-gap must be in [0, 1]")


async def run(args: argparse.Namespace) -> None:
    base = EmbodiedExperimentConfig.from_yaml(args.config)
    _validate_args(args, base)
    profile = runtime_profile_for_policy(base.policy.type)
    require_compatible_runtime(profile=profile)
    prepare_libero_runtime_paths()
    validate_libero_runtime_imports()
    validate_libero_task_assets(LiberoSettings.from_config(base))
    validate_runtime_device_availability(base, include_training_devices=False)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    temperatures = args.temperatures or [
        float(base.policy.rollout_generation.temperature)
    ]
    greedy = _condition_config(
        base,
        output_dir=output_dir,
        label="greedy",
        do_sample=False,
        temperature=1.0,
        episodes=args.episodes,
    )
    if args.greedy_outcomes is None:
        greedy_report, greedy_diagnostics = await _evaluate_condition(
            greedy,
            measured_baseline_path=None,
        )
    else:
        greedy_report = args.greedy_outcomes.expanduser().resolve()
        greedy_diagnostics = None
        if not greedy_report.is_file():
            raise FileNotFoundError(
                f"Greedy outcome report does not exist: {greedy_report}"
            )

    candidates: list[dict[str, Any]] = []
    for temperature in temperatures:
        label = _temperature_label(temperature)
        sampled = _condition_config(
            base,
            output_dir=output_dir,
            label=label,
            do_sample=True,
            temperature=temperature,
            episodes=args.episodes,
        )
        sampled_report, sampled_diagnostics = await _evaluate_condition(
            sampled,
            measured_baseline_path=greedy_report,
        )
        paired = compare_paired_evaluation_reports(
            greedy_report,
            sampled_report,
        )
        gap_passed = (
            args.max_absolute_success_rate_gap is None
            or abs(float(paired["success_rate_lift"]))
            <= args.max_absolute_success_rate_gap
        )
        candidates.append(
            {
                "temperature": temperature,
                "outcomes": str(sampled_report),
                "sampler_diagnostics": sampled_diagnostics,
                "paired_vs_greedy": paired,
                "gap_gate_passed": gap_passed,
            }
        )

    payload = {
        "schema_version": 1,
        "kind": "pi0_fast_same_reset_sampler_calibration",
        "config": str(args.config.expanduser().resolve()),
        "policy_path": base.policy.path,
        "policy_revision": base.policy.revision,
        "episodes": greedy.evaluation.episodes,
        "fixed_scenarios": list(greedy.evaluation.fixed_scenarios),
        "greedy_outcomes": str(greedy_report),
        "greedy_sampler_diagnostics": greedy_diagnostics,
        "candidates": candidates,
        "max_absolute_success_rate_gap": args.max_absolute_success_rate_gap,
    }
    payload["passed"] = all(item["gap_gate_passed"] for item in candidates)
    summary_path = output_dir / "pi0-fast-sampler-calibration.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
