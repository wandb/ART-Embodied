"""Compare native Flow ODE and exploratory Flow-SDE on fixed scenarios.

The diagnostic supports every ART-Embodied flow policy that exposes the
native-evaluation and stochastic-training rollout phases. It intentionally
bypasses W&B/Weave and writes a bounded, paired evidence bundle under one
operator-selected directory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Literal

from art_embodied import EmbodiedExperimentConfig, make_policy
from art_embodied.compatibility import (
    require_compatible_runtime,
    runtime_profile_for_policy,
)
from art_embodied.evaluation import (
    FixedScenarioEvaluator,
    compare_paired_evaluation_reports,
)
from art_embodied.experiment import ExperimentProgress
from art_embodied.rollout_process import LocalProcessRolloutPool
from art_embodied.types import LocalTrainResult
from examples.embodied.libero.components import (
    LiberoSettings,
    build_evaluation_scenarios,
    prepare_libero_runtime_paths,
    validate_libero_runtime_imports,
    validate_libero_task_assets,
)

_SUPPORTED_FLOW_POLICIES = frozenset({"pi0", "pi05", "smolvla"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-path",
        "--adapter-path",
        dest="checkpoint_path",
        type=Path,
        help=(
            "Optional ART policy snapshot to apply before calibration. "
            "--adapter-path remains as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--noise-level",
        action="append",
        type=float,
        dest="noise_levels",
        help=(
            "Flow-SDE noise level to compare. Repeat for a sweep. The YAML "
            "value is used when omitted."
        ),
    )
    parser.add_argument(
        "--denoise-steps",
        action="append",
        type=int,
        dest="denoise_steps",
        help=(
            "Denoising-step count to compare. Repeat for a sweep. Each value "
            "gets its own native-ODE baseline because changing the numerical "
            "integration schedule also changes deterministic behavior."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        help="Optional bounded prefix of the YAML's fixed evaluation plan.",
    )
    parser.add_argument(
        "--policy-seed-repetitions",
        type=int,
        default=1,
        help=(
            "Repeat every fixed scenario with independently derived policy seeds. "
            "This estimates the sampler gap without changing the reset population."
        ),
    )
    parser.add_argument(
        "--max-absolute-success-rate-gap",
        type=float,
        help=(
            "Fail after writing evidence if any observed SDE-minus-ODE success-rate "
            "gap exceeds this fraction. Omit for a report-only diagnostic."
        ),
    )
    return parser.parse_args()


def _diagnostic_config(
    base: EmbodiedExperimentConfig,
    *,
    output_dir: Path,
    noise_level: float | None,
    denoise_steps: int | None,
    episodes: int | None,
    policy_seed_repetitions: int = 1,
) -> EmbodiedExperimentConfig:
    algorithm = base.algorithm
    if noise_level is not None or denoise_steps is not None:
        assert algorithm.flow_sde is not None
        flow_sde_updates = {}
        if noise_level is not None:
            flow_sde_updates["noise_level"] = float(noise_level)
        if denoise_steps is not None:
            flow_sde_updates["num_denoise_steps"] = int(denoise_steps)
        flow_sde = algorithm.flow_sde.model_copy(update=flow_sde_updates)
        algorithm = algorithm.model_copy(update={"flow_sde": flow_sde})

    evaluation_kwargs = dict(base.evaluation.kwargs)
    seed_contract = dict(evaluation_kwargs.get("seed_contract", {}))
    # ODE and SDE share each state and policy seed. Distinct states still use
    # independent policy noise, avoiding one repeated Gaussian stream.
    seed_contract["policy_mode"] = "derived"
    evaluation_kwargs["seed_contract"] = seed_contract
    fixed_scenario_count = len(base.evaluation.fixed_scenarios)
    requested_episodes = int(episodes or base.evaluation.episodes)
    if policy_seed_repetitions > 1:
        if episodes is not None:
            raise ValueError(
                "--episodes and --policy-seed-repetitions cannot be combined"
            )
        requested_episodes = fixed_scenario_count * policy_seed_repetitions
    evaluation = base.evaluation.model_copy(
        update={
            "baseline_outcomes_path": None,
            "data_role": "diagnostic",
            "episodes": requested_episodes,
            "kwargs": evaluation_kwargs,
        }
    )
    storage = base.storage.model_copy(update={"output_dir": output_dir})
    return base.model_copy(
        update={
            "algorithm": algorithm,
            "evaluation": evaluation,
            "storage": storage,
        }
    )


def _load_checkpoint(policy: object, path: Path, *, policy_type: str) -> None:
    """Apply an ART policy snapshot before the rollout pool snapshots it."""

    loader = getattr(policy, "load_checkpoint", None)
    if not callable(loader):
        raise TypeError(
            f"policy.type={policy_type!r} does not support checkpoint loading"
        )
    loader(path.expanduser().resolve())


def _sampler_gap_passes(
    comparison: dict[str, float],
    *,
    maximum_absolute_gap: float | None,
) -> bool:
    if maximum_absolute_gap is None:
        return True
    return abs(float(comparison["success_rate_lift"])) <= maximum_absolute_gap


async def _evaluate_sampler(
    config: EmbodiedExperimentConfig,
    *,
    phase: Literal["train", "eval"],
    checkpoint_path: Path | None,
) -> Path:
    async def log_progress(
        progress: ExperimentProgress,
        _config: EmbodiedExperimentConfig,
    ) -> None:
        completed = progress.completed
        total = progress.total
        if completed is not None and total is not None and (
            completed == total or completed % 10 == 0
        ):
            print(
                f"[sampler-calibration] sampler={phase} "
                f"completed={completed}/{total}",
                flush=True,
            )

    policy = make_policy(config)
    if checkpoint_path is not None:
        _load_checkpoint(policy, checkpoint_path, policy_type=config.policy.type)
    scenarios = build_evaluation_scenarios(config)
    pool = LocalProcessRolloutPool(config=config, policy=policy)
    try:
        evaluator = FixedScenarioEvaluator(
            config=config,
            scenarios=scenarios,
            rollout=pool.for_phase(phase),
            log_progress=log_progress,
        )
        result = await evaluator(
            0,
            LocalTrainResult(step=0, metrics={}),
            config,
        )
    finally:
        await pool.close()
    return Path(result.artifacts["episode_outcomes_json"])


async def run(args: argparse.Namespace) -> None:
    base = EmbodiedExperimentConfig.from_yaml(args.config)
    if base.policy.type not in _SUPPORTED_FLOW_POLICIES:
        supported = ", ".join(sorted(_SUPPORTED_FLOW_POLICIES))
        raise ValueError(
            f"sampler calibration requires a supported flow policy ({supported})"
        )
    if base.algorithm.flow_sde is None:
        raise ValueError("sampler calibration requires algorithm.flow_sde")
    if args.episodes is not None and args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if args.policy_seed_repetitions < 1:
        raise ValueError("--policy-seed-repetitions must be positive")
    if args.noise_levels and any(level <= 0.0 for level in args.noise_levels):
        raise ValueError("--noise-level values must be positive")
    if args.denoise_steps and any(steps < 1 for steps in args.denoise_steps):
        raise ValueError("--denoise-steps values must be positive")
    if args.max_absolute_success_rate_gap is not None and not (
        0.0 <= args.max_absolute_success_rate_gap <= 1.0
    ):
        raise ValueError("--max-absolute-success-rate-gap must be in [0, 1]")

    profile = runtime_profile_for_policy(base.policy.type)
    require_compatible_runtime(profile=profile)
    prepare_libero_runtime_paths()
    validate_libero_runtime_imports()
    validate_libero_task_assets(LiberoSettings.from_config(base))

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    noise_levels = args.noise_levels or [base.algorithm.flow_sde.noise_level]
    denoise_steps_values = args.denoise_steps or [
        base.algorithm.flow_sde.num_denoise_steps
    ]
    schedules = []
    for denoise_steps in denoise_steps_values:
        schedule_dir = output_dir / f"denoise-{denoise_steps}"
        ode_config = _diagnostic_config(
            base,
            output_dir=schedule_dir / "native-ode",
            noise_level=None,
            denoise_steps=denoise_steps,
            episodes=args.episodes,
            policy_seed_repetitions=args.policy_seed_repetitions,
        )
        ode_report = await _evaluate_sampler(
            ode_config,
            phase="eval",
            checkpoint_path=args.checkpoint_path,
        )
        comparisons = []
        for noise_level in noise_levels:
            label = str(float(noise_level)).replace(".", "p")
            sde_config = _diagnostic_config(
                base,
                output_dir=schedule_dir / f"flow-sde-noise-{label}",
                noise_level=noise_level,
                denoise_steps=denoise_steps,
                episodes=args.episodes,
                policy_seed_repetitions=args.policy_seed_repetitions,
            )
            sde_report = await _evaluate_sampler(
                sde_config,
                phase="train",
                checkpoint_path=args.checkpoint_path,
            )
            paired = compare_paired_evaluation_reports(ode_report, sde_report)
            comparisons.append(
                {
                    "noise_level": float(noise_level),
                    "sde_outcomes": str(sde_report),
                    "paired_vs_ode": paired,
                    "gap_gate_passed": _sampler_gap_passes(
                        paired,
                        maximum_absolute_gap=args.max_absolute_success_rate_gap,
                    ),
                }
            )
        schedules.append(
            {
                "denoise_steps": denoise_steps,
                "ode_outcomes": str(ode_report),
                "comparisons": comparisons,
            }
        )

    payload = {
        "schema_version": 1,
        "config": str(args.config.expanduser().resolve()),
        "policy_type": base.policy.type,
        "policy_path": base.policy.path,
        "policy_revision": base.policy.revision,
        "checkpoint_path": (
            str(args.checkpoint_path.expanduser().resolve())
            if args.checkpoint_path is not None
            else None
        ),
        "episodes": ode_config.evaluation.episodes,
        "policy_seed_repetitions": args.policy_seed_repetitions,
        "max_absolute_success_rate_gap": args.max_absolute_success_rate_gap,
        "model_horizon": base.policy.load_kwargs.get("model_chunk_size"),
        "execution_horizon": base.policy.load_kwargs.get("execution_horizon"),
        "schedules": schedules,
    }
    payload["passed"] = all(
        comparison["gap_gate_passed"]
        for schedule in schedules
        for comparison in schedule["comparisons"]
    )
    summary_path = output_dir / "sampler-calibration.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
