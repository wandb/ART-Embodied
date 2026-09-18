"""Evaluate an ordered policy series in one coherent W&B run."""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
from pathlib import Path
from typing import Literal

import pydantic

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
from examples.embodied.libero.components import (
    LiberoSettings,
    build_evaluation_scenarios,
    prepare_libero_runtime_paths,
    validate_libero_runtime_imports,
    validate_libero_task_assets,
)


class _StrictModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True, strict=True)


class PolicySeriesCandidate(_StrictModel):
    """One policy source and its logical training step."""

    step: int = pydantic.Field(ge=0)
    name: str = pydantic.Field(min_length=1)
    checkpoint_role: Literal["sft_baseline", "sft_candidate", "candidate"]
    path: str = pydantic.Field(min_length=1)
    revision: str | None = None
    local: bool


class PolicySeriesManifest(_StrictModel):
    """Single source of truth for a checkpoint evaluation campaign."""

    schema_version: Literal[1]
    workspace_root: str = pydantic.Field(min_length=1)
    base_config: str = pydantic.Field(min_length=1)
    candidates: tuple[PolicySeriesCandidate, ...]

    @pydantic.field_validator("candidates", mode="before")
    @classmethod
    def _freeze_candidates(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @pydantic.model_validator(mode="after")
    def _validate_candidates(self) -> "PolicySeriesManifest":
        if not self.candidates:
            raise ValueError("policy series requires at least one candidate")
        steps = [candidate.step for candidate in self.candidates]
        if steps != sorted(steps):
            raise ValueError("policy series candidates must use increasing steps")
        if len(steps) != len(set(steps)):
            raise ValueError("policy series candidate steps must be unique")
        names = [candidate.name for candidate in self.candidates]
        if len(names) != len(set(names)):
            raise ValueError("policy series candidate names must be unique")
        if self.candidates[0].step != 0:
            raise ValueError("policy series must begin with the Step 0 baseline")
        if self.candidates[0].checkpoint_role != "sft_baseline":
            raise ValueError("policy series first candidate must be sft_baseline")
        if any(
            candidate.checkpoint_role == "sft_baseline"
            for candidate in self.candidates[1:]
        ):
            raise ValueError("policy series may declare only one sft_baseline")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "PolicySeriesManifest":
        import yaml

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign",
        type=Path,
        required=True,
        help="Policy-series YAML containing the config and ordered checkpoints.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate the campaign and assets without loading policies.",
    )
    return parser.parse_args()


def resolve_campaign(
    campaign_path: Path,
) -> tuple[PolicySeriesManifest, Path, tuple[PolicySeriesCandidate, ...]]:
    """Resolve local paths while preserving Hub identifiers verbatim."""

    campaign_path = campaign_path.expanduser().resolve()
    manifest = PolicySeriesManifest.from_yaml(campaign_path)
    workspace = (campaign_path.parent / manifest.workspace_root).resolve()
    config_path = (campaign_path.parent / manifest.base_config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Policy-series base config is missing: {config_path}")

    resolved = []
    for candidate in manifest.candidates:
        if candidate.local:
            source = Path(candidate.path).expanduser()
            if not source.is_absolute():
                source = workspace / source
            source = source.resolve()
            if not source.is_dir():
                raise FileNotFoundError(
                    f"Policy-series candidate is missing: {source}"
                )
            resolved.append(candidate.model_copy(update={"path": str(source)}))
        else:
            resolved.append(candidate)
    return manifest, config_path, tuple(resolved)


async def run(campaign_path: Path, *, preflight: bool = False) -> None:
    manifest, config_path, candidates = resolve_campaign(campaign_path)
    config = EmbodiedExperimentConfig.from_yaml(config_path)
    if config.policy.lora.enabled:
        raise ValueError(
            "Policy-series evaluation requires policy.lora.enabled=false. "
            "A full SFT checkpoint synchronized through a LoRA-only snapshot "
            "would silently discard its changed base weights."
        )
    if config.policy.force_trainable_float32:
        raise ValueError(
            "Policy-series evaluation requires "
            "policy.force_trainable_float32=false. Evaluation must preserve the "
            "serialized checkpoint dtypes instead of changing policy behavior "
            "while selecting a nominal trainable surface."
        )

    profile = runtime_profile_for_policy(config.policy.type)
    require_compatible_runtime(profile=profile)
    prepare_libero_runtime_paths()
    settings = LiberoSettings.from_config(config)
    assets = validate_libero_task_assets(settings)
    evaluation_scenarios = build_evaluation_scenarios(config)
    if preflight:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "campaign": str(campaign_path.resolve()),
                    "config": str(config_path),
                    "assets": assets,
                    "evaluation_scenarios": len(evaluation_scenarios),
                    "candidates": [
                        candidate.model_dump(mode="json") for candidate in candidates
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    validate_runtime_device_availability(config, include_training_devices=False)
    observer = WandbWeaveObserver.start(config)
    run_error: BaseException | None = None
    measured_baseline_path: Path | None = None
    try:
        validate_libero_runtime_imports()
        for candidate in candidates:
            await observer.log_progress(
                ExperimentProgress(
                    update=candidate.step,
                    phase="initialization",
                    status="started",
                    message=f"Loading policy series candidate {candidate.name}",
                ),
                config,
            )
            candidate_policy_config = config.model_copy(
                update={
                    "policy": config.policy.model_copy(
                        update={
                            "path": candidate.path,
                            "revision": candidate.revision,
                        }
                    )
                }
            )
            policy = make_policy(candidate_policy_config)
            try:
                await observer.log_progress(
                    ExperimentProgress(
                        update=candidate.step,
                        phase="initialization",
                        status="completed",
                        message=f"Loaded policy series candidate {candidate.name}",
                    ),
                    config,
                )
                result = await run_lerobot_evaluation(
                    config=config,
                    policy=policy,
                    evaluation_scenarios=evaluation_scenarios,
                    step=candidate.step,
                    checkpoint_path=(candidate.path if candidate.local else None),
                    observer=observer,
                    checkpoint_role=candidate.checkpoint_role,
                    use_native_wandb_step=True,
                    measured_baseline_path=measured_baseline_path,
                )
                if measured_baseline_path is None:
                    outcome_path = result.evaluation.artifacts.get(
                        "episode_outcomes_json"
                    )
                    if outcome_path is None:
                        raise RuntimeError(
                            "Baseline evaluation did not return an outcome report"
                        )
                    measured_baseline_path = Path(outcome_path).resolve()
            finally:
                to = getattr(policy, "to", None)
                if callable(to):
                    to("cpu")
                del policy
                gc.collect()
                try:
                    import torch

                    torch.cuda.empty_cache()
                except ImportError:  # pragma: no cover - preflight-only installs.
                    pass
    except BaseException as exc:
        run_error = exc
        raise
    finally:
        observer.close(exit_code=1 if run_error is not None else 0)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args.campaign, preflight=args.preflight))
