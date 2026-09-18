"""Auditable ownership for every field in a resolved experiment config."""

from __future__ import annotations

from typing import Any

import pydantic

_SECTION_OWNERS = {
    "experiment": "runner_and_observability",
    "policy": "policy_factory_and_native_adapter",
    "environment": "rollout_components",
    "reward": "lerobot_rollout_adapter",
    "algorithm": "policy_gradient_backend",
    "rollout": "experiment_and_rollout_runtime",
    "training": "training_schedule_and_backend",
    "evaluation": "fixed_scenario_evaluator",
    "observability": "wandb_weave_observability",
    "storage": "backend_checkpoint_and_artifacts",
    "runtime": "runner_and_process_runtime",
}

_FIELD_OVERRIDES: dict[str, tuple[str, str, str]] = {
    "schema_version": (
        "config_parser",
        "assertion",
        "Selects the exact accepted experiment schema.",
    ),
    "training.optimizer.type": (
        "policy_gradient_backend",
        "assertion",
        "The built-in backend currently accepts AdamW only.",
    ),
    "training.microbatch_size": (
        "policy_gradient_backend",
        "executed",
        "Canonical gradient-enabled action-token scorer microbatch.",
    ),
    "algorithm.train_logprob_microbatch_size": (
        "experiment_preflight",
        "assertion",
        "Schema-v1 compatibility value; must equal training.microbatch_size.",
    ),
    "algorithm.flow_sde": (
        "continuous_flow_backend",
        "executed",
        "Defines the stochastic sampler and teacher-forced rescore probability model.",
    ),
    "algorithm.pad_fixed_horizon_examples": (
        "experiment_preflight",
        "assertion",
        "Must match whether the selected schedule executes fixed-horizon rows.",
    ),
    "rollout.action_payload.require_old_logprobs": (
        "experiment_preflight",
        "assertion",
        "Built-in action-token policy-gradient objectives require old logprobs.",
    ),
    "evaluation.runtime": (
        "fixed_scenario_evaluator",
        "assertion",
        "The built-in evaluator executes the native LeRobot runtime only.",
    ),
    "evaluation.evaluate_before_training": (
        "embodied_runner",
        "behavior",
        "Evaluates the actual initialized SFT policy before the first rollout.",
    ),
    "evaluation.evaluate_after_first_update": (
        "embodied_runner",
        "behavior",
        "Evaluates update 1 independently of the periodic evaluation cadence.",
    ),
    "evaluation.pre_training_success_gate": (
        "embodied_runner",
        "behavior",
        "Refuses optimizer work when Step-0 success has no useful RL signal or headroom.",
    ),
    "evaluation.split": (
        "evaluation_report_and_observability",
        "metadata",
        "Labels evidence; fixed_scenarios and seeds select the evaluated data.",
    ),
    "evaluation.data_role": (
        "fixed_scenario_evaluator",
        "assertion",
        "Separates diagnostic/development evaluation from a sealed final test.",
    ),
    "evaluation.checkpoint_selection": (
        "wandb_artifact_publisher",
        "telemetry",
        "Controls W&B checkpoint aliases, not local checkpoint retention.",
    ),
    "observability.wandb.log_evaluation_artifacts": (
        "wandb_artifact_publisher",
        "telemetry",
        "Publishes raw outcomes and their evidence identity as one artifact.",
    ),
    "observability.wandb.log_input_model_artifact": (
        "wandb_artifact_publisher",
        "telemetry",
        "Backs up a completed local warm-start checkpoint and declares it as run input.",
    ),
    "observability.wandb.input_model_artifact_ref": (
        "wandb_artifact_publisher",
        "telemetry",
        "Uses a pre-registered immutable model artifact without uploading it on GPU time.",
    ),
    "observability.delivery_failure_policy": (
        "wandb_weave_observability",
        "behavior",
        "Chooses fail-closed claim logging or best-effort optional telemetry.",
    ),
    "observability.wandb.run_id": (
        "wandb_observer",
        "telemetry",
        "Identifies the interrupted writer recovered with connection=resume.",
    ),
    "observability.wandb.resume": (
        "wandb_observer",
        "telemetry",
        "Controls recovery of an interrupted W&B writer; not distributed attachment.",
    ),
    "observability.wandb.connection": (
        "wandb_observer",
        "telemetry",
        "Selects a regular primary writer, restricted shared telemetry, or crash recovery.",
    ),
    "observability.wandb.writer_label": (
        "wandb_observer",
        "telemetry",
        "Labels console and system metrics when restricted shared mode is used.",
    ),
    "observability.wandb.console_multipart": (
        "wandb_observer",
        "telemetry",
        "Stores console output as immutable timestamped chunks.",
    ),
    "observability.wandb.native_update_steps": (
        "wandb_observer",
        "telemetry",
        "Commits one native W&B row per completed update, including train and evaluation media.",
    ),
    "observability.wandb.console_chunk_max_bytes": (
        "wandb_observer",
        "telemetry",
        "Bounds each immutable W&B console chunk by size.",
    ),
    "observability.wandb.console_chunk_max_seconds": (
        "wandb_observer",
        "telemetry",
        "Flushes immutable console chunks during long-running jobs.",
    ),
    "environment.observation_processor": (
        "rollout_components",
        "plugin",
        "Consumed by a user components factory; built-in OpenVLA requires an empty mapping.",
    ),
    "environment.action_processor": (
        "rollout_components",
        "plugin",
        "Consumed by a user components factory; built-in OpenVLA requires an empty mapping.",
    ),
    "environment.kwargs": (
        "rollout_components",
        "plugin",
        "Validated and consumed by the selected environment/components factory.",
    ),
    "environment.reset": (
        "rollout_components",
        "plugin",
        "Applied by the selected environment adapter during replayable reset.",
    ),
    "reward.kwargs": (
        "rollout_components",
        "plugin",
        "Reserved for a selected custom reward/components implementation.",
    ),
    "policy.load_kwargs": (
        "policy_factory_and_native_adapter",
        "plugin",
        "Validated by the selected policy-family loader.",
    ),
    "runtime.rollout_execution.actor_kwargs": (
        "rollout_actor_factory",
        "plugin",
        "Passed to the selected rollout actor factory.",
    ),
}


def config_consumption_report(config: pydantic.BaseModel) -> dict[str, Any]:
    """Return one ownership row for every resolved public config field."""

    fields = []
    for path, value in _iter_resolved_fields(config):
        owner, status, note = _classify(path, config=config)
        fields.append(
            {
                "path": path,
                "owner": owner,
                "status": status,
                "note": note,
                "value_type": type(value).__name__,
            }
        )
    counts: dict[str, int] = {}
    for row in fields:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    return {
        "field_count": len(fields),
        "status_counts": dict(sorted(counts.items())),
        "unowned_fields": [row["path"] for row in fields if row["status"] == "unowned"],
        "fields": fields,
    }


def _iter_resolved_fields(
    model: pydantic.BaseModel,
    *,
    prefix: str = "",
) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    for field_name in type(model).model_fields:
        value = getattr(model, field_name)
        path = f"{prefix}.{field_name}" if prefix else field_name
        if isinstance(value, pydantic.BaseModel):
            rows.extend(_iter_resolved_fields(value, prefix=path))
        else:
            rows.append((path, value))
    return rows


def _classify(
    path: str,
    *,
    config: pydantic.BaseModel,
) -> tuple[str, str, str]:
    action_kind = _resolved_action_kind(config)
    token_sampling_fields = (
        "policy.rollout_generation",
        "policy.train_generation",
        "policy.evaluation_generation",
        "rollout.temperature",
        "rollout.deterministic",
        "evaluation.temperature",
        "evaluation.deterministic",
    )
    if action_kind == "continuous" and any(
        path == prefix or path.startswith(prefix + ".")
        for prefix in token_sampling_fields
    ):
        return (
            "none",
            "not_applicable",
            "Token sampling control; continuous policies use their native "
            "flow/diffusion probability contract instead.",
        )
    override = _FIELD_OVERRIDES.get(path)
    if override is not None:
        return override
    section = path.split(".", 1)[0]
    owner = _SECTION_OWNERS.get(section)
    if owner is None:
        return ("none", "unowned", "No runtime owner is registered.")
    return (
        owner,
        "owned",
        "Assigned to the section runtime; field-level behavior remains covered "
        "by that component's contract tests.",
    )


def _resolved_action_kind(config: pydantic.BaseModel) -> str | None:
    rollout = getattr(config, "rollout", None)
    payload = getattr(rollout, "action_payload", None)
    value = getattr(payload, "kind", None)
    return value if isinstance(value, str) else None
