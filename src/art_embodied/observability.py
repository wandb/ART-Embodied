"""W&B Models and Weave observability for embodied experiments."""

from __future__ import annotations

from importlib import import_module
import json
import math
import os
from pathlib import Path
import socket
from statistics import fmean
import sys
import time
from typing import Any, Callable, Sequence

from .checkpointing import CHECKPOINT_COMPLETE_MARKER
from .config import EmbodiedExperimentConfig
from .experiment import EvaluationResult, ExperimentProgress
from .media import local_video_media_refs, wandb_video_payload
from .trajectories import EmbodiedTrajectory, EmbodiedTrajectoryGroup
from .types import TrainResult
from .utils import make_json_safe, write_json_atomic

_TRAINING_MONITOR_METRIC_SUFFIXES = frozenset(
    {
        "advantage_std",
        "approx_kl_abs_mean",
        "alignment_approx_kl_abs_mean",
        "alignment_old_new_logprobs_aligned",
        "alignment_ratio_mean",
        "active_previous_abs_delta_max",
        "active_previous_abs_delta_mean",
        "active_previous_ratio_mean",
        "approximate_kl",
        "approximate_kl_per_primitive",
        "clip_fraction",
        "distributed_elapsed_seconds",
        "distributed_adapter_refreshes",
        "effective_group_fraction",
        "grad_norm",
        "grad_norm_before_clip",
        "group_relative_signal_available",
        "groups_filtered",
        "groups_kept",
        "groups_total",
        "groups_with_reward_variance",
        "groups_with_zero_reward_variance",
        "loss",
        "lora_a_parameter_delta_norm",
        "lora_b_parameter_delta_norm",
        "objective_direction_advantage_logprob_delta_product_mean",
        "objective_direction_sign_agreement_fraction",
        "old_logprob_rescore_previous_abs_delta_max",
        "old_logprob_rescore_previous_abs_delta_mean",
        "old_new_logprobs_aligned",
        "optimizer_step_completed",
        "parameter_delta_norm",
        "payload_cache_hit_fraction",
        "policy_parameters_updated",
        "pre_update_alignment_abs_delta_max",
        "pre_update_alignment_abs_delta_mean",
        "pre_update_alignment_ratio_mean",
        "pre_update_active_alignment_abs_delta_max",
        "pre_update_active_alignment_abs_delta_mean",
        "pre_update_active_alignment_ratio_mean",
        "optimization_old_policy_abs_delta_mean",
        "previous_abs_delta_max",
        "previous_abs_delta_mean",
        "ratio_max",
        "ratio_mean",
        "ratio_min",
        "train_total_seconds",
        "alignment_guard_seconds",
        "alignment_guard_seconds_max",
        "train_logprob_forward_seconds",
        "train_logprob_forward_seconds_max",
        "train_loss_backward_seconds",
        "train_loss_backward_seconds_max",
        "gradient_apply_seconds",
        "distributed_workers",
        "distributed_worker_seconds_max",
        "distributed_worker_seconds_mean",
        "worker_seconds_max",
        "worker_gradient_effective_aligned_workers",
        "worker_gradient_noise_to_signal_ratio",
        "worker_gradient_pairwise_cosine_mean",
        "worker_gradient_resultant_ratio",
        "worker_gradient_signal_to_rms_ratio",
        "zero_advantage_fraction",
    }
)

# Generated W&B workspaces stop being useful when backend-specific diagnostics
# create hundreds of distinct columns. Keep the chart-ready history bounded;
# complete TrainResult metrics remain in checkpoint artifact metadata and Weave.
_MAX_WANDB_TRAINING_HISTORY_METRICS = 96

_ROLLOUT_MONITOR_METRIC_KEYS = frozenset(
    {
        "completed",
        "completion_rate",
        "elapsed_seconds",
        "groups_completed",
        "groups_per_second",
        "reward_mean",
        "rollout/elapsed_seconds",
        "rollout/groups_per_second",
        "rollout/trajectories_per_second",
        "success_count",
        "success_denominator",
        "success_rate",
        "total",
        "trajectories_completed",
        "trajectories_per_second",
    }
)

_PRIMARY_EVALUATION_METRICS = {
    "success_rate",
    "episodes",
    "task_macro_success_rate",
    "paired/baseline_success_rate",
    "paired/candidate_success_rate",
    "paired/success_rate_lift",
    "paired/success_rate_lift_ci95_low",
    "paired/success_rate_lift_ci95_high",
    "paired/baseline_task_macro_success_rate",
    "paired/candidate_task_macro_success_rate",
    "paired/task_macro_success_rate_lift",
    "paired/task_macro_success_rate_lift_ci95_low",
    "paired/task_macro_success_rate_lift_ci95_high",
    "paired/mcnemar_exact_p_value",
    "paired/improved_pairs",
    "paired/regressed_pairs",
    "paired/unchanged_success_pairs",
    "paired/unchanged_failure_pairs",
}

# W&B groups panels by the first path component. Keep ``train/*`` limited to
# the two curves a human needs for a first-pass learning-health decision;
# backend internals belong on separate top-level surfaces.
_PRIMARY_TRAIN_METRIC_KEYS = frozenset({"train/reward_mean", "train/success_rate"})
_WANDB_HISTORY_METRIC_NAMESPACES = (
    "train/*",
    "validation/*",
    "test/*",
    "optimization/*",
    "performance/*",
    "signal/*",
    "train_details/*",
    "eval_details/*",
    "train_tasks/*",
    "eval_tasks/*",
    "telemetry/*",
)
_PERFORMANCE_METRIC_MARKERS = (
    "_seconds",
    "elapsed_",
    "throughput",
    "worker_",
    "workers",
    "cache_",
    "adapter_refresh",
    "policy_load",
)
_OPTIMIZATION_METRIC_MARKERS = (
    "loss",
    "kl",
    "ratio",
    "clip",
    "grad",
    "optimizer",
    "parameter",
    "logprob",
    "trust_region",
    "checkpoint",
    "probe",
)
_SIGNAL_METRIC_MARKERS = (
    "advantage",
    "reward",
    "group",
    "example",
    "trajectory",
    "action_span",
    "action_token",
    "tokens_",
    "zero_variance",
    "fixed_horizon_rows",
)


def _wandb_artifact_metadata(value: Any) -> Any:
    """Return JSON-safe artifact metadata without non-finite numbers."""

    if isinstance(value, bool | int | str) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _wandb_artifact_metadata(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_wandb_artifact_metadata(item) for item in value]
    return make_json_safe(value)


def _progress_monitor_metrics(
    event: ExperimentProgress,
) -> dict[str, float | int]:
    """Select the small live-health surface published through W&B summary.

    Primary rollout outcomes are committed once under ``train/*``; backend
    details use separate top-level namespaces. The mutable ``monitor/*``
    summary answers whether the current update is healthy without mirroring
    every token percentile, parameter counter, or scenario.
    """

    metrics = event.metrics or {}
    if event.phase == "rollout":
        return {
            key: value
            for key, value in metrics.items()
            if key in _ROLLOUT_MONITOR_METRIC_KEYS
        }
    if event.phase == "evaluation":
        return {
            key: value
            for key, value in metrics.items()
            if not _is_evaluation_detail_metric(key)
        }
    if event.phase != "training":
        return {}
    return {
        key: value
        for key, value in metrics.items()
        if _is_wandb_training_history_metric(key)
    }


def _is_wandb_training_history_metric(key: str) -> bool:
    """Return whether one backend metric belongs on generated W&B charts."""

    normalized = key.removeprefix("train/").removeprefix("eval/")
    suffix = normalized.rsplit("/", 1)[-1]
    for boundary in ("_first_subupdate", "_last_subupdate"):
        if suffix.endswith(boundary):
            suffix = suffix.removesuffix(boundary)
            break
    return (
        normalized.startswith(("rollout/", "training/"))
        or "_schedule/" in normalized
        or suffix in _TRAINING_MONITOR_METRIC_SUFFIXES
    )


def _wandb_training_history_payload(
    metrics: dict[str, Any],
) -> dict[str, int | float | bool]:
    """Return a stable, bounded history surface while retaining metric names."""

    payload = {
        _wandb_training_metric_key(key): value
        for key, value in metrics.items()
        if isinstance(value, int | float | bool)
        and _is_wandb_training_history_metric(key)
    }
    if len(payload) > _MAX_WANDB_TRAINING_HISTORY_METRICS:
        raise RuntimeError(
            "W&B training history metric budget exceeded: "
            f"{len(payload)} > {_MAX_WANDB_TRAINING_HISTORY_METRICS}. "
            "Keep detailed backend diagnostics in artifacts or Weave instead."
        )
    return payload


def _bounded_pending_wandb_payload(
    payload: dict[str, str | int | float | bool | None],
) -> dict[str, str | int | float | bool | None]:
    """Sanitize pending rows created before history metric budgeting existed."""

    bounded: dict[str, str | int | float | bool | None] = {}
    for key, value in payload.items():
        if key == "experiment/update" or key.startswith(
            ("telemetry/", "validation/", "test/", "eval_details/")
        ):
            bounded[key] = value
            continue
        if key.startswith(("optimization/", "signal/", "performance/")):
            raw_key = key.split("/", 1)[1]
            if _is_wandb_training_history_metric(raw_key):
                bounded[key] = value
    return bounded


def _import_observability_dependency(module_name: str, *, integration: str) -> Any:
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        raise RuntimeError(
            f"{integration} is enabled but {module_name} is not installed. "
            "Install art-embodied[observability], or disable the integration "
            "in observability settings."
        ) from exc


def _define_wandb_history_schema(wandb_run: Any) -> None:
    """Declare the stable chart schema instead of relying on UI inference."""

    define_metric = getattr(wandb_run, "define_metric", None)
    if not callable(define_metric):
        raise RuntimeError("W&B logging requires Run.define_metric")
    update_metric = define_metric("experiment/update", hidden=True)
    for namespace in _WANDB_HISTORY_METRIC_NAMESPACES:
        define_metric(namespace, step_metric=update_metric)


class WandbWeaveObserver:
    """Experiment logger implementing the ``EmbodiedExperiment`` step hook.

    Only the coordinator should own this object in distributed runs.  Worker
    processes return metrics and media references to the coordinator instead
    of creating competing W&B runs or disconnected Weave traces.
    """

    def __init__(
        self,
        config: EmbodiedExperimentConfig,
        *,
        wandb_run: Any | None = None,
        wandb_module: Any | None = None,
        weave_client: Any | None = None,
        weave_module: Any | None = None,
        enforce_initial_history_commit: bool = False,
    ) -> None:
        self.config = config
        self._wandb_module = wandb_module
        self._weave_module = weave_module
        self.wandb_run = wandb_run
        self.weave_client = weave_client
        self._enforce_initial_history_commit = enforce_initial_history_commit
        self._native_update_history = None
        if config.observability.wandb.native_update_steps and wandb_run is not None:
            from .wandb_native_steps import NativeUpdateHistory

            self._native_update_history = NativeUpdateHistory(wandb_run)
        self._best_logged_evaluation_success: float | None = None
        self._best_summary_evaluation_success: dict[str, float] = {}
        self._progress_event_index = 0
        self._resume_config_recorded = False
        self._update_started_at: dict[int, float] = {}
        self._weave_update_calls: dict[int, Any] = {}
        self._pending_wandb_policy_version: int | None = None
        self._pending_wandb_payload: dict[str, str | int | float | bool | None] = {}
        self._pending_wandb_path = (
            Path(config.storage.output_dir) / "wandb" / "pending-history.json"
        )
        self._telemetry_failure_count = 0
        self._telemetry_failure_counts: dict[str, int] = {"wandb": 0, "weave": 0}
        if config.observability.wandb.connection == "resume":
            if (
                self._native_update_history is not None
                and self._pending_wandb_path.exists()
            ):
                raise RuntimeError(
                    "Unresolved W&B transaction exists; recover it before optimizer resume"
                )
            self._restore_pending_wandb_history()
            summary = getattr(wandb_run, "summary", None)
            if summary is not None:
                for namespace in ("validation", "test"):
                    value = summary.get(f"{namespace}/best_success_rate")
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        self._best_summary_evaluation_success[namespace] = float(value)
                namespace = (
                    "test"
                    if config.evaluation.data_role == "sealed_test"
                    else "validation"
                )
                self._best_logged_evaluation_success = (
                    self._best_summary_evaluation_success.get(namespace)
                )

    @classmethod
    def start(cls, config: EmbodiedExperimentConfig) -> "WandbWeaveObserver":
        # wandb.init() can automatically initialize Weave. Apply the cache
        # contract before importing either SDK so that auto-init cannot touch
        # a shared, process-global /tmp cache owned by another cluster user.
        if config.observability.weave.enabled:
            _configure_weave_server_cache(config)
        wandb_module = (
            _import_observability_dependency("wandb", integration="W&B logging")
            if config.observability.wandb.enabled
            else None
        )
        weave_module = (
            _import_observability_dependency("weave", integration="Weave tracing")
            if config.observability.weave.enabled
            else None
        )

        wandb_run = None
        if wandb_module is not None:
            wandb_dir = Path(config.storage.output_dir) / "wandb"
            wandb_dir.mkdir(parents=True, exist_ok=True)
            run_id = config.observability.wandb.run_id or os.environ.get("WANDB_RUN_ID")
            resume = config.observability.wandb.resume or os.environ.get("WANDB_RESUME")
            connection = config.observability.wandb.connection
            writer_label = (
                config.observability.wandb.writer_label
                or _default_wandb_writer_label(connection)
            )
            settings_type = getattr(wandb_module, "Settings", None)
            if settings_type is None:
                raise RuntimeError(
                    "W&B logging requires wandb.Settings for connection and "
                    "multipart-console safety"
                )
            settings_kwargs: dict[str, Any] = {
                "console_multipart": config.observability.wandb.console_multipart,
                "console_chunk_max_bytes": (
                    config.observability.wandb.console_chunk_max_bytes
                ),
                "console_chunk_max_seconds": (
                    config.observability.wandb.console_chunk_max_seconds
                ),
                "x_disable_stats": (not config.observability.wandb.log_system_metrics),
            }
            if config.observability.wandb.mode == "online" and connection.startswith(
                "shared_"
            ):
                is_primary = connection == "shared_primary"
                settings_kwargs.update(
                    mode="shared",
                    x_label=writer_label,
                    x_primary=is_primary,
                    x_update_finish_state=is_primary,
                )
            else:
                settings_kwargs["mode"] = config.observability.wandb.mode
            init_kwargs: dict[str, Any] = dict(
                entity=config.observability.wandb.entity,
                project=config.observability.wandb.project,
                dir=str(wandb_dir),
                save_code=config.observability.wandb.save_code,
                settings=settings_type(**settings_kwargs),
            )
            if run_id is not None:
                init_kwargs["id"] = run_id
                if connection == "resume":
                    init_kwargs["resume"] = resume
            else:
                init_kwargs.update(
                    name=config.experiment.run,
                    group=config.observability.wandb.group,
                    job_type=config.observability.wandb.job_type,
                    tags=config.experiment.tags,
                    config=config.model_dump(mode="json"),
                )
            wandb_run = wandb_module.init(**init_kwargs)
            if wandb_run is None:
                raise RuntimeError("wandb.init returned no Run")
            _define_wandb_history_schema(wandb_run)
            if config.observability.wandb.log_input_model_artifact:
                _declare_input_model_artifact(
                    config=config,
                    wandb_module=wandb_module,
                    wandb_run=wandb_run,
                )

        weave_client = None
        if weave_module is not None:
            weave_client = weave_module.init(config.observability.weave.project)

        return cls(
            config,
            wandb_run=wandb_run,
            wandb_module=wandb_module,
            weave_client=weave_client,
            weave_module=weave_module,
            enforce_initial_history_commit=(
                wandb_run is not None
                and config.observability.wandb.mode == "online"
                and config.observability.wandb.connection == "primary"
            ),
        )

    async def log_progress(
        self,
        event: ExperimentProgress,
        config: EmbodiedExperimentConfig,
    ) -> None:
        """Publish live phase and rollout progress before an update finishes."""

        if config.fingerprint != self.config.fingerprint:
            raise ValueError("Observer config does not match experiment config")
        if event.phase == "update" and event.status == "started":
            if (
                self.config.observability.wandb.connection == "resume"
                and not self._resume_config_recorded
            ):
                self._deliver(
                    "wandb",
                    "record_resume_config",
                    event.update,
                    lambda: self._record_resume_config(event.update),
                )
            self._update_started_at[event.update] = time.monotonic()
            self._deliver(
                "weave",
                "start_update",
                event.update,
                lambda: self._start_weave_update(event.update),
            )
        should_report_progress = self._should_log_progress(event)
        if should_report_progress:
            self._log_terminal_progress(event)
        if self.wandb_run is not None and should_report_progress:
            self._update_wandb_progress_summary(event)
        if event.phase == "update" and event.status == "failed":
            self._deliver(
                "weave",
                "fail_update",
                event.update,
                lambda: self._fail_weave_update(event.update, event.message),
            )
        if event.phase == "update" and event.status in {"completed", "failed"}:
            self._update_started_at.pop(event.update, None)

    def _record_resume_config(self, update: int) -> None:
        """Preserve configuration history before updating a recovered writer.

        The first update-start event follows backend checkpoint restoration.
        Updating at observer startup would mislabel a run if restoration failed.
        """

        current = self.config.model_dump(mode="json")
        previous = dict(self.wandb_run.config)
        record = {
            "run_id": self.wandb_run.id,
            "restored_policy_version": update - 1,
            "next_optimizer_update": update,
            "timestamp_ns": time.time_ns(),
            "previous": previous,
            "current": current,
        }
        path = (
            Path(self.config.storage.output_dir)
            / "wandb"
            / "configuration-history"
            / f"resume-{update - 1}-{record['timestamp_ns']}.json"
        )
        write_json_atomic(path, record, indent=2, sort_keys=True)
        artifact = self._wandb_module.Artifact(
            name=f"run-{self.wandb_run.id}-configuration",
            type="experiment-config",
            metadata={
                "restored_policy_version": update - 1,
                "config_fingerprint": self.config.fingerprint,
                "resume_contract_fingerprint": self.config.resume_contract_fingerprint,
            },
        )
        artifact.add_file(str(path), name="configuration.json")
        self.wandb_run.log_artifact(
            artifact, aliases=["latest", f"resume-update-{update - 1}"]
        )
        self.wandb_run.config.update(current, allow_val_change=True)
        self._resume_config_recorded = True

    async def log_step(
        self,
        step: int,
        groups: Sequence[EmbodiedTrajectoryGroup],
        train_result: TrainResult,
        evaluation: EvaluationResult | None,
        config: EmbodiedExperimentConfig,
    ) -> None:
        if config.fingerprint != self.config.fingerprint:
            raise ValueError("Observer config does not match experiment config")
        if self.wandb_run is not None:
            self._log_wandb(step, groups, train_result, evaluation)
        if self.weave_client is not None:
            self._deliver(
                "weave",
                "log_training_update",
                step,
                lambda: self._log_weave(step, groups, train_result, evaluation),
            )

    async def log_rollout(
        self,
        policy_version: int,
        groups: Sequence[EmbodiedTrajectoryGroup],
        config: EmbodiedExperimentConfig,
    ) -> None:
        """Commit rollout behavior under the policy version that produced it."""

        if config.fingerprint != self.config.fingerprint:
            raise ValueError("Observer config does not match experiment config")
        evidence = self._persist_rollout_group_evidence(policy_version, groups)
        if self.wandb_run is not None:
            self._log_wandb_rollout(policy_version, groups, evidence=evidence)

    async def log_evaluation(
        self,
        step: int,
        evaluation: EvaluationResult,
        config: EmbodiedExperimentConfig,
    ) -> None:
        """Log a candidate evaluation not attached to a training update."""

        await self._log_standalone_evaluation(
            step,
            evaluation,
            config,
            checkpoint_role="candidate",
        )

    async def log_evaluation_checkpoint(
        self,
        step: int,
        evaluation: EvaluationResult,
        config: EmbodiedExperimentConfig,
        *,
        checkpoint_role: str,
        use_native_wandb_step: bool = False,
    ) -> None:
        """Log one member of an ordered checkpoint-evaluation campaign.

        ``use_native_wandb_step`` is reserved for a fresh, evaluation-only run
        whose checkpoints are visited in increasing order. It makes W&B's
        built-in Step axis equal the policy update without requiring users to
        configure a custom x-axis. Training observers must leave it disabled:
        their pre-update evaluation and optimizer metrics share one history row.
        """

        await self._log_standalone_evaluation(
            step,
            evaluation,
            config,
            checkpoint_role=checkpoint_role,
            native_wandb_step=use_native_wandb_step,
        )

    async def log_initial_evaluation(
        self,
        evaluation: EvaluationResult,
        config: EmbodiedExperimentConfig,
    ) -> None:
        """Commit the measured SFT baseline at experiment/update=0 immediately."""

        await self._log_standalone_evaluation(
            0,
            evaluation,
            config,
            checkpoint_role="sft_baseline",
            commit=True,
        )

    async def _log_standalone_evaluation(
        self,
        step: int,
        evaluation: EvaluationResult,
        config: EmbodiedExperimentConfig,
        *,
        checkpoint_role: str,
        commit: bool = True,
        native_wandb_step: bool = False,
    ) -> None:

        if config.fingerprint != self.config.fingerprint:
            raise ValueError("Observer config does not match evaluation config")
        if self.wandb_run is not None:
            payload: dict[str, Any] = {"experiment/update": step}
            self._add_wandb_evaluation(
                payload=payload,
                step=step,
                evaluation=evaluation,
                checkpoint_role=checkpoint_role,
            )
            self._persist_pending_wandb_history(step, payload)
            if native_wandb_step and not commit:
                raise ValueError(
                    "native W&B steps require a committed evaluation history row"
                )
            delivered = self._deliver(
                "wandb",
                "log_evaluation",
                step,
                lambda: (
                    self._native_update_history.commit(step, payload)
                    if self._native_update_history is not None
                    else self._log_native_evaluation_row(step, payload)
                    if native_wandb_step
                    else self.wandb_run.log(
                        payload,
                        commit=commit,
                    )
                ),
            )
            if not delivered and (
                self._native_update_history is not None or native_wandb_step
            ):
                raise RuntimeError("Native W&B evaluation row was not committed")
            if delivered:
                self._pending_wandb_policy_version = None if commit else step
                if commit:
                    self._clear_pending_wandb_history()
            if self.config.observability.wandb.log_evaluation_artifacts:
                self._deliver(
                    "wandb",
                    "log_evaluation_artifact",
                    step,
                    lambda: self._log_evaluation_artifact(step, evaluation),
                )
        if self.weave_client is not None:
            self._deliver(
                "weave",
                "log_evaluation",
                step,
                lambda: self._log_weave_evaluation_run(step, evaluation),
            )

    def _log_native_evaluation_row(self, step: int, payload: dict[str, Any]) -> None:
        run = self.wandb_run
        if type(step) is not int or step < 0 or run.step > step:
            raise ValueError("Standalone evaluation cannot overwrite a native W&B step")
        # W&B binds media using run.step before processing log(step=...). Advance
        # the pending row without committing an empty row before binding videos.
        if run.step < step:
            run.log({}, step=step, commit=False)
        if run.step != step:
            raise RuntimeError("W&B did not prepare the requested evaluation step")
        run.log(payload, step=step, commit=True)
        if run.step != step + 1:
            raise RuntimeError("W&B did not commit exactly one evaluation row")

    def close(self, *, exit_code: int = 0) -> None:
        if self._native_update_history is not None:
            self._native_update_history.discard_unfinished()
        for update in tuple(self._weave_update_calls):
            self._deliver(
                "weave",
                "close_incomplete_update",
                update,
                lambda update=update: self._fail_weave_update(
                    update,
                    "Observer closed before the training update completed",
                ),
            )
        if self.wandb_run is not None:
            if (
                self._pending_wandb_policy_version is not None
                and exit_code == 0
                and self._native_update_history is None
            ):
                pending_version = self._pending_wandb_policy_version
                delivered = self._deliver(
                    "wandb",
                    "commit_pending_history",
                    pending_version,
                    lambda: self.wandb_run.log(
                        {
                            **self._pending_wandb_payload,
                            "experiment/update": pending_version,
                        },
                        commit=True,
                    ),
                )
                if delivered:
                    self._pending_wandb_policy_version = None
                    self._clear_pending_wandb_history()
            # On an interrupted run, keep the durable pending payload local.
            # A resumed observer merges it into that policy version's rollout
            # row instead of consuming a separate native W&B history step.
            finish = getattr(self.wandb_run, "finish", None)
            if callable(finish):
                self._deliver(
                    "wandb",
                    "finish_run",
                    None,
                    lambda: finish(exit_code=exit_code),
                )

    def _deliver(
        self,
        integration: str,
        operation: str,
        step: int | None,
        callback: Callable[[], Any],
    ) -> bool:
        """Deliver telemetry according to the experiment's durability contract."""

        if integration == "wandb" and self.wandb_run is None:
            return False
        if integration == "weave" and self.weave_client is None:
            return False
        try:
            callback()
            return True
        except Exception as exc:
            self._record_delivery_failure(
                integration=integration,
                operation=operation,
                step=step,
                exc=exc,
            )
            if self.config.observability.delivery_failure_policy == "fail_run":
                raise
            return False

    def _record_delivery_failure(
        self,
        *,
        integration: str,
        operation: str,
        step: int | None,
        exc: Exception,
    ) -> None:
        self._telemetry_failure_count += 1
        self._telemetry_failure_counts[integration] = (
            self._telemetry_failure_counts.get(integration, 0) + 1
        )
        record = {
            "timestamp": time.time(),
            "integration": integration,
            "operation": operation,
            "step": step,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "failure_count": self._telemetry_failure_count,
            "config_fingerprint": self.config.fingerprint,
        }
        try:
            _append_bounded_jsonl(
                self.config.storage.output_dir / "telemetry_failures.jsonl",
                record,
                max_bytes=self.config.storage.max_log_file_mb * 1024 * 1024,
            )
        except OSError as write_error:
            record["local_record_error"] = (
                f"{type(write_error).__name__}: {write_error}"
            )
        count = self._telemetry_failure_counts[integration]
        if count == 1 or count & (count - 1) == 0:
            print(
                "[ART-Embodied] telemetry delivery degraded "
                + json.dumps(record, sort_keys=True),
                file=sys.stderr,
                flush=True,
            )

    def _should_log_progress(self, event: ExperimentProgress) -> bool:
        if event.status != "progress":
            return True
        if event.phase == "rollout":
            if not self.config.observability.wandb.log_rollout_progress:
                return False
            cadence = self.config.observability.wandb.rollout_progress_every_groups
        elif event.phase == "evaluation":
            if not self.config.observability.wandb.log_evaluation_progress:
                return False
            cadence = self.config.observability.wandb.evaluation_progress_every_episodes
        else:
            return True
        return event.completed is not None and (
            event.completed % cadence == 0 or event.completed == event.total
        )

    def _update_wandb_progress_summary(self, event: ExperimentProgress) -> None:
        """Update live run state without consuming a W&B history step."""

        self._progress_event_index += 1
        payload: dict[str, Any] = {
            "monitor/event_index": self._progress_event_index,
            "monitor/update": event.update,
            "monitor/phase": event.phase,
            "monitor/status": event.status,
        }
        if event.completed is not None:
            payload["monitor/completed"] = event.completed
        if event.total is not None:
            payload["monitor/total"] = event.total
            if event.completed is not None and event.total > 0:
                payload["monitor/progress"] = event.completed / event.total
        started_at = self._update_started_at.get(event.update)
        if started_at is not None:
            payload["monitor/update_elapsed_seconds"] = time.monotonic() - started_at
        if event.message is not None:
            payload["monitor/message"] = event.message
        for key, value in _progress_monitor_metrics(event).items():
            payload[f"monitor/{key}"] = value
        summary = getattr(self.wandb_run, "summary", None)
        update_summary = getattr(summary, "update", None)
        if not callable(update_summary):
            raise RuntimeError(
                "W&B progress monitoring requires wandb_run.summary.update(...); "
                "using run.log() here would consume optimizer history steps"
            )
        payload["monitor/telemetry_degraded"] = int(self._telemetry_failure_count > 0)
        payload["monitor/telemetry_delivery_failures"] = self._telemetry_failure_count
        self._deliver(
            "wandb",
            "update_progress_summary",
            event.update,
            lambda: update_summary(payload),
        )

    def _log_terminal_progress(self, event: ExperimentProgress) -> None:
        fields = [
            "[ART-Embodied]",
            f"update={event.update}",
            f"phase={event.phase}",
            f"status={event.status}",
        ]
        if event.completed is not None and event.total is not None:
            fields.append(f"completed={event.completed}/{event.total}")
        metrics = event.metrics or {}
        success_count = metrics.get("success_count")
        success_denominator = metrics.get("success_denominator")
        success_rate = metrics.get("success_rate")
        if success_count is not None and success_denominator is not None:
            success = f"success={int(success_count)}/{int(success_denominator)}"
            if success_rate is not None:
                success += f" ({float(success_rate):.1%})"
            fields.append(success)
        if "trajectories_completed" in metrics:
            fields.append(f"trajectories={int(metrics['trajectories_completed'])}")
        if "trajectories_per_second" in metrics:
            fields.append(
                f"trajectories_per_second={float(metrics['trajectories_per_second']):.3f}"
            )
        if event.message:
            fields.append(f"message={event.message}")
        print(" ".join(fields), flush=True)

    def _start_weave_update(self, update: int) -> None:
        if (
            self.weave_client is None
            or not self.config.observability.weave.trace_trajectories
            or update in self._weave_update_calls
        ):
            return
        self._weave_update_calls[update] = self.weave_client.create_call(
            "art_embodied.training_update",
            inputs={
                "update": update,
                "config_fingerprint": self.config.fingerprint,
                "policy": _policy_contract(self.config),
                "wandb": self._wandb_identity(),
            },
            display_name=f"training update {update}",
            use_stack=False,
        )

    def _fail_weave_update(self, update: int, message: str | None) -> None:
        root = self._weave_update_calls.pop(update, None)
        if root is None or self.weave_client is None:
            return
        self.weave_client.finish_call(
            root,
            exception=RuntimeError(message or "Training update failed"),
        )

    def _wandb_identity(self) -> dict[str, Any] | None:
        if self.wandb_run is None:
            return None
        return {
            "entity": getattr(self.wandb_run, "entity", None),
            "project": getattr(self.wandb_run, "project", None),
            "run_id": getattr(self.wandb_run, "id", None),
            "run_url": getattr(self.wandb_run, "url", None),
        }

    def _log_wandb(
        self,
        step: int,
        groups: Sequence[EmbodiedTrajectoryGroup],
        train_result: TrainResult,
        evaluation: EvaluationResult | None,
    ) -> None:
        payload: dict[str, Any] = {
            "experiment/update": step,
            "telemetry/degraded": int(self._telemetry_failure_count > 0),
            "telemetry/delivery_failures_total": self._telemetry_failure_count,
            "telemetry/wandb_delivery_failures": (
                self._telemetry_failure_counts["wandb"]
            ),
            "telemetry/weave_delivery_failures": (
                self._telemetry_failure_counts["weave"]
            ),
            **_wandb_training_history_payload(train_result.metrics),
        }
        if evaluation is not None:
            self._add_wandb_evaluation(
                payload=payload,
                step=step,
                evaluation=evaluation,
                checkpoint_role="candidate",
            )
        _validate_primary_training_namespace(payload)
        if self._native_update_history is not None:
            payload = self._native_update_history.training_payload(step, payload)
        # Legacy recipes use split rows. Native mode commits this update's
        # pre-update rollout, optimizer metrics and post-update evaluation once.
        self._persist_pending_wandb_history(step, payload)
        delivered = self._deliver(
            "wandb",
            "log_training_update",
            step,
            lambda: (
                self._native_update_history.commit(step, payload)
                if self._native_update_history is not None
                else self.wandb_run.log(payload, commit=True)
            ),
        )
        if not delivered and self._native_update_history is not None:
            raise RuntimeError("Native W&B training row was not committed")
        if delivered:
            self._clear_pending_wandb_history()

        checkpoint_path = getattr(train_result, "checkpoint_path", None)
        should_log_model = (
            self.config.observability.wandb.log_model_artifacts
            and checkpoint_path
            and (
                step % self.config.training.checkpoint_every_updates == 0
                or step in self.config.storage.retain_checkpoint_updates
            )
            and Path(checkpoint_path).exists()
        )
        if should_log_model:
            aliases = ["latest", f"update-{step}"]
            evaluation_success = _evaluation_success_rate(evaluation)
            is_best = (
                self.config.evaluation.checkpoint_selection == "evaluation_success"
                and evaluation_success is not None
                and (
                    self._best_logged_evaluation_success is None
                    or evaluation_success > self._best_logged_evaluation_success
                )
            )
            if is_best:
                aliases.append("best")
            delivered = self._deliver(
                "wandb",
                "log_model_artifact",
                step,
                lambda: self._log_model_artifact(
                    checkpoint_path=Path(checkpoint_path),
                    step=step,
                    aliases=aliases,
                    train_result=train_result,
                    evaluation=evaluation,
                ),
            )
            if delivered and is_best:
                self._best_logged_evaluation_success = evaluation_success
        if (
            evaluation is not None
            and self.config.observability.wandb.log_evaluation_artifacts
        ):
            self._deliver(
                "wandb",
                "log_evaluation_artifact",
                step,
                lambda: self._log_evaluation_artifact(step, evaluation),
            )

    def _log_wandb_rollout(
        self,
        policy_version: int,
        groups: Sequence[EmbodiedTrajectoryGroup],
        *,
        evidence: dict[str, Any],
    ) -> None:
        trajectories = [trajectory for group in groups for trajectory in group]
        rewards = [trajectory.reward for trajectory in trajectories]
        details = _trajectory_metric_payload("train_details", trajectories)
        success_rate = details.pop("train_details/success_rate", None)
        payload: dict[str, Any] = {
            **self._pending_wandb_payload,
            "experiment/update": policy_version,
            "signal/rollout_groups": len(groups),
            "signal/rollout_trajectories": len(rewards),
            "signal/mixed_reward_groups": evidence["summary"]["mixed_reward_groups"],
            "signal/all_failure_groups": evidence["summary"]["all_failure_groups"],
            "signal/all_success_groups": evidence["summary"]["all_success_groups"],
            "signal/mixed_reward_group_fraction": evidence["summary"][
                "mixed_reward_group_fraction"
            ],
            "signal/reward_range_mean": fmean(
                max(row["rewards"]) - min(row["rewards"])
                for row in evidence["groups"]
                if row["rewards"]
            )
            if evidence["groups"]
            else 0.0,
            "train/reward_mean": fmean(rewards) if rewards else 0.0,
            **details,
        }
        table_type = getattr(self._wandb_module, "Table", None)
        if callable(table_type):
            table_columns = [
                "policy_version",
                "group_index",
                "scenario_id",
                "task",
                "environment_seed",
                "completed_trajectories",
                "failed_trajectories",
                "success_count",
                "success_rate",
                "reward_mean",
                "signal_class",
            ]
            payload["train_details/group_outcomes"] = table_type(
                columns=table_columns,
                data=[
                    [row[column] for column in table_columns]
                    for row in evidence["groups"]
                ],
            )
        if success_rate is not None:
            payload["train/success_rate"] = success_rate
        videos_remaining = self.config.observability.videos_per_update
        train_videos_logged = 0
        indexed_trajectories = [
            ((group_index, trajectory_index), trajectory)
            for group_index, group in enumerate(groups)
            for trajectory_index, trajectory in enumerate(group)
        ]
        for _, trajectory in _representative_trajectories(
            indexed_trajectories,
            limit=videos_remaining,
        ):
            videos = wandb_video_payload(
                trajectory,
                prefix="media/simulation/train",
                max_videos=videos_remaining,
                start_index=train_videos_logged,
                wandb_module=self._wandb_module,
                media_role="simulation",
            )
            videos_remaining -= len(videos)
            train_videos_logged += len(videos)
            payload.update(videos)
        lookahead_settings = self.config.observability.lookahead_preview
        lookahead_remaining = (
            lookahead_settings.videos_per_update if lookahead_settings.enabled else 0
        )
        lookahead_logged = 0
        for _, trajectory in _representative_trajectories(
            indexed_trajectories,
            limit=lookahead_remaining,
        ):
            videos = wandb_video_payload(
                trajectory,
                prefix="media/lookahead/train",
                max_videos=lookahead_remaining,
                start_index=lookahead_logged,
                wandb_module=self._wandb_module,
                media_role="lookahead_preview",
            )
            lookahead_remaining -= len(videos)
            lookahead_logged += len(videos)
            payload.update(videos)
        if self.config.observability.require_train_video and train_videos_logged == 0:
            raise RuntimeError(
                "Required W&B train video is missing. Check env.render(), video "
                "capture limits, and local media retention."
            )
        _validate_primary_training_namespace(payload)
        if self._native_update_history is not None:
            self._native_update_history.stage_rollout(policy_version, payload)
            return
        if (
            self._pending_wandb_policy_version is not None
            and self._pending_wandb_policy_version != policy_version
        ):
            raise RuntimeError(
                "Pending W&B history belongs to a different policy version: "
                f"pending={self._pending_wandb_policy_version}, "
                f"rollout={policy_version}"
            )
        expected_native_step = (
            int(self.wandb_run.step) + 1
            if policy_version == 0 and self._enforce_initial_history_commit
            else None
        )
        delivered = self._deliver(
            "wandb",
            "log_rollout_policy_version",
            policy_version,
            lambda: self.wandb_run.log(payload, commit=True),
        )
        if policy_version == 0 and self._enforce_initial_history_commit:
            if not delivered:
                raise RuntimeError(
                    "W&B Step-0 history commit failed; refusing to start the "
                    "first optimizer update without its baseline and rollout row"
                )
            assert expected_native_step is not None
            self._verify_initial_history_commit(expected_native_step)
        if delivered:
            self._pending_wandb_policy_version = None
            self._clear_pending_wandb_history()

    def _persist_rollout_group_evidence(
        self,
        policy_version: int,
        groups: Sequence[EmbodiedTrajectoryGroup],
    ) -> dict[str, Any]:
        """Persist every reward group before relying on remote telemetry."""

        rows = [
            _rollout_group_evidence_row(policy_version, index, group)
            for index, group in enumerate(groups)
        ]
        mixed = sum(row["signal_class"] == "mixed" for row in rows)
        all_failure = sum(row["signal_class"] == "all_failure" for row in rows)
        all_success = sum(row["signal_class"] == "all_success" for row in rows)
        payload = {
            "schema_version": 1,
            "policy_version": int(policy_version),
            "config_fingerprint": self.config.fingerprint,
            "summary": {
                "groups": len(rows),
                "trajectories": sum(int(row["completed_trajectories"]) for row in rows),
                "mixed_reward_groups": mixed,
                "all_failure_groups": all_failure,
                "all_success_groups": all_success,
                "mixed_reward_group_fraction": mixed / len(rows) if rows else 0.0,
            },
            "groups": rows,
        }
        path = (
            Path(self.config.storage.output_dir)
            / "rollout-evidence"
            / f"policy-version-{policy_version:06d}.json"
        )
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != payload:
                raise FileExistsError(
                    "Refusing to overwrite different rollout evidence for "
                    f"policy version {policy_version}: {path}"
                )
            return payload
        write_json_atomic(path, payload, indent=2, sort_keys=True)
        return payload

    def _verify_initial_history_commit(self, expected_native_step: int) -> None:
        """Fail closed unless the primary Run acknowledged the rollout commit.

        W&B history upload is asynchronous. Reading ``Run.step`` waits for the
        SDK service to process the committed row. The measured initial
        evaluation may already occupy a separate row at experiment/update=0.
        Server-side history queries are not
        used here because their materialized views can lag the live UI.
        """

        if self.wandb_run is None:
            raise RuntimeError("W&B Step-0 commit gate has no primary Run")
        identity = self._wandb_identity() or {}
        missing = [
            key
            for key in ("entity", "project", "run_id", "run_url")
            if not identity.get(key)
        ]
        if missing:
            raise RuntimeError(
                "W&B primary Run is missing identity fields after Step-0 commit: "
                + ", ".join(missing)
            )
        try:
            native_step = int(self.wandb_run.step)
        except Exception as exc:
            raise RuntimeError(
                "W&B did not acknowledge the committed Step-0 history row"
            ) from exc
        if native_step != expected_native_step:
            raise RuntimeError(
                f"W&B native Step must be {expected_native_step} after committing policy version 0; "
                f"observed {native_step}"
            )
        write_json_atomic(
            Path(self.config.storage.output_dir)
            / "wandb"
            / "initial-history-commit.json",
            {
                "schema_version": 1,
                "policy_version": 0,
                "native_step_after_commit": native_step,
                "committed_at_unix_seconds": time.time(),
                "wandb": identity,
                "config_fingerprint": self.config.fingerprint,
            },
            indent=2,
            sort_keys=True,
        )

    def _persist_pending_wandb_history(
        self,
        policy_version: int,
        payload: dict[str, Any],
    ) -> None:
        scalar_payload = {
            key: value
            for key, value in payload.items()
            if value is None or isinstance(value, str | int | float | bool)
        }
        write_json_atomic(
            self._pending_wandb_path,
            {
                "schema_version": 1,
                "policy_version": policy_version,
                "resume_contract_fingerprint": (
                    self.config.resume_contract_fingerprint
                ),
                "payload": scalar_payload,
            },
        )
        self._pending_wandb_policy_version = policy_version
        self._pending_wandb_payload = scalar_payload

    def _restore_pending_wandb_history(self) -> None:
        if not self._pending_wandb_path.is_file():
            return
        raw = json.loads(self._pending_wandb_path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != 1:
            raise ValueError("Unsupported pending W&B history schema")
        if raw.get("resume_contract_fingerprint") != (
            self.config.resume_contract_fingerprint
        ):
            raise ValueError(
                "Pending W&B history does not match the resumed experiment contract"
            )
        policy_version = raw.get("policy_version")
        payload = raw.get("payload")
        if not isinstance(policy_version, int) or not isinstance(payload, dict):
            raise ValueError("Pending W&B history is malformed")
        if any(
            not isinstance(key, str)
            or not (value is None or isinstance(value, str | int | float | bool))
            for key, value in payload.items()
        ):
            raise ValueError("Pending W&B history contains a non-scalar value")
        self._pending_wandb_policy_version = policy_version
        self._pending_wandb_payload = _bounded_pending_wandb_payload(payload)

    def _clear_pending_wandb_history(self) -> None:
        self._pending_wandb_policy_version = None
        self._pending_wandb_payload = {}
        self._pending_wandb_path.unlink(missing_ok=True)

    def _add_wandb_evaluation(
        self,
        *,
        payload: dict[str, Any],
        step: int,
        evaluation: EvaluationResult,
        checkpoint_role: str,
    ) -> None:
        # Keep the decision surface deliberately small and stable. Detailed task
        # and scenario metrics live under eval_tasks/ and eval_details/; putting
        # the primary curve beside them makes the result effectively invisible
        # in a real W&B workspace.
        primary_namespace = (
            "test"
            if self.config.evaluation.data_role == "sealed_test"
            else "validation"
        )
        primary_payload = _wandb_primary_evaluation_payload(
            evaluation,
            namespace=primary_namespace,
        )
        payload.update(primary_payload)
        payload[f"{primary_namespace}/checkpoint_role"] = checkpoint_role
        if self.wandb_run is not None:
            summary = getattr(self.wandb_run, "summary", None)
            if summary is not None:
                summary.update(primary_payload)
                latest_success_rate = primary_payload.get(
                    f"{primary_namespace}/success_rate"
                )
                is_summary_best = latest_success_rate is not None and (
                    primary_namespace not in self._best_summary_evaluation_success
                    or latest_success_rate
                    > self._best_summary_evaluation_success[primary_namespace]
                )
                summary.update(
                    {
                        f"{primary_namespace}/latest_update": step,
                        f"{primary_namespace}/split": self.config.evaluation.split,
                        f"{primary_namespace}/data_role": (
                            self.config.evaluation.data_role
                        ),
                        **(
                            {
                                f"{primary_namespace}/latest_success_rate": (
                                    latest_success_rate
                                )
                            }
                            if latest_success_rate is not None
                            else {}
                        ),
                        **(
                            {
                                f"{primary_namespace}/best_update": step,
                                f"{primary_namespace}/best_success_rate": (
                                    latest_success_rate
                                ),
                            }
                            if is_summary_best
                            else {}
                        ),
                    }
                )
                if is_summary_best:
                    self._best_summary_evaluation_success[primary_namespace] = (
                        latest_success_rate
                    )
        # Per-scenario and per-task scalars create hundreds of W&B panels and
        # bury the decision surface. Keep operational aggregate metrics in
        # history; preserve episode/task detail in Tables and the immutable
        # evaluation Artifact instead.
        payload.update(
            {
                _metric_key("eval", key): value
                for key, value in evaluation.metrics.items()
                if key not in _PRIMARY_EVALUATION_METRICS
                and not _is_evaluation_detail_metric(key)
            }
        )
        videos_remaining = self.config.observability.videos_per_evaluation
        evaluation_videos_logged = 0
        for _, trajectory in _representative_trajectories(
            evaluation.trajectories,
            limit=videos_remaining,
        ):
            videos = wandb_video_payload(
                trajectory,
                prefix="media/simulation/eval",
                max_videos=videos_remaining,
                start_index=evaluation_videos_logged,
                wandb_module=self._wandb_module,
                media_role="simulation",
            )
            videos_remaining -= len(videos)
            evaluation_videos_logged += len(videos)
            payload.update(videos)
        lookahead_settings = self.config.observability.lookahead_preview
        lookahead_remaining = (
            lookahead_settings.videos_per_evaluation
            if lookahead_settings.enabled
            else 0
        )
        lookahead_logged = 0
        for _, trajectory in _representative_trajectories(
            evaluation.trajectories,
            limit=lookahead_remaining,
        ):
            videos = wandb_video_payload(
                trajectory,
                prefix="media/lookahead/eval",
                max_videos=lookahead_remaining,
                start_index=lookahead_logged,
                wandb_module=self._wandb_module,
                media_role="lookahead_preview",
            )
            lookahead_remaining -= len(videos)
            lookahead_logged += len(videos)
            payload.update(videos)
        if (
            self.config.observability.require_evaluation_video
            and evaluation_videos_logged == 0
        ):
            raise RuntimeError(
                "Required W&B evaluation video is missing. Check env.render(), "
                "video capture limits, and local media retention."
            )
        evaluation_table = self._evaluation_table(step, evaluation)
        if evaluation_table is not None:
            payload["eval_details/episodes_table"] = evaluation_table
        task_table = self._evaluation_task_table(step, evaluation)
        if task_table is not None:
            payload["eval_details/task_summary_table"] = task_table

    def _evaluation_table(
        self,
        step: int,
        evaluation: EvaluationResult,
    ) -> Any | None:
        if (
            not self.config.observability.wandb.log_evaluation_table
            or self._wandb_module is None
            or self.config.observability.wandb.max_evaluation_table_rows == 0
        ):
            return None
        table_type = getattr(self._wandb_module, "Table", None)
        if not callable(table_type):
            return None
        columns = [
            "update",
            "episode",
            "scenario_id",
            "task",
            "reward",
            "success",
            "episode_steps",
            "duration_seconds",
            "environment_seed",
            "policy_seed",
            "completed",
            "error_type",
        ]
        rows = []
        limit = self.config.observability.wandb.max_evaluation_table_rows
        outcome_rows = _evaluation_outcome_rows(evaluation)
        for outcome in outcome_rows[:limit]:
            rows.append(
                [
                    step,
                    outcome.get("episode"),
                    outcome.get("scenario_id"),
                    outcome.get("task"),
                    outcome.get("reward"),
                    outcome.get("success"),
                    outcome.get("episode_steps"),
                    outcome.get("duration_seconds"),
                    outcome.get("environment_seed"),
                    outcome.get("policy_seed"),
                    outcome.get("completed"),
                    outcome.get("error_type"),
                ]
            )
        if not rows:
            for episode, trajectory in enumerate(evaluation.trajectories[:limit]):
                rows.append(
                    [
                        step,
                        episode,
                        trajectory.metadata.get("scenario_id"),
                        trajectory.task,
                        trajectory.reward,
                        trajectory.metrics.get("success"),
                        trajectory.metrics.get(
                            "episode_steps", len(trajectory.actions)
                        ),
                        trajectory.metrics.get("duration"),
                        trajectory.metadata.get("environment_seed"),
                        trajectory.metadata.get("policy_seed"),
                        True,
                        None,
                    ]
                )
        return table_type(columns=columns, data=rows)

    def _evaluation_task_table(
        self,
        step: int,
        evaluation: EvaluationResult,
    ) -> Any | None:
        """Return one compact task-level table instead of scalar panel spam."""

        if (
            not self.config.observability.wandb.log_evaluation_table
            or self._wandb_module is None
        ):
            return None
        table_type = getattr(self._wandb_module, "Table", None)
        if not callable(table_type):
            return None
        task_rows: dict[str, dict[str, Any]] = {}
        for key, value in evaluation.metrics.items():
            if not isinstance(value, int | float | bool):
                continue
            if key.startswith("paired/task/"):
                rest = key.removeprefix("paired/task/")
                task, separator, metric = rest.rpartition("/")
                if separator:
                    task_rows.setdefault(task, {})[metric] = value
            elif key.startswith("task/"):
                rest = key.removeprefix("task/")
                task, separator, metric = rest.rpartition("/")
                if separator:
                    task_rows.setdefault(task, {})[f"candidate_{metric}"] = value
        if not task_rows:
            return None
        columns = [
            "update",
            "task",
            "episodes",
            "baseline_success_rate",
            "candidate_success_rate",
            "success_rate_lift",
        ]
        rows = []
        for task, metrics in sorted(task_rows.items()):
            rows.append(
                [
                    step,
                    task,
                    metrics.get("episodes"),
                    metrics.get("baseline_success_rate"),
                    metrics.get("candidate_success_rate"),
                    metrics.get("success_rate_lift"),
                ]
            )
        return table_type(columns=columns, data=rows)

    def _log_model_artifact(
        self,
        *,
        checkpoint_path: Path,
        step: int,
        aliases: list[str],
        train_result: TrainResult,
        evaluation: EvaluationResult | None,
    ) -> None:
        artifact_type = (
            getattr(self._wandb_module, "Artifact", None)
            if self._wandb_module is not None
            else None
        )
        log_artifact = getattr(self.wandb_run, "log_artifact", None)
        if callable(artifact_type) and callable(log_artifact):
            artifact = artifact_type(
                name=f"{self.config.experiment.run}-checkpoint",
                type="model",
                metadata=_wandb_artifact_metadata(
                    {
                        "update": step,
                        "config_fingerprint": self.config.fingerprint,
                        "algorithm": self.config.algorithm.type,
                        "policy_type": self.config.policy.type,
                        "policy": _policy_contract(self.config),
                        "checkpoint": _checkpoint_manifest(checkpoint_path),
                        "train_metrics": {
                            str(key): value
                            for key, value in train_result.metrics.items()
                            if isinstance(value, int | float | bool | str)
                        },
                        "evaluation_metrics": (
                            evaluation.metrics if evaluation is not None else None
                        ),
                    }
                ),
            )
            if checkpoint_path.is_dir():
                artifact.add_dir(str(checkpoint_path))
            else:
                artifact.add_file(str(checkpoint_path))
            log_artifact(artifact, aliases=aliases)
            return
        self.wandb_run.log_model(
            path=str(checkpoint_path),
            name=f"{self.config.experiment.run}-checkpoint",
            aliases=aliases,
        )

    def _log_evaluation_artifact(
        self,
        step: int,
        evaluation: EvaluationResult,
    ) -> None:
        artifact_type = (
            getattr(self._wandb_module, "Artifact", None)
            if self._wandb_module is not None
            else None
        )
        log_artifact = getattr(self.wandb_run, "log_artifact", None)
        if not callable(artifact_type) or not callable(log_artifact):
            return
        evidence_path_value = evaluation.artifacts.get("evaluation_evidence_json")
        if not evidence_path_value:
            return
        evidence_path = Path(evidence_path_value)
        if not evidence_path.is_file():
            raise FileNotFoundError(
                f"Evaluation evidence artifact is missing: {evidence_path}"
            )
        if (
            evidence_path.stat().st_size
            > self.config.storage.max_log_file_mb * 1024 * 1024
        ):
            raise ValueError(
                f"Evaluation evidence artifact is too large: {evidence_path}"
            )
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        artifact = artifact_type(
            name=f"{self.config.experiment.run}-evaluation",
            type="evaluation",
            metadata={
                "update": step,
                "config_fingerprint": self.config.fingerprint,
                "split": self.config.evaluation.split,
                "data_role": self.config.evaluation.data_role,
                "metrics": evaluation.metrics,
                "identity": evidence.get("identity"),
                "outcomes_sha256": (evidence.get("outcomes") or {}).get("sha256"),
            },
        )
        added: set[Path] = set()
        artifact_names: set[str] = set()
        for artifact_key, artifact_path_value in evaluation.artifacts.items():
            artifact_path = Path(artifact_path_value).expanduser()
            if artifact_path in added or not artifact_path.is_file():
                continue
            artifact_name = artifact_path.name
            if artifact_name in artifact_names:
                artifact_name = f"{artifact_key}/{artifact_path.name}"
                artifact.add_file(str(artifact_path), name=artifact_name)
            else:
                artifact.add_file(str(artifact_path))
            added.add(artifact_path)
            artifact_names.add(artifact_name)
        aliases = ["latest", f"update-{step}"]
        if self.config.evaluation.data_role == "sealed_test":
            aliases.append("sealed-test")
        log_artifact(artifact, aliases=aliases)

    def _log_weave(
        self,
        step: int,
        groups: Sequence[EmbodiedTrajectoryGroup],
        train_result: TrainResult,
        evaluation: EvaluationResult | None,
    ) -> None:
        if not self.config.observability.weave.trace_trajectories:
            return
        root = self._weave_update_calls.get(step)
        if root is None:
            self._start_weave_update(step)
            root = self._weave_update_calls.get(step)
        if root is None:
            return
        try:
            max_groups = self.config.observability.weave.max_groups_per_update
            for group_index, group in _representative_groups(
                groups,
                limit=max_groups,
            ):
                group_call = self.weave_client.create_call(
                    "art_embodied.trajectory_group",
                    inputs={
                        "update": step,
                        "group_index": group_index,
                        "metadata": group.metadata,
                    },
                    parent=root,
                    display_name=f"group {group_index}",
                    use_stack=False,
                )
                try:
                    max_trajectories = (
                        self.config.observability.weave.max_trajectories_per_group
                    )
                    for trajectory_index, trajectory in _representative_trajectories(
                        group.trajectories,
                        limit=max_trajectories,
                        require_video=False,
                    ):
                        trajectory_call = self.weave_client.create_call(
                            "art_embodied.trajectory",
                            inputs={
                                "update": step,
                                "group_index": group_index,
                                "trajectory_index": trajectory_index,
                                "task": trajectory.task,
                            },
                            parent=group_call,
                            display_name=(
                                f"trajectory {group_index}.{trajectory_index}"
                            ),
                            use_stack=False,
                        )
                        self.weave_client.finish_call(
                            trajectory_call,
                            output=self._weave_trajectory_output(trajectory),
                        )
                    self.weave_client.finish_call(
                        group_call,
                        output={
                            "rewards": group.rewards(),
                            "metrics": group.metrics,
                            "exceptions": len(group.exceptions),
                        },
                    )
                except BaseException as exc:
                    self.weave_client.finish_call(group_call, exception=exc)
                    raise
            output: dict[str, Any] = {
                "train_metrics": train_result.metrics,
                "checkpoint_path": getattr(train_result, "checkpoint_path", None),
                "checkpoint": _checkpoint_manifest_from_result(train_result),
                "wandb": self._wandb_identity(),
            }
            if evaluation is not None:
                output["evaluation"] = {
                    "metrics": evaluation.metrics,
                    "artifacts": evaluation.artifacts,
                }
                self._trace_weave_evaluation(root, step, evaluation)
            self.weave_client.finish_call(root, output=output)
            self._weave_update_calls.pop(step, None)
        except BaseException as exc:
            self.weave_client.finish_call(root, exception=exc)
            self._weave_update_calls.pop(step, None)
            raise

    def _log_weave_evaluation_run(
        self,
        step: int,
        evaluation: EvaluationResult,
    ) -> None:
        if not self.config.observability.weave.trace_trajectories:
            return
        root = self.weave_client.create_call(
            "art_embodied.evaluation_run",
            inputs={
                "update": step,
                "split": self.config.evaluation.split,
                "data_role": self.config.evaluation.data_role,
                "episodes": self.config.evaluation.episodes,
                "config_fingerprint": self.config.fingerprint,
                "policy": _policy_contract(self.config),
                "wandb": self._wandb_identity(),
            },
            display_name=f"evaluation run {step}",
            use_stack=False,
        )
        try:
            self._trace_weave_evaluation(root, step, evaluation)
            self.weave_client.finish_call(
                root,
                output={
                    "metrics": evaluation.metrics,
                    "artifacts": evaluation.artifacts,
                    "evidence": _evaluation_evidence_summary(evaluation),
                    "wandb": self._wandb_identity(),
                },
            )
        except BaseException as exc:
            self.weave_client.finish_call(root, exception=exc)
            raise

    def _trace_weave_evaluation(
        self,
        parent: Any,
        step: int,
        evaluation: EvaluationResult,
    ) -> None:
        evaluation_call = self.weave_client.create_call(
            "art_embodied.evaluation",
            inputs={
                "update": step,
                "split": self.config.evaluation.split,
                "data_role": self.config.evaluation.data_role,
                "episodes": self.config.evaluation.episodes,
            },
            parent=parent,
            display_name=f"evaluation update {step}",
            use_stack=False,
        )
        try:
            max_trajectories = min(
                len(evaluation.trajectories),
                self.config.observability.weave.max_evaluation_trajectories,
            )
            for trajectory_index, trajectory in _representative_trajectories(
                evaluation.trajectories,
                limit=max_trajectories,
                require_video=False,
            ):
                trajectory_call = self.weave_client.create_call(
                    "art_embodied.evaluation_trajectory",
                    inputs={
                        "update": step,
                        "trajectory_index": trajectory_index,
                        "task": trajectory.task,
                    },
                    parent=evaluation_call,
                    display_name=f"evaluation trajectory {trajectory_index}",
                    use_stack=False,
                )
                self.weave_client.finish_call(
                    trajectory_call,
                    output=self._weave_trajectory_output(trajectory),
                )
            self.weave_client.finish_call(
                evaluation_call,
                output={
                    "metrics": evaluation.metrics,
                    "artifacts": evaluation.artifacts,
                    "evidence": _evaluation_evidence_summary(evaluation),
                },
            )
        except BaseException as exc:
            self.weave_client.finish_call(evaluation_call, exception=exc)
            raise

    def _weave_trajectory_output(
        self, trajectory: EmbodiedTrajectory
    ) -> dict[str, Any]:
        output: dict[str, Any] = {
            "task": trajectory.task,
            "reward": trajectory.reward,
            "summary": {
                "success": trajectory.metrics.get("success"),
                "episode_steps": trajectory.metrics.get(
                    "episode_steps", len(trajectory.actions)
                ),
                "terminated": trajectory.metrics.get("terminated"),
                "truncated": trajectory.metrics.get("truncated"),
                "duration": trajectory.metrics.get("duration"),
            },
            "metrics": trajectory.metrics,
            "metadata": _compact_trace_metadata(trajectory.metadata),
            "reward_events": [
                event.model_dump(mode="json") for event in trajectory.rewards
            ],
            "actions": [
                {
                    "step": action.step,
                    "kind": action.kind,
                    "decoded": action.model_dump(mode="json")["decoded"],
                    "metadata": _compact_trace_metadata(action.metadata),
                }
                for action in trajectory.actions
            ],
            "steps": _weave_step_timeline(trajectory),
        }
        if self._weave_module is None:
            return output
        videos: dict[str, Any] = {}
        for index, (media, path) in enumerate(local_video_media_refs(trajectory)[:1]):
            videos[f"video_{index}"] = self._weave_module.Content.from_path(
                path,
                mimetype=media.mime_type,
                metadata={
                    "caption": media.caption,
                    "step": media.step,
                    **media.metadata,
                },
            )
        if videos:
            output["media"] = videos
        return output


def _metric_key(prefix: str, key: str) -> str:
    normalized = key.removeprefix("train/").removeprefix("eval/")
    if prefix == "eval":
        if normalized.startswith(("task/", "paired/task/")):
            return f"eval_tasks/{normalized}"
        return f"eval_details/{normalized}"
    return f"{prefix}/{normalized}"


def _rollout_group_evidence_row(
    policy_version: int,
    fallback_group_index: int,
    group: EmbodiedTrajectoryGroup,
) -> dict[str, Any]:
    rewards = [float(trajectory.reward) for trajectory in group.trajectories]
    successes: list[int] = []
    for trajectory in group.trajectories:
        value = trajectory.metrics.get("success")
        if isinstance(value, bool):
            successes.append(int(value))
        elif isinstance(value, int | float) and float(value) in (0.0, 1.0):
            successes.append(int(value))
        else:
            successes.append(int(float(trajectory.reward) > 0.0))
    success_count = sum(successes)
    if not successes or success_count == 0:
        signal_class = "all_failure"
    elif success_count == len(successes):
        signal_class = "all_success"
    else:
        signal_class = "mixed"
    tasks = sorted({trajectory.task for trajectory in group.trajectories})
    metadata = make_json_safe(group.metadata)
    return {
        "policy_version": int(policy_version),
        "group_index": int(metadata.get("group_index", fallback_group_index)),
        "scenario_id": str(metadata.get("scenario_id", "")),
        "task": tasks[0] if len(tasks) == 1 else ",".join(tasks),
        "environment_seed": metadata.get("environment_seed"),
        "completed_trajectories": len(group.trajectories),
        "failed_trajectories": len(group.exceptions),
        "success_count": success_count,
        "success_rate": success_count / len(successes) if successes else 0.0,
        "reward_mean": fmean(rewards) if rewards else 0.0,
        "signal_class": signal_class,
        "rewards": rewards,
        "successes": successes,
        "group_metadata": metadata,
    }


def _wandb_training_metric_key(key: str) -> str:
    """Route backend metrics without burying the primary training curves."""

    normalized = key.removeprefix("train/").removeprefix("eval/")
    lowered = normalized.lower()
    if "worker_gradient_" in lowered:
        namespace = "signal"
    elif normalized.startswith(("rollout/", "training/")) or any(
        marker in lowered for marker in _PERFORMANCE_METRIC_MARKERS
    ):
        namespace = "performance"
    elif any(marker in lowered for marker in _OPTIMIZATION_METRIC_MARKERS):
        namespace = "optimization"
    elif any(marker in lowered for marker in _SIGNAL_METRIC_MARKERS):
        namespace = "signal"
    else:
        namespace = "optimization"
    return f"{namespace}/{normalized}"


def _validate_primary_training_namespace(payload: dict[str, Any]) -> None:
    """Fail before W&B history is polluted by non-primary train metrics."""

    unexpected = sorted(
        key
        for key in payload
        if key.startswith("train/") and key not in _PRIMARY_TRAIN_METRIC_KEYS
    )
    if unexpected:
        raise ValueError(
            "W&B train/* is reserved for train/success_rate and "
            f"train/reward_mean; route these metrics elsewhere: {unexpected}"
        )


def _wandb_primary_evaluation_payload(
    evaluation: EvaluationResult,
    *,
    namespace: str = "validation",
) -> dict[str, float]:
    """Return the small, chart-ready fixed-evaluation decision surface.

    These keys are intentionally independent of detailed evaluator output so
    every checkpoint evaluation in one training run lands on the same W&B
    charts. An ad-hoc calibration from another run must never be inserted into
    this series.
    """

    metrics = evaluation.metrics
    payload: dict[str, float] = {}
    aliases = {
        "success_rate": f"{namespace}/success_rate",
        "task_macro_success_rate": f"{namespace}/task_macro_success_rate",
        "paired/candidate_success_rate": f"{namespace}/success_rate",
        "paired/success_rate_lift": f"{namespace}/success_rate_lift",
        "paired/success_rate_lift_ci95_low": (
            f"{namespace}/success_rate_lift_ci95_low"
        ),
        "paired/success_rate_lift_ci95_high": (
            f"{namespace}/success_rate_lift_ci95_high"
        ),
        "paired/candidate_task_macro_success_rate": (
            f"{namespace}/task_macro_success_rate"
        ),
        "paired/task_macro_success_rate_lift": (
            f"{namespace}/task_macro_success_rate_lift"
        ),
        "paired/task_macro_success_rate_lift_ci95_low": (
            f"{namespace}/task_macro_success_rate_lift_ci95_low"
        ),
        "paired/task_macro_success_rate_lift_ci95_high": (
            f"{namespace}/task_macro_success_rate_lift_ci95_high"
        ),
        "paired/mcnemar_exact_p_value": f"{namespace}/mcnemar_exact_p_value",
        "paired/improved_pairs": f"{namespace}/improved_pairs",
        "paired/regressed_pairs": f"{namespace}/regressed_pairs",
        "paired/unchanged_success_pairs": f"{namespace}/unchanged_success_pairs",
        "paired/unchanged_failure_pairs": f"{namespace}/unchanged_failure_pairs",
    }
    for source, destination in aliases.items():
        value = metrics.get(source)
        if isinstance(value, int | float | bool):
            payload[destination] = float(value)
    for prefix in ("category", "difficulty"):
        marker = f"{prefix}/"
        for key, value in metrics.items():
            if (
                key.startswith(marker)
                and key.endswith("/success_rate")
                and isinstance(value, int | float | bool)
            ):
                payload[f"{namespace}/{key}"] = float(value)
    episodes = metrics.get("episodes")
    if isinstance(episodes, int | float | bool):
        payload[f"{namespace}/episodes"] = float(episodes)
    return payload


def _default_wandb_writer_label(connection: str) -> str:
    """Identify one W&B console/system-metrics writer within a shared run."""

    role = "eval" if connection == "shared_worker" else "coordinator"
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        return f"{role}-slurm-{job_id}"
    return f"{role}-{socket.gethostname()}-{os.getpid()}"


def _is_evaluation_detail_metric(key: str) -> bool:
    return key.startswith(
        ("scenario/", "task/", "category/", "difficulty/", "paired/task/")
    )


def _policy_contract(config: EmbodiedExperimentConfig) -> dict[str, Any]:
    return {
        "type": config.policy.type,
        "path": config.policy.path,
        "revision": config.policy.revision,
        "dtype": config.policy.dtype,
        "trainable_parameter_strategy": config.policy.trainable_parameter_strategy,
        "config_fingerprint": config.fingerprint,
    }


def _declare_input_model_artifact(
    *,
    config: EmbodiedExperimentConfig,
    wandb_module: Any,
    wandb_run: Any,
) -> None:
    """Back up and declare a completed local warm-start policy as run input.

    This intentionally uses ``Run.use_artifact`` rather than reopening the SFT
    run after training. The resulting lineage records exactly which immutable
    checkpoint the trajectory-RL run consumed without appending console or
    history records to an already completed producer run.
    """

    checkpoint = Path(config.policy.path).expanduser().resolve()
    marker_path = checkpoint / "art_embodied_sft_complete.json"
    if not checkpoint.is_dir():
        raise FileNotFoundError(
            f"W&B input-model artifact requires a local policy directory: {checkpoint}"
        )
    if not marker_path.is_file() or marker_path.stat().st_size > 1024 * 1024:
        raise FileNotFoundError(
            "W&B input-model artifact requires a bounded SFT completion marker: "
            f"{marker_path}"
        )
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid SFT completion marker: {marker_path}") from exc
    if marker.get("schema_version") != 1 or marker.get("status") != "complete":
        raise ValueError(
            "SFT completion marker must have schema_version=1 and status='complete'"
        )

    use_artifact = getattr(wandb_run, "use_artifact", None)
    if not callable(use_artifact):
        raise RuntimeError("W&B input-model lineage requires Run.use_artifact support")
    artifact_ref = config.observability.wandb.input_model_artifact_ref
    if artifact_ref is not None:
        use_artifact(artifact_ref, use_as="warm_start_policy")
        return

    artifact_type = getattr(wandb_module, "Artifact", None)
    if not callable(artifact_type):
        raise RuntimeError("W&B input-model backup requires Artifact support")
    artifact = artifact_type(
        name=f"{config.experiment.run}-warm-start",
        type="model",
        metadata={
            "role": "warm_start_policy",
            "policy": _policy_contract(config),
            "sft_completion": marker,
        },
    )
    artifact.add_dir(str(checkpoint))
    use_artifact(artifact, use_as="warm_start_policy")


def _checkpoint_manifest_from_result(
    train_result: TrainResult,
) -> dict[str, Any] | None:
    checkpoint_path = getattr(train_result, "checkpoint_path", None)
    if not checkpoint_path:
        return None
    return _checkpoint_manifest(Path(checkpoint_path))


def _checkpoint_manifest(checkpoint_path: Path) -> dict[str, Any] | None:
    if not checkpoint_path.is_dir():
        return None
    path = checkpoint_path / "art_embodied_checkpoint.json"
    try:
        if not path.is_file() or path.stat().st_size > 64 * 1024:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    allowed = (
        "schema_version",
        "type",
        "base_model_id",
        "base_model_revision",
        "model_loader",
    )
    result = {key: payload.get(key) for key in allowed if key in payload}
    completion_path = checkpoint_path / CHECKPOINT_COMPLETE_MARKER
    try:
        if completion_path.is_file() and completion_path.stat().st_size <= 1024 * 1024:
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            files = completion.get("files")
            result["transaction"] = {
                "schema_version": completion.get("schema_version"),
                "complete": completion.get("complete"),
                "config_fingerprint": completion.get("config_fingerprint"),
                "resume_contract_fingerprint": completion.get(
                    "resume_contract_fingerprint"
                ),
                "metadata": completion.get("metadata"),
                "file_count": len(files) if isinstance(files, list) else None,
            }
    except (OSError, json.JSONDecodeError):
        # Artifact upload still carries the marker itself; malformed metadata is
        # rejected by resume validation rather than hidden by observability.
        pass
    return result


def _evaluation_success_rate(evaluation: EvaluationResult | None) -> float | None:
    if evaluation is None:
        return None
    value = evaluation.metrics.get("success_rate")
    if not isinstance(value, int | float):
        return None
    return float(value)


def _evaluation_outcome_rows(
    evaluation: EvaluationResult,
) -> list[dict[str, Any]]:
    path_value = evaluation.artifacts.get("episode_outcomes_json")
    if not path_value:
        return []
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(
            f"Evaluation outcome artifact is missing before W&B logging: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported evaluation outcome artifact schema: {path}")
    rows = payload.get("episodes")
    if not isinstance(rows, list):
        raise ValueError(f"Evaluation outcome artifact has no episode rows: {path}")
    return [dict(row) for row in rows]


def _evaluation_evidence_summary(
    evaluation: EvaluationResult,
) -> dict[str, Any] | None:
    path_value = evaluation.artifacts.get("evaluation_evidence_json")
    if not path_value:
        return None
    path = Path(path_value)
    try:
        if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return {
        "schema_version": payload.get("schema_version"),
        "kind": payload.get("kind"),
        "config_fingerprint": payload.get("config_fingerprint"),
        "identity": payload.get("identity"),
        "outcomes": payload.get("outcomes"),
        "baseline_outcomes": payload.get("baseline_outcomes"),
    }


def _trajectory_metric_payload(
    prefix: str,
    trajectories: Sequence[EmbodiedTrajectory],
) -> dict[str, float | int]:
    """Aggregate user-facing episode metrics independently of the backend."""

    payload: dict[str, float | int] = {}
    successes = [
        bool(trajectory.metrics["success"])
        for trajectory in trajectories
        if "success" in trajectory.metrics
    ]
    if successes:
        payload[f"{prefix}/success_count"] = sum(successes)
        payload[f"{prefix}/success_rate"] = sum(successes) / len(successes)
        payload[f"{prefix}/success_denominator"] = len(successes)
    steps = [
        float(trajectory.metrics.get("episode_steps", len(trajectory.actions)))
        for trajectory in trajectories
    ]
    if steps:
        payload[f"{prefix}/episode_steps_mean"] = fmean(steps)
    durations = [
        float(trajectory.metrics["duration"])
        for trajectory in trajectories
        if "duration" in trajectory.metrics
    ]
    if durations:
        payload[f"{prefix}/duration_mean"] = fmean(durations)
    token_actions = [
        action
        for trajectory in trajectories
        for action in trajectory.actions
        if action.kind == "token"
    ]
    if token_actions:
        grammar_flags = [
            bool(action.metadata["action_grammar_valid"])
            for action in token_actions
            if "action_grammar_valid" in action.metadata
        ]
        if grammar_flags:
            payload[f"{prefix}/action_grammar_valid_rate"] = sum(grammar_flags) / len(
                grammar_flags
            )
            invalid_episodes = sum(
                any(
                    action.kind == "token"
                    and action.metadata.get(
                        "action_decode_valid",
                        action.metadata.get("action_grammar_valid"),
                    )
                    is False
                    for action in trajectory.actions
                )
                for trajectory in trajectories
            )
            payload[f"{prefix}/episodes_with_invalid_action_rate"] = (
                invalid_episodes / len(trajectories) if trajectories else 0.0
            )
        decode_flags = [
            bool(action.metadata["action_decode_valid"])
            for action in token_actions
            if "action_decode_valid" in action.metadata
        ]
        if decode_flags:
            payload[f"{prefix}/action_decode_valid_rate"] = sum(decode_flags) / len(
                decode_flags
            )
        rollout_logprob_flags = [
            bool(action.metadata["rollout_logprobs_computed"])
            for action in token_actions
            if "rollout_logprobs_computed" in action.metadata
        ]
        if rollout_logprob_flags:
            payload[f"{prefix}/rollout_logprobs_computed_rate"] = sum(
                rollout_logprob_flags
            ) / len(rollout_logprob_flags)
        for metadata_key, metric_name in (
            ("generated_token_count", "generated_token_count_mean"),
            (
                "post_termination_tokens_discarded",
                "post_termination_tokens_discarded_mean",
            ),
        ):
            values = [
                float(action.metadata[metadata_key])
                for action in token_actions
                if metadata_key in action.metadata
            ]
            if values:
                payload[f"{prefix}/{metric_name}"] = fmean(values)
    for metric_name in (
        "training_reward",
        "progress_reward",
        "shared_prefix_action_chunks",
        "shared_prefix_environment_steps",
        "trainable_suffix_action_chunks",
    ):
        values = [
            float(trajectory.metrics[metric_name])
            for trajectory in trajectories
            if metric_name in trajectory.metrics
        ]
        if values:
            payload[f"{prefix}/{metric_name}_mean"] = fmean(values)
    for metric_name in (
        "progress_grasp_reached",
        "progress_placement_reached",
        "progress_success_reached",
        "shared_prefix_branch_verified",
        "shared_prefix_completed_episode",
    ):
        values = [
            bool(trajectory.metrics[metric_name])
            for trajectory in trajectories
            if metric_name in trajectory.metrics
        ]
        if values:
            payload[f"{prefix}/{metric_name}_rate"] = sum(values) / len(values)
    return payload


def _representative_trajectories(
    trajectories: Sequence[EmbodiedTrajectory]
    | Sequence[tuple[Any, EmbodiedTrajectory]],
    *,
    limit: int,
    require_video: bool = True,
) -> list[tuple[Any, EmbodiedTrajectory]]:
    """Select bounded media rows while preserving both observed outcomes.

    Metrics alone rarely explain an embodied failure. When both successful and
    failed episodes exist, logging only the first trajectories can hide one
    outcome entirely. Selection is deterministic and task-agnostic: take the
    first success and failure, then fill remaining slots in source order.
    Trajectories without local video media are skipped before applying the
    limit, so configured media slots are not wasted.
    """

    if limit <= 0:
        return []
    indexed: list[tuple[Any, EmbodiedTrajectory]] = []
    for source_index, item in enumerate(trajectories):
        if isinstance(item, tuple):
            index, trajectory = item
        else:
            index, trajectory = source_index, item
        if not require_video or local_video_media_refs(trajectory):
            indexed.append((index, trajectory))
    if not require_video:
        indexed.sort(key=lambda item: not bool(local_video_media_refs(item[1])))

    selected: list[tuple[Any, EmbodiedTrajectory]] = []
    selected_indices: set[Any] = set()
    for outcome in (True, False):
        match = next(
            (
                item
                for item in indexed
                if "success" in item[1].metrics
                and bool(item[1].metrics["success"]) is outcome
            ),
            None,
        )
        if match is not None and match[0] not in selected_indices:
            selected.append(match)
            selected_indices.add(match[0])
            if len(selected) == limit:
                return selected

    for item in indexed:
        if item[0] in selected_indices:
            continue
        selected.append(item)
        selected_indices.add(item[0])
        if len(selected) == limit:
            break
    return selected


def _representative_groups(
    groups: Sequence[EmbodiedTrajectoryGroup],
    *,
    limit: int,
) -> list[tuple[int, EmbodiedTrajectoryGroup]]:
    """Prefer counterfactual groups while retaining stable source indices."""

    if limit <= 0:
        return []
    indexed = list(enumerate(groups))
    if not indexed:
        return []
    selected: list[tuple[int, EmbodiedTrajectoryGroup]] = []
    selected_indices: set[int] = set()

    def outcomes(group: EmbodiedTrajectoryGroup) -> set[bool]:
        return {
            bool(trajectory.metrics["success"])
            for trajectory in group
            if "success" in trajectory.metrics
        }

    first = min(
        indexed,
        key=lambda item: (
            not (
                _group_has_local_video(item[1]) and outcomes(item[1]) == {True, False}
            ),
            not _group_has_local_video(item[1]),
            outcomes(item[1]) != {True, False},
            item[0],
        ),
    )
    selected.append(first)
    selected_indices.add(first[0])

    while len(selected) < limit and len(selected_indices) < len(indexed):
        represented = set().union(*(outcomes(item[1]) for item in selected))
        missing_outcomes = {True, False}.difference(represented)
        candidates = [item for item in indexed if item[0] not in selected_indices]
        match = min(
            candidates,
            key=lambda item: (
                not bool(outcomes(item[1]).intersection(missing_outcomes)),
                not _group_has_local_video(item[1]),
                outcomes(item[1]) != {True, False},
                item[0],
            ),
        )
        selected.append(match)
        selected_indices.add(match[0])
    return selected


def _group_has_local_video(group: EmbodiedTrajectoryGroup) -> bool:
    return any(local_video_media_refs(trajectory) for trajectory in group)


def _configure_weave_server_cache(config: EmbodiedExperimentConfig) -> None:
    """Apply the YAML-owned Weave cache contract before importing the SDK."""

    weave = config.observability.weave
    os.environ["WEAVE_USE_SERVER_CACHE"] = "true" if weave.use_server_cache else "false"
    if not weave.use_server_cache:
        os.environ.pop("WEAVE_SERVER_CACHE_DIR", None)
        os.environ.pop("WEAVE_SERVER_CACHE_SIZE_LIMIT", None)
        return
    assert weave.server_cache_dir is not None
    cache_dir = weave.server_cache_dir.expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WEAVE_SERVER_CACHE_DIR"] = str(cache_dir.resolve())
    os.environ["WEAVE_SERVER_CACHE_SIZE_LIMIT"] = str(
        int(weave.server_cache_size_mb) * 1024 * 1024
    )


def _weave_step_timeline(trajectory: EmbodiedTrajectory) -> list[dict[str, Any]]:
    """Build a compact chronological trace without embedding image tensors."""

    step_ids = {
        item.step
        for collection in (
            trajectory.observations,
            trajectory.actions,
            trajectory.rewards,
            trajectory.tool_calls,
        )
        for item in collection
        if item.step is not None
    }
    timeline: list[dict[str, Any]] = []
    for step in sorted(step_ids):
        timeline.append(
            {
                "step": step,
                "observations": [
                    {
                        "kind": observation.kind,
                        "value": _compact_trace_value(observation.value),
                        "metadata": _compact_trace_metadata(observation.metadata),
                        "media": [
                            media.model_dump(mode="json") for media in observation.media
                        ],
                    }
                    for observation in trajectory.observations
                    if observation.step == step
                ],
                "actions": [
                    {
                        "kind": action.kind,
                        "decoded": _compact_trace_value(action.decoded),
                        "metadata": _compact_trace_metadata(action.metadata),
                    }
                    for action in trajectory.actions
                    if action.step == step
                ],
                "rewards": [
                    reward.model_dump(mode="json")
                    for reward in trajectory.rewards
                    if reward.step == step
                ],
                "tool_calls": [
                    tool.model_dump(mode="json")
                    for tool in trajectory.tool_calls
                    if tool.step == step
                ],
            }
        )
    return timeline


def _compact_trace_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    shape = getattr(value, "shape", None)
    if shape is not None:
        return {
            "type": type(value).__name__,
            "shape": [int(dimension) for dimension in shape],
            "dtype": str(getattr(value, "dtype", "unknown")),
        }
    if isinstance(value, dict):
        return {
            str(key): _compact_trace_value(item)
            for key, item in list(value.items())[:32]
        }
    if isinstance(value, list | tuple):
        if len(value) <= 32:
            return [_compact_trace_value(item) for item in value]
        return {
            "type": type(value).__name__,
            "length": len(value),
            "sample": [_compact_trace_value(item) for item in value[:8]],
        }
    return {"type": type(value).__name__, "repr": repr(value)[:256]}


def _compact_trace_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Exclude optimizer-only payloads while preserving human trace context."""

    return {
        str(key): _compact_trace_value(value)
        for key, value in metadata.items()
        if not str(key).startswith("_art_embodied_transient_")
    }


def _append_bounded_jsonl(
    path: Path,
    record: dict[str, Any],
    *,
    max_bytes: int,
) -> None:
    """Append one diagnostic while bounding coordinator-local telemetry state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    if len(encoded) > max_bytes:
        compact = dict(record)
        compact["message"] = str(compact.get("message", ""))[:1024]
        compact["record_truncated"] = True
        encoded = (json.dumps(compact, sort_keys=True) + "\n").encode("utf-8")
    current_size = path.stat().st_size if path.exists() else 0
    if current_size + len(encoded) > max_bytes:
        replacement = path.with_name(path.name + ".tmp")
        replacement.write_bytes(encoded[-max_bytes:])
        replacement.replace(path)
        return
    with path.open("ab") as handle:
        handle.write(encoded)
