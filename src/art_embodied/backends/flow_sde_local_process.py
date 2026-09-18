"""Local multi-GPU gradient handoff for PI Flow-SDE GRPO."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import math
from pathlib import Path
import pickle
import shutil
import subprocess
import tempfile
import time
from typing import Any, TextIO

import torch

from art_embodied.checkpointing import CheckpointManager
from art_embodied.config import EmbodiedExperimentConfig, GR00TN17FlowLoadConfig
from art_embodied.trajectories import EmbodiedTrajectoryGroup
from art_embodied.types import LocalTrainResult, TrainResult
from art_embodied.utils import worker_python_command, write_json_atomic

from .action_token_gradients import (
    apply_action_token_gradient_payloads,
    load_action_token_gradient_payload,
)
from .flow_sde import (
    FlowSDEExample,
    FlowSDEGRPOBackend,
    _collate_flow_sde_audit_rows,
    _flow_sde_kl_guard_scope,
    _flow_sde_max_approximate_kl,
    _flow_sde_subupdates,
    _should_stop_flow_sde_before_optimizer,
    flow_sde_replay_selection_counts,
    precalculate_flow_sde_logprobs,
    prepare_flow_sde_examples,
)
from .flow_sde_progress import (
    emit_flow_sde_alignment,
    emit_flow_sde_training_progress,
    emit_flow_sde_trust_region_stop,
)
from .local_process import (
    _guard_file_size,
    _move_policy_model,
    _save_policy_snapshot,
    _stop_process,
    _worker_environment,
)


@dataclass(slots=True)
class _FlowWorker:
    index: int
    device: str
    process: subprocess.Popen[str]
    stdin: TextIO
    stdout: TextIO
    stderr: TextIO
    directory: Path
    ready_path: Path


class LocalProcessFlowSDEBackend:
    """Shard Flow-SDE rescoring while keeping one authoritative optimizer."""

    def __init__(
        self,
        *,
        config: EmbodiedExperimentConfig,
        policy: Any,
        backend: FlowSDEGRPOBackend,
    ) -> None:
        if not config.runtime.distributed_training:
            raise ValueError(
                "LocalProcessFlowSDEBackend requires distributed_training=true"
            )
        self.config = config
        self.policy = policy
        self.backend = backend
        self.update_step = 0
        self._persistent_worker_pool: _FlowGradientWorkerPool | None = None
        if config.storage.resume_from_checkpoint is not None:
            self._restore_checkpoint(config.storage.resume_from_checkpoint)

    def _restore_checkpoint(self, path: Path) -> None:
        """Restore the coordinator policy, Adam state, and update cursor.

        Gradient workers are intentionally not restored. They are disposable
        replicas and receive an authoritative coordinator snapshot before the
        next subupdate.
        """

        CheckpointManager().validate(
            path,
            expected_resume_contract_fingerprint=(
                self.config.resume_contract_fingerprint
            ),
            allow_legacy=self.config.storage.allow_legacy_checkpoint_resume,
            require_training_state=True,
        )
        policy_path = Path(path) / "policy"
        load = getattr(self.policy, "load_checkpoint", None)
        if not callable(load):
            raise TypeError("Flow-SDE resume requires policy.load_checkpoint(path)")
        load(policy_path)
        _move_policy_model(self.policy, "cpu")

        state_path = Path(path) / "art_embodied_training_state.pt"
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            raise ValueError(f"Invalid Flow-SDE training state: {state_path}")
        step = state.get("step")
        optimizer_state = state.get("optimizer")
        if not isinstance(step, int) or not isinstance(optimizer_state, dict):
            raise ValueError(
                "Flow-SDE training state must contain integer step and optimizer"
            )
        self.backend.optimizer.load_state_dict(optimizer_state)
        self.update_step = step
        self.backend.step = step
        if self.update_step >= self.config.training.updates:
            raise ValueError(
                "Resumed update must be smaller than training.updates: "
                f"resumed={self.update_step}, "
                f"target={self.config.training.updates}"
            )

    def save_snapshot(self, path: Path, *, update: int) -> None:
        if int(update) != self.update_step:
            raise ValueError(
                "Rollout snapshot update does not match Flow-SDE backend state: "
                f"requested={update}, backend={self.update_step}"
            )
        _save_policy_snapshot(self.policy, path)

    def offload(self) -> None:
        _move_policy_model(self.policy, "cpu")

    def restore(self, device: str) -> None:
        _move_policy_model(self.policy, device)

    async def train(
        self,
        trajectory_groups: list[EmbodiedTrajectoryGroup],
        **kwargs: Any,
    ) -> TrainResult:
        if kwargs:
            raise TypeError("Distributed Flow-SDE training accepts YAML config only")
        return await asyncio.to_thread(self._train_sync, list(trajectory_groups))

    def _train_sync(self, groups: list[EmbodiedTrajectoryGroup]) -> LocalTrainResult:
        replay_eligible, replay_selected = flow_sde_replay_selection_counts(groups)
        examples, advantages, kept_groups = prepare_flow_sde_examples(
            groups,
            group_size=self.backend.group_size,
            advantage_epsilon=self.backend.advantage_epsilon,
            advantage_std_unbiased=self.backend.advantage_std_unbiased,
            filter_rewards=self.backend.filter_rewards,
            rewards_lower_bound=self.backend.rewards_lower_bound,
            rewards_upper_bound=self.backend.rewards_upper_bound,
        )
        subupdates = _flow_sde_subupdates(
            examples,
            advantages,
            group_count=len(groups),
            group_size=self.backend.group_size,
            optimizer_steps_per_update=self.backend.optimizer_steps_per_update,
            schedule=self.backend.training_schedule,
            max_episode_steps=self.backend.max_episode_steps,
        )
        started = time.perf_counter()
        rescore_metrics: dict[str, float] = {}
        if self.backend.precalculate_logprobs:
            self.policy.train()
            examples, subupdates, rescore_metrics = (
                _precalculate_distributed_flow_sde_logprobs(
                    self.policy,
                    examples,
                    subupdates,
                    workers=len(self.config.runtime.training_devices),
                    microbatch_size=self.backend.microbatch_size,
                    device=self.backend.device,
                    task_keys=_task_gradient_keys(self.config),
                )
            )
        reuse_workers = self.config.runtime.training_worker_lifecycle == "cpu_offload"
        if reuse_workers:
            if self._persistent_worker_pool is None:
                self._persistent_worker_pool = _FlowGradientWorkerPool(
                    config=self.config,
                    policy=self.policy,
                    update_step=self.update_step,
                )
            pool = self._persistent_worker_pool
        else:
            pool = _FlowGradientWorkerPool(
                config=self.config,
                policy=self.policy,
                update_step=self.update_step,
            )
        pool.begin_update(update_step=self.update_step)
        metrics_by_subupdate: list[dict[str, float]] = []
        trust_region_stop: dict[str, Any] | None = None
        subupdate_audits: list[dict[str, Any]] | None = (
            [] if self.backend.diagnostics_dir is not None else None
        )
        sft_replay_samples: list[dict[str, Any]] = []
        try:
            for index, (
                sub_examples,
                sub_advantages,
                denominator,
                length_normalized,
            ) in enumerate(subupdates):
                payload_cache_key, stage_payload = _flow_payload_cache_plan(
                    subupdate_index=index,
                    subupdate_count=len(subupdates),
                    update_epochs=int(
                        getattr(self.backend.training_schedule, "update_epochs", 1)
                    ),
                )
                results = pool.compute(
                    sub_examples,
                    sub_advantages,
                    subupdate_index=index,
                    loss_denominator=denominator,
                    length_normalized=length_normalized,
                    backend=self.backend,
                    payload_cache_key=payload_cache_key,
                    stage_payload=stage_payload,
                )
                if index == 0:
                    joint_alignment = _weighted_worker_metric(
                        results,
                        "active_previous_abs_delta_mean",
                        weight="valid_rows",
                    )
                    joint_alignment_max = max(
                        result["metrics"]["active_previous_abs_delta_max"]
                        for result in results
                    )
                    alignment = _weighted_worker_metric(
                        results,
                        "active_previous_abs_delta_per_primitive_mean",
                        weight="valid_rows",
                    )
                    alignment_max = max(
                        result["metrics"]["active_previous_abs_delta_per_primitive_max"]
                        for result in results
                    )
                    alignment_ratio = _weighted_worker_metric(
                        results,
                        "active_previous_ratio_mean",
                        weight="valid_rows",
                        empty_value=1.0,
                    )
                    tolerance = self.config.algorithm.pre_update_logprob_kl_tolerance
                    ratio_tolerance = self.config.algorithm.pre_update_ratio_tolerance
                    if tolerance is not None or ratio_tolerance is not None:
                        passed = (
                            math.isfinite(alignment)
                            and math.isfinite(alignment_ratio)
                            and (tolerance is None or alignment <= tolerance)
                            and (
                                ratio_tolerance is None
                                or abs(alignment_ratio - 1.0) <= ratio_tolerance
                            )
                        )
                        emit_flow_sde_alignment(
                            update=self.update_step + 1,
                            mean_abs_delta=alignment,
                            max_abs_delta=alignment_max,
                            tolerance=(
                                tolerance if tolerance is not None else float("inf")
                            ),
                            ratio_mean=alignment_ratio,
                            ratio_tolerance=ratio_tolerance,
                            passed=passed,
                        )
                        if not passed:
                            raise RuntimeError(
                                "Distributed Flow-SDE rollout/rescore alignment "
                                "failed before optimizer step: "
                                "mean_abs_delta_per_primitive="
                                f"{alignment:.6g}, "
                                f"joint_mean_abs_delta={joint_alignment:.6g}, "
                                f"joint_max_abs_delta={joint_alignment_max:.6g}, "
                                f"tolerance={tolerance}, "
                                f"ratio_mean={alignment_ratio:.6g}, "
                                f"ratio_tolerance={ratio_tolerance}"
                            )
                probe_approximate_kl = _weighted_worker_metric(
                    results, "approximate_kl", weight="valid_rows"
                )
                probe_approximate_kl_per_primitive = _weighted_worker_metric(
                    results, "approximate_kl_per_primitive", weight="valid_rows"
                )
                if _should_stop_flow_sde_before_optimizer(
                    self.backend.training_schedule,
                    subupdate_index=index,
                    approximate_kl=probe_approximate_kl,
                    approximate_kl_per_primitive=(probe_approximate_kl_per_primitive),
                ):
                    threshold = _flow_sde_max_approximate_kl(
                        self.backend.training_schedule
                    )
                    assert threshold is not None
                    guard_scope = _flow_sde_kl_guard_scope(
                        self.backend.training_schedule
                    )
                    trust_region_stop = {
                        "approximate_kl": probe_approximate_kl,
                        "approximate_kl_per_primitive": (
                            probe_approximate_kl_per_primitive
                        ),
                        "scope": guard_scope,
                        "before_subupdate": float(index + 1),
                    }
                    emit_flow_sde_trust_region_stop(
                        update=self.update_step + 1,
                        applied_subupdates=len(metrics_by_subupdate),
                        planned_subupdates=len(subupdates),
                        approximate_kl=probe_approximate_kl,
                        approximate_kl_per_primitive=(
                            probe_approximate_kl_per_primitive
                        ),
                        scope=guard_scope,
                        threshold=threshold,
                    )
                    break
                apply_metrics = apply_action_token_gradient_payloads(
                    self.policy,
                    self.backend.optimizer,
                    [result["gradient_payload"] for result in results],
                    max_grad_norm=self.backend.max_grad_norm,
                    skip_optimizer_step_without_policy_gradient_signal=(
                        self.config.algorithm.skip_optimizer_step_without_policy_gradient_signal
                    ),
                    prefix="embodied_flow_sde_grpo",
                    gradient_aggregation=(_gradient_aggregation(self.config)),
                )
                metrics_by_subupdate.append(
                    _aggregate_flow_worker_metrics(
                        results,
                        apply_metrics=apply_metrics,
                        subupdate_index=index,
                    )
                )
                if self.backend.sft_replay_coefficient > 0.0:
                    for result in results:
                        sample = result.get("sft_replay_sample")
                        if not isinstance(sample, dict):
                            raise RuntimeError(
                                "Positive SFT replay did not return its sample audit"
                            )
                        sft_replay_samples.append(
                            {
                                "subupdate_index": index,
                                "worker_index": int(result["worker_index"]),
                                **sample,
                            }
                        )
                emit_flow_sde_training_progress(
                    update=self.update_step + 1,
                    completed=index + 1,
                    total=len(subupdates),
                    metrics=metrics_by_subupdate[-1],
                    started_at=started,
                )
                if subupdate_audits is not None:
                    audit_rows = [
                        result["audit_row"]
                        for result in results
                        if result.get("audit_row") is not None
                    ]
                    subupdate_audits.append(
                        _collate_flow_sde_audit_rows(
                            audit_rows,
                            subupdate_index=index,
                            loss_denominator=denominator,
                            length_normalized=length_normalized,
                            metrics=metrics_by_subupdate[-1],
                        )
                    )
                    subupdate_audits[-1]["worker_metrics"] = [
                        {
                            "worker_index": int(result["worker_index"]),
                            "rows": int(result["metrics"]["rows"]),
                            "valid_rows": int(result["metrics"]["valid_rows"]),
                            "loss": float(result["metrics"]["loss"]),
                            "ratio_mean": float(result["metrics"]["ratio_mean"]),
                            "approximate_kl": float(
                                result["metrics"]["approximate_kl"]
                            ),
                            "approximate_kl_per_primitive": float(
                                result["metrics"]["approximate_kl_per_primitive"]
                            ),
                            "clip_fraction": float(result["metrics"]["clip_fraction"]),
                            "microbatch_losses": result["audit_row"][
                                "microbatch_losses"
                            ].tolist(),
                            "row_weights": result["audit_row"]["row_weights"],
                        }
                        for result in results
                    ]
        finally:
            if reuse_workers:
                pool.finish_update()
            else:
                pool.close()

        if subupdate_audits is not None:
            self.backend._write_batch_audit(
                groups,
                examples=examples,
                advantages=advantages,
                kept_groups=kept_groups,
                subupdates=subupdate_audits,
            )
        if self.backend.sft_replay_coefficient > 0.0:
            expected_samples = len(metrics_by_subupdate) * len(
                self.config.runtime.training_devices
            )
            if len(sft_replay_samples) != expected_samples:
                raise RuntimeError(
                    "SFT replay audit coverage differs from applied gradients: "
                    f"samples={len(sft_replay_samples)}, expected={expected_samples}"
                )
            write_json_atomic(
                self.config.storage.output_dir
                / "sft_replay_audits"
                / f"update_{self.update_step + 1:06d}.json",
                {
                    "schema_version": 1,
                    "kind": "art_embodied_gr00t_n1d7_sft_replay_update_audit",
                    "training_update_index": self.update_step,
                    "checkpoint_update": self.update_step + 1,
                    "coefficient": self.backend.sft_replay_coefficient,
                    "applied_subupdates": len(metrics_by_subupdate),
                    "training_workers": len(self.config.runtime.training_devices),
                    "samples": sft_replay_samples,
                },
                indent=2,
                sort_keys=True,
            )
        self.update_step += 1
        self.backend.step = self.update_step
        checkpoint = self.backend._save_checkpoint()
        metrics = _aggregate_flow_subupdates(metrics_by_subupdate)
        metrics.update(
            {
                f"embodied_flow_sde_grpo/{key}": value
                for key, value in rescore_metrics.items()
            }
        )
        metrics.update(
            {
                "embodied_flow_sde_grpo/groups": float(len(groups)),
                "embodied_flow_sde_grpo/groups_kept": float(kept_groups),
                "embodied_flow_sde_grpo/examples": float(len(examples)),
                "embodied_flow_sde_grpo/replay_actions_eligible": float(
                    replay_eligible
                ),
                "embodied_flow_sde_grpo/replay_actions_selected": float(
                    replay_selected
                ),
                "embodied_flow_sde_grpo/replay_selection_fraction": (
                    replay_selected / replay_eligible
                ),
                "embodied_flow_sde_grpo/optimizer_subupdates": float(
                    len(metrics_by_subupdate)
                ),
                "embodied_flow_sde_grpo/optimizer_subupdates_planned": float(
                    len(subupdates)
                ),
                "embodied_flow_sde_grpo/trust_region_early_stop": float(
                    trust_region_stop is not None
                ),
                "embodied_flow_sde_grpo/trust_region_stop_approximate_kl": (
                    trust_region_stop["approximate_kl"]
                    if trust_region_stop is not None
                    else 0.0
                ),
                "embodied_flow_sde_grpo/trust_region_stop_abs_approximate_kl": (
                    abs(trust_region_stop["approximate_kl"])
                    if trust_region_stop is not None
                    else 0.0
                ),
                "embodied_flow_sde_grpo/trust_region_stop_approximate_kl_per_primitive": (
                    trust_region_stop["approximate_kl_per_primitive"]
                    if trust_region_stop is not None
                    else 0.0
                ),
                "embodied_flow_sde_grpo/trust_region_stop_before_subupdate": (
                    trust_region_stop["before_subupdate"]
                    if trust_region_stop is not None
                    else 0.0
                ),
                "embodied_flow_sde_grpo/distributed_training": 1.0,
                "embodied_flow_sde_grpo/training_workers": float(
                    len(self.config.runtime.training_devices)
                ),
                "embodied_flow_sde_grpo/training_worker_reused": float(
                    pool.reused_workers
                ),
                "embodied_flow_sde_grpo/worker_startup_seconds": float(
                    pool.last_startup_seconds
                ),
                "embodied_flow_sde_grpo/worker_offload_seconds": float(
                    pool.last_offload_seconds
                ),
                "embodied_flow_sde_grpo/distributed_elapsed_seconds": float(
                    time.perf_counter() - started
                ),
            }
        )
        return LocalTrainResult(
            step=self.update_step,
            metrics=metrics,
            checkpoint_path=str(checkpoint),
        )

    async def close(self) -> None:
        if self._persistent_worker_pool is not None:
            self._persistent_worker_pool.close()
            self._persistent_worker_pool = None
        await self.backend.close()


class _FlowGradientWorkerPool:
    def __init__(
        self,
        *,
        config: EmbodiedExperimentConfig,
        policy: Any,
        update_step: int,
    ) -> None:
        self.config = config
        self.policy = policy
        self.update_step = update_step
        self.run_dir: Path | None = None
        self.initial_snapshot: Path | None = None
        self.workers: list[_FlowWorker] = []
        self.offloaded = False
        self.reused_workers = False
        self.last_startup_seconds = 0.0
        self.last_offload_seconds = 0.0

    def begin_update(self, *, update_step: int) -> None:
        if self.workers and not self.offloaded:
            raise RuntimeError("Flow-SDE training workers are already active")
        started = time.perf_counter()
        self.update_step = int(update_step)
        self.reused_workers = bool(self.workers)
        if self.run_dir is None:
            root = self.config.runtime.worker_handoff_dir.expanduser()
            root.mkdir(parents=True, exist_ok=True)
            self.run_dir = Path(tempfile.mkdtemp(prefix="flow-training-", dir=root))
        self.initial_snapshot = (
            self.run_dir / "snapshots" / f"update-{self.update_step:04d}" / "initial"
        )
        _save_policy_snapshot(self.policy, self.initial_snapshot)
        _move_policy_model(self.policy, "cpu")
        try:
            if not self.workers:
                for index, device in enumerate(self.config.runtime.training_devices):
                    self.workers.append(self._start_worker(index=index, device=device))
                self._wait_ready()
            else:
                self._restore_workers()
            self.offloaded = False
        except Exception:
            self.close()
            raise
        self.last_startup_seconds = time.perf_counter() - started

    def _start_worker(self, *, index: int, device: str) -> _FlowWorker:
        assert self.run_dir is not None
        assert self.initial_snapshot is not None
        directory = self.run_dir / f"worker-{index:02d}"
        directory.mkdir(parents=True)
        ready_path = directory / "ready.json"
        spec_path = directory / "bootstrap.json"
        spec_path.write_text(
            json.dumps(
                {
                    "config": self.config.model_dump(mode="json"),
                    "policy_snapshot": str(self.initial_snapshot),
                    "ready_path": str(ready_path),
                    "worker_index": index,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        stdout = (directory / "stdout.log").open("w", encoding="utf-8")
        stderr = (directory / "stderr.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            worker_python_command(
                configured_executable=self.config.runtime.worker_python_executable,
                module="art_embodied.backends.flow_sde_worker",
                spec_path=spec_path,
            ),
            cwd=str(Path.cwd()),
            env=_worker_environment(device),
            text=True,
            stdin=subprocess.PIPE,
            stdout=stdout,
            stderr=stderr,
            bufsize=1,
        )
        if process.stdin is None:
            _stop_process(process)
            raise RuntimeError("Flow-SDE worker has no stdin pipe")
        return _FlowWorker(
            index=index,
            device=device,
            process=process,
            stdin=process.stdin,
            stdout=stdout,
            stderr=stderr,
            directory=directory,
            ready_path=ready_path,
        )

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.config.runtime.worker_timeout_seconds
        pending = {worker.index: worker for worker in self.workers}
        while pending:
            for index, worker in list(pending.items()):
                if worker.ready_path.exists():
                    payload = json.loads(worker.ready_path.read_text(encoding="utf-8"))
                    if not payload.get("ok"):
                        raise RuntimeError(f"Flow-SDE worker startup failed: {payload}")
                    pending.pop(index)
                elif worker.process.poll() is not None:
                    raise RuntimeError(self._worker_failure(worker))
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Flow-SDE workers timed out: {sorted(pending)}")
            time.sleep(0.1)

    def compute(
        self,
        examples: list[FlowSDEExample],
        advantages: list[float],
        *,
        subupdate_index: int,
        loss_denominator: int,
        length_normalized: bool,
        backend: FlowSDEGRPOBackend,
        payload_cache_key: str | None = None,
        stage_payload: bool = True,
    ) -> list[dict[str, Any]]:
        if self.offloaded or not self.workers:
            raise RuntimeError("Flow-SDE workers are not active for this update")
        assert self.run_dir is not None
        assert self.initial_snapshot is not None
        if subupdate_index == 0:
            snapshot = self.initial_snapshot
        else:
            snapshot = (
                self.run_dir
                / "snapshots"
                / f"update-{self.update_step:04d}"
                / f"subupdate-{subupdate_index:04d}"
            )
            _save_policy_snapshot(self.policy, snapshot)
        task_keys = _task_gradient_keys(self.config)
        partitions = _partition_flow_examples(
            examples,
            advantages,
            workers=len(self.workers),
            task_keys=task_keys,
        )
        jobs = []
        for worker, (worker_examples, worker_advantages) in zip(
            self.workers, partitions, strict=True
        ):
            if not worker_examples:
                continue
            directory = (
                worker.directory
                / f"update-{self.update_step:04d}"
                / f"subupdate-{subupdate_index:04d}"
            )
            directory.mkdir(parents=True)
            if payload_cache_key is None:
                examples_path = directory / "examples.pkl"
            else:
                payload_directory = (
                    worker.directory
                    / f"update-{self.update_step:04d}"
                    / "payload-cache"
                )
                payload_directory.mkdir(parents=True, exist_ok=True)
                examples_path = payload_directory / f"{payload_cache_key}.pkl"
            if stage_payload:
                with examples_path.open("wb") as handle:
                    pickle.dump(
                        {
                            "examples": worker_examples,
                            "advantages": worker_advantages,
                        },
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                _guard_file_size(
                    examples_path, max_mb=self.config.runtime.max_worker_handoff_mb
                )
            task_key = None if task_keys is None else task_keys[worker.index]
            if task_key is not None and {
                example.task_key for example in worker_examples
            } != {task_key}:
                raise RuntimeError(
                    "task_pcgrad worker partition is not task-owned: "
                    f"worker={worker.index}, expected={task_key!r}"
                )
            gradient_path = directory / "gradients.pt"
            audit_path = directory / "audit.pt"
            result_path = directory / "result.json"
            command = {
                "op": "gradient",
                "worker_index": worker.index,
                "policy_snapshot": str(snapshot),
                "gradient_path": str(gradient_path),
                "result_path": str(result_path),
                "microbatch_size": backend.microbatch_size,
                "loss_denominator": loss_denominator,
                "length_normalized": length_normalized,
                "max_episode_steps": backend.max_episode_steps,
                "clip_epsilon_low": backend.clip_epsilon_low,
                "clip_epsilon_high": backend.clip_epsilon_high,
                "clip_ratio_c": backend.clip_ratio_c,
                "reference_kl_coefficient": backend.reference_kl_coefficient,
                "sft_replay_coefficient": backend.sft_replay_coefficient,
                "worker_count": len(self.workers),
                "update_index": self.update_step,
                "subupdate_index": subupdate_index,
            }
            command["examples_path"] = str(examples_path)
            if payload_cache_key is not None:
                command["payload_cache_key"] = payload_cache_key
                command["payload_cache_hit"] = not stage_payload
            if backend.diagnostics_dir is not None:
                command["audit_path"] = str(audit_path)
            worker.stdin.write(json.dumps(command, sort_keys=True) + "\n")
            worker.stdin.flush()
            jobs.append((worker, result_path, gradient_path))
        if (backend.sft_replay_coefficient > 0.0 or task_keys is not None) and len(
            jobs
        ) != len(self.workers):
            raise RuntimeError(
                "Task-owned training requires every gradient worker in every "
                f"subupdate: active={len(jobs)}, expected={len(self.workers)}"
            )
        self._wait_jobs(jobs)
        results = []
        for worker, result_path, gradient_path in jobs:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if not payload.get("ok"):
                raise RuntimeError(
                    "Flow-SDE gradient worker failed:\n"
                    + str(payload.get("traceback") or payload)
                )
            _guard_file_size(
                gradient_path, max_mb=self.config.runtime.max_worker_handoff_mb
            )
            audit_row = None
            if backend.diagnostics_dir is not None:
                audit_row = _load_flow_worker_audit(
                    result_path,
                    max_mb=self.config.runtime.max_worker_handoff_mb,
                )
            gradient_payload = load_action_token_gradient_payload(gradient_path)
            if task_keys is not None:
                gradient_payload["task_key"] = task_keys[worker.index]
            results.append(
                {
                    "worker_index": worker.index,
                    "metrics": {
                        str(key): float(value)
                        for key, value in payload["metrics"].items()
                    },
                    "elapsed_seconds": float(payload["elapsed_seconds"]),
                    "adapter_refreshed": bool(payload.get("adapter_refreshed")),
                    "payload_cache_hit": bool(payload.get("payload_cache_hit")),
                    "gradient_payload": gradient_payload,
                    "audit_row": audit_row,
                    "sft_replay_sample": payload.get("sft_replay_sample"),
                }
            )
        if not self.config.runtime.keep_worker_handoffs:
            _cleanup_flow_subupdate_artifacts(
                result_paths=[result_path for _, result_path, _ in jobs],
                snapshot=snapshot,
                initial_snapshot=self.initial_snapshot,
            )
        return results

    def _restore_workers(self) -> None:
        assert self.initial_snapshot is not None
        jobs: list[tuple[_FlowWorker, Path, Path]] = []
        for worker in self.workers:
            result_path = (
                worker.directory / f"update-{self.update_step:04d}-restore.json"
            )
            worker.stdin.write(
                json.dumps(
                    {
                        "op": "restore",
                        "policy_snapshot": str(self.initial_snapshot),
                        "result_path": str(result_path),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            worker.stdin.flush()
            jobs.append((worker, result_path, result_path))
        self._wait_jobs(jobs)
        self._require_control_jobs_ok(jobs, operation="restore")

    def finish_update(self) -> None:
        if self.offloaded or not self.workers:
            return
        started = time.perf_counter()
        jobs: list[tuple[_FlowWorker, Path, Path]] = []
        for worker in self.workers:
            result_path = (
                worker.directory / f"update-{self.update_step:04d}-offload.json"
            )
            worker.stdin.write(
                json.dumps({"op": "offload", "result_path": str(result_path)}) + "\n"
            )
            worker.stdin.flush()
            jobs.append((worker, result_path, result_path))
        self._wait_jobs(jobs)
        self._require_control_jobs_ok(jobs, operation="offload")
        self.offloaded = True
        self.last_offload_seconds = time.perf_counter() - started
        if not self.config.runtime.keep_worker_handoffs:
            self._prune_completed_update_handoffs()

    @staticmethod
    def _require_control_jobs_ok(
        jobs: list[tuple[_FlowWorker, Path, Path]], *, operation: str
    ) -> None:
        for worker, result_path, _unused in jobs:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if not payload.get("ok"):
                raise RuntimeError(
                    f"Flow-SDE worker {worker.index} failed to {operation}: "
                    f"{payload.get('traceback') or payload}"
                )

    def _prune_completed_update_handoffs(self) -> None:
        assert self.run_dir is not None
        shutil.rmtree(
            self.run_dir / "snapshots" / f"update-{self.update_step:04d}",
            ignore_errors=True,
        )
        for worker in self.workers:
            shutil.rmtree(
                worker.directory / f"update-{self.update_step:04d}",
                ignore_errors=True,
            )
            for suffix in ("restore", "offload"):
                result_path = (
                    worker.directory / f"update-{self.update_step:04d}-{suffix}.json"
                )
                result_path.unlink(missing_ok=True)

    def _wait_jobs(self, jobs: list[tuple[_FlowWorker, Path, Path]]) -> None:
        deadline = time.monotonic() + self.config.runtime.worker_timeout_seconds
        pending = {worker.index: (worker, result) for worker, result, _ in jobs}
        while pending:
            for index, (worker, result) in list(pending.items()):
                if result.exists():
                    pending.pop(index)
                elif worker.process.poll() is not None:
                    raise RuntimeError(self._worker_failure(worker))
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Flow-SDE gradient jobs timed out: {sorted(pending)}"
                )
            time.sleep(0.1)

    @staticmethod
    def _worker_failure(worker: _FlowWorker) -> str:
        path = worker.directory / "stderr.log"
        tail = path.read_text(encoding="utf-8", errors="replace")[-4000:]
        return f"worker {worker.index} on {worker.device} failed:\n{tail}"

    def close(self) -> None:
        for worker in self.workers:
            if worker.process.poll() is None:
                try:
                    worker.stdin.write('{"op":"shutdown"}\n')
                    worker.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
        for worker in self.workers:
            if worker.process.poll() is None:
                try:
                    worker.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    _stop_process(worker.process)
            worker.stdin.close()
            worker.stdout.close()
            worker.stderr.close()
        self.workers.clear()
        self.offloaded = False
        if self.run_dir is not None and not self.config.runtime.keep_worker_handoffs:
            shutil.rmtree(self.run_dir, ignore_errors=True)


def _load_flow_worker_audit(result_path: Path, *, max_mb: int) -> dict[str, Any]:
    audit_path = result_path.with_name("audit.pt")
    if not audit_path.is_file():
        raise RuntimeError(
            "Flow-SDE worker did not produce its required retained audit payload: "
            f"{audit_path}"
        )
    _guard_file_size(audit_path, max_mb=max_mb)
    return torch.load(audit_path, map_location="cpu", weights_only=True)


def _cleanup_flow_subupdate_artifacts(
    *,
    result_paths: list[Path],
    snapshot: Path,
    initial_snapshot: Path,
) -> None:
    """Remove consumed multi-GB handoffs while preserving explicit audits.

    Gradient and audit tensors have already been loaded into coordinator memory
    before this function runs. Persistent workers have also completed loading
    ``snapshot``, so only the initial bootstrap snapshot must survive until the
    worker pool closes.
    """

    for result_path in result_paths:
        shutil.rmtree(result_path.parent, ignore_errors=True)
    if snapshot != initial_snapshot:
        shutil.rmtree(snapshot, ignore_errors=True)


def _partition_flow_examples(
    examples: list[FlowSDEExample],
    advantages: list[float],
    *,
    workers: int,
    task_keys: list[str] | None = None,
) -> list[tuple[list[FlowSDEExample], list[float]]]:
    if len(examples) != len(advantages):
        raise ValueError("Flow-SDE examples and advantages do not align")
    if workers < 1:
        raise ValueError("Flow-SDE partition requires at least one worker")
    if task_keys is not None:
        if len(task_keys) != workers or len(set(task_keys)) != len(task_keys):
            raise ValueError(
                "Task-owned Flow-SDE partition requires one unique task per worker"
            )
        task_to_worker = {task_key: index for index, task_key in enumerate(task_keys)}
    else:
        task_to_worker = None
    partitions = [([], []) for _ in range(workers)]
    for index, (example, advantage) in enumerate(
        zip(examples, advantages, strict=True)
    ):
        if task_to_worker is None:
            target = index % workers
        else:
            try:
                target = task_to_worker[example.task_key]
            except KeyError as exc:
                raise ValueError(
                    "Flow-SDE example has no configured task_pcgrad worker: "
                    f"{example.task_key!r}"
                ) from exc
        partitions[target][0].append(example)
        partitions[target][1].append(float(advantage))
    return partitions


def _task_gradient_keys(config: EmbodiedExperimentConfig) -> list[str] | None:
    aggregation = _gradient_aggregation(config)
    if aggregation == "sum":
        return None
    if aggregation != "task_pcgrad":
        raise ValueError(f"Unsupported gradient aggregation: {aggregation!r}")
    raw_tasks = config.environment.kwargs.get("task_ids")
    if not isinstance(raw_tasks, list):
        raise ValueError("task_pcgrad requires environment.kwargs.task_ids")
    return [str(value) for value in raw_tasks]


def _gradient_aggregation(config: EmbodiedExperimentConfig) -> str:
    if config.policy.type != "gr00t_n1d7":
        return "sum"
    return GR00TN17FlowLoadConfig.model_validate(
        config.policy.load_kwargs
    ).gradient_aggregation


def _precalculate_distributed_flow_sde_logprobs(
    policy: Any,
    examples: list[FlowSDEExample],
    subupdates: list[tuple[list[FlowSDEExample], list[float], int, bool]],
    *,
    workers: int,
    microbatch_size: int,
    device: str,
    task_keys: list[str] | None = None,
) -> tuple[
    list[FlowSDEExample],
    list[tuple[list[FlowSDEExample], list[float], int, bool]],
    dict[str, float],
]:
    """Rescore each row under its first exact distributed worker geometry."""

    started = time.perf_counter()
    replacements: dict[tuple[int, int, int], FlowSDEExample] = {}
    delta_weighted_sum = 0.0
    delta_max = 0.0
    rescored_rows = 0
    scorer_batches = 0
    for sub_examples, sub_advantages, _denominator, _length_normalized in subupdates:
        for worker_examples, worker_advantages in _partition_flow_examples(
            sub_examples,
            sub_advantages,
            workers=workers,
            task_keys=task_keys,
        ):
            unseen = [
                (example, advantage)
                for example, advantage in zip(
                    worker_examples, worker_advantages, strict=True
                )
                if _flow_example_coordinate(example) not in replacements
            ]
            if not unseen:
                continue
            pending_examples = [example for example, _advantage in unseen]
            pending_advantages = [advantage for _example, advantage in unseen]
            refreshed, report = precalculate_flow_sde_logprobs(
                policy,
                pending_examples,
                pending_advantages,
                microbatch_size=microbatch_size,
                device=device,
            )
            rows = int(report["old_logprob_rescore_rows"])
            delta_weighted_sum += (
                report["old_logprob_rescore_previous_abs_delta_mean"] * rows
            )
            delta_max = max(
                delta_max,
                report["old_logprob_rescore_previous_abs_delta_max"],
            )
            rescored_rows += rows
            scorer_batches += int(report["old_logprob_rescore_microbatches"])
            replacements.update(
                (_flow_example_coordinate(example), example) for example in refreshed
            )
    if len(replacements) != len(examples):
        raise RuntimeError(
            "Distributed Flow-SDE old-logprob rescore did not visit every row: "
            f"visited={len(replacements)}, expected={len(examples)}"
        )

    def refreshed_rows(rows: list[FlowSDEExample]) -> list[FlowSDEExample]:
        return [replacements[_flow_example_coordinate(example)] for example in rows]

    refreshed_subupdates = [
        (
            refreshed_rows(sub_examples),
            sub_advantages,
            denominator,
            length_normalized,
        )
        for sub_examples, sub_advantages, denominator, length_normalized in subupdates
    ]
    return (
        refreshed_rows(examples),
        refreshed_subupdates,
        {
            "old_logprob_rescore_rows": float(rescored_rows),
            "old_logprob_rescore_microbatches": float(scorer_batches),
            "old_logprob_rescore_previous_abs_delta_mean": (
                delta_weighted_sum / rescored_rows if rescored_rows else 0.0
            ),
            "old_logprob_rescore_previous_abs_delta_max": delta_max,
            "old_logprob_rescore_elapsed_seconds": time.perf_counter() - started,
        },
    )


def _flow_example_coordinate(example: FlowSDEExample) -> tuple[int, int, int]:
    return (example.group_index, example.trajectory_index, example.action_index)


def _flow_payload_cache_plan(
    *,
    subupdate_index: int,
    subupdate_count: int,
    update_epochs: int,
) -> tuple[str | None, bool]:
    """Reuse identical RLinf epoch slots without changing their ordering."""

    if update_epochs <= 1:
        return None, True
    if subupdate_count < 1 or subupdate_count % update_epochs:
        raise ValueError("Flow-SDE subupdates do not divide across update epochs")
    cache_period = subupdate_count // update_epochs
    if subupdate_index < 0 or subupdate_index >= subupdate_count:
        raise ValueError("Flow-SDE subupdate index is outside the update")
    return f"slot-{subupdate_index % cache_period:04d}", subupdate_index < cache_period


def _weighted_worker_metric(
    results: list[dict[str, Any]],
    key: str,
    *,
    weight: str,
    empty_value: float = 0.0,
) -> float:
    total = sum(float(result["metrics"][weight]) for result in results)
    if total <= 0:
        return float(empty_value)
    return (
        sum(
            float(result["metrics"].get(key, empty_value))
            * float(result["metrics"][weight])
            for result in results
        )
        / total
    )


def _aggregate_flow_worker_metrics(
    results: list[dict[str, Any]],
    *,
    apply_metrics: dict[str, float],
    subupdate_index: int,
) -> dict[str, float]:
    sft_replay_loss = sum(
        result["metrics"].get("sft_replay_loss", 0.0) for result in results
    ) / len(results)
    sft_replay_weighted_loss = sum(
        result["metrics"].get("sft_replay_weighted_loss", 0.0) for result in results
    ) / len(results)
    metrics = dict(apply_metrics)
    metrics.update(
        {
            "loss": (
                sum(result["metrics"]["loss"] for result in results)
                + sft_replay_weighted_loss
            ),
            "policy_loss": sum(
                result["metrics"].get("policy_loss", result["metrics"]["loss"])
                for result in results
            ),
            "reference_kl": _weighted_worker_metric(
                results, "reference_kl", weight="valid_rows"
            ),
            "reference_kl_loss": sum(
                result["metrics"].get("reference_kl_loss", 0.0) for result in results
            ),
            "reference_kl_coefficient": float(
                results[0]["metrics"].get("reference_kl_coefficient", 0.0)
            ),
            "sft_replay_loss": sft_replay_loss,
            "sft_replay_weighted_loss": sft_replay_weighted_loss,
            "sft_replay_coefficient": float(
                results[0]["metrics"].get("sft_replay_coefficient", 0.0)
            ),
            "sft_replay_examples": sum(
                result["metrics"].get("sft_replay_examples", 0.0) for result in results
            ),
            "ratio_mean": _weighted_worker_metric(
                results, "ratio_mean", weight="valid_rows", empty_value=1.0
            ),
            "approximate_kl": _weighted_worker_metric(
                results, "approximate_kl", weight="valid_rows"
            ),
            "approximate_kl_per_primitive": _weighted_worker_metric(
                results, "approximate_kl_per_primitive", weight="valid_rows"
            ),
            "clip_fraction": _weighted_worker_metric(
                results, "clip_fraction", weight="valid_rows"
            ),
            "previous_abs_delta_mean": _weighted_worker_metric(
                results, "previous_abs_delta_mean", weight="rows"
            ),
            "previous_abs_delta_max": max(
                result["metrics"]["previous_abs_delta_max"] for result in results
            ),
            "previous_ratio_mean": _weighted_worker_metric(
                results, "previous_ratio_mean", weight="rows", empty_value=1.0
            ),
            "active_previous_abs_delta_mean": _weighted_worker_metric(
                results, "active_previous_abs_delta_mean", weight="valid_rows"
            ),
            "active_previous_abs_delta_max": max(
                result["metrics"]["active_previous_abs_delta_max"] for result in results
            ),
            "active_previous_ratio_mean": _weighted_worker_metric(
                results,
                "active_previous_ratio_mean",
                weight="valid_rows",
                empty_value=1.0,
            ),
            "previous_abs_delta_per_primitive_mean": _weighted_worker_metric(
                results,
                "previous_abs_delta_per_primitive_mean",
                weight="rows",
            ),
            "previous_abs_delta_per_primitive_max": max(
                result["metrics"]["previous_abs_delta_per_primitive_max"]
                for result in results
            ),
            "active_previous_abs_delta_per_primitive_mean": (
                _weighted_worker_metric(
                    results,
                    "active_previous_abs_delta_per_primitive_mean",
                    weight="valid_rows",
                )
            ),
            "active_previous_abs_delta_per_primitive_max": max(
                result["metrics"]["active_previous_abs_delta_per_primitive_max"]
                for result in results
            ),
            "worker_seconds_max": max(result["elapsed_seconds"] for result in results),
            "payload_cache_hit_fraction": sum(
                float(result["payload_cache_hit"]) for result in results
            )
            / len(results),
            "workers": float(len(results)),
            "subupdate_index": float(subupdate_index),
        }
    )
    return metrics


def _aggregate_flow_subupdates(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    count = float(len(items))
    keys = (
        "loss",
        "policy_loss",
        "reference_kl",
        "reference_kl_loss",
        "reference_kl_coefficient",
        "sft_replay_loss",
        "sft_replay_weighted_loss",
        "sft_replay_coefficient",
        "sft_replay_examples",
        "ratio_mean",
        "approximate_kl",
        "approximate_kl_per_primitive",
        "clip_fraction",
        "previous_abs_delta_mean",
        "previous_abs_delta_per_primitive_mean",
        "previous_ratio_mean",
        "worker_seconds_max",
        "payload_cache_hit_fraction",
    )
    result = {
        f"embodied_flow_sde_grpo/{key}": (
            sum(item.get(key, 0.0) for item in items) / count
        )
        for key in keys
    }
    result["embodied_flow_sde_grpo/previous_abs_delta_max"] = max(
        item["previous_abs_delta_max"] for item in items
    )
    result["embodied_flow_sde_grpo/active_previous_abs_delta_max"] = max(
        item.get("active_previous_abs_delta_max", item["previous_abs_delta_max"])
        for item in items
    )
    result["embodied_flow_sde_grpo/active_previous_abs_delta_mean"] = (
        sum(
            item.get("active_previous_abs_delta_mean", item["previous_abs_delta_mean"])
            for item in items
        )
        / count
    )
    result["embodied_flow_sde_grpo/previous_abs_delta_per_primitive_max"] = max(
        item["previous_abs_delta_per_primitive_max"] for item in items
    )
    result["embodied_flow_sde_grpo/active_previous_abs_delta_per_primitive_mean"] = (
        sum(item["active_previous_abs_delta_per_primitive_mean"] for item in items)
        / count
    )
    result["embodied_flow_sde_grpo/active_previous_abs_delta_per_primitive_max"] = max(
        item["active_previous_abs_delta_per_primitive_max"] for item in items
    )
    result["embodied_flow_sde_grpo/active_previous_ratio_mean"] = (
        sum(
            item.get("active_previous_ratio_mean", item["previous_ratio_mean"])
            for item in items
        )
        / count
    )
    # Subupdate zero is the only clean rollout/rescore alignment measurement.
    # Later subupdates intentionally observe a policy that has moved away from
    # the rollout policy, so their average is optimizer-path drift instead.
    result.update(
        {
            "embodied_flow_sde_grpo/pre_update_alignment_abs_delta_mean": items[0][
                "previous_abs_delta_mean"
            ],
            "embodied_flow_sde_grpo/pre_update_alignment_abs_delta_max": items[0][
                "previous_abs_delta_max"
            ],
            "embodied_flow_sde_grpo/pre_update_alignment_ratio_mean": items[0][
                "previous_ratio_mean"
            ],
            "embodied_flow_sde_grpo/pre_update_active_alignment_abs_delta_mean": (
                items[0].get(
                    "active_previous_abs_delta_mean",
                    items[0]["previous_abs_delta_mean"],
                )
            ),
            "embodied_flow_sde_grpo/pre_update_active_alignment_abs_delta_max": (
                items[0].get(
                    "active_previous_abs_delta_max",
                    items[0]["previous_abs_delta_max"],
                )
            ),
            "embodied_flow_sde_grpo/"
            "pre_update_active_alignment_abs_delta_per_primitive_mean": (
                items[0]["active_previous_abs_delta_per_primitive_mean"]
            ),
            "embodied_flow_sde_grpo/"
            "pre_update_active_alignment_abs_delta_per_primitive_max": (
                items[0]["active_previous_abs_delta_per_primitive_max"]
            ),
            "embodied_flow_sde_grpo/pre_update_active_alignment_ratio_mean": (
                items[0].get(
                    "active_previous_ratio_mean",
                    items[0]["previous_ratio_mean"],
                )
            ),
            "embodied_flow_sde_grpo/optimization_old_policy_abs_delta_mean": result[
                "embodied_flow_sde_grpo/previous_abs_delta_mean"
            ],
        }
    )
    coherence_suffixes = (
        "worker_gradient_pairwise_cosine_mean",
        "worker_gradient_resultant_ratio",
        "worker_gradient_signal_to_rms_ratio",
        "worker_gradient_noise_to_signal_ratio",
        "worker_gradient_effective_aligned_workers",
    )
    for suffix in coherence_suffixes:
        key = f"embodied_flow_sde_grpo/{suffix}"
        if not all(key in item for item in items):
            continue
        result[f"embodied_flow_sde_grpo/pre_update_{suffix}"] = items[0][key]
        result[f"{key}_across_subupdates"] = sum(item[key] for item in items) / count
    for key, value in items[-1].items():
        if key.startswith("embodied_flow_sde_grpo/"):
            result[key] = value
    return result
