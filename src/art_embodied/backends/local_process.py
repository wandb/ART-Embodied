"""Local multi-GPU action-token training with isolated worker processes."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import pickle
import random
import shutil
import subprocess
import tempfile
import time
from typing import Any, TextIO

from art_embodied.checkpointing import CheckpointManager
from art_embodied.config import (
    EmbodiedExperimentConfig,
    FullUpdateScheduleConfig,
    RlinfActorBatchScheduleConfig,
    TrajectoryMinibatchScheduleConfig,
)
from art_embodied.trajectories import EmbodiedTrajectoryGroup
from art_embodied.types import LocalTrainResult, TrainResult
from art_embodied.utils import make_json_safe, worker_python_command

from ..conformance.rlinf import prepare_rlinf_action_token_update
from .action_token import (
    ActionTokenExample,
    ActionTokenGRPOBackend,
    _example_advantage_sign_token_counts,
    _example_loss_denominator_count,
    _loss_denominator_example_count,
    _trajectory_group_signal_metrics,
    prepare_action_token_examples,
)
from .action_token_advantages import _example_advantages
from .action_token_gradients import (
    apply_action_token_gradient_payloads,
    load_action_token_gradient_payload,
)

_SUBUPDATE_BOUNDARY_METRIC_SUFFIXES = frozenset(
    {
        "approx_kl_abs_mean",
        "clip_fraction",
        "loss",
        "objective_direction_advantage_logprob_delta_product_mean",
        "objective_direction_sign_agreement_fraction",
        "ratio_max",
        "ratio_mean",
        "ratio_min",
    }
)
_CHECKPOINT_MANAGER = CheckpointManager()


@dataclass(frozen=True, slots=True)
class _GradientBatch:
    index: int
    examples: tuple[ActionTokenExample, ...]
    denominator_examples: int
    denominator_tokens: int
    positive_tokens: int
    negative_tokens: int


@dataclass(frozen=True, slots=True)
class _WorkerResult:
    index: int
    device: str
    metrics: dict[str, float]
    gradient_payload: dict[str, Any]
    elapsed_seconds: float
    policy_loads: int
    adapter_refreshes: int


@dataclass(slots=True)
class _WorkerProcess:
    index: int
    device: str
    process: subprocess.Popen[str]
    stdin: TextIO
    stdout: TextIO
    stderr: TextIO
    worker_dir: Path
    ready_path: Path


class LocalProcessActionTokenBackend:
    """Execute one action-token update across local CUDA worker processes.

    The coordinator prepares rewards, masks, advantages, and RLinf actor-batch
    geometry exactly once. Each worker receives a disjoint slice of those
    prepared examples, computes its globally normalized loss, and returns only
    trainable-parameter gradients. The coordinator sums the gradients and owns
    the single persistent optimizer state.

    This backend is local-machine infrastructure, not a Slurm integration.
    Slurm, Kubernetes, or a plain shell may allocate the machine; the YAML
    ``runtime.training_devices`` list is the complete device contract.
    """

    def __init__(
        self,
        *,
        config: EmbodiedExperimentConfig,
        policy: Any,
        backend: ActionTokenGRPOBackend,
    ) -> None:
        if not config.runtime.distributed_training:
            raise ValueError(
                "LocalProcessActionTokenBackend requires "
                "runtime.distributed_training=true"
            )
        if config.policy.type not in {"openvla_oft", "pi0_fast"}:
            raise NotImplementedError(
                "The built-in local-process trainer currently supports "
                "policy.type='openvla_oft' and policy.type='pi0_fast'. "
                "Register a model-native backend for other policy families."
            )
        self.config = config
        self.policy = policy
        self.backend = backend
        self.update_step = 0
        self._persistent_worker_pool: _PersistentGradientWorkerPool | None = None
        if config.storage.resume_from_checkpoint is not None:
            self._restore_checkpoint(config.storage.resume_from_checkpoint)

    def _restore_checkpoint(self, path: Path) -> None:
        validation = _CHECKPOINT_MANAGER.validate(
            path,
            expected_resume_contract_fingerprint=(
                self.config.resume_contract_fingerprint
            ),
            allow_legacy=self.config.storage.allow_legacy_checkpoint_resume,
            require_training_state=True,
        )
        load = getattr(self.policy, "load_checkpoint", None)
        if not callable(load):
            raise TypeError("Checkpoint resume requires policy.load_checkpoint(path)")
        load({"path": str(path)})
        # The coordinator applies aggregated gradients on CPU while CUDA is
        # owned by model workers. Establish that residency before constructing
        # and restoring Adam state; otherwise PyTorch casts restored moments to
        # CUDA and the first resumed CPU optimizer step fails.
        _move_policy_model(self.policy, "cpu")
        self.backend._ensure_optimizer()
        if self.backend.optimizer is None:
            raise RuntimeError("Action-token optimizer was not initialized")
        state = _load_training_state(
            path,
            optimizer=self.backend.optimizer,
            expected_resume_contract_fingerprint=(
                self.config.resume_contract_fingerprint
            ),
            allow_missing_resume_contract=validation.legacy,
        )
        self.update_step = int(state["update_step"])
        self.backend.step = int(state["backend_step"])
        if self.update_step >= self.config.training.updates:
            raise ValueError(
                "Resumed update must be smaller than training.updates: "
                f"resumed={self.update_step}, target={self.config.training.updates}"
            )

    def save_snapshot(self, path: Path, *, update: int) -> None:
        """Publish the coordinator policy through the rollout wire contract."""

        if int(update) != self.update_step:
            raise ValueError(
                "Rollout snapshot update does not match backend policy state: "
                f"requested={update}, backend={self.update_step}"
            )
        _save_policy_snapshot(self.policy, path)

    def offload(self) -> None:
        """Release the coordinator policy while rollout workers own CUDA."""

        _move_policy_model(self.policy, "cpu")

    def restore(self, device: str) -> None:
        """Restore the coordinator policy before the next optimizer phase."""

        _move_policy_model(self.policy, device)

    async def train(
        self,
        trajectory_groups: Sequence[EmbodiedTrajectoryGroup],
        **kwargs: Any,
    ) -> TrainResult:
        if kwargs:
            unknown = ", ".join(sorted(str(key) for key in kwargs))
            raise TypeError(
                "LocalProcessActionTokenBackend does not accept hidden train "
                f"overrides; put them in YAML: {unknown}"
            )
        return await asyncio.to_thread(
            self._train_sync,
            list(trajectory_groups),
        )

    def _train_sync(
        self,
        groups: list[EmbodiedTrajectoryGroup],
    ) -> LocalTrainResult:
        batches, reward_filter_report = _prepare_gradient_batches(
            groups,
            config=self.config,
            backend=self.backend,
            update_step=self.update_step,
        )
        self.backend._ensure_optimizer()
        if self.backend.optimizer is None:
            raise RuntimeError("Action-token optimizer was not initialized")

        started = time.perf_counter()
        batch_metrics: list[dict[str, float]] = []
        reuse_workers = self.config.runtime.training_worker_lifecycle == "cpu_offload"
        if reuse_workers:
            if self._persistent_worker_pool is None:
                self._persistent_worker_pool = _PersistentGradientWorkerPool(
                    config=self.config,
                    policy=self.policy,
                    update_step=self.update_step,
                )
            worker_pool = self._persistent_worker_pool
        else:
            worker_pool = _PersistentGradientWorkerPool(
                config=self.config,
                policy=self.policy,
                update_step=self.update_step,
            )
        worker_pool.begin_update(update_step=self.update_step)
        try:
            rescore_metrics: dict[str, float] = {}
            if self.config.algorithm.precalculate_logprobs:
                if isinstance(
                    self.config.training.schedule,
                    FullUpdateScheduleConfig,
                ):
                    # Full-update epochs reuse one rollout batch. Recompute its
                    # old likelihoods once, then keep them frozen while later
                    # epochs compare the updated policy against that behavior
                    # policy.
                    rescored, rescore_metrics = worker_pool.rescore_batches(
                        batches[:1],
                        source=self.config.algorithm.rollout_logprob_source,
                    )
                    first = rescored[0]
                    batches = [
                        _gradient_batch_with_examples(
                            batch,
                            examples=first.examples,
                        )
                        for batch in batches
                    ]
                else:
                    batches, rescore_metrics = worker_pool.rescore_batches(
                        batches,
                        source=self.config.algorithm.rollout_logprob_source,
                    )
            for batch in batches:
                workers = worker_pool.compute(
                    batch,
                    reward_filter_report=reward_filter_report,
                )
                _raise_for_worker_alignment_rejection(workers, batch_index=batch.index)
                apply_started = time.perf_counter()
                apply_metrics = apply_action_token_gradient_payloads(
                    self.policy,
                    self.backend.optimizer,
                    [worker.gradient_payload for worker in workers],
                    max_grad_norm=self.backend.max_grad_norm,
                    skip_optimizer_step_without_policy_gradient_signal=(
                        self.backend.skip_optimizer_step_without_policy_gradient_signal
                    ),
                )
                apply_metrics["embodied_action_token_grpo/gradient_apply_seconds"] = (
                    time.perf_counter() - apply_started
                )
                self.backend.step += int(
                    apply_metrics.get(
                        "embodied_action_token_grpo/optimizer_step_completed",
                        0.0,
                    )
                    > 0.0
                )
                batch_metrics.append(
                    _aggregate_worker_metrics(
                        workers,
                        apply_metrics=apply_metrics,
                        batch_index=batch.index,
                    )
                )
        finally:
            if reuse_workers:
                worker_pool.finish_update()
            else:
                worker_pool.close()

        self.update_step += 1
        checkpoint_path = self._maybe_checkpoint()
        metrics = _aggregate_batch_metrics(batch_metrics)
        metrics.update(rescore_metrics)
        # Worker shards do not contain complete groups. Recompute group-level
        # diagnostics once from the coordinator's original trajectories rather
        # than averaging per-worker fractions with unequal denominators.
        metrics.update(
            _trajectory_group_signal_metrics(
                groups,
                prefix="embodied_action_token_grpo",
            )
        )
        if self.backend.importance_sampling_level == "sequence":
            # Worker processes only compute gradients, so their GSPO aliases
            # correctly report optimizer_step_completed=0. The coordinator owns
            # the actual optimizer step; re-alias its final GRPO metrics after
            # aggregation so the public GSPO namespace reflects that update.
            _expose_coordinator_gspo_metrics(metrics)
        metrics.update(
            {
                "embodied_action_token_schedule/update": float(self.update_step),
                "embodied_action_token_schedule/subupdates": float(len(batches)),
                "embodied_action_token_schedule/distributed_training": 1.0,
                "embodied_action_token_schedule/training_workers": float(
                    len(self.config.runtime.training_devices)
                ),
                "embodied_action_token_schedule/training_worker_reused": float(
                    worker_pool.reused_workers
                ),
                "embodied_action_token_schedule/worker_startup_seconds": float(
                    worker_pool.last_startup_seconds
                ),
                "embodied_action_token_schedule/worker_offload_seconds": float(
                    worker_pool.last_offload_seconds
                ),
                "embodied_action_token_schedule/elapsed_seconds": float(
                    time.perf_counter() - started
                ),
            }
        )
        return LocalTrainResult(
            step=self.update_step,
            metrics=metrics,
            checkpoint_path=checkpoint_path,
        )

    def _maybe_checkpoint(self) -> str | None:
        retained_updates = set(self.config.storage.retain_checkpoint_updates)
        periodic = self.update_step % self.config.training.checkpoint_every_updates == 0
        if not periodic and self.update_step not in retained_updates:
            return None
        path = (
            self.config.storage.output_dir
            / "checkpoints"
            / f"step_{self.update_step:06d}"
        )

        def write_checkpoint(staging: Path) -> None:
            _save_policy_snapshot(self.policy, staging)
            if self.config.storage.save_training_state:
                _save_training_state(
                    staging,
                    optimizer=self.backend.optimizer,
                    update_step=self.update_step,
                    backend_step=self.backend.step,
                    config_fingerprint=self.config.fingerprint,
                    resume_contract_fingerprint=(
                        self.config.resume_contract_fingerprint
                    ),
                )

        _CHECKPOINT_MANAGER.publish(
            path,
            writer=write_checkpoint,
            config_fingerprint=self.config.fingerprint,
            resume_contract_fingerprint=self.config.resume_contract_fingerprint,
            metadata={
                "update_step": self.update_step,
                "backend_step": self.backend.step,
                "contains_training_state": self.config.storage.save_training_state,
            },
        )
        checkpoints = sorted(path.parent.glob("step_*"))
        unprotected = [
            checkpoint
            for checkpoint in checkpoints
            if _checkpoint_step(checkpoint) not in retained_updates
        ]
        for stale in unprotected[: -self.config.storage.keep_last_checkpoints]:
            shutil.rmtree(stale, ignore_errors=True)
        return str(path)

    async def close(self) -> None:
        if self._persistent_worker_pool is not None:
            self._persistent_worker_pool.close()
            self._persistent_worker_pool = None
        await self.backend.close()


class _PersistentGradientWorkerPool:
    """Keep one loaded VLA process per training device for a complete update."""

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
        self.workers: list[_WorkerProcess] = []
        self.job_index = 0
        self.offloaded = False
        self.reused_workers = False
        self.last_startup_seconds = 0.0
        self.last_offload_seconds = 0.0
        self._policy_loads_on_first_job = False

    def __enter__(self) -> "_PersistentGradientWorkerPool":
        self.begin_update(update_step=self.update_step)
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def begin_update(self, *, update_step: int) -> None:
        if self.workers and not self.offloaded:
            raise RuntimeError("Training workers are already active for an update")
        started = time.perf_counter()
        self.update_step = update_step
        self.job_index = 0
        self.reused_workers = bool(self.workers)
        self._policy_loads_on_first_job = not self.workers
        if self.run_dir is None:
            handoff_root = self.config.runtime.worker_handoff_dir.expanduser()
            handoff_root.mkdir(parents=True, exist_ok=True)
            self.run_dir = Path(
                tempfile.mkdtemp(prefix="training-workers-", dir=handoff_root)
            )
        self.initial_snapshot = (
            self.run_dir / "snapshots" / f"update-{self.update_step:04d}" / "initial"
        )
        _save_policy_snapshot(self.policy, self.initial_snapshot)
        _move_policy_model(self.policy, "cpu")
        try:
            if not self.workers:
                for index, device in enumerate(self.config.runtime.training_devices):
                    self.workers.append(self._start_worker(index=index, device=device))
                self._wait_until_ready()
            self.offloaded = False
        except Exception:
            self.close()
            raise
        self.last_startup_seconds = time.perf_counter() - started

    def _start_worker(self, *, index: int, device: str) -> _WorkerProcess:
        assert self.run_dir is not None
        assert self.initial_snapshot is not None
        worker_dir = self.run_dir / f"worker-{index:02d}"
        worker_dir.mkdir(parents=True)
        ready_path = worker_dir / "ready.json"
        spec_path = worker_dir / "bootstrap.json"
        anchor_tasks = _pi0_fast_sft_anchor_tasks_for_worker(
            self.config,
            worker_index=index,
            worker_count=len(self.config.runtime.training_devices),
        )
        spec_path.write_text(
            json.dumps(
                {
                    "config": self.config.model_dump(mode="json"),
                    "policy_snapshot": str(self.initial_snapshot),
                    "ready_path": str(ready_path),
                    "worker_index": index,
                    "sft_anchor_task_indices": anchor_tasks,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        stdout = (worker_dir / "stdout.log").open("w", encoding="utf-8")
        stderr = (worker_dir / "stderr.log").open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                worker_python_command(
                    configured_executable=(
                        self.config.runtime.worker_python_executable
                    ),
                    module="art_embodied.backends.action_token_worker",
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
        except Exception:
            stdout.close()
            stderr.close()
            raise
        if process.stdin is None:
            _stop_process(process)
            stdout.close()
            stderr.close()
            raise RuntimeError("Persistent action-token worker has no stdin pipe")
        return _WorkerProcess(
            index=index,
            device=device,
            process=process,
            stdin=process.stdin,
            stdout=stdout,
            stderr=stderr,
            worker_dir=worker_dir,
            ready_path=ready_path,
        )

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.config.runtime.worker_timeout_seconds
        pending = {worker.index: worker for worker in self.workers}
        while pending:
            for index, worker in list(pending.items()):
                if worker.ready_path.exists():
                    payload = json.loads(worker.ready_path.read_text(encoding="utf-8"))
                    if not payload.get("ok"):
                        raise RuntimeError(
                            f"Persistent worker {index} failed during startup: {payload}"
                        )
                    pending.pop(index)
                    continue
                return_code = worker.process.poll()
                if return_code is not None:
                    raise RuntimeError(self._worker_failure(worker, return_code))
            if time.monotonic() >= deadline:
                names = ", ".join(str(index) for index in sorted(pending))
                raise RuntimeError(
                    "Persistent action-token workers timed out during model load: "
                    f"workers={names}"
                )
            time.sleep(0.1)

    def rescore_batches(
        self,
        batches: Sequence[_GradientBatch],
        *,
        source: str,
    ) -> tuple[list[_GradientBatch], dict[str, float]]:
        """Recompute old logprobs on all workers before any optimizer step."""

        if self.job_index != 0:
            raise RuntimeError("Old-logprob rescore must precede every gradient job")
        rescored: list[_GradientBatch] = []
        worker_seconds: list[float] = []
        previous_means: list[tuple[float, int]] = []
        previous_max = 0.0
        calls = 0
        tokens = 0
        started = time.perf_counter()
        for batch in batches:
            updated, reports = self._rescore_batch(batch, source=source)
            rescored.append(updated)
            for report in reports:
                worker_seconds.append(float(report.get("elapsed_seconds") or 0.0))
                report_payload = report.get("report") or {}
                report_tokens = int(report_payload.get("tokens_updated") or 0)
                tokens += report_tokens
                calls += int(report_payload.get("policy_logprob_calls") or 0)
                previous_means.append(
                    (
                        float(report_payload.get("previous_abs_delta_mean") or 0.0),
                        report_tokens,
                    )
                )
                previous_max = max(
                    previous_max,
                    float(report_payload.get("previous_abs_delta_max") or 0.0),
                )
        weighted_delta = sum(value * count for value, count in previous_means)
        return rescored, {
            "embodied_action_token_schedule/old_logprob_rescore_enabled": 1.0,
            "embodied_action_token_schedule/old_logprob_rescore_batches": float(
                len(rescored)
            ),
            "embodied_action_token_schedule/old_logprob_rescore_tokens": float(tokens),
            "embodied_action_token_schedule/old_logprob_rescore_calls": float(calls),
            "embodied_action_token_schedule/old_logprob_rescore_previous_abs_delta_mean": (
                weighted_delta / float(tokens) if tokens else 0.0
            ),
            "embodied_action_token_schedule/old_logprob_rescore_previous_abs_delta_max": (
                previous_max
            ),
            "embodied_action_token_schedule/old_logprob_rescore_worker_seconds_mean": (
                sum(worker_seconds) / float(len(worker_seconds))
                if worker_seconds
                else 0.0
            ),
            "embodied_action_token_schedule/old_logprob_rescore_elapsed_seconds": float(
                time.perf_counter() - started
            ),
        }

    def _rescore_batch(
        self,
        batch: _GradientBatch,
        *,
        source: str,
    ) -> tuple[_GradientBatch, list[dict[str, Any]]]:
        assert self.run_dir is not None
        assert self.initial_snapshot is not None
        partitions = _partition_examples(batch.examples, workers=len(self.workers))
        jobs: list[tuple[_WorkerProcess, Path, Path]] = []
        for worker, examples in zip(self.workers, partitions, strict=True):
            if not examples:
                continue
            job_dir = (
                worker.worker_dir
                / f"update-{self.update_step:04d}-rescore-{batch.index:04d}"
            )
            job_dir.mkdir(parents=True)
            examples_path = job_dir / "examples.pkl"
            output_examples_path = job_dir / "rescored-examples.pkl"
            result_path = job_dir / "result.json"
            with examples_path.open("wb") as handle:
                pickle.dump(examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
            _guard_file_size(
                examples_path,
                max_mb=self.config.runtime.max_worker_handoff_mb,
            )
            worker.stdin.write(
                json.dumps(
                    {
                        "op": "rescore",
                        "policy_snapshot": str(self.initial_snapshot),
                        "examples_path": str(examples_path),
                        "output_examples_path": str(output_examples_path),
                        "result_path": str(result_path),
                        "source": source,
                        "worker_index": worker.index,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            worker.stdin.flush()
            jobs.append((worker, result_path, output_examples_path))
        self._wait_for_jobs(jobs)
        examples_by_index: dict[tuple[int, int], ActionTokenExample] = {}
        reports: list[dict[str, Any]] = []
        for worker, result_path, output_examples_path in jobs:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if not payload.get("ok"):
                raise RuntimeError(
                    "Persistent action-token rescore worker failed:\n"
                    + str(payload.get("traceback") or payload)
                )
            _guard_file_size(
                output_examples_path,
                max_mb=self.config.runtime.max_worker_handoff_mb,
            )
            with output_examples_path.open("rb") as handle:
                worker_examples = pickle.load(handle)
            for example in worker_examples:
                stable_key = (
                    int(example.trajectory_index),
                    int(example.action_index),
                )
                if stable_key in examples_by_index:
                    raise RuntimeError(
                        f"Distributed rescore returned duplicate example={stable_key}"
                    )
                examples_by_index[stable_key] = example
            reports.append(payload)
        ordered = tuple(
            examples_by_index[
                (int(example.trajectory_index), int(example.action_index))
            ]
            for example in batch.examples
        )
        return (
            _GradientBatch(
                index=batch.index,
                examples=ordered,
                denominator_examples=batch.denominator_examples,
                denominator_tokens=batch.denominator_tokens,
                positive_tokens=batch.positive_tokens,
                negative_tokens=batch.negative_tokens,
            ),
            reports,
        )

    def compute(
        self,
        batch: _GradientBatch,
        *,
        reward_filter_report: dict[str, Any],
    ) -> list[_WorkerResult]:
        assert self.run_dir is not None
        assert self.initial_snapshot is not None
        rank_partition = self.config.policy.lora.rank_partition
        task_keys = (
            tuple(rank_partition.task_keys) if rank_partition is not None else None
        )
        partitions = _partition_examples(
            batch.examples,
            workers=len(self.workers),
            task_keys=task_keys,
        )
        if not any(partitions):
            raise RuntimeError("Distributed action-token batch has no examples")

        base_job_index = self.job_index
        if base_job_index == 0:
            snapshot = self.initial_snapshot
        else:
            snapshot = self.run_dir / "snapshots" / f"job-{base_job_index:04d}"
            _save_policy_snapshot(self.policy, snapshot)

        results: list[_WorkerResult] = []
        waves = 0
        for wave_start in range(0, len(partitions), len(self.workers)):
            wave = partitions[wave_start : wave_start + len(self.workers)]
            job_index = base_job_index + waves
            active = [
                (worker, examples)
                for worker, examples in zip(self.workers, wave, strict=False)
                if examples
            ]
            jobs: list[tuple[_WorkerProcess, Path, Path]] = []
            for worker, worker_examples in active:
                observed_tasks = {str(example.task) for example in worker_examples}
                task_key = (
                    next(iter(observed_tasks)) if len(observed_tasks) == 1 else None
                )
                if task_keys is not None and task_key is None:
                    raise RuntimeError(
                        "Task-partitioned action-token worker received mixed tasks"
                    )
                job_dir = (
                    worker.worker_dir
                    / f"update-{self.update_step:04d}-job-{job_index:04d}"
                )
                job_dir.mkdir(parents=True)
                examples_path = job_dir / "examples.pkl"
                with examples_path.open("wb") as handle:
                    pickle.dump(
                        worker_examples, handle, protocol=pickle.HIGHEST_PROTOCOL
                    )
                _guard_file_size(
                    examples_path,
                    max_mb=self.config.runtime.max_worker_handoff_mb,
                )
                gradient_path = job_dir / "gradients.pt"
                result_path = job_dir / "result.json"
                command = {
                    "op": "gradient",
                    "policy_snapshot": str(snapshot),
                    "examples_path": str(examples_path),
                    "gradient_path": str(gradient_path),
                    "result_path": str(result_path),
                    "reward_filter_report": make_json_safe(reward_filter_report),
                    "global_example_count": batch.denominator_examples,
                    "global_token_count": batch.denominator_tokens,
                    "global_positive_token_count": batch.positive_tokens,
                    "global_negative_token_count": batch.negative_tokens,
                    "enforce_pre_update_alignment": self._guard_alignment_for_job(),
                    "worker_index": worker.index,
                    "update_index": self.update_step,
                    "subupdate_index": batch.index,
                }
                if task_key is not None:
                    command["task_key"] = task_key
                worker.stdin.write(json.dumps(command, sort_keys=True) + "\n")
                worker.stdin.flush()
                jobs.append((worker, result_path, gradient_path))

            self._wait_for_jobs(jobs)
            for worker, result_path, gradient_path in jobs:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                if not payload.get("ok"):
                    raise RuntimeError(
                        "Persistent action-token worker failed:\n"
                        + str(payload.get("traceback") or payload)
                    )
                _guard_file_size(
                    gradient_path,
                    max_mb=self.config.runtime.max_worker_handoff_mb,
                )
                refresh = payload.get("adapter_refresh")
                results.append(
                    _WorkerResult(
                        index=worker.index,
                        device=worker.device,
                        metrics={
                            str(key): float(value)
                            for key, value in (payload.get("metrics") or {}).items()
                            if isinstance(value, int | float | bool)
                        },
                        gradient_payload=load_action_token_gradient_payload(
                            gradient_path
                        ),
                        elapsed_seconds=float(payload.get("elapsed_seconds") or 0.0),
                        policy_loads=int(
                            self._policy_loads_on_first_job
                            and base_job_index == 0
                            and waves == 0
                        ),
                        adapter_refreshes=int(isinstance(refresh, dict)),
                    )
                )
            waves += 1
        self.job_index += waves
        return results

    def finish_update(self) -> None:
        if self.offloaded or not self.workers:
            return
        started = time.perf_counter()
        jobs: list[tuple[_WorkerProcess, Path, Path]] = []
        for worker in self.workers:
            result_path = (
                worker.worker_dir / f"update-{self.update_step:04d}-offload.json"
            )
            worker.stdin.write(
                json.dumps({"op": "offload", "result_path": str(result_path)}) + "\n"
            )
            worker.stdin.flush()
            jobs.append((worker, result_path, result_path))
        self._wait_for_jobs(jobs)
        for worker, result_path, _unused in jobs:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if not payload.get("ok"):
                raise RuntimeError(
                    f"Persistent worker {worker.index} failed to offload: {payload}"
                )
        _move_policy_model(self.policy, self.config.policy.device)
        self.offloaded = True
        self.last_offload_seconds = time.perf_counter() - started
        if not self.config.runtime.keep_worker_handoffs:
            self._prune_completed_update_handoffs()

    def _prune_completed_update_handoffs(self) -> None:
        assert self.run_dir is not None
        snapshot_dir = self.run_dir / "snapshots" / f"update-{self.update_step:04d}"
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        pattern = f"update-{self.update_step:04d}-*"
        for worker in self.workers:
            for path in worker.worker_dir.glob(pattern):
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)

    def _guard_alignment_for_job(self) -> bool:
        schedule = self.config.training.schedule
        if self.job_index != 0:
            return False
        if isinstance(schedule, FullUpdateScheduleConfig):
            # A full update has exactly one gradient job. Honor the configured
            # fail-closed old/new logprob contract before applying its gradient.
            return True
        if self.config.algorithm.type == "gspo" or isinstance(
            schedule, TrajectoryMinibatchScheduleConfig
        ):
            # The first minibatch must reproduce rollout likelihoods. Later
            # minibatches intentionally use the refreshed current policy
            # against the same old rollout policy and are therefore off-policy.
            return True
        return bool(
            isinstance(schedule, RlinfActorBatchScheduleConfig)
            and schedule.pre_update_alignment_guard == "first_subupdate"
        )

    def _wait_for_jobs(
        self,
        jobs: Sequence[tuple[_WorkerProcess, Path, Path]],
    ) -> None:
        deadline = time.monotonic() + self.config.runtime.worker_timeout_seconds
        pending = {worker.index: (worker, result) for worker, result, _ in jobs}
        while pending:
            for index, (worker, result_path) in list(pending.items()):
                if result_path.exists():
                    pending.pop(index)
                    continue
                return_code = worker.process.poll()
                if return_code is not None:
                    raise RuntimeError(self._worker_failure(worker, return_code))
            if time.monotonic() >= deadline:
                names = ", ".join(str(index) for index in sorted(pending))
                raise RuntimeError(
                    "Persistent action-token gradient jobs timed out: "
                    f"workers={names}, job={self.job_index}"
                )
            time.sleep(0.1)

    @staticmethod
    def _worker_failure(worker: _WorkerProcess, return_code: int) -> str:
        stderr_path = worker.worker_dir / "stderr.log"
        tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        return f"worker {worker.index} on {worker.device}: exit={return_code}\n{tail}"

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
        _move_policy_model(self.policy, self.config.policy.device)
        self.offloaded = False
        if self.run_dir is not None and not self.config.runtime.keep_worker_handoffs:
            shutil.rmtree(self.run_dir, ignore_errors=True)


def _prepare_gradient_batches(
    groups: list[EmbodiedTrajectoryGroup],
    *,
    config: EmbodiedExperimentConfig,
    backend: ActionTokenGRPOBackend,
    update_step: int = 0,
) -> tuple[list[_GradientBatch], dict[str, Any]]:
    if isinstance(config.training.schedule, RlinfActorBatchScheduleConfig):
        prepared = prepare_rlinf_action_token_update(
            groups,
            backend=backend,
            config=config,
        )
        if not _has_prepared_token_advantages(prepared.examples):
            # Worker shards do not necessarily contain every sibling from a
            # GRPO group. Freeze the globally computed group advantages before
            # partitioning so workers never re-normalize partial groups.
            _attach_full_update_advantages(prepared.examples, backend=backend)
        return [
            _gradient_batch_from_prepared_subupdate(subupdate)
            for subupdate in prepared.subupdates
        ], prepared.reward_filter_report

    examples, report = prepare_action_token_examples(groups, backend=backend)
    _attach_full_update_advantages(examples, backend=backend)
    if isinstance(config.training.schedule, TrajectoryMinibatchScheduleConfig):
        schedule = config.training.schedule
        # Action-token GRPO has multiple examples per trajectory. Partition
        # complete trajectories, after freezing their full-group advantages.
        by_trajectory: dict[int, list[ActionTokenExample]] = {}
        for example in examples:
            by_trajectory.setdefault(example.trajectory_index, []).append(example)
        if len(by_trajectory) != config.trajectories_per_update:
            raise ValueError(
                "Trajectory minibatch preparation requires the complete "
                "rollout batch: "
                f"trajectories={len(by_trajectory)}, "
                f"expected={config.trajectories_per_update}"
            )
        positions = list(by_trajectory)
        random.Random(schedule.shuffle_seed + int(update_step)).shuffle(positions)
        batches = []
        for index, start in enumerate(
            range(0, len(positions), schedule.minibatch_trajectories)
        ):
            selected_positions = positions[
                start : start + schedule.minibatch_trajectories
            ]
            if len(selected_positions) != schedule.minibatch_trajectories:
                raise ValueError(
                    "Trajectory minibatches must all be complete: "
                    f"subupdate={index}, trajectories={len(selected_positions)}, "
                    f"expected={schedule.minibatch_trajectories}"
                )
            selected = tuple(
                example
                for position in selected_positions
                for example in by_trajectory[position]
            )
            batches.append(_gradient_batch_from_examples(index, selected, backend))
        if len(batches) != config.training.optimizer_steps_per_update:
            raise ValueError(
                "Trajectory minibatch count does not match "
                "optimizer_steps_per_update: "
                f"batches={len(batches)}, "
                f"optimizer_steps={config.training.optimizer_steps_per_update}"
            )
        return batches, report

    batch = _gradient_batch_from_examples(0, tuple(examples), backend)
    if isinstance(config.training.schedule, FullUpdateScheduleConfig):
        return [
            _gradient_batch_with_examples(batch, examples=batch.examples, index=index)
            for index in range(config.training.schedule.update_epochs)
        ], report
    return [batch], report


def _gradient_batch_with_examples(
    batch: _GradientBatch,
    *,
    examples: tuple[ActionTokenExample, ...],
    index: int | None = None,
) -> _GradientBatch:
    """Reuse globally prepared examples without changing loss denominators."""

    return _GradientBatch(
        index=batch.index if index is None else int(index),
        examples=examples,
        denominator_examples=batch.denominator_examples,
        denominator_tokens=batch.denominator_tokens,
        positive_tokens=batch.positive_tokens,
        negative_tokens=batch.negative_tokens,
    )


def _gradient_batch_from_examples(
    index: int,
    examples: tuple[ActionTokenExample, ...],
    backend: ActionTokenGRPOBackend,
) -> _GradientBatch:
    if backend.loss_aggregation == "task_balanced_trajectory_mean":
        _attach_task_balance_weights(examples)
    positive = 0
    negative = 0
    for example in examples:
        counts = _example_advantage_sign_token_counts(example)
        positive += counts["positive"]
        negative += counts["negative"]
    denominator_examples = _loss_denominator_example_count(
        examples, loss_aggregation=backend.loss_aggregation
    )
    return _GradientBatch(
        index=index,
        examples=examples,
        denominator_examples=denominator_examples,
        denominator_tokens=sum(
            _example_loss_denominator_count(
                example,
                loss_aggregation=backend.loss_aggregation,
            )
            for example in examples
        ),
        positive_tokens=positive,
        negative_tokens=negative,
    )


def _attach_task_balance_weights(
    examples: Sequence[ActionTokenExample],
) -> None:
    """Make the global objective an equal mean of per-task example means."""

    counts: dict[str, int] = {}
    for example in examples:
        task = str(example.task)
        counts[task] = counts.get(task, 0) + 1
    if len(counts) < 2:
        raise ValueError(
            "task_balanced_trajectory_mean requires at least two tasks per update"
        )
    ordered_tasks = sorted(counts)
    task_indices = {task: index for index, task in enumerate(ordered_tasks)}
    groups_by_task: dict[str, set[Any]] = {task: set() for task in ordered_tasks}
    signal_groups_by_task: dict[str, set[Any]] = {task: set() for task in ordered_tasks}
    signal_examples_by_task = {task: 0 for task in ordered_tasks}
    for example in examples:
        task = str(example.task)
        group_index = example.metadata.get("group_index")
        groups_by_task[task].add(group_index)
        advantage = float(example.metadata.get("group_advantage", example.reward))
        if abs(advantage) > 1.0e-8:
            signal_groups_by_task[task].add(group_index)
            signal_examples_by_task[task] += 1

    total = len(examples)
    task_count = len(counts)
    for example in examples:
        task = str(example.task)
        count = counts[task]
        example.metadata["task_balance_weight"] = float(total) / float(
            task_count * count
        )
        example.metadata["task_balance_index"] = task_indices[task]
        example.metadata["task_balance_global_examples"] = total
        example.metadata["task_balance_task_examples"] = count
        example.metadata["task_balance_task_count"] = task_count
        example.metadata["task_balance_task_groups"] = len(groups_by_task[task])
        example.metadata["task_balance_task_signal_examples"] = signal_examples_by_task[
            task
        ]
        example.metadata["task_balance_task_signal_groups"] = len(
            signal_groups_by_task[task]
        )


def _has_prepared_token_advantages(
    examples: Sequence[ActionTokenExample],
) -> bool:
    return bool(examples) and all(
        isinstance(example.metadata.get("token_advantages"), list | tuple)
        for example in examples
    )


def _gradient_batch_from_prepared_subupdate(subupdate: Any) -> _GradientBatch:
    positive = 0
    negative = 0
    for example in subupdate.examples:
        counts = _example_advantage_sign_token_counts(example)
        positive += counts["positive"]
        negative += counts["negative"]
    return _GradientBatch(
        index=subupdate.index,
        examples=subupdate.examples,
        denominator_examples=subupdate.denominator_examples,
        denominator_tokens=subupdate.denominator_tokens,
        positive_tokens=positive,
        negative_tokens=negative,
    )


def _attach_full_update_advantages(
    examples: Sequence[ActionTokenExample],
    *,
    backend: ActionTokenGRPOBackend,
) -> None:
    """Freeze globally prepared scalar advantages before worker partitioning."""

    normalized = _example_advantages(
        list(examples),
        normalize=backend.normalize_advantages,
        scope=backend.advantage_normalization_scope,
        std_unbiased=backend.advantage_std_unbiased,
        eps=backend.advantage_epsilon,
        device="cpu",
    ).tolist()
    for example, advantage in zip(examples, normalized, strict=True):
        if backend.importance_sampling_level in ("sequence", "action_chunk"):
            # Both event-level objectives need a frozen scalar advantage,
            # not worker-local normalization or a token-level objective.
            example.metadata["group_advantage"] = float(advantage)
            example.metadata["group_advantage_prepared"] = True
            example.metadata.pop("token_advantages", None)
        else:
            token_count = len(example.logprobs or example.tokens)
            example.metadata["token_advantages"] = [float(advantage)] * token_count
            example.metadata.setdefault("token_loss_mask", [True] * token_count)


def _normalize_positions(
    values: list[float],
    *,
    positions: list[int],
    unbiased: bool,
) -> None:
    if len(positions) <= 1:
        for position in positions:
            values[position] = 0.0
        return
    selected = [values[position] for position in positions]
    mean = sum(selected) / len(selected)
    denominator = len(selected) - 1 if unbiased and len(selected) > 1 else len(selected)
    variance = sum((value - mean) ** 2 for value in selected) / denominator
    std = math.sqrt(variance)
    if std <= 1.0e-8:
        for position in positions:
            values[position] = 0.0
        return
    for position in positions:
        values[position] = (values[position] - mean) / (std + 1.0e-8)


def _partition_examples(
    examples: Sequence[ActionTokenExample],
    *,
    workers: int,
    task_keys: Sequence[str] | None = None,
) -> list[list[ActionTokenExample]]:
    if task_keys is not None:
        ordered_keys = tuple(str(value) for value in task_keys)
        buckets = {task_key: [] for task_key in ordered_keys}
        for example in examples:
            try:
                buckets[str(example.task)].append(example)
            except KeyError as exc:
                raise ValueError(
                    "Action-token example has no configured LoRA rank block: "
                    f"{example.task!r}"
                ) from exc
        missing = [task_key for task_key, rows in buckets.items() if not rows]
        if missing:
            raise ValueError(
                "Every task-partitioned update must contain every task: "
                f"missing={missing}"
            )
        return [buckets[task_key] for task_key in ordered_keys]
    partitions: list[list[ActionTokenExample]] = [[] for _ in range(workers)]
    loads = [0] * workers
    weighted = sorted(
        enumerate(examples),
        key=lambda item: (-max(1, len(item[1].tokens)), item[0]),
    )
    for _, example in weighted:
        target = min(range(workers), key=lambda index: (loads[index], index))
        partitions[target].append(example)
        loads[target] += max(1, len(example.tokens))
    return partitions


def _pi0_fast_sft_anchor_tasks_for_worker(
    config: EmbodiedExperimentConfig,
    *,
    worker_index: int,
    worker_count: int,
) -> list[int]:
    if config.policy.type != "pi0_fast":
        return []
    raw = config.policy.load_kwargs.get("sft_anchor")
    if not isinstance(raw, Mapping):
        return []
    tasks = [int(value) for value in raw.get("task_indices", [])]
    if worker_count < 1 or not 0 <= worker_index < worker_count:
        raise ValueError("Invalid SFT anchor worker geometry")
    assigned = tasks[worker_index::worker_count]
    if worker_index < min(worker_count, len(tasks)) and not assigned:
        raise RuntimeError("SFT anchor task assignment unexpectedly produced no tasks")
    return assigned


def _worker_environment(device: str) -> dict[str, str]:
    if not device.startswith("cuda:"):
        raise ValueError(
            "Distributed local training devices must use explicit cuda:N names: "
            f"{device!r}"
        )
    index = device.split(":", 1)[1]
    if not index.isdigit():
        raise ValueError(f"Invalid CUDA device: {device!r}")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = index
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env.setdefault(name, "1")
    return env


def _save_policy_snapshot(policy: Any, path: Path) -> None:
    save = getattr(policy, "save_checkpoint", None)
    if not callable(save):
        raise TypeError(
            "Distributed local training requires policy.save_checkpoint(path)"
        )
    path.mkdir(parents=True, exist_ok=True)
    save(str(path))


def _save_training_state(
    path: Path,
    *,
    optimizer: Any,
    update_step: int,
    backend_step: int,
    config_fingerprint: str | None = None,
    resume_contract_fingerprint: str | None = None,
) -> None:
    """Persist enough coordinator state for an exact optimizer continuation."""

    import numpy as np
    import torch

    if optimizer is None:
        raise RuntimeError("Cannot save training state before optimizer initialization")
    payload = {
        "schema_version": 2,
        "update_step": int(update_step),
        "backend_step": int(backend_step),
        "config_fingerprint": config_fingerprint,
        "resume_contract_fingerprint": resume_contract_fingerprint,
        "optimizer_state_dict": optimizer.state_dict(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }
    state_path = path / "art_embodied_training_state.pt"
    state_tmp = path / "art_embodied_training_state.pt.tmp"
    torch.save(payload, state_tmp)
    state_tmp.replace(state_path)
    summary_path = path / "art_embodied_training_state.json"
    summary_tmp = path / "art_embodied_training_state.json.tmp"
    summary_tmp.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "update_step": int(update_step),
                "backend_step": int(backend_step),
                "config_fingerprint": config_fingerprint,
                "resume_contract_fingerprint": resume_contract_fingerprint,
                "contains_optimizer_state": True,
                "contains_rng_state": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_tmp.replace(summary_path)


def _load_training_state(
    path: Path,
    *,
    optimizer: Any,
    expected_resume_contract_fingerprint: str | None = None,
    allow_missing_resume_contract: bool = False,
) -> dict[str, Any]:
    """Restore coordinator optimizer and RNG state from one milestone."""

    import numpy as np
    import torch

    state_path = path / "art_embodied_training_state.pt"
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") not in {1, 2}:
        raise ValueError(f"Unsupported embodied training state: {state_path}")
    required = {
        "update_step",
        "backend_step",
        "optimizer_state_dict",
        "python_random_state",
        "numpy_random_state",
        "torch_rng_state",
        "torch_cuda_rng_state_all",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError("Embodied training state is incomplete: " + ", ".join(missing))
    if expected_resume_contract_fingerprint is not None:
        saved_contract = payload.get("resume_contract_fingerprint")
        if saved_contract is None and not allow_missing_resume_contract:
            raise ValueError(
                "Embodied training state has no resume contract fingerprint"
            )
        if (
            saved_contract is not None
            and saved_contract != expected_resume_contract_fingerprint
        ):
            raise ValueError(
                "Training-state resume contract does not match the current "
                f"experiment: saved={saved_contract!r}, "
                f"current={expected_resume_contract_fingerprint!r}"
            )
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    cuda_states = payload["torch_cuda_rng_state_all"]
    if torch.cuda.is_available() and cuda_states:
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "CUDA device count differs from the saved RNG state: "
                f"saved={len(cuda_states)}, current={torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all(cuda_states)
    return payload


def _checkpoint_step(path: Path) -> int | None:
    prefix = "step_"
    if not path.name.startswith(prefix):
        return None
    suffix = path.name.removeprefix(prefix)
    return int(suffix) if suffix.isdigit() else None


def _move_policy_model(policy: Any, device: str) -> None:
    model = getattr(policy, "model", None)
    if model is None or not callable(getattr(model, "to", None)):
        return
    model.to(device)
    if device == "cpu":
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass


def _guard_file_size(path: Path, *, max_mb: int) -> None:
    size = path.stat().st_size
    limit = int(max_mb) * 1024 * 1024
    if size > limit:
        raise RuntimeError(
            f"Worker handoff exceeds configured cap: path={path}, "
            f"size={size}, max={limit}"
        )


def _stop_process(process: subprocess.Popen[str]) -> None:
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def _aggregate_worker_metrics(
    workers: Sequence[_WorkerResult],
    *,
    apply_metrics: Mapping[str, float],
    batch_index: int,
) -> dict[str, float]:
    metrics = dict(apply_metrics)
    metrics.update(
        {
            "embodied_action_token_grpo/distributed_workers": float(len(workers)),
            "embodied_action_token_grpo/distributed_batch_index": float(batch_index),
            "embodied_action_token_grpo/distributed_worker_seconds_max": max(
                worker.elapsed_seconds for worker in workers
            ),
            "embodied_action_token_grpo/distributed_worker_seconds_mean": sum(
                worker.elapsed_seconds for worker in workers
            )
            / len(workers),
            "embodied_action_token_grpo/distributed_policy_loads": float(
                sum(worker.policy_loads for worker in workers)
            ),
            "embodied_action_token_grpo/distributed_adapter_refreshes": float(
                sum(worker.adapter_refreshes for worker in workers)
            ),
        }
    )
    additive_suffixes = (
        "/examples",
        "/tokens_total",
        "/logprob_microbatches",
    )
    keys = {key for worker in workers for key in worker.metrics}
    for key in keys:
        values = [worker.metrics[key] for worker in workers if key in worker.metrics]
        if key.endswith(additive_suffixes) or key.endswith("/loss"):
            metrics[key] = sum(values)
        elif key.endswith("_min"):
            metrics[key] = min(values)
        elif key.endswith("_max"):
            metrics[key] = max(values)
        else:
            metrics[key] = sum(values) / len(values)
        if key.endswith("_seconds"):
            # Preserve the slowest worker, not just the average, for bottlenecks.
            metrics[f"{key}_max"] = max(values)
    metrics.update(apply_metrics)
    return metrics


def _raise_for_worker_alignment_rejection(
    workers: Sequence[_WorkerResult],
    *,
    batch_index: int,
) -> None:
    rejected_workers = [
        worker
        for worker in workers
        if worker.metrics.get(
            "embodied_action_token_grpo/optimizer_step_skipped_logprob_misalignment",
            0.0,
        )
        > 0.0
    ]
    if rejected_workers:
        rejected = [worker.index for worker in rejected_workers]
        diagnostic_names = (
            "alignment_approx_kl_mean",
            "alignment_approx_kl_abs_mean",
            "alignment_ratio_mean",
            "alignment_ratio_min",
            "alignment_ratio_max",
            "alignment_ratio_outside_tolerance_fraction",
        )
        diagnostics = {
            worker.index: {
                name: worker.metrics[f"embodied_action_token_grpo/{name}"]
                for name in diagnostic_names
                if f"embodied_action_token_grpo/{name}" in worker.metrics
            }
            for worker in rejected_workers
        }
        raise RuntimeError(
            "Distributed action-token alignment guard rejected a worker shard; "
            "the coordinator will not apply a partial gradient: "
            f"subupdate={batch_index}, workers={rejected}, "
            f"diagnostics={diagnostics}"
        )


def _aggregate_batch_metrics(
    batches: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    if not batches:
        return {}
    result: dict[str, float] = {}
    keys = {key for batch in batches for key in batch}
    for key in keys:
        values = [float(batch[key]) for batch in batches if key in batch]
        if key.endswith(
            (
                "/optimizer_state_entries_after",
                "/optimizer_state_step_min_after",
                "/optimizer_state_step_max_after",
                "/optimizer_state_step_mean_after",
            )
        ):
            # These metrics describe coordinator state, not a minibatch
            # distribution. Publish the state after the final subupdate.
            result[key] = values[-1]
        elif key.endswith("/optimizer_state_entries_before"):
            result[key] = values[0]
        elif key.endswith("/optimizer_state_step_min_before"):
            result[key] = min(values)
        elif key.endswith("/optimizer_state_step_max_before"):
            result[key] = max(values)
        elif key.endswith("_min"):
            result[key] = min(values)
        elif key.endswith("_max"):
            result[key] = max(values)
        elif key.endswith(
            (
                "/optimizer_step_completed",
                "/distributed_policy_loads",
                "/distributed_adapter_refreshes",
            )
        ):
            result[key] = sum(values)
        else:
            result[key] = sum(values) / len(values)
        if (
            len(values) > 1
            and key.rsplit("/", 1)[-1] in _SUBUPDATE_BOUNDARY_METRIC_SUFFIXES
        ):
            result[f"{key}_first_subupdate"] = values[0]
            result[f"{key}_last_subupdate"] = values[-1]
    return result


def _expose_coordinator_gspo_metrics(metrics: dict[str, float]) -> None:
    """Publish final coordinator metrics as GSPO without duplicate GRPO panels."""

    grpo_prefix = "embodied_action_token_grpo/"
    gspo_prefix = "embodied_action_token_gspo/"
    aliases = {
        gspo_prefix + key.removeprefix(grpo_prefix): value
        for key, value in metrics.items()
        if key.startswith(grpo_prefix)
    }
    for key in tuple(metrics):
        if key.startswith(grpo_prefix):
            del metrics[key]
    metrics.update(aliases)
