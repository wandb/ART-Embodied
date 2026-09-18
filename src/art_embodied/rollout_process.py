"""Process-isolated rollout actors for local multi-GPU collection."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Literal, Protocol, TextIO

from .config import EmbodiedExperimentConfig
from .experiment import EmbodiedScenario, RolloutContext
from .trajectories import EmbodiedTrajectory
from .utils import worker_python_command


@dataclass(frozen=True, slots=True)
class RolloutActorProcessContext:
    """Stable process identity passed to an importable actor factory."""

    worker_index: int
    configured_device: str
    local_device: str


class PolicySnapshotProvider(Protocol):
    """Publish policy state for rollout workers without owning rollout logic."""

    def save_snapshot(self, path: Path, *, update: int) -> None: ...

    def offload(self) -> None: ...

    def restore(self, device: str) -> None: ...


class LocalPolicySnapshotProvider:
    """Adapt a coordinator-owned policy to the rollout snapshot contract."""

    def __init__(self, policy: Any) -> None:
        self.policy = policy

    def save_snapshot(self, path: Path, *, update: int) -> None:
        del update
        save = getattr(self.policy, "save_checkpoint", None)
        if not callable(save):
            raise TypeError(
                "checkpoint-synchronized rollout actors require either "
                "policy.save_checkpoint(path) or a PolicySnapshotProvider"
            )
        save(str(path))

    def offload(self) -> None:
        _move_policy_model(self.policy, "cpu")

    def restore(self, device: str) -> None:
        _move_policy_model(self.policy, device)


@dataclass(slots=True)
class _ActorWorker:
    index: int
    configured_device: str
    process: subprocess.Popen[str]
    stdin: TextIO
    stdout: TextIO
    stderr: TextIO
    worker_dir: Path
    ready_path: Path


@dataclass(slots=True)
class _InferenceWorker:
    index: int
    configured_device: str
    process: subprocess.Popen[str]
    stdin: TextIO
    stdout: TextIO
    stderr: TextIO
    worker_dir: Path
    ready_path: Path
    socket_path: Path


class _RolloutPhaseView:
    def __init__(
        self,
        pool: "LocalProcessRolloutPool",
        phase: Literal["train", "eval"],
    ) -> None:
        self.pool = pool
        self.phase = phase

    @property
    def supports_group_rollout(self) -> bool:
        return self.pool.execution.group_batching

    @property
    def lifecycle_metrics(self) -> dict[str, float]:
        """Return timing for the most recently completed rollout lifecycle."""

        return dict(self.pool.last_lifecycle_metrics)

    async def prepare_update(self, *, update: int) -> None:
        await self.pool.prepare_update(update=update)

    async def finish_update(self, *, update: int) -> None:
        await self.pool.finish_update(update=update)

    async def __call__(
        self,
        scenario: EmbodiedScenario,
        context: RolloutContext,
    ) -> EmbodiedTrajectory:
        return await self.pool.rollout(
            scenario,
            context,
            phase=self.phase,
        )

    async def rollout_group(
        self,
        scenario: EmbodiedScenario,
        contexts: tuple[RolloutContext, ...],
    ) -> list[EmbodiedTrajectory]:
        return await self.pool.rollout_group(
            scenario,
            contexts,
            phase=self.phase,
        )


class LocalProcessRolloutPool:
    """Run episodes in persistent process/device-isolated actors.

    The configured ``actor_factory`` is imported inside every child process.
    It receives ``config`` and :class:`RolloutActorProcessContext`, constructs
    the native LeRobot policy, processor pipeline, and environment there, and
    returns an actor exposing ``prepare_update`` and ``rollout``. This avoids
    pickling CUDA objects or reimplementing model-family logic in ART.
    """

    def __init__(
        self,
        *,
        config: EmbodiedExperimentConfig,
        policy: Any | None = None,
        snapshot_provider: PolicySnapshotProvider | None = None,
    ) -> None:
        execution = config.runtime.rollout_execution
        if execution.mode != "local_process":
            raise ValueError(
                "LocalProcessRolloutPool requires "
                "runtime.rollout_execution.mode='local_process'"
            )
        self.config = config
        if snapshot_provider is not None and policy is not None:
            raise ValueError("Pass policy or snapshot_provider, not both")
        if snapshot_provider is None:
            if policy is None:
                raise ValueError("policy or snapshot_provider is required")
            snapshot_provider = LocalPolicySnapshotProvider(policy)
        self.snapshot_provider = snapshot_provider
        self.execution = execution
        max_concurrent_rollouts = execution.actor_kwargs.get(
            "max_concurrent_rollouts",
            config.rollout.workers,
        )
        if (
            isinstance(max_concurrent_rollouts, bool)
            or not isinstance(max_concurrent_rollouts, int)
            or not 1 <= max_concurrent_rollouts <= config.rollout.workers
        ):
            raise ValueError(
                "actor_kwargs.max_concurrent_rollouts must be an integer in "
                f"[1, {config.rollout.workers}]"
            )
        self.max_concurrent_rollouts = max_concurrent_rollouts
        self._rollout_slots = asyncio.Semaphore(max_concurrent_rollouts)
        self.workers: list[_ActorWorker] = []
        self.inference_workers: list[_InferenceWorker] = []
        self.run_dir: Path | None = None
        self.available: asyncio.Queue[_ActorWorker] | None = None
        self.prepared_update: int | None = None
        self.job_index = 0
        self._job_index_lock = threading.Lock()
        self._handoff_metrics_lock = threading.Lock()
        self._trajectory_handoff_bytes = 0
        self._trajectory_handoff_files = 0
        self._trajectory_handoff_count = 0
        self._policy_offloaded = False
        self._inference_offloaded = False
        self._actors_offloaded = False
        self._lifecycle_lock = asyncio.Lock()
        self.last_lifecycle_metrics: dict[str, float] = {}

    def for_phase(self, phase: Literal["train", "eval"]) -> _RolloutPhaseView:
        return _RolloutPhaseView(self, phase)

    async def prepare_update(self, *, update: int) -> None:
        async with self._lifecycle_lock:
            if self.prepared_update == update and self.workers:
                return
            prepare_started = time.perf_counter()
            metrics: dict[str, float] = {}
            with self._handoff_metrics_lock:
                self._trajectory_handoff_bytes = 0
                self._trajectory_handoff_files = 0
                self._trajectory_handoff_count = 0
            if self.workers and self.execution.lifecycle == "per_update":
                await asyncio.to_thread(self._stop_sync)
            if not self.workers:
                startup_started = time.perf_counter()
                await asyncio.to_thread(self._start_sync)
                metrics["worker_startup_seconds"] = (
                    time.perf_counter() - startup_started
                )
                self.available = asyncio.Queue()
                for worker in self.workers:
                    self.available.put_nowait(worker)

            snapshot_started = time.perf_counter()
            snapshot = await asyncio.to_thread(self._save_snapshot_sync, update)
            metrics["snapshot_seconds"] = time.perf_counter() - snapshot_started
            try:
                if (
                    self.execution.lifecycle == "cpu_offload"
                    and not self._policy_offloaded
                ):
                    coordinator_offload_started = time.perf_counter()
                    await asyncio.to_thread(self.snapshot_provider.offload)
                    metrics["coordinator_offload_seconds"] = (
                        time.perf_counter() - coordinator_offload_started
                    )
                    self._policy_offloaded = True
                if self.inference_workers:
                    inference_prepare_started = time.perf_counter()
                    await asyncio.gather(
                        *(
                            asyncio.to_thread(
                                self._prepare_inference_worker_sync,
                                worker,
                                update,
                                snapshot,
                            )
                            for worker in self.inference_workers
                        )
                    )
                    metrics["inference_prepare_seconds"] = (
                        time.perf_counter() - inference_prepare_started
                    )
                    self._inference_offloaded = False
                if (
                    self.execution.lifecycle == "cpu_offload"
                    and self.execution.inference_mode == "embedded"
                    and self._actors_offloaded
                ):
                    actor_restore_started = time.perf_counter()
                    await asyncio.gather(
                        *(
                            asyncio.to_thread(self._restore_worker_sync, worker)
                            for worker in self.workers
                        )
                    )
                    metrics["actor_restore_seconds"] = (
                        time.perf_counter() - actor_restore_started
                    )
                    self._actors_offloaded = False
                actor_prepare_started = time.perf_counter()
                await asyncio.gather(
                    *(
                        asyncio.to_thread(
                            self._prepare_worker_sync,
                            worker,
                            update,
                            snapshot,
                        )
                        for worker in self.workers
                    )
                )
                metrics["actor_prepare_seconds"] = (
                    time.perf_counter() - actor_prepare_started
                )
            except Exception:
                await asyncio.to_thread(self._stop_sync)
                raise
            self.prepared_update = update
            if not self.config.runtime.keep_worker_handoffs:
                await asyncio.to_thread(self._prune_snapshots_sync, snapshot)
            metrics["prepare_seconds"] = time.perf_counter() - prepare_started
            self.last_lifecycle_metrics = metrics

    async def finish_update(self, *, update: int) -> None:
        del update
        if self.execution.lifecycle == "persistent":
            return
        async with self._lifecycle_lock:
            finish_started = time.perf_counter()
            if self.execution.lifecycle == "per_update":
                await asyncio.to_thread(self._stop_sync)
                self.last_lifecycle_metrics["worker_shutdown_seconds"] = (
                    time.perf_counter() - finish_started
                )
                self.last_lifecycle_metrics["finish_seconds"] = (
                    time.perf_counter() - finish_started
                )
                return
            dead_workers = [
                worker for worker in self.workers if worker.process.poll() is not None
            ]
            dead_inference_workers = [
                worker
                for worker in self.inference_workers
                if worker.process.poll() is not None
            ]
            if dead_workers or dead_inference_workers:
                # Do not send lifecycle commands to a dead pipe. Tear down the
                # remaining actors so the next phase starts a complete pool.
                await asyncio.to_thread(
                    self._archive_failed_worker_logs_sync,
                    [*dead_workers, *dead_inference_workers],
                )
                await asyncio.to_thread(self._stop_sync)
                self.last_lifecycle_metrics["unexpected_worker_exits"] = float(
                    len(dead_workers) + len(dead_inference_workers)
                )
                self.last_lifecycle_metrics["worker_shutdown_seconds"] = (
                    time.perf_counter() - finish_started
                )
                self.last_lifecycle_metrics["finish_seconds"] = (
                    time.perf_counter() - finish_started
                )
                return
            if self.inference_workers and not self._inference_offloaded:
                inference_offload_started = time.perf_counter()
                await asyncio.gather(
                    *(
                        asyncio.to_thread(
                            self._offload_inference_worker_sync,
                            worker,
                        )
                        for worker in self.inference_workers
                    )
                )
                self.last_lifecycle_metrics["inference_offload_seconds"] = (
                    time.perf_counter() - inference_offload_started
                )
                self._inference_offloaded = True
            if (
                self.execution.inference_mode == "embedded"
                and not self._actors_offloaded
            ):
                actor_offload_started = time.perf_counter()
                await asyncio.gather(
                    *(
                        asyncio.to_thread(self._offload_worker_sync, worker)
                        for worker in self.workers
                    )
                )
                self.last_lifecycle_metrics["actor_offload_seconds"] = (
                    time.perf_counter() - actor_offload_started
                )
                self._actors_offloaded = True
            if self._policy_offloaded:
                coordinator_restore_started = time.perf_counter()
                await asyncio.to_thread(
                    self.snapshot_provider.restore,
                    self.config.runtime.training_devices[0],
                )
                self.last_lifecycle_metrics["coordinator_restore_seconds"] = (
                    time.perf_counter() - coordinator_restore_started
                )
                self._policy_offloaded = False
            # The next phase may use the same numeric update after training.
            # Force adapter refresh instead of treating the CPU-offloaded model
            # as already prepared.
            self.prepared_update = None
            self.last_lifecycle_metrics["finish_seconds"] = (
                time.perf_counter() - finish_started
            )

    async def rollout(
        self,
        scenario: EmbodiedScenario,
        context: RolloutContext,
        *,
        phase: Literal["train", "eval"],
    ) -> EmbodiedTrajectory:
        if self.prepared_update is None or not self.workers or self.available is None:
            raise RuntimeError(
                "Local process rollout actors are not synchronized; "
                "prepare_update must complete before collection"
            )
        async with self._rollout_slots:
            worker = await self.available.get()
            try:
                trajectory = await asyncio.to_thread(
                    self._rollout_sync,
                    worker,
                    scenario,
                    context,
                    phase,
                )
            finally:
                # A native simulator or CUDA failure can terminate one actor. Never
                # recycle its dead pipe into the queue for every remaining episode.
                if worker.process.poll() is None:
                    self.available.put_nowait(worker)
        trajectory.metadata.setdefault("rollout_execution", "local_process")
        trajectory.metadata.setdefault("rollout_actor_index", worker.index)
        trajectory.metadata.setdefault("rollout_actor_device", worker.configured_device)
        trajectory.metadata.setdefault("rollout_policy_update", self.prepared_update)
        return trajectory

    async def rollout_group(
        self,
        scenario: EmbodiedScenario,
        contexts: tuple[RolloutContext, ...],
        *,
        phase: Literal["train", "eval"],
    ) -> list[EmbodiedTrajectory]:
        if not contexts:
            raise ValueError("rollout_group requires at least one context")
        if self.prepared_update is None or not self.workers or self.available is None:
            raise RuntimeError(
                "Local process rollout actors are not synchronized; "
                "prepare_update must complete before collection"
            )
        async with self._rollout_slots:
            worker = await self.available.get()
            try:
                trajectories = await asyncio.to_thread(
                    self._rollout_group_sync,
                    worker,
                    scenario,
                    contexts,
                    phase,
                )
            finally:
                if worker.process.poll() is None:
                    self.available.put_nowait(worker)
        for trajectory in trajectories:
            trajectory.metadata.setdefault("rollout_execution", "local_process")
            trajectory.metadata.setdefault("rollout_actor_index", worker.index)
            trajectory.metadata.setdefault(
                "rollout_actor_device", worker.configured_device
            )
            trajectory.metadata.setdefault(
                "rollout_policy_update", self.prepared_update
            )
            trajectory.metadata.setdefault("rollout_group_batch", True)
        return trajectories

    async def close(self) -> None:
        async with self._lifecycle_lock:
            await asyncio.to_thread(self._stop_sync)

    def _start_sync(self) -> None:
        handoff_root = self.config.runtime.worker_handoff_dir.expanduser()
        handoff_root.mkdir(parents=True, exist_ok=True)
        self.run_dir = Path(
            tempfile.mkdtemp(prefix="rollout-actors-", dir=handoff_root)
        )
        try:
            if self._time_shares_training_device():
                self.snapshot_provider.offload()
                self._policy_offloaded = True
            if self.execution.inference_mode == "batched_server":
                inference_index = 0
                for device in self.config.runtime.rollout_devices:
                    for _ in range(self.execution.inference_replicas_per_device):
                        self.inference_workers.append(
                            self._start_inference_worker_sync(
                                index=inference_index,
                                configured_device=device,
                            )
                        )
                        inference_index += 1
                self._wait_inference_ready_sync()
            worker_index = 0
            for device in self.config.runtime.rollout_devices:
                device_inference_workers = [
                    worker
                    for worker in self.inference_workers
                    if worker.configured_device == device
                ]
                for actor_index in range(self.execution.actors_per_device):
                    inference_socket = (
                        str(
                            device_inference_workers[
                                actor_index % len(device_inference_workers)
                            ].socket_path
                        )
                        if device_inference_workers
                        else None
                    )
                    self.workers.append(
                        self._start_worker_sync(
                            index=worker_index,
                            configured_device=device,
                            inference_socket=inference_socket,
                        )
                    )
                    worker_index += 1
            self._wait_ready_sync()
            self._actors_offloaded = False
        except Exception:
            self._stop_sync()
            raise

    def _start_worker_sync(
        self,
        *,
        index: int,
        configured_device: str,
        inference_socket: str | None,
    ) -> _ActorWorker:
        assert self.run_dir is not None
        worker_dir = self.run_dir / f"worker-{index:03d}"
        worker_dir.mkdir(parents=True)
        ready_path = worker_dir / "ready.json"
        bootstrap = worker_dir / "bootstrap.json"
        local_device = (
            "cuda:0" if configured_device.startswith("cuda:") else configured_device
        )
        bootstrap.write_text(
            json.dumps(
                {
                    "config": self.config.model_dump(mode="json"),
                    "actor_factory": self.execution.actor_factory,
                    "worker_index": index,
                    "configured_device": configured_device,
                    "local_device": local_device,
                    "inference_socket": inference_socket,
                    "ready_path": str(ready_path),
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
                        self.execution.actor_python_executable
                        or self.config.runtime.worker_python_executable
                    ),
                    module="art_embodied.rollout_worker",
                    spec_path=bootstrap,
                ),
                cwd=str(Path.cwd()),
                env=_worker_environment(configured_device),
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
            raise RuntimeError("Rollout actor has no stdin pipe")
        return _ActorWorker(
            index=index,
            configured_device=configured_device,
            process=process,
            stdin=process.stdin,
            stdout=stdout,
            stderr=stderr,
            worker_dir=worker_dir,
            ready_path=ready_path,
        )

    def _start_inference_worker_sync(
        self,
        *,
        index: int,
        configured_device: str,
    ) -> _InferenceWorker:
        assert self.run_dir is not None
        worker_dir = self.run_dir / f"inference-{index:03d}"
        worker_dir.mkdir(parents=True)
        ready_path = worker_dir / "ready.json"
        # A short, owner-only directory protects pickle IPC regardless of umask
        # while staying within the Linux AF_UNIX path length limit.
        socket_dir = Path(tempfile.mkdtemp(prefix="artemb-"))
        socket_path = socket_dir / "policy.sock"
        bootstrap = worker_dir / "bootstrap.json"
        local_device = (
            "cuda:0" if configured_device.startswith("cuda:") else configured_device
        )
        bootstrap.write_text(
            json.dumps(
                {
                    "config": self.config.model_dump(mode="json"),
                    "inference_factory": self.execution.inference_factory,
                    "server_index": index,
                    "configured_device": configured_device,
                    "local_device": local_device,
                    "ready_path": str(ready_path),
                    "socket_path": str(socket_path),
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
                        self.execution.inference_python_executable
                        or self.config.runtime.worker_python_executable
                    ),
                    module="art_embodied.rollout_inference_worker",
                    spec_path=bootstrap,
                ),
                cwd=str(Path.cwd()),
                env=_worker_environment(configured_device),
                text=True,
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
                bufsize=1,
            )
        except Exception:
            stdout.close()
            stderr.close()
            socket_dir.rmdir()
            raise
        if process.stdin is None:
            _stop_process(process)
            stdout.close()
            stderr.close()
            socket_path.unlink(missing_ok=True)
            socket_dir.rmdir()
            raise RuntimeError("Batched inference server has no stdin pipe")
        return _InferenceWorker(
            index=index,
            configured_device=configured_device,
            process=process,
            stdin=process.stdin,
            stdout=stdout,
            stderr=stderr,
            worker_dir=worker_dir,
            ready_path=ready_path,
            socket_path=socket_path,
        )

    def _wait_ready_sync(self) -> None:
        deadline = time.monotonic() + self.execution.startup_timeout_seconds
        pending = {worker.index: worker for worker in self.workers}
        while pending:
            for index, worker in list(pending.items()):
                if worker.ready_path.exists():
                    payload = json.loads(worker.ready_path.read_text(encoding="utf-8"))
                    if not payload.get("ok"):
                        raise RuntimeError(
                            f"Rollout actor {index} failed during startup: {payload}"
                        )
                    pending.pop(index)
                    continue
                return_code = worker.process.poll()
                if return_code is not None:
                    raise RuntimeError(self._worker_failure(worker, return_code))
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Rollout actors timed out during startup: "
                    + ", ".join(str(index) for index in sorted(pending))
                )
            time.sleep(0.05)

    def _wait_inference_ready_sync(self) -> None:
        deadline = time.monotonic() + self.execution.startup_timeout_seconds
        pending = {worker.index: worker for worker in self.inference_workers}
        while pending:
            for index, worker in list(pending.items()):
                if worker.ready_path.exists():
                    payload = json.loads(worker.ready_path.read_text(encoding="utf-8"))
                    if not payload.get("ok"):
                        raise RuntimeError(
                            f"Inference server {index} failed during startup: {payload}"
                        )
                    pending.pop(index)
                    continue
                return_code = worker.process.poll()
                if return_code is not None:
                    raise RuntimeError(self._worker_failure(worker, return_code))
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Inference servers timed out during startup: "
                    + ", ".join(str(index) for index in sorted(pending))
                )
            time.sleep(0.05)

    def _save_snapshot_sync(self, update: int) -> Path:
        assert self.run_dir is not None
        snapshot = self.run_dir / "snapshots" / f"update-{update:06d}"
        ready = snapshot / ".art-embodied-snapshot-ready.json"
        if ready.is_file():
            payload = json.loads(ready.read_text(encoding="utf-8"))
            if int(payload.get("update", -1)) != update:
                raise RuntimeError(
                    "Rollout snapshot readiness marker has the wrong policy "
                    f"version: path={snapshot}, expected={update}, "
                    f"found={payload.get('update')!r}"
                )
            _guard_size(
                snapshot,
                max_mb=self.config.runtime.max_worker_handoff_mb,
            )
            return snapshot
        if snapshot.exists():
            # A previous save was interrupted before publishing its readiness
            # marker. No rollout can be using it while the lifecycle lock is held.
            shutil.rmtree(snapshot)
        snapshot.mkdir(parents=True, exist_ok=False)
        self.snapshot_provider.save_snapshot(snapshot, update=update)
        _guard_size(
            snapshot,
            max_mb=self.config.runtime.max_worker_handoff_mb,
        )
        ready.write_text(
            json.dumps({"schema_version": 1, "update": update}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return snapshot

    def _prepare_worker_sync(
        self,
        worker: _ActorWorker,
        update: int,
        snapshot: Path,
    ) -> None:
        result = worker.worker_dir / f"prepare-{update:06d}.json"
        self._send_command_sync(
            worker,
            {
                "op": "prepare",
                "update": update,
                "policy_snapshot": str(snapshot),
                "result_path": str(result),
            },
            result_path=result,
        )

    def _prepare_inference_worker_sync(
        self,
        worker: _InferenceWorker,
        update: int,
        snapshot: Path,
    ) -> None:
        result = worker.worker_dir / f"prepare-{update:06d}.json"
        self._send_command_sync(
            worker,
            {
                "op": "prepare",
                "update": update,
                "policy_snapshot": str(snapshot),
                "result_path": str(result),
            },
            result_path=result,
        )

    def _offload_inference_worker_sync(self, worker: _InferenceWorker) -> None:
        result = worker.worker_dir / f"offload-{self.job_index:08d}.json"
        self._send_command_sync(
            worker,
            {
                "op": "offload",
                "result_path": str(result),
            },
            result_path=result,
        )

    def _offload_worker_sync(self, worker: _ActorWorker) -> None:
        result = worker.worker_dir / f"offload-{self.job_index:08d}.json"
        self._send_command_sync(
            worker,
            {"op": "offload", "result_path": str(result)},
            result_path=result,
        )

    def _restore_worker_sync(self, worker: _ActorWorker) -> None:
        result = worker.worker_dir / f"restore-{self.job_index:08d}.json"
        self._send_command_sync(
            worker,
            {"op": "restore", "result_path": str(result)},
            result_path=result,
        )

    def _rollout_sync(
        self,
        worker: _ActorWorker | _InferenceWorker,
        scenario: EmbodiedScenario,
        context: RolloutContext,
        phase: Literal["train", "eval"],
    ) -> EmbodiedTrajectory:
        with self._job_index_lock:
            job = self.job_index
            self.job_index += 1
        job_dir = worker.worker_dir / f"job-{job:08d}"
        job_dir.mkdir(parents=True)
        request_path = job_dir / "request.pkl"
        trajectory_path = job_dir / "trajectory.pkl"
        result_path = job_dir / "result.json"
        with request_path.open("wb") as handle:
            pickle.dump(
                {
                    "scenario": scenario.model_dump(mode="python"),
                    "context": {
                        "update": context.update,
                        "group_index": context.group_index,
                        "attempt_index": context.attempt_index,
                        "environment_seed": context.environment_seed,
                        "policy_seed": context.policy_seed,
                        "config_fingerprint": context.config_fingerprint,
                    },
                    "phase": phase,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        _guard_size(
            request_path,
            max_mb=self.config.runtime.max_worker_handoff_mb,
        )
        self._send_command_sync(
            worker,
            {
                "op": "rollout",
                "request_path": str(request_path),
                "trajectory_path": str(trajectory_path),
                "result_path": str(result_path),
            },
            result_path=result_path,
        )
        _guard_size(
            trajectory_path,
            max_mb=self.config.runtime.max_worker_handoff_mb,
        )
        self._record_trajectory_handoff(
            byte_count=trajectory_path.stat().st_size,
            trajectory_count=1,
        )
        with trajectory_path.open("rb") as handle:
            trajectory = pickle.load(handle)
        if not isinstance(trajectory, EmbodiedTrajectory):
            raise TypeError(
                "Rollout actor returned an invalid trajectory type: "
                f"{type(trajectory).__name__}"
            )
        if not self.config.runtime.keep_worker_handoffs:
            shutil.rmtree(job_dir, ignore_errors=True)
        return trajectory

    def _rollout_group_sync(
        self,
        worker: _ActorWorker | _InferenceWorker,
        scenario: EmbodiedScenario,
        contexts: tuple[RolloutContext, ...],
        phase: Literal["train", "eval"],
    ) -> list[EmbodiedTrajectory]:
        with self._job_index_lock:
            job = self.job_index
            self.job_index += 1
        job_dir = worker.worker_dir / f"job-{job:08d}"
        job_dir.mkdir(parents=True)
        request_path = job_dir / "request.pkl"
        trajectories_path = job_dir / "trajectories.pkl"
        result_path = job_dir / "result.json"
        with request_path.open("wb") as handle:
            pickle.dump(
                {
                    "scenario": scenario.model_dump(mode="python"),
                    "contexts": [
                        {
                            "update": context.update,
                            "group_index": context.group_index,
                            "attempt_index": context.attempt_index,
                            "environment_seed": context.environment_seed,
                            "policy_seed": context.policy_seed,
                            "config_fingerprint": context.config_fingerprint,
                        }
                        for context in contexts
                    ],
                    "phase": phase,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        _guard_size(request_path, max_mb=self.config.runtime.max_worker_handoff_mb)
        self._send_command_sync(
            worker,
            {
                "op": "rollout_group",
                "request_path": str(request_path),
                "trajectories_path": str(trajectories_path),
                "result_path": str(result_path),
            },
            result_path=result_path,
        )
        _guard_size(
            trajectories_path,
            max_mb=self.config.runtime.max_worker_handoff_mb,
        )
        self._record_trajectory_handoff(
            byte_count=trajectories_path.stat().st_size,
            trajectory_count=len(contexts),
        )
        with trajectories_path.open("rb") as handle:
            trajectories = pickle.load(handle)
        if not isinstance(trajectories, list) or any(
            not isinstance(item, EmbodiedTrajectory) for item in trajectories
        ):
            raise TypeError("Rollout actor returned an invalid trajectory group")
        if len(trajectories) != len(contexts):
            raise RuntimeError(
                "Rollout actor returned the wrong trajectory-group size: "
                f"returned={len(trajectories)}, expected={len(contexts)}"
            )
        if not self.config.runtime.keep_worker_handoffs:
            shutil.rmtree(job_dir, ignore_errors=True)
        return trajectories

    def _record_trajectory_handoff(
        self,
        *,
        byte_count: int,
        trajectory_count: int,
    ) -> None:
        with self._handoff_metrics_lock:
            self._trajectory_handoff_bytes += int(byte_count)
            self._trajectory_handoff_files += 1
            self._trajectory_handoff_count += int(trajectory_count)
            total_bytes = self._trajectory_handoff_bytes
            total_files = self._trajectory_handoff_files
            total_trajectories = self._trajectory_handoff_count
            self.last_lifecycle_metrics.update(
                {
                    "trajectory_handoff_bytes": float(total_bytes),
                    "trajectory_handoff_mb": float(total_bytes / (1024 * 1024)),
                    "trajectory_handoff_files": float(total_files),
                    "trajectory_handoff_trajectories": float(total_trajectories),
                    "trajectory_handoff_bytes_mean": float(
                        total_bytes / total_trajectories
                    ),
                }
            )

    def _send_command_sync(
        self,
        worker: _ActorWorker | _InferenceWorker,
        command: dict[str, Any],
        *,
        result_path: Path,
    ) -> dict[str, Any]:
        (worker.worker_dir / "last-command.json").write_text(
            json.dumps(command, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return_code = worker.process.poll()
        if return_code is not None:
            self._archive_failed_worker_logs_sync([worker])
            raise RuntimeError(self._worker_failure(worker, return_code))
        try:
            worker.stdin.write(json.dumps(command, sort_keys=True) + "\n")
            worker.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            # Give the OS a moment to publish the child's final return code and
            # stderr before constructing the durable diagnostic.
            try:
                return_code = worker.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                return_code = worker.process.poll()
            self._archive_failed_worker_logs_sync([worker])
            raise RuntimeError(self._worker_failure(worker, return_code)) from exc
        deadline = time.monotonic() + self.execution.request_timeout_seconds
        while not result_path.exists():
            return_code = worker.process.poll()
            if return_code is not None:
                self._archive_failed_worker_logs_sync([worker])
                raise RuntimeError(self._worker_failure(worker, return_code))
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Rollout actor {worker.index} timed out for {command['op']!r}"
                )
            time.sleep(0.02)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if not payload.get("ok"):
            raise RuntimeError(
                f"Rollout actor {worker.index} failed: "
                f"{payload.get('error_type')}: {payload.get('error')}\n"
                f"{payload.get('traceback', '')}"
            )
        return payload

    def _prune_snapshots_sync(self, current: Path) -> None:
        snapshots = current.parent
        for path in snapshots.iterdir():
            if path != current:
                shutil.rmtree(path, ignore_errors=True)

    def _stop_sync(self) -> None:
        unexpectedly_dead = [
            worker
            for worker in [*self.workers, *self.inference_workers]
            if worker.process.poll() is not None
        ]
        self._archive_failed_worker_logs_sync(unexpectedly_dead)
        for worker in self.workers:
            if worker.process.poll() is None:
                try:
                    worker.stdin.write('{"op":"shutdown"}\n')
                    worker.stdin.flush()
                    worker.process.wait(timeout=30)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    _stop_process(worker.process)
            _close_worker_streams(worker)
        self.workers.clear()
        for worker in self.inference_workers:
            if worker.process.poll() is None:
                try:
                    worker.stdin.write('{"op":"shutdown"}\n')
                    worker.stdin.flush()
                    worker.process.wait(timeout=30)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    _stop_process(worker.process)
            _close_worker_streams(worker)
            worker.socket_path.unlink(missing_ok=True)
            worker.socket_path.parent.rmdir()
        self.inference_workers.clear()
        self._inference_offloaded = False
        self._actors_offloaded = False
        self.available = None
        self.prepared_update = None
        if self.run_dir is not None and not self.config.runtime.keep_worker_handoffs:
            shutil.rmtree(self.run_dir, ignore_errors=True)
        self.run_dir = None
        if self._policy_offloaded:
            self.snapshot_provider.restore(self.config.runtime.training_devices[0])
            self._policy_offloaded = False

    def _time_shares_training_device(self) -> bool:
        if self.execution.lifecycle not in {"per_update", "cpu_offload"}:
            return False
        return bool(
            set(self.config.runtime.rollout_devices).intersection(
                self.config.runtime.training_devices
            )
        )

    @staticmethod
    def _worker_failure(
        worker: _ActorWorker | _InferenceWorker,
        return_code: int | None,
    ) -> str:
        worker.stderr.flush()
        stderr_path = worker.worker_dir / "stderr.log"
        stderr = (
            stderr_path.read_text(encoding="utf-8", errors="replace")
            if stderr_path.exists()
            else ""
        )
        return (
            f"Rollout actor {worker.index} exited with code {return_code}.\n"
            f"{stderr[-8000:]}"
        )

    def _archive_failed_worker_logs_sync(
        self,
        workers: list[_ActorWorker | _InferenceWorker],
    ) -> None:
        if not workers or self.run_dir is None:
            return
        destination = (
            Path(self.config.storage.output_dir).expanduser()
            / "worker-failures"
            / self.run_dir.name
        )
        for worker in workers:
            worker_destination = destination / worker.worker_dir.name
            worker_destination.mkdir(parents=True, exist_ok=True)
            for name in (
                "bootstrap.json",
                "last-command.json",
                "ready.json",
                "stdout.log",
                "stderr.log",
            ):
                source = worker.worker_dir / name
                if source.is_file():
                    shutil.copy2(source, worker_destination / name)
            (worker_destination / "exit.json").write_text(
                json.dumps(
                    {
                        "configured_device": worker.configured_device,
                        "return_code": worker.process.poll(),
                        "worker_index": worker.index,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )


def _worker_environment(device: str) -> dict[str, str]:
    env = dict(os.environ)
    if device.startswith("cuda:"):
        index = device.split(":", 1)[1]
        if not index.isdigit():
            raise ValueError(f"Invalid CUDA rollout device: {device!r}")
        env["CUDA_VISIBLE_DEVICES"] = index
        # MuJoCo's EGL backend enumerates every physical EGL device and does
        # not use CUDA_VISIBLE_DEVICES for selection. Pin rendering alongside
        # the actor's model instead of letting every process choose GPU zero.
        env["MUJOCO_EGL_DEVICE_ID"] = index
    elif device != "cpu":
        raise ValueError(
            f"Local rollout devices must use explicit 'cuda:N' or 'cpu': {device!r}"
        )
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env.setdefault(name, "1")
    return env


def _guard_size(path: Path, *, max_mb: int) -> None:
    if path.is_dir():
        size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    else:
        size = path.stat().st_size
    limit = int(max_mb) * 1024 * 1024
    if size > limit:
        raise RuntimeError(
            f"Rollout worker handoff exceeds configured cap: path={path}, "
            f"size={size}, max={limit}"
        )


def _stop_process(process: subprocess.Popen[str]) -> None:
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def _close_worker_streams(worker: _ActorWorker | _InferenceWorker) -> None:
    for stream in (worker.stdin, worker.stdout, worker.stderr):
        try:
            stream.close()
        except OSError:
            pass


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
