"""Lifecycle client for the process-isolated RoboCasa runtime."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any

from art_embodied.inference_transport import BatchedPolicyClient

from .settings import RoboCasaSettings


class RoboCasaSimulatorProcess:
    """Own RoboCasa without importing its pinned MuJoCo stack into GR00T."""

    def __init__(
        self,
        *,
        settings: RoboCasaSettings,
        startup_timeout_seconds: int,
    ) -> None:
        self.settings = settings
        self.startup_timeout_seconds = int(startup_timeout_seconds)
        self.run_dir: Path | None = None
        self.process: subprocess.Popen[str] | None = None
        self.client: BatchedPolicyClient | None = None
        self.stdout_handle: Any | None = None
        self.stderr_handle: Any | None = None
        self.failed = False

    async def request(self, value: dict[str, Any]) -> Any:
        try:
            if self.client is None:
                await self.start()
            assert self.client is not None
            return await self.client.predict(value)
        except Exception as exc:
            self.failed = True
            location = f"; simulator logs: {self.run_dir}" if self.run_dir else ""
            exc.add_note(f"RoboCasa simulator request failed{location}")
            raise

    async def start(self) -> None:
        if self.process is not None:
            return
        executable = _absolute_executable(
            self.settings.simulator_python_executable,
            cwd=Path.cwd(),
        )
        if not executable.is_file():
            raise FileNotFoundError(
                f"RoboCasa Python executable is missing: {executable}"
            )
        run_dir = Path(tempfile.mkdtemp(prefix="art-embodied-robocasa-"))
        socket_path = run_dir / "simulator.sock"
        ready_path = run_dir / "ready.json"
        spec_path = run_dir / "spec.json"
        spec_path.write_text(
            json.dumps(
                {
                    "socket_path": str(socket_path),
                    "ready_path": str(ready_path),
                    "source_revision": self.settings.manifest.source_revision,
                    "max_environment_steps": self.settings.max_environment_steps,
                    "allowed_tasks": [task.id for task in self.settings.tasks],
                    "reward_mode": self.settings.reward_mode,
                    "progress_grasp_reward": self.settings.progress_grasp_reward,
                    "progress_placement_reward": (
                        self.settings.progress_placement_reward
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self.stdout_handle = (run_dir / "stdout.log").open("w", encoding="utf-8")
        self.stderr_handle = (run_dir / "stderr.log").open("w", encoding="utf-8")
        environment = dict(os.environ)
        environment.setdefault("MUJOCO_GL", "egl")
        environment.setdefault("PYOPENGL_PLATFORM", "egl")
        repository_root = Path.cwd()
        python_paths = [str(repository_root / "src"), str(repository_root)]
        inherited_python_path = environment.get("PYTHONPATH")
        if inherited_python_path:
            python_paths.append(inherited_python_path)
        environment["PYTHONPATH"] = os.pathsep.join(python_paths)
        self.process = subprocess.Popen(
            [
                str(executable),
                "-m",
                "examples.embodied.robocasa.simulator_worker",
                "--serve-spec",
                str(spec_path),
            ],
            cwd=str(Path.cwd()),
            env=environment,
            text=True,
            stdout=self.stdout_handle,
            stderr=self.stderr_handle,
        )
        self.run_dir = run_dir
        await asyncio.to_thread(self._wait_ready, ready_path)
        self.client = BatchedPolicyClient(str(socket_path))

    def _wait_ready(self, ready_path: Path) -> None:
        assert self.process is not None
        deadline = time.monotonic() + self.startup_timeout_seconds
        while time.monotonic() < deadline:
            if ready_path.is_file():
                payload = json.loads(ready_path.read_text(encoding="utf-8"))
                if payload.get("ok"):
                    return
                raise RuntimeError(
                    "RoboCasa simulator failed during startup: "
                    f"{payload.get('error_type')}: {payload.get('error')}\n"
                    f"{payload.get('traceback', '')}"
                )
            return_code = self.process.poll()
            if return_code is not None:
                stderr = ""
                if self.run_dir is not None:
                    path = self.run_dir / "stderr.log"
                    if path.is_file():
                        stderr = path.read_text(encoding="utf-8", errors="replace")
                raise RuntimeError(
                    f"RoboCasa simulator exited during startup: {return_code}\n{stderr}"
                )
            time.sleep(0.1)
        raise TimeoutError("RoboCasa simulator timed out during startup")

    async def close(self) -> None:
        if self.client is not None:
            try:
                await self.client.predict({"op": "shutdown"})
            except Exception:
                pass
            await self.client.close()
            self.client = None
        if self.process is not None:
            try:
                await asyncio.wait_for(asyncio.to_thread(self.process.wait), timeout=30)
            except TimeoutError:
                self.process.terminate()
                await asyncio.to_thread(self.process.wait)
            self.process = None
        for handle_name in ("stdout_handle", "stderr_handle"):
            handle = getattr(self, handle_name)
            if handle is not None:
                handle.close()
                setattr(self, handle_name, None)
        if self.run_dir is not None:
            if not self.failed:
                shutil.rmtree(self.run_dir, ignore_errors=True)
            self.run_dir = None


def _absolute_executable(executable: Path, *, cwd: Path) -> Path:
    value = executable.expanduser()
    if value.is_absolute():
        return value
    return Path(os.path.abspath(cwd / value))
