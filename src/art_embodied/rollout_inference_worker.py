"""Per-device dynamic-batching inference server for rollout actors."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import importlib
import inspect
import json
from pathlib import Path
import sys
import traceback
from typing import Any

from .config import EmbodiedExperimentConfig
from .inference_transport import read_message, write_message
from .rollout_process import RolloutActorProcessContext
from .utils import write_json_atomic
from .worker_config import discard_coordinator_resume


@dataclass(slots=True)
class _PendingRequest:
    value: Any
    future: asyncio.Future[Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve-spec", type=Path, required=True)
    return parser.parse_args()


def _resolve_factory(reference: str) -> Any:
    module_name, separator, qualname = reference.partition(":")
    if not separator or not module_name or not qualname:
        raise ValueError("inference_factory must use the 'module:callable' form")
    value: Any = importlib.import_module(module_name)
    for component in qualname.split("."):
        value = getattr(value, component)
    if not callable(value):
        raise TypeError(f"Inference factory is not callable: {reference}")
    return value


async def _await_if_needed(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _worker_config(spec: dict[str, Any]) -> EmbodiedExperimentConfig:
    """Validate the experiment contract after removing parent-only state."""

    raw = discard_coordinator_resume(spec["config"])
    return EmbodiedExperimentConfig.model_validate(raw)


async def _batch_loop(
    queue: asyncio.Queue[_PendingRequest],
    engine: Any,
    *,
    max_batch_size: int,
    max_wait_seconds: float,
    engine_lock: asyncio.Lock,
) -> None:
    while True:
        first = await queue.get()
        batch = [first]
        deadline = asyncio.get_running_loop().time() + max_wait_seconds
        while len(batch) < max_batch_size:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(queue.get(), timeout=remaining))
            except TimeoutError:
                break
        try:
            async with engine_lock:
                values = await _await_if_needed(
                    engine.predict_batch([item.value for item in batch])
                )
            values = list(values)
            if len(values) != len(batch):
                raise RuntimeError(
                    "predict_batch returned a different number of results: "
                    f"requests={len(batch)}, results={len(values)}"
                )
        except Exception as exc:
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)
        else:
            for item, value in zip(batch, values, strict=True):
                if not item.future.done():
                    item.future.set_result(value)


async def _handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    queue: asyncio.Queue[_PendingRequest],
) -> None:
    try:
        while True:
            try:
                value = await read_message(reader)
            except (asyncio.IncompleteReadError, ConnectionResetError):
                break
            future = asyncio.get_running_loop().create_future()
            await queue.put(_PendingRequest(value=value, future=future))
            try:
                result = await future
                response = {"ok": True, "value": result}
            except Exception as exc:
                response = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            await write_message(writer, response)
    finally:
        writer.close()
        await writer.wait_closed()


async def _command_loop(
    engine: Any,
    *,
    engine_lock: asyncio.Lock,
) -> None:
    while line := await asyncio.to_thread(sys.stdin.readline):
        command = json.loads(line)
        if command.get("op") == "shutdown":
            return
        result_path = Path(command["result_path"])
        try:
            operation = command.get("op")
            if operation == "offload":
                offload = getattr(engine, "offload", None)
                if not callable(offload):
                    raise TypeError(
                        "Inference engine must expose offload() for the "
                        "cpu_offload lifecycle"
                    )
                value = await _await_if_needed(offload())
                result = {"ok": True, "value": value}
                write_json_atomic(result_path, result, indent=2, sort_keys=True)
                continue
            if operation != "prepare":
                raise ValueError(f"Unknown inference command: {command.get('op')!r}")
            async with engine_lock:
                await _await_if_needed(
                    engine.prepare_update(
                        update=int(command["update"]),
                        policy_snapshot=Path(command["policy_snapshot"]),
                    )
                )
            result = {"ok": True}
        except Exception as exc:
            result = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        write_json_atomic(result_path, result, indent=2, sort_keys=True)


async def _serve(spec: dict[str, Any]) -> None:
    config = _worker_config(spec)
    context = RolloutActorProcessContext(
        worker_index=int(spec["server_index"]),
        configured_device=str(spec["configured_device"]),
        local_device=str(spec["local_device"]),
    )
    ready_path = Path(spec["ready_path"])
    socket_path = Path(spec["socket_path"])
    socket_path.unlink(missing_ok=True)
    try:
        factory = _resolve_factory(str(spec["inference_factory"]))
        engine = await _await_if_needed(factory(config=config, context=context))
        if not callable(getattr(engine, "prepare_update", None)):
            raise TypeError("Inference engine must expose prepare_update")
        if not callable(getattr(engine, "predict_batch", None)):
            raise TypeError("Inference engine must expose predict_batch")
        queue: asyncio.Queue[_PendingRequest] = asyncio.Queue()
        engine_lock = asyncio.Lock()
        server = await asyncio.start_unix_server(
            lambda reader, writer: _handle_client(reader, writer, queue),
            path=str(socket_path),
        )
        socket_path.chmod(0o600)
    except Exception as exc:
        write_json_atomic(
            ready_path,
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            indent=2,
            sort_keys=True,
        )
        return

    write_json_atomic(
        ready_path,
        {"ok": True, "socket_path": str(socket_path)},
    )
    execution = config.runtime.rollout_execution
    batch_task = asyncio.create_task(
        _batch_loop(
            queue,
            engine,
            max_batch_size=execution.inference_max_batch_size,
            max_wait_seconds=execution.inference_max_wait_ms / 1000.0,
            engine_lock=engine_lock,
        )
    )
    try:
        await _command_loop(engine, engine_lock=engine_lock)
    finally:
        server.close()
        await server.wait_closed()
        batch_task.cancel()
        try:
            await batch_task
        except asyncio.CancelledError:
            pass
        close = getattr(engine, "close", None)
        if callable(close):
            await _await_if_needed(close())
        socket_path.unlink(missing_ok=True)


def main() -> None:
    """Serve batched policy inference from a coordinator-written specification."""

    args = _parse_args()
    spec = json.loads(args.serve_spec.read_text(encoding="utf-8"))
    asyncio.run(_serve(spec))


if __name__ == "__main__":
    main()
