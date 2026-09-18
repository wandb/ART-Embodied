"""Local framed transport for batched embodied policy inference.

Messages use pickle because actors and the policy server exchange internal
tensor-rich Python objects on a private Unix socket. This is intentionally not
a network or untrusted-input protocol: both peers must belong to the same
ART-Embodied run and execute trusted code.
"""

from __future__ import annotations

import asyncio
import pickle
import socket
import struct
from typing import Any

_HEADER = struct.Struct("!Q")


async def write_message(writer: asyncio.StreamWriter, value: Any) -> None:
    """Write one length-prefixed internal Python message."""

    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    writer.write(_HEADER.pack(len(payload)))
    writer.write(payload)
    await writer.drain()


async def read_message(reader: asyncio.StreamReader) -> Any:
    """Read one complete trusted message from a run-local peer."""

    header = await reader.readexactly(_HEADER.size)
    (size,) = _HEADER.unpack(header)
    payload = await reader.readexactly(size)
    return pickle.loads(payload)


def write_message_sync(connection: socket.socket, value: Any) -> None:
    """Write one framed message without requiring an asyncio event loop."""

    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    connection.sendall(_HEADER.pack(len(payload)) + payload)


def read_message_sync(connection: socket.socket) -> Any:
    """Read one framed message from a blocking run-local socket."""

    header = _receive_exact(connection, _HEADER.size)
    (size,) = _HEADER.unpack(header)
    payload = _receive_exact(connection, size)
    return pickle.loads(payload)


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("Run-local peer closed the framed transport")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class BatchedPolicyClient:
    """Serialize requests over one persistent per-device policy connection.

    The lock preserves request/response ordering when concurrent actor tasks
    share a client; the server protocol has no request IDs for reordering.
    """

    def __init__(self, socket_path: str) -> None:
        self.socket_path = socket_path
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None
        self._lock = asyncio.Lock()

    async def predict(self, request: Any) -> Any:
        async with self._lock:
            if self.reader is None or self.writer is None:
                self.reader, self.writer = await asyncio.open_unix_connection(
                    self.socket_path
                )
            await write_message(self.writer, request)
            response = await read_message(self.reader)
            if not isinstance(response, dict) or "ok" not in response:
                raise RuntimeError("Batched policy server returned an invalid response")
            if not response["ok"]:
                raise RuntimeError(
                    "Batched policy inference failed: "
                    f"{response.get('error_type')}: {response.get('error')}"
                )
            return response.get("value")

    async def close(self) -> None:
        if self.writer is None:
            return
        self.writer.close()
        await self.writer.wait_closed()
        self.reader = None
        self.writer = None
