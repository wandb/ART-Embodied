"""Process-local RNG streams for shared embodied inference servers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import random
from typing import Any, Iterator


@dataclass(slots=True)
class _RngState:
    python: object
    numpy: tuple[Any, ...]
    torch_cpu: Any
    torch_cuda: list[Any] | None


class PolicyRngStreams:
    """Isolate stochastic policy state for interleaved rollout groups."""

    def __init__(self) -> None:
        self._states: dict[str, _RngState] = {}

    def clear(self) -> None:
        self._states.clear()

    def release(self, stream_id: str) -> bool:
        return self._states.pop(stream_id, None) is not None

    @contextmanager
    def use(self, stream_id: str, *, seed: int, reset: bool) -> Iterator[None]:
        ambient = _capture_rng_state()
        try:
            if reset or stream_id not in self._states:
                seed_process_rng(seed)
            else:
                _restore_rng_state(self._states[stream_id])
            yield
            self._states[stream_id] = _capture_rng_state()
        finally:
            _restore_rng_state(ambient)


def seed_process_rng(seed: int) -> None:
    """Seed every process-local RNG used by policy construction or sampling."""

    import numpy as np

    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _capture_rng_state() -> _RngState:
    import numpy as np
    import torch

    return _RngState(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.random.get_rng_state(),
        torch_cuda=(
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    )


def _restore_rng_state(state: _RngState) -> None:
    import numpy as np
    import torch

    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.random.set_rng_state(state.torch_cpu)
    if state.torch_cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state.torch_cuda)
