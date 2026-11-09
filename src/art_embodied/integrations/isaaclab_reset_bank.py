"""Immutable, tensor-only scene banks for fixed Isaac Lab evaluations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path

from .isaaclab import isaaclab_reset_fingerprints


def _flatten(state, path=()):
    import torch

    if isinstance(state, Mapping):
        for key, value in state.items():
            if not isinstance(key, str):
                raise TypeError("Scene keys must be strings")
            yield from _flatten(value, (*path, key))
    elif isinstance(state, torch.Tensor) and state.ndim > 0 and path:
        yield json.dumps(path), state.detach().cpu().contiguous()
    else:
        raise TypeError("Scene state must contain batched tensors")


def _unflatten(tensors):
    state = {}
    for key, value in tensors.items():
        path = json.loads(key)
        if (
            not isinstance(path, list)
            or not path
            or any(not isinstance(part, str) for part in path)
        ):
            raise ValueError("Invalid scene tensor key")
        node = state
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = value
    return state


class IsaacLabResetBank:
    """Scene snapshots plus an explicit task/physics contract.

    Banks contain physical state only. The task must restore its other state and
    verify observations after reset. Generating a sealed bank does not evaluate
    a policy on it; callers must keep it out of development selection.
    """

    def __init__(self, tensors, *, contract, split):
        if split not in {"train", "development", "sealed"}:
            raise ValueError("Unknown reset bank split")
        if not tensors or not isinstance(contract, dict) or not contract:
            raise ValueError("Scene tensors and a task/physics contract are required")
        count = next(iter(tensors.values())).shape[0]
        self.hashes = isaaclab_reset_fingerprints(_unflatten(tensors), num_envs=count)
        if len(set(self.hashes)) != count:
            raise ValueError("Reset bank contains duplicate initial states")
        self._tensors = tensors
        self.contract = json.loads(json.dumps(contract))
        self.split = split

    def __len__(self):
        return len(self.hashes)

    @property
    def fingerprint(self):
        document = {
            "contract": self.contract,
            "split": self.split,
            "hashes": self.hashes,
        }
        return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_batches(cls, batches, *, contract, split):
        import torch

        batches = [dict(_flatten(batch)) for batch in batches]
        if not batches or not batches[0]:
            raise ValueError("Empty scene bank")
        for batch in batches:
            if batch.keys() != batches[0].keys():
                raise ValueError("Scene batch schemas differ")
            count = next(iter(batch.values())).shape[0]
            isaaclab_reset_fingerprints(_unflatten(batch), num_envs=count)
            if any(
                batch[key].dtype != reference.dtype
                or batch[key].shape[1:] != reference.shape[1:]
                for key, reference in batches[0].items()
            ):
                raise ValueError("Scene batch tensor shapes or dtypes differ")
        tensors = {
            key: torch.cat([batch[key] for batch in batches]) for key in batches[0]
        }
        return cls(tensors, contract=contract, split=split)

    def select(self, indices: Sequence[int], *, device="cpu"):
        import torch

        if not indices or any(
            type(index) is not int or not 0 <= index < len(self) for index in indices
        ):
            raise ValueError("Reset bank indices must be nonempty and in range")
        index = torch.tensor(indices, dtype=torch.long)
        return _unflatten(
            {key: value[index].to(device) for key, value in self._tensors.items()}
        )

    def assert_disjoint(self, other):
        if set(self.hashes) & set(other.hashes):
            raise ValueError("Reset banks overlap")

    def save(self, directory):
        from safetensors.torch import save_file

        from art_embodied.utils import write_json_atomic

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        filename = directory / "states.safetensors"
        save_file(self._tensors, str(filename))
        digest = hashlib.sha256(filename.read_bytes()).hexdigest()
        write_json_atomic(
            directory / "manifest.json",
            {
                "format_version": 1,
                "split": self.split,
                "contract": self.contract,
                "states_sha256": digest,
                "reset_hashes": self.hashes,
            },
        )

    @classmethod
    def load(cls, directory, *, contract, split):
        from safetensors.torch import load_file

        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if (
            manifest["format_version"] != 1
            or manifest["split"] != split
            or manifest["contract"] != contract
        ):
            raise ValueError("Reset bank task/physics contract or split differs")
        filename = directory / "states.safetensors"
        if (
            hashlib.sha256(filename.read_bytes()).hexdigest()
            != manifest["states_sha256"]
        ):
            raise ValueError("Reset bank checksum mismatch")
        bank = cls(load_file(str(filename)), contract=contract, split=split)
        if list(bank.hashes) != manifest["reset_hashes"]:
            raise ValueError("Reset bank fingerprints differ")
        return bank
