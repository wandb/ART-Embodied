from __future__ import annotations

from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from art_embodied import EmbodiedExperimentConfig
from art_embodied.backends import local_process as local_process_module
from art_embodied.backends.local_process import LocalProcessActionTokenBackend

ROOT = Path(__file__).parents[1]


def _config(tmp_path: Path) -> EmbodiedExperimentConfig:
    config = EmbodiedExperimentConfig.from_yaml(
        ROOT / "examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml"
    )
    return config.model_copy(
        update={
            "training": config.training.model_copy(
                update={"checkpoint_every_updates": 2}
            ),
            "storage": config.storage.model_copy(
                update={
                    "output_dir": tmp_path,
                    "keep_last_checkpoints": 1,
                    "retain_checkpoint_updates": [2, 3],
                    "save_training_state": False,
                }
            ),
        }
    )


def test_checkpoint_retention_protects_explicit_milestones(
    tmp_path: Path,
    monkeypatch,
) -> None:
    backend = object.__new__(LocalProcessActionTokenBackend)
    backend.config = _config(tmp_path)
    backend.policy = object()
    backend.backend = SimpleNamespace(optimizer=None, step=0)

    def save_snapshot(_policy, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_model.safetensors").write_bytes(b"adapter")

    monkeypatch.setattr(local_process_module, "_save_policy_snapshot", save_snapshot)

    written = []
    for step in range(1, 7):
        backend.update_step = step
        written.append(backend._maybe_checkpoint())

    assert written[0] is None
    assert written[1] is not None  # periodic and protected
    assert written[2] is not None  # protected even though not periodic
    assert written[4] is None
    assert [path.name for path in sorted((tmp_path / "checkpoints").iterdir())] == [
        "step_000002",
        "step_000003",
        "step_000006",
    ]


def test_training_state_contains_optimizer_and_rng_state(tmp_path: Path) -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-4)
    parameter.grad = torch.tensor([0.5])
    optimizer.step()

    local_process_module._save_training_state(
        tmp_path,
        optimizer=optimizer,
        update_step=20,
        backend_step=80,
    )

    payload = torch.load(
        tmp_path / "art_embodied_training_state.pt",
        weights_only=False,
    )
    assert payload["update_step"] == 20
    assert payload["backend_step"] == 80
    assert payload["optimizer_state_dict"]["state"]
    assert payload["torch_rng_state"].numel() > 0
    assert (tmp_path / "art_embodied_training_state.json").is_file()


def test_training_state_restores_optimizer_and_rng_state(tmp_path: Path) -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-4)
    parameter.grad = torch.tensor([0.5])
    optimizer.step()
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    local_process_module._save_training_state(
        tmp_path,
        optimizer=optimizer,
        update_step=20,
        backend_step=80,
        config_fingerprint="saved-config",
    )
    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = float(torch.rand(()))

    restored_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    restored_optimizer = torch.optim.AdamW([restored_parameter], lr=9.9e-3)
    random.seed(21)
    np.random.seed(22)
    torch.manual_seed(23)

    payload = local_process_module._load_training_state(
        tmp_path,
        optimizer=restored_optimizer,
    )

    assert payload["update_step"] == 20
    assert payload["backend_step"] == 80
    assert payload["config_fingerprint"] == "saved-config"
    assert restored_optimizer.param_groups[0]["lr"] == 1.0e-4
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    assert float(torch.rand(())) == expected_torch


def test_local_backend_restores_policy_optimizer_and_update_cursor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "resume" / "step_000020"
    checkpoint.mkdir(parents=True)
    (checkpoint / "art_embodied_checkpoint.json").write_text("{}")
    (checkpoint / "art_embodied_training_state.pt").write_bytes(b"state")
    (checkpoint / "art_embodied_training_state.json").write_text("{}")
    config = _config(tmp_path).model_copy(
        update={
            "training": _config(tmp_path).training.model_copy(update={"updates": 100}),
            "storage": _config(tmp_path).storage.model_copy(
                update={
                    "resume_from_checkpoint": checkpoint,
                    "allow_legacy_checkpoint_resume": True,
                }
            ),
        }
    )
    loaded = []

    class Policy:
        def load_checkpoint(self, ref) -> None:
            loaded.append(ref)

    optimizer = object()
    action_backend = SimpleNamespace(optimizer=None, step=0)

    def ensure_optimizer() -> None:
        action_backend.optimizer = optimizer

    action_backend._ensure_optimizer = ensure_optimizer
    monkeypatch.setattr(
        local_process_module,
        "_load_training_state",
        lambda path, *, optimizer, **_kwargs: {
            "update_step": 20,
            "backend_step": 80,
        },
    )

    backend = LocalProcessActionTokenBackend(
        config=config,
        policy=Policy(),
        backend=action_backend,
    )

    assert loaded == [{"path": str(checkpoint)}]
    assert backend.update_step == 20
    assert backend.backend.step == 80


def test_local_backend_moves_parent_to_cpu_before_restoring_optimizer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "resume" / "step_000010"
    checkpoint.mkdir(parents=True)
    (checkpoint / "art_embodied_checkpoint.json").write_text("{}")
    (checkpoint / "art_embodied_training_state.pt").write_bytes(b"state")
    (checkpoint / "art_embodied_training_state.json").write_text("{}")
    config = _config(tmp_path).model_copy(
        update={
            "training": _config(tmp_path).training.model_copy(update={"updates": 100}),
            "storage": _config(tmp_path).storage.model_copy(
                update={
                    "resume_from_checkpoint": checkpoint,
                    "allow_legacy_checkpoint_resume": True,
                }
            ),
        }
    )
    events: list[str] = []

    class Model(torch.nn.Module):
        def to(self, *args, **kwargs):
            events.append(f"model:{args[0]}")
            return super().to(*args, **kwargs)

    class Policy:
        def __init__(self) -> None:
            self.model = Model()

        def load_checkpoint(self, _ref) -> None:
            events.append("policy:load")

    action_backend = SimpleNamespace(optimizer=None, step=0)

    def ensure_optimizer() -> None:
        events.append("optimizer:create")
        action_backend.optimizer = object()

    action_backend._ensure_optimizer = ensure_optimizer

    def load_training_state(_path, *, optimizer, **_kwargs):
        assert optimizer is action_backend.optimizer
        events.append("optimizer:restore")
        return {"update_step": 10, "backend_step": 40}

    monkeypatch.setattr(
        local_process_module,
        "_load_training_state",
        load_training_state,
    )

    LocalProcessActionTokenBackend(
        config=config,
        policy=Policy(),
        backend=action_backend,
    )

    assert events == [
        "policy:load",
        "model:cpu",
        "optimizer:create",
        "optimizer:restore",
    ]


def test_resume_rejects_contract_mismatch_before_loading_policy(
    tmp_path: Path,
) -> None:
    base = _config(tmp_path)

    def write_checkpoint(path: Path) -> None:
        (path / "art_embodied_checkpoint.json").write_text("{}")
        (path / "art_embodied_training_state.pt").write_bytes(b"state")

    checkpoint = local_process_module._CHECKPOINT_MANAGER.publish(
        tmp_path / "resume" / "step_000010",
        writer=write_checkpoint,
        config_fingerprint="other-config",
        resume_contract_fingerprint="other-contract",
    )
    config = base.model_copy(
        update={
            "training": base.training.model_copy(update={"updates": 100}),
            "storage": base.storage.model_copy(
                update={"resume_from_checkpoint": checkpoint}
            ),
        }
    )
    loaded: list[object] = []

    class Policy:
        def load_checkpoint(self, ref) -> None:
            loaded.append(ref)

    with pytest.raises(ValueError, match="resume contract does not match"):
        LocalProcessActionTokenBackend(
            config=config,
            policy=Policy(),
            backend=SimpleNamespace(optimizer=None, step=0),
        )

    assert loaded == []
