from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from art_embodied.backends.flow_sde import TRANSIENT_FLOW_SDE_ROLLOUT_KEY
from art_embodied.integrations.isaaclab_reset_bank import IsaacLabResetBank
from art_embodied.policies.flow_policy import FlowModelInputs, FlowSDERollout
from art_embodied.policies.flow_replay_io import save_flow_replay
from art_embodied.policies.flow_sde import FlowSDESchedule, FlowSDETransitionRecord
from examples.embodied.isaaclab.droid_environment import DroidSceneConfig
from examples.embodied.isaaclab.droid_training import logging_groups, read_droid_groups


@pytest.fixture
def collection(tmp_path):
    settings = DroidSceneConfig("soup", "bowl", "Pick up soup.", num_envs=4)
    bank = IsaacLabResetBank.from_batches(
        [{"root": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}],
        contract=settings.reset_contract(),
        split="train",
    )
    actions = torch.arange(4 * 15 * 8, dtype=torch.float32).reshape(4, 15, 8) / 100
    rollout = FlowSDERollout(
        actions=actions,
        transition=FlowSDETransitionRecord(
            torch.zeros(4, 15, 32),
            torch.ones(4, 15, 32),
            torch.ones(4, dtype=torch.long),
            torch.zeros(4, 15, 32),
        ),
        inputs=FlowModelInputs(
            images=(torch.zeros(4, 3, 2, 2),),
            image_masks=(torch.ones(4, dtype=torch.bool),),
            language_tokens=torch.ones(4, 2, dtype=torch.long),
            language_masks=torch.ones(4, 2, dtype=torch.bool),
            state=torch.zeros(4, 32),
        ),
    )
    (tmp_path / "records").mkdir()
    save_flow_replay(tmp_path / "records/chunk-000000.pt", rollout, valid_count=4)
    report = {
        "settings": asdict(settings),
        "model_server": {
            "mode": "sde",
            "training_updates": 0,
            "batch_size": 4,
            "noise_level": 0.5,
            "denoising_steps": 10,
        },
        "optimizer_updates": 0,
        "split": "train",
        "group_size": 2,
        "initial_state_and_camera_replay_exact": True,
        "episodes": 4,
    }
    episodes = [
        {
            "episode": i,
            "batch": 0,
            "row": i,
            "group": i // 2,
            "bank_index": i // 2,
            "success": i % 2 == 0,
            "terminated": True,
            "truncated": False,
            "termination_reason": "success" if i % 2 == 0 else "failure",
            "reset_sha256": bank.hashes[i // 2],
            "observation_sha256": f"image-{i // 2}",
            "env_steps": 2,
            "policy_seed_base": i + 1,
            "environment_seed": 42,
            "media": [],
        }
        for i in range(4)
    ]
    chunks = [
        {
            "record_id": "chunk-000000",
            "active_rows": list(range(4)),
            "batch": 0,
            "chunk": 0,
            "actions": actions.tolist(),
            "execution_mask": [[True, True] + [False] * 13 for _ in range(4)],
        }
    ]
    for name, data in [("report", report), ("episodes", episodes), ("actions", chunks)]:
        (tmp_path / f"{name}.json").write_text(json.dumps(data))
    policy = SimpleNamespace(
        batch_size=4,
        bridge=SimpleNamespace(
            execution_horizon=15,
            schedule=FlowSDESchedule(10, 0.5),
            postprocess=lambda inputs, values: values[0].numpy(),
        ),
    )
    return tmp_path, bank, policy


def load(collection, **overrides):
    path, bank, policy = collection
    return read_droid_groups(
        path,
        path / "records",
        bank=bank,
        policy=policy,
        policy_version=overrides.get("policy_version", 0),
        group_size=2,
    )


def mutate(collection, name, edit):
    path = collection[0] / f"{name}.json"
    value = json.loads(path.read_text())
    edit(value)
    path.write_text(json.dumps(value))


def test_collected_groups_preserve_rewards_actions_masks_and_logging(collection):
    groups = load(collection)
    assert len(groups) == 2
    assert groups[0].rewards() == [1.0, 0.0]
    assert groups[1].metadata["group_index"] == 1
    for group in groups:
        for trajectory in group:
            assert trajectory.metrics["environment_steps"] == 2
            assert trajectory.actions[0].metadata["primitive_loss_mask_sum"] == 2
            assert TRANSIENT_FLOW_SDE_ROLLOUT_KEY in trajectory.actions[0].metadata
    exported = logging_groups(groups)
    assert (
        TRANSIENT_FLOW_SDE_ROLLOUT_KEY
        not in exported[0].trajectories[0].actions[0].metadata
    )
    assert (
        TRANSIENT_FLOW_SDE_ROLLOUT_KEY in groups[0].trajectories[0].actions[0].metadata
    )
    json.dumps([g.model_dump(mode="json") for g in exported])


def test_collector_rejects_stale_policy(collection):
    with pytest.raises(ValueError, match="policy contract"):
        load(collection, policy_version=1)


@pytest.mark.parametrize(
    "name,edit,match",
    [
        (
            "episodes",
            lambda rows: rows[0].update(reset_sha256="wrong"),
            "reset differs",
        ),
        ("episodes", lambda rows: rows[0].update(env_steps=3), "omits executed"),
        ("episodes", lambda rows: rows[0].update(terminated=False), "not finished"),
        (
            "episodes",
            lambda rows: rows[0].update(policy_seed_base=2),
            "reused a policy seed",
        ),
        (
            "episodes",
            lambda rows: rows[0].update(observation_sha256="different"),
            "different initial",
        ),
        ("actions", lambda rows: rows[0].update(record_id="../other"), "record ID"),
        ("actions", lambda rows: rows[0].update(chunk=1), "missing or duplicated"),
        (
            "actions",
            lambda rows: rows[0]["actions"][0][0].__setitem__(0, 9),
            "executed command",
        ),
        (
            "actions",
            lambda rows: rows[0]["execution_mask"][0].__setitem__(0, 1),
            "execution loss mask",
        ),
    ],
)
def test_collector_fails_closed_on_mismatched_behavior(collection, name, edit, match):
    mutate(collection, name, edit)
    with pytest.raises(ValueError, match=match):
        load(collection)


def test_server_configuration_preserves_frozen_policy_settings(tmp_path):
    import yaml

    from art_embodied.config import EmbodiedExperimentConfig
    from examples.embodied.isaaclab.openpi_server import inference_configuration

    reference = (
        Path(__file__).parents[1]
        / "examples/embodied/pi05_libero_long_flow_sde_grpo_rlinf_reference_1024.yaml"
    )
    original = EmbodiedExperimentConfig.model_validate(
        yaml.safe_load(reference.read_text())
    )
    path = tmp_path / "config.json"
    path.write_text(original.model_dump_json())
    actual = inference_configuration(path, device="cuda:7")
    assert actual.policy.device == "cuda:7"
    assert original.policy.device != "cuda:7"
    assert actual.policy.lora == original.policy.lora
    assert path.read_text() == original.model_dump_json()


def test_inference_server_verifies_actual_checkpoint_update(tmp_path):
    from art_embodied.checkpointing import CheckpointManager
    from examples.embodied.isaaclab.openpi_server import validate_adapter_checkpoint

    def writer(path):
        (path / "policy").mkdir()
        (path / "policy/weights").write_bytes(b"test")
        (path / "art_embodied_training_state.pt").write_bytes(b"test")

    checkpoint = CheckpointManager().publish(
        tmp_path / "update",
        writer=writer,
        config_fingerprint="config",
        resume_contract_fingerprint="contract",
        metadata={"step": 2},
    )
    config = SimpleNamespace(resume_contract_fingerprint="contract")
    validate_adapter_checkpoint(checkpoint / "policy", config=config, policy_version=2)
    with pytest.raises(ValueError, match="policy version"):
        validate_adapter_checkpoint(
            checkpoint / "policy", config=config, policy_version=3
        )
    (checkpoint / "policy/weights").write_bytes(b"changed")
    with pytest.raises(ValueError):
        validate_adapter_checkpoint(
            checkpoint / "policy", config=config, policy_version=2
        )
