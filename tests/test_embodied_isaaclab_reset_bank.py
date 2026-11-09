import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from art_embodied.integrations.isaaclab import isaaclab_reset_fingerprints
from art_embodied.integrations.isaaclab_reset_bank import IsaacLabResetBank


def scene(start=0):
    return {
        "robot": {"joint/pos": torch.arange(start, start + 12).reshape(4, 3).float()}
    }


def bank(start=0, split="train"):
    return IsaacLabResetBank.from_batches(
        [scene(start)], contract={"task": "pick", "armature": 0.1}, split=split
    )


def test_bank_roundtrip_group_replay_and_non_aliasing(tmp_path):
    original = bank()
    path = tmp_path / "bank"
    original.save(path)
    restored = IsaacLabResetBank.load(path, contract=original.contract, split="train")
    assert restored.fingerprint == original.fingerprint
    assert bank(100).fingerprint != original.fingerprint
    selected = restored.select([2, 2, 0, 0])
    assert isaaclab_reset_fingerprints(selected, num_envs=4) == (
        original.hashes[2],
        original.hashes[2],
        original.hashes[0],
        original.hashes[0],
    )
    selected["robot"]["joint/pos"].zero_()
    assert restored.hashes == original.hashes
    torch.testing.assert_close(
        restored.select([2])["robot"]["joint/pos"], scene()["robot"]["joint/pos"][2:3]
    )
    with pytest.raises(FileExistsError):
        original.save(path)


@pytest.mark.parametrize("indices", [[], [-1], [4], [True], [0.5]])
def test_bank_rejects_invalid_indices(indices):
    with pytest.raises(ValueError, match="indices"):
        bank().select(indices)


def test_bank_split_and_physics_are_checked(tmp_path):
    original = bank()
    original.save(tmp_path / "bank")
    for contract, split in [
        (original.contract, "sealed"),
        ({"task": "other"}, "train"),
    ]:
        with pytest.raises(ValueError, match="contract or split"):
            IsaacLabResetBank.load(tmp_path / "bank", contract=contract, split=split)
    bank().assert_disjoint(bank(100, "development"))
    with pytest.raises(ValueError, match="overlap"):
        bank().assert_disjoint(bank(0, "sealed"))


def test_bank_checks_data_and_fingerprint_tampering(tmp_path):
    original = bank()
    path = tmp_path / "bank"
    original.save(path)
    manifest = path / "manifest.json"
    data = json.loads(manifest.read_text())
    data["reset_hashes"][0] = "bad"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="fingerprints"):
        IsaacLabResetBank.load(path, contract=original.contract, split="train")
    with (path / "states.safetensors").open("ab") as stream:
        stream.write(b"bad")
    with pytest.raises(ValueError, match="checksum"):
        IsaacLabResetBank.load(path, contract=original.contract, split="train")


def test_bank_rejects_duplicate_nonfinite_and_mismatched_batches():
    with pytest.raises(ValueError, match="duplicate"):
        IsaacLabResetBank.from_batches(
            [scene(), scene()], contract={"task": "a"}, split="train"
        )
    invalid = scene()
    invalid["robot"]["joint/pos"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        IsaacLabResetBank.from_batches([invalid], contract={"task": "a"}, split="train")
    with pytest.raises(ValueError, match="schemas"):
        IsaacLabResetBank.from_batches(
            [scene(), {"other": torch.ones(4)}], contract={"task": "a"}, split="train"
        )


def test_bank_scene_contract_excludes_only_execution_details():
    from examples.embodied.isaaclab.droid_environment import DroidSceneConfig

    a = DroidSceneConfig("soup", "bowl", "Pick up soup.")
    b = DroidSceneConfig(
        "soup", "bowl", "Pick up soup.", num_envs=8, seed=17, device="cuda:2"
    )
    c = DroidSceneConfig("soup", "bowl", "Pick up soup.", panda_armature=0.2)
    assert a.reset_contract() == b.reset_contract()
    assert a.reset_contract() != c.reset_contract()
