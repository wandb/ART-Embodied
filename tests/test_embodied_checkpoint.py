from __future__ import annotations

from pathlib import Path

from art_embodied.backends.action_token import _save_policy_checkpoint
from art_embodied.checkpointing import CHECKPOINT_COMPLETE_MARKER


def test_policy_checkpoint_contract_precedes_full_state_dict_fallback(
    tmp_path: Path,
) -> None:
    class AdapterPolicy:
        def save_checkpoint(self, path: str):
            output = Path(path)
            (output / "adapter_model.safetensors").write_bytes(b"adapter")
            return {"path": str(output), "type": "adapter"}

        def state_dict(self):
            raise AssertionError("full state_dict fallback must not run")

    output = tmp_path / "checkpoint"

    saved = _save_policy_checkpoint(AdapterPolicy(), output)

    assert saved == output
    assert (output / "adapter_model.safetensors").read_bytes() == b"adapter"
    assert not (output / "policy_state.pt").exists()
    assert (output / CHECKPOINT_COMPLETE_MARKER).is_file()
