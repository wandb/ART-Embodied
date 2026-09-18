"""Reset provenance, not a complete simulator checkpoint or equality proof."""

import hashlib
from typing import Any, Mapping

import numpy as np


def array_fingerprint(values: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        array = np.asarray(value)
        digest.update(f"{key}:{array.dtype}:{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def reset_fingerprints(
    observation: Mapping[str, Any], sim: Any
) -> dict[str, str | int]:
    # Fixture placement lives in model arrays, outside the flattened sim state.
    # Hash both without changing the native benchmark reset distribution.
    return {
        "schema_version": 1,
        "policy_observation_sha256": array_fingerprint(observation),
        "flattened_sim_state_sha256": array_fingerprint(
            {"state": sim.get_state().flatten()}
        ),
        "model_body_transforms_sha256": array_fingerprint(
            {"body_pos": sim.model.body_pos, "body_quat": sim.model.body_quat}
        ),
    }
