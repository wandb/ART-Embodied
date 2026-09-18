from types import SimpleNamespace

import numpy as np

from examples.embodied.libero.reset_evidence import (
    array_fingerprint,
    reset_fingerprints,
)


def test_fixture_changes_are_visible_even_with_identical_sim_state():
    state = np.array([0.1, 0.2])
    model = SimpleNamespace(body_pos=np.zeros((2, 3)), body_quat=np.ones((2, 4)))
    sim = SimpleNamespace(get_state=lambda: state, model=model)
    observation = {"image": np.zeros((4, 4, 3), dtype=np.uint8)}
    first = reset_fingerprints(observation, sim)
    second = reset_fingerprints(observation, sim)
    assert first == second
    np.testing.assert_array_equal(state, [0.1, 0.2])
    np.testing.assert_array_equal(model.body_pos, np.zeros((2, 3)))
    model.body_pos[0, 0] = 0.01
    changed = reset_fingerprints(observation, sim)
    assert first["flattened_sim_state_sha256"] == changed["flattened_sim_state_sha256"]
    assert (
        first["model_body_transforms_sha256"] != changed["model_body_transforms_sha256"]
    )
    assert first["policy_observation_sha256"] == changed["policy_observation_sha256"]


def test_fingerprint_tracks_shape_dtype_and_values_not_mapping_order():
    a = {"b": np.zeros((2, 3)), "a": np.ones(2)}
    assert array_fingerprint(a) == array_fingerprint(dict(reversed(list(a.items()))))
    for replacement in (
        np.zeros((3, 2)),
        np.zeros((2, 3), dtype=np.float32),
        np.ones((2, 3)),
    ):
        assert array_fingerprint(a) != array_fingerprint(a | {"b": replacement})
