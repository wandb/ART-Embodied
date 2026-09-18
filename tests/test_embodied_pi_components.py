from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from examples.embodied.libero.pi_components import create_pi_components


def test_sampling_override_rejects_shared_inference():
    config = SimpleNamespace(
        policy=SimpleNamespace(type="pi0"),
        runtime=SimpleNamespace(
            rollout_execution=SimpleNamespace(inference_mode="batched_server")
        ),
    )

    with pytest.raises(ValueError, match="embedded inference"):
        create_pi_components(
            config=config,
            context=SimpleNamespace(local_device="cuda:0"),
            sampling_mode_override="eval",
        )
