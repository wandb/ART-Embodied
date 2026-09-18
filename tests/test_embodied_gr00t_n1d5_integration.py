from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from art_embodied.config import EmbodiedExperimentConfig  # noqa: E402
from art_embodied.integrations.gr00t_n1d5 import (  # noqa: E402
    GR00T_N1D5_EMBODIMENT_IDS,
    GR00TN15EmbodimentTag,
)
from examples.embodied.libero.gr00t_components import (  # noqa: E402
    create_gr00t_n1d5_components,
)

ROOT = Path(__file__).parents[1]
CONFIG = (
    ROOT
    / "examples/embodied/gr00t_n1d5_libero_spatial_flow_sde_grpo_positive_control.yaml"
)


def test_gr00t_positive_control_declares_the_rlinf_sampler_contract() -> None:
    config = EmbodiedExperimentConfig.from_yaml(CONFIG)

    assert config.policy.path == "RLinf/RLinf-Gr00t-SFT-Spatial"
    assert config.policy.revision == "73f710e70e7d571f8d828e51e0a428f5a1e0ac22"
    assert config.policy.load_kwargs["embodiment_tag"] == "libero_franka"
    assert config.policy.load_kwargs["execution_horizon"] == 5
    assert config.policy.load_kwargs["model_action_horizon"] == 16
    assert config.policy.load_kwargs["language_padding_length"] == 570
    assert config.algorithm.flow_sde.num_denoise_steps == 4
    assert config.algorithm.flow_sde.noise_level == pytest.approx(0.5)
    assert config.runtime.rollout_execution.inference_mode == "embedded"


def test_gr00t_n1d5_libero_embodiment_id_is_immutable() -> None:
    assert GR00TN15EmbodimentTag.LIBERO_FRANKA.value == "libero_franka"
    assert GR00T_N1D5_EMBODIMENT_IDS["libero_franka"] == 31


def test_gr00t_components_fail_closed_for_unaudited_shared_inference() -> None:
    config = SimpleNamespace(
        policy=SimpleNamespace(type="gr00t_n1d5"),
        runtime=SimpleNamespace(
            rollout_execution=SimpleNamespace(inference_mode="batched_server")
        ),
    )

    with pytest.raises(ValueError, match="embedded inference"):
        create_gr00t_n1d5_components(
            config=config,
            context=SimpleNamespace(local_device="cuda:0"),
        )
