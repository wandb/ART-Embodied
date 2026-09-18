from __future__ import annotations

from pathlib import Path
import pickle
from types import SimpleNamespace

import pytest
import yaml

torch = pytest.importorskip("torch")

from art_embodied.backends.action_token_gradients import (  # noqa: E402
    load_action_token_gradient_payload,
)
from art_embodied.backends.flow_sde import FlowSDEExample  # noqa: E402
from art_embodied.backends.flow_sde_local_process import (  # noqa: E402
    _aggregate_flow_worker_metrics,
)
from art_embodied.backends.flow_sde_worker import (  # noqa: E402
    _compute_gradient_job,
)
from art_embodied.config import EmbodiedExperimentConfig  # noqa: E402
from art_embodied.policies.flow_sde import (  # noqa: E402
    FlowSDETransitionRecord,
)
from art_embodied.policies.gr00t_n1d7_sft_replay import (  # noqa: E402
    sft_replay_sample_seed,
)
from art_embodied.policies.pi_flow_sde import (  # noqa: E402
    PIFlowModelInputs,
    PIFlowSDERollout,
)

TASKS = [
    "PnPCupToDrawerClose",
    "PnPMilkToMicrowaveClose",
    "PnPPotatoToMicrowaveClose",
    "PosttrainPnPNovelFromCuttingboardToPanSplitA",
    "PosttrainPnPNovelFromPlacematToBasketSplitA",
    "PosttrainPnPNovelFromPlateToBowlSplitA",
    "PosttrainPnPNovelFromTrayToPotSplitA",
    "PosttrainPnPNovelFromTrayToTieredbasketSplitA",
]


class _ReplayPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def flow_sde_logprobs(self, rollout: PIFlowSDERollout) -> torch.Tensor:
        return rollout.transition.old_logprobs + self.weight

    def sft_replay_loss(self, step_data, *, seed: int) -> torch.Tensor:
        assert step_data == ["fixed-training-example"]
        assert seed == 1234
        return (self.weight - 2.0).square()


class _ReplayProvider:
    def sample(self, *, update_index: int, subupdate_index: int):
        assert (update_index, subupdate_index) == (3, 5)
        return SimpleNamespace(
            step_data="fixed-training-example",
            task_id=TASKS[0],
            episode_index=17,
            step_index=23,
            seed=1234,
            source_info_sha256="a" * 64,
        )


def _example() -> FlowSDEExample:
    rollout = PIFlowSDERollout(
        actions=torch.zeros(1, 2, 1),
        transition=FlowSDETransitionRecord(
            previous_states=torch.zeros(1, 2, 1),
            next_states=torch.zeros(1, 2, 1),
            selected_indices=torch.zeros(1, dtype=torch.long),
            old_logprobs=torch.zeros(1, 2, 1),
        ),
        inputs=PIFlowModelInputs(
            images=(torch.zeros(1, 3, 2, 2),),
            image_masks=(torch.ones(1, dtype=torch.bool),),
            language_tokens=torch.ones(1, 1, dtype=torch.long),
            language_masks=torch.ones(1, 1, dtype=torch.bool),
            state=None,
        ),
    )
    return FlowSDEExample(
        rollout=rollout,
        group_index=0,
        trajectory_index=0,
        action_index=0,
        reward=0.0,
        loss_mask=True,
        trajectory_primitive_steps=2,
    )


def _config(tmp_path: Path, *, coefficient: float) -> EmbodiedExperimentConfig:
    path = (
        Path(__file__).parents[1]
        / "examples/embodied/gr00t_n1d7_robocasa_gr1_tabletop_sft_anchor_b010_development.yaml"
    )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["algorithm"]["kl_coefficient"] = 0.0
    raw["policy"]["load_kwargs"]["sft_replay"] = {
        "coefficient": coefficient,
        "dataset_root": str(tmp_path / "prepared"),
        "task_ids": TASKS,
        "seed": 20260814,
        "samples_per_worker": 1,
        "sampling_strategy": "worker_task_balanced",
        "processor_mode": "eval",
        "dataset_prefix": "gr1_unified",
    }
    raw["storage"]["output_dir"] = str(tmp_path / "output")
    return EmbodiedExperimentConfig.model_validate(raw)


def _command(tmp_path: Path, *, coefficient: float) -> dict:
    payload_path = tmp_path / f"examples-{coefficient}.pkl"
    with payload_path.open("wb") as handle:
        pickle.dump(
            {"examples": [_example()], "advantages": [0.0]},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return {
        "worker_index": 0,
        "worker_count": 8,
        "update_index": 3,
        "subupdate_index": 5,
        "examples_path": str(payload_path),
        "gradient_path": str(tmp_path / f"gradients-{coefficient}.pt"),
        "microbatch_size": 1,
        "loss_denominator": 1,
        "length_normalized": False,
        "max_episode_steps": 2,
        "clip_epsilon_low": 0.2,
        "clip_epsilon_high": 0.2,
        "clip_ratio_c": 3.0,
        "sft_replay_coefficient": coefficient,
    }


def test_sft_replay_seed_is_stable_and_coordinate_specific() -> None:
    first = sft_replay_sample_seed(
        seed=7, update_index=3, subupdate_index=5, worker_index=2
    )
    assert first == sft_replay_sample_seed(
        seed=7, update_index=3, subupdate_index=5, worker_index=2
    )
    assert first != sft_replay_sample_seed(
        seed=7, update_index=3, subupdate_index=6, worker_index=2
    )
    assert first != sft_replay_sample_seed(
        seed=7, update_index=3, subupdate_index=5, worker_index=3
    )


def test_positive_sft_replay_requires_exact_development_task_panel(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, coefficient=0.1)
    assert config.execution_summary()["sft_replay"]["task_ids"] == TASKS

    raw = config.model_dump(mode="python")
    raw["policy"]["load_kwargs"]["sft_replay"]["task_ids"] = TASKS[::-1]
    with pytest.raises(ValueError, match="must exactly match"):
        EmbodiedExperimentConfig.model_validate(raw)


def test_zero_coefficient_is_exact_worker_parity(tmp_path: Path) -> None:
    implicit_policy = _ReplayPolicy()
    explicit_policy = _ReplayPolicy()
    implicit = _command(tmp_path, coefficient=0.0)
    implicit.pop("sft_replay_coefficient")
    implicit["gradient_path"] = str(tmp_path / "implicit.pt")
    explicit = _command(tmp_path, coefficient=0.0)

    implicit_result = _compute_gradient_job(implicit, policy=implicit_policy)
    explicit_result = _compute_gradient_job(explicit, policy=explicit_policy)
    implicit_gradient = load_action_token_gradient_payload(implicit["gradient_path"])
    explicit_gradient = load_action_token_gradient_payload(explicit["gradient_path"])

    assert implicit_result["metrics"] == explicit_result["metrics"]
    torch.testing.assert_close(
        implicit_gradient["gradients"]["weight"],
        explicit_gradient["gradients"]["weight"],
        rtol=0,
        atol=0,
    )


def test_sft_replay_adds_worker_normalized_native_gradient(tmp_path: Path) -> None:
    policy = _ReplayPolicy()
    config = _config(tmp_path, coefficient=0.5)
    command = _command(tmp_path, coefficient=0.5)

    result = _compute_gradient_job(
        command,
        policy=policy,
        config=config,
        sft_replay_provider=_ReplayProvider(),
    )
    gradient = load_action_token_gradient_payload(command["gradient_path"])

    assert gradient["gradients"]["weight"].item() == pytest.approx(-0.25)
    assert result["metrics"]["policy_loss"] == pytest.approx(0.0)
    assert result["metrics"]["sft_replay_loss"] == pytest.approx(4.0)
    assert result["metrics"]["sft_replay_weighted_loss"] == pytest.approx(2.0)
    assert result["sft_replay_sample"]["episode_index"] == 17

    aggregate = _aggregate_flow_worker_metrics(
        [result], apply_metrics={}, subupdate_index=5
    )
    assert aggregate["loss"] == pytest.approx(2.0)
    assert aggregate["sft_replay_examples"] == 1.0
