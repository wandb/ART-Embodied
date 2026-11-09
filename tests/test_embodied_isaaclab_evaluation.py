import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from art_embodied.checkpointing import CheckpointManager
from art_embodied.experiment import EvaluationResult
from art_embodied.integrations.isaaclab_reset_bank import IsaacLabResetBank
from art_embodied.trajectories import Action, EmbodiedTrajectory
from examples.embodied.isaaclab.droid_environment import DroidSceneConfig
from examples.embodied.isaaclab.evaluate_droid import (
    evaluation_config,
    outcome_rows,
    record_evidence,
    verify_campaign,
)
from examples.embodied.isaaclab.prepare_droid import build_config


@pytest.fixture
def reference(tmp_path):
    model = tmp_path / "model"
    (model / "assets/droid").mkdir(parents=True)
    (model / "model.safetensors").write_bytes(b"model")
    (model / "assets/droid/norm_stats.json").write_text("{}")
    scene = DroidSceneConfig(
        "alphabet_soup_can_hope_robolab",
        "bowl_ycb_robolab",
        "Pick up the soup can and place it in the bowl.",
    )
    banks = tmp_path / "banks"
    offset = 0
    for name, count in (("train", 512), ("development", 128), ("sealed", 256)):
        IsaacLabResetBank.from_batches(
            [
                {
                    "robot": {
                        "joint": torch.arange(offset, offset + count)[:, None].float()
                    }
                }
            ],
            contract=scene.reset_contract(),
            split=name,
        ).save(banks / name)
        offset += count
    config = build_config(
        checkpoint=model,
        banks=banks,
        output_dir=tmp_path / "training",
        policy_python=Path("/policy/bin/python"),
        entity="test",
    )
    return config, banks


@pytest.mark.parametrize(
    "split,count,role",
    [("development", 128, "development"), ("sealed", 256, "sealed_test")],
)
def test_evaluation_uses_disjoint_bank_and_keeps_policy_contract(
    reference, tmp_path, split, count, role
):
    training, banks = reference
    config, bank = evaluation_config(
        training, banks=banks, output=tmp_path / "evaluation", split=split
    )
    assert config.resume_contract_fingerprint == training.resume_contract_fingerprint
    assert config.evaluation.data_role == role and len(bank) == count
    assert config.evaluation.kwargs["reset_bank_sha256"] == bank.fingerprint
    assert config.evaluation.fixed_scenarios == [f"{split}-{i}" for i in range(count)]
    assert not config.observability.wandb.native_update_steps
    assert config.observability.wandb.connection == "primary"
    assert config.observability.wandb.run_id is None
    assert config.observability.videos_per_evaluation == count
    assert config.observability.wandb.max_evaluation_table_rows == count
    assert config.storage.resume_from_checkpoint is None


def test_evaluation_rejects_training_bank_and_changed_identity(reference, tmp_path):
    training, banks = reference
    with pytest.raises(ValueError, match="development or sealed"):
        evaluation_config(
            training, banks=banks, output=tmp_path / "eval", split="train"
        )
    training.environment.kwargs["train_reset_bank_sha256"] = "changed"
    with pytest.raises(ValueError, match="differ from"):
        evaluation_config(
            training, banks=banks, output=tmp_path / "eval", split="sealed"
        )


def result(step, successes, *, policy_seed=33000):
    trajectories = []
    for index, success in enumerate(successes):
        trajectories.append(
            EmbodiedTrajectory(
                task="move",
                reward=float(success),
                metrics={"success": float(success), "environment_steps": 1},
                metadata={
                    "bank_index": index,
                    "scenario_id": f"sealed-{index}",
                    "environment_seed": 33000,
                    "policy_seed": policy_seed + index,
                    "reset_sha256": f"state-{index}",
                    "termination_reason": "success" if success else "time_limit",
                },
                actions=[Action(step=0, kind="continuous", raw=[[0.0] * 8])],
            )
        )
    return EvaluationResult(
        step=step,
        metrics={
            "success_rate": sum(successes) / len(successes),
            "episodes": len(successes),
            "successes": sum(successes),
        },
        artifacts={},
        trajectories=tuple(reversed(trajectories)),
    )


def test_evidence_sorts_completion_order_and_reuses_paired_statistics(
    reference, tmp_path
):
    training, banks = reference
    config, _ = evaluation_config(
        training, banks=banks, output=tmp_path / "evaluation", split="sealed"
    )
    config = config.model_copy(
        update={
            "evaluation": config.evaluation.model_copy(
                update={"episodes": 2, "fixed_scenarios": ["sealed-0", "sealed-1"]}
            )
        }
    )
    baseline = record_evidence(
        result(0, [False, True]),
        config=config,
        bank_path=banks / "sealed",
        checkpoint=None,
    )
    checkpoint = tmp_path / "step-5"
    CheckpointManager().publish(
        checkpoint,
        writer=lambda path: (
            (path / "weights").write_bytes(b"adapter"),
            (path / "art_embodied_training_state.pt").write_bytes(b"state"),
        ),
        config_fingerprint=training.fingerprint,
        resume_contract_fingerprint=training.resume_contract_fingerprint,
        metadata={"step": 5},
    )
    candidate = record_evidence(
        result(5, [True, True]),
        config=config,
        bank_path=banks / "sealed",
        checkpoint=checkpoint,
        baseline=baseline.artifacts["episode_outcomes_json"],
    )
    assert candidate.metrics["success_rate_lift"] == 0.5
    assert candidate.metrics["improved_pairs"] == 1
    assert candidate.metrics["regressed_pairs"] == 0
    rows = outcome_rows(candidate)
    assert [row["episode"] for row in rows] == [0, 1]
    evidence = json.loads(
        Path(candidate.artifacts["evaluation_evidence_json"]).read_text()
    )
    assert evidence["evaluation"]["data_role"] == "sealed_test"
    assert evidence["identity"]["policy"]["evaluated_state"]["status"] == "checkpoint"
    config = config.model_copy(
        update={
            "storage": config.storage.model_copy(
                update={"output_dir": tmp_path / "different-seed"}
            )
        }
    )
    with pytest.raises(ValueError, match="episode mismatch"):
        record_evidence(
            result(5, [True, True], policy_seed=33001),
            config=config,
            bank_path=banks / "sealed",
            checkpoint=checkpoint,
            baseline=baseline.artifacts["episode_outcomes_json"],
        )


def test_evidence_rejects_partial_population_or_invented_update(reference, tmp_path):
    training, banks = reference
    config, _ = evaluation_config(
        training, banks=banks, output=tmp_path / "eval", split="sealed"
    )
    with pytest.raises(ValueError, match="episode population"):
        record_evidence(
            result(0, [True]),
            config=config,
            bank_path=banks / "sealed",
            checkpoint=None,
        )
    config = config.model_copy(
        update={
            "evaluation": config.evaluation.model_copy(
                update={"episodes": 1, "fixed_scenarios": ["sealed-0"]}
            )
        }
    )
    with pytest.raises(ValueError, match="selected checkpoint"):
        record_evidence(
            result(5, [True]),
            config=config,
            bank_path=banks / "sealed",
            checkpoint=None,
        )
    invalid = result(0, [True])
    invalid.metrics["success_rate"] = 0
    with pytest.raises(ValueError, match="episode outcomes"):
        record_evidence(
            invalid, config=config, bank_path=banks / "sealed", checkpoint=None
        )


@pytest.mark.parametrize(
    "defect", [None, "extra_row", "wrong_axis", "wrong_metric", "missing_media"]
)
def test_evaluation_campaign_native_axes_media_and_artifacts(monkeypatch, defect):
    import wandb

    expected = {
        0: {"success_rate": 0.5, "episodes": 1},
        5: {"success_rate": 1.0, "episodes": 1},
    }
    rows = [
        {
            "_step": step,
            "experiment/update": step,
            "test/success_rate": values["success_rate"],
            "media/simulation/eval/0": {"path": f"video-{step}"},
        }
        for step, values in expected.items()
    ]
    if defect == "extra_row":
        rows.insert(1, {"_step": 1})
    elif defect == "wrong_axis":
        rows[1]["experiment/update"] = 4
    elif defect == "wrong_metric":
        rows[1]["test/success_rate"] = 0.9
    elif defect == "missing_media":
        rows[1].pop("media/simulation/eval/0")
    remote = SimpleNamespace(
        scan_history=lambda: rows,
        files=lambda **kwargs: [
            SimpleNamespace(name=f"video-{step}", size=123) for step in expected
        ],
        logged_artifacts=lambda: [
            SimpleNamespace(
                name=f"evaluation:v{i}", type="evaluation", metadata={"update": step}
            )
            for i, step in enumerate(expected)
        ],
    )
    monkeypatch.setattr(
        wandb, "Api", lambda **kwargs: SimpleNamespace(run=lambda path: remote)
    )
    if defect:
        with pytest.raises(RuntimeError):
            verify_campaign("entity/project/run", expected, namespace="test", timeout=1)
    else:
        verified = verify_campaign(
            "entity/project/run", expected, namespace="test", timeout=1
        )
        assert verified["native_steps"] == [0, 5]
        assert not verified["ui_checked"]
