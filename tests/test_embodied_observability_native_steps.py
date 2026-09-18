import asyncio
import json

import pytest

from art_embodied.config import EmbodiedExperimentConfig
from art_embodied.experiment import EvaluationResult
from art_embodied.observability import WandbWeaveObserver
from art_embodied.trajectories import (
    EmbodiedTrajectory,
    EmbodiedTrajectoryGroup,
    MediaRef,
)
from art_embodied.types import LocalTrainResult
from tests.test_embodied_observability import (
    _CommittedRunWithNativeStep,
    _config,
    _Wandb,
)


def native_config(tmp_path):
    raw = _config(tmp_path).model_dump(mode="python")
    raw["observability"]["wandb"]["native_update_steps"] = True
    raw["observability"]["delivery_failure_policy"] = "fail_run"
    return EmbodiedExperimentConfig.model_validate(raw)


def test_native_observer_new_resume_eval_and_close(tmp_path):
    config = native_config(tmp_path)
    run = _CommittedRunWithNativeStep()
    observer = WandbWeaveObserver(
        config, wandb_run=run, wandb_module=_Wandb, enforce_initial_history_commit=True
    )
    path = tmp_path / "video.gif"
    path.write_bytes(b"GIF89a")
    trajectory = EmbodiedTrajectory(
        task="pick cube",
        reward=1.0,
        metrics={"success": True},
        media=[MediaRef(uri=path.as_uri(), kind="video", mime_type="image/gif")],
    )
    groups = [EmbodiedTrajectoryGroup([trajectory])]
    evaluation = lambda step: EvaluationResult(
        step=step,
        metrics={"success_rate": 0.75},
        artifacts={},
        trajectories=(trajectory,),
    )
    asyncio.run(observer.log_initial_evaluation(evaluation(0), config))
    for step in range(1, 6):
        if step == 3:
            observer.close()
            assert run.step == 3
            raw = config.model_dump(mode="python")
            raw["observability"]["wandb"].update(
                connection="resume", run_id=run.id, resume="must"
            )
            config = EmbodiedExperimentConfig.model_validate(raw)
            observer = WandbWeaveObserver(config, wandb_run=run, wandb_module=_Wandb)
            assert run.step == 3
        asyncio.run(observer.log_rollout(step - 1, groups, config))
        assert run.step == step
        asyncio.run(
            observer.log_step(
                step,
                groups,
                LocalTrainResult(step=step, metrics={"loss": 0.1}),
                evaluation(step) if step == 5 else None,
                config,
            )
        )
        assert run.step == step + 1
    observer.close()
    assert run.step == 6
    assert [s for _, s in run.logs] == list(range(6))
    assert all(s == p["experiment/update"] for p, s in run.logs)
    assert "media/simulation/eval/0" in run.history[0]
    assert all("media/simulation/train/0" in row for row in run.history[1:])
    assert "media/simulation/eval/0" in run.history[5]
    assert run.history[5]["train_details/policy_update"] == 4
    assert "train/success_rate" not in run.history[0]


def test_unfinished_rollout_close_does_not_create_a_fake_update(tmp_path):
    config = native_config(tmp_path)
    run = _CommittedRunWithNativeStep()
    observer = WandbWeaveObserver(config, wandb_run=run, wandb_module=_Wandb)
    asyncio.run(
        observer.log_initial_evaluation(
            EvaluationResult(step=0, metrics={"success_rate": 0.5}, artifacts={}),
            config,
        )
    )
    asyncio.run(
        observer.log_rollout(
            0,
            [EmbodiedTrajectoryGroup([EmbodiedTrajectory(task="x", reward=1.0)])],
            config,
        )
    )
    observer.close(exit_code=1)
    assert run.step == 1 and len(run.history) == 1


def test_legacy_recipe_fingerprint_is_unchanged(tmp_path):
    import hashlib

    config = _config(tmp_path)
    document = config.model_dump(mode="json")
    document["observability"]["wandb"].pop("native_update_steps")
    document["observability"]["wandb"].pop("input_model_artifact_ref", None)
    if document["rollout"].get("shared_prefix_action_chunks") == 0:
        document["rollout"].pop("shared_prefix_action_chunks")
    assert (
        config.fingerprint
        == hashlib.sha256(
            json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
    )


def test_shared_writer_cannot_use_native_steps(tmp_path):
    raw = native_config(tmp_path).model_dump(mode="python")
    raw["observability"]["wandb"]["connection"] = "shared_primary"
    with pytest.raises(ValueError, match="single W&B history writer"):
        EmbodiedExperimentConfig.model_validate(raw)
