from types import SimpleNamespace

import pytest

from art_embodied.wandb_native_steps import NativeUpdateHistory


class Run:
    def __init__(self, step=0):
        self.step = step
        self.history = []

    def log(self, payload, *, step, commit):
        assert step == self.step and commit is True
        self.history.append({**payload, "_step": step})
        self.step += 1


def test_baseline_training_evaluation_and_media_share_native_updates():
    run = Run()
    history = NativeUpdateHistory(run)
    history.commit(0, {"experiment/update": 0, "validation/success_rate": 0.7})
    video = object()
    for step in range(1, 7):
        history.stage_rollout(
            step - 1, {"train/success_rate": 0.8, "media/train": video}
        )
        assert run.step == step
        payload = {"optimization/loss": 0.1}
        if step == 5:
            payload.update({"validation/success_rate": 0.9, "media/eval": video})
        history.commit(step, history.training_payload(step, payload))
    assert [r["_step"] for r in run.history] == list(range(7))
    assert all(r["_step"] == r["experiment/update"] for r in run.history)
    assert run.history[5]["media/train"] is run.history[5]["media/eval"] is video
    assert run.history[5]["train_details/policy_update"] == 4


def test_resume_has_no_initialization_or_close_row():
    run = Run(step=18)
    history = NativeUpdateHistory(run)
    history.stage_rollout(17, {"media/train": object()})
    history.discard_unfinished()
    assert run.step == 18 and not run.history
    resumed = NativeUpdateHistory(run)
    resumed.stage_rollout(17, {"train/success_rate": 0.8})
    resumed.commit(18, resumed.training_payload(18, {"optimization/loss": 0.1}))
    resumed.discard_unfinished()
    assert len(run.history) == 1 and run.history[0]["_step"] == 18
    assert run.step == 19


@pytest.mark.parametrize("native_step", [17, 19, 34])
def test_resume_rejects_native_step_off_by_one_or_doubled(native_step):
    run = Run(step=native_step)
    with pytest.raises(RuntimeError, match="native step mismatch"):
        NativeUpdateHistory(run).stage_rollout(17, {})
    assert not run.history


def test_wrong_policy_or_missing_rollout_cannot_be_committed():
    history = NativeUpdateHistory(Run(step=1))
    with pytest.raises(RuntimeError, match="no matching"):
        history.training_payload(1, {})
    history.stage_rollout(0, {})
    with pytest.raises(RuntimeError, match="already waiting"):
        history.stage_rollout(0, {})
    with pytest.raises(RuntimeError, match="no matching"):
        history.training_payload(2, {})


def test_native_axis_mismatch_is_rejected():
    run = Run()
    with pytest.raises(RuntimeError, match="must agree"):
        NativeUpdateHistory(run).commit(0, {"experiment/update": 1})
    assert not run.history


def test_noncommitting_sdk_is_not_accepted():
    run = SimpleNamespace(step=0, log=lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="did not acknowledge"):
        NativeUpdateHistory(run).commit(0, {"experiment/update": 0})
