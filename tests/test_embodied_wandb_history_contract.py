from types import SimpleNamespace

import pytest

from art_embodied.wandb_history_contract import inspect_metric_rows, verify_run_history


def rows(versions):
    return [
        {"_step": 2 * v + 1, "experiment/update": v, "train/success_rate": 0.5}
        for v in versions
    ]


def test_sdk_steps_are_not_policy_update_numbers():
    result = inspect_metric_rows(
        rows([0, 1, 2, 3]), dict.fromkeys(range(4), 0.5), metric="train/success_rate"
    )
    assert result["storage_verified"]


@pytest.mark.parametrize(
    "versions,kind", [([0, 2, 3], "missing"), ([0, 1, 1, 2, 3], "duplicate")]
)
def test_missing_or_duplicate_updates_fail(versions, kind):
    result = inspect_metric_rows(
        rows(versions), dict.fromkeys(range(4), 0.5), metric="train/success_rate"
    )
    assert not result["storage_verified"]
    assert result["errors"] == [{"kind": kind, "update": 1, "rows": versions.count(1)}]


@pytest.mark.parametrize("axis", [None, True, -1, 1.5, float("nan")])
def test_invalid_or_missing_custom_axis_fails(axis):
    row = rows([0])[0] | {"experiment/update": axis}
    result = inspect_metric_rows([row], {0: 0.5}, metric="train/success_rate")
    assert not result["storage_verified"]
    assert result["errors"][0]["kind"] == "invalid_update_axis"


@pytest.mark.parametrize("value", [0.6, True, float("nan"), float("inf")])
def test_wrong_values_fail(value):
    row = rows([0])[0] | {"train/success_rate": value}
    result = inspect_metric_rows([row], {0: 0.5}, metric="train/success_rate")
    assert result["errors"][0]["kind"] == "value_mismatch"


def test_storage_pass_never_claims_ui_or_experiment_acceptance():
    run = SimpleNamespace(
        url="test/run", state="finished", scan_history=lambda **kwargs: iter(rows([0]))
    )
    result = verify_run_history(run, {0: {"train/success_rate": 0.5}})
    assert result["storage_verified"]
    assert not result["app_rendering_verified"]
    assert not result["experiment_accepted"]


def test_native_contract_rejects_doubled_steps_even_with_correct_custom_axis():
    run = SimpleNamespace(
        url="test/run",
        state="running",
        scan_history=lambda **kwargs: iter(rows([0, 1, 2])),
    )
    report = verify_run_history(
        run,
        {i: {"train/success_rate": 0.5} for i in range(3)},
        require_native_steps=True,
    )
    assert not report["storage_verified"]
    assert not report["native_steps_verified"]
    assert report["native_step_errors"]


def test_native_contract_checks_all_rows_including_an_extra_resume_row():
    values = [
        {"_step": 0, "experiment/update": 0, "validation/success_rate": 0.7},
        {"_step": 1, "experiment/update": 1, "train/success_rate": 0.8},
    ]
    expected = {0: {"validation/success_rate": 0.7}, 1: {"train/success_rate": 0.8}}
    run = SimpleNamespace(
        url="test/run", state="running", scan_history=lambda **kwargs: iter(values)
    )
    assert verify_run_history(run, expected, require_native_steps=True)[
        "native_steps_verified"
    ]
    values.append({"_step": 2, "experiment/update": 1})
    assert not verify_run_history(run, expected, require_native_steps=True)[
        "storage_verified"
    ]
