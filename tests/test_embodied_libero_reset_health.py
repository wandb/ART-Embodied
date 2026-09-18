from copy import deepcopy
from types import SimpleNamespace

import pytest

from examples.embodied.libero.audit_reset import (
    TASK5_BDDL_SHA256,
    TASK5_RELATION,
    classify,
    exit_code,
    manifest_task_states,
    select_indices,
)
from examples.embodied.libero.compare_reset_audits import compare
from examples.embodied.libero.compare_reset_images import image_metrics


def test_sampling_is_outcome_independent():
    assert select_indices(50, 3) == [0, 24, 49]
    assert select_indices(50, None) == list(range(50))
    for n in (0, -1, 51):
        with pytest.raises(ValueError):
            select_indices(50, n)


def test_manifest_audit_preserves_order_and_rejects_task_mismatch():
    entries = [
        SimpleNamespace(task_id=0, state_key="z", bddl_sha256="bddl"),
        SimpleNamespace(task_id=1, state_key="other", bddl_sha256="other-bddl"),
        SimpleNamespace(task_id=0, state_key="a", bddl_sha256="bddl"),
    ]
    manifest = SimpleNamespace(
        entries=entries, states={"z": [3], "a": [1], "other": [2]}
    )
    selected, states = manifest_task_states(manifest, 0, "bddl")
    assert [e.state_key for e in selected] == ["z", "a"]
    assert states == [[3], [1]]
    with pytest.raises(ValueError):
        manifest_task_states(manifest, 0, "wrong")
    with pytest.raises(ValueError):
        manifest_task_states(manifest, 2, "bddl")


def test_reference_replay_rejects_seed_or_population_drift():
    from examples.embodied.pi05_libero_reset_audit import validate_reference

    manifest = SimpleNamespace(
        path="/repo/dev/manifest.json", entries=[SimpleNamespace(id="state")]
    )
    config = {
        "environment": {
            "task": "libero_10",
            "reset": {"reset_gripper_open": True},
            "kwargs": {
                "task_ids": list(range(10)),
                "wait_steps_after_reset": 15,
                "control_mode": "unchanged",
                "observation_height": 256,
                "observation_width": 256,
                "rotate_images_180": True,
                "evaluation_state_manifest": "dev/manifest.json",
            },
        },
        "evaluation": {
            "seeds": [0],
            "episodes": 100,
            "fixed_scenarios": ["state"],
            "kwargs": {
                "seed_contract": {
                    "environment_mode": "fixed",
                    "fixed_environment_seed": 0,
                }
            },
        },
    }
    validate_reference(config, manifest)
    changed = deepcopy(config)
    changed["evaluation"]["fixed_scenarios"] = ["another-state"]
    with pytest.raises(ValueError):
        validate_reference(changed, manifest)
    config["environment"]["kwargs"]["wait_steps_after_reset"] = 10
    with pytest.raises(ValueError):
        validate_reference(config, manifest)


def test_image_comparison_does_not_wrap_unsigned_values():
    import numpy as np

    a = np.zeros((2, 2, 3), dtype=np.uint8)
    b = np.full_like(a, 255)
    assert image_metrics(a, b)["mae_0_255"] == 255
    assert image_metrics(a, a)["review_needed"] is False
    with pytest.raises(ValueError):
        image_metrics(a, b[:1])
    with pytest.raises(ValueError):
        image_metrics(a.astype(float), b)


def test_known_support_failure_is_an_error():
    p = [{"predicate": list(TASK5_RELATION), "value": False, "error": None}]
    result = classify(TASK5_BDDL_SHA256, p, finite=True, initial_success=False)
    assert result["failures"] == ["spatial_task5_bowl_not_on_ramekin"]
    report = {"completed": True, "tasks": [{"episodes": [result]}]}
    assert exit_code(report) == 2
    assert exit_code(report, report_only=True) == 0
    assert exit_code({"completed": False}, report_only=True) == 2


def test_unknown_or_changed_bddl_is_not_certified():
    p = [{"predicate": ["on", "bowl", "region"], "value": False}]
    result = classify("unreviewed-bddl", p, finite=True, initial_success=False)
    assert not result["failures"]
    assert result["semantic_qualification"] == "not_qualified"
    assert result["unvalidated_false_predicates"] == [["on", "bowl", "region"]]


def test_correct_support_does_not_force_all_placement_predicates():
    p = [
        {"predicate": list(TASK5_RELATION), "value": True},
        {"predicate": ["on", "fixture", "region"], "value": False},
    ]
    result = classify(TASK5_BDDL_SHA256, p, finite=True, initial_success=False)
    assert not result["failures"]
    assert len(result["unvalidated_false_predicates"]) == 1


def test_missing_known_predicate_and_errors_are_not_success():
    assert classify(TASK5_BDDL_SHA256, [], finite=True, initial_success=False)[
        "failures"
    ]
    result = classify(
        "unknown",
        [{"predicate": ["on"], "error": "unsupported"}],
        finite=False,
        initial_success=True,
    )
    assert set(result["failures"]) == {
        "predicate_check_incomplete",
        "non_finite_state",
        "already_successful_at_reset",
    }


def test_cli_report_only_cannot_hide_execution_error(monkeypatch, tmp_path, capsys):
    from examples.embodied.libero import audit_reset

    def fail(args):
        raise RuntimeError("missing simulator")

    monkeypatch.setattr(audit_reset, "audit", fail)
    assert audit_reset.main(["--output", str(tmp_path), "--report-only"]) == 2
    assert "missing simulator" in capsys.readouterr().err


def test_cli_rejects_duplicate_suites_before_simulator_import(tmp_path):
    from examples.embodied.libero import audit_reset

    with pytest.raises(SystemExit) as exc:
        audit_reset.main(
            ["--output", str(tmp_path), "--suites", "libero_10", "libero_10"]
        )
    assert exc.value.code == 2


def audit():
    return {
        "completed": True,
        "reset": {"seed": 0, "wait_steps": 10},
        "versions": {"mujoco": "3.3.0", "robosuite": "1.4.0"},
        "tasks": [
            {
                "suite": "libero_spatial",
                "task_id": 5,
                "task_name": "bowl",
                "bddl_sha256": "bddl",
                "bank_sha256": "bank",
                "bank_size": 50,
                "episodes": [
                    {
                        "index": 0,
                        "source_state_sha256": "state",
                        "predicates": [{"predicate": ["on", "a", "b"], "value": True}],
                        "objects": {"a": {"position": [0, 0, 0]}},
                    }
                ],
            }
        ],
    }


def test_comparison_detects_drift_without_claiming_which_is_correct():
    left, right = audit(), audit()
    right["versions"]["mujoco"] = "3.8.1"
    right["tasks"][0]["episodes"][0]["predicates"][0]["value"] = False
    right["tasks"][0]["episodes"][0]["objects"]["a"]["position"][0] = 0.02
    before = deepcopy(left)
    result = compare(left, right)
    assert left == before
    assert result["review_tasks"] == 1
    assert result["tasks"][0]["max_object_position_delta_m"] == 0.02


def test_timeline_comparison_separates_transient_and_remaining_difference():
    from examples.embodied.summarize_pi05_reset_audit import timeline_comparison

    left, right = audit(), audit()
    for report in (left, right):
        report["tasks"][0]["controller_use_delta"] = [True]
        report["tasks"][0]["episodes"][0]["reset_timeline"] = [
            {"control_step": step, "objects": {"a": {"position": [0, 0, 0]}}}
            for step in (0, 10, 15)
        ]
    right["tasks"][0]["episodes"][0]["reset_timeline"][1]["objects"]["a"]["position"][
        2
    ] = 0.54
    result = timeline_comparison(left, right)["tasks"][0]["timeline"]
    assert result[1]["max_position_delta_m"] == 0.54
    assert result[1]["states_with_delta_above_5cm"] == 1
    assert result[2]["max_position_delta_m"] == 0
    right["tasks"][0]["controller_use_delta"] = [False]
    with pytest.raises(ValueError):
        timeline_comparison(left, right)


@pytest.mark.parametrize("field", ["bddl_sha256", "bank_sha256", "bank_size"])
def test_comparison_rejects_confounders(field):
    left, right = audit(), audit()
    right["tasks"][0][field] = "changed"
    with pytest.raises(ValueError):
        compare(left, right)


def test_comparison_rejects_changed_reset_or_other_package():
    for section, key, value in [
        ("reset", "seed", 1),
        ("versions", "robosuite", "other"),
    ]:
        left, right = audit(), audit()
        right[section][key] = value
        with pytest.raises(ValueError):
            compare(left, right)
