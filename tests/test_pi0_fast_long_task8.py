from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from art_embodied.config import EmbodiedExperimentConfig
from examples.embodied.libero.state_manifest import (
    load_state_manifest,
    state_sha256,
    write_state_manifest,
)
from examples.embodied.pi0_fast_long_task8_run import (
    check_transfer,
    combine_development,
    configure,
    generation_config,
)

SOURCE = Path("outputs/multitask/pi0-fast-long-single-engine-1891")
DEV = Path("examples/embodied/libero/state_manifests/pi05_long_dev_v1/manifest.json")


@pytest.mark.skipif(
    not (SOURCE / "grpo.yaml").exists(), reason="Requires the local measured Long recipe"
)
def test_single_task_preserves_learning_rule(tmp_path):
    from examples.embodied.libero.components import build_train_scenarios

    original = yaml.safe_load((SOURCE / "grpo.yaml").read_text())
    saved = deepcopy(original)
    raw = configure(original, tmp_path)
    assert original == saved
    for key in ("algorithm", "training", "reward", "rollout"):
        assert raw[key] == original[key]
    policy = deepcopy(raw["policy"])
    policy["load_kwargs"]["warm_start_checkpoint"] = original["policy"]["load_kwargs"][
        "warm_start_checkpoint"
    ]
    assert policy == original["policy"]
    raw["environment"]["kwargs"]["evaluation_state_manifest"] = str(DEV.resolve())
    config = EmbodiedExperimentConfig.model_validate(raw)
    assert config.environment.kwargs["training_task_ids"] == [8]
    assert config.environment.kwargs["task_ids"] == [8]
    scenarios = build_train_scenarios(config)
    assert len(scenarios) == 12000
    assert {s.payload["task_id"] for s in scenarios} == {8}
    assert config.evaluation.every_updates == 5
    assert not config.evaluation.evaluate_after_first_update
    assert config.evaluation.episodes == 100
    assert config.storage.resume_from_checkpoint is None
    generation = EmbodiedExperimentConfig.model_validate(generation_config(raw))
    assert not generation.evaluation.enabled
    assert "evaluation_state_manifest" not in generation.environment.kwargs
    assert generation.environment.kwargs["task_ids"] == [8]


def test_development_extension_preserves_original_ten(tmp_path):
    old = load_state_manifest(DEV)
    template = next(e for e in old.metadata["entries"] if e["task_id"] == 8)
    entries, states = [], {}
    for i in range(90):
        state = old.states[template["state_key"]].copy()
        state[0] += 1000 + i
        key = f"extra_{i}"
        states[key] = state
        entries.append(
            dict(
                template,
                id=f"extra-{i}",
                state_key=key,
                state_sha256=state_sha256(state),
            )
        )
    extension = write_state_manifest(
        tmp_path / "extra",
        suite_name=old.suite_name,
        simulator_compatibility=old.simulator_compatibility,
        states=states,
        entries=entries,
        generator={"test_only": True},
    )
    combined = load_state_manifest(
        combine_development(DEV, extension, tmp_path / "combined")
    )
    assert len(combined.entries) == 100
    assert [e.state_sha256 for e in combined.entries[:10]] == [
        e.state_sha256 for e in old.entries if e.task_id == 8
    ]
    assert len({e.id for e in combined.entries}) == 100
    assert not combined.metadata["generator"]["selection_uses_model_outcomes"]


def test_transfer_requires_exact_initial_actions(tmp_path):
    source, root = tmp_path / "source", tmp_path / "run"
    old = [
        {
            "scenario_id": f"task/task-08/state-{i}",
            "success": i < 6,
            "environment_seed": 0,
            "action_sequence_sha256": str(i),
        }
        for i in range(10)
    ]
    new = deepcopy(old) + [
        {
            "scenario_id": f"task/task-08/new-{i}",
            "success": False,
            "environment_seed": 0,
            "action_sequence_sha256": "extra",
        }
        for i in range(90)
    ]
    for directory, rows in ((source, old), (root, new)):
        file = directory / "grpo/evaluation/update_000000_episode_outcomes.json"
        file.parent.mkdir(parents=True)
        file.write_text(json.dumps({"episodes": rows}))
    assert check_transfer(root, source)["successes"] == 6
    new[0]["action_sequence_sha256"] = "different"
    (root / "grpo/evaluation/update_000000_episode_outcomes.json").write_text(
        json.dumps({"episodes": new})
    )
    with pytest.raises(ValueError, match="action_sequence_sha256"):
        check_transfer(root, source)
