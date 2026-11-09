import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.embodied.isaaclab.droid_evaluate import bank_indices, policy_seed_base
from examples.embodied.isaaclab.run_droid import (
    assigned_devices,
    partition_indices,
    read_evaluation,
    read_logging_groups,
    resume_configuration,
    train_indices,
    update_limit,
    verify_history,
    wandb_run_path,
)


def test_bounded_run_keeps_the_original_training_target_for_resume():
    assert update_limit(completed=0, total=100, stop_after=1) == 1
    assert update_limit(completed=1, total=100, stop_after=2) == 2
    assert update_limit(completed=2, total=100) == 100


@pytest.mark.parametrize(
    "completed,total,stop", [(0, 100, 0), (0, 100, 101), (1, 100, 1), (100, 100, None)]
)
def test_bounded_run_rejects_empty_or_out_of_range_work(completed, total, stop):
    with pytest.raises(ValueError, match="Stop update"):
        update_limit(completed=completed, total=total, stop_after=stop)


def test_rng_resume_check_detects_changed_state(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    import random

    import numpy as np
    from safetensors.torch import save_file

    from examples.embodied.isaaclab.train_update import assert_restored_rng

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    directory = tmp_path / "policy"
    directory.mkdir()
    original_torch = torch.get_rng_state()
    original_python, original_numpy = random.getstate(), np.random.get_state()
    save_file({"cpu": original_torch}, str(directory / "rng.safetensors"))
    (directory / "rng.json").write_text(
        json.dumps(
            {
                "python": original_python,
                "numpy": [
                    original_numpy[0],
                    original_numpy[1].tolist(),
                    *original_numpy[2:],
                ],
            }
        )
    )
    try:
        assert_restored_rng(tmp_path)
        torch.rand(1)
        with pytest.raises(ValueError, match="Torch RNG"):
            assert_restored_rng(tmp_path)
        torch.set_rng_state(original_torch)
        random.random()
        with pytest.raises(ValueError, match="Python/NumPy RNG"):
            assert_restored_rng(tmp_path)
        random.setstate(original_python)
        np.random.random()
        with pytest.raises(ValueError, match="Python/NumPy RNG"):
            assert_restored_rng(tmp_path)
    finally:
        torch.set_rng_state(original_torch)
        random.setstate(original_python)
        np.random.set_state(original_numpy)


def test_collection_partition_preserves_distinct_groups_and_seed_identity():
    indices = train_indices(512, 64, seed=17, version=0)
    assert indices == train_indices(512, 64, seed=17, version=0)
    assert indices != train_indices(512, 64, seed=17, version=1)
    parts = partition_indices(indices, group_size=8, batch_size=32, devices=8)
    assert [len(p) for p in parts] == [8] * 8
    assert set(sum(parts, [])) == set(indices)
    seeds = {
        policy_seed_base(17000, i, attempt, 8)
        for p in parts
        for i in p
        for attempt in range(8)
    }
    assert len(seeds) == 512
    single = partition_indices(indices, group_size=8, batch_size=32, devices=1)
    assert seeds == {
        policy_seed_base(17000, i, a, 8) for i in single[0] for a in range(8)
    }
    assert partition_indices(
        list(range(128)), group_size=1, batch_size=32, devices=8
    ) == [list(range(i, i + 32)) for i in range(0, 128, 32)]


def test_device_mapping_respects_allocation():
    assert assigned_devices(["cuda:1", "cuda:0"], "GPU-a,GPU-b") == ["GPU-b", "GPU-a"]
    assert assigned_devices(["cuda:3"], None) == ["3"]
    for devices, visible in [
        (["cuda:2"], "0,1"),
        (["cpu"], None),
        (["cuda:0", "cuda:0"], None),
    ]:
        with pytest.raises(ValueError):
            assigned_devices(devices, visible)


@pytest.fixture
def droid_config(tmp_path):
    from pathlib import Path

    import yaml

    from art_embodied.config import EmbodiedExperimentConfig

    recipe = (
        Path(__file__).parents[1]
        / "examples/embodied/isaaclab/pi05_droid_soup_grpo.yaml"
    )
    raw = yaml.safe_load(recipe.read_text())
    raw["storage"]["output_dir"] = str(tmp_path / "outputs")
    raw["evaluation"]["fixed_scenarios"] = [f"development-{i}" for i in range(128)]
    return EmbodiedExperimentConfig.model_validate(raw)


@pytest.mark.parametrize("setting", ["horizon", "physics", "batch"])
def test_collection_rejects_unsupported_settings_before_startup(
    droid_config, setting, monkeypatch
):
    from examples.embodied.isaaclab.run_droid import DroidRunner

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    raw = droid_config.model_dump(mode="json")
    if setting == "horizon":
        raw["rollout"].update(max_episode_steps=150, max_policy_steps=10)
        raw["training"]["optimizer_steps_per_update"] = 4
    elif setting == "physics":
        raw["environment"]["kwargs"]["physics_condition"] = "different"
    else:
        raw["environment"]["kwargs"]["num_envs"] = 16
    droid_config = type(droid_config).model_validate(raw)
    with pytest.raises(ValueError, match="DROID collection"):
        DroidRunner(SimpleNamespace(), droid_config)
    assert not Path(droid_config.storage.output_dir).exists()


def test_collection_forwards_configured_sampler_settings(
    droid_config, tmp_path, monkeypatch
):
    from contextlib import nullcontext

    from examples.embodied.isaaclab import run_droid

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    raw = droid_config.model_dump(mode="json")
    raw["algorithm"]["flow_sde"].update(noise_level=0.35, num_denoise_steps=12)
    droid_config = type(droid_config).model_validate(raw)
    bank = tmp_path / "banks/train"
    bank.mkdir(parents=True)
    (bank / "manifest.json").write_text(json.dumps({"reset_hashes": list(range(512))}))
    calls = []

    def start(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            poll=lambda: None, terminate=lambda: None, wait=lambda **kwargs: 0
        )

    monkeypatch.setattr(run_droid.subprocess, "Popen", start)
    monkeypatch.setattr(run_droid.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_droid, "urlopen", lambda *args, **kwargs: nullcontext())
    runner = run_droid.DroidRunner(
        SimpleNamespace(
            banks=tmp_path / "banks",
            policy_python="policy",
            sim_python="sim",
            port=8000,
        ),
        droid_config,
    )
    roots, _ = runner.collect(version=0, checkpoint=None, training=True)
    assert len(roots) == len(calls) == 8
    for command in calls:
        assert command[command.index("--noise-level") + 1] == "0.35"
        assert command[command.index("--denoising-steps") + 1] == "12"
        assert (
            command[command.index("--config-name") + 1] == "pi05_droid_jointpos_polaris"
        )


def test_wandb_sdk_and_public_api_run_paths():
    assert (
        wandb_run_path(SimpleNamespace(path="entity/project/run"))
        == "entity/project/run"
    )
    assert (
        wandb_run_path(SimpleNamespace(path=["entity", "project", "run"]))
        == "entity/project/run"
    )
    for path in ["project/run", "entity//run", ["entity", "project", "run/extra"]]:
        with pytest.raises(ValueError):
            wandb_run_path(SimpleNamespace(path=path))


@pytest.mark.parametrize("training", [True, False])
def test_collection_survives_pruned_resume_source(
    droid_config, tmp_path, monkeypatch, training
):
    from contextlib import nullcontext
    import shutil

    from art_embodied.checkpointing import CheckpointManager
    from examples.embodied.isaaclab import run_droid
    from examples.embodied.isaaclab.openpi_server import (
        inference_configuration,
        validate_adapter_checkpoint,
    )

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    origin = tmp_path / "old-resume"
    origin.mkdir()
    (origin / "art_embodied_training_state.pt").write_bytes(b"optimizer")
    resumed = resume_configuration(droid_config, checkpoint=origin, run_id="run")
    fingerprint = resumed.resume_contract_fingerprint
    shutil.rmtree(origin)

    def writer(path):
        (path / "policy").mkdir()
        (path / "policy/weights").write_bytes(b"adapter")
        (path / "art_embodied_training_state.pt").write_bytes(b"optimizer")

    current = CheckpointManager().publish(
        tmp_path / "current",
        writer=writer,
        config_fingerprint=resumed.fingerprint,
        resume_contract_fingerprint=fingerprint,
        metadata={"step": 35},
    )
    split = "train" if training else "development"
    bank = tmp_path / "banks" / split
    bank.mkdir(parents=True)
    (bank / "manifest.json").write_text(json.dumps({"reset_hashes": list(range(512))}))
    commands = []

    def start(command, **kwargs):
        commands.append(command)
        path = Path(command[command.index("--config") + 1])
        raw = json.loads(path.read_text())
        assert raw["storage"]["resume_from_checkpoint"] is None
        actual = inference_configuration(path, device="cuda:0")
        assert actual.resume_contract_fingerprint == fingerprint
        adapter = Path(command[command.index("--adapter-checkpoint") + 1])
        assert adapter == current / "policy"
        validate_adapter_checkpoint(adapter, config=actual, policy_version=35)
        return SimpleNamespace(
            poll=lambda: None, terminate=lambda: None, wait=lambda **kwargs: 0
        )

    monkeypatch.setattr(run_droid.subprocess, "Popen", start)
    monkeypatch.setattr(run_droid.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_droid, "urlopen", lambda *args, **kwargs: nullcontext())
    runner = run_droid.DroidRunner(
        SimpleNamespace(
            banks=tmp_path / "banks",
            policy_python="policy",
            sim_python="sim",
            port=8000,
        ),
        resumed,
    )
    roots, _ = runner.collect(version=35, checkpoint=current, training=training)
    assert len(roots) == len(commands) == (8 if training else 4)
    assert resumed.storage.resume_from_checkpoint == origin


def test_inference_can_read_historical_config_after_resume_source_is_pruned(
    droid_config, tmp_path
):
    from examples.embodied.isaaclab.openpi_server import inference_configuration

    raw = droid_config.model_dump(mode="json")
    raw["storage"]["resume_from_checkpoint"] = str(tmp_path / "deleted-checkpoint")
    raw["evaluation"]["evaluate_before_training"] = False
    path = tmp_path / "historical.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="resume"):
        type(droid_config).model_validate(raw)
    inference = inference_configuration(path, device="cuda:0")
    assert inference.storage.resume_from_checkpoint is None
    assert (
        inference.resume_contract_fingerprint
        == droid_config.resume_contract_fingerprint
    )
    assert json.loads(path.read_text()) == raw
    raw["algorithm"]["flow_sde"]["noise_level"] = -1
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        inference_configuration(path, device="cuda:0")


def test_logging_handoff_restores_typed_trajectories_without_dropping_them(tmp_path):
    from art_embodied.trajectories import (
        Action,
        EmbodiedTrajectory,
        EmbodiedTrajectoryGroup,
    )

    members = [
        EmbodiedTrajectory(
            task="move",
            reward=reward,
            actions=[Action(step=0, kind="continuous", raw=[[0.0] * 8])],
        )
        for reward in (0.0, 1.0)
    ]
    original = EmbodiedTrajectoryGroup(members, metadata={"scenario_id": "train-0"})
    path = tmp_path / "groups.json"
    path.write_text(json.dumps([original.model_dump(mode="json")]))
    restored = read_logging_groups(
        path, expected_groups=1, expected_trajectories=2, expected_chunks=2
    )
    assert restored[0].rewards() == [0.0, 1.0]
    assert restored[0].model_dump(mode="json") == original.model_dump(mode="json")
    with pytest.raises(ValueError, match="lost"):
        read_logging_groups(
            path, expected_groups=1, expected_trajectories=4, expected_chunks=2
        )


@pytest.mark.parametrize("corruption", [None, "reward", "scenario", "summary"])
def test_recovery_preserves_rollout_provenance_and_rejects_changes(
    droid_config, tmp_path, monkeypatch, corruption
):
    import asyncio

    from art_embodied.observability import WandbWeaveObserver
    from art_embodied.trajectories import EmbodiedTrajectory, EmbodiedTrajectoryGroup
    from examples.embodied.isaaclab.recover_droid_update import restore_rollout

    groups = [
        EmbodiedTrajectoryGroup(
            [EmbodiedTrajectory(task="move", reward=r) for r in (0.0, 1.0)],
            metadata={"scenario_id": "train-0"},
        )
    ]
    original = WandbWeaveObserver(droid_config)
    asyncio.run(original.log_rollout(34, groups, droid_config))
    path = (
        droid_config.storage.output_dir / "rollout-evidence/policy-version-000034.json"
    )
    before = path.read_bytes()
    checkpoint = tmp_path / "checkpoint35"
    checkpoint.mkdir()
    (checkpoint / "art_embodied_training_state.pt").write_bytes(b"optimizer")
    config = resume_configuration(
        droid_config, checkpoint=checkpoint, run_id="same-run"
    )
    assert config.fingerprint != droid_config.fingerprint
    observer = WandbWeaveObserver(config)
    with pytest.raises(FileExistsError, match="different rollout evidence"):
        asyncio.run(observer.log_rollout(34, groups, config))
    delivered = []
    observer.wandb_run = object()
    monkeypatch.setattr(
        observer,
        "_log_wandb_rollout",
        lambda version, groups, *, evidence: delivered.append(evidence),
    )
    if corruption == "reward":
        groups[0].trajectories[0].reward = 1.0
    elif corruption == "scenario":
        groups[0].metadata["scenario_id"] = "train-1"
    elif corruption == "summary":
        saved = json.loads(before)
        saved["summary"]["trajectories"] += 1
        path.write_text(json.dumps(saved))
        before = path.read_bytes()
    recovery = restore_rollout(
        observer, policy_version=34, groups=groups, config=config
    )
    if corruption:
        with pytest.raises(FileExistsError, match="different rollout evidence"):
            asyncio.run(recovery)
        assert delivered == []
    else:
        asyncio.run(recovery)
        assert delivered == [json.loads(before)]
        assert delivered[0]["config_fingerprint"] == droid_config.fingerprint
    assert path.read_bytes() == before


def test_resume_preserves_contract_without_repeating_initial_evaluation(tmp_path):
    from pathlib import Path

    import yaml

    from art_embodied.config import EmbodiedExperimentConfig

    reference = (
        Path(__file__).parents[1]
        / "examples/embodied/pi05_libero_long_flow_sde_grpo_rlinf_reference_1024.yaml"
    )
    raw = yaml.safe_load(reference.read_text())
    raw["storage"]["output_dir"] = str(tmp_path / "outputs")
    config = EmbodiedExperimentConfig.model_validate(raw)
    checkpoint = tmp_path / "step-000001"
    checkpoint.mkdir()
    (checkpoint / "art_embodied_training_state.pt").write_bytes(b"state")
    resumed = resume_configuration(config, checkpoint=checkpoint, run_id="existing")
    assert resumed.resume_contract_fingerprint == config.resume_contract_fingerprint
    assert not resumed.evaluation.evaluate_before_training
    assert resumed.observability.wandb.run_id == "existing"
    assert resumed.observability.wandb.resume == "must"


def test_reset_index_validation():
    assert bank_indices(count=2, start=4, bank_size=6) == [4, 5]
    assert bank_indices(count=2, selected=[5, 2], bank_size=6) == [5, 2]
    for selected in [[1, 1], [1], [-1, 0], [1, True], [1, 6]]:
        with pytest.raises(ValueError):
            bank_indices(count=2, selected=selected, bank_size=6)
    with pytest.raises(ValueError):
        partition_indices([0, 1], group_size=8, batch_size=32, devices=1)
    with pytest.raises(ValueError):
        train_indices(32, 64, seed=17, version=0)


def test_evaluation_keeps_actions_initial_state_and_policy_version(tmp_path):
    report = {
        "model_server": {"mode": "ode"},
        "optimizer_updates": 5,
        "split": "development",
        "initial_state_and_camera_replay_exact": True,
        "episodes": 1,
    }
    episode = {
        "bank_index": 0,
        "reset_sha256": "hash",
        "batch": 0,
        "row": 0,
        "success": True,
        "env_steps": 1,
        "environment_seed": 23000,
        "policy_seed_base": 10,
        "termination_reason": "success",
        "media": [],
    }
    chunk = {
        "batch": 0,
        "chunk": 0,
        "active_rows": [0],
        "actions": [[[0.0] * 8]],
        "execution_mask": [[True]],
    }
    for name, data in [
        ("report", report),
        ("episodes", [episode]),
        ("actions", [chunk]),
    ]:
        (tmp_path / f"{name}.json").write_text(json.dumps(data))
    config = SimpleNamespace(
        environment=SimpleNamespace(kwargs={"instruction": "move"}),
        evaluation=SimpleNamespace(episodes=1),
    )
    bank = SimpleNamespace(
        hashes=["hash"], select=lambda indices: {"position": [[1, 2, 3]]}
    )
    result = read_evaluation([tmp_path], version=5, config=config, bank=bank)
    assert result.metrics["success_rate"] == 1
    assert result.trajectories[0].initial_policy_version == 5
    assert result.trajectories[0].metadata["reset_state"] == {"position": [[1, 2, 3]]}
    assert result.trajectories[0].actions[0].raw == [[0.0] * 8]
    report["split"] = "sealed"
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="fixed evaluation reset"):
        read_evaluation([tmp_path], version=5, config=config, bank=bank)
    sealed = read_evaluation(
        [tmp_path], version=5, config=config, bank=bank, split="sealed"
    )
    assert sealed.trajectories[0].metadata["scenario_id"] == "sealed-0"
    assert sealed.trajectories[0].metadata["environment_seed"] == 23000
    report["split"] = "development"
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        read_evaluation([tmp_path, tmp_path], version=5, config=config, bank=bank)
    with pytest.raises(ValueError):
        read_evaluation([tmp_path], version=4, config=config, bank=bank)


def test_history_verifier_rejects_native_step_gaps(monkeypatch):
    wandb = pytest.importorskip("wandb")

    remote = SimpleNamespace(scan_history=lambda: [{"_step": 0}, {"_step": 2}])
    monkeypatch.setattr(
        wandb, "Api", lambda **kwargs: SimpleNamespace(run=lambda _: remote)
    )
    with pytest.raises(RuntimeError, match="native W&B steps"):
        verify_history("entity/project/run", last_step=2, expected={}, timeout=1)


def test_collector_refuses_training_on_sealed_states():
    from examples.embodied.isaaclab.run_droid import DroidRunner

    runner = DroidRunner.__new__(DroidRunner)
    runner.config = SimpleNamespace()
    with pytest.raises(ValueError, match="cannot be used for training"):
        runner.collect(
            version=0, checkpoint=None, training=True, evaluation_split="sealed"
        )


def test_history_verifier_checks_values_media_and_models(monkeypatch):
    wandb = pytest.importorskip("wandb")

    rows = [
        {
            "_step": 0,
            "experiment/update": 0,
            "validation/success_rate": 0.5,
            "media/simulation/eval/0": {"path": "video.gif"},
        }
    ]
    remote = SimpleNamespace(
        scan_history=lambda: rows,
        files=lambda **kwargs: [SimpleNamespace(name="video.gif", size=100)],
        logged_artifacts=lambda: [],
    )
    monkeypatch.setattr(
        wandb, "Api", lambda **kwargs: SimpleNamespace(run=lambda _: remote)
    )
    expected = {"0": {"metrics": {"validation/success_rate": 0.5}, "media": 1}}
    assert verify_history("e/p/r", last_step=0, expected=expected, timeout=1)[
        "native_steps"
    ] == [0]
    rows[0]["validation/success_rate"] = 0.6
    with pytest.raises(RuntimeError, match="metric mismatch"):
        verify_history("e/p/r", last_step=0, expected=expected, timeout=1)


def test_history_verifier_requires_model_for_each_actual_update(monkeypatch):
    wandb = pytest.importorskip("wandb")
    from examples.embodied.isaaclab import run_droid

    models = [SimpleNamespace(type="model", name="wrong:v0", metadata={"update": 2})]
    remote = SimpleNamespace(
        scan_history=lambda: [
            {"_step": step, "experiment/update": step} for step in (0, 1)
        ],
        files=lambda **kwargs: [],
        logged_artifacts=lambda: models,
    )
    monkeypatch.setattr(
        wandb, "Api", lambda **kwargs: SimpleNamespace(run=lambda _: remote)
    )
    times = iter([0, 0, 2])
    monkeypatch.setattr(run_droid.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(run_droid.time, "sleep", lambda _: None)
    with pytest.raises(TimeoutError):
        verify_history("e/p/r", last_step=1, expected={}, timeout=1)
    models[0].metadata["update"] = 1
    times = iter([0, 0])
    verified = verify_history("e/p/r", last_step=1, expected={}, timeout=1)
    assert verified["model_artifacts_by_update"] == {1: "wrong:v0"}


def test_history_readback_retries_incomplete_file_page_without_losing_checks(
    monkeypatch,
):
    wandb = pytest.importorskip("wandb")
    from examples.embodied.isaaclab import run_droid

    calls = []

    def files(*, per_page):
        calls.append(per_page)
        yield SimpleNamespace(name="first.gif", size=100)
        if len(calls) == 1:
            raise TypeError("'NoneType' object is not subscriptable")
        yield SimpleNamespace(name="second.gif", size=200)

    row = {
        "_step": 0,
        "experiment/update": 0,
        "validation/success_rate": 0.5,
        "media/simulation/eval/0": {"path": "first.gif"},
        "media/simulation/eval/1": {"path": "second.gif"},
    }
    remote = SimpleNamespace(
        scan_history=lambda: [row], files=files, logged_artifacts=lambda: []
    )
    monkeypatch.setattr(
        wandb, "Api", lambda **kwargs: SimpleNamespace(run=lambda _: remote)
    )
    monkeypatch.setattr(run_droid.time, "sleep", lambda _: None)
    expected = {"0": {"metrics": {"validation/success_rate": 0.5}, "media": 2}}
    result = verify_history("e/p/r", last_step=0, expected=expected, timeout=1)
    assert calls == [500, 500]
    assert result["native_steps"] == [0]
    row["validation/success_rate"] = 0.6
    with pytest.raises(RuntimeError, match="metric mismatch"):
        verify_history("e/p/r", last_step=0, expected=expected, timeout=1)
    assert calls == [500, 500, 500]


@pytest.mark.parametrize("kind", ["communication", "timeout", "null_page"])
def test_wandb_readback_retries_only_reads_and_has_a_deadline(monkeypatch, kind):
    wandb = pytest.importorskip("wandb")
    from requests.exceptions import Timeout

    from examples.embodied.isaaclab import run_droid

    error = {
        "communication": wandb.errors.CommError("temporarily unavailable"),
        "timeout": Timeout("request timed out"),
        "null_page": TypeError("'NoneType' object is not subscriptable"),
    }[kind]
    now, calls = [0], []
    monkeypatch.setattr(run_droid.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        run_droid.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )

    def operation():
        calls.append(now[0])
        raise error

    with pytest.raises(TimeoutError, match="did not recover") as caught:
        run_droid.read_wandb(operation, deadline=25, label="media files")
    assert calls == [0, 10, 20, 25]
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("kind", ["authentication", "programming", "validation"])
def test_wandb_readback_does_not_retry_auth_or_code_errors(monkeypatch, kind):
    wandb = pytest.importorskip("wandb")
    from examples.embodied.isaaclab import run_droid

    error = {
        "authentication": wandb.errors.AuthenticationError("invalid credentials"),
        "programming": TypeError("unexpected argument"),
        "validation": RuntimeError("W&B metric mismatch"),
    }[kind]
    monkeypatch.setattr(
        run_droid.time, "sleep", lambda _: pytest.fail("Must not retry")
    )

    def operation():
        raise error

    with pytest.raises(type(error)) as caught:
        run_droid.read_wandb(operation, deadline=100, label="history")
    assert caught.value is error


def test_resume_check_compares_optimizer_moments_not_only_step():
    torch = pytest.importorskip("torch")

    from examples.embodied.isaaclab.train_update import (
        assert_state_equal,
        optimizer_steps,
    )

    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter])
    parameter.sum().backward()
    optimizer.step()
    assert optimizer_steps(optimizer) == [1]
    restored = torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))])
    restored.load_state_dict(optimizer.state_dict())
    assert_state_equal(restored.state_dict(), optimizer.state_dict())
    changed = {"state": {"exp_avg": torch.tensor([0.1, 0.2])}, "step": 1}
    with pytest.raises(ValueError, match="tensor differs"):
        assert_state_equal(
            changed, {"state": {"exp_avg": torch.tensor([0.1, 0.3])}, "step": 1}
        )


@pytest.mark.parametrize(
    "extra", [["--resume-from", "checkpoint"], ["--run-id", "existing"]]
)
def test_resume_cli_requires_checkpoint_and_same_run_together(monkeypatch, extra):
    from examples.embodied.isaaclab.run_droid import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_droid",
            "--config",
            "unused.json",
            "--banks",
            "unused",
            "--policy-python",
            "python",
            "--sim-python",
            "python",
            *extra,
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
