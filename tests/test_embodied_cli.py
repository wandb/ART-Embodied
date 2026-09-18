from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path

from typer.testing import CliRunner
import yaml

from art_embodied.cli import app
from art_embodied.compatibility import TESTED_ART_VERSION, VALIDATED_PI_PACKAGES

runner = CliRunner()


def _positive_control() -> Path:
    return (
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml"
    )


def _sft_baseline() -> Path:
    return (
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_libero_object_sft_baseline_eval.yaml"
    )


def _public_grpo_evaluation() -> Path:
    return (
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_libero_object_public_grpo_eval.yaml"
    )


def _pi05_single_gpu() -> Path:
    return (
        Path(__file__).parents[1]
        / "examples/embodied/pi05_libero_object_flow_sde_grpo_single_gpu.yaml"
    )


def test_embodied_doctor_reports_control_plane_compatibility(monkeypatch) -> None:
    versions = {
        "openpipe-art": str(TESTED_ART_VERSION),
        "lerobot": "0.4.4",
        "diffusers": "0.38.0",
    }
    monkeypatch.setattr(metadata, "version", versions.__getitem__)

    result = runner.invoke(
        app,
        ["doctor", "--require-lerobot", "--json"],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["mode"] == "control-plane"
    assert report["art_version"] == str(TESTED_ART_VERSION)
    assert report["lerobot_version"] == "0.4.4"
    assert report["compatible"] is True
    assert report["issues"] == []


def test_embodied_doctor_reports_pi_runtime_compatibility(monkeypatch) -> None:
    versions = {
        "openpipe-art": str(TESTED_ART_VERSION),
        "lerobot": "0.6.0",
        **VALIDATED_PI_PACKAGES,
    }
    monkeypatch.setattr(metadata, "version", versions.__getitem__)

    result = runner.invoke(app, ["doctor", "--profile", "pi", "--json"])

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["profile"] == "pi"
    assert report["lerobot_version"] == "0.6.0"
    assert report["compatible"] is True


def test_embodied_doctor_fails_with_actionable_version_issue(monkeypatch) -> None:
    versions = {"openpipe-art": "0.6.0"}

    def version(name: str) -> str:
        try:
            return versions[name]
        except KeyError as exc:
            raise metadata.PackageNotFoundError(name) from exc

    monkeypatch.setattr(metadata, "version", version)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "status: incompatible" in result.output
    assert f"expected {TESTED_ART_VERSION}" in result.output
    assert "found 0.6.0" in result.output


def test_embodied_doctor_worker_mode_does_not_require_art(monkeypatch) -> None:
    def missing(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing)

    result = runner.invoke(app, ["doctor", "--worker", "--json"])

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["mode"] == "worker"
    assert report["art_version"] is None
    assert report["compatible"] is True


def test_embodied_doctor_validates_pinned_libero_worker_profile(monkeypatch) -> None:
    from art_embodied.compatibility import VALIDATED_OPENVLA_OFT_V01_PACKAGES

    versions = dict(VALIDATED_OPENVLA_OFT_V01_PACKAGES)

    def version(name: str) -> str:
        if name == "openpipe-art":
            return str(TESTED_ART_VERSION)
        try:
            return versions[name]
        except KeyError as exc:
            raise metadata.PackageNotFoundError(name) from exc

    monkeypatch.setattr(metadata, "version", version)

    result = runner.invoke(app, ["doctor", "--profile", "libero", "--json"])

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["profile"] == "libero"
    assert report["package_versions"] == versions
    assert report["compatible"] is True


def test_embodied_doctor_reports_libero_profile_drift(monkeypatch) -> None:
    from art_embodied.compatibility import VALIDATED_OPENVLA_OFT_V01_PACKAGES

    versions = dict(VALIDATED_OPENVLA_OFT_V01_PACKAGES)
    versions["transformers"] = "4.57.3"

    def version(name: str) -> str:
        if name == "openpipe-art":
            return str(TESTED_ART_VERSION)
        try:
            return versions[name]
        except KeyError as exc:
            raise metadata.PackageNotFoundError(name) from exc

    monkeypatch.setattr(metadata, "version", version)

    result = runner.invoke(app, ["doctor", "--profile", "libero"])

    assert result.exit_code == 1
    assert "profile: libero" in result.output
    assert "transformers==4.40.1; found 4.57.3" in result.output


def test_embodied_doctor_rejects_lerobot_requirement_in_libero_profile() -> None:
    result = runner.invoke(
        app,
        ["doctor", "--profile", "libero", "--require-lerobot"],
    )

    assert result.exit_code == 2
    assert "isolated profile" in result.output


def test_embodied_validate_prints_executed_geometry() -> None:
    result = runner.invoke(app, ["validate", str(_positive_control())])

    assert result.exit_code == 0, result.output
    assert "Valid ART-Embodied experiment" in result.output
    assert "1024 trajectories" in result.output
    assert "65536 fixed-horizon rows = 65536 optimizer rows" in result.output
    assert "every 128 microbatches, capped at 64 MiB" in result.output
    assert "4 devices, 4 actors (4 active max), 4 model replicas" in result.output
    assert "32 environment slots" in result.output
    assert "training resources: 4 devices, 4 model replicas" in result.output
    assert "policy worker Python: current interpreter" in result.output


def test_embodied_validate_can_emit_machine_readable_summary() -> None:
    result = runner.invoke(
        app,
        ["validate", "--json", str(_positive_control())],
    )

    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["schedule"] == "rlinf_actor_global_batch"
    assert summary["trajectories_per_update"] == 1024
    assert summary["total_trajectories"] == 102400
    assert summary["action_token_progress_enabled"] is True
    assert summary["action_token_progress_every_microbatches"] == 128
    assert summary["max_log_file_mb"] == 64
    assert summary["config_consumption"]["unowned_fields"] == []
    assert summary["config_consumption"]["field_count"] > 100


def test_storage_preflight_accepts_output_under_required_root(
    tmp_path: Path,
) -> None:
    source = yaml.safe_load(_positive_control().read_text(encoding="utf-8"))
    storage_root = tmp_path / "data"
    output = storage_root / "experiments" / "run"
    storage_root.mkdir()
    source["storage"]["output_dir"] = str(output)
    config = tmp_path / "experiment.yaml"
    config.write_text(yaml.safe_dump(source), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "storage-preflight",
            str(config),
            "--required-root",
            str(storage_root),
            "--min-free-gib",
            "0",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["status"] == "passed"
    assert report["resolved_output_dir"] == str(output)
    assert report["required_root"] == str(storage_root)


def test_storage_preflight_rejects_output_outside_required_root(
    tmp_path: Path,
) -> None:
    source = yaml.safe_load(_positive_control().read_text(encoding="utf-8"))
    storage_root = tmp_path / "data"
    storage_root.mkdir()
    source["storage"]["output_dir"] = str(tmp_path / "home" / "run")
    config = tmp_path / "experiment.yaml"
    config.write_text(yaml.safe_dump(source), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "storage-preflight",
            str(config),
            "--required-root",
            str(storage_root),
            "--min-free-gib",
            "0",
        ],
    )

    assert result.exit_code == 1
    assert "outside the required storage root" in result.output


def test_embodied_validate_explains_serial_one_gpu_phase_reuse() -> None:
    result = runner.invoke(app, ["validate", str(_pi05_single_gpu())])

    assert result.exit_code == 0, result.output
    assert "training resources: 1 devices, 1 model replicas" in result.output
    assert (
        "serial device reuse: initial_evaluation -> train_rollout -> training -> "
        "periodic_evaluation on cuda:0"
    ) in result.output
    assert "paired against measured Step 0 policy" in result.output


def test_embodied_compare_evaluations_reports_paired_lift(tmp_path: Path) -> None:
    rows = [
        {
            "episode": index,
            "scenario_id": f"scenario-{index}",
            "environment_seed": index + 10,
            "policy_seed": index + 20,
            "success": success,
        }
        for index, success in enumerate((0.0, 1.0))
    ]
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(
        json.dumps({"schema_version": 1, "episodes": rows}),
        encoding="utf-8",
    )
    candidate_rows = [dict(row) for row in rows]
    candidate_rows[0]["success"] = 1.0
    candidate.write_text(
        json.dumps({"schema_version": 1, "episodes": candidate_rows}),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "compare-evaluations",
            "--json",
            str(baseline),
            str(candidate),
        ],
    )

    assert result.exit_code == 0, result.output
    metrics = json.loads(result.output)
    assert metrics["success_rate_lift"] == 0.5
    assert metrics["success_rate_lift_ci95_low"] <= 0.5
    assert metrics["success_rate_lift_ci95_high"] >= 0.5
    assert metrics["task_macro_success_rate_lift"] == 0.5
    assert metrics["improved_pairs"] == 1.0


def test_embodied_validate_evaluation_pair_reports_fixed_contract() -> None:
    result = runner.invoke(
        app,
        [
            "validate-evaluation-pair",
            "--json",
            str(_sft_baseline()),
            str(_public_grpo_evaluation()),
        ],
    )

    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["episodes"] == 100
    assert summary["baseline_step"] == 0
    assert summary["baseline_policy_path"].startswith("Haozhan72/")
    assert summary["candidate_policy_path"].startswith("RLinf/")
    assert summary["baseline_report_path"].endswith(
        "openvla-oft-libero-object-sft-baseline-eval/"
        "evaluation/update_000000_episode_outcomes.json"
    )


def test_embodied_prepare_evaluation_candidate_preserves_native_contract(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    output_config = tmp_path / "candidate.yaml"
    output_dir = tmp_path / "candidate-run"

    result = runner.invoke(
        app,
        [
            "prepare-evaluation-candidate",
            str(_sft_baseline()),
            str(output_config),
            "--run",
            "art-lora-held-out",
            "--output-dir",
            str(output_dir),
            "--adapter-path",
            str(adapter),
            "--baseline-wait-timeout-seconds",
            "3600",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = yaml.safe_load(output_config.read_text(encoding="utf-8"))
    assert payload["experiment"]["run"] == "art-lora-held-out"
    assert payload["policy"]["load_kwargs"]["peft_adapter_path"] == str(adapter)
    assert payload["policy"]["load_kwargs"]["attn_implementation"] is None
    assert payload["storage"]["output_dir"] == str(output_dir)
    assert payload["evaluation"]["baseline_outcomes_path"].endswith(
        "openvla-oft-libero-object-sft-baseline-eval/"
        "evaluation/update_000000_episode_outcomes.json"
    )
    assert payload["evaluation"]["baseline_wait_timeout_seconds"] == 3600
