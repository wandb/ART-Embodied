from __future__ import annotations

from importlib import metadata
import subprocess
import sys

import pytest

from art_embodied.compatibility import (
    TESTED_ART_VERSION,
    VALIDATED_GR00T_N1D5_PACKAGES,
    VALIDATED_GR00T_N1D7_PACKAGES,
    VALIDATED_PI0_FAST_PACKAGES,
    VALIDATED_PI_PACKAGES,
    VALIDATED_SMOLVLA_PACKAGES,
    require_compatible_runtime,
    require_compatible_worker_runtime,
    runtime_compatibility_report,
    runtime_profile_for_policy,
)


@pytest.mark.parametrize(
    "python_version, expected",
    [("3.11.14", "0.5.18"), ("3.12.13", "0.5.20"), ("3.13.0", "0.5.20")],
)
@pytest.mark.parametrize("installed", ["0.5.18", "0.5.20"])
def test_runtime_art_pin_follows_python(
    monkeypatch, python_version, expected, installed
):
    monkeypatch.setattr(
        "art_embodied.compatibility.platform.python_version", lambda: python_version
    )

    def version(name):
        if name == "openpipe-art":
            return installed
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", version)
    report = runtime_compatibility_report()
    assert report.python_version == python_version
    supported = installed in {"0.5.18", expected}
    assert report.compatible is supported
    if not supported:
        assert f"expected {expected}, found {installed}" in report.issues[0]


def test_runtime_report_accepts_tested_art_and_optional_lerobot(monkeypatch) -> None:
    versions = {"openpipe-art": str(TESTED_ART_VERSION)}

    def version(name: str) -> str:
        try:
            return versions[name]
        except KeyError as exc:
            raise metadata.PackageNotFoundError(name) from exc

    monkeypatch.setattr(metadata, "version", version)

    report = runtime_compatibility_report()

    assert report.compatible is True
    assert report.art_version == str(TESTED_ART_VERSION)
    assert report.lerobot_version is None
    assert report.issues == ()


@pytest.mark.parametrize("installed", [None, "0.35.1", "0.35.2", "0.38.0"])
@pytest.mark.parametrize("profile", ["lerobot", "gr00t_n1d7"])
def test_diffusers_security_version_is_checked(monkeypatch, profile, installed):
    versions = {"openpipe-art": str(TESTED_ART_VERSION)}
    if profile == "gr00t_n1d7":
        versions.update(VALIDATED_GR00T_N1D7_PACKAGES)
    else:
        versions["lerobot"] = "0.4.4"
    versions.pop("diffusers", None)
    if installed is not None:
        versions["diffusers"] = installed

    def version(name):
        if name not in versions:
            raise metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(metadata, "version", version)
    report = runtime_compatibility_report(profile=profile)
    assert report.compatible is (installed == "0.38.0")
    assert report.package_versions["diffusers"] == installed
    if not report.compatible:
        assert any("diffusers==0.38.0" in issue for issue in report.issues)


def test_control_plane_ignores_unrequested_lerobot_version(monkeypatch) -> None:
    versions = {"openpipe-art": str(TESTED_ART_VERSION), "lerobot": "0.6.0"}
    monkeypatch.setattr(metadata, "version", versions.__getitem__)

    report = runtime_compatibility_report()

    assert report.compatible is True
    assert report.lerobot_version == "0.6.0"


def test_runtime_report_rejects_untested_art_and_lerobot(monkeypatch) -> None:
    versions = {"openpipe-art": "0.6.0", "lerobot": "0.5.0"}
    monkeypatch.setattr(metadata, "version", versions.__getitem__)

    report = runtime_compatibility_report(require_lerobot=True)

    assert report.compatible is False
    assert len(report.issues) == 2
    assert f"expected {TESTED_ART_VERSION}" in report.issues[0]
    assert "expected >=0.4.4,<0.5" in report.issues[1]


def test_runtime_report_accepts_pi_lerobot_060_profile(monkeypatch) -> None:
    versions = {
        "openpipe-art": str(TESTED_ART_VERSION),
        "lerobot": "0.6.0",
        **VALIDATED_PI_PACKAGES,
    }
    monkeypatch.setattr(metadata, "version", versions.__getitem__)

    report = runtime_compatibility_report(profile="pi")

    assert report.compatible is True
    assert report.profile == "pi"
    assert report.lerobot_version == "0.6.0"


def test_runtime_report_rejects_pi_lerobot_version_drift(monkeypatch) -> None:
    versions = {
        "openpipe-art": str(TESTED_ART_VERSION),
        "lerobot": "0.6.1",
        **VALIDATED_PI_PACKAGES,
    }
    monkeypatch.setattr(metadata, "version", versions.__getitem__)

    report = runtime_compatibility_report(profile="pi")

    assert report.compatible is False
    assert "expected 0.6.0" in report.issues[0]


def test_runtime_report_rejects_missing_pi_peft(monkeypatch) -> None:
    versions = {
        "openpipe-art": str(TESTED_ART_VERSION),
        "lerobot": "0.6.0",
        **VALIDATED_PI_PACKAGES,
    }
    versions.pop("peft")

    def version(name: str) -> str:
        try:
            return versions[name]
        except KeyError as exc:
            raise metadata.PackageNotFoundError(name) from exc

    monkeypatch.setattr(metadata, "version", version)

    report = runtime_compatibility_report(profile="pi")

    assert report.compatible is False
    assert "PI worker requires peft==0.20.0" in report.issues[0]


def test_runtime_report_accepts_pi0_fast_lerobot_060_profile(monkeypatch) -> None:
    versions = {
        "openpipe-art": str(TESTED_ART_VERSION),
        "lerobot": "0.6.0",
        **VALIDATED_PI0_FAST_PACKAGES,
    }
    monkeypatch.setattr(metadata, "version", versions.__getitem__)

    report = runtime_compatibility_report(profile="pi0_fast")

    assert report.compatible is True
    assert report.profile == "pi0_fast"
    assert runtime_profile_for_policy("PI0_FAST") == "pi0_fast"


@pytest.mark.parametrize("policy_type", ["pi0", "pi05", "PI05"])
def test_runtime_profile_for_pi_policy(policy_type: str) -> None:
    assert runtime_profile_for_policy(policy_type) == "pi"


def test_runtime_profile_for_action_token_policy() -> None:
    assert runtime_profile_for_policy("openvla_oft") == "control"


def test_runtime_report_accepts_smolvla_lerobot_060_profile(monkeypatch) -> None:
    versions = {
        "openpipe-art": str(TESTED_ART_VERSION),
        "lerobot": "0.6.0",
        **VALIDATED_SMOLVLA_PACKAGES,
    }
    monkeypatch.setattr(metadata, "version", versions.__getitem__)

    report = runtime_compatibility_report(profile="smolvla")

    assert report.compatible is True
    assert report.profile == "smolvla"
    assert runtime_profile_for_policy("SmolVLA") == "smolvla"


def test_runtime_report_accepts_gr00t_n1d5_profile(monkeypatch) -> None:
    versions = {
        "openpipe-art": str(TESTED_ART_VERSION),
        **VALIDATED_GR00T_N1D5_PACKAGES,
    }

    def version(name: str) -> str:
        try:
            return versions[name]
        except KeyError as exc:
            raise metadata.PackageNotFoundError(name) from exc

    monkeypatch.setattr(metadata, "version", version)

    report = runtime_compatibility_report(profile="gr00t_n1d5")

    assert report.compatible is True
    assert report.profile == "gr00t_n1d5"
    assert runtime_profile_for_policy("GR00T_N1D5") == "gr00t_n1d5"


def test_runtime_report_rejects_gr00t_n1d5_dependency_drift(monkeypatch) -> None:
    versions = {
        "openpipe-art": str(TESTED_ART_VERSION),
        **VALIDATED_GR00T_N1D5_PACKAGES,
        "transformers": "4.52.0",
    }

    def version(name: str) -> str:
        try:
            return versions[name]
        except KeyError as exc:
            raise metadata.PackageNotFoundError(name) from exc

    monkeypatch.setattr(metadata, "version", version)

    report = runtime_compatibility_report(profile="gr00t_n1d5")

    assert report.compatible is False
    assert "GR00T N1.5 worker requires transformers==4.51.3" in report.issues[0]


def test_require_runtime_explains_missing_packages(monkeypatch) -> None:
    def missing(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing)

    with pytest.raises(RuntimeError, match="OpenPipe ART is not installed"):
        require_compatible_runtime(require_lerobot=True)


def test_worker_runtime_does_not_require_art(monkeypatch) -> None:
    def missing(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing)

    report = runtime_compatibility_report(require_art=False)

    assert report.compatible is True
    require_compatible_worker_runtime()


def test_package_import_does_not_initialize_art_control_plane() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import art_embodied; "
                "assert 'art_embodied.art_compat' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
