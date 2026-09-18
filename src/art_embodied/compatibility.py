"""Machine-readable compatibility checks for the ART-Embodied control plane."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
import platform

from packaging.version import Version


def art_version_for_python(python_version: str) -> Version:
    """Select the default ART pin; native GR00T workers retain 0.5.18."""

    return Version("0.5.18" if Version(python_version) < Version("3.12") else "0.5.20")


TESTED_ART_VERSION = art_version_for_python(platform.python_version())
MIN_LEROBOT_VERSION = Version("0.4.4")
MAX_LEROBOT_VERSION = Version("0.5")
PI_LEROBOT_VERSION = Version("0.6.0")
PI0_FAST_LEROBOT_VERSION = Version("0.6.0")
SMOLVLA_LEROBOT_VERSION = Version("0.6.0")
RUNTIME_PROFILES = frozenset(
    {
        "control",
        "gr00t_n1d5",
        "gr00t_n1d7",
        "lerobot",
        "libero",
        "pi",
        "pi0_fast",
        "smolvla",
    }
)
VALIDATED_OPENVLA_OFT_V01_PACKAGES = {
    "accelerate": "1.14.0",
    "hf-libero": "0.1.4",
    "h5py": "3.14.0",
    "mujoco": "3.8.1",
    "numba": "0.61.2",
    "numpy": "1.26.4",
    "peft": "0.11.1",
    "safetensors": "0.7.0",
    "sentencepiece": "0.2.1",
    "timm": "0.9.10",
    "tokenizers": "0.19.1",
    "torch": "2.6.0",
    "torchvision": "0.21.0",
    "transformers": "4.40.1",
}
VALIDATED_PI_PACKAGES = {
    "accelerate": "1.14.0",
    "hf-libero": "0.1.4",
    "mujoco": "3.8.1",
    "numba": "0.61.2",
    "numpy": "2.2.6",
    "peft": "0.20.0",
    "robosuite": "1.4.0",
    "torch": "2.10.0",
    "torchvision": "0.25.0",
    "transformers": "5.5.4",
}
VALIDATED_SMOLVLA_PACKAGES = {
    package: VALIDATED_PI_PACKAGES[package]
    for package in (
        "accelerate",
        "numpy",
        "peft",
        "torch",
        "torchvision",
        "transformers",
    )
}
VALIDATED_PI0_FAST_PACKAGES = dict(VALIDATED_PI_PACKAGES)
VALIDATED_GR00T_N1D5_PACKAGES = {
    "accelerate": "1.2.1",
    "diffusers": "0.30.2",
    "flash-attn": "2.7.1.post4",
    "gr00t": "1.1.0",
    "hf-libero": "0.1.4",
    "mujoco": "3.8.1",
    "numpy": "1.26.4",
    "peft": "0.17.0",
    "pydantic": "2.10.6",
    "protobuf": "4.25.1",
    "robosuite": "1.4.0",
    "torch": "2.5.1",
    "torchvision": "0.20.1",
    "transformers": "4.51.3",
    "typing-extensions": "4.12.2",
    "wandb": "0.24.2",
    "weave": "0.52.37",
}
VALIDATED_GR00T_N1D7_PACKAGES = {
    "diffusers": "0.38.0",
    "flash-attn": "2.8.3",
    "gr00t": "0.1.0",
    "hf-libero": "0.1.4",
    "numpy": "1.26.4",
    "peft": "0.17.1",
    "safetensors": "0.8.0",
    "torch": "2.9.0",
    "torchvision": "0.24.0",
    "transformers": "4.57.3",
}


def runtime_profile_for_policy(policy_type: str) -> str:
    """Return the compatibility profile owned by a native policy family."""

    normalized = policy_type.strip().lower()
    if normalized in {"pi0", "pi05"}:
        return "pi"
    if normalized == "pi0_fast":
        return "pi0_fast"
    if normalized == "smolvla":
        return "smolvla"
    if normalized == "gr00t_n1d5":
        return "gr00t_n1d5"
    if normalized == "gr00t_n1d7":
        return "gr00t_n1d7"
    return "control"


@dataclass(frozen=True, slots=True)
class RuntimeCompatibilityReport:
    """Installed versions and actionable compatibility failures."""

    python_version: str
    profile: str
    art_version: str | None
    lerobot_version: str | None
    package_versions: dict[str, str | None]
    compatible: bool
    issues: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "python_version": self.python_version,
            "profile": self.profile,
            "art_version": self.art_version,
            "lerobot_version": self.lerobot_version,
            "package_versions": dict(self.package_versions),
            "compatible": self.compatible,
            "issues": list(self.issues),
        }


def runtime_compatibility_report(
    *,
    require_art: bool = True,
    require_lerobot: bool = False,
    profile: str = "control",
) -> RuntimeCompatibilityReport:
    """Inspect the control-plane runtime without importing robot ML stacks."""

    if profile not in RUNTIME_PROFILES:
        choices = ", ".join(sorted(RUNTIME_PROFILES))
        raise ValueError(
            f"Unknown runtime profile {profile!r}; expected one of {choices}"
        )
    if profile in {"lerobot", "pi", "pi0_fast", "smolvla"}:
        require_lerobot = True

    issues: list[str] = []
    expected_art_version = art_version_for_python(platform.python_version())
    art_version = _distribution_version("openpipe-art")
    lerobot_version = _distribution_version("lerobot")
    package_versions: dict[str, str | None] = {}

    if require_art:
        if art_version is None:
            issues.append(
                "OpenPipe ART is not installed; install "
                f"openpipe-art=={expected_art_version} before "
                "using the ART-Embodied control plane"
            )
        elif Version(art_version) not in {Version("0.5.18"), expected_art_version}:
            issues.append(
                "Unsupported OpenPipe ART version: "
                f"expected {expected_art_version}"
                + (" or 0.5.18" if expected_art_version != Version("0.5.18") else "")
                + f", found {art_version}"
            )

    if require_lerobot and lerobot_version is None:
        issues.append(
            "LeRobot is not installed; install art-embodied[lerobot] or the "
            "simulator-specific extra"
        )
    elif require_lerobot and lerobot_version is not None:
        installed = Version(lerobot_version)
        if profile == "pi" and installed != PI_LEROBOT_VERSION:
            issues.append(
                "Unsupported LeRobot version for PI0/PI0.5: "
                f"expected {PI_LEROBOT_VERSION}, found {lerobot_version}"
            )
        elif profile == "pi0_fast" and installed != PI0_FAST_LEROBOT_VERSION:
            issues.append(
                "Unsupported LeRobot version for pi0-FAST: "
                f"expected {PI0_FAST_LEROBOT_VERSION}, found {lerobot_version}"
            )
        elif profile == "smolvla" and installed != SMOLVLA_LEROBOT_VERSION:
            issues.append(
                "Unsupported LeRobot version for SmolVLA: "
                f"expected {SMOLVLA_LEROBOT_VERSION}, found {lerobot_version}"
            )
        elif profile not in {"pi", "pi0_fast", "smolvla"} and not (
            MIN_LEROBOT_VERSION <= installed < MAX_LEROBOT_VERSION
        ):
            issues.append(
                "Unsupported LeRobot version: expected >=0.4.4,<0.5, "
                f"found {lerobot_version}"
            )

    if profile == "lerobot":
        installed = _distribution_version("diffusers")
        package_versions["diffusers"] = installed
        if installed != "0.38.0":
            issues.append(
                f"LeRobot profile requires diffusers==0.38.0; found {installed}. "
                "Use uv sync --extra lerobot to apply the security override"
            )
    elif profile == "libero":
        for package, expected in VALIDATED_OPENVLA_OFT_V01_PACKAGES.items():
            installed = _distribution_version(package)
            package_versions[package] = installed
            normalized = installed.split("+", 1)[0] if installed is not None else None
            if normalized != expected:
                found = installed if installed is not None else "not installed"
                issues.append(
                    f"OpenVLA-OFT worker requires {package}=={expected}; found {found}. "
                    "Install the isolated art-embodied[libero] profile"
                )
    elif profile == "pi":
        for package, expected in VALIDATED_PI_PACKAGES.items():
            installed = _distribution_version(package)
            package_versions[package] = installed
            normalized = installed.split("+", 1)[0] if installed is not None else None
            if normalized != expected:
                found = installed if installed is not None else "not installed"
                issues.append(
                    f"PI worker requires {package}=={expected}; found {found}. "
                    "Install the isolated art-embodied[pi-libero] profile"
                )
    elif profile == "smolvla":
        for package, expected in VALIDATED_SMOLVLA_PACKAGES.items():
            installed = _distribution_version(package)
            package_versions[package] = installed
            normalized = installed.split("+", 1)[0] if installed is not None else None
            if normalized != expected:
                found = installed if installed is not None else "not installed"
                issues.append(
                    f"SmolVLA worker requires {package}=={expected}; found {found}. "
                    "Install the isolated art-embodied[smolvla-libero] profile"
                )
    elif profile == "pi0_fast":
        for package, expected in VALIDATED_PI0_FAST_PACKAGES.items():
            installed = _distribution_version(package)
            package_versions[package] = installed
            normalized = installed.split("+", 1)[0] if installed is not None else None
            if normalized != expected:
                found = installed if installed is not None else "not installed"
                issues.append(
                    f"pi0-FAST worker requires {package}=={expected}; found {found}. "
                    "Install the isolated art-embodied[pi0-fast-libero] profile"
                )
    elif profile == "gr00t_n1d5":
        for package, expected in VALIDATED_GR00T_N1D5_PACKAGES.items():
            installed = _distribution_version(package)
            package_versions[package] = installed
            normalized = installed.split("+", 1)[0] if installed is not None else None
            if normalized != expected:
                found = installed if installed is not None else "not installed"
                issues.append(
                    f"GR00T N1.5 worker requires {package}=={expected}; "
                    f"found {found}. Run scripts/install-gr00t-n1d5-runtime.sh"
                )
    elif profile == "gr00t_n1d7":
        for package, expected in VALIDATED_GR00T_N1D7_PACKAGES.items():
            installed = _distribution_version(package)
            package_versions[package] = installed
            normalized = installed.split("+", 1)[0] if installed is not None else None
            if normalized != expected:
                found = installed if installed is not None else "not installed"
                issues.append(
                    f"GR00T N1.7 worker requires {package}=={expected}; "
                    f"found {found}. Run scripts/install-gr00t-n1d7-runtime.sh"
                )

    return RuntimeCompatibilityReport(
        python_version=platform.python_version(),
        profile=profile,
        art_version=art_version,
        lerobot_version=lerobot_version,
        package_versions=package_versions,
        compatible=not issues,
        issues=tuple(issues),
    )


def require_compatible_runtime(
    *,
    require_lerobot: bool = False,
    profile: str = "control",
) -> None:
    """Fail before policy loading when installed package versions are unsupported."""

    report = runtime_compatibility_report(
        require_lerobot=require_lerobot,
        profile=profile,
    )
    if report.compatible:
        return
    details = "\n- ".join(report.issues)
    raise RuntimeError(f"ART-Embodied runtime is incompatible:\n- {details}")


def require_compatible_worker_runtime(
    *,
    require_lerobot: bool = False,
    profile: str = "control",
) -> None:
    """Validate a policy worker without requiring ART's control-plane package."""

    report = runtime_compatibility_report(
        require_art=False,
        require_lerobot=require_lerobot,
        profile=profile,
    )
    if report.compatible:
        return
    details = "\n- ".join(report.issues)
    raise RuntimeError(f"ART-Embodied worker runtime is incompatible:\n- {details}")


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None
