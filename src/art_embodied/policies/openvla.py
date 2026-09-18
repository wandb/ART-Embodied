"""OpenVLA policy adapter.

This adapter follows the official OpenVLA Hugging Face interface:
`AutoProcessor.from_pretrained`, `AutoModelForVision2Seq.from_pretrained`, and
`model.predict_action(...)`. It is intentionally lazy so importing ART-Embodied
does not download or initialize an 8B VLA model.
"""

from __future__ import annotations

from enum import Enum
from importlib import import_module, metadata
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from art_embodied.backends.action_token import (
    TRANSIENT_RLINF_ENV_OBS_METADATA_KEY,
    TRANSIENT_RLINF_FORWARD_INPUTS_METADATA_KEY,
    ActionTokenExample,
)
from art_embodied.compatibility import VALIDATED_OPENVLA_OFT_V01_PACKAGES
from art_embodied.trajectories import Action, Observation
from art_embodied.utils import make_json_safe


def _install_openvla_oft_prismatic_compat_if_needed(
    *,
    hint: str = "",
    robot_platform: str | None = None,
) -> bool:
    """Install a tiny OpenVLA-OFT ``prismatic`` compatibility shim if absent.

    Some OpenVLA-OFT Hugging Face checkpoints import a handful of constants and
    mask helpers from the training repository's ``prismatic`` package even for
    inference. Pulling that package's import-time TensorFlow and observability
    stack into every policy worker is heavy and brittle, so the native runtime
    installs the exact minimal surface before loading remote code. A module that
    an application explicitly loaded first is preserved.
    """

    import sys
    import types

    if "prismatic.training.train_utils" in sys.modules:
        return False

    class NormalizationType(str, Enum):
        NORMAL = "normal"
        BOUNDS = "bounds"
        BOUNDS_Q99 = "bounds_q99"

    platform = _detect_openvla_oft_robot_platform(
        hint,
        robot_platform=robot_platform,
    )
    constants_by_platform = {
        "LIBERO": {
            "NUM_ACTIONS_CHUNK": 8,
            "ACTION_DIM": 7,
            "PROPRIO_DIM": 8,
            "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
        },
        "ALOHA": {
            "NUM_ACTIONS_CHUNK": 25,
            "ACTION_DIM": 14,
            "PROPRIO_DIM": 14,
            "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
        },
        "BRIDGE": {
            "NUM_ACTIONS_CHUNK": 5,
            "ACTION_DIM": 7,
            "PROPRIO_DIM": 7,
            "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
        },
    }
    selected = constants_by_platform[platform]

    prismatic_mod = sys.modules.setdefault("prismatic", types.ModuleType("prismatic"))
    prismatic_mod.__path__ = getattr(prismatic_mod, "__path__", [])
    training_mod = sys.modules.setdefault(
        "prismatic.training", types.ModuleType("prismatic.training")
    )
    training_mod.__path__ = getattr(training_mod, "__path__", [])
    vla_mod = sys.modules.setdefault("prismatic.vla", types.ModuleType("prismatic.vla"))
    vla_mod.__path__ = getattr(vla_mod, "__path__", [])

    constants_mod = types.ModuleType("prismatic.vla.constants")
    constants_mod.IGNORE_INDEX = -100
    constants_mod.ACTION_TOKEN_BEGIN_IDX = 31743
    constants_mod.STOP_INDEX = 2
    constants_mod.NormalizationType = NormalizationType
    constants_mod.ROBOT_PLATFORM = platform
    for key, value in selected.items():
        setattr(constants_mod, key, value)

    train_utils_mod = types.ModuleType("prismatic.training.train_utils")

    def get_current_action_mask(token_ids: Any):
        import torch

        newline_positions = (token_ids != constants_mod.IGNORE_INDEX).to(torch.int64)
        cumsum = torch.cumsum(newline_positions, dim=1)
        mask = (1 <= cumsum) & (cumsum <= constants_mod.ACTION_DIM)
        action_tokens_only_mask = token_ids > constants_mod.ACTION_TOKEN_BEGIN_IDX
        return action_tokens_only_mask & mask

    def get_next_actions_mask(token_ids: Any):
        import torch

        newline_positions = (token_ids != constants_mod.IGNORE_INDEX).to(torch.int64)
        cumsum = torch.cumsum(newline_positions, dim=1)
        mask = cumsum > constants_mod.ACTION_DIM
        action_tokens_only_mask = token_ids > constants_mod.ACTION_TOKEN_BEGIN_IDX
        return action_tokens_only_mask & mask

    train_utils_mod.get_current_action_mask = get_current_action_mask
    train_utils_mod.get_next_actions_mask = get_next_actions_mask

    sys.modules["prismatic.vla.constants"] = constants_mod
    sys.modules["prismatic.training.train_utils"] = train_utils_mod
    setattr(vla_mod, "constants", constants_mod)
    setattr(training_mod, "train_utils", train_utils_mod)
    setattr(prismatic_mod, "vla", vla_mod)
    setattr(prismatic_mod, "training", training_mod)
    return True


def _detect_openvla_oft_robot_platform(
    hint: str = "",
    *,
    robot_platform: str | None = None,
) -> str:
    import sys

    if robot_platform is not None:
        normalized = robot_platform.strip().upper()
        if normalized not in {"LIBERO", "ALOHA", "BRIDGE"}:
            raise ValueError(
                "OpenVLA-OFT robot_platform must be libero, aloha, or bridge; "
                f"got {robot_platform!r}"
            )
        return normalized
    text = " ".join([hint, *sys.argv]).lower()
    if "aloha" in text:
        return "ALOHA"
    if "bridge" in text:
        return "BRIDGE"
    return "LIBERO"


def check_openvla_runtime_requirements(*, require_cuda: bool = True) -> list[str]:
    """Return missing or incompatible runtime requirements for OpenVLA inference.

    OpenVLA uses Hugging Face remote code. Failing early is much cheaper than
    discovering a missing or incompatible dependency after model download or
    partial load. Warnings that do not block loading are reported separately by
    :func:`openvla_runtime_warnings`.
    """

    issues: list[str] = []
    for module_name in ("PIL", "torch", "transformers"):
        try:
            __import__(module_name)
        except Exception:
            issues.append(module_name)

    try:
        __import__("timm")
        timm_version = metadata.version("timm")
    except Exception:
        issues.append("timm>=0.9.10,<1.0.0")
    else:
        if not _is_supported_openvla_timm(timm_version):
            issues.append(f"timm>=0.9.10,<1.0.0 (found {timm_version})")

    if require_cuda and not any(issue.startswith("torch") for issue in issues):
        import torch

        if not torch.cuda.is_available():
            issues.append("cuda")
    return issues


def openvla_runtime_warnings() -> list[str]:
    """Return non-fatal OpenVLA dependency warnings for experiment logging."""

    warnings: list[str] = []
    for package, expected in (
        ("transformers", "4.40.1"),
        ("tokenizers", "0.19.1"),
    ):
        try:
            found = metadata.version(package)
        except Exception:
            continue
        if found != expected:
            warnings.append(f"{package}: expected {expected}, found {found}")
    return warnings


def openvla_oft_v01_runtime_issues(*, require_peft: bool) -> list[str]:
    """Return drift from the runtime that reproduces OpenVLA-OFT v0.1.

    Loading the checkpoint successfully is not sufficient: newer Transformers
    versions produce different action logits for identical model inputs.  Keep
    the validated positive-control runtime explicit until another combination
    has passed the same fixed held-out evaluation contract.
    """

    packages = (
        "torch",
        "transformers",
        "tokenizers",
        "timm",
        *(("peft",) if require_peft else ()),
    )
    issues: list[str] = []
    for package in packages:
        expected = VALIDATED_OPENVLA_OFT_V01_PACKAGES[package]
        try:
            found = metadata.version(package)
        except metadata.PackageNotFoundError:
            issues.append(f"{package}=={expected} (not installed)")
            continue
        normalized = found.split("+", 1)[0] if package == "torch" else found
        if normalized != expected:
            issues.append(f"{package}=={expected} (found {found})")
    return issues


def raise_for_openvla_oft_v01_runtime(*, require_peft: bool) -> None:
    """Fail closed when the validated OpenVLA-OFT runtime has drifted."""

    issues = openvla_oft_v01_runtime_issues(require_peft=require_peft)
    if not issues:
        return
    raise RuntimeError(
        "OpenVLA-OFT runtime is not behavior-compatible with the validated "
        "v0.1 contract: "
        + ", ".join(issues)
        + ". A checkpoint can load under newer libraries while producing "
        "different action logits. Run this model family in the pinned "
        "ART-owned OpenVLA-OFT runtime, or set "
        "policy.load_kwargs.runtime_contract='unchecked' only for an "
        "explicit diagnostic that does not make parity or performance claims."
    )


def raise_for_openvla_runtime_requirements(*, require_cuda: bool = True) -> None:
    """Fail with an actionable message when generic OpenVLA imports cannot run."""

    issues = check_openvla_runtime_requirements(require_cuda=require_cuda)
    if not issues:
        return
    install_hint = (
        "OpenVLA runtime requirements are missing or incompatible: "
        + ", ".join(issues)
        + ". For the upstream OpenVLA remote code, install a compatible TIMM "
        "release, for example `python -m pip install --force-reinstall --no-deps "
        "timm==0.9.16`."
    )
    raise RuntimeError(install_hint)


def _is_supported_openvla_timm(version: str) -> bool:
    parts = []
    for part in version.split(".")[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        parts.append(int(digits or 0))
    while len(parts) < 3:
        parts.append(0)
    return (0, 9, 10) <= tuple(parts) < (1, 0, 0)


def _resolve_rlinf_local_model_path(model_id: str) -> Path:
    """Resolve a Hub id to a local snapshot for RLinf's stats lookup."""

    local = Path(model_id).expanduser()
    if local.is_dir():
        return local.resolve()
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError(
            f"RLinf model_loader requires a local path or huggingface_hub for {model_id!r}."
        ) from exc
    return Path(snapshot_download(repo_id=model_id)).resolve()


def _ensure_rlinf_openvla_oft_prismatic_shim(*, model_id: str) -> None:
    """Expose OpenVLA-OFT HF remote code under RLinf's `prismatic.*` imports."""

    import importlib.util
    import os
    import shutil
    import sys
    import tempfile

    try:
        existing = importlib.util.find_spec("prismatic.extern.hf.modeling_prismatic")
    except ModuleNotFoundError:
        existing = None
    if existing is not None:
        return

    code_dir = _find_openvla_oft_remote_code_dir(model_id)
    if code_dir is None:
        raise RuntimeError(
            "Could not locate OpenVLA-OFT remote code files needed by RLinf's "
            f"prismatic imports for model_id={model_id!r}."
        )

    root = Path(tempfile.gettempdir()) / "art-embodied-openvla-oft-prismatic-shim"
    hf_dir = root / "prismatic" / "extern" / "hf"
    hf_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "configuration_prismatic.py",
        "modeling_prismatic.py",
        "processing_prismatic.py",
    ):
        shutil.copy2(code_dir / name, hf_dir / name)
    for package in (
        root / "prismatic",
        root / "prismatic" / "extern",
        hf_dir,
        root / "prismatic" / "training",
        root / "prismatic" / "vla",
        root / "prismatic" / "models",
    ):
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
    constants_text = (
        "from enum import Enum\n\n"
        "IGNORE_INDEX = -100\n"
        "ACTION_TOKEN_BEGIN_IDX = 31743\n"
        "STOP_INDEX = 2\n\n"
        "class NormalizationType(str, Enum):\n"
        '    NORMAL = "normal"\n'
        '    BOUNDS = "bounds"\n'
        '    BOUNDS_Q99 = "bounds_q99"\n\n'
        "NUM_ACTIONS_CHUNK = 8\n"
        "ACTION_DIM = 7\n"
        "PROPRIO_DIM = 8\n"
        "ACTION_PROPRIO_NORMALIZATION_TYPE = NormalizationType.BOUNDS_Q99\n"
    )
    (root / "prismatic" / "vla" / "constants.py").write_text(
        constants_text, encoding="utf-8"
    )
    train_utils_text = (
        "import torch\n"
        "from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX\n\n"
        "def get_current_action_mask(token_ids):\n"
        "    newline_positions = token_ids != IGNORE_INDEX\n"
        "    cumsum = torch.cumsum(newline_positions, dim=1)\n"
        "    mask = (1 <= cumsum) & (cumsum <= ACTION_DIM)\n"
        "    return (token_ids > ACTION_TOKEN_BEGIN_IDX) * mask\n\n"
        "def get_next_actions_mask(token_ids):\n"
        "    newline_positions = token_ids != IGNORE_INDEX\n"
        "    cumsum = torch.cumsum(newline_positions, dim=1)\n"
        "    mask = cumsum > ACTION_DIM\n"
        "    return (token_ids > ACTION_TOKEN_BEGIN_IDX) * mask\n"
    )
    (root / "prismatic" / "training" / "train_utils.py").write_text(
        train_utils_text, encoding="utf-8"
    )
    projectors_text = (
        "import torch.nn as nn\n\n"
        "class ProprioProjector(nn.Module):\n"
        "    def __init__(self, llm_dim: int, proprio_dim: int) -> None:\n"
        "        super().__init__()\n"
        "        self.fc1 = nn.Linear(proprio_dim, llm_dim, bias=True)\n"
        "        self.fc2 = nn.Linear(llm_dim, llm_dim, bias=True)\n\n"
        "    def forward(self, proprio):\n"
        "        return self.fc2(nn.functional.gelu(self.fc1(proprio)))\n"
    )
    (root / "prismatic" / "models" / "projectors.py").write_text(
        projectors_text, encoding="utf-8"
    )
    for module_name in list(sys.modules):
        if module_name == "prismatic" or module_name.startswith("prismatic."):
            del sys.modules[module_name]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.environ["PYTHONPATH"] = str(root) + (
        os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""
    )


def _find_openvla_oft_remote_code_dir(model_id: str) -> Path | None:
    local = Path(model_id).expanduser()
    if local.is_dir() and all(
        (local / name).is_file()
        for name in (
            "configuration_prismatic.py",
            "modeling_prismatic.py",
            "processing_prismatic.py",
        )
    ):
        return local.resolve()
    try:
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                repo_id=model_id,
                allow_patterns=[
                    "configuration_prismatic.py",
                    "modeling_prismatic.py",
                    "processing_prismatic.py",
                    "config.json",
                    "preprocessor_config.json",
                    "processor_config.json",
                    "tokenizer_config.json",
                    "tokenizer.model",
                    "tokenizer.json",
                    "special_tokens_map.json",
                    "added_tokens.json",
                ],
            )
        )
        if all(
            (snapshot / name).is_file()
            for name in (
                "configuration_prismatic.py",
                "modeling_prismatic.py",
                "processing_prismatic.py",
            )
        ):
            return snapshot
    except Exception:
        pass
    return None


def _load_rlinf_openvla_oft_model(
    *,
    model_id: str,
    torch_dtype: Any,
    device: str,
    unnorm_key: str,
    max_prompt_length: int,
    num_images_in_input: int,
    attn_implementation: str | None,
    peft_adapter_path: str | None,
) -> Any:
    """Load OpenVLA-OFT through RLinf's prior-art wrapper.

    This is intentionally opt-in. It gives ART-Embodied a parity mode for
    reproducing RLinf/OpenVLA-OFT LIBERO results before we claim GRPO/GSPO gains.
    """

    try:
        from omegaconf import OmegaConf
    except Exception as exc:  # pragma: no cover - optional integration dependency.
        raise RuntimeError(
            "OpenVLAPolicy(model_loader='rlinf') requires RLinf and OmegaConf on PYTHONPATH."
        ) from exc

    get_model, get_vla_model_config_and_processor, loader_api = (
        _resolve_rlinf_openvla_oft_loaders()
    )

    resolved_model_id = str(_resolve_rlinf_local_model_path(model_id))
    _ensure_rlinf_openvla_oft_prismatic_shim(model_id=resolved_model_id)
    cfg = OmegaConf.create(
        {
            "model_path": resolved_model_id,
            "precision": "bf16" if str(torch_dtype).endswith("bfloat16") else "fp32",
            "model_type": "openvla_oft",
            "implement_version": "rlinf",
            "action_dim": 7,
            "num_action_chunks": 8,
            "add_value_head": False,
            "value_type": "action_level",
            "proprio_dim": 8,
            "use_proprio": False,
            "use_film": False,
            "max_prompt_length": int(max_prompt_length),
            "unnorm_key": unnorm_key,
            "num_images_in_input": int(num_images_in_input),
            "center_crop": True,
            "is_lora": False,
            "lora_rank": 32,
            "lora_path": None,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "attn_implementation": attn_implementation,
            "policy_setup": "libero",
            "reward_coef": 5.0,
        }
    )
    runtime_cfg = OmegaConf.create(
        {
            "actor": {
                "model": OmegaConf.to_container(cfg, resolve=True),
                "tokenizer": {
                    "tokenizer_model": resolved_model_id,
                    "use_fast": False,
                    "trust_remote_code": True,
                    "padding_side": "right",
                },
            },
            "runner": {"max_prompt_length": int(max_prompt_length)},
        }
    )

    def load_once() -> Any:
        if loader_api == "v0.1":
            if (
                get_vla_model_config_and_processor is None
            ):  # pragma: no cover - resolver invariant.
                raise RuntimeError("RLinf v0.1 processor loader was not resolved")
            # RLinf v0.1 derives dtype from cfg.precision and initializes the
            # processor separately in HuggingFaceRolloutWorker.init_worker().
            # Preserve the official worker's order: model first, then processor
            # construction and setup. get_model(cfg, torch_dtype=...) is a later API.
            loaded_model = get_model(cfg)
            model_config, input_processor = get_vla_model_config_and_processor(
                runtime_cfg.actor
            )
            loaded_model.setup_config_and_processor(
                model_config,
                runtime_cfg,
                input_processor,
            )
            return loaded_model.to(device)
        return get_model(cfg, torch_dtype=torch_dtype).to(device)

    try:
        # Keep the positive-control path identical to RLinf's native loader by
        # default.  The compatibility patch below changes Transformers'
        # attention capability checks and can change rollout logits on newer
        # stacks, so only use it as a last-resort load fallback.
        model = load_once()
    except Exception:
        _patch_rlinf_openvla_oft_transformers_attn_support()
        model = load_once()
    if peft_adapter_path:
        model = _attach_peft_adapter(model, peft_adapter_path, device=device)
    # RLinf v0.1 and later releases use different names for the primary image
    # in env_obs. Keep the resolved API on the final (possibly PEFT-wrapped)
    # model so every native rollout and rescore path uses the same contract.
    setattr(model, "_art_rlinf_loader_api", loader_api)
    model.eval()
    return model


def _resolve_rlinf_openvla_oft_loaders() -> tuple[Any, Any | None, str]:
    """Resolve the versioned RLinf OpenVLA-OFT model/processor API.

    RLinf v0.1 exports ``get_model`` and the separate rollout processor loader
    from ``rlinf.models``. Later releases moved a self-contained loader under
    ``rlinf.models.embodiment.openvla_oft`` and accept ``torch_dtype`` directly.
    Resolve the API before model construction so a model-load dependency error
    can never be misclassified as a version fallback.
    """

    modern_module_name = "rlinf.models.embodiment.openvla_oft"
    try:
        modern_module = import_module(modern_module_name)
    except ModuleNotFoundError as exc:
        # Only a missing RLinf/module path identifies the v0.1 layout. A nested
        # dependency failure in a present modern module must remain a hard error.
        if exc.name not in {
            "rlinf",
            "rlinf.models",
            "rlinf.models.embodiment",
            modern_module_name,
        }:
            raise RuntimeError(
                f"Failed to import present RLinf OpenVLA-OFT module {modern_module_name!r}"
            ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import RLinf OpenVLA-OFT module {modern_module_name!r}"
        ) from exc
    else:
        get_model = getattr(modern_module, "get_model", None)
        if not callable(get_model):
            raise RuntimeError(
                f"{modern_module_name!r} does not export callable get_model"
            )
        return get_model, None, "modern"

    try:
        v01_module = import_module("rlinf.models")
    except Exception as exc:  # pragma: no cover - optional integration dependency.
        raise RuntimeError(
            "OpenVLAPolicy(model_loader='rlinf') requires a supported RLinf checkout "
            "on PYTHONPATH (v0.1 or later OpenVLA-OFT layout)."
        ) from exc
    get_model = getattr(v01_module, "get_model", None)
    get_processor = getattr(v01_module, "get_vla_model_config_and_processor", None)
    if not callable(get_model) or not callable(get_processor):
        raise RuntimeError(
            "RLinf v0.1 layout must export callable get_model and "
            "get_vla_model_config_and_processor from rlinf.models"
        )
    return get_model, get_processor, "v0.1"


def _patch_rlinf_openvla_oft_transformers_attn_support() -> None:
    """Patch RLinf OpenVLA-OFT classes for newer Transformers init checks.

    Transformers 4.57 asks remote-code ``PreTrainedModel`` subclasses for
    ``_supports_sdpa`` / ``_supports_flash_attn_2`` during construction.  The
    RLinf OpenVLA-OFT wrappers used for parity were authored against older
    Transformers and do not define these attributes, which makes host-side
    diagnostics fail before any ART code runs. This compatibility patch only
    declares capabilities; it does not select an attention override.
    OpenVLA-OFT v0.1 action logits require the checkpoint/Transformers default
    because its action placeholders use a bidirectional mask.
    """

    module_names = (
        "rlinf.models.embodiment.openvla_oft_action_model",
        "rlinf.models.embodiment.openvla_oft.rlinf.openvla_oft_action_model",
        "rlinf.models.embodiment.openvla_oft.official.openvla_oft_action_model",
    )
    for module_name in module_names:
        try:
            module = __import__(
                module_name, fromlist=["OpenVLAOFTForRLActionPrediction"]
            )
            cls = getattr(module, "OpenVLAOFTForRLActionPrediction", None)
        except Exception:
            cls = None
        if cls is None:
            continue
        # Do this unconditionally.  Some OpenVLA-OFT remote-code parents expose
        # these as properties that delegate back through ``nn.Module`` instance
        # lookup and raise during initialization.  A concrete bool on the final
        # class shadows the brittle parent property.
        setattr(cls, "_supports_sdpa", False)
        setattr(cls, "_supports_flash_attn_2", True)


def _predict_rlinf_native_policy_action(
    model: Any,
    *,
    observation: Observation,
    instruction: str,
    do_sample: bool,
    temperature: float,
    num_images_in_input: int | None,
    use_proprio: bool | None,
    env_obs_override: dict[str, Any] | None = None,
) -> tuple[Any, list[int], list[float], dict[str, Any]]:
    """Run RLinf's native OpenVLA-OFT rollout method from an ART observation."""

    if env_obs_override is None:
        env_obs, obs_report = _rlinf_native_env_obs_from_observation(
            observation,
            instruction=instruction,
            num_images_in_input=num_images_in_input,
            use_proprio=use_proprio,
            primary_image_key=_rlinf_primary_image_key(model),
        )
    else:
        env_obs = _copy_rlinf_env_obs_for_model(
            env_obs_override, instructions=[instruction]
        )
        obs_report = _rlinf_env_obs_report(env_obs, source="context_passthrough")
    raw_actions, result = model.predict_action_batch(
        env_obs=env_obs,
        do_sample=bool(do_sample),
        temperature=float(temperature),
        top_k=-1,
        calculate_values=False,
    )
    forward_inputs = (
        result.get("forward_inputs", {}) if isinstance(result, dict) else {}
    )
    action_tokens = forward_inputs.get("action_tokens")
    prev_logprobs = result.get("prev_logprobs") if isinstance(result, dict) else None
    if action_tokens is None:
        raise RuntimeError(
            "RLinf OpenVLA-OFT result did not include forward_inputs.action_tokens"
        )

    token_ids = _flatten_first_batch_values(action_tokens, dtype=int)
    token_logprobs = (
        _flatten_first_batch_values(prev_logprobs, dtype=float)
        if prev_logprobs is not None
        else []
    )
    if token_logprobs and len(token_logprobs) != len(token_ids):
        raise RuntimeError(
            "RLinf OpenVLA-OFT token/logprob length mismatch: "
            f"tokens={len(token_ids)} logprobs={len(token_logprobs)}"
        )
    if not token_logprobs:
        token_logprobs = [0.0 for _ in token_ids]

    action = _first_batch_value(raw_actions)
    metadata = {
        "model_loader": "rlinf",
        "rlinf_native_predict_action_batch": True,
        "rlinf_forward_input_keys": sorted(str(key) for key in forward_inputs),
        "rlinf_observation_report": obs_report,
        TRANSIENT_RLINF_FORWARD_INPUTS_METADATA_KEY: _slice_forward_inputs_row(
            forward_inputs, row_index=0
        ),
    }
    return action, token_ids, token_logprobs, metadata


def _predict_rlinf_native_policy_actions_batch(
    model: Any,
    *,
    observations: list[Observation],
    instructions: list[str],
    do_sample: bool,
    temperature: float,
    num_images_in_input: int | None,
    use_proprio: bool | None,
    env_obs_overrides: list[dict[str, Any] | None] | None = None,
) -> list[tuple[Any, list[int], list[float], dict[str, Any]]]:
    """Run RLinf's native rollout method for a whole ART observation batch.

    The single-observation path above is the parity-critical OpenVLA-OFT/RLinf
    contract.  Batched rollout must call the same ``predict_action_batch`` API,
    not ART's independent action-logit reconstruction, otherwise faster rollout
    silently becomes a different policy.
    """

    if len(observations) != len(instructions):
        raise ValueError(
            "RLinf native batch prediction requires matching observations/instructions"
        )
    if not observations:
        return []

    import torch

    if env_obs_overrides is not None and any(
        item is not None for item in env_obs_overrides
    ):
        if len(env_obs_overrides) != len(observations):
            raise ValueError(
                "RLinf env_obs overrides must match observation batch length"
            )
        if any(item is None for item in env_obs_overrides):
            raise ValueError(
                "RLinf env_obs overrides must be provided for the whole batch or none of it"
            )
        env_obs_rows = [
            _copy_rlinf_env_obs_for_model(item, instructions=[instruction])
            for item, instruction in zip(env_obs_overrides, instructions, strict=True)
            if item is not None
        ]
        env_obs, obs_reports = _merge_rlinf_env_obs_rows(
            env_obs_rows, instructions=instructions
        )
    else:
        per_row: list[tuple[dict[str, Any], dict[str, Any]]] = [
            _rlinf_native_env_obs_from_observation(
                observation,
                instruction=instruction,
                num_images_in_input=num_images_in_input,
                use_proprio=use_proprio,
                primary_image_key=_rlinf_primary_image_key(model),
            )
            for observation, instruction in zip(observations, instructions, strict=True)
        ]
        env_obs_rows = [item[0] for item in per_row]
        obs_reports = [item[1] for item in per_row]
        env_obs, _obs_reports = _merge_rlinf_env_obs_rows(
            env_obs_rows, instructions=instructions
        )

    raw_actions, result = model.predict_action_batch(
        env_obs=env_obs,
        do_sample=bool(do_sample),
        temperature=float(temperature),
        top_k=-1,
        calculate_values=False,
    )
    forward_inputs = (
        result.get("forward_inputs", {}) if isinstance(result, dict) else {}
    )
    action_tokens = forward_inputs.get("action_tokens")
    prev_logprobs = result.get("prev_logprobs") if isinstance(result, dict) else None
    if action_tokens is None:
        raise RuntimeError(
            "RLinf OpenVLA-OFT result did not include forward_inputs.action_tokens"
        )

    predictions: list[tuple[Any, list[int], list[float], dict[str, Any]]] = []
    for row_index in range(len(observations)):
        token_ids = _flatten_batch_row_values(
            action_tokens, row_index=row_index, dtype=int
        )
        token_logprobs = (
            _flatten_batch_row_values(prev_logprobs, row_index=row_index, dtype=float)
            if prev_logprobs is not None
            else []
        )
        if token_logprobs and len(token_logprobs) != len(token_ids):
            raise RuntimeError(
                "RLinf OpenVLA-OFT batched token/logprob length mismatch: "
                f"row={row_index} tokens={len(token_ids)} logprobs={len(token_logprobs)}"
            )
        if not token_logprobs:
            token_logprobs = [0.0 for _ in token_ids]
        metadata = {
            "model_loader": "rlinf",
            "rlinf_native_predict_action_batch": True,
            "rlinf_native_batched_policy_call": True,
            "rlinf_forward_input_keys": sorted(str(key) for key in forward_inputs),
            "rlinf_observation_report": obs_reports[row_index],
            TRANSIENT_RLINF_FORWARD_INPUTS_METADATA_KEY: _slice_forward_inputs_row(
                forward_inputs,
                row_index=row_index,
            ),
        }
        predictions.append(
            (
                _batch_row_value(raw_actions, row_index=row_index),
                token_ids,
                token_logprobs,
                metadata,
            )
        )
    return predictions


def _copy_rlinf_env_obs_for_model(
    env_obs: dict[str, Any],
    *,
    instructions: list[str],
) -> dict[str, Any]:
    """Copy an RLinf env_obs payload before passing it to predict_action_batch.

    RLinf's OpenVLA-OFT implementation mutates the dict by adding a time
    dimension when image tensors are rank-4.  A shallow-but-tensor-safe copy
    keeps ART rollouts from corrupting the environment-owned observation.
    """

    import numpy as np

    copied: dict[str, Any] = {}
    for key, value in env_obs.items():
        if hasattr(value, "clone"):
            copied[key] = value.clone()
        elif isinstance(value, np.ndarray):
            copied[key] = value.copy()
        elif isinstance(value, list):
            copied[key] = list(value)
        elif isinstance(value, tuple):
            copied[key] = list(value)
        else:
            copied[key] = value
    if "task_descriptions" not in copied:
        copied["task_descriptions"] = list(instructions)
    return copied


def _merge_rlinf_env_obs_rows(
    rows: list[dict[str, Any]],
    *,
    instructions: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import numpy as np
    import torch

    if not rows:
        return {"task_descriptions": []}, []

    if all("main_images" in row for row in rows):
        primary_image_key = "main_images"
    elif all("images" in row for row in rows):
        primary_image_key = "images"
    else:
        schemas = [sorted(str(key) for key in row) for row in rows]
        raise ValueError(
            "RLinf env_obs rows must use one consistent primary image key "
            f"('main_images' for later releases or 'images' for v0.1); got {schemas}"
        )

    def concat_values(values: list[Any]) -> Any:
        if all(hasattr(value, "detach") for value in values):
            return torch.cat(list(values), dim=0)
        arrays = [np.asarray(value) for value in values]
        return torch.as_tensor(np.concatenate(arrays, axis=0))

    task_descriptions: list[str] = []
    for row, instruction in zip(rows, instructions, strict=True):
        value = row.get("task_descriptions")
        if isinstance(value, (list, tuple)) and value:
            task_descriptions.append(str(value[0]))
        elif value is not None:
            array = np.asarray(value, dtype=object)
            task_descriptions.append(
                str(array.reshape(-1)[0]) if array.size else instruction
            )
        else:
            task_descriptions.append(instruction)

    env_obs: dict[str, Any] = {
        primary_image_key: concat_values([row[primary_image_key] for row in rows]),
        "task_descriptions": task_descriptions,
        "states": concat_values([row["states"] for row in rows]),
    }
    if all("wrist_images" in row for row in rows):
        env_obs["wrist_images"] = concat_values([row["wrist_images"] for row in rows])

    return env_obs, [
        _rlinf_env_obs_report(row, source="context_passthrough") for row in rows
    ]


def _rlinf_env_obs_report(env_obs: dict[str, Any], *, source: str) -> dict[str, Any]:
    def shape_of(value: Any) -> list[int] | None:
        shape = getattr(value, "shape", None)
        return [int(dim) for dim in shape] if shape is not None else None

    primary_image_key = (
        "main_images"
        if "main_images" in env_obs
        else "images"
        if "images" in env_obs
        else None
    )
    return {
        "source": source,
        "primary_image_key": primary_image_key,
        "main_images_shape": shape_of(
            env_obs.get(primary_image_key) if primary_image_key is not None else None
        ),
        "wrist_images_shape": shape_of(env_obs.get("wrist_images")),
        "states_shape": shape_of(env_obs.get("states")),
        "has_task_descriptions": "task_descriptions" in env_obs,
    }


def _rlinf_primary_image_key(model: Any) -> str:
    """Return the env_obs image key required by the resolved RLinf API."""

    return (
        "images"
        if getattr(model, "_art_rlinf_loader_api", None) == "v0.1"
        else "main_images"
    )


def _rlinf_native_env_obs_from_observation(
    observation: Observation,
    *,
    instruction: str,
    num_images_in_input: int | None,
    use_proprio: bool | None,
    primary_image_key: str = "main_images",
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    import torch

    images = _limit_openvla_image_count(
        _observation_to_openvla_images(observation),
        num_images_in_input,
    )
    main_image = _image_to_uint8_hwc(images[0])
    if primary_image_key not in {"images", "main_images"}:
        raise ValueError(f"Unsupported RLinf primary image key: {primary_image_key!r}")
    if primary_image_key == "images":
        # RLinf v0.1's prismatic processor accepts uint8 BCHW tensors.
        main_image_batch = np.transpose(main_image, (2, 0, 1))[None, ...]
    else:
        # Later RLinf releases accept the simulator-native uint8 BHWC layout.
        main_image_batch = main_image[None, ...]
    env_obs: dict[str, Any] = {
        primary_image_key: torch.from_numpy(np.array(main_image_batch, copy=True)),
        "task_descriptions": [instruction],
    }
    if len(images) > 1:
        wrist_image = _image_to_uint8_hwc(images[1])
        wrist_image_batch = (
            np.transpose(wrist_image, (2, 0, 1))[None, ...]
            if primary_image_key == "images"
            else wrist_image[None, ...]
        )
        env_obs["wrist_images"] = torch.from_numpy(
            np.array(wrist_image_batch, copy=True)
        )

    proprio = _observation_proprio_state(observation)
    if proprio is None:
        # RLinf's input processor expects the key even when use_proprio=False.
        # Keep this fallback explicit in metadata so it cannot masquerade as a
        # faithful robot-state path in future benchmarks.
        proprio_array = np.zeros((8,), dtype=np.float32)
        proprio_source = "zeros_fallback"
    else:
        proprio_array = np.asarray(proprio, dtype=np.float32).reshape(-1)
        proprio_source = "observation"
    env_obs["states"] = torch.from_numpy(np.array(proprio_array[None, ...], copy=True))
    return env_obs, {
        "primary_image_key": primary_image_key,
        "main_images_shape": list(env_obs[primary_image_key].shape),
        "wrist_images_shape": list(env_obs["wrist_images"].shape)
        if "wrist_images" in env_obs
        else None,
        "states_shape": list(env_obs["states"].shape),
        "proprio_source": proprio_source,
        "num_images": len(images),
        "use_proprio_override": use_proprio,
    }


def _rlinf_native_env_obs_batch_from_observations(
    model: Any,
    observations: list[Observation],
    *,
    instructions: list[str],
    num_images_in_input: int | None,
    use_proprio: bool | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    if len(observations) != len(instructions):
        raise ValueError(
            "RLinf native batch conversion requires matching observations/instructions"
        )
    per_row: list[tuple[dict[str, Any], dict[str, Any]]] = [
        _rlinf_native_env_obs_from_observation(
            observation,
            instruction=instruction,
            num_images_in_input=num_images_in_input,
            use_proprio=use_proprio,
            primary_image_key=_rlinf_primary_image_key(model),
        )
        for observation, instruction in zip(observations, instructions, strict=True)
    ]
    env_obs_rows = [item[0] for item in per_row]
    obs_reports = [item[1] for item in per_row]
    primary_image_key = _rlinf_primary_image_key(model)
    env_obs: dict[str, Any] = {
        primary_image_key: torch.cat(
            [row[primary_image_key] for row in env_obs_rows],
            dim=0,
        ),
        "task_descriptions": list(instructions),
        "states": torch.cat([row["states"] for row in env_obs_rows], dim=0),
    }
    if all("wrist_images" in row for row in env_obs_rows):
        env_obs["wrist_images"] = torch.cat(
            [row["wrist_images"] for row in env_obs_rows], dim=0
        )
    return env_obs, obs_reports


def _rlinf_native_action_token_logprobs_batch(
    model: Any,
    *,
    observations: list[Observation],
    instructions: list[str],
    token_rows: list[list[Any]],
    temperature: float,
    num_images_in_input: int | None,
    use_proprio: bool | None,
    env_obs_overrides: list[dict[str, Any] | None] | None = None,
    forward_input_overrides: list[dict[str, Any] | None] | None = None,
) -> list[Any]:
    """Score action tokens through RLinf's native OpenVLA-OFT rollout contract.

    The rollout path records old logprobs from
    ``OpenVLAOFTForRLActionPrediction.predict_action_batch``.  RLinf's
    ``default_forward`` can score a different action-logit slice from rollout,
    so the old-logprob contract must be matched to rollout itself, not merely to
    a train convenience method.  For parity runs we let RLinf build the exact
    model-facing inputs used by rollout, then rerun the same predict-style logit
    slice with gradients enabled for the requested action token rows.
    """

    import torch

    if not observations:
        return []
    if len(observations) != len(instructions) or len(observations) != len(token_rows):
        raise ValueError(
            "RLinf native logprob rescore requires aligned observations/instructions/tokens"
        )

    forward_inputs = _rlinf_native_action_token_forward_inputs_batch(
        model,
        observations=observations,
        instructions=instructions,
        temperature=float(temperature),
        num_images_in_input=num_images_in_input,
        use_proprio=use_proprio,
        env_obs_overrides=env_obs_overrides,
        forward_input_overrides=forward_input_overrides,
    )
    token_ids = _rlinf_native_token_ids_tensor(
        model,
        token_rows=token_rows,
        device=forward_inputs["input_ids"].device,
    )
    logprobs = _rlinf_predict_style_action_token_logprobs_from_forward_inputs(
        model,
        forward_inputs=forward_inputs,
        token_ids=token_ids,
        temperature=float(temperature),
    )
    if tuple(logprobs.shape[:2]) != tuple(token_ids.shape):
        raise ValueError(
            "RLinf native logprob shape mismatch: "
            f"logprobs={tuple(logprobs.shape)} tokens={tuple(token_ids.shape)}"
        )
    return [logprobs[index] for index in range(logprobs.shape[0])]


def _rlinf_native_action_token_forward_inputs_batch(
    model: Any,
    *,
    observations: list[Observation],
    instructions: list[str],
    temperature: float,
    num_images_in_input: int | None,
    use_proprio: bool | None,
    env_obs_overrides: list[dict[str, Any] | None] | None = None,
    forward_input_overrides: list[dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Build RLinf OpenVLA-OFT rollout inputs without tying them to sampled tokens."""

    if not observations:
        raise ValueError("Cannot build RLinf native forward inputs for an empty batch")
    if len(observations) != len(instructions):
        raise ValueError(
            "RLinf native forward input capture requires aligned observations/instructions"
        )
    if forward_input_overrides is not None and any(
        item is not None for item in forward_input_overrides
    ):
        if len(forward_input_overrides) != len(observations):
            raise ValueError(
                "RLinf native forward_input overrides must match observation batch length"
            )
        if any(item is None for item in forward_input_overrides):
            raise ValueError(
                "RLinf native forward_input overrides must be provided for the whole batch or none of it"
            )
        return _merge_rlinf_forward_input_rows(
            [item for item in forward_input_overrides if item is not None],
            model=model,
        )
    if env_obs_overrides is not None and any(
        item is not None for item in env_obs_overrides
    ):
        if len(env_obs_overrides) != len(observations):
            raise ValueError(
                "RLinf native env_obs overrides must match observation batch length"
            )
        if any(item is None for item in env_obs_overrides):
            raise ValueError(
                "RLinf native env_obs overrides must be provided for the whole batch or none of it"
            )
        env_obs_rows = [
            _copy_rlinf_env_obs_for_model(item, instructions=[instruction])
            for item, instruction in zip(env_obs_overrides, instructions, strict=True)
            if item is not None
        ]
        env_obs, _obs_reports = _merge_rlinf_env_obs_rows(
            env_obs_rows, instructions=instructions
        )
    else:
        env_obs, _obs_reports = _rlinf_native_env_obs_batch_from_observations(
            model,
            observations,
            instructions=instructions,
            num_images_in_input=num_images_in_input,
            use_proprio=use_proprio,
        )

    # Let RLinf construct exactly the same model-facing inputs as rollout.  The
    # sampled/argmax action tokens from this no-grad call are placeholders; they
    # are overwritten below by the tokens from the trajectory examples.
    _raw_actions, rollout_result = model.predict_action_batch(
        env_obs=env_obs,
        do_sample=False,
        temperature=float(temperature),
        top_k=-1,
        calculate_values=False,
    )
    forward_inputs = dict(rollout_result.get("forward_inputs", {}))
    required_keys = {"input_ids", "attention_mask", "pixel_values"}
    missing = sorted(required_keys.difference(forward_inputs))
    if missing:
        raise RuntimeError(
            f"RLinf native rollout did not return forward_inputs keys: {missing}"
        )
    # ``action_tokens`` is model-dependent placeholder output from the no-grad
    # rollout call.  Scoring callers overwrite or ignore it with their fixed
    # action token rows, so keep only input tensors that define the observation
    # and prompt.  This makes fixed-forward-input diagnostics explicit.
    forward_inputs.pop("action_tokens", None)
    return forward_inputs


def _rlinf_native_token_ids_tensor(
    model: Any,
    *,
    token_rows: list[list[Any]],
    device: Any,
) -> Any:
    import torch

    constants = _openvla_oft_constants(model, unnorm_key=None)
    action_dim = int(getattr(model, "action_dim", constants["ACTION_DIM"]))
    num_chunks = int(
        getattr(model, "num_action_chunks", constants["NUM_ACTIONS_CHUNK"])
    )
    expected_tokens = action_dim * num_chunks
    lengths = {len(row) for row in token_rows}
    if lengths != {expected_tokens}:
        raise ValueError(
            "RLinf native OpenVLA-OFT token rows must match "
            f"action_dim*num_action_chunks={expected_tokens}; got {sorted(lengths)}"
        )
    token_ids = torch.as_tensor(
        [[_coerce_int_token(token) for token in row] for row in token_rows],
        dtype=torch.long,
        device=device,
    )
    action_start, action_end = _openvla_oft_action_token_range(model)
    if bool(((token_ids < action_start) | (token_ids >= action_end)).any()):
        raise ValueError(
            "RLinf native OpenVLA-OFT action token ids must be inside "
            f"[{action_start}, {action_end})"
        )
    return token_ids


def _detach_forward_inputs_to_cpu(forward_inputs: dict[str, Any]) -> dict[str, Any]:
    cached: dict[str, Any] = {}
    for key, value in forward_inputs.items():
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            cached[key] = value.detach().cpu()
        else:
            cached[key] = value
    return cached


def _forward_inputs_to_device(
    forward_inputs: dict[str, Any], *, device: Any
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in forward_inputs.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def _slice_forward_inputs_row(
    forward_inputs: dict[str, Any], *, row_index: int
) -> dict[str, Any]:
    import numpy as np

    row: dict[str, Any] = {}
    for key, value in forward_inputs.items():
        if key == "action_tokens":
            continue
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            row[key] = value[row_index : row_index + 1].detach().cpu().contiguous()
        elif isinstance(value, np.ndarray):
            row[key] = value[row_index : row_index + 1].copy()
        else:
            row[key] = value
    return row


def _merge_rlinf_forward_input_rows(
    rows: list[dict[str, Any]],
    *,
    model: Any,
) -> dict[str, Any]:
    import numpy as np
    import torch

    if not rows:
        raise ValueError("Cannot merge empty RLinf forward_input rows")
    required_keys = {"input_ids", "attention_mask", "pixel_values"}
    missing = sorted(
        key for key in required_keys if any(key not in row for row in rows)
    )
    if missing:
        raise RuntimeError(
            f"RLinf native forward_input overrides are missing keys: {missing}"
        )
    common_keys = set(rows[0])
    for row in rows[1:]:
        common_keys.intersection_update(row)
    common_keys.discard("action_tokens")
    ordered_keys = [
        key
        for key in ("input_ids", "attention_mask", "pixel_values")
        if key in common_keys
    ]
    ordered_keys.extend(
        sorted(key for key in common_keys if key not in set(ordered_keys))
    )
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = None
    merged: dict[str, Any] = {}
    for key in ordered_keys:
        values = [row[key] for row in rows]
        tensor_values: list[Any] = []
        for value in values:
            if hasattr(value, "detach"):
                tensor = value.detach()
            elif isinstance(value, np.ndarray):
                tensor = torch.as_tensor(value)
            else:
                tensor = torch.as_tensor(value)
            tensor_values.append(tensor)
        merged_tensor = torch.cat(tensor_values, dim=0)
        if device is not None:
            merged_tensor = merged_tensor.to(device=device)
        merged[key] = merged_tensor
    return merged


def _rlinf_predict_style_action_token_logprobs_from_forward_inputs(
    model: Any,
    *,
    forward_inputs: dict[str, Any],
    token_ids: Any,
    temperature: float,
) -> Any:
    """Differentiate the exact OpenVLA-OFT/RLinf rollout logprob definition.

    RLinf rollout computes old logprobs inside ``predict_action_batch`` from:

    ``outputs.logits[:, n_patches + (input_len - 1) : ... + action_tokens]``.

    Its ``default_forward`` path scores a suffix slice instead. Those are meant
    to coincide only if the wrapper's prepared prompt/action layout is exactly
    suffix-aligned. The strict ``previous_abs_delta`` gate showed this assumption
    is too weak for ART parity. This helper is intentionally close to RLinf's
    rollout code and returns logprobs for caller-supplied action-token ids.
    """

    import torch

    input_ids = forward_inputs["input_ids"]
    attention_mask = forward_inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(dtype=torch.long)
    pixel_values = forward_inputs["pixel_values"]

    n_prompt_tokens = input_ids.shape[-1] - 1
    n_patches = (
        model.vision_backbone.get_num_patches()
        * model.vision_backbone.get_num_images_in_input()
    )
    action_token_count = int(model.action_dim) * int(model.num_action_chunks)
    if token_ids.shape != (input_ids.shape[0], action_token_count):
        raise ValueError(
            "RLinf predict-style token shape mismatch: "
            f"tokens={tuple(token_ids.shape)} expected={(input_ids.shape[0], action_token_count)}"
        )

    input_ids, attention_mask = model._prepare_input_for_action_prediction(
        input_ids, attention_mask
    )
    multimodal_embeddings, multimodal_attention_mask = model._build_embedding(
        input_ids,
        attention_mask,
        pixel_values,
    )
    multimodal_position_ids = multimodal_attention_mask.cumsum(dim=1) - 1
    outputs = model.language_model(
        input_ids=None,
        attention_mask=multimodal_attention_mask,
        position_ids=multimodal_position_ids,
        past_key_values=None,
        inputs_embeds=multimodal_embeddings,
        labels=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    logits = outputs.logits[
        :,
        n_patches + n_prompt_tokens : n_patches + n_prompt_tokens + action_token_count,
        :,
    ].clone()
    logits[..., : model.vocab_size - model.config.n_action_bins] = -torch.inf
    logits[..., model.vocab_size :] = -torch.inf
    scaled_logits = logits / max(float(temperature), 1e-6)
    import torch.nn.functional as F

    batch_shape = scaled_logits.shape[:-1]
    vocab_shape = scaled_logits.shape[-1]
    return (
        -F.cross_entropy(
            scaled_logits.reshape(-1, vocab_shape),
            token_ids.reshape(-1),
            reduction="none",
        )
        .view(*batch_shape)
        .float()
    )


def _image_to_uint8_hwc(value: Any) -> Any:
    import numpy as np

    image = _value_to_image(value)
    array = np.asarray(image.convert("RGB"))
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _first_batch_value(value: Any) -> Any:
    value = _numpy_compatible_tensor_value(value)
    try:
        import numpy as np

        array = np.asarray(value)
        if array.shape and array.shape[0] == 1:
            return array[0]
        return array
    except Exception:
        if isinstance(value, (list, tuple)) and len(value) == 1:
            return value[0]
        return value


def _flatten_first_batch_values(value: Any, *, dtype: type) -> list[Any]:
    import numpy as np

    value = _numpy_compatible_tensor_value(value)
    array = np.asarray(value)
    if array.shape and array.shape[0] == 1:
        array = array[0]
    flat = array.reshape(-1)
    return [dtype(item) for item in flat.tolist()]


def _flatten_batch_row_values(value: Any, *, row_index: int, dtype: type) -> list[Any]:
    import numpy as np

    value = _numpy_compatible_tensor_value(value)
    array = np.asarray(value)
    if not array.shape:
        return [dtype(array.item())]
    if row_index >= array.shape[0]:
        raise IndexError(f"batch row {row_index} out of range for shape {array.shape}")
    flat = array[row_index].reshape(-1)
    return [dtype(item) for item in flat.tolist()]


def _batch_row_value(value: Any, *, row_index: int) -> Any:
    import numpy as np

    value = _numpy_compatible_tensor_value(value)
    array = np.asarray(value)
    if array.shape and row_index < array.shape[0]:
        return array[row_index]
    return value


def _numpy_compatible_tensor_value(value: Any) -> Any:
    """Detach tensors and promote NumPy-unsupported BF16 values to FP32."""

    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if str(getattr(value, "dtype", "")) == "torch.bfloat16":
            value = value.float()
    return value


def _transient_rlinf_env_obs_override_from_example(
    example: Any,
) -> dict[str, Any] | None:
    """Return exact rollout env_obs carried only for native RLinf parity scoring."""

    metadata = getattr(example, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    direct = metadata.get(TRANSIENT_RLINF_ENV_OBS_METADATA_KEY)
    if isinstance(direct, dict):
        return direct
    action_metadata = metadata.get("action_metadata")
    if isinstance(action_metadata, dict):
        value = action_metadata.get(TRANSIENT_RLINF_ENV_OBS_METADATA_KEY)
        if isinstance(value, dict):
            return value
    action_span = metadata.get("action_span")
    if isinstance(action_span, dict):
        span_action_metadata = action_span.get("action_metadata")
        if isinstance(span_action_metadata, dict):
            value = span_action_metadata.get(TRANSIENT_RLINF_ENV_OBS_METADATA_KEY)
            if isinstance(value, dict):
                return value
    return None


def _transient_rlinf_forward_inputs_override_from_example(
    example: Any,
) -> dict[str, Any] | None:
    """Return exact rollout forward_inputs carried only for RLinf parity scoring."""

    metadata = getattr(example, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    direct = metadata.get(TRANSIENT_RLINF_FORWARD_INPUTS_METADATA_KEY)
    if isinstance(direct, dict):
        return direct
    action_metadata = metadata.get("action_metadata")
    if isinstance(action_metadata, dict):
        value = action_metadata.get(TRANSIENT_RLINF_FORWARD_INPUTS_METADATA_KEY)
        if isinstance(value, dict):
            return value
    action_span = metadata.get("action_span")
    if isinstance(action_span, dict):
        span_action_metadata = action_span.get("action_metadata")
        if isinstance(span_action_metadata, dict):
            value = span_action_metadata.get(
                TRANSIENT_RLINF_FORWARD_INPUTS_METADATA_KEY
            )
            if isinstance(value, dict):
                return value
        value = action_span.get(TRANSIENT_RLINF_FORWARD_INPUTS_METADATA_KEY)
        if isinstance(value, dict):
            return value
    return None


class OpenVLAPolicy:
    """Adapter for OpenVLA and OpenVLA-OFT action-token policies.

    ``model_loader="native"`` is the supported OpenVLA-OFT RL path. It records
    sampled action tokens, rollout-time log probabilities, decoded action
    chunks, and a bounded runtime contract for later rescoring and observability.
    The generic Transformers path remains available for inference-only OpenVLA
    checkpoints; the optional RLinf loader exists only as a conformance oracle.
    """

    def __init__(
        self,
        *,
        model_id: str = "openvla/openvla-7b",
        revision: str | None = None,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        unnorm_key: str | None = "bridge_orig",
        robot_platform: str | None = None,
        attn_implementation: str | None = None,
        trust_remote_code: bool = True,
        load_on_init: bool = False,
        capture_action_tokens: bool = True,
        action_output_kind: str = "token",
        do_sample: bool = False,
        temperature: float = 1.0,
        peft_adapter_path: str | None = None,
        dataset_statistics_path: str | Path | None = None,
        logprob_batch_size: int | None = None,
        strict_batched_logprobs: bool = False,
        max_prompt_length: int | None = None,
        prompt_template: str | None = None,
        lowercase_instruction: bool = True,
        num_images_in_input: int | None = None,
        use_proprio: bool | None = None,
        action_dim: int = 7,
        num_action_chunks: int = 8,
        model_loader: str = "transformers",
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.dtype = dtype
        self.unnorm_key = unnorm_key
        self.robot_platform = robot_platform
        self.attn_implementation = attn_implementation
        self.trust_remote_code = trust_remote_code
        if model_loader not in ("native", "transformers", "rlinf"):
            raise ValueError(
                "model_loader must be 'native', 'transformers', or 'rlinf'"
            )
        self.model_loader = model_loader
        if action_output_kind not in ("token", "continuous"):
            raise ValueError("action_output_kind must be 'token' or 'continuous'")
        if action_output_kind == "continuous" and not capture_action_tokens:
            raise ValueError(
                "action_output_kind='continuous' requires capture_action_tokens=True"
            )
        self.capture_action_tokens = capture_action_tokens
        self.action_output_kind = action_output_kind
        self.do_sample = bool(do_sample)
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.peft_adapter_path = peft_adapter_path
        self.dataset_statistics_path = (
            Path(dataset_statistics_path).expanduser()
            if dataset_statistics_path is not None
            else None
        )
        if logprob_batch_size is not None and int(logprob_batch_size) <= 0:
            raise ValueError("logprob_batch_size must be a positive integer or None")
        self.logprob_batch_size = (
            int(logprob_batch_size) if logprob_batch_size is not None else None
        )
        self.strict_batched_logprobs = bool(strict_batched_logprobs)
        self._last_batched_action_token_logprob_error: dict[str, Any] | None = None
        if max_prompt_length is not None and int(max_prompt_length) <= 0:
            raise ValueError("max_prompt_length must be a positive integer or None")
        self.max_prompt_length = (
            int(max_prompt_length) if max_prompt_length is not None else None
        )
        self.prompt_template = prompt_template
        self.lowercase_instruction = bool(lowercase_instruction)
        if num_images_in_input is not None and int(num_images_in_input) <= 0:
            raise ValueError("num_images_in_input must be a positive integer or None")
        self.num_images_in_input = (
            int(num_images_in_input) if num_images_in_input is not None else None
        )
        self.use_proprio = None if use_proprio is None else bool(use_proprio)
        if int(action_dim) <= 0 or int(num_action_chunks) <= 0:
            raise ValueError("action_dim and num_action_chunks must be positive")
        self.action_dim = int(action_dim)
        self.num_action_chunks = int(num_action_chunks)
        self.processor: Any | None = None
        self.model: Any | None = None
        self._rlinf_native_loader: bool = False
        self.native_runtime_report: dict[str, Any] | None = None
        self.dataset_statistics_report: dict[str, Any] | None = None
        self.trainable_report: dict[str, Any] | None = None
        if load_on_init:
            self.load()

    def load(self) -> None:
        if self.peft_adapter_path and self.model_loader != "rlinf":
            adapter_path = Path(self.peft_adapter_path).expanduser()
            if adapter_path.exists():
                _validate_or_adopt_adapter_base_contract(self, adapter_path)
        raise_for_openvla_runtime_requirements(
            require_cuda=self.device.startswith("cuda")
        )
        for warning in openvla_runtime_warnings():
            print(f"[OpenVLA runtime warning] {warning}")

        import torch

        torch_dtype = getattr(torch, self.dtype)
        if self.model_loader == "rlinf":
            if self.revision is not None:
                raise ValueError(
                    "model_loader='rlinf' does not support policy revision pinning; "
                    "use model_loader='native' or a revision-pinned local snapshot"
                )
            self.model = _load_rlinf_openvla_oft_model(
                model_id=self.model_id,
                torch_dtype=torch_dtype,
                device=self.device,
                unnorm_key=self.unnorm_key or "libero",
                max_prompt_length=self.max_prompt_length or 128,
                num_images_in_input=self.num_images_in_input or 2,
                attn_implementation=self.attn_implementation,
                peft_adapter_path=self.peft_adapter_path,
            )
            self.processor = getattr(self.model, "input_processor", None)
            self._rlinf_native_loader = True
            self.dataset_statistics_report = _merge_external_dataset_statistics(
                self.model,
                self.model_id,
                explicit_path=self.dataset_statistics_path,
            )
            self.model.eval()
            return

        _install_openvla_oft_prismatic_compat_if_needed(
            hint=self.model_id,
            robot_platform=self.robot_platform,
        )
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
        )
        _configure_rlinf_openvla_oft_processor_if_present(self.processor)
        kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            "revision": self.revision,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.attn_implementation is not None:
            kwargs["attn_implementation"] = self.attn_implementation
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            **kwargs,
        ).to(device=self.device, dtype=torch_dtype)
        _attach_processor_to_openvla_oft_model(self.model, self.processor)
        self._rlinf_native_loader = False
        self.dataset_statistics_report = _merge_external_dataset_statistics(
            self.model,
            self.model_id,
            explicit_path=self.dataset_statistics_path,
            revision=self.revision,
        )
        if self.model_loader == "native":
            from art_embodied.policies.openvla_oft_native import (
                configure_native_openvla_oft_model,
            )

            oft_model = _openvla_oft_action_model(self.model)
            if oft_model is None:
                raise TypeError(
                    "model_loader='native' requires an OpenVLA-OFT action-token checkpoint"
                )
            self.native_runtime_report = configure_native_openvla_oft_model(
                oft_model,
                self.processor,
                action_dim=self.action_dim,
                num_action_chunks=self.num_action_chunks,
                max_prompt_length=self.max_prompt_length or 128,
                num_images_in_input=self.num_images_in_input or 1,
                unnorm_key=self.unnorm_key,
            )
        if self.peft_adapter_path:
            self.model = _attach_peft_adapter(
                self.model,
                self.peft_adapter_path,
                device=self.device,
            )
        self.model.eval()

    def act(self, observation: Observation, context: dict[str, Any]) -> Action:
        if self.processor is None or self.model is None:
            self.load()
        assert self.processor is not None
        assert self.model is not None
        import torch

        self.model.eval()

        instruction = _instruction_from_context(context)
        resolved_unnorm_key = _resolve_unnorm_key(self.model, self.unnorm_key)
        prompt = _prompt_from_task(
            instruction,
            template=self.prompt_template,
            lowercase_instruction=self.lowercase_instruction,
        )
        if self._rlinf_native_loader and _openvla_oft_supports_rlinf_rollout_style(
            self.model
        ):
            with torch.no_grad():
                action, token_ids, token_logprobs, output_metadata = (
                    _predict_rlinf_native_policy_action(
                        self.model,
                        observation=observation,
                        instruction=instruction,
                        do_sample=self.do_sample,
                        temperature=self.temperature,
                        num_images_in_input=_openvla_resolved_num_images(
                            self.model, self.num_images_in_input
                        ),
                        use_proprio=self.use_proprio,
                        env_obs_override=context.get("rlinf_env_obs"),
                    )
                )
            distribution_metadata = _openvla_binned_categorical_distribution_metadata(
                self.model,
                token_ids=token_ids,
                unnorm_key=resolved_unnorm_key,
                temperature=self.temperature,
            )
            raw = {
                "model_id": self.model_id,
                "revision": self.revision,
                "prompt": prompt,
                "tokens": token_ids,
                "do_sample": self.do_sample,
                "temperature": self.temperature,
                "model_loader": self.model_loader,
            }
            if self.action_output_kind == "continuous":
                logprobs = {
                    "action_logprobs": token_logprobs,
                    "token_logprobs": token_logprobs,
                    "policy_distribution": distribution_metadata,
                }
                kind = "continuous"
            else:
                logprobs = {"token_logprobs": token_logprobs}
                kind = "token"
            action_metadata = {
                "policy": "openvla",
                "model_id": self.model_id,
                "revision": self.revision,
                "model_loader": self.model_loader,
                "peft_adapter_path": self.peft_adapter_path,
                "unnorm_key": resolved_unnorm_key,
                "configured_unnorm_key": self.unnorm_key,
                "available_unnorm_keys": _available_norm_stat_keys(self.model),
                "dataset_statistics": make_json_safe(self.dataset_statistics_report),
                "native_runtime": make_json_safe(self.native_runtime_report),
                "num_images_in_input_override": self.num_images_in_input,
                "use_proprio_override": self.use_proprio,
                "action_logit_extraction": _openvla_oft_logit_extraction_method(
                    self.model
                ),
                **_openvla_oft_action_range_metadata(self.model),
                "rollout_model_mode": "eval",
                "observation_step": observation.step,
                **output_metadata,
            }
            return Action(
                step=int(context.get("step", 0)),
                kind=kind,  # type: ignore[arg-type]
                raw=raw,
                decoded=make_json_safe(action),
                logprobs=logprobs,
                metadata=action_metadata,
            )

        inputs = _prepare_openvla_inputs(
            self.processor,
            observation=observation,
            prompt=prompt,
            device=self.device,
            dtype=self.dtype,
            max_prompt_length=self.max_prompt_length,
            max_images=_openvla_resolved_num_images(
                self.model, self.num_images_in_input
            ),
            use_proprio=self.use_proprio,
            prefer_tensor_image_processing=self.model_loader == "native",
        )

        with torch.no_grad():
            if self.capture_action_tokens:
                action, token_ids, token_logprobs = _predict_action_with_tokens(
                    self.model,
                    inputs,
                    unnorm_key=resolved_unnorm_key,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                )
                raw: dict[str, Any] = {
                    "model_id": self.model_id,
                    "revision": self.revision,
                    "prompt": prompt,
                    "tokens": token_ids,
                    "do_sample": self.do_sample,
                    "temperature": self.temperature,
                }
                distribution_metadata = (
                    _openvla_binned_categorical_distribution_metadata(
                        self.model,
                        token_ids=token_ids,
                        unnorm_key=resolved_unnorm_key,
                        temperature=self.temperature,
                    )
                )
                output_metadata = {"policy_distribution": distribution_metadata}
                if self.action_output_kind == "continuous":
                    logprobs = {
                        "action_logprobs": token_logprobs,
                        "token_logprobs": token_logprobs,
                        "policy_distribution": distribution_metadata,
                    }
                    kind = "continuous"
                else:
                    logprobs = {"token_logprobs": token_logprobs}
                    kind = "token"
            else:
                predict_kwargs: dict[str, Any] = {"do_sample": False}
                if resolved_unnorm_key is not None:
                    predict_kwargs["unnorm_key"] = resolved_unnorm_key
                action_output = self.model.predict_action(**inputs, **predict_kwargs)
                action, output_metadata = _primary_action_from_predict_action_output(
                    action_output
                )
                raw = {
                    "model_id": self.model_id,
                    "revision": self.revision,
                    "prompt": prompt,
                }
                logprobs = None
                kind = "continuous"

        action_metadata = {
            "policy": "openvla",
            "model_id": self.model_id,
            "revision": self.revision,
            "model_loader": self.model_loader,
            "peft_adapter_path": self.peft_adapter_path,
            "unnorm_key": resolved_unnorm_key,
            "configured_unnorm_key": self.unnorm_key,
            "available_unnorm_keys": _available_norm_stat_keys(self.model),
            "dataset_statistics": make_json_safe(self.dataset_statistics_report),
            "native_runtime": make_json_safe(self.native_runtime_report),
            "num_images_in_input_override": self.num_images_in_input,
            "use_proprio_override": self.use_proprio,
            "action_logit_extraction": _openvla_oft_logit_extraction_method(self.model),
            **_openvla_oft_action_range_metadata(self.model),
            "rollout_model_mode": "eval",
            "observation_step": observation.step,
        }
        if output_metadata:
            action_metadata.update(output_metadata)

        return Action(
            step=int(context.get("step", 0)),
            kind=kind,  # type: ignore[arg-type]
            raw=raw,
            decoded=make_json_safe(action),
            logprobs=logprobs,
            metadata=action_metadata,
        )

    def act_batch(
        self,
        observations: list[Observation],
        contexts: list[dict[str, Any]] | None = None,
    ) -> list[Action]:
        """Predict actions for multiple active environments with one policy call.

        This is the rollout-side counterpart to batched action-token logprob
        recomputation. OpenVLA-OFT placeholder-token models can produce action
        logits for a whole image/prompt batch, which is the minimum building
        block needed for ART-style grouped embodied rollouts. Vanilla OpenVLA
        generation remains on the safe sequential fallback until it has a tested
        teacher-forced/batched generation contract.
        """

        if not observations:
            return []
        contexts = contexts or [{} for _ in observations]
        if len(contexts) != len(observations):
            raise ValueError(
                "OpenVLAPolicy.act_batch requires one context per observation "
                f"({len(contexts)} contexts for {len(observations)} observations)"
            )
        max_policy_batch_size = self.logprob_batch_size
        if (
            max_policy_batch_size is not None
            and len(observations) > max_policy_batch_size
        ):
            actions: list[Action] = []
            for start in range(0, len(observations), max_policy_batch_size):
                actions.extend(
                    self.act_batch(
                        observations[start : start + max_policy_batch_size],
                        contexts[start : start + max_policy_batch_size],
                    )
                )
            return actions
        if self.processor is None or self.model is None:
            self.load()
        assert self.processor is not None
        assert self.model is not None
        import torch

        self.model.eval()

        oft_model = _openvla_oft_action_model(self.model)
        if oft_model is None or not self.capture_action_tokens:
            return [
                self.act(observation, context)
                for observation, context in zip(observations, contexts, strict=True)
            ]

        instructions = [_instruction_from_context(context) for context in contexts]
        prompts = [
            _prompt_from_task(
                instruction,
                template=self.prompt_template,
                lowercase_instruction=self.lowercase_instruction,
            )
            for instruction in instructions
        ]
        resolved_unnorm_key = _resolve_unnorm_key(self.model, self.unnorm_key)
        if self._rlinf_native_loader and _openvla_oft_supports_rlinf_rollout_style(
            oft_model
        ):
            with torch.no_grad():
                predictions = _predict_rlinf_native_policy_actions_batch(
                    oft_model,
                    observations=observations,
                    instructions=instructions,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                    num_images_in_input=_openvla_resolved_num_images(
                        self.model, self.num_images_in_input
                    ),
                    use_proprio=self.use_proprio,
                    env_obs_overrides=[
                        context.get("rlinf_env_obs") for context in contexts
                    ],
                )
        else:
            inputs = _prepare_openvla_batch_inputs(
                self.processor,
                observations=observations,
                prompts=prompts,
                device=self.device,
                dtype=self.dtype,
                max_prompt_length=self.max_prompt_length,
                max_images=_openvla_resolved_num_images(
                    self.model, self.num_images_in_input
                ),
                use_proprio=self.use_proprio,
                prefer_tensor_image_processing=self.model_loader == "native",
            )
            with torch.no_grad():
                predictions = _predict_oft_actions_with_tokens_batch(
                    oft_model,
                    inputs,
                    unnorm_key=resolved_unnorm_key,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                )
        actions: list[Action] = []
        available_norm_keys = _available_norm_stat_keys(self.model)
        logit_method = _openvla_oft_logit_extraction_method(self.model)
        range_metadata = _openvla_oft_action_range_metadata(self.model)
        for index, prediction in enumerate(predictions):
            if len(prediction) == 4:
                action, token_ids, token_logprobs, output_metadata = prediction
            else:
                action, token_ids, token_logprobs = prediction
                output_metadata = {}
            if not isinstance(output_metadata, dict):
                output_metadata = {"policy_output_metadata": output_metadata}
            safe_output_metadata = make_json_safe(output_metadata)
            if TRANSIENT_RLINF_FORWARD_INPUTS_METADATA_KEY in output_metadata:
                safe_output_metadata[TRANSIENT_RLINF_FORWARD_INPUTS_METADATA_KEY] = (
                    output_metadata[TRANSIENT_RLINF_FORWARD_INPUTS_METADATA_KEY]
                )
            context = contexts[index]
            observation = observations[index]
            actions.append(
                Action(
                    step=int(context.get("step", 0)),
                    kind=self.action_output_kind,  # type: ignore[arg-type]
                    raw={
                        "model_id": self.model_id,
                        "revision": self.revision,
                        "prompt": prompts[index],
                        "tokens": token_ids,
                        "do_sample": self.do_sample,
                        "temperature": self.temperature,
                        "batch_index": index,
                        "batch_size": len(observations),
                    },
                    decoded=make_json_safe(action),
                    logprobs=_openvla_action_logprobs_payload(
                        token_logprobs=token_logprobs,
                        distribution_metadata=_openvla_binned_categorical_distribution_metadata(
                            self.model,
                            token_ids=token_ids,
                            unnorm_key=resolved_unnorm_key,
                            temperature=self.temperature,
                        ),
                        action_output_kind=self.action_output_kind,
                    ),
                    metadata={
                        "policy": "openvla",
                        "model_id": self.model_id,
                        "revision": self.revision,
                        "model_loader": self.model_loader,
                        "peft_adapter_path": self.peft_adapter_path,
                        "unnorm_key": resolved_unnorm_key,
                        "configured_unnorm_key": self.unnorm_key,
                        "available_unnorm_keys": available_norm_keys,
                        "dataset_statistics": make_json_safe(
                            self.dataset_statistics_report
                        ),
                        "native_runtime": make_json_safe(self.native_runtime_report),
                        "action_logit_extraction": logit_method,
                        **range_metadata,
                        "policy_distribution": _openvla_binned_categorical_distribution_metadata(
                            self.model,
                            token_ids=token_ids,
                            unnorm_key=resolved_unnorm_key,
                            temperature=self.temperature,
                        ),
                        "rollout_model_mode": "eval",
                        "observation_step": observation.step,
                        "batched_policy_call": True,
                        "batch_index": index,
                        "batch_size": len(observations),
                        **safe_output_metadata,
                    },
                )
            )
        return actions

    def action_token_logprobs(self, examples: list[Any]) -> list[Any]:
        """Recompute current action-token logprobs for GRPO/GSPO backends.

        OpenVLA-OFT exposes action-token placeholder logits instead of the
        vanilla OpenVLA ``generate`` interface. This method keeps the backend
        contract policy-specific: action-token examples carry prompts,
        observations, and old rollout logprobs; the policy adapter reconstructs
        the same image+prompt forward pass and gathers current logprobs for the
        selected action tokens. Gradients flow through the remote-code model.
        """

        if self.processor is None or self.model is None:
            self.load()
        assert self.processor is not None
        assert self.model is not None

        if any(_is_trajectory_action_token_example(example) for example in examples):
            batched = self._batched_trajectory_action_token_logprobs_for_examples(
                examples
            )
            if batched is not None:
                return batched
        else:
            batched = self._batched_action_token_logprobs_for_examples(examples)
            if batched is not None:
                return batched

        rows: list[Any] = []
        for example in examples:
            rows.append(self._action_token_logprobs_for_example(example))
        return rows

    def capture_rlinf_native_action_token_forward_inputs(
        self,
        examples: list[Any],
    ) -> dict[str, Any] | None:
        """Capture fixed RLinf OpenVLA-OFT forward inputs for scorer diagnostics.

        This is intentionally not the normal training path.  It lets the Wan
        train loop ask a sharper question during trust-region backtracking:
        does a scaled LoRA update still produce large KL when the image/prompt
        tensors are held fixed, or is the apparent KL created by rebuilding
        forward inputs differently between probes?
        """

        self._last_fixed_forward_input_capture_error = None
        if not examples:
            self._last_fixed_forward_input_capture_error = {"reason": "empty_examples"}
            return None
        if any(_is_trajectory_action_token_example(example) for example in examples):
            self._last_fixed_forward_input_capture_error = {
                "reason": "trajectory_level_examples_not_supported"
            }
            return None
        if any(getattr(example, "observation", None) is None for example in examples):
            self._last_fixed_forward_input_capture_error = {
                "reason": "missing_observation"
            }
            return None
        token_lengths = {
            len(getattr(example, "tokens", []) or []) for example in examples
        }
        if len(token_lengths) != 1:
            self._last_fixed_forward_input_capture_error = {
                "reason": "mixed_token_lengths",
                "token_lengths": sorted(token_lengths),
            }
            return None
        if self.processor is None or self.model is None:
            self.load()
        assert self.model is not None
        oft_model = _openvla_oft_action_model(self.model)
        if (
            not self._rlinf_native_loader
            or oft_model is None
            or not _openvla_oft_supports_rlinf_rollout_style(oft_model)
        ):
            self._last_fixed_forward_input_capture_error = {
                "reason": "not_rlinf_native_openvla_oft",
                "rlinf_native_loader": bool(self._rlinf_native_loader),
                "has_oft_model": oft_model is not None,
                "supports_rlinf_rollout_style": bool(
                    oft_model is not None
                    and _openvla_oft_supports_rlinf_rollout_style(oft_model)
                ),
            }
            return None
        prompts = [
            getattr(example, "prompt", None)
            or _prompt_from_task(
                getattr(example, "task", ""),
                template=self.prompt_template,
                lowercase_instruction=self.lowercase_instruction,
            )
            for example in examples
        ]
        instructions = [
            _rlinf_instruction_from_example(example, prompt=prompt)
            for example, prompt in zip(examples, prompts, strict=True)
        ]
        model_training = bool(getattr(self.model, "training", False))
        oft_training = bool(getattr(oft_model, "training", False))
        try:
            if hasattr(self.model, "eval"):
                self.model.eval()
            if hasattr(oft_model, "eval"):
                oft_model.eval()
            import torch

            with torch.no_grad():
                forward_inputs = _rlinf_native_action_token_forward_inputs_batch(
                    oft_model,
                    observations=[example.observation for example in examples],
                    instructions=instructions,
                    temperature=self.temperature,
                    num_images_in_input=_openvla_resolved_num_images(
                        self.model, self.num_images_in_input
                    ),
                    use_proprio=self.use_proprio,
                    env_obs_overrides=[
                        _transient_rlinf_env_obs_override_from_example(example)
                        for example in examples
                    ],
                    forward_input_overrides=[
                        _transient_rlinf_forward_inputs_override_from_example(example)
                        for example in examples
                    ],
                )
        finally:
            if model_training and hasattr(self.model, "train"):
                self.model.train()
            if oft_training and hasattr(oft_model, "train"):
                oft_model.train()
        return {
            "forward_inputs": _detach_forward_inputs_to_cpu(forward_inputs),
            "token_rows": [
                [_coerce_int_token(token) for token in example.tokens]
                for example in examples
            ],
            "old_logprob_rows": [
                [float(value) for value in (getattr(example, "logprobs", None) or [])]
                for example in examples
            ],
            "example_count": len(examples),
            "token_length": next(iter(token_lengths)),
            "temperature": float(self.temperature),
            "model_loader": self.model_loader,
        }

    def action_token_logprobs_from_rlinf_native_forward_inputs(
        self,
        cache: dict[str, Any],
    ) -> list[Any]:
        """Score cached RLinf OpenVLA-OFT forward inputs under current weights."""

        if self.processor is None or self.model is None:
            self.load()
        assert self.model is not None
        oft_model = _openvla_oft_action_model(self.model)
        if (
            not self._rlinf_native_loader
            or oft_model is None
            or not _openvla_oft_supports_rlinf_rollout_style(oft_model)
        ):
            raise RuntimeError(
                "Fixed forward-input scorer requires RLinf native OpenVLA-OFT policy"
            )
        try:
            device = next(oft_model.parameters()).device
        except StopIteration:
            device = self.device
        forward_inputs = _forward_inputs_to_device(
            dict(cache["forward_inputs"]),
            device=device,
        )
        token_rows = cache.get("token_rows")
        if not isinstance(token_rows, list) or not token_rows:
            raise ValueError("Fixed forward-input cache is missing token_rows")
        token_ids = _rlinf_native_token_ids_tensor(
            oft_model,
            token_rows=token_rows,
            device=device,
        )
        model_training = bool(getattr(self.model, "training", False))
        oft_training = bool(getattr(oft_model, "training", False))
        try:
            if hasattr(self.model, "eval"):
                self.model.eval()
            if hasattr(oft_model, "eval"):
                oft_model.eval()
            logprobs = _rlinf_predict_style_action_token_logprobs_from_forward_inputs(
                oft_model,
                forward_inputs=forward_inputs,
                token_ids=token_ids,
                temperature=float(cache.get("temperature", self.temperature)),
            )
        finally:
            if model_training and hasattr(self.model, "train"):
                self.model.train()
            if oft_training and hasattr(oft_model, "train"):
                oft_model.train()
        return [logprobs[index] for index in range(logprobs.shape[0])]

    def continuous_action_logprobs(self, examples: list[Any]) -> list[Any]:
        """Recompute logprobs for OpenVLA actions emitted as continuous actions.

        OpenVLA-OFT samples action-bin tokens and decodes them into continuous
        robot actions. When ``action_output_kind="continuous"`` is used, the
        executed action is continuous, but the stochastic policy distribution is
        still the binned categorical action-token distribution. This method
        reconstructs action-token examples from the continuous-action records so
        GRPO/GSPO optimizes the same sampled bins that produced the executed
        action.
        """

        token_examples: list[ActionTokenExample] = []
        for example in examples:
            raw_action = getattr(example, "raw_action", None)
            if not isinstance(raw_action, dict):
                raw_action = {}
            metadata = getattr(example, "metadata", None)
            action_metadata = (
                metadata.get("action_metadata", {})
                if isinstance(metadata, dict)
                else {}
            )
            distribution = (
                action_metadata.get("policy_distribution", {})
                if isinstance(action_metadata, dict)
                else {}
            )
            tokens = (
                raw_action.get("tokens")
                or distribution.get("token_ids")
                or distribution.get("tokens")
            )
            if tokens is None:
                raise ValueError(
                    "OpenVLAPolicy continuous_action_logprobs requires raw_action['tokens'] "
                    "or action_metadata.policy_distribution.token_ids."
                )
            prompt = raw_action.get("prompt") or _prompt_from_task(
                getattr(example, "task", ""),
                template=self.prompt_template,
                lowercase_instruction=self.lowercase_instruction,
            )
            token_examples.append(
                ActionTokenExample(
                    task=getattr(example, "task", ""),
                    trajectory_index=int(getattr(example, "trajectory_index", 0)),
                    action_index=int(getattr(example, "action_index", 0)),
                    step=int(getattr(example, "step", 0)),
                    tokens=[int(token) for token in tokens],
                    reward=float(getattr(example, "reward", 0.0)),
                    prompt=prompt,
                    observation=getattr(example, "observation", None),
                    decoded_action=getattr(example, "action", None),
                    logprobs=getattr(example, "logprobs", None),
                    metadata={
                        "source": "openvla_continuous_action",
                        "action_metadata": make_json_safe(action_metadata),
                    },
                )
            )
        return self.action_token_logprobs(token_examples)

    def _batched_action_token_logprobs_for_examples(
        self, examples: list[Any]
    ) -> list[Any] | None:
        """Fast path for action-level GRPO microbatches.

        The backend owns the microbatch size. When it passes multiple action
        examples at once, we can run one OpenVLA-OFT forward pass for the whole
        image+prompt batch instead of one forward per action. This keeps the RL
        objective unchanged while avoiding the pathological single-example
        training loop that makes full LIBERO groups impractically slow.
        """

        if not examples:
            return []
        forward_input_overrides = [
            _transient_rlinf_forward_inputs_override_from_example(example)
            for example in examples
        ]
        has_forward_input_overrides = any(
            item is not None for item in forward_input_overrides
        )
        if has_forward_input_overrides and any(
            item is None for item in forward_input_overrides
        ):
            if self.strict_batched_logprobs:
                raise ValueError(
                    "Batched OpenVLA-OFT action-token logprob recomputation "
                    "requires forward-input overrides for every example when "
                    "any override is present."
                )
            return None
        if not has_forward_input_overrides and any(
            getattr(example, "observation", None) is None for example in examples
        ):
            if self.strict_batched_logprobs:
                raise ValueError(
                    "Batched OpenVLA-OFT action-token logprob recomputation "
                    "requires observations for every example."
                )
            return None
        token_lengths = {
            len(getattr(example, "tokens", []) or []) for example in examples
        }
        if len(token_lengths) != 1:
            if self.strict_batched_logprobs:
                raise ValueError(
                    "Batched OpenVLA-OFT action-token logprob recomputation "
                    f"requires equal token lengths; got {sorted(token_lengths)}."
                )
            return None
        prompts = [
            getattr(example, "prompt", None)
            or _prompt_from_task(
                getattr(example, "task", ""),
                template=self.prompt_template,
                lowercase_instruction=self.lowercase_instruction,
            )
            for example in examples
        ]
        observations = [example.observation for example in examples]
        token_rows = [example.tokens for example in examples]
        env_obs_overrides = [
            _transient_rlinf_env_obs_override_from_example(example)
            for example in examples
        ]
        try:
            oft_model = _openvla_oft_action_model(self.model)
            if (
                self._rlinf_native_loader
                and oft_model is not None
                and _openvla_oft_supports_rlinf_rollout_style(oft_model)
            ):
                instructions = [
                    _rlinf_instruction_from_example(example, prompt=prompt)
                    for example, prompt in zip(examples, prompts, strict=True)
                ]
                return _rlinf_native_action_token_logprobs_batch(
                    oft_model,
                    observations=observations,
                    instructions=instructions,
                    token_rows=token_rows,
                    temperature=self.temperature,
                    num_images_in_input=_openvla_resolved_num_images(
                        self.model, self.num_images_in_input
                    ),
                    use_proprio=self.use_proprio,
                    env_obs_overrides=env_obs_overrides,
                    forward_input_overrides=forward_input_overrides,
                )
            inputs = _prepare_openvla_batch_inputs(
                self.processor,
                observations=observations,
                prompts=prompts,
                device=self.device,
                dtype=self.dtype,
                max_prompt_length=self.max_prompt_length,
                max_images=_openvla_resolved_num_images(
                    self.model, self.num_images_in_input
                ),
                use_proprio=self.use_proprio,
                prefer_tensor_image_processing=self.model_loader == "native",
            )
            resolved_unnorm_key = _resolve_unnorm_key(self.model, self.unnorm_key)
            return _action_token_logprobs_for_token_rows(
                self.model,
                inputs,
                token_rows=token_rows,
                unnorm_key=resolved_unnorm_key,
                temperature=self.temperature,
            )
        except Exception as exc:
            self._last_batched_action_token_logprob_error = {
                "batch_size": len(examples),
                "token_lengths": sorted(token_lengths),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            if self.strict_batched_logprobs:
                raise RuntimeError(
                    "Batched OpenVLA-OFT action-token logprob recomputation "
                    f"failed for batch_size={len(examples)}, "
                    f"token_lengths={sorted(token_lengths)}. Sequential fallback "
                    "is disabled because it can break rollout old/current "
                    "logprob alignment after policy updates."
                ) from exc
            # Some remote-code processors only support single examples. Falling
            # back preserves compatibility; strict GRPO/GSPO runs should enable
            # strict_batched_logprobs so this failure is surfaced immediately.
            return None

    def _batched_trajectory_action_token_logprobs_for_examples(
        self, examples: list[Any]
    ) -> list[Any] | None:
        """Recompute trajectory-level action-token logprobs with batched span forwards.

        Trajectory-level GRPO/GSPO uses one training example per rollout, but the
        VLA model still scores each action span from its own observation+prompt.
        The naive implementation loops over every trajectory and then every span,
        causing a full forward pass per action. Full LIBERO groups have enough
        spans that this effectively serializes rollout-logprob precomputation.

        This helper keeps the trajectory-level objective intact while flattening
        spans into action-level rows, batching equal-length rows through the
        existing OpenVLA-OFT fast path, and concatenating differentiable rows
        back into one tensor per trajectory.
        """

        if not examples:
            return []
        if not all(
            _is_trajectory_action_token_example(example) for example in examples
        ):
            return None

        flat_examples_by_length: dict[int, list[SimpleNamespace]] = {}
        owners_by_length: dict[int, list[tuple[int, int]]] = {}
        for example_index, example in enumerate(examples):
            spans = example.metadata.get("action_spans") or []
            observations = example.metadata.get("action_observations") or []
            prompts = example.metadata.get("prompts") or []
            if not spans:
                return None
            for span_index, span in enumerate(spans):
                token_start = int(span.get("token_start", 0))
                token_end = int(span.get("token_end", token_start))
                tokens = example.tokens[token_start:token_end]
                if not tokens:
                    return None
                observation = (
                    observations[span_index]
                    if span_index < len(observations)
                    else example.observation
                )
                if observation is None:
                    return None
                prompt = (
                    prompts[span_index] if span_index < len(prompts) else example.prompt
                )
                row = SimpleNamespace(
                    task=example.task,
                    prompt=prompt
                    or _prompt_from_task(
                        example.task,
                        template=self.prompt_template,
                        lowercase_instruction=self.lowercase_instruction,
                    ),
                    observation=observation,
                    tokens=tokens,
                    metadata={
                        "action_span": span,
                        "action_metadata": span.get("action_metadata")
                        if isinstance(span, dict)
                        else None,
                        TRANSIENT_RLINF_FORWARD_INPUTS_METADATA_KEY: (
                            span.get(TRANSIENT_RLINF_FORWARD_INPUTS_METADATA_KEY)
                            if isinstance(span, dict)
                            else None
                        ),
                    },
                )
                length = len(tokens)
                flat_examples_by_length.setdefault(length, []).append(row)
                owners_by_length.setdefault(length, []).append(
                    (example_index, span_index)
                )

        import torch

        parts_by_example: list[list[tuple[int, Any]]] = [[] for _ in examples]
        max_batch_size = self.logprob_batch_size
        for length, flat_examples in flat_examples_by_length.items():
            owners = owners_by_length[length]
            chunk_size = max_batch_size or len(flat_examples)
            for start in range(0, len(flat_examples), chunk_size):
                chunk = flat_examples[start : start + chunk_size]
                owner_chunk = owners[start : start + chunk_size]
                rows = self._batched_action_token_logprobs_for_examples(chunk)
                if rows is None:
                    return None
                for (owner, span_index), row in zip(owner_chunk, rows, strict=True):
                    parts_by_example[owner].append((span_index, row.reshape(-1)))

        if any(not parts for parts in parts_by_example):
            return None
        return [
            torch.cat(
                [row for _span_index, row in sorted(parts, key=lambda item: item[0])],
                dim=0,
            )
            for parts in parts_by_example
        ]

    def _action_token_logprobs_for_example(self, example: Any) -> Any:
        if _is_trajectory_action_token_example(example):
            parts = []
            spans = example.metadata.get("action_spans") or []
            observations = example.metadata.get("action_observations") or []
            prompts = example.metadata.get("prompts") or []
            for index, span in enumerate(spans):
                token_start = int(span.get("token_start", 0))
                token_end = int(span.get("token_end", token_start))
                tokens = example.tokens[token_start:token_end]
                observation = (
                    observations[index]
                    if index < len(observations)
                    else example.observation
                )
                prompt = prompts[index] if index < len(prompts) else example.prompt
                env_obs_override = None
                forward_input_override = None
                if isinstance(span, dict):
                    action_metadata = span.get("action_metadata")
                    if isinstance(action_metadata, dict):
                        value = action_metadata.get(
                            TRANSIENT_RLINF_ENV_OBS_METADATA_KEY
                        )
                        if isinstance(value, dict):
                            env_obs_override = value
                        value = action_metadata.get(
                            TRANSIENT_RLINF_FORWARD_INPUTS_METADATA_KEY
                        )
                        if isinstance(value, dict):
                            forward_input_override = value
                    value = span.get(TRANSIENT_RLINF_FORWARD_INPUTS_METADATA_KEY)
                    if isinstance(value, dict):
                        forward_input_override = value
                parts.append(
                    self._action_token_logprobs_for_observation_prompt(
                        observation=observation,
                        prompt=prompt
                        or _prompt_from_task(
                            example.task,
                            template=self.prompt_template,
                            lowercase_instruction=self.lowercase_instruction,
                        ),
                        tokens=tokens,
                        env_obs_override=env_obs_override,
                        forward_input_override=forward_input_override,
                    )
                )
            if not parts:
                raise ValueError(
                    "trajectory action-token example contains no action spans"
                )
            import torch

            return torch.cat([part.reshape(-1) for part in parts], dim=0)

        return self._action_token_logprobs_for_observation_prompt(
            observation=example.observation,
            prompt=example.prompt
            or _prompt_from_task(
                example.task,
                template=self.prompt_template,
                lowercase_instruction=self.lowercase_instruction,
            ),
            tokens=example.tokens,
            env_obs_override=_transient_rlinf_env_obs_override_from_example(example),
            forward_input_override=_transient_rlinf_forward_inputs_override_from_example(
                example
            ),
        )

    def _action_token_logprobs_for_observation_prompt(
        self,
        *,
        observation: Observation | None,
        prompt: str,
        tokens: list[Any],
        env_obs_override: dict[str, Any] | None = None,
        forward_input_override: dict[str, Any] | None = None,
    ) -> Any:
        if observation is None and forward_input_override is None:
            raise ValueError(
                "OpenVLAPolicy action-token logprob recomputation requires observations"
            )
        assert self.processor is not None
        assert self.model is not None
        oft_model = _openvla_oft_action_model(self.model)
        if (
            self._rlinf_native_loader
            and oft_model is not None
            and _openvla_oft_supports_rlinf_rollout_style(oft_model)
        ):
            return _rlinf_native_action_token_logprobs_batch(
                oft_model,
                observations=[observation],
                instructions=[_rlinf_instruction_from_prompt(prompt)],
                token_rows=[tokens],
                temperature=self.temperature,
                num_images_in_input=_openvla_resolved_num_images(
                    self.model, self.num_images_in_input
                ),
                use_proprio=self.use_proprio,
                env_obs_overrides=[env_obs_override],
                forward_input_overrides=[forward_input_override],
            )[0]
        inputs = _prepare_openvla_inputs(
            self.processor,
            observation=observation,
            prompt=prompt,
            device=self.device,
            dtype=self.dtype,
            max_prompt_length=self.max_prompt_length,
            max_images=_openvla_resolved_num_images(
                self.model, self.num_images_in_input
            ),
            use_proprio=self.use_proprio,
            prefer_tensor_image_processing=self.model_loader == "native",
        )
        resolved_unnorm_key = _resolve_unnorm_key(self.model, self.unnorm_key)
        return _action_token_logprobs_for_tokens(
            self.model,
            inputs,
            tokens=tokens,
            unnorm_key=resolved_unnorm_key,
            temperature=self.temperature,
        )

    def save_checkpoint(self, path: str) -> Any:
        if self.model is None:
            raise RuntimeError(
                "Cannot save an OpenVLA checkpoint before loading the policy"
            )
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        is_peft_adapter = _model_has_peft_adapter(self.model)
        if self.model is not None and hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(output_path)
        if self.processor is not None and hasattr(self.processor, "save_pretrained"):
            self.processor.save_pretrained(output_path)
        checkpoint_type = (
            "openvla-peft-adapter" if is_peft_adapter else "openvla-policy"
        )
        manifest = {
            "schema_version": 1,
            "type": checkpoint_type,
            "base_model_id": self.model_id,
            "base_model_revision": self.revision,
            "model_loader": self.model_loader,
        }
        (output_path / "art_embodied_checkpoint.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result = {"path": str(output_path), **manifest}
        if is_peft_adapter:
            result["peft_adapter_path"] = str(output_path)
        return result

    def load_checkpoint(self, ref: Any) -> None:
        if isinstance(ref, (str, Path)):
            ref = {"path": str(ref)}
        if not isinstance(ref, dict):
            raise TypeError("OpenVLA checkpoint reference must be a path or mapping")

        adapter_path = ref.get("peft_adapter_path")
        checkpoint_path = ref.get("path")
        if adapter_path is None and checkpoint_path is not None:
            candidate = Path(str(checkpoint_path)).expanduser()
            if _is_peft_adapter_checkpoint(candidate):
                adapter_path = str(candidate)

        if adapter_path is not None:
            self.peft_adapter_path = str(adapter_path)
            adapter_checkpoint = Path(self.peft_adapter_path).expanduser()
            adapter_exists = adapter_checkpoint.exists()
            if adapter_exists:
                _validate_or_adopt_adapter_base_contract(self, adapter_checkpoint)
            if self.model is not None and adapter_exists:
                if _model_has_peft_adapter(self.model):
                    refresh_openvla_peft_adapter(self, self.peft_adapter_path)
                else:
                    self.model = _attach_peft_adapter(
                        self.model,
                        self.peft_adapter_path,
                        device=self.device,
                    )
                    self.model.eval()
            elif self.model is None:
                self.load()
        elif checkpoint_path is not None:
            self.model_id = str(checkpoint_path)
            self.revision = None
            self.processor = None
            self.model = None
            self.load()
        else:
            raise ValueError(
                "OpenVLA checkpoint reference requires 'path' or 'peft_adapter_path'"
            )

    def parameters(self, *args: Any, **kwargs: Any) -> Any:
        if self.model is None:
            return iter(())
        return self.model.parameters(*args, **kwargs)

    def named_parameters(self, *args: Any, **kwargs: Any) -> Any:
        if self.model is None:
            return iter(())
        return self.model.named_parameters(*args, **kwargs)

    def named_modules(self, *args: Any, **kwargs: Any) -> Any:
        if self.model is None:
            return iter(())
        return self.model.named_modules(*args, **kwargs)

    def train(self, mode: bool = True) -> "OpenVLAPolicy":
        if self.model is not None and hasattr(self.model, "train"):
            self.model.train(mode)
        return self

    def eval(self) -> "OpenVLAPolicy":
        return self.train(False)

    def set_generation(self, *, do_sample: bool, temperature: float) -> None:
        """Update sampling controls without rebuilding the loaded policy."""

        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        if self.model is None:
            return {}
        return self.model.state_dict(*args, **kwargs)


def _model_has_peft_adapter(model: Any | None) -> bool:
    if model is None:
        return False
    return any(
        hasattr(candidate, "peft_config")
        for candidate in _model_unwrap_candidates(model)
    )


def _is_peft_adapter_checkpoint(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "adapter_config.json").is_file():
        return True
    manifest = _read_art_embodied_checkpoint_manifest(path)
    return manifest.get("type") == "openvla-peft-adapter"


def _read_art_embodied_checkpoint_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / "art_embodied_checkpoint.json"
    if not manifest_path.is_file():
        return {}
    try:
        manifest_size = manifest_path.stat().st_size
        if manifest_size > 64 * 1024:
            raise ValueError(
                "OpenVLA checkpoint manifest exceeds the 64 KiB safety limit: "
                f"{manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(
            f"Could not read OpenVLA checkpoint manifest: {manifest_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid OpenVLA checkpoint manifest JSON: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError(
            f"OpenVLA checkpoint manifest must be a JSON object: {manifest_path}"
        )
    required = {
        "schema_version",
        "type",
        "base_model_id",
        "base_model_revision",
        "model_loader",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(
            "OpenVLA checkpoint manifest is missing required fields "
            f"{missing}: {manifest_path}"
        )
    if manifest["schema_version"] != 1:
        raise ValueError(
            "Unsupported OpenVLA checkpoint manifest schema_version "
            f"{manifest['schema_version']!r}: {manifest_path}"
        )
    if manifest["type"] not in {"openvla-peft-adapter", "openvla-policy"}:
        raise ValueError(
            f"Unsupported OpenVLA checkpoint type {manifest['type']!r}: {manifest_path}"
        )
    if manifest["model_loader"] not in {"native", "transformers", "rlinf"}:
        raise ValueError(
            "Unsupported OpenVLA checkpoint model_loader "
            f"{manifest['model_loader']!r}: {manifest_path}"
        )
    return manifest


def _validate_or_adopt_adapter_base_contract(
    policy: OpenVLAPolicy,
    adapter_path: Path,
) -> None:
    manifest = _read_art_embodied_checkpoint_manifest(adapter_path)
    if not manifest:
        return
    manifest_model_id = manifest.get("base_model_id")
    if manifest_model_id is not None and str(manifest_model_id) != policy.model_id:
        raise ValueError(
            "OpenVLA adapter base model mismatch: "
            f"checkpoint={manifest_model_id!r}, policy={policy.model_id!r}"
        )
    manifest_revision = manifest.get("base_model_revision")
    if manifest_revision is None:
        return
    manifest_revision = str(manifest_revision)
    if policy.model is None and policy.revision is None:
        policy.revision = manifest_revision
        return
    if policy.revision != manifest_revision:
        raise ValueError(
            "OpenVLA adapter base revision mismatch: "
            f"checkpoint={manifest_revision!r}, policy={policy.revision!r}"
        )


def refresh_openvla_peft_adapter(
    policy: OpenVLAPolicy,
    path: str | Path,
) -> dict[str, Any]:
    """Refresh a loaded OpenVLA LoRA adapter without reloading the base VLA."""

    adapter_path = Path(path).expanduser()
    if not adapter_path.exists():
        raise FileNotFoundError(f"PEFT adapter path does not exist: {adapter_path}")
    _validate_or_adopt_adapter_base_contract(policy, adapter_path)
    model = policy.model
    if model is None:
        raise RuntimeError(
            "Cannot refresh an OpenVLA adapter before loading the policy"
        )
    peft_model = next(
        (
            candidate
            for candidate in _model_unwrap_candidates(model)
            if hasattr(candidate, "peft_config")
        ),
        None,
    )
    if peft_model is None:
        raise RuntimeError("OpenVLA adapter refresh requires a loaded PEFT model")

    from art_embodied.vla_trainable import (
        _disable_incompatible_optional_peft_dispatches,
        _shield_broken_transformer_engine_for_peft,
    )

    _shield_broken_transformer_engine_for_peft()
    _disable_incompatible_optional_peft_dispatches()
    from peft import set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights

    active_adapter = getattr(peft_model, "active_adapter", "default")
    if isinstance(active_adapter, (list, tuple)):
        if len(active_adapter) != 1:
            raise RuntimeError(
                "OpenVLA adapter refresh requires exactly one active adapter, "
                f"found {active_adapter!r}"
            )
        active_adapter = active_adapter[0]
    state = load_peft_weights(str(adapter_path), device="cpu")
    load_result = set_peft_model_state_dict(
        peft_model,
        state,
        adapter_name=str(active_adapter),
    )
    policy.peft_adapter_path = str(adapter_path)
    model.to(policy.device)
    if callable(getattr(model, "eval", None)):
        model.eval()
    return {
        "adapter_path": str(adapter_path),
        "base_model_reloaded": False,
        "missing_keys": len(list(getattr(load_result, "missing_keys", ()) or ())),
        "unexpected_keys": len(list(getattr(load_result, "unexpected_keys", ()) or ())),
    }


def _attach_processor_to_openvla_oft_model(model: Any, processor: Any) -> None:
    """Expose the processor on OpenVLA-OFT remote-code models when needed.

    RLinf's OpenVLA-OFT wrapper stores the processor on the model and its
    action-prediction embedding path reads ``self.processor.tokenizer``. Hugging
    Face remote-code checkpoints do not always attach it automatically, so ART
    does it explicitly before extracting RLinf-compatible action logits.
    """

    for candidate in _model_unwrap_candidates(model):
        setter = getattr(candidate, "set_processor", None)
        try:
            if callable(setter):
                setter(processor)
            else:
                setattr(candidate, "processor", processor)
        except Exception:
            continue


def _attach_peft_adapter(model: Any, adapter_path: str, *, device: str) -> Any:
    """Attach a PEFT adapter checkpoint to an OpenVLA/OFT base model."""

    adapter = Path(adapter_path).expanduser()
    if not adapter.exists():
        raise FileNotFoundError(f"PEFT adapter path does not exist: {adapter}")
    try:
        from art_embodied.vla_trainable import (
            _disable_incompatible_optional_peft_dispatches,
            _shield_broken_transformer_engine_for_peft,
        )

        _shield_broken_transformer_engine_for_peft()
        from peft import PeftModel

        _disable_incompatible_optional_peft_dispatches()
    except Exception as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError(
            "peft is required to load an OpenVLA PEFT adapter checkpoint"
        ) from exc
    loaded = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    return loaded.to(device)


def _merge_external_dataset_statistics(
    model: Any,
    model_id: str,
    *,
    explicit_path: Path | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    """Merge OpenVLA-OFT ``dataset_statistics.json`` into model norm stats.

    Some OpenVLA-OFT checkpoints store task-specific LIBERO action statistics in
    a sibling file instead of embedding them in ``config.json``.  The remote-code
    model then cannot resolve keys such as ``libero_object_no_noops`` unless the
    application explicitly merges that file.
    """

    path = (
        explicit_path
        if explicit_path is not None
        else _dataset_statistics_path(model_id, revision=revision)
    )
    if path is None:
        return {"loaded": False, "reason": "dataset_statistics_not_found"}
    try:
        stats = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - optional compatibility file.
        return {"loaded": False, "path": str(path), "error": str(exc)}
    if not isinstance(stats, dict):
        return {
            "loaded": False,
            "path": str(path),
            "error": "dataset_statistics_json_is_not_object",
        }

    merged_keys: list[str] = []
    for candidate in _model_unwrap_candidates(model):
        existing = getattr(candidate, "norm_stats", None)
        if isinstance(existing, dict):
            before = set(existing)
            existing.update(stats)
            merged_keys.extend(sorted(set(existing) - before))
        config = getattr(candidate, "config", None)
        config_stats = getattr(config, "norm_stats", None)
        if isinstance(config_stats, dict):
            before = set(config_stats)
            config_stats.update(stats)
            merged_keys.extend(sorted(set(config_stats) - before))
    return {
        "loaded": True,
        "path": str(path),
        "keys": sorted(stats.keys()),
        "merged_keys": sorted(set(merged_keys)),
    }


def _dataset_statistics_path(
    model_id: str,
    *,
    revision: str | None = None,
) -> Path | None:
    local = Path(model_id).expanduser()
    if local.exists():
        path = local / "dataset_statistics.json"
        return path if path.exists() else None
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            model_id,
            "dataset_statistics.json",
            revision=revision,
        )
        return Path(downloaded)
    except Exception:
        return None


def _primary_action_from_predict_action_output(
    output: Any,
) -> tuple[Any, dict[str, Any]]:
    """Split OpenVLA/OFT ``predict_action`` output into action and metadata.

    OpenVLA-OFT checkpoints may return ``(actions, action_hidden_states)``. The
    hidden states are useful for specialized trainers, but they are not the robot
    action and can explode trajectory logs if serialized as ``Action.decoded``.
    """

    if isinstance(output, tuple) and output:
        return output[0], {
            "predict_action_output_type": "tuple",
            "predict_action_extra_outputs": len(output) - 1,
        }
    if (
        isinstance(output, list)
        and output
        and _looks_like_auxiliary_predict_output(output)
    ):
        return output[0], {
            "predict_action_output_type": "list",
            "predict_action_extra_outputs": len(output) - 1,
        }
    return output, {"predict_action_output_type": type(output).__name__}


def _looks_like_auxiliary_predict_output(output: list[Any]) -> bool:
    if len(output) < 2:
        return False
    first = output[0]
    second = output[1]
    if hasattr(second, "shape"):
        return True
    if isinstance(second, (list, tuple)) and len(second) > 32:
        return True
    if hasattr(first, "shape"):
        shape = tuple(getattr(first, "shape", ()))
        return len(shape) <= 3
    return False


def _instruction_from_context(context: dict[str, Any]) -> str:
    scenario = context.get("scenario") or {}
    if isinstance(scenario, dict):
        task = scenario.get("task")
        if task:
            return str(task)
    return str(context.get("task") or "complete the task")


def _available_norm_stat_keys(model: Any) -> list[str]:
    """Return OpenVLA action-normalization keys exposed by a model."""

    keys: list[str] = []
    containers = [
        getattr(model, "norm_stats", None),
        getattr(getattr(model, "config", None), "norm_stats", None),
        getattr(getattr(model, "config", None), "action_norm_stats", None),
    ]
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in container:
            if isinstance(key, str) and key not in keys:
                keys.append(key)
    return keys


def _resolve_unnorm_key(model: Any, unnorm_key: str | None) -> str | None:
    """Resolve ``unnorm_key='auto'`` from model metadata.

    OpenVLA checkpoints are often trained with different action-normalization
    statistics (Bridge, LIBERO, RT-X, etc.). A hard-coded default makes prior-art
    checkpoints fail for reasons unrelated to ART-Embodied. ``auto`` keeps the
    condition explicit while choosing a deterministic key from the model.
    """

    if not isinstance(unnorm_key, str) or unnorm_key.lower() != "auto":
        return unnorm_key
    keys = _available_norm_stat_keys(model)
    if not keys:
        return None

    lower_to_key = {key.lower(): key for key in keys}
    preferred_exact = (
        "libero_90",
        "libero",
        "bridge_orig",
        "bridge",
        "rt_1_x",
        "rtx",
    )
    for preferred in preferred_exact:
        if preferred in lower_to_key:
            return lower_to_key[preferred]

    preferred_fragments = ("libero", "bridge", "rt")
    for fragment in preferred_fragments:
        for key in keys:
            if fragment in key.lower():
                return key

    return sorted(keys)[0]


def _observation_to_pil_image(observation: Observation):
    from PIL import Image

    value = observation.value
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, str):
        return Image.open(value).convert("RGB")
    if isinstance(value, dict):
        for key in ("image", "rgb", "frame", "path", "agentview_image", "main_image"):
            if key in value:
                return _value_to_image(value[key])
    if observation.media:
        uri = observation.media[0].uri
        if uri.startswith("file://"):
            uri = uri[len("file://") :]
        if Path(uri).exists():
            return Image.open(uri).convert("RGB")
    return _value_to_image(value)


def _observation_to_openvla_images(observation: Observation) -> list[Any]:
    """Return the ordered image streams for an OpenVLA-family observation.

    Vanilla OpenVLA consumes a single image, while OpenVLA-OFT LIBERO models can
    consume a primary camera plus wrist camera.  Keep the ART observation schema
    generic by recognizing common image payload keys rather than baking LIBERO
    into the policy adapter.
    """

    value = observation.value
    if isinstance(value, dict):
        if "images" in value:
            images = _image_payload_to_images(value["images"])
            if images:
                return images
        images: list[Any] = []
        primary_keys = (
            "image",
            "rgb",
            "frame",
            "path",
            "agentview_image",
            "main_image",
        )
        for key in primary_keys:
            if key in value:
                images.append(value[key])
                break
        for key in (
            "wrist_image",
            "wrist",
            "robot0_eye_in_hand_image",
            "eye_in_hand_image",
        ):
            if key in value:
                images.append(value[key])
                break
        if "wrist_images" in value:
            images.extend(_image_payload_to_images(value["wrist_images"]))
        if images:
            return [_value_to_image(image) for image in images]
    if len(observation.media) > 1:
        from PIL import Image

        media_images = []
        for media in observation.media:
            uri = media.uri
            if uri.startswith("file://"):
                uri = uri[len("file://") :]
            if Path(uri).exists():
                media_images.append(Image.open(uri).convert("RGB"))
        if media_images:
            return media_images
    return [_observation_to_pil_image(observation)]


def _image_payload_to_images(payload: Any) -> list[Any]:
    if isinstance(payload, dict):
        ordered_keys = (
            "image",
            "rgb",
            "frame",
            "path",
            "agentview_image",
            "main_image",
            "wrist_image",
            "robot0_eye_in_hand_image",
            "eye_in_hand_image",
        )
        images = [payload[key] for key in ordered_keys if key in payload]
        if images:
            return images
        return [payload[key] for key in sorted(payload)]
    if isinstance(payload, (list, tuple)):
        return list(payload)
    return [payload]


def _limit_openvla_image_count(images: list[Any], max_images: int | None) -> list[Any]:
    if max_images is None:
        return images
    resolved = max(1, int(max_images))
    if len(images) < resolved:
        raise ValueError(
            f"OpenVLA model expects {resolved} image stream(s), but observation provides {len(images)}"
        )
    return images[:resolved]


def _openvla_resolved_num_images(
    model: Any, configured_num_images: int | None = None
) -> int | None:
    if configured_num_images is not None:
        return max(1, int(configured_num_images))
    return _openvla_expected_num_images(model)


def _openvla_expected_num_images(model: Any) -> int | None:
    oft_model = _openvla_oft_action_model(model)
    target = oft_model or model
    vision_backbone = getattr(target, "vision_backbone", None)
    getter = getattr(vision_backbone, "get_num_images_in_input", None)
    if not callable(getter):
        return None
    try:
        return max(1, int(getter()))
    except Exception:
        return None


def _value_to_image(value: Any):
    from PIL import Image

    if isinstance(value, Image.Image):
        return value
    if isinstance(value, str):
        return Image.open(value).convert("RGB")
    try:
        import numpy as np

        array = np.asarray(value)
        if array.dtype != "uint8":
            array = array.clip(0, 255).astype("uint8")
        return Image.fromarray(array).convert("RGB")
    except Exception as exc:
        raise ValueError(
            "OpenVLAPolicy requires an image observation, image path, or media ref"
        ) from exc


def _predict_action_with_tokens(
    model: Any,
    inputs: Any,
    *,
    unnorm_key: str | None,
    do_sample: bool,
    temperature: float = 1.0,
) -> tuple[Any, list[int], list[float]]:
    oft_model = _openvla_oft_action_model(model)
    if oft_model is not None:
        return _predict_oft_action_with_tokens(
            oft_model,
            inputs,
            unnorm_key=unnorm_key,
            do_sample=do_sample,
            temperature=temperature,
        )
    return _predict_vanilla_action_with_tokens(
        model,
        inputs,
        unnorm_key=unnorm_key,
        do_sample=do_sample,
        temperature=temperature,
    )


def _is_trajectory_action_token_example(example: Any) -> bool:
    metadata = getattr(example, "metadata", None)
    return isinstance(metadata, dict) and metadata.get("training_unit") == "trajectory"


def _prompt_from_task(
    task: Any,
    *,
    template: str | None = None,
    lowercase_instruction: bool = True,
) -> str:
    instruction = str(task)
    if lowercase_instruction:
        instruction = instruction.lower()
    if template:
        rendered = template.format(instruction=instruction)
        # OpenVLA-OFT was trained with a literal space token after ``Out:``.
        # YAML block scalars intentionally strip trailing whitespace, so restore
        # the model-level token contract here instead of asking users to encode
        # invisible experimental state in configuration files.
        return rendered if rendered.endswith(" ") else rendered + " "
    return f"In: What action should the robot take to {instruction}?\nOut: "


def _rlinf_instruction_from_example(example: Any, *, prompt: str | None) -> str:
    task = str(getattr(example, "task", "") or "").strip()
    if task:
        return task
    return _rlinf_instruction_from_prompt(prompt)


def _rlinf_instruction_from_prompt(prompt: str | None) -> str:
    text = str(prompt or "").strip()
    prefix = "In: What action should the robot take to "
    suffix = "\nOut:"
    if text.startswith(prefix) and text.endswith(suffix):
        return text[len(prefix) : -len(suffix)].strip()
    return text


def _prepare_openvla_inputs(
    processor: Any,
    *,
    observation: Observation,
    prompt: str,
    device: str,
    dtype: str,
    max_prompt_length: int | None = None,
    max_images: int | None = None,
    use_proprio: bool | None = None,
    prefer_tensor_image_processing: bool = False,
) -> Any:
    images = _limit_openvla_image_count(
        _observation_to_openvla_images(observation), max_images
    )
    processor_kwargs: dict[str, Any] = {}
    if max_prompt_length is not None:
        processor_kwargs.update(
            {"padding": "max_length", "max_length": int(max_prompt_length)}
        )
    proprio_state = _observation_proprio_state(observation)
    processor_uses_proprio = _processor_has_proprio_states_argument(processor)
    if proprio_state is None and (
        processor_uses_proprio or prefer_tensor_image_processing
    ):
        import numpy as np

        # RLinf/OpenVLA-OFT processors keep proprio_states in the call signature
        # even when the model config has use_proprio=False.  A zero vector keeps
        # vanilla logprob recomputation callable; real RLinf LIBERO observations
        # normally provide the compact robot state explicitly.
        proprio_state = np.zeros((8,), dtype=np.float32)
    if prefer_tensor_image_processing:
        inputs = _try_prepare_rlinf_openvla_oft_tensor_inputs(
            processor,
            prompts=[prompt],
            images=[images],
            proprio_states=[proprio_state] if proprio_state is not None else None,
            processor_kwargs=processor_kwargs,
        )
        if inputs is None:
            raise RuntimeError(
                "ART-native OpenVLA-OFT requires tensor image preprocessing, "
                "but the checkpoint processor does not expose the required transform contract"
            )
        if hasattr(inputs, "to"):
            import torch

            inputs = inputs.to(device, dtype=getattr(torch, dtype))
        return inputs
    if proprio_state is not None and (
        use_proprio is not False or processor_uses_proprio
    ):
        inputs = _try_prepare_openvla_oft_inputs_with_proprio(
            processor,
            prompts=[prompt],
            images=[images],
            proprio_states=[proprio_state],
            processor_kwargs=processor_kwargs,
        )
        if inputs is not None:
            _apply_rlinf_openvla_oft_token_layout(inputs, processor)
            if hasattr(inputs, "to"):
                import torch

                inputs = inputs.to(device, dtype=getattr(torch, dtype))
            return inputs
    inputs = processor(prompt, images[0], **processor_kwargs)
    _apply_rlinf_openvla_oft_token_layout(inputs, processor)
    if hasattr(inputs, "to"):
        import torch

        inputs = inputs.to(device, dtype=getattr(torch, dtype))
    return inputs


def _prepare_openvla_batch_inputs(
    processor: Any,
    *,
    observations: list[Observation],
    prompts: list[str],
    device: str,
    dtype: str,
    max_prompt_length: int | None = None,
    max_images: int | None = None,
    use_proprio: bool | None = None,
    prefer_tensor_image_processing: bool = False,
) -> Any:
    image_groups = [
        _limit_openvla_image_count(
            _observation_to_openvla_images(observation), max_images
        )
        for observation in observations
    ]
    primary_images = [images[0] for images in image_groups]
    processor_kwargs: dict[str, Any] = {}
    if max_prompt_length is not None:
        processor_kwargs.update(
            {"padding": "max_length", "max_length": int(max_prompt_length)}
        )
    proprio_states = [
        _observation_proprio_state(observation) for observation in observations
    ]
    processor_uses_proprio = _processor_has_proprio_states_argument(processor)
    if (processor_uses_proprio or prefer_tensor_image_processing) and any(
        state is None for state in proprio_states
    ):
        import numpy as np

        proprio_states = [
            np.zeros((8,), dtype=np.float32) if state is None else state
            for state in proprio_states
        ]
    if prefer_tensor_image_processing:
        inputs = _try_prepare_rlinf_openvla_oft_tensor_inputs(
            processor,
            prompts=prompts,
            images=image_groups,
            proprio_states=[state for state in proprio_states if state is not None],
            processor_kwargs=processor_kwargs,
        )
        if inputs is None:
            raise RuntimeError(
                "ART-native OpenVLA-OFT requires tensor image preprocessing, "
                "but the checkpoint processor does not expose the required transform contract"
            )
        if hasattr(inputs, "to"):
            import torch

            inputs = inputs.to(device, dtype=getattr(torch, dtype))
        return inputs
    if (use_proprio is not False or processor_uses_proprio) and all(
        state is not None for state in proprio_states
    ):
        inputs = _try_prepare_openvla_oft_inputs_with_proprio(
            processor,
            prompts=prompts,
            images=image_groups,
            proprio_states=[state for state in proprio_states if state is not None],
            processor_kwargs=processor_kwargs,
        )
        if inputs is not None:
            _apply_rlinf_openvla_oft_token_layout(inputs, processor)
            if hasattr(inputs, "to"):
                import torch

                inputs = inputs.to(device, dtype=getattr(torch, dtype))
            return inputs
    inputs = processor(prompts, primary_images, **processor_kwargs)
    _apply_rlinf_openvla_oft_token_layout(inputs, processor)
    if hasattr(inputs, "to"):
        import torch

        inputs = inputs.to(device, dtype=getattr(torch, dtype))
    return inputs


def _processor_has_proprio_states_argument(processor: Any) -> bool:
    try:
        import inspect

        signature = inspect.signature(processor.__call__)
    except (TypeError, ValueError):
        return False
    return "proprio_states" in signature.parameters


def _observation_proprio_state(observation: Observation) -> Any | None:
    """Return optional robot proprioception stored with an observation.

    RLinf's OpenVLA-OFT LIBERO path feeds both the camera image and a compact
    robot state vector into the processor. ART observations stay generic, so the
    state can be carried either in the value payload or in metadata.
    """

    value = observation.value
    if isinstance(value, dict):
        for key in ("proprio_state", "proprio_states", "state", "robot_state"):
            if key in value:
                return value[key]
    for key in ("proprio_state", "proprio_states", "state", "robot_state"):
        if key in observation.metadata:
            return observation.metadata[key]
    return None


def _try_prepare_openvla_oft_inputs_with_proprio(
    processor: Any,
    *,
    prompts: list[str],
    images: list[Any],
    proprio_states: list[Any],
    processor_kwargs: dict[str, Any],
) -> Any | None:
    """Use the OpenVLA-OFT/RLinf processor contract when proprio is available.

    Vanilla OpenVLA processors generally accept ``processor(prompt, image)``.
    OpenVLA-OFT as used by RLinf accepts keyword inputs with
    ``text``, ``images={"images": ...}``, and ``proprio_states``.  Try the richer
    path only when a state vector is present and fall back cleanly for vanilla
    processors or tests.
    """

    inputs = _try_prepare_rlinf_openvla_oft_tensor_inputs(
        processor,
        prompts=prompts,
        images=images,
        proprio_states=proprio_states,
        processor_kwargs=processor_kwargs,
    )
    if inputs is not None:
        return inputs

    try:
        import numpy as np
        import torch

        image_groups = _normalize_openvla_image_groups(images)
        image_batches = []
        for group in image_groups:
            image_tensors = []
            for image in group:
                array = np.asarray(image).copy()
                if array.dtype != np.uint8:
                    array = np.clip(array, 0, 255).astype(np.uint8)
                tensor = torch.as_tensor(array)
                if tensor.shape[-1] in (1, 3):
                    tensor = tensor.permute(2, 0, 1)
                image_tensors.append(tensor)
            image_batches.append(torch.stack(image_tensors, dim=0))
        # RLinf passes [B, N, C, H, W] under images["images"].
        image_batch = torch.stack(image_batches, dim=0)
        proprio_batch = torch.as_tensor(np.asarray(proprio_states), dtype=torch.float32)
        return processor(
            text=prompts,
            images={"images": image_batch},
            proprio_states=proprio_batch,
            **processor_kwargs,
        )
    except (TypeError, ValueError):
        return None


def _try_prepare_rlinf_openvla_oft_tensor_inputs(
    processor: Any,
    *,
    prompts: list[str],
    images: list[Any],
    proprio_states: list[Any] | None = None,
    processor_kwargs: dict[str, Any],
) -> Any | None:
    """Prepare OpenVLA-OFT inputs with RLinf's tensor image transform.

    The public OpenVLA-OFT Hugging Face processor is PIL-oriented. RLinf's
    reported LIBERO numbers use a tensor-based ``MultiInputPrismaticProcessor``
    that applies ``torchvision.transforms.functional`` directly to
    ``[B, N, C, H, W]`` image tensors. PIL and tensor bicubic resize are close
    but not identical; for manipulation policies that difference is large enough
    to invalidate prior-art parity.  Reproduce only that input contract here and
    leave simulator/task-specific logic in the benchmark layer.
    """

    image_processor = getattr(processor, "image_processor", None)
    tokenizer = getattr(processor, "tokenizer", None)
    if image_processor is None or tokenizer is None:
        return None
    required_attrs = (
        "input_sizes",
        "tvf_resize_params",
        "tvf_crop_params",
        "tvf_normalize_params",
    )
    if not all(hasattr(image_processor, attr) for attr in required_attrs):
        return None

    try:
        try:
            from transformers.image_processing_utils import BatchFeature
        except ImportError:
            BatchFeature = None

        pixel_values = _rlinf_openvla_oft_tensor_pixel_values(
            image_processor,
            images=images,
        )
        text_inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=processor_kwargs.get("padding", False),
            truncation=processor_kwargs.get("truncation", None),
            max_length=processor_kwargs.get("max_length", None),
        )
        data = {**text_inputs, "pixel_values": pixel_values}
        if proprio_states is not None:
            import numpy as np
            import torch

            data["proprio"] = torch.as_tensor(
                np.asarray(proprio_states), dtype=torch.float32
            )
        inputs = BatchFeature(data=data) if BatchFeature is not None else data
        _apply_rlinf_openvla_oft_token_layout(inputs, processor)
        return inputs
    except (AttributeError, TypeError, ValueError):
        return None


def _normalize_openvla_image_groups(images: list[Any]) -> list[list[Any]]:
    groups: list[list[Any]] = []
    for item in images:
        if isinstance(item, dict):
            group = _image_payload_to_images(item)
        elif isinstance(item, (list, tuple)):
            group = list(item)
        else:
            group = [item]
        if not group:
            raise ValueError("OpenVLA image groups must contain at least one image")
        groups.append(group)
    if not groups:
        raise ValueError(
            "OpenVLA image preprocessing requires at least one observation"
        )
    num_images = len(groups[0])
    if any(len(group) != num_images for group in groups):
        raise ValueError(
            "OpenVLA batch image groups must have the same number of images"
        )
    return groups


def _rlinf_openvla_oft_tensor_pixel_values(
    image_processor: Any, *, images: list[Any]
) -> Any:
    """Return pixel values matching RLinf's tensor Prismatic image processor."""

    import numpy as np
    from PIL import Image
    import torch
    import torchvision.transforms.functional as TVF

    image_groups = _normalize_openvla_image_groups(images)
    image_tensors = []
    for group in image_groups:
        group_tensors = []
        for image in group:
            if isinstance(image, Image.Image):
                array = np.asarray(image.convert("RGB")).copy()
            else:
                array = np.asarray(image).copy()
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            if array.ndim != 3:
                raise ValueError(
                    "OpenVLA-OFT tensor image preprocessing expects HWC or CHW RGB images"
                )
            tensor = torch.as_tensor(array)
            if tensor.shape[-1] in (1, 3):
                tensor = tensor.permute(2, 0, 1)
            elif tensor.shape[0] not in (1, 3):
                raise ValueError(
                    "OpenVLA-OFT tensor image preprocessing expects an RGB channel dimension"
                )
            group_tensors.append(tensor)
        image_tensors.append(torch.stack(group_tensors, dim=0))

    # RLinf's MultiInputPrismaticProcessor receives image streams as
    # [B, N, C, H, W], applies fused-backbone transforms to tensor images, and
    # returns [B, N, C * num_backbones, H, W].
    image_batch = torch.stack(image_tensors, dim=0)
    batch_size, num_images = image_batch.shape[:2]
    flat_images = image_batch.reshape(-1, *image_batch.shape[2:])

    transformed = []
    for idx in range(len(image_processor.input_sizes)):
        image_idx = TVF.resize(flat_images, **image_processor.tvf_resize_params[idx])
        image_idx = TVF.center_crop(image_idx, **image_processor.tvf_crop_params[idx])
        if isinstance(image_idx, Image.Image):
            image_idx = TVF.to_tensor(image_idx)
        image_idx = image_idx / 255.0
        image_idx = TVF.normalize(
            image_idx, **image_processor.tvf_normalize_params[idx]
        )
        transformed.append(image_idx)

    stacked = torch.cat(transformed, dim=1)
    return stacked.reshape(batch_size, num_images, *stacked.shape[1:])


def _configure_rlinf_openvla_oft_processor_if_present(processor: Any) -> None:
    """Match RLinf's OpenVLA-OFT tokenizer padding contract when possible.

    RLinf constructs the OpenVLA-OFT tokenizer with ``padding_side="left"`` and
    then rewrites the padded prompt so ``input_ids[:, 0]`` is BOS while the
    actual prompt still ends with the space token. The public Hugging Face
    processor defaults to right padding, which shifts the action-token logits
    away from the positions used by RLinf's reported LIBERO results.
    """

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"


def _apply_rlinf_openvla_oft_token_layout(inputs: Any, processor: Any) -> None:
    """Apply RLinf's max-prompt token layout in-place when tensors are present."""

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if pad_token_id is None or bos_token_id is None:
        return
    try:
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
    except Exception:
        return
    if not hasattr(input_ids, "ndim") or int(input_ids.ndim) != 2:
        return
    if not hasattr(attention_mask, "ndim") or int(attention_mask.ndim) != 2:
        return

    import torch

    pad_token_id = int(pad_token_id)
    bos_token_id = int(bos_token_id)
    nonpad_mask = attention_mask.ne(0)
    if not bool(nonpad_mask.any()):
        return
    first_nonpad = torch.argmax(nonpad_mask.to(dtype=torch.int64), dim=1, keepdim=True)
    first_tokens = input_ids.gather(1, first_nonpad)
    if not torch.all(first_tokens == bos_token_id):
        return

    input_ids.scatter_(1, first_nonpad, pad_token_id)
    attention_mask.scatter_(1, first_nonpad, 0)
    input_ids[:, 0] = bos_token_id
    attention_mask[:, 0] = 1


def _openvla_oft_pixel_values_for_model(inputs: Any) -> Any:
    """Return pixel values with RLinf/OpenVLA-OFT's model-facing rank.

    The OpenVLA-OFT processor can return temporal/multi-camera image tensors as
    ``[B, N, C, H, W]``. RLinf flattens the image axis into channels before
    calling the model's vision path. Matching that convention is important for
    LIBERO parity and harmless for the usual ``[B, C, H, W]`` tensors.
    """

    pixel_values = inputs["pixel_values"]
    if hasattr(pixel_values, "ndim") and int(pixel_values.ndim) == 5:
        batch, num_images, channels, height, width = pixel_values.shape
        return pixel_values.reshape(batch, num_images * channels, height, width)
    return pixel_values


def _action_token_logprobs_for_tokens(
    model: Any,
    inputs: Any,
    *,
    tokens: list[Any],
    unnorm_key: str | None,
    temperature: float,
) -> Any:
    oft_model = _openvla_oft_action_model(model)
    if oft_model is not None:
        return _oft_action_token_logprobs(
            oft_model,
            inputs,
            tokens=tokens,
            unnorm_key=unnorm_key,
            temperature=temperature,
        )
    raise TypeError(
        "OpenVLAPolicy action_token_logprobs currently supports OpenVLA-OFT "
        "placeholder action-token models. Vanilla OpenVLA generation logprob "
        "training needs a separate teacher-forcing adapter."
    )


def _action_token_logprobs_for_token_rows(
    model: Any,
    inputs: Any,
    *,
    token_rows: list[list[Any]],
    unnorm_key: str | None,
    temperature: float,
) -> list[Any]:
    oft_model = _openvla_oft_action_model(model)
    if oft_model is not None:
        return _oft_action_token_logprobs_batch(
            oft_model,
            inputs,
            token_rows=token_rows,
            unnorm_key=unnorm_key,
            temperature=temperature,
        )
    raise TypeError(
        "OpenVLAPolicy action_token_logprobs currently supports OpenVLA-OFT "
        "placeholder action-token models. Vanilla OpenVLA generation logprob "
        "training needs a separate teacher-forcing adapter."
    )


def _openvla_oft_action_model(model: Any) -> Any | None:
    for candidate in _model_unwrap_candidates(model):
        if _is_openvla_oft_action_model(candidate):
            return candidate
    return None


def _model_unwrap_candidates(model: Any) -> list[Any]:
    """Return likely underlying model objects without losing PEFT-injected weights.

    HF/PEFT wrappers often proxy the useful remote-code methods via nested
    ``base_model`` / ``model`` attributes.  We must inspect those objects to
    resolve OpenVLA-OFT action constants consistently for baseline and adapter
    checkpoints.  The nested modules still contain the injected adapter layers.
    """

    candidates: list[Any] = []
    stack = [model]
    seen: set[int] = set()
    while stack:
        candidate = stack.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        candidates.append(candidate)
        for attr in ("base_model", "model", "module"):
            child = getattr(candidate, attr, None)
            if child is not None and id(child) not in seen:
                stack.append(child)

    # Some PEFT wrappers expose the remote-code model only as a registered
    # submodule rather than a simple ``.base_model.model`` chain.  Walking
    # named_modules keeps adapter-reload inference on the OpenVLA-OFT
    # placeholder-token path instead of silently falling back to vanilla
    # seven-token generation.
    named_modules = getattr(model, "named_modules", None)
    if callable(named_modules):
        try:
            for _name, module in named_modules():
                if module is not None and id(module) not in seen:
                    seen.add(id(module))
                    candidates.append(module)
        except Exception:
            pass
    return candidates


def _is_openvla_oft_action_model(model: Any) -> bool:
    if _openvla_oft_supports_rlinf_rollout_style(model):
        return True
    return all(
        hasattr(model, name)
        for name in (
            "_prepare_input_for_action_prediction",
            "_prepare_labels_for_action_prediction",
            "_process_action_masks",
            "_build_multimodal_attention",
        )
    )


def _openvla_oft_constants(model: Any, *, unnorm_key: str | None) -> dict[str, int]:
    action_dim = _first_openvla_oft_constant(model, "ACTION_DIM", ("action_dim",))
    if action_dim is None:
        for candidate in _model_unwrap_candidates(model):
            if hasattr(candidate, "get_action_dim"):
                action_dim = candidate.get_action_dim(unnorm_key)
                break
    if action_dim is None:
        raise ValueError("Could not infer OpenVLA-OFT ACTION_DIM")

    num_chunks = _first_openvla_oft_constant(
        model,
        "NUM_ACTIONS_CHUNK",
        ("num_actions_chunk", "num_action_chunks"),
    )
    if num_chunks is None:
        num_chunks = 1

    ignore_index = _first_openvla_oft_constant(model, "IGNORE_INDEX", ("ignore_index",))
    if ignore_index is None:
        ignore_index = -100
    stop_index = _first_openvla_oft_constant(model, "STOP_INDEX", ("stop_index",))
    if stop_index is None:
        stop_index = 2
    action_token_begin_idx = _first_openvla_oft_constant(
        model,
        "ACTION_TOKEN_BEGIN_IDX",
        ("action_token_begin_idx",),
    )
    if action_token_begin_idx is None:
        action_token_begin_idx = 31743
    return {
        "ACTION_DIM": int(action_dim),
        "NUM_ACTIONS_CHUNK": int(num_chunks),
        "IGNORE_INDEX": int(ignore_index),
        "STOP_INDEX": int(stop_index),
        "ACTION_TOKEN_BEGIN_IDX": int(action_token_begin_idx),
    }


def _first_openvla_oft_constant(
    model: Any, module_name: str, attr_names: tuple[str, ...]
) -> Any:
    import sys

    for candidate in _model_unwrap_candidates(model):
        module = sys.modules.get(candidate.__class__.__module__)
        if module is not None and hasattr(module, module_name):
            return getattr(module, module_name)
        for attr in attr_names:
            if hasattr(candidate, attr):
                value = getattr(candidate, attr)
                if value is not None:
                    return value
    return None


def _oft_action_logits(model: Any, inputs: Any, *, unnorm_key: str | None) -> Any:
    # RLinf's implement_version=rlinf rollout wrapper has a distinct indexing
    # contract: prompts are max-length padded and action logits start at
    # n_patches + (input_ids.shape[-1] - 1), not at the non-pad prompt length.
    # Prefer this exact path when the wrapper shape is detected.
    if _openvla_oft_supports_rlinf_rollout_style(model):
        return _oft_action_logits_rlinf_rollout_style(
            model, inputs, unnorm_key=unnorm_key
        )
    # When the checkpoint exposes the VERL rollout helper, mirror that path next.
    if hasattr(model, "generate_action_verl"):
        return _oft_action_logits_verl_style(model, inputs, unnorm_key=unnorm_key)
    if _openvla_oft_supports_rlinf_main_style(model):
        return _oft_action_logits_rlinf_main_style(model, inputs, unnorm_key=unnorm_key)
    return _oft_action_logits_predict_action_style(model, inputs, unnorm_key=unnorm_key)


def _openvla_oft_supports_rlinf_rollout_style(model: Any) -> bool:
    return all(
        hasattr(model, name)
        for name in (
            "input_processor",
            "_prepare_input_for_action_prediction",
            "_build_embedding",
            "language_model",
            "vision_backbone",
            "action_dim",
            "num_action_chunks",
            "vocab_size",
        )
    )


def _oft_action_logits_rlinf_rollout_style(
    model: Any, inputs: Any, *, unnorm_key: str | None
) -> Any:
    """Extract logits exactly like RLinf's implement_version=rlinf rollout.

    This path intentionally does not sort padding or use non-pad token counts.
    RLinf's rollout wrapper pads prompts to ``max_prompt_length`` and indexes
    action logits with ``input_ids.shape[-1] - 1``. Using a more intuitive
    non-pad length silently shifts every action token position.
    """

    import torch

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(dtype=torch.long)
    pixel_values = _openvla_oft_pixel_values_for_model(inputs)

    n_prompt_tokens = input_ids.shape[-1] - 1
    n_patches = (
        model.vision_backbone.get_num_patches()
        * model.vision_backbone.get_num_images_in_input()
    )
    action_token_count = int(model.action_dim) * int(model.num_action_chunks)

    input_ids, attention_mask = model._prepare_input_for_action_prediction(
        input_ids, attention_mask
    )
    multimodal_embeddings, multimodal_attention_mask = model._build_embedding(
        input_ids,
        attention_mask,
        pixel_values,
    )
    multimodal_position_ids = multimodal_attention_mask.cumsum(dim=1) - 1
    outputs = model.language_model(
        input_ids=None,
        attention_mask=multimodal_attention_mask,
        position_ids=multimodal_position_ids,
        past_key_values=None,
        inputs_embeds=multimodal_embeddings,
        labels=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    logits = outputs.logits[
        :,
        n_patches + n_prompt_tokens : n_patches + n_prompt_tokens + action_token_count,
        :,
    ].clone()
    logits[..., : model.vocab_size - model.config.n_action_bins] = -torch.inf
    logits[..., model.vocab_size :] = -torch.inf
    return logits


def _openvla_oft_logit_extraction_method(model: Any) -> str | None:
    oft_model = _openvla_oft_action_model(model)
    if oft_model is None:
        return None
    if _openvla_oft_supports_rlinf_rollout_style(oft_model):
        if bool(getattr(oft_model, "_art_native_openvla_oft", False)):
            return "art_native_openvla_oft_rollout"
        return "rlinf_rollout_predict_action_batch_compatible"
    if hasattr(oft_model, "generate_action_verl"):
        return "native_generate_action_verl_compatible"
    if _openvla_oft_supports_rlinf_main_style(oft_model):
        return "rlinf_main_predict_action_batch_compatible"
    return "native_predict_action_compatible"


def _openvla_oft_supports_rlinf_main_style(model: Any) -> bool:
    return all(
        hasattr(model, name)
        for name in (
            "_prepare_input_for_action_prediction",
            "_process_vision_features",
            "_build_multimodal_attention",
            "language_model",
            "get_input_embeddings",
        )
    )


def _openvla_oft_action_range_metadata(model: Any) -> dict[str, Any]:
    oft_model = _openvla_oft_action_model(model)
    if oft_model is None:
        return {}
    token_start, token_end = _openvla_oft_action_token_range(oft_model)
    logit_start, logit_end = _openvla_oft_action_logit_range(oft_model)
    return {
        "openvla_oft_action_token_range": [int(token_start), int(token_end)],
        "openvla_oft_action_logit_range": [int(logit_start), int(logit_end)],
        "openvla_oft_action_logit_to_token_offset": int(token_start - logit_start),
    }


def _oft_action_logits_rlinf_main_style(
    model: Any, inputs: Any, *, unnorm_key: str | None
) -> Any:
    """Extract OpenVLA-OFT action logits like RLinf's main rollout wrapper.

    RLinf computes action logits from the *unpadded prompt length* even after
    left-padding prompts to ``max_prompt_length``. Using the padded sequence
    length shifts every action position and silently samples from the wrong time
    steps. Keep this function intentionally close to
    ``OpenVLAOFTForRLActionPrediction.predict_action_batch``.
    """

    import torch

    constants = _openvla_oft_constants(model, unnorm_key=unnorm_key)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(dtype=torch.long)

    if not torch.all(input_ids[:, -1] == 29871):
        spacer = torch.full(
            (input_ids.shape[0], 1),
            29871,
            device=input_ids.device,
            dtype=torch.long,
        )
        input_ids = torch.cat((input_ids, spacer), dim=1)
        attention_mask = torch.cat(
            (attention_mask, torch.ones_like(spacer, dtype=attention_mask.dtype)),
            dim=1,
        )

    pad_token_id = _openvla_oft_padding_idx(inputs)
    processor = getattr(model, "processor", None)
    tokenizer = getattr(processor, "tokenizer", None)
    if getattr(tokenizer, "pad_token_id", None) is not None:
        pad_token_id = int(tokenizer.pad_token_id)
    num_prompt_tokens = input_ids.ne(pad_token_id).sum(dim=1) - 1
    labels = input_ids.clone()
    labels[:] = constants["IGNORE_INDEX"]
    pixel_values = _openvla_oft_pixel_values_for_model(inputs)
    proprio = inputs.get("proprio") if hasattr(inputs, "get") else None

    if hasattr(model, "_build_embedding"):
        multimodal_embeddings, multimodal_attention_mask, multimodal_position_ids = (
            model._build_embedding(
                input_ids,
                attention_mask,
                pixel_values,
                labels,
                proprio,
            )
        )
    else:
        input_ids, attention_mask = model._prepare_input_for_action_prediction(
            input_ids,
            attention_mask,
        )
        labels = model._prepare_labels_for_action_prediction(labels, input_ids)
        padding_mask = input_ids.ne(pad_token_id)
        sorted_indices = torch.argsort(
            padding_mask.int(),
            dim=1,
            descending=True,
            stable=True,
        )
        input_ids = torch.gather(input_ids, 1, sorted_indices)
        attention_mask = torch.gather(attention_mask, 1, sorted_indices)
        labels = torch.gather(labels, 1, sorted_indices)
        input_embeddings = model.get_input_embeddings()(input_ids)
        all_actions_mask = model._process_action_masks(labels)
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0],
            -1,
            input_embeddings.shape[2],
        )
        projected_patch_embeddings = model._process_vision_features(
            pixel_values,
            language_embeddings,
            getattr(model, "use_film", False),
        )
        use_proprio = (
            getattr(model, "proprio_projector", None) is not None
            and proprio is not None
        )
        if use_proprio and hasattr(model, "_process_proprio_features"):
            proprio = torch.as_tensor(
                proprio,
                device=projected_patch_embeddings.device,
                dtype=projected_patch_embeddings.dtype,
            )
            projected_patch_embeddings = model._process_proprio_features(
                projected_patch_embeddings,
                proprio,
                model.proprio_projector,
            )
        input_embeddings = input_embeddings * ~all_actions_mask.unsqueeze(-1)
        multimodal_embeddings, multimodal_attention_mask = (
            model._build_multimodal_attention(
                input_embeddings,
                projected_patch_embeddings,
                attention_mask,
            )
        )
        multimodal_position_ids = (
            multimodal_attention_mask.long().cumsum(-1) - 1
        ).masked_fill(multimodal_attention_mask == 0, 1)

    language_model_output = model.language_model(
        input_ids=None,
        attention_mask=multimodal_attention_mask,
        position_ids=multimodal_position_ids,
        past_key_values=None,
        inputs_embeds=multimodal_embeddings,
        labels=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )

    num_patches = (
        model.vision_backbone.get_num_patches()
        * model.vision_backbone.get_num_images_in_input()
    )
    if getattr(model, "proprio_projector", None) is not None and proprio is not None:
        num_patches += 1
    action_token_count = constants["ACTION_DIM"] * constants["NUM_ACTIONS_CHUNK"]
    start_indices = (num_patches + num_prompt_tokens).unsqueeze(1)
    position_offsets = torch.arange(
        action_token_count,
        device=language_model_output.logits.device,
    ).unsqueeze(0)
    seq_indices = start_indices + position_offsets
    batch_indices = torch.arange(
        language_model_output.logits.shape[0],
        device=language_model_output.logits.device,
    ).unsqueeze(-1)
    return language_model_output.logits[batch_indices, seq_indices, :]


def _oft_action_logits_predict_action_style(
    model: Any, inputs: Any, *, unnorm_key: str | None
) -> Any:
    import torch

    constants = _openvla_oft_constants(model, unnorm_key=unnorm_key)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    if not torch.all(input_ids[:, -1] == 29871):
        spacer = torch.full(
            (input_ids.shape[0], 1),
            29871,
            device=input_ids.device,
            dtype=torch.long,
        )
        input_ids = torch.cat((input_ids, spacer), dim=1)
        attention_mask = torch.cat(
            (attention_mask, torch.ones_like(spacer, dtype=attention_mask.dtype)),
            dim=1,
        )

    labels = input_ids.clone()
    labels[:] = constants["IGNORE_INDEX"]
    padding_idx = _openvla_oft_padding_idx(inputs)
    num_prompt_tokens = input_ids.ne(padding_idx).sum(dim=1) - 1

    input_ids, attention_mask = model._prepare_input_for_action_prediction(
        input_ids,
        attention_mask,
    )
    labels = model._prepare_labels_for_action_prediction(labels, input_ids)

    input_embeddings = model.get_input_embeddings()(input_ids)
    all_actions_mask = model._process_action_masks(labels)
    language_embeddings = input_embeddings[~all_actions_mask].reshape(
        input_embeddings.shape[0],
        -1,
        input_embeddings.shape[2],
    )

    pixel_values = _openvla_oft_pixel_values_for_model(inputs)
    projected_patch_embeddings = model._process_vision_features(
        pixel_values,
        language_embeddings,
        False,
    )

    action_mask = all_actions_mask.unsqueeze(-1)
    input_embeddings = input_embeddings * ~action_mask
    multimodal_embeddings, multimodal_attention_mask = (
        model._build_multimodal_attention(
            input_embeddings,
            projected_patch_embeddings,
            attention_mask,
        )
    )

    language_model_output = model.language_model(
        input_ids=None,
        attention_mask=multimodal_attention_mask,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=multimodal_embeddings,
        labels=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )

    num_patches = (
        model.vision_backbone.get_num_patches()
        * model.vision_backbone.get_num_images_in_input()
    )
    count = constants["ACTION_DIM"] * constants["NUM_ACTIONS_CHUNK"]
    start_indices = (num_patches + num_prompt_tokens).unsqueeze(1)
    position_offsets = torch.arange(
        count, device=language_model_output.logits.device
    ).unsqueeze(0)
    seq_indices = start_indices + position_offsets
    batch_indices = torch.arange(
        language_model_output.logits.shape[0],
        device=language_model_output.logits.device,
    ).unsqueeze(-1)
    return language_model_output.logits[batch_indices, seq_indices, :]


def _oft_action_logits_verl_style(
    model: Any, inputs: Any, *, unnorm_key: str | None
) -> Any:
    import torch

    constants = _openvla_oft_constants(model, unnorm_key=unnorm_key)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    padding_idx = _openvla_oft_padding_idx(inputs)
    num_prompt_tokens = input_ids.ne(padding_idx).sum(dim=1) - 1

    labels = input_ids.clone()
    labels[:] = constants["IGNORE_INDEX"]
    input_ids, attention_mask = model._prepare_input_for_action_prediction(
        input_ids,
        attention_mask,
    )
    labels = model._prepare_labels_for_action_prediction(labels, input_ids)

    # RLinf's VERL/OpenVLA-OFT path moves prompt tokens before padding after
    # appending placeholder action tokens. Matching this ordering is necessary
    # for action logits, rollout old logprobs, and current-policy logprobs to
    # describe the same distribution.
    padding_mask = input_ids.ne(padding_idx)
    sorted_indices = torch.argsort(
        padding_mask.int(),
        dim=1,
        descending=True,
        stable=True,
    )
    input_ids = torch.gather(input_ids, 1, sorted_indices)
    attention_mask = torch.gather(attention_mask, 1, sorted_indices)
    labels = torch.gather(labels, 1, sorted_indices)

    input_embeddings = model.get_input_embeddings()(input_ids)
    all_actions_mask = model._process_action_masks(labels)
    language_embeddings = input_embeddings[~all_actions_mask].reshape(
        input_embeddings.shape[0],
        -1,
        input_embeddings.shape[2],
    )

    pixel_values = _openvla_oft_pixel_values_for_model(inputs)
    projected_patch_embeddings = model._process_vision_features(
        pixel_values,
        language_embeddings,
        False,
    )

    action_mask = all_actions_mask.unsqueeze(-1)
    input_embeddings = input_embeddings * ~action_mask
    multimodal_embeddings, multimodal_attention_mask = (
        model._build_multimodal_attention(
            input_embeddings,
            projected_patch_embeddings,
            attention_mask,
        )
    )
    language_model_output = model.language_model(
        input_ids=None,
        attention_mask=multimodal_attention_mask,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=multimodal_embeddings,
        labels=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )

    num_patches = (
        model.vision_backbone.get_num_patches()
        * model.vision_backbone.get_num_images_in_input()
    )
    action_token_count = constants["ACTION_DIM"] * constants["NUM_ACTIONS_CHUNK"]
    start_indices = (num_prompt_tokens + num_patches).unsqueeze(1)
    position_offsets = torch.arange(
        action_token_count,
        device=language_model_output.logits.device,
    ).unsqueeze(0)
    seq_indices = start_indices + position_offsets
    batch_indices = torch.arange(
        language_model_output.logits.shape[0],
        device=language_model_output.logits.device,
    ).unsqueeze(-1)
    return language_model_output.logits[batch_indices, seq_indices, :]


def _openvla_oft_padding_idx(inputs: Any) -> int:
    tokenizer = getattr(inputs, "tokenizer", None)
    padding_idx = getattr(tokenizer, "pad_token_id", None)
    if padding_idx is not None:
        return int(padding_idx)
    return 0


def _try_rlinf_native_oft_prediction(
    model: Any,
    inputs: Any,
    *,
    unnorm_key: str | None,
    do_sample: bool,
    temperature: float,
    top_k: int = -1,
) -> dict[str, Any] | None:
    """Use RLinf/OpenVLA-OFT's native discrete action path when available.

    The exact action-token positions for OpenVLA-OFT are easy to get subtly
    wrong because prompt padding, action placeholders, vision patches, and
    padded vocab rows interact. RLinf's prior-art wrapper exposes the canonical
    `_build_embedding` + `_discrete_prediction` path; use it directly instead
    of reconstructing the same indexing logic when that method is present.
    """

    if not (
        hasattr(model, "_build_embedding") and hasattr(model, "_discrete_prediction")
    ):
        return None

    import torch

    constants = _openvla_oft_constants(model, unnorm_key=unnorm_key)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(dtype=torch.long)

    if not torch.all(input_ids[:, -1] == 29871):
        spacer = torch.full(
            (input_ids.shape[0], 1),
            29871,
            device=input_ids.device,
            dtype=torch.long,
        )
        input_ids = torch.cat((input_ids, spacer), dim=1)
        attention_mask = torch.cat(
            (attention_mask, torch.ones_like(spacer, dtype=attention_mask.dtype)),
            dim=1,
        )

    pad_token_id = _openvla_oft_padding_idx(inputs)
    processor = getattr(model, "processor", None)
    tokenizer = getattr(processor, "tokenizer", None)
    if getattr(tokenizer, "pad_token_id", None) is not None:
        pad_token_id = int(tokenizer.pad_token_id)
    num_prompt_tokens = input_ids.ne(pad_token_id).sum(dim=1) - 1

    labels = input_ids.clone()
    labels[:] = constants["IGNORE_INDEX"]
    pixel_values = _openvla_oft_pixel_values_for_model(inputs)
    proprio = inputs.get("proprio") if hasattr(inputs, "get") else None

    multimodal_embeddings, multimodal_attention_mask, multimodal_position_ids = (
        model._build_embedding(
            input_ids,
            attention_mask,
            pixel_values,
            labels,
            proprio,
        )
    )
    num_patches = (
        model.vision_backbone.get_num_patches()
        * model.vision_backbone.get_num_images_in_input()
    )
    if getattr(model, "proprio_projector", None) is not None and proprio is not None:
        num_patches += 1

    normalized_actions, action_logits, local_action_tokens, last_hidden_states = (
        model._discrete_prediction(
            multimodal_embeddings,
            multimodal_attention_mask,
            multimodal_position_ids,
            num_patches,
            num_prompt_tokens,
            do_sample=bool(do_sample),
            temperature=float(temperature),
            top_k=int(top_k),
        )
    )
    actions = model._unnormalize_actions(normalized_actions, unnorm_key)
    actions = actions.reshape(
        -1, constants["NUM_ACTIONS_CHUNK"], constants["ACTION_DIM"]
    )
    return {
        "actions": actions,
        "action_logits": action_logits,
        "local_action_tokens": local_action_tokens,
        "last_hidden_states": last_hidden_states,
        "logits_already_sample_scaled": bool(do_sample),
    }


def _logprobs_from_action_bin_logits(
    action_logits: Any,
    local_token_ids: Any,
    *,
    temperature: float,
    logits_already_sample_scaled: bool = False,
) -> Any:
    import torch

    scaled_logits = (
        action_logits
        if logits_already_sample_scaled
        else action_logits / max(float(temperature), 1e-6)
    )
    return (
        torch.log_softmax(scaled_logits, dim=-1)
        .gather(
            -1,
            local_token_ids.unsqueeze(-1),
        )
        .squeeze(-1)
    )


def _sample_rlinf_oft_action_tokens(
    logits: Any,
    *,
    action_logit_start: int,
    action_logit_end: int,
    do_sample: bool,
    temperature: float,
) -> tuple[Any, Any]:
    """Sample full-vocabulary action IDs with RLinf's exact geometry.

    RLinf masks non-action vocabulary rows, applies temperature over the full
    padded vocabulary, flattens batch and action positions, and calls
    ``torch.multinomial``. Sampling a sliced 256-bin ``Categorical`` is
    distributionally equivalent, but it consumes a different RNG stream and
    breaks fixed-seed rollout conformance. Keep the full-vocabulary operation
    here so ART and the positive-control implementation make the same draw.
    """

    import torch

    if logits.ndim < 2:
        raise ValueError(
            f"OpenVLA-OFT action logits must have rank >= 2, got {logits.shape}"
        )
    vocab_size = int(logits.shape[-1])
    if not (0 <= int(action_logit_start) < int(action_logit_end) <= vocab_size):
        raise ValueError(
            "Invalid OpenVLA-OFT action-logit range: "
            f"start={action_logit_start}, end={action_logit_end}, vocab={vocab_size}"
        )

    masked_logits = logits.clone()
    masked_logits[..., :action_logit_start] = -torch.inf
    masked_logits[..., action_logit_end:] = -torch.inf
    processed_logits = (
        masked_logits / max(float(temperature), 1e-6) if do_sample else masked_logits
    )
    logprobs = torch.log_softmax(processed_logits, dim=-1)
    if do_sample:
        probabilities = torch.exp(logprobs)
        sampled = torch.multinomial(
            probabilities.reshape(-1, vocab_size),
            num_samples=1,
            replacement=True,
        ).reshape(logits.shape[:-1])
    else:
        sampled = processed_logits.argmax(dim=-1)
    selected_logprobs = logprobs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    return sampled, selected_logprobs


def _predict_oft_action_with_tokens(
    model: Any,
    inputs: Any,
    *,
    unnorm_key: str | None,
    do_sample: bool,
    temperature: float,
) -> tuple[Any, list[int], list[float]]:
    import torch

    native = _try_rlinf_native_oft_prediction(
        model,
        inputs,
        unnorm_key=unnorm_key,
        do_sample=do_sample,
        temperature=temperature,
        top_k=-1,
    )
    action_token_start, _action_token_end = _openvla_oft_action_token_range(model)
    if native is not None:
        local_token_tensor = native["local_action_tokens"][0].to(dtype=torch.long)
        token_logprobs_tensor = _logprobs_from_action_bin_logits(
            native["action_logits"][0],
            local_token_tensor,
            temperature=temperature,
            logits_already_sample_scaled=bool(
                native.get("logits_already_sample_scaled")
            ),
        )
        token_tensor = local_token_tensor + action_token_start
        token_ids = [int(token_id) for token_id in token_tensor.detach().cpu().tolist()]
        token_logprobs = [
            float(value) for value in token_logprobs_tensor.detach().cpu().tolist()
        ]
        return native["actions"][0], token_ids, token_logprobs

    logits = _oft_action_logits(model, inputs, unnorm_key=unnorm_key)[0]
    action_logit_start, action_logit_end = _openvla_oft_action_logit_range(model)
    token_tensor, token_logprobs_tensor = _sample_rlinf_oft_action_tokens(
        logits,
        action_logit_start=action_logit_start,
        action_logit_end=action_logit_end,
        do_sample=do_sample,
        temperature=temperature,
    )
    if action_token_start != action_logit_start:
        token_tensor = token_tensor - action_logit_start + action_token_start
    token_ids = [int(token_id) for token_id in token_tensor.detach().cpu().tolist()]
    token_logprobs = [
        float(value) for value in token_logprobs_tensor.detach().cpu().tolist()
    ]
    actions = _openvla_oft_tokens_to_actions(model, token_ids, unnorm_key=unnorm_key)
    return actions, token_ids, token_logprobs


def _predict_oft_actions_with_tokens_batch(
    model: Any,
    inputs: Any,
    *,
    unnorm_key: str | None,
    do_sample: bool,
    temperature: float,
) -> list[tuple[Any, list[int], list[float]]]:
    import torch

    action_token_start, _action_token_end = _openvla_oft_action_token_range(model)
    native = _try_rlinf_native_oft_prediction(
        model,
        inputs,
        unnorm_key=unnorm_key,
        do_sample=do_sample,
        temperature=temperature,
        top_k=-1,
    )
    if native is not None:
        local_token_tensor = native["local_action_tokens"].to(dtype=torch.long)
        token_logprobs_tensor = _logprobs_from_action_bin_logits(
            native["action_logits"],
            local_token_tensor,
            temperature=temperature,
            logits_already_sample_scaled=bool(
                native.get("logits_already_sample_scaled")
            ),
        )
        token_tensor = local_token_tensor + action_token_start
        predictions: list[tuple[Any, list[int], list[float]]] = []
        for row_index in range(token_tensor.shape[0]):
            token_ids = [
                int(token_id)
                for token_id in token_tensor[row_index].detach().cpu().tolist()
            ]
            token_logprobs = [
                float(value)
                for value in token_logprobs_tensor[row_index].detach().cpu().tolist()
            ]
            predictions.append(
                (native["actions"][row_index], token_ids, token_logprobs)
            )
        return predictions

    logits = _oft_action_logits(model, inputs, unnorm_key=unnorm_key)
    action_logit_start, action_logit_end = _openvla_oft_action_logit_range(model)
    token_tensor, token_logprobs_tensor = _sample_rlinf_oft_action_tokens(
        logits,
        action_logit_start=action_logit_start,
        action_logit_end=action_logit_end,
        do_sample=do_sample,
        temperature=temperature,
    )
    if action_token_start != action_logit_start:
        token_tensor = token_tensor - action_logit_start + action_token_start
    predictions: list[tuple[Any, list[int], list[float]]] = []
    for row_index in range(token_tensor.shape[0]):
        token_ids = [
            int(token_id)
            for token_id in token_tensor[row_index].detach().cpu().tolist()
        ]
        token_logprobs = [
            float(value)
            for value in token_logprobs_tensor[row_index].detach().cpu().tolist()
        ]
        actions = _openvla_oft_tokens_to_actions(
            model,
            token_ids,
            unnorm_key=unnorm_key,
        )
        predictions.append((actions, token_ids, token_logprobs))
    return predictions


def _oft_action_token_logprobs(
    model: Any,
    inputs: Any,
    *,
    tokens: list[Any],
    unnorm_key: str | None,
    temperature: float,
) -> Any:
    import torch

    token_ids = torch.as_tensor(
        [_coerce_int_token(token) for token in tokens],
        dtype=torch.long,
        device=inputs["input_ids"].device,
    )
    logits = _oft_action_logits(model, inputs, unnorm_key=unnorm_key)[0]
    if logits.shape[0] != token_ids.shape[0]:
        raise ValueError(
            "OpenVLA-OFT token count mismatch: "
            f"logits={int(logits.shape[0])} tokens={int(token_ids.shape[0])}"
        )
    action_token_start, action_token_end = _openvla_oft_action_token_range(model)
    if bool(((token_ids < action_token_start) | (token_ids >= action_token_end)).any()):
        raise ValueError(
            "OpenVLA-OFT action token ids must be inside the action-token "
            f"range [{action_token_start}, {action_token_end}); got "
            f"{[int(token) for token in token_ids.detach().cpu().tolist()]}"
        )
    local_token_ids = token_ids - action_token_start
    action_logit_start, action_logit_end = _openvla_oft_action_logit_range(model)
    scaled_logits = logits[:, action_logit_start:action_logit_end] / max(
        float(temperature), 1e-6
    )
    return (
        torch.log_softmax(scaled_logits, dim=-1)
        .gather(
            -1,
            local_token_ids.unsqueeze(-1),
        )
        .squeeze(-1)
    )


def _oft_action_token_logprobs_batch(
    model: Any,
    inputs: Any,
    *,
    token_rows: list[list[Any]],
    unnorm_key: str | None,
    temperature: float,
) -> list[Any]:
    import torch

    if not token_rows:
        return []
    lengths = {len(row) for row in token_rows}
    if len(lengths) != 1:
        raise ValueError(
            "Batched OpenVLA-OFT action-token logprobs require equal token lengths"
        )
    token_ids = torch.as_tensor(
        [[_coerce_int_token(token) for token in row] for row in token_rows],
        dtype=torch.long,
        device=inputs["input_ids"].device,
    )
    logits = _oft_action_logits(model, inputs, unnorm_key=unnorm_key)
    if logits.shape[:2] != token_ids.shape:
        raise ValueError(
            "OpenVLA-OFT batched token count mismatch: "
            f"logits={tuple(logits.shape[:2])} tokens={tuple(token_ids.shape)}"
        )
    action_token_start, action_token_end = _openvla_oft_action_token_range(model)
    if bool(((token_ids < action_token_start) | (token_ids >= action_token_end)).any()):
        raise ValueError(
            "OpenVLA-OFT action token ids must be inside the action-token "
            f"range [{action_token_start}, {action_token_end})"
        )
    local_token_ids = token_ids - action_token_start
    action_logit_start, action_logit_end = _openvla_oft_action_logit_range(model)
    scaled_logits = logits[:, :, action_logit_start:action_logit_end] / max(
        float(temperature), 1e-6
    )
    logprobs = (
        torch.log_softmax(scaled_logits, dim=-1)
        .gather(
            -1,
            local_token_ids.unsqueeze(-1),
        )
        .squeeze(-1)
    )
    return [logprobs[index] for index in range(logprobs.shape[0])]


def _coerce_int_token(token: Any) -> int:
    if isinstance(token, int):
        return int(token)
    if hasattr(token, "item") and callable(token.item):
        return int(token.item())
    return int(str(token))


def _openvla_oft_tokens_to_actions(
    model: Any,
    token_ids: list[int],
    *,
    unnorm_key: str | None,
) -> Any:
    import numpy as np

    constants = _openvla_oft_constants(model, unnorm_key=unnorm_key)
    action_dim = constants["ACTION_DIM"]
    num_chunks = constants["NUM_ACTIONS_CHUNK"]
    token_array = np.asarray(token_ids)
    discretized_actions = model.vocab_size - token_array
    discretized_actions = np.clip(
        discretized_actions - 1,
        a_min=0,
        a_max=model.bin_centers.shape[0] - 1,
    )
    normalized_actions = model.bin_centers[discretized_actions].reshape(
        num_chunks,
        action_dim,
    )
    if hasattr(model, "_unnormalize_actions"):
        return model._unnormalize_actions(normalized_actions, unnorm_key)

    action_norm_stats = model.get_action_stats(unnorm_key)
    mask = action_norm_stats.get(
        "mask",
        np.ones_like(
            action_norm_stats.get("q01", action_norm_stats.get("min")), dtype=bool
        ),
    )
    high = np.asarray(action_norm_stats.get("q99", action_norm_stats.get("max")))
    low = np.asarray(action_norm_stats.get("q01", action_norm_stats.get("min")))
    return np.where(
        mask,
        0.5 * (normalized_actions + 1) * (high - low + 1e-8) + low,
        normalized_actions,
    )


def _openvla_oft_action_token_range(model: Any) -> tuple[int, int]:
    """Return the contiguous OpenVLA-OFT emitted action-token id range.

    OpenVLA-OFT emits action ids in the final action bin range and detokenizes
    them as ``vocab_size - token_id``. Some checkpoints, including the RLinf
    VERL path, read the *logits* from a different contiguous slice that starts at
    ``ACTION_TOKEN_BEGIN_IDX + 1`` and may be offset by trailing special tokens.
    Keep the emitted-id range separate from :func:`_openvla_oft_action_logit_range`;
    conflating the two silently trains on the wrong distribution.
    """

    action_vocab_size = _openvla_oft_action_vocab_size(model)
    vocab_size = int(getattr(model, "vocab_size"))
    action_token_start = vocab_size - action_vocab_size
    if action_token_start < 0:
        raise ValueError(
            "Invalid OpenVLA-OFT action token range: "
            f"vocab_size={vocab_size}, action_vocab_size={action_vocab_size}"
        )
    return action_token_start, vocab_size


def _openvla_oft_action_logit_range(model: Any) -> tuple[int, int]:
    """Return the logits slice used to sample OpenVLA-OFT action bins.

    OpenVLA-OFT remote code samples from
    ``logits[..., -n_action_bins-pad_to_multiple_of:-pad_to_multiple_of]`` and
    then maps the local bin to emitted action-token ids starting at
    ``model.vocab_size - n_action_bins``.  ``model.vocab_size`` excludes padded
    vocabulary rows, while ``language_model_output.logits.shape[-1]`` usually
    includes them.  Therefore this range must be based on the language-model
    logit vocabulary size, not on ``model.vocab_size``.
    """

    action_vocab_size = _openvla_oft_action_vocab_size(model)
    config = getattr(model, "config", None)
    pad_to_multiple_of = int(getattr(config, "pad_to_multiple_of", 0) or 0)
    logit_vocab_size = _openvla_oft_logit_vocab_size(model)
    begin = logit_vocab_size - action_vocab_size - pad_to_multiple_of
    end = logit_vocab_size - pad_to_multiple_of
    if 0 <= begin < end <= logit_vocab_size and (end - begin) == action_vocab_size:
        return begin, end
    return _openvla_oft_action_token_range(model)


def _openvla_oft_logit_vocab_size(model: Any) -> int:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    for candidate in (
        text_config,
        config,
        getattr(getattr(model, "language_model", None), "config", None),
    ):
        value = (
            getattr(candidate, "vocab_size", None) if candidate is not None else None
        )
        if value:
            return int(value)
    vocab_size = int(getattr(model, "vocab_size"))
    pad_to_multiple_of = int(getattr(config, "pad_to_multiple_of", 0) or 0)
    return vocab_size + pad_to_multiple_of


def _openvla_oft_action_vocab_size(model: Any) -> int:
    config = getattr(model, "config", None)
    action_vocab_size = int(getattr(config, "n_action_bins", 0) or 0)
    if action_vocab_size <= 0:
        bin_centers = getattr(model, "bin_centers", [])
        action_vocab_size = int(len(bin_centers) if len(bin_centers) else 256)
    return action_vocab_size


def _openvla_action_logprobs_payload(
    *,
    token_logprobs: list[float],
    distribution_metadata: dict[str, Any],
    action_output_kind: str,
) -> dict[str, Any]:
    """Return rollout-time old-logprob payload for OpenVLA action bins.

    OpenVLA-OFT samples discrete action-bin tokens and decodes them into the
    continuous action executed by the environment. For token-mode ART backends we
    keep the historical payload exactly as token logprobs. For continuous-mode
    GRPO/GSPO, we additionally expose those same per-bin logprobs as
    ``action_logprobs`` and attach the binned-categorical distribution metadata
    that makes the continuous action contract explicit.
    """

    token_values = [float(value) for value in token_logprobs]
    if action_output_kind == "continuous":
        return {
            "action_logprobs": token_values,
            "token_logprobs": token_values,
            "policy_distribution": distribution_metadata,
        }
    return {"token_logprobs": token_values}


def _openvla_binned_categorical_distribution_metadata(
    model: Any,
    *,
    token_ids: list[int],
    unnorm_key: str | None,
    temperature: float,
) -> dict[str, Any]:
    """Describe the categorical action-bin distribution behind OpenVLA actions.

    The action executed by the simulator is continuous, but OpenVLA-OFT chooses it
    by sampling/classifying one token per action dimension from a finite set of
    action bins. Capturing this metadata avoids pretending that the rollout came
    from a Gaussian policy while still giving continuous-action GRPO/GSPO a real
    old-logprob contract.
    """

    action_model = _openvla_oft_action_model(model) or model
    clean_token_ids = [int(token) for token in token_ids]
    distribution: dict[str, Any] = {
        "family": "binned_categorical",
        "token_ids": clean_token_ids,
        "tokens": clean_token_ids,
        "temperature": float(temperature),
        "logprob_unit": "action_dim",
        "unnorm_key": unnorm_key,
    }

    try:
        token_start, token_end = _openvla_oft_action_token_range(action_model)
        distribution["action_token_range"] = [int(token_start), int(token_end)]
    except Exception:
        pass
    try:
        logit_start, logit_end = _openvla_oft_action_logit_range(action_model)
        distribution["action_logit_range"] = [int(logit_start), int(logit_end)]
    except Exception:
        pass
    try:
        constants = _openvla_oft_constants(action_model, unnorm_key=unnorm_key)
        distribution["action_dim"] = int(constants["ACTION_DIM"])
        distribution["num_action_chunks"] = int(constants["NUM_ACTIONS_CHUNK"])
        distribution["action_shape"] = [
            int(constants["NUM_ACTIONS_CHUNK"]),
            int(constants["ACTION_DIM"]),
        ]
    except Exception:
        pass
    bin_centers = getattr(action_model, "bin_centers", None)
    if bin_centers is not None:
        try:
            distribution["bin_centers"] = make_json_safe(bin_centers)
        except Exception:
            try:
                distribution["bin_centers"] = [float(value) for value in bin_centers]
            except Exception:
                pass
    if "bin_centers" not in distribution:
        try:
            distribution["num_bins"] = int(_openvla_oft_action_vocab_size(action_model))
        except Exception:
            pass

    return make_json_safe(distribution)


def _predict_vanilla_action_with_tokens(
    model: Any,
    inputs: Any,
    *,
    unnorm_key: str | None,
    do_sample: bool,
    temperature: float,
) -> tuple[Any, list[int], list[float]]:
    """Run OpenVLA generation while preserving action tokens and logprobs.

    This mirrors OpenVLA's upstream `predict_action` implementation, but asks
    Hugging Face generation to return per-token scores. The decoded continuous
    action remains identical in structure to `predict_action`; the extra token
    data is what lets an ART backend train action-token policies from complete
    embodied trajectories.
    """

    import numpy as np
    import torch

    input_ids = inputs["input_ids"]
    generation_inputs = dict(inputs)
    generation_inputs.pop("input_ids", None)

    # OpenVLA inserts this token after OUT:/ASSISTANT: if it is absent.
    if not torch.all(input_ids[:, -1] == 29871):
        spacer = torch.tensor([[29871]], device=input_ids.device, dtype=torch.long)
        input_ids = torch.cat((input_ids, spacer), dim=1)

    action_dim = model.get_action_dim(unnorm_key)
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": action_dim,
        "do_sample": do_sample,
        "output_scores": True,
        "return_dict_in_generate": True,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
    generated = model.generate(
        input_ids,
        **generate_kwargs,
        **generation_inputs,
    )
    sequences = generated.sequences
    predicted_action_token_ids = sequences[0, -action_dim:]
    token_ids = [
        int(token_id) for token_id in predicted_action_token_ids.detach().cpu().tolist()
    ]

    token_logprobs: list[float] = []
    for score, token_id in zip(
        generated.scores[-action_dim:], predicted_action_token_ids
    ):
        score = score[0] / max(float(temperature), 1e-6)
        logprob = torch.log_softmax(score, dim=-1)[token_id]
        token_logprobs.append(float(logprob.detach().cpu().item()))

    discretized_actions = (
        model.vocab_size - predicted_action_token_ids.detach().cpu().numpy()
    )
    discretized_actions = np.clip(
        discretized_actions - 1,
        a_min=0,
        a_max=model.bin_centers.shape[0] - 1,
    )
    normalized_actions = model.bin_centers[discretized_actions]

    action_norm_stats = model.get_action_stats(unnorm_key)
    mask = action_norm_stats.get(
        "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
    )
    action_high = np.array(action_norm_stats["q99"])
    action_low = np.array(action_norm_stats["q01"])
    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )
    return actions, token_ids, token_logprobs
