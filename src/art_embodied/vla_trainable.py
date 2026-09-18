"""Trainable-surface selection for VLA policy RL.

The embodied RL runners use this module to avoid the most common failure mode
in VLA RL experiments: selecting a parameter surface that produces gradients but
barely changes the policy.  The defaults mirror the public VLA training stacks:
flow/action-expert models update the action expert and action/state projections;
autoregressive FAST models update the language-token policy through PEFT/LoRA or
late language-model blocks, not the giant tied embedding matrix by default.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
import importlib.machinery
import re
import sys
import types
from typing import Any

_ACTION_PROJECTIONS = (
    "state_proj",
    "action_in_proj",
    "action_out_proj",
    "action_time_mlp_in",
    "action_time_mlp_out",
)


@dataclass(frozen=True)
class _LayerSelection:
    enabled: bool
    last_n_layers: int
    selected_layers: set[int]


def configure_vla_trainable_parameters(
    policy: Any,
    *,
    algorithm_cfg: dict[str, Any],
    policy_type: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Configure trainable parameters and return ``(possibly_wrapped_policy, report)``.

    ``algorithm_cfg`` is intentionally the single source of truth so experiment
    conditions remain visible in YAML.  Environment variables are not consulted.
    """

    strategy = str(algorithm_cfg.get("trainable_parameter_strategy") or "regex")
    peft_cfg = algorithm_cfg.get("peft") or {}
    if not isinstance(peft_cfg, dict):
        return policy, _error_report(strategy, "algorithm.peft must be a mapping")

    if bool(peft_cfg.get("enabled", False)):
        return _configure_peft_lora(
            policy,
            strategy=strategy,
            algorithm_cfg=algorithm_cfg,
            peft_cfg=peft_cfg,
            policy_type=policy_type,
        )

    patterns = [
        str(pattern)
        for pattern in (algorithm_cfg.get("trainable_parameter_patterns") or [])
    ]
    allow_embeddings = bool(algorithm_cfg.get("allow_embedding_training", False))
    force_float32 = bool(algorithm_cfg.get("force_trainable_float32", False))
    predicate, expanded_patterns, layer_selection, warnings = _selection_predicate(
        strategy=strategy,
        explicit_patterns=patterns,
        policy=policy,
        allow_embeddings=allow_embeddings,
        algorithm_cfg=algorithm_cfg,
    )
    _freeze_all(policy)
    report = _apply_trainable_predicate(
        policy,
        predicate=predicate,
        force_float32=force_float32,
    )
    gradient_checkpointing_report = _configure_gradient_checkpointing(
        policy, algorithm_cfg, policy_type=policy_type
    )
    report.update(
        {
            "strategy": strategy,
            "method": "raw_parameters",
            "patterns": expanded_patterns,
            "explicit_patterns": patterns,
            "allow_embedding_training": allow_embeddings,
            "force_trainable_float32": force_float32,
            "layer_selection": _layer_selection_report(layer_selection),
            "gradient_checkpointing": gradient_checkpointing_report,
            "warnings": warnings,
        }
    )
    return policy, report


def _configure_gradient_checkpointing(
    policy: Any,
    algorithm_cfg: dict[str, Any],
    *,
    policy_type: str | None = None,
) -> dict[str, Any]:
    enabled = bool(algorithm_cfg.get("gradient_checkpointing", False))
    normalized_policy_type = (policy_type or "").lower().replace("-", "_")
    report: dict[str, Any] = {
        "enabled": enabled,
        "applied_to": [],
        "input_require_grads_applied_to": [],
        "use_cache_disabled_on": [],
        "errors": [],
        "warnings": [],
        "skipped": False,
    }
    if not enabled:
        return report
    if normalized_policy_type == "smolvla":
        report["skipped"] = True
        report["warnings"].append(
            "generic gradient checkpointing is skipped for SmolVLA because it can "
            "disable cache assumptions required by rollout/eval sampling; use a "
            "SmolVLA-specific train-only checkpointing mode instead"
        )
        return report

    for name, target in _gradient_checkpointing_targets(policy):
        config = getattr(target, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            try:
                setattr(config, "use_cache", False)
                report["use_cache_disabled_on"].append(name)
            except (
                Exception
            ) as exc:  # pragma: no cover - defensive for remote-code configs.
                report["errors"].append(f"{name}.config.use_cache: {exc}")

        enable_input_require_grads = getattr(target, "enable_input_require_grads", None)
        if callable(enable_input_require_grads):
            try:
                enable_input_require_grads()
                report["input_require_grads_applied_to"].append(name)
            except Exception as exc:  # pragma: no cover - optional HF hook.
                report["errors"].append(f"{name}.enable_input_require_grads: {exc}")

        if hasattr(target, "gradient_checkpointing"):
            try:
                setattr(target, "gradient_checkpointing", True)
            except Exception as exc:  # pragma: no cover - optional HF attribute.
                report["errors"].append(f"{name}.gradient_checkpointing: {exc}")

        gradient_checkpointing_enable = getattr(
            target, "gradient_checkpointing_enable", None
        )
        if callable(gradient_checkpointing_enable):
            try:
                gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                report["applied_to"].append(name)
            except TypeError:
                try:
                    gradient_checkpointing_enable()
                    report["applied_to"].append(name)
                except Exception as exc:  # pragma: no cover - optional HF hook.
                    report["errors"].append(
                        f"{name}.gradient_checkpointing_enable: {exc}"
                    )
            except Exception as exc:  # pragma: no cover - optional HF hook.
                report["errors"].append(f"{name}.gradient_checkpointing_enable: {exc}")

    report["ok"] = bool(report["applied_to"]) and not bool(report["errors"])
    return report


def _gradient_checkpointing_targets(policy: Any) -> list[tuple[str, Any]]:
    queue: list[tuple[str, Any]] = [("policy", policy)]
    deduped: list[tuple[str, Any]] = []
    seen: set[int] = set()
    child_attrs = (
        "language_model",
        "vision_tower",
        "projector",
        "base_model",
        "model",
        "module",
    )

    while queue:
        name, target = queue.pop(0)
        target_id = id(target)
        if target_id in seen:
            continue
        seen.add(target_id)
        deduped.append((name, target))
        for attr in child_attrs:
            child = getattr(target, attr, None)
            if child is not None and id(child) not in seen:
                queue.append((f"{name}.{attr}", child))

    return deduped


def _configure_peft_lora(
    policy: Any,
    *,
    strategy: str,
    algorithm_cfg: dict[str, Any],
    peft_cfg: dict[str, Any],
    policy_type: str | None,
) -> tuple[Any, dict[str, Any]]:
    model = getattr(policy, "model", None)
    if model is None:
        return policy, _error_report(
            strategy, "PEFT/LoRA requires policy.model to exist"
        )
    force_float32 = bool(algorithm_cfg.get("force_trainable_float32", False))
    rank_partition = peft_cfg.get("rank_partition")
    if hasattr(model, "peft_config"):
        if rank_partition is not None:
            return policy, _error_report(
                strategy,
                "Task-partitioned LoRA requires an adapter-free SFT policy. "
                "Do not merge or reuse an existing RL adapter; start from the "
                "declared SFT checkpoint.",
            )
        _freeze_all(policy)
        trainable_names: list[str] = []
        for name, parameter in policy.named_parameters():
            if (
                "lora_" in name
                or ".modules_to_save." in name
                or "modules_to_save." in name
            ):
                parameter.requires_grad_(True)
                if force_float32 and hasattr(parameter, "data"):
                    parameter.data = parameter.data.float()
                trainable_names.append(name)
        gradient_checkpointing_report = _configure_gradient_checkpointing(
            policy, algorithm_cfg, policy_type=policy_type
        )
        report = _trainable_report(policy)
        report.update(
            {
                "ok": report["trainable_tensors"] > 0,
                "strategy": strategy,
                "method": "peft_lora_attached_adapter",
                "peft": {
                    "enabled": True,
                    "attached_adapter": True,
                    "trainable_name_count": len(trainable_names),
                },
                "patterns": [],
                "explicit_patterns": [],
                "allow_embedding_training": False,
                "force_trainable_float32": force_float32,
                "layer_selection": _layer_selection_report(
                    _peft_layer_selection(
                        policy,
                        strategy=strategy,
                        policy_type=policy_type,
                        algorithm_cfg=algorithm_cfg,
                    )
                ),
                "gradient_checkpointing": gradient_checkpointing_report,
                "warnings": [],
            }
        )
        return policy, report

    _shield_broken_transformer_engine_for_peft()
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as exc:  # pragma: no cover - optional dependency.
        return policy, _error_report(
            strategy, f"PEFT/LoRA requested but peft import failed: {exc}"
        )
    _disable_incompatible_optional_peft_dispatches()

    layer_selection = _peft_layer_selection(
        policy, strategy=strategy, policy_type=policy_type, algorithm_cfg=algorithm_cfg
    )
    configured_targets = peft_cfg.get("target_modules") or []
    if configured_targets == ["auto"]:
        base_target_modules = _default_lora_targets(strategy, policy_type)
    elif configured_targets == ["auto_full_action_expert"]:
        base_target_modules = _full_action_expert_lora_targets(strategy, policy_type)
    elif configured_targets == ["auto_full_language_model"]:
        base_target_modules = _full_language_model_lora_targets(
            strategy, policy_type
        )
    else:
        base_target_modules = configured_targets
    target_modules = _restrict_lora_targets_to_layers(
        strategy=strategy,
        policy_type=policy_type,
        target_modules=base_target_modules,
        layer_selection=layer_selection,
    )
    target_modules_restricted = target_modules != base_target_modules
    warnings: list[str] = []
    if target_modules_restricted:
        warnings.append(
            "PEFT target_modules were restricted by layer selection. Set "
            "algorithm.peft.apply_layer_selection=false if explicit target_modules "
            "must be used as-is."
        )
    if not target_modules:
        return policy, _error_report(
            strategy, "No LoRA target_modules resolved for this strategy/policy"
        )

    _freeze_all(policy)
    lora_dropout = _first_present_float(
        peft_cfg, "lora_dropout", "dropout", default=0.05
    )
    lora_rank = int(peft_cfg.get("r") or peft_cfg.get("rank") or 8)
    lora_alpha = int(peft_cfg.get("lora_alpha") or peft_cfg.get("alpha") or 16)
    common_lora_config = {
        "lora_dropout": lora_dropout,
        "bias": str(peft_cfg.get("bias") or "none"),
        "target_modules": target_modules,
        "modules_to_save": list(peft_cfg.get("modules_to_save") or []),
        "init_lora_weights": peft_cfg.get("init_lora_weights", True),
    }
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        **common_lora_config,
    )
    try:
        partition_report = None
        if rank_partition is None:
            policy.model = get_peft_model(model, lora_config)
        else:
            from art_embodied.lora_rank_partition import balanced_lora_rank_blocks

            if rank_partition.get("mode") != "task_all_active":
                raise ValueError(
                    "Unsupported LoRA rank-partition mode: "
                    f"{rank_partition.get('mode')!r}"
                )
            blocks = balanced_lora_rank_blocks(
                lora_rank,
                rank_partition.get("task_keys") or [],
            )
            first, *remaining = blocks
            first_config = LoraConfig(
                r=first.rank,
                lora_alpha=first.rank,
                **common_lora_config,
            )
            partitioned_model = get_peft_model(
                model,
                first_config,
                adapter_name=first.adapter_name,
            )
            for block in remaining:
                partitioned_model.add_adapter(
                    block.adapter_name,
                    LoraConfig(
                        r=block.rank,
                        lora_alpha=block.rank,
                        **common_lora_config,
                    ),
                )
            adapter_names = [block.adapter_name for block in blocks]
            # PeftModel.set_adapter accepts one adapter, while its native tuner
            # supports an additive list. All blocks must remain active so each
            # task loss observes the complete policy before backward routing.
            partitioned_model.base_model.set_adapter(adapter_names)
            policy.model = partitioned_model
            partition_report = {
                "mode": "task_all_active",
                "allocation": "balanced",
                "total_rank": lora_rank,
                "active_adapters": adapter_names,
                "blocks": [
                    {
                        "task_key": block.task_key,
                        "adapter_name": block.adapter_name,
                        "rank": block.rank,
                        "start": block.start,
                        "stop": block.stop,
                    }
                    for block in blocks
                ],
            }
    except Exception as exc:
        return policy, _error_report(strategy, f"get_peft_model failed: {exc}")

    initialization_report = _verify_fresh_lora_zero_delta(policy)
    if not initialization_report["ok"]:
        return policy, _error_report(
            strategy,
            "Fresh LoRA initialization changed the policy at attachment time: "
            f"{initialization_report['error']}",
        )

    warm_start_report = None
    if rank_partition is not None and rank_partition.get("warm_start_sources"):
        try:
            from art_embodied.lora_rank_partition import warm_start_partitioned_lora

            warm_start_report = warm_start_partitioned_lora(
                policy,
                blocks,
                rank_partition["warm_start_sources"],
                forward_routing=str(
                    rank_partition.get("forward_routing") or "all_active"
                ),
            )
            admission = rank_partition.get("composition_admission")
            if admission is not None:
                from art_embodied.lora_composition import (
                    validate_partitioned_lora_admission,
                )

                warm_start_report["composition_admission"] = (
                    validate_partitioned_lora_admission(
                        warm_start_report=warm_start_report,
                        admission=admission,
                    )
                )
            assert partition_report is not None
            partition_report["warm_start"] = warm_start_report
        except Exception as exc:
            return policy, _error_report(
                strategy, f"Task-partitioned LoRA warm start failed: {exc}"
            )

    if force_float32:
        for _name, parameter in policy.named_parameters():
            if bool(getattr(parameter, "requires_grad", False)) and hasattr(
                parameter, "data"
            ):
                parameter.data = parameter.data.float()

    gradient_checkpointing_report = _configure_gradient_checkpointing(
        policy, algorithm_cfg, policy_type=policy_type
    )
    report = _trainable_report(policy)
    report.update(
        {
            "ok": report["trainable_tensors"] > 0,
            "strategy": strategy,
            "method": "peft_lora",
            "peft": {
                "enabled": True,
                "r": lora_config.r,
                "lora_alpha": lora_config.lora_alpha,
                "lora_dropout": lora_config.lora_dropout,
                "bias": lora_config.bias,
                "base_target_modules": base_target_modules,
                "target_modules": target_modules,
                "target_modules_restricted_by_layer_selection": target_modules_restricted,
                "modules_to_save": list(lora_config.modules_to_save or []),
                "init_lora_weights": lora_config.init_lora_weights,
                "initialization": initialization_report,
                "rank_partition": partition_report,
            },
            "patterns": [],
            "explicit_patterns": [],
            "allow_embedding_training": False,
            "force_trainable_float32": force_float32,
            "layer_selection": _layer_selection_report(layer_selection),
            "gradient_checkpointing": gradient_checkpointing_report,
            "warnings": warnings,
        }
    )
    return policy, report


def _shield_broken_transformer_engine_for_peft() -> None:
    """Prevent optional Transformer Engine import failures from disabling PEFT.

    Recent PEFT versions opportunistically import ``transformer_engine`` to add
    LoRA dispatch support for TE layers.  ART-Embodied's OpenVLA-OFT LoRA target
    does not need TE, but shared clusters can have a TE package whose CUDA
    runtime/CUDNN libraries are not loadable in the current environment.  In that
    case PEFT import fails before it can build ordinary LoRA modules.

    If TE imports cleanly, leave it alone.  If it is absent, leave it alone.  If
    it exists but cannot import due to CUDA shared-library errors, install a
    minimal module without a ``pytorch`` attribute so PEFT's
    ``is_te_pytorch_available`` returns False.
    """

    try:
        import transformer_engine  # noqa: F401

        return
    except ModuleNotFoundError:
        return
    except Exception as exc:  # noqa: BLE001 - optional acceleration path.
        message = str(exc)
        if not any(
            fragment in message
            for fragment in ("cudart", "cudnn", "libcuda", "shared object")
        ):
            return
    dummy = types.ModuleType("transformer_engine")
    dummy.__spec__ = importlib.machinery.ModuleSpec("transformer_engine", loader=None)
    sys.modules["transformer_engine"] = dummy


def _disable_incompatible_optional_peft_dispatches() -> None:
    """Disable optional PEFT LoRA dispatches that are present but unusable.

    PEFT probes optional backends such as torchao inside ``get_peft_model``.
    If a shared cluster image has an older torchao package installed, PEFT can
    raise before it reaches ordinary Linear-layer LoRA dispatch.  We only mask
    torchao when its installed version is below PEFT's stated minimum.
    """

    try:
        version_text = importlib_metadata.version("torchao")
    except importlib_metadata.PackageNotFoundError:
        return
    try:
        major, minor, *_rest = [int(part) for part in version_text.split(".")[:2]]
    except Exception:
        return
    if (major, minor) >= (0, 16):
        return

    try:
        import peft.import_utils as peft_import_utils

        if hasattr(peft_import_utils.is_torchao_available, "cache_clear"):
            peft_import_utils.is_torchao_available.cache_clear()
        peft_import_utils.is_torchao_available = lambda: False
    except Exception:
        pass
    try:
        import peft.tuners.lora.torchao as peft_lora_torchao

        peft_lora_torchao.is_torchao_available = lambda: False
    except Exception:
        pass


def _first_present_float(config: dict[str, Any], *keys: str, default: float) -> float:
    for key in keys:
        if key in config and config[key] is not None:
            return float(config[key])
    return float(default)


def _peft_layer_selection(
    policy: Any,
    *,
    strategy: str,
    policy_type: str | None,
    algorithm_cfg: dict[str, Any],
) -> _LayerSelection:
    peft_cfg = (
        algorithm_cfg.get("peft") if isinstance(algorithm_cfg.get("peft"), dict) else {}
    )
    if peft_cfg.get("apply_layer_selection") is False:
        return _LayerSelection(enabled=False, last_n_layers=0, selected_layers=set())
    normalized = (policy_type or "").lower().replace("-", "_")
    if strategy == "pi0_fast_lora" or normalized == "pi0_fast":
        return _pi0_fast_layer_selection(policy, algorithm_cfg)
    if strategy == "openvla_oft_lora" or normalized == "openvla_oft":
        return _openvla_oft_layer_selection(policy, algorithm_cfg)
    return _LayerSelection(enabled=False, last_n_layers=0, selected_layers=set())


def _restrict_lora_targets_to_layers(
    *,
    strategy: str,
    policy_type: str | None,
    target_modules: Any,
    layer_selection: _LayerSelection,
) -> Any:
    normalized = (policy_type or "").lower().replace("-", "_")
    if not layer_selection.enabled:
        return target_modules
    layer_alternation = "|".join(
        str(index) for index in sorted(layer_selection.selected_layers)
    )
    if strategy == "pi0_fast_lora" or normalized == "pi0_fast":
        return (
            r".*paligemma_with_expert\.paligemma\.(?:model\.)?"
            rf"language_model\.layers\.(?:{layer_alternation})\.self_attn\.(?:q_proj|v_proj)"
        )
    if strategy == "openvla_oft_lora" or normalized == "openvla_oft":
        return (
            r".*language_model\.(?:model\.)?"
            rf"layers\.(?:{layer_alternation})\.self_attn\.(?:q_proj|v_proj)"
        )
    return target_modules


def _selection_predicate(
    *,
    strategy: str,
    explicit_patterns: list[str],
    policy: Any,
    allow_embeddings: bool,
    algorithm_cfg: dict[str, Any],
) -> tuple[Callable[[str, Any], bool], list[str], _LayerSelection, list[str]]:
    warnings: list[str] = []
    patterns = list(explicit_patterns)
    layer_selection = _LayerSelection(
        enabled=False, last_n_layers=0, selected_layers=set()
    )

    if strategy == "regex":
        if not patterns:
            warnings.append(
                "regex strategy selected but no trainable_parameter_patterns were provided"
            )
    elif strategy in {"pi0_action_expert", "pi05_action_expert"}:
        patterns.extend(_pi0_action_expert_patterns())
    elif strategy == "smolvla_action_expert":
        patterns.extend(_smolvla_action_expert_patterns())
    elif strategy == "smolvla_action_head":
        patterns.extend(_action_projection_patterns())
    elif strategy == "smolvla_action_out_only":
        patterns.extend(_action_out_projection_patterns())
    elif strategy == "pi0_fast_lm_late":
        layer_selection = _pi0_fast_layer_selection(policy, algorithm_cfg)
        patterns.extend(_pi0_fast_lm_late_patterns())
    elif strategy == "pi0_fast_lm_all":
        patterns.extend(_pi0_fast_lm_late_patterns())
        layer_selection = _LayerSelection(
            enabled=False, last_n_layers=0, selected_layers=set()
        )
    elif strategy == "openvla_oft_lm_late":
        layer_selection = _openvla_oft_layer_selection(policy, algorithm_cfg)
        patterns.extend(_openvla_oft_lm_late_patterns())
    elif strategy == "openvla_oft_lm_all":
        patterns.extend(_openvla_oft_lm_late_patterns())
        layer_selection = _LayerSelection(
            enabled=False, last_n_layers=0, selected_layers=set()
        )
    else:
        warnings.append(
            f"unknown trainable_parameter_strategy={strategy!r}; using explicit regex patterns only"
        )

    compiled = [re.compile(pattern) for pattern in patterns]

    def predicate(name: str, parameter: Any) -> bool:
        del parameter
        if not allow_embeddings and _looks_like_embedding_or_lm_head(name):
            return False
        if layer_selection.enabled and _is_pi0_fast_lm_layer_parameter(name):
            layer_index = _pi0_fast_layer_index(name)
            if (
                layer_index is not None
                and layer_index not in layer_selection.selected_layers
            ):
                return False
        if layer_selection.enabled and _is_openvla_oft_lm_layer_parameter(name):
            layer_index = _openvla_oft_layer_index(name)
            if (
                layer_index is not None
                and layer_index not in layer_selection.selected_layers
            ):
                return False
        return any(pattern.search(name) for pattern in compiled)

    return predicate, patterns, layer_selection, warnings


def _pi0_action_expert_patterns() -> list[str]:
    projections = "|".join(_ACTION_PROJECTIONS)
    return [
        r".*paligemma_with_expert\.gemma_expert\..*",
        rf".*(^|\.)(?:{projections})\.(?:weight|bias)$",
    ]


def _smolvla_action_expert_patterns() -> list[str]:
    return [
        r".*vlm_with_expert\.lm_expert\..*",
        *_action_projection_patterns(),
    ]


def _action_projection_patterns() -> list[str]:
    projections = "|".join(_ACTION_PROJECTIONS)
    return [rf".*(^|\.)(?:{projections})\.(?:weight|bias)$"]


def _action_out_projection_patterns() -> list[str]:
    return [r".*(^|\.)action_out_proj\.(?:weight|bias)$"]


def _pi0_fast_lm_late_patterns() -> list[str]:
    return [
        r".*paligemma_with_expert\.paligemma\.(?:model\.)?language_model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)\.(?:weight|bias)$",
        r".*paligemma_with_expert\.paligemma\.(?:model\.)?language_model\.layers\.\d+\.mlp\.(?:gate_proj|up_proj|down_proj)\.(?:weight|bias)$",
        r".*paligemma_with_expert\.paligemma\.(?:model\.)?language_model\.(?:norm|final_layernorm)\.(?:weight|bias)$",
    ]


def _default_lora_targets(strategy: str, policy_type: str | None) -> str:
    normalized = (policy_type or "").lower().replace("-", "_")
    if strategy == "pi0_fast_lora" or normalized == "pi0_fast":
        return r".*paligemma_with_expert\.paligemma\.(?:model\.)?language_model\.layers\.\d+\.self_attn\.(?:q_proj|v_proj)"
    if strategy in {
        "pi0_action_expert_lora",
        "pi05_action_expert_lora",
    } or normalized in {"pi0", "pi05"}:
        projections = "|".join(_ACTION_PROJECTIONS)
        return rf".*gemma_expert\..*\.self_attn\.(?:q_proj|v_proj)|.*(?:{projections})"
    if strategy == "smolvla_action_expert_lora" or normalized == "smolvla":
        projections = "|".join(_ACTION_PROJECTIONS)
        return rf".*lm_expert\..*\.(?:q_proj|v_proj)|.*(?:{projections})"
    if strategy in {
        "gr00t_n1d5_action_head_lora",
        "gr00t_n1d7_action_head_lora",
    } or normalized in {"gr00t_n1d5", "gr00t_n1d7"}:
        # GR00T uses Diffusers Attention/FeedForward modules inside action_head.model.
        # CategorySpecificLinear projectors store rank-3 parameters directly and
        # are intentionally excluded from PEFT's Linear-only adapter surface.
        return (
            r".*action_head\.model\..*(?:to_q|to_k|to_v|to_out\.0|"
            r"ff\.net\.0\.proj|ff\.net\.2|proj_out_1|proj_out_2|linear)"
        )
    if strategy == "openvla_oft_lora" or normalized == "openvla_oft":
        return (
            r".*language_model\.(?:model\.)?layers\.\d+\.self_attn\.(?:q_proj|v_proj)"
        )
    return ""


def _full_action_expert_lora_targets(strategy: str, policy_type: str | None) -> str:
    """Resolve every Linear projection in a policy's action expert only."""

    normalized = (policy_type or "").lower().replace("-", "_")
    if strategy == "smolvla_action_expert_lora" or normalized == "smolvla":
        expert_projections = "q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj"
        action_projections = "|".join(_ACTION_PROJECTIONS)
        return (
            rf".*lm_expert\..*\.(?:{expert_projections})"
            rf"|.*(?:{action_projections})"
        )
    raise ValueError(
        "target_modules=['auto_full_action_expert'] is only supported for "
        "the SmolVLA action-expert LoRA strategy"
    )


def _full_language_model_lora_targets(
    strategy: str, policy_type: str | None
) -> str:
    """Resolve attention and MLP projections in an autoregressive action LM."""

    normalized = (policy_type or "").lower().replace("-", "_")
    if strategy == "pi0_fast_lora" or normalized == "pi0_fast":
        return (
            r".*paligemma_with_expert\.paligemma\.(?:model\.)?"
            r"language_model\.layers\.\d+\."
            r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
            r"mlp\.(?:gate_proj|up_proj|down_proj))"
        )
    raise ValueError(
        "target_modules=['auto_full_language_model'] is only supported for "
        "the pi0-FAST language-model LoRA strategy"
    )


def _pi0_fast_layer_selection(
    policy: Any, algorithm_cfg: dict[str, Any]
) -> _LayerSelection:
    last_n = int(algorithm_cfg.get("trainable_last_n_layers") or 4)
    layer_indices = sorted(
        {
            index
            for name in _pi0_fast_layer_index_source_names(policy)
            for index in [_pi0_fast_layer_index(name)]
            if index is not None
        }
    )
    if not layer_indices:
        return _LayerSelection(
            enabled=False, last_n_layers=last_n, selected_layers=set()
        )
    selected = set(layer_indices[-max(1, last_n) :])
    return _LayerSelection(enabled=True, last_n_layers=last_n, selected_layers=selected)


def _pi0_fast_layer_index_source_names(policy: Any) -> list[str]:
    names: set[str] = set()
    named_modules = getattr(policy, "named_modules", None)
    if callable(named_modules):
        names.update(name for name, _module in named_modules())
    named_parameters = getattr(policy, "named_parameters", None)
    if callable(named_parameters):
        names.update(name for name, _parameter in named_parameters())
    model = getattr(policy, "model", None)
    model_named_modules = getattr(model, "named_modules", None)
    if callable(model_named_modules):
        names.update(f"model.{name}" for name, _module in model_named_modules())
    model_named_parameters = getattr(model, "named_parameters", None)
    if callable(model_named_parameters):
        names.update(f"model.{name}" for name, _parameter in model_named_parameters())
    return sorted(names)


def _pi0_fast_layer_index(name: str) -> int | None:
    match = re.search(r"language_model\.layers\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def _is_pi0_fast_lm_layer_parameter(name: str) -> bool:
    return (
        "paligemma_with_expert.paligemma" in name and "language_model.layers." in name
    )


def _openvla_oft_lm_late_patterns() -> list[str]:
    return [
        r".*language_model\.(?:model\.)?layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)\.(?:weight|bias)$",
        r".*language_model\.(?:model\.)?layers\.\d+\.mlp\.(?:gate_proj|up_proj|down_proj)\.(?:weight|bias)$",
        r".*language_model\.(?:model\.)?(?:norm|final_layernorm)\.(?:weight|bias)$",
    ]


def _openvla_oft_layer_selection(
    policy: Any, algorithm_cfg: dict[str, Any]
) -> _LayerSelection:
    last_n = int(algorithm_cfg.get("trainable_last_n_layers") or 1)
    layer_indices = sorted(
        {
            index
            for name in _openvla_oft_layer_index_source_names(policy)
            for index in [_openvla_oft_layer_index(name)]
            if index is not None
        }
    )
    if not layer_indices:
        return _LayerSelection(
            enabled=False, last_n_layers=last_n, selected_layers=set()
        )
    selected = set(layer_indices[-max(1, last_n) :])
    return _LayerSelection(enabled=True, last_n_layers=last_n, selected_layers=selected)


def _openvla_oft_layer_index_source_names(policy: Any) -> list[str]:
    names: set[str] = set()
    named_modules = getattr(policy, "named_modules", None)
    if callable(named_modules):
        names.update(name for name, _module in named_modules())
    named_parameters = getattr(policy, "named_parameters", None)
    if callable(named_parameters):
        names.update(name for name, _parameter in named_parameters())
    model = getattr(policy, "model", None)
    model_named_modules = getattr(model, "named_modules", None)
    if callable(model_named_modules):
        names.update(f"model.{name}" for name, _module in model_named_modules())
    model_named_parameters = getattr(model, "named_parameters", None)
    if callable(model_named_parameters):
        names.update(f"model.{name}" for name, _parameter in model_named_parameters())
    return sorted(names)


def _openvla_oft_layer_index(name: str) -> int | None:
    match = re.search(r"language_model\.(?:model\.)?layers\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def _is_openvla_oft_lm_layer_parameter(name: str) -> bool:
    return "language_model." in name and ".layers." in name


def _looks_like_embedding_or_lm_head(name: str) -> bool:
    return any(
        selector in name
        for selector in ("embed_tokens", "lm_head", "embed_language_tokens")
    )


def _freeze_all(policy: Any) -> None:
    for _name, parameter in policy.named_parameters():
        parameter.requires_grad_(False)


def _apply_trainable_predicate(
    policy: Any,
    *,
    predicate: Callable[[str, Any], bool],
    force_float32: bool,
) -> dict[str, Any]:
    for name, parameter in policy.named_parameters():
        is_trainable = bool(predicate(name, parameter))
        parameter.requires_grad_(is_trainable)
        if is_trainable and force_float32 and hasattr(parameter, "data"):
            parameter.data = parameter.data.float()
    report = _trainable_report(policy)
    report["ok"] = report["trainable_tensors"] > 0
    return report


def _trainable_report(policy: Any) -> dict[str, Any]:
    matched: list[dict[str, Any]] = []
    name_sample: list[str] = []
    trainable_tensors = 0
    trainable_parameters = 0
    total_parameters = 0
    dtype_summary: dict[str, dict[str, int]] = {}
    for name, parameter in policy.named_parameters():
        if len(name_sample) < 80:
            name_sample.append(name)
        numel = int(parameter.numel())
        total_parameters += numel
        dtype = str(parameter.dtype)
        dtype_row = dtype_summary.setdefault(
            dtype,
            {
                "tensors": 0,
                "parameters": 0,
                "trainable_tensors": 0,
                "trainable_parameters": 0,
            },
        )
        dtype_row["tensors"] += 1
        dtype_row["parameters"] += numel
        if bool(getattr(parameter, "requires_grad", False)):
            trainable_tensors += 1
            trainable_parameters += numel
            dtype_row["trainable_tensors"] += 1
            dtype_row["trainable_parameters"] += numel
            if len(matched) < 40:
                matched.append(
                    {
                        "name": name,
                        "shape": list(parameter.shape),
                        "numel": numel,
                        "dtype": str(parameter.dtype),
                    }
                )
    return {
        "ok": trainable_tensors > 0,
        "policy_class": type(policy).__name__,
        "trainable_tensors": trainable_tensors,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_fraction": float(trainable_parameters / total_parameters)
        if total_parameters
        else 0.0,
        "dtype_summary": dict(sorted(dtype_summary.items())),
        "matched_sample": matched,
        "parameter_name_sample": name_sample,
    }


def _verify_fresh_lora_zero_delta(policy: Any) -> dict[str, Any]:
    """Verify PEFT's identity-at-attachment invariant for linear LoRA layers.

    ART initializes LoRA-A from either Kaiming or a Gaussian distribution while
    LoRA-B must be exactly zero. A nonzero B matrix changes rollout behavior
    before the first optimizer step and invalidates the Step 0 baseline.
    """

    zero_factors: list[tuple[str, Any]] = [
        (name, parameter)
        for name, parameter in policy.named_parameters()
        if "lora_B" in name
    ]
    if not zero_factors:
        return {
            "ok": False,
            "checked_tensors": 0,
            "max_abs": None,
            "error": "no LoRA-B tensors were found after PEFT attachment",
        }

    max_abs = 0.0
    nonzero_names: list[str] = []
    for name, parameter in zero_factors:
        tensor_max = float(parameter.detach().abs().max().item())
        max_abs = max(max_abs, tensor_max)
        if tensor_max != 0.0:
            nonzero_names.append(name)
    if nonzero_names:
        return {
            "ok": False,
            "checked_tensors": len(zero_factors),
            "max_abs": max_abs,
            "nonzero_name_sample": nonzero_names[:20],
            "error": f"{len(nonzero_names)} LoRA-B tensors are nonzero",
        }
    return {
        "ok": True,
        "checked_tensors": len(zero_factors),
        "max_abs": 0.0,
        "nonzero_name_sample": [],
        "error": None,
    }


def _layer_selection_report(selection: _LayerSelection) -> dict[str, Any] | None:
    if not selection.enabled:
        return None
    return {
        "last_n_layers": selection.last_n_layers,
        "selected_layers": sorted(selection.selected_layers),
    }


def _error_report(strategy: str, message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "strategy": strategy,
        "method": None,
        "error": message,
        "trainable_tensors": 0,
        "trainable_parameters": 0,
        "total_parameters": 0,
        "trainable_fraction": 0.0,
        "dtype_summary": {},
        "matched_sample": [],
        "parameter_name_sample": [],
        "warnings": [message],
    }
