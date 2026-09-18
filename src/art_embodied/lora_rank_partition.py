"""Task-partitioned LoRA contracts with all rank blocks active in forward passes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

_ADAPTER_PARAMETER = re.compile(
    r"(?:^|\.)lora_(?:A|B|embedding_A|embedding_B)\.([^.]+)\."
)


@dataclass(frozen=True, slots=True)
class LoraRankBlock:
    """One task-owned rank interval represented by one native PEFT adapter."""

    task_key: str
    adapter_name: str
    rank: int
    start: int
    stop: int


def balanced_lora_rank_blocks(
    total_rank: int,
    task_keys: Sequence[str],
) -> tuple[LoraRankBlock, ...]:
    """Assign remainder ranks to earlier tasks deterministically.

    For rank 128 and ten tasks this yields eight blocks of rank 13 followed by
    two blocks of rank 12. The contiguous intervals document equivalence to a
    single concatenated LoRA even though PEFT stores each block independently.
    """

    keys = tuple(str(value) for value in task_keys)
    if total_rank < 1:
        raise ValueError("LoRA total rank must be positive")
    if len(keys) < 2:
        raise ValueError("Task-partitioned LoRA requires at least two task keys")
    if len(set(keys)) != len(keys) or any(not key for key in keys):
        raise ValueError("Task-partitioned LoRA task keys must be unique and non-empty")
    if total_rank < len(keys):
        raise ValueError("LoRA total rank must provide at least one rank per task")

    quotient, remainder = divmod(total_rank, len(keys))
    blocks = []
    start = 0
    for index, task_key in enumerate(keys):
        rank = quotient + int(index < remainder)
        stop = start + rank
        blocks.append(
            LoraRankBlock(
                task_key=task_key,
                adapter_name=f"task_{index:03d}",
                rank=rank,
                start=start,
                stop=stop,
            )
        )
        start = stop
    assert start == total_rank
    return tuple(blocks)


def rank_blocks_from_lora_config(lora: Any) -> tuple[LoraRankBlock, ...]:
    """Resolve the strict config model without coupling helpers to Pydantic."""

    partition = getattr(lora, "rank_partition", None)
    if partition is None:
        return ()
    if getattr(partition, "mode", None) != "task_all_active":
        raise ValueError(f"Unsupported LoRA rank-partition mode: {partition.mode!r}")
    return balanced_lora_rank_blocks(int(lora.rank), partition.task_keys)


def task_adapter_map(blocks: Iterable[LoraRankBlock]) -> dict[str, str]:
    return {block.task_key: block.adapter_name for block in blocks}


def warm_start_partitioned_lora(
    policy: Any,
    blocks: Sequence[LoraRankBlock],
    source_by_task: Mapping[str, Mapping[str, Any]],
    *,
    forward_routing: str = "all_active",
) -> dict[str, Any]:
    """Load one complete single-adapter checkpoint into each task rank block.

    Source checkpoints are validated before any source weights are loaded. The
    optimizer is intentionally outside this operation: callers must construct a
    fresh optimizer after composition rather than importing incompatible moments.
    """

    expected_tasks = {block.task_key for block in blocks}
    provided_tasks = {str(task_key) for task_key in source_by_task}
    if provided_tasks != expected_tasks:
        raise ValueError(
            "Warm-start checkpoints must cover every LoRA rank block exactly; "
            f"missing={sorted(expected_tasks - provided_tasks)}, "
            f"unexpected={sorted(provided_tasks - expected_tasks)}"
        )
    resolved = {
        str(task_key): {
            **dict(source),
            "checkpoint": Path(source["checkpoint"]).expanduser().resolve(),
            "source_config": Path(source["source_config"]).expanduser().resolve(),
        }
        for task_key, source in source_by_task.items()
    }
    checkpoints = [source["checkpoint"] for source in resolved.values()]
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("Warm-start checkpoints must be unique per task")

    model = getattr(policy, "model", None)
    if model is None or not hasattr(model, "peft_config"):
        raise ValueError("Partitioned LoRA warm start requires an attached PEFT model")

    descriptors = [
        _validate_warm_start_checkpoint(policy, model, block, resolved[block.task_key])
        for block in blocks
    ]

    from peft.utils.save_and_load import (
        get_peft_model_state_dict,
        load_peft_weights,
        set_peft_model_state_dict,
    )

    loaded = []
    for block, descriptor in zip(blocks, descriptors, strict=True):
        state = load_peft_weights(str(descriptor["policy_path"]), device="cpu")
        result = set_peft_model_state_dict(
            model,
            state,
            adapter_name=block.adapter_name,
        )
        unexpected = list(getattr(result, "unexpected_keys", ()) or ())
        if unexpected:
            raise RuntimeError(
                f"Warm-start adapter {block.task_key!r} has unexpected keys: "
                f"{unexpected[:20]}"
            )
        installed = get_peft_model_state_dict(model, adapter_name=block.adapter_name)
        if set(installed) != set(state):
            raise RuntimeError(
                f"Warm-start adapter {block.task_key!r} parameter names differ after "
                "loading"
            )
        mismatched = []
        for name, source_tensor in state.items():
            target_tensor = installed[name].detach().to(device="cpu")
            if not source_tensor.detach().to(device="cpu").equal(target_tensor):
                mismatched.append(name)
        if mismatched:
            raise RuntimeError(
                f"Warm-start adapter {block.task_key!r} was not copied exactly: "
                f"{mismatched[:20]}"
            )
        loaded.append(
            {
                "task_key": block.task_key,
                "adapter_name": block.adapter_name,
                "rank": block.rank,
                "checkpoint": str(descriptor["checkpoint"]),
                "source_config": str(descriptor["source_config"]),
                "final_step": descriptor["final_step"],
                "checkpoint_manifest_sha256": descriptor["checkpoint_manifest_sha256"],
                "source_config_sha256": descriptor["source_config_sha256"],
                "adapter_config_sha256": descriptor["adapter_config_sha256"],
                "adapter_weights_sha256": descriptor["adapter_weights_sha256"],
                "source_kind": descriptor["source_kind"],
                **descriptor["evidence"],
                "parameter_tensors": len(state),
                "exact_copy_verified": True,
            }
        )
        del installed, state

    adapter_names = [block.adapter_name for block in blocks]
    if forward_routing == "all_active":
        model.base_model.set_adapter(adapter_names)
        active_adapters = adapter_names
    elif forward_routing == "task_owned":
        model.base_model.set_adapter(adapter_names[0])
        # PEFT deactivates gradients for non-active adapters. The coordinator
        # optimizer must still own every block because later task-homogeneous
        # worker gradients update different adapters in the same global step.
        enable_partitioned_adapter_gradients(policy, adapter_names)
        active_adapters = [adapter_names[0]]
    else:
        raise ValueError(
            f"Unsupported LoRA rank-partition forward routing: {forward_routing!r}"
        )
    policy._art_task_adapter_map = task_adapter_map(blocks)
    policy._art_partition_forward_routing = forward_routing
    return {
        "enabled": True,
        "source": "complete_single_task_checkpoints",
        "optimizer_state_imported": False,
        "forward_routing": forward_routing,
        "active_adapters": active_adapters,
        "blocks": loaded,
    }


def _validate_warm_start_checkpoint(
    policy: Any,
    model: Any,
    block: LoraRankBlock,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = Path(source["checkpoint"])
    source_config_path = Path(source["source_config"])
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError(
            f"Warm-start checkpoint must be a real directory: {checkpoint}"
        )
    marker_path = checkpoint / "art_embodied_checkpoint_complete.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Warm-start checkpoint has no valid completion marker: {marker_path}"
        ) from exc
    marker_sha256 = _sha256(marker_path)
    if marker_sha256 != source["checkpoint_manifest_sha256"]:
        raise ValueError(
            f"Warm-start checkpoint manifest SHA256 mismatch: {marker_path}"
        )
    if source_config_path.is_symlink() or not source_config_path.is_file():
        raise ValueError(
            f"Warm-start source config must be a regular file: {source_config_path}"
        )
    source_config_sha256 = _sha256(source_config_path)
    if source_config_sha256 != source["source_config_sha256"]:
        raise ValueError(
            f"Warm-start source config SHA256 mismatch: {source_config_path}"
        )

    from art_embodied.config import EmbodiedExperimentConfig

    source_config = EmbodiedExperimentConfig.from_yaml(source_config_path)
    source_kind = str(source.get("source_kind") or "validated_grpo")
    expected_policy_type = {
        "validated_grpo": "gr00t_n1d7",
        "partition_sft": "pi0_fast",
    }.get(source_kind)
    if expected_policy_type is None:
        raise ValueError(f"Unsupported warm-start source kind: {source_kind!r}")
    if source_config.policy.type != expected_policy_type:
        raise ValueError(
            "Warm-start source policy does not match its evidence kind: "
            f"kind={source_kind!r}, policy={source_config.policy.type!r}"
        )
    if source_config.policy.lora.rank_partition is not None:
        raise ValueError("Warm-start source must be a single unpartitioned LoRA")
    if (
        source_config.policy.lora.rank != block.rank
        or source_config.policy.lora.alpha != block.rank
    ):
        raise ValueError(
            f"Warm-start source rank mismatch for {block.task_key!r}: "
            f"rank={source_config.policy.lora.rank}, "
            f"alpha={source_config.policy.lora.alpha}, target={block.rank}"
        )
    final_step = int(source["final_step"])
    if source_config.training.updates != final_step:
        raise ValueError(
            "Warm-start source config final update does not match final_step: "
            f"config={source_config.training.updates}, declared={final_step}"
        )
    if marker.get("config_fingerprint") != source_config.fingerprint:
        raise ValueError("Warm-start checkpoint does not belong to its source config")
    if marker.get("metadata", {}).get("step") != final_step:
        raise ValueError(
            "Warm-start checkpoint step does not match final_step: "
            f"checkpoint={marker.get('metadata', {}).get('step')!r}, "
            f"declared={final_step}"
        )
    resume_contract = marker.get("resume_contract_fingerprint")
    if not isinstance(resume_contract, str) or not resume_contract:
        raise ValueError(f"Warm-start checkpoint has no resume contract: {marker_path}")
    if resume_contract != source_config.resume_contract_fingerprint:
        raise ValueError(
            "Warm-start checkpoint resume contract does not match its source config"
        )

    from art_embodied.checkpointing import CheckpointManager

    CheckpointManager().validate(
        checkpoint,
        expected_resume_contract_fingerprint=resume_contract,
        require_training_state=False,
    )
    if source_kind == "validated_grpo":
        source_tasks = source_config.environment.kwargs.get("task_ids")
        if source_tasks != [block.task_key]:
            raise ValueError(
                f"Warm-start source task mismatch for {block.task_key!r}: "
                f"source={source_tasks!r}"
            )
        evidence = _validate_source_adjudications(
            source=source,
            checkpoint=checkpoint,
            marker_sha256=marker_sha256,
            source_config_path=source_config_path,
            source_config_sha256=source_config_sha256,
            resume_contract=resume_contract,
        )
        snapshot_name = "art_embodied_gr00t_n1d7_snapshot.json"
    else:
        evidence = _validate_partition_sft_completion(
            source=source,
            checkpoint=checkpoint,
            marker_sha256=marker_sha256,
            source_config=source_config,
            source_config_path=source_config_path,
            source_config_sha256=source_config_sha256,
            task_key=block.task_key,
            final_step=final_step,
        )
        snapshot_name = "art_embodied_pi0_fast_snapshot.json"
    policy_path = checkpoint / "policy"
    config_path = policy_path / "adapter_config.json"
    weights_path = policy_path / "adapter_model.safetensors"
    snapshot_path = policy_path / snapshot_name
    for required in (config_path, weights_path, snapshot_path):
        if required.is_symlink() or not required.is_file():
            raise ValueError(f"Warm-start checkpoint is missing {required}")

    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if snapshot.get("format") != "peft_adapter" or snapshot.get("adapter_names") != [
        "default"
    ]:
        raise ValueError(
            "Warm-start source must contain exactly one default PEFT adapter: "
            f"{snapshot_path}"
        )
    source_snapshot_mismatches = {}
    source_snapshot_expected = {
        "model_id": str(source_config.policy.path),
        "revision": source_config.policy.revision,
    }
    if source_kind == "validated_grpo":
        source_snapshot_expected.update(
            {
                "checkpoint_subfolder": source_config.policy.load_kwargs.get(
                    "checkpoint_subfolder"
                ),
                "embodiment_tag": source_config.policy.load_kwargs.get(
                    "embodiment_tag"
                ),
                "processor_action_horizon": source_config.policy.load_kwargs.get(
                    "processor_action_horizon"
                ),
            }
        )
    for field, expected in source_snapshot_expected.items():
        if snapshot.get(field) != expected:
            source_snapshot_mismatches[field] = {
                "snapshot": snapshot.get(field),
                "source_config": expected,
            }
    if source_snapshot_mismatches:
        raise ValueError(
            "Warm-start source snapshot does not match its source config: "
            f"{source_snapshot_mismatches}"
        )
    target_config = model.peft_config.get(block.adapter_name)
    if target_config is None:
        raise ValueError(f"Target PEFT adapter is missing: {block.adapter_name}")
    geometry_mismatches = {}
    expected_geometry = {
        "peft_type": "LORA",
        "r": block.rank,
        "lora_alpha": block.rank,
        "lora_dropout": float(getattr(target_config, "lora_dropout", 0.0)),
        "bias": "none",
        "modules_to_save": list(getattr(target_config, "modules_to_save", None) or []),
        "target_modules": getattr(target_config, "target_modules", None),
    }
    for key, expected in expected_geometry.items():
        observed = adapter_config.get(key)
        if key == "target_modules":
            observed = _canonical_target_modules(observed)
            expected = _canonical_target_modules(expected)
        elif key == "modules_to_save":
            observed = list(observed or [])
        if observed != expected:
            geometry_mismatches[key] = {"source": observed, "target": expected}
    if geometry_mismatches:
        raise ValueError(
            f"Warm-start LoRA geometry mismatch for {block.task_key!r}: "
            f"{geometry_mismatches}"
        )
    if float(adapter_config["lora_alpha"]) / int(adapter_config["r"]) != 1.0:
        raise ValueError("Warm-start LoRA scaling must be alpha/rank=1")

    contract_fields = (
        "family",
        "model_id",
        "revision",
        "checkpoint_subfolder",
        "embodiment_tag",
        "processor_action_horizon",
    )
    contract_mismatches = {}
    for field in contract_fields:
        if hasattr(policy, field):
            expected = getattr(policy, field)
            if snapshot.get(field) != expected:
                contract_mismatches[field] = {
                    "source": snapshot.get(field),
                    "target": expected,
                }
    if hasattr(policy, "action_components"):
        expected_components = [list(value) for value in policy.action_components]
        if snapshot.get("action_components") != expected_components:
            contract_mismatches["action_components"] = {
                "source": snapshot.get("action_components"),
                "target": expected_components,
            }
    if contract_mismatches:
        raise ValueError(
            f"Warm-start base policy contract mismatch for {block.task_key!r}: "
            f"{contract_mismatches}"
        )
    base_model = adapter_config.get("base_model_name_or_path")
    if hasattr(policy, "model_id") and base_model != policy.model_id:
        raise ValueError(
            f"Warm-start base SFT mismatch for {block.task_key!r}: "
            f"source={base_model!r}, target={policy.model_id!r}"
        )
    return {
        "checkpoint": checkpoint,
        "policy_path": policy_path,
        "source_config": source_config_path,
        "final_step": final_step,
        "checkpoint_manifest_sha256": marker_sha256,
        "source_config_sha256": source_config_sha256,
        "adapter_config_sha256": _sha256(config_path),
        "adapter_weights_sha256": _sha256(weights_path),
        "source_kind": source_kind,
        "evidence": evidence,
    }


def _validate_partition_sft_completion(
    *,
    source: Mapping[str, Any],
    checkpoint: Path,
    marker_sha256: str,
    source_config: Any,
    source_config_path: Path,
    source_config_sha256: str,
    task_key: str,
    final_step: int,
) -> dict[str, str]:
    completion_path = Path(source["sft_completion"])
    if completion_path.is_symlink() or not completion_path.is_file():
        raise ValueError(
            f"Warm-start SFT completion must be a regular file: {completion_path}"
        )
    completion_sha256 = _sha256(completion_path)
    if completion_sha256 != source["sft_completion_sha256"]:
        raise ValueError(
            f"Warm-start SFT completion SHA256 mismatch: {completion_path}"
        )
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    config_ref = completion.get("source_config") or {}
    checkpoint_ref = completion.get("checkpoint") or {}
    wandb_ref = completion.get("wandb") or {}
    source_tasks = source_config.environment.kwargs.get("task_ids")
    if (
        completion.get("kind") != "pi0_fast_partition_sft_completion"
        or completion.get("status") != "passed"
        or completion.get("task_key") != task_key
        or completion.get("final_step") != final_step
        or source_tasks != [completion.get("task_index")]
        or Path(config_ref.get("path", "")).expanduser().resolve() != source_config_path
        or config_ref.get("sha256") != source_config_sha256
        or Path(checkpoint_ref.get("path", "")).expanduser().resolve() != checkpoint
        or checkpoint_ref.get("marker_sha256") != marker_sha256
        or wandb_ref.get("history_readback_verified") is not True
        or not isinstance(wandb_ref.get("run_url"), str)
        or not wandb_ref["run_url"]
        or completion.get("finite_loss_verified") is not True
    ):
        raise ValueError(
            f"Warm-start source did not pass partition SFT completion: {completion_path}"
        )
    return {
        "sft_completion": str(completion_path),
        "sft_completion_sha256": completion_sha256,
    }


def _validate_source_adjudications(
    *,
    source: Mapping[str, Any],
    checkpoint: Path,
    marker_sha256: str,
    source_config_path: Path,
    source_config_sha256: str,
    resume_contract: str,
) -> dict[str, str]:
    development_path = Path(source["development_adjudication"])
    sealed_path = Path(source["sealed_adjudication"])
    for label, path, expected_sha256 in (
        (
            "development",
            development_path,
            source["development_adjudication_sha256"],
        ),
        ("sealed", sealed_path, source["sealed_adjudication_sha256"]),
    ):
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"Warm-start {label} adjudication must be a regular file: {path}"
            )
        if _sha256(path) != expected_sha256:
            raise ValueError(f"Warm-start {label} adjudication SHA256 mismatch: {path}")

    development = json.loads(development_path.read_text(encoding="utf-8"))
    selected_checkpoint = development.get("selected_checkpoint")
    development_config = development.get("config") or {}
    if (
        development.get("kind") != "gr00t_n1d7_robocasa_one_task_u100_adjudication"
        or development.get("status") != "passed"
        or development.get("development_lift_established") is not True
        or development.get("sealed_eligible") is not True
        or not isinstance(selected_checkpoint, str)
        or Path(selected_checkpoint).expanduser().resolve() != checkpoint
        or development_config.get("sha256") != source_config_sha256
        or Path(development_config.get("path", "")).expanduser().resolve()
        != source_config_path
        or development_config.get("resume_contract_fingerprint") != resume_contract
    ):
        raise ValueError(
            "Warm-start source did not pass the required development adjudication"
        )

    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    sealed_checkpoint = sealed.get("candidate_checkpoint") or {}
    if (
        sealed.get("kind") != "gr00t_n1d7_robocasa_one_task_u100_sealed_adjudication"
        or sealed.get("status") != "passed"
        or sealed.get("no_post_sealed_tuning") is not True
        or Path(sealed_checkpoint.get("path", "")).expanduser().resolve() != checkpoint
        or sealed_checkpoint.get("marker_sha256") != marker_sha256
    ):
        raise ValueError(
            "Warm-start source did not pass the required sealed adjudication"
        )
    return {
        "development_adjudication": str(development_path),
        "development_adjudication_sha256": _sha256(development_path),
        "sealed_adjudication": str(sealed_path),
        "sealed_adjudication_sha256": _sha256(sealed_path),
    }


def _canonical_target_modules(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(sorted(str(item) for item in (value or ())))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_from_parameter_name(name: str) -> str | None:
    """Return a PEFT adapter name from one LoRA parameter name."""

    match = _ADAPTER_PARAMETER.search(str(name))
    return match.group(1) if match is not None else None


def enable_partitioned_adapter_gradients(
    policy: Any,
    adapter_names: Sequence[str],
) -> None:
    """Keep every partition optimizer-owned after PEFT selects one forward adapter."""

    configured = {str(name) for name in adapter_names}
    for name, parameter in policy.named_parameters():
        if adapter_from_parameter_name(str(name)) in configured:
            parameter.requires_grad_(True)


def require_partitioned_trainable_parameters(
    policy: Any,
    blocks: Sequence[LoraRankBlock],
) -> dict[str, str]:
    """Verify that every trainable tensor belongs to exactly one rank block."""

    expected = {block.adapter_name for block in blocks}
    parameter_adapters: dict[str, str] = {}
    unknown = []
    observed = set()
    for name, parameter in policy.named_parameters():
        if not bool(getattr(parameter, "requires_grad", False)):
            continue
        adapter = adapter_from_parameter_name(name)
        if adapter is None or adapter not in expected:
            unknown.append(str(name))
            continue
        parameter_adapters[str(name)] = adapter
        observed.add(adapter)
    if unknown:
        preview = ", ".join(unknown[:5])
        raise ValueError(
            "Task-partitioned LoRA requires an adapter-only trainable surface; "
            f"unowned parameters: {preview}"
        )
    missing = sorted(expected.difference(observed))
    if missing:
        raise ValueError(
            f"Task-partitioned LoRA adapters have no parameters: {missing}"
        )
    return parameter_adapters


class TaskAdapterGradientGate:
    """Keep all adapters active in forward while routing backward by task.

    Hooks run before gradients accumulate into ``Parameter.grad``. Therefore a
    task contributes only to its own adapter block, while the loss still sees
    the summed output of every adapter and can account for their interactions.
    """

    def __init__(
        self,
        policy: Any,
        blocks: Sequence[LoraRankBlock],
        *,
        forward_routing: str = "all_active",
    ) -> None:
        self._policy = policy
        self._forward_routing = str(forward_routing)
        self._task_to_adapter = task_adapter_map(blocks)
        self._active_adapter: str | None = None
        self._handles = []
        if self._forward_routing == "task_owned":
            enable_partitioned_adapter_gradients(
                policy,
                tuple(self._task_to_adapter.values()),
            )
        parameter_adapters = require_partitioned_trainable_parameters(policy, blocks)
        for name, parameter in policy.named_parameters():
            adapter = parameter_adapters.get(str(name))
            if adapter is None:
                continue
            self._handles.append(parameter.register_hook(self._make_hook(adapter)))

    @property
    def adapter_names(self) -> frozenset[str]:
        return frozenset(self._task_to_adapter.values())

    def activate(self, task_key: str) -> str:
        try:
            adapter = self._task_to_adapter[str(task_key)]
        except KeyError as exc:
            raise ValueError(
                f"No LoRA rank block is configured for task {task_key!r}"
            ) from exc
        self._active_adapter = adapter
        if self._forward_routing == "task_owned":
            model = getattr(self._policy, "model", None)
            base_model = getattr(model, "base_model", None)
            set_adapter = getattr(base_model, "set_adapter", None)
            if not callable(set_adapter):
                raise RuntimeError(
                    "Task-owned forward routing requires PEFT set_adapter"
                )
            set_adapter(adapter)
            enable_partitioned_adapter_gradients(
                self._policy,
                tuple(self._task_to_adapter.values()),
            )
        elif self._forward_routing != "all_active":
            raise ValueError(
                "Unsupported LoRA rank-partition forward routing: "
                f"{self._forward_routing!r}"
            )
        return adapter

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._active_adapter = None

    def _make_hook(self, parameter_adapter: str):
        def route(gradient: Any) -> Any:
            if self._active_adapter is None:
                raise RuntimeError(
                    "Task-partitioned LoRA backward ran without an active task"
                )
            if parameter_adapter == self._active_adapter:
                return gradient
            return gradient.new_zeros(gradient.shape)

        return route
