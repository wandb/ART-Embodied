"""Pinned NVIDIA GR00T N1.7 policy plugin for Flow-SDE GRPO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .flow_policy import FlowSDERollout, GR00TFlowModelInputs
from .flow_sde import FlowSDESchedule
from .gr00t import _native_gr00t_model, _replace_dropout_with_identity
from .gr00t_n1d7_flow_sde import GR00TN17FlowSDEBridge


def _resolve_checkpoint_path(
    *,
    model_id: str,
    revision: str | None,
    checkpoint_subfolder: str,
    snapshot_download: Any,
) -> Path:
    """Resolve either a repository-root or nested N1.7 checkpoint."""

    source = Path(model_id).expanduser()
    checkpoint_at_root = checkpoint_subfolder == "."
    if source.exists():
        model_path = source if checkpoint_at_root else source / checkpoint_subfolder
    else:
        download_kwargs: dict[str, Any] = {
            "repo_id": model_id,
            "revision": revision,
            "repo_type": "model",
        }
        if not checkpoint_at_root:
            download_kwargs.update(
                {
                    "allow_patterns": (f"{checkpoint_subfolder}/**",),
                    "ignore_patterns": (f"{checkpoint_subfolder}/global_step*/**",),
                }
            )
        snapshot = Path(snapshot_download(**download_kwargs))
        model_path = snapshot if checkpoint_at_root else snapshot / checkpoint_subfolder
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"GR00T N1.7 checkpoint subfolder does not exist: {model_path}"
        )
    return model_path


class GR00TN17FlowPolicy:
    """Own the official N1.7 runtime while exposing ART's flow-policy API."""

    family = "gr00t_n1d7"

    def __init__(
        self,
        *,
        model_id: str,
        revision: str | None,
        checkpoint_subfolder: str,
        device: str,
        embodiment_tag: str,
        execution_horizon: int,
        processor_action_horizon: int,
        action_dim: int,
        action_components: tuple[tuple[str, int, bool, int | None], ...],
        model_action_horizon: int,
        schedule: FlowSDESchedule,
        disable_dropout: bool,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.checkpoint_subfolder = checkpoint_subfolder
        self.device = device
        self.embodiment_tag = embodiment_tag
        self.execution_horizon = int(execution_horizon)
        self.processor_action_horizon = int(processor_action_horizon)
        self.action_dim = int(action_dim)
        self.action_components = tuple(action_components)
        self.model_action_horizon = int(model_action_horizon)
        self.schedule = schedule
        self.disable_dropout = bool(disable_dropout)
        self.native_policy: Any | None = None
        self.bridge: GR00TN17FlowSDEBridge | None = None
        self.trainable_report: dict[str, Any] | None = None

    @property
    def model(self) -> Any:
        self._require_loaded()
        assert self.native_policy is not None
        return self.native_policy.model

    @model.setter
    def model(self, value: Any) -> None:
        self._require_loaded()
        assert self.native_policy is not None
        self.native_policy.model = value
        action_head = _native_gr00t_model(value).action_head
        self.bridge = GR00TN17FlowSDEBridge(
            action_head,
            schedule=self.schedule,
            execution_horizon=self.execution_horizon,
            action_dim=self.action_dim,
        )

    def load(self) -> None:
        if self.native_policy is not None:
            return
        from .transformers_compat import ensure_art_mask_patch_compatibility

        ensure_art_mask_patch_compatibility()
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        from huggingface_hub import snapshot_download

        model_path = _resolve_checkpoint_path(
            model_id=self.model_id,
            revision=self.revision,
            checkpoint_subfolder=self.checkpoint_subfolder,
            snapshot_download=snapshot_download,
        )
        native_policy = Gr00tPolicy(
            model_path=str(model_path),
            embodiment_tag=self.embodiment_tag,
            device=self.device,
            strict=True,
        )
        model = native_policy.model
        action_head = _native_gr00t_model(model).action_head
        model_config = action_head.config
        if int(model_config.action_horizon) != self.model_action_horizon:
            raise RuntimeError(
                "N1.7 model action horizon differs from YAML: "
                f"model={model_config.action_horizon}, "
                f"declared={self.model_action_horizon}"
            )
        processor_horizon = len(native_policy.modality_configs["action"].delta_indices)
        if processor_horizon != self.processor_action_horizon:
            raise RuntimeError(
                "N1.7 processor action horizon differs from YAML: "
                f"processor={processor_horizon}, "
                f"declared={self.processor_action_horizon}"
            )
        if self.action_dim > int(action_head.action_dim):
            raise RuntimeError("N1.7 action_dim exceeds the loaded action head")
        processor_keys = tuple(native_policy.modality_configs["action"].modality_keys)
        declared_keys = tuple(
            key for key, _size, _executed, _order in self.action_components
        )
        if processor_keys != declared_keys:
            raise RuntimeError(
                "N1.7 processor action keys differ from YAML: "
                f"processor={processor_keys}, declared={declared_keys}"
            )
        if self.disable_dropout:
            _replace_dropout_with_identity(model)
        elif any(isinstance(module, torch.nn.Dropout) for module in model.modules()):
            raise RuntimeError("N1.7 Flow-SDE requires dropout to be disabled")
        self.native_policy = native_policy
        self.bridge = GR00TN17FlowSDEBridge(
            action_head,
            schedule=self.schedule,
            execution_horizon=self.execution_horizon,
            action_dim=self.action_dim,
        )

    @torch.no_grad()
    def prepare_flow_inputs(
        self,
        native_input: dict[str, Any],
    ) -> GR00TFlowModelInputs:
        """Cache only frozen N1.7 vision-language conditioning."""

        self._require_loaded()
        model = _native_gr00t_model(self.model)
        _require_frozen_n1d7_conditioner(model)
        collated = self._collate_native_input(native_input)
        raw_inputs = collated.get("inputs")
        if raw_inputs is None:
            raise KeyError("N1.7 collator output is missing 'inputs'")
        backbone_input, action_input = model.prepare_input(raw_inputs)
        backbone_output = model.backbone(backbone_input)
        backbone_output = model.action_head.process_backbone_output(backbone_output)
        return GR00TFlowModelInputs(
            vision_language_features=backbone_output.backbone_features,
            vision_language_attention_mask=backbone_output.backbone_attention_mask,
            state=action_input.state,
            embodiment_id=action_input.embodiment_id,
            image_mask=getattr(backbone_output, "image_mask", None),
            model_family=self.family,
        )

    def sample_flow_sde(
        self,
        inputs: GR00TFlowModelInputs,
        **kwargs: Any,
    ) -> FlowSDERollout:
        self._require_loaded()
        assert self.bridge is not None
        return self.bridge.sample(inputs, **kwargs)

    def flow_sde_logprobs(self, rollout: FlowSDERollout) -> torch.Tensor:
        self._require_loaded()
        assert self.bridge is not None
        return self.bridge.rescore(rollout)

    @torch.no_grad()
    def flow_sde_reference_logprobs(
        self,
        rollout: FlowSDERollout,
    ) -> torch.Tensor:
        """Rescore a sampled transition under the immutable SFT base model."""

        self._require_loaded()
        assert self.bridge is not None
        disable_adapter = getattr(self.model, "disable_adapter", None)
        if not callable(disable_adapter):
            raise RuntimeError(
                "N1.7 SFT-reference scoring requires a PEFT model with "
                "disable_adapter()"
            )
        with disable_adapter():
            return self.bridge.rescore(rollout).detach()

    def sft_replay_loss(
        self,
        step_data: list[Any],
        *,
        seed: int,
        audit: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Evaluate native masked action-flow MSE on fixed training examples."""

        self._require_loaded()
        assert self.native_policy is not None
        processor = self.native_policy.processor
        if bool(getattr(processor, "training", False)):
            raise RuntimeError("SFT replay requires the processor in eval mode")
        from gr00t.data.types import MessageType
        from gr00t.policy.gr00t_policy import _rec_to_dtype

        processed = [
            processor([{"type": MessageType.EPISODE_STEP.value, "content": example}])
            for example in step_data
        ]
        collated = self.native_policy.collate_fn(processed)
        collated = _rec_to_dtype(collated, dtype=torch.bfloat16)
        collated["inputs"] = _detach_tensor_tree(collated["inputs"])
        input_tensors = list(_iter_tensor_tree(collated["inputs"]))
        if audit is not None:
            audit.update(
                {
                    "input_tensor_count": len(input_tensors),
                    "input_requires_grad_count": sum(
                        int(tensor.requires_grad) for tensor in input_tensors
                    ),
                }
            )
        model_device = next(iter(self.model.parameters())).device
        fork_devices = (
            [
                model_device.index
                if model_device.index is not None
                else torch.cuda.current_device()
            ]
            if model_device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(int(seed))
            outputs = self.model(**collated)
        loss = outputs["loss"]
        if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"Invalid native SFT replay loss: {loss}")
        return loss

    def predict_native_action_chunk(self, native_input: dict[str, Any]) -> Any:
        self._require_loaded()
        assert self.native_policy is not None
        actions, _ = self.native_policy.get_action(native_input)
        return actions

    def decode_action_transforms(
        self,
        actions: torch.Tensor,
        native_input: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Decode sampled actions using the state that conditioned the sample."""

        self._require_loaded()
        assert self.native_policy is not None
        states = {
            key: np.asarray(value) for key, value in native_input["state"].items()
        }
        decoded = self.native_policy.processor.decode_action(
            actions.float().cpu().numpy(),
            self.native_policy.embodiment_tag,
            states,
        )
        return {
            key: np.asarray(value, dtype=np.float32) for key, value in decoded.items()
        }

    def _collate_native_input(self, native_input: dict[str, Any]) -> dict[str, Any]:
        assert self.native_policy is not None
        self.native_policy.check_observation(native_input)
        from gr00t.data.types import MessageType
        from gr00t.policy.gr00t_policy import _rec_to_dtype

        processed = []
        for observation in self.native_policy._unbatch_observation(native_input):
            step_data = self.native_policy._to_vla_step_data(observation)
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": step_data}]
            processed.append(self.native_policy.processor(messages))
        collated = self.native_policy.collate_fn(processed)
        # Match Gr00tPolicy._get_action exactly before either native inference or
        # Flow-SDE conditioning. In particular, image tensors must enter the
        # frozen backbone as BF16 rather than relying on downstream coercion.
        return _rec_to_dtype(collated, dtype=torch.bfloat16)

    def parameters(self, *args: Any, **kwargs: Any):
        return self.model.parameters(*args, **kwargs)

    def named_parameters(self, *args: Any, **kwargs: Any):
        return self.model.named_parameters(*args, **kwargs)

    def train(self, mode: bool = True) -> "GR00TN17FlowPolicy":
        self.model.train(mode)
        return self

    def eval(self) -> "GR00TN17FlowPolicy":
        return self.train(False)

    def to(self, device: str) -> "GR00TN17FlowPolicy":
        self.model.to(device)
        self.device = str(device)
        if self.device == "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    def save_pretrained(self, path: str, **kwargs: Any) -> Any:
        return self.model.save_pretrained(path, **kwargs)

    def save_checkpoint(self, path: str) -> None:
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "peft_config"):
            self.model.save_pretrained(destination)
            snapshot_format = "peft_adapter"
            adapter_names = list(self.model.peft_config)
        else:
            from safetensors.torch import save_model

            save_model(self.model, destination / "model.safetensors")
            snapshot_format = "full_model"
            adapter_names = []
        (destination / "art_embodied_gr00t_n1d7_snapshot.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "format": snapshot_format,
                    "family": self.family,
                    "model_id": self.model_id,
                    "revision": self.revision,
                    "checkpoint_subfolder": self.checkpoint_subfolder,
                    "embodiment_tag": self.embodiment_tag,
                    "adapter_names": adapter_names,
                    "processor_action_horizon": self.processor_action_horizon,
                    "action_components": self.action_components,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def load_checkpoint(self, checkpoint: dict[str, Any] | str | Path) -> None:
        raw_path = (
            checkpoint.get("path") if isinstance(checkpoint, dict) else checkpoint
        )
        source = Path(str(raw_path)).expanduser()
        metadata = json.loads(
            (source / "art_embodied_gr00t_n1d7_snapshot.json").read_text(
                encoding="utf-8"
            )
        )
        expected = {
            "family": self.family,
            "model_id": self.model_id,
            "revision": self.revision,
            "checkpoint_subfolder": self.checkpoint_subfolder,
            "embodiment_tag": self.embodiment_tag,
            "processor_action_horizon": self.processor_action_horizon,
            "action_components": [
                list(component) for component in self.action_components
            ],
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"GR00T N1.7 snapshot contract mismatch: {mismatches}")
        snapshot_format = metadata.get("format")
        if snapshot_format == "peft_adapter":
            if not hasattr(self.model, "peft_config"):
                raise RuntimeError("N1.7 PEFT snapshot requires an attached adapter")
            from peft.utils.save_and_load import (
                load_peft_weights,
                set_peft_model_state_dict,
            )

            expected_adapters = list(self.model.peft_config)
            recorded_adapters = list(metadata.get("adapter_names") or ["default"])
            if recorded_adapters != expected_adapters:
                raise RuntimeError(
                    "N1.7 adapter snapshot geometry mismatch: "
                    f"recorded={recorded_adapters}, expected={expected_adapters}"
                )
            for adapter_name in expected_adapters:
                adapter_path = (
                    source if adapter_name == "default" else source / adapter_name
                )
                state = load_peft_weights(str(adapter_path), device=self.device)
                result = set_peft_model_state_dict(
                    self.model,
                    state,
                    adapter_name=adapter_name,
                )
                unexpected = list(getattr(result, "unexpected_keys", ()) or ())
                if unexpected:
                    raise RuntimeError(
                        "N1.7 adapter snapshot has unexpected keys for "
                        f"{adapter_name!r}: {unexpected}"
                    )
            if len(expected_adapters) > 1:
                self.model.base_model.set_adapter(expected_adapters)
        elif snapshot_format == "full_model":
            from safetensors.torch import load_model

            missing, unexpected = load_model(
                self.model,
                source / "model.safetensors",
                strict=True,
                device=self.device,
            )
            if missing or unexpected:
                raise RuntimeError(
                    "N1.7 full snapshot mismatch: "
                    f"missing={missing}, unexpected={unexpected}"
                )
        else:
            raise ValueError(f"Unsupported N1.7 snapshot format: {snapshot_format!r}")

    def _require_loaded(self) -> None:
        if self.native_policy is None:
            raise RuntimeError("GR00TN17FlowPolicy.load() must be called first")


def _require_frozen_n1d7_conditioner(model: Any) -> None:
    """Reject trainable modules whose cached output would detach gradients."""

    offenders = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        cached_action_head_path = name.startswith(
            ("action_head.vlln.", "action_head.vl_self_attention.")
        )
        if not name.startswith("action_head.") or cached_action_head_path:
            offenders.append(name)
    if offenders:
        raise RuntimeError(
            "N1.7 retained replay requires the backbone, VLLN, and VL self-attention "
            "to remain frozen; trainable parameters include: "
            + ", ".join(offenders[:5])
        )


def _detach_tensor_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {key: _detach_tensor_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detach_tensor_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_tensor_tree(item) for item in value)
    return value


def _iter_tensor_tree(value: Any):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensor_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensor_tree(item)
