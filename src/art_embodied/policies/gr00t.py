"""Lazy NVIDIA GR00T N1.5 policy plugin for Flow-SDE GRPO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .flow_policy import FlowSDERollout, GR00TFlowModelInputs
from .flow_sde import FlowSDESchedule
from .gr00t_flow_sde import GR00TN15FlowSDEBridge


class GR00TN15FlowPolicy:
    """Own the official N1.5 runtime while exposing ART's flow-policy API."""

    family = "gr00t_n1d5"

    def __init__(
        self,
        *,
        model_id: str,
        revision: str | None,
        device: str,
        data_config: str,
        embodiment_tag: str,
        execution_horizon: int,
        action_dim: int,
        model_action_horizon: int,
        language_padding_length: int,
        schedule: FlowSDESchedule,
        disable_dropout: bool,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.data_config = data_config
        self.embodiment_tag = embodiment_tag
        self.execution_horizon = int(execution_horizon)
        self.action_dim = int(action_dim)
        self.model_action_horizon = int(model_action_horizon)
        self.language_padding_length = int(language_padding_length)
        self.schedule = schedule
        self.disable_dropout = bool(disable_dropout)
        self.native_policy: Any | None = None
        self.bridge: GR00TN15FlowSDEBridge | None = None
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
        self.bridge = GR00TN15FlowSDEBridge(
            action_head,
            schedule=self.schedule,
            execution_horizon=self.execution_horizon,
            action_dim=self.action_dim,
        )

    def load(self) -> None:
        if self.native_policy is not None:
            return
        from art_embodied.integrations.gr00t_n1d5 import (
            install_gr00t_n1d5_embodiment_compatibility,
        )

        install_gr00t_n1d5_embodiment_compatibility()
        from gr00t.experiment.data_config import load_data_config
        from gr00t.model.policy import Gr00tPolicy
        from huggingface_hub import snapshot_download

        model_path = self.model_id
        if not Path(model_path).expanduser().exists():
            model_path = snapshot_download(
                repo_id=self.model_id,
                revision=self.revision,
                repo_type="model",
            )
        data_config = load_data_config(self.data_config)
        native_policy = Gr00tPolicy(
            model_path=model_path,
            embodiment_tag=self.embodiment_tag,
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            denoising_steps=self.schedule.num_steps,
            device=self.device,
        )
        model = native_policy.model
        if int(model.action_head.config.action_horizon) != self.model_action_horizon:
            raise RuntimeError(
                "GR00T action horizon differs from the declared checkpoint contract: "
                f"model={model.action_head.config.action_horizon}, "
                f"declared={self.model_action_horizon}"
            )
        if self.action_dim > int(model.action_head.config.action_dim):
            raise RuntimeError("GR00T action_dim exceeds the loaded action head")
        if self.disable_dropout:
            _replace_dropout_with_identity(model)
        elif any(isinstance(module, torch.nn.Dropout) for module in model.modules()):
            raise RuntimeError("GR00T Flow-SDE parity requires dropout to be disabled")
        self.native_policy = native_policy
        self.bridge = GR00TN15FlowSDEBridge(
            model.action_head,
            schedule=self.schedule,
            execution_horizon=self.execution_horizon,
            action_dim=self.action_dim,
        )

    @torch.no_grad()
    def prepare_flow_inputs(
        self,
        normalized_input: dict[str, Any],
    ) -> GR00TFlowModelInputs:
        """Cache only the frozen Eagle output for rollout replay."""

        self._require_loaded()
        model = _native_gr00t_model(self.model)
        _require_frozen_gr00t_conditioner(model)
        normalized_input = self._pad_language_inputs(normalized_input)
        backbone_input, action_input = model.prepare_input(normalized_input)
        backbone_output = model.backbone(backbone_input)
        backbone_output = model.action_head.process_backbone_output(backbone_output)
        return GR00TFlowModelInputs(
            vision_language_features=backbone_output.backbone_features,
            vision_language_attention_mask=backbone_output.backbone_attention_mask,
            state=action_input.state,
            embodiment_id=action_input.embodiment_id,
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

    def predict_native_action_chunk(self, observations: dict[str, Any]) -> Any:
        self._require_loaded()
        assert self.native_policy is not None
        normalized = self.native_policy.apply_transforms(observations.copy())
        normalized = self._pad_language_inputs(normalized)
        model = _native_gr00t_model(self.model)
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=torch.device(self.device).type,
                dtype=torch.bfloat16,
                enabled=torch.device(self.device).type == "cuda",
            ),
        ):
            actions = model.get_action(normalized)["action_pred"].float()
        return self.native_policy.unapply_transforms({"action": actions.cpu()})

    def apply_observation_transforms(self, observations: dict[str, Any]) -> Any:
        self._require_loaded()
        assert self.native_policy is not None
        return self.native_policy.apply_transforms(observations)

    def unapply_action_transforms(self, actions: torch.Tensor) -> Any:
        self._require_loaded()
        assert self.native_policy is not None
        # N1.5's inverse StateActionToTensor calls Tensor.numpy(), which has no
        # BF16 NumPy representation. The official policy likewise converts its
        # normalized action prediction to FP32 before unapplying transforms.
        return self.native_policy.unapply_transforms({"action": actions.float().cpu()})

    def parameters(self, *args: Any, **kwargs: Any):
        return self.model.parameters(*args, **kwargs)

    def named_parameters(self, *args: Any, **kwargs: Any):
        return self.model.named_parameters(*args, **kwargs)

    def train(self, mode: bool = True) -> "GR00TN15FlowPolicy":
        self.model.train(mode)
        return self

    def eval(self) -> "GR00TN15FlowPolicy":
        return self.train(False)

    def to(self, device: str) -> "GR00TN15FlowPolicy":
        self.model.to(device)
        self.device = str(device)
        if self.device == "cpu":
            torch.cuda.empty_cache()
        return self

    def save_pretrained(self, path: str, **kwargs: Any) -> Any:
        return self.model.save_pretrained(path, **kwargs)

    def save_checkpoint(self, path: str) -> None:
        """Save a rollout-worker delta and its immutable model contract."""

        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "peft_config"):
            self.model.save_pretrained(destination)
            snapshot_format = "peft_adapter"
        else:
            from safetensors.torch import save_model

            save_model(self.model, destination / "model.safetensors")
            snapshot_format = "full_model"
        (destination / "art_embodied_gr00t_n1d5_snapshot.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "format": snapshot_format,
                    "family": self.family,
                    "model_id": self.model_id,
                    "revision": self.revision,
                    "embodiment_tag": self.embodiment_tag,
                    "data_config": self.data_config,
                    "language_padding_length": self.language_padding_length,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def load_checkpoint(self, checkpoint: dict[str, Any] | str | Path) -> None:
        """Load an ART delta into an already constructed N1.5 policy."""

        raw_path = (
            checkpoint.get("path") if isinstance(checkpoint, dict) else checkpoint
        )
        source = Path(str(raw_path)).expanduser()
        metadata_path = source / "art_embodied_gr00t_n1d5_snapshot.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "family": self.family,
            "model_id": self.model_id,
            "revision": self.revision,
            "embodiment_tag": self.embodiment_tag,
            "data_config": self.data_config,
            "language_padding_length": self.language_padding_length,
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"GR00T snapshot contract mismatch: {mismatches}")
        snapshot_format = metadata.get("format")
        if snapshot_format == "peft_adapter":
            if not hasattr(self.model, "peft_config"):
                raise RuntimeError("GR00T PEFT snapshot requires an attached adapter")
            from peft.utils.save_and_load import (
                load_peft_weights,
                set_peft_model_state_dict,
            )

            state = load_peft_weights(str(source), device=self.device)
            result = set_peft_model_state_dict(self.model, state)
            unexpected = list(getattr(result, "unexpected_keys", ()) or ())
            if unexpected:
                raise RuntimeError(
                    f"GR00T adapter snapshot has unexpected keys: {unexpected}"
                )
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
                    "GR00T full snapshot mismatch: "
                    f"missing={missing}, unexpected={unexpected}"
                )
        else:
            raise ValueError(f"Unsupported GR00T snapshot format: {snapshot_format!r}")

    def _require_loaded(self) -> None:
        if self.native_policy is None:
            raise RuntimeError("GR00TN15FlowPolicy.load() must be called first")

    def _pad_language_inputs(self, normalized_input: dict[str, Any]) -> dict[str, Any]:
        """Apply RLinf N1.5's fixed Eagle sequence geometry."""

        result = dict(normalized_input)
        for key in ("eagle_input_ids", "eagle_attention_mask"):
            value = result.get(key)
            if value is None:
                raise KeyError(f"GR00T transformed input is missing {key!r}")
            current = int(value.shape[-1])
            if current > self.language_padding_length:
                raise ValueError(
                    f"GR00T {key} length {current} exceeds fixed padding "
                    f"length {self.language_padding_length}"
                )
            result[key] = torch.nn.functional.pad(
                value,
                (0, self.language_padding_length - current),
                mode="constant",
                value=0,
            )
        return result


def _native_gr00t_model(model: Any) -> Any:
    """Unwrap PEFT without importing it into the GR00T runtime boundary."""

    get_base_model = getattr(model, "get_base_model", None)
    return get_base_model() if callable(get_base_model) else model


def _replace_dropout_with_identity(module: torch.nn.Module) -> None:
    """Disable stochastic dropout exactly once before rollout and rescore."""

    for name, child in tuple(module.named_children()):
        if isinstance(child, torch.nn.Dropout):
            setattr(module, name, torch.nn.Identity())
        else:
            _replace_dropout_with_identity(child)


def _require_frozen_gr00t_conditioner(model: Any) -> None:
    """Reject trainable conditioning paths that retained replay would detach."""

    offenders = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("action_head.")
    ]
    if offenders:
        preview = ", ".join(offenders[:5])
        raise RuntimeError(
            "GR00T retained replay requires the backbone and conditioning path "
            f"to remain frozen; trainable parameters include: {preview}"
        )
