"""Lazy LeRobot SmolVLA policy plugin for Flow-SDE RL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .flow_policy import FlowSDERollout
from .flow_sde import FlowSDESchedule
from .smolvla_flow_sde import SmolVLAFlowSDEBridge
from .transformers_compat import ensure_art_mask_patch_compatibility


class SmolVLAFlowPolicy:
    """Own SmolVLA's native policy, processors, and sampler-aligned bridge."""

    family = "smolvla"

    def __init__(
        self,
        *,
        model_id: str,
        revision: str | None,
        device: str,
        execution_horizon: int,
        action_dim: int,
        observation_key_map: dict[str, str],
        schedule: FlowSDESchedule,
        strict_weights: bool,
        compile_model: bool,
        train_expert_only: bool,
        load_on_init: bool = False,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.execution_horizon = int(execution_horizon)
        self.action_dim = int(action_dim)
        self.observation_key_map = dict(observation_key_map)
        self.schedule = schedule
        self.strict_weights = bool(strict_weights)
        self.compile_model = bool(compile_model)
        self.train_expert_only = bool(train_expert_only)
        self.policy: Any | None = None
        self.bridge: SmolVLAFlowSDEBridge | None = None
        self.preprocessor: Any | None = None
        self.postprocessor: Any | None = None
        self.trainable_report: dict[str, Any] | None = None
        if load_on_init:
            self.load()

    @property
    def model(self) -> Any:
        self._require_loaded()
        assert self.policy is not None
        return self.policy.model

    @model.setter
    def model(self, value: Any) -> None:
        self._require_loaded()
        assert self.policy is not None
        self.policy.model = value

    @property
    def config(self) -> Any:
        self._require_loaded()
        assert self.policy is not None
        return self.policy.config

    def load(self) -> None:
        if self.policy is not None:
            return
        ensure_art_mask_patch_compatibility()
        try:
            from lerobot.configs import PreTrainedConfig
        except ImportError:  # pragma: no cover - LeRobot 0.4 compatibility.
            from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        config = PreTrainedConfig.from_pretrained(
            self.model_id,
            revision=self.revision,
        )
        if getattr(config, "type", None) != "smolvla":
            raise ValueError(
                f"Checkpoint {self.model_id!r} is not a SmolVLA policy"
            )
        config.device = self.device
        config.num_steps = self.schedule.num_steps
        config.n_action_steps = self.execution_horizon
        config.compile_model = self.compile_model
        config.train_expert_only = self.train_expert_only
        policy = SmolVLAPolicy.from_pretrained(
            self.model_id,
            config=config,
            revision=self.revision,
            strict=self.strict_weights,
        )
        if self.execution_horizon > int(policy.config.chunk_size):
            raise ValueError("execution_horizon exceeds checkpoint chunk_size")
        action_feature = getattr(policy.config, "action_feature", None)
        feature_shape = getattr(action_feature, "shape", None)
        if feature_shape is None or self.action_dim > int(feature_shape[0]):
            raise ValueError(
                "action_dim exceeds or cannot be verified against checkpoint output"
            )
        self.policy = policy
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=self.model_id,
            pretrained_revision=self.revision,
            # Serialized checkpoints commonly store the generic "cuda" device.
            # Override it so process-isolated replicas on cuda:N do not funnel
            # every observation into cuda:0 before model inference.
            preprocessor_overrides={
                "device_processor": {"device": self.device},
            },
        )
        self.bridge = SmolVLAFlowSDEBridge(
            policy,
            schedule=self.schedule,
            execution_horizon=self.execution_horizon,
            action_dim=self.action_dim,
        )

    def reset(self) -> None:
        self._require_loaded()
        for component in (self.policy, self.preprocessor, self.postprocessor):
            reset = getattr(component, "reset", None)
            if callable(reset):
                reset()

    def sample_flow_sde(self, processed_batch: dict[str, Any], **kwargs: Any):
        self._require_loaded()
        assert self.bridge is not None
        return self.bridge.sample(processed_batch, **kwargs)

    def predict_native_action_chunk(
        self,
        processed_batch: dict[str, Any],
        *,
        initial_noise: Any | None = None,
    ):
        self._require_loaded()
        assert self.policy is not None
        return self.policy.predict_action_chunk(processed_batch, noise=initial_noise)

    def predict_bridge_ode_action_chunk(
        self,
        processed_batch: dict[str, Any],
        *,
        initial_noise: Any,
    ):
        self._require_loaded()
        assert self.bridge is not None
        return self.bridge.sample_native_ode(
            processed_batch,
            initial_noise=initial_noise,
        )

    def flow_sde_logprobs(self, rollout: FlowSDERollout):
        self._require_loaded()
        assert self.bridge is not None
        return self.bridge.rescore(rollout)

    def parameters(self, *args: Any, **kwargs: Any):
        self._require_loaded()
        assert self.policy is not None
        return self.policy.parameters(*args, **kwargs)

    def named_parameters(self, *args: Any, **kwargs: Any):
        self._require_loaded()
        assert self.policy is not None
        return self.policy.named_parameters(*args, **kwargs)

    def train(self, mode: bool = True):
        self._require_loaded()
        assert self.policy is not None
        self.policy.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def save_pretrained(self, path: str, **kwargs: Any) -> Any:
        self._require_loaded()
        assert self.policy is not None
        return self.policy.save_pretrained(path, **kwargs)

    def save_checkpoint(self, path: str) -> None:
        """Publish a rollout snapshot without duplicating frozen base weights."""

        self._require_loaded()
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
        (destination / "art_embodied_smolvla_snapshot.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "format": snapshot_format,
                    "family": self.family,
                    "model_id": self.model_id,
                    "revision": self.revision,
                    "adapter_names": adapter_names,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def load_checkpoint(self, checkpoint: dict[str, Any] | str | Path) -> None:
        self._require_loaded()
        raw_path = checkpoint.get("path") if isinstance(checkpoint, dict) else checkpoint
        source = Path(str(raw_path)).expanduser()
        metadata_path = source / "art_embodied_smolvla_snapshot.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"SmolVLA snapshot metadata is missing: {metadata_path}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("family") != self.family:
            raise ValueError("SmolVLA snapshot family mismatch")
        snapshot_format = metadata.get("format")
        if snapshot_format == "peft_adapter":
            if not hasattr(self.model, "peft_config"):
                raise RuntimeError(
                    "PEFT snapshot requires the rollout policy to attach LoRA first"
                )
            from peft.utils.save_and_load import (
                load_peft_weights,
                set_peft_model_state_dict,
            )

            expected_adapters = list(self.model.peft_config)
            recorded_adapters = list(metadata.get("adapter_names") or ["default"])
            if recorded_adapters != expected_adapters:
                raise RuntimeError(
                    "SmolVLA adapter snapshot geometry mismatch: "
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
                        "SmolVLA adapter snapshot has unexpected keys for "
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
                    "SmolVLA full snapshot mismatch: "
                    f"missing={missing}, unexpected={unexpected}"
                )
        else:
            raise ValueError(f"Unsupported SmolVLA snapshot format: {snapshot_format!r}")

    def to(self, device: str) -> "SmolVLAFlowPolicy":
        self._require_loaded()
        assert self.policy is not None
        self.policy.to(device)
        self.device = str(device)
        if str(device) == "cpu":
            import torch

            torch.cuda.empty_cache()
        return self

    def _require_loaded(self) -> None:
        if self.policy is None:
            raise RuntimeError("SmolVLAFlowPolicy.load() must be called first")
