"""Trainable OpenPI action-expert LoRA with sampler-aligned replay."""

from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import pydantic
import torch

from ..utils import write_json_atomic
from ..vla_trainable import configure_vla_trainable_parameters
from .flow_policy import FlowSDERollout
from .flow_sde import FlowSDESchedule
from .openpi_flow_sde import OpenPIFlowSDEBridge


class OpenPIFlowLoadConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    config_name: str = "pi05_droid_jointpos_polaris"
    checkpoint_sha256: str = pydantic.Field(pattern=r"^[a-f0-9]{64}$")
    normalization_sha256: str = pydantic.Field(pattern=r"^[a-f0-9]{64}$")
    execution_horizon: int = 15
    action_dim: int = 8


def validate_openpi_config(config):
    settings = OpenPIFlowLoadConfig.model_validate(config.policy.load_kwargs)
    if settings.config_name != "pi05_droid_jointpos_polaris" or (
        settings.execution_horizon,
        settings.action_dim,
    ) != (15, 8):
        raise ValueError(
            "OpenPI DROID currently requires the PI0.5 Polaris 15x8 contract"
        )
    if config.algorithm.flow_sde is None or config.algorithm.type != "grpo":
        raise ValueError("OpenPI DROID requires Flow-SDE GRPO")
    if config.policy.dtype != "bfloat16" or not config.policy.force_trainable_float32:
        raise ValueError("OpenPI uses its native BF16 backbone and FP32 trainable LoRA")
    if config.policy.trainable_parameter_strategy != "pi05_action_expert_lora":
        raise ValueError("OpenPI DROID requires action-expert LoRA")
    if (
        not config.policy.lora.enabled
        or config.policy.lora.dropout != 0
        or config.policy.lora.rank_partition is not None
    ):
        raise ValueError("OpenPI DROID requires shared LoRA with zero dropout")
    flow = config.algorithm.flow_sde
    if (
        flow.joint_logprob
        or flow.stochastic_transitions_per_sample != 1
        or flow.selected_step_sampling != "uniform"
    ):
        raise ValueError("OpenPI DROID requires one uniformly sampled SDE transition")
    if (
        config.rollout.action_payload.kind != "continuous"
        or not config.rollout.action_payload.require_old_logprobs
        or config.rollout.action_payload.trainable_action_selection != "all"
        or config.environment.kwargs.get("action_chunk_size")
        != settings.execution_horizon
        or config.rollout.max_policy_steps
        != (config.rollout.max_episode_steps + settings.execution_horizon - 1)
        // settings.execution_horizon
        or getattr(
            config.training.schedule, "action_chunk_size", settings.execution_horizon
        )
        != settings.execution_horizon
    ):
        raise ValueError("OpenPI rollout and training action geometry must match")
    algorithm = config.algorithm
    if (
        algorithm.training_unit != "action"
        or algorithm.action_advantage_mode != "example"
        or algorithm.score_source != "trajectory_reward"
        or algorithm.loss_aggregation != "rlinf_chunk_mean"
        or algorithm.rollout_logprob_source != "rollout_action"
        or not algorithm.advantage.normalize
        or algorithm.advantage.scope != "group"
        or algorithm.advantage.extra_global_normalization
    ):
        raise ValueError(
            "OpenPI requires the shared Flow-SDE group-advantage objective"
        )
    if (
        config.training.schedule.type == "full_update"
        and config.training.optimizer_steps_per_update != 1
    ):
        raise ValueError(
            "Use rlinf_actor_global_batch for multiple Flow-SDE optimizer steps"
        )
    return settings


def make_openpi_flow_policy(config, *, load=True):
    import hashlib

    from .factory import _trainable_parameter_config
    from .openpi_flow_sde import load_openpi_pytorch

    settings = validate_openpi_config(config)
    if not load:
        raise ValueError(
            "OpenPI constructs its processors together with loaded weights"
        )
    checkpoint = Path(config.policy.path)
    with (checkpoint / "model.safetensors").open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != settings.checkpoint_sha256:
        raise ValueError("OpenPI base checkpoint checksum does not match the recipe")
    with (checkpoint / "assets/droid/norm_stats.json").open("rb") as stream:
        normalization = hashlib.file_digest(stream, "sha256").hexdigest()
    if normalization != settings.normalization_sha256:
        raise ValueError("OpenPI normalization checksum does not match the recipe")
    native = load_openpi_pytorch(
        checkpoint, config_name=settings.config_name, device=config.policy.device
    )
    return OpenPIFlowPolicy(
        native,
        base_identity=f"{actual}:{normalization}",
        config_name=settings.config_name,
        schedule=FlowSDESchedule(
            config.algorithm.flow_sde.num_denoise_steps,
            config.algorithm.flow_sde.noise_level,
            deterministic_sampler="native_euler",
        ),
        batch_size=config.training.microbatch_size,
        execution_horizon=settings.execution_horizon,
        action_dim=settings.action_dim,
        lora_config=_trainable_parameter_config(config)["peft"],
    )


class _NativeLoRARoot(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.native = model
        # PEFT reads a Hugging Face-style config mapping. OpenPI itself needs its
        # original dataclass, so place the mapping on a wrapper instead.
        self.config = {"model_type": "pi05_openpi"}

    def forward(self, *args, **kwargs):
        return self.native(*args, **kwargs)


class OpenPIFlowPolicy(torch.nn.Module):
    """Update only the action expert; preserve the native vision-language model."""

    def __init__(
        self,
        native: Any,
        *,
        base_identity: str,
        config_name: str,
        schedule: FlowSDESchedule,
        batch_size: int,
        execution_horizon: int,
        action_dim: int,
        lora_config: dict[str, Any],
    ) -> None:
        super().__init__()
        if not base_identity or not config_name:
            raise ValueError("OpenPI checkpoint identity and config name are required")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not getattr(native._model, "pi05", False):
            raise ValueError(
                "This trainable OpenPI integration currently requires PI0.5"
            )
        if not lora_config.get("enabled") or lora_config.get("rank_partition"):
            raise ValueError("OpenPI requires shared action-expert LoRA")
        self.model = _NativeLoRARoot(native._model)
        self.batch_size = batch_size
        self.bridge = OpenPIFlowSDEBridge(
            native,
            schedule=schedule,
            execution_horizon=execution_horizon,
            action_dim=action_dim,
        )
        _, report = configure_vla_trainable_parameters(
            self,
            policy_type="pi05",
            algorithm_cfg={
                "trainable_parameter_strategy": "pi05_action_expert_lora",
                "force_trainable_float32": True,
                "gradient_checkpointing": False,
                "peft": lora_config,
            },
        )
        if not report.get("ok"):
            raise RuntimeError(f"OpenPI LoRA configuration failed: {report}")
        if self.model.get_base_model().native is not self.bridge.model:
            raise RuntimeError("OpenPI sampler and trainer do not share the same model")
        trainable = [
            name for name, value in self.named_parameters() if value.requires_grad
        ]
        if not trainable or any(
            "lora_" not in name or "paligemma." in name for name in trainable
        ):
            raise RuntimeError(
                "OpenPI trainable parameters escaped the action-expert LoRA"
            )
        self.trainable_report = report
        self._worker_index = 0
        self._contract = {
            "format_version": 1,
            "base_identity": base_identity,
            "config_name": config_name,
            "batch_size": batch_size,
            "execution_horizon": execution_horizon,
            "action_dim": action_dim,
            "schedule": asdict(schedule),
            "lora": copy.deepcopy(lora_config),
        }
        self.eval()

    def flow_sde_logprobs(self, rollout: FlowSDERollout) -> torch.Tensor:
        count = rollout.inputs.batch_size
        if not 0 < count <= self.batch_size:
            raise ValueError("Replay exceeds the fixed behavior batch size")
        if count < self.batch_size:
            rollout = FlowSDERollout.concatenate(
                [rollout] + [rollout.select(count - 1)] * (self.batch_size - count)
            )
        return self.bridge.rescore(rollout)[:count]

    def save_checkpoint(self, path: str | Path) -> None:
        """Store LoRA and RNG state without pickled Python objects."""
        from safetensors.torch import save_file

        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        if any(destination.iterdir()):
            raise FileExistsError(f"Nonempty checkpoint destination: {destination}")
        self.model.save_pretrained(destination, safe_serialization=True)
        states = {"cpu": torch.get_rng_state()}
        if torch.cuda.is_available():
            states.update(
                {
                    f"cuda_{index}": state
                    for index, state in enumerate(torch.cuda.get_rng_state_all())
                }
            )
        save_file(states, str(destination / "rng.safetensors"))
        numpy_state = np.random.get_state()
        write_json_atomic(
            destination / "rng.json",
            {
                "python": random.getstate(),
                "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            },
        )
        # Written last so partially saved checkpoints cannot pass contract checks.
        write_json_atomic(destination / "runtime.json", self._contract)

    def load_checkpoint(self, path: str | Path) -> None:
        self.restore_checkpoint(path, cuda_rng_source_index=self._worker_index)

    def set_worker_index(self, index: int) -> None:
        """Select this replica's CUDA RNG state in a coordinator snapshot."""
        if type(index) is not int or index < 0:
            raise ValueError("worker index must be a nonnegative integer")
        self._worker_index = index

    def restore_checkpoint(
        self,
        path: str | Path,
        *,
        restore_rng: bool = True,
        cuda_rng_source_index: int = 0,
    ) -> None:
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        from safetensors.torch import load_file

        source = Path(path)
        if json.loads((source / "runtime.json").read_text()) != self._contract:
            raise ValueError("OpenPI checkpoint does not match the runtime contract")
        state = load_file(
            str(source / "adapter_model.safetensors"), device=str(self.bridge.device)
        )
        expected = get_peft_model_state_dict(self.model)
        if state.keys() != expected.keys() or any(
            state[key].shape != expected[key].shape for key in state
        ):
            raise ValueError("OpenPI checkpoint adapter keys or shapes do not match")
        if any(not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("OpenPI checkpoint contains nonfinite weights")
        rng_states = rng_json = None
        if restore_rng:
            rng_states = load_file(str(source / "rng.safetensors"), device="cpu")
            rng_json = json.loads((source / "rng.json").read_text())
            cuda_states = [
                rng_states[key]
                for key in sorted(
                    (key for key in rng_states if key.startswith("cuda_")),
                    key=lambda key: int(key[5:]),
                )
            ]
            device_count = torch.cuda.device_count()
            if len(cuda_states) != device_count:
                if device_count == 1 and 0 <= cuda_rng_source_index < len(cuda_states):
                    cuda_states = [cuda_states[cuda_rng_source_index]]
                else:
                    raise ValueError("OpenPI checkpoint CUDA RNG topology differs")
        result = set_peft_model_state_dict(self.model, state)
        if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
            raise RuntimeError("Incomplete OpenPI checkpoint restore")
        if rng_states is not None:
            torch.set_rng_state(rng_states["cpu"])
            if cuda_states:
                torch.cuda.set_rng_state_all(cuda_states)
            python_state = rng_json["python"]
            random.setstate((python_state[0], tuple(python_state[1]), python_state[2]))
            numpy_state = rng_json["numpy"]
            np.random.set_state(
                (
                    numpy_state[0],
                    np.array(numpy_state[1], dtype=np.uint32),
                    *numpy_state[2:],
                )
            )
