"""Lazy LeRobot PI0/PI0.5 policy plugin for Flow-SDE RL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from art_embodied.policies.flow_sde import FlowSDESchedule
from art_embodied.policies.pi_flow_sde import PIFlowSDEBridge, PIFlowSDERollout
from art_embodied.policies.transformers_compat import (
    ensure_art_mask_patch_compatibility,
)

PI_TOKENIZER_ID = "google/paligemma-3b-pt-224"


def _preload_pi_tokenizer() -> None:
    """Resolve PI's gated tokenizer before downloading policy weights."""

    from transformers import AutoTokenizer

    try:
        AutoTokenizer.from_pretrained(PI_TOKENIZER_ID)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "PI0/PI0.5 require access to the gated Hugging Face repository "
            f"{PI_TOKENIZER_ID!r}. Accept its terms at "
            f"https://huggingface.co/{PI_TOKENIZER_ID}, then run "
            "`hf auth login` or set HF_TOKEN before launching ART-Embodied. "
            "Tokenizer access is checked before policy weights are downloaded."
        ) from exc


class PIFlowPolicy:
    """Own a native LeRobot PI policy and its sampler-aligned RL bridge."""

    def __init__(
        self,
        *,
        family: str,
        model_id: str,
        revision: str | None,
        device: str,
        dtype: str,
        model_format: str,
        execution_horizon: int,
        action_dim: int,
        processor_path: str | None,
        processor_revision: str | None,
        model_chunk_size: int | None,
        normalization_stats_file: str | None,
        discrete_state_input: bool | None,
        extra_delta_transform: bool | None,
        observation_key_map: dict[str, str],
        schedule: FlowSDESchedule,
        strict_weights: bool,
        compile_model: bool,
        gradient_checkpointing: bool,
        train_expert_only: bool,
        load_on_init: bool = False,
    ) -> None:
        if family not in {"pi0", "pi05"}:
            raise ValueError("family must be 'pi0' or 'pi05'")
        self.family = family
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.dtype = dtype
        self.model_format = model_format
        self.execution_horizon = int(execution_horizon)
        self.action_dim = int(action_dim)
        self.processor_path = processor_path
        self.processor_revision = processor_revision
        self.model_chunk_size = model_chunk_size
        self.normalization_stats_file = normalization_stats_file
        self.discrete_state_input = discrete_state_input
        self.extra_delta_transform = extra_delta_transform
        self.observation_key_map = dict(observation_key_map)
        self.schedule = schedule
        self.strict_weights = bool(strict_weights)
        self.compile_model = bool(compile_model)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.train_expert_only = bool(train_expert_only)
        self.policy: Any | None = None
        self.bridge: PIFlowSDEBridge | None = None
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
        # LeRobot's PI processor hard-codes this gated PaliGemma tokenizer.
        # Resolve the small tokenizer first so an unapproved Hub account fails
        # before downloading a multi-gigabyte policy checkpoint. The
        # coordinator loads before rollout workers and warms their shared cache.
        _preload_pi_tokenizer()
        try:
            from lerobot.configs import PreTrainedConfig
        except ImportError:  # LeRobot 0.4.x compatibility for diagnostics.
            from lerobot.configs.policies import PreTrainedConfig

        if self.family == "pi05":
            from lerobot.policies.pi05 import PI05Policy as NativePolicy
        else:
            from lerobot.policies.pi0 import PI0Policy as NativePolicy
        from lerobot.policies.factory import make_pre_post_processors

        config_source = self.processor_path or self.model_id
        config_revision = self.processor_revision or self.revision
        config = PreTrainedConfig.from_pretrained(
            config_source,
            revision=config_revision,
        )
        config.device = self.device
        config.dtype = self.dtype
        config.compile_model = self.compile_model
        config.gradient_checkpointing = self.gradient_checkpointing
        config.train_expert_only = self.train_expert_only
        dataset_stats = None
        load_serialized_processors = self.model_format == "lerobot"
        if self.model_format == "rlinf_openpi_safetensors":
            _configure_rlinf_openpi_contract(
                config,
                family=self.family,
                execution_horizon=self.execution_horizon,
                action_dim=self.action_dim,
                model_chunk_size=self.model_chunk_size,
                num_inference_steps=self.schedule.num_steps,
                extra_delta_transform=bool(self.extra_delta_transform),
            )
            assert self.normalization_stats_file is not None
            dataset_stats = _load_rlinf_openpi_normalization_stats(
                model_id=self.model_id,
                revision=self.revision,
                filename=self.normalization_stats_file,
                state_dim=int(config.max_state_dim),
                action_dim=self.action_dim,
            )
        if self.model_format == "lerobot":
            policy = NativePolicy.from_pretrained(
                self.model_id,
                config=config,
                revision=self.revision,
                strict=self.strict_weights,
            )
        elif self.model_format == "rlinf_openpi_safetensors":
            policy = NativePolicy(config)
            _load_rlinf_openpi_weights(
                policy,
                model_id=self.model_id,
                revision=self.revision,
                strict=self.strict_weights,
            )
            from art_embodied.policies.openpi_runtime import (
                install_openpi_vision_embedding,
            )

            install_openpi_vision_embedding(policy.model)
        else:  # pragma: no cover - PIFlowLoadConfig owns validation.
            raise ValueError(f"Unsupported PI model format: {self.model_format!r}")
        if self.execution_horizon > int(policy.config.chunk_size):
            raise ValueError(
                "policy.load_kwargs.execution_horizon exceeds checkpoint chunk_size"
            )
        output_features = getattr(policy.config, "output_features", {})
        action_feature = output_features.get("action")
        feature_shape = getattr(action_feature, "shape", None)
        if feature_shape and self.action_dim > int(feature_shape[0]):
            raise ValueError(
                "policy.load_kwargs.action_dim exceeds checkpoint action feature"
            )
        self.policy = policy
        if (
            self.model_format == "rlinf_openpi_safetensors"
            and self.family == "pi05"
            and self.discrete_state_input is False
        ):
            # RLinf's public PI0.5 SFT config explicitly disables OpenPI's
            # discrete-state prompt. Use the PI0 text pipeline so the model sees
            # the training-time plain instruction plus newline, while retaining
            # PI0.5's state-free action-expert architecture.
            from lerobot.policies.pi0.processor_pi0 import (
                make_pi0_pre_post_processors,
            )

            self.preprocessor, self.postprocessor = make_pi0_pre_post_processors(
                policy.config,
                dataset_stats=dataset_stats,
            )
            _set_tokenizer_padding_side(self.preprocessor, "right")
        else:
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                policy.config,
                pretrained_path=(
                    self.processor_path or self.model_id
                    if load_serialized_processors
                    else None
                ),
                pretrained_revision=(
                    self.processor_revision if load_serialized_processors else None
                ),
                dataset_stats=dataset_stats,
            )
        self.bridge = PIFlowSDEBridge(
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

    def predict_native_action_chunk(self, processed_batch: dict[str, Any]):
        """Run LeRobot's deployment sampler for deterministic evaluation."""

        self._require_loaded()
        assert self.policy is not None
        return self.policy.predict_action_chunk(processed_batch)

    def flow_sde_logprobs(self, rollout: PIFlowSDERollout):
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
        """Publish a rollout-worker snapshot without duplicating base weights."""

        self._require_loaded()
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "peft_config"):
            self.model.save_pretrained(destination)
            snapshot_format = "peft_adapter"
        else:
            from safetensors.torch import save_model

            save_model(self.model, destination / "model.safetensors")
            snapshot_format = "full_model"
        (destination / "art_embodied_pi_snapshot.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "format": snapshot_format,
                    "family": self.family,
                    "model_id": self.model_id,
                    "revision": self.revision,
                    "model_format": self.model_format,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def load_checkpoint(self, checkpoint: dict[str, Any] | str | Path) -> None:
        """Load an ART rollout snapshot into an already constructed PI policy."""

        self._require_loaded()
        raw_path = (
            checkpoint.get("path") if isinstance(checkpoint, dict) else checkpoint
        )
        source = Path(str(raw_path)).expanduser()
        metadata_path = source / "art_embodied_pi_snapshot.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"PI snapshot metadata is missing: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("family") != self.family:
            raise ValueError(
                "PI snapshot family mismatch: "
                f"{metadata.get('family')!r} != {self.family!r}"
            )
        snapshot_format = metadata.get("format")
        if snapshot_format == "peft_adapter":
            if not hasattr(self.model, "peft_config"):
                raise RuntimeError(
                    "PEFT PI snapshot requires the rollout policy to attach the "
                    "same LoRA surface before loading"
                )
            from peft.utils.save_and_load import (
                load_peft_weights,
                set_peft_model_state_dict,
            )

            state = load_peft_weights(str(source), device=self.device)
            result = set_peft_model_state_dict(self.model, state)
            unexpected = list(getattr(result, "unexpected_keys", ()) or ())
            if unexpected:
                raise RuntimeError(
                    f"PI adapter snapshot has unexpected parameters: {unexpected}"
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
                    "PI full snapshot mismatch: "
                    f"missing={missing}, unexpected={unexpected}"
                )
        else:
            raise ValueError(f"Unsupported PI snapshot format: {snapshot_format!r}")

    def to(self, device: str) -> "PIFlowPolicy":
        self._require_loaded()
        assert self.policy is not None
        self.policy.to(device)
        self.device = str(device)
        if str(device) == "cpu":
            # Long-running local-process schedules alternate rollout and
            # training replicas on the same GPUs. Moving weights is not enough:
            # release PyTorch's cached CUDA blocks so the other phase can use
            # the full device without restarting this expensive model process.
            import torch

            torch.cuda.empty_cache()
        return self

    def _require_loaded(self) -> None:
        if self.policy is None:
            raise RuntimeError("PIFlowPolicy.load() must be called before model access")


def _load_rlinf_openpi_weights(
    policy: Any,
    *,
    model_id: str,
    revision: str | None,
    strict: bool,
) -> None:
    """Load RLinf's raw OpenPI safetensors into LeRobot's native PI module."""

    from pathlib import Path

    from safetensors.torch import load_model

    checkpoints, declared_keys = _resolve_rlinf_openpi_safetensors(
        model_id=model_id,
        revision=revision,
    )
    checkpoint_dtypes = _read_safetensors_dtypes(checkpoints)
    _apply_checkpoint_dtypes(policy.model, checkpoint_dtypes)
    unexpected: set[str] = set()
    if declared_keys is None:
        missing, shard_unexpected = load_model(
            policy.model,
            checkpoints[0],
            # OpenPI stores one side of PaliGemma's tied embedding/lm_head pair.
            # Validate that single alias explicitly below rather than weakening
            # the checkpoint contract for any other tensor.
            strict=False,
            device=policy.config.device,
        )
        missing_set = set(missing)
        unexpected.update(shard_unexpected)
    else:
        actual_keys = set(checkpoint_dtypes)
        for checkpoint in checkpoints:
            _, shard_unexpected = load_model(
                policy.model,
                checkpoint,
                strict=False,
                device=policy.config.device,
            )
            unexpected.update(shard_unexpected)
        if actual_keys != declared_keys:
            raise RuntimeError(
                "RLinf OpenPI sharded checkpoint index does not match shard "
                f"contents: missing={sorted(declared_keys - actual_keys)}, "
                f"undeclared={sorted(actual_keys - declared_keys)}"
            )
        missing_set = set(policy.model.state_dict()) - declared_keys
    embedding_key = (
        "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    )
    allowed_missing = {embedding_key}
    missing_embedding_only = missing_set == allowed_missing
    unexpected_policy_keys = {
        key for key in unexpected if not key.startswith("value_head.")
    }
    if missing_embedding_only:
        _copy_paligemma_lm_head_to_embedding(policy.model)
    if strict and (
        unexpected_policy_keys or (missing_set and not missing_embedding_only)
    ):
        raise RuntimeError(
            "RLinf OpenPI checkpoint did not strictly match the LeRobot PI model: "
            f"missing={sorted(missing_set)}, "
            f"unexpected={sorted(unexpected_policy_keys)}"
        )


def _read_safetensors_dtypes(checkpoints: list[Path]) -> dict[str, str]:
    """Read tensor dtype metadata without materializing checkpoint tensors."""

    from safetensors import safe_open

    dtypes: dict[str, str] = {}
    for checkpoint in checkpoints:
        with safe_open(checkpoint, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                dtype = str(handle.get_slice(key).get_dtype())
                previous = dtypes.setdefault(key, dtype)
                if previous != dtype:
                    raise RuntimeError(
                        "RLinf OpenPI checkpoint declares conflicting dtypes for "
                        f"{key!r}: {previous} vs {dtype}"
                    )
    return dtypes


def _apply_checkpoint_dtypes(model: Any, dtypes: dict[str, str]) -> None:
    """Preserve OpenPI's mixed-precision parameter contract during import.

    ``safetensors.torch.load_model`` copies into the receiving module's dtype.
    LeRobot 0.6 keeps its complete vision path in FP32, whereas RLinf v0.1's
    OpenPI checkpoint stores a deliberate mixture of FP32 and BF16 tensors.
    Casting receiving tensors first prevents this silent policy change.
    """

    if not dtypes:
        return

    import torch

    torch_dtypes = {
        "BOOL": torch.bool,
        "U8": torch.uint8,
        "I8": torch.int8,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F64": torch.float64,
    }
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    for key, serialized_dtype in dtypes.items():
        target_dtype = torch_dtypes.get(serialized_dtype)
        if target_dtype is None:
            raise ValueError(
                f"Unsupported OpenPI safetensors dtype {serialized_dtype!r} "
                f"for {key!r}"
            )
        tensor = parameters.get(key)
        if tensor is None:
            tensor = buffers.get(key)
        if tensor is None or tensor.dtype == target_dtype:
            continue
        tensor.data = tensor.data.to(dtype=target_dtype)


def _resolve_rlinf_openpi_safetensors(
    *,
    model_id: str,
    revision: str | None,
) -> tuple[list[Path], set[str] | None]:
    """Resolve one safetensors file or a validated Hugging Face shard index."""

    source = Path(model_id).expanduser()
    if source.is_file():
        return [source], None
    if source.is_dir():
        single = source / "model.safetensors"
        index = source / "model.safetensors.index.json"
        if single.is_file():
            return [single], None
        if index.is_file():
            return _resolve_local_safetensors_index(index)
        raise FileNotFoundError(
            f"PI checkpoint directory contains neither model.safetensors nor "
            f"model.safetensors.index.json: {source}"
        )

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        single = Path(
            hf_hub_download(
                repo_id=model_id,
                filename="model.safetensors",
                revision=revision,
            )
        )
    except EntryNotFoundError:
        index = Path(
            hf_hub_download(
                repo_id=model_id,
                filename="model.safetensors.index.json",
                revision=revision,
            )
        )
        return _resolve_local_safetensors_index(
            index,
            repo_id=model_id,
            revision=revision,
        )
    return [single], None


def _resolve_local_safetensors_index(
    index: Path,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
) -> tuple[list[Path], set[str]]:
    payload = json.loads(index.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"Invalid safetensors weight_map: {index}")
    declared_keys = {str(key) for key in weight_map}
    shard_names = sorted({str(name) for name in weight_map.values()})
    checkpoints = []
    for name in shard_names:
        checkpoint = index.parent / name
        if not checkpoint.is_file() and repo_id is not None:
            from huggingface_hub import hf_hub_download

            checkpoint = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=name,
                    revision=revision,
                )
            )
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing safetensors shard: {checkpoint}")
        checkpoints.append(checkpoint)
    return checkpoints, declared_keys


def _copy_paligemma_lm_head_to_embedding(model: Any) -> None:
    """Apply LeRobot's OpenPI compatibility transform for PaliGemma weights."""

    import torch

    try:
        paligemma = model.paligemma_with_expert.paligemma
        embedding = paligemma.model.language_model.embed_tokens.weight
        lm_head = paligemma.lm_head.weight
    except AttributeError as error:
        raise RuntimeError(
            "RLinf PI checkpoint omitted the PaliGemma embedding, but the "
            "LeRobot model does not expose the expected lm_head/embedding pair"
        ) from error
    if embedding.shape != lm_head.shape:
        raise RuntimeError(
            "RLinf PI PaliGemma lm_head/embedding shape mismatch: "
            f"lm_head={tuple(lm_head.shape)}, embedding={tuple(embedding.shape)}"
        )
    with torch.no_grad():
        embedding.copy_(lm_head)


def _configure_rlinf_openpi_contract(
    config: Any,
    *,
    family: str,
    execution_horizon: int,
    action_dim: int,
    model_chunk_size: int | None,
    num_inference_steps: int,
    extra_delta_transform: bool,
) -> None:
    """Replace checkpoint-local LeRobot defaults with RLinf/OpenPI geometry."""

    if (
        model_chunk_size is None
    ):  # guarded by PIFlowLoadConfig; keeps this helper total.
        raise ValueError("RLinf/OpenPI contract requires model_chunk_size")
    if execution_horizon > model_chunk_size:
        raise ValueError("execution_horizon cannot exceed model_chunk_size")
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be positive")
    if family == "pi05":
        from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
        from lerobot.utils.constants import ACTION, OBS_STATE

        config.normalization_mapping = {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
        config.input_features[OBS_STATE] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(int(config.max_state_dim),),
        )
        config.output_features[ACTION] = PolicyFeature(
            type=FeatureType.ACTION,
            shape=(action_dim,),
        )
    config.use_relative_actions = bool(extra_delta_transform)
    if extra_delta_transform:
        if action_dim != 7:
            raise ValueError(
                "RLinf/OpenPI LIBERO extra_delta_transform requires the "
                "6-DoF-plus-gripper action contract (action_dim=7)"
            )
        # LeRobot derives the delta mask from these names. OpenPI's exact mask is
        # make_bool_mask(6, -1): six relative pose dimensions and one absolute
        # gripper dimension.
        config.relative_exclude_joints = ["gripper"]
        config.action_feature_names = [
            "x",
            "y",
            "z",
            "roll",
            "pitch",
            "yaw",
            "gripper",
        ]
    config.chunk_size = int(model_chunk_size)
    config.n_action_steps = int(execution_horizon)
    config.num_inference_steps = int(num_inference_steps)


def _set_tokenizer_padding_side(processor: Any, side: str) -> None:
    """Set the tokenizer object, not only LeRobot's serialized hint.

    Transformers' PaliGemma tokenizer defaults to left padding and ignores the
    extra ``padding_side`` keyword passed by LeRobot 0.6 at call time. OpenPI
    builds right-padded arrays, so raw OpenPI checkpoints require changing the
    tokenizer instance itself.
    """

    if side not in {"left", "right"}:
        raise ValueError("tokenizer padding side must be 'left' or 'right'")
    matches = 0
    for step in getattr(processor, "steps", ()):
        tokenizer = getattr(step, "input_tokenizer", None)
        if tokenizer is None:
            continue
        tokenizer.padding_side = side
        matches += 1
    if matches != 1:
        raise RuntimeError(
            "RLinf/OpenPI processor must contain exactly one initialized text "
            f"tokenizer; found {matches}"
        )


def _load_rlinf_openpi_normalization_stats(
    *,
    model_id: str,
    revision: str | None,
    filename: str,
    state_dim: int,
    action_dim: int,
) -> dict[str, dict[str, Any]]:
    """Load OpenPI norm stats and expose LeRobot's canonical feature names."""

    import json
    from pathlib import Path

    import torch

    source = Path(model_id).expanduser()
    if source.is_dir():
        stats_path = source / filename
    else:
        from huggingface_hub import hf_hub_download

        stats_path = Path(
            hf_hub_download(
                repo_id=model_id,
                filename=filename,
                revision=revision,
            )
        )
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    raw = payload.get("norm_stats", payload)
    try:
        state = raw["state"]
        action = raw["actions"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Invalid RLinf/OpenPI normalization stats: {stats_path}"
        ) from exc

    def convert(values: Any, *, size: int, label: str) -> dict[str, torch.Tensor]:
        if not isinstance(values, dict):
            raise ValueError(f"Normalization entry {label!r} must be a mapping")
        converted = {}
        for key in ("mean", "std", "q01", "q99"):
            tensor = torch.as_tensor(values[key], dtype=torch.float32)
            if tensor.ndim != 1 or tensor.numel() < size:
                raise ValueError(
                    f"Normalization entry {label}.{key} has shape "
                    f"{tuple(tensor.shape)}; expected at least {size} values"
                )
            converted[key] = tensor[:size].clone()
        return converted

    from lerobot.utils.constants import ACTION, OBS_STATE

    return {
        OBS_STATE: convert(state, size=state_dim, label="state"),
        ACTION: convert(action, size=action_dim, label="actions"),
    }
