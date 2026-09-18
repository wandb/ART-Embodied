"""Policy construction from the embodied experiment contract.

Built-in policies validate their own loader-specific configuration. External
LeRobot integrations can register another factory without adding model-family
conditionals to the experiment runner.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeAlias

from ..config import (
    EmbodiedExperimentConfig,
    GR00TFlowLoadConfig,
    GR00TN17FlowLoadConfig,
    OpenVLALoadConfig,
    PI0FastLoadConfig,
    PIFlowLoadConfig,
    SmolVLAFlowLoadConfig,
)
from ..policy_capabilities import PolicyCapabilities
from ..vla_trainable import configure_vla_trainable_parameters
from .openvla import OpenVLAPolicy, raise_for_openvla_oft_v01_runtime
from .rng_streams import seed_process_rng

if TYPE_CHECKING:
    from .pi import PIFlowPolicy

PolicyFactory: TypeAlias = Callable[..., Any]
PolicyCapabilitiesFactory: TypeAlias = Callable[
    [EmbodiedExperimentConfig], PolicyCapabilities
]

_POLICY_FACTORIES: dict[str, PolicyFactory] = {}
_POLICY_CAPABILITY_FACTORIES: dict[str, PolicyCapabilitiesFactory] = {}


def register_policy_factory(
    policy_type: str,
    factory: PolicyFactory,
    *,
    capabilities: PolicyCapabilitiesFactory | None = None,
    replace: bool = False,
) -> None:
    """Register a policy constructor for ``policy.type``.

    Factories receive ``(config, load=...)``. Registration is intentionally a
    small Python extension point: experiment conditions still live in YAML,
    while model-family code remains owned by its integration package.
    """

    normalized = policy_type.strip()
    if not normalized:
        raise ValueError("policy_type cannot be empty")
    if not callable(factory):
        raise TypeError("policy factory must be callable")
    if normalized in _POLICY_FACTORIES and not replace:
        raise ValueError(f"policy factory already registered: {normalized}")
    _POLICY_FACTORIES[normalized] = factory
    if capabilities is not None:
        _POLICY_CAPABILITY_FACTORIES[normalized] = capabilities
    elif replace:
        _POLICY_CAPABILITY_FACTORIES.pop(normalized, None)


def registered_policy_types() -> tuple[str, ...]:
    """Return policy types available in this Python process."""

    return tuple(sorted(_POLICY_FACTORIES))


def policy_capabilities(config: EmbodiedExperimentConfig) -> PolicyCapabilities:
    """Resolve the declared model-family contract before loading weights."""

    try:
        factory = _POLICY_CAPABILITY_FACTORIES[config.policy.type]
    except KeyError as exc:
        raise ValueError(
            f"Policy {config.policy.type!r} has no probability capability declaration. "
            "Register one with register_policy_factory(..., capabilities=...)."
        ) from exc
    capabilities = factory(config)
    if capabilities.policy_type != config.policy.type:
        raise ValueError(
            "Policy capability declaration returned the wrong policy_type: "
            f"{capabilities.policy_type!r} != {config.policy.type!r}"
        )
    return capabilities


def make_policy(
    config: EmbodiedExperimentConfig,
    *,
    load: bool = True,
) -> Any:
    """Construct the policy selected by the explicit YAML ``policy.type``."""

    # LoRA A matrices and other trainable adapters are initialized while the
    # policy is built. Tie that initialization to the declared experiment seed;
    # rollout seeds alone cannot make two training runs follow the same path.
    seed_process_rng(config.experiment.seed)
    policy_type = config.policy.type
    try:
        factory = _POLICY_FACTORIES[policy_type]
    except KeyError as exc:
        available = ", ".join(registered_policy_types()) or "<none>"
        raise ValueError(
            f"No embodied policy factory registered for {policy_type!r}. "
            f"Registered types: {available}"
        ) from exc
    return factory(config, load=load)


def make_openvla_oft_policy(
    config: EmbodiedExperimentConfig,
    *,
    load: bool = True,
) -> OpenVLAPolicy:
    """Build and configure the supported OpenVLA-OFT policy from YAML.

    LoRA is attached here, before backend construction, so the declared
    trainable surface cannot be accidentally omitted by an example runner.
    """

    if config.policy.type != "openvla_oft":
        raise ValueError("make_openvla_oft_policy requires policy.type='openvla_oft'")
    load_config = OpenVLALoadConfig.model_validate(config.policy.load_kwargs)
    if config.policy.revision is not None and load_config.model_loader == "rlinf":
        raise ValueError(
            "OpenVLA-OFT revision pinning is supported by the native/Transformers "
            "loader only; use a revision-pinned local snapshot with model_loader='rlinf'"
        )

    _validate_openvla_generation(config)
    generation = config.policy.rollout_generation
    policy = OpenVLAPolicy(
        model_id=config.policy.path,
        revision=config.policy.revision,
        device=config.policy.device,
        dtype=config.policy.dtype,
        unnorm_key=config.policy.unnorm_key,
        robot_platform=load_config.robot_platform,
        attn_implementation=load_config.attn_implementation,
        trust_remote_code=config.policy.trust_remote_code,
        load_on_init=False,
        capture_action_tokens=True,
        action_output_kind="token",
        do_sample=generation.do_sample,
        temperature=generation.temperature,
        peft_adapter_path=load_config.peft_adapter_path,
        dataset_statistics_path=load_config.dataset_statistics_path,
        logprob_batch_size=load_config.logprob_batch_size,
        strict_batched_logprobs=load_config.strict_batched_logprobs,
        max_prompt_length=load_config.max_prompt_length,
        prompt_template=load_config.prompt_template,
        lowercase_instruction=load_config.lowercase_instruction,
        num_images_in_input=load_config.num_images_in_input,
        use_proprio=load_config.use_proprio,
        action_dim=load_config.action_dim,
        num_action_chunks=load_config.num_action_chunks,
        model_loader=load_config.model_loader,
    )
    if load:
        if load_config.runtime_contract == "openvla_oft_v01":
            raise_for_openvla_oft_v01_runtime(
                require_peft=(
                    config.policy.lora.enabled
                    or load_config.peft_adapter_path is not None
                )
            )
        policy.load()
        policy, report = configure_vla_trainable_parameters(
            policy,
            algorithm_cfg=_trainable_parameter_config(config),
            policy_type="openvla_oft",
        )
        if not report.get("ok"):
            raise RuntimeError(
                "OpenVLA-OFT trainable parameter configuration failed: "
                f"{report.get('error') or report.get('warnings')}"
            )
        policy.trainable_report = report
    return policy


def make_pi_flow_policy(
    config: EmbodiedExperimentConfig,
    *,
    load: bool = True,
) -> "PIFlowPolicy":
    """Build a LeRobot PI0/PI0.5 policy with an explicit Flow-SDE contract."""

    if config.policy.type not in {"pi0", "pi05"}:
        raise ValueError("make_pi_flow_policy requires policy.type='pi0' or 'pi05'")
    load_config = PIFlowLoadConfig.model_validate(config.policy.load_kwargs)
    flow_sde = config.algorithm.flow_sde
    if flow_sde is None:
        raise ValueError(
            f"policy.type={config.policy.type!r} requires algorithm.flow_sde"
        )
    if config.algorithm.type != "grpo":
        raise ValueError("PI Flow-SDE currently supports algorithm.type='grpo' only")
    if config.rollout.action_payload.kind != "continuous":
        raise ValueError(
            "PI Flow-SDE requires rollout.action_payload.kind='continuous'"
        )
    if config.policy.dtype not in {"bfloat16", "float32"}:
        raise ValueError("LeRobot PI policies require dtype='bfloat16' or 'float32'")
    from .flow_sde import FlowSDESchedule
    from .pi import PIFlowPolicy

    policy = PIFlowPolicy(
        family=config.policy.type,
        model_id=config.policy.path,
        revision=config.policy.revision,
        device=config.policy.device,
        dtype=config.policy.dtype,
        model_format=load_config.model_format,
        execution_horizon=load_config.execution_horizon,
        action_dim=load_config.action_dim,
        processor_path=load_config.processor_path,
        processor_revision=load_config.processor_revision,
        model_chunk_size=load_config.model_chunk_size,
        normalization_stats_file=load_config.normalization_stats_file,
        discrete_state_input=load_config.discrete_state_input,
        extra_delta_transform=load_config.extra_delta_transform,
        observation_key_map=load_config.observation_key_map,
        schedule=FlowSDESchedule(
            num_steps=flow_sde.num_denoise_steps,
            noise_level=flow_sde.noise_level,
        ),
        strict_weights=load_config.strict_weights,
        compile_model=load_config.compile_model,
        gradient_checkpointing=load_config.gradient_checkpointing,
        train_expert_only=load_config.train_expert_only,
    )
    if load:
        policy.load()
        policy, report = configure_vla_trainable_parameters(
            policy,
            algorithm_cfg=_trainable_parameter_config(config),
            policy_type=config.policy.type,
        )
        if not report.get("ok"):
            raise RuntimeError(
                "PI trainable parameter configuration failed: "
                f"{report.get('error') or report.get('warnings')}"
            )
        policy.trainable_report = report
    return policy


def make_smolvla_flow_policy(
    config: EmbodiedExperimentConfig,
    *,
    load: bool = True,
) -> Any:
    """Build SmolVLA from its serialized LeRobot checkpoint contract."""

    if config.policy.type != "smolvla":
        raise ValueError("make_smolvla_flow_policy requires policy.type='smolvla'")
    load_config = SmolVLAFlowLoadConfig.model_validate(config.policy.load_kwargs)
    flow_sde = config.algorithm.flow_sde
    if flow_sde is None:
        raise ValueError("policy.type='smolvla' requires algorithm.flow_sde")
    from .flow_sde import FlowSDESchedule
    from .smolvla import SmolVLAFlowPolicy

    policy = SmolVLAFlowPolicy(
        model_id=config.policy.path,
        revision=config.policy.revision,
        device=config.policy.device,
        execution_horizon=load_config.execution_horizon,
        action_dim=load_config.action_dim,
        observation_key_map=load_config.observation_key_map,
        schedule=FlowSDESchedule(
            num_steps=flow_sde.num_denoise_steps,
            noise_level=flow_sde.noise_level,
            deterministic_sampler="native_euler",
        ),
        strict_weights=load_config.strict_weights,
        compile_model=load_config.compile_model,
        train_expert_only=load_config.train_expert_only,
    )
    if load:
        policy.load()
        policy, report = configure_vla_trainable_parameters(
            policy,
            algorithm_cfg=_trainable_parameter_config(config),
            policy_type="smolvla",
        )
        if not report.get("ok"):
            raise RuntimeError(
                "SmolVLA trainable parameter configuration failed: "
                f"{report.get('error') or report.get('warnings')}"
            )
        policy.trainable_report = report
    return policy


def make_pi0_fast_policy(
    config: EmbodiedExperimentConfig,
    *,
    load: bool = True,
) -> Any:
    """Build LeRobot pi0-FAST with exact categorical action probabilities."""

    if config.policy.type != "pi0_fast":
        raise ValueError("make_pi0_fast_policy requires policy.type='pi0_fast'")
    load_config = PI0FastLoadConfig.model_validate(config.policy.load_kwargs)
    if config.algorithm.flow_sde is not None:
        raise ValueError("pi0-FAST requires algorithm.flow_sde=null")
    if config.rollout.action_payload.kind != "token":
        raise ValueError("pi0-FAST requires token action payloads")
    from .pi0_fast import PI0FastPolicy

    policy = PI0FastPolicy(
        model_id=config.policy.path,
        revision=config.policy.revision,
        device=config.policy.device,
        execution_horizon=load_config.execution_horizon,
        action_dim=load_config.action_dim,
        max_decoding_steps=load_config.max_decoding_steps,
        action_tokenizer_revision=load_config.action_tokenizer_revision,
        observation_key_map=load_config.observation_key_map,
        strict_weights=load_config.strict_weights,
        compile_model=load_config.compile_model,
        gradient_checkpointing=load_config.gradient_checkpointing,
        use_kv_cache=load_config.use_kv_cache,
        model_compute_dtype=load_config.model_compute_dtype,
        training_loss_scale=load_config.training_loss_scale,
        training_logprob_mode=load_config.training_logprob_mode,
        rl_token_scope=load_config.rl_token_scope,
    )
    if load:
        policy.load()
        policy, report = configure_vla_trainable_parameters(
            policy,
            algorithm_cfg=_trainable_parameter_config(config),
            policy_type="pi0_fast",
        )
        if not report.get("ok"):
            raise RuntimeError(
                "pi0-FAST trainable parameter configuration failed: "
                f"{report.get('error') or report.get('warnings')}"
            )
        report["model_compute_dtype"] = load_config.model_compute_dtype
        report["training_loss_scale"] = load_config.training_loss_scale
        report["training_logprob_mode"] = load_config.training_logprob_mode
        if load_config.model_compute_dtype == "fp16_residual":
            from .pi0_fast_precision import install_fp16_residual

            report["precision_installation"] = install_fp16_residual(
                policy.model,
                policy.model.paligemma_with_expert.paligemma.model.language_model,
            )
        if load_config.warm_start_checkpoint is not None:
            from art_embodied.checkpointing import CheckpointManager

            validation = CheckpointManager().validate_payload(
                load_config.warm_start_checkpoint
            )
            assert validation.manifest is not None
            metadata = validation.manifest.get("metadata") or {}
            if metadata.get("backend") != "pi0_fast_language_sft":
                raise ValueError(
                    "pi0-FAST warm start must be a language-SFT checkpoint"
                )
            policy.load_checkpoint(validation.path / "policy")
            report["warm_start"] = {
                "kind": "pi0_fast_language_sft",
                "checkpoint": str(validation.path.resolve()),
                "step": metadata.get("step"),
                "task_index": metadata.get("task_index"),
            }
        policy.trainable_report = report
    return policy


def make_gr00t_n1d5_flow_policy(
    config: EmbodiedExperimentConfig,
    *,
    load: bool = True,
) -> Any:
    """Build the audited NVIDIA N1.5 Flow-SDE policy boundary."""

    if config.policy.type != "gr00t_n1d5":
        raise ValueError(
            "make_gr00t_n1d5_flow_policy requires policy.type='gr00t_n1d5'"
        )
    load_config = GR00TFlowLoadConfig.model_validate(config.policy.load_kwargs)
    flow_sde = config.algorithm.flow_sde
    if flow_sde is None:
        raise ValueError("policy.type='gr00t_n1d5' requires algorithm.flow_sde")
    from .flow_sde import FlowSDESchedule
    from .gr00t import GR00TN15FlowPolicy

    policy = GR00TN15FlowPolicy(
        model_id=config.policy.path,
        revision=config.policy.revision,
        device=config.policy.device,
        data_config=load_config.data_config,
        embodiment_tag=load_config.embodiment_tag,
        execution_horizon=load_config.execution_horizon,
        action_dim=load_config.action_dim,
        model_action_horizon=load_config.model_action_horizon,
        language_padding_length=load_config.language_padding_length,
        schedule=FlowSDESchedule(
            num_steps=flow_sde.num_denoise_steps,
            noise_level=flow_sde.noise_level,
            noise_time="zero",
        ),
        disable_dropout=load_config.disable_dropout,
    )
    if load:
        policy.load()
        policy, report = configure_vla_trainable_parameters(
            policy,
            algorithm_cfg=_trainable_parameter_config(config),
            policy_type="gr00t_n1d5",
        )
        if not report.get("ok"):
            raise RuntimeError(
                "GR00T N1.5 trainable parameter configuration failed: "
                f"{report.get('error') or report.get('warnings')}"
            )
        policy.trainable_report = report
    return policy


def make_gr00t_n1d7_flow_policy(
    config: EmbodiedExperimentConfig,
    *,
    load: bool = True,
) -> Any:
    """Build the pinned NVIDIA N1.7 Flow-SDE policy boundary."""

    if config.policy.type != "gr00t_n1d7":
        raise ValueError(
            "make_gr00t_n1d7_flow_policy requires policy.type='gr00t_n1d7'"
        )
    load_config = GR00TN17FlowLoadConfig.model_validate(config.policy.load_kwargs)
    flow_sde = config.algorithm.flow_sde
    if flow_sde is None:
        raise ValueError("policy.type='gr00t_n1d7' requires algorithm.flow_sde")
    from .flow_sde import FlowSDESchedule
    from .gr00t_n1d7 import GR00TN17FlowPolicy

    policy = GR00TN17FlowPolicy(
        model_id=config.policy.path,
        revision=config.policy.revision,
        checkpoint_subfolder=load_config.checkpoint_subfolder,
        device=config.policy.device,
        embodiment_tag=load_config.embodiment_tag,
        execution_horizon=load_config.execution_horizon,
        processor_action_horizon=load_config.processor_action_horizon,
        action_dim=load_config.action_dim,
        action_components=tuple(
            (
                component.key,
                component.size,
                component.executed,
                component.execution_order,
            )
            for component in load_config.action_components
        ),
        model_action_horizon=load_config.model_action_horizon,
        schedule=FlowSDESchedule(
            num_steps=flow_sde.num_denoise_steps,
            noise_level=flow_sde.noise_level,
            noise_time="zero",
        ),
        disable_dropout=load_config.disable_dropout,
    )
    if load:
        policy.load()
        policy, report = configure_vla_trainable_parameters(
            policy,
            algorithm_cfg=_trainable_parameter_config(config),
            policy_type="gr00t_n1d7",
        )
        if not report.get("ok"):
            raise RuntimeError(
                "GR00T N1.7 trainable parameter configuration failed: "
                f"{report.get('error') or report.get('warnings')}"
            )
        policy.trainable_report = report
    return policy


def _validate_openvla_generation(config: EmbodiedExperimentConfig) -> None:
    # OpenVLAPolicy currently implements categorical temperature sampling but
    # not top-p truncation. Keep this limitation out of the generic config so
    # another registered policy can implement top-p correctly.
    for name, generation in (
        ("rollout_generation", config.policy.rollout_generation),
        ("train_generation", config.policy.train_generation),
        ("evaluation_generation", config.policy.evaluation_generation),
    ):
        if generation.top_p != 1.0:
            raise ValueError(
                f"policy.{name}.top_p must be 1.0 for the built-in "
                "OpenVLA-OFT policy until top-p sampling is implemented"
            )


def _trainable_parameter_config(
    config: EmbodiedExperimentConfig,
) -> dict[str, Any]:
    lora = config.policy.lora
    return {
        "trainable_parameter_strategy": config.policy.trainable_parameter_strategy,
        "force_trainable_float32": config.policy.force_trainable_float32,
        "trainable_parameter_patterns": [],
        "allow_embedding_training": False,
        "gradient_checkpointing": False,
        "peft": {
            "enabled": lora.enabled,
            "r": lora.rank,
            "lora_alpha": lora.alpha,
            "lora_dropout": lora.dropout,
            "bias": "none",
            "target_modules": list(lora.target_modules),
            "modules_to_save": [],
            "init_lora_weights": ("gaussian" if lora.init == "gaussian" else True),
            "rank_partition": (
                lora.rank_partition.model_dump(mode="python")
                if lora.rank_partition is not None
                else None
            ),
            # Explicit targets in YAML are the complete trainable surface.
            "apply_layer_selection": False,
        },
    }


def _openvla_oft_capabilities(
    config: EmbodiedExperimentConfig,
) -> PolicyCapabilities:
    load_config = OpenVLALoadConfig.model_validate(config.policy.load_kwargs)
    return PolicyCapabilities(
        policy_type="openvla_oft",
        action_kind="token",
        probability_model="categorical_tokens",
        action_shape=(load_config.num_action_chunks, load_config.action_dim),
        chunk_horizon=load_config.num_action_chunks,
        exact_logprobs=True,
        teacher_forced_rescore=True,
        rng_replay=True,
        batched_inference=True,
        checkpoint_delta="either",
        observation_normalization_revision=(
            config.policy.revision or "unversioned-model-statistics"
        ),
    )


def _pi_flow_capabilities(
    config: EmbodiedExperimentConfig,
) -> PolicyCapabilities:
    load_config = PIFlowLoadConfig.model_validate(config.policy.load_kwargs)
    if config.algorithm.flow_sde is None:
        raise ValueError(
            f"policy.type={config.policy.type!r} requires algorithm.flow_sde"
        )
    return PolicyCapabilities(
        policy_type=config.policy.type,
        action_kind="continuous",
        probability_model="gaussian_flow_sde",
        action_shape=(
            load_config.execution_horizon,
            load_config.action_dim,
        ),
        chunk_horizon=load_config.execution_horizon,
        exact_logprobs=True,
        teacher_forced_rescore=True,
        rng_replay=True,
        batched_inference=True,
        checkpoint_delta="either",
        observation_normalization_revision=(
            config.policy.revision or "unversioned-lerobot-processors"
        ),
    )


def _smolvla_flow_capabilities(
    config: EmbodiedExperimentConfig,
) -> PolicyCapabilities:
    load_config = SmolVLAFlowLoadConfig.model_validate(config.policy.load_kwargs)
    if config.algorithm.flow_sde is None:
        raise ValueError("policy.type='smolvla' requires algorithm.flow_sde")
    return PolicyCapabilities(
        policy_type="smolvla",
        action_kind="continuous",
        probability_model="gaussian_flow_sde",
        action_shape=(load_config.execution_horizon, load_config.action_dim),
        chunk_horizon=load_config.execution_horizon,
        exact_logprobs=True,
        teacher_forced_rescore=True,
        rng_replay=True,
        batched_inference=True,
        checkpoint_delta="either",
        observation_normalization_revision=(
            config.policy.revision or "unversioned-lerobot-processors"
        ),
    )


def _pi0_fast_capabilities(
    config: EmbodiedExperimentConfig,
) -> PolicyCapabilities:
    load_config = PI0FastLoadConfig.model_validate(config.policy.load_kwargs)
    return PolicyCapabilities(
        policy_type="pi0_fast",
        action_kind="token",
        probability_model="categorical_tokens",
        action_shape=(load_config.execution_horizon, load_config.action_dim),
        chunk_horizon=load_config.execution_horizon,
        exact_logprobs=True,
        teacher_forced_rescore=True,
        rng_replay=True,
        batched_inference=True,
        checkpoint_delta="either",
        observation_normalization_revision=(
            config.policy.revision or "unversioned-lerobot-processors"
        ),
    )


def _gr00t_n1d5_flow_capabilities(
    config: EmbodiedExperimentConfig,
) -> PolicyCapabilities:
    load_config = GR00TFlowLoadConfig.model_validate(config.policy.load_kwargs)
    if config.algorithm.flow_sde is None:
        raise ValueError("policy.type='gr00t_n1d5' requires algorithm.flow_sde")
    return PolicyCapabilities(
        policy_type="gr00t_n1d5",
        action_kind="continuous",
        probability_model="gaussian_flow_sde",
        action_shape=(load_config.execution_horizon, load_config.action_dim),
        chunk_horizon=load_config.execution_horizon,
        exact_logprobs=True,
        teacher_forced_rescore=True,
        rng_replay=True,
        batched_inference=True,
        checkpoint_delta="either",
        observation_normalization_revision=(
            config.policy.revision or "unversioned-gr00t-experiment-metadata"
        ),
    )


def _gr00t_n1d7_flow_capabilities(
    config: EmbodiedExperimentConfig,
) -> PolicyCapabilities:
    load_config = GR00TN17FlowLoadConfig.model_validate(config.policy.load_kwargs)
    if config.algorithm.flow_sde is None:
        raise ValueError("policy.type='gr00t_n1d7' requires algorithm.flow_sde")
    return PolicyCapabilities(
        policy_type="gr00t_n1d7",
        action_kind="continuous",
        probability_model="gaussian_flow_sde",
        action_shape=(
            load_config.execution_horizon,
            load_config.execution_action_dim,
        ),
        chunk_horizon=load_config.execution_horizon,
        exact_logprobs=True,
        teacher_forced_rescore=True,
        rng_replay=True,
        batched_inference=True,
        checkpoint_delta="either",
        observation_normalization_revision=(
            config.policy.revision or "unversioned-gr00t-n1d7-processor"
        ),
    )


register_policy_factory(
    "openvla_oft",
    make_openvla_oft_policy,
    capabilities=_openvla_oft_capabilities,
)
register_policy_factory(
    "pi0",
    make_pi_flow_policy,
    capabilities=_pi_flow_capabilities,
)
register_policy_factory(
    "pi05",
    make_pi_flow_policy,
    capabilities=_pi_flow_capabilities,
)
register_policy_factory(
    "smolvla",
    make_smolvla_flow_policy,
    capabilities=_smolvla_flow_capabilities,
)
register_policy_factory(
    "pi0_fast",
    make_pi0_fast_policy,
    capabilities=_pi0_fast_capabilities,
)
register_policy_factory(
    "gr00t_n1d5",
    make_gr00t_n1d5_flow_policy,
    capabilities=_gr00t_n1d5_flow_capabilities,
)
register_policy_factory(
    "gr00t_n1d7",
    make_gr00t_n1d7_flow_policy,
    capabilities=_gr00t_n1d7_flow_capabilities,
)
