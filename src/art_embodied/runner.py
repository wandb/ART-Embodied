"""High-level, low-friction entry point for LeRobot-owned experiments."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .backends.factory import make_embodied_backend
from .config import EmbodiedExperimentConfig
from .evaluation import FixedScenarioEvaluator
from .experiment import (
    EmbodiedExperiment,
    EmbodiedScenario,
    EvaluationFunction,
    EvaluationResult,
    ExperimentProgress,
    ExperimentStepResult,
    RolloutFunction,
)
from .integrations.lerobot import (
    LeRobotPolicyAdapterProtocol,
    SharedLeRobotPolicyAdapterFactory,
)
from .integrations.lerobot_rollout import (
    EnvironmentFactory,
    LeRobotEpisodeRollout,
    ObservationRecorder,
    PolicyAdapterFactory,
    SuccessFunction,
    TransitionRecorder,
)
from .observability import WandbWeaveObserver
from .rollout_process import LocalProcessRolloutPool
from .types import LocalTrainResult

if TYPE_CHECKING:
    from .art_compat import EmbodiedBackend


def _backend_snapshot_provider(backend: Any) -> Any | None:
    """Return a backend-owned rollout snapshot boundary when available."""

    candidate = getattr(backend, "backend", backend)
    required = ("save_snapshot", "offload", "restore")
    if all(callable(getattr(candidate, name, None)) for name in required):
        return candidate
    return None


@dataclass(frozen=True, slots=True)
class EmbodiedRunResult:
    """Completed update results and the immutable experiment fingerprint."""

    config_fingerprint: str
    steps: tuple[ExperimentStepResult, ...]


@dataclass(frozen=True, slots=True)
class EmbodiedEvaluationRunResult:
    """One fixed held-out evaluation and its immutable config fingerprint."""

    config_fingerprint: str
    evaluation: EvaluationResult


def validate_runtime_device_availability(
    config: EmbodiedExperimentConfig,
    *,
    include_training_devices: bool = True,
) -> None:
    """Fail before logging or model work when YAML names invisible CUDA devices."""

    devices = list(config.runtime.rollout_devices)
    if include_training_devices:
        devices.extend(config.runtime.training_devices)
    configured = {device for device in devices if device.startswith("cuda:")}
    if not configured:
        return
    import torch

    visible_count = int(torch.cuda.device_count())
    invalid = sorted(
        device for device in configured if int(device.split(":", 1)[1]) >= visible_count
    )
    if invalid:
        raise RuntimeError(
            "ART-Embodied YAML requests CUDA devices that are not visible to "
            f"this process: {', '.join(invalid)}; visible cuda device count="
            f"{visible_count}. Fix the requested runtime devices or the "
            "resource allocation."
        )


async def run_lerobot_evaluation(
    *,
    config: EmbodiedExperimentConfig,
    policy: Any,
    evaluation_scenarios: Sequence[EmbodiedScenario],
    environment_factory: EnvironmentFactory | None = None,
    policy_adapter_factory: PolicyAdapterFactory | None = None,
    policy_adapter: LeRobotPolicyAdapterProtocol | None = None,
    success_fn: SuccessFunction | None = None,
    observation_recorder: ObservationRecorder | None = None,
    transition_recorder: TransitionRecorder | None = None,
    success_metric: str = "success",
    step: int = 0,
    checkpoint_path: str | Path | None = None,
    observer: WandbWeaveObserver | None = None,
    checkpoint_role: str = "candidate",
    use_native_wandb_step: bool = False,
    measured_baseline_path: str | Path | None = None,
) -> EmbodiedEvaluationRunResult:
    """Evaluate a LeRobot policy without constructing a training backend."""

    if config.runtime.rollout_execution.mode == "local_process":
        supplied = sorted(
            name
            for name, value in {
                "environment_factory": environment_factory,
                "policy_adapter_factory": policy_adapter_factory,
                "policy_adapter": policy_adapter,
                "success_fn": success_fn,
                "observation_recorder": observation_recorder,
                "transition_recorder": transition_recorder,
            }.items()
            if value is not None
        )
        if supplied:
            raise ValueError(
                "local-process evaluation actors own their LeRobot components; "
                "remove parent-process inputs: " + ", ".join(supplied)
            )
        return await run_embodied_evaluation(
            config=config,
            policy=policy,
            evaluation_scenarios=evaluation_scenarios,
            rollout=None,
            success_metric=success_metric,
            step=step,
            checkpoint_path=checkpoint_path,
            observer=observer,
            checkpoint_role=checkpoint_role,
            use_native_wandb_step=use_native_wandb_step,
            measured_baseline_path=measured_baseline_path,
        )

    if environment_factory is None:
        raise ValueError("in-process LeRobot evaluation requires environment_factory")
    resolved_policy_adapter_factory = _resolve_policy_adapter_factory(
        config=config,
        policy=policy,
        policy_adapter=policy_adapter,
        policy_adapter_factory=policy_adapter_factory,
        label="evaluation",
    )
    rollout = LeRobotEpisodeRollout.from_config(
        config,
        environment_factory=environment_factory,
        policy_adapter_factory=resolved_policy_adapter_factory,
        training_policy=policy,
        phase="eval",
        success_fn=success_fn,
        observation_recorder=observation_recorder,
        transition_recorder=transition_recorder,
    )
    return await run_embodied_evaluation(
        config=config,
        policy=policy,
        evaluation_scenarios=evaluation_scenarios,
        rollout=rollout,
        success_metric=success_metric,
        step=step,
        checkpoint_path=checkpoint_path,
        observer=observer,
        checkpoint_role=checkpoint_role,
        use_native_wandb_step=use_native_wandb_step,
        measured_baseline_path=measured_baseline_path,
    )


async def run_embodied_evaluation(
    *,
    config: EmbodiedExperimentConfig,
    policy: Any,
    evaluation_scenarios: Sequence[EmbodiedScenario],
    rollout: RolloutFunction | None,
    success_metric: str = "success",
    step: int = 0,
    checkpoint_path: str | Path | None = None,
    observer: WandbWeaveObserver | None = None,
    checkpoint_role: str = "candidate",
    use_native_wandb_step: bool = False,
    measured_baseline_path: str | Path | None = None,
) -> EmbodiedEvaluationRunResult:
    """Run fixed held-out evaluation with W&B/Weave, but no optimizer state."""

    if not config.evaluation.enabled:
        raise ValueError("run_embodied_evaluation requires evaluation.enabled=true")
    if step < 0:
        raise ValueError("evaluation step cannot be negative")
    if config.runtime.rollout_execution.mode == "local_process" or rollout is None:
        validate_runtime_device_availability(
            config,
            include_training_devices=False,
        )

    if observer is None and config.observability.wandb.enabled:
        raise ValueError(
            "Standalone evaluation cannot safely attach W&B metrics to a training "
            "run: shared mode cannot preserve ART-Embodied's native policy-version "
            "steps or upload evaluation Artifacts, while resume cannot backfill "
            "past history. Pass the live training coordinator observer so it owns "
            "both training and evaluation logging, or disable W&B and retain the "
            "local evidence bundle."
        )

    owns_observer = observer is None
    resolved_observer = observer or WandbWeaveObserver.start(config)
    process_rollout_pool: LocalProcessRolloutPool | None = None
    run_error: BaseException | None = None
    try:
        resolved_rollout = rollout
        if config.runtime.rollout_execution.mode == "local_process":
            if rollout is not None:
                raise ValueError(
                    "runtime.rollout_execution.mode='local_process' owns the "
                    "evaluation rollout; pass rollout=None"
                )
            process_rollout_pool = LocalProcessRolloutPool(
                config=config,
                policy=policy,
            )
            resolved_rollout = process_rollout_pool.for_phase("eval")
        elif resolved_rollout is None:
            raise ValueError(
                "runtime.rollout_execution.mode='in_process' requires rollout"
            )
        assert resolved_rollout is not None

        evaluator = FixedScenarioEvaluator(
            config=config,
            scenarios=evaluation_scenarios,
            rollout=resolved_rollout,
            success_metric=success_metric,
            log_progress=getattr(resolved_observer, "log_progress", None),
            measured_baseline_path=measured_baseline_path,
        )
        with _policy_generation(
            policy,
            do_sample=config.policy.evaluation_generation.do_sample,
            temperature=config.policy.evaluation_generation.temperature,
        ):
            evaluation = await evaluator(
                step,
                LocalTrainResult(
                    step=step,
                    metrics={},
                    checkpoint_path=(
                        str(Path(checkpoint_path).expanduser().resolve())
                        if checkpoint_path is not None
                        else None
                    ),
                ),
                config,
            )
        log_checkpoint = getattr(resolved_observer, "log_evaluation_checkpoint", None)
        if callable(log_checkpoint):
            await log_checkpoint(
                step,
                evaluation,
                config,
                checkpoint_role=checkpoint_role,
                use_native_wandb_step=use_native_wandb_step,
            )
        else:  # Preserve the existing structural observer protocol.
            await resolved_observer.log_evaluation(step, evaluation, config)
    except BaseException as exc:
        run_error = exc
        raise
    finally:
        if process_rollout_pool is not None:
            await process_rollout_pool.close()
        if owns_observer:
            resolved_observer.close(exit_code=1 if run_error is not None else 0)

    return EmbodiedEvaluationRunResult(
        config_fingerprint=config.fingerprint,
        evaluation=evaluation,
    )


async def run_lerobot_experiment(
    *,
    config: EmbodiedExperimentConfig,
    policy: Any,
    train_scenarios: Sequence[EmbodiedScenario],
    environment_factory: EnvironmentFactory | None = None,
    policy_adapter_factory: PolicyAdapterFactory | None = None,
    policy_adapter: LeRobotPolicyAdapterProtocol | None = None,
    evaluation_scenarios: Sequence[EmbodiedScenario] | None = None,
    evaluation_environment_factory: EnvironmentFactory | None = None,
    evaluation_policy_adapter_factory: PolicyAdapterFactory | None = None,
    success_fn: SuccessFunction | None = None,
    evaluation_success_fn: SuccessFunction | None = None,
    observation_recorder: ObservationRecorder | None = None,
    evaluation_observation_recorder: ObservationRecorder | None = None,
    transition_recorder: TransitionRecorder | None = None,
    evaluation_transition_recorder: TransitionRecorder | None = None,
    evaluation_success_metric: str = "success",
    backend: Any | None = None,
    observer: WandbWeaveObserver | None = None,
) -> EmbodiedRunResult:
    """Run a Gymnasium-compatible LeRobot experiment from native factories.

    The helper keeps train and evaluation episode runners separate so phase
    metadata, video limits, and output paths remain correct without requiring
    users to duplicate runner plumbing.
    """

    if config.runtime.rollout_execution.mode == "local_process":
        process_owned_inputs = {
            "environment_factory": environment_factory,
            "policy_adapter_factory": policy_adapter_factory,
            "policy_adapter": policy_adapter,
            "evaluation_environment_factory": evaluation_environment_factory,
            "evaluation_policy_adapter_factory": evaluation_policy_adapter_factory,
            "success_fn": success_fn,
            "evaluation_success_fn": evaluation_success_fn,
            "observation_recorder": observation_recorder,
            "evaluation_observation_recorder": evaluation_observation_recorder,
            "transition_recorder": transition_recorder,
            "evaluation_transition_recorder": evaluation_transition_recorder,
        }
        supplied = sorted(
            name for name, value in process_owned_inputs.items() if value is not None
        )
        if supplied:
            raise ValueError(
                "local-process rollout actors construct their native LeRobot "
                "policy, environment, processors, and reward hooks inside the "
                "configured actor_factory; remove parent-process inputs: "
                + ", ".join(supplied)
            )
        return await run_embodied_experiment(
            config=config,
            policy=policy,
            train_scenarios=train_scenarios,
            rollout=None,
            evaluation_scenarios=evaluation_scenarios,
            evaluation_rollout=None,
            evaluation_success_metric=evaluation_success_metric,
            backend=backend,
            observer=observer,
        )

    if environment_factory is None:
        raise ValueError("in-process LeRobot rollout requires environment_factory")

    resolved_policy_adapter_factory = _resolve_policy_adapter_factory(
        config=config,
        policy=policy,
        policy_adapter=policy_adapter,
        policy_adapter_factory=policy_adapter_factory,
        label="training",
    )
    train_rollout = LeRobotEpisodeRollout.from_config(
        config,
        environment_factory=environment_factory,
        policy_adapter_factory=resolved_policy_adapter_factory,
        training_policy=policy,
        phase="train",
        success_fn=success_fn,
        observation_recorder=observation_recorder,
        transition_recorder=transition_recorder,
    )
    evaluation_rollout = None
    if config.evaluation.enabled:
        resolved_evaluation_policy_adapter_factory = (
            _resolve_policy_adapter_factory(
                config=config,
                policy=policy,
                policy_adapter=None,
                policy_adapter_factory=evaluation_policy_adapter_factory,
                label="evaluation",
            )
            if evaluation_policy_adapter_factory is not None
            else resolved_policy_adapter_factory
        )
        evaluation_rollout = LeRobotEpisodeRollout.from_config(
            config,
            environment_factory=(evaluation_environment_factory or environment_factory),
            policy_adapter_factory=(resolved_evaluation_policy_adapter_factory),
            training_policy=policy,
            phase="eval",
            success_fn=evaluation_success_fn or success_fn,
            observation_recorder=(
                evaluation_observation_recorder or observation_recorder
            ),
            transition_recorder=(evaluation_transition_recorder or transition_recorder),
        )
    return await run_embodied_experiment(
        config=config,
        policy=policy,
        train_scenarios=train_scenarios,
        rollout=train_rollout,
        evaluation_scenarios=evaluation_scenarios,
        evaluation_rollout=evaluation_rollout,
        evaluation_success_metric=evaluation_success_metric,
        backend=backend,
        observer=observer,
    )


def _resolve_policy_adapter_factory(
    *,
    config: EmbodiedExperimentConfig,
    policy: Any,
    policy_adapter: LeRobotPolicyAdapterProtocol | None,
    policy_adapter_factory: PolicyAdapterFactory | None,
    label: str,
) -> PolicyAdapterFactory:
    if policy_adapter is not None and policy_adapter_factory is not None:
        raise ValueError(
            f"Pass either {label} policy_adapter or policy_adapter_factory, not both"
        )
    if policy_adapter is not None:
        if config.rollout.workers != 1:
            raise ValueError(
                "A shared LeRobot policy_adapter requires rollout.workers=1. "
                "For parallel rollout, provide a factory with "
                "prepare_update(policy=..., update=...) that synchronizes "
                "episode-isolated replicas."
            )
        if policy_adapter.policy is not policy:
            raise ValueError(
                f"The {label} policy_adapter must wrap the policy trained by ART"
            )
        return SharedLeRobotPolicyAdapterFactory(policy_adapter)
    if policy_adapter_factory is None:
        raise ValueError(
            f"run_lerobot_experiment requires a {label} policy_adapter or "
            "policy_adapter_factory"
        )
    prepare = getattr(policy_adapter_factory, "prepare_update", None)
    bound_policy = getattr(policy_adapter_factory, "training_policy", None)
    if not callable(prepare) and bound_policy is not policy:
        raise ValueError(
            f"The {label} policy_adapter_factory must expose "
            "prepare_update(policy=..., update=...) or training_policy bound "
            "to the policy trained by ART"
        )
    return policy_adapter_factory


async def run_embodied_experiment(
    *,
    config: EmbodiedExperimentConfig,
    policy: Any,
    train_scenarios: Sequence[EmbodiedScenario],
    rollout: RolloutFunction | None,
    evaluation_scenarios: Sequence[EmbodiedScenario] | None = None,
    evaluation_rollout: RolloutFunction | None = None,
    evaluate: EvaluationFunction | None = None,
    evaluation_success_metric: str = "success",
    backend: Any | None = None,
    observer: WandbWeaveObserver | None = None,
) -> EmbodiedRunResult:
    """Run action-token RL while preserving native LeRobot environment code.

    This is the ergonomic path. Advanced users may instantiate
    :class:`EmbodiedExperiment` directly. The helper deliberately does not
    infer model, reward, reset, or optimizer settings outside the YAML config.
    """

    if config.runtime.rollout_execution.mode == "local_process" or backend is None:
        validate_runtime_device_availability(config)
    # Evaluation-only workers intentionally run without ART. Load the ART
    # lifecycle only when an optimizer-backed experiment is actually started.
    from .art_compat import EmbodiedBackend, EmbodiedTrainableModel

    owns_backend = backend is None
    owns_observer = observer is None
    native_backend = backend or make_embodied_backend(config, policy=policy)
    resolved_backend: EmbodiedBackend | None = None
    resolved_observer: WandbWeaveObserver | None = None
    process_rollout_pool: LocalProcessRolloutPool | None = None
    backend_close_attempted = False
    run_error: BaseException | None = None

    try:
        resolved_rollout = rollout
        resolved_evaluation_rollout = evaluation_rollout
        execution = config.runtime.rollout_execution
        if execution.mode == "local_process":
            if rollout is not None or evaluation_rollout is not None:
                raise ValueError(
                    "runtime.rollout_execution.mode='local_process' owns the "
                    "rollout implementation through actor_factory; pass "
                    "rollout=None and evaluation_rollout=None"
                )
            snapshot_provider = _backend_snapshot_provider(native_backend)
            process_rollout_pool = LocalProcessRolloutPool(
                config=config,
                policy=policy if snapshot_provider is None else None,
                snapshot_provider=snapshot_provider,
            )
            resolved_rollout = process_rollout_pool.for_phase("train")
            resolved_evaluation_rollout = process_rollout_pool.for_phase("eval")
        elif resolved_rollout is None:
            raise ValueError(
                "runtime.rollout_execution.mode='in_process' requires rollout"
            )
        assert resolved_rollout is not None

        resolved_observer = observer or WandbWeaveObserver.start(config)
        model = EmbodiedTrainableModel(
            policy=policy,
            config=config,
            observer=resolved_observer,
        )
        resolved_backend = (
            native_backend
            if isinstance(native_backend, EmbodiedBackend)
            else EmbodiedBackend(native_backend, config=config)
        )
        await model.register(resolved_backend)

        if evaluate is not None and evaluation_scenarios is not None:
            raise ValueError("Pass either evaluate or evaluation_scenarios, not both")
        resolved_evaluate = evaluate
        if config.evaluation.enabled and resolved_evaluate is None:
            if evaluation_scenarios is None:
                raise ValueError(
                    "evaluation.enabled=true requires evaluate or evaluation_scenarios"
                )
            fixed_evaluator = FixedScenarioEvaluator(
                config=config,
                scenarios=evaluation_scenarios,
                rollout=resolved_evaluation_rollout or resolved_rollout,
                success_metric=evaluation_success_metric,
                log_progress=getattr(resolved_observer, "log_progress", None),
            )
            if resolved_evaluation_rollout is None:

                async def evaluate_with_generation(*args: Any) -> Any:
                    with _policy_generation(
                        policy,
                        do_sample=config.policy.evaluation_generation.do_sample,
                        temperature=config.policy.evaluation_generation.temperature,
                    ):
                        return await fixed_evaluator(*args)

                resolved_evaluate = evaluate_with_generation
            else:
                resolved_evaluate = fixed_evaluator

        if config.evaluation.evaluate_before_training:
            assert resolved_evaluate is not None
            await resolved_observer.log_progress(
                ExperimentProgress(
                    update=0,
                    phase="evaluation",
                    status="started",
                    completed=0,
                    total=config.evaluation.episodes,
                ),
                config,
            )
            initial_evaluation = await resolved_evaluate(
                0,
                LocalTrainResult(step=0, metrics={}),
                config,
            )
            await resolved_observer.log_initial_evaluation(
                initial_evaluation,
                config,
            )
            await resolved_observer.log_progress(
                ExperimentProgress(
                    update=0,
                    phase="evaluation",
                    status="completed",
                    completed=config.evaluation.episodes,
                    total=config.evaluation.episodes,
                    metrics={
                        str(key): float(value)
                        for key, value in initial_evaluation.metrics.items()
                        if isinstance(value, int | float | bool)
                    },
                ),
                config,
            )
            await _apply_pre_training_success_gate(
                config=config,
                evaluation=initial_evaluation,
                observer=resolved_observer,
            )

        async def log_model_step(
            step: int,
            groups: Any,
            train_result: Any,
            evaluation: Any,
            _config: EmbodiedExperimentConfig,
        ) -> None:
            if _config.fingerprint != model.config.fingerprint:
                raise ValueError("Model logger config does not match experiment config")
            model._last_train_result = train_result
            await model.log(
                groups,
                split="train",
                metrics=train_result.metrics,
                step=step,
                evaluation=evaluation,
            )

        experiment = EmbodiedExperiment(
            config=config,
            scenarios=train_scenarios,
            rollout=resolved_rollout,
            backend=resolved_backend,
            evaluate=resolved_evaluate,
            log_rollout=getattr(resolved_observer, "log_rollout", None),
            log_step=log_model_step,
            log_progress=getattr(resolved_observer, "log_progress", None),
        )
        try:
            steps = await experiment.run(close_backend=owns_backend)
        finally:
            backend_close_attempted = owns_backend
    except BaseException as exc:
        run_error = exc
        raise
    finally:
        if process_rollout_pool is not None:
            await process_rollout_pool.close()
        if owns_observer and resolved_observer is not None:
            resolved_observer.close(exit_code=1 if run_error is not None else 0)
        if (
            owns_backend
            and not backend_close_attempted
            and resolved_backend is not None
        ):
            await resolved_backend.close()
    return EmbodiedRunResult(
        config_fingerprint=config.fingerprint,
        steps=tuple(steps),
    )


async def _apply_pre_training_success_gate(
    *,
    config: EmbodiedExperimentConfig,
    evaluation: EvaluationResult,
    observer: Any,
) -> None:
    gate = config.evaluation.pre_training_success_gate
    if gate is None:
        return
    raw_success_rate = evaluation.metrics.get("success_rate")
    if not isinstance(raw_success_rate, int | float) or isinstance(
        raw_success_rate, bool
    ):
        message = "Step-0 evaluation did not publish a numeric success_rate"
        await observer.log_progress(
            ExperimentProgress(
                update=0,
                phase="training_gate",
                status="failed",
                message=message,
            ),
            config,
        )
        raise RuntimeError(message)
    success_rate = float(raw_success_rate)
    minimum = gate.minimum_success_rate
    maximum = gate.maximum_success_rate
    passed = minimum <= success_rate <= maximum
    message = (
        f"Step-0 success_rate={success_rate:.4f}; "
        f"required_range=[{minimum:.4f}, {maximum:.4f}]"
    )
    await observer.log_progress(
        ExperimentProgress(
            update=0,
            phase="training_gate",
            status="completed" if passed else "failed",
            message=message,
            metrics={
                "success_rate": success_rate,
                "minimum_success_rate": minimum,
                "maximum_success_rate": maximum,
            },
        ),
        config,
    )
    if not passed:
        raise RuntimeError(
            "Pre-training success gate rejected optimizer start: " + message
        )


@contextmanager
def _policy_generation(policy: Any, *, do_sample: bool, temperature: float):
    setter = getattr(policy, "set_generation", None)
    if not callable(setter):
        yield
        return
    previous_do_sample = getattr(policy, "do_sample")
    previous_temperature = getattr(policy, "temperature")
    setter(do_sample=do_sample, temperature=temperature)
    try:
        yield
    finally:
        setter(
            do_sample=previous_do_sample,
            temperature=previous_temperature,
        )
