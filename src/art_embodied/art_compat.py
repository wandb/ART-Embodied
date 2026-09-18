"""ART-compatible lifecycle wrappers for native embodied policies.

The native policy remains owned by LeRobot (or another robotics stack).  This
module only adapts its training lifecycle to the public ART model/backend shape;
it does not pretend that a robot policy is an OpenAI-compatible text model.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import pydantic

try:
    from art import TrainableModel
except ImportError as exc:  # pragma: no cover - exercised by split runtimes
    raise RuntimeError(
        "ART-Embodied requires a normally initialized OpenPipe ART package. "
        "Remove diagnostic sitecustomize shims that synthesize an empty 'art' "
        "module, or launch the embodied worker through the supported runtime "
        "boundary."
    ) from exc

from .compatibility import require_compatible_runtime, runtime_profile_for_policy
from .config import EmbodiedExperimentConfig
from .experiment import EvaluationResult
from .observability import WandbWeaveObserver
from .trajectories import EmbodiedTrajectory, EmbodiedTrajectoryGroup
from .types import LocalTrainResult, TrainResult


class EmbodiedTrainableModel(TrainableModel[EmbodiedExperimentConfig, dict[str, Any]]):
    """An ART ``TrainableModel`` whose native policy is owned by LeRobot.

    ART's text-model inference client is intentionally not initialized. Rollout
    adapters call :attr:`policy` through the policy's native processors and
    sampler, while ART still owns model identity, backend registration, step
    tracking, logging, and checkpoint lifecycle.
    """

    _policy: Any = pydantic.PrivateAttr()
    _observer: WandbWeaveObserver | None = pydantic.PrivateAttr(default=None)
    _owns_observer: bool = pydantic.PrivateAttr(default=False)
    _last_train_result: TrainResult | None = pydantic.PrivateAttr(default=None)

    def __init__(
        self,
        *,
        policy: Any,
        config: EmbodiedExperimentConfig,
        name: str | None = None,
        run_name: str | None = None,
        project: str | None = None,
        entity: str | None = None,
        observer: WandbWeaveObserver | None = None,
    ) -> None:
        require_compatible_runtime(
            profile=runtime_profile_for_policy(config.policy.type)
        )
        # ART 0.5.20 separates durable run identity from the serving name.
        # Preserve the existing embodied identity on both versions.
        model_name = run_name or name or config.experiment.run
        identity = {"name": model_name}
        if "run_name" in TrainableModel.model_fields:
            identity["run_name"] = model_name
        super().__init__(
            **identity,
            project=project or config.experiment.project,
            entity=entity or config.observability.wandb.entity,
            config=config,
            base_model=config.policy.path,
            base_path=str(config.storage.output_dir),
            report_metrics=[],
        )
        self._policy = policy
        self._observer = observer
        self._owns_observer = False

    @property
    def policy(self) -> Any:
        """Return the unmodified native policy used by LeRobot adapters."""

        return self._policy

    async def register(  # type: ignore[override]
        self,
        backend: Any,
        _openai_client_config: Any | None = None,
    ) -> None:
        """Register without creating ART's text/OpenAI inference endpoint."""

        del _openai_client_config
        object.__setattr__(self, "_backend", backend)
        await backend.register(self)

    async def log(  # type: ignore[override]
        self,
        trajectories: Iterable[EmbodiedTrajectoryGroup] | None = None,
        split: str = "val",
        *,
        metrics: dict[str, float] | None = None,
        step: int | None = None,
        evaluation: EvaluationResult | None = None,
    ) -> None:
        """Log embodied groups through the same model-owned lifecycle as ART.

        The W&B/Weave observer remains embodied-specific because it renders
        videos, observations, actions, and nested trajectory traces that ART's
        text logger does not understand.
        """

        if split != "train":
            raise ValueError(
                "EmbodiedTrainableModel.log currently accepts split='train'; "
                "use log_evaluation(...) for fixed native-policy evaluation"
            )
        resolved_step = await self.get_step() if step is None else int(step)
        groups = list(trajectories or ())
        train_result = self._last_train_result
        if train_result is None or train_result.step != resolved_step:
            train_result = LocalTrainResult(
                step=resolved_step,
                metrics=dict(metrics or {}),
            )
        elif metrics:
            train_result.metrics.update(metrics)
        await self._get_observer().log_step(
            resolved_step,
            groups,
            train_result,
            evaluation,
            self.config,
        )

    async def log_evaluation(self, evaluation: EvaluationResult) -> None:
        """Log a fixed native-policy evaluation without an optimizer update."""

        await self._get_observer().log_evaluation(
            evaluation.step,
            evaluation,
            self.config,
        )

    async def delete_checkpoints(  # type: ignore[override]
        self,
        best_checkpoint_metric: str = "eval/success_rate",
    ) -> None:
        """Delegate retention to the embodied backend's explicit policy."""

        del best_checkpoint_metric
        backend = self.backend()
        delete = getattr(backend, "delete_checkpoints", None)
        if callable(delete):
            await delete(self)

    async def close(self, *, exit_code: int = 0) -> None:
        """Close model-owned observability and the registered backend."""

        if self._owns_observer and self._observer is not None:
            self._observer.close(exit_code=exit_code)
            self._observer = None
            self._owns_observer = False
        if self._backend is not None:
            await self._backend.close()

    def _get_observer(self) -> WandbWeaveObserver:
        if self._observer is None:
            self._observer = WandbWeaveObserver.start(self.config)
            self._owns_observer = True
        return self._observer


class EmbodiedBackend:
    """Expose an existing embodied backend through ART's backend contract."""

    def __init__(
        self,
        backend: Any,
        *,
        config: EmbodiedExperimentConfig,
    ) -> None:
        self.backend = backend
        self.config = config
        self._model: EmbodiedTrainableModel | None = None
        self._last_step = _backend_step(backend)

    async def register(self, model: EmbodiedTrainableModel) -> None:
        if model.config.fingerprint != self.config.fingerprint:
            raise ValueError("Embodied model and backend configs do not match")
        backend_policy = getattr(self.backend, "policy", None)
        if backend_policy is not None and backend_policy is not model.policy:
            raise ValueError("Embodied backend and model must share one native policy")
        self._model = model

    async def _get_step(self, model: EmbodiedTrainableModel) -> int:
        self._require_model(model)
        return self._last_step

    async def train(
        self,
        model: EmbodiedTrainableModel | Sequence[EmbodiedTrajectoryGroup],
        trajectory_groups: Sequence[EmbodiedTrajectoryGroup] | None = None,
        *,
        learning_rate: float | None = None,
        **kwargs: Any,
    ) -> TrainResult:
        """Train with ART's ``train(model, groups, ...)`` call shape."""

        if trajectory_groups is None:
            if self._model is None:
                raise ValueError(
                    "Embodied backend has no registered model; call "
                    "await model.register(backend) first"
                )
            resolved_model = self._model
            resolved_groups = model
            if isinstance(resolved_groups, EmbodiedTrainableModel):
                raise TypeError("trajectory_groups are required")
        else:
            if not isinstance(model, EmbodiedTrainableModel):
                raise TypeError(
                    "ART-style training requires train(model, trajectory_groups)"
                )
            resolved_model = model
            resolved_groups = trajectory_groups
        self._require_model(resolved_model)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(
                "Embodied training options are YAML-owned; unsupported train "
                f"overrides: {unknown}"
            )
        configured_lr = self.config.training.optimizer.learning_rate
        if learning_rate is not None and learning_rate != configured_lr:
            raise ValueError(
                "learning_rate must match training.optimizer.learning_rate in "
                f"the experiment YAML ({configured_lr}); received {learning_rate}"
            )
        result = await self.backend.train(resolved_groups)
        self._last_step = int(result.step)
        resolved_model._last_train_result = result
        return result

    @property
    def update_step(self) -> int:
        """Expose the registered ART step to the existing experiment loop."""

        return self._last_step

    async def delete_checkpoints(self, model: EmbodiedTrainableModel) -> None:
        self._require_model(model)
        # Native backends enforce keep-last and protected-update retention when
        # checkpoints are written. There is no second implicit deletion pass.

    async def close(self) -> None:
        await self.backend.close()

    def _require_model(self, model: EmbodiedTrainableModel) -> None:
        if self._model is not model:
            raise ValueError(
                "Embodied model is not registered with this backend; call "
                "await model.register(backend) first"
            )


async def trajectory_group(
    trajectories: Iterable[Any],
    *,
    return_exceptions: bool = False,
    metadata: dict[str, Any] | None = None,
) -> EmbodiedTrajectoryGroup:
    """Gather embodied rollout coroutines using ART's trajectory-group idiom."""

    import asyncio

    items = list(trajectories)
    results = await asyncio.gather(*items, return_exceptions=return_exceptions)
    if not return_exceptions:
        for result in results:
            if not isinstance(result, EmbodiedTrajectory):
                raise TypeError(
                    "embodied.trajectory_group rollouts must return "
                    f"EmbodiedTrajectory, got {type(result).__name__}"
                )
    return EmbodiedTrajectoryGroup(results, metadata=metadata)


async def gather_trajectory_groups(
    groups: Iterable[Any],
    *,
    max_exceptions: int | float = 0,
) -> list[EmbodiedTrajectoryGroup]:
    """Gather embodied groups with ART-compatible exception accounting."""

    import asyncio

    completed = list(await asyncio.gather(*list(groups)))
    if not all(isinstance(group, EmbodiedTrajectoryGroup) for group in completed):
        invalid = next(
            group
            for group in completed
            if not isinstance(group, EmbodiedTrajectoryGroup)
        )
        raise TypeError(
            "gather_trajectory_groups expected EmbodiedTrajectoryGroup, got "
            f"{type(invalid).__name__}"
        )
    exception_count = sum(len(group.exceptions) for group in completed)
    trajectory_count = sum(len(group.trajectories) for group in completed)
    if isinstance(max_exceptions, float):
        if not 0.0 <= max_exceptions <= 1.0:
            raise ValueError("fractional max_exceptions must be between 0 and 1")
        limit = max_exceptions * max(1, exception_count + trajectory_count)
    else:
        limit = max_exceptions
    if exception_count > limit:
        raise RuntimeError(
            "Too many embodied rollout exceptions: "
            f"observed={exception_count}, allowed={limit}"
        )
    return completed


def _backend_step(backend: Any) -> int:
    if hasattr(backend, "update_step"):
        return int(backend.update_step)
    if hasattr(backend, "step"):
        return int(backend.step)
    return 0
