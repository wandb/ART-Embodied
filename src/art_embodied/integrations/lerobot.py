"""Thin LeRobot integration that preserves LeRobot's policy contract.

ART owns trajectory grouping and training.  LeRobot continues to own policy
inference and its pre/post-processing pipelines.  The integration is lazy so
users can inspect configs and trajectory data without installing LeRobot.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from art_embodied.trajectories import Action
from art_embodied.utils import make_json_safe


@runtime_checkable
class TrainableActionTokenPolicy(Protocol):
    """Capability required by the action-token GRPO/GSPO backend."""

    def action_token_logprobs(
        self,
        examples: Sequence[Any],
    ) -> Any: ...

    def parameters(self) -> Any: ...

    def train(self, mode: bool = True) -> Any: ...

    def eval(self) -> Any: ...


PredictAction = Callable[..., Any]
RecordAction = Callable[..., Action]
SeedFunction = Callable[[int], None]


@dataclass(frozen=True, slots=True)
class LeRobotActionPrediction:
    """Executed action, trajectory record, and optional full chunk prediction.

    ``predicted_action_chunk`` is intentionally separate from ``action``. It is
    transient observability data: the trajectory stores only the prefix that
    was actually sent to the environment. Chunk policies can expose their
    unused lookahead without changing the learning or replay contract.
    """

    native_action: Any
    action: Action
    predicted_action_chunk: Any | None = None
    execution_horizon: int | None = None


@runtime_checkable
class LeRobotPolicyAdapterProtocol(Protocol):
    """Structural contract for a policy used by the LeRobot episode loop.

    The stock adapter calls LeRobot's processors and ``predict_action``. Model
    families with a native batched sampler, such as OpenVLA-OFT, can implement
    the same narrow contract without inheriting from that concrete adapter.
    """

    policy: Any

    def reset(self, *, seed: int | None = None) -> None: ...

    def predict(
        self,
        observation: Mapping[str, Any],
        *,
        task: str | None,
        step: int,
        seed: int | None = None,
    ) -> LeRobotActionPrediction: ...

    def stateful_component_ids(self) -> tuple[int, ...]: ...


@dataclass(slots=True)
class LeRobotPolicyAdapter:
    """Run a native LeRobot policy while recording ART embodied actions.

    ``preprocessor`` and ``postprocessor`` are the same processor pipelines
    passed to LeRobot's own rollout/evaluation utilities.  This avoids a second
    observation or action conversion path inside ART.
    """

    policy: Any
    preprocessor: Any
    postprocessor: Any
    device: Any
    use_amp: bool = False
    robot_type: str | None = None
    record_action_fn: RecordAction | None = None
    seed_fn: SeedFunction | None = None
    _predict_action_fn: PredictAction | None = None

    def reset(self, *, seed: int | None = None) -> None:
        """Reset all native episode state and optionally seed policy sampling."""

        if seed is not None:
            self._seed(seed)
        for component in (self.policy, self.preprocessor, self.postprocessor):
            reset = getattr(component, "reset", None)
            if callable(reset):
                reset()

    def act(
        self,
        observation: Mapping[str, Any],
        *,
        task: str | None,
        step: int,
        seed: int | None = None,
    ) -> Action:
        """Return the serializable ART action record.

        Use :meth:`predict` in an environment loop so ``env.step`` receives the
        native tensor returned by LeRobot rather than a JSON-converted copy.
        """

        return self.predict(
            observation,
            task=task,
            step=step,
            seed=seed,
        ).action

    def predict(
        self,
        observation: Mapping[str, Any],
        *,
        task: str | None,
        step: int,
        seed: int | None = None,
    ) -> LeRobotActionPrediction:
        """Run LeRobot's native processor pipeline and record the action."""

        if seed is not None:
            self._seed(seed)
        predict_action = self._predict_action_fn or _load_lerobot_predict_action()
        native_action = predict_action(
            observation=dict(observation),
            policy=self.policy,
            device=self.device,
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
            use_amp=self.use_amp,
            task=task,
            robot_type=self.robot_type,
        )
        if self.record_action_fn is not None:
            recorded = self.record_action_fn(
                native_action=native_action,
                observation=observation,
                task=task,
                step=step,
                policy=self.policy,
            )
        else:
            decoded = make_json_safe(native_action)
            recorded = Action(
                step=step,
                kind="continuous",
                raw=decoded,
                decoded=decoded,
                metadata={
                    "framework": "lerobot",
                    "policy_type": _policy_type(self.policy),
                    "task": task,
                    "robot_type": self.robot_type,
                },
            )
        if seed is not None:
            recorded.metadata.setdefault("policy_seed", seed)
        return LeRobotActionPrediction(
            native_action=native_action,
            action=recorded,
        )

    def stateful_component_ids(self) -> tuple[int, ...]:
        """Identity keys used to reject concurrent queue/state sharing."""

        return tuple(
            id(component)
            for component in (self.policy, self.preprocessor, self.postprocessor)
            if component is not None
        )

    def _seed(self, seed: int) -> None:
        seed_fn = self.seed_fn or _load_lerobot_set_seed()
        seed_fn(int(seed))


@dataclass(frozen=True, slots=True)
class SharedLeRobotPolicyAdapterFactory:
    """Serial adapter provider bound to the policy trained by ART.

    This is the low-friction in-process path. The runner restricts it to one
    rollout worker because LeRobot policies and processors may contain episode
    queues and other mutable state.
    """

    adapter: LeRobotPolicyAdapterProtocol

    @property
    def training_policy(self) -> Any:
        return self.adapter.policy

    def prepare_update(self, *, policy: Any, update: int) -> None:
        del update
        if policy is not self.adapter.policy:
            raise RuntimeError(
                "The shared LeRobot adapter is not bound to the policy being trained"
            )

    def __call__(self, _scenario: Any, _context: Any) -> LeRobotPolicyAdapterProtocol:
        return self.adapter


def _load_lerobot_predict_action() -> PredictAction:
    try:
        from lerobot.utils.control_utils import predict_action
    except ImportError as exc:  # pragma: no cover - exercised without optional extra.
        raise RuntimeError(
            "LeRobotPolicyAdapter requires LeRobot. Install the ART-Embodied "
            "LeRobot extra in the policy runtime."
        ) from exc
    return predict_action


def _load_lerobot_set_seed() -> SeedFunction:
    try:
        from lerobot.utils.random_utils import set_seed
    except ImportError as exc:  # pragma: no cover - exercised without optional extra.
        raise RuntimeError(
            "LeRobotPolicyAdapter requires LeRobot. Install the ART-Embodied "
            "LeRobot extra in the policy runtime."
        ) from exc
    return set_seed


def _policy_type(policy: Any) -> str:
    config = getattr(policy, "config", None)
    for value in (
        getattr(config, "type", None),
        getattr(config, "name", None),
        getattr(policy, "name", None),
    ):
        if isinstance(value, str) and value:
            return value
    return type(policy).__name__
