"""Shared-model inference engine for LeRobot PI0/PI0.5 rollouts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..config import EmbodiedExperimentConfig
from ..integrations.pi_flow_sde import PIFlowSDEPolicyAdapter
from ..rollout_process import RolloutActorProcessContext
from .factory import make_policy
from .pi import PIFlowPolicy
from .rng_streams import PolicyRngStreams

PolicyFactory = Callable[[EmbodiedExperimentConfig], PIFlowPolicy]


class PIBatchedInferenceEngine:
    """Share one PI model across multiple environment actor processes.

    A request remains one complete counterfactual group. Multiple actor
    processes overlap LIBERO stepping with model inference without changing
    the B=group-size sampler contract used by training rescoring.
    """

    def __init__(
        self,
        *,
        config: EmbodiedExperimentConfig,
        context: RolloutActorProcessContext,
        policy_factory: PolicyFactory = make_policy,
    ) -> None:
        if config.policy.type not in {"pi0", "pi05"}:
            raise ValueError("PIBatchedInferenceEngine requires PI0 or PI0.5")
        self.base_config = config
        self.context = context
        self.policy_factory = policy_factory
        self.policy: PIFlowPolicy | None = None
        self.policy_update: int | None = None
        self.adapter_refreshes = 0
        self._rng_streams = PolicyRngStreams()

    def prepare_update(self, *, update: int, policy_snapshot: Path) -> None:
        if self.policy is None:
            local_config = self.base_config.model_copy(
                update={
                    "policy": self.base_config.policy.model_copy(
                        update={"device": self.context.local_device}
                    )
                }
            )
            self.policy = self.policy_factory(local_config)
        else:
            self.policy.to(self.context.local_device)
            self.adapter_refreshes += 1
        self.policy.load_checkpoint({"path": str(policy_snapshot)})
        self.policy.eval()
        self.policy_update = int(update)
        self._rng_streams.clear()

    def offload(self) -> dict[str, Any]:
        if self.policy is None:
            return {"offloaded": False, "policy_loaded": False}
        self.policy.to("cpu")
        self._rng_streams.clear()
        return {"offloaded": True, "policy_loaded": True}

    def predict_batch(self, requests: list[Any]) -> list[dict[str, Any]]:
        if self.policy is None or self.policy_update is None:
            raise RuntimeError("PI inference was used before prepare_update")
        return [self._predict_group(request, len(requests)) for request in requests]

    def _predict_group(
        self,
        request: Any,
        server_request_batch_size: int,
    ) -> dict[str, Any]:
        assert self.policy is not None
        assert self.policy_update is not None
        if not isinstance(request, dict):
            raise TypeError("PI inference requests must be mappings")
        operation = request.get("op", "predict_group")
        stream_id = request.get("rng_stream_id")
        if operation == "release_rng_stream":
            if not isinstance(stream_id, str) or not stream_id:
                raise ValueError("release_rng_stream requires rng_stream_id")
            return {
                "released": self._rng_streams.release(stream_id),
                "policy_update": self.policy_update,
            }
        if operation != "predict_group":
            raise ValueError(f"Unsupported PI inference operation: {operation!r}")

        observations = request.get("raw_observations")
        contexts = request.get("contexts")
        if not isinstance(observations, list) or not observations:
            raise ValueError("PI predict_group requires raw_observations")
        if not isinstance(contexts, list) or len(contexts) != len(observations):
            raise ValueError("PI predict_group requires one context per observation")
        phase = str(request.get("phase", "train"))
        if phase not in {"train", "eval"}:
            raise ValueError(f"Unsupported PI inference phase: {phase!r}")
        tasks = [dict(context).get("task") for context in contexts]
        step = int(dict(contexts[0]).get("step", 0))
        if any(int(dict(context).get("step", step)) != step for context in contexts):
            raise ValueError("PI group request contains inconsistent policy steps")
        adapter = PIFlowSDEPolicyAdapter(
            policy=self.policy,
            robot_type="panda",
            sampling_mode=phase,
        )
        if not isinstance(stream_id, str) or not stream_id:
            raise ValueError("PI inference requests require rng_stream_id")
        reset_stream = bool(request.get("reset_rng_stream", False))
        # Native Flow-ODE evaluation is deterministic only after its initial
        # Gaussian noise is fixed. Isolate both train and eval RNG state so an
        # actor's result cannot depend on request ordering at the shared server.
        with self._rng_streams.use(
            stream_id,
            seed=int(request["rng_seed"]),
            reset=reset_stream,
        ):
            if reset_stream:
                # Match the embedded adapter: seed first, then reset any
                # episode-local policy or processor state.
                self.policy.reset()
            predictions = adapter.predict_batch(
                observations,
                tasks=tasks,
                step=step,
            )
        return {
            "actions": [_cpu_action(item.action) for item in predictions],
            "native_actions": [
                _to_cpu_tree(item.native_action) for item in predictions
            ],
            "predicted_action_chunks": [
                _to_cpu_tree(getattr(item, "predicted_action_chunk", None))
                for item in predictions
            ],
            "prediction_execution_horizons": [
                getattr(item, "execution_horizon", None) for item in predictions
            ],
            "policy_update": self.policy_update,
            "model_batch_size": len(predictions),
            "server_request_batch_size": server_request_batch_size,
        }


def create_pi_batched_inference_engine(
    *,
    config: EmbodiedExperimentConfig,
    context: RolloutActorProcessContext,
) -> PIBatchedInferenceEngine:
    """Importable YAML factory for the built-in shared PI engine."""

    return PIBatchedInferenceEngine(config=config, context=context)


def _cpu_action(action: Any) -> Any:
    return action.model_copy(
        update={
            "raw": _to_cpu_tree(action.raw),
            "decoded": _to_cpu_tree(action.decoded),
            "logprobs": _to_cpu_tree(action.logprobs),
            "metadata": _to_cpu_tree(action.metadata),
        }
    )


def _to_cpu_tree(value: Any) -> Any:
    detach = getattr(value, "detach", None)
    cpu = getattr(value, "cpu", None)
    if callable(detach) and callable(cpu):
        return detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_tree(item) for item in value)
    return value
