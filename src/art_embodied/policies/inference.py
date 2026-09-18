"""Model-native inference engines for process-isolated rollout actors."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

from ..config import EmbodiedExperimentConfig
from ..rollout_process import RolloutActorProcessContext
from ..trajectories import Action, Observation
from .factory import make_policy
from .openvla import OpenVLAPolicy, refresh_openvla_peft_adapter

PolicyFactory = Callable[[EmbodiedExperimentConfig], Any]


@dataclass(slots=True)
class _RngState:
    python: object
    numpy: tuple[Any, ...]
    torch_cpu: Any
    torch_cuda: list[Any] | None


class OpenVLABatchedInferenceEngine:
    """Serve native OpenVLA-OFT ``act_batch`` calls on one rollout device."""

    def __init__(
        self,
        *,
        config: EmbodiedExperimentConfig,
        context: RolloutActorProcessContext,
        policy_factory: PolicyFactory = make_policy,
    ) -> None:
        if config.policy.type != "openvla_oft":
            raise ValueError(
                "OpenVLABatchedInferenceEngine requires policy.type='openvla_oft'"
            )
        self.base_config = config
        self.context = context
        self.policy_factory = policy_factory
        self.policy: OpenVLAPolicy | None = None
        self.policy_update: int | None = None
        self.adapter_refreshes = 0
        self._rng_streams: dict[str, _RngState] = {}

    def prepare_update(self, *, update: int, policy_snapshot: Path) -> None:
        if self.policy is None:
            self.policy = self.policy_factory(
                _rollout_policy_config(
                    self.base_config,
                    local_device=self.context.local_device,
                    policy_snapshot=policy_snapshot,
                )
            )
        else:
            refresh_openvla_peft_adapter(self.policy, policy_snapshot)
            self.adapter_refreshes += 1
        self.policy_update = int(update)
        self._rng_streams.clear()

    def offload(self) -> dict[str, Any]:
        """Release rollout VRAM while retaining the loaded base model on CPU."""

        if self.policy is None:
            return {"offloaded": False, "policy_loaded": False}
        model = getattr(self.policy, "model", None)
        if model is not None and callable(getattr(model, "to", None)):
            model.to("cpu")
        import torch

        torch.cuda.empty_cache()
        self._rng_streams.clear()
        return {"offloaded": True, "policy_loaded": model is not None}

    def predict_batch(self, requests: list[Any]) -> list[Any]:
        if self.policy is None or self.policy_update is None:
            raise RuntimeError("OpenVLA inference was used before prepare_update")
        if all(_is_single_action_request(request) for request in requests):
            return self._predict_single_action_batch(requests)
        return [
            self._predict_group_request(request, len(requests)) for request in requests
        ]

    def _predict_single_action_batch(
        self,
        requests: list[Any],
    ) -> list[dict[str, Any]]:
        assert self.policy is not None
        assert self.policy_update is not None
        observations: list[Observation] = []
        contexts: list[dict[str, Any]] = []
        for request in requests:
            if not isinstance(request, dict):
                raise TypeError("OpenVLA inference requests must be mappings")
            observations.append(Observation.model_validate(request["observation"]))
            context = dict(request.get("context", {}))
            context.setdefault("policy_update", self.policy_update)
            contexts.append(context)
        actions = self.policy.act_batch(observations, contexts)
        return [
            {
                "native_action": action.decoded,
                "action": _cpu_action(action),
                "policy_update": self.policy_update,
                "batch_size": len(actions),
            }
            for action in actions
        ]

    def _predict_group_request(
        self,
        request: Any,
        server_batch_size: int,
    ) -> dict[str, Any]:
        assert self.policy is not None
        assert self.policy_update is not None
        if not isinstance(request, dict):
            raise TypeError("OpenVLA inference requests must be mappings")
        operation = request.get("op", "predict_group")
        stream_id = request.get("rng_stream_id")
        if operation == "release_rng_stream":
            if not isinstance(stream_id, str) or not stream_id:
                raise ValueError("release_rng_stream requires rng_stream_id")
            released = self._rng_streams.pop(stream_id, None) is not None
            return {"released": released, "policy_update": self.policy_update}
        if operation != "predict_group":
            raise ValueError(f"Unsupported OpenVLA inference operation: {operation!r}")

        raw_observations = request.get("observations")
        raw_contexts = request.get("contexts")
        if not isinstance(raw_observations, list) or not raw_observations:
            raise ValueError("predict_group requires a non-empty observations list")
        if not isinstance(raw_contexts, list) or len(raw_contexts) != len(
            raw_observations
        ):
            raise ValueError("predict_group requires one context for each observation")
        phase = str(request.get("phase", "train"))
        generation = _generation_for_phase(self.base_config, phase)
        self.policy.set_generation(
            do_sample=generation.do_sample,
            temperature=generation.temperature,
        )
        observations = [Observation.model_validate(item) for item in raw_observations]
        contexts = [dict(item) for item in raw_contexts]
        for context in contexts:
            context.setdefault("policy_update", self.policy_update)

        if generation.do_sample:
            if not isinstance(stream_id, str) or not stream_id:
                raise ValueError(
                    "stochastic predict_group requests require rng_stream_id"
                )
            seed = int(request["rng_seed"])
            reset = bool(request.get("reset_rng_stream", False))
            with self._use_rng_stream(stream_id, seed=seed, reset=reset):
                actions = self.policy.act_batch(observations, contexts)
        else:
            actions = self.policy.act_batch(observations, contexts)
        return {
            "actions": [_cpu_action(action) for action in actions],
            "native_actions": [_to_cpu_tree(action.decoded) for action in actions],
            "policy_update": self.policy_update,
            "model_batch_size": len(actions),
            "server_request_batch_size": server_batch_size,
        }

    @contextmanager
    def _use_rng_stream(self, stream_id: str, *, seed: int, reset: bool):
        ambient = _capture_rng_state()
        try:
            if reset or stream_id not in self._rng_streams:
                _seed_rng(seed)
            else:
                _restore_rng_state(self._rng_streams[stream_id])
            yield
            self._rng_streams[stream_id] = _capture_rng_state()
        finally:
            _restore_rng_state(ambient)


def create_openvla_batched_inference_engine(
    *,
    config: EmbodiedExperimentConfig,
    context: RolloutActorProcessContext,
) -> OpenVLABatchedInferenceEngine:
    """Importable YAML factory for the built-in OpenVLA-OFT batch engine."""

    return OpenVLABatchedInferenceEngine(config=config, context=context)


def _rollout_policy_config(
    config: EmbodiedExperimentConfig,
    *,
    local_device: str,
    policy_snapshot: Path,
) -> EmbodiedExperimentConfig:
    raw = config.model_dump(mode="python")
    policy = dict(raw["policy"])
    load_kwargs = dict(policy["load_kwargs"])
    load_kwargs["peft_adapter_path"] = str(policy_snapshot)
    policy["load_kwargs"] = load_kwargs
    policy["device"] = local_device
    raw["policy"] = policy
    return EmbodiedExperimentConfig.model_validate(raw)


def _cpu_action(action: Action) -> Action:
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


def _is_single_action_request(request: Any) -> bool:
    return isinstance(request, dict) and "observation" in request


def _generation_for_phase(config: EmbodiedExperimentConfig, phase: str) -> Any:
    if phase == "train":
        return config.policy.rollout_generation
    if phase == "eval":
        return config.policy.evaluation_generation
    raise ValueError(f"Unsupported rollout phase: {phase!r}")


def _seed_rng(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _capture_rng_state() -> _RngState:
    import numpy as np
    import torch

    return _RngState(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.random.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    )


def _restore_rng_state(state: _RngState) -> None:
    import numpy as np
    import torch

    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.random.set_rng_state(state.torch_cpu)
    if state.torch_cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state.torch_cuda)
