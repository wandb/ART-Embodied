"""LeRobot-first episode adapter for SmolVLA Flow-SDE rollouts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import numbers
import random
from typing import Any, Literal

from art_embodied.backends.flow_sde import TRANSIENT_FLOW_SDE_ROLLOUT_KEY
from art_embodied.trajectories import Action
from art_embodied.utils import make_json_safe

from .lerobot import LeRobotActionPrediction


@dataclass(slots=True)
class SmolVLAFlowSDEPolicyAdapter:
    """Preserve SmolVLA processors while retaining exact Flow-SDE evidence."""

    policy: Any
    robot_type: str | None = None
    seed_fn: Callable[[int], None] | None = None
    prepare_observation_fn: Callable[..., Any] | None = None
    sampling_mode: Literal["train", "eval"] = "train"

    def __post_init__(self) -> None:
        if self.policy.preprocessor is None or self.policy.postprocessor is None:
            raise RuntimeError(
                "SmolVLAFlowPolicy must load its processors before rollout"
            )

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            (self.seed_fn or _load_lerobot_set_seed())(int(seed))
        self.policy.reset()

    def predict(
        self,
        observation: Mapping[str, Any],
        *,
        task: str | None,
        step: int,
        seed: int | None = None,
    ) -> LeRobotActionPrediction:
        del seed
        prepared = self._prepare(observation, task=task)
        processed = self.policy.preprocessor(prepared)
        predictions = self._predict_processed_batch(
            processed,
            tasks=[task],
            step=step,
        )
        return predictions[0]

    def predict_batch(
        self,
        observations: Sequence[Mapping[str, Any]],
        *,
        tasks: Sequence[str | None],
        step: int,
        selected_indices: Any | None = None,
        initial_noise: Any | None = None,
    ) -> list[LeRobotActionPrediction]:
        """Sample same-reset siblings in one native SmolVLA model batch."""

        if not observations:
            return []
        if len(observations) != len(tasks):
            raise ValueError("SmolVLA observations and tasks must have equal length")
        processed_rows = [
            self.policy.preprocessor(self._prepare(observation, task=task))
            for observation, task in zip(observations, tasks, strict=True)
        ]
        processed = _concatenate_processed_batches(processed_rows)
        return self._predict_processed_batch(
            processed,
            tasks=tasks,
            step=step,
            selected_indices=selected_indices,
            initial_noise=initial_noise,
        )

    def stateful_component_ids(self) -> tuple[int, ...]:
        return tuple(
            id(component)
            for component in (
                self.policy,
                self.policy.policy,
                self.policy.preprocessor,
                self.policy.postprocessor,
            )
            if component is not None
        )

    def _prepare(
        self,
        observation: Mapping[str, Any],
        *,
        task: str | None,
    ) -> Any:
        if self.prepare_observation_fn is not None:
            import torch

            return self.prepare_observation_fn(
                dict(observation),
                torch.device(self.policy.device),
                task,
                self.robot_type,
            )
        return _prepare_smolvla_observation(
            observation,
            device=self.policy.device,
            task=task,
            robot_type=self.robot_type,
            observation_key_map=getattr(self.policy, "observation_key_map", {}),
        )

    def _predict_processed_batch(
        self,
        processed: dict[str, Any],
        *,
        tasks: Sequence[str | None],
        step: int,
        selected_indices: Any | None = None,
        initial_noise: Any | None = None,
    ) -> list[LeRobotActionPrediction]:
        import torch

        batch_size = _batch_size(processed)
        if batch_size != len(tasks):
            raise ValueError(
                f"SmolVLA processor batch {batch_size} does not match {len(tasks)} tasks"
            )
        rollout = None
        if self.sampling_mode == "eval":
            normalized_chunks = self.policy.predict_native_action_chunk(processed)
            probability_model = "native_flow_ode"
        else:
            if selected_indices is None:
                selected_index = random.randint(0, self.policy.schedule.num_steps - 1)
                selected_indices = torch.full(
                    (batch_size,),
                    selected_index,
                    dtype=torch.long,
                    device=torch.device(self.policy.device),
                )
            else:
                selected_indices = torch.as_tensor(
                    selected_indices,
                    dtype=torch.long,
                    device=torch.device(self.policy.device),
                )
                if selected_indices.shape != (batch_size,):
                    raise ValueError(
                        "SmolVLA selected_indices must have one value per sample"
                    )
            sample_kwargs: dict[str, Any] = {"selected_index": selected_indices}
            if initial_noise is not None:
                sample_kwargs["initial_noise"] = initial_noise
            rollout = self.policy.sample_flow_sde(processed, **sample_kwargs)
            normalized_chunks = rollout.actions
            probability_model = "gaussian_flow_sde"

        predicted_chunks = self.policy.postprocessor(normalized_chunks)
        if not isinstance(predicted_chunks, torch.Tensor):
            raise TypeError("SmolVLA postprocessor must return an action tensor")
        if predicted_chunks.ndim != 3 or predicted_chunks.shape[0] != batch_size:
            raise ValueError(
                "SmolVLA rollout must return [batch, horizon, action_dim], "
                f"got {tuple(predicted_chunks.shape)}"
            )
        if predicted_chunks.shape[1] < self.policy.execution_horizon:
            raise ValueError(
                "SmolVLA returned fewer actions than execution_horizon"
            )

        predictions = []
        for index, task in enumerate(tasks):
            predicted_action_chunk = predicted_chunks[index].detach().cpu()
            native_action = predicted_action_chunk[: self.policy.execution_horizon]
            metadata: dict[str, Any] = {
                "framework": "lerobot",
                "policy_type": "smolvla",
                "probability_model": probability_model,
                "execution_horizon": self.policy.execution_horizon,
                "action_dim": self.policy.action_dim,
                "task": task,
                "model_batch_size": batch_size,
            }
            logprobs = None
            if rollout is not None:
                item = rollout.select(index).cpu()
                old_logprobs = item.transition.old_logprobs[
                    :, : self.policy.execution_horizon, : self.policy.action_dim
                ]
                assert selected_indices is not None
                metadata.update(
                    {
                        "selected_denoise_index": int(
                            selected_indices[index].item()
                        ),
                        TRANSIENT_FLOW_SDE_ROLLOUT_KEY: item,
                    }
                )
                logprobs = {
                    "element_logprobs": make_json_safe(old_logprobs),
                    "chunk_logprob": float(old_logprobs.sum().detach().cpu()),
                }
            recorded = Action(
                step=step,
                kind="continuous",
                raw=make_json_safe(native_action),
                decoded=make_json_safe(native_action),
                logprobs=logprobs,
                metadata=metadata,
            )
            predictions.append(
                LeRobotActionPrediction(
                    native_action=native_action,
                    action=recorded,
                    predicted_action_chunk=predicted_action_chunk,
                    execution_horizon=self.policy.execution_horizon,
                )
            )
        return predictions


def _prepare_smolvla_observation(
    observation: Mapping[str, Any],
    *,
    device: str,
    task: str | None,
    robot_type: str | None,
    observation_key_map: Mapping[str, str],
) -> Any:
    """Prepare raw environment values using LeRobot's inference boundary."""

    import numpy as np
    import torch

    try:
        from lerobot.policies import prepare_observation_for_inference
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("SmolVLA rollout requires LeRobot 0.6") from exc
    contiguous = _map_smolvla_observation(
        observation,
        observation_key_map=observation_key_map,
    )
    return prepare_observation_for_inference(
        contiguous,
        torch.device(device),
        task,
        robot_type,
    )


def _map_smolvla_observation(
    observation: Mapping[str, Any],
    *,
    observation_key_map: Mapping[str, str],
) -> dict[str, Any]:
    """Map environment keys and normalize NumPy strides at the policy boundary."""

    import numpy as np

    return {
        observation_key_map.get(key, key): (
            np.ascontiguousarray(value) if isinstance(value, np.ndarray) else value
        )
        for key, value in observation.items()
    }


def _batch_size(processed: Mapping[str, Any]) -> int:
    for value in processed.values():
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) > 0:
            return int(shape[0])
    raise ValueError("SmolVLA preprocessor produced no batched tensors")


def _concatenate_processed_batches(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Combine independently processed singleton observations without state leak."""

    import torch

    if not rows:
        raise ValueError("SmolVLA processed batch cannot be empty")
    keys = set(rows[0])
    if any(set(row) != keys for row in rows[1:]):
        raise ValueError("SmolVLA processor produced inconsistent batch keys")
    combined: dict[str, Any] = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        if all(value is None for value in values):
            combined[key] = None
        elif all(isinstance(value, torch.Tensor) for value in values):
            if any(value.ndim == 0 or value.shape[0] != 1 for value in values):
                raise ValueError(
                    "SmolVLA processor rows require singleton leading batches; "
                    f"key={key!r}"
                )
            combined[key] = torch.cat(values, dim=0)
        elif all(isinstance(value, numbers.Number) for value in values):
            combined[key] = torch.as_tensor(values)
        elif all(value == values[0] for value in values[1:]):
            combined[key] = values[0]
        else:
            raise TypeError(
                "SmolVLA processor produced non-batchable metadata; "
                f"key={key!r}"
            )
    return combined


def _load_lerobot_set_seed() -> Callable[[int], None]:
    try:
        from lerobot.utils.random_utils import set_seed
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("SmolVLA rollout requires LeRobot 0.6") from exc
    return set_seed
