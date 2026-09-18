"""LeRobot-first episode adapter for PI0/PI0.5 Flow-SDE rollouts."""

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
class PIFlowSDEPolicyAdapter:
    """Run the native LeRobot processor pipeline and retain Flow-SDE evidence.

    State, language, action normalization, and native LeRobot image handling
    remain in the checkpoint processor pipeline. Imported OpenPI checkpoints
    additionally receive OpenPI's compatibility resize before that pipeline;
    without it LeRobot's fallback PyTorch resize changes the visual policy.
    """

    policy: Any
    robot_type: str | None = None
    seed_fn: Callable[[int], None] | None = None
    prepare_observation_fn: Callable[..., Any] | None = None
    sampling_mode: Literal["train", "eval"] = "train"

    def __post_init__(self) -> None:
        if self.policy.preprocessor is None or self.policy.postprocessor is None:
            raise RuntimeError(
                "PIFlowPolicy must load its LeRobot processor pipelines before rollout"
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
        # reset() seeds one episode-local stream. Re-seeding every policy step
        # would repeat noise and diverge from RLinf's rollout probability model.
        del seed
        import torch

        if self.prepare_observation_fn is not None:
            prepared = self.prepare_observation_fn(
                dict(observation),
                torch.device(self.policy.device),
                task,
                self.robot_type,
            )
        else:
            prepared = _prepare_pi_observation(
                observation,
                policy=self.policy,
                task=task,
                robot_type=self.robot_type,
            )
        processed = self.policy.preprocessor(prepared)
        if self.sampling_mode == "eval":
            normalized_chunk = self.policy.predict_native_action_chunk(processed)
            native_chunk = _postprocess_pi_actions(
                self.policy,
                normalized_chunk,
                prepared_rows=[prepared],
            )
            if not isinstance(native_chunk, torch.Tensor):
                raise TypeError("LeRobot PI postprocessor must return an action tensor")
            native_action = _single_native_action(
                native_chunk,
                execution_horizon=self.policy.execution_horizon,
            )
            recorded = Action(
                step=step,
                kind="continuous",
                raw=make_json_safe(native_action),
                decoded=make_json_safe(native_action),
                logprobs=None,
                metadata={
                    "framework": "lerobot",
                    "policy_type": self.policy.family,
                    "probability_model": "native_flow_ode",
                    "execution_horizon": self.policy.execution_horizon,
                    "action_dim": self.policy.action_dim,
                    "task": task,
                },
            )
            return LeRobotActionPrediction(
                native_action=native_action,
                action=recorded,
                predicted_action_chunk=native_chunk[0].detach().cpu(),
                execution_horizon=self.policy.execution_horizon,
            )
        batch_size = _batch_size(processed)
        # RLinf uses Python's episode-seeded RNG for the non-joint denoise
        # index and reserves Torch's RNG stream for initial/action noise.
        selected_index = random.randint(0, self.policy.schedule.num_steps - 1)
        selected_indices = torch.full(
            (batch_size,),
            selected_index,
            dtype=torch.long,
            device=torch.device(self.policy.device),
        )
        rollout = self.policy.sample_flow_sde(
            processed,
            selected_index=selected_indices,
        )
        native_chunk = _postprocess_pi_actions(
            self.policy,
            rollout.actions,
            prepared_rows=[prepared],
        )
        if not isinstance(native_chunk, torch.Tensor):
            raise TypeError("LeRobot PI postprocessor must return an action tensor")
        native_action = _single_native_action(
            native_chunk,
            execution_horizon=self.policy.execution_horizon,
        )
        old_logprobs = rollout.transition.old_logprobs[
            :, : self.policy.execution_horizon, : self.policy.action_dim
        ]
        recorded = Action(
            step=step,
            kind="continuous",
            raw=make_json_safe(native_action),
            decoded=make_json_safe(native_action),
            logprobs={
                "element_logprobs": make_json_safe(old_logprobs),
                "chunk_logprob": float(old_logprobs.sum().detach().cpu()),
            },
            metadata={
                "framework": "lerobot",
                "policy_type": self.policy.family,
                "probability_model": "gaussian_flow_sde",
                "selected_denoise_index": int(selected_indices[0].item()),
                "execution_horizon": self.policy.execution_horizon,
                "action_dim": self.policy.action_dim,
                "task": task,
                # Rollouts can outlive the actor call for hundreds of policy
                # decisions. Keep replay evidence off GPU until its training
                # microbatch is rescored.
                TRANSIENT_FLOW_SDE_ROLLOUT_KEY: rollout.cpu(),
            },
        )
        return LeRobotActionPrediction(
            native_action=native_action,
            action=recorded,
            predicted_action_chunk=native_chunk[0].detach().cpu(),
            execution_horizon=self.policy.execution_horizon,
        )

    def predict_batch(
        self,
        observations: Sequence[Mapping[str, Any]],
        *,
        tasks: Sequence[str | None],
        step: int,
        selected_indices: Any | None = None,
        initial_noise: Any | None = None,
    ) -> list[LeRobotActionPrediction]:
        """Predict one action chunk per sibling with one PI model forward.

        Counterfactual siblings share a reset state and retain independent
        initial and transition noise.  RLinf's non-joint Flow-SDE sampler also
        shares the selected denoising step across the model batch.  Sharing the
        step keeps sibling likelihoods on the same denoising transition while
        the Gaussian draws still provide independent action samples.
        """

        if not observations:
            return []
        if len(observations) != len(tasks):
            raise ValueError("PI batch observations and tasks must have equal length")
        import torch

        prepared_rows = []
        processed_rows = []
        for observation, task in zip(observations, tasks, strict=True):
            if self.prepare_observation_fn is not None:
                prepared = self.prepare_observation_fn(
                    dict(observation),
                    torch.device(self.policy.device),
                    task,
                    self.robot_type,
                )
            else:
                prepared = _prepare_pi_observation(
                    observation,
                    policy=self.policy,
                    task=task,
                    robot_type=self.robot_type,
                )
            prepared_rows.append(prepared)
            processed_rows.append(self.policy.preprocessor(prepared))
        processed = _concatenate_processed_batches(processed_rows)

        if self.sampling_mode == "eval":
            normalized_chunks = self.policy.predict_native_action_chunk(processed)
            native_chunks = _postprocess_pi_actions(
                self.policy,
                normalized_chunks,
                prepared_rows=prepared_rows,
            )
            probability_model = "native_flow_ode"
            rollout = None
            selected_indices = None
        else:
            if selected_indices is None:
                selected_index = random.randint(
                    0,
                    self.policy.schedule.num_steps - 1,
                )
                selected_indices = torch.tensor(
                    [selected_index] * len(observations),
                    dtype=torch.long,
                    device=torch.device(self.policy.device),
                )
            else:
                selected_indices = torch.as_tensor(
                    selected_indices,
                    dtype=torch.long,
                    device=torch.device(self.policy.device),
                )
                if selected_indices.shape != (len(observations),):
                    raise ValueError(
                        "PI selected_indices must have one value per observation"
                    )
            sample_kwargs: dict[str, Any] = {"selected_index": selected_indices}
            if initial_noise is not None:
                sample_kwargs["initial_noise"] = initial_noise
            rollout = self.policy.sample_flow_sde(processed, **sample_kwargs)
            native_chunks = _postprocess_pi_actions(
                self.policy,
                rollout.actions,
                prepared_rows=prepared_rows,
            )
            probability_model = "gaussian_flow_sde"
        if not isinstance(native_chunks, torch.Tensor):
            raise TypeError("LeRobot PI postprocessor must return an action tensor")
        if native_chunks.ndim != 3 or native_chunks.shape[0] != len(observations):
            raise ValueError(
                "Batched PI rollout must return [batch, horizon, action_dim], "
                f"got {tuple(native_chunks.shape)}"
            )
        if native_chunks.shape[1] < self.policy.execution_horizon:
            raise ValueError(
                "PI postprocessor returned fewer actions than execution_horizon: "
                f"horizon={native_chunks.shape[1]}, "
                f"execution_horizon={self.policy.execution_horizon}"
            )
        predictions = []
        for index, task in enumerate(tasks):
            predicted_action_chunk = native_chunks[index].detach().cpu()
            native_action = predicted_action_chunk[: self.policy.execution_horizon]
            metadata: dict[str, Any] = {
                "framework": "lerobot",
                "policy_type": self.policy.family,
                "probability_model": probability_model,
                "execution_horizon": self.policy.execution_horizon,
                "action_dim": self.policy.action_dim,
                "task": task,
                "model_batch_size": len(observations),
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
                        "selected_denoise_index": int(selected_indices[index].item()),
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


def _batch_size(processed: Mapping[str, Any]) -> int:
    for value in processed.values():
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) > 0:
            return int(shape[0])
    raise ValueError("PI preprocessor produced no batched tensors")


def _concatenate_processed_batches(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Concatenate processor outputs that each carry a singleton batch."""

    import torch

    if not rows:
        raise ValueError("PI processed batch cannot be empty")
    keys = set(rows[0])
    if any(set(row) != keys for row in rows[1:]):
        raise ValueError("PI processor produced inconsistent keys across batch rows")
    combined: dict[str, Any] = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        if all(value is None for value in values):
            combined[key] = None
            continue
        if all(isinstance(value, torch.Tensor) for value in values):
            if any(value.ndim == 0 or value.shape[0] != 1 for value in values):
                raise ValueError(
                    "PI processor rows must carry a singleton leading batch; "
                    f"key={key!r}, shapes={[tuple(value.shape) for value in values]}"
                )
            combined[key] = torch.cat(values, dim=0)
            continue
        if all(isinstance(value, numbers.Number) for value in values):
            # LeRobot processors preserve scalar transition metadata such as
            # ``next.reward``. Match PyTorch's default-collate semantics instead
            # of rejecting metadata that the PI model itself simply ignores.
            combined[key] = torch.as_tensor(values)
            continue
        if all(value == values[0] for value in values[1:]):
            # Uniform non-tensor metadata (for example a robot identifier) is
            # a batch-level invariant, not a model batch axis.
            combined[key] = values[0]
            continue
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError(
                "PI processor batch values must be tensors, numbers, uniformly "
                "None, or equal metadata; "
                f"key={key!r}, types={[type(value).__name__ for value in values]}"
            )
    return combined


def _postprocess_pi_actions(
    policy: Any,
    actions: Any,
    *,
    prepared_rows: Sequence[Mapping[str, Any]],
) -> Any:
    """Postprocess actions against the matching per-row reference states.

    LeRobot's relative-action processor caches only the most recently
    preprocessed state. ART preprocesses dynamic-batch rows independently, so
    calling the postprocessor directly would add the final row's state to
    every action chunk. Restore the full batch immediately before the paired
    absolute-action step instead.
    """

    reference_states = _stack_pi_reference_states(prepared_rows)
    _set_pi_relative_action_state(policy, reference_states)
    return policy.postprocessor(actions)


def _stack_pi_reference_states(
    prepared_rows: Sequence[Mapping[str, Any]],
) -> Any | None:
    import torch

    states = []
    for index, row in enumerate(prepared_rows):
        value = row.get("observation.state")
        if value is None:
            return None
        state = torch.as_tensor(value)
        if state.ndim == 2 and state.shape[0] == 1:
            state = state[0]
        if state.ndim != 1:
            raise ValueError(
                "PI prepared reference state must be a vector or singleton "
                f"batch; row={index}, shape={tuple(state.shape)}"
            )
        states.append(state)
    return torch.stack(states, dim=0) if states else None


def _set_pi_relative_action_state(policy: Any, reference_states: Any | None) -> None:
    if reference_states is None:
        return
    config = getattr(policy, "config", None)
    if not bool(getattr(config, "use_relative_actions", False)):
        return
    postprocessor = getattr(policy, "postprocessor", None)
    steps = getattr(postprocessor, "steps", ())
    paired_steps = [
        step
        for step in steps
        if bool(getattr(step, "enabled", False))
        and getattr(step, "relative_step", None) is not None
    ]
    if len(paired_steps) != 1:
        raise RuntimeError(
            "PI relative-action postprocessor must expose exactly one enabled "
            f"paired step; found {len(paired_steps)}"
        )
    # LeRobot has no public setter for this processor state. The shared
    # RelativeActionsProcessorStep owns the cache by contract.
    paired_steps[0].relative_step._last_state = reference_states


def _single_native_action(native_chunk: Any, *, execution_horizon: int):
    import torch

    if not isinstance(native_chunk, torch.Tensor):
        raise TypeError("LeRobot PI postprocessor must return an action tensor")
    if native_chunk.ndim != 3 or native_chunk.shape[0] != 1:
        raise ValueError(
            "Single-environment PI rollout must return [1, horizon, action_dim], "
            f"got {tuple(native_chunk.shape)}"
        )
    if native_chunk.shape[1] < execution_horizon:
        raise ValueError(
            "PI postprocessor returned fewer actions than execution_horizon: "
            f"horizon={native_chunk.shape[1]}, execution_horizon={execution_horizon}"
        )
    return native_chunk[0, :execution_horizon].detach().cpu()


def _load_prepare_observation() -> Callable[..., Any]:
    try:
        from lerobot.policies import prepare_observation_for_inference
    except ImportError as exc:  # pragma: no cover - optional dependency boundary.
        raise RuntimeError(
            "PI Flow-SDE rollout requires the supported LeRobot runtime"
        ) from exc
    return prepare_observation_for_inference


def _prepare_pi_observation(
    observation: Mapping[str, Any],
    *,
    policy: Any,
    task: str | None,
    robot_type: str | None,
) -> Any:
    """Map environment keys into LeRobot's declared PI feature contract."""

    import numpy as np
    import torch

    key_map = getattr(policy, "observation_key_map", {})
    # LIBERO's camera contract may expose rotated NumPy views with negative
    # strides. LeRobot converts observations with torch.from_numpy(), which
    # requires non-negative strides, so normalize only the adapter boundary.
    mapped = {
        key_map.get(key, key): (
            np.ascontiguousarray(value) if isinstance(value, np.ndarray) else value
        )
        for key, value in observation.items()
    }
    if getattr(policy, "model_format", None) == "rlinf_openpi_safetensors":
        mapped = _resize_openpi_checkpoint_images(mapped, policy=policy)
    state_key = "observation.state"
    if state_key in mapped:
        state = np.asarray(mapped[state_key])
        expected = int(policy.config.max_state_dim)
        if state.ndim != 1 or state.shape[0] > expected:
            raise ValueError(
                f"PI state must be a vector of at most {expected} values; "
                f"got {state.shape}"
            )
        if state.shape[0] < expected:
            state = np.pad(state, (0, expected - state.shape[0]))
        mapped[state_key] = np.ascontiguousarray(state.astype(np.float32, copy=False))
    prepare = _load_prepare_observation()
    return prepare(
        mapped,
        torch.device(policy.device),
        task,
        robot_type,
    )


def _resize_openpi_checkpoint_images(
    observation: Mapping[str, Any],
    *,
    policy: Any,
) -> dict[str, Any]:
    """Restore OpenPI's image contract before LeRobot's model boundary.

    LeRobot 0.6 converts images to float tensors and resizes with PyTorch
    bilinear interpolation inside the model. Imported RLinf/OpenPI checkpoints
    were trained with OpenPI's JAX uint8 resize. Pre-resizing the declared
    visual features makes LeRobot skip its incompatible resize while retaining
    the native LeRobot processor pipeline for all other features.
    """

    import numpy as np

    from art_embodied.policies.openpi_images import resize_openpi_uint8_image

    resolution = tuple(int(value) for value in policy.config.image_resolution)
    if len(resolution) != 2:
        raise ValueError(
            "PI image_resolution must contain height and width; "
            f"got {policy.config.image_resolution!r}"
        )
    image_features = getattr(policy.config, "image_features", {})
    result = dict(observation)
    for key in image_features:
        value = result.get(key)
        if value is None:
            continue
        if not isinstance(value, np.ndarray):
            # prepare_observation_for_inference accepts tensors too, but the
            # environment boundary is expected to provide raw uint8 arrays.
            raise TypeError(
                "OpenPI-compatible image preprocessing requires NumPy arrays; "
                f"key={key!r}, type={type(value).__name__}"
            )
        result[key] = resize_openpi_uint8_image(
            value,
            height=resolution[0],
            width=resolution[1],
        )
    return result


def _load_lerobot_set_seed() -> Callable[[int], None]:
    try:
        from lerobot.utils.random_utils import set_seed
    except ImportError as exc:  # pragma: no cover - optional dependency boundary.
        raise RuntimeError(
            "PI Flow-SDE rollout requires the supported LeRobot runtime"
        ) from exc
    return set_seed
