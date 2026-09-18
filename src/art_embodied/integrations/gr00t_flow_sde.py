"""LIBERO rollout adapter for NVIDIA GR00T N1.5 Flow-SDE GRPO."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import random
from typing import Any, Literal

import numpy as np

from art_embodied.backends.flow_sde import (
    FLOW_SDE_ACTION_DIMENSION_MASK_KEY,
    TRANSIENT_FLOW_SDE_ROLLOUT_KEY,
)
from art_embodied.integrations.lerobot import LeRobotActionPrediction
from art_embodied.trajectories import Action
from art_embodied.utils import make_json_safe


@dataclass(slots=True)
class GR00TN15FlowSDEPolicyAdapter:
    """Preserve the official N1.5 transforms around ART's Flow-SDE sampler."""

    policy: Any
    sampling_mode: Literal["train", "eval"] = "train"

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            _seed_policy(seed)

    def predict(
        self,
        observation: Mapping[str, Any],
        *,
        task: str | None,
        step: int,
        seed: int | None = None,
    ) -> LeRobotActionPrediction:
        """Predict one chunk through the same path used by batched rollout."""

        if seed is not None:
            _seed_policy(seed)
        return self.predict_batch(
            [observation],
            tasks=[task],
            step=step,
        )[0]

    def predict_batch(
        self,
        observations: Sequence[Mapping[str, Any]],
        *,
        tasks: Sequence[str | None],
        step: int,
        selected_indices: Any | None = None,
        initial_noise: Any | None = None,
    ) -> list[LeRobotActionPrediction]:
        if not observations:
            return []
        if len(observations) != len(tasks):
            raise ValueError("GR00T observations and tasks must have equal length")
        import torch

        native_input = _libero_observation_batch(observations, tasks)
        rollout = None
        if self.sampling_mode == "eval":
            action_components = self.policy.predict_native_action_chunk(native_input)
            probability_model = "native_flow_ode"
        else:
            normalized = self.policy.apply_observation_transforms(native_input)
            flow_inputs = self.policy.prepare_flow_inputs(normalized)
            if selected_indices is None:
                selected_index = random.randint(0, self.policy.schedule.num_steps - 1)
                selected_indices = torch.full(
                    (len(observations),),
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
            if selected_indices.shape != (len(observations),):
                raise ValueError("GR00T selected_indices must have one row per sample")
            sample_kwargs: dict[str, Any] = {"selected_index": selected_indices}
            if initial_noise is not None:
                sample_kwargs["initial_noise"] = initial_noise
            rollout = self.policy.sample_flow_sde(flow_inputs, **sample_kwargs)
            action_components = self.policy.unapply_action_transforms(rollout.actions)
            probability_model = "gaussian_flow_sde"

        predicted_chunks = _libero_action_batch(action_components)
        predictions = []
        for index, task in enumerate(tasks):
            predicted_chunk = predicted_chunks[index]
            native_action = predicted_chunk[: self.policy.execution_horizon]
            metadata: dict[str, Any] = {
                "framework": "nvidia-gr00t",
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
                metadata.update(
                    {
                        "selected_denoise_index": int(selected_indices[index].item()),
                        TRANSIENT_FLOW_SDE_ROLLOUT_KEY: item,
                    }
                )
                logprobs = {
                    "element_logprobs": make_json_safe(old_logprobs),
                    "chunk_logprob": float(old_logprobs.sum().item()),
                }
            action = Action(
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
                    action=action,
                    predicted_action_chunk=predicted_chunk,
                    execution_horizon=self.policy.execution_horizon,
                )
            )
        return predictions

    def stateful_component_ids(self) -> tuple[int, ...]:
        return (id(self.policy), id(self.policy.native_policy), id(self.policy.model))


@dataclass(slots=True)
class GR00TN17FlowSDEPolicyAdapter:
    """Map one declared N1.7 embodiment through its native processor contract."""

    policy: Any
    sampling_mode: Literal["train", "eval", "stochastic_eval"] = "train"
    runtime_profile: Literal[
        "libero",
        "droid",
        "robocasa_gr1_tabletop",
    ] = "libero"

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            _seed_policy(seed)

    def predict(
        self,
        observation: Mapping[str, Any],
        *,
        task: str | None,
        step: int,
        seed: int | None = None,
    ) -> LeRobotActionPrediction:
        if seed is not None:
            _seed_policy(seed)
        return self.predict_batch([observation], tasks=[task], step=step)[0]

    def predict_batch(
        self,
        observations: Sequence[Mapping[str, Any]],
        *,
        tasks: Sequence[str | None],
        step: int,
        selected_indices: Any | None = None,
        initial_noise: Any | None = None,
    ) -> list[LeRobotActionPrediction]:
        if not observations:
            return []
        if len(observations) != len(tasks):
            raise ValueError("GR00T N1.7 observations and tasks must have equal length")
        import torch

        if self.runtime_profile == "libero":
            native_input = _libero_n1d7_observation_batch(observations, tasks)
        elif self.runtime_profile == "droid":
            native_input = _droid_n1d7_observation_batch(observations, tasks)
        else:
            native_input = _robocasa_gr1_n1d7_observation_batch(observations)
        rollout = None
        if self.sampling_mode == "eval":
            action_components = self.policy.predict_native_action_chunk(native_input)
            probability_model = "native_flow_ode"
        else:
            flow_inputs = self.policy.prepare_flow_inputs(native_input)
            if selected_indices is None:
                selected_index = random.randint(0, self.policy.schedule.num_steps - 1)
                selected_indices = torch.full(
                    (len(observations),),
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
            if selected_indices.shape != (len(observations),):
                raise ValueError("N1.7 selected_indices must have one row per sample")
            sample_kwargs: dict[str, Any] = {"selected_index": selected_indices}
            if initial_noise is not None:
                sample_kwargs["initial_noise"] = initial_noise
            rollout = self.policy.sample_flow_sde(flow_inputs, **sample_kwargs)
            action_components = self.policy.decode_action_transforms(
                rollout.actions, native_input
            )
            probability_model = "gaussian_flow_sde"

        if self.runtime_profile == "libero":
            predicted_chunks = _libero_action_batch(action_components)
            action_dimension_mask = tuple(True for _ in range(self.policy.action_dim))
        elif self.runtime_profile == "droid":
            predicted_chunks, action_dimension_mask = _droid_action_batch(
                action_components,
                action_components_layout=self.policy.action_components,
            )
        else:
            predicted_chunks, action_dimension_mask = _robocasa_gr1_action_batch(
                action_components,
                action_components_layout=self.policy.action_components,
            )
        predictions = []
        for index, task in enumerate(tasks):
            predicted_chunk = predicted_chunks[index]
            native_action = predicted_chunk[: self.policy.execution_horizon]
            metadata: dict[str, Any] = {
                "framework": "nvidia-gr00t",
                "policy_type": self.policy.family,
                "probability_model": probability_model,
                "model_action_horizon": self.policy.model_action_horizon,
                "processor_action_horizon": self.policy.processor_action_horizon,
                "execution_horizon": self.policy.execution_horizon,
                "action_dim": self.policy.action_dim,
                "execution_action_dim": int(predicted_chunk.shape[-1]),
                "task": task,
                "model_batch_size": len(observations),
                FLOW_SDE_ACTION_DIMENSION_MASK_KEY: list(action_dimension_mask),
            }
            logprobs = None
            if rollout is not None and self.sampling_mode == "train":
                item = rollout.select(index).cpu()
                old_logprobs = item.transition.old_logprobs[
                    :, : self.policy.execution_horizon, : self.policy.action_dim
                ]
                metadata.update(
                    {
                        "selected_denoise_index": int(selected_indices[index].item()),
                        TRANSIENT_FLOW_SDE_ROLLOUT_KEY: item,
                    }
                )
                scored_logprobs = old_logprobs[..., list(action_dimension_mask)]
                logprobs = {
                    "element_logprobs": make_json_safe(scored_logprobs),
                    "chunk_logprob": float(scored_logprobs.sum().item()),
                }
            elif rollout is not None:
                metadata["selected_denoise_index"] = int(
                    selected_indices[index].item()
                )
            action = Action(
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
                    action=action,
                    predicted_action_chunk=predicted_chunk,
                    execution_horizon=self.policy.execution_horizon,
                )
            )
        return predictions

    def stateful_component_ids(self) -> tuple[int, ...]:
        return (id(self.policy), id(self.policy.native_policy), id(self.policy.model))


def _libero_observation_batch(
    observations: Sequence[Mapping[str, Any]],
    tasks: Sequence[str | None],
) -> dict[str, np.ndarray]:
    if any(task is None or not str(task).strip() for task in tasks):
        raise ValueError("GR00T LIBERO requires non-empty task descriptions")
    images = np.stack([np.asarray(row["image"]) for row in observations])[:, None]
    wrist_images = np.stack([np.asarray(row["wrist_image"]) for row in observations])[
        :, None
    ]
    states = np.stack([np.asarray(row["proprio_state"]) for row in observations])[
        :, None
    ]
    if states.shape[-1] != 8:
        raise ValueError(f"GR00T LIBERO expects 8 state values, got {states.shape}")
    # RLinf quantizes simulator state at its actor boundary before applying
    # checkpoint normalization. Reproduce that published rollout contract.
    import torch

    states = torch.from_numpy(states).to(torch.bfloat16).float().numpy()
    return {
        "video.image": images,
        "video.wrist_image": wrist_images,
        "state.x": states[:, :, 0:1],
        "state.y": states[:, :, 1:2],
        "state.z": states[:, :, 2:3],
        "state.roll": states[:, :, 3:4],
        "state.pitch": states[:, :, 4:5],
        "state.yaw": states[:, :, 5:6],
        "state.gripper": states[:, :, 6:],
        "annotation.human.action.task_description": np.asarray(tasks),
    }


def _libero_n1d7_observation_batch(
    observations: Sequence[Mapping[str, Any]],
    tasks: Sequence[str | None],
) -> dict[str, dict[str, Any]]:
    if any(task is None or not str(task).strip() for task in tasks):
        raise ValueError("GR00T N1.7 LIBERO requires non-empty task descriptions")
    images = np.stack([np.asarray(row["image"]) for row in observations])[:, None]
    wrist_images = np.stack([np.asarray(row["wrist_image"]) for row in observations])[
        :, None
    ]
    states = np.stack([np.asarray(row["proprio_state"]) for row in observations])[
        :, None
    ].astype(np.float32, copy=False)
    if states.shape[-1] != 8:
        raise ValueError(
            f"GR00T N1.7 LIBERO expects 8 state values, got {states.shape}"
        )
    return {
        "video": {"image": images, "wrist_image": wrist_images},
        "state": {
            "x": states[:, :, 0:1],
            "y": states[:, :, 1:2],
            "z": states[:, :, 2:3],
            "roll": states[:, :, 3:4],
            "pitch": states[:, :, 4:5],
            "yaw": states[:, :, 5:6],
            "gripper": states[:, :, 6:],
        },
        "language": {
            "annotation.human.action.task_description": [[str(task)] for task in tasks]
        },
    }


def _droid_n1d7_observation_batch(
    observations: Sequence[Mapping[str, Any]],
    tasks: Sequence[str | None],
) -> dict[str, dict[str, Any]]:
    """Build the exact N1.7 DROID request used by RoboLab's official client."""

    if any(task is None or not str(task).strip() for task in tasks):
        raise ValueError("GR00T N1.7 DROID requires non-empty task descriptions")
    if len(observations) != len(tasks):
        raise ValueError("GR00T DROID observations and tasks must have equal length")
    exterior = np.stack(
        [np.asarray(row["external_image"], dtype=np.uint8) for row in observations]
    )[:, None]
    wrist = np.stack(
        [np.asarray(row["wrist_image"], dtype=np.uint8) for row in observations]
    )[:, None]
    if exterior.shape[-3:] != (180, 320, 3) or wrist.shape[-3:] != (180, 320, 3):
        raise ValueError(
            "GR00T N1.7 DROID transport images must be HWC uint8 at 180x320"
        )

    def state(name: str, size: int) -> np.ndarray:
        value = np.stack(
            [np.asarray(row[name], dtype=np.float32) for row in observations]
        )[:, None]
        if value.shape[-1] != size:
            raise ValueError(
                f"GR00T N1.7 DROID {name} must have {size} values, got {value.shape}"
            )
        return value

    return {
        "video": {
            "exterior_image_1_left": exterior,
            "wrist_image_left": wrist,
        },
        "state": {
            "eef_9d": state("eef_9d", 9),
            "gripper_position": state("gripper_position", 1),
            "joint_position": state("joint_position", 7),
        },
        "language": {
            "annotation.language.language_instruction": [[str(task)] for task in tasks]
        },
    }


_ROBOCASA_GR1_STATE_COMPONENTS = (
    ("left_arm", 7),
    ("right_arm", 7),
    ("left_hand", 6),
    ("right_hand", 6),
    ("waist", 3),
)


def _robocasa_gr1_n1d7_observation_batch(
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Rebuild the exact N1.7 GR-1 Tabletop processor request.

    The official simulator wrapper publishes namespaced, flat Gymnasium keys.
    ART keeps those keys intact at the environment boundary and nests them only
    here, immediately before invoking NVIDIA's serialized processor.
    """

    if not observations:
        raise ValueError("GR00T N1.7 RoboCasa requires at least one observation")

    def array(key: str, *, dtype: Any) -> np.ndarray:
        try:
            value = np.stack(
                [np.asarray(row[key], dtype=dtype) for row in observations]
            )
        except KeyError as exc:
            raise KeyError(f"RoboCasa observation is missing {key!r}") from exc
        return value[:, None]

    state: dict[str, np.ndarray] = {}
    for name, size in _ROBOCASA_GR1_STATE_COMPONENTS:
        value = array(f"state.{name}", dtype=np.float32)
        if value.shape[-1] != size:
            raise ValueError(
                f"RoboCasa state.{name} must have {size} values, got {value.shape}"
            )
        state[name] = value

    simulator_language_key = "annotation.human.coarse_action"
    language = []
    for observation in observations:
        value = observation.get(simulator_language_key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "RoboCasa observation requires a non-empty "
                f"{simulator_language_key!r} string"
            )
        language.append([value])

    background_video = "video.ego_view_bg_crop_pad_res256_freq20"
    video = {
        "ego_view_bg_crop_pad_res256_freq20": array(background_video, dtype=np.uint8),
    }
    for key, value in video.items():
        if value.shape[-3:] != (256, 256, 3):
            raise ValueError(
                f"RoboCasa video.{key} must be HWC uint8 at 256x256, got {value.shape}"
            )

    return {
        "video": video,
        "state": state,
        # NVIDIA's sim compatibility wrapper maps the flat annotation key to
        # the built-in RoboCasa processor's canonical language modality.
        "language": {"task": language},
    }


def _libero_action_batch(
    action_components: Mapping[str, Any],
) -> np.ndarray:
    prefixed_keys = (
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    )
    bare_keys = tuple(key.removeprefix("action.") for key in prefixed_keys)
    keys = (
        prefixed_keys
        if all(key in action_components for key in prefixed_keys)
        else bare_keys
    )
    missing = [key for key in keys if key not in action_components]
    if missing:
        raise KeyError("GR00T action output is missing: " + ", ".join(missing))
    chunks = np.concatenate(
        [np.asarray(action_components[key]) for key in keys],
        axis=-1,
    )
    if chunks.shape[-1] != 7:
        raise ValueError(f"GR00T LIBERO action must have 7 values, got {chunks.shape}")
    chunks = chunks.copy()
    chunks[..., -1] = np.sign(1.0 - 2.0 * chunks[..., -1])
    return chunks


def _droid_action_batch(
    action_components: Mapping[str, Any],
    *,
    action_components_layout: Sequence[tuple[str, int, bool, int | None]],
) -> tuple[np.ndarray, tuple[bool, ...]]:
    """Decode processor-ordered actions into RoboLab's joint/gripper command.

    N1.7 DROID predicts EEF, gripper, and joint representations. RoboLab applies
    only joint positions followed by the gripper command. The returned mask stays
    in processor order so the GRPO objective scores only those causal dimensions.
    """

    if not action_components_layout:
        raise ValueError("GR00T N1.7 DROID requires an explicit action layout")
    decoded: dict[str, np.ndarray] = {}
    dimension_mask: list[bool] = []
    execution_rows: list[tuple[int, str, np.ndarray]] = []
    for key, size, executed, execution_order in action_components_layout:
        value = _action_component(action_components, key)
        if value.shape[-1] != size:
            raise ValueError(
                f"GR00T DROID action component {key!r} expected {size} values, "
                f"got {value.shape}"
            )
        decoded[key] = value
        dimension_mask.extend([bool(executed)] * size)
        if executed:
            if execution_order is None:
                raise ValueError(f"Executed DROID action {key!r} has no order")
            execution_rows.append((execution_order, key, value))
    execution_rows.sort(key=lambda row: row[0])
    if [row[0] for row in execution_rows] != list(range(len(execution_rows))):
        raise ValueError("DROID execution orders must be contiguous from zero")
    chunks = []
    for _order, key, value in execution_rows:
        value = value.copy()
        if key == "gripper_position":
            value[...] = (value > 0.5).astype(value.dtype)
        chunks.append(value)
    return np.concatenate(chunks, axis=-1), tuple(dimension_mask)


def _robocasa_gr1_action_batch(
    action_components: Mapping[str, Any],
    *,
    action_components_layout: Sequence[tuple[str, int, bool, int | None]],
) -> tuple[np.ndarray, tuple[bool, ...]]:
    """Concatenate the official 29D live GR-1 action in processor order.

    Unlike the DROID profile, every declared RoboCasa component is physically
    executed and therefore every decoded dimension remains in the GRPO score.
    """

    expected_layout = tuple(
        (name, size, True, index)
        for index, (name, size) in enumerate(_ROBOCASA_GR1_STATE_COMPONENTS)
    )
    if tuple(action_components_layout) != expected_layout:
        raise ValueError(
            "RoboCasa GR-1 requires the exact official 29D action layout: "
            f"expected={expected_layout}, actual={tuple(action_components_layout)}"
        )
    chunks = []
    for name, size, _executed, _order in expected_layout:
        value = _action_component(action_components, name)
        if value.shape[-1] != size:
            raise ValueError(
                f"RoboCasa action component {name!r} expected {size} values, "
                f"got {value.shape}"
            )
        chunks.append(value)
    return np.concatenate(chunks, axis=-1), (True,) * 29


def _action_component(
    action_components: Mapping[str, Any],
    key: str,
) -> np.ndarray:
    prefixed = f"action.{key}"
    if prefixed in action_components:
        return np.asarray(action_components[prefixed], dtype=np.float32)
    if key in action_components:
        return np.asarray(action_components[key], dtype=np.float32)
    raise KeyError(
        f"GR00T action output is missing {prefixed!r} or {key!r}; "
        f"keys={sorted(action_components)}"
    )


def _seed_policy(seed: int) -> None:
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
