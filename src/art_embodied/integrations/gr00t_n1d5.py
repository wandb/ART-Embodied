"""Official-GR00T data contract used by the N1.5 LIBERO positive control.

The transform geometry is adapted from RLinf release/v0.1 at commit
9df6dc80dc729a6caccab92dd676004e51b1d3a2 (Apache-2.0, Copyright 2025 The
RLinf Authors), while this module depends only on NVIDIA's GR00T runtime. The
lazy imports, immutable compatibility boundary, tuple-based schema, and
validation are ART-Embodied modifications. Keeping the contract here makes the
normalization, crop, horizon, and key mapping inspectable without importing
the RLinf training stack.
"""

from __future__ import annotations

from enum import Enum
import sys


class GR00TN15EmbodimentTag(Enum):
    """N1.5 tags required by the published RLinf checkpoint family."""

    GR1 = "gr1"
    OXE_DROID = "oxe_droid"
    AGIBOT_GENIE1 = "agibot_genie1"
    LIBERO_FRANKA = "libero_franka"
    MANISKILL_WIDOWX = "maniskill_widowx"
    ISAACLAB_FRANKA = "isaaclab_franka"
    NEW_EMBODIMENT = "new_embodiment"


GR00T_N1D5_EMBODIMENT_IDS = {
    GR00TN15EmbodimentTag.LIBERO_FRANKA.value: 31,
    GR00TN15EmbodimentTag.OXE_DROID.value: 17,
    GR00TN15EmbodimentTag.AGIBOT_GENIE1.value: 26,
    GR00TN15EmbodimentTag.GR1.value: 24,
    GR00TN15EmbodimentTag.MANISKILL_WIDOWX.value: 30,
    GR00TN15EmbodimentTag.ISAACLAB_FRANKA.value: 31,
    GR00TN15EmbodimentTag.NEW_EMBODIMENT.value: 10,
}


def install_gr00t_n1d5_embodiment_compatibility() -> None:
    """Expose checkpoint tags missing from NVIDIA's frozen N1.5 branch.

    RLinf patches the same process-global symbols before loading its N1.5
    checkpoints. Keeping this narrow compatibility boundary in ART avoids a
    runtime dependency on RLinf while preserving the published checkpoint's
    immutable embodiment IDs.
    """

    from gr00t.data import embodiment_tags

    embodiment_tags.EmbodimentTag = GR00TN15EmbodimentTag
    embodiment_tags.EMBODIMENT_TAG_MAPPING = GR00T_N1D5_EMBODIMENT_IDS
    for module_name in (
        "gr00t.data.dataset",
        "gr00t.data.schema",
        "gr00t.model.policy",
        "gr00t.model.transforms",
    ):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        module.EmbodimentTag = GR00TN15EmbodimentTag
        if hasattr(module, "EMBODIMENT_TAG_MAPPING"):
            module.EMBODIMENT_TAG_MAPPING = GR00T_N1D5_EMBODIMENT_IDS


class LiberoFrankaDataConfig:
    """Construct the N1.5 LIBERO modality and invertible transform contract."""

    video_keys = ("video.image", "video.wrist_image")
    state_keys = (
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.gripper",
    )
    action_keys = (
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    )
    language_keys = ("annotation.human.action.task_description",)
    observation_indices = (0,)
    action_indices = tuple(range(16))

    def modality_config(self):
        from gr00t.data.dataset import ModalityConfig

        return {
            "video": ModalityConfig(
                delta_indices=list(self.observation_indices),
                modality_keys=list(self.video_keys),
            ),
            "state": ModalityConfig(
                delta_indices=list(self.observation_indices),
                modality_keys=list(self.state_keys),
            ),
            "action": ModalityConfig(
                delta_indices=list(self.action_indices),
                modality_keys=list(self.action_keys),
            ),
            "language": ModalityConfig(
                delta_indices=list(self.observation_indices),
                modality_keys=list(self.language_keys),
            ),
        }

    def transform(self):
        from gr00t.data.transform.base import ComposedModalityTransform
        from gr00t.data.transform.concat import ConcatTransform
        from gr00t.data.transform.state_action import (
            StateActionToTensor,
            StateActionTransform,
        )
        from gr00t.data.transform.video import (
            VideoColorJitter,
            VideoCrop,
            VideoResize,
            VideoToNumpy,
            VideoToTensor,
        )
        from gr00t.model.transforms import GR00TTransform

        state_modes = dict.fromkeys(self.state_keys, "min_max")
        action_modes = dict.fromkeys(self.action_keys, "min_max")
        transforms = [
            VideoToTensor(apply_to=list(self.video_keys)),
            VideoCrop(apply_to=list(self.video_keys), scale=0.95),
            VideoResize(
                apply_to=list(self.video_keys),
                height=224,
                width=224,
                interpolation="linear",
            ),
            VideoColorJitter(
                apply_to=list(self.video_keys),
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=list(self.video_keys)),
            StateActionToTensor(apply_to=list(self.state_keys)),
            StateActionTransform(
                apply_to=list(self.state_keys),
                normalization_modes=state_modes,
            ),
            StateActionToTensor(apply_to=list(self.action_keys)),
            StateActionTransform(
                apply_to=list(self.action_keys),
                normalization_modes=action_modes,
            ),
            ConcatTransform(
                video_concat_order=list(self.video_keys),
                state_concat_order=list(self.state_keys),
                action_concat_order=list(self.action_keys),
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)
