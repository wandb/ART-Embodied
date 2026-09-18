"""Compare SFT and rollout preprocessing on identical teacher inputs, on CPU."""

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def rollout_input(sample, key_map):
    """Repackage an already oriented teacher image; do not rotate it again."""
    result = {}
    for source, target in key_map.items():
        value = sample[target]
        if target.startswith("observation.images."):
            if value.ndim != 3 or value.shape[0] != 3 or value.dtype != torch.uint8:
                raise ValueError("Expected a CHW uint8 teacher image")
            value = value.permute(1, 2, 0)
        result[source] = np.ascontiguousarray(value.cpu().numpy())
    return result


def compare_observations(teacher, rollout):
    keys = sorted(key for key in teacher if key.startswith("observation."))
    other = sorted(key for key in rollout if key.startswith("observation."))
    if not keys or keys != other:
        raise ValueError("Observation key sets differ or are empty")
    result = {}
    for key in keys:
        a, b = teacher[key], rollout[key]
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise TypeError(f"Non-tensor observation: {key}")
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"Observation shape/dtype mismatch: {key}")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(f"Nonfinite observation: {key}")
        result[key] = {
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "exact": torch.equal(a, b),
            "max_abs_difference": (a.double() - b.double()).abs().max().item(),
        }
    return result


def environment_oracle(preprocessor, task, *, random_states=4096, seed=20260909):
    """Synthetic raw-observation comparison against installed LeRobot, not physics."""
    from lerobot.processor.env_processor import LiberoProcessorStep

    from examples.embodied.libero.environment import _image, _proprio_state

    rng = np.random.default_rng(seed)
    quaternions = rng.normal(size=(random_states, 4))
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    quaternions = np.concatenate(
        [
            quaternions,
            [[0, 0, 0, 1], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        ]
    )
    count = len(quaternions)
    positions = rng.uniform(-0.5, 0.5, (count, 3))
    grippers = rng.uniform(-0.04, 0.04, (count, 2))
    native = LiberoProcessorStep().observation(
        {
            "observation.robot_state": {
                "eef": {
                    "pos": torch.tensor(positions),
                    "quat": torch.tensor(quaternions),
                },
                "gripper": {"qpos": torch.tensor(grippers)},
            }
        }
    )["observation.state"]
    ours = torch.from_numpy(
        np.stack(
            [
                _proprio_state(
                    {
                        "robot0_eef_pos": pos,
                        "robot0_eef_quat": quat,
                        "robot0_gripper_qpos": gripper,
                    }
                )
                for pos, quat, gripper in zip(
                    positions, quaternions, grippers, strict=True
                )
            ]
        )
    )
    delta = (native - ours).abs()
    tokens = []
    for states in (native, ours):
        processed = preprocessor({"observation.state": states, "task": [task] * count})
        tokens.append(processed["observation.language.tokens"].clone())
    changed = (tokens[0] != tokens[1]).any(dim=1)
    cameras = {}
    for key in ("observation.images.image", "observation.images.image2"):
        raw = rng.integers(0, 256, size=(13, 17, 3), dtype=np.uint8)
        native_image = LiberoProcessorStep().observation(
            {key: torch.from_numpy(raw).permute(2, 0, 1)[None].float() / 255}
        )[key]
        ours_image = torch.from_numpy(
            np.ascontiguousarray(_image(raw, rotate_180=True))
        )
        ours_image = ours_image.permute(2, 0, 1)[None].float() / 255
        cameras[key] = torch.equal(native_image, ours_image)
    return {
        "scope": "Synthetic raw observations, not real simulator states or returns",
        "seed": seed,
        "random_states": random_states,
        "identity_and_axis_edge_cases": 5,
        "samples": count,
        "state_max_abs_difference": delta.max().item(),
        "state_allclose_atol_rtol_1e_6": torch.allclose(
            native, ours, atol=1e-6, rtol=1e-6
        ),
        "language_token_changed_rows": changed.sum().item(),
        "language_token_changed_indices": changed.nonzero().flatten().tolist(),
        "camera_rotation_exact": cameras,
    }


def main(inventory, frame_check, bootstrap, output):
    from huggingface_hub import snapshot_download
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors

    from art_embodied.policies.pi0_fast import PI0FastPolicy
    from art_embodied.policies.pi0_fast_sft_anchor import (
        _collate_one,
        _prepare_batch,
    )

    if output.exists():
        raise FileExistsError(output)
    spec = json.loads(inventory.read_text())
    prior = json.loads(frame_check.read_text())
    policy_spec = json.loads(bootstrap.read_text())["config"]["policy"]
    cfg = PreTrainedConfig.from_pretrained(
        policy_spec["path"], revision=policy_spec["revision"]
    )
    cfg.device = "cpu"
    tokenizer = snapshot_download(
        repo_id=cfg.action_tokenizer_name,
        revision=policy_spec["load_kwargs"]["action_tokenizer_revision"],
        local_files_only=True,
    )

    def processors():
        return make_pre_post_processors(
            cfg,
            pretrained_path=policy_spec["path"],
            pretrained_revision=policy_spec["revision"],
            preprocessor_overrides={
                "action_tokenizer_processor": {"action_tokenizer_name": tokenizer},
                "device_processor": {"device": "cpu"},
            },
        )

    # Only processors are loaded. The stub exposes config to the real adapter.
    adapter = PI0FastPolicy(
        model_id=policy_spec["path"],
        revision=policy_spec["revision"],
        device="cpu",
        observation_key_map=policy_spec["load_kwargs"]["observation_key_map"],
        **{
            key: policy_spec["load_kwargs"][key]
            for key in (
                "execution_horizon",
                "action_dim",
                "max_decoding_steps",
                "action_tokenizer_revision",
                "strict_weights",
                "compile_model",
                "gradient_checkpointing",
                "use_kv_cache",
            )
        },
    )
    adapter.policy = SimpleNamespace(config=cfg)
    adapter.preprocessor, adapter.postprocessor = processors()
    teacher_preprocessor, _ = processors()
    teacher = SimpleNamespace(preprocessor=teacher_preprocessor)
    raw_observation_comparison = environment_oracle(teacher_preprocessor, spec["task"])
    data = LeRobotDataset(
        "lerobot/libero",
        root=Path(spec["root"]),
        revision=spec["revision"],
        episodes=spec["episodes"],
        delta_timestamps=prior["delta_timestamps"],
        return_uint8=True,
        video_backend="pyav",
    )
    locations = defaultdict(list)
    for index, episode in enumerate(data.hf_dataset["episode_index"]):
        locations[int(episode)].append(index)
    if {str(k): len(v) for k, v in locations.items()} != spec["episode_lengths"]:
        raise ValueError("Teacher population differs from inventory")
    checked = []
    for episode, indices in sorted(locations.items()):
        for index in (indices[0], indices[len(indices) // 2], indices[-1]):
            sample = data[index]
            if sample["task"] != spec["task"]:
                raise ValueError("Different teacher task")
            sft = _prepare_batch(
                _collate_one(deepcopy(sample), data), teacher, data.meta.camera_keys
            )
            rollout = adapter.preprocess_observation(
                rollout_input(sample, adapter.observation_key_map),
                task=sample["task"],
            )
            compared = compare_observations(sft, rollout)
            checked.append(
                {
                    "episode": episode,
                    "frame": int(sample["frame_index"]),
                    "fields": compared,
                    "exact": all(v["exact"] for v in compared.values()),
                }
            )
        print(json.dumps({"episode": episode, "checked": len(checked)}), flush=True)
    result = {
        "scope": __doc__,
        "inputs": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (inventory, frame_check, bootstrap)
        },
        "model": policy_spec["path"],
        "revision": policy_spec["revision"],
        "action_tokenizer_revision": policy_spec["load_kwargs"][
            "action_tokenizer_revision"
        ],
        "use_relative_actions": cfg.use_relative_actions,
        "key_map": adapter.observation_key_map,
        "processor_steps": [type(s).__name__ for s in teacher_preprocessor.steps],
        "synthetic_raw_observation_comparison": raw_observation_comparison,
        "checked": checked,
        "all_exact": bool(checked) and all(row["exact"] for row in checked),
        "model_loaded": False,
        "gpu_used": False,
        "sealed_accessed": False,
        "limitations": [
            "Same-input observation comparison only; not simulator-to-demonstration physical equivalence.",
            "No neural forward, attention mask, action decoding, gradient or closed-loop success check.",
            "Teacher images are 256px, runtime renders are 360px; rendering geometry is not tested.",
        ],
    }
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    if not result["all_exact"]:
        raise RuntimeError("Preprocessing differences recorded in output")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inventory", "frame-check", "bootstrap", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    main(args.inventory, args.frame_check, args.bootstrap, args.output)
