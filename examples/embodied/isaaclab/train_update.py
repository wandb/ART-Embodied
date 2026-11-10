"""Train one complete DROID update through the shared multi-GPU backend.

Rollout collection runs in the simulator runtime. Run this command in the
OpenPI policy runtime, using the same config and a matching reset bank.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import random
import time


def optimizer_steps(optimizer):
    return sorted(
        {int(value["step"]) for value in optimizer.state.values() if "step" in value}
    )


def assert_state_equal(actual, expected):
    import torch

    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or not torch.equal(
            actual.cpu(), expected.cpu()
        ):
            raise ValueError("Restored optimizer tensor differs from checkpoint")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError("Restored optimizer keys differ from checkpoint")
        for key in expected:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise ValueError("Restored optimizer sequence differs from checkpoint")
        for a, b in zip(actual, expected, strict=True):
            assert_state_equal(a, b)
    elif actual != expected:
        raise ValueError("Restored optimizer value differs from checkpoint")


def assert_restored_rng(checkpoint):
    import json

    import numpy as np
    from safetensors.torch import load_file
    import torch

    directory = Path(checkpoint) / "policy"
    tensors = load_file(str(directory / "rng.safetensors"), device="cpu")
    actual = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        actual.update(
            {
                f"cuda_{i}": value
                for i, value in enumerate(torch.cuda.get_rng_state_all())
            }
        )
    if actual.keys() != tensors.keys() or any(
        not torch.equal(actual[key], value) for key, value in tensors.items()
    ):
        raise ValueError("Restored Torch RNG state differs from checkpoint")
    expected = json.loads((directory / "rng.json").read_text())
    numpy_state = np.random.get_state()
    actual_json = {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
    }
    if json.loads(json.dumps(actual_json)) != expected:
        raise ValueError("Restored Python/NumPy RNG state differs from checkpoint")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rollout-dir", type=Path, action="append", required=True)
    parser.add_argument("--record-dir", type=Path, action="append", required=True)
    parser.add_argument("--reset-bank", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.rollout_dir) != len(args.record_dir):
        parser.error("Each rollout directory needs one record directory")

    import json

    import numpy as np
    import torch

    from art_embodied.backends.factory import make_embodied_backend
    from art_embodied.config import EmbodiedExperimentConfig
    from art_embodied.integrations.isaaclab_reset_bank import IsaacLabResetBank
    from art_embodied.policies.factory import make_policy
    from art_embodied.utils import write_json_atomic
    from examples.embodied.isaaclab.droid_environment import DroidSceneConfig
    from examples.embodied.isaaclab.droid_training import (
        logging_groups,
        read_droid_groups,
    )

    config = EmbodiedExperimentConfig.model_validate_json(args.config.read_text())
    if (
        config.policy.type != "openpi_pi05_droid"
        or not config.runtime.distributed_training
    ):
        parser.error(
            "This entrypoint requires OpenPI DROID and the distributed Flow-SDE backend"
        )
    args.result_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.manual_seed(config.experiment.seed)
    random.seed(config.experiment.seed)
    np.random.seed(config.experiment.seed)
    policy = make_policy(config)
    backend = make_embodied_backend(config, policy=policy)
    policy_version = backend.update_step
    optimizer_before = optimizer_steps(backend.backend.optimizer)
    if config.storage.resume_from_checkpoint:
        saved = torch.load(
            Path(config.storage.resume_from_checkpoint)
            / "art_embodied_training_state.pt",
            map_location="cpu",
            weights_only=True,
        )
        assert_state_equal(backend.backend.optimizer.state_dict(), saved["optimizer"])
        assert_restored_rng(config.storage.resume_from_checkpoint)
        if saved["step"] != policy_version:
            raise ValueError("Restored update counter differs from checkpoint")
    load_seconds = time.monotonic() - started
    report = json.loads((args.rollout_dir[0] / "report.json").read_text())
    scene = DroidSceneConfig(**report["settings"])
    bank = IsaacLabResetBank.load(
        args.reset_bank, contract=scene.reset_contract(), split="train"
    )
    if config.environment.kwargs.get("train_reset_bank_sha256") != bank.fingerprint:
        raise ValueError(
            "Training reset bank differs from the experiment configuration"
        )
    groups = []
    begin = time.monotonic()
    for rollout, records in zip(args.rollout_dir, args.record_dir, strict=True):
        groups.extend(
            read_droid_groups(
                rollout,
                records,
                bank=bank,
                policy=policy,
                policy_version=policy_version,
                group_size=config.algorithm.group_size,
                group_offset=len(groups),
            )
        )
    if len(groups) != config.rollout.groups_per_update:
        raise ValueError(
            "Collected group count differs from the training configuration"
        )
    chunks = sum(len(t.actions) for g in groups for t in g.trajectories)
    write_json_atomic(
        args.result_dir / "trace-groups.json",
        [g.model_dump(mode="json") for g in logging_groups(groups)],
    )
    decode_seconds = time.monotonic() - begin
    begin = time.monotonic()
    try:
        result = asyncio.run(backend.train(groups))
    finally:
        if backend._persistent_worker_pool is not None:
            backend._persistent_worker_pool.close()
    if result.step != policy_version + 1 or not all(
        np.isfinite(v) for v in result.metrics.values()
    ):
        raise RuntimeError("Training did not produce the expected finite update")
    write_json_atomic(
        args.result_dir / "result.json",
        {
            "step": result.step,
            "checkpoint_path": str(result.checkpoint_path),
            "metrics": result.metrics,
            "groups": len(groups),
            "trajectories": sum(len(g) for g in groups),
            "chunks": chunks,
            "successes": sum(t.reward for g in groups for t in g.trajectories),
            "model_load_seconds": load_seconds,
            "rollout_decode_seconds": decode_seconds,
            "backend_update_seconds": time.monotonic() - begin,
            "wall_seconds": time.monotonic() - started,
            "optimizer_steps_before": optimizer_before,
            "optimizer_steps_after": optimizer_steps(backend.backend.optimizer),
            "optimizer_resume_exact": bool(config.storage.resume_from_checkpoint),
            "rng_resume_exact": bool(config.storage.resume_from_checkpoint),
        },
    )


if __name__ == "__main__":
    main()
