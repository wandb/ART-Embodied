"""Run the model-backed GR00T N1.5 Flow-SDE conformance gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from art_embodied.backends.flow_sde import TRANSIENT_FLOW_SDE_ROLLOUT_KEY
from art_embodied.config import EmbodiedExperimentConfig
from art_embodied.integrations.gr00t_flow_sde import GR00TN15FlowSDEPolicyAdapter
from art_embodied.policies.factory import make_policy


def run(config_path: Path) -> dict[str, object]:
    """Load real weights and verify rollout/rescore plus native ODE inference."""

    config = EmbodiedExperimentConfig.from_yaml(config_path)
    if config.policy.type != "gr00t_n1d5":
        raise ValueError("GR00T conformance requires policy.type='gr00t_n1d5'")
    policy = make_policy(config)
    observation = {
        "image": np.zeros((256, 256, 3), dtype=np.uint8),
        "wrist_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "proprio_state": np.zeros(8, dtype=np.float32),
    }
    task = "pick up the black bowl and place it on the plate"

    policy.eval()
    eval_prediction = GR00TN15FlowSDEPolicyAdapter(
        policy=policy,
        sampling_mode="eval",
    ).predict(observation, task=task, step=0, seed=11)

    rollout_prediction = GR00TN15FlowSDEPolicyAdapter(policy=policy).predict(
        observation,
        task=task,
        step=0,
        seed=17,
    )
    retained = rollout_prediction.action.metadata[TRANSIENT_FLOW_SDE_ROLLOUT_KEY]
    policy.train()
    rescored = policy.flow_sde_logprobs(retained.to(policy.device))
    old = retained.transition.old_logprobs[
        :, : policy.execution_horizon, : policy.action_dim
    ].to(rescored.device)
    maximum_delta = float((rescored.detach() - old).abs().max().item())
    loss = -rescored.mean()
    loss.backward()
    gradient_tensors = [
        parameter.grad
        for parameter in policy.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradient_tensors:
        raise RuntimeError("GR00T conformance produced no trainable gradients")
    all_gradients_finite = all(
        bool(torch.isfinite(gradient).all()) for gradient in gradient_tensors
    )
    trainable_names = [
        name for name, parameter in policy.named_parameters() if parameter.requires_grad
    ]
    if any("action_head" not in name for name in trainable_names):
        raise RuntimeError("GR00T LoRA escaped the audited action-head surface")

    return {
        "schema_version": 1,
        "model_id": policy.model_id,
        "revision": policy.revision,
        "execution_horizon": policy.execution_horizon,
        "model_action_horizon": policy.model_action_horizon,
        "native_action_shape": list(np.asarray(eval_prediction.native_action).shape),
        "predicted_action_shape": list(
            np.asarray(eval_prediction.predicted_action_chunk).shape
        ),
        "rollout_rescore_max_abs_delta": maximum_delta,
        "trainable_tensor_count": len(trainable_names),
        "gradient_tensor_count": len(gradient_tensors),
        "all_gradients_finite": all_gradients_finite,
        "ok": maximum_delta <= 1.0e-5 and all_gradients_finite,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.config)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
