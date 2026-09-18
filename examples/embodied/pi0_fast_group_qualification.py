"""Bounded, same-failure-group FAST numerical qualification before GRPO."""

import hashlib
import json
import math
import pickle
import time

import torch

from art_embodied.backends.action_token import ActionTokenExample
from art_embodied.backends.loss_scaling import (
    backward_policy_loss,
    unscale_policy_gradients,
)
from art_embodied.experiment import EmbodiedScenario, RolloutContext
from art_embodied.integrations.pi0_fast import PI0FastPolicyAdapter
from art_embodied.trajectories import Observation
from examples.embodied.libero.environment import LiberoTaskCatalog
from examples.embodied.libero.rollout import rollout_libero_group
from examples.embodied.libero.settings import LiberoSettings


def score_detached_batches(policy, examples, batch_size=8):
    scores = []
    for start in range(0, len(examples), batch_size):
        with torch.enable_grad():
            values = policy.action_token_logprobs(examples[start : start + batch_size])
        scores.extend(value.detach().cpu() for value in values)
        # Release this graph before evaluating the next forward's RHS.
        del values
        print(
            f"FAST full-group scoring {min(start + batch_size, len(examples))}/{len(examples)}",
            flush=True,
        )
    return torch.cat(scores)


async def qualify(root, config, policy):
    started = time.monotonic()
    with (root / "active-row-failure/request.pkl").open("rb") as file:
        request = pickle.load(file)
    settings = LiberoSettings.from_config(config)
    catalog = LiberoTaskCatalog(settings)
    policy.eval()
    adapter = PI0FastPolicyAdapter(
        policy=policy,
        robot_type="panda",
        do_sample=True,
        temperature=0.2,
        compute_rollout_logprobs=True,
        action_decoder="native",
        invalid_action_handling="terminate_episode",
        model_batch_size=8,
    )
    examples = []

    def predict(observations, *, tasks, step):
        predictions = adapter.predict_batch(observations, tasks=tasks, step=step)
        for index, (observation, prediction) in enumerate(
            zip(observations, predictions, strict=True)
        ):
            action = prediction.action
            examples.append(
                ActionTokenExample(
                    task=tasks[index],
                    trajectory_index=index,
                    action_index=step,
                    step=step,
                    reward=0.0,
                    prompt=tasks[index],
                    tokens=action.raw["tokens"],
                    logprobs=action.logprobs,
                    observation=Observation(
                        kind="custom", step=step, value=observation
                    ),
                    metadata={"action_metadata": action.metadata},
                )
            )
        print(
            f"FAST full-group qualification action={step} active={len(observations)}",
            flush=True,
        )
        return predictions

    trajectories = await rollout_libero_group(
        config=config,
        policy=policy,
        catalog=catalog,
        settings=settings,
        scenario=EmbodiedScenario.model_validate(request["scenario"]),
        contexts=tuple(RolloutContext(**item) for item in request["contexts"]),
        phase="train",
        embedded_batch_predictor=predict,
        embedded_batch_reset=lambda seed: adapter.reset(seed=seed),
    )
    if len(trajectories) != 8 or not examples:
        raise ValueError("Incomplete failure-group qualification")
    with (root / "full-group-qualification-examples.pkl").open("wb") as file:
        pickle.dump(examples, file)
    actual = score_detached_batches(policy, examples)
    old = torch.cat([torch.tensor(example.logprobs) for example in examples])
    delta = actual - old
    selected = [example for example in examples if 8 <= len(example.tokens) <= 64][:2]
    if len(selected) != 2:
        raise ValueError("No ordinary action rows for gradient qualification")
    trainable = [p for p in policy.parameters() if p.requires_grad]
    digest_before = hashlib.sha256(
        b"".join(p.detach().cpu().numpy().tobytes() for p in trainable)
    ).hexdigest()
    policy.model.zero_grad(set_to_none=True)
    loss = -torch.cat(policy.action_token_logprobs(selected)).mean()
    backward_policy_loss(policy, loss)
    unscale_policy_gradients(policy)
    gradient_l2 = math.sqrt(
        sum(
            float(p.grad.float().square().sum())
            for p in trainable
            if p.grad is not None
        )
    )
    digest_after = hashlib.sha256(
        b"".join(p.detach().cpu().numpy().tobytes() for p in trainable)
    ).hexdigest()
    policy.model.zero_grad(set_to_none=True)
    report = {
        "completed_trajectories": len(trajectories),
        "actions": len(examples),
        "tokens": len(old),
        "finite": bool(torch.isfinite(actual).all()),
        "logprob_abs_mean": float(delta.abs().mean()),
        "ratio_mean": float(delta.exp().mean()),
        "gradient_l2": gradient_l2,
        "weights_unchanged": digest_before == digest_after,
        "optimizer_updates": 0,
        "seconds": time.monotonic() - started,
    }
    (root / "full-group-qualification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report), flush=True)
    if not (
        report["finite"]
        and report["weights_unchanged"]
        and math.isfinite(gradient_l2)
        and gradient_l2 > 0
        and report["logprob_abs_mean"] <= 0.02
        and abs(report["ratio_mean"] - 1) <= 0.02
    ):
        raise ValueError("Full failure group fails numerical qualification")
    return report
