"""Independent finite-tree oracle for the production GRPO backend, CPU only.

This tests algebra, not pi0-FAST transformer numerics or LIBERO learning. The
oracle uses scalar probabilities and central differences, not backend helpers.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import math
from pathlib import Path

import torch

from art_embodied import (
    Action,
    ActionTokenGRPOBackend,
    EmbodiedTrajectory,
    EmbodiedTrajectoryGroup,
)
from art_embodied.backends.action_token import extract_action_token_examples
from art_embodied.backends.local_process import _attach_full_update_advantages

# Token 2 terminates immediately; otherwise a second token ends the episode.
LEAVES = ((2,),) + tuple(itertools.product(range(2), range(3)))
INITIAL = (0.12, -0.07, 0.03, -0.04, 0.13, 0.02, 0.09, -0.11, 0.04)


def reward(leaf):
    # Two distinct latent strings decode to the same successful physical action.
    return float(leaf in ((0, 0), (0, 1), (1, 2)))


def contexts(leaf):
    return (0,) if len(leaf) == 1 else (0, 1 + leaf[0])


def token_probabilities(weights, leaf, temperature):
    result = []
    for context, token in zip(contexts(leaf), leaf, strict=True):
        row = [v / temperature for v in weights[3 * context : 3 * context + 3]]
        values = [math.exp(v - max(row)) for v in row]
        result.append(values[token] / sum(values))
    return result


def success_probability(weights, temperature):
    return sum(
        reward(leaf) * math.prod(token_probabilities(weights, leaf, temperature))
        for leaf in LEAVES
    )


def finite_difference(function, weights, delta=1e-5):
    result = []
    for index in range(len(weights)):
        plus, minus = list(weights), list(weights)
        plus[index] += delta
        minus[index] -= delta
        result.append((function(plus) - function(minus)) / (2 * delta))
    return torch.tensor(result, dtype=torch.float64)


def scalar_surrogate(weights, leaves, temperature, aggregation, normalize):
    rewards = [reward(leaf) for leaf in leaves]
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((v - mean) ** 2 for v in rewards) / (len(rewards) - 1))
    total = 0.0
    for leaf, value in zip(leaves, rewards, strict=True):
        advantage = value - mean
        if normalize:
            advantage = advantage / (std + 1e-6) if std > 1e-6 else 0.0
        old = token_probabilities(INITIAL, leaf, temperature)
        new = token_probabilities(weights, leaf, temperature)
        losses = []
        for before, after in zip(old, new, strict=True):
            ratio = after / before
            clipped = min(1.28, max(0.8, ratio))
            losses.append(max(-advantage * ratio, -advantage * clipped))
        total += sum(losses) / (len(leaf) if aggregation == "trajectory_mean" else 1)
    return total / len(leaves)


class TreePolicy(torch.nn.Module):
    def __init__(self, weights, temperature):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.tensor(weights, dtype=torch.float64))
        self.temperature = temperature

    def action_token_logprobs(self, examples):
        table = torch.log_softmax(self.weights.reshape(3, 3) / self.temperature, -1)
        return [
            table[e.metadata["action_metadata"]["contexts"], e.tokens] for e in examples
        ]


def make_group(leaves, temperature, layout, *, pad=False):
    trajectories = []
    for leaf in leaves:
        trajectory = EmbodiedTrajectory(task="finite token tree", reward=reward(leaf))
        logs = [math.log(p) for p in token_probabilities(INITIAL, leaf, temperature)]
        ctx = contexts(leaf)
        spans = (
            [(0, len(leaf))]
            if layout == "tokens"
            else [(k, k + 1) for k in range(len(leaf))]
        )
        for step, (start, stop) in enumerate(spans):
            tokens, old, rows = (
                list(leaf[start:stop]),
                logs[start:stop],
                list(ctx[start:stop]),
            )
            mask = [True] * len(tokens)
            if pad:
                tokens += [1]
                old += [math.log(token_probabilities(INITIAL, (1,), temperature)[0])]
                rows += [0]
                mask += [False]
            trajectory.actions.append(
                Action(
                    step=step,
                    kind="token",
                    raw={"tokens": tokens, "prompt": trajectory.task},
                    logprobs=old,
                    metadata={"contexts": rows, "token_loss_mask": mask},
                )
            )
        trajectories.append(trajectory)
    return EmbodiedTrajectoryGroup(trajectories)


async def backend_gradient(
    leaves,
    *,
    weights=INITIAL,
    temperature=0.2,
    aggregation="seq_mean_token_sum",
    normalize=False,
    layout="tokens",
    microbatch=2,
    shards=1,
    pad=False,
):
    policy = TreePolicy(weights, temperature)
    backend = ActionTokenGRPOBackend(
        policy,
        optimizer=torch.optim.SGD(policy.parameters(), lr=0),
        normalize_advantages=normalize,
        advantage_normalization_scope="group",
        advantage_std_unbiased=True,
        advantage_epsilon=1e-6,
        rlinf_action_level_score_source="trajectory_reward",
        training_unit="action",
        loss_aggregation=aggregation,
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.28,
        logprob_microbatch_size=microbatch,
        skip_optimizer_step_without_policy_gradient_signal=False,
    )
    group = make_group(leaves, temperature, layout, pad=pad)
    examples = extract_action_token_examples(
        [group], require_logprobs=True, score_source="trajectory_reward"
    )
    _attach_full_update_advantages(examples, backend=backend)
    gradient = torch.zeros_like(policy.weights)
    for index in range(shards):
        shard = examples[index::shards]
        if not shard:
            continue
        await backend.train(
            [],
            _action_token_grpo_return_gradients=True,
            _action_token_grpo_precomputed_examples=shard,
            _action_token_grpo_precomputed_examples_prepared=True,
            _action_token_grpo_global_example_count=len(leaves),
            _action_token_grpo_global_token_count=sum(len(e.tokens) for e in examples),
        )
        gradient += policy.weights.grad.detach()
    return gradient


async def exact_score_identity(temperature, normalize):
    expected = torch.zeros(9, dtype=torch.float64)
    mass = 0.0
    for pair in itertools.product(LEAVES, repeat=2):
        probability = math.prod(
            math.prod(token_probabilities(INITIAL, leaf, temperature)) for leaf in pair
        )
        mass += probability
        expected -= probability * await backend_gradient(
            pair, temperature=temperature, normalize=normalize
        )
    true_gradient = finite_difference(
        lambda w: success_probability(w, temperature), INITIAL
    )
    # Self-including group mean gives (G-1)/G. For binary rewards and G=2,
    # every mixed group has the same sample std; no such claim for general G.
    scale = 0.5 / (math.sqrt(0.5) + 1e-6) if normalize else 0.5
    return {
        "temperature": temperature,
        "normalize": normalize,
        "enumerated_groups": 49,
        "probability_mass": mass,
        "true_success_gradient": true_gradient.tolist(),
        "expected_backend_ascent": expected.tolist(),
        "expected_scale": scale,
        "max_abs_error": float((expected - scale * true_gradient).abs().max()),
    }


async def audit():
    return {
        "scope": "Finite-tree algebra, not transformer parity or LIBERO learning",
        "identities": [
            await exact_score_identity(t, n) for t in (0.2, 1.0) for n in (False, True)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(audit())
    assert all(
        abs(r["probability_mass"] - 1) < 1e-12 and r["max_abs_error"] < 1e-6
        for r in result["identities"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
