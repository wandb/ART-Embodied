"""LeRobot pi0-FAST rollout adapter for categorical action-token GRPO."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from art_embodied.policies.pi0_fast import (
    PI0FastPolicy,
    _concatenate_processed_batches,
)
from art_embodied.trajectories import Action
from art_embodied.utils import make_json_safe

from .lerobot import LeRobotActionPrediction


@dataclass(slots=True)
class PI0FastPolicyAdapter:
    """Generate FAST tokens while retaining exact rollout probabilities."""

    policy: PI0FastPolicy
    robot_type: str | None = "panda"
    sampling_mode: Literal["train", "eval"] = "train"
    do_sample: bool = True
    temperature: float = 1.0
    compute_rollout_logprobs: bool = True
    model_batch_size: int | None = None
    invalid_action: tuple[float, ...] | None = None
    invalid_action_handling: Literal["raise", "terminate_episode"] = "raise"
    action_decoder: Literal["strict", "native"] = "strict"

    def __post_init__(self) -> None:
        if self.action_decoder not in ("strict", "native"):
            raise ValueError("Unsupported action_decoder")
        if self.action_decoder == "native" and (
            self.invalid_action is not None
            or getattr(self.policy, "rl_token_scope", "generated_sequence")
            != "generated_sequence"
        ):
            raise ValueError(
                "Native decoding requires generated_sequence and no fallback action"
            )
        if self.invalid_action_handling not in ("raise", "terminate_episode"):
            raise ValueError("Unsupported invalid_action_handling")
        if self.invalid_action_handling == "terminate_episode":
            if self.invalid_action is not None:
                raise ValueError("Termination cannot also execute a fallback action")
            if (
                getattr(self.policy, "rl_token_scope", "generated_sequence")
                != "generated_sequence"
            ):
                raise ValueError(
                    "terminate_episode requires generated_sequence token scope"
                )

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            _seed_lerobot(seed)
        self.policy.reset()

    def predict(
        self,
        observation: Mapping[str, Any],
        *,
        task: str | None,
        step: int,
        seed: int | None = None,
    ) -> LeRobotActionPrediction:
        if seed is not None:
            _seed_lerobot(seed)
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
        **_kwargs: Any,
    ) -> list[LeRobotActionPrediction]:
        if not observations:
            return []
        if len(observations) != len(tasks):
            raise ValueError("pi0-FAST observations and tasks must have equal length")
        if self.model_batch_size is not None:
            if self.model_batch_size <= 0:
                raise ValueError("pi0-FAST model_batch_size must be positive")
            if len(observations) > self.model_batch_size:
                predictions: list[LeRobotActionPrediction] = []
                for start in range(0, len(observations), self.model_batch_size):
                    stop = start + self.model_batch_size
                    predictions.extend(
                        self.predict_batch(
                            observations[start:stop],
                            tasks=tasks[start:stop],
                            step=step,
                        )
                    )
                return predictions
            if len(observations) < self.model_batch_size:
                padding = self.model_batch_size - len(observations)
                padded_observations = list(observations) + [observations[-1]] * padding
                padded_tasks = list(tasks) + [tasks[-1]] * padding
                return replace(self, model_batch_size=None).predict_batch(
                    padded_observations,
                    tasks=padded_tasks,
                    step=step,
                )[: len(observations)]
        processed_rows = [
            self.policy.preprocess_observation(
                observation,
                task=task,
                robot_type=self.robot_type,
            )
            for observation, task in zip(observations, tasks, strict=True)
        ]
        processed = _concatenate_processed_batches(processed_rows)
        # The generation contract, not the rollout phase label, determines
        # whether decoding is stochastic. Normal evaluation remains greedy
        # because evaluation_generation.do_sample defaults to false, while
        # paired sampler calibration can explicitly request sampled eval.
        sampling_temperature = float(self.temperature) if self.do_sample else 0.0

        import torch

        with torch.no_grad():
            sampler_logprobs = None
            if sampling_temperature > 0.0 and self.compute_rollout_logprobs:
                token_tensor, sampler_logprobs = (
                    self.policy.sample_action_tokens_with_logprobs(
                        processed,
                        temperature=sampling_temperature,
                    )
                )
            else:
                token_tensor = self.policy.sample_action_tokens(
                    processed,
                    temperature=sampling_temperature,
                )
            token_rows, grammar_valid, discarded_token_counts, payload_masks = (
                self.policy.prepare_generated_action_tokens(
                    token_tensor,
                    **(
                        {"native_decoder": True}
                        if self.action_decoder == "native"
                        else {}
                    ),
                )
            )
            old_logprobs = (
                [
                    sampler_logprobs[index, : len(row)]
                    for index, row in enumerate(token_rows)
                ]
                if sampler_logprobs is not None
                else [None] * len(observations)
            )
            decode_errors: list[str | None] = [None] * len(observations)
            decode_valid = grammar_valid
            if self.action_decoder == "native":
                predicted_chunks, decode_errors = (
                    self.policy.decode_action_tokens_native(token_tensor)
                )
                decode_valid = [error is None for error in decode_errors]
                if not all(decode_valid) and self.invalid_action_handling == "raise":
                    raise ValueError(f"Native FAST decoding failed: {decode_errors}")
            elif self.invalid_action_handling == "terminate_episode" and not all(
                grammar_valid
            ):
                # Rejected decisions carry tokens and a terminal outcome, but
                # no physical action. Never inverse-normalize a pretend zero.
                predicted_chunks = [
                    torch.empty((0, self.policy.action_dim)) for _ in observations
                ]
                valid_indices = [i for i, valid in enumerate(grammar_valid) if valid]
                if valid_indices:
                    decoded = self.policy.decode_action_tokens_safe(
                        token_tensor[valid_indices],
                        grammar_valid=[True] * len(valid_indices),
                    )
                    for i, chunk in zip(valid_indices, decoded, strict=True):
                        predicted_chunks[i] = chunk
            else:
                predicted_chunks = self.policy.decode_action_tokens_safe(
                    token_tensor,
                    grammar_valid=grammar_valid,
                    **(
                        {"invalid_action": self.invalid_action}
                        if self.invalid_action is not None
                        else {}
                    ),
                )

        predictions: list[LeRobotActionPrediction] = []
        for index, task in enumerate(tasks):
            rejected = (
                not decode_valid[index]
                and self.invalid_action_handling == "terminate_episode"
            )
            predicted_chunk = predicted_chunks[index].detach().cpu()
            native_action = predicted_chunk[: self.policy.execution_horizon]
            token_row = token_rows[index]
            payload_mask = payload_masks[index]
            logprob_row = old_logprobs[index]
            rl_token_scope = getattr(
                self.policy, "rl_token_scope", "generated_sequence"
            )
            token_loss_mask = (
                payload_mask
                if rl_token_scope == "fast_payload"
                else [True] * len(token_row)
            )
            recorded = Action(
                step=step,
                kind="token",
                raw={
                    "tokens": [int(token) for token in token_row],
                    "prompt": task,
                },
                decoded=make_json_safe(native_action),
                logprobs=(
                    [float(value) for value in logprob_row.detach().cpu().tolist()]
                    if logprob_row is not None
                    else None
                ),
                metadata={
                    "framework": "lerobot",
                    "policy_type": "pi0_fast",
                    "probability_model": (
                        "categorical_tokens" if sampling_temperature > 0.0 else "greedy"
                    ),
                    "sampling_temperature": sampling_temperature,
                    "rollout_logprobs_computed": bool(
                        sampling_temperature > 0.0 and self.compute_rollout_logprobs
                    ),
                    "execution_horizon": 0
                    if rejected
                    else self.policy.execution_horizon,
                    "model_horizon": int(self.policy.config.chunk_size),
                    "action_dim": self.policy.action_dim,
                    "generated_token_count": len(token_row),
                    "rl_token_scope": rl_token_scope,
                    "token_loss_mask": token_loss_mask,
                    "rl_objective_token_count": sum(token_loss_mask),
                    "fast_payload_token_count": sum(payload_mask),
                    "format_token_count": len(token_row) - sum(payload_mask),
                    "sampled_token_count": int(token_tensor.shape[1]),
                    "post_termination_tokens_discarded": discarded_token_counts[index],
                    "action_grammar_valid": grammar_valid[index],
                    "action_decoder": self.action_decoder,
                    "action_decode_valid": decode_valid[index],
                    "action_decode_error": decode_errors[index],
                    # Explicit malformed-action fallbacks are not model outputs
                    # and must never receive a policy-gradient
                    # update. Otherwise a successful episode can reinforce the
                    # invalid token sequence that happened to precede the fallback.
                    **(
                        {
                            "terminate_episode": True,
                            "termination_reason": "invalid_action_tokens",
                            "executed_primitive_count": 0,
                        }
                        if rejected
                        else {"primitive_loss_mask_sum": 0}
                        if not decode_valid[index]
                        else {}
                    ),
                    "task": task,
                    "model_batch_size": len(observations),
                },
            )
            predictions.append(
                LeRobotActionPrediction(
                    native_action=native_action,
                    action=recorded,
                    predicted_action_chunk=predicted_chunk,
                    execution_horizon=0 if rejected else self.policy.execution_horizon,
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


def _seed_lerobot(seed: int) -> None:
    try:
        from lerobot.utils.random_utils import set_seed
    except ImportError as exc:  # pragma: no cover - optional dependency boundary.
        raise RuntimeError("pi0-FAST requires LeRobot 0.6.0") from exc
    set_seed(int(seed))
