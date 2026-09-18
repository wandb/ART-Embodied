"""Lazy LeRobot pi0-FAST policy with exact action-token likelihoods."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any, Literal


def _sample_actions_fast_kv_cache_until_end(
    model: Any,
    images: Any,
    image_masks: Any,
    tokens: Any,
    masks: Any,
    *,
    max_decoding_steps: int,
    temperature: float,
    end_token_id: int,
    return_logprobs: bool = False,
) -> Any:
    """Generate FAST tokens, removing ended rows and their KV cache immediately."""

    import torch

    batch_size = int(tokens.shape[0])
    device = tokens.device
    lm_head = model.paligemma_with_expert.paligemma.lm_head
    active_rows = torch.arange(batch_size, device=device)
    sampled_logprobs: list[Any] = []

    if return_logprobs and temperature <= 0.0:
        raise ValueError("Sampled token logprobs require temperature > 0")

    def sample_next(logits: Any) -> Any:
        if temperature > 0:
            log_probabilities = torch.log_softmax(
                logits[:, -1].float() / temperature, dim=-1
            )
            probabilities = log_probabilities.exp()
            token = torch.multinomial(probabilities, num_samples=1)
            if return_logprobs:
                row_logprobs = log_probabilities.new_zeros(batch_size)
                row_logprobs[active_rows] = log_probabilities.gather(-1, token).squeeze(
                    -1
                )
                sampled_logprobs.append(row_logprobs)
            return token
        return torch.argmax(logits[:, -1], dim=-1, keepdim=True)

    def result(generated: Any) -> Any:
        if not return_logprobs:
            return generated
        return generated, torch.stack(sampled_logprobs, dim=1)

    bos_token = torch.full(
        (batch_size, 1),
        model._paligemma_tokenizer.bos_token_id,
        dtype=torch.long,
        device=device,
    )
    input_tokens = torch.cat([tokens, bos_token], dim=1)
    input_masks = torch.cat(
        [masks, torch.ones((batch_size, 1), dtype=torch.bool, device=device)],
        dim=1,
    )
    prefix_embs, prefix_pad_masks, prefix_att_masks, _, _ = model.embed_prefix_fast(
        images,
        image_masks,
        input_tokens,
        input_masks,
        fast_action_tokens=None,
        fast_action_masks=None,
    )
    language_dtype = model.paligemma_with_expert.paligemma.model.language_model.layers[
        0
    ].self_attn.q_proj.weight.dtype
    if language_dtype == torch.bfloat16:
        prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

    position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    attention_mask = model._prepare_attention_masks_4d(
        prefix_att_masks,
        dtype=prefix_embs.dtype,
    )
    (prefix_out, _), past_key_values = model.paligemma_with_expert.forward(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
        adarms_cond=[None, None],
    )

    logits = lm_head(prefix_out[:, -1:, :])
    next_token = sample_next(logits)

    generated = torch.zeros(
        (batch_size, max_decoding_steps),
        dtype=torch.long,
        device=device,
    )
    generated[:, 0] = next_token.squeeze(-1)
    current_pad_mask = prefix_pad_masks
    for step in range(1, max_decoding_steps):
        keep = torch.nonzero(next_token.squeeze(-1) != end_token_id).flatten()
        if keep.numel() == 0:
            return result(generated[:, :step])
        if keep.numel() != active_rows.numel():
            # Ended rows have no future actions or likelihood terms. Continuing
            # them can enter out-of-domain text and even overflow an unused row.
            past_key_values.batch_select_indices(keep)
            active_rows = active_rows[keep]
            current_pad_mask = current_pad_mask[keep]
            next_token = next_token[keep]
        next_token_emb = model.paligemma_with_expert.embed_language_tokens(next_token)
        if prefix_embs.dtype == torch.bfloat16:
            next_token_emb = next_token_emb.to(dtype=torch.bfloat16)

        current_pad_mask = torch.cat(
            [
                current_pad_mask,
                torch.ones(
                    (active_rows.numel(), 1),
                    dtype=torch.bool,
                    device=device,
                ),
            ],
            dim=1,
        )
        current_position_ids = (
            torch.sum(current_pad_mask, dim=1, keepdim=True) - 1
        ).long()
        step_attention_mask = model._prepare_attention_masks_4d(
            current_pad_mask.unsqueeze(1),
            dtype=next_token_emb.dtype,
        )
        (step_out, _), past_key_values = model.paligemma_with_expert.forward(
            attention_mask=step_attention_mask,
            position_ids=current_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[next_token_emb, None],
            use_cache=True,
            adarms_cond=[None, None],
        )
        logits = lm_head(step_out[:, -1:, :])
        next_token = sample_next(logits)
        generated[active_rows, step] = next_token.squeeze(-1)

    return result(generated)


def _sample_actions_fast_no_cache_until_end(
    model: Any,
    images: Any,
    image_masks: Any,
    tokens: Any,
    masks: Any,
    *,
    max_decoding_steps: int,
    temperature: float,
    end_token_id: int,
    return_logprobs: bool = False,
) -> Any:
    """Generate FAST tokens with the native full-sequence attention pattern."""

    import torch

    batch_size = int(tokens.shape[0])
    device = tokens.device
    lm_head = model.paligemma_with_expert.paligemma.lm_head
    if return_logprobs and temperature <= 0.0:
        raise ValueError("Sampled token logprobs require temperature > 0")
    bos_token = torch.full(
        (batch_size, 1),
        model._paligemma_tokenizer.bos_token_id,
        dtype=torch.long,
        device=device,
    )
    input_tokens = torch.cat([tokens, bos_token], dim=1)
    input_masks = torch.cat(
        [masks, torch.ones((batch_size, 1), dtype=torch.bool, device=device)],
        dim=1,
    )
    embeddings, pad_masks, attention_masks, _, _ = model.embed_prefix_fast(
        images,
        image_masks,
        input_tokens,
        input_masks,
        fast_action_tokens=None,
        fast_action_masks=None,
    )
    language_dtype = model.paligemma_with_expert.paligemma.model.language_model.layers[
        0
    ].self_attn.q_proj.weight.dtype
    if language_dtype == torch.bfloat16:
        embeddings = embeddings.to(dtype=torch.bfloat16)

    generated = torch.zeros(
        (batch_size, max_decoding_steps), dtype=torch.long, device=device
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    sampled_logprobs: list[Any] = []
    for step in range(max_decoding_steps):
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        attention_4d = model._prepare_attention_masks_4d(
            attention_masks, dtype=embeddings.dtype
        )
        (hidden, _), _ = model.paligemma_with_expert.forward(
            attention_mask=attention_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[embeddings, None],
            use_cache=False,
            adarms_cond=[None, None],
        )
        logits = lm_head(hidden[:, -1:, :])
        if temperature > 0:
            log_probabilities = torch.log_softmax(
                logits[:, -1].float() / temperature, dim=-1
            )
            probabilities = log_probabilities.exp()
            next_token = torch.multinomial(probabilities, num_samples=1)
            if return_logprobs:
                sampled_logprobs.append(
                    log_probabilities.gather(-1, next_token).squeeze(-1)
                )
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
        generated[:, step] = next_token.squeeze(-1)
        finished |= next_token.squeeze(-1) == end_token_id
        if bool(finished.all()):
            width = step + 1
            if not return_logprobs:
                return generated[:, :width]
            return generated[:, :width], torch.stack(sampled_logprobs, dim=1)

        next_embedding = model.paligemma_with_expert.embed_language_tokens(next_token)
        if embeddings.dtype == torch.bfloat16:
            next_embedding = next_embedding.to(dtype=torch.bfloat16)
        embeddings = torch.cat([embeddings, next_embedding], dim=1)
        pad_masks = torch.cat(
            [
                pad_masks,
                torch.ones((batch_size, 1), dtype=torch.bool, device=device),
            ],
            dim=1,
        )
        old_length = int(attention_masks.shape[1])
        grown = torch.zeros(
            (batch_size, old_length + 1, old_length + 1),
            dtype=torch.bool,
            device=device,
        )
        grown[:, :old_length, :old_length] = attention_masks
        grown[:, -1, :] = pad_masks
        attention_masks = grown

    if not return_logprobs:
        return generated
    return generated, torch.stack(sampled_logprobs, dim=1)


class PI0FastPolicy:
    """Own LeRobot pi0-FAST, its serialized processors, and GRPO evidence."""

    family = "pi0_fast"

    def __init__(
        self,
        *,
        model_id: str,
        revision: str | None,
        device: str,
        execution_horizon: int,
        action_dim: int,
        max_decoding_steps: int,
        action_tokenizer_revision: str,
        observation_key_map: Mapping[str, str],
        strict_weights: bool,
        compile_model: bool,
        gradient_checkpointing: bool,
        use_kv_cache: bool,
        model_compute_dtype: Literal[
            "checkpoint", "float32", "fp16_residual"
        ] = "checkpoint",
        training_loss_scale: float = 1.0,
        training_logprob_mode: Literal["kv", "full_sequence"] = "kv",
        rl_token_scope: Literal["generated_sequence", "fast_payload"] = (
            "generated_sequence"
        ),
        stop_on_action_end: bool = True,
        load_on_init: bool = False,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.execution_horizon = int(execution_horizon)
        self.action_dim = int(action_dim)
        self.max_decoding_steps = int(max_decoding_steps)
        self.action_tokenizer_revision = action_tokenizer_revision
        self.observation_key_map = dict(observation_key_map)
        self.strict_weights = bool(strict_weights)
        self.compile_model = bool(compile_model)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.use_kv_cache = bool(use_kv_cache)
        self.model_compute_dtype = model_compute_dtype
        self.training_loss_scale = float(training_loss_scale)
        if training_logprob_mode not in ("kv", "full_sequence"):
            raise ValueError("Unsupported pi0-FAST training_logprob_mode")
        self.training_logprob_mode = training_logprob_mode
        self.rl_token_scope = rl_token_scope
        self.stop_on_action_end = bool(stop_on_action_end)
        self.policy: Any | None = None
        self.preprocessor: Any | None = None
        self.postprocessor: Any | None = None
        self.trainable_report: dict[str, Any] | None = None
        if load_on_init:
            self.load()

    @property
    def model(self) -> Any:
        self._require_loaded()
        assert self.policy is not None
        return self.policy.model

    @model.setter
    def model(self, value: Any) -> None:
        self._require_loaded()
        assert self.policy is not None
        self.policy.model = value

    @property
    def config(self) -> Any:
        self._require_loaded()
        assert self.policy is not None
        return self.policy.config

    def load(self) -> None:
        if self.policy is not None:
            return
        from .transformers_compat import ensure_art_mask_patch_compatibility

        ensure_art_mask_patch_compatibility()
        try:
            from lerobot.configs import PreTrainedConfig
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.pi0_fast.modeling_pi0_fast import (
                PI0FastPolicy as NativePI0FastPolicy,
            )
        except ImportError as exc:  # pragma: no cover - optional runtime boundary.
            raise RuntimeError(
                "pi0-FAST requires LeRobot 0.6.0; install art-embodied[pi0-fast]"
            ) from exc

        config = PreTrainedConfig.from_pretrained(
            self.model_id,
            revision=self.revision,
        )
        if getattr(config, "type", None) != "pi0_fast":
            raise ValueError(f"Checkpoint {self.model_id!r} is not pi0-FAST")
        config.device = self.device
        config.compile_model = self.compile_model
        config.gradient_checkpointing = self.gradient_checkpointing
        config.use_kv_cache = self.use_kv_cache
        config.max_decoding_steps = self.max_decoding_steps
        from huggingface_hub import snapshot_download

        action_tokenizer_path = snapshot_download(
            repo_id=str(config.action_tokenizer_name),
            revision=self.action_tokenizer_revision,
        )
        config.action_tokenizer_name = action_tokenizer_path
        resolved_weights: str | None = None
        if self.strict_weights:
            from transformers.utils import cached_file

            resolved_weights = cached_file(
                self.model_id,
                "model.safetensors",
                revision=self.revision,
            )
            if resolved_weights is None:
                raise RuntimeError(
                    f"pi0-FAST checkpoint weights were not found: {self.model_id!r}"
                )
        policy = NativePI0FastPolicy.from_pretrained(
            self.model_id,
            config=config,
            revision=self.revision,
            strict=self.strict_weights,
        )
        if resolved_weights is not None:
            _assert_pretrained_weights_loaded(policy, resolved_weights)
        chunk_size = int(policy.config.chunk_size)
        if self.execution_horizon > chunk_size:
            raise ValueError("pi0-FAST execution_horizon exceeds checkpoint chunk_size")
        output_features = getattr(policy.config, "output_features", {})
        action_feature = output_features.get("action")
        feature_shape = getattr(action_feature, "shape", None)
        if not feature_shape or self.action_dim > int(feature_shape[0]):
            raise ValueError("pi0-FAST action_dim exceeds checkpoint action feature")
        if self.model_compute_dtype == "float32":
            policy.model.float()

        self.policy = policy
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=self.model_id,
            pretrained_revision=self.revision,
            preprocessor_overrides={
                "action_tokenizer_processor": {
                    "action_tokenizer_name": action_tokenizer_path,
                },
                "device_processor": {"device": self.device},
            },
        )

    def reset(self) -> None:
        self._require_loaded()
        for component in (self.policy, self.preprocessor, self.postprocessor):
            reset = getattr(component, "reset", None)
            if callable(reset):
                reset()

    def prepare_observation(
        self,
        observation: Mapping[str, Any],
        *,
        task: str | None,
        robot_type: str | None,
    ) -> dict[str, Any]:
        """Apply the checkpoint's feature names before native preprocessing."""

        self._require_loaded()
        import numpy as np
        import torch

        try:
            from lerobot.policies import prepare_observation_for_inference
        except ImportError as exc:  # pragma: no cover - optional dependency.
            raise RuntimeError("pi0-FAST rollout requires LeRobot 0.6.0") from exc

        mapped = {
            self.observation_key_map.get(key, key): (
                np.ascontiguousarray(value) if isinstance(value, np.ndarray) else value
            )
            for key, value in observation.items()
        }
        state_key = "observation.state"
        if state_key in mapped:
            state = np.asarray(mapped[state_key])
            expected = int(self.config.max_state_dim)
            if state.ndim != 1 or state.shape[0] > expected:
                raise ValueError(
                    f"pi0-FAST state must have at most {expected} values; "
                    f"got {state.shape}"
                )
            mapped[state_key] = np.ascontiguousarray(
                state.astype(np.float32, copy=False)
            )
        return prepare_observation_for_inference(
            mapped,
            torch.device(self.device),
            task,
            robot_type,
        )

    def preprocess_observation(
        self,
        observation: Mapping[str, Any],
        *,
        task: str | None,
        robot_type: str | None = "panda",
    ) -> dict[str, Any]:
        self._require_loaded()
        self.activate_task_adapter(task)
        assert self.preprocessor is not None
        prepared = self.prepare_observation(
            observation,
            task=task,
            robot_type=robot_type,
        )
        return self.preprocessor(prepared)

    def activate_task_adapter(self, task: str | None) -> str | None:
        """Route a task-owned partition before inference or token rescoring."""

        if getattr(self, "_art_partition_forward_routing", None) != "task_owned":
            return None
        mapping = getattr(self, "_art_task_adapter_map", None)
        if not isinstance(mapping, Mapping):
            raise RuntimeError("Task-owned LoRA routing has no task-adapter map")
        try:
            adapter = str(mapping[str(task)])
        except KeyError as exc:
            raise ValueError(f"No LoRA adapter is assigned to task {task!r}") from exc
        self.model.base_model.set_adapter(adapter)
        return adapter

    def sample_action_tokens(
        self,
        processed_batch: Mapping[str, Any],
        *,
        temperature: float,
    ) -> Any:
        """Sample with LeRobot's deployment sampler exactly once."""

        self._require_loaded()
        assert self.policy is not None
        images, image_masks = self.policy._preprocess_images(dict(processed_batch))
        language_tokens_key, language_attention_mask_key = _language_batch_keys()
        tokens = processed_batch[language_tokens_key]
        masks = processed_batch[language_attention_mask_key]
        if self.use_kv_cache and self.stop_on_action_end:
            tokenizer = self.policy._paligemma_tokenizer
            return _sample_actions_fast_kv_cache_until_end(
                self.model,
                images,
                image_masks,
                tokens,
                masks,
                max_decoding_steps=self.max_decoding_steps,
                temperature=float(temperature),
                end_token_id=int(tokenizer.convert_tokens_to_ids("|")),
            )
        sampler = self.model.sample_actions_fast_kv_cache
        if not self.use_kv_cache:
            sampler = self.model.sample_actions_fast
        return sampler(
            images,
            image_masks,
            tokens,
            masks,
            max_decoding_steps=self.max_decoding_steps,
            temperature=float(temperature),
        )

    def sample_action_tokens_with_logprobs(
        self,
        processed_batch: Mapping[str, Any],
        *,
        temperature: float,
    ) -> tuple[Any, Any]:
        """Sample tokens and retain the probabilities used by the KV sampler."""

        self._require_loaded()
        assert self.policy is not None
        if not self.stop_on_action_end:
            raise ValueError("Sampler-native logprobs require action-end stopping")
        if temperature <= 0.0:
            raise ValueError("Sampled token logprobs require temperature > 0")
        images, image_masks = self.policy._preprocess_images(dict(processed_batch))
        language_tokens_key, language_attention_mask_key = _language_batch_keys()
        tokens = processed_batch[language_tokens_key]
        masks = processed_batch[language_attention_mask_key]
        tokenizer = self.policy._paligemma_tokenizer
        sampler = (
            _sample_actions_fast_kv_cache_until_end
            if self.use_kv_cache
            else _sample_actions_fast_no_cache_until_end
        )
        sampled = sampler(
            self.model,
            images,
            image_masks,
            tokens,
            masks,
            max_decoding_steps=self.max_decoding_steps,
            temperature=float(temperature),
            end_token_id=int(tokenizer.convert_tokens_to_ids("|")),
            return_logprobs=True,
        )
        if not isinstance(sampled, tuple):
            raise RuntimeError("pi0-FAST sampler did not return token probabilities")
        return sampled

    def decode_action_tokens(self, tokens: Any) -> Any:
        self._require_loaded()
        assert self.policy is not None
        normalized = self.policy.detokenize_actions(
            tokens,
            action_horizon=int(self.policy.config.chunk_size),
            action_dim=self.action_dim,
        )
        assert self.postprocessor is not None
        return self.postprocessor(normalized)

    def prepare_generated_action_tokens(
        self,
        tokens: Any,
        *,
        native_decoder: bool = False,
    ) -> tuple[list[list[int]], list[bool], list[int], list[list[bool]]]:
        """Keep only generated tokens that determine the decoded action."""

        self._require_loaded()
        assert self.policy is not None
        tokenizer = self.policy._paligemma_tokenizer
        prefix_ids = [
            int(token)
            for token in tokenizer.encode("Action: ", add_special_tokens=False)
        ]
        end_token_id = int(tokenizer.convert_tokens_to_ids("|"))
        action_vocab_size = int(self.policy.action_tokenizer.vocab_size)
        action_token_max_id = (
            int(tokenizer.vocab_size) - 1 - int(self.policy.config.fast_skip_tokens)
        )
        action_token_min_id = action_token_max_id - action_vocab_size + 1
        return _objective_action_token_rows(
            tokens.detach().cpu().tolist(),
            prefix_ids=prefix_ids,
            end_token_id=end_token_id,
            action_token_min_id=action_token_min_id,
            action_token_max_id=action_token_max_id,
            trim_invalid_prefix=not native_decoder,
        )

    def decode_action_tokens_native(
        self, tokens: Any
    ) -> tuple[list[Any], list[str | None]]:
        """Use LeRobot's decoding contract, but reject its silent error fallback.

        LeRobot 0.6.0 reports caught FAST decoding errors through root logging
        before returning zeros. Observe only that decoder's warnings on this
        thread; do not suppress logs or mistake legitimate zero actions for errors.
        Token cleanup and relaxed DCT decoding remain entirely upstream-owned.
        """
        self._require_loaded()
        assert self.policy is not None
        assert self.postprocessor is not None
        import logging
        import threading

        import torch

        decoder = self.policy.decode_actions_with_fast
        code = decoder.__func__.__code__
        thread = threading.get_ident()
        warnings: list[str] = []

        class Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                if (
                    record.thread == thread
                    and record.pathname == code.co_filename
                    and record.funcName == code.co_name
                ):
                    warnings.append(record.getMessage())

        handler = Capture(level=logging.WARNING)
        logger = logging.getLogger()
        if not logger.isEnabledFor(logging.WARNING):
            raise RuntimeError("Native FAST error detection requires WARNING logging")
        logger.addHandler(handler)
        chunks, errors = [], []
        try:
            for row in tokens:
                warnings.clear()
                try:
                    normalized = self.policy.detokenize_actions(
                        row.unsqueeze(0),
                        action_horizon=int(self.policy.config.chunk_size),
                        action_dim=self.action_dim,
                    )
                    if warnings:
                        raise ValueError(
                            "Native FAST decoder reported a decoding error"
                        )
                    expected = (1, int(self.policy.config.chunk_size), self.action_dim)
                    if tuple(normalized.shape) != expected or not bool(
                        torch.isfinite(normalized).all()
                    ):
                        raise ValueError("Native FAST decoder returned invalid actions")
                    action = self.postprocessor(normalized)
                    if tuple(action.shape) != expected or not bool(
                        torch.isfinite(action).all()
                    ):
                        raise ValueError("FAST postprocessor returned invalid actions")
                    chunks.append(action[0])
                    errors.append(None)
                except (ValueError, AssertionError, OverflowError) as exc:
                    chunks.append(
                        torch.empty((0, self.action_dim), device=tokens.device)
                    )
                    errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            logger.removeHandler(handler)
        return chunks, errors

    def decode_action_tokens_safe(
        self,
        tokens: Any,
        *,
        grammar_valid: Sequence[bool],
        invalid_action: Sequence[float] | None = None,
    ) -> Any:
        """Decode FAST rows; malformed rows require an explicit physical action.

        There is no robot-independent no-op, especially for absolute controls
        and grippers. The caller must supply an environment-appropriate action
        in postprocessed units or malformed generation is rejected.
        """

        self._require_loaded()
        assert self.policy is not None
        import torch

        if len(grammar_valid) != int(tokens.shape[0]):
            raise ValueError("pi0-FAST grammar flags do not match token batch")
        valid_indices = [index for index, valid in enumerate(grammar_valid) if valid]
        invalid_indices = [
            index for index, valid in enumerate(grammar_valid) if not valid
        ]
        fallback = None
        if invalid_indices:
            if invalid_action is None:
                raise ValueError(
                    "Malformed FAST generation requires an explicit physical "
                    "invalid_action; normalized zero is not a safe no-op"
                )
            fallback = torch.as_tensor(
                invalid_action, dtype=torch.float32, device=tokens.device
            )
            if fallback.shape != (self.action_dim,) or not bool(
                torch.isfinite(fallback).all()
            ):
                raise ValueError("invalid_action must be a finite action_dim vector")
        normalized = torch.zeros(
            (
                int(tokens.shape[0]),
                int(self.policy.config.chunk_size),
                self.action_dim,
            ),
            dtype=torch.float32,
            device=tokens.device,
        )
        if valid_indices:
            decoded = self.policy.detokenize_actions(
                tokens[valid_indices],
                action_horizon=int(self.policy.config.chunk_size),
                action_dim=self.action_dim,
            )
            normalized[valid_indices] = decoded
        assert self.postprocessor is not None
        actions = self.postprocessor(normalized)
        if fallback is not None:
            actions[invalid_indices] = fallback.to(
                device=actions.device, dtype=actions.dtype
            )
        return actions

    def processed_action_token_logprobs(
        self,
        processed_batch: Mapping[str, Any],
        token_rows: Sequence[Sequence[int]],
        *,
        temperature: float,
    ) -> list[Any]:
        """Return one differentiable likelihood row per generated token row."""

        self._require_loaded()
        assert self.policy is not None
        if temperature <= 0.0:
            raise ValueError("Categorical token logprobs require temperature > 0")
        # Only teacher-forced scoring changes; sampling keeps its KV-cache path.
        scorer = (
            _pi0_fast_token_logprobs
            if self.training_logprob_mode == "full_sequence"
            else _pi0_fast_kv_token_logprobs
        )
        return scorer(
            native_policy=self.policy,
            processed_batch=processed_batch,
            token_rows=token_rows,
            temperature=float(temperature),
        )

    def processed_action_token_logprobs_full_sequence(
        self,
        processed_batch: Mapping[str, Any],
        token_rows: Sequence[Sequence[int]],
        *,
        temperature: float,
    ) -> list[Any]:
        """Explicit full-sequence scorer for comparison with the KV-cache path."""

        self._require_loaded()
        assert self.policy is not None
        return _pi0_fast_token_logprobs(
            native_policy=self.policy,
            processed_batch=processed_batch,
            token_rows=token_rows,
            temperature=float(temperature),
        )

    def processed_action_token_logprobs_kv(
        self,
        processed_batch: Mapping[str, Any],
        token_rows: Sequence[Sequence[int]],
        *,
        temperature: float,
    ) -> list[Any]:
        """Return likelihoods from the sampler-equivalent KV-cache path."""

        self._require_loaded()
        assert self.policy is not None
        return _pi0_fast_kv_token_logprobs(
            native_policy=self.policy,
            processed_batch=processed_batch,
            token_rows=token_rows,
            temperature=float(temperature),
        )

    def action_token_logprobs(self, examples: Sequence[Any]) -> list[Any]:
        """Teacher-force trajectory tokens against their source observations."""

        if not examples:
            return []
        processed_rows: list[Mapping[str, Any]] = []
        token_rows: list[list[int]] = []
        temperatures: list[float] = []
        for example in examples:
            observation = getattr(example, "observation", None)
            value = getattr(observation, "value", None)
            if not isinstance(value, Mapping):
                raise ValueError("pi0-FAST rescoring requires mapping observations")
            prompt = getattr(example, "prompt", None)
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("pi0-FAST rescoring requires the rollout prompt")
            tokens = [int(token) for token in getattr(example, "tokens", ())]
            if not tokens:
                raise ValueError("pi0-FAST rescoring requires generated tokens")
            action_metadata = getattr(example, "metadata", {}).get(
                "action_metadata", {}
            )
            temperature = action_metadata.get("sampling_temperature")
            if temperature is None:
                raise ValueError(
                    "pi0-FAST action metadata must record sampling_temperature"
                )
            processed_rows.append(
                self.preprocess_observation(value, task=prompt, robot_type="panda")
            )
            token_rows.append(tokens)
            temperatures.append(float(temperature))
        if any(value != temperatures[0] for value in temperatures[1:]):
            raise ValueError("pi0-FAST scorer batches require one sampling temperature")
        return self.processed_action_token_logprobs(
            _concatenate_processed_batches(processed_rows),
            token_rows,
            temperature=temperatures[0],
        )

    def parameters(self, *args: Any, **kwargs: Any) -> Any:
        self._require_loaded()
        assert self.policy is not None
        return self.policy.parameters(*args, **kwargs)

    def named_parameters(self, *args: Any, **kwargs: Any) -> Any:
        self._require_loaded()
        assert self.policy is not None
        return self.policy.named_parameters(*args, **kwargs)

    def train(self, mode: bool = True) -> "PI0FastPolicy":
        self._require_loaded()
        assert self.policy is not None
        self.policy.train(mode)
        return self

    def eval(self) -> "PI0FastPolicy":
        return self.train(False)

    def to(self, device: str) -> "PI0FastPolicy":
        self._require_loaded()
        assert self.policy is not None
        self.device = str(device)
        self.policy.to(device)
        self.policy.config.device = str(device)
        if str(device) == "cpu":
            import torch

            torch.cuda.empty_cache()
        return self

    def save_checkpoint(self, path: str | Path) -> None:
        """Publish trainable state without duplicating frozen base weights."""

        self._require_loaded()
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "peft_config"):
            # The native PI0Fast module is assembled from a pinned Hub snapshot
            # rather than loaded through Transformers' PreTrainedModel API, so
            # PEFT cannot infer this provenance itself. Persist it explicitly;
            # task-partition composition rejects an unbound adapter.
            for adapter_config in self.model.peft_config.values():
                adapter_config.base_model_name_or_path = self.model_id
            self.model.save_pretrained(destination)
            snapshot_format = "peft_adapter"
            adapter_names = sorted(str(name) for name in self.model.peft_config)
            active_adapters = [
                str(name) for name in getattr(self.model, "active_adapters", ())
            ]
        else:
            from safetensors.torch import save_model

            save_model(self.model, destination / "model.safetensors")
            snapshot_format = "full_model"
            adapter_names = []
            active_adapters = []
        (destination / "art_embodied_pi0_fast_snapshot.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "format": snapshot_format,
                    "family": self.family,
                    "model_id": self.model_id,
                    "revision": self.revision,
                    "model_compute_dtype": self.model_compute_dtype,
                    "training_loss_scale": self.training_loss_scale,
                    "adapter_names": adapter_names,
                    "active_adapters": active_adapters,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def load_checkpoint(self, checkpoint: Mapping[str, Any] | str | Path) -> None:
        self._require_loaded()
        raw_path = (
            checkpoint.get("path") if isinstance(checkpoint, Mapping) else checkpoint
        )
        source = Path(str(raw_path)).expanduser()
        metadata_path = source / "art_embodied_pi0_fast_snapshot.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("family") != self.family:
            raise ValueError("pi0-FAST snapshot family mismatch")
        saved_precision = metadata.get("model_compute_dtype", "checkpoint")
        if "fp16_residual" in (saved_precision, self.model_compute_dtype):
            if (
                saved_precision != self.model_compute_dtype
                or metadata.get("training_loss_scale") != self.training_loss_scale
            ):
                raise ValueError("pi0-FAST snapshot precision/loss scale mismatch")
        if (
            metadata.get("model_id") != self.model_id
            or metadata.get("revision") != self.revision
        ):
            raise ValueError("pi0-FAST snapshot base model contract mismatch")
        if metadata.get("format") == "peft_adapter":
            if not hasattr(self.model, "peft_config"):
                raise RuntimeError(
                    "pi0-FAST PEFT snapshot requires LoRA to be attached"
                )
            from peft.utils.save_and_load import (
                get_peft_model_state_dict,
                load_peft_weights,
                set_peft_model_state_dict,
            )

            adapter_names = metadata.get("adapter_names")
            if not isinstance(adapter_names, list) or not adapter_names:
                # Backward compatibility for snapshots written before adapter
                # names were explicit.
                adapter_names = ["default"]
            configured = set(str(name) for name in self.model.peft_config)
            if set(adapter_names) != configured:
                raise ValueError(
                    "pi0-FAST snapshot adapter set mismatch: "
                    f"snapshot={sorted(adapter_names)}, configured={sorted(configured)}"
                )
            validated_states = []
            for adapter_name in adapter_names:
                adapter_path = (
                    source if adapter_name == "default" else source / adapter_name
                )
                state = load_peft_weights(str(adapter_path), device=self.device)
                expected = get_peft_model_state_dict(
                    self.model, adapter_name=str(adapter_name)
                )
                missing = sorted(expected.keys() - state.keys())
                extra = sorted(state.keys() - expected.keys())
                mismatched = sorted(
                    key
                    for key in expected.keys() & state.keys()
                    if expected[key].shape != state[key].shape
                )
                if missing or extra or mismatched:
                    raise RuntimeError(
                        f"pi0-FAST adapter {adapter_name!r} state mismatch: "
                        f"missing={missing[:20]}, unexpected={extra[:20]}, "
                        f"shape_mismatches={mismatched[:20]}"
                    )
                validated_states.append((adapter_name, state))
            # Validate every adapter before mutating any of the live weights.
            for adapter_name, state in validated_states:
                result = set_peft_model_state_dict(
                    self.model,
                    state,
                    adapter_name=str(adapter_name),
                )
                unexpected = list(getattr(result, "unexpected_keys", ()) or ())
                if unexpected:
                    raise RuntimeError(
                        f"pi0-FAST adapter {adapter_name!r} has unexpected keys: "
                        f"{unexpected[:20]}"
                    )
            active_adapters = metadata.get("active_adapters") or adapter_names
            active_set = set(str(name) for name in active_adapters)
            routing = getattr(self, "_art_partition_forward_routing", None)
            if routing == "task_owned":
                if len(active_set) != 1 or not active_set.issubset(configured):
                    raise ValueError(
                        "pi0-FAST task-owned snapshot must have one configured "
                        "active adapter"
                    )
            elif active_set != configured:
                raise ValueError("pi0-FAST snapshot active adapter set mismatch")
            self.model.base_model.set_adapter([str(name) for name in active_adapters])
            if routing == "task_owned":
                from art_embodied.lora_rank_partition import (
                    enable_partitioned_adapter_gradients,
                )

                enable_partitioned_adapter_gradients(self, adapter_names)
        elif metadata.get("format") == "full_model":
            from safetensors.torch import load_model

            load_model(self.model, source / "model.safetensors", strict=True)
        else:
            raise ValueError("Unsupported pi0-FAST snapshot format")

    def _require_loaded(self) -> None:
        if self.policy is None:
            raise RuntimeError("pi0-FAST policy is not loaded")


def _pi0_fast_token_logprobs(
    *,
    native_policy: Any,
    processed_batch: Mapping[str, Any],
    token_rows: Sequence[Sequence[int]],
    temperature: float,
) -> list[Any]:
    """Score tokens under the exact autoregressive rollout attention pattern."""

    import torch

    if not token_rows:
        return []
    batch_size = len(token_rows)
    language_tokens_key, language_attention_mask_key = _language_batch_keys()
    language_tokens = processed_batch[language_tokens_key]
    language_masks = processed_batch[language_attention_mask_key]
    if int(language_tokens.shape[0]) != batch_size:
        raise ValueError("pi0-FAST token rows do not match processor batch")

    model = native_policy.model
    device = language_tokens.device
    bos_id = int(native_policy._paligemma_tokenizer.bos_token_id)
    max_generated = max(len(row) for row in token_rows)
    if max_generated == 0 or any(not row for row in token_rows):
        raise ValueError("pi0-FAST token rows cannot be empty")
    generated_tokens = torch.zeros(
        (batch_size, max_generated), dtype=torch.long, device=device
    )
    generated_masks = torch.zeros_like(generated_tokens, dtype=torch.bool)
    for index, row in enumerate(token_rows):
        generated_tokens[index, : len(row)] = torch.as_tensor(
            row, dtype=torch.long, device=device
        )
        generated_masks[index, : len(row)] = True

    images, image_masks = native_policy._preprocess_images(dict(processed_batch))
    bos_tokens = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    sampler_tokens = torch.cat([language_tokens, bos_tokens], dim=1)
    sampler_masks = torch.cat(
        [
            language_masks,
            torch.ones((batch_size, 1), dtype=torch.bool, device=device),
        ],
        dim=1,
    )
    prefix_embeddings, prefix_pad_masks, prefix_attention_masks, _, _ = (
        model.embed_prefix_fast(
            images,
            image_masks,
            sampler_tokens,
            sampler_masks,
            fast_action_tokens=None,
            fast_action_masks=None,
        )
    )
    q_proj = model.paligemma_with_expert.paligemma.model.language_model.layers[
        0
    ].self_attn.q_proj
    if q_proj.weight.dtype == torch.bfloat16:
        prefix_embeddings = prefix_embeddings.to(dtype=torch.bfloat16)

    # The prefix predicts token 0. Each generated input token predicts the next
    # token, so the final generated token is a target but never an input.
    previous_tokens = generated_tokens[:, :-1]
    previous_masks = generated_masks[:, :-1]
    if previous_tokens.shape[1] > 0:
        previous_embeddings = (
            model.paligemma_with_expert.embed_language_tokens(previous_tokens)
        ).to(dtype=prefix_embeddings.dtype)
        embeddings = torch.cat([prefix_embeddings, previous_embeddings], dim=1)
        pad_masks = torch.cat([prefix_pad_masks, previous_masks], dim=1)
    else:
        embeddings = prefix_embeddings
        pad_masks = prefix_pad_masks

    prefix_length = int(prefix_embeddings.shape[1])
    sequence_length = int(embeddings.shape[1])
    attention_masks = torch.zeros(
        (batch_size, sequence_length, sequence_length),
        dtype=torch.bool,
        device=device,
    )
    attention_masks[:, :prefix_length, :prefix_length] = prefix_attention_masks
    for offset in range(previous_tokens.shape[1]):
        row_index = prefix_length + offset
        valid_query = previous_masks[:, offset].view(batch_size, 1)
        attention_masks[:, row_index, : row_index + 1] = (
            pad_masks[:, : row_index + 1] & valid_query
        )

    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    attention_4d = model._prepare_attention_masks_4d(
        attention_masks,
        dtype=embeddings.dtype,
    )
    (hidden_states, _), _ = model.paligemma_with_expert.forward(
        attention_mask=attention_4d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[embeddings, None],
        use_cache=False,
        adarms_cond=[None, None],
    )
    prediction_start = prefix_length - 1
    prediction_hidden = hidden_states[
        :, prediction_start : prediction_start + max_generated, :
    ]
    logits = model.paligemma_with_expert.paligemma.lm_head(prediction_hidden)
    gathered = (
        torch.log_softmax(logits.float() / float(temperature), dim=-1)
        .gather(
            -1,
            generated_tokens.unsqueeze(-1),
        )
        .squeeze(-1)
    )
    return [gathered[index, : len(row)] for index, row in enumerate(token_rows)]


def _pi0_fast_kv_token_logprobs(
    *,
    native_policy: Any,
    processed_batch: Mapping[str, Any],
    token_rows: Sequence[Sequence[int]],
    temperature: float,
) -> list[Any]:
    """Score target tokens through the same KV-cache path used for sampling."""

    import torch

    if temperature <= 0.0:
        raise ValueError("Categorical token logprobs require temperature > 0")
    if not token_rows or any(not row for row in token_rows):
        raise ValueError("pi0-FAST token rows cannot be empty")
    batch_size = len(token_rows)
    language_tokens_key, language_attention_mask_key = _language_batch_keys()
    language_tokens = processed_batch[language_tokens_key]
    language_masks = processed_batch[language_attention_mask_key]
    if int(language_tokens.shape[0]) != batch_size:
        raise ValueError("pi0-FAST token rows do not match processor batch")

    model = native_policy.model
    device = language_tokens.device
    max_generated = max(len(row) for row in token_rows)
    generated_tokens = torch.zeros(
        (batch_size, max_generated), dtype=torch.long, device=device
    )
    generated_masks = torch.zeros_like(generated_tokens, dtype=torch.bool)
    for index, row in enumerate(token_rows):
        generated_tokens[index, : len(row)] = torch.as_tensor(
            row, dtype=torch.long, device=device
        )
        generated_masks[index, : len(row)] = True

    images, image_masks = native_policy._preprocess_images(dict(processed_batch))
    bos_tokens = torch.full(
        (batch_size, 1),
        int(native_policy._paligemma_tokenizer.bos_token_id),
        dtype=torch.long,
        device=device,
    )
    sampler_tokens = torch.cat([language_tokens, bos_tokens], dim=1)
    sampler_masks = torch.cat(
        [
            language_masks,
            torch.ones((batch_size, 1), dtype=torch.bool, device=device),
        ],
        dim=1,
    )
    prefix_embeddings, prefix_pad_masks, prefix_attention_masks, _, _ = (
        model.embed_prefix_fast(
            images,
            image_masks,
            sampler_tokens,
            sampler_masks,
            fast_action_tokens=None,
            fast_action_masks=None,
        )
    )
    q_proj = model.paligemma_with_expert.paligemma.model.language_model.layers[
        0
    ].self_attn.q_proj
    if q_proj.weight.dtype == torch.bfloat16:
        prefix_embeddings = prefix_embeddings.to(dtype=torch.bfloat16)
    position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    attention_4d = model._prepare_attention_masks_4d(
        prefix_attention_masks, dtype=prefix_embeddings.dtype
    )
    (hidden, _), past_key_values = model.paligemma_with_expert.forward(
        attention_mask=attention_4d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embeddings, None],
        use_cache=True,
        adarms_cond=[None, None],
    )
    lm_head = model.paligemma_with_expert.paligemma.lm_head
    current_pad_mask = prefix_pad_masks
    gathered_steps: list[Any] = []
    for target_index in range(max_generated):
        logits = lm_head(hidden[:, -1:, :])[:, -1]
        log_probabilities = torch.log_softmax(
            logits.float() / float(temperature), dim=-1
        )
        gathered_steps.append(
            log_probabilities.gather(
                -1, generated_tokens[:, target_index : target_index + 1]
            ).squeeze(-1)
        )
        if target_index == max_generated - 1:
            break

        next_token = generated_tokens[:, target_index : target_index + 1]
        next_embedding = model.paligemma_with_expert.embed_language_tokens(next_token)
        if prefix_embeddings.dtype == torch.bfloat16:
            next_embedding = next_embedding.to(dtype=torch.bfloat16)
        next_valid = generated_masks[:, target_index : target_index + 1]
        current_pad_mask = torch.cat([current_pad_mask, next_valid], dim=1)
        current_position_ids = (
            torch.sum(current_pad_mask, dim=1, keepdim=True) - 1
        ).long()
        step_attention = model._prepare_attention_masks_4d(
            current_pad_mask.unsqueeze(1), dtype=next_embedding.dtype
        )
        (hidden, _), past_key_values = model.paligemma_with_expert.forward(
            attention_mask=step_attention,
            position_ids=current_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[next_embedding, None],
            use_cache=True,
            adarms_cond=[None, None],
        )
    gathered = torch.stack(gathered_steps, dim=1)
    return [gathered[index, : len(row)] for index, row in enumerate(token_rows)]


def _objective_action_token_rows(
    token_rows: Sequence[Sequence[int]],
    *,
    prefix_ids: Sequence[int],
    end_token_id: int,
    action_token_min_id: int | None = None,
    action_token_max_id: int | None = None,
    trim_invalid_prefix: bool = True,
) -> tuple[list[list[int]], list[bool], list[int], list[list[bool]]]:
    """Trim sampled FAST rows to the tokens that affect the environment action."""

    if not prefix_ids:
        raise ValueError("pi0-FAST action prefix cannot be empty")
    if (action_token_min_id is None) != (action_token_max_id is None):
        raise ValueError("pi0-FAST action token bounds must be provided together")
    if (
        action_token_min_id is not None
        and action_token_max_id is not None
        and action_token_min_id > action_token_max_id
    ):
        raise ValueError("pi0-FAST action token bounds are reversed")
    objective_rows: list[list[int]] = []
    grammar_valid: list[bool] = []
    discarded_counts: list[int] = []
    payload_masks: list[list[bool]] = []
    for raw_row in token_rows:
        row = [int(token) for token in raw_row]
        prefix_valid = row[: len(prefix_ids)] == list(prefix_ids)
        if prefix_valid or not trim_invalid_prefix:
            try:
                objective_length = row.index(int(end_token_id)) + 1
            except ValueError:
                objective_length = len(row)
            payload_end = (
                objective_length - 1
                if objective_length and row[objective_length - 1] == end_token_id
                else objective_length
            )
            payload = row[len(prefix_ids) : payload_end]
            payload_valid = bool(payload)
            if action_token_min_id is not None and action_token_max_id is not None:
                payload_valid = payload_valid and all(
                    action_token_min_id <= token <= action_token_max_id
                    for token in payload
                )
        else:
            # Once the fixed action prefix diverges, the continuation cannot be
            # decoded. Penalize the causal prefix without assigning task credit
            # to the arbitrary text conditioned on it.
            objective_length = min(len(row), len(prefix_ids))
            payload_valid = False
        grammar_valid.append(prefix_valid and payload_valid)
        objective_row = row[:objective_length]
        objective_rows.append(objective_row)
        payload_masks.append(
            [
                prefix_valid
                and payload_valid
                and len(prefix_ids) <= index < payload_end
                for index in range(len(objective_row))
            ]
        )
        discarded_counts.append(len(row) - objective_length)
    return objective_rows, grammar_valid, discarded_counts, payload_masks


def _language_batch_keys() -> tuple[str, str]:
    """Resolve LeRobot's canonical language feature names lazily."""

    try:
        from lerobot.utils.constants import (
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency boundary.
        raise RuntimeError("pi0-FAST requires LeRobot 0.6.0") from exc
    return OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK


def _assert_pretrained_weights_loaded(native_policy: Any, path: str | Path) -> None:
    """Detect LeRobot 0.6.0's silent fallback to an uninitialized model."""

    import math

    from safetensors import safe_open
    import torch

    state = native_policy.state_dict()
    candidates: list[tuple[int, str, str]] = []
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        for source_key in checkpoint.keys():
            target_key = (
                source_key if source_key.startswith("model.") else f"model.{source_key}"
            )
            target = state.get(target_key)
            if target is None:
                continue
            shape = tuple(checkpoint.get_slice(source_key).get_shape())
            if tuple(target.shape) == shape:
                candidates.append((math.prod(shape), source_key, target_key))
        if not candidates:
            raise RuntimeError(
                "pi0-FAST checkpoint and instantiated model share no verifiable tensors"
            )
        _, source_key, target_key = min(candidates)
        expected = checkpoint.get_tensor(source_key)
    actual = state[target_key].detach().cpu()
    if not torch.equal(actual, expected.to(dtype=actual.dtype)):
        raise RuntimeError(
            "pi0-FAST pretrained weights were not loaded exactly; refusing to run "
            "with LeRobot's uninitialized fallback model"
        )


def _concatenate_processed_batches(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    import numbers

    import torch

    if not rows:
        raise ValueError("pi0-FAST processed batch cannot be empty")
    keys = set(rows[0])
    if any(set(row) != keys for row in rows[1:]):
        raise ValueError("pi0-FAST processor produced inconsistent batch keys")
    combined: dict[str, Any] = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        if all(value is None for value in values):
            combined[key] = None
        elif all(isinstance(value, torch.Tensor) for value in values):
            if any(value.ndim == 0 or value.shape[0] != 1 for value in values):
                raise ValueError(
                    "pi0-FAST processor rows require singleton leading batches; "
                    f"key={key!r}"
                )
            combined[key] = torch.cat(values, dim=0)
        elif all(isinstance(value, numbers.Number) for value in values):
            combined[key] = torch.as_tensor(values)
        elif all(value == values[0] for value in values[1:]):
            combined[key] = values[0]
        else:
            combined[key] = values
    return combined
