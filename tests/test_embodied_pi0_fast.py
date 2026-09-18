from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")

from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)

from art_embodied.integrations.pi0_fast import PI0FastPolicyAdapter
from art_embodied.policies.pi0_fast import (
    PI0FastPolicy,
    _objective_action_token_rows,
    _pi0_fast_kv_token_logprobs,
    _pi0_fast_token_logprobs,
    _sample_actions_fast_kv_cache_until_end,
    _sample_actions_fast_no_cache_until_end,
)


class _FakeAttention:
    def __init__(self) -> None:
        self.q_proj = SimpleNamespace(weight=torch.zeros(1, dtype=torch.float32))


class _FakeLayer:
    def __init__(self) -> None:
        self.self_attn = _FakeAttention()


class _FakeLanguageModel:
    def __init__(self) -> None:
        self.layers = [_FakeLayer()]


class _FakePaliGemma:
    def __init__(self, model: "_FakeModel") -> None:
        self.model = SimpleNamespace(language_model=_FakeLanguageModel())
        self.lm_head = lambda hidden: hidden + model.logit_bias


class _FakePaliGemmaWithExpert:
    def __init__(self, model: "_FakeModel") -> None:
        self.paligemma = _FakePaliGemma(model)

    def forward(self, *, inputs_embeds, **_kwargs):
        return (inputs_embeds[0], None), None

    def embed_language_tokens(self, tokens):
        hidden = torch.zeros(*tokens.shape, 6)
        hidden[tokens == 2, 4] = 2.0
        hidden[tokens == 4, 5] = 1.0
        return hidden


class _FakeModel:
    def __init__(self) -> None:
        self.logit_bias = torch.nn.Parameter(torch.zeros(6))
        self.paligemma_with_expert = _FakePaliGemmaWithExpert(self)

    def embed_prefix_fast(
        self,
        _images,
        _image_masks,
        _language_tokens,
        _language_masks,
        *,
        fast_action_tokens,
        fast_action_masks,
    ):
        assert fast_action_tokens is None
        assert fast_action_masks is None
        batch_size, language_length = _language_tokens.shape
        hidden = torch.zeros(batch_size, language_length, 6)
        hidden[:, -1, 2] = 3.0
        pad_masks = _language_masks
        attention = torch.ones(
            batch_size,
            hidden.shape[1],
            hidden.shape[1],
            dtype=torch.bool,
        )
        return hidden, pad_masks, attention, 0, 0

    def _prepare_attention_masks_4d(self, masks, *, dtype):
        return masks[:, None].to(dtype=dtype)


class _FakeNativePolicy:
    def __init__(self) -> None:
        self.model = _FakeModel()
        self._paligemma_tokenizer = SimpleNamespace(
            bos_token_id=1,
            pad_token_id=0,
        )

    def _preprocess_images(self, _batch):
        return [], []


class _EarlyStopCache:
    def __init__(self, size):
        self.rows = torch.arange(size)

    def batch_select_indices(self, indices):
        self.rows = self.rows[indices]


class _EarlyStopModel:
    def __init__(self, token_steps: list[list[int]]) -> None:
        self.token_steps = token_steps
        self.forward_calls = 0
        self.forward_batch_sizes = []
        language_model = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(
                        q_proj=SimpleNamespace(
                            weight=torch.zeros(1, dtype=torch.float32)
                        )
                    )
                )
            ]
        )
        paligemma = SimpleNamespace(
            lm_head=lambda hidden: hidden,
            model=SimpleNamespace(language_model=language_model),
        )
        self.paligemma_with_expert = SimpleNamespace(
            paligemma=paligemma,
            forward=self._forward,
            embed_language_tokens=self._embed_language_tokens,
        )
        self._paligemma_tokenizer = SimpleNamespace(bos_token_id=1)

    def embed_prefix_fast(
        self,
        _images,
        _image_masks,
        tokens,
        masks,
        *,
        fast_action_tokens,
        fast_action_masks,
    ):
        assert fast_action_tokens is None
        assert fast_action_masks is None
        hidden = torch.zeros(tokens.shape[0], tokens.shape[1], 16)
        attention = torch.ones(
            tokens.shape[0], tokens.shape[1], tokens.shape[1], dtype=torch.bool
        )
        return hidden, masks, attention, 0, 0

    def _prepare_attention_masks_4d(self, masks, *, dtype):
        return masks.to(dtype=dtype)

    def _embed_language_tokens(self, tokens):
        return torch.zeros(tokens.shape[0], tokens.shape[1], 16)

    def _forward(self, **kwargs):
        step_tokens = self.token_steps[self.forward_calls]
        self.forward_calls += 1
        cache = kwargs.get("past_key_values") or _EarlyStopCache(len(step_tokens))
        self.forward_batch_sizes.append(len(cache.rows))
        hidden = torch.full((len(cache.rows), 1, 16), -1000.0)
        for row, original_row in enumerate(cache.rows):
            token = step_tokens[original_row]
            if token < 0:
                hidden[row] = float("nan")
            else:
                hidden[row, 0, token] = 1000.0
        return (hidden, None), cache


@pytest.mark.parametrize("temperature", [0.0, 0.2])
def test_pi0_fast_kv_sampler_stops_after_every_row_ends(
    temperature: float,
) -> None:
    model = _EarlyStopModel([[4, 5], [9, 6], [0, 9]])
    tokens = torch.tensor([[7], [8]])
    masks = torch.ones_like(tokens, dtype=torch.bool)

    generated = _sample_actions_fast_kv_cache_until_end(
        model,
        [],
        [],
        tokens,
        masks,
        max_decoding_steps=8,
        temperature=temperature,
        end_token_id=9,
    )

    assert model.forward_calls == 3
    assert model.forward_batch_sizes == [2, 2, 1]
    assert generated.tolist() == [
        [4, 9, 0],
        [5, 6, 9],
    ]


def test_pi0_fast_kv_sampler_returns_probabilities_used_for_sampling() -> None:
    model = _EarlyStopModel([[4, 5], [9, 6], [0, 9]])
    tokens = torch.tensor([[7], [8]])
    masks = torch.ones_like(tokens, dtype=torch.bool)

    generated, logprobs = _sample_actions_fast_kv_cache_until_end(
        model,
        [],
        [],
        tokens,
        masks,
        max_decoding_steps=8,
        temperature=0.2,
        end_token_id=9,
        return_logprobs=True,
    )

    assert generated.tolist() == [[4, 9, 0], [5, 6, 9]]
    assert tuple(logprobs.shape) == (2, 3)
    assert torch.allclose(logprobs, torch.zeros_like(logprobs))


@pytest.mark.parametrize("temperature", [0.0, 0.2])
def test_pi0_fast_kv_sampler_never_decodes_ended_rows(temperature):
    model = _EarlyStopModel([[9, 4, 5], [-1, 9, 6], [-1, -1, 9]])
    tokens = torch.tensor([[7], [8], [7]])
    result = _sample_actions_fast_kv_cache_until_end(
        model,
        [],
        [],
        tokens,
        torch.ones_like(tokens, dtype=torch.bool),
        max_decoding_steps=8,
        temperature=temperature,
        end_token_id=9,
        return_logprobs=temperature > 0,
    )
    generated = result[0] if temperature > 0 else result
    assert generated.tolist() == [[9, 0, 0], [4, 9, 0], [5, 6, 9]]
    assert model.forward_batch_sizes == [3, 2, 1]
    if temperature > 0:
        assert torch.isfinite(result[1]).all()
        assert torch.equal(result[1], torch.zeros(3, 3))


def test_pi0_fast_kv_sampler_rejects_greedy_logprobs() -> None:
    model = _EarlyStopModel([[4]])

    with pytest.raises(ValueError, match="temperature"):
        _sample_actions_fast_kv_cache_until_end(
            model,
            [],
            [],
            torch.tensor([[7]]),
            torch.ones(1, 1, dtype=torch.bool),
            max_decoding_steps=1,
            temperature=0.0,
            end_token_id=9,
            return_logprobs=True,
        )


def test_pi0_fast_no_cache_sampler_returns_probabilities_used_for_sampling() -> None:
    model = _EarlyStopModel([[4, 5], [9, 6], [0, 9]])
    tokens = torch.tensor([[7], [8]])
    masks = torch.ones_like(tokens, dtype=torch.bool)

    generated, logprobs = _sample_actions_fast_no_cache_until_end(
        model,
        [],
        [],
        tokens,
        masks,
        max_decoding_steps=8,
        temperature=0.2,
        end_token_id=9,
        return_logprobs=True,
    )

    assert model.forward_calls == 3
    assert generated.tolist() == [[4, 9, 0], [5, 6, 9]]
    assert tuple(logprobs.shape) == (2, 3)
    assert torch.allclose(logprobs, torch.zeros_like(logprobs))


def test_pi0_fast_routes_task_owned_adapter() -> None:
    policy = PI0FastPolicy(
        model_id="test/pi0-fast",
        revision="revision",
        device="cpu",
        execution_horizon=1,
        action_dim=1,
        max_decoding_steps=1,
        action_tokenizer_revision="tokenizer-revision",
        observation_key_map={},
        strict_weights=False,
        compile_model=False,
        gradient_checkpointing=False,
        use_kv_cache=False,
    )
    activated: list[str] = []
    fake_model = SimpleNamespace(
        base_model=SimpleNamespace(set_adapter=activated.append)
    )
    policy.policy = SimpleNamespace(model=fake_model)
    policy._art_partition_forward_routing = "task_owned"
    policy._art_task_adapter_map = {"task a": "task_000", "task b": "task_001"}

    assert policy.activate_task_adapter("task b") == "task_001"
    assert activated == ["task_001"]
    with pytest.raises(ValueError, match="No LoRA adapter"):
        policy.activate_task_adapter("unknown")


def test_pi0_fast_cpu_offload_releases_cuda_allocator_cache(monkeypatch) -> None:
    policy = PI0FastPolicy(
        model_id="test/pi0-fast",
        revision="revision",
        device="cuda",
        execution_horizon=1,
        action_dim=1,
        max_decoding_steps=1,
        action_tokenizer_revision="tokenizer-revision",
        observation_key_map={},
        strict_weights=False,
        compile_model=False,
        gradient_checkpointing=False,
        use_kv_cache=False,
    )
    moved_to: list[str] = []
    empty_cache_calls: list[bool] = []
    policy.policy = SimpleNamespace(
        model=object(),
        config=SimpleNamespace(device="cuda"),
        to=moved_to.append,
    )
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: empty_cache_calls.append(True)
    )

    assert policy.to("cpu") is policy
    assert moved_to == ["cpu"]
    assert policy.device == "cpu"
    assert policy.policy.config.device == "cpu"
    assert empty_cache_calls == [True]


def test_pi0_fast_teacher_forcing_scores_every_generated_token() -> None:
    native = _FakeNativePolicy()
    processed = {
        OBS_LANGUAGE_TOKENS: torch.tensor([[7], [8]]),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 1, dtype=torch.bool),
    }

    rows = _pi0_fast_token_logprobs(
        native_policy=native,
        processed_batch=processed,
        token_rows=[[2, 4], [2, 4, 5]],
        temperature=2.0,
    )

    expected_logits = torch.tensor([0.0, 0.0, 3.0, 0.0, 0.0, 0.0]) / 2.0
    expected_first = torch.log_softmax(expected_logits, dim=-1)[2]
    assert [tuple(row.shape) for row in rows] == [(2,), (3,)]
    assert torch.allclose(rows[0][0], expected_first)
    assert all(row.requires_grad for row in rows)
    (-sum(row.sum() for row in rows)).backward()
    assert native.model.logit_bias.grad is not None


def test_pi0_fast_kv_teacher_forcing_scores_every_generated_token() -> None:
    native = _FakeNativePolicy()
    processed = {
        OBS_LANGUAGE_TOKENS: torch.tensor([[7], [8]]),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 1, dtype=torch.bool),
    }

    rows = _pi0_fast_kv_token_logprobs(
        native_policy=native,
        processed_batch=processed,
        token_rows=[[2, 4], [2, 4, 5]],
        temperature=2.0,
    )

    assert [tuple(row.shape) for row in rows] == [(2,), (3,)]
    assert all(row.requires_grad for row in rows)
    (-sum(row.sum() for row in rows)).backward()
    assert native.model.logit_bias.grad is not None


class _FakeRolloutPolicy:
    execution_horizon = 2
    action_dim = 7
    config = SimpleNamespace(chunk_size=3)
    policy = object()
    preprocessor = object()
    postprocessor = object()

    def reset(self) -> None:
        pass

    def preprocess_observation(self, observation, *, task, robot_type):
        del observation, task, robot_type
        return {
            OBS_LANGUAGE_TOKENS: torch.tensor([[1]]),
            OBS_LANGUAGE_ATTENTION_MASK: torch.tensor([[True]]),
        }

    def sample_action_tokens(self, processed, *, temperature):
        assert processed[OBS_LANGUAGE_TOKENS].shape[0] == 2
        assert temperature == 1.5
        return torch.tensor([[10, 11], [12, 13]])

    def sample_action_tokens_with_logprobs(self, processed, *, temperature):
        tokens = self.sample_action_tokens(processed, temperature=temperature)
        return tokens, torch.tensor([[-0.2, -0.3], [-0.4, -0.5]])

    def processed_action_token_logprobs(self, *_args, **_kwargs):
        pytest.fail("rollout must retain the sampler-native logprobs")

    def prepare_generated_action_tokens(self, tokens):
        rows = tokens.detach().cpu().tolist()
        return (
            rows,
            [True] * len(rows),
            [0] * len(rows),
            [[True] * len(row) for row in rows],
        )

    def decode_action_tokens(self, tokens):
        assert tokens.shape[1] == 2
        batch_size = int(tokens.shape[0])
        return torch.arange(batch_size * 3 * 7, dtype=torch.float32).reshape(
            batch_size, 3, 7
        )

    def decode_action_tokens_safe(self, tokens, *, grammar_valid):
        assert all(grammar_valid)
        return self.decode_action_tokens(tokens)


def test_pi0_fast_adapter_records_tokens_and_executes_decoded_prefix() -> None:
    adapter = PI0FastPolicyAdapter(
        policy=_FakeRolloutPolicy(),
        sampling_mode="train",
        do_sample=True,
        temperature=1.5,
    )

    predictions = adapter.predict_batch(
        [{"image": 1}, {"image": 2}],
        tasks=["task one", "task two"],
        step=4,
    )

    assert len(predictions) == 2
    first = predictions[0]
    assert first.action.kind == "token"
    assert first.action.raw == {"tokens": [10, 11], "prompt": "task one"}
    assert first.action.logprobs == pytest.approx([-0.2, -0.3])
    assert first.action.metadata["sampling_temperature"] == 1.5
    assert first.action.metadata["action_grammar_valid"] is True
    assert first.action.metadata["post_termination_tokens_discarded"] == 0
    assert tuple(first.native_action.shape) == (2, 7)
    assert tuple(first.predicted_action_chunk.shape) == (3, 7)


@pytest.mark.parametrize("valid", [[False, True], [False, False]])
def test_pi0_fast_rejected_tokens_never_decode_or_execute(valid):
    from art_embodied.backends.action_token import _is_fully_masked_action

    policy = _FakeRolloutPolicy()
    policy.prepare_generated_action_tokens = lambda tokens: (
        tokens.tolist(),
        valid,
        [0, 0],
        [[False, False], [True, True]],
    )
    original_decode = policy.decode_action_tokens_safe
    calls = []

    def decode(tokens, *, grammar_valid):
        calls.append(len(tokens))
        return original_decode(tokens, grammar_valid=grammar_valid)

    policy.decode_action_tokens_safe = decode
    adapter = PI0FastPolicyAdapter(
        policy=policy, invalid_action_handling="terminate_episode", temperature=1.5
    )
    predictions = adapter.predict_batch(
        [{"image": 1}, {"image": 2}], tasks=["pick"] * 2, step=0
    )
    assert calls == ([sum(valid)] if any(valid) else [])
    for ok, prediction in zip(valid, predictions, strict=True):
        if not ok:
            assert prediction.native_action.numel() == 0
            assert prediction.execution_horizon == 0
            assert prediction.action.metadata["terminate_episode"] is True
            assert not _is_fully_masked_action(prediction.action)
            assert all(prediction.action.metadata["token_loss_mask"])
            assert prediction.action.logprobs is not None
        else:
            assert prediction.native_action.shape == (2, 7)


def test_pi0_fast_termination_requires_complete_token_scope():
    policy = _FakeRolloutPolicy()
    policy.rl_token_scope = "fast_payload"
    with pytest.raises(ValueError, match="generated_sequence"):
        PI0FastPolicyAdapter(policy=policy, invalid_action_handling="terminate_episode")


def test_pi0_fast_adapter_can_limit_rl_objective_to_fast_payload() -> None:
    policy = _FakeRolloutPolicy()
    policy.rl_token_scope = "fast_payload"
    policy.prepare_generated_action_tokens = lambda _tokens: (
        [[10, 11, 40, 41, 12], [10, 11, 42, 43, 12]],
        [True, True],
        [0, 0],
        [
            [False, False, True, True, False],
            [False, False, True, True, False],
        ],
    )
    policy.sample_action_tokens_with_logprobs = lambda _processed, *, temperature: (
        torch.tensor([[10, 11, 40, 41, 12], [10, 11, 42, 43, 12]]),
        torch.tensor(
            [
                [-0.1, -0.2, -0.3, -0.4, -0.5],
                [-0.6, -0.7, -0.8, -0.9, -1.0],
            ]
        ),
    )
    policy.decode_action_tokens_safe = lambda tokens, *, grammar_valid: torch.zeros(
        len(grammar_valid), 3, 7
    )

    prediction = PI0FastPolicyAdapter(
        policy=policy,
        sampling_mode="train",
        do_sample=True,
        temperature=0.2,
    ).predict_batch(
        [{"image": 1}, {"image": 2}],
        tasks=["task one", "task two"],
        step=0,
    )[0]

    assert prediction.action.metadata["rl_token_scope"] == "fast_payload"
    assert prediction.action.metadata["token_loss_mask"] == [
        False,
        False,
        True,
        True,
        False,
    ]
    assert prediction.action.metadata["rl_objective_token_count"] == 2
    assert prediction.action.metadata["format_token_count"] == 3


def test_pi0_fast_adapter_splits_generation_into_fixed_model_batches() -> None:
    policy = _FakeRolloutPolicy()
    observed_batch_sizes: list[int] = []

    def sample_with_logprobs(processed, *, temperature):
        batch_size = int(processed[OBS_LANGUAGE_TOKENS].shape[0])
        observed_batch_sizes.append(batch_size)
        assert temperature == 1.5
        tokens = torch.tensor([[10, 11]] * batch_size)
        return tokens, torch.tensor([[-0.2, -0.3]] * batch_size)

    policy.sample_action_tokens_with_logprobs = sample_with_logprobs
    adapter = PI0FastPolicyAdapter(
        policy=policy,
        sampling_mode="train",
        do_sample=True,
        temperature=1.5,
        model_batch_size=2,
    )

    predictions = adapter.predict_batch(
        [{"image": index} for index in range(5)],
        tasks=[f"task {index}" for index in range(5)],
        step=0,
    )

    assert len(predictions) == 5
    assert observed_batch_sizes == [2, 2, 2]
    assert [
        prediction.action.metadata["model_batch_size"] for prediction in predictions
    ] == [2, 2, 2, 2, 2]


def test_pi0_fast_adapter_masks_malformed_action_tokens_from_training() -> None:
    policy = _FakeRolloutPolicy()
    policy.sample_action_tokens = lambda _processed, *, temperature: (
        torch.tensor([[10, 11], [12, 13]])
        if temperature == 0.1
        else pytest.fail("the calibration temperature must be preserved")
    )
    policy.processed_action_token_logprobs = (
        lambda _processed, _token_rows, *, temperature: (
            [torch.tensor([-0.2, -0.3]), torch.tensor([-0.4, -0.5])]
            if temperature == 0.1
            else pytest.fail("logprobs must use the rollout temperature")
        )
    )
    policy.prepare_generated_action_tokens = lambda _tokens: (
        [[10, 11], [12, 13]],
        [True, False],
        [0, 0],
        [[True, True], [False, False]],
    )
    policy.decode_action_tokens_safe = lambda tokens, *, grammar_valid: (
        policy.decode_action_tokens(tokens)
    )
    adapter = PI0FastPolicyAdapter(
        policy=policy,
        sampling_mode="train",
        do_sample=True,
        temperature=0.1,
    )

    predictions = adapter.predict_batch(
        [{"observation.state": [0.0]}, {"observation.state": [1.0]}],
        tasks=["task one", "task two"],
        step=0,
    )

    assert "primitive_loss_mask_sum" not in predictions[0].action.metadata
    assert predictions[1].action.metadata["primitive_loss_mask_sum"] == 0


def test_pi0_fast_eval_is_greedy_and_does_not_claim_logprobs() -> None:
    policy = _FakeRolloutPolicy()
    adapter = PI0FastPolicyAdapter(
        policy=policy,
        sampling_mode="eval",
        do_sample=False,
        temperature=1.5,
    )
    policy.sample_action_tokens = lambda _processed, *, temperature: (
        torch.tensor([[10, 11]])
        if temperature == 0.0
        else pytest.fail("eval must be greedy")
    )

    prediction = adapter.predict_batch(
        [{"image": 1}],
        tasks=["task"],
        step=0,
    )[0]

    assert prediction.action.logprobs is None
    assert prediction.action.metadata["probability_model"] == "greedy"


def test_pi0_fast_eval_can_explicitly_sample_for_calibration() -> None:
    policy = _FakeRolloutPolicy()
    adapter = PI0FastPolicyAdapter(
        policy=policy,
        sampling_mode="eval",
        do_sample=True,
        temperature=1.5,
    )

    prediction = adapter.predict_batch(
        [{"image": 1}, {"image": 2}],
        tasks=["task one", "task two"],
        step=0,
    )[0]

    assert prediction.action.logprobs == pytest.approx([-0.2, -0.3])
    assert prediction.action.metadata["probability_model"] == "categorical_tokens"
    assert prediction.action.metadata["sampling_temperature"] == 1.5


def test_pi0_fast_adapter_can_defer_rollout_logprobs() -> None:
    policy = _FakeRolloutPolicy()
    policy.processed_action_token_logprobs = lambda *_args, **_kwargs: pytest.fail(
        "deferred rollout logprobs must not invoke the scorer"
    )
    adapter = PI0FastPolicyAdapter(
        policy=policy,
        sampling_mode="train",
        do_sample=True,
        temperature=1.5,
        compute_rollout_logprobs=False,
    )

    predictions = adapter.predict_batch(
        [{"image": 1}, {"image": 2}],
        tasks=["task one", "task two"],
        step=0,
    )

    assert all(prediction.action.logprobs is None for prediction in predictions)
    assert all(
        prediction.action.metadata["rollout_logprobs_computed"] is False
        for prediction in predictions
    )


def test_pi0_fast_objective_stops_at_action_terminator() -> None:
    rows, valid, discarded, payload_masks = _objective_action_token_rows(
        [
            [10, 11, 40, 41, 12, 99, 98],
            [10, 11, 42, 43],
            [7, 11, 12, 97],
        ],
        prefix_ids=[10, 11],
        end_token_id=12,
    )

    assert rows == [
        [10, 11, 40, 41, 12],
        [10, 11, 42, 43],
        [7, 11],
    ]
    assert valid == [True, True, False]
    assert discarded == [2, 0, 2]
    assert payload_masks == [
        [False, False, True, True, False],
        [False, False, True, True],
        [False, False],
    ]


def test_pi0_fast_grammar_rejects_tokens_outside_action_vocabulary() -> None:
    rows, valid, discarded, payload_masks = _objective_action_token_rows(
        [
            [10, 11, 40, 41, 12, 99],
            [10, 11, 40, 55, 12, 98],
            [10, 11, 12, 97],
        ],
        prefix_ids=[10, 11],
        end_token_id=12,
        action_token_min_id=40,
        action_token_max_id=49,
    )

    assert rows == [
        [10, 11, 40, 41, 12],
        [10, 11, 40, 55, 12],
        [10, 11, 12],
    ]
    assert valid == [True, False, False]
    assert discarded == [1, 1, 1]
    assert payload_masks == [
        [False, False, True, True, False],
        [False, False, False, False, False],
        [False, False, False],
    ]


@pytest.mark.parametrize("corruption", ["missing", "extra", "shape"])
def test_pi0_fast_rejects_incomplete_adapter_before_weight_mutation(
    tmp_path, corruption
):
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file, save_file

    model = get_peft_model(
        torch.nn.Sequential(torch.nn.Linear(2, 2)),
        LoraConfig(r=1, lora_alpha=1, target_modules=["0"]),
    )
    policy = PI0FastPolicy.__new__(PI0FastPolicy)
    policy.policy = SimpleNamespace(model=model)
    policy.model_id = "local-test"
    policy.revision = "test-revision"
    policy.device = "cpu"
    policy.model_compute_dtype = "checkpoint"
    policy.training_loss_scale = 1.0
    policy.save_checkpoint(tmp_path)
    state = load_file(tmp_path / "adapter_model.safetensors")
    key = next(key for key in state if "lora_B" in key)
    if corruption == "missing":
        state.pop(key)
    elif corruption == "extra":
        state["unexpected.weight"] = torch.zeros(1)
    else:
        state[key] = torch.zeros(9, 9)
    save_file(state, tmp_path / "adapter_model.safetensors")
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.fill_(7)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    with pytest.raises(RuntimeError, match="state mismatch"):
        policy.load_checkpoint(tmp_path)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name])


def test_malformed_fast_requires_physical_fallback_and_preserves_valid_rows():
    policy = PI0FastPolicy.__new__(PI0FastPolicy)
    policy.policy = SimpleNamespace(
        config=SimpleNamespace(chunk_size=2),
        detokenize_actions=lambda tokens, **kw: torch.ones(len(tokens), 2, 7),
    )
    policy.action_dim = 7
    mean = torch.tensor([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.5])
    policy.postprocessor = lambda normalized: normalized * 2 + mean
    tokens = torch.tensor([[123], [456]])
    with pytest.raises(ValueError, match="explicit physical"):
        policy.decode_action_tokens_safe(tokens, grammar_valid=[True, False])
    fallback = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    actions = policy.decode_action_tokens_safe(
        tokens, grammar_valid=[True, False], invalid_action=fallback
    )
    torch.testing.assert_close(actions[0], (torch.ones(2, 7) * 2 + mean))
    torch.testing.assert_close(actions[1], torch.tensor(fallback).expand(2, 7))
    for invalid in ([0.0], [float("nan")] * 7):
        with pytest.raises(ValueError, match="finite action_dim"):
            policy.decode_action_tokens_safe(
                tokens, grammar_valid=[True, False], invalid_action=invalid
            )
