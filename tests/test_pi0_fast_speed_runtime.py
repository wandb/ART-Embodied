from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from examples.embodied.pi0_fast_speed_runtime import (
    compact_decode_rows,
    split_bf16_frozen_linears,
)


def test_compaction_removes_finished_cache_rows_and_restores_methods():
    class Cache:
        def __init__(self):
            self.rows = torch.tensor([0, 1])

        def batch_select_indices(self, indices):
            self.rows = self.rows[indices]

    cache = Cache()
    batches = []

    def forward(**kwargs):
        hidden = kwargs["inputs_embeds"][0]
        batches.append(hidden.shape[0])
        return (hidden + 1, None), cache

    def embed(tokens):
        return tokens.unsqueeze(-1).float()

    transformer = SimpleNamespace(forward=forward, embed_language_tokens=embed)
    policy = SimpleNamespace(
        family="pi0_fast",
        use_kv_cache=True,
        model=SimpleNamespace(paligemma_with_expert=transformer),
        policy=SimpleNamespace(
            _paligemma_tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda _: 9)
        ),
    )
    with compact_decode_rows(policy) as counters:
        transformer.forward(
            inputs_embeds=[torch.zeros(2, 3, 1), None], past_key_values=None
        )
        embedded = transformer.embed_language_tokens(torch.tensor([[9], [4]]))
        (hidden, _), _ = transformer.forward(
            inputs_embeds=[embedded, None],
            past_key_values=cache,
            position_ids=torch.ones(2, 1),
            attention_mask=torch.ones(2, 1, 1, 4),
        )
        assert batches == [2, 1]
        assert cache.rows.tolist() == [1]
        assert hidden[:, 0, 0].tolist() == [0, 5]
        assert counters == {"decode_rows_requested": 2, "decode_rows_computed": 1}
        # A fresh prediction batch must reset the completed-row state.
        transformer.forward(
            inputs_embeds=[torch.zeros(2, 3, 1), None], past_key_values=None
        )
        transformer.embed_language_tokens(torch.tensor([[4], [5]]))
        transformer.forward(
            inputs_embeds=[torch.ones(2, 1, 1), None],
            past_key_values=cache,
            position_ids=torch.ones(2, 1),
            attention_mask=torch.ones(2, 1, 1, 4),
        )
        assert batches[-1] == 2
    assert transformer.forward is forward
    assert transformer.embed_language_tokens is embed


def test_compaction_does_not_accept_other_policy_families():
    with pytest.raises(ValueError, match="FAST"):
        with compact_decode_rows(SimpleNamespace(family="pi0")):
            pass


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA BF16 output_dtype required"
)
def test_split_frozen_linear_cuda_forward_backward():
    torch.manual_seed(123)
    torch.set_float32_matmul_precision("highest")
    model = torch.nn.Sequential(torch.nn.Linear(257, 131)).cuda()
    model.requires_grad_(False)
    x = torch.randn(2, 7, 257, device="cuda", requires_grad=True)
    weights = torch.randn(2, 7, 131, device="cuda")
    y = model(x)
    baseline_grad = torch.autograd.grad((y * weights).sum(), x)[0]
    original = model[0].forward
    with split_bf16_frozen_linears(
        SimpleNamespace(family="pi0_fast", model=model)
    ) as count:
        assert count == 1
        candidate = model(x)
        grad = torch.autograd.grad((candidate * weights).sum(), x)[0]
    assert model[0].forward == original
    assert ((y - candidate).norm() / y.norm()).item() < 3e-5
    assert ((baseline_grad - grad).norm() / baseline_grad.norm()).item() < 3e-5
