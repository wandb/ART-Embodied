import asyncio
import json
from types import SimpleNamespace

import pytest

from art_embodied.config import PI0FastLoadConfig


def load_kwargs(**changes):
    return (
        dict(
            runtime_contract="lerobot_v060",
            execution_horizon=10,
            action_dim=7,
            max_decoding_steps=256,
            action_tokenizer_revision="a" * 40,
            observation_key_map={"image": "observation.images.image"},
            strict_weights=True,
            compile_model=False,
            gradient_checkpointing=True,
            use_kv_cache=True,
        )
        | changes
    )


@pytest.mark.parametrize("scale", [0, 3, float("nan"), float("inf")])
def test_invalid_loss_scale_rejected(scale):
    with pytest.raises(ValueError):
        PI0FastLoadConfig.model_validate(
            load_kwargs(model_compute_dtype="fp16_residual", training_loss_scale=scale)
        )


def test_precision_is_explicit_and_compile_not_silently_enabled():
    config = PI0FastLoadConfig.model_validate(load_kwargs())
    assert config.model_compute_dtype == "checkpoint"
    assert config.training_loss_scale == 1
    for kwargs in (
        {"training_loss_scale": 128},
        {"model_compute_dtype": "fp16_residual", "compile_model": True},
    ):
        with pytest.raises(ValueError):
            PI0FastLoadConfig.model_validate(load_kwargs(**kwargs))


@pytest.mark.parametrize(
    "source,target,source_scale,target_scale",
    [
        ("checkpoint", "fp16_residual", 1, 128),
        ("float32", "fp16_residual", 1, 128),
        ("fp16_residual", "checkpoint", 128, 1),
        ("fp16_residual", "fp16_residual", 128, 64),
    ],
)
def test_checkpoint_precision_mismatch_rejected_before_mutation(
    tmp_path, source, target, source_scale, target_scale
):
    torch = pytest.importorskip("torch")
    from art_embodied.policies.pi0_fast import PI0FastPolicy

    policy = PI0FastPolicy.__new__(PI0FastPolicy)
    policy.policy = SimpleNamespace(model=torch.nn.Linear(2, 2))
    policy.model_id, policy.revision, policy.device = "local-test", "test", "cpu"
    policy.model_compute_dtype, policy.training_loss_scale = source, source_scale
    policy.save_checkpoint(tmp_path)
    metadata = json.loads(
        (tmp_path / "art_embodied_pi0_fast_snapshot.json").read_text()
    )
    assert metadata["model_compute_dtype"] == source
    assert metadata["training_loss_scale"] == source_scale
    policy.model_compute_dtype, policy.training_loss_scale = target, target_scale
    before = {name: p.detach().clone() for name, p in policy.model.named_parameters()}
    with pytest.raises(ValueError, match="precision/loss scale mismatch"):
        policy.load_checkpoint(tmp_path)
    for name, p in policy.model.named_parameters():
        assert torch.equal(p, before[name])


@pytest.mark.parametrize("training_unit", ["action", "trajectory"])
@pytest.mark.parametrize("gradient_only", [False, True])
def test_real_backend_unscales_before_handoff_and_clip(training_unit, gradient_only):
    torch = pytest.importorskip("torch")
    from art_embodied import (
        Action,
        ActionTokenGRPOBackend,
        EmbodiedTrajectory,
        EmbodiedTrajectoryGroup,
    )

    class Policy(torch.nn.Module):
        family = "pi0_fast"
        training_loss_scale = 128.0

        def __init__(self, precision):
            super().__init__()
            self.model_compute_dtype = precision
            self.logits = torch.nn.Parameter(torch.tensor([0.05, -0.05]))

        def action_token_logprobs(self, examples):
            scores = self.logits.log_softmax(-1)
            return [scores[torch.tensor(e.tokens)] for e in examples]

    members = []
    initial = torch.tensor([0.05, -0.05]).log_softmax(-1).tolist()
    for token, reward in [(0, 0.0), (1, 1.0)]:
        trajectory = EmbodiedTrajectory(task="pick", reward=reward)
        trajectory.actions = [
            Action(
                step=step,
                kind="token",
                raw={"tokens": [token, token], "prompt": "pick"},
                logprobs={"token_logprobs": [initial[token]] * 2},
            )
            for step in range(2)
        ]
        members.append(trajectory)
    groups = [EmbodiedTrajectoryGroup(members)]
    policies = [Policy(precision) for precision in ("checkpoint", "fp16_residual")]
    for policy in policies:
        backend = ActionTokenGRPOBackend(
            policy,
            training_unit=training_unit,
            max_grad_norm=0.01,
            optimizer=torch.optim.AdamW(policy.parameters(), lr=0.01),
            train_logprob_microbatch_size=1,
        )
        asyncio.run(
            backend.train(groups, _action_token_grpo_return_gradients=gradient_only)
        )
    assert torch.equal(policies[0].logits, policies[1].logits)
    assert torch.equal(policies[0].logits.grad, policies[1].logits.grad)
    assert torch.isfinite(policies[1].logits.grad).all()
    assert policies[1].logits.grad.abs().sum() > 0
