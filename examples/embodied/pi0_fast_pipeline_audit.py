"""Trace retained rewards through a real gradient shard and snapshot reload.

No optimizer update or environment rollout is performed. Only trusted local
experiment evidence is unpickled. Passing this audit does not prove RL lift.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


def reward_advantage(rewards, index, epsilon=1e-6):
    values = np.asarray(rewards, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Invalid reward group")
    return float((values[index] - values.mean()) / (values.std(ddof=1) + epsilon))


def validate_example(example, groups):
    m = example.metadata
    group = groups[int(m["group_index"])]
    i = int(m["trajectory_index_in_group"])
    reward = group["rewards"][i]
    if example.reward != reward or m["trajectory_reward"] != reward:
        raise ValueError("Reward correspondence failed")
    tm = m["trajectory_metadata"]
    if tm["scenario_id"] != group["scenario_id"]:
        raise ValueError("Scenario correspondence failed")
    if tm["environment_seed"] != group["environment_seed"]:
        raise ValueError("Reset correspondence failed")
    if example.observation.step != example.step or example.action_index != example.step:
        raise ValueError("Observation/action step correspondence failed")
    a = reward_advantage(group["rewards"], i)
    supplied = np.asarray(m["token_advantages"])
    if len(supplied) != len(example.tokens) or not np.allclose(supplied, a, atol=2e-6):
        raise ValueError("Advantage differs from independently computed group rewards")
    mask = np.asarray(m["token_loss_mask"], dtype=bool)
    if mask.shape != supplied.shape or not mask.all():
        raise ValueError("This audit requires the generated-sequence objective")
    action = m["action_metadata"]
    if action["sampling_temperature"] != 0.2:
        raise ValueError("Unexpected sampling temperature")
    if action.get("action_decode_valid"):
        decoded = np.asarray(example.decoded_action)
        processed = np.asarray(action["processed_action_chunk"])
        executed = np.asarray(action["executed_action_vectors"])
        if not np.array_equal(decoded, processed):
            raise ValueError("Recorded action differs from simulator input chunk")
        if not np.allclose(processed[: len(executed)], executed, rtol=0, atol=1e-7):
            raise ValueError("Simulator executed different primitive actions")
    return a


def snapshot_comparison(policy, snapshot):
    from safetensors.torch import load_file
    import torch

    expected = load_file(snapshot / "adapter_model.safetensors")
    actual = {
        n.removeprefix("model.").replace(".default.weight", ".weight"): p.detach().cpu()
        for n, p in policy.named_parameters()
        if p.requires_grad
    }
    if set(actual) != set(expected):
        raise ValueError("Live adapter keys differ from snapshot")
    if not all(torch.equal(actual[k], expected[k]) for k in expected):
        raise ValueError("Live adapter values differ from snapshot")
    return {
        "tensors": len(actual),
        "bitwise_equal": True,
        "sha256": hashlib.sha256(
            (snapshot / "adapter_model.safetensors").read_bytes()
        ).hexdigest(),
    }


def main():
    import torch
    import wandb

    from art_embodied.backends.action_token_worker import _build_runtime
    from examples.embodied.pi0_fast_worker_audit import compare_gradients

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    initial = args.workers / "snapshots/update-0000/initial"
    candidate = args.experiment / "checkpoints/step_000001"
    worker = args.workers / "worker-00"
    job = worker / "update-0000-job-0000"
    spec = json.loads((worker / "bootstrap.json").read_text())
    spec["policy_snapshot"] = str(initial.resolve())
    evidence = json.loads(
        (args.experiment / "rollout-evidence/policy-version-000000.json").read_text()
    )
    groups = {int(g["group_index"]): g for g in evidence["groups"]}
    reference_metrics = json.loads((job / "result.json").read_text())["metrics"]
    denominator = reference_metrics[
        "embodied_action_token_grpo/global_loss_denominator_examples"
    ]
    if denominator != 128:
        raise ValueError("Unexpected minibatch denominator")
    with (job / "examples.pkl").open("rb") as f:
        examples = pickle.load(f)
    run = wandb.init(
        entity="wandb-japan",
        project="art-embodied-pi0-fast-spatial",
        name="pi0-fast-first-principles-pipeline-audit",
        job_type="gradient-diagnostic",
        config={
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "workers": str(args.workers),
            "optimizer_steps": 0,
        },
    )
    run.define_metric("diagnostics/stage")
    run.define_metric("diagnostics/*", step_metric="diagnostics/stage")
    report = {"run_id": run.id, "success_rate_measured": False}
    code = 1
    try:
        run.log({"diagnostics/stage": 0, "diagnostics/started": 1})
        advantages = [validate_example(e, groups) for e in examples]
        report["upstream"] = {
            "examples": len(examples),
            "trajectories": len({e.trajectory_index for e in examples}),
            "denominator": denominator,
        }
        run.log({"diagnostics/stage": 1, "diagnostics/source_examples": len(examples)})
        config, policy, backend = _build_runtime(spec)
        policy.eval()
        if policy.model is not policy.policy.model:
            raise ValueError("Sampler and trainer do not reference the same model")
        report["initial_live_weights"] = snapshot_comparison(policy, initial)
        named = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
        for _, p in named:
            p.grad = None
        old_error = 0.0
        for start in range(0, len(examples), 2):
            batch = examples[start : start + 2]
            with torch.enable_grad():
                scores = policy.action_token_logprobs(batch)
                losses = []
                for offset, (e, score) in enumerate(zip(batch, scores, strict=True)):
                    old = torch.as_tensor(
                        e.logprobs, device=score.device, dtype=score.dtype
                    )
                    old_error = max(
                        old_error, float((score.detach() - old).abs().max())
                    )
                    ratio = (score - old).exp()
                    advantage = advantages[start + offset]
                    losses.append(
                        -torch.minimum(
                            ratio * advantage, ratio.clamp(0.8, 1.28) * advantage
                        ).sum()
                        / denominator
                    )
                torch.stack(losses).sum().backward()
            del scores, losses, score, ratio
            if start % 64 == 0:
                print(f"independent-gradient {start}/{len(examples)}", flush=True)
        gradients = {
            n: p.grad.detach().cpu()
            if p.grad is not None
            else torch.zeros_like(p, device="cpu")
            for n, p in named
        }
        saved = torch.load(job / "gradients.pt", map_location="cpu", weights_only=False)
        comparison = compare_gradients(
            saved,
            {
                "gradients": gradients,
                "missing_gradients": [n for n, p in named if p.grad is None],
            },
        )
        report["gradient"] = comparison | {"old_logprob_abs_max": old_error}
        if comparison["gradient_relative_l2"] > 0.005 or old_error > 0.02:
            raise ValueError(f"Actual worker gradient disagrees: {report['gradient']}")
        run.log(
            {
                "diagnostics/stage": 2,
                "diagnostics/gradient_relative_l2": comparison["gradient_relative_l2"],
                "diagnostics/old_logprob_abs_max": old_error,
            }
        )
        for _, p in named:
            p.grad = None
        selected = [examples[i] for i in sorted(set([0, len(examples) // 2]))]

        def score_selected():
            with torch.enable_grad():
                return [
                    x.detach().cpu() for x in policy.action_token_logprobs(selected)
                ]

        before = score_selected()
        policy.load_checkpoint(candidate)
        report["candidate_live_weights"] = snapshot_comparison(policy, candidate)
        after = score_selected()
        export = args.output / "roundtrip-snapshot"
        policy.save_checkpoint(export)
        report["exported_live_weights"] = snapshot_comparison(policy, export)
        policy.load_checkpoint(initial)
        restored = score_selected()
        report["restored_live_weights"] = snapshot_comparison(policy, initial)
        restore_error = max(
            float((a - b).abs().max()) for a, b in zip(before, restored, strict=True)
        )
        policy.load_checkpoint(export)
        reloaded = score_selected()
        reload_error = max(
            float((a - b).abs().max()) for a, b in zip(after, reloaded, strict=True)
        )
        change = max(
            float((a - b).abs().max()) for a, b in zip(before, after, strict=True)
        )
        report["roundtrip"] = {
            "restore_logprob_abs_max": restore_error,
            "reload_logprob_abs_max": reload_error,
            "updated_logprob_change_abs_max": change,
        }
        if restore_error > 1e-5 or reload_error > 1e-5 or change <= 1e-5:
            raise ValueError(f"Snapshot propagation failed: {report['roundtrip']}")
        run.log(
            {
                "diagnostics/stage": 3,
                **{"diagnostics/" + k: v for k, v in report["roundtrip"].items()},
            }
        )
        code = 0
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(report, indent=2))
        a = wandb.Artifact("pi0-fast-pipeline-audit", type="diagnostic")
        a.add_file(str(args.output / "result.json"))
        a.add_file(__file__)
        run.log_artifact(a).wait()
        run.summary["diagnostics/completed"] = code == 0
        run.finish(exit_code=code)


if __name__ == "__main__":
    main()
