"""Frozen-data, initial-state-disjoint gradient and learning-rate diagnostics.

No environment rollout, optimizer continuation, or exportable policy is created.
Held-out surrogates are not success rates and cannot qualify a learning recipe.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def wait_file(path, seconds=1200):
    deadline = time.monotonic() + seconds
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(str(path))
        time.sleep(1)


def make_plan(records):
    trajectories = {}
    for rows in records:
        for tid, record in rows.items():
            if tid in trajectories and trajectories[tid] != record:
                raise ValueError("Inconsistent trajectory metadata")
            trajectories[tid] = record
    keys = sorted({r["initial_state"] for r in trajectories.values()})
    if len(keys) < 4:
        raise ValueError("Too few distinct initial states")
    assignment = {k: ("fit" if i % 2 == 0 else "heldout") for i, k in enumerate(keys)}
    splits = {tid: assignment[r["initial_state"]] for tid, r in trajectories.items()}
    return {
        "splits": splits,
        "initial_states": assignment,
        "counts": {s: sum(v == s for v in splits.values()) for s in ("fit", "heldout")},
        "trajectories": trajectories,
    }


def history_variants(weight, gradients, history, role_a):
    """Change epsilon by role without changing LR, moments, or the input model."""
    import torch

    from examples.embodied.pi0_fast_optimizer_calibration_audit import adam_delta

    fit = gradients["fit_positive"] + gradients["fit_negative"]
    result = {"zero_control": torch.zeros_like(weight)}
    info = {}
    for label, epsilon_a, epsilon_b, g in (
        ("current", 1e-5, 1e-5, fit),
        ("eps_all", 1e-8, 1e-8, fit),
        ("eps_A_only", 1e-8, 1e-5, fit),
        ("eps_B_only", 1e-5, 1e-8, fit),
        ("without_long", 1e-5, 1e-5, fit - gradients["fit_long"]),
    ):
        clipped = g * min(1.0, 1.0 / (float(g.norm()) + 1e-6))
        eps = torch.empty_like(weight).fill_(epsilon_b)
        eps[role_a] = epsilon_a
        d, _, _, rms = adam_delta(
            weight,
            clipped,
            history["first"].double(),
            history["second"].double(),
            history["step"] + 1,
            lr=2e-6,
            eps=eps,
        )
        result[label] = d
        info[label] = {
            "A_delta_norm": float(d[role_a].norm()),
            "B_delta_norm": float(d[~role_a].norm()),
            "A_rms_median": float(rms[role_a].median()),
            "B_rms_median": float(rms[~role_a].median()),
        }
    return result, info


def copy_shared_initial(named, source):
    """Preserve the Q/V coordinates and require every added B matrix to be zero."""
    import torch

    seen = set()
    with torch.no_grad():
        for name, p in named:
            key = name.removeprefix("model.").replace(".default.weight", ".weight")
            if key in source:
                if p.shape != source[key].shape:
                    raise ValueError(f"Shared adapter shape differs: {key}")
                p.copy_(source[key].to(p))
                seen.add(key)
            elif "lora_B" in name and torch.count_nonzero(p):
                raise ValueError(f"Added adapter changes the initial policy: {name}")
    if seen != set(source):
        raise ValueError("Expanded surface does not contain all shared parameters")


def surface_variants(weight, gradients, qv_mask, mlp_mask):
    """Compare surfaces at equal LR and equal fit-only linearized improvement."""
    import torch

    fit = gradients["fit_positive"] + gradients["fit_negative"]

    def first_delta(g, mask, lr):
        selected = g * mask
        clipped = selected * min(1.0, 1.0 / (float(selected.norm()) + 1e-6))
        return (-lr * clipped / (clipped.abs() + 1e-5) - lr * 0.01 * weight) * mask

    all_mask = torch.ones_like(qv_mask)
    qv = first_delta(fit, qv_mask, 2e-6)
    full = first_delta(fit, all_mask, 2e-6)
    mlp = first_delta(fit, mlp_mask, 2e-6)
    target, full_gain, mlp_gain = (float(-fit @ d) for d in (qv, full, mlp))
    if min(target, full_gain, mlp_gain) <= 0:
        raise ValueError("Cannot match a nonpositive first-order fit improvement")
    scales = {
        "full_matched_fit": target / full_gain,
        "mlp_matched_fit": target / mlp_gain,
    }
    return {
        "zero_control": torch.zeros_like(weight),
        "qv_current": qv,
        "full_current": full,
        "full_matched_fit": full * scales["full_matched_fit"],
        "mlp_matched_fit": mlp * scales["mlp_matched_fit"],
        "full_lr1e-5": full * 5,
    }, scales


def worker(args):
    import torch

    from art_embodied.backends.action_token_worker import _build_runtime
    from examples.embodied.pi0_fast_score_sum_direction import token_statistics

    spec = json.loads(
        (args.workers / f"worker-{args.worker:02d}/bootstrap.json").read_text()
    )
    spec["policy_snapshot"] = str(
        (args.workers / f"snapshots/update-{args.update:04d}/initial").resolve()
    )
    if args.surface == "full-lm":
        from safetensors.torch import load_file, save_file

        from art_embodied.backends.action_token_worker import _worker_config
        from art_embodied.policies.factory import make_policy

        spec["config"]["policy"]["lora"]["target_modules"] = [
            "auto_full_language_model"
        ]
        config = _worker_config(spec)
        policy, backend = make_policy(config), None
        named = sorted((n, p) for n, p in policy.named_parameters() if p.requires_grad)
        source = load_file(Path(spec["policy_snapshot"]) / "adapter_model.safetensors")
        if any(torch.count_nonzero(v) for k, v in source.items() if "lora_B" in k):
            raise ValueError("Surface comparison requires a zero-delta initial policy")
        copy_shared_initial(named, source)
        weights = {
            n.removeprefix("model.").replace(".default.weight", ".weight"): p.detach()
            .cpu()
            .contiguous()
            for n, p in named
        }
        fingerprint = hashlib.sha256()
        for k, v in weights.items():
            fingerprint.update(k.encode())
            fingerprint.update(v.numpy().tobytes())
        write(
            args.output / f"surface-{args.worker}.json",
            {
                "initial_hash": fingerprint.hexdigest(),
                "parameters": sum(p.numel() for _, p in named),
                "names": [n for n, _ in named],
            },
        )
        if args.worker == 0:
            save_file(weights, args.output / "initial-full.tmp")
            (args.output / "initial-full.tmp").replace(
                args.output / "initial-full.safetensors"
            )
        del weights, source
    else:
        config, policy, backend = _build_runtime(spec)
    policy.eval()
    examples = []
    for j in range(4):
        with (
            args.workers
            / f"worker-{args.worker:02d}/update-{args.update:04d}-job-{j:04d}/examples.pkl"
        ).open("rb") as f:
            examples.extend(pickle.load(f))
    records = {}
    for e in examples:
        reset = e.metadata["trajectory_metadata"]["reset_info"]
        records[str(e.trajectory_index)] = {
            "initial_state": f"{reset['suite']}/{reset['task_id']}/{reset['init_state_index']}",
            "group": int(e.metadata["group_index"]),
            "reward": float(e.reward),
        }
    write(args.output / f"records-{args.worker}.json", records)
    wait_file(args.output / "plan.json")
    plan = json.loads((args.output / "plan.json").read_text())
    named = sorted((n, p) for n, p in policy.named_parameters() if p.requires_grad)
    params = [p for _, p in named]
    if args.checkpoint and args.worker == 0:
        from safetensors.torch import load_file

        expected = load_file(args.checkpoint / "adapter_model.safetensors")
        for n, p in named:
            key = n.removeprefix("model.").replace(".default.weight", ".weight")
            if not torch.equal(p.detach().cpu(), expected[key]):
                raise ValueError(f"Checkpoint/snapshot mismatch: {n}")
        backend._ensure_optimizer()
        state = torch.load(
            args.checkpoint / "art_embodied_training_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        if state["update_step"] != args.update:
            raise ValueError("Optimizer state is not from this rollout's policy")
        backend.optimizer.load_state_dict(state["optimizer_state_dict"])
        for group in backend.optimizer.param_groups:
            if (
                group["lr"] != 2e-6
                or tuple(group["betas"]) != (0.9, 0.999)
                or group["eps"] != 1e-5
                or group["weight_decay"] != 0.01
                or group.get("amsgrad", False)
                or group.get("maximize", False)
            ):
                raise ValueError("Optimizer history does not match the audited recipe")
        states = [backend.optimizer.state[p] for p in params]
        steps = {int(s["step"]) for s in states}
        if len(steps) != 1 or next(iter(steps)) != 4 * args.update:
            raise ValueError("Unexpected optimizer history step")
        history = {
            "names": [n for n, _ in named],
            "step": next(iter(steps)),
            "first": torch.cat([s["exp_avg"].flatten().cpu() for s in states]),
            "second": torch.cat([s["exp_avg_sq"].flatten().cpu() for s in states]),
        }
        torch.save(history, args.output / "history.tmp")
        (args.output / "history.tmp").replace(args.output / "history.pt")
    total = sum(p.numel() for p in params)
    gradients = {
        s + "_" + sign: torch.zeros(total)
        for s in ("fit", "heldout")
        for sign in ("positive", "negative")
    }
    if args.checkpoint:
        gradients.update({s + "_long": torch.zeros(total) for s in ("fit", "heldout")})
    baseline = [None] * len(examples)
    old_error = 0.0
    for bucket in gradients:
        split, sign = bucket.split("_")
        wanted = [
            i
            for i, e in enumerate(examples)
            if plan["splits"][str(e.trajectory_index)] == split
            and (
                len(e.tokens) >= 128 and e.metadata["group_advantage"] != 0
                if sign == "long"
                else e.metadata["group_advantage"] > 0
                if sign == "positive"
                else e.metadata["group_advantage"] < 0
            )
        ]
        for p in params:
            p.grad = None
        for offset in range(0, len(wanted), 2):
            ids = wanted[offset : offset + 2]
            scores = policy.action_token_logprobs([examples[i] for i in ids])
            losses = []
            for i, score in zip(ids, scores, strict=True):
                e = examples[i]
                old = torch.tensor(e.logprobs, device=score.device)
                advantage = torch.tensor(
                    e.metadata["token_advantages"], device=score.device
                )
                mask = torch.tensor(e.metadata["token_loss_mask"], device=score.device)
                old_error = max(old_error, float((score.detach() - old).abs().max()))
                if old_error > 0.02:
                    raise ValueError(f"Initial old-logprob mismatch: {old_error}")
                baseline[i] = score.detach().cpu().double().numpy()
                ratio = (score - old).exp()
                clipped = ratio.clamp(
                    1 - config.algorithm.clip_epsilon_low,
                    1 + config.algorithm.clip_epsilon_high,
                )
                losses.append(
                    (
                        torch.maximum(-advantage * ratio, -advantage * clipped) * mask
                    ).sum()
                )
            loss = sum(losses) / plan["counts"][split]
            loss.backward()
            del scores, score, loss, losses, ratio, clipped
            if offset % 128 == 0:
                print(f"grad {bucket}: {offset}/{len(wanted)}", flush=True)
        gradients[bucket] = torch.cat(
            [
                p.grad.detach().float().flatten().cpu()
                if p.grad is not None
                else torch.zeros(p.numel())
                for p in params
            ]
        )
    gradient_path = args.output / f"gradient-{args.worker}.pt"
    torch.save(
        {
            "names": [n for n, _ in named],
            "gradients": gradients,
            "old_error": old_error,
        },
        gradient_path.with_suffix(".tmp"),
    )
    gradient_path.with_suffix(".tmp").replace(gradient_path)
    for p in params:
        p.grad = None
    wait_file(args.output / "deltas.pt")
    deltas = torch.load(
        args.output / "deltas.pt", map_location="cpu", weights_only=True
    )
    before = [p.detach().clone() for p in params]
    import numpy as np

    ids = [i for i in range(len(examples)) if baseline[i] is not None]
    raw_metadata = [
        {
            "index": i,
            "trajectory": examples[i].trajectory_index,
            "action_index": examples[i].action_index,
            "group": examples[i].metadata["group_index"],
            "split": plan["splits"][str(examples[i].trajectory_index)],
            "tokens": list(examples[i].tokens),
        }
        for i in ids
    ]
    write(args.output / f"raw-metadata-{args.worker}.json", raw_metadata)
    np.savez_compressed(
        args.output / f"raw-baseline-{args.worker}.npz",
        lengths=np.array([len(examples[i].tokens) for i in ids]),
        old=np.concatenate([np.asarray(examples[i].logprobs) for i in ids]),
        baseline=np.concatenate([baseline[i] for i in ids]),
        advantage=np.concatenate(
            [np.asarray(examples[i].metadata["token_advantages"]) for i in ids]
        ),
        mask=np.concatenate(
            [np.asarray(examples[i].metadata["token_loss_mask"]) for i in ids]
        ),
    )
    output = {"worker": args.worker, "old_error": old_error, "variants": {}}
    try:
        for label, delta in deltas.items():
            pieces = delta.split([p.numel() for p in params])
            with torch.no_grad():
                for p, base, d in zip(params, before, pieces, strict=True):
                    p.copy_(base + d.reshape_as(p).to(p))
            sums = {
                s: {
                    "gain": 0.0,
                    "score_gain": 0.0,
                    "positive_delta": 0.0,
                    "negative_delta": 0.0,
                    "active_tokens": 0,
                    "clipped_tokens": 0,
                }
                for s in ("fit", "heldout")
            }
            raw_scores = []
            for offset in range(0, len(ids), 2):
                batch_ids = ids[offset : offset + 2]
                with torch.enable_grad():
                    scores = policy.action_token_logprobs(
                        [examples[i] for i in batch_ids]
                    )
                    scores = [s.detach().cpu().double().numpy() for s in scores]
                raw_scores.extend(scores)
                for i, score in zip(batch_ids, scores, strict=True):
                    e = examples[i]
                    stats = token_statistics(
                        e.logprobs,
                        baseline[i],
                        score,
                        e.metadata["token_advantages"],
                        e.metadata["token_loss_mask"],
                        config.algorithm.clip_epsilon_low,
                        config.algorithm.clip_epsilon_high,
                    )
                    s = sums[plan["splits"][str(e.trajectory_index)]]
                    s["gain"] += stats["clipped_surrogate_gain_sum"]
                    s["score_gain"] += stats["advantage_logprob_delta_sum"]
                    s[
                        "positive_delta"
                        if e.metadata["group_advantage"] > 0
                        else "negative_delta"
                    ] += stats["logprob_delta_sum"]
                    import numpy as np

                    ratio = np.exp(score - np.asarray(e.logprobs))
                    mask = np.asarray(e.metadata["token_loss_mask"], dtype=bool)
                    s["active_tokens"] += int(mask.sum())
                    s["clipped_tokens"] += int(
                        (
                            (
                                (ratio < 1 - config.algorithm.clip_epsilon_low)
                                | (ratio > 1 + config.algorithm.clip_epsilon_high)
                            )
                            & mask
                        ).sum()
                    )
                if offset % 256 == 0:
                    print(f"score {label}: {offset}/{len(ids)}", flush=True)
            output["variants"][label] = sums
            np.savez_compressed(
                args.output / f"raw-{label}-{args.worker}.npz",
                logprobs=np.concatenate(raw_scores),
            )
            write(args.output / f"scores-{args.worker}.json", output)
    finally:
        with torch.no_grad():
            for p, base in zip(params, before, strict=True):
                p.copy_(base)


def parent(args):
    import torch
    import wandb

    args.output.mkdir(parents=True, exist_ok=False)
    run = wandb.init(
        entity="wandb-japan",
        project="art-embodied-pi0-fast-spatial",
        name=f"pi0-fast-state-disjoint-update{args.update}-{'surface' if args.surface == 'full-lm' else 'epsilon' if args.checkpoint else 'lr'}-diagnostic",
        job_type="gradient-diagnostic",
        config={
            "scope": __doc__,
            "workers": str(args.workers),
            "new_rollouts": 0,
            "optimizer_continuation": False,
            "split_unit": "official task initial-state index",
            "policy_update": args.update,
            "optimizer_checkpoint": str(args.checkpoint),
            "surface": args.surface,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )
    write(args.output / "run.json", {"id": run.id, "url": run.url})
    run.define_metric("diagnostics/stage")
    run.define_metric("diagnostics/*", step_metric="diagnostics/stage")
    run.log({"diagnostics/stage": 0, "diagnostics/started": 1})
    processes, files = [], []
    exit_code = 1
    result = {"scope": __doc__}
    try:
        visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if len(visible) != 8:
            raise ValueError("Expected eight allocated GPUs on one node")
        for i, device in enumerate(visible):
            f = (args.output / f"worker-{i}.log").open("w")
            files.append(f)
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "examples.embodied.pi0_fast_group_crossfit_audit",
                        *sys.argv[1:],
                        "--worker",
                        str(i),
                    ],
                    env=os.environ | {"CUDA_VISIBLE_DEVICES": device},
                    stdout=f,
                    stderr=subprocess.STDOUT,
                )
            )
        deadline = time.monotonic() + 1500

        def await_paths(paths):
            while not all(p.exists() for p in paths):
                if (
                    any(p.poll() not in (None, 0) for p in processes)
                    or time.monotonic() > deadline
                ):
                    raise RuntimeError("Worker failed or exceeded the 25-minute bound")
                time.sleep(2)

        paths = [args.output / f"records-{i}.json" for i in range(8)]
        await_paths(paths)
        plan = make_plan([json.loads(p.read_text()) for p in paths])
        if len(plan["splits"]) != 512:
            raise ValueError("Incomplete retained trajectory population")
        write(args.output / "plan.json", plan)
        run.log(
            {
                "diagnostics/stage": 1,
                "diagnostics/fit_trajectories": plan["counts"]["fit"],
                "diagnostics/heldout_trajectories": plan["counts"]["heldout"],
                "diagnostics/distinct_initial_states": len(plan["initial_states"]),
            }
        )
        paths = [args.output / f"gradient-{i}.pt" for i in range(8)]
        await_paths(paths)
        payloads = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
        if any(p["names"] != payloads[0]["names"] for p in payloads):
            raise ValueError("Gradient parameter order differs")
        gradients = {
            k: sum(p["gradients"][k].double() for p in payloads)
            for k in payloads[0]["gradients"]
        }
        fit = gradients["fit_positive"] + gradients["fit_negative"]
        heldout = gradients["heldout_positive"] + gradients["heldout_negative"]

        def cosine(a, b):
            return float(a @ b) / max(float(a.norm() * b.norm()), 1e-30)

        result["gradient"] = {
            "fit_norm": float(fit.norm()),
            "heldout_norm": float(heldout.norm()),
            "fit_heldout_cosine": cosine(fit, heldout),
            "fit_positive_negative_cosine": cosine(
                gradients["fit_positive"], gradients["fit_negative"]
            ),
            "heldout_positive_negative_cosine": cosine(
                gradients["heldout_positive"], gradients["heldout_negative"]
            ),
            "positive_cross_split_cosine": cosine(
                gradients["fit_positive"], gradients["heldout_positive"]
            ),
            "negative_cross_split_cosine": cosine(
                gradients["fit_negative"], gradients["heldout_negative"]
            ),
        }
        from safetensors.torch import load_file

        if args.surface == "full-lm":
            reports = [
                json.loads((args.output / f"surface-{i}.json").read_text())
                for i in range(8)
            ]
            if any(r != reports[0] for r in reports):
                raise ValueError("Workers have different expanded initial adapters")
            result["surface"] = reports[0]
            weights = load_file(args.output / "initial-full.safetensors")
        else:
            weights = load_file(
                args.workers
                / f"snapshots/update-{args.update:04d}/initial/adapter_model.safetensors"
            )
        w = torch.cat(
            [
                weights[n.removeprefix("model.").replace(".default.weight", ".weight")]
                .flatten()
                .double()
                for n in payloads[0]["names"]
            ]
        )
        deltas = {"zero_control": torch.zeros_like(w, dtype=torch.float32)}
        result["predictions"] = {}
        for label, lr, eps, g in (
            ("lr2e-6", 2e-6, 1e-5, fit),
            ("lr1e-5", 1e-5, 1e-5, fit),
            ("lr2.5e-5", 2.5e-5, 1e-5, fit),
            ("eps1e-8", 2e-6, 1e-8, fit),
            ("positive_only", 2e-6, 1e-5, gradients["fit_positive"]),
        ):
            clipped = g * min(1.0, 1.0 / (float(g.norm()) + 1e-6))
            delta = -lr * clipped / (clipped.abs() + eps) - lr * 0.01 * w
            deltas[label] = delta.float()
            result["predictions"][label] = {
                "lr": lr,
                "epsilon": eps,
                "fit_gain": float(-fit @ delta),
                "heldout_gain": float(-heldout @ delta),
                "heldout_positive_gain": float(-gradients["heldout_positive"] @ delta),
                "heldout_negative_gain": float(-gradients["heldout_negative"] @ delta),
                "delta_norm": float(delta.norm()),
            }
        if args.surface == "full-lm":
            names = payloads[0]["names"]
            masks = {
                role: torch.cat(
                    [
                        torch.full(
                            (
                                weights[
                                    n.removeprefix("model.").replace(
                                        ".default.weight", ".weight"
                                    )
                                ].numel(),
                            ),
                            (".q_proj." in n or ".v_proj." in n)
                            if role == "qv"
                            else ".mlp." in n,
                            dtype=torch.bool,
                        )
                        for n in names
                    ]
                )
                for role in ("qv", "mlp")
            }
            deltas, scales = surface_variants(w, gradients, masks["qv"], masks["mlp"])
            result["matched_fit_scales"] = scales
            result["predictions"] = {
                label: {
                    "fit_gain": float(-fit @ d),
                    "heldout_gain": float(-heldout @ d),
                    "heldout_positive_gain": float(-gradients["heldout_positive"] @ d),
                    "delta_norm": float(d.norm()),
                }
                for label, d in deltas.items()
            }
            result["surface_gradient"] = {
                role: {
                    "fit_norm": float(fit[mask].norm()),
                    "heldout_norm": float(heldout[mask].norm()),
                    "fit_heldout_cosine": cosine(fit[mask], heldout[mask]),
                }
                for role, mask in masks.items()
            }
            deltas = {k: v.float() for k, v in deltas.items()}
        if args.checkpoint:
            await_paths([args.output / "history.pt"])
            history = torch.load(
                args.output / "history.pt", map_location="cpu", weights_only=True
            )
            if history["names"] != payloads[0]["names"]:
                raise ValueError("Optimizer moment ordering does not match gradients")
            role_a = torch.cat(
                [
                    torch.full(
                        (
                            weights[
                                n.removeprefix("model.").replace(
                                    ".default.weight", ".weight"
                                )
                            ].numel(),
                        ),
                        "lora_A" in n,
                        dtype=torch.bool,
                    )
                    for n in history["names"]
                ]
            )
            deltas, info = history_variants(w, gradients, history, role_a)
            result["role_updates"] = info
            result["optimizer_history_step"] = history["step"]
            result["predictions"] = {
                label: {
                    "fit_gain": float(-fit @ d),
                    "heldout_gain": float(-heldout @ d),
                    "delta_norm": float(d.norm()),
                }
                for label, d in deltas.items()
            }
            result["long_output_gradient"] = {
                s: {
                    "long_norm": float(gradients[s + "_long"].norm()),
                    "long_total_cosine": cosine(
                        gradients[s + "_long"],
                        gradients[s + "_positive"] + gradients[s + "_negative"],
                    ),
                }
                for s in ("fit", "heldout")
            }
        write(args.output / "result.json", result)
        torch.save(deltas, args.output / "deltas.tmp")
        (args.output / "deltas.tmp").replace(args.output / "deltas.pt")
        run.log(
            {
                "diagnostics/stage": 2,
                **{"diagnostics/" + k: v for k, v in result["gradient"].items()},
            }
        )
        paths = [args.output / f"scores-{i}.json" for i in range(8)]
        await_paths(paths)
        while any(p.poll() is None for p in processes):
            if (
                any(p.poll() not in (None, 0) for p in processes)
                or time.monotonic() > deadline
            ):
                raise RuntimeError("Scoring failed or timed out")
            time.sleep(2)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError("Scoring worker failed")
        payloads = [json.loads(p.read_text()) for p in paths]
        result["measured"] = {}
        for index, label in enumerate(deltas, start=3):
            metrics = {}
            for split in ("fit", "heldout"):
                sums = {
                    k: sum(p["variants"][label][split][k] for p in payloads)
                    for k in payloads[0]["variants"][label][split]
                }
                for k, v in sums.items():
                    metrics[split + "_" + k] = (
                        v / plan["counts"][split] if "tokens" not in k else v
                    )
                metrics[split + "_clip_fraction"] = sums["clipped_tokens"] / max(
                    1, sums["active_tokens"]
                )
            result["measured"][label] = metrics
            run.log(
                {
                    "diagnostics/stage": index,
                    **{"diagnostics/" + k: v for k, v in metrics.items()},
                }
            )
        exit_code = 0
    except BaseException as exc:
        result["error"] = repr(exc)
        raise
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        for f in files:
            f.close()
        write(args.output / "result.json", result)
        try:
            a = wandb.Artifact(
                "pi0-fast-state-disjoint-gradient-lr-audit", type="diagnostic"
            )
            a.add_file(str(args.output / "result.json"))
            a.add_file(__file__)
            if (args.output / "plan.json").exists():
                a.add_file(str(args.output / "plan.json"))
            run.log_artifact(a).wait()
            run.summary["diagnostics/completed"] = exit_code == 0
        finally:
            run.finish(exit_code=exit_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--update", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--surface", choices=("qv", "full-lm"), default="qv")
    args = parser.parse_args()
    if args.update < 0 or (args.update != 0 and args.checkpoint is None):
        parser.error("Noninitial policies require their matching optimizer checkpoint")
    if args.surface == "full-lm" and (args.update != 0 or args.checkpoint is not None):
        parser.error("Surface comparison is restricted to the initial policy")
    worker(args) if args.worker is not None else parent(args)
