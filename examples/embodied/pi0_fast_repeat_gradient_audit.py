"""Same-reset independent G=8 gradient replication at the unchanged initial policy.

Use the first 16 retained groups, selected without rewards. Collect one new group
per start with different policy seeds. No optimizer step or sealed evaluation.
"""

import argparse
import asyncio
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

from examples.embodied.pi0_fast_group_crossfit_audit import write
from examples.embodied.pi0_fast_pipeline_provenance import observation_digest


def new_seeds(group, old):
    seeds = [
        int.from_bytes(
            hashlib.sha256(f"repeat-20260908/{group}/{i}".encode()).digest()[:4], "big"
        )
        % (2**31)
        for i in range(8)
    ]
    if len(set(seeds)) != 8 or set(seeds).intersection(old):
        raise ValueError("Policy seeds must be distinct from retained seeds")
    return seeds


def plan(args):
    manifest = json.loads(args.manifest.read_text())
    groups = sorted(manifest["groups"], key=lambda g: g["group_index"])
    if [g["group_index"] for g in groups] != list(range(64)):
        raise ValueError("Expected complete retained population")
    old_seeds = [seed for g in groups for seed in g["policy_seeds"]]
    selected = []
    for group in groups[:16]:
        source = Path(group["source"])
        with source.with_name("request.pkl").open("rb") as stream:
            request = pickle.load(stream)
        contexts = request["contexts"]
        if (
            request["phase"] != "train"
            or len(contexts) != 8
            or any(c["update"] != 0 for c in contexts)
        ):
            raise ValueError("Wrong retained rollout coordinates")
        if [c["policy_seed"] for c in contexts] != group["policy_seeds"]:
            raise ValueError("Manifest seeds differ from source")
        selected.append(
            group | {"new_policy_seeds": new_seeds(group["group_index"], old_seeds)}
        )
    if len({s for g in selected for s in g["new_policy_seeds"]}) != 128:
        raise ValueError("New seeds overlap across groups")
    result = {
        "groups": selected,
        "scope": __doc__,
        "selection": "first 16 group IDs, no reward filtering",
    }
    if getattr(args, "pooled_source", None):
        from examples.embodied.pi0_fast_grouping_comparison import pooling_plan

        result = pooling_plan(result, args.pooled_source, old_seeds)
    if getattr(args, "bos_source", None):
        if getattr(args, "pooled_source", None) or getattr(args, "reuse", None):
            raise ValueError("BOS control cannot be combined with pooling or reuse")
        from examples.embodied.pi0_fast_bos_gradient_control import aligned_plan

        result = aligned_plan(result, args.bos_source)
    return result


def vector(payload):
    import torch

    values = payload["gradients"]
    if not values:
        raise ValueError("Empty gradient surface")
    result = torch.cat([values[n].double().flatten() for n in sorted(values)])
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite gradient")
    return result


def cosine(a, b):
    if a.norm() == 0 or b.norm() == 0:
        return None
    return float(a @ b / (a.norm() * b.norm()))


def check_starts(trajectories, expected_hash, expected_fingerprints=None):
    if len(trajectories) != 8:
        raise ValueError("Incomplete G=8 rollout")
    for t in trajectories:
        if observation_digest(t.observations[0].value) != expected_hash:
            raise ValueError("Initial observation differs from retained group")
        if (
            expected_fingerprints is not None
            and t.metadata["reset_info"]["reset_fingerprints"] != expected_fingerprints
        ):
            raise ValueError("New reset state/model fingerprints differ")


def diagnostic_video(output, wandb_module=None):
    from art_embodied.media import wandb_video_payload_from_paths

    entries = [
        {"path": str(p)}
        for p in sorted((output / "runtime-0/videos").rglob("*"))
        if p.is_file() and p.suffix.lower() in (".gif", ".mp4", ".webm", ".ogg")
    ]
    payload = wandb_video_payload_from_paths(
        entries, prefix="media/rollout", max_videos=1, wandb_module=wandb_module
    )
    if not payload:
        raise ValueError("No recorded diagnostic video")
    return payload


def reuse_completed(output, previous):
    import torch

    current_plan = json.loads((output / "plan.json").read_text())
    if current_plan != json.loads((previous / "plan.json").read_text()):
        raise ValueError("Cannot reuse pairs from a different plan")
    reused = []
    extra_sides = ("separate",) if "pooling_source" in current_plan else ()
    for path in sorted(previous.glob("pair-*.json")):
        row = json.loads(path.read_text())
        gid = row["group_index"]
        if not row["policy_weights_unchanged"]:
            raise ValueError("Cannot reuse changed weights")
        for side in ("original", "repeated", *extra_sides):
            vector(
                torch.load(
                    previous / f"{side}-{gid}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
            )
        for name in (
            path.name,
            f"original-{gid}.pt",
            f"repeated-{gid}.pt",
            f"repeated-{gid}.pkl",
            *(f"{side}-{gid}.pt" for side in extra_sides),
        ):
            shutil.copy2(previous / name, output / name)
        reused.append(gid)
    if 0 in reused:
        shutil.copytree(previous / "runtime-0/videos", output / "runtime-0/videos")
    write(output / "reuse.json", {"source": str(previous.resolve()), "groups": reused})
    return reused


def gradient(policy, backend, trajectories, output):
    import numpy as np
    import torch

    from art_embodied.backends.action_token import prepare_action_token_examples
    from art_embodied.backends.action_token_gradients import _trainable_gradient_payload
    from art_embodied.backends.local_process import _attach_full_update_advantages
    from art_embodied.trajectories import EmbodiedTrajectoryGroup

    examples, _ = prepare_action_token_examples(
        [EmbodiedTrajectoryGroup(trajectories)], backend=backend
    )
    _attach_full_update_advantages(examples, backend=backend)
    rewards = np.array([t.reward for t in trajectories])
    expected = (rewards - rewards.mean()) / (
        rewards.std(ddof=1) + backend.advantage_epsilon
    )
    for e in examples:
        if not np.allclose(
            e.metadata["token_advantages"],
            expected[e.metadata["trajectory_index_in_group"]],
            rtol=0,
            atol=2e-6,
        ):
            raise ValueError("Group advantage disagrees with independent equation")
    started = time.monotonic()
    result = asyncio.run(
        backend.train(
            [],
            _action_token_grpo_precomputed_examples=examples,
            _action_token_grpo_precomputed_examples_prepared=True,
            _action_token_grpo_global_example_count=len(trajectories),
            _action_token_grpo_return_gradients=True,
        )
    )
    payload = _trainable_gradient_payload(policy)
    flat = vector(payload)
    active = len(set(rewards.tolist())) > 1
    if active and (flat.norm() == 0 or payload["missing_gradients"]):
        raise ValueError("Active group produced missing gradients")
    if not active and flat.norm() != 0:
        raise ValueError("Zero-advantage group produced nonzero gradient")
    torch.save(payload, output)
    return flat, {
        "rewards": rewards.tolist(),
        "active": active,
        "norm": float(flat.norm()),
        "examples": len(examples),
        "seconds": time.monotonic() - started,
        "metrics": result.metrics,
    }


def worker(args):
    from art_embodied.backends.action_token_worker import _build_runtime
    from examples.embodied.pi0_fast_teacher_path_control import bos_boundary

    spec = json.loads((args.workers / "worker-00/bootstrap.json").read_text())
    snapshot = args.workers / "snapshots/update-0000/initial"
    spec["policy_snapshot"] = str(snapshot.resolve())
    spec["config"]["storage"]["output_dir"] = str(
        (args.output / f"runtime-{args.worker}").resolve()
    )
    config, policy, backend = _build_runtime(spec)
    if backend.precalculate_logprobs or backend.importance_sampling_level != "token":
        raise ValueError(
            "Requires unmodified native token ratio and actual rollout old logprobs"
        )
    with bos_boundary(policy.policy.model, "sft" if args.bos_source else "sampler"):
        worker_groups(args, config, policy, backend, snapshot)


def worker_groups(args, config, policy, backend, snapshot):
    from art_embodied.experiment import EmbodiedScenario, RolloutContext
    from art_embodied.integrations.pi0_fast import PI0FastPolicyAdapter
    from examples.embodied.libero.environment import LiberoTaskCatalog
    from examples.embodied.libero.rollout import rollout_libero_group
    from examples.embodied.libero.settings import LiberoSettings
    from examples.embodied.pi0_fast_pipeline_audit import snapshot_comparison

    policy.eval()
    policy.rollout_update = 0
    snapshot_comparison(policy, snapshot)
    settings = LiberoSettings.from_config(config)
    catalog = LiberoTaskCatalog(settings)
    selected = json.loads((args.output / "plan.json").read_text())["groups"]
    for row in selected[args.worker :: 8]:
        gid = row["group_index"]
        if (args.output / f"pair-{gid}.json").exists():
            print(f"Reusing completed pair {gid}", flush=True)
            continue
        if gid >= 8:
            deadline = time.monotonic() + 1200
            while not (args.output / "CONTINUE").exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("First-wave observability gate not accepted")
                time.sleep(2)
        source = Path(row["source"])
        trajectory_source = (
            args.bos_source / f"changed-{gid}.pkl" if args.bos_source else source
        )
        with trajectory_source.open("rb") as stream:
            original = pickle.load(stream)
        if args.bos_source:
            if [t.metadata["policy_seed"] for t in original] != row[
                "bos_retained_policy_seeds"
            ]:
                raise ValueError("Retained BOS trajectory seeds differ")
        check_starts(original, row["initial_observation_sha256"])
        if args.pooled_source:
            with (args.pooled_source / f"repeated-{gid}.pkl").open("rb") as stream:
                additional = pickle.load(stream)
            check_starts(additional, row["initial_observation_sha256"])
            if [t.metadata["policy_seed"] for t in additional] != row[
                "pooled_policy_seeds"
            ]:
                raise ValueError("Pooling source seeds differ")
            original.extend(additional)
            del additional
        with source.with_name("request.pkl").open("rb") as stream:
            request = pickle.load(stream)
        contexts = tuple(RolloutContext(**c) for c in request["contexts"])
        scenario = EmbodiedScenario.model_validate(request["scenario"])
        env = catalog.make_environment(scenario, contexts[0])
        try:
            obs, info = env.reset(
                seed=contexts[0].environment_seed,
                options=scenario.payload["reset_options"],
            )
            if observation_digest(obs) != row["initial_observation_sha256"]:
                raise ValueError("Fresh reset does not reproduce retained observation")
        finally:
            env.close()
        contexts = tuple(
            replace(c, policy_seed=s)
            for c, s in zip(contexts, row["new_policy_seeds"], strict=True)
        )
        generation = config.policy.rollout_generation
        adapter = PI0FastPolicyAdapter(
            policy=policy,
            robot_type="panda",
            sampling_mode="train",
            do_sample=generation.do_sample,
            temperature=generation.temperature,
            compute_rollout_logprobs=True,
            action_decoder="native",
            invalid_action_handling="terminate_episode",
            model_batch_size=int(
                config.runtime.rollout_execution.inference_max_batch_size
            ),
        )
        started = time.monotonic()
        repeated = asyncio.run(
            rollout_libero_group(
                config=config,
                policy=policy,
                catalog=catalog,
                settings=settings,
                scenario=scenario,
                contexts=contexts,
                phase="train",
                embedded_batch_predictor=adapter.predict_batch,
                embedded_batch_reset=lambda seed: adapter.reset(seed=seed),
            )
        )
        rollout_seconds = time.monotonic() - started
        check_starts(
            repeated, row["initial_observation_sha256"], info["reset_fingerprints"]
        )
        with (args.output / f"repeated-{gid}.pkl").open("wb") as stream:
            pickle.dump(repeated, stream)
        a, left = gradient(
            policy, backend, original, args.output / f"original-{gid}.pt"
        )
        b, right = gradient(
            policy, backend, repeated, args.output / f"repeated-{gid}.pt"
        )
        extra = {}
        if args.pooled_source:
            from examples.embodied.pi0_fast_grouping_comparison import separate_gradient

            separate = separate_gradient(args.pooled_source, gid, args.output)
            extra = {
                "separate_gradient_cosine": cosine(separate, b),
                "pooled_vs_separate_cosine": cosine(a, separate),
            }
        snapshot_comparison(policy, snapshot)
        write(
            args.output / f"pair-{gid}.json",
            {
                "group_index": gid,
                "original": left,
                "repeated": right,
                "gradient_cosine": cosine(a, b),
                "rollout_seconds": rollout_seconds,
                "reset_fingerprints": info["reset_fingerprints"],
                "policy_weights_unchanged": True,
                **extra,
            },
        )
        print(
            f"pair {gid}: cosine={cosine(a, b)} rewards={left['rewards']} -> {right['rewards']}",
            flush=True,
        )
        del original, repeated, a, b


def aggregate(output):
    import torch

    rows = [json.loads((output / f"pair-{i}.json").read_text()) for i in range(16)]
    vectors = {
        side: [
            vector(
                torch.load(
                    output / f"{side}-{i}.pt", map_location="cpu", weights_only=False
                )
            )
            for i in range(16)
        ]
        for side in ("original", "repeated")
    }
    a, b = vectors["original"], vectors["repeated"]
    return {
        "scope": __doc__,
        "pairs": rows,
        "same_start_cosines": [cosine(x, y) for x, y in zip(a, b, strict=True)],
        "different_start_cosines": [[cosine(x, y) for y in b] for x in a],
        "aggregate_gradient_cosine": cosine(sum(a), sum(b)),
        "original_active_groups": sum(r["original"]["active"] for r in rows),
        "repeated_active_groups": sum(r["repeated"]["active"] for r in rows),
        "both_active_groups": sum(
            r["original"]["active"] and r["repeated"]["active"] for r in rows
        ),
        "root_cause_confirmed": False,
    }


def verify(run_id, records, *, finished=False):
    import wandb

    from examples.embodied.pi0_fast_chunk_optimizer_audit import verify_rows

    deadline = time.monotonic() + 180
    while True:
        try:
            remote = wandb.Api(timeout=30).run(
                f"wandb-japan/art-embodied-pi0-fast-spatial/{run_id}"
            )
            verify_rows(list(remote.scan_history()), records)
            if finished and remote.state != "finished":
                raise ValueError("Diagnostic not finished")
            if not any(f.name.startswith("media/videos/") for f in remote.files()):
                raise ValueError("Diagnostic video not uploaded")
            return remote
        except Exception:
            if time.monotonic() > deadline:
                raise
            time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workers", "manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--reuse", type=Path)
    parser.add_argument("--pooled-source", type=Path)
    parser.add_argument("--bos-source", type=Path)
    args = parser.parse_args()
    if args.worker is not None:
        return worker(args)
    selected = plan(args)
    if args.preflight:
        print(
            json.dumps(
                {
                    "groups": len(selected["groups"]),
                    "new_rollouts": 128,
                    "selection": selected["selection"],
                }
            )
        )
        return
    import wandb

    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if (
        len(devices) != 8
        or not all(d.isdigit() for d in devices)
        or os.environ.get("SLURM_JOB_NUM_NODES") != "1"
    ):
        raise ValueError("Requires one Slurm node with eight indexed GPUs")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "plan.json", selected)
    reused = reuse_completed(args.output, args.reuse) if args.reuse else []
    run = wandb.init(
        entity="wandb-japan",
        project="art-embodied-pi0-fast-spatial",
        name=f"pi0-fast-{'bos-repeat' if args.bos_source else 'grouping' if args.pooled_source else 'repeat'}-gradient-{os.environ['SLURM_JOB_ID']}",
        job_type="grouping-diagnostic"
        if args.pooled_source
        else "gradient-replication-diagnostic",
        config={
            "scope": selected["scope"],
            "groups": 16,
            "group_size": 8,
            "new_rollouts": 128,
            "optimizer_steps": 0,
            "reused_groups": reused,
            "gradient_reference_group_size": 16 if args.pooled_source else 8,
            "pooled_source": str(args.pooled_source) if args.pooled_source else None,
            "bos_source": str(args.bos_source) if args.bos_source else None,
            "bos_boundary": "sft" if args.bos_source else "sampler",
        },
    )
    run.define_metric("diagnostics/stage")
    run.define_metric("diagnostics/*", step_metric="diagnostics/stage")
    write(args.output / "run.json", {"id": run.id, "url": run.url})
    records, processes, logs = [], [], []

    def log(row, **extra):
        row = {"diagnostics/stage": len(records)} | row
        records.append(row)
        write(args.output / "history.json", records)
        run.log(row | extra)

    code = 1
    try:
        log({"diagnostics/started": 1})
        for i, device in enumerate(devices):
            stream = (args.output / f"worker-{i}.log").open("w")
            logs.append(stream)
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "examples.embodied.pi0_fast_repeat_gradient_audit",
                        *sys.argv[1:],
                        "--worker",
                        str(i),
                    ],
                    env=os.environ
                    | {"CUDA_VISIBLE_DEVICES": device, "MUJOCO_EGL_DEVICE_ID": device},
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            )
        seen = set()
        deadline = time.monotonic() + 2400
        for wave in range(2):
            while len(seen) < (wave + 1) * 8:
                if time.monotonic() > deadline or any(
                    p.poll() not in (None, 0) for p in processes
                ):
                    raise RuntimeError(
                        "Diagnostic worker failed or exceeded time budget"
                    )
                for gid in range(wave * 8, (wave + 1) * 8):
                    path = args.output / f"pair-{gid}.json"
                    if gid in seen or not path.exists():
                        continue
                    row = json.loads(path.read_text())
                    extra = {}
                    if gid == 0:
                        extra = diagnostic_video(args.output)
                    values = {
                        "diagnostics/group_index": gid,
                        "diagnostics/original_success_rate": sum(
                            row["original"]["rewards"]
                        )
                        / len(row["original"]["rewards"]),
                        "diagnostics/repeated_success_rate": sum(
                            row["repeated"]["rewards"]
                        )
                        / len(row["repeated"]["rewards"]),
                        "diagnostics/rollout_seconds": row["rollout_seconds"],
                        "diagnostics/gradient_seconds": row["original"]["seconds"]
                        + row["repeated"]["seconds"],
                        "diagnostics/both_active": int(
                            row["gradient_cosine"] is not None
                        ),
                    }
                    if row["gradient_cosine"] is not None:
                        values["diagnostics/gradient_cosine"] = row["gradient_cosine"]
                    for key in (
                        "separate_gradient_cosine",
                        "pooled_vs_separate_cosine",
                    ):
                        if row.get(key) is not None:
                            values[f"diagnostics/{key}"] = row[key]
                    log(values, **extra)
                    seen.add(gid)
                time.sleep(2)
            verify(run.id, records)
            if wave == 0:
                write(
                    args.output / "first-wave-verification.json",
                    {
                        "history_verified": True,
                        "video_uploaded": True,
                        "rendering_verified": False,
                    },
                )
                (args.output / "CONTINUE").touch()
        for p in processes:
            if p.wait(timeout=30) != 0:
                raise RuntimeError("Worker shutdown failed")
        result = aggregate(args.output)
        comparison_metrics = {}
        if args.pooled_source:
            from examples.embodied.pi0_fast_grouping_comparison import compare

            result["scope"] = selected["scope"]
            result["grouping_comparison"] = compare(args.output, args.workers)
            comparison_metrics = {
                f"diagnostics/{key}": value
                for key, value in result["grouping_comparison"].items()
                if isinstance(value, (int, float))
            }
        write(args.output / "result.json", result)
        log(
            {
                "diagnostics/aggregate_gradient_cosine": result[
                    "aggregate_gradient_cosine"
                ],
                "diagnostics/both_active_groups": result["both_active_groups"],
                **comparison_metrics,
            }
        )
        # Do not close the producer with an unverified final history row.
        verify(run.id, records)
        code = 0
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        for stream in logs:
            stream.close()
        artifact = wandb.Artifact(
            f"pi0-fast-repeat-gradient-{run.id}", type="diagnostic"
        )
        for path in args.output.glob("*.json"):
            artifact.add_file(str(path), name=path.name)
        for path in args.output.glob("*.pt"):
            artifact.add_file(str(path), name=path.name)
        artifact.add_file(__file__, name="audit.py")
        if args.bos_source:
            for name in (
                "pi0_fast_bos_gradient_control.py",
                "pi0_fast_teacher_path_control.py",
            ):
                artifact.add_file(str(Path(__file__).with_name(name)), name=name)
        if args.pooled_source:
            artifact.add_file(
                str(Path(__file__).with_name("pi0_fast_grouping_comparison.py")),
                name="grouping.py",
            )
        run.log_artifact(artifact).wait()
        run.finish(exit_code=code)
    remote = verify(run.id, records, finished=True)
    saved = next(a for a in remote.logged_artifacts() if a.type == "diagnostic")
    if saved.state != "COMMITTED":
        raise ValueError("Uncommitted diagnostic artifact")
    path = Path(
        saved.get_entry("result.json").download(root=str(args.output / "remote-result"))
    )
    if json.loads(path.read_text()) != result:
        raise ValueError("Artifact result differs")
    write(
        args.output / "remote-verification.json",
        {
            "history_verified": True,
            "artifact_verified": True,
            "video_uploaded": True,
            "rendering_verified": False,
            "recovery_sync_used": False,
        },
    )


if __name__ == "__main__":
    main()
