"""Run corrected-reset GRPO with automatic in-allocation W&B acceptance."""

import argparse
import asyncio
import json
from pathlib import Path
import shutil
import time

from art_embodied.checkpointing import CheckpointManager
from art_embodied.config import EmbodiedExperimentConfig
from art_embodied.observability import (
    WandbWeaveObserver,
    _wandb_training_history_payload,
)
from art_embodied.wandb_history_contract import (
    expected_training_history,
    verify_run_history,
)
from examples.embodied.libero.state_manifest import file_sha256
from examples.embodied.pi0_fast_spatial_teacher_control import write


class VerifiedObserver(WandbWeaveObserver):
    async def log_step(self, step, groups, train_result, evaluation, config):
        await super().log_step(step, groups, train_result, evaluation, config)
        self.last_verified_step = step
        if step == 1 or step % 5 == 0:
            self.verify(step, train_result)

    def verify(self, step, train_result=None):
        import wandb

        output = self.config.storage.output_dir
        native_steps = self.config.observability.wandb.native_update_steps
        expected = expected_training_history(
            output, last_policy_version=step - 1, completed_update_steps=native_steps
        )
        evaluation_steps = []
        for path in sorted((output / "evaluation").glob("update_*_evidence.json")):
            version = int(path.name.split("_")[1])
            if version <= step:
                evaluation_steps.append(version)
                evidence = json.loads(path.read_text())
                expected.setdefault(version, {})["validation/success_rate"] = evidence[
                    "metrics"
                ]["success_rate"]
        if train_result is not None:
            expected.setdefault(step, {}).update(
                {
                    k: v
                    for k, v in _wandb_training_history_payload(
                        train_result.metrics
                    ).items()
                    if not isinstance(v, bool)
                }
            )
        checkpoint = output / f"checkpoints/step_{step:06d}"
        CheckpointManager().validate_payload(checkpoint)
        deadline = time.monotonic() + 180
        while True:
            try:
                remote = wandb.Api(timeout=30).run(self.wandb_run.path)
                report = verify_run_history(
                    remote, expected, require_native_steps=native_steps
                )
                if not report["storage_verified"]:
                    raise ValueError("W&B history differs from measured results")
                videos = [
                    f for f in remote.files() if f.name.startswith("media/videos/")
                ]
                if (
                    len(videos)
                    < self.config.observability.videos_per_update
                    + self.config.observability.videos_per_evaluation
                ):
                    raise ValueError("Required train/evaluation videos missing")
                artifacts = list(remote.logged_artifacts())
                models = [
                    a
                    for a in artifacts
                    if a.type == "model" and a.metadata.get("update") == step
                ]
                if len(models) != 1 or not any(
                    a.type == "evaluation" for a in artifacts
                ):
                    raise ValueError("Required model/evaluation artifact missing")
                # Read back exact model bytes; do not mistake enqueued upload for delivery.
                local = checkpoint / "policy/adapter_model.safetensors"
                if not local.exists():
                    local = checkpoint / "adapter_model.safetensors"
                key = str(local.relative_to(checkpoint))
                downloaded = (
                    models[0]
                    .get_entry(key)
                    .download(root=str(output / "acceptance/model"))
                )
                if file_sha256(downloaded) != file_sha256(local):
                    raise ValueError("Model artifact bytes differ")
                for file in (videos[0], videos[-1]):
                    handle = file.download(
                        root=str(output / "acceptance/media"), replace=True
                    )
                    path = Path(handle.name)
                    handle.close()
                    import av

                    with av.open(str(path)) as container:
                        next(container.decode(video=0))
                if native_steps:
                    from art_embodied.wandb_media_contract import (
                        VideoExpectation,
                        verify_run_videos,
                    )

                    expectations = [
                        VideoExpectation(
                            step,
                            "media/simulation/train",
                            self.config.observability.videos_per_update,
                        )
                    ]
                    if evaluation_steps:
                        expectations.append(
                            VideoExpectation(
                                max(evaluation_steps),
                                "media/simulation/eval",
                                self.config.observability.videos_per_evaluation,
                            )
                        )
                    media = verify_run_videos(
                        remote,
                        expectations,
                        download_root=output / "acceptance/native-media",
                        require_native_steps=True,
                    )
                    write(output / f"acceptance/native-media-{step:06d}.json", media)
                    if not media["media_verified"]:
                        raise ValueError("W&B native media steps or bytes differ")
                    report["native_media_steps_verified"] = True
                report.update(
                    first_update_verified=step >= 1,
                    model_bytes_verified=True,
                    video_decode_verified=True,
                    human_confirmation_required=False,
                )
                write(output / f"acceptance/update-{step:06d}.json", report)
                print(
                    f"Automatic W&B acceptance passed at update {step}; App rendering not asserted",
                    flush=True,
                )
                return
            except Exception as exc:
                write(
                    output / "acceptance/latest-attempt.json",
                    {"step": step, "error": repr(exc)},
                )
                if time.monotonic() >= deadline:
                    raise
                time.sleep(10)

    def close(self, *, exit_code=0):
        super().close(exit_code=exit_code)
        if not exit_code and getattr(self, "last_verified_step", 0):
            self.verify(self.last_verified_step)


def publish_sft(root, config):
    complete = json.loads((root / "sft/complete.json").read_text())
    if (
        complete["updates"] != 100
        or not (root / "sft/FIRST_UPDATE_VERIFIED.json").exists()
    ):
        raise ValueError("SFT is not complete and verified")
    initial = root / "sft/checkpoint-0100"
    destination = root / "sft-warm-start"
    CheckpointManager().publish(
        destination,
        writer=lambda staging: shutil.copytree(initial, staging / "policy"),
        config_fingerprint=config.fingerprint,
        resume_contract_fingerprint=config.resume_contract_fingerprint,
        metadata={
            "backend": "pi0_fast_language_sft",
            "step": 100,
            "task_index": 32,
            "source": "native task5 teacher control; canonical language, no paraphrases",
            "source_sha256": file_sha256(initial / "adapter_model.safetensors"),
        },
    )


def main(root):
    hold = root / "HOLD_BEFORE_GRPO"
    if hold.exists():
        status = {"status": "held_before_grpo", "reason": hold.read_text().strip()}
        write(root / "grpo-held.json", status)
        print(json.dumps(status), flush=True)
        return

    from examples.embodied.libero import train_openvla_oft as entry

    path = root / "grpo.yaml"
    config = EmbodiedExperimentConfig.from_yaml(path)
    publish_sft(root, config)
    original = entry.WandbWeaveObserver
    entry.WandbWeaveObserver = VerifiedObserver
    try:
        asyncio.run(entry.run(path))
    finally:
        entry.WandbWeaveObserver = original


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    main(p.parse_args().root.resolve())
