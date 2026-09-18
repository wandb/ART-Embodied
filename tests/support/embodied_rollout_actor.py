"""CPU-only rollout actor imported by process-pool tests."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from art_embodied.integrations.lerobot import LeRobotPolicyAdapter
from art_embodied.integrations.lerobot_process import LeRobotProcessComponents
from art_embodied.trajectories import Action, EmbodiedTrajectory, Observation


class FakeRolloutActor:
    def __init__(self, *, context) -> None:
        self.context = context
        self.policy_version: int | None = None

    def prepare_update(self, *, update: int, policy_snapshot: Path) -> None:
        recorded = int((policy_snapshot / "policy_version.txt").read_text())
        if recorded != update:
            raise RuntimeError(
                f"Snapshot/update mismatch: recorded={recorded}, update={update}"
            )
        self.policy_version = recorded

    async def rollout(
        self,
        scenario,
        context,
        *,
        phase: str,
        policy_client=None,
    ):
        await asyncio.sleep(float(scenario.payload.get("sleep_seconds", 0.05)))
        inference = None
        if policy_client is not None:
            inference = await policy_client.predict(
                {
                    "scenario_id": scenario.id,
                    "attempt_index": context.attempt_index,
                }
            )
        return EmbodiedTrajectory(
            task=scenario.task,
            reward=float(context.attempt_index % 2),
            metrics={"success": context.attempt_index % 2 == 0},
            metadata={
                "fixture_actor": self.context.worker_index,
                "fixture_device": self.context.local_device,
                "fixture_policy_version": self.policy_version,
                "fixture_phase": phase,
                "fixture_inference": inference,
            },
        ).finish()

    def close(self) -> None:
        return None


def create_actor(*, config, context):
    del config
    return FakeRolloutActor(context=context)


class CrashingRolloutActor(FakeRolloutActor):
    async def rollout(self, scenario, *args, **kwargs):
        if scenario.payload.get("crash_worker"):
            os._exit(17)
        return await super().rollout(scenario, *args, **kwargs)


def create_crashing_actor(*, config, context):
    del config
    return CrashingRolloutActor(context=context)


class FakeObservationRolloutActor(FakeRolloutActor):
    async def rollout(self, *args, **kwargs):
        trajectory = await super().rollout(*args, **kwargs)
        scenario = args[0]
        observation_bytes = int(scenario.payload.get("observation_bytes", 1024))
        trajectory.observations.append(
            Observation(
                step=0,
                kind="image",
                value={"pixels": bytearray(observation_bytes)},
                metadata={"camera": "agentview"},
            )
        )
        return trajectory


def create_observation_actor(*, config, context):
    del config
    return FakeObservationRolloutActor(context=context)


class FakeBatchedInferenceEngine:
    def __init__(self, *, context) -> None:
        self.context = context
        self.policy_version: int | None = None
        self.offload_count = 0

    def prepare_update(self, *, update: int, policy_snapshot: Path) -> None:
        recorded = int((policy_snapshot / "policy_version.txt").read_text())
        if recorded != update:
            raise RuntimeError(
                f"Inference snapshot mismatch: recorded={recorded}, update={update}"
            )
        self.policy_version = recorded

    async def predict_batch(self, requests):
        await asyncio.sleep(0.02)
        results = []
        for request in requests:
            result = {
                "batch_size": len(requests),
                "policy_version": self.policy_version,
                "server_index": self.context.worker_index,
                "offload_count": self.offload_count,
            }
            if "attempt_indices" in request:
                result["attempt_indices"] = list(request["attempt_indices"])
            else:
                result["attempt_index"] = request["attempt_index"]
            results.append(result)
        return results

    def offload(self):
        self.offload_count += 1
        return {"offloaded": True, "offload_count": self.offload_count}

    def close(self) -> None:
        return None


def create_inference_engine(*, config, context):
    del config
    return FakeBatchedInferenceEngine(context=context)


class _FixturePolicy:
    def __init__(self, *, label: str) -> None:
        self.label = label
        self.version: int | None = None
        self.offload_count = 0
        self.restore_count = 0


class _FixtureEnvironment:
    def reset(self, *, seed, options):
        return {"state": [seed], "options": str(options)}, {"seed": seed}

    def step(self, native_action):
        version = int(native_action[0])
        return (
            {"state": [version]},
            float(version + 1),
            True,
            False,
            {"is_success": True},
        )

    def close(self) -> None:
        return None


def create_lerobot_components(*, config, context, label: str):
    del config, context
    policy = _FixturePolicy(label=label)

    def load_policy_snapshot(*, update: int, policy_snapshot: Path) -> None:
        recorded = int((policy_snapshot / "policy_version.txt").read_text())
        if recorded != update:
            raise RuntimeError(
                f"Snapshot/update mismatch: recorded={recorded}, update={update}"
            )
        policy.version = recorded

    def predict_action(**_kwargs):
        if policy.version is None:
            raise RuntimeError(
                "Fixture policy was used before snapshot synchronization"
            )
        return [policy.version]

    def record_action(*, native_action, step, **_kwargs):
        return Action(
            step=step,
            kind="token",
            raw={"tokens": [int(native_action[0])]},
            decoded=native_action,
            logprobs={"token_logprobs": [-0.1]},
            metadata={
                "fixture_label": policy.label,
                "fixture_offload_count": policy.offload_count,
                "fixture_restore_count": policy.restore_count,
            },
        )

    def offload() -> None:
        policy.offload_count += 1

    def restore() -> None:
        policy.restore_count += 1

    def policy_adapter_factory(_scenario, _rollout_context):
        return LeRobotPolicyAdapter(
            policy=policy,
            preprocessor=None,
            postprocessor=None,
            device="cpu",
            record_action_fn=record_action,
            seed_fn=lambda _seed: None,
            _predict_action_fn=predict_action,
        )

    return LeRobotProcessComponents(
        environment_factory=lambda _scenario, _context: _FixtureEnvironment(),
        policy_adapter_factory=policy_adapter_factory,
        load_policy_snapshot=load_policy_snapshot,
        offload=offload,
        restore=restore,
    )


def create_shared_lerobot_components(*, config, context, label: str):
    del config, context

    async def group_rollout(*, scenario, contexts, phase, policy_client):
        if policy_client is None:
            raise RuntimeError("shared fixture requires policy_client")
        inference = await policy_client.predict(
            {"attempt_indices": [item.attempt_index for item in contexts]}
        )
        return [
            EmbodiedTrajectory(
                task=scenario.task,
                reward=float(item.attempt_index % 2),
                metrics={"success": item.attempt_index % 2 == 0},
                metadata={
                    "fixture_label": label,
                    "fixture_phase": phase,
                    "fixture_inference": inference,
                    "attempt_index": item.attempt_index,
                },
            ).finish()
            for item in contexts
        ]

    def embedded_only_adapter(*_args, **_kwargs):
        raise RuntimeError("shared fixture must not create an embedded adapter")

    return LeRobotProcessComponents(
        environment_factory=lambda _scenario, _context: None,
        policy_adapter_factory=embedded_only_adapter,
        load_policy_snapshot=lambda **_kwargs: None,
        group_rollout=group_rollout,
    )
