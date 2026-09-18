from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from art_embodied.config import EmbodiedExperimentConfig
from art_embodied.experiment import EmbodiedScenario, RolloutContext
from art_embodied.integrations.lerobot import (
    LeRobotActionPrediction,
    LeRobotPolicyAdapter,
)
from art_embodied.integrations.lerobot_process import (
    LeRobotProcessComponents,
    LeRobotProcessRolloutActor,
)
from art_embodied.integrations.lerobot_rollout import LeRobotEpisodeRollout
from art_embodied.lookahead import LookaheadFrame
from art_embodied.trajectories import Action, EmbodiedTrajectory


class _ArrayLike:
    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return [0.1, -0.2, 0.3]


class _Policy:
    config = SimpleNamespace(type="smolvla")

    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


class _Resettable:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


def test_adapter_uses_lerobot_inference_contract_without_reprocessing() -> None:
    calls = []

    def predict_action(**kwargs):
        calls.append(kwargs)
        return _ArrayLike()

    policy = _Policy()
    preprocessor = _Resettable()
    postprocessor = _Resettable()
    seeds = []
    adapter = LeRobotPolicyAdapter(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device="cuda:0",
        use_amp=True,
        robot_type="so101",
        seed_fn=seeds.append,
        _predict_action_fn=predict_action,
    )

    adapter.reset(seed=11)
    action = adapter.act(
        {"observation.state": [1.0, 2.0]},
        task="pick up the cube",
        step=3,
        seed=12,
    )

    assert policy.reset_count == 1
    assert preprocessor.reset_count == 1
    assert postprocessor.reset_count == 1
    assert seeds == [11, 12]
    assert action.kind == "continuous"
    assert action.decoded == [0.1, -0.2, 0.3]
    assert action.metadata["framework"] == "lerobot"
    assert action.metadata["policy_type"] == "smolvla"
    assert action.metadata["policy_seed"] == 12
    assert calls == [
        {
            "observation": {"observation.state": [1.0, 2.0]},
            "policy": policy,
            "device": "cuda:0",
            "preprocessor": preprocessor,
            "postprocessor": postprocessor,
            "use_amp": True,
            "task": "pick up the cube",
            "robot_type": "so101",
        }
    ]


def test_adapter_predict_preserves_native_action_for_environment() -> None:
    native = _ArrayLike()
    adapter = LeRobotPolicyAdapter(
        policy=_Policy(),
        preprocessor=object(),
        postprocessor=object(),
        device="cpu",
        _predict_action_fn=lambda **_kwargs: native,
    )

    prediction = adapter.predict(
        {"observation.state": np.array([1.0])},
        task="pick cube",
        step=0,
    )

    assert prediction.native_action is native
    assert prediction.action.decoded == [0.1, -0.2, 0.3]


class _Environment:
    def __init__(self) -> None:
        self.actions = []
        self.closed = False
        self.step_index = 0

    def reset(self, *, seed, options):
        assert seed == 123
        assert options == {"reset_id": 4}
        return {
            "observation.state": np.array([0.0], dtype=np.float32),
            "observation.image": np.zeros((8, 8, 3), dtype=np.uint8),
        }, {"reset_id": 4}

    def step(self, action):
        self.actions.append(action)
        self.step_index += 1
        success = self.step_index == 2
        return (
            {"observation.state": np.array([self.step_index], dtype=np.float32)},
            float(success),
            success,
            False,
            {"success": success},
        )

    def render(self):
        return np.full((8, 8, 3), self.step_index * 30, dtype=np.uint8)

    def close(self):
        self.closed = True


def _config(tmp_path: Path) -> EmbodiedExperimentConfig:
    source = (
        Path(__file__).parents[1]
        / "examples/embodied/lerobot_action_token_grpo.template.yaml"
    )
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["storage"]["output_dir"] = str(tmp_path / "output")
    raw["rollout"]["max_episode_steps"] = 3
    raw["rollout"]["max_policy_steps"] = 3
    raw["rollout"]["action_payload"].update(
        {
            "kind": "continuous",
            "require_old_logprobs": False,
            "require_prompt": False,
            "require_observation": False,
        }
    )
    return EmbodiedExperimentConfig.model_validate(raw)


def test_lerobot_episode_rollout_records_native_episode_and_video(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    environment = _Environment()
    native_actions = [_ArrayLike(), _ArrayLike()]
    action_index = 0
    policy_seeds = []

    def make_adapter(_scenario, _context):
        def predict(**_kwargs):
            nonlocal action_index
            action = native_actions[action_index]
            action_index += 1
            return action

        return LeRobotPolicyAdapter(
            policy=_Policy(),
            preprocessor=object(),
            postprocessor=object(),
            device="cpu",
            seed_fn=policy_seeds.append,
            _predict_action_fn=predict,
        )

    rollout = LeRobotEpisodeRollout.from_config(
        config,
        environment_factory=lambda _scenario, _context: environment,
        policy_adapter_factory=make_adapter,
    )
    trajectory = asyncio.run(
        rollout(
            EmbodiedScenario(
                id="pick-4",
                task="pick cube",
                payload={"reset_options": {"reset_id": 4}},
            ),
            RolloutContext(
                update=1,
                group_index=0,
                attempt_index=0,
                environment_seed=123,
                policy_seed=456,
                config_fingerprint=config.fingerprint,
            ),
        )
    )

    assert environment.actions == native_actions
    assert policy_seeds == [456, 457, 458]
    assert environment.closed is True
    assert trajectory.metrics["success"] is True
    assert trajectory.metrics["episode_steps"] == 2
    assert trajectory.reward == 1.0
    assert len(trajectory.observations) == 3
    assert trajectory.observations[0].value is None
    assert trajectory.observations[0].metadata["fields"]["observation.image"] == {
        "type": "ndarray",
        "shape": [8, 8, 3],
        "dtype": "uint8",
    }
    assert len(trajectory.media) == 1
    assert trajectory.media[0].kind == "video"
    assert Path(trajectory.media[0].metadata["path"]).is_file()
    assert trajectory.metadata["video_capture_selected"] is True


def test_process_actor_prepares_train_and_eval_policy_phases(tmp_path: Path) -> None:
    config = _config(tmp_path)
    phases = []

    async def group_rollout(*, scenario, contexts, phase, policy_client):
        del policy_client
        return [
            EmbodiedTrajectory(task=scenario.task, metrics={"success": True})
            for _context in contexts
        ]

    actor = LeRobotProcessRolloutActor(
        config=config,
        components=LeRobotProcessComponents(
            environment_factory=lambda scenario, context: None,
            policy_adapter_factory=lambda scenario, context: None,
            load_policy_snapshot=lambda **kwargs: None,
            prepare_phase=phases.append,
            group_rollout=group_rollout,
        ),
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    asyncio.run(actor.prepare_update(update=4, policy_snapshot=snapshot))
    scenario = EmbodiedScenario(id="phase", task="pick", payload={})
    contexts = (
        RolloutContext(
            update=4,
            group_index=0,
            attempt_index=0,
            environment_seed=1,
            policy_seed=2,
            config_fingerprint=config.fingerprint,
        ),
    )

    asyncio.run(actor.rollout_group(scenario, contexts, phase="train"))
    asyncio.run(actor.rollout_group(scenario, contexts, phase="eval"))

    assert phases == ["train", "eval"]


def test_process_actor_uses_group_hook_for_shared_inference_evaluation(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    policy_client = object()
    received = []

    async def group_rollout(*, scenario, contexts, phase, policy_client):
        received.append((contexts, phase, policy_client))
        return [
            EmbodiedTrajectory(
                task=scenario.task,
                metrics={"success": True},
            ).finish()
        ]

    actor = LeRobotProcessRolloutActor(
        config=config,
        components=LeRobotProcessComponents(
            environment_factory=lambda scenario, context: None,
            policy_adapter_factory=lambda scenario, context: None,
            load_policy_snapshot=lambda **kwargs: None,
            group_rollout=group_rollout,
        ),
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    asyncio.run(actor.prepare_update(update=4, policy_snapshot=snapshot))
    context = RolloutContext(
        update=4,
        group_index=3,
        attempt_index=0,
        environment_seed=11,
        policy_seed=12,
        config_fingerprint=config.fingerprint,
    )

    trajectory = asyncio.run(
        actor.rollout(
            EmbodiedScenario(id="eval", task="pick", payload={}),
            context,
            phase="eval",
            policy_client=policy_client,
        )
    )

    assert trajectory.metrics["success"] is True
    assert received == [((context,), "eval", policy_client)]


def test_process_actor_uses_group_hook_for_embedded_evaluation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    received = []

    async def group_rollout(*, scenario, contexts, phase, policy_client):
        received.append((contexts, phase, policy_client))
        return [
            EmbodiedTrajectory(task=scenario.task, metrics={"success": True}).finish()
        ]

    actor = LeRobotProcessRolloutActor(
        config=config,
        components=LeRobotProcessComponents(
            environment_factory=lambda scenario, context: None,
            policy_adapter_factory=lambda scenario, context: None,
            load_policy_snapshot=lambda **kwargs: None,
            group_rollout=group_rollout,
        ),
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    asyncio.run(actor.prepare_update(update=4, policy_snapshot=snapshot))
    context = RolloutContext(
        update=4,
        group_index=3,
        attempt_index=0,
        environment_seed=11,
        policy_seed=12,
        config_fingerprint=config.fingerprint,
    )

    trajectory = asyncio.run(
        actor.rollout(
            EmbodiedScenario(id="eval", task="pick", payload={}),
            context,
            phase="eval",
        )
    )

    assert trajectory.metrics["success"] is True
    assert received == [((context,), "eval", None)]


def test_lerobot_episode_rollout_calls_transition_recorder(tmp_path: Path) -> None:
    config = _config(tmp_path)
    observed = []

    def record_transition(**kwargs):
        observed.append(kwargs)
        kwargs["action"].metadata["transition_recorded"] = True

    rollout = LeRobotEpisodeRollout.from_config(
        config,
        environment_factory=lambda _scenario, _context: _Environment(),
        policy_adapter_factory=lambda _scenario, _context: LeRobotPolicyAdapter(
            policy=_Policy(),
            preprocessor=object(),
            postprocessor=object(),
            device="cpu",
            seed_fn=lambda _seed: None,
            _predict_action_fn=lambda **_kwargs: _ArrayLike(),
        ),
        transition_recorder=record_transition,
    )

    trajectory = rollout.run(
        EmbodiedScenario(
            id="transition-hook",
            task="pick cube",
            payload={"reset_options": {"reset_id": 4}},
        ),
        RolloutContext(
            update=0,
            group_index=0,
            attempt_index=0,
            environment_seed=123,
            policy_seed=456,
            config_fingerprint=config.fingerprint,
        ),
    )

    assert len(observed) == 2
    assert observed[0]["policy_step"] == 0
    assert observed[1]["info"]["success"] is True
    assert all(action.metadata["transition_recorded"] for action in trajectory.actions)


def test_lerobot_episode_rollout_bounds_video_creation_before_rendering(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    raw = config.model_dump(mode="python")
    raw["observability"]["videos_per_update"] = 1
    config = EmbodiedExperimentConfig.model_validate(raw)

    def make_adapter(_scenario, _context):
        return LeRobotPolicyAdapter(
            policy=_Policy(),
            preprocessor=object(),
            postprocessor=object(),
            device="cpu",
            seed_fn=lambda _seed: None,
            _predict_action_fn=lambda **_kwargs: _ArrayLike(),
        )

    rollout = LeRobotEpisodeRollout.from_config(
        config,
        environment_factory=lambda _scenario, _context: _Environment(),
        policy_adapter_factory=make_adapter,
    )
    scenario = EmbodiedScenario(
        id="pick",
        task="pick cube",
        payload={"reset_options": {"reset_id": 4}},
    )

    first = rollout.run(
        scenario,
        RolloutContext(
            update=0,
            group_index=0,
            attempt_index=0,
            environment_seed=123,
            policy_seed=456,
            config_fingerprint=config.fingerprint,
        ),
    )
    second = rollout.run(
        scenario,
        RolloutContext(
            update=0,
            group_index=0,
            attempt_index=1,
            environment_seed=123,
            policy_seed=457,
            config_fingerprint=config.fingerprint,
        ),
    )

    assert first.metadata["video_capture_selected"] is True
    assert len(first.media) == 1
    assert second.metadata["video_capture_selected"] is False
    assert second.media == []
    assert (
        len(list((config.storage.output_dir / "videos" / "train").glob("*.gif"))) == 1
    )


def test_lerobot_rollout_records_unused_chunk_lookahead_without_stepping_source(
    tmp_path: Path,
) -> None:
    raw = _config(tmp_path).model_dump(mode="python")
    raw["observability"]["videos_per_update"] = 0
    raw["observability"]["lookahead_preview"].update(
        {
            "enabled": True,
            "videos_per_update": 1,
            "max_future_frames": 2,
            "future_stride": 1,
        }
    )
    config = EmbodiedExperimentConfig.model_validate(raw)
    environments = []

    class PreviewEnvironment(_Environment):
        def __init__(self) -> None:
            super().__init__()
            self.preview_calls = 0

        def preview_action_chunk(
            self,
            action_chunk,
            *,
            source_environment,
            execution_horizon,
            frame_stride,
            max_frames,
        ):
            assert self is not source_environment
            assert len(action_chunk) == 4
            assert execution_horizon == 1
            assert frame_stride == 1
            assert max_frames == 2
            self.preview_calls += 1
            return [
                LookaheadFrame(
                    image=np.full((8, 8, 3), 80, dtype=np.uint8),
                    action_index=1,
                ),
                LookaheadFrame(
                    image=np.full((8, 8, 3), 160, dtype=np.uint8),
                    action_index=2,
                ),
            ]

    def environment_factory(_scenario, _context):
        environment = PreviewEnvironment()
        environments.append(environment)
        return environment

    class Adapter:
        def reset(self, *, seed=None):
            del seed

        def stateful_component_ids(self):
            return ()

        def predict(self, _observation, *, task, step, seed=None):
            del task, seed
            full_chunk = np.zeros((4, 3), dtype=np.float32)
            action = Action(
                step=step,
                kind="continuous",
                raw=full_chunk[:1].tolist(),
                decoded=full_chunk[:1].tolist(),
            )
            return LeRobotActionPrediction(
                native_action=full_chunk[:1],
                action=action,
                predicted_action_chunk=full_chunk,
                execution_horizon=1,
            )

    rollout = LeRobotEpisodeRollout.from_config(
        config,
        environment_factory=environment_factory,
        policy_adapter_factory=lambda _scenario, _context: Adapter(),
    )
    trajectory = rollout.run(
        EmbodiedScenario(
            id="lookahead",
            task="pick cube",
            payload={"reset_options": {"reset_id": 4}},
        ),
        RolloutContext(
            update=0,
            group_index=0,
            attempt_index=0,
            environment_seed=123,
            policy_seed=456,
            config_fingerprint=config.fingerprint,
        ),
    )

    assert len(environments) == 2
    source, preview = environments
    assert source.step_index == 2
    assert preview.step_index == 0
    assert preview.preview_calls == 2
    assert all(environment.closed for environment in environments)
    assert [media.metadata["role"] for media in trajectory.media] == [
        "lookahead_preview"
    ]


def test_lerobot_rollout_requires_visible_success_key(tmp_path: Path) -> None:
    config = _config(tmp_path)
    raw = config.model_dump(mode="python")
    raw["environment"]["kwargs"] = {}
    without_key = EmbodiedExperimentConfig.model_validate(raw)

    try:
        LeRobotEpisodeRollout.from_config(
            without_key,
            environment_factory=lambda _scenario, _context: _Environment(),
            policy_adapter_factory=lambda _scenario, _context: None,
        )
    except ValueError as exc:
        assert "environment.kwargs.success_key" in str(exc)
    else:
        raise AssertionError("missing success_key must fail closed")


def test_lerobot_rollout_rejects_action_representation_mismatch(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    raw = config.model_dump(mode="python")
    raw["rollout"]["action_payload"]["kind"] = "token"
    raw["rollout"]["action_payload"]["require_old_logprobs"] = True
    config = EmbodiedExperimentConfig.model_validate(raw)
    rollout = LeRobotEpisodeRollout.from_config(
        config,
        environment_factory=lambda _scenario, _context: _Environment(),
        policy_adapter_factory=lambda _scenario, _context: LeRobotPolicyAdapter(
            policy=_Policy(),
            preprocessor=object(),
            postprocessor=object(),
            device="cpu",
            seed_fn=lambda _seed: None,
            _predict_action_fn=lambda **_kwargs: _ArrayLike(),
        ),
    )

    with pytest.raises(RuntimeError, match="action_payload.kind"):
        rollout.run(
            EmbodiedScenario(
                id="mismatch",
                task="pick cube",
                payload={"reset_options": {"reset_id": 4}},
            ),
            RolloutContext(
                update=0,
                group_index=0,
                attempt_index=0,
                environment_seed=123,
                policy_seed=456,
                config_fingerprint=config.fingerprint,
            ),
        )


def test_lerobot_rollout_rejects_concurrent_state_sharing(tmp_path: Path) -> None:
    config = _config(tmp_path)
    adapter = LeRobotPolicyAdapter(
        policy=_Policy(),
        preprocessor=_Resettable(),
        postprocessor=_Resettable(),
        device="cpu",
        seed_fn=lambda _seed: None,
        _predict_action_fn=lambda **_kwargs: _ArrayLike(),
    )
    rollout = LeRobotEpisodeRollout.from_config(
        config,
        environment_factory=lambda _scenario, _context: _Environment(),
        policy_adapter_factory=lambda _scenario, _context: adapter,
    )

    lease = rollout._acquire_adapter(adapter)
    try:
        with pytest.raises(RuntimeError, match="episode-isolated"):
            rollout._acquire_adapter(adapter)
    finally:
        rollout._release_adapter(lease)


def test_lerobot_rollout_closes_environment_when_adapter_factory_fails(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    environment = _Environment()

    def fail_adapter(_scenario, _context):
        raise RuntimeError("adapter construction failed")

    rollout = LeRobotEpisodeRollout.from_config(
        config,
        environment_factory=lambda _scenario, _context: environment,
        policy_adapter_factory=fail_adapter,
    )

    with pytest.raises(RuntimeError, match="adapter construction failed"):
        rollout.run(
            EmbodiedScenario(id="adapter-failure", task="pick cube", payload={}),
            RolloutContext(
                update=0,
                group_index=0,
                attempt_index=0,
                environment_seed=123,
                policy_seed=456,
                config_fingerprint=config.fingerprint,
            ),
        )

    assert environment.closed is True
