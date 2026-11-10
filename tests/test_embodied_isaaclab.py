from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from art_embodied.experiment import EmbodiedScenario, RolloutContext  # noqa: E402
from art_embodied.integrations.isaaclab import (  # noqa: E402
    IsaacLabChunkEnvironment,
    isaaclab_reset_fingerprints,
    repeat_isaaclab_scene_state,
)
from art_embodied.integrations.isaaclab_rollout import (
    IsaacLabGroupRollout,  # noqa: E402
)
from art_embodied.integrations.lerobot import LeRobotActionPrediction  # noqa: E402
from art_embodied.trajectories import Action  # noqa: E402


class VectorEnv:
    """Same-step resets with reused buffers, matching the manager-based API."""

    num_envs = 3
    device = "cpu"
    single_action_space = SimpleNamespace(shape=(2,))

    def __init__(self):
        self.unwrapped = self
        self.cfg = SimpleNamespace(compute_final_obs=True)
        self.actions = []
        self.close_count = 0
        self.success = torch.zeros(3, dtype=torch.bool)
        self.state = torch.zeros(3, 2)
        self.omit_final = False

    def reset(self, *, seed):
        self.seed = seed
        self.tick = 0
        self.state.zero_()
        self.success.zero_()
        self.actions.clear()
        return self.observation(), {}

    def observation(self):
        return {
            "policy": {
                "state": self.state,
                "rgb": self.state[:, :1, None, None].expand(3, 4, 4, 3).byte(),
            }
        }

    def step(self, actions):
        self.actions.append(actions.clone())
        self.tick += 1
        self.state.fill_(self.tick)
        terminated = torch.tensor([self.tick == 1, False, self.tick == 3])
        truncated = torch.tensor([False, self.tick == 2, False])
        self.success = torch.tensor([self.tick == 1, False, False])
        final = {
            "policy": {
                key: value.clone()
                for key, value in self.observation()["policy"].items()
            }
        }
        self.state[terminated | truncated] = -100
        info = {} if self.omit_final else {"final_obs": final}
        return (
            self.observation(),
            torch.tensor([1.0, 2.0, 3.0]),
            terminated,
            truncated,
            info,
        )

    def close(self):
        self.close_count += 1


def adapter(env=None, *, horizon=4, limit=20, success_fn=None):
    return IsaacLabChunkEnvironment(
        env or VectorEnv(),
        execution_horizon=horizon,
        max_environment_steps=limit,
        success_fn=success_fn or (lambda env, obs, info: env.success),
    )


def test_chunk_preserves_terminal_observations_and_masks_finished_rows():
    env = adapter()
    initial, _ = env.reset(seed=41)
    obs, reward, terminated, truncated, info = env.step(torch.ones(3, 4, 2))
    assert initial["policy"]["state"].tolist() == [[0, 0]] * 3
    assert obs["policy"]["state"].tolist() == [[1, 1], [2, 2], [3, 3]]
    assert reward.tolist() == [1, 4, 9]
    assert terminated.tolist() == [True, False, True]
    assert truncated.tolist() == [False, True, False]
    assert info["success"].tolist() == [True, False, False]
    assert info["env_steps"].tolist() == [1, 2, 3]
    assert info["primitive_loss_mask"].tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
    ]
    assert len(env.env.actions) == 3
    assert env.env.actions[1][0].tolist() == [0, 0]
    assert env.env.actions[2][1].tolist() == [0, 0]
    assert info["executed_actions"][0].tolist() == [[1, 1], [0, 0], [0, 0], [0, 0]]
    with pytest.raises(RuntimeError, match="All Isaac Lab episodes finished"):
        env.step(torch.ones(3, 4, 2))


def test_truncation_at_primitive_budget_is_not_success_or_termination():
    env = adapter(limit=2)
    env.reset(seed=1)
    obs, _, terminated, truncated, info = env.step(torch.ones(3, 4, 2))
    assert info["env_steps"].tolist() == [1, 2, 2]
    assert terminated.tolist() == [True, False, False]
    assert truncated.tolist() == [False, True, True]
    assert obs["policy"]["state"][2].tolist() == [2, 2]


def test_idle_commands_preserve_active_rows_and_stay_out_of_episode_records():
    raw = VectorEnv()
    calls = []

    def hold_current_state(env):
        calls.append(env.tick)
        return env.state.clone()

    env = IsaacLabChunkEnvironment(
        raw,
        execution_horizon=4,
        max_environment_steps=20,
        success_fn=lambda env, *_: env.success,
        inactive_action_fn=hold_current_state,
    )
    env.reset(seed=1)
    obs, reward, _, _, info = env.step(torch.ones(3, 4, 2))
    assert calls == [1, 2]
    assert raw.actions[1][0].tolist() == [-100, -100]
    assert raw.actions[1][1:].tolist() == [[1, 1], [1, 1]]
    assert raw.actions[2][1].tolist() == [-100, -100]
    assert obs["policy"]["state"].tolist() == [[1, 1], [2, 2], [3, 3]]
    assert reward.tolist() == [1, 4, 9]
    assert not info["primitive_loss_mask"][0, 1:].any()
    assert not info["executed_actions"][0, 1:].any()


@pytest.mark.parametrize(
    "idle, message",
    [
        (torch.ones(2), "num_envs, action_dim"),
        (torch.full((3, 2), float("nan")), "nonfinite"),
        (torch.full((3, 2), float("inf")), "nonfinite"),
    ],
)
def test_invalid_idle_commands_stop_before_another_physics_step(idle, message):
    raw = VectorEnv()
    env = IsaacLabChunkEnvironment(
        raw,
        execution_horizon=4,
        max_environment_steps=20,
        success_fn=lambda env, *_: env.success,
        inactive_action_fn=lambda _: idle,
    )
    env.reset(seed=1)
    with pytest.raises(ValueError, match=message):
        env.step(torch.ones(3, 4, 2))
    assert len(raw.actions) == 1
    with pytest.raises(RuntimeError, match="Reset Isaac Lab"):
        env.step(torch.ones(3, 4, 2))


def test_idle_callback_must_be_callable():
    with pytest.raises(TypeError, match="inactive_action_fn must be callable"):
        IsaacLabChunkEnvironment(
            VectorEnv(),
            execution_horizon=1,
            max_environment_steps=20,
            success_fn=lambda env, *_: env.success,
            inactive_action_fn=42,
        )


def test_finished_observations_survive_later_chunks():
    env = adapter(horizon=1)
    env.reset(seed=1)
    first, *_ = env.step(torch.ones(3, 1, 2))
    second, _, _, _, info = env.step(torch.ones(3, 1, 2))
    assert first["policy"]["state"][0].tolist() == [1, 1]
    assert second["policy"]["state"][0].tolist() == [1, 1]
    assert info["primitive_loss_mask"].tolist() == [[False], [True], [True]]


def test_missing_final_observation_fails_and_requires_reset():
    raw = VectorEnv()
    raw.omit_final = True
    env = adapter(raw)
    env.reset(seed=1)
    with pytest.raises(RuntimeError, match="omitted final_obs"):
        env.step(torch.ones(3, 4, 2))
    with pytest.raises(RuntimeError, match="Reset Isaac Lab"):
        env.step(torch.ones(3, 4, 2))


def test_final_observation_support_required_at_construction():
    env = VectorEnv()
    env.cfg.compute_final_obs = False
    with pytest.raises(ValueError, match="compute_final_obs"):
        adapter(env)


@pytest.mark.parametrize(
    "renderer,refresh",
    [("newton_warp", False), ("newton_warp", True), ("rtx", False), (None, False)],
)
def test_warp_cameras_require_pose_refresh_before_collection(renderer, refresh):
    env = VectorEnv()
    env.scene = SimpleNamespace(
        sensors={
            "wrist": SimpleNamespace(
                cfg=SimpleNamespace(
                    renderer_cfg=SimpleNamespace(renderer_type=renderer),
                    update_latest_camera_pose=refresh,
                )
            )
        }
    )
    if renderer == "newton_warp" and not refresh:
        with pytest.raises(ValueError, match="wrist.*update_latest_camera_pose=True"):
            adapter(env)
        assert env.actions == []
    else:
        adapter(env)


@pytest.mark.parametrize(
    "actions",
    [torch.ones(3, 2), torch.ones(1, 4, 2), torch.ones(3, 3, 2), torch.ones(3, 4, 1)],
)
def test_action_shape_rejected_before_physics(actions):
    env = adapter()
    env.reset(seed=1)
    with pytest.raises(ValueError, match="Expected"):
        env.step(actions)
    assert not env.env.actions


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_action_rejected_before_physics(value):
    env = adapter()
    env.reset(seed=1)
    with pytest.raises(ValueError, match="nonfinite"):
        env.step(torch.full((3, 4, 2), value))
    assert not env.env.actions


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_reset_observation_requires_another_reset(value):
    env = adapter()

    def invalid_reset(raw, seed):
        observation, info = raw.reset(seed=seed)
        observation["policy"]["state"][2, 0] = value
        return observation, info

    with pytest.raises(ValueError, match="observation contains nonfinite"):
        env.reset(seed=1, reset_fn=invalid_reset)
    with pytest.raises(RuntimeError, match="Reset Isaac Lab"):
        env.step(torch.ones(3, 4, 2))
    assert not env.env.actions
    env.reset(seed=1)
    env.step(torch.ones(3, 4, 2))


@pytest.mark.parametrize("terminal", [False, True])
def test_nonfinite_observation_stops_at_first_bad_primitive_step(terminal):
    raw = VectorEnv()
    original = raw.step

    def invalid_step(actions):
        observation, reward, terminated, truncated, info = original(actions)
        if terminal:
            info["final_obs"]["policy"]["state"][0, 0] = float("nan")
        else:
            observation["policy"]["state"][2, 0] = float("inf")
        return observation, reward, terminated, truncated, info

    raw.step = invalid_step
    env = adapter(raw)
    env.reset(seed=1)
    with pytest.raises(ValueError, match="observation contains nonfinite"):
        env.step(torch.ones(3, 4, 2))
    assert len(raw.actions) == 1
    with pytest.raises(RuntimeError, match="Reset Isaac Lab"):
        env.step(torch.ones(3, 4, 2))


def test_nonfinite_autoreset_row_does_not_replace_valid_final_observation():
    raw = VectorEnv()
    original = raw.step

    def reset_row_nan(actions):
        observation, reward, terminated, truncated, info = original(actions)
        observation["policy"]["state"][0, 0] = float("nan")
        return observation, reward, terminated, truncated, info

    raw.step = reset_row_nan
    env = adapter(raw)
    env.reset(seed=1)
    obs, *_ = env.step(torch.ones(3, 4, 2))
    assert obs["policy"]["state"].tolist() == [[1, 1], [2, 2], [3, 3]]


def test_positive_dense_reward_does_not_imply_success():
    env = adapter(success_fn=lambda *_: torch.zeros(3, dtype=torch.bool))
    env.reset(seed=1)
    _, rewards, _, _, info = env.step(torch.ones(3, 4, 2))
    assert (rewards > 0).all()
    assert not info["success"].any()


def test_success_probability_not_accepted_as_boolean():
    env = adapter(success_fn=lambda *_: torch.full((3,), 0.2))
    env.reset(seed=1)
    with pytest.raises(TypeError, match="must be boolean"):
        env.step(torch.ones(3, 4, 2))


def test_reset_and_close_lifecycle():
    env = adapter()
    with pytest.raises(RuntimeError, match="Reset"):
        env.step(torch.ones(3, 4, 2))
    env.reset(seed=1)
    env.step(torch.ones(3, 4, 2))
    env.reset(seed=2)
    assert env.env.seed == 2
    env.step(torch.ones(3, 4, 2))
    env.close()
    env.close()
    assert env.env.close_count == 1
    with pytest.raises(RuntimeError, match="closed"):
        env.reset(seed=3)


def test_repeat_relative_scene_state_and_fingerprints():
    state = {
        "articulation": {"robot": {"joint_position": torch.arange(6).reshape(3, 2)}}
    }
    repeated = repeat_isaaclab_scene_state(state, count=3, source_index=1)
    assert repeated["articulation"]["robot"]["joint_position"].tolist() == [[2, 3]] * 3
    hashes = isaaclab_reset_fingerprints(repeated, num_envs=3)
    assert len(set(hashes)) == 1
    repeated["articulation"]["robot"]["joint_position"][0, 0] = 7
    assert state["articulation"]["robot"]["joint_position"][1, 0] == 2
    assert repeated["articulation"]["robot"]["joint_position"][1, 0] == 2
    assert len(set(isaaclab_reset_fingerprints(repeated, num_envs=3))) == 2


@pytest.mark.parametrize(
    "state",
    [
        {},
        {"scalar": 3},
        {"wrong_batch": torch.zeros(2, 4)},
        {"bad": torch.full((3, 1), float("nan"))},
    ],
)
def test_invalid_comparison_state_rejected(state):
    with pytest.raises((ValueError, TypeError)):
        isaaclab_reset_fingerprints(state, num_envs=3)


class BatchPolicy:
    def __init__(self):
        self.calls = []

    def prepare_update(self, *, update):
        self.update = update

    def reset(self, *, seeds):
        self.reset_seeds = seeds

    def predict_batch(self, observations, *, task, step, seeds):
        self.calls.append((len(observations), seeds))
        return [
            LeRobotActionPrediction(
                native_action=torch.ones(1, 2),
                action=Action(step=step, kind="chunk", raw=[[1, 1]], logprobs=[-0.5]),
                execution_horizon=1,
            )
            for _ in observations
        ]


def contexts():
    return [RolloutContext(1, 0, i, 17, 100 + i * 10, "config") for i in range(3)]


def collector(tmp_path, *, state_fn=None, policy=None):
    return IsaacLabGroupRollout(
        environment=adapter(horizon=1),
        policy=policy or BatchPolicy(),
        reset_fn=lambda env, scenario, seed: env.reset(seed=seed),
        reset_state_fn=state_fn or (lambda env: {"state": env.state}),
        observation_fn=lambda observation: observation["policy"],
        output_dir=tmp_path,
        expected_action_kind="chunk",
        videos_per_update=3,
        frame_fn=lambda observation: observation["rgb"].cpu().numpy(),
    )


@pytest.mark.asyncio
async def test_group_rollout_retains_seed_identity_records_and_videos(tmp_path):
    rollout = collector(tmp_path)
    await rollout.prepare_update(update=1)
    trajectories = await rollout.rollout_group(
        EmbodiedScenario(id="test", task="pick", payload={}), contexts()
    )
    assert [len(t.actions) for t in trajectories] == [1, 2, 3]
    assert [t.reward for t in trajectories] == [1, 0, 0]
    assert [t.metrics["environment_return"] for t in trajectories] == [1, 4, 9]
    assert rollout.policy.calls == [(3, [101, 111, 121]), (2, [112, 122]), (1, [123])]
    assert rollout.policy.reset_seeds == [100, 110, 120]
    for t in trajectories:
        assert t.initial_policy_version == t.final_policy_version == 1
        assert t.media and t.media[0].kind == "video"
        assert t.actions[0].logprobs == [-0.5]
        assert t.actions[0].metadata["isaaclab_execution"]["primitive_loss_mask"] == [
            True
        ]
        assert t.metadata["reset_info"]["comparison_state_sha256"]
        assert len(t.observations) == len(t.actions) + 1
    assert len(list(tmp_path.glob("*.gif"))) == 3
    rollout.close()


@pytest.mark.asyncio
async def test_group_rejects_nonidentical_task_commands_before_policy_sampling(
    tmp_path,
):
    rollout = collector(
        tmp_path, state_fn=lambda env: {"scene": env.state, "command": torch.arange(3)}
    )
    await rollout.prepare_update(update=1)
    with pytest.raises(RuntimeError, match="initial conditions differ"):
        await rollout.rollout_group(
            EmbodiedScenario(id="test", task="pick", payload={}), contexts()
        )
    assert rollout.policy.calls == []
    assert rollout.environment.env.actions == []


@pytest.mark.asyncio
async def test_group_requires_explicit_policy_sync(tmp_path):
    rollout = collector(tmp_path)
    with pytest.raises(RuntimeError, match="prepare_update"):
        await rollout.rollout_group(
            EmbodiedScenario(id="test", task="pick", payload={}), contexts()
        )


@pytest.mark.asyncio
async def test_wrong_batch_count_does_not_execute_physics(tmp_path):
    policy = BatchPolicy()
    policy.predict_batch = lambda *args, **kwargs: []
    rollout = collector(tmp_path, policy=policy)
    await rollout.prepare_update(update=1)
    with pytest.raises(ValueError, match="prediction count"):
        await rollout.rollout_group(
            EmbodiedScenario(id="test", task="pick", payload={}), contexts()
        )
    assert rollout.environment.env.actions == []


@pytest.mark.asyncio
async def test_group_reset_receives_scenario_and_records_state(tmp_path):
    rollout = collector(tmp_path)
    scenario = EmbodiedScenario(id="reset-42", task="pick", payload={"state_id": 42})
    calls = []

    def reset(env, received, seed):
        calls.append((received, seed))
        return env.reset(seed=seed)

    rollout.reset_fn = reset
    await rollout.prepare_update(update=1)
    trajectories = await rollout.rollout_group(scenario, contexts())
    assert calls == [(scenario, 17)]
    assert trajectories[0].metadata["reset_state"] == {"state": [0.0, 0.0]}


@pytest.mark.parametrize(
    "change",
    [
        {"environment_seed": 18},
        {"update": 2},
        {"group_index": 1},
        {"config_fingerprint": "different"},
        {"attempt_index": 1},
        {"policy_seed": 110},
    ],
)
@pytest.mark.asyncio
async def test_invalid_group_context_rejected_before_reset(tmp_path, change):
    rollout = collector(tmp_path)
    batch = contexts()
    batch[0] = replace(batch[0], **change)
    await rollout.prepare_update(update=1)
    with pytest.raises(ValueError):
        await rollout.rollout_group(
            EmbodiedScenario(id="test", task="pick", payload={}), batch
        )
    assert not hasattr(rollout.environment.env, "seed")


@pytest.mark.asyncio
async def test_existing_experiment_collects_vector_groups_without_new_logging_path(
    tmp_path,
):
    from art_embodied.config import EmbodiedExperimentConfig
    from art_embodied.experiment import EmbodiedExperiment

    config = EmbodiedExperimentConfig.from_yaml(
        Path(__file__).parents[1]
        / "examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml"
    )
    config = config.model_copy(
        update={
            "algorithm": config.algorithm.model_copy(update={"group_size": 3}),
            "rollout": config.rollout.model_copy(
                update={
                    "groups_per_update": 2,
                    "epochs_per_update": 1,
                    "workers": 2,
                    "minimum_completed_attempts_per_group": 3,
                }
            ),
            "evaluation": config.evaluation.model_copy(update={"enabled": False}),
        }
    )
    rollout = collector(tmp_path)
    experiment = EmbodiedExperiment(
        config=config,
        rollout=rollout,
        backend=object(),
        scenarios=[EmbodiedScenario(id="test", task="pick", payload={})],
    )
    groups = await experiment.collect(update=0)
    assert len(groups) == 2
    assert [len(group.trajectories) for group in groups] == [3, 3]
    assert groups[0].metadata["group_index"] == 0
    assert groups[1].metadata["group_index"] == 1
    for group in groups:
        assert [t.reward for t in group.trajectories] == [1, 0, 0]
        assert all(t.initial_policy_version == 0 for t in group.trajectories)
        assert all(t.final_policy_version == 0 for t in group.trajectories)
        assert len({t.metadata["environment_seed"] for t in group.trajectories}) == 1
        assert len({t.metadata["policy_seed"] for t in group.trajectories}) == 3
        assert all(t.metadata["simulator"] == "isaaclab" for t in group.trajectories)
    assert all(t.media for t in groups[0].trajectories)
    assert all(not t.media for t in groups[1].trajectories)
