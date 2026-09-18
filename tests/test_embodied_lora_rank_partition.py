from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

torch = pytest.importorskip("torch")

from art_embodied.backends.action_token_gradients import (  # noqa: E402
    apply_action_token_gradient_payloads,
)
from art_embodied.checkpointing import CheckpointManager  # noqa: E402
from art_embodied.config import (  # noqa: E402
    EmbodiedExperimentConfig,
    LoraRankPartitionConfig,
    LoraWarmStartSourceConfig,
)
from art_embodied.lora_rank_partition import (  # noqa: E402
    TaskAdapterGradientGate,
    adapter_from_parameter_name,
    balanced_lora_rank_blocks,
    rank_blocks_from_lora_config,
)
from art_embodied.vla_trainable import (  # noqa: E402
    configure_vla_trainable_parameters,
)


class _PartitionedPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = torch.nn.ModuleDict(
            {
                "task_000": torch.nn.Linear(1, 1, bias=False),
                "task_001": torch.nn.Linear(1, 1, bias=False),
            }
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return sum(layer(value) for layer in self.lora_A.values())


def _blocks():
    return balanced_lora_rank_blocks(2, ["task-a", "task-b"])


def test_balanced_rank_partition_assigns_all_128_ranks() -> None:
    blocks = balanced_lora_rank_blocks(128, [f"task-{index}" for index in range(10)])

    assert [block.rank for block in blocks] == [13] * 8 + [12] * 2
    assert [(block.start, block.stop) for block in blocks] == [
        (0, 13),
        (13, 26),
        (26, 39),
        (39, 52),
        (52, 65),
        (65, 78),
        (78, 91),
        (91, 104),
        (104, 116),
        (116, 128),
    ]
    assert [block.adapter_name for block in blocks] == [
        f"task_{index:03d}" for index in range(10)
    ]


def test_rank_blocks_resolve_from_strict_config_shape() -> None:
    lora = SimpleNamespace(
        rank=2,
        rank_partition=SimpleNamespace(
            mode="task_all_active",
            task_keys=["task-a", "task-b"],
        ),
    )

    assert rank_blocks_from_lora_config(lora) == _blocks()


def test_all_adapters_affect_forward_but_backward_updates_only_task_block() -> None:
    policy = _PartitionedPolicy()
    with torch.no_grad():
        policy.lora_A["task_000"].weight.fill_(2.0)
        policy.lora_A["task_001"].weight.fill_(3.0)
    gate = TaskAdapterGradientGate(policy, _blocks())
    gate.activate("task-a")

    output = policy(torch.tensor([[1.0]]))
    output.backward()
    gate.close()

    assert output.item() == pytest.approx(5.0)
    assert policy.lora_A["task_000"].weight.grad.item() == pytest.approx(1.0)
    assert policy.lora_A["task_001"].weight.grad.item() == pytest.approx(0.0)


def test_task_owned_switch_keeps_all_partitions_optimizer_owned() -> None:
    policy = _PartitionedPolicy()
    activated: list[str] = []

    def set_adapter(adapter: str) -> None:
        activated.append(adapter)
        for name, parameter in policy.named_parameters():
            parameter.requires_grad_(f".{adapter}." in name)

    policy.model = SimpleNamespace(base_model=SimpleNamespace(set_adapter=set_adapter))
    # Rescoring selects one PEFT adapter and freezes the others before training.
    set_adapter("task_001")
    for task in ("task-a", "task-b"):
        gate = TaskAdapterGradientGate(
            policy,
            _blocks(),
            forward_routing="task_owned",
        )
        gate.activate(task)
        gate.close()
        assert {
            adapter_from_parameter_name(name)
            for name, parameter in policy.named_parameters()
            if parameter.requires_grad
        } == {"task_000", "task_001"}

    assert activated == ["task_001", "task_000", "task_001"]


def test_optimizer_does_not_touch_inactive_adapter_value_or_state() -> None:
    policy = _PartitionedPolicy()
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=0.1,
        betas=(0.9, 0.95),
        weight_decay=0.2,
    )
    names = dict(policy.named_parameters())
    task_a_name = next(name for name in names if ".task_000." in name)
    task_b_name = next(name for name in names if ".task_001." in name)

    def payload(active: str, a_grad: float, b_grad: float):
        return {
            "format": "art_embodied_action_token_grpo_gradients_v1",
            "gradients": {
                name: torch.full_like(names[name], value)
                for name, value in ((task_a_name, a_grad), (task_b_name, b_grad))
                if f".{active}." in name
            },
            "shapes": {
                task_a_name: tuple(names[task_a_name].shape),
                task_b_name: tuple(names[task_b_name].shape),
            },
            "missing_gradients": [],
            "rank_partition_mode": "task_all_active",
            "active_adapters": [active],
        }

    apply_action_token_gradient_payloads(
        policy,
        optimizer,
        [payload("task_001", 0.0, 1.0)],
        skip_optimizer_step_without_policy_gradient_signal=False,
    )
    task_b_before = names[task_b_name].detach().clone()
    task_b_state_before = {
        key: value.detach().clone() if torch.is_tensor(value) else value
        for key, value in optimizer.state[names[task_b_name]].items()
    }

    apply_action_token_gradient_payloads(
        policy,
        optimizer,
        [payload("task_000", 1.0, 0.0)],
        skip_optimizer_step_without_policy_gradient_signal=False,
    )

    torch.testing.assert_close(names[task_b_name], task_b_before)
    task_b_state_after = optimizer.state[names[task_b_name]]
    assert task_b_state_after.keys() == task_b_state_before.keys()
    for key, before in task_b_state_before.items():
        after = task_b_state_after[key]
        if torch.is_tensor(before):
            torch.testing.assert_close(after, before)
        else:
            assert after == before
    assert adapter_from_parameter_name(task_a_name) == "task_000"


def test_peft_partition_keeps_every_task_adapter_active() -> None:
    pytest.importorskip("peft")

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(2, 2, bias=False)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.proj(value)

    class Policy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Model()

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.model(value)

    policy = Policy()
    baseline = policy(torch.ones(1, 2)).detach().clone()
    configured, report = configure_vla_trainable_parameters(
        policy,
        policy_type="smolvla",
        algorithm_cfg={
            "trainable_parameter_strategy": "smolvla_action_expert_lora",
            "force_trainable_float32": True,
            "peft": {
                "enabled": True,
                "r": 128,
                "lora_alpha": 128,
                "lora_dropout": 0.0,
                "bias": "none",
                "target_modules": ["proj"],
                "modules_to_save": [],
                "init_lora_weights": "gaussian",
                "apply_layer_selection": False,
                "rank_partition": {
                    "mode": "task_all_active",
                    "allocation": "balanced",
                    "task_keys": [f"task-{index}" for index in range(10)],
                },
            },
        },
    )

    assert report["ok"] is True
    partition = report["peft"]["rank_partition"]
    assert [block["rank"] for block in partition["blocks"]] == [13] * 8 + [12] * 2
    assert configured.model.active_adapters == [
        f"task_{index:03d}" for index in range(10)
    ]
    torch.testing.assert_close(configured(torch.ones(1, 2)), baseline)
    trainable_names = [
        name
        for name, parameter in configured.named_parameters()
        if parameter.requires_grad
    ]
    assert trainable_names
    assert all(
        adapter_from_parameter_name(name) is not None for name in trainable_names
    )


def test_rank_partition_rejects_reusing_an_existing_adapter() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(2, 2, bias=False)
            self.peft_config = {"existing_rl_adapter": object()}

    policy = SimpleNamespace(model=Model())
    _, report = configure_vla_trainable_parameters(
        policy,
        policy_type="smolvla",
        algorithm_cfg={
            "trainable_parameter_strategy": "smolvla_action_expert_lora",
            "force_trainable_float32": True,
            "peft": {
                "enabled": True,
                "r": 2,
                "lora_alpha": 2,
                "target_modules": ["proj"],
                "rank_partition": {
                    "mode": "task_all_active",
                    "allocation": "balanced",
                    "task_keys": ["task-a", "task-b"],
                },
            },
        },
    )

    assert report["ok"] is False
    assert "Do not merge or reuse an existing RL adapter" in report["error"]


def _warm_start_source(
    root: Path,
    *,
    task_key: str,
    a_value: float,
    b_value: float,
    rank: int = 1,
) -> dict[str, object]:
    peft = pytest.importorskip("peft")

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(2, 2, bias=False)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.proj(value)

    base = Model()
    with torch.no_grad():
        base.proj.weight.zero_()
    source = peft.get_peft_model(
        base,
        peft.LoraConfig(
            r=rank,
            lora_alpha=rank,
            lora_dropout=0.0,
            bias="none",
            target_modules=["proj"],
            modules_to_save=[],
            init_lora_weights="gaussian",
        ),
    )
    with torch.no_grad():
        source.base_model.model.proj.lora_A["default"].weight.fill_(a_value)
        source.base_model.model.proj.lora_B["default"].weight.fill_(b_value)

    source_config_path = root.with_suffix(".yaml")
    source_config_raw = yaml.safe_load(
        Path(
            "examples/embodied/"
            "gr00t_n1d7_robocasa_gr1_tabletop_cuttingboard_pan_noise01_u20_development.yaml"
        ).read_text(encoding="utf-8")
    )
    source_config_raw["experiment"]["run"] = f"warm-start-{task_key}"
    source_config_raw["environment"]["kwargs"]["task_ids"] = [task_key]
    source_config_raw["policy"]["lora"]["rank"] = rank
    source_config_raw["policy"]["lora"]["alpha"] = rank
    source_config_raw["training"]["updates"] = 100
    source_config_path.write_text(
        yaml.safe_dump(source_config_raw, sort_keys=False), encoding="utf-8"
    )
    source_config = EmbodiedExperimentConfig.from_yaml(source_config_path)

    def write(staging: Path) -> None:
        policy_path = staging / "policy"
        source.save_pretrained(policy_path)
        adapter_config_path = policy_path / "adapter_config.json"
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        adapter_config["base_model_name_or_path"] = str(source_config.policy.path)
        adapter_config_path.write_text(
            json.dumps(adapter_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (policy_path / "art_embodied_gr00t_n1d7_snapshot.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "format": "peft_adapter",
                    "family": "gr00t_n1d7",
                    "model_id": str(source_config.policy.path),
                    "revision": None,
                    "checkpoint_subfolder": source_config.policy.load_kwargs[
                        "checkpoint_subfolder"
                    ],
                    "embodiment_tag": source_config.policy.load_kwargs[
                        "embodiment_tag"
                    ],
                    "adapter_names": ["default"],
                    "processor_action_horizon": source_config.policy.load_kwargs[
                        "processor_action_horizon"
                    ],
                    "action_components": [["arm", 2, True, 0]],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    checkpoint = CheckpointManager().publish(
        root,
        writer=write,
        config_fingerprint=source_config.fingerprint,
        resume_contract_fingerprint=source_config.resume_contract_fingerprint,
        metadata={"backend": "flow_sde_grpo", "step": 100},
    )
    marker = checkpoint / "art_embodied_checkpoint_complete.json"
    development_adjudication = root.with_name(f"{root.name}-development.json")
    development_adjudication.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "gr00t_n1d7_robocasa_one_task_u100_adjudication",
                "status": "passed",
                "development_lift_established": True,
                "sealed_eligible": True,
                "selected_checkpoint": str(checkpoint),
                "config": {
                    "path": str(source_config_path),
                    "sha256": _sha256(source_config_path),
                    "resume_contract_fingerprint": (
                        source_config.resume_contract_fingerprint
                    ),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    sealed_adjudication = root.with_name(f"{root.name}-sealed.json")
    sealed_adjudication.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": ("gr00t_n1d7_robocasa_one_task_u100_sealed_adjudication"),
                "status": "passed",
                "no_post_sealed_tuning": True,
                "candidate_checkpoint": {
                    "path": str(checkpoint),
                    "marker_sha256": _sha256(marker),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "checkpoint": checkpoint,
        "source_config": source_config_path,
        "checkpoint_manifest_sha256": _sha256(
            checkpoint / "art_embodied_checkpoint_complete.json"
        ),
        "source_config_sha256": _sha256(source_config_path),
        "development_adjudication": development_adjudication,
        "development_adjudication_sha256": _sha256(development_adjudication),
        "sealed_adjudication": sealed_adjudication,
        "sealed_adjudication_sha256": _sha256(sealed_adjudication),
        "final_step": 100,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _warm_start_policy() -> torch.nn.Module:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(2, 2, bias=False)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.proj(value)

    class Policy(torch.nn.Module):
        family = "gr00t_n1d7"
        model_id = (
            "outputs/gr00t-n1d7-robocasa-sft/"
            "gr00t-n1d7-robocasa-gr1-tabletop-sft-u60000/checkpoint-60000"
        )
        revision = None
        checkpoint_subfolder = "."
        embodiment_tag = "ROBOCASA_GR1_TABLETOP"
        processor_action_horizon = 8
        action_components = (("arm", 2, True, 0),)

        def __init__(self) -> None:
            super().__init__()
            self.model = Model()
            with torch.no_grad():
                self.model.proj.weight.zero_()

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.model(value)

    return Policy()


def test_partition_warm_start_exactly_composes_single_task_adapters(
    tmp_path: Path,
) -> None:
    pytest.importorskip("peft")
    source_a = _warm_start_source(
        tmp_path / "source-a", task_key="task-a", a_value=1.0, b_value=2.0
    )
    source_b = _warm_start_source(
        tmp_path / "source-b", task_key="task-b", a_value=3.0, b_value=1.0
    )
    policy = _warm_start_policy()

    configured, report = configure_vla_trainable_parameters(
        policy,
        policy_type="gr00t_n1d7",
        algorithm_cfg={
            "trainable_parameter_strategy": "gr00t_n1d7_action_head_lora",
            "force_trainable_float32": True,
            "peft": {
                "enabled": True,
                "r": 2,
                "lora_alpha": 2,
                "lora_dropout": 0.0,
                "bias": "none",
                "target_modules": ["proj"],
                "modules_to_save": [],
                "init_lora_weights": "gaussian",
                "apply_layer_selection": False,
                "rank_partition": {
                    "mode": "task_all_active",
                    "allocation": "balanced",
                    "task_keys": ["task-a", "task-b"],
                    "warm_start_sources": {
                        "task-a": source_a,
                        "task-b": source_b,
                    },
                },
            },
        },
    )

    assert report["ok"] is True
    warm_start = report["peft"]["rank_partition"]["warm_start"]
    assert warm_start["optimizer_state_imported"] is False
    assert [block["task_key"] for block in warm_start["blocks"]] == [
        "task-a",
        "task-b",
    ]
    assert all(block["exact_copy_verified"] for block in warm_start["blocks"])
    assert configured.model.active_adapters == ["task_000", "task_001"]
    # task-a contributes 2 * (1 + 1) and task-b contributes 1 * (3 + 3).
    torch.testing.assert_close(
        configured(torch.ones(1, 2)),
        torch.full((1, 2), 10.0),
    )


def test_partition_warm_start_task_owned_keeps_all_blocks_trainable(
    tmp_path: Path,
) -> None:
    pytest.importorskip("peft")
    source_a = _warm_start_source(
        tmp_path / "source-a", task_key="task-a", a_value=1.0, b_value=2.0
    )
    source_b = _warm_start_source(
        tmp_path / "source-b", task_key="task-b", a_value=3.0, b_value=1.0
    )

    configured, report = configure_vla_trainable_parameters(
        _warm_start_policy(),
        policy_type="gr00t_n1d7",
        algorithm_cfg={
            "trainable_parameter_strategy": "gr00t_n1d7_action_head_lora",
            "force_trainable_float32": True,
            "peft": {
                "enabled": True,
                "r": 2,
                "lora_alpha": 2,
                "lora_dropout": 0.0,
                "bias": "none",
                "target_modules": ["proj"],
                "modules_to_save": [],
                "init_lora_weights": "gaussian",
                "apply_layer_selection": False,
                "rank_partition": {
                    "mode": "task_all_active",
                    "allocation": "balanced",
                    "forward_routing": "task_owned",
                    "task_keys": ["task-a", "task-b"],
                    "warm_start_sources": {
                        "task-a": source_a,
                        "task-b": source_b,
                    },
                },
            },
        },
    )

    assert report["ok"] is True
    warm_start = report["peft"]["rank_partition"]["warm_start"]
    assert warm_start["forward_routing"] == "task_owned"
    assert configured.model.active_adapters == ["task_000"]
    adapter_trainable = {
        adapter_from_parameter_name(name)
        for name, parameter in configured.named_parameters()
        if parameter.requires_grad
    }
    assert adapter_trainable == {"task_000", "task_001"}


def test_partition_warm_start_rejects_source_rank_mismatch(tmp_path: Path) -> None:
    pytest.importorskip("peft")
    source_a = _warm_start_source(
        tmp_path / "source-a",
        task_key="task-a",
        a_value=1.0,
        b_value=2.0,
        rank=2,
    )
    source_b = _warm_start_source(
        tmp_path / "source-b", task_key="task-b", a_value=3.0, b_value=1.0
    )

    _, report = configure_vla_trainable_parameters(
        _warm_start_policy(),
        policy_type="gr00t_n1d7",
        algorithm_cfg={
            "trainable_parameter_strategy": "gr00t_n1d7_action_head_lora",
            "force_trainable_float32": True,
            "peft": {
                "enabled": True,
                "r": 2,
                "lora_alpha": 2,
                "lora_dropout": 0.0,
                "bias": "none",
                "target_modules": ["proj"],
                "modules_to_save": [],
                "init_lora_weights": "gaussian",
                "apply_layer_selection": False,
                "rank_partition": {
                    "mode": "task_all_active",
                    "allocation": "balanced",
                    "task_keys": ["task-a", "task-b"],
                    "warm_start_sources": {
                        "task-a": source_a,
                        "task-b": source_b,
                    },
                },
            },
        },
    )

    assert report["ok"] is False
    assert "warm start failed" in report["error"]
    assert "source rank mismatch" in report["error"]


def test_partition_warm_start_rejects_checkpoint_changed_after_publish(
    tmp_path: Path,
) -> None:
    pytest.importorskip("peft")
    source_a = _warm_start_source(
        tmp_path / "source-a", task_key="task-a", a_value=1.0, b_value=2.0
    )
    source_b = _warm_start_source(
        tmp_path / "source-b", task_key="task-b", a_value=3.0, b_value=1.0
    )
    with (Path(source_b["checkpoint"]) / "policy/adapter_model.safetensors").open(
        "ab"
    ) as stream:
        stream.write(b"changed-after-transactional-publish")

    _, report = configure_vla_trainable_parameters(
        _warm_start_policy(),
        policy_type="gr00t_n1d7",
        algorithm_cfg={
            "trainable_parameter_strategy": "gr00t_n1d7_action_head_lora",
            "force_trainable_float32": True,
            "peft": {
                "enabled": True,
                "r": 2,
                "lora_alpha": 2,
                "lora_dropout": 0.0,
                "bias": "none",
                "target_modules": ["proj"],
                "modules_to_save": [],
                "init_lora_weights": "gaussian",
                "apply_layer_selection": False,
                "rank_partition": {
                    "mode": "task_all_active",
                    "allocation": "balanced",
                    "task_keys": ["task-a", "task-b"],
                    "warm_start_sources": {
                        "task-a": source_a,
                        "task-b": source_b,
                    },
                },
            },
        },
    )

    assert report["ok"] is False
    assert "warm start failed" in report["error"]
    assert "payload integrity check failed" in report["error"]
    assert "policy/adapter_model.safetensors" in report["error"]


@pytest.mark.parametrize(
    ("adjudication_key", "sha256_key", "expected_error"),
    [
        (
            "development_adjudication",
            "development_adjudication_sha256",
            "required development adjudication",
        ),
        (
            "sealed_adjudication",
            "sealed_adjudication_sha256",
            "required sealed adjudication",
        ),
    ],
)
def test_partition_warm_start_rejects_unpassed_source_adjudication(
    tmp_path: Path,
    adjudication_key: str,
    sha256_key: str,
    expected_error: str,
) -> None:
    pytest.importorskip("peft")
    source_a = _warm_start_source(
        tmp_path / "source-a", task_key="task-a", a_value=1.0, b_value=2.0
    )
    source_b = _warm_start_source(
        tmp_path / "source-b", task_key="task-b", a_value=3.0, b_value=1.0
    )
    adjudication_path = Path(source_b[adjudication_key])
    adjudication = json.loads(adjudication_path.read_text(encoding="utf-8"))
    adjudication["status"] = "rejected"
    adjudication_path.write_text(
        json.dumps(adjudication, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_b[sha256_key] = _sha256(adjudication_path)

    _, report = configure_vla_trainable_parameters(
        _warm_start_policy(),
        policy_type="gr00t_n1d7",
        algorithm_cfg={
            "trainable_parameter_strategy": "gr00t_n1d7_action_head_lora",
            "force_trainable_float32": True,
            "peft": {
                "enabled": True,
                "r": 2,
                "lora_alpha": 2,
                "lora_dropout": 0.0,
                "bias": "none",
                "target_modules": ["proj"],
                "modules_to_save": [],
                "init_lora_weights": "gaussian",
                "apply_layer_selection": False,
                "rank_partition": {
                    "mode": "task_all_active",
                    "allocation": "balanced",
                    "task_keys": ["task-a", "task-b"],
                    "warm_start_sources": {
                        "task-a": source_a,
                        "task-b": source_b,
                    },
                },
            },
        },
    )

    assert report["ok"] is False
    assert expected_error in report["error"]


def test_partition_config_requires_one_unique_checkpoint_per_task(
    tmp_path: Path,
) -> None:
    def source(path: Path) -> LoraWarmStartSourceConfig:
        return LoraWarmStartSourceConfig(
            checkpoint=path,
            source_config=path.with_suffix(".yaml"),
            checkpoint_manifest_sha256="0" * 64,
            source_config_sha256="1" * 64,
            development_adjudication=path.with_name("development.json"),
            development_adjudication_sha256="2" * 64,
            sealed_adjudication=path.with_name("sealed.json"),
            sealed_adjudication_sha256="3" * 64,
            final_step=100,
        )

    with pytest.raises(ValueError, match="cover every task key exactly"):
        LoraRankPartitionConfig(
            mode="task_all_active",
            task_keys=["task-a", "task-b"],
            warm_start_sources={"task-a": source(tmp_path / "same")},
        )

    with pytest.raises(ValueError, match="unique per task"):
        LoraRankPartitionConfig(
            mode="task_all_active",
            task_keys=["task-a", "task-b"],
            warm_start_sources={
                "task-a": source(tmp_path / "same"),
                "task-b": source(tmp_path / "same"),
            },
        )

    with pytest.raises(ValueError, match="requires warm_start_sources"):
        LoraRankPartitionConfig(
            mode="task_all_active",
            task_keys=["task-a", "task-b"],
            composition_admission={
                "checkpoint": tmp_path / "composition",
                "checkpoint_manifest_sha256": "2" * 64,
                "retention_adjudication": tmp_path / "retention.json",
                "retention_adjudication_sha256": "3" * 64,
            },
        )
