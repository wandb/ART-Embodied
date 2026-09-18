# ART-Embodied

以 [OpenPipe ART](https://github.com/OpenPipe/ART) 为基础，为 Physical AI
提供 trajectory-aware reinforcement learning，并以
[LeRobot](https://github.com/huggingface/lerobot) 工作流程为核心设计的实验性框架。

[English](README.md) · [日本语](README.ja.md) · [한국어](README.ko.md) ·
[繁體中文](README.zh-TW.md) ·
[Embodied RL 指南](docs/experimental/embodied-rl.mdx) ·
[示例](examples/embodied/README.md) ·
[发布验证](docs/experimental/embodied-release-validation.mdx) ·
[上游 ART 文档](https://art.openpipe.ai)

<p align="center">
  <img src="docs/assets/art-embodied-trajectory-rl-dashboard.gif" alt="展示 PI0.5 成功率曲线与四组 LIBERO Long 机器人 rollout 的 W&B 仪表板" width="920">
</p>
<p align="center">
  <strong>使用 trajectory-level Flow-SDE GRPO 训练 PI0.5 / LIBERO Long。</strong><br>
  在同一个实时实验视图中追踪成功率曲线与 grouped rollout。
</p>

> [!WARNING]
> **ART-Embodied 目前处于研究预览阶段。**
>
> - **OpenVLA-OFT / GRPO：** 已在 LIBERO 验证轨迹采集、分布式 LoRA 训练、检查点恢复和评估。
> - **OpenVLA-OFT / GSPO：** 已验证分布式执行和检查点；目标函数验证与学习效果比较仍在进行。
> - **PI0、PI0.5、SmolVLA / Flow-SDE GRPO：** 固定开发集上的成功率有所提升，多随机种子与封存测试（sealed test）验证尚未完成。
> - **GR00T N1.7 / Flow-SDE GRPO：** 已完成 RoboCasa 单任务训练及 192 回合的封存测试。
> - **PI0-FAST / GRPO：** 已完成 LIBERO Long 单任务训练，开发评估与封存测试各 100 回合。结果的不确定性与验证范围见下文。

## 添加了什么

LeRobot 提供策略、前后处理器、数据集和机器人环境。
ART-Embodied 增加轨迹分组采集、GRPO/GSPO 训练、检查点管理，
以及评估和 W&B、Weave 记录功能。

```text
LeRobot policy + processors + environment
                   │
                   ▼
       grouped trajectories and rewards
                   │
                   ▼
      trajectory/action-token GRPO or GSPO
                   │
                   ▼
       versioned LoRA checkpoints
                   │
                   ▼
 fixed evaluation + W&B Models + Weave
```

我们以策略在原生环境中的任务成功率衡量训练效果。

## ART lifecycle 与 LeRobot 的责任边界

LeRobot 负责策略、前后处理、环境和动作采样。
ART-Embodied 使用 ART 的模型管理 API，管理轨迹分组、训练、更新步、
检查点、评估和日志。

| 功能 | API |
| --- | --- |
| 模型管理 | 继承 `art.TrainableModel` 的 `EmbodiedTrainableModel` |
| 训练 | `backend.train(model, trajectory_groups, learning_rate=...)` |
| 轨迹分组采集 | `trajectory_group(...)` 和 `gather_trajectory_groups(...)` |
| 结果与更新步 | `TrainResult` 和 `get_step()` |
| 日志 | 按模型记录 W&B 指标、视频和 Weave 轨迹追踪 |
| 策略执行 | LeRobot 前后处理器与动作采样器 |

专用后端处理图像、机器人状态、动作块和采样器对应的似然。
OpenVLA 后端独立于 ART 的 `LocalBackend`、AOM 和 Serverless Training 运行。

可以用 LeRobot 风格的高层 API 执行完整训练循环，也可以用 ART 风格的低层 API
逐步控制轨迹采集和更新。两者共用模型注册逻辑和训练后端。

## 已验证结果

在相同初始状态下，比较 SFT 与训练后策略的成功次数：

| Policy / suite | Objective | Update | SFT | ART-Embodied | Paired lift |
| --- | --- | ---: | ---: | ---: | ---: |
| [OpenVLA-OFT / LIBERO Object](https://wandb.ai/wandb-japan/art-embodied-openvla) | Action-token GRPO | 200 | 34/100 | **100/100** | **+66 个百分点** |
| [OpenVLA-OFT / LIBERO Spatial](https://wandb.ai/wandb-japan/art-embodied-openvla-spatial) | Action-token GRPO | 100 | 48/100 | **88/100** | **+40 个百分点** |
| [PI0 / LIBERO Spatial](https://wandb.ai/wandb-japan/art-embodied-pi0-positive-control-reference/runs/941byojx) | Flow-SDE GRPO | 100 | 63/100 | **99/100** | **+36 个百分点** |
| [PI0.5 / LIBERO Long](https://wandb.ai/wandb-japan/art-embodied-pi05-positive-control-reference/runs/t0a9mnd3) | Flow-SDE GRPO | 250（best dev） | 48/100 | **84/100** | **+36 个百分点** |
| [SmolVLA / LIBERO Long](https://wandb.ai/wandb-japan/art-embodied-smolvla-positive-control-v3/runs/s4xwc2jm) | Flow-SDE GRPO | 180（best dev） | 42/100 | **69/100** | **+27 个百分点** |
| GR00T N1.7 / RoboCasa Cuttingboard-to-Pan（[开发 run](https://wandb.ai/wandb-japan/art-embodied-gr00t-n1d7-robocasa-single-task/runs/1970sjop)、[sealed test](https://wandb.ai/wandb-japan/art-embodied-gr00t-n1d7-robocasa-single-task-sealed/runs/kjpzsj11)） | Flow-SDE GRPO | 100（sealed） | 111/192 | **139/192** | **+14.6 个百分点** |
| [PI0-FAST / LIBERO Long (单任务)](https://wandb.ai/wandb-japan/art-embodied-pi0-fast-long/runs/a1hx8rsf) | Action-token GRPO | 100 (开发评估) | 70/100 | **89/100** | **+19 个百分点** |
| [PI0-FAST / LIBERO Long (单任务)](https://wandb.ai/wandb-japan/art-embodied-pi0-fast-long/runs/ww9b5upu) | Action-token GRPO | 100 (封存测试) | [73/100](https://wandb.ai/wandb-japan/art-embodied-pi0-fast-long/runs/r56aa76y) | **83/100** | **+10 个百分点** |

**评估条件。** 前五行使用固定的 100 个开发场景，初始状态与训练集分离，
任务和指令与训练时相同。这些场景在开发期间反复使用；多随机种子和封存测试
验证尚未完成。PI0.5 和 SmolVLA 展示开发评估最好的检查点，最终得分见下文。

GR00T 在选定检查点后，使用开发期间未接触的 192 个环境种子进行封存测试。
结果覆盖 RoboCasa 的一个任务，多任务和未见任务评估尚未开展。

PI0-FAST 在 LIBERO Long 的“将两个摩卡壶放到炉灶上”任务中，使用一个训练种子和
隔离的 MuJoCo 3.3 环境。表中两行均评估封存测试前选定的最终更新 100 检查点。
封存测试提升 +10 个百分点，配对 95% 置信区间为 [0, 20] 个百分点（p=0.099），
尚未达到 5% 水平的统计显著性。多任务与鲁棒性评估有待开展。
详见[配方与完整结果](docs/experimental/pi0-fast-long-result.md)。

<details>
<summary>训练条件与配对评估详情</summary>

- **OpenVLA-OFT Object：** rank 32/alpha 32 LoRA，200 次更新。100 个评估初始状态独立于 500 个训练状态生成。最终 100/100 与公开的 RLinf GRPO 检查点相同。
- **OpenVLA-OFT Spatial：** 使用相同后端，分别设置 SFT 检查点、任务集、评估集和 W&B 项目。更新 30 时为 82/100，更新 100 时为 88/100。本次比较以 SFT 为基准。
- **PI0 / PI0.5：** 使用 K4/noise 0.5 Flow-SDE 采样器，每次更新采集 1,024 条轨迹。PI0 到更新 130 仍保持 97--99/100。PI0.5 使用 rank 32/alpha 32 LoRA，完成 300 次更新；更新 250 时最高为 84/100，最终为 81/100。
- **SmolVLA：** 共 200 次更新，在更新 100 后扩大动作专家的适配器范围和 rank。更新 180 时最高为 69/100，最终更新 200 时降至 61/100。
- **GR00T N1.7：** RoboCasa GR1 `PnPCounterToCab` 的 Cuttingboard-to-Pan 任务。从按 NVIDIA 配方训练 60k 步的固定 SFT 检查点开始，以 rank 64/alpha 64 LoRA 连续训练 100 次更新。用固定的 64 回合开发集选定检查点，独立审核后进行一次封存评估：[SFT 111/192](https://wandb.ai/wandb-japan/art-embodied-gr00t-n1d7-robocasa-single-task-sealed/runs/j11avofl)，[GRPO 139/192](https://wandb.ai/wandb-japan/art-embodied-gr00t-n1d7-robocasa-single-task-sealed/runs/kjpzsj11)。训练使用 Flow-SDE，评估使用官方 ODE 采样器，并保留官方处理器、机器人构型和归一化设置。

上方的 W&B 链接包含指标、视频、模型制品和轨迹追踪。
复现配方和初始状态清单位于 `examples/embodied/`。

| 比较 | 改善 / 退步回合数 | 配对 95% 置信区间（百分点） | 精确 McNemar 检验 p |
| --- | ---: | ---: | ---: |
| OpenVLA-OFT Object | 66 / 0 | | `2.71e-20` |
| OpenVLA-OFT Spatial | 43 / 3 | | `4.62e-10` |
| PI0 | 36 / 0 | `[+27,+46]` | `2.91e-11` |
| PI0.5 | 38 / 2 | `[+26,+46]` | `1.49e-9` |
| SmolVLA | 33 / 6 | `[+16,+38]` | `1.43e-5` |
| GR00T N1.7 | 净改善 28 回合 | `[+5.2,+24.0]` | `0.00335` |

</details>

## 目前支持情况

| 功能 | 状态 |
| --- | --- |
| LeRobot / Gymnasium 轨迹采集适配器 | 已实现 |
| 固定初始状态的配对评估 | 已实现 |
| 单 GPU 与本地多 GPU LoRA 训练 | 已实现 |
| 多个 actor 共享策略的批量推理 | 已实现 |
| 包含优化器和 RNG 状态的检查点与恢复 | 已实现 |
| W&B 指标、视频、Table 和模型制品 | 已实现 |
| Weave 轨迹追踪与视频链接 | 已实现 |

### 支持单一 GPU 的资源配置

单 GPU 运行时，将两个设备列表设为同一张 GPU，并使用 `per_update`。
评估、轨迹采集和训练依次执行，训练前释放轨迹采集使用的模型副本。

```yaml
runtime:
  rollout_devices: [cuda:0]
  training_devices: [cuda:0]
  distributed_training: false
  rollout_execution:
    lifecycle: per_update
```

`per_update` 在每个阶段结束时关闭工作进程，减少 GPU 和主机内存占用。
`cpu_offload` 将工作进程保留在主机 RAM 中，以减少启动时间。
详见 [PI0.5 单 GPU 示例](examples/embodied/pi05_libero_object_flow_sde_grpo_single_gpu.yaml)。

## 安装

> **LIBERO 初始状态检查：** 依赖检查通过并不保证初始场景符合预期。
> MuJoCo 更新可能改变 reset 期间的物体位置（Spatial task 5 已有报告）。
> 请同时复现配方的模拟器版本和 reset 设置，并参阅无需模型的[检查与复现指南](docs/experimental/libero-reset-health.md)。
> 检查目前需要显式运行，不会修改既有配方或自动修正初始状态。

### 已验证的兼容性

ART-Embodied 与 OpenPipe ART 一起安装，包名为 `art_embodied`。
`0.1.0rc2` 根据运行环境支持 ART 0.5.18 和 0.5.20：

| 运行环境 | OpenPipe ART | 用途 |
| --- | --- | --- |
| Python 3.12 及以上默认环境 | `0.5.20` | PI0 / PI0.5、PI0-FAST、SmolVLA 配置 |
| Python 3.11 | `0.5.18` | 既有 OpenVLA-OFT 配置（LeRobot `>=0.4.4,<0.5`） |
| GR00T N1.7 专用 Python 3.12 环境 | `0.5.18` | NVIDIA 固定依赖的原生环境 |

以下安装命令指定 ART 版本。NVIDIA 要求 SciPy 1.15.3，而 ART 0.5.20 要求
SciPy 1.17，因此 GR00T 安装脚本保留 ART 0.5.18。既有 ART 0.5.18 环境仍受支持。
ART 0.5.20 已通过包安装、数值计算、更新与恢复及 W&B 检查；具体范围请参阅
[兼容性验证报告](docs/experimental/upstream-art-compatibility.md)。
标准环境使用 `constraints/security.txt`，NVIDIA 专用环境保留单独的版本约束。
安装方式的差异及注意事项见[依赖配置与安全](docs/experimental/dependency-security.md)。

通过 `import art` 导入 ART，通过 `import art_embodied as embodied` 导入附加组件。

ART 0.5.18 和 ART-Embodied 均支持 Python 3.11。已验证的 OpenVLA-OFT
配置可在同一个隔离环境中运行 ART 和策略。

请使用隔离环境。Robotics policy stack 经常固定与 ART LLM backend 不同的
Torch、Transformers 与 simulator 版本。

若要在干净环境中重现 add-on 的安装顺序：

```bash
git clone https://github.com/wandb/ART-Embodied.git
cd ART-Embodied
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints/security.txt 'openpipe-art==0.5.18'
python -m pip install -c constraints/security.txt '.[libero]'
art-embodied doctor --profile libero
```

此安装过程已在全新的 Python 3.11 环境中验证：从安装包加载 OpenVLA-OFT，
完成一次 LoRA GRPO 更新，保存适配器与训练状态，并恢复检查点。

若在 checkout 中以 uv 开发：

请使用 uv 0.12.0 或更新版本。`uv sync --locked` 会覆盖依赖约束，使用
LiteLLM 1.101.0 和 Diffusers 0.38.0。GR00T N1.7 安装脚本只覆盖 Diffusers，
并使用 Safetensors 0.8.0。这些设置替换上游的旧版本约束，不修改上游包。
普通 pip 安装不会应用这些覆盖设置。

```bash
git clone https://github.com/wandb/ART-Embodied.git
cd ART-Embodied
uv sync --python 3.11 --extra lerobot
```

若使用已验证的 LIBERO integration：

```bash
uv sync --python 3.11 --extra libero
```

PI0/PI0.5 Flow-SDE 请使用独立的 Python 3.12 环境，安装
LeRobot 0.6 和 Transformers 5：

```bash
python3.12 -m venv .venv-pi
source .venv-pi/bin/activate
python -m pip install -c constraints/security.txt 'openpipe-art==0.5.20' '.[pi-libero]'
art-embodied doctor --profile pi
```

请为不同用途分别创建环境：

- `lerobot`：不含模型专用及模拟器额外依赖的 LeRobot。
- `libero`：已验证的 OpenVLA-OFT/LIBERO 配置，使用 Torch 2.6.0、Transformers 4.40.1、PEFT 0.11.1 和 NumPy 1.26.4。
- `pi` / `pi-libero`：LeRobot 0.6 与 PI 原生采样器。

请勿混装这些配置。用 `lerobot[libero,peft]` 替换 `libero` 会在 LeRobot 0.4.4 下
改变 Transformers 和 PEFT 版本，即使检查点能正常加载，动作 logits 也可能发生变化。

在配置 GPU 前，请先检查已安装的 control-plane stack：

```bash
uv run art-embodied doctor

# 在通用 LeRobot worker profile 中使用。
uv run art-embodied doctor --require-lerobot

# 为 OpenVLA-OFT/LIBERO profile 配置 GPU 前使用。
uv run art-embodied doctor --profile libero
```

对 process-isolated policy environment 使用 `--worker`；它会验证选定的
package profile，而不要求 worker process 中存在 ART。只有通用 LeRobot
profile 才加上 `--require-lerobot`。在 CI 或 launch script 中加入 `--json`，
即可用程序读取相同报告。

OpenVLA-OFT v0.1 需要专用 `libero` 环境或相同配置的容器。
启动时会检查并拒绝可能影响推理结果的依赖版本变化。

若 native policy dependency 仍需 process isolation，请在两个环境都安装
`art-embodied` wheel，并明确选择 policy environment：

```yaml
runtime:
  # Used by rollout actors, batched inference servers, and training workers.
  worker_python_executable: /opt/venv/openvla/bin/python
```

`null` 使用当前 Python。独立工作进程环境需要安装 `art-embodied` 及策略、模拟器依赖。
工作进程直接以参数列表启动，不经过 shell；加载策略依赖时不初始化 ART。

对于 GR00T N1.7 与 RoboCasa，请将 policy 与 simulator 分别置于两个固定的
Python 3.12 环境中：

GR00T N1.7 安装脚本需要 Git LFS、micromamba、CMake 和 C++ 构建工具。

```bash
./scripts/install-gr00t-n1d7-runtime.sh
./scripts/install-robocasa-gr1-runtime.sh
./scripts/download-robocasa-gr1-dataset.sh
```

安装脚本将 NVIDIA Isaac-GR00T 和 CUDA/Torch 与 RoboCasa、robosuite、MuJoCo
分别固定版本。数据集也按指定修订版本下载，并生成供训练脚本使用的验证报告。
开发、继续训练和封存评估的步骤见
[GR00T 示例](examples/embodied/README.md#gr00t-n17--robocasa-flow-sde)。

### 可携式 Slurm 启动

使用 Slurm 时，仓库中的启动脚本会定位代码目录并读取 `.env`。
请将集群特定路径和账号设置放在可复用作业文件之外：

```bash
sbatch --gres=gpu:h100:8 --cpus-per-task=96 --mem=690G \
  scripts/slurm/run-in-repo.sh \
  uv run python examples/embodied/libero/train_openvla_oft.py \
  --config examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml
```

只有在 secret 位于 checkout 外部时，才于提交时设置
`ART_EMBODIED_ENV_FILE`：

```bash
ART_EMBODIED_ENV_FILE="${HOME}/.config/art-embodied/secrets.env" \
  sbatch --export=ALL scripts/slurm/run-in-repo.sh COMMAND [ARG ...]
```

Slurm 缺省会导出提交时的环境。因此重新 clone 到其他用户、home directory、
cluster 或 region 后，wrapper 仍可运作，不需修改 job file。Experiment
condition 应保留在 YAML；只有 secret 与 machine-local location 可以使用
environment override。

## 运行 OpenVLA-OFT control

Experiment condition 存放于 YAML。Environment variable 仅用于
`WANDB_API_KEY` 等 secret。

```bash
uv run art-embodied validate \
  examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml

uv run python examples/embodied/libero/train_openvla_oft.py \
  --config examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml \
  --preflight

uv run python examples/embodied/libero/train_openvla_oft.py \
  --config examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml
```

`validate` 和 `--preflight` 会在模型加载前检查轨迹形状、优化器行、设备、
评估场景和模拟器资源。添加新集成可从
[`lerobot_action_token_grpo.template.yaml`](examples/embodied/lerobot_action_token_grpo.template.yaml) 开始。

训练前，创建与训练初始状态分离的固定评估清单，并评估一次 SFT。
定期评估复用同一清单和 SFT 结果。选择配方或检查点时，使用
`evaluation.data_role: development`。

确定方法和检查点选择规则后，使用新清单进行 `data_role: sealed_test`、
`checkpoint_selection: last` 评估。系统会拒绝从封存结果选择 `best` 的配置。
清单准备步骤见示例指南。

每次固定评估保存原始结果和复现信息：结果、清单、源代码的哈希，Git 状态，
依赖及运行环境、容器版本，还有被评估的模型、适配器或检查点标识。
启用 W&B 时，`log_evaluation_artifacts: true` 将两个文件上传为带版本的评估制品。
相同文件也保存在本地。

不创建 optimizer，直接评估固定 SFT baseline：

```bash
uv run python examples/embodied/libero/train_openvla_oft.py \
  --config examples/embodied/openvla_oft_libero_object_sft_baseline_eval.yaml \
  --evaluate-only --evaluation-step 0
```

不恢复 optimizer，直接评估已保存的 ART-Embodied policy snapshot：

```bash
uv run python examples/embodied/libero/train_openvla_oft.py \
  --config experiment-eval.yaml --evaluate-only --evaluation-step 5 \
  --policy-checkpoint outputs/my-run/checkpoints/step-000005/policy
```

评估 YAML 中的策略类型、基础模型修订版本、前后处理器、环境和固定场景
必须与检查点一致。

Candidate preparation、paired evaluation、generated state manifest 与
conformance tool 请见
[`examples/embodied/README.md`](examples/embodied/README.md)。

## 连接既有 LeRobot 工作流程

将应用中的 LeRobot 策略、前后处理器和环境传给 `run_lerobot_experiment`。
它管理模型注册、轨迹分组采集、训练、日志、检查点和清理。
对于动作 token 策略，适配器记录采样 token 及采集时的对数概率。

```python
import asyncio

import art_embodied as embodied
from my_robot_app import (
    evaluation_scenarios,
    make_environment,
    policy,
    postprocessor,
    preprocessor,
    record_action_tokens,
    train_scenarios,
)


async def main() -> None:
    config = embodied.EmbodiedExperimentConfig.from_yaml("experiment.yaml")
    adapter = embodied.LeRobotPolicyAdapter(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=config.policy.device,
        record_action_fn=record_action_tokens,
    )
    result = await embodied.run_lerobot_experiment(
        config=config,
        policy=policy,
        train_scenarios=train_scenarios,
        evaluation_scenarios=evaluation_scenarios,
        environment_factory=make_environment,
        policy_adapter=adapter,
    )
    print(result.config_fingerprint)


asyncio.run(main())
```

如需自行控制轨迹分组采集和训练更新，请使用低层 API：

```python
import art_embodied as embodied

config = embodied.EmbodiedExperimentConfig.from_yaml("experiment.yaml")

model = embodied.EmbodiedTrainableModel(policy=policy, config=config)
native_backend = embodied.make_embodied_backend(config, policy=policy)
backend = embodied.EmbodiedBackend(native_backend, config=config)
await model.register(backend)

groups = await embodied.gather_trajectory_groups(
    [
        embodied.trajectory_group(
            (rollout(model.policy, scenario) for _ in range(config.algorithm.group_size)),
            metadata={"scenario_id": scenario.id},
        )
        for scenario in train_scenarios
    ]
)
result = await backend.train(
    model,
    groups,
    learning_rate=config.training.optimizer.learning_rate,
)
await model.log(groups, split="train", metrics=result.metrics, step=result.step)
await model.close()
```

`EmbodiedTrainableModel` 继承 `art.TrainableModel`。
各策略后端使用与原生采样器对应的训练目标函数。

## 不改变实验即可扩展规模

Slurm 为可选项。通过 YAML 设置设备、actor 数、推理副本数、工作进程生命周期和
微批量大小，可在工作站或集群上运行同一实验。

1. 先用每张采集 GPU 一个 actor、一个模型验证一次更新。
2. 增加 actor，让模拟与推理并行。
3. 使用 `batched_server` 在多个 actor 之间共享策略。
4. 根据 GPU 内存余量和实测吞吐量增加推理副本。
5. 使用本地分布式训练缩短优化器耗时。
6. 采集与训练共用 GPU 时使用 `cpu_offload`，并为所有副本预留主机 RAM。

在 80 GB H100 上，OpenVLA-OFT 的实测配置为每 GPU 三个推理副本、六个 actor、
批量大小 2、训练微批量 12。1,024 条轨迹的采集速度为 `1.536 trajectories/s`，
超过初始实现的两倍。请根据策略和硬件调整，并保留 10–20% 的 GPU 内存余量。

## W&B 与 Weave

W&B 将 SFT 评估记在 Step 0，后续定期评估添加到同一条
`validation/success_rate` 曲线。每次训练更新记录一行历史，并保存视频、
评估 Table 和带版本的模型及训练状态制品。Weave 按更新、分组、轨迹组织追踪，
并链接对应视频。

| 分区 | 内容 |
| --- | --- |
| `train/*` | `train/success_rate`、`train/reward_mean` |
| `validation/*` | 汇总评估指标 |
| `signal/*` | 奖励、advantage、有效分组 |
| `optimization/*` | 损失、KL、似然比、梯度 |
| `performance/*` | 耗时、吞吐量、内存 |
| `train_details/*` | 回合数和长度 |
| `media/simulation/*` | 训练与评估视频 |

各任务、各回合的详细结果保存在 Table 和评估制品中。
W&B 支持需要安装 `observability` extra。ART 0.5.18 已包含 Weave 客户端；
两种集成都只在 YAML 启用后发送数据。

```yaml
observability:
  delivery_failure_policy: fail_run
  wandb:
    enabled: true
    project: art-embodied
    mode: online
    log_model_artifacts: true
    log_evaluation_table: true
  weave:
    enabled: true
    project: art-embodied
    trace_trajectories: true
    max_groups_per_update: 4
    max_trajectories_per_group: 4
  videos_per_update: 2
  videos_per_evaluation: 8
```

关闭两种集成后，训练仍会在本地保存检查点、视频和 JSON 评估结果。

检查点包含策略、优化器、随机数生成器（RNG）状态及各自的哈希。
写入完成标记并原子发布后，检查点即可用于恢复。恢复前会检查文件完整性和训练设置兼容性。

示例使用 `delivery_failure_policy: fail_run`：W&B 或 Weave 发送失败时，
将错误写入有大小限制的 `telemetry_failures.jsonl`，然后停止运行。
如需在发送失败时继续训练并保留本地记录，可使用 `best_effort`。
两种模式都会保留已完成的更新和检查点。

恢复已有 Step 0 评估的实验时，将 `evaluation.baseline_outcomes_path` 指向
该结果文件，并关闭 `evaluate_before_training`。启动前会检查此引用，
以便继续计算配对比较、置信区间和 McNemar 检验。

## 边界

- GRPO 更新需要完整的轨迹组。
- 训练诊断指标与策略的原生评估结果分开记录。
- 视频和追踪数量可在 YAML 中设置。
- 模拟器特有行为由环境适配器处理。

## 文档

- [Embodied RL 概念与设置](docs/experimental/embodied-rl.mdx)
- [Runtime 与 scaling architecture](docs/experimental/embodied-runtime-architecture.mdx)
- [Release validation contract](docs/experimental/embodied-release-validation.mdx)
- [OpenVLA-OFT 与 LIBERO 示例](examples/embodied/README.md)

## 致谢

感谢 [RLinf](https://github.com/RLinf/RLinf) 作者公开实现、训练配方和检查点。
我们参考这些资料验证 OpenVLA-OFT 轨迹采集、动作 token 掩码、advantage 与损失聚合，
以及 LIBERO 评估。

`rlinf_v01` 是对应参考条件的验证配置名称。ART-Embodied 使用自己的运行时
（`model_loader: native`），运行时不依赖 RLinf。

## 与 ART 的关系

ART-Embodied 是 OpenPipe ART 的附加组件，机器人相关依赖可按需安装。
欢迎贡献可复现的基准、策略和模拟器适配器，以及 W&B、Weave 集成。
详见 [CONTRIBUTING.md](CONTRIBUTING.md)。
