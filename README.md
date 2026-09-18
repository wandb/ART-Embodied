# ART-Embodied

Trajectory-aware reinforcement learning for Physical AI, built on
[OpenPipe ART](https://github.com/OpenPipe/ART) and designed for
[LeRobot](https://github.com/huggingface/lerobot) workflows.

[日本語](README.ja.md) · [한국어](README.ko.md) ·
[简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) ·
[Embodied RL guide](docs/experimental/embodied-rl.mdx) ·
[Examples](examples/embodied/README.md) ·
[Release validation](docs/experimental/embodied-release-validation.mdx) ·
[Upstream ART docs](https://art.openpipe.ai)

<p align="center">
  <img src="docs/assets/art-embodied-trajectory-rl-dashboard.gif" alt="W&B dashboard showing PI0.5 success-rate curves and four LIBERO Long robot rollouts" width="920">
</p>
<p align="center">
  <strong>PI0.5 / LIBERO Long with trajectory-level Flow-SDE GRPO.</strong><br>
  Success-rate curves and grouped rollouts in one live experiment view.
</p>

> [!WARNING]
> **ART-Embodied is a research preview.**
>
> - **OpenVLA-OFT / GRPO:** Rollout, distributed LoRA training, checkpoint reload, and evaluation tested on LIBERO.
> - **OpenVLA-OFT / GSPO:** Distributed execution and checkpoints tested; objective validation and learning comparisons are ongoing.
> - **PI0, PI0.5, SmolVLA / Flow-SDE GRPO:** Success rates improved on fixed development sets. Multi-seed and sealed-test validation remain open.
> - **GR00T N1.7 / Flow-SDE GRPO:** Single-task RoboCasa result with a 192-episode sealed test.
> - **PI0-FAST / GRPO:** Single-task LIBERO Long result with 100 development and 100 sealed episodes. See the results below for uncertainty and scope.

## What it adds

LeRobot provides policies, processors, datasets, and robot environments.
ART-Embodied adds grouped rollouts, GRPO/GSPO training, checkpoint management,
and evaluation with W&B and Weave.

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

We measure progress by task success in the policy's native environment.

## ART lifecycle, LeRobot ownership

LeRobot runs the policy, processors, environment, and action sampler.
ART-Embodied groups trajectories, trains the policy, and tracks updates,
checkpoints, evaluations, and logs through ART's model lifecycle.

| Component | API |
| --- | --- |
| Model lifecycle | `EmbodiedTrainableModel`, a subclass of `art.TrainableModel` |
| Training | `backend.train(model, trajectory_groups, learning_rate=...)` |
| Rollout groups | `trajectory_group(...)` and `gather_trajectory_groups(...)` |
| Results and updates | `TrainResult` and `get_step()` |
| Logging | Model-owned W&B metrics, videos, and Weave traces |
| Policy execution | LeRobot processors and action sampler |

The embodied backends handle images, robot states, action chunks, and
sampler-specific likelihoods. The OpenVLA backend runs independently of ART's
`LocalBackend`, AOM, and Serverless Training.

Choose the LeRobot-style high-level helper for a complete training loop, or the
ART-style low-level API to control rollout collection and updates. Both use the
same model registration and training backend.

## Validated results

Success counts before and after training, evaluated on matched initial states:

| Policy / suite | Objective | Update | SFT | ART-Embodied | Paired lift |
| --- | --- | ---: | ---: | ---: | ---: |
| [OpenVLA-OFT / LIBERO Object](https://wandb.ai/wandb-japan/art-embodied-openvla) | Action-token GRPO | 200 | 34/100 | **100/100** | **+66 points** |
| [OpenVLA-OFT / LIBERO Spatial](https://wandb.ai/wandb-japan/art-embodied-openvla-spatial) | Action-token GRPO | 100 | 48/100 | **88/100** | **+40 points** |
| [PI0 / LIBERO Spatial](https://wandb.ai/wandb-japan/art-embodied-pi0-positive-control-reference/runs/941byojx) | Flow-SDE GRPO | 100 | 63/100 | **99/100** | **+36 points** |
| [PI0.5 / LIBERO Long](https://wandb.ai/wandb-japan/art-embodied-pi05-positive-control-reference/runs/t0a9mnd3) | Flow-SDE GRPO | 250 (best dev) | 48/100 | **84/100** | **+36 points** |
| [SmolVLA / LIBERO Long](https://wandb.ai/wandb-japan/art-embodied-smolvla-positive-control-v3/runs/s4xwc2jm) | Flow-SDE GRPO | 180 (best dev) | 42/100 | **69/100** | **+27 points** |
| GR00T N1.7 / RoboCasa Cuttingboard-to-Pan ([development run](https://wandb.ai/wandb-japan/art-embodied-gr00t-n1d7-robocasa-single-task/runs/1970sjop), [sealed test](https://wandb.ai/wandb-japan/art-embodied-gr00t-n1d7-robocasa-single-task-sealed/runs/kjpzsj11)) | Flow-SDE GRPO | 100 (sealed) | 111/192 | **139/192** | **+14.6 points** |
| [PI0-FAST / LIBERO Long (single task)](https://wandb.ai/wandb-japan/art-embodied-pi0-fast-long/runs/a1hx8rsf) | Action-token GRPO | 100 (development) | 70/100 | **89/100** | **+19 points** |
| [PI0-FAST / LIBERO Long (single task)](https://wandb.ai/wandb-japan/art-embodied-pi0-fast-long/runs/ww9b5upu) | Action-token GRPO | 100 (sealed) | [73/100](https://wandb.ai/wandb-japan/art-embodied-pi0-fast-long/runs/r56aa76y) | **83/100** | **+10 points** |

**How to read the results.** The first five rows use 100 fixed development
scenarios, with initial states held out from training. The tasks and instructions
stay the same. These sets were used throughout development; multi-seed and
sealed-test validation are still pending. PI0.5 and SmolVLA report their best
development checkpoint; their final scores are listed below.

GR00T uses 192 sealed environment seeds, evaluated after checkpoint selection.
Its result covers one RoboCasa task; multitask and held-out-task evaluations
remain open.

PI0-FAST uses one LIBERO Long task, "put both moka pots on the stove", with one
training seed and an isolated MuJoCo 3.3 runtime. Both rows evaluate the last
checkpoint at update 100, selected before sealed testing. The sealed +10-point
estimate has a paired 95% CI of [0, 20] points (p=0.099), so the comparison is
inconclusive at the 5% significance level. Multitask and robustness evaluations
remain open. See the [recipe and full results](docs/experimental/pi0-fast-long-result.md).

<details>
<summary>Recipe and paired-evaluation details</summary>

- **OpenVLA-OFT Object:** Rank-32/alpha-32 LoRA, 200 updates. The 100 evaluation states were generated independently of the 500 training states. The final 100/100 matched the public RLinf GRPO checkpoint.
- **OpenVLA-OFT Spatial:** The same backend, with a separate SFT checkpoint, suite, evaluation set, and W&B project. It scored 82/100 at update 30 and 88/100 at update 100. This run was evaluated against SFT.
- **PI0 / PI0.5:** K4/noise-0.5 Flow-SDE sampler and 1,024 trajectories per update. PI0 stayed at 97--99/100 through update 130. PI0.5 used rank-32/alpha-32 LoRA and finished 300 updates at 81/100, after peaking at 84/100 at update 250.
- **SmolVLA:** 200 updates, with the action-expert adapter scope and rank expanded after update 100. Success peaked at 69/100 at update 180 and fell to 61/100 at update 200.
- **GR00T N1.7:** RoboCasa GR1 `PnPCounterToCab`, Cuttingboard-to-Pan. Rank-64/alpha-64 LoRA, 100 uninterrupted updates from the pinned 60k-step SFT checkpoint trained with NVIDIA's recipe. Checkpoint selection used 64 development episodes, followed by an independent audit and one sealed evaluation: [SFT 111/192](https://wandb.ai/wandb-japan/art-embodied-gr00t-n1d7-robocasa-single-task-sealed/runs/j11avofl) versus [GRPO 139/192](https://wandb.ai/wandb-japan/art-embodied-gr00t-n1d7-robocasa-single-task-sealed/runs/kjpzsj11). Training uses Flow-SDE; evaluation uses the native ODE sampler with the official processor, embodiment, and normalization.

The W&B links above include metrics, videos, model artifacts, and trajectory
traces. Reproduction recipes and state manifests are under `examples/embodied/`.

| Comparison | Improved / regressed episodes | Paired 95% CI (points) | Exact McNemar p |
| --- | ---: | ---: | ---: |
| OpenVLA-OFT Object | 66 / 0 | | `2.71e-20` |
| OpenVLA-OFT Spatial | 43 / 3 | | `4.62e-10` |
| PI0 | 36 / 0 | `[+27,+46]` | `2.91e-11` |
| PI0.5 | 38 / 2 | `[+26,+46]` | `1.49e-9` |
| SmolVLA | 33 / 6 | `[+16,+38]` | `1.43e-5` |
| GR00T N1.7 | 28 net improvements | `[+5.2,+24.0]` | `0.00335` |

</details>

## Current support

| Capability | Status |
| --- | --- |
| LeRobot Gymnasium rollout adapter | Implemented |
| Fixed and paired evaluation | Implemented |
| Single-GPU and local multi-GPU LoRA training | Implemented |
| Batched shared-policy rollout | Implemented |
| Checkpoint/resume with optimizer and RNG state | Implemented |
| W&B metrics, videos, Tables, and Models artifacts | Implemented |
| Weave nested trajectory traces and file-backed video | Implemented |

### One GPU is a supported resource profile

To run on one GPU, set both device lists to the same accelerator and use
`per_update`. Evaluation, rollout collection, and training run in sequence;
rollout replicas are released before training:

```yaml
runtime:
  rollout_devices: [cuda:0]
  training_devices: [cuda:0]
  distributed_training: false
  rollout_execution:
    lifecycle: per_update
```

`per_update` shuts down rollout workers between phases to reduce GPU and host
memory use. `cpu_offload` keeps workers in host RAM to reduce startup time.
See the [single-GPU PI0.5 example](examples/embodied/pi05_libero_object_flow_sde_grpo_single_gpu.yaml).

## Install

> **LIBERO reset health:** A passing dependency check does not guarantee the
> intended initial scene: newer MuJoCo versions can change object placement during
> reset (reported for Spatial task 5). Keep the recipe's simulator version and reset
> settings together; see the policy-free [check and reproduction guide](docs/experimental/libero-reset-health.md).
> This check is currently opt-in; it does not change existing recipes or repair states.

### Validated compatibility

ART-Embodied installs the `art_embodied` package alongside OpenPipe ART.
Release `0.1.0rc2` supports ART 0.5.18 and 0.5.20 in separate runtime profiles:

| Environment | OpenPipe ART | Use |
| --- | --- | --- |
| Python 3.12+ default | `0.5.20` | PI0 / PI0.5, PI0-FAST, SmolVLA profiles |
| Python 3.11 | `0.5.18` | Existing OpenVLA-OFT profile (LeRobot `>=0.4.4,<0.5`) |
| GR00T N1.7 dedicated Python 3.12 environment | `0.5.18` | NVIDIA's pinned native runtime |

The install commands below select the ART version; the GR00T installer retains
0.5.18 because NVIDIA pins SciPy 1.15.3, while ART 0.5.20 requires SciPy 1.17.
Existing ART 0.5.18 environments remain supported. ART 0.5.20 passed package,
numerical, update/resume, and W&B checks; see the
[compatibility report](docs/experimental/upstream-art-compatibility.md) for coverage.
Standard installs apply `constraints/security.txt`; native NVIDIA environments
keep separate pins. See [dependency profiles and security](docs/experimental/dependency-security.md)
for installation differences and remaining precautions.

Import ART with `import art` and the add-on with `import art_embodied as embodied`.

ART 0.5.18 and ART-Embodied support Python 3.11. The validated OpenVLA-OFT
profile runs ART and the native policy in one isolated environment.

Use an isolated environment. Robotics policy stacks often pin Torch,
Transformers, and simulator versions that differ from ART's LLM backends.

To reproduce the add-on installation order in a clean environment:

```bash
git clone https://github.com/wandb/ART-Embodied.git
cd ART-Embodied
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints/security.txt 'openpipe-art==0.5.18'
python -m pip install -c constraints/security.txt '.[libero]'
art-embodied doctor --profile libero
```

This installation was tested in a fresh Python 3.11 environment: OpenVLA-OFT
loaded, completed one LoRA GRPO update, saved its adapter and training state,
and reloaded the checkpoint from the installed package.

For uv-based development in the checkout:

Use uv 0.12.0 or newer. `uv sync --locked` applies compatibility overrides for
LiteLLM 1.101.0 and Diffusers 0.38.0. The GR00T N1.7 installer applies only the
Diffusers override, with Safetensors 0.8.0. These overrides replace older upstream
requirements without editing upstream packages. Plain pip does not apply them.

```bash
git clone https://github.com/wandb/ART-Embodied.git
cd ART-Embodied
uv sync --python 3.11 --extra lerobot
```

For the validated LIBERO integration:

```bash
uv sync --python 3.11 --extra libero
```

Use a separate Python 3.12 environment for PI0/PI0.5 Flow-SDE, with
LeRobot 0.6 and Transformers 5:

```bash
python3.12 -m venv .venv-pi
source .venv-pi/bin/activate
python -m pip install -c constraints/security.txt 'openpipe-art==0.5.20' '.[pi-libero]'
art-embodied doctor --profile pi
```

LeRobot's PI processor uses the gated
[`google/paligemma-3b-pt-224`](https://huggingface.co/google/paligemma-3b-pt-224)
tokenizer. Accept its terms once, then run `hf auth login` or export `HF_TOKEN`.
ART-Embodied resolves this small tokenizer before the multi-gigabyte policy
checkpoint and reports an actionable error if access is missing. The
coordinator completes the initial download before worker startup, so users do
not need to prewarm each rollout worker manually.

Choose a separate environment for each profile:

- `lerobot`: LeRobot without model-specific or simulator extras.
- `libero`: The tested OpenVLA-OFT/LIBERO stack: Torch 2.6.0, Transformers 4.40.1, PEFT 0.11.1, and NumPy 1.26.4.
- `pi` / `pi-libero`: LeRobot 0.6 with native PI samplers.

Keep these profiles isolated. Replacing `libero` with `lerobot[libero,peft]`
changes Transformers and PEFT versions under LeRobot 0.4.4 and can change action
logits even when the checkpoint loads successfully.

Check the installed control-plane stack before allocating a GPU:

```bash
uv run art-embodied doctor

# Use this in the generic LeRobot worker profile.
uv run art-embodied doctor --require-lerobot

# Use this before allocating a GPU for the OpenVLA-OFT/LIBERO profile.
uv run art-embodied doctor --profile libero
```

For a process-isolated policy environment, use `--worker`; this validates the
selected package profile without requiring ART in that worker process. Add
`--require-lerobot` only to the generic LeRobot profile. Add `--json` in CI or
launch scripts to consume the same report programmatically.

OpenVLA-OFT v0.1 requires the dedicated `libero` environment or an equivalent
container. Startup checks reject dependency versions that can change inference.

When native policy dependencies still require process isolation, install the
`art-embodied` wheel in both environments and select the policy environment
explicitly:

```yaml
runtime:
  # Used by rollout actors, batched inference servers, and training workers.
  worker_python_executable: /opt/venv/openvla/bin/python
```

`null` uses the current Python interpreter. A separate worker environment needs
`art-embodied` and the policy/simulator dependencies. Workers launch directly as
an argument list and load policy dependencies without initializing ART.

For GR00T N1.7 with RoboCasa, keep the policy and simulator in separate pinned
Python 3.12 environments:

The GR00T N1.7 installer requires Git LFS, micromamba, CMake, and C++ build tools.

```bash
./scripts/install-gr00t-n1d7-runtime.sh
./scripts/install-robocasa-gr1-runtime.sh
./scripts/download-robocasa-gr1-dataset.sh
```

The installers pin NVIDIA Isaac-GR00T and CUDA/Torch separately from RoboCasa,
robosuite, and MuJoCo. The dataset download is revision-pinned and writes a
verification report for the training scripts. See the
[GR00T examples](examples/embodied/README.md#gr00t-n17--robocasa-flow-sde)
for development, continuation, and sealed evaluation.

### Portable Slurm launch

For Slurm, use the repository wrapper to resolve the checkout and load its
`.env`. Keep cluster-specific paths and account settings outside reusable job files:

```bash
sbatch --gres=gpu:h100:8 --cpus-per-task=96 --mem=690G \
  scripts/slurm/run-in-repo.sh \
  uv run python examples/embodied/libero/train_openvla_oft.py \
  --config examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml
```

Set `ART_EMBODIED_ENV_FILE` at submission time only when secrets live outside
the checkout:

```bash
ART_EMBODIED_ENV_FILE="${HOME}/.config/art-embodied/secrets.env" \
  sbatch --export=ALL scripts/slurm/run-in-repo.sh COMMAND [ARG ...]
```

Slurm exports the submission environment and original `SLURM_SUBMIT_DIR` by
default. The wrapper uses that directory after Slurm copies the script into its
spool, and falls back to the script location when run directly. It therefore
works after cloning under another user, home directory, cluster, or region
without editing the job file. Set `ART_EMBODIED_REPO_ROOT` only when submitting
from outside the checkout. Experiment conditions remain in YAML; only secrets
and machine-local locations may use environment overrides.

The GPU image must expose the NVIDIA driver and a GLVND EGL loader
(`libEGL.so.1`; package `libegl1` on Ubuntu) for headless LIBERO rendering.
Cluster images should install it system-wide. For an immutable image, an
administrator-provided compatible loader may instead be placed in the ignored
repo-relative `.runtime/lib/` directory; the portable wrapper prepends that
directory without recording a machine-specific path in YAML.

## Run the OpenVLA-OFT control

Experiment conditions live in YAML. Environment variables are reserved for
secrets such as `WANDB_API_KEY`.

```bash
uv run art-embodied validate \
  examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml

uv run python examples/embodied/libero/train_openvla_oft.py \
  --config examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml \
  --preflight

uv run python examples/embodied/libero/train_openvla_oft.py \
  --config examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml
```

`validate` and `--preflight` check trajectory shapes, optimizer rows, devices,
evaluation scenarios, and simulator assets before loading the model. For a new
integration, start with
[`lerobot_action_token_grpo.template.yaml`](examples/embodied/lerobot_action_token_grpo.template.yaml).

Before training, create a fixed evaluation manifest with initial states held out
from training and evaluate the SFT baseline once. Reuse that manifest and baseline
for periodic evaluations. Use `evaluation.data_role: development` while selecting
recipes or checkpoints.

After choosing the method and checkpoint selection rule, evaluate a fresh manifest
with `data_role: sealed_test` and `checkpoint_selection: last`. Selecting `best`
from sealed outcomes is rejected. See the examples guide for manifest preparation.

Each fixed evaluation saves raw outcomes and a provenance index: outcome,
manifest, and source hashes; Git state; dependency and runtime/container versions;
and the evaluated model/adapter or checkpoint identity. With W&B enabled,
`log_evaluation_artifacts: true` uploads both files as a versioned evaluation
artifact. The files are also saved locally.

Evaluate the fixed SFT baseline without constructing an optimizer:

```bash
uv run python examples/embodied/libero/train_openvla_oft.py \
  --config examples/embodied/openvla_oft_libero_object_sft_baseline_eval.yaml \
  --evaluate-only --evaluation-step 0
```

Evaluate a saved ART-Embodied policy snapshot without resuming its optimizer:

```bash
uv run python examples/embodied/libero/train_openvla_oft.py \
  --config experiment-eval.yaml --evaluate-only --evaluation-step 5 \
  --policy-checkpoint outputs/my-run/checkpoints/step-000005/policy
```

The evaluation YAML must match the checkpoint's policy family, base revision,
processors, environment, and fixed scenarios.

See [`examples/embodied/README.md`](examples/embodied/README.md) for candidate
preparation, paired evaluation, generated state manifests, and conformance tools.

## Connect an existing LeRobot workflow

Use your application's LeRobot policy, preprocessor, postprocessor, and
environment with `run_lerobot_experiment`. The helper manages registration,
grouped rollouts, training, logging, checkpoints, and cleanup. For action-token
policies, the adapter records sampled tokens and their rollout-time log probabilities.

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

Use the lower-level API to manage rollout groups and training updates yourself:

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

`EmbodiedTrainableModel` inherits from `art.TrainableModel`. Each policy backend
uses the native sampler and its corresponding training objective.

## Scale without changing the experiment

Slurm is optional. Set devices, actors, inference replicas, worker lifecycle,
and microbatch size in YAML to scale the same experiment on a workstation or cluster.

1. Check one update with one actor and one model replica per rollout device.
2. Increase actors to overlap simulation and inference.
3. Use `batched_server` to share a policy across actors.
4. Add inference replicas when GPU memory and measured throughput allow.
5. Use local distributed training to reduce optimizer time.
6. Use `cpu_offload` when rollout and training share GPUs, allowing enough host RAM for all replicas.

On 80 GB H100s, an OpenVLA-OFT profile with three inference replicas and six actors
per GPU, batch size two, and training microbatch 12 collected 1,024 trajectories at
`1.536 trajectories/s`, over twice the original implementation's throughput.
Tune for your policy and hardware, keeping 10–20% GPU memory headroom.

## W&B and Weave

W&B records the SFT baseline at Step 0 and periodic evaluations on the same
`validation/success_rate` curve. Each completed training update adds one history
row, with videos, evaluation Tables, and versioned model/training-state artifacts.
Weave groups traces by update, rollout group, and trajectory, with links to videos.

| Section | Contents |
| --- | --- |
| `train/*` | `train/success_rate`, `train/reward_mean` |
| `validation/*` | Aggregate evaluation metrics |
| `signal/*` | Rewards, advantages, and useful groups |
| `optimization/*` | Loss, KL, likelihood ratios, and gradients |
| `performance/*` | Timing, throughput, and memory |
| `train_details/*` | Episode counts and durations |
| `media/simulation/*` | Rollout and evaluation videos |

Per-task and per-episode results are stored in Tables and evaluation artifacts.
Install the `observability` extra for W&B support. ART 0.5.18 includes the Weave
client; both integrations send data only when enabled in YAML.

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

With both integrations disabled, training still saves local checkpoints, videos,
and JSON evaluation results.

Checkpoints include hashes of the policy, optimizer, and RNG state. A completion
marker and atomic publication make each saved checkpoint ready for resume.
Resume checks file integrity and compatible training settings before loading.

The example recipes use `delivery_failure_policy: fail_run`: W&B or Weave
upload errors are recorded in the size-limited `telemetry_failures.jsonl` and stop
the run. Use `best_effort` to continue with local records when uploads fail.
Completed updates and checkpoints are retained in both modes.

To resume with an existing Step-0 evaluation, set
`evaluation.baseline_outcomes_path` to its outcome file and disable
`evaluate_before_training`. Preflight requires this reference to preserve paired
comparisons, confidence intervals, and McNemar statistics.

## Boundaries

- Each GRPO update requires complete rollout groups.
- Training diagnostics and native policy evaluations are recorded separately.
- Video and trace limits are configurable in YAML.
- Simulator adapters handle environment-specific behavior.

## Documentation

- [Embodied RL concepts and configuration](docs/experimental/embodied-rl.mdx)
- [Runtime and scaling architecture](docs/experimental/embodied-runtime-architecture.mdx)
- [Release validation contract](docs/experimental/embodied-release-validation.mdx)
- [OpenVLA-OFT and LIBERO examples](examples/embodied/README.md)

## Acknowledgements

We thank the [RLinf](https://github.com/RLinf/RLinf) authors for their public
implementations, recipes, and checkpoints. We used them to check OpenVLA-OFT
rollouts, action-token masking, advantage/loss aggregation, and LIBERO evaluation.

`rlinf_v01` names the corresponding conformance profile. ART-Embodied uses its own
runtime (`model_loader: native`); RLinf is not a dependency.

## Relationship to ART

ART-Embodied is an add-on to OpenPipe ART, with optional robotics dependencies.
Contributions are welcome for reproducible benchmarks, policy and simulator
adapters, and W&B/Weave integration. See [CONTRIBUTING.md](CONTRIBUTING.md).
