# ART-Embodied OpenVLA-OFT positive control

This directory contains the strict OpenVLA-OFT action-token GRPO contract used
to validate the embodied API. It is intentionally compute-heavy and should not
be presented as the eventual LeRobot-first default or as a requirement for
other policy families.

Install `art-embodied[lerobot]` for the simulator-neutral LeRobot path. This
LIBERO positive control additionally requires
`art-embodied[libero]`; normal users should not install the LIBERO
extra unless their environment needs it.
The extra pins the validated `hf-libero==0.1.4` product runtime; no RLinf
source checkout is required for normal execution. It also pins the validated
OpenVLA-OFT ML stack instead of composing LeRobot 0.4.4's newer
`libero,peft` extras, whose Transformers and PEFT constraints are behaviorally
incompatible with this checkpoint family.
Run `art-embodied doctor --profile libero` in that environment before allocating
a GPU; the command checks every behavior-pinned distribution without importing
or loading the model.
The entry point validates all task BDDL and initial-state files before loading
the policy. A valid explicit `LIBERO_CONFIG_PATH` is preserved; otherwise the
paths are derived from the installed `hf-libero` wheel into a process-local
config. Machine-specific cache paths are neither required nor silently trusted.

File and package validation does not guarantee equivalent reset physics.
Before training, also run the opt-in, policy-free
[LIBERO reset audit](../../docs/experimental/libero-reset-health.md).
In particular, Spatial task 5 can start with a displaced bowl under newer
MuJoCo versions. The audit does not silently change dependencies or state banks.

`lerobot_action_token_grpo.template.yaml` is the non-RLinf starting point for a
registered LeRobot action-token policy. It is complete by design: copy it,
change visible experiment conditions, and register the policy factory in the
application. Do not inherit the positive-control actor geometry unless the
experiment intentionally reproduces that prior work.

`openvla_oft_libero_spatial_grpo_rlinf_positive_control.yaml` is the second
suite control. It preserves the Object control's algorithm, LoRA, rollout,
training, and runtime geometry while changing only the pinned SFT model and
LIBERO suite contract. See
`docs/experimental/embodied-release-validation.mdx` before making a learning or
public-demo claim.

The experimental PI Flow-SDE controls are isolated in the `pi-libero` runtime:

- `pi05_libero_object_flow_sde_grpo_rlinf_positive_control.yaml`
- `pi05_libero_spatial_flow_sde_grpo_rlinf_positive_control.yaml`
- `pi05_libero_long_flow_sde_grpo_rlinf_positive_control.yaml`
- `pi0_libero_object_flow_sde_grpo_rlinf_positive_control.yaml`
- `pi0_libero_spatial_flow_sde_grpo_rlinf_positive_control.yaml`
- `pi0_libero_long_flow_sde_grpo_rlinf_positive_control.yaml`

Larger-budget variants are available as
`pi0_libero_spatial_flow_sde_grpo_1024.yaml` and
`pi05_libero_long_flow_sde_grpo_1024.yaml`. They double the number of
independent same-reset groups from 64 to 128 (1024 trajectories at group size
8), while halving the number of passes over each rollout batch. The optimizer
therefore performs the same number of subupdates as in the 512-trajectory
profiles. This isolates lower-variance rollout sampling from a larger policy
update; it is not a license to double both trajectories and optimizer work.

The executed RLinf-reference positive controls are frozen separately as
`pi0_libero_spatial_flow_sde_grpo_rlinf_reference_1024.yaml` and
`pi05_libero_long_flow_sde_grpo_rlinf_reference_1024.yaml`. Both use four
Flow-SDE transitions, noise `0.5`, group size 8, and 1,024 trajectories. Their
hash-verified generated-state development manifests live under
`libero/state_manifests/`. PI0 Spatial improved `63/100` to `75/100` at update
5 and `87/100` at update 35; PI0.5 Long improved `43/100` to `54/100` at update
5. Both reference recipes continue to 100 updates because these early gates do
not establish convergence. These are development
results, not sealed-test or multi-seed claims. The K8/K16 recipes above are
more conservative behavioral-calibration profiles and must not be presented as
the executed K4 reference condition.

These are complete research recipes, not aliases for the OpenVLA-OFT control.
They execute the native PI ODE sampler for evaluation and a retained stochastic
Flow-SDE transition for training/rescoring. PI0 and PI0.5 deliberately keep
different checkpoint, processor, action-coordinate, noise-level, and
update-epoch contracts. PI0 denoises its full 50-action OpenPI horizon and
executes only the suite-specific leading 5 or 10 actions; PI0.5 denoises a
10-action horizon. These model and execution horizons are not interchangeable.
In particular, RLinf's older public LIBERO PI0
checkpoints require `extra_delta_transform: true`: the first six action
dimensions are relative to the current state while the gripper stays absolute.
PI0.5 uses `false`. This is a checkpoint compatibility contract, not a default
that should be guessed for arbitrary PI checkpoints. Run these recipes only
after `art-embodied doctor --profile pi`; a passing config or sampler
conformance test is not evidence of success-rate lift.

```bash
art-embodied validate \
  examples/embodied/pi05_libero_long_flow_sde_grpo_rlinf_positive_control.yaml

python -m examples.embodied.libero.train_openvla_oft \
  --config examples/embodied/pi05_libero_long_flow_sde_grpo_rlinf_positive_control.yaml
```

## SmolVLA Flow-SDE (experimental)

The SmolVLA path loads the serialized LeRobot 0.6 policy and processor
pipelines directly. It denoises the checkpoint's full 50-action model chunk,
executes the configured leading prefix, and retains one stochastic transition
for exact teacher-forced rescore. SmolVLA uses its native Python-timestep Euler
order for deterministic transitions; the PI/RLinf interpolation contract is
not silently reused across model families.

The real `HuggingFaceVLA/smolvla_libero` checkpoint has passed native ODE,
rollout/rescore, LoRA-gradient, and one-update LIBERO lifecycle conformance on
one H100. Frozen image/language prefix embeddings are retained instead of the
resized float camera tensors; the trainable state projection is recomputed, so
exact rescore is preserved while replay handoff falls from roughly 6.3 MB to
0.56 MB per policy step on the public checkpoint. Long-horizon recipes can
additionally set `trainable_action_selection: uniform_grid` and an explicit
`max_trainable_actions_per_trajectory`. This keeps every lightweight action and
video frame for evidence while retaining exact Flow-SDE replay tensors at a
deterministic, endpoint-inclusive subset of policy decisions. The resulting
objective is a systematic approximation to the all-action objective, so the
eligible count, selected count, and selection fraction are logged explicitly.
Strict RLinf fixed-row geometry rejects this option rather than silently
changing its denominator.

This is not yet a success-rate-lift claim. Run the conformance gate before any
long experiment:

```bash
uv sync --extra smolvla-libero --extra dev
art-embodied doctor --profile smolvla

python examples/embodied/smolvla_flow_sde_conformance.py \
  examples/embodied/smolvla_libero_object_flow_sde_grpo_smoke_h100.yaml

# Compare native ODE and exploratory Flow-SDE on identical states and seeds.
python examples/embodied/flow_sde_sampler_calibration.py \
  --config examples/embodied/smolvla_libero_object_flow_sde_grpo_smoke_h100.yaml \
  --output-dir outputs/smolvla-sampler-calibration \
  --episodes 20 \
  --max-absolute-success-rate-gap 0.10

python -m examples.embodied.libero.train_openvla_oft \
  --config examples/embodied/smolvla_libero_object_flow_sde_grpo_smoke_h100.yaml
```

After that gate passes, the one-node positive-control recipe uses 64
trajectories per update and one fixed 100-episode evaluation curve:

```bash
art-embodied validate \
  examples/embodied/smolvla_libero_object_flow_sde_grpo_positive_control_h100.yaml

python -m examples.embodied.libero.train_openvla_oft \
  --config examples/embodied/smolvla_libero_object_flow_sde_grpo_positive_control_h100.yaml
```

## GR00T N1.5 Flow-SDE (experimental)

GR00T N1.5 uses an isolated runtime because NVIDIA's archived N1.5 release pins
an older Torch, Transformers, and protobuf stack than the LeRobot 0.6 policy
families. Install it without modifying another policy environment:

```bash
./scripts/install-gr00t-n1d5-runtime.sh
```

The adapter preserves the official processor, `libero_franka` embodiment,
normalization, 16-action model horizon, 5-action execution prefix, and native
ODE evaluator. Training follows RLinf's N1.5 Flow-SDE transition with four
denoising steps and noise level 0.5, but replaces PPO/GAE with ART's same-reset
group-relative GRPO advantage. It is therefore an RLinf-derived GRPO control,
not a PPO reproduction. Only actions actually executed in the environment
receive trajectory credit.

Run real-weight conformance before allocating a long job:

```bash
.venv-gr00t-n1d5/bin/python -P \
  examples/embodied/gr00t_n1d5_conformance.py \
  examples/embodied/gr00t_n1d5_libero_spatial_flow_sde_grpo_positive_control.yaml

.venv-gr00t-n1d5/bin/python -P -m \
  examples.embodied.libero.train_openvla_oft \
  --config examples/embodied/gr00t_n1d5_libero_spatial_flow_sde_grpo_positive_control.yaml
```

The positive-control recipe uses one node, eight rollout/training GPUs, 512
trajectories per update, and a fixed 100-episode diagnostic evaluation. The
baseline and every later evaluation use one `validation/success_rate` series
in the same W&B run. This N1.5 path remains experimental until it demonstrates
a clear native-evaluation lift. GR00T N1.7 and RoboCasa use a separate adapter
and evidence boundary; neither is implied by passing the N1.5 LIBERO gate.

## GR00T N1.7 + RoboCasa Flow-SDE

The validated GR00T N1.7 path uses two isolated Python 3.12 environments: a
pinned NVIDIA Isaac-GR00T policy runtime and a pinned RoboCasa/robosuite/MuJoCo
simulator runtime. Install and verify them before allocating a training job:

```bash
./scripts/install-gr00t-n1d7-runtime.sh
./scripts/install-robocasa-gr1-runtime.sh
./scripts/download-robocasa-gr1-dataset.sh

sbatch scripts/slurm/run-gr00t-n1d7-robocasa-policy-conformance.sh
sbatch scripts/slurm/run-robocasa-gr1-conformance.sh
```

The opened learning path is deliberately narrow: GR1
`PnPCounterToCab` Cuttingboard-to-Pan, initialized from a pinned 60k-step SFT
checkpoint produced with NVIDIA's training recipe. The rank-64/alpha-64 adapter
used learning rate `3e-5`, groups of
eight trajectories, four denoising steps, Flow-SDE noise `0.1`, and 100
uninterrupted updates. The
exact development and continuation contracts are:

```text
examples/embodied/gr00t_n1d7_robocasa_gr1_tabletop_cuttingboard_pan_noise01_u20_development.yaml
examples/embodied/robocasa/development_experiments/gr1_tabletop_cuttingboard_pan_noise01_u100_continuation_v1.yaml
```

The update-100 candidate improved the fixed 64-episode development set from
29/64 to 47/64. An independent read-only audit verified all 100 rollout groups,
27 evaluations, checkpoint lineage, and outcome hashes before the sealed test.
The candidate and SFT baseline were then each evaluated once on the same 192
previously unused simulator seeds from:

```text
examples/embodied/robocasa/sealed_tests/gr1_tabletop_cuttingboard_pan_u100_v1.json
```

The SFT baseline scored 111/192 and update 100 scored 139/192, a paired lift of
`+14.6` points with 95% CI `[+5.2,+24.0]` and exact McNemar `p=0.00335`. Public
W&B runs:

- [SFT sealed baseline](https://wandb.ai/wandb-japan/art-embodied-gr00t-n1d7-robocasa-single-task-sealed/runs/j11avofl)
- [Update-100 sealed candidate](https://wandb.ai/wandb-japan/art-embodied-gr00t-n1d7-robocasa-single-task-sealed/runs/kjpzsj11)
- [100-update development run](https://wandb.ai/wandb-japan/art-embodied-gr00t-n1d7-robocasa-single-task/runs/1970sjop)

This validates policy loading, official preprocessing and embodiment handling,
native ODE evaluation, Flow-SDE rollout collection, trajectory-level GRPO,
LoRA checkpointing/resume, RoboCasa simulation, paired evaluation, and sealed
adjudication for one task. It does not establish eight-task or task-held-out
performance. Those are separate gated expansion problems, not part of this
result.

## Validate the contract

```bash
art-embodied validate \
  examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml
```

The validator checks more than YAML syntax. In particular, it proves that the
declared fixed-horizon rollout rows equal the optimizer rows actually consumed
per update and prints the total rollout cost before any model or simulator is
loaded. Use `--json` in launch automation.

## Run the complete LIBERO control

After validation, run the same visible YAML through the public ART entry point:

```bash
python examples/embodied/libero/train_openvla_oft.py \
  --config examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml
```

The profile declares four rollout and training GPUs while retaining RLinf's
logical eight-actor reset topology, and intentionally produces 1,024
trajectories per update. It requires the `libero` extra.
W&B and Weave start online, training and evaluation video are required, and
checkpoint/log retention follows the limits in this YAML. To use fewer GPUs or
a different simulator, create another complete YAML rather than applying
hidden environment-variable defaults.

`openvla_oft_libero_object_grpo_lora_lr2e4.yaml` is a complete experimental
high-update profile. It differs from the Object positive control only in run
identity, output path, and learning rate (`2e-4` instead of `5e-5`). Keep the
control YAML unchanged so the comparison remains reviewable.

`openvla_oft_libero_object_grpo_lora_lr2e4_shared_rollout.yaml` keeps that
algorithm, optimizer, and evaluation contract while changing only rollout
resource placement. Two simulator actors per rollout device share one OpenVLA
inference server. On the declared four devices, validation reports eight actors,
four rollout model replicas, and 64 concurrent environment slots. Treat this as
a throughput profile: benchmark it against the embedded control before using it
for a claim-facing run. ART logs rollout elapsed time, groups/second,
trajectories/second, and training elapsed time so the two profiles can be
compared directly in W&B without parsing scheduler logs.

The H100 profiles are explicit resource-tuned variants, not portable defaults:

- `openvla_oft_libero_object_grpo_lora_lr2e4_h100.yaml`
- `openvla_oft_libero_spatial_grpo_lora_lr2e4_h100.yaml`
- `openvla_oft_libero_spatial_sft_baseline_eval_h100.yaml`

They declare eight devices, three inference replicas and six simulator actors
per device, bounded media retention, and online W&B/Weave output. The Spatial
pair changes the pinned SFT model, suite, scenarios, and run identity while
preserving the validated Object algorithm and optimizer geometry. Run the SFT
evaluation first and point the training profile's
`evaluation.baseline_outcomes_path` at that immutable episode report. A
completed baseline is not evidence that Spatial GRPO improves the policy; the
fixed post-update evaluations are the acceptance criterion.

For a resource-constrained workstation, use
`pi05_libero_object_flow_sde_grpo_single_gpu.yaml`. It declares `cuda:0` for
both rollout and training, runs initial fixed evaluation before the first
rollout, and reuses that device serially for rollout, optimization, and periodic
evaluation. It deliberately uses `lifecycle: per_update`: this costs worker
startup time but does not require a second GPU or retain an idle model replica.
Run `art-embodied validate ... --json` and confirm
`serial_phase_device_reuse: true` before allocating the accelerator.

## Evaluate a baseline or checkpoint without training

Use the same public entry point with `--evaluate-only`:

```bash
python examples/embodied/libero/train_openvla_oft.py \
  --config examples/embodied/openvla_oft_libero_object_sft_baseline_eval.yaml \
  --evaluate-only --evaluation-step 0
```

To evaluate a saved adapter without restoring optimizer state, add
`--policy-checkpoint PATH_TO_CHECKPOINT/policy`. The evaluation YAML remains the
single source of truth for policy family, processor, environment, scenarios,
seeds, observability, and evidence output.

The command never constructs a training backend. It uses the fixed scenarios,
seeds, sampler, checkpoint, media limits, W&B project, and Weave project from
the YAML. Local-process actors switch explicitly to the YAML's evaluation
sampler rather than inheriting training generation settings. The bounded
episode report is written under `storage.output_dir/evaluation/`.
OpenVLA-OFT examples also declare `runtime_contract: openvla_oft_v01`. This
requires Torch 2.6.0, Transformers 4.40.1, tokenizers 0.19.1, TIMM 0.9.10, and
PEFT 0.11.1. Loading under a newer LeRobot stack is not evidence of behavioral
compatibility: it can change action logits. Use `unchecked` only for a named
diagnostic, never for release evaluation or training.
`robot_platform` must be declared separately so custom checkpoint paths do not
silently inherit LIBERO action constants.
The public GRPO positive-control profile references that exact path, so the
claim-facing baseline and candidate conditions remain reviewable before either
job starts. Both training and claim-facing evaluation preserve RLinf v0.1's
unspecified attention selection. Explicit `eager` and `flash_attention_2`
overrides both change logits for the bidirectional action-placeholder sequence
and are rejected by the validated runtime contract.

### Freeze additional evaluation initial states

Official LIBERO reset banks are finite and can become train-matched when every
state is sampled during RL. Generate a model-independent state manifest before
starting a claim-facing long run, not after selecting a promising checkpoint:

```bash
python -m examples.embodied.libero.generate_state_manifest \
  --config examples/embodied/openvla_oft_libero_object_sft_baseline_eval.yaml \
  --output-dir outputs/evaluation-states/libero-object-v1 \
  --states-per-task 10 \
  --base-seed 20260718
```

Then set `environment.kwargs.evaluation_state_manifest` in both baseline and
training/candidate YAML files to the generated `manifest.json`, set
`evaluation.split: held_out` and `evaluation.data_role: development`, and point
the training YAML at the immutable SFT
`evaluation.baseline_outcomes_path`. This makes the periodic evaluation the
claim-facing held-out series from update 0 onward. Keep train-matched evaluation
only as a separately named diagnostic; rollout success already provides the
normal train series. The archive is hash
verified, rejects exact official-state duplicates and initially solved states,
and records the simulator runtime and generation seed. Generation loads no
policy and does not select states using model outcomes. It measures
initialization-state generalization under the same task definitions; a new-task
claim additionally requires new BDDL tasks and instructions.

Once recipe, checkpoint-selection rule, and all hyperparameters are frozen,
generate a separate untouched manifest and mark both baseline and candidate
configs as `evaluation.data_role: sealed_test`. A sealed config must use
`checkpoint_selection: last`; the schema rejects evaluation-driven selection
against the final test. Evaluation writes raw outcomes and an evidence index
locally, and `observability.wandb.log_evaluation_artifacts: true` publishes both
as a W&B evaluation artifact with stable `latest`/`update-N` aliases.

Also run `openvla_oft_libero_object_public_grpo_eval.yaml` before interpreting
a clean-path training result. It pins the known-good public RLinf GRPO model
while preserving the baseline's fixed scenarios, sampler, and simulator
contract. If that model does not substantially beat SFT, debug the evaluator
instead of launching another training run.

```bash
art-embodied validate-evaluation-pair \
  examples/embodied/openvla_oft_libero_object_sft_baseline_eval.yaml \
  examples/embodied/openvla_oft_libero_object_public_grpo_eval.yaml
```

This command treats the model/checkpoint as the experimental treatment and
fails before GPU allocation if the native loader/runtime, attention backend,
environment/processors, horizon, fixed scenarios, seeds, sampler, or expected
baseline artifact differs. Evaluate an ART LoRA through an equivalent
claim-facing profile that preserves this runtime contract and changes only the
adapter/checkpoint and run/output identity.

Generate that ART LoRA profile without hand-editing the evaluation controls:

```bash
art-embodied prepare-evaluation-candidate \
  examples/embodied/openvla_oft_libero_object_sft_baseline_eval.yaml \
  outputs/my-run/held-out-eval.yaml \
  --run my-run-held-out \
  --output-dir outputs/my-run/held-out \
  --adapter-path outputs/my-run/checkpoints/accepted/peft_adapter
```

The Spatial second positive control has the same paired profiles:

- `openvla_oft_libero_spatial_sft_baseline_eval.yaml`
- `openvla_oft_libero_spatial_grpo_rlinf_positive_control.yaml`

## Run the real-policy smoke

The one-GPU smoke loads the SFT OpenVLA-OFT checkpoint, attaches the declared
LoRA surface, samples action tokens, recomputes their likelihoods, performs one
GRPO optimizer step, and writes a checkpoint:

```bash
python examples/embodied/openvla_oft_action_token_smoke.py
```

Its synthetic preference only proves implementation wiring. It is not a
robotics benchmark and must not be cited as task-performance evidence. The
complete conditions live in `openvla_oft_action_token_smoke.yaml`; the saved
adapter is reloaded through the public policy contract, then successful
temporary checkpoints are removed unless `--keep-checkpoint` is passed.

GSPO profiles fail closed unless old action-token logprobs are recomputed with
the training scorer contract. Keep `precalculate_logprobs: true` and
`rollout_logprob_source: recomputed_current_policy`; the local multi-GPU
backend performs this pre-update rescore across its persistent training workers
before the first trajectory minibatch changes the policy.

## Integrate an existing LeRobot rollout

For a Gymnasium-compatible LeRobot environment, use the LeRobot-first runner:

```python
import art_embodied

policy_adapter = art_embodied.LeRobotPolicyAdapter(
    policy=policy,
    preprocessor=preprocessor,
    postprocessor=postprocessor,
    device=config.policy.device,
    record_action_fn=record_action_tokens,
)

result = await art_embodied.run_lerobot_experiment(
    config=config,
    policy=policy,
    train_scenarios=train_scenarios,
    evaluation_scenarios=held_out_scenarios,
    environment_factory=make_environment,
    policy_adapter=policy_adapter,
)
```

The local default binds the adapter to the exact policy trained by ART and uses
`rollout.workers: 1`. ART calls LeRobot's own preprocessor, `predict_action`, and
postprocessor; the native action goes directly to `env.step`. The runner records
the trajectory, explicit success field, reward events, compact observation
summaries, and a bounded rollout video. Policy, preprocessor, and postprocessor
state are reset and policy seeds are applied for every episode. Parallel
factories must synchronize episode-isolated replicas through
`prepare_update(policy=..., update=...)`; stale or shared state fails closed.
A custom rollout remains available for non-Gymnasium environments.

For concurrent simulation, switch the YAML execution contract to
`runtime.rollout_execution.mode: local_process`. Point `actor_factory` at the
built-in `create_lerobot_process_actor`, then provide one application-owned
`components_factory` under `actor_kwargs`. It constructs the native environment,
policy adapter, processors, and checkpoint loader inside each child process;
users do not rewrite grouping, episode control, success/reward capture, videos,
or evaluation. Embedded mode creates one policy replica per actor.

`batched_server` mode creates one policy inference engine per rollout GPU and
allows application-owned group rollouts to share it across several environment
actors. The bundled LIBERO/OpenVLA-OFT group path uses this contract without
moving simulator-specific code into the generic runtime. Stochastic groups keep
independent continuous RNG streams; model calls remain serialized while actor
processes overlap simulator work.

On a high-memory GPU where one engine leaves compute headroom, set
`inference_replicas_per_device` above one to distribute actors across independent
model processes on that GPU. Keep it at one unless a fixed rollout benchmark
shows a material throughput gain and safe peak memory.
The explicit OpenVLA-OFT H100 profile is
[`openvla_oft_libero_object_grpo_lora_lr2e4_h100.yaml`](openvla_oft_libero_object_grpo_lora_lr2e4_h100.yaml).
It uses three inference replicas and six actors per H100; it is not a portable
default. On the validated H100 80 GB workload, the third replica improved
fixed-workload rollout throughput by 17.6% over two replicas while retaining
about 18 GB of HBM headroom. Increasing the request batch limit from two to
four did not materially improve the three-replica result.
Distributed training uses an action-logprob microbatch of 12. Microbatch 16
exhausted the 80 GB device during recomputation, while 12 completed two
consecutive rollout/train updates and reduced the measured four-subupdate
training window by roughly 31% relative to microbatch 8.
Choose `persistent` with separate rollout/training GPUs, `cpu_offload` to retain
batched inference processes while time-sharing GPUs, or `per_update` when host
memory is constrained. See the
[runtime architecture](../../docs/experimental/embodied-runtime-architecture.mdx)
for the complete factory contract and YAML example.
The YAML also declares whether one failed episode invalidates the update and
the minimum completed attempts needed for a trainable group; partial groups are
never accepted implicitly.
`runtime.rollout_execution.group_batching` is a separate, explicit switch.
Leave it `false` for normal episode actors and dynamic per-request inference.
Set it `true` only when the actor implements a group-native path. The LIBERO
positive control batches the eight OpenVLA forwards belonging to one shared
reset while retaining separate environments, action records, rewards, masks,
success fields, and videos. ART never silently changes this rollout geometry.
It separately declares the low-level `max_episode_steps` and policy-decision
`max_policy_steps`. These are equal for single-action policies and differ for
policies that emit chunks; strict conformance profiles reject inconsistent
horizons before allocating a model.
Video limits apply before rendering and file creation, not only when selecting
media for W&B or Weave.

Policy-specific action-token adapters attach generated tokens and rollout-time
log probabilities through `LeRobotPolicyAdapter.record_action_fn`. Continuous,
diffusion, and flow policies require their own native objective adapters rather
than being mislabeled as token policies.

Action-level objectives read `primitive_rewards` and an equal-length
`primitive_loss_mask` from each action's metadata. These names are intentionally
model- and simulator-neutral; application processors own their construction.

The helper constructs distinct train/evaluation episode runners, preserving
the correct video limits, output paths, and trajectory phase metadata. For a
non-Gymnasium environment, use the generic high-level entry point instead:

```python
result = await art_embodied.run_embodied_experiment(
    config=config,
    policy=policy,
    train_scenarios=train_scenarios,
    rollout=rollout,
    evaluation_scenarios=held_out_scenarios,
)
```

For the validated OpenVLA-OFT path, construct the trainable policy directly
from the same YAML:

```python
policy = art_embodied.make_policy(config)
```

The YAML selects the registered `openvla_oft` factory. This applies the
declared rollout sampler and LoRA target surface. It fails
closed if the trainable surface cannot be attached, rather than starting a run
with a frozen or differently configured policy.

This does not replace LeRobot's environment or processor pipeline. The rollout
continues to call the native policy and environment; ART only owns grouping,
the trajectory-level update schedule, fixed evaluation, and observability.

The built-in OpenVLA-OFT action-token backend supports one-device training and
local multi-GPU gradient workers. For the latter, set
`runtime.distributed_training: true`, list explicit `cuda:N` devices, and place
ephemeral handoffs on node-local storage such as `/tmp/art-embodied`. The
coordinator prepares the complete update and owns the optimizer; workers load
the base VLA once per update, refresh LoRA state between subupdates, and return
gradients only. Set `runtime.training_worker_lifecycle: cpu_offload` to retain
workers and CPU-resident base models between updates while time-sharing their
GPUs with rollout. Keep `per_update` when host memory is constrained. The
two-GPU contract can be checked with:

Long action-token jobs can expose bounded local progress without hidden
environment variables. Set `training.log_action_token_progress: true` and
`training.action_token_progress_every_microbatches` in YAML. Records are
written below `storage.output_dir/diagnostics/` and stop at the explicit
`storage.max_log_file_mb` limit.

```bash
python examples/embodied/openvla_oft_action_token_smoke.py \
  --config examples/embodied/openvla_oft_action_token_distributed_smoke.yaml
```

Measure native OpenVLA-OFT rollout batching on one GPU with a retained LoRA
adapter:

```bash
python examples/embodied/openvla_oft_rollout_batch_benchmark.py \
  --config examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml \
  --adapter-path /path/to/model_adapter \
  --batch-sizes 1,2,4,8
```

The benchmark compares sequential `act` calls with the exact native
`act_batch` policy path and reports actions/second, speedup, realized native
batch metadata, and peak GPU memory. It is a runtime benchmark, not a task
success evaluation.

Applications can still
inject another distributed backend through
`run_embodied_experiment(..., backend=...)` without changing the trajectory
contract.

In W&B, each run exposes live rollout completion and provisional success
numerator/denominator through Summary, plus update-level reward and optimizer
history, bounded train/eval videos, held-out episode Tables, and metadata-rich
checkpoint Models artifacts. The mutable Summary intentionally keeps only
operational health signals such as throughput, optimizer completion, KL,
clipping, and parameter movement. Full diagnostics remain in the committed
  `train/*` history. Per-task and per-scenario evaluation detail lives in a
  bounded Table and the immutable evaluation Artifact; neither surface
  overwhelms the live monitor.
When both outcomes are available, the bounded video sample includes a success
and a failure before filling remaining slots in rollout order; users do not
have to store every episode to compare behavior at the same update.
Videos use stable `media/simulation/train/{slot}` and
`media/simulation/eval/{slot}` keys at every update, so W&B reuses a bounded
set of panels instead of creating one panel per update, task, or outcome.
Task and outcome remain visible in each video's caption and in the episode
Table.
Provisional progress does not create W&B history rows. For a paired run, the
SFT report is the first `validation/success_rate` point at native Step 0. Each
completed optimizer update then creates one row at the same numeric default W&B
Step; no custom x-axis is required. In Weave, one update is a trace root opened before rollout with group and
trajectory children; each trajectory includes a compact step timeline and
file-backed video content. Bounded traces prioritize a group containing both
outcomes and preserve a success and failure within each selected group. The
root links back to its W&B run and records mid-update failures explicitly.
The YAML also owns Weave's optional response cache. Release profiles set
`use_server_cache: false`; enabling it requires an explicit writable
`server_cache_dir` and bounded `server_cache_size_mb`, so shared home storage
cannot grow through an invisible SDK default.

For clean release acceptance, set `require_train_video: true` and, when fixed
evaluation is enabled, `require_evaluation_video: true`. These checks fail
closed: metrics-only logging cannot hide a broken `env.render()` path or an
empty W&B video payload. Keep them false when an observability outage must not
abort a production training run.

W&B and Weave are optional. A non-W&B user can set `wandb.enabled: false`,
`wandb.mode: disabled`, `weave.enabled: false`, and both `require_*_video`
fields to `false`. The same runner still writes local checkpoints, bounded
media, and machine-readable evaluation outcomes. For long runs, declare
`storage.retain_checkpoint_updates` and keep `storage.save_training_state: true`
so selected milestones include optimizer and RNG state; enabled W&B Models
logging backs up those milestone directories as versioned artifacts. Set
`storage.resume_from_checkpoint` to one milestone directory to restore policy,
optimizer, RNG streams, and the update cursor. `training.updates` remains the
final target update rather than an additional-update count.

Milestones are published transactionally. The coordinator writes policy and
training state to a same-filesystem staging directory, records every file size
and SHA-256 digest in `art_embodied_checkpoint_complete.json`, then atomically
renames the complete directory. Resume validates those digests and a resume
contract fingerprint before loading policy or optimizer state. Operational
changes such as a larger final update target, a new output directory, or a new
GPU allocation are allowed; policy, reward, rollout, objective, schedule, and
optimizer changes are rejected. A trusted checkpoint produced before this
contract can be opened only with the explicit
`storage.allow_legacy_checkpoint_resume: true` escape hatch.

Runtime W&B or Weave delivery errors do not roll back a completed optimizer
update or its local checkpoint. Public recipes use
`observability.delivery_failure_policy: fail_run`: each failure is written to
bounded `storage.output_dir/telemetry_failures.jsonl` and then aborts the
claim-facing run before more optimization occurs. Set `best_effort` explicitly
only when continuing from durable local evidence is more important than a
complete external experiment record. Video requirements and other declared
observability contract violations also fail closed.

See the [embodied RL guide](../../docs/experimental/embodied-rl.mdx) for the
complete structure and current support boundary, and the
[release validation contract](../../docs/experimental/embodied-release-validation.mdx)
for multi-suite success-rate and media requirements.

## Evidence boundary

The positive-control research established that `reset_gripper_open` must be
`false`. Under that contract, the retained rank-32/alpha-32/LR-1e-4 run reached
100/100 at update 200 on 100 simulator-generated initial states absent from its
500-state training stream. SFT scored 34/100 and the public RLinf checkpoint
also scored 100/100 on the same states. The ART checkpoint improved 66 paired
episodes without a regression. This validates state-held-out transfer within
the same tasks and instructions, not held-out tasks or another policy family.
Repeated use makes this a development manifest rather than a sealed final test.
The positive-control YAML records the demonstrated ART LoRA geometry.
