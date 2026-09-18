# PI0-FAST: single-task LIBERO Long result

This is a measured learning result, not a claim that every retained experimental
helper is qualified. The task is LIBERO Long task 8, "put both moka pots
on the stove". SFT used all ten Long tasks; GRPO used only this task. No task-ID
adapter routing, replay, instruction perturbations, or SFT anchor was used.

## Results

| Evaluation | SFT | Last GRPO checkpoint (100 updates) |
| --- | ---: | ---: |
| Fixed development states, 100 episodes | 70/100 | 89/100 |
| Separate sealed states, 100 episodes | 73/100 | 83/100 |

- [Development: full training curve, evaluations, videos, and model artifacts](https://wandb.ai/wandb-japan/art-embodied-pi0-fast-long/runs/a1hx8rsf).
- [Sealed SFT baseline](https://wandb.ai/wandb-japan/art-embodied-pi0-fast-long/runs/r56aa76y).
- [Sealed last GRPO checkpoint](https://wandb.ai/wandb-japan/art-embodied-pi0-fast-long/runs/ww9b5upu).

Development reached 76% at update 20, 85% at 30, and 90% at 35. Its best result
was 92%, but the reported final checkpoint is update 100, not the best checkpoint.
The last checkpoint was selected before reading sealed outcomes. The sealed
physical-state fingerprints have no overlap with the development set.

The sealed comparison has 20 improved and 10 regressed pairs: +10 percentage
points, paired 95% bootstrap CI [0, 20] points, exact two-sided McNemar p=0.0987.
This is encouraging independent evidence, not a statistically significant
improvement at the 5% level. Only one training seed was tested. Multitask RL,
language robustness, and broad policy coverage remain separate research work.

## Measured Recipe

| Setting | Value |
| --- | --- |
| Model | `lerobot/pi0fast-libero` at `840f4b503f4c09110421c33c810a85b6684fd658` |
| Action tokenizer revision | `79ae83e3cbd8786dcb84b628569f8d076ca8151e` |
| Warm start | Fresh 400-update multitask SFT, shared rank-32/alpha-32 LoRA |
| GRPO | 100 updates, 120 groups x 8 trajectories = 960/update |
| Optimizer | AdamW, LR 2e-6, epsilon 1e-5, max gradient norm 1 |
| Optimizer schedule | 120-trajectory minibatches, eight optimizer steps/update |
| Likelihood | All generated action tokens; full-sequence teacher forcing |
| Generation | Native decoder, sampling temperature 0.2; greedy evaluation |
| Executed action horizon | 10 |
| Precision | `fp16_residual`, static training loss scale 128 |
| Simulator | Isolated MuJoCo 3.3.0, unchanged controller, 15 neutral settle steps |
| Development evaluation | 100 fixed episodes at updates 0, 5, ..., 100 |
| Hardware | One node, eight H100 GPUs; 32 rollout actors, eight training workers |
| W&B | One native history row per update; `_step == experiment/update` |

`fp16_residual` is not an all-FP16 cast. It preserves sensitive residual/MLP
output operations and other designated modules in FP32, with scaled matrix
products and correct gradient unscaling. Rollout KV-cache and autograd scoring
were compared on identical inputs. Do not infer that BF16 is generally unsuitable
for RL, or attribute the measured lift to any one repair without an ablation.

## Reproduction Boundary

The package's existing public LIBERO profile still pins MuJoCo 3.8.1. **Installing
`pi0-fast-libero` alone does not reproduce this 3.3.0 experiment.** Keep the
isolated runtime, state manifests, reset protocol, precision, and checkpoint
provenance together. Do not replace the simulator in an existing qualified
OpenVLA-OFT, PI0/PI0.5, or GR00T environment.

The run's W&B configuration records the resolved experiment configuration;
the model and evaluation artifacts retain checkpoint and outcome provenance.
The following entrypoints are research tools, not a one-command public recipe:

- `examples/embodied/pi0_fast_long_restart.py`: exact isolated runtime validation.
- `examples/embodied/pi0_fast_long_task8_run.py`: single-task experiment planning.
- `examples/embodied/pi0_fast_native_precision_restart.py`: fresh SFT planning
  and measured actor-capacity checks. Its checkpoint-precision defaults are
  **not** the final `fp16_residual` recipe without explicit overrides.
- `scripts/slurm/run-pi0-fast-native-step-grpo.sh`: one-node launch wrapper;
  requires a materialized and qualified recipe, rather than inventing defaults.
- `examples/embodied/pi0_fast_long_sealed_eval.py`: separately selected baseline
  and candidate evaluation, exact runtime/checkpoint checks and native media axes.

Some historical helpers/configurations remain because the SFT/GRPO entrypoints
or regression tests depend on them;
they must not be presented as validated defaults. Local checkpoint
paths and generated simulator state banks are not distributed in this PR.
Packaging a portable, independently rerun reproduction bundle is still pending.
See the [reset-health guide](libero-reset-health.md) before experimenting with a
different simulator runtime or state bank.

## Observability and Performance

Read-back audits checked all 101 development history rows, 368 development
videos, 100 model artifacts, and all 23 development/sealed evaluation tables
(2,300 episode rows). Each sealed run has eight videos at its actual checkpoint
step, without dummy intermediate rows. App rendering was not automatically
inspected; storage/media verification is not a claim of rendering verification.

The final update took 16.44 minutes for rollout and 9.01 minutes for training,
excluding evaluation/logging. The last five save-completion intervals averaged
27.34 minutes. The requested 20-minute target has not been met.
