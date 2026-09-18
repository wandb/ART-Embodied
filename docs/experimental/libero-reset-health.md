# LIBERO Reset Health

Package/API compatibility is not behavioral equivalence. In LIBERO Spatial
task 5, official initial states depend on physics settling to seat the target
bowl on a ramekin. MuJoCo changes can leave it displaced before the first policy
observation. A successful `doctor` check does not detect this.

The issue affects environment initialization, not only pi0-FAST. Do not assume
every other task is affected, or that previous successful experiments are
invalid. Qualify the actual task, state bank, reset protocol and runtime together.

## What To Reproduce

For published results, keep the recipe's exact simulator dependencies, state
bank and reset settings together. Do not upgrade MuJoCo independently or
downgrade a shared environment: a successful PI0.5 result is not evidence that
another policy is insensitive to the same shift.

For Spatial task 5, an [upstream report](https://github.com/huggingface/lerobot/issues/4390)
locates a seating change at MuJoCo 3.4.0; our isolated 3.3.0 check retained the
intended support relation on all 50 official states. This is reset evidence,
**not** qualification of every LIBERO task or a complete pi0-FAST training stack.
The [dependency-fix PR](https://github.com/huggingface/lerobot/pull/4465) was still
open when checked on 2026-09-09. An API-compatible installation alone is not a
behaviorally qualified installation.

Prefer unmodified official states in one qualified simulator version for a new
reference recipe. Preparing states under 3.3.0 and running them under 3.8.1 is a
separate causal diagnostic, not the original official benchmark. The
`run-pi0-fast-corrected-reset-sft-grpo.sh` research script uses that mixed-version
condition; it is **not a first-user quickstart**.

## Check Before Training

Also run these checks when learning stalls, even if another policy previously
learned the task. Different policies can have very different sensitivity to the
same reset shift. Include robot/gripper state, controller mode and settling
actions in the comparison, not only object coordinates. Compare against actual
teacher observations and the current train/dev reset paths. A successful policy
of another family is not a same-policy reset ablation.

From an ART-Embodied source checkout, use the Python environment that will run
the simulator. No policy, checkpoint, optimizer, or teacher dataset is loaded.
Without `--render`, no GPU rendering is requested, although robosuite still
needs its OpenGL libraries to import.

```bash
python -m examples.embodied.libero.audit_reset \
  --suites libero_spatial libero_object libero_goal libero_10 \
  --states-per-task 3 \
  --output outputs/reset-health-screen
```

`libero_10` is LIBERO Long. Screening selects evenly spaced official bank
indices (0, 24, 49 for a 50-state bank), without using policy outcomes.
For a full initial-state audit of the affected task:

```bash
python -m examples.embodied.libero.audit_reset \
  --suites libero_spatial --task-ids 5 --all-states \
  --render --output outputs/reset-health-spatial-task5
```

`--render` saves one post-reset image per task and needs a working renderer
(for example, NVIDIA/EGL on a GPU allocation). The default reset uses seed 0,
10 neutral control steps, an open gripper, and relative OSC control. Match
`--seed` and `--wait-steps` to the intended experiment; other reset profiles
are not implicitly qualified by this command.

Outputs contain runtime versions, task BDDL and state-bank hashes, sampled state
hashes, initial predicate values, object poses and optional images. Existing
output directories are rejected. The command never rewrites an initial-state
bank or repairs/filter-selects states.

To audit an existing generated manifest instead of the official bank:

```bash
python -m examples.embodied.libero.audit_reset \
  --suites libero_10 \
  --state-manifest examples/embodied/libero/state_manifests/pi05_long_dev_v1/manifest.json \
  --all-states --wait-steps 15 --control-mode unchanged --trace-reset --render \
  --output outputs/reset-health-pi05-dev
```

This checks archive and per-state hashes, suite membership and BDDL hashes before
using each task's states. Manifest order and state IDs are preserved. A manifest
requires exactly one suite; use `--task-ids` if it contains only selected tasks.
Do not point diagnostics at a sealed manifest unless the evaluation protocol
explicitly authorizes access. The tool cannot infer the split from a filename.
The generic report therefore records `sealed_accessed: null`, not a claim that
no sealed population was accessed. A study-specific wrapper must establish that
from the authorized inputs and their provenance.

`--trace-reset` records object poses, maximum absolute generalized velocity and
contact counts after loading the state and at control steps 1, 5, 10 and the final
wait step. This helps distinguish a transient bounce from a persistent placement
change. `--control-mode unchanged` preserves the simulator's controller default;
the actual controller `use_delta` values are recorded. Neither option repairs
states or changes a production recipe.

## Interpret Results

- A violated, reviewed Spatial-task5 support rule returns exit code 2. It is
  bound to the known BDDL SHA-256, rather than assuming task IDs mean the same
  thing in every installation.
- Nonfinite state, an already achieved goal, or a predicate-evaluation error
  is a failure, not a successful reset check.
- Other false initial predicates are reported as **unvalidated warnings**.
  Placement-region conditions are not universally valid after settling, so
  blindly requiring every BDDL initial predicate would generate false alarms.
- `semantic_qualification: not_qualified` means there is no reviewed
  task-specific semantic rule. Exit code 0 means no checked failure was found,
  not that the task, all seeds, cameras, policy or training are certified.
- A three-state screen cannot rule out rare failures. Use `--all-states` and
  appropriate additional seeds after investigating differences.
- `--report-only` records known failures without a nonzero exit status. It is
  for diagnosis, not a way to make a training gate pass. Incomplete execution
  remains a failure.

## Compare Environments

Run the same command in two isolated simulator environments with identical
assets and reset settings. Do not downgrade the shared environment used by
other validated policies just to run this comparison.

To reproduce the task-5 reset comparison without a policy, use the wrapper
below from a source checkout. Replace the two Python paths with **existing**
simulator environments (for example, MuJoCo 3.3.0 and 3.8.1, with all other
dependencies and LIBERO assets matched). Both need the source checkout's audit
dependencies and OpenGL import libraries. This script does not install or modify
either environment, download assets, render videos, or repair initial states.

```bash
bash scripts/reproduce-libero-reset.sh \
  /absolute/path/to/mujoco-3.3.0-env/bin/python \
  /absolute/path/to/mujoco-3.8.1-env/bin/python \
  outputs/reset-version-comparison
```

It audits all 50 official Spatial task-5 states with seed 0, 10 wait steps and
relative control, then compares state hashes, dependencies and reset contracts.
Each arm's `report.json` records the actual runtime and reset failures;
`comparison.json` summarizes differences. It deliberately uses `--report-only`
to retain both sides of a known failure: **exit 0 is not a training approval**.
Incomplete audits or mismatched comparison inputs still fail. For another
recipe, use the individual commands with that recipe's reset settings instead
of assuming these defaults apply (the successful PI0.5 Long recipe uses 15 wait
steps and `--control-mode unchanged`).

```bash
python -m examples.embodied.libero.compare_reset_audits \
  --left outputs/reset-old/report.json \
  --right outputs/reset-current/report.json \
  --output outputs/reset-comparison.json
```

The comparison refuses different state banks, tasks, source states, reset
settings or non-MuJoCo package versions. It flags changed predicate values
and object-position differences greater than 5 mm for review. The 5 mm threshold
is a screening heuristic, not a universal physical validity criterion. Both
versions may be wrong, or may differ harmlessly; a flag alone is not proof of a
regression. Rendering and dynamic behavior during an episode need separate checks.

For audits captured with `--render`, also compare the images:

```bash
python -m examples.embodied.libero.compare_reset_images \
  --left outputs/reset-old/report.json \
  --right outputs/reset-current/report.json \
  --output outputs/reset-image-comparison.json
```

This first validates the same physical audit contract, then reports RGB mean
absolute differences and the fraction of changed pixels. MAE above 5/255 is
flagged for review, not automatically labeled a policy regression. The images
are paired at the same sampled state index. MuJoCo-dependent floor rendering
changes have also been [reported upstream](https://github.com/Lifelong-Robot-Learning/LIBERO/issues/88);
passing a contact/placement check alone cannot rule out visual distribution shift.

## Remediation And Scope

Our 40-task cross-version screen
also found large post-reset position differences in Long tasks 0, 1 and 7,
and floor-rendering differences in all ten Object tasks. Their causes and policy
impact are not yet fully qualified. The single-environment audit does **not**
yet enforce semantic rules for these additional cases; use the paired comparisons
to expose differences rather than treating a zero exit code as certification.

A replay of the successful PI0.5 reset profile
shows why the wait protocol matters: for the affected Long tasks in its dev bank,
the large difference at 10 control steps shrinks substantially by 15 steps.
This does not fix Spatial task 5, whose displaced support relation persists longer.
Neither a reset warning nor its absence alone predicts whether a policy can learn.

An affected experiment should stop before expensive policy training. Confirm
the intended configuration against reference data and inspect reset images.
Either qualify an isolated simulator version or qualify a new, explicitly
versioned pre-settled state bank. Apply the same protocol to training and
evaluation; do not silently fix only validation or rewrite sealed data after
seeing policy outcomes. Preserve the original evidence and distinguish old/new
conditions in W&B.

This is currently an **opt-in source-checkout preflight**, not an automatically
enforced check in every training entry point. Production integration and
cross-policy qualification remain separate work. The command audits official
banks and validated ART-Embodied state manifests, not arbitrary custom simulations.

Background: [LeRobot issue #4390](https://github.com/huggingface/lerobot/issues/4390),
[RLinf issue #1460](https://github.com/RLinf/RLinf/issues/1460).
