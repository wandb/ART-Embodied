# Contributing to ART-Embodied

ART-Embodied is a LeRobot-first add-on for OpenPipe ART. Contributions should
preserve that boundary: ART owns experiment/model lifecycle, LeRobot owns native
policies and environments, and each policy family owns its sampler-aligned RL
objective.

## Development setup

ART-Embodied supports Python 3.11 and 3.12 and uses `uv`:

```bash
git clone https://github.com/wandb/ART-Embodied.git
cd ART-Embodied
git config core.hooksPath .githooks
uv sync --extra dev
```

The package imports as `art_embodied`. Do not add modules to the upstream
`art` namespace or vendor OpenPipe ART into this distribution.

## Before opening a pull request

Keep internal notes and operational records outside Git, in the owner-only
directory specified by `ART_EMBODIED_PRIVATE_DIR`. This directory must be outside
every Git checkout. Do not add these records to research branches.
Local hooks and CI reject private paths and marked internal content, including
content added in an earlier commit and subsequently removed. Review unmarked
documents manually; automated checks cannot recognize all internal information.

Run the add-on-owned CPU checks:

```bash
uv run --extra dev ruff check \
  src/art_embodied \
  tests/test_embodied_*.py \
  examples/embodied/*.py \
  examples/embodied/libero/*.py

uv run --extra dev pytest -q \
  tests/test_embodied_art_compat.py \
  tests/test_embodied_cli.py \
  tests/test_embodied_config.py \
  tests/test_embodied_local_process_backend.py \
  tests/test_embodied_observability.py \
  tests/test_embodied_package_boundary.py \
  tests/test_embodied_rollout_process.py \
  tests/test_embodied_runner.py \
  tests/test_embodied_utils.py

uv build
```

Changes to policy loading, action likelihoods, distributed training,
checkpointing, or simulator integration require the relevant accelerator gate.
For OpenVLA-OFT that means at least one real H100 optimizer update followed by
checkpoint reload. Changes that claim learning quality also require the fixed
native evaluation described in
`docs/experimental/embodied-release-validation.mdx`; a loader smoke or lower
surrogate loss is not sufficient evidence.

### Empirical policy-opening gate

A new policy or simulator path must pass a real-task positive control before it
is tested with harder tasks or additional research variables. Unit tests,
loader smoke tests, finite gradients, checkpoint reloads, and one successful
optimizer update establish mechanical correctness only; they do not establish
that the RL implementation can improve a policy.

The positive control must:

- use the released checkpoint and native runtime that users will actually run;
- use one real benchmark task with measured headroom and prior evidence that the
  base policy can perform the task;
- exclude instruction or visual perturbations, domain randomization, multitask
  training, task routing, replay, branching, and other experimental mechanisms;
- run enough updates and fixed development evaluations to demonstrate a
  sustained success-rate lift rather than a favorable single evaluation;
- log standalone `train/*` and `validation/*` metrics, media, configuration, and
  checkpoints to W&B, with the sealed evaluation pair prepared in advance.

Only after this gate passes should experiments add one failure mode at a time.
For example, qualify single-task learning before multitask balancing, and
qualify the unperturbed task before testing language or camera robustness. If
the positive control fails, treat the policy integration or base RL recipe as
unqualified and debug it before interpreting results from harder settings.

## Design rules

- Keep generic control-plane code independent of LIBERO, Slurm, and any one
  policy family.
- Put simulator/task-specific behavior in its integration or example package.
- Keep YAML as the reproducible experiment contract. Environment variables are
  for secrets and explicit operational overrides only.
- W&B and Weave emission must be no-op when disabled. When enabled, metrics,
  videos, artifacts, and trajectory traces must share the same update identity.
- Preserve native policy samplers and their action likelihoods.
- Do not claim a model family as supported until a warm-start baseline and an
  RL checkpoint have been compared with the same fixed native evaluator.
- Do not combine policy bring-up with robustness or multitask experiments. Pass
  the empirical policy-opening gate first, then change one variable at a time.

## Pull requests

Describe the user-visible behavior, compatibility impact, tests run, and any
remaining unsupported path. Include W&B run links or compact local evidence for
learning-quality changes, but never commit API keys, `.env` files, model weights,
large rollout payloads, or generated videos.

### Keep experiments PR-ready from the start

Do not accumulate a research diary in the feature branch and sort it out only
after an experiment succeeds. Separate the deliverable from scratch work when
creating files:

- Keep reusable implementation in `src/`, regression tests in `tests/`, and
  maintained reproduction entrypoints/recipes in `examples/` or `scripts/`.
- Keep one-off probes, intermediate recipe variants, run-specific launch files
  and investigation notes in `ART_EMBODIED_PRIVATE_DIR`, outside every Git
  checkout. Ignore rules are a fallback, not a place to store internal records.
- Never make shipped entrypoints depend on scratch helpers. Extract reusable
  functionality into a maintained module with tests instead of importing an
  entire one-off investigation script.
- Stage explicit deliverable paths and review the staged name list and diffstat
  after each implementation increment. Do not use blanket `git add .` or
  `git add -A` to package an experiment. Ignore rules are not a secrets scanner.
- Preserve full provenance in experiment outputs and W&B. On success, add the
  final recipe, concise result/reproduction documentation and evidence links,
  rather than copying every intermediate report into the PR.

The intended outcome is that a successful experiment is already running the
PR-ready implementation. Final review and checks should verify that boundary,
not trigger a large post-success cleanup and another experiment solely because
research files became runtime dependencies.

### Final-tree validation order

Finish the release boundary before spending GPU time on release validation:

1. Select the implementation, recipes, regression tests and concise user docs.
   Keep one-off investigations outside Git; retain all runtime dependencies.
2. Test a separate Git checkout of the proposed merge tree with no research
   outputs or archived helpers. Run core and optional numerical/W&B suites,
   packaging checks and entrypoint checks before allocating GPUs.
3. Run the affected policy's accelerator gate from that exact checkout. Record
   its commit, source checksums, resolved recipe, runtime versions, input model
   identity and W&B run. Do not change unrelated qualified policy environments.
4. Read back the optimizer/checkpoint and W&B history, native update/media axes,
   evaluation records, videos and model bytes. Report storage verification and
   App-rendering verification separately. Keep the PR unqualified on failures.

For a packaging/pruning regression gate, retain the successful recipe's real
batch size, parallelism, precision and evaluation panel, and use a separate
run/output directory. One full update tests execution, not learning quality;
it does not replace the empirical policy-opening gate or sealed comparison.
Do not overwrite or resume a published result merely to test packaging.

Pruning after this GPU gate defeats the ordering: changes to runtime code,
imports, recipes, assets or dependencies invalidate the affected evidence and
require requalification. Documentation-only follow-ups can reuse it when the
unchanged execution inputs are explicitly verified. Link the final-tree proof
in the PR rather than asking the user to request a GPU check separately.
