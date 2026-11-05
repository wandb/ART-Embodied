# Upstream ART Compatibility

The current checkout selects ART 0.5.24 for Python 3.12+. Python 3.11 and
native GR00T environments retain ART 0.5.18. Existing environments are not
upgraded in place. Release `0.1.0rc2` predates this update.

## Runtime Selection

| Environment | ART version | Reason |
| --- | --- | --- |
| Python 3.12+ default | 0.5.24 in `uv.lock` | Validated default runtime |
| Python 3.11 | 0.5.18 | Newer ART versions require Python 3.12 |
| GR00T N1.7 native installer | 0.5.18 | NVIDIA pins SciPy 1.15.3; ART 0.5.24 requires >=1.17,<1.18 |
| Existing qualified environments | 0.5.18 and 0.5.20 remain supported | No in-place environment upgrades |

On Python 3.12+, package metadata allows 0.5.18, 0.5.20, and 0.5.24. The
GR00T installer constrains ART, SciPy, Weave, and W&B separately and does not
inherit the checkout's dependency overrides.

## November 2026 Update

[ART 0.5.24](https://github.com/OpenPipe/ART/releases/tag/v0.5.24) requires
Weave 0.54.0 to support the OpenTelemetry 1.45 exporter. The checkout updates
both, together with the security floors described in
[Dependency Profiles and Security](dependency-security.md). Policy-specific
Torch, Transformers, Tokenizers, and LeRobot versions are unchanged.

- Core tests: 717 passed on each of Python 3.11 and 3.12, with 80 optional
  tests skipped per environment.
- ART 0.5.24 numerical and offline W&B tests: 1,202 passed, 13 skipped.
- OpenVLA-OFT environment: 1,139 CPU tests passed, 33 skipped. Tests for PI's
  LeRobot processors and newer Transformers helpers run in their own profile.
- Fresh wheel installation and runtime checks passed on both Python versions.
  Dependency resolution also passed for GR00T's separately constrained profile.
- Pi0-FAST initial evaluation: all 100 action sequences and outcomes match
  ART 0.5.20. Both policies succeed in 70 episodes. The first rollout's 120
  groups also match in inputs and outcomes (960 trajectories).
- [Pi0-FAST update/resume run](https://wandb.ai/wandb-japan/art-embodied-pi0-fast-long/runs/6o7qgupb):
  two updates passed, with checkpoint resume between them. Optimizer counters
  advance from 8 to 16; weights and optimizer state are finite, and RNG state
  is retained. W&B native steps are **0, 1, 2**. Metric values, media steps,
  downloaded videos, and model artifact bytes passed automatic checks.
  App rendering was not inspected.
- Equal-work Pi0-FAST timing versus ART 0.5.20: first-update rollout
  **17.7 vs 18.2 minutes**, training **9.6 vs 9.6 minutes**. The second update
  took 18.2 minutes for rollout and 9.7 minutes for training.
- PI0 fixed-input GPU comparison: all **85 tensors** match bitwise with
  ART 0.5.20, including actions, rollout/rescored logprobs, and LoRA gradients.
- OpenVLA-OFT fixed-input GPU comparison: predictions, rescored logprobs, and
  all **852 gradient tensors** match bitwise after the dependency updates.
  This profile retains ART 0.5.18.
- GR00T's Pillow-only fixed-input GPU comparison: ODE/SDE actions,
  rollout/rescored logprobs, all **452 gradient tensors**, and post-update
  scores match bitwise. Checkpoint reload also passed. This checks Pillow 13
  in the separately pinned native runtime; it does not upgrade native ART
  or apply the root lockfile to that environment.

These are compatibility checks, not a new long-run learning comparison.
Pi0-FAST's updated weights differ from the ART 0.5.20 reference by at most
9.11e-6, within the historical repeatability range reported below. No sealed
outcomes were used.

## September 2026 Baseline

The following results describe ART 0.5.20 and the RC2 release, not a new
long-run learning comparison for ART 0.5.24.

### Reproduced Failures and Fixes

- **Model identity:** ART 0.5.20 requires `TrainableModel.run_name`. Without the
  bridge, five lifecycle tests failed during construction. Both fields now use
  the same resolved embodied run identity. Tests cover registration, explicit
  names, checkpoint lineage, and resume from update 7 to 8.
- **Transformers masking:** ART 0.5.20 wraps the Transformers 5.5 signature.
  Importing it before a Transformers 4.57 Llama caused
  `AttributeError: 'DynamicCache' object has no attribute 'shape'` because
  positional arguments shifted. The bridge preserves native argument order and
  packed position IDs. GR00T N1.7 now invokes it before loading its native model.
- **Dependencies:** Python 3.11 and GR00T's SciPy/Weave pins cannot resolve with
  ART 0.5.20. They retain ART 0.5.18 instead of forcing incompatible dependencies.

Source: [ART 0.5.20 on PyPI](https://pypi.org/project/openpipe-art/0.5.20/).
Published 0.5.18 and 0.5.20 wheel hashes were checked against PyPI metadata.
ART 0.5.20 and 0.5.24 restrict LiteLLM to <=1.82.0. The current checkout
overrides this with LiteLLM 1.101.3 through uv. See the
[dependency profiles](dependency-security.md) for installation differences.

### CPU and Package Checks

| Check | Result |
| --- | --- |
| Python 3.11 / ART 0.5.18 core suite | 649 passed, 80 skipped |
| Python 3.12 / ART 0.5.20 core suite | 649 passed, 80 skipped |
| ART 0.5.20 numerical and offline W&B suite | 1,121 passed, 15 skipped |
| Attention-boundary matrix | 11 passed per combination |
| Wheel installation on Python 3.11 and 3.12 | Passed |
| Isolated worker-profile resolution and package ownership | Passed |

These tests use the real package metadata and compatibility guard. The
attention matrix covers ART 0.5.20 with Transformers 5.5.4 and 4.57.3, and ART
0.5.18 with Transformers 4.57.3. Native and repaired small-model outputs and
gradients match at zero tolerance. Optional GPU and external-asset tests are
reported as skipped in the CPU suites.

Logs, environment freezes, and diagnostic scripts stay outside the source tree.

### GPU Acceptance

Use existing checkpoints and fixed development inputs, with the same
precision, actor count, batch size, and episode budget as the
qualified recipe. Do not retrain every supported policy for 100-200 updates.

1. Compare actions, likelihoods, and gradients at affected boundaries.
2. Run a real optimizer update on representative or directly affected paths.
3. Resume with the same run ID, optimizer/RNG state, and completed-update axes.
4. Verify live W&B metric values, media steps/bytes, and model artifacts against
   local evidence. App rendering is unverified unless actually inspected.
5. Compare equal-work rollout and training time with the existing baseline.

#### Pi0-FAST: Update and Resume

[W&B run](https://wandb.ai/wandb-japan/art-embodied-pi0-fast-long/runs/sq2slg05)
used the qualified SFT checkpoint, 960 trajectories per update, eight training
GPUs, and four rollout actors per GPU. Only ART and its new transitive
dependencies changed; ML, simulator, and W&B versions were retained.

- Initial evaluation: all 100 episode action sequences and outcomes match the
  ART 0.5.18 reference; success rate is 70% in both.
- First rollout: all 120 groups match the reference inputs, rewards, and
  successes (960 trajectories).
- Two completed updates, with checkpoint resume between them: optimizer step
  counters advance from 8 to 16, with finite weights and optimizer state.
- W&B retains one run and native steps **0, 1, 2**. Measured metrics, media
  steps, downloaded videos, and model artifact bytes passed automatic checks.
  App rendering was not inspected.
- Equal-work first-update timing: rollout **18.2 vs 17.9 minutes**, training
  **9.6 vs 9.5 minutes** (candidate vs ART 0.5.18).

Updated adapters are not bitwise identical. The candidate/reference weight
difference has maximum absolute value **8.37e-6** and L2 norm **7.36e-4**;
two existing ART 0.5.18 executions differ by **1.08e-5** and **9.71e-4**,
respectively. The candidate's post-update evaluation is 73%, versus 77% in
the reference. The observed weight difference is within this measured
old-version repeatability range; this small comparison does not establish
long-run learning equivalence.

#### PI0: Fixed-Input GPU Comparison

Using the same checkpoint, seeds, precision, and synthetic observation, all
**85 tensors** match bitwise between ART 0.5.18 and 0.5.20: actions, rollout and
rescored logprobs, and 82 LoRA gradients. Checkpoint save/load also passed.
This checks the full-policy numerical path, not simulator learning.

Subsequent changes pinned the GR00T installer's W&B version and updated
packaging, documentation, and tests.
The follow-up passed 34 package/compatibility tests (one optional test skipped),
dependency resolution, shell syntax, lint, lock, and distribution-build checks.
No sealed outcomes were used. Other policies were covered by CPU/interface
tests and dependency checks, not new long GPU training runs.

Longer learning runs are needed if these checks expose meaningful numerical
or learning-path changes.

### RC2 Packaging

The `0.1.0rc2` wheel installs in clean Python 3.11 and 3.12 environments,
selecting ART 0.5.18 and 0.5.20 respectively. Dependency checks and the installed
`art-embodied doctor` pass in both. All 82 packaged runtime files match the
GPU-qualified development wheel byte for byte.

All five READMEs describe the same runtime selection and install commands and
are included in the source distribution. A regression test checks these
against the package version. Worker-profile resolution retains the GR00T pins.

RC2 validation passed 650 core tests on each Python version (80 skipped per
environment) and 1,122 numerical/W&B tests (15 skipped). The final source-
distribution change also passed all 12 package-boundary tests.

The current checkout has additional dependency updates, including
SentencePiece, LiteLLM, and Diffusers. Their installation scope and validation
are summarized in [Dependency Profiles and Security](dependency-security.md).
The existing RC2 assets predate those updates.
