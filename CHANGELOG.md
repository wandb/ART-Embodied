# Changelog

## Unreleased

## 0.1.0rc2

### Compatibility

- Support OpenPipe ART 0.5.20 on Python 3.12+, retaining ART 0.5.18 for Python
  3.11 and the GR00T N1.7 native runtime in one ART-Embodied package.
- Adapt ART model identity and Transformers attention-mask signatures across
  the supported versions. Preserve GR00T's SciPy, Weave, and W&B pins.
- Verify the upgrade with CPU regressions, PI0 fixed-input GPU actions and
  gradients, and PI0-FAST optimizer updates, resume, and W&B read-back checks.
  See [validation details](docs/experimental/upstream-art-compatibility.md).

### Security

- Update the standard lockfile's GitPython, aiohttp, Pillow, cryptography, and
  hydra-core versions to patched releases. Enforce security floors in uv and
  the documented pip installation commands.
- Address 56 of the 105 advisory matches in the 2026-09-13 snapshot. Document
  49 retained matches, upstream constraints, and the separate native NVIDIA
  environment in the [security review](docs/experimental/dependency-security.md).

### Policies and Experiments

- Add Flow-SDE GRPO integrations for PI0/PI0.5, SmolVLA, and GR00T N1.7,
  including the RoboCasa adapter. See the [README results](README.md) for
  each policy's evaluation scope and reproduction recipe.
- Record PI0-FAST trajectory-level GRPO on one LIBERO Long task: 70/100 to
  89/100 development success after 100 updates; a separate sealed comparison
  improved from 73/100 to 83/100 (paired 95% CI [0, 20] percentage points).
  This experiment uses an isolated MuJoCo 3.3.0 runtime, not the default LIBERO
  dependency profile. See `docs/experimental/pi0-fast-long-result.md`.
- Add explicit PI0-FAST precision/loss-scaling contracts, native decoding with
  completed-row removal, full-sequence scoring, and measured multi-actor training
  diagnostics. Keep experimental precision changes scoped to PI0-FAST.
- Add opt-in native W&B update-step transactions and resume checks so scalar,
  evaluation, and video steps remain aligned; validate checkpoint payloads and
  history/media read-back contracts.
- Add opt-in LIBERO reset-health diagnostics and experimental LIBERO-Plus and
  task-balancing research tools; these are not additional learning-quality claims.

- Latest generated-state development validation reached Object `100/100` at
  update 200 (SFT `34/100`) and Spatial `88/100` at update 100 (SFT
  `48/100`). The earlier RC milestones below remain part of the result history.
- Post-RC Spatial validation reached `82/100` at update 30 on the frozen
  generated-state set, versus `48/100` for SFT and `80/100` at update 20. The
  SFT comparison is conclusive (37 improvements, three regressions, exact
  McNemar `p=1.95e-8`); the update-20 to update-30 increment is not (seven
  improvements, five regressions, `p=0.774`).

## 0.1.0rc1 - 2026-07-19

First release candidate of the standalone ART-Embodied add-on.

### Validated

- Install after published `openpipe-art==0.5.18` without sharing the `art`
  package namespace.
- Run OpenVLA-OFT action-token trajectory-level GRPO with native LeRobot/LIBERO
  policy execution, grouped rollouts, distributed LoRA training, bounded
  checkpoint/resume, and fixed paired evaluation.
- Improve LIBERO Object generated-state success from SFT `34/100` to
  ART-Embodied `97/100` at update 100.
- Improve LIBERO Spatial generated-state success from SFT `48/100` to
  ART-Embodied `80/100` at update 20 without importing RLinf at runtime.
- Record optimizer-step metrics, stable representative videos, versioned model
  artifacts, and nested trajectory traces through optional W&B and Weave
  observability.
- Complete an ART-first install in an empty Python 3.11 environment and a real
  H100 OpenVLA-OFT optimizer/checkpoint/reload acceptance test.

### Release-candidate boundaries

- OpenVLA-OFT action-token LoRA GRPO is the only validated policy-training
  family in this candidate.
- SmolVLA flow matching and pi0/pi0.5 training are not supported paths yet.
- The validated compatibility target is OpenPipe ART 0.5.18 and LeRobot
  `>=0.4.4,<0.5`; later versions require the same regression gates before the
  compatibility window is widened.
- The OpenVLA-OFT backend is an embodied extension and does not execute through
  upstream ART LocalBackend, AOM, or Serverless Training.
