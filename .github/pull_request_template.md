## Change

Describe the user-visible change and compatibility impact.

## Validation

- [ ] Scratch probes/configurations/diaries stayed separate during development;
      only implementation, tests, maintained recipes and concise docs are staged.
- [ ] Shipped entrypoints do not depend on scratch or archived research helpers;
      any existing dependency requiring retention is explicitly identified.
- [ ] Clean-checkout tests, optional numerical/W&B tests and package checks pass.
- [ ] For accelerator-path changes or pruning: final-tree commit/source hashes,
      resolved recipe, runtime/input-model identity and GPU/W&B evidence linked.
      Otherwise explain why the accelerator gate is not applicable.
- [ ] Checkpoint, history, native update/media axes, evaluation, videos and model
      uploads verified; App rendering marked verified or explicitly unverified.
- [ ] Execution inputs have not changed since the linked accelerator evidence.
- [ ] Learning claims include real-task development and separate sealed evidence,
      or explicitly state that this PR makes no new learning-quality claim.

List remaining limitations and skipped checks. See CONTRIBUTING.md for the
final-tree validation order. Never include secrets or large generated assets.
