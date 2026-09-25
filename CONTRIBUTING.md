# Contributing to ART-Embodied

Bug reports, documentation improvements, tests, and code contributions are
welcome. ART-Embodied adds trajectory-level reinforcement learning and experiment
management to LeRobot workflows using OpenPipe ART.

## Getting started

For small fixes, open a pull request directly. For a new policy, simulator,
training objective, or substantial API change, open an issue first so we can
agree on the scope and validation plan.

When reporting a bug, include the recipe, package versions, expected behavior,
and a minimal reproduction. Include relevant logs or experiment links when you
can share them, but remove credentials and private information.

## Development setup

Fork the repository on GitHub, then clone your fork and create a branch:

```bash
git clone https://github.com/YOUR-USERNAME/ART-Embodied.git
cd ART-Embodied
git switch -c my-change
git config core.hooksPath .githooks
uv sync --python 3.12 --locked --extra dev
```

Core development supports Python 3.11 and 3.12. The setup above is enough for
core tests. Policy and simulator changes need the corresponding environment
from the [README](README.md#install). Keep those environments separate.

The package imports as `art_embodied`. OpenPipe ART is a dependency, so changes
belong in this package rather than in a bundled copy of `art`.

## Testing your changes

Run the core checks from the repository root:

```bash
uv run --locked --extra dev ruff check \
  src/art_embodied \
  tests/test_embodied_*.py \
  examples/embodied/*.py \
  examples/embodied/libero/*.py
uv run --locked --extra dev pytest -q
uv build
```

Tests requiring optional dependencies may be skipped in the core environment.
CI also runs numerical and W&B regression tests with additional dependencies.
Add regression tests for fixes and describe which checks you ran in your PR.

Validation depends on the change:

| Change | Expected validation |
| --- | --- |
| Documentation | Check instructions, examples, and links. No GPU is needed. |
| Core APIs, configuration, or utilities | Relevant unit tests and the core checks above. |
| Policy execution, likelihoods, distributed training, checkpointing, or simulator behavior | Relevant CPU tests and GPU validation agreed with a maintainer, including an update and checkpoint reload where applicable. |
| New policy or simulator support, or learning-performance claims | Real-task rollouts and fixed evaluations comparing the starting and trained policies. Distinguish development results from a separate held-out final test. |

You do not need access to a GPU to submit a contribution. Maintainers can run
GPU validation when needed on hardware appropriate for the affected policy.
Mention unavailable checks in your PR so we can arrange them before merging.

For learning changes, start with a controlled benchmark before adding further
experimental variables. Share the recipe, checkpoint identity, and evaluation
results. W&B metrics, videos, and artifacts should refer to the same training
update, including after resume. See the
[validation guide](docs/experimental/embodied-release-validation.mdx) for details.

## Design guidelines

- Keep shared training and experiment-management code independent of any one
  policy, simulator, or cluster scheduler.
- Put policy-specific sampling and likelihood logic in policy adapters, and
  environment-specific behavior in simulator integrations.
- Use YAML for reproducible experiment settings. Reserve environment variables
  for credentials and machine-specific configuration.
- Preserve existing policy behavior and checkpoint compatibility, or explain
  the migration required by your change.
- Keep W&B and Weave optional. When enabled, preserve consistent metrics,
  media, evaluation records, and trajectory traces across adapters.

## Submitting a pull request

Open your PR against `main` and describe the change, any related issue,
compatibility impact, and test results. Small, focused PRs are easier to review.
Draft PRs are welcome when you need feedback or help with validation.

Keep internal notes, operational records, and private data outside Git. Submit
reusable code, tests, maintained recipes, and concise documentation. Do not
commit credentials, model weights, generated videos, or large rollout files.
Hooks and CI check for some private content, but review your diff as well.

The `art-embodied` team reviews contributions. A maintainer will help identify
any additional checks needed before merging. If you change execution-related
code after validation, rerun the affected checks or ask for help doing so.

Only submit material you have permission to contribute, and retain required
third-party license notices. Maintainers will confirm any required contribution
agreement before merging.

## Contributing

Contributors must agree to the [CoreWeave CLA](./CLA.md) when pushing code to this project.

Agreement with the CoreWeave CLA must signified by including a `Signed-Off-By` trailer in every submitted Git commit to this repository. By signing off, you certify that you have the right to submit the contribution and that you agree to and are bound by the CoreWeave Contributor License Agreement in effect at the date of your submission, found as [CLA.md](./CLA.md) , which governs your submission. If you are contributing on behalf of an entity, you further certify that you are authorized to bind that entity to the CLA.

Individual commits can be signed using `--signoff` option to [`git commit`](https://git-scm.com/docs/git-commit#Documentation/git-commit.txt---signoff); or a repo as a whole can use the `commit.signoff` configuration option.
