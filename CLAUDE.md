# ART-Embodied development notes

Use `uv` for dependency management and commands. Read `CONTRIBUTING.md` before
changing package boundaries, policy objectives, evaluation, or release claims.

The public package is `art-embodied`, imported as `art_embodied`, and depends on
published `openpipe-art`. Do not add add-on code to the upstream `art` namespace.
Generic code must remain independent of Slurm, LIBERO, and any single VLA family.

Run add-on-owned Ruff and CPU tests before committing. Policy-loading,
likelihood, distributed-training, checkpoint, or simulator changes also require
the corresponding H100/fixed-evaluation gate. Never infer learning quality from
a smoke test or train rollout success alone.

Do not publish releases, push branches, stop long-running experiments, or delete
retained checkpoints without explicit user approval.
