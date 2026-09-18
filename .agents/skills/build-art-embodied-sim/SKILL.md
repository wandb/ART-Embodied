---
name: build-art-embodied-sim
description: Experimentally scaffold, repair, and validate candidate MuJoCo simulations from task photos or videos, natural-language instructions, measurements, existing robot profiles, CAD, or generated assets. Use for real-to-sim prototyping, task-specific digital-twin prototypes, MJCF and Gymnasium/LeRobot EnvHub authoring, machine-checkable success and reset logic, uncertain physics estimates, or candidate embodied-RL environment packages.
---

# Build an ART-Embodied simulation

> **Experimental and unvalidated.** This Skill is an internal research
> prototype, not a supported ART-Embodied capability. It has not demonstrated
> end-to-end generation from photographs through successful RL. Never describe
> its output as validated merely because the scaffold or individual checks
> complete; report exactly which gates ran, which passed, and what still
> requires expert review or physical calibration.

Produce a self-contained **Use Case Pack**, not merely a 3D asset. Continue
until the requested validation level passes or a concrete physical ambiguity
requires human input.

## Read only what is needed

- Read `references/contracts.md` before authoring SceneIR, TaskSpec, physics
  priors, or evidence.
- Read `references/lerobot-art-contract.md` before implementing `env.py` or the
  ART-Embodied experiment.
- Read `references/validation.md` before claiming any readiness level.

## Non-negotiable rules

1. Reuse a verified robot profile by default. Generate task objects, fixtures,
   and task-relevant workspace geometry; do not regenerate the robot.
2. Search customer-provided CAD/specifications and official manufacturer
   sources before generating geometry. Record source, license, hash, match
   confidence, and retrieval date in the evidence ledger.
3. Treat photo-derived dimensions and physical properties as uncertain priors.
   Store value, plausible range, confidence, and source. Never present mass,
   friction, damping, or inertia inferred from appearance as measured truth.
4. Use SI units, right-handed Z-up coordinates, explicit frame names, and
   simple stable collision geometry.
5. Never upload private images to an external service without explicit user
   authorization.
6. Retain approved assets locally, with their source and license recorded.
   Training must work without access to the asset-generation service.
7. Generate declarative TaskSpec predicates first. Do not let generated Python
   silently define success, safety, or reward semantics.
8. Do not claim `LeRobot-ready` until `make_env()` passes reset/step tests. Do
   not claim `ART-Embodied-ready` until the experiment YAML validates. Do not
   claim `RL-ready` without a machine-checkable success signal and deterministic
   reset coverage.

## Build loop

### 1. Inspect and scaffold

Identify:

- task instruction and completion semantics;
- selected verified robot profile;
- manipulated objects, fixtures, target regions, and safety constraints;
- available photos/videos, at least one known scale when possible, CAD/specs,
  and teleoperation trajectories;
- target policy family, simulator, and validation level.

Create a pack:

```bash
python .agents/skills/build-art-embodied-sim/scripts/init_use_case.py \
  --output workcells/my-task \
  --name my-task \
  --task "Place the blue part into the fixture" \
  --robot-profile franka-panda \
  --known-scale-m 0.10 \
  --known-scale-description "fixture opening width" \
  --reference-image ./inputs/front.jpg
```

Do not overwrite an existing pack. Revise it in place and preserve evidence.
`--known-scale-description` must identify the exact visible dimension; a bare
number is not sufficient. The scaffold hashes local references but does not
authorize or perform an upload. Add `--authorize-reference-upload` only after
the user explicitly authorizes an external provider to receive them.

### 2. Resolve evidence and assets

Use this order:

1. customer CAD/BOM/SOP and measurements;
2. manufacturer CAD, datasheet, and manual;
3. verified simulation catalog;
4. parametric reconstruction;
5. approved generated assets;
6. primitive approximation.

Use world knowledge to form hypotheses and search queries, not as numeric
ground truth. Ask the user only when a parameter is both highly uncertain and
highly sensitive to task feasibility or safety.

### 3. Prepare local assets

Copy or compose approved CAD, MJCF, or mesh assets into `scene/`.
Record their source, license, and hashes in `evidence/ledger.yaml`.
Check dimensions, coordinate frames, and collision geometry before using them
in the task.

### 4. Author the task contract

Complete:

- `scene/scene_ir.yaml`;
- `task/task_spec.yaml`;
- `robot/robot_profile.yaml`;
- `physics/priors.yaml`;
- `physics/overrides.yaml`;
- `randomization/domain_randomization.yaml`;
- `evidence/ledger.yaml`.

Separate visual and collision geometry. Prioritize parameters by task
sensitivity. Convert unresolved but plausible uncertainty into domain
randomization ranges.

### 5. Compile and repair

Implement `lerobot/env.py` as a standard Gymnasium environment and expose:

```python
make_env(n_envs: int = 1, use_async_envs: bool = False)
```

Return a `gym.vector.VectorEnv`, a single `gym.Env`, or the documented LeRobot
multi-task mapping. Ensure every step exposes `info["is_success"]`. Implement
deterministic seeded reset, observation/action schemas, limits, truncation,
positive and negative success tests, and safe cleanup.

Create `art_embodied/experiment.yaml` from the closest validated repository
recipe. Keep environment-specific code inside the Use Case Pack; do not add
task-specific behavior to generic ART-Embodied modules.

Inspect renders and contacts. Prefer directly repairing MJCF, TaskSpec, or
adapter code over repeatedly regenerating a nearly-correct scene.

### 6. Validate progressively

```bash
validator=.agents/skills/build-art-embodied-sim/scripts/validate_use_case.py
python "$validator" workcells/my-task --level structure
python "$validator" workcells/my-task --level sim
python "$validator" workcells/my-task --level lerobot --random-steps 1000
python "$validator" workcells/my-task --level art
```

Fix failures and rerun. Preserve `validation/report.json`. Then run task-specific
reachability, scripted success, contact stability, reset diversity, and short
rollout/training probes described in `references/validation.md`.

## Completion report

Report:

- achieved readiness level and failed/skipped gates;
- robot/profile revision and simulator/version;
- scene/task hashes and asset provenance without credentials;
- evidence-backed, overridden, calibrated, and randomized parameters;
- unresolved high-sensitivity uncertainties;
- exact commands for LeRobot and ART-Embodied;
- representative render/video and validation report paths.

Do not hide uncertainty behind a single “generated successfully” result.
