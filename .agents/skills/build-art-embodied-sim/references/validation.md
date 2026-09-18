# Validation levels

Generation is not validation. Advance only after the prior level passes.

## Level 1: structure

- required pack files exist;
- YAML parses and required IDs/units are present;
- no `TODO`, fake path, or unresolved high-sensitivity value is hidden;
- evidence and license status exist for external assets;
- success and failure predicates are non-empty;
- randomization ranges are finite, ordered, and physically plausible.

## Level 2: sim

- MJCF XML parses and `mujoco.MjModel.from_xml_path()` succeeds;
- referenced meshes/textures exist;
- initial state has no unintended penetrations or explosive contacts;
- at least 1,000 zero/random-action steps produce no NaN or unstable energy;
- repeated seeded resets are valid and diverse as specified;
- robot workspace reaches pre-grasp and target poses;
- joints move on the intended axes and respect limits;
- grasp/contact geometry is feasible.

## Level 3: LeRobot

- `make_env(n_envs=2, use_async_envs=False)` works;
- reset and 1,000 random steps match declared spaces;
- observations/actions have stable keys, shapes, dtypes, units, and frames;
- termination and truncation are distinct;
- `info["is_success"]` is always available;
- constructed positive and negative states exercise every predicate;
- vector environments do not share mutable simulator state.

## Level 4: ART-Embodied

- experiment YAML passes `art-embodied validate`;
- fixed baseline evaluation completes;
- same-reset grouped rollouts contain valid trajectories;
- reward and success are logged separately;
- a short update changes intended policy parameters and reloads;
- fixed validation still runs after checkpoint reload;
- W&B metrics, bounded videos, Weave traces, and checkpoint artifact share the
  same update index.

## Level 5: evidence-ready

- scripted or teleoperated policy can achieve success;
- failure cases remain reachable;
- short RL smoke collects non-degenerate reward/advantage signal;
- representative success/failure videos are inspectable;
- package manifest contains source/config/environment hashes;
- results can be regenerated from the pack without an external generation service;
- unresolved sim-to-real gaps and calibration requirements are explicit.

The bundled validator automates contract, XML, MuJoCo, Gymnasium, and
ART-Embodied checks. Reachability, contact quality, scripted success, and
sim-to-real calibration remain task-specific tests that Codex must implement.
