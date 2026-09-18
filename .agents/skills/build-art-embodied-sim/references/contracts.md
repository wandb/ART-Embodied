# Real-to-sim contracts

## Information precedence

Use the strongest available source per parameter:

| Parameter | Preferred evidence |
|---|---|
| geometry/dimensions | measured value, customer CAD, manufacturer CAD/spec, multi-view estimate |
| mass | measured value, exact datasheet, density x volume estimate |
| inertia | CAD mass distribution, then bounded primitive approximation |
| joint axis/limits | CAD/manual/observed motion, then category prior |
| friction/damping | system identification, then broad material/mechanism prior |
| appearance | current site photos/video |
| success/safety | user SOP and explicit task instruction |

General model knowledge is a prior, never a measurement.

Every known scale must identify the physical feature it measures, such as
`blue_part.width` or `drawer_opening.inner_width`. Do not use an unlabeled
scale bar to infer all scene dimensions.

## ArtSceneIR

Use SI units and a right-handed Z-up world. Every task-relevant object needs:

- stable ID and semantic role;
- parent frame and transform;
- visual and collision geometry sources;
- dimensions with provenance;
- dynamic/static classification;
- mass, friction, restitution, and inertia where dynamic;
- articulation with local axis, anchor, limits, damping, and friction;
- uncertainty and human override state.

Each inferred scalar uses:

```yaml
estimate: 0.48
range: [0.25, 0.75]
unit: dimensionless
confidence: 0.22
source: visual_material_prior
status: provisional
task_sensitivity: high
human_override: null
calibrated_value: null
```

Never encode unknown as fake precision. Use `null` plus a plausible range or
block validation if no defensible range exists.

## TaskSpec

Keep task semantics declarative:

- natural-language instruction;
- manipulated objects and target regions;
- initial-state distributions;
- success predicates and hold duration;
- failure predicates and timeout;
- reward components and their relationship to terminal success;
- safety constraints;
- observation and action contracts.

Predicates use a bounded vocabulary such as `inside`, `contact`, `grasped`,
`released`, `joint_above`, `pose_within`, `upright`, `not`, `all`, and `any`.
Extend the vocabulary explicitly and test both true and false cases.

## Evidence ledger

For every external fact or asset record:

- source type and URL/path;
- exact model/variant match confidence;
- extracted value and unit;
- retrieval timestamp;
- content hash;
- license and redistribution status;
- whether the source may enter a public pack;
- conflicts and resolution.

Separate public-web research from private customer inputs. Never send private
CAD, SOPs, images, or telemetry to public search or generation providers
without explicit authorization.

## Calibration

Use teleoperation data to fit high-sensitivity parameters where possible:

- pushed distance and object motion for friction;
- current/torque during lift for mass;
- drawer trajectory for joint axis, damping, and friction;
- grasp slip for contact properties.

Keep the initial prior, fitted value, objective, dataset hash, and residual.
Use remaining uncertainty as domain-randomization ranges.
