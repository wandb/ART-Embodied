# LeRobot and ART-Embodied contract

Primary LeRobot reference:

- <https://huggingface.co/docs/lerobot/envhub>

## LeRobot EnvHub boundary

The pack's `lerobot/env.py` must expose:

```python
def make_env(n_envs: int = 1, use_async_envs: bool = False):
    ...
```

It may return:

- `gym.vector.VectorEnv` (preferred);
- one `gym.Env`, which LeRobot can wrap;
- `{suite_name: {task_id: VectorEnv}}` for a multi-task benchmark.

Implement standard Gymnasium `reset()` and `step()` behavior. Pin a revision
when publishing through EnvHub because loading a Hub environment executes
remote Python.

## Observation and action contract

Record:

- image keys, shape, dtype, color order, camera frame, and timestamp;
- proprioceptive state ordering, units, and normalization;
- action ordering, units, frame, limits, control frequency, and chunk horizon;
- simulator-to-real processor mapping.

Use the selected LeRobot policy's native pre/post-processors. Do not silently
transpose, normalize, clip, or change coordinate frames inside unrelated code.

Every episode must expose a machine-readable terminal signal:

```python
info["is_success"]
```

Keep environment reward distinct from success. ART-Embodied may optimize dense
reward, sparse verification, or both, but evaluation reports success directly.

## ART-Embodied boundary

Start from the closest validated recipe in `examples/embodied/` and adapt:

- environment factory and kwargs;
- policy/checkpoint and processors;
- action representation;
- reward and verifier;
- fixed evaluation/reset manifests;
- rollout and training resource geometry;
- W&B/Weave destination and bounded media.

Run:

```bash
uv run art-embodied validate art_embodied/experiment.yaml
```

Task-specific predicates, reset distributions, simulator code, and assets stay
inside the Use Case Pack. Generic ART-Embodied code should only receive typed
environment, trajectory, reward, and policy interfaces.

Use fixed development and sealed test reset manifests. Log baseline at Step 0,
then training and validation success under stable metric names in one W&B run.
Representative videos use stable keys so panels do not multiply per step.
