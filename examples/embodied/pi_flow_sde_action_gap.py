"""Measure native-ODE versus Flow-SDE action drift on fixed observations.

Success-rate calibration can hide compensating action errors. This diagnostic
therefore compares action chunks before any environment transition, using the
same initial observation for one deterministic sample and multiple stochastic
samples. It writes only bounded summary statistics and compact NumPy arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np

from art_embodied import EmbodiedExperimentConfig, make_policy
from art_embodied.compatibility import (
    require_compatible_runtime,
    runtime_profile_for_policy,
)
from art_embodied.experiment import RolloutContext
from art_embodied.integrations.pi_flow_sde import (
    _postprocess_pi_actions,
    _prepare_pi_observation,
)
from examples.embodied.libero.components import (
    LiberoSettings,
    LiberoTaskCatalog,
    build_evaluation_scenarios,
    prepare_libero_runtime_paths,
    validate_libero_runtime_imports,
    validate_libero_task_assets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenarios", type=int, default=100)
    parser.add_argument("--sde-samples", type=int, default=8)
    parser.add_argument("--noise-level", type=float)
    parser.add_argument("--denoise-steps", type=int)
    return parser.parse_args()


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.blake2b(
        "\x1f".join(str(part) for part in parts).encode(),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


def _seed_all(seed: int) -> None:
    from lerobot.utils.random_utils import set_seed
    import torch

    set_seed(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _configured_schedule(
    config: EmbodiedExperimentConfig,
    *,
    noise_level: float | None,
    denoise_steps: int | None,
) -> EmbodiedExperimentConfig:
    if config.algorithm.flow_sde is None:
        raise ValueError("action-gap diagnostic requires algorithm.flow_sde")
    updates: dict[str, Any] = {}
    if noise_level is not None:
        updates["noise_level"] = float(noise_level)
    if denoise_steps is not None:
        updates["num_denoise_steps"] = int(denoise_steps)
    flow_sde = config.algorithm.flow_sde.model_copy(update=updates)
    algorithm = config.algorithm.model_copy(update={"flow_sde": flow_sde})
    return config.model_copy(update={"algorithm": algorithm})


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def action_gap_summary(ode: np.ndarray, sde: np.ndarray) -> dict[str, Any]:
    """Summarize paired `[state,sample,horizon,dim]` ODE and SDE chunks."""

    if ode.ndim != 4 or sde.ndim != 4 or sde.shape != ode.shape:
        raise ValueError("paired action arrays must have identical four-dimensional shapes")
    sample_delta = sde - ode
    mean_delta = sde.mean(axis=1) - ode.mean(axis=1)
    ode_sample_std = ode.std(axis=1)
    sde_sample_std = sde.std(axis=1)
    absolute_delta = np.abs(sample_delta)

    def rms(value: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(value))))

    return {
        "states": int(ode.shape[0]),
        "sde_samples_per_state": int(sde.shape[1]),
        "horizon": int(ode.shape[2]),
        "action_dim": int(ode.shape[3]),
        "sde_mean_vs_ode_bias_rmse": rms(mean_delta),
        "paired_sde_vs_ode_rmse": rms(sample_delta),
        "ode_within_state_std_rms": rms(ode_sample_std),
        "sde_within_state_std_rms": rms(sde_sample_std),
        "absolute_sample_delta_quantiles": {
            str(quantile): float(np.quantile(absolute_delta, quantile))
            for quantile in (0.5, 0.9, 0.95, 0.99)
        },
        "per_action_dimension": [
            {
                "dimension": dimension,
                "mean_bias": float(mean_delta[..., dimension].mean()),
                "bias_rmse": rms(mean_delta[..., dimension]),
                "paired_rmse": rms(sample_delta[..., dimension]),
                "ode_sample_std_rms": rms(ode_sample_std[..., dimension]),
                "sde_sample_std_rms": rms(sde_sample_std[..., dimension]),
                "ode_saturation_rate": float(
                    (np.abs(ode[..., dimension]) >= 0.999).mean()
                ),
                "sde_saturation_rate": float(
                    (np.abs(sde[..., dimension]) >= 0.999).mean()
                ),
            }
            for dimension in range(ode.shape[-1])
        ],
    }


def main() -> None:
    args = parse_args()
    if args.scenarios < 1 or args.sde_samples < 2:
        raise ValueError("--scenarios must be positive and --sde-samples must be >= 2")
    if args.noise_level is not None and args.noise_level <= 0:
        raise ValueError("--noise-level must be positive")
    if args.denoise_steps is not None and args.denoise_steps < 1:
        raise ValueError("--denoise-steps must be positive")

    config = _configured_schedule(
        EmbodiedExperimentConfig.from_yaml(args.config),
        noise_level=args.noise_level,
        denoise_steps=args.denoise_steps,
    )
    if config.policy.type not in {"pi0", "pi05"}:
        raise ValueError("action-gap diagnostic requires a PI0 or PI0.5 policy")
    require_compatible_runtime(profile=runtime_profile_for_policy(config.policy.type))
    prepare_libero_runtime_paths()
    validate_libero_runtime_imports()
    validate_libero_task_assets(LiberoSettings.from_config(config))

    scenarios = build_evaluation_scenarios(config)[: args.scenarios]
    if len(scenarios) != args.scenarios:
        raise ValueError("the evaluation plan contains fewer requested scenarios")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    policy = make_policy(config)
    policy.eval()
    assert policy.preprocessor is not None
    assert policy.postprocessor is not None
    assert policy.schedule is not None
    catalog = LiberoTaskCatalog(LiberoSettings.from_config(config))
    normalized_ode = []
    normalized_sde = []
    native_ode = []
    native_sde = []
    scenario_ids = []

    import torch

    with torch.inference_mode():
        for scenario_index, scenario in enumerate(scenarios):
            environment_seed = _stable_seed(scenario.id, "action-gap-environment")
            context = RolloutContext(
                update=0,
                group_index=scenario_index,
                attempt_index=0,
                environment_seed=environment_seed,
                policy_seed=0,
                config_fingerprint="pi-flow-sde-action-gap",
            )
            env = catalog.make_environment(scenario, context)
            try:
                observation, _ = env.reset(
                    seed=environment_seed,
                    options=scenario.payload.get("reset_options"),
                )
            finally:
                env.close()
            prepared = _prepare_pi_observation(
                observation,
                policy=policy,
                task=scenario.task,
                robot_type="panda",
            )
            processed = policy.preprocessor(prepared)

            state_normalized_ode = []
            state_normalized_sde = []
            state_native_ode = []
            state_native_sde = []
            for sample_index in range(args.sde_samples):
                sample_seed = _stable_seed(scenario.id, sample_index, "action-gap")
                _seed_all(sample_seed)
                policy.reset()
                ode_normalized = policy.predict_native_action_chunk(processed)
                ode_native = _postprocess_pi_actions(
                    policy,
                    ode_normalized,
                    prepared_rows=[prepared],
                )

                _seed_all(sample_seed)
                policy.reset()
                selected_index = random.randint(0, policy.schedule.num_steps - 1)
                selected_indices = torch.full(
                    (1,), selected_index, dtype=torch.long, device=policy.device
                )
                rollout = policy.sample_flow_sde(
                    processed,
                    selected_index=selected_indices,
                )
                state_normalized_ode.append(_to_numpy(ode_normalized[0]))
                state_normalized_sde.append(_to_numpy(rollout.actions[0]))
                state_native_ode.append(_to_numpy(ode_native[0]))
                state_native_sde.append(
                    _to_numpy(
                        _postprocess_pi_actions(
                            policy,
                            rollout.actions,
                            prepared_rows=[prepared],
                        )[0]
                    )
                )

            horizon = int(policy.execution_horizon)
            action_dim = int(policy.action_dim)
            normalized_ode.append(
                np.asarray(state_normalized_ode)[:, :horizon, :action_dim]
            )
            native_ode.append(np.asarray(state_native_ode)[:, :horizon, :action_dim])
            normalized_sde.append(
                np.asarray(state_normalized_sde)[:, :horizon, :action_dim]
            )
            native_sde.append(np.asarray(state_native_sde)[:, :horizon, :action_dim])
            scenario_ids.append(scenario.id)
            if (scenario_index + 1) % 10 == 0:
                print(
                    f"[action-gap] completed={scenario_index + 1}/{len(scenarios)}",
                    flush=True,
                )

    arrays = {
        "normalized_ode": np.asarray(normalized_ode),
        "normalized_sde": np.asarray(normalized_sde),
        "native_ode": np.asarray(native_ode),
        "native_sde": np.asarray(native_sde),
    }
    np.savez_compressed(output_dir / "action-chunks.npz", **arrays)
    assert config.algorithm.flow_sde is not None
    report = {
        "schema_version": 1,
        "config": str(args.config.expanduser().resolve()),
        "policy_type": config.policy.type,
        "policy_path": config.policy.path,
        "noise_level": config.algorithm.flow_sde.noise_level,
        "denoise_steps": config.algorithm.flow_sde.num_denoise_steps,
        "scenario_ids": scenario_ids,
        "normalized_action_gap": action_gap_summary(
            arrays["normalized_ode"], arrays["normalized_sde"]
        ),
        "native_action_gap": action_gap_summary(
            arrays["native_ode"], arrays["native_sde"]
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (output_dir / "action-gap.json").write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
