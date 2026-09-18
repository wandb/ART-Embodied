"""Compare ART Flow-SDE transitions with RLinf's source implementation.

The reference function is compiled directly from RLinf's
``OpenPi0ForRLActionPrediction.sample_mean_var_val`` AST.  This avoids both a
runtime dependency on RLinf and the weaker test of copying its equations into
another helper and comparing two copies.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import json
from pathlib import Path
from types import FunctionType
from typing import Any

import torch

from art_embodied.policies.flow_sde import FlowSDESchedule, flow_sde_transition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rlinf-source",
        type=Path,
        required=True,
        help="RLinf checkout containing openpi_action_model.py",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


@dataclass
class _ReferenceConfig:
    noise_level: float
    noise_anneal: bool = False
    noise_params: tuple[float, float, int] = (0.7, 0.3, 400)
    add_value_head: bool = False
    value_after_vlm: bool = False
    chunk_critic_input: bool = False
    detach_critic_input: bool = False
    noise_method: str = "flow_sde"


class _ReferenceModel:
    """Minimal receiver for RLinf's unmodified transition function."""

    def __init__(self, *, velocity: torch.Tensor, noise_level: float) -> None:
        self.config = _ReferenceConfig(noise_level=noise_level)
        self.global_step = 0
        self._velocity = velocity
        self.action_out_proj = torch.nn.Identity()

    def get_suffix_out(self, *_args: Any, **_kwargs: Any) -> torch.Tensor:
        return self._velocity


def _load_reference_function(rlinf_source: Path) -> FunctionType:
    source = rlinf_source.expanduser().resolve()
    model_path = source / "rlinf/models/embodiment/openpi_action_model.py"
    if not model_path.is_file():
        raise FileNotFoundError(f"RLinf OpenPI model source not found: {model_path}")
    tree = ast.parse(model_path.read_text(encoding="utf-8"), filename=str(model_path))
    function = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "OpenPi0ForRLActionPrediction":
            function = next(
                (
                    item
                    for item in node.body
                    if isinstance(item, ast.FunctionDef)
                    and item.name == "sample_mean_var_val"
                ),
                None,
            )
            break
    if function is None:
        raise RuntimeError("RLinf sample_mean_var_val function was not found")
    function.decorator_list = []
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {"torch": torch}
    exec(compile(module, str(model_path), "exec"), namespace)  # noqa: S102
    return namespace["sample_mean_var_val"]


def run(rlinf_source: Path) -> dict[str, Any]:
    reference = _load_reference_function(rlinf_source)
    generator = torch.Generator(device="cpu").manual_seed(20260721)
    conditions = []
    maxima = {"mean": 0.0, "std": 0.0}
    for num_steps in (3, 4, 8, 16):
        for noise_level in (0.3, 0.5):
            schedule = FlowSDESchedule(num_steps=num_steps, noise_level=noise_level)
            for step in range(num_steps):
                x_t = torch.randn((3, 5, 7), generator=generator)
                velocity = torch.randn((3, 5, 7), generator=generator)
                receiver = _ReferenceModel(
                    velocity=velocity,
                    noise_level=noise_level,
                )
                reference_mean, reference_std, _ = reference(
                    receiver,
                    x_t,
                    step,
                    torch.zeros((3, 1)),
                    torch.zeros((3, 1), dtype=torch.bool),
                    None,
                    "train",
                    num_steps,
                    False,
                )
                candidate = flow_sde_transition(
                    x_t,
                    velocity,
                    step,
                    schedule,
                    stochastic=True,
                )
                mean_delta = float((candidate.mean - reference_mean).abs().max())
                std_delta = float((candidate.std - reference_std).abs().max())
                maxima["mean"] = max(maxima["mean"], mean_delta)
                maxima["std"] = max(maxima["std"], std_delta)
                conditions.append(
                    {
                        "num_steps": num_steps,
                        "noise_level": noise_level,
                        "step": step,
                        "mean_max_abs_delta": mean_delta,
                        "std_max_abs_delta": std_delta,
                    }
                )
    tolerance = 1.0e-7
    return {
        "schema_version": 1,
        "status": "ok" if max(maxima.values()) <= tolerance else "mismatch",
        "reference": str(
            rlinf_source.expanduser().resolve()
            / "rlinf/models/embodiment/openpi_action_model.py"
        ),
        "reference_function": "OpenPi0ForRLActionPrediction.sample_mean_var_val",
        "tolerance": tolerance,
        "max_deltas": maxima,
        "conditions": conditions,
    }


def main() -> None:
    args = parse_args()
    result = run(args.rlinf_source)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
