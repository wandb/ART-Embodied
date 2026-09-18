"""Compare tensor bundles emitted by ``pi_cross_runtime_dump.py``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--art", type=Path, required=True)
    parser.add_argument("--rlinf", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def tensor_delta(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.shape != right.shape:
        return {
            "shape_match": False,
            "art_shape": list(left.shape),
            "rlinf_shape": list(right.shape),
        }
    left_float = left.float()
    right_float = right.float()
    delta = left_float - right_float
    return {
        "shape_match": True,
        "shape": list(left.shape),
        "dtype_match": left.dtype == right.dtype,
        "art_dtype": str(left.dtype),
        "rlinf_dtype": str(right.dtype),
        "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
        "mean_abs": float(delta.abs().mean()) if delta.numel() else 0.0,
        "rmse": float(delta.square().mean().sqrt()) if delta.numel() else 0.0,
    }


def compare_bundles(art: dict[str, Any], rlinf: dict[str, Any]) -> dict[str, Any]:
    if art.get("schema_version") != 1 or rlinf.get("schema_version") != 1:
        raise ValueError("unsupported PI runtime bundle schema")
    results: dict[str, Any] = {}
    for key in (
        "language_tokens",
        "language_masks",
        "model_state",
        "initial_noise",
        "flow_states",
        "velocities",
        "normalized_actions",
        "native_actions",
    ):
        left, right = art[key], rlinf[key]
        if left is None or right is None:
            results[key] = {"both_none": left is None and right is None}
        else:
            results[key] = tensor_delta(left, right)
    art_native = art["native_actions"]
    rlinf_native = rlinf["native_actions"]
    if (
        art_native.ndim == 3
        and rlinf_native.ndim == 3
        and art_native.shape[0] == rlinf_native.shape[0]
        and art_native.shape[2] == rlinf_native.shape[2]
    ):
        # RLinf applies its configured execution horizon in output_transform;
        # ART retains the full model horizon until the rollout adapter boundary.
        # The common prefix is the action sequence actually sent to LIBERO.
        execution_horizon = min(art_native.shape[1], rlinf_native.shape[1])
        results["native_actions_executed"] = tensor_delta(
            art_native[:, :execution_horizon],
            rlinf_native[:, :execution_horizon],
        )
    for key in ("images", "image_masks"):
        left_values, right_values = art[key], rlinf[key]
        results[key] = {
            "count_match": len(left_values) == len(right_values),
            "art_count": len(left_values),
            "rlinf_count": len(right_values),
            "items": [
                tensor_delta(left, right)
                for left, right in zip(left_values, right_values, strict=False)
            ],
        }
    art_intermediates = art.get("intermediates")
    rlinf_intermediates = rlinf.get("intermediates")
    if art_intermediates is not None and rlinf_intermediates is not None:
        if set(art_intermediates) != set(rlinf_intermediates):
            raise ValueError("PI runtime intermediate keys do not match")
        compared = {}
        for key, left in art_intermediates.items():
            right = rlinf_intermediates[key]
            if isinstance(left, list) and isinstance(right, list):
                compared[key] = {
                    "count_match": len(left) == len(right),
                    "items": [
                        tensor_delta(a, b)
                        for a, b in zip(left, right, strict=False)
                    ],
                }
            elif isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
                compared[key] = tensor_delta(left, right)
            else:
                raise TypeError(
                    f"Unsupported PI intermediate type for {key!r}: "
                    f"{type(left).__name__}, {type(right).__name__}"
                )
        results["intermediates"] = compared
    return {"schema_version": 1, "comparisons": results}


def main() -> None:
    args = parse_args()
    art = torch.load(args.art, map_location="cpu", weights_only=True)
    rlinf = torch.load(args.rlinf, map_location="cpu", weights_only=True)
    result = compare_bundles(art, rlinf)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
