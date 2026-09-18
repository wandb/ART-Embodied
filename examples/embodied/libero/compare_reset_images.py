"""Measure paired reset image differences; thresholds are screening heuristics."""

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_reset import write_report
from .compare_reset_audits import compare
from .state_manifest import file_sha256


def image_metrics(left, right):
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] != 3 or a.size == 0:
        raise ValueError("Expected equally sized nonempty RGB images")
    if a.dtype != np.uint8 or b.dtype != np.uint8:
        raise ValueError("Expected uint8 image arrays")
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "mae_0_255": float(delta.mean()),
        "fraction_pixels_any_channel_delta_gt_10": float(
            (delta.max(axis=-1) > 10).mean()
        ),
        "review_needed": bool(delta.mean() > 5),
    }


def compare_images(left_path, right_path):
    from PIL import Image

    left = json.loads(left_path.read_text())
    right = json.loads(right_path.read_text())
    physics = compare(left, right)
    paired = {(t["suite"], t["task_id"]): t for t in right["tasks"]}
    rows = []
    for a in left["tasks"]:
        b = paired[(a["suite"], a["task_id"])]
        if "image" not in a or "image" not in b:
            raise ValueError("Both reports need --render images for every task")
        paths = [
            (root.parent / task["image"]).resolve()
            for root, task in ((left_path, a), (right_path, b))
        ]
        for root, path in zip((left_path, right_path), paths, strict=True):
            if path.parent != root.parent.resolve():
                raise ValueError("Image must be beside its report")
        with Image.open(paths[0]) as x, Image.open(paths[1]) as y:
            metrics = image_metrics(
                np.asarray(x.convert("RGB")), np.asarray(y.convert("RGB"))
            )
        rows.append(
            {
                "suite": a["suite"],
                "task_id": a["task_id"],
                **metrics,
                "left_image_sha256": file_sha256(paths[0]),
                "right_image_sha256": file_sha256(paths[1]),
            }
        )
    return {
        "scope": __doc__,
        "physics_comparison": physics,
        "images": rows,
        "review_images": sum(r["review_needed"] for r in rows),
        "policy_performance_measured": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("left", "right", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    args = p.parse_args()
    result = compare_images(args.left, args.right)
    write_report(args.output, result)
    print(json.dumps({"review_images": result["review_images"]}))
