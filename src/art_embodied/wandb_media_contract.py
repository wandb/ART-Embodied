"""Read-only verification of numbered videos attached to exact policy updates.

This checks W&B history attachments and downloaded bytes, not App rendering.
It neither changes history nor controls a running experiment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class VideoExpectation:
    update: int
    prefix: str
    count: int

    def __post_init__(self) -> None:
        if type(self.update) is not int or self.update < 0:
            raise ValueError("update must be a nonnegative integer")
        if type(self.count) is not int or self.count < 1:
            raise ValueError("count must be a positive integer")
        if (
            not isinstance(self.prefix, str)
            or not self.prefix
            or self.prefix.endswith("/")
        ):
            raise ValueError("prefix must be nonempty and have no trailing slash")


def _decode_video(path: Path) -> dict[str, int]:
    import av

    frames = 0
    width = height = 0
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames += 1
            width, height = frame.width, frame.height
    if frames == 0 or width <= 0 or height <= 0:
        raise ValueError("Video has no decodable frames")
    return {"frames": frames, "width": width, "height": height}


def _video_metadata(value: Any) -> tuple[str, str]:
    if not isinstance(value, dict) or value.get("_type") != "video-file":
        raise ValueError("Expected W&B video-file metadata")
    name, digest = value.get("path"), value.get("sha256")
    if not isinstance(name, str) or "\\" in name:
        raise ValueError("Invalid video path")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or str(path) != name
        or path.parts[:2] != ("media", "videos")
        or len(path.parts) < 3
    ):
        raise ValueError("Video must have a normalized relative media/videos path")
    if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
        raise ValueError("Missing or invalid video SHA-256")
    return name, digest


def verify_run_videos(
    run: Any,
    expectations: Iterable[VideoExpectation],
    *,
    download_root: Path,
    decode_video: Callable[[Path], dict[str, int]] = _decode_video,
    require_native_steps: bool = False,
) -> dict[str, Any]:
    expectations = tuple(expectations)
    if not expectations:
        raise ValueError("At least one video expectation is required")
    identities = [(item.update, item.prefix) for item in expectations]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate update/prefix expectation")

    attachments: dict[tuple[int, str], dict[str, list[Any]]] = {
        identity: {} for identity in identities
    }
    native_errors: dict[tuple[int, str], list[dict[str, Any]]] = {
        identity: [] for identity in identities
    }
    # Scan once without intersecting optional keys from distinct history rows.
    for row in run.scan_history():
        update = row.get("experiment/update")
        if isinstance(update, bool) or not isinstance(update, (int, float)):
            continue
        for item in expectations:
            if update != item.update:
                continue
            target = attachments[(item.update, item.prefix)]
            for key, value in row.items():
                if key.startswith(item.prefix + "/") and value is not None:
                    target.setdefault(key, []).append(value)
                    if require_native_steps and row.get("_step") != update:
                        native_errors[(item.update, item.prefix)].append(
                            {
                                "kind": "native_step_mismatch",
                                "key": key,
                                "native_step": row.get("_step"),
                                "update": update,
                            }
                        )

    download_root = download_root.resolve()
    download_root.mkdir(parents=True, exist_ok=True)
    results = []
    for item in expectations:
        found = attachments[(item.update, item.prefix)]
        expected = {f"{item.prefix}/{index}" for index in range(item.count)}
        errors: list[dict[str, Any]] = list(native_errors[(item.update, item.prefix)])
        checked = []
        for key in sorted(set(found) - expected):
            errors.append({"key": key, "kind": "unexpected_attachment"})
        for key in sorted(expected):
            values = found.get(key, [])
            if len(values) != 1:
                errors.append(
                    {"key": key, "kind": "missing" if not values else "duplicate"}
                )
                continue
            try:
                name, digest = _video_metadata(values[0])
                destination = (download_root / name).resolve()
                if not destination.is_relative_to(download_root):
                    raise ValueError("Video destination escapes download root")
                handle = run.file(name).download(root=str(download_root), replace=True)
                path = Path(handle.name).resolve()
                handle.close()
                if path != destination:
                    raise ValueError("Downloaded video path differs from metadata")
                size = values[0].get("size")
                # Historical API pages may encode integral byte counts as floats.
                if size is not None and (
                    isinstance(size, bool)
                    or not isinstance(size, (int, float))
                    or size != path.stat().st_size
                ):
                    raise ValueError("Video size differs from history metadata")
                with path.open("rb") as file:
                    hasher = hashlib.sha256()
                    for chunk in iter(lambda: file.read(1024 * 1024), b""):
                        hasher.update(chunk)
                if hasher.hexdigest() != digest:
                    raise ValueError("Video SHA-256 differs from history metadata")
                decoded = decode_video(path)
                if any(
                    type(decoded.get(field)) is not int or decoded[field] <= 0
                    for field in ("frames", "width", "height")
                ):
                    raise ValueError("Invalid video decode result")
                checked.append({"key": key, "path": name, "sha256": digest, **decoded})
            except Exception as exc:
                errors.append({"key": key, "kind": "invalid", "error": repr(exc)})
        results.append(
            {
                "update": item.update,
                "prefix": item.prefix,
                "expected_count": item.count,
                "videos": checked,
                "errors": errors,
                "verified": not errors,
            }
        )
    return {
        "run_url": run.url,
        "media_verified": all(result["verified"] for result in results),
        "app_rendering_verified": False,
        "expectations": results,
    }


def _parse_expectation(value: str) -> VideoExpectation:
    try:
        update, remainder = value.split(":", 1)
        prefix, count = remainder.rsplit(":", 1)
        return VideoExpectation(int(update), prefix, int(count))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use UPDATE:METRIC_PREFIX:COUNT, e.g. 5:media/simulation/eval:8"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="entity/project/run_id")
    parser.add_argument(
        "--expect", type=_parse_expectation, action="append", required=True
    )
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    import wandb

    report = verify_run_videos(
        wandb.Api(timeout=30).run(args.run),
        args.expect,
        download_root=args.download_root,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if not report["media_verified"]:
        raise SystemExit("W&B media verification failed; see report")


if __name__ == "__main__":
    main()
