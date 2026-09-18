"""Media capture and integration helpers for ART-Embodied.

The core trajectory model stores media as ``MediaRef`` objects. This module is
where we bridge simulator renders to local files and, optionally, W&B/Weave SDK
objects. The split is deliberate: rollout code should be able to save videos in
Slurm smoke tests without importing networked logging SDKs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .lookahead import LookaheadFrame
from .trajectories import EmbodiedTrajectory, MediaRef

_VIDEO_FORMAT_MIME_TYPES = {
    "gif": "image/gif",
    "mp4": "video/mp4",
    "webm": "video/webm",
}
_WANDB_VIDEO_FORMATS = {"gif", "mp4", "webm", "ogg"}


class RolloutVideoRecorder:
    """Capture ``env.render()`` frames and attach a video ``MediaRef``.

    The first implementation intentionally writes GIFs using Pillow because GIF
    support is available in lightweight CPU environments and is accepted by both
    W&B ``wandb.Video`` and Weave's video handling. MP4/WebM can be added later
    through optional encoders without changing the trajectory contract.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        filename_prefix: str = "rollout",
        fps: int = 4,
        max_frames: int = 300,
        video_format: str = "gif",
        capture_every: int = 1,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if capture_every <= 0:
            raise ValueError("capture_every must be positive")
        normalized_format = video_format.lower().lstrip(".")
        if normalized_format != "gif":
            raise ValueError(
                "RolloutVideoRecorder currently writes GIF files only. "
                "Use video_format='gif' or add an explicit encoder adapter."
            )
        self.output_dir = Path(output_dir)
        self.filename_prefix = _safe_filename_prefix(filename_prefix)
        self.fps = int(fps)
        self.max_frames = int(max_frames)
        self.video_format = normalized_format
        self.capture_every = int(capture_every)
        self._frames: list[tuple[int | None, Any]] = []
        self._errors: list[dict[str, Any]] = []

    @property
    def frames_captured(self) -> int:
        return len(self._frames)

    def capture(self, env: Any, *, step: int | None = None) -> None:
        """Try to append the current render frame.

        Rendering failures are recorded as metadata instead of interrupting a
        rollout. That keeps the RL path robust while making missing videos
        diagnosable in traces and reports.
        """

        if len(self._frames) >= self.max_frames:
            return
        if step is not None and step % self.capture_every != 0:
            return
        if not hasattr(env, "render"):
            self._errors.append({"step": step, "error": "environment has no render()"})
            return
        try:
            frame = env.render()
            self.capture_frame(frame, step=step)
        except Exception as exc:  # pragma: no cover - exact simulator errors vary.
            self._errors.append({"step": step, "error": repr(exc)})

    def capture_frame(self, frame: Any, *, step: int | None = None) -> None:
        """Append an already rendered frame under the recorder's bounds."""

        if len(self._frames) >= self.max_frames:
            return
        if step is not None and step % self.capture_every != 0:
            return
        image = _frame_to_pil_image(frame)
        if image is None:
            self._errors.append({"step": step, "error": "render() returned None"})
            return
        self._frames.append((step, image))

    def capture_error(self, error: BaseException | str, *, step: int | None) -> None:
        """Record a non-fatal media error without interrupting the rollout."""

        value = error if isinstance(error, str) else repr(error)
        self._errors.append({"step": step, "error": value})

    def finalize(self, trajectory: EmbodiedTrajectory | None = None) -> list[MediaRef]:
        """Write a video file and return media refs to attach to a trajectory."""

        if not self._frames:
            if not self._errors:
                return []
            return [
                MediaRef(
                    uri="memory://rollout-video/error",
                    kind="custom",
                    caption="rollout video capture failed",
                    metadata={"errors": self._errors},
                )
            ]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = _unique_path(
            self.output_dir / f"{self.filename_prefix}.{self.video_format}"
        )
        duration_ms = max(1, int(1000 / self.fps))
        images = [image for _, image in self._frames]
        first, *rest = images
        first.save(
            output_path,
            save_all=True,
            append_images=rest,
            duration=duration_ms,
            loop=0,
            format="GIF",
        )
        steps = [step for step, _ in self._frames]
        metadata: dict[str, Any] = {
            "format": self.video_format,
            "fps": self.fps,
            "frames": len(self._frames),
            "capture_every": self.capture_every,
            "max_frames": self.max_frames,
            "path": str(output_path),
            "steps": steps,
            "role": "simulation",
        }
        if self._errors:
            metadata["capture_errors"] = self._errors
        if trajectory is not None:
            metadata["task"] = trajectory.task
            metadata["scenario_id"] = trajectory.metadata.get("scenario_id")
            metadata["environment_seed"] = trajectory.metadata.get("environment_seed")
            metadata["policy_seed"] = trajectory.metadata.get("policy_seed")
            metadata["success"] = trajectory.metrics.get("success")
        return [
            MediaRef(
                uri=output_path.resolve().as_uri(),
                kind="video",
                step=steps[-1] if steps else None,
                mime_type=_VIDEO_FORMAT_MIME_TYPES[self.video_format],
                caption=_video_caption(trajectory),
                metadata=metadata,
            )
        ]


class LookaheadPreviewRecorder(RolloutVideoRecorder):
    """Record a time-ordered filmstrip of counterfactual future renders."""

    def __init__(
        self,
        *args: Any,
        max_panels: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if max_panels < 1:
            raise ValueError("max_panels must be positive")
        self.max_panels = int(max_panels)

    def capture_lookahead(
        self,
        current_frame: Any,
        future_frames: list[LookaheadFrame],
        *,
        step: int,
        execution_horizon: int,
        model_horizon: int,
    ) -> None:
        if not future_frames:
            return
        preview = compose_lookahead_filmstrip(
            current_frame,
            future_frames,
            execution_horizon=execution_horizon,
            model_horizon=model_horizon,
            max_panels=self.max_panels,
        )
        self.capture_frame(preview, step=step)

    def finalize(self, trajectory: EmbodiedTrajectory | None = None) -> list[MediaRef]:
        refs = super().finalize(trajectory)
        for ref in refs:
            ref.metadata["role"] = "lookahead_preview"
            ref.metadata["semantics"] = "simulated_unexecuted_action_chunk_tail"
            ref.metadata["experimental"] = True
            ref.metadata["display"] = "current_plus_future_filmstrip"
            ref.metadata["max_panels"] = self.max_panels
            if ref.caption:
                ref.caption = f"lookahead preview | {ref.caption}"
        return refs


def compose_lookahead_filmstrip(
    current_frame: Any,
    future_frames: list[LookaheadFrame],
    *,
    execution_horizon: int,
    model_horizon: int,
    max_panels: int = 4,
) -> Any:
    """Place the current render beside representative speculative futures.

    Full-frame alpha compositing makes a moving robot and manipulated objects
    appear as undifferentiated ghosts. A filmstrip preserves temporal order and
    lets a human compare concrete simulated states without implying that the
    unexecuted futures occurred in the rollout.
    """

    from PIL import Image, ImageDraw

    current_image = _frame_to_pil_image(current_frame)
    if current_image is None:
        raise ValueError("Lookahead preview requires a non-empty current frame")
    if not future_frames:
        raise ValueError("Lookahead preview requires at least one future frame")
    if max_panels < 1:
        raise ValueError("max_panels must be positive")

    selected = [future_frames[index] for index in _panel_indices(len(future_frames), max_panels)]
    width, height = current_image.size
    columns = 2 if len(selected) > 1 else 1
    rows = (len(selected) + columns - 1) // columns
    panel_width = width // columns
    panel_height = height // rows
    canvas = Image.new("RGB", (width * 2, height), color=(18, 18, 18))
    canvas.paste(current_image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    _draw_panel_label(draw, "CURRENT (executed rollout)", x=0, y=0)

    for panel_index, frame in enumerate(selected):
        future_image = _frame_to_pil_image(frame.image)
        if future_image is None:
            raise ValueError("Lookahead preview requires non-empty future frames")
        resized = future_image.resize((panel_width, panel_height), Image.Resampling.BILINEAR)
        column = panel_index % columns
        row = panel_index // columns
        x = width + column * panel_width
        y = row * panel_height
        canvas.paste(resized, (x, y))
        _draw_panel_label(
            draw,
            f"UNEXECUTED action {frame.action_index}/{model_horizon - 1}",
            x=x,
            y=y,
        )
        draw.rectangle(
            (x, y, x + panel_width - 1, y + panel_height - 1),
            outline=(245, 180, 35),
            width=2,
        )

    _draw_panel_label(
        draw,
        f"predicted tail starts at action {execution_horizon}",
        x=width,
        y=height - 18,
    )
    return canvas


def _panel_indices(frame_count: int, max_panels: int) -> list[int]:
    """Choose evenly spaced frames while always retaining both endpoints."""

    if frame_count <= max_panels:
        return list(range(frame_count))
    if max_panels == 1:
        return [frame_count - 1]
    return [
        round(index * (frame_count - 1) / (max_panels - 1))
        for index in range(max_panels)
    ]


def _draw_panel_label(draw: Any, label: str, *, x: int, y: int) -> None:
    bounds = draw.textbbox((0, 0), label)
    draw.rectangle(
        (x + 3, y + 3, x + bounds[2] + 9, y + bounds[3] + 9),
        fill=(0, 0, 0),
    )
    draw.text((x + 6, y + 6), label, fill=(255, 255, 255))


def media_ref_local_path(media: MediaRef) -> Path | None:
    """Resolve a file-backed media ref to a local path when possible."""

    parsed = urlparse(media.uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if parsed.scheme:
        return None
    return Path(media.uri)


def _video_caption(trajectory: EmbodiedTrajectory | None) -> str:
    if trajectory is None:
        return "rollout video"
    success = trajectory.metrics.get("success")
    outcome = "unknown"
    if success is not None:
        outcome = "success" if bool(success) else "failure"
    values = [f"task={trajectory.task}", f"outcome={outcome}"]
    for key in ("scenario_id", "environment_seed", "policy_seed"):
        value = trajectory.metadata.get(key)
        if value is not None:
            values.append(f"{key}={value}")
    return " | ".join(values)


def local_video_media_refs(
    trajectory: EmbodiedTrajectory,
    *,
    role: str | None = None,
) -> list[tuple[MediaRef, Path]]:
    """Return local file-backed video refs that still exist on disk."""

    refs: list[tuple[MediaRef, Path]] = []
    for media in trajectory.media:
        if media.kind != "video":
            continue
        media_role = str(media.metadata.get("role", "simulation"))
        if role is not None and media_role != role:
            continue
        path = media_ref_local_path(media)
        if path is not None and path.is_file():
            refs.append((media, path))
    return refs


def wandb_video_payload(
    trajectory: EmbodiedTrajectory,
    *,
    prefix: str = "embodied/video",
    max_videos: int = 4,
    start_index: int = 0,
    wandb_module: Any | None = None,
    media_role: str | None = None,
) -> dict[str, Any]:
    """Build a W&B payload with ``wandb.Video`` objects for local videos.

    W&B documents videos as the ``wandb.Video`` data type. Importing W&B only
    here keeps normal rollout and test paths SDK-free.
    """

    entries = []
    for media, path in local_video_media_refs(trajectory, role=media_role):
        entries.append(
            {
                "path": str(path),
                "caption": media.caption,
                "fps": media.metadata.get("fps"),
                "format": media.metadata.get("format"),
            }
        )
    return wandb_video_payload_from_paths(
        entries,
        prefix=prefix,
        max_videos=max_videos,
        start_index=start_index,
        wandb_module=wandb_module,
    )


def wandb_video_payload_from_paths(
    video_entries: list[dict[str, Any]],
    *,
    prefix: str = "embodied/video",
    max_videos: int = 4,
    start_index: int = 0,
    wandb_module: Any | None = None,
) -> dict[str, Any]:
    """Build a W&B video payload from report/discovery video entries."""

    if max_videos <= 0:
        return {}

    candidates: list[tuple[dict[str, Any], Path, str]] = []
    for entry in video_entries:
        if len(candidates) >= max_videos:
            break
        raw_path = entry.get("path") or entry.get("uri")
        if not raw_path:
            continue
        path = Path(str(raw_path))
        if not path.is_file():
            continue
        video_format = str(
            entry.get("format") or path.suffix.lstrip(".") or "gif"
        ).lower()
        if video_format not in _WANDB_VIDEO_FORMATS:
            video_format = "gif"
        candidates.append((entry, path, video_format))

    if not candidates:
        return {}
    if wandb_module is None:
        import importlib

        wandb_module = importlib.import_module("wandb")
    payload: dict[str, Any] = {}
    for logged, (entry, path, video_format) in enumerate(candidates):
        # W&B reads frame timing from file-backed videos. Passing ``fps`` with
        # a path is ignored by the SDK and emits one warning per video. The
        # recorder already encodes GIF timing, so keep launch logs quiet.
        payload[f"{prefix}/{start_index + logged}"] = wandb_module.Video(
            str(path),
            caption=entry.get("caption"),
            format=video_format,
        )
    return payload


def weave_video_content_paths(
    trajectory: EmbodiedTrajectory,
    *,
    prefix: str = "embodied/video",
    max_videos: int = 4,
) -> dict[str, str]:
    """Return local video paths for Weave file-backed ``Content`` ops.

    Returning paths here lets callers create a tiny ``@weave.op`` that returns
    ``Content.from_path(path, metadata=...)`` without importing Weave in ART
    core or serializing GIF/MP4 bytes into JSON traces.
    """

    if max_videos <= 0:
        return {}
    return {
        f"{prefix}/{index}": str(path)
        for index, (_, path) in enumerate(
            local_video_media_refs(trajectory)[:max_videos]
        )
    }


def weave_video_content_paths_from_entries(
    video_entries: list[dict[str, Any]],
    *,
    prefix: str = "embodied/video",
    max_videos: int = 4,
) -> dict[str, str]:
    """Return local video paths from report/discovery entries for Weave ops."""

    if max_videos <= 0:
        return {}
    payload: dict[str, str] = {}
    logged = 0
    for entry in video_entries:
        if logged >= max_videos:
            break
        raw_path = entry.get("path") or entry.get("uri")
        if not raw_path:
            continue
        path = Path(str(raw_path))
        if not path.is_file():
            continue
        payload[f"{prefix}/{logged}"] = str(path)
        logged += 1
    return payload


def _frame_to_pil_image(frame: Any):
    if frame is None:
        return None
    import numpy as np
    from PIL import Image

    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu()
    if hasattr(frame, "numpy"):
        frame = frame.numpy()
    array = np.asarray(frame)
    if array.size == 0:
        return None
    if (
        array.ndim == 3
        and array.shape[0] in (1, 3, 4)
        and array.shape[-1] not in (1, 3, 4)
    ):
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 4:
        array = array[0]
    if array.dtype.kind == "f":
        max_value = float(np.nanmax(array)) if array.size else 0.0
        if max_value <= 1.0:
            array = array * 255.0
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    array = np.clip(array, 0, 255).astype("uint8")
    if array.ndim == 2:
        return Image.fromarray(array, mode="L").convert("RGB")
    if array.ndim == 3 and array.shape[-1] == 1:
        return Image.fromarray(array[..., 0], mode="L").convert("RGB")
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        return Image.fromarray(array).convert("RGB")
    raise ValueError(f"unsupported render frame shape: {getattr(array, 'shape', None)}")


def _safe_filename_prefix(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value).strip(
        ".-"
    )
    return cleaned or "rollout"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 10_000):
        candidate = path.with_name(f"{stem}-{index:04d}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not allocate a unique media path under {path.parent}")
