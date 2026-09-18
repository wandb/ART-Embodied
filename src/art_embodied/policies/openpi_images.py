"""OpenPI-compatible image transforms for imported PI checkpoints."""

from __future__ import annotations

import numpy as np


def resize_openpi_uint8_image(
    image: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Match OpenPI's JAX linear resize and centered zero padding for RGB images.

    OpenPI applies antialiased ``jax.image.resize(..., method="linear")`` to
    uint8 HWC images, rounds the result back to uint8, and only then normalizes
    it for SigLIP. Reproduce JAX's half-pixel triangular weights directly so
    imported checkpoints do not require JAX at runtime.
    """

    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] not in {1, 3, 4}:
        raise ValueError(
            "OpenPI image input must be HWC with 1, 3, or 4 channels; "
            f"got {value.shape}"
        )
    if value.dtype != np.uint8:
        raise ValueError(
            f"OpenPI image input must be uint8; got {value.dtype}"
        )
    if height < 1 or width < 1:
        raise ValueError("OpenPI image target dimensions must be positive")
    current_height, current_width = value.shape[:2]
    if (current_height, current_width) == (height, width):
        return np.ascontiguousarray(value)

    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    if resized_height < 1 or resized_width < 1:
        raise ValueError(
            "OpenPI resize produced an empty image; "
            f"input={value.shape}, target={(height, width)}"
        )

    resized = _jax_linear_resize_uint8(
        value,
        height=resized_height,
        width=resized_width,
    )
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w1 = pad_w0 + remainder_w
    padded = np.pad(
        resized,
        ((pad_h0, pad_h1), (pad_w0, pad_w1), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    return np.ascontiguousarray(padded)


def _jax_linear_resize_uint8(
    image: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Apply JAX 0.5's separable antialiased triangle resize."""

    height_weights = _jax_triangle_weights(image.shape[0], height)
    width_weights = _jax_triangle_weights(image.shape[1], width)
    image_float = image.astype(np.float32)
    resized_height = np.einsum(
        "hwc,hi->iwc",
        image_float,
        height_weights,
        dtype=np.float32,
        optimize=True,
    )
    resized = np.einsum(
        "iwc,wj->ijc",
        resized_height,
        width_weights,
        dtype=np.float32,
        optimize=True,
    )
    return np.rint(resized).clip(0, 255).astype(np.uint8)


def _jax_triangle_weights(input_size: int, output_size: int) -> np.ndarray:
    scale = np.float32(output_size / input_size)
    inverse_scale = np.float32(1.0) / scale
    kernel_scale = np.maximum(inverse_scale, np.float32(1.0))
    sample = (
        (np.arange(output_size, dtype=np.float32) + np.float32(0.5))
        * inverse_scale
        - np.float32(0.5)
    )
    distance = np.abs(
        sample[np.newaxis, :]
        - np.arange(input_size, dtype=np.float32)[:, np.newaxis]
    ) / kernel_scale
    weights = np.maximum(np.float32(0.0), np.float32(1.0) - distance)
    total = weights.sum(axis=0, keepdims=True, dtype=np.float32)
    threshold = np.float32(1000.0 * np.finfo(np.float32).eps)
    return np.where(
        np.abs(total) > threshold,
        weights / np.where(total != 0, total, np.float32(1.0)),
        np.float32(0.0),
    ).astype(np.float32)
