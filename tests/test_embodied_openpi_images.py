from __future__ import annotations

import numpy as np
import pytest

from art_embodied.policies.openpi_images import resize_openpi_uint8_image


def test_openpi_image_resize_preserves_aspect_ratio_and_centers_padding() -> None:
    image = np.full((2, 4, 3), 255, dtype=np.uint8)

    resized = resize_openpi_uint8_image(image, height=4, width=4)

    assert resized.shape == (4, 4, 3)
    np.testing.assert_array_equal(resized[0], np.zeros((4, 3), dtype=np.uint8))
    np.testing.assert_array_equal(resized[-1], np.zeros((4, 3), dtype=np.uint8))
    np.testing.assert_array_equal(
        resized[1:3], np.full((2, 4, 3), 255, dtype=np.uint8)
    )


def test_openpi_image_resize_returns_contiguous_identity_copy() -> None:
    source = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    view = source[:, ::-1]
    assert not view.flags.c_contiguous

    resized = resize_openpi_uint8_image(view, height=4, width=5)

    assert resized.flags.c_contiguous
    np.testing.assert_array_equal(resized, view)


@pytest.mark.parametrize(
    ("image", "message"),
    [
        (np.zeros((4, 4), dtype=np.uint8), "must be HWC"),
        (np.zeros((4, 4, 3), dtype=np.float32), "must be uint8"),
    ],
)
def test_openpi_image_resize_rejects_invalid_contract(
    image: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        resize_openpi_uint8_image(image, height=4, width=4)
