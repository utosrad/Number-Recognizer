"""Two ways of turning an arbitrary image into a 28x28 tensor.

`naive` is the path most MNIST demos use: grayscale, resize the whole frame to
28x28, divide by 255. It ignores the fact that MNIST images are not raw
drawings.

`mnist_style` reproduces the normalisation the MNIST authors applied to the
original NIST scans: isolate the ink, crop to the digit's bounding box, scale
the long side to 20 pixels with the aspect ratio preserved, paste into a 28x28
field, then translate so the centre of mass of the ink sits at the centre of
the field. Polarity is detected and corrected first, because MNIST is bright
ink on a dark background and most drawings are the other way round.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

TARGET_SIZE = 28
"""Side length of the field the model consumes."""

DIGIT_BOX = 20
"""Side length of the box the digit is scaled to fit inside the 28x28 field."""

INK_THRESHOLD = 0.2
"""Ink cutoff, applied after the image is normalised to [0, 1] with bright ink."""

ImageLike = Union[str, Path, Image.Image, np.ndarray]


def to_gray_float(image: ImageLike) -> np.ndarray:
    """Return a 2D float32 array in [0, 1]. Accepts a path, a PIL image or an array."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as handle:
            arr = np.asarray(handle.convert("L"), dtype=np.float32) / 255.0
        return arr
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0

    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D or 3D image, got shape {arr.shape}")
    arr = arr.astype(np.float32)
    if arr.max(initial=0.0) > 1.0:
        arr = arr / 255.0
    return arr


def stretch_contrast(arr: np.ndarray) -> np.ndarray:
    """Map the image's own min and max onto 0 and 1. A flat image is returned unchanged."""
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-6:
        return arr.astype(np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def background_is_light(arr: np.ndarray) -> bool:
    """True when the image is dark ink on a light background, the opposite of MNIST.

    The border ring is treated as background. It is compared against the midpoint
    of the image's own dynamic range, so the test survives contrast changes that
    leave the image confined to a narrow band of grey.
    """
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-6:
        return False
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    return float(border.mean()) > (lo + hi) / 2.0


def ensure_bright_ink(arr: np.ndarray) -> np.ndarray:
    """Normalise to [0, 1] and invert if the ink is darker than the background."""
    out = stretch_contrast(arr)
    if background_is_light(arr):
        out = 1.0 - out
    return out


def ink_bbox(arr: np.ndarray, threshold: float = INK_THRESHOLD):
    """Tightest (top, left, height, width) box containing pixels at or above `threshold`.

    Returns None when no pixel clears the threshold.
    """
    mask = arr >= threshold
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    top, bottom = int(rows[0]), int(rows[-1])
    left, right = int(cols[0]), int(cols[-1])
    return top, left, bottom - top + 1, right - left + 1


def crop_to_ink(arr: np.ndarray, threshold: float = INK_THRESHOLD) -> np.ndarray:
    """Crop to the ink bounding box. Returns the input unchanged if there is no ink."""
    box = ink_bbox(arr, threshold)
    if box is None:
        return arr
    top, left, height, width = box
    return arr[top : top + height, left : left + width]


def scale_long_side(arr: np.ndarray, box: int = DIGIT_BOX) -> np.ndarray:
    """Scale so the longer side is `box` pixels, preserving the aspect ratio.

    The shorter side is rounded to the nearest pixel and never collapses below 1.
    """
    height, width = arr.shape
    if height == 0 or width == 0:
        raise ValueError("cannot scale an empty image")
    scale = box / float(max(height, width))
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))
    resized = Image.fromarray(arr).resize((new_w, new_h), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def paste_centered(arr: np.ndarray, size: int = TARGET_SIZE) -> np.ndarray:
    """Paste `arr` into the middle of a zero-filled `size` x `size` field."""
    height, width = arr.shape
    if height > size or width > size:
        raise ValueError(f"{height}x{width} does not fit in {size}x{size}")
    field = np.zeros((size, size), dtype=np.float32)
    top = (size - height) // 2
    left = (size - width) // 2
    field[top : top + height, left : left + width] = arr
    return field


def center_of_mass(arr: np.ndarray):
    """Intensity-weighted (row, col) centroid. Returns None for an all-zero image."""
    total = float(arr.sum())
    if total <= 0.0:
        return None
    rows = np.arange(arr.shape[0], dtype=np.float64)
    cols = np.arange(arr.shape[1], dtype=np.float64)
    row = float((arr.sum(axis=1) * rows).sum() / total)
    col = float((arr.sum(axis=0) * cols).sum() / total)
    return row, col


def shift(arr: np.ndarray, d_row: int, d_col: int) -> np.ndarray:
    """Translate by whole pixels, filling vacated space with zeros."""
    out = np.zeros_like(arr)
    height, width = arr.shape
    src_r0, dst_r0 = (0, d_row) if d_row >= 0 else (-d_row, 0)
    src_c0, dst_c0 = (0, d_col) if d_col >= 0 else (-d_col, 0)
    rows = max(0, height - abs(d_row))
    cols = max(0, width - abs(d_col))
    if rows and cols:
        out[dst_r0 : dst_r0 + rows, dst_c0 : dst_c0 + cols] = arr[
            src_r0 : src_r0 + rows, src_c0 : src_c0 + cols
        ]
    return out


def center_by_mass(arr: np.ndarray) -> np.ndarray:
    """Translate so the centre of mass lands on the centre of the field.

    The centre of an N-wide field is taken as (N - 1) / 2. The shift is rounded
    to whole pixels, so the residual offset is at most half a pixel per axis.
    """
    com = center_of_mass(arr)
    if com is None:
        return arr
    row, col = com
    target = (arr.shape[0] - 1) / 2.0, (arr.shape[1] - 1) / 2.0
    return shift(arr, int(round(target[0] - row)), int(round(target[1] - col)))


def preprocess_naive(image: ImageLike) -> np.ndarray:
    """Grayscale, resize the whole frame to 28x28, scale to [0, 1]. Nothing else.

    This is the path the earlier version of this repo used. No cropping, no
    centring, no polarity correction.
    """
    arr = to_gray_float(image)
    resized = Image.fromarray(arr).resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
    return np.clip(np.asarray(resized, dtype=np.float32), 0.0, 1.0)


def preprocess_mnist_style(image: ImageLike, threshold: float = INK_THRESHOLD) -> np.ndarray:
    """The MNIST normalisation recipe, start to finish.

    Steps: grayscale, stretch contrast, invert if the ink is dark on light, crop
    to the ink bounding box, scale the long side to 20px, paste into 28x28,
    translate the centre of mass to the centre. An image with no ink comes back
    as an all-zero field rather than raising.
    """
    arr = ensure_bright_ink(to_gray_float(image))
    if ink_bbox(arr, threshold) is None:
        return np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
    cropped = crop_to_ink(arr, threshold)
    scaled = scale_long_side(cropped, DIGIT_BOX)
    field = paste_centered(scaled, TARGET_SIZE)
    return np.clip(center_by_mass(field), 0.0, 1.0)


PREPROCESSORS = {
    "naive": preprocess_naive,
    "mnist_style": preprocess_mnist_style,
}


def preprocess(image: ImageLike, mode: str = "mnist_style") -> np.ndarray:
    """Run one of the two paths by name."""
    if mode not in PREPROCESSORS:
        raise ValueError(f"unknown mode {mode!r}, expected one of {sorted(PREPROCESSORS)}")
    return PREPROCESSORS[mode](image)
