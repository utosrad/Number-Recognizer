"""MNIST access and the synthetic "in the wild" set.

The distorted set in this module is SYNTHETIC. It is built by applying capture
distortions to MNIST *test* images. It contains no photographs and no real
scanned handwriting. Treat its numbers as a measurement of how much the
preprocessing gap costs under known, controlled distortions, not as a
real-world accuracy estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from torchvision import datasets

from .model import data_root

VALID_SPLITS = ("train", "test")


def load_mnist(split: str, root: Path | None = None, download: bool = True):
    """Return (images uint8 NxHxW, labels int64 N) for one MNIST split."""
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
    root = Path(root) if root is not None else data_root()
    dataset = datasets.MNIST(
        root=str(root), train=(split == "train"), download=download
    )
    return dataset.data.numpy().astype(np.uint8), dataset.targets.numpy().astype(np.int64)


@dataclass(frozen=True)
class Distortion:
    """The parameters drawn for one synthetic capture."""

    canvas: int
    scale: float
    top: int
    left: int
    thicken: bool
    ink_level: float
    background_level: float
    inverted: bool


def digit_size(canvas: int, scale: float) -> int:
    """Side length of the scaled digit, clamped to leave a margin inside the canvas."""
    return min(max(4, int(round(28 * scale))), canvas - 2)


def sample_distortion(rng: np.random.Generator) -> Distortion:
    """Draw one set of capture parameters.

    Ranges are deliberately wide but plausible for someone photographing or
    screenshotting a digit: the frame is bigger than the digit, the digit is
    somewhere inside it, the pen may be thicker, the exposure may be flat, and
    the paper may be white.
    """
    canvas = int(rng.integers(36, 65))
    scale = float(rng.uniform(0.55, 1.5))
    span = canvas - digit_size(canvas, scale)
    return Distortion(
        canvas=canvas,
        scale=scale,
        top=int(rng.integers(0, span + 1)),
        left=int(rng.integers(0, span + 1)),
        thicken=bool(rng.random() < 0.5),
        ink_level=float(rng.uniform(0.45, 1.0)),
        background_level=float(rng.uniform(0.0, 0.25)),
        inverted=bool(rng.random() < 0.5),
    )


def apply_distortion(digit: np.ndarray, params: Distortion) -> np.ndarray:
    """Apply one distortion to a 28x28 uint8 MNIST digit. Returns a uint8 canvas."""
    img = Image.fromarray(digit.astype(np.uint8), mode="L")
    if params.thicken:
        img = img.filter(ImageFilter.MaxFilter(3))
    size = digit_size(params.canvas, params.scale)
    img = img.resize((size, size), Image.BILINEAR)

    canvas = np.zeros((params.canvas, params.canvas), dtype=np.float32)
    patch = np.asarray(img, dtype=np.float32) / 255.0
    canvas[params.top : params.top + size, params.left : params.left + size] = patch

    lo = params.background_level
    hi = max(params.ink_level, lo + 0.2)
    canvas = lo + (hi - lo) * canvas
    if params.inverted:
        canvas = 1.0 - canvas
    return np.clip(canvas * 255.0, 0, 255).astype(np.uint8)


def build_wild_set(images: np.ndarray, labels: np.ndarray, seed: int = 0):
    """Distort every image in `images`. Returns (list of uint8 canvases, labels, params).

    SYNTHETIC. The output is MNIST test data pushed through the distortions
    above, not real captures.
    """
    rng = np.random.default_rng(seed)
    canvases = []
    all_params = []
    for digit in images:
        params = sample_distortion(rng)
        canvases.append(apply_distortion(digit, params))
        all_params.append(params)
    return canvases, np.asarray(labels), all_params


def wild_set_from_mnist(limit: int | None = None, seed: int = 0, root: Path | None = None):
    """Build the synthetic distorted set from the MNIST *test* split only.

    Training images are never read here. That is the point: the distorted set is
    held out from anything the model saw.
    """
    images, labels = load_mnist("test", root=root)
    if limit is not None:
        images, labels = images[:limit], labels[:limit]
    return build_wild_set(images, labels, seed=seed)
