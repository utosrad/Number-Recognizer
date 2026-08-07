"""Predict digits for every image in a folder the caller names."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .model import load_model, predict_batch
from .preprocess import preprocess

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}


def find_images(folder: Path) -> list[Path]:
    """Every image file directly inside `folder`, sorted by name."""
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} is not a directory")
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def predict_folder(folder: Path, mode: str = "mnist_style", weights: Path | None = None):
    """Return a list of (path, digit, confidence) for the images in `folder`."""
    paths = find_images(folder)
    if not paths:
        return []
    model = load_model(weights)
    batch = np.stack([preprocess(p, mode) for p in paths])
    labels, probs = predict_batch(model, batch)
    return [
        (path, int(label), float(prob[label]))
        for path, label, prob in zip(paths, labels, probs)
    ]


def format_predictions(rows, mode: str) -> str:
    if not rows:
        return "no images found"
    width = max(len(p.name) for p, _, _ in rows)
    lines = [f"preprocessing: {mode}"]
    lines += [
        f"{path.name:<{width}}  {digit}  ({confidence * 100:.1f}%)"
        for path, digit, confidence in rows
    ]
    return "\n".join(lines)
