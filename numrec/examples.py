"""Write a handful of sample images so `predict` can be demonstrated.

These are SYNTHETIC: MNIST test digits with the same capture distortions the
evaluation uses. They are not photographs of handwriting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .data import wild_set_from_mnist


def write_examples(folder: Path, count: int = 3, seed: int = 7, pool: int = 40):
    """Write `count` distorted digits into `folder`. Returns [(path, true label)]."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    canvases, labels, _ = wild_set_from_mnist(limit=pool, seed=seed)

    chosen = []
    seen: set[int] = set()
    for canvas, label in zip(canvases, labels):
        digit = int(label)
        if digit in seen:
            continue
        seen.add(digit)
        chosen.append((canvas, digit))
        if len(chosen) == count:
            break

    written = []
    for index, (canvas, digit) in enumerate(chosen):
        path = folder / f"sample_{index}_true_{digit}.png"
        Image.fromarray(np.asarray(canvas, dtype=np.uint8), mode="L").save(path)
        written.append((path, digit))
    return written
