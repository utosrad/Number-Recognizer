"""Measure what preprocessing is worth.

The same trained model is run twice over the same images, once through the
`naive` path and once through `mnist_style`. The difference is entirely
preprocessing.

The distorted set used here is SYNTHETIC: MNIST *test* digits with capture
distortions applied programmatically. It is not photographs and not real
scanned handwriting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import load_mnist, wild_set_from_mnist
from .model import load_model, predict_batch
from .preprocess import preprocess

MODES = ("naive", "mnist_style")


def accuracy_for(model, images, labels, mode: str) -> np.ndarray:
    """Per-image correctness under one preprocessing path."""
    batch = np.stack([preprocess(img, mode) for img in images])
    predicted, _ = predict_batch(model, batch)
    return predicted == labels


def evaluate(
    limit: int | None = 10000,
    seed: int = 0,
    weights: Path | None = None,
    root: Path | None = None,
) -> dict:
    """Run both preprocessing paths on clean MNIST test and on the synthetic set."""
    model = load_model(weights)

    clean_images, clean_labels = load_mnist("test", root=root)
    if limit is not None:
        clean_images, clean_labels = clean_images[:limit], clean_labels[:limit]

    wild_images, wild_labels, params = wild_set_from_mnist(
        limit=limit, seed=seed, root=root
    )
    inverted = np.array([p.inverted for p in params], dtype=bool)

    results = {
        "n": int(len(wild_labels)),
        "seed": seed,
        "clean": {},
        "distorted": {},
        "distorted_upright": {},
        "distorted_inverted": {},
    }
    for mode in MODES:
        clean_hits = accuracy_for(model, clean_images, clean_labels, mode)
        wild_hits = accuracy_for(model, wild_images, wild_labels, mode)
        results["clean"][mode] = float(clean_hits.mean())
        results["distorted"][mode] = float(wild_hits.mean())
        results["distorted_upright"][mode] = float(wild_hits[~inverted].mean())
        results["distorted_inverted"][mode] = float(wild_hits[inverted].mean())
    return results


def format_report(results: dict) -> str:
    """A plain text table of the numbers in `results`."""
    rows = [
        ("clean MNIST test", "clean"),
        ("synthetic distorted", "distorted"),
        ("  of which upright", "distorted_upright"),
        ("  of which inverted", "distorted_inverted"),
    ]
    lines = [
        f"n = {results['n']} images per set, distortion seed {results['seed']}",
        "",
        f"{'set':<22}{'naive':>10}{'mnist_style':>14}{'delta':>10}",
        "-" * 56,
    ]
    for label, key in rows:
        naive = results[key]["naive"]
        smart = results[key]["mnist_style"]
        lines.append(
            f"{label:<22}{naive * 100:>9.2f}%{smart * 100:>13.2f}%{(smart - naive) * 100:>9.2f}"
        )
    lines.append("")
    lines.append("The distorted set is synthetic: MNIST test digits with capture")
    lines.append("distortions applied in code. Not photographs, not real handwriting.")
    return "\n".join(lines)
