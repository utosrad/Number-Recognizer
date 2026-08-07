"""Tests for the synthetic distortion pipeline and the train/test split guard."""

from __future__ import annotations

import numpy as np
import pytest

from numrec import data
from numrec.data import (
    Distortion,
    apply_distortion,
    build_wild_set,
    digit_size,
    load_mnist,
    sample_distortion,
    wild_set_from_mnist,
)
from numrec.preprocess import ink_bbox, preprocess_mnist_style


def fake_digit():
    """A 28x28 stand-in for an MNIST digit: bright ink on black, roughly centred."""
    arr = np.zeros((28, 28), dtype=np.uint8)
    arr[6:22, 11:17] = 255
    return arr


def params(**overrides) -> Distortion:
    base = dict(
        canvas=48,
        scale=1.0,
        top=3,
        left=5,
        thicken=False,
        ink_level=1.0,
        background_level=0.0,
        inverted=False,
    )
    base.update(overrides)
    return Distortion(**base)


def test_distorted_canvas_has_the_requested_size():
    out = apply_distortion(fake_digit(), params(canvas=52))
    assert out.shape == (52, 52)


def test_distorted_canvas_is_uint8():
    assert apply_distortion(fake_digit(), params()).dtype == np.uint8


def test_the_digit_lands_at_the_requested_offset():
    out = apply_distortion(fake_digit(), params(canvas=48, scale=1.0, top=3, left=5))
    top, left, _, _ = ink_bbox(out.astype(np.float32) / 255.0)
    assert top == 3 + 6
    assert left == 5 + 11


def test_scaling_up_makes_the_ink_box_bigger():
    small = apply_distortion(fake_digit(), params(scale=0.6, top=0, left=0))
    large = apply_distortion(fake_digit(), params(scale=1.4, top=0, left=0))
    _, _, small_h, _ = ink_bbox(small.astype(np.float32) / 255.0)
    _, _, large_h, _ = ink_bbox(large.astype(np.float32) / 255.0)
    assert large_h > small_h


def test_thickening_adds_ink():
    plain = apply_distortion(fake_digit(), params(thicken=False))
    thick = apply_distortion(fake_digit(), params(thicken=True))
    assert int(thick.sum()) > int(plain.sum())


def test_inversion_flips_the_background_to_bright():
    upright = apply_distortion(fake_digit(), params(inverted=False))
    inverted = apply_distortion(fake_digit(), params(inverted=True))
    assert upright[0, 0] < 128
    assert inverted[0, 0] > 128


def test_contrast_settings_bound_the_pixel_range():
    out = apply_distortion(
        fake_digit(), params(ink_level=0.6, background_level=0.2)
    ).astype(np.float32) / 255.0
    assert out.min() == pytest.approx(0.2, abs=0.01)
    assert out.max() == pytest.approx(0.6, abs=0.01)


def test_mnist_style_recovers_a_centred_digit_from_a_distorted_canvas():
    out = apply_distortion(
        fake_digit(), params(canvas=60, scale=0.7, top=1, left=2, inverted=True)
    )
    top, left, height, width = ink_bbox(preprocess_mnist_style(out))
    assert max(height, width) <= 20
    assert 2 <= top <= 6 and 2 <= left <= 14


def test_digit_size_leaves_a_margin_inside_the_canvas():
    for canvas in (36, 48, 64):
        assert digit_size(canvas, 3.0) <= canvas - 2


def test_digit_size_tracks_the_scale_factor():
    assert digit_size(64, 1.0) == 28
    assert digit_size(64, 0.5) == 14


def test_sampled_positions_keep_the_digit_inside_the_canvas():
    rng = np.random.default_rng(1)
    for _ in range(200):
        drawn = sample_distortion(rng)
        size = digit_size(drawn.canvas, drawn.scale)
        assert drawn.top + size <= drawn.canvas
        assert drawn.left + size <= drawn.canvas


def test_the_same_seed_produces_the_same_distorted_set():
    images = np.stack([fake_digit(), fake_digit()])
    labels = np.array([3, 3])
    first, _, _ = build_wild_set(images, labels, seed=11)
    second, _, _ = build_wild_set(images, labels, seed=11)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_different_seeds_produce_different_distortions():
    images = np.stack([fake_digit()] * 8)
    labels = np.zeros(8, dtype=np.int64)
    first, _, _ = build_wild_set(images, labels, seed=1)
    second, _, _ = build_wild_set(images, labels, seed=2)
    assert any(a.shape != b.shape or not np.array_equal(a, b) for a, b in zip(first, second))


def test_build_wild_set_preserves_labels_and_length():
    images = np.stack([fake_digit()] * 5)
    labels = np.array([0, 1, 2, 3, 4])
    canvases, out_labels, drawn = build_wild_set(images, labels, seed=3)
    assert len(canvases) == 5
    assert len(drawn) == 5
    assert np.array_equal(out_labels, labels)


def test_roughly_half_the_set_is_inverted():
    images = np.stack([fake_digit()] * 400)
    labels = np.zeros(400, dtype=np.int64)
    _, _, drawn = build_wild_set(images, labels, seed=5)
    fraction = sum(p.inverted for p in drawn) / len(drawn)
    assert 0.4 < fraction < 0.6


def test_load_mnist_rejects_an_unknown_split():
    with pytest.raises(ValueError):
        load_mnist("validation")


def test_the_wild_set_is_built_from_the_test_split_only(monkeypatch):
    """Leakage guard: evaluation data must never come from the training split."""
    requested = []

    def spy(split, root=None, download=True):
        requested.append(split)
        return np.stack([fake_digit()] * 4), np.arange(4)

    monkeypatch.setattr(data, "load_mnist", spy)
    wild_set_from_mnist(limit=4, seed=0)
    assert requested == ["test"]
    assert "train" not in requested
