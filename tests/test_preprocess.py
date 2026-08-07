"""Tests for the two preprocessing paths.

Everything here is checked against a fact that can be derived by hand from a
constructed input. There are no golden files and no recorded outputs.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from numrec.preprocess import (
    DIGIT_BOX,
    TARGET_SIZE,
    background_is_light,
    center_by_mass,
    center_of_mass,
    crop_to_ink,
    ensure_bright_ink,
    ink_bbox,
    paste_centered,
    preprocess,
    preprocess_mnist_style,
    preprocess_naive,
    scale_long_side,
    shift,
    stretch_contrast,
    to_gray_float,
)

CENTER = (TARGET_SIZE - 1) / 2.0


def blob(canvas=(60, 60), top=4, left=6, height=20, width=10, value=1.0):
    """A solid rectangle of ink on a black field, at a position we choose."""
    arr = np.zeros(canvas, dtype=np.float32)
    arr[top : top + height, left : left + width] = value
    return arr


# --- centre of mass and centring -------------------------------------------


def test_center_of_mass_of_a_symmetric_rectangle_is_its_geometric_centre():
    arr = blob(top=4, left=6, height=20, width=10)
    row, col = center_of_mass(arr)
    assert row == pytest.approx(4 + (20 - 1) / 2.0)
    assert col == pytest.approx(6 + (10 - 1) / 2.0)


def test_center_of_mass_is_none_for_an_empty_image():
    assert center_of_mass(np.zeros((28, 28), dtype=np.float32)) is None


def test_center_by_mass_puts_an_off_centre_blob_within_a_pixel_of_the_centre():
    arr = np.zeros((28, 28), dtype=np.float32)
    arr[2:8, 3:9] = 1.0
    centred = center_by_mass(arr)
    row, col = center_of_mass(centred)
    assert abs(row - CENTER) <= 1.0
    assert abs(col - CENTER) <= 1.0


def test_center_by_mass_residual_is_at_most_half_a_pixel_when_nothing_clips():
    arr = np.zeros((28, 28), dtype=np.float32)
    arr[3:10, 4:9] = 1.0
    row, col = center_of_mass(center_by_mass(arr))
    assert abs(row - CENTER) <= 0.5 + 1e-6
    assert abs(col - CENTER) <= 0.5 + 1e-6


def test_center_by_mass_preserves_total_ink_when_nothing_clips():
    arr = np.zeros((28, 28), dtype=np.float32)
    arr[5:11, 6:12] = 1.0
    assert center_by_mass(arr).sum() == pytest.approx(arr.sum())


def test_center_by_mass_leaves_an_already_centred_image_alone():
    arr = np.zeros((28, 28), dtype=np.float32)
    arr[11:17, 11:17] = 1.0
    assert np.array_equal(center_by_mass(arr), arr)


def test_shift_moves_content_by_exactly_the_requested_offset():
    arr = np.zeros((10, 10), dtype=np.float32)
    arr[2, 3] = 1.0
    moved = shift(arr, 4, -1)
    assert moved[6, 2] == 1.0
    assert moved.sum() == pytest.approx(1.0)


# --- bounding box and cropping ---------------------------------------------


def test_ink_bbox_is_tight_around_a_known_rectangle():
    arr = blob(top=7, left=11, height=9, width=5)
    assert ink_bbox(arr) == (7, 11, 9, 5)


def test_ink_bbox_is_none_when_no_pixel_clears_the_threshold():
    arr = np.full((20, 20), 0.05, dtype=np.float32)
    assert ink_bbox(arr) is None


def test_crop_to_ink_returns_exactly_the_bounding_box():
    arr = blob(top=3, left=9, height=12, width=6)
    cropped = crop_to_ink(arr)
    assert cropped.shape == (12, 6)
    assert cropped.min() == 1.0


def test_crop_to_ink_leaves_a_blank_image_untouched():
    arr = np.zeros((15, 17), dtype=np.float32)
    assert crop_to_ink(arr).shape == (15, 17)


# --- scaling ----------------------------------------------------------------


def test_scale_long_side_puts_the_long_side_at_exactly_twenty_pixels():
    assert scale_long_side(np.ones((40, 17), dtype=np.float32)).shape[0] == DIGIT_BOX
    assert scale_long_side(np.ones((11, 33), dtype=np.float32)).shape[1] == DIGIT_BOX


def test_scale_long_side_preserves_aspect_ratio_to_the_nearest_pixel():
    for height, width in [(40, 20), (30, 15), (9, 27), (25, 25)]:
        out_h, out_w = scale_long_side(np.ones((height, width), dtype=np.float32)).shape
        expected = DIGIT_BOX * min(height, width) / max(height, width)
        assert abs(min(out_h, out_w) - expected) <= 0.5 + 1e-6


def test_scale_long_side_never_collapses_a_thin_stroke_to_zero():
    out = scale_long_side(np.ones((60, 1), dtype=np.float32))
    assert out.shape == (DIGIT_BOX, 1)


def test_scale_long_side_rejects_an_empty_image():
    with pytest.raises(ValueError):
        scale_long_side(np.zeros((0, 5), dtype=np.float32))


# --- pasting ----------------------------------------------------------------


def test_paste_centered_places_content_with_balanced_margins():
    field = paste_centered(np.ones((20, 10), dtype=np.float32))
    assert field.shape == (TARGET_SIZE, TARGET_SIZE)
    assert ink_bbox(field) == (4, 9, 20, 10)


def test_paste_centered_rejects_content_larger_than_the_field():
    with pytest.raises(ValueError):
        paste_centered(np.ones((30, 10), dtype=np.float32))


# --- polarity ---------------------------------------------------------------


def test_background_is_light_detects_dark_ink_on_light_paper():
    assert background_is_light(1.0 - blob()) is True


def test_background_is_light_is_false_for_mnist_polarity():
    assert background_is_light(blob()) is False


def test_background_is_light_survives_a_low_contrast_capture():
    faint = 0.30 + 0.25 * (1.0 - blob())
    assert background_is_light(faint) is True


def test_background_is_light_is_false_for_a_flat_image():
    assert background_is_light(np.full((20, 20), 0.4, dtype=np.float32)) is False


def test_ensure_bright_ink_makes_ink_the_maximum_in_both_polarities():
    dark_on_light = 1.0 - blob()
    light_on_dark = blob()
    for arr in (dark_on_light, light_on_dark):
        out = ensure_bright_ink(arr)
        assert out[10, 8] == pytest.approx(1.0)
        assert out[0, 0] == pytest.approx(0.0)


def test_stretch_contrast_maps_a_narrow_band_onto_the_full_range():
    arr = 0.3 + 0.2 * blob()
    out = stretch_contrast(arr)
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)


def test_stretch_contrast_leaves_a_flat_image_alone():
    arr = np.full((8, 8), 0.7, dtype=np.float32)
    assert np.array_equal(stretch_contrast(arr), arr)


def test_mnist_style_gives_the_same_output_for_both_polarities():
    upright = blob()
    inverted = 1.0 - upright
    assert np.allclose(preprocess_mnist_style(upright), preprocess_mnist_style(inverted))


# --- the full paths ---------------------------------------------------------


def test_both_paths_return_a_28x28_float_array_in_unit_range():
    arr = blob()
    for mode in ("naive", "mnist_style"):
        out = preprocess(arr, mode)
        assert out.shape == (TARGET_SIZE, TARGET_SIZE)
        assert out.dtype == np.float32
        assert 0.0 <= out.min() and out.max() <= 1.0


def test_the_two_paths_disagree_on_an_off_centre_input():
    arr = blob(canvas=(64, 64), top=2, left=3, height=18, width=9)
    assert not np.allclose(preprocess_naive(arr), preprocess_mnist_style(arr))


def test_mnist_style_centres_an_off_centre_input_but_naive_does_not():
    arr = blob(canvas=(64, 64), top=2, left=3, height=18, width=9)
    smart_row, smart_col = center_of_mass(preprocess_mnist_style(arr))
    naive_row, naive_col = center_of_mass(preprocess_naive(arr))
    smart_offset = max(abs(smart_row - CENTER), abs(smart_col - CENTER))
    naive_offset = max(abs(naive_row - CENTER), abs(naive_col - CENTER))
    assert smart_offset <= 1.0
    assert naive_offset > 3.0


def test_mnist_style_fits_the_digit_inside_a_twenty_pixel_box():
    arr = blob(canvas=(64, 64), top=2, left=3, height=40, width=12)
    _, _, height, width = ink_bbox(preprocess_mnist_style(arr))
    assert max(height, width) <= DIGIT_BOX


def test_mnist_style_is_scale_invariant_for_the_same_shape():
    small = blob(canvas=(40, 40), top=5, left=6, height=10, width=5)
    large = blob(canvas=(80, 80), top=30, left=9, height=40, width=20)
    difference = np.abs(preprocess_mnist_style(small) - preprocess_mnist_style(large)).max()
    assert difference < 0.05


def test_a_blank_image_does_not_crash_either_path():
    blank = np.zeros((50, 50), dtype=np.float32)
    assert preprocess_mnist_style(blank).sum() == 0.0
    assert preprocess_naive(blank).sum() == 0.0


def test_a_uniformly_white_image_does_not_crash_mnist_style():
    assert preprocess_mnist_style(np.ones((50, 50), dtype=np.float32)).shape == (28, 28)


def test_preprocess_rejects_an_unknown_mode():
    with pytest.raises(ValueError):
        preprocess(blob(), "sharpen")


# --- input handling ---------------------------------------------------------


def test_to_gray_float_scales_uint8_input_into_unit_range():
    out = to_gray_float(np.array([[0, 128, 255]], dtype=np.uint8))
    assert out.max() == pytest.approx(1.0)
    assert out.min() == pytest.approx(0.0)


def test_to_gray_float_averages_colour_channels():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    assert to_gray_float(rgb)[0, 0] == pytest.approx(1 / 3, abs=1e-6)


def test_to_gray_float_rejects_a_one_dimensional_input():
    with pytest.raises(ValueError):
        to_gray_float(np.zeros(9, dtype=np.float32))


def test_both_paths_accept_a_file_path(tmp_path):
    path = tmp_path / "digit.png"
    Image.fromarray((blob() * 255).astype(np.uint8), mode="L").save(path)
    for mode in ("naive", "mnist_style"):
        assert preprocess(path, mode).shape == (TARGET_SIZE, TARGET_SIZE)


def test_a_path_and_the_matching_array_give_the_same_result(tmp_path):
    arr = blob()
    path = tmp_path / "digit.png"
    Image.fromarray((arr * 255).astype(np.uint8), mode="L").save(path)
    assert np.allclose(preprocess_mnist_style(path), preprocess_mnist_style(arr))
