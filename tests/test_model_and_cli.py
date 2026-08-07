"""Tests for the model plumbing, the folder predictor and the CLI wiring."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from numrec.cli import build_parser
from numrec.evaluate import format_report
from numrec.model import SmallCNN, load_model, predict_batch, save_model, to_batch
from numrec.predict import find_images, format_predictions, predict_folder


def test_forward_pass_returns_one_logit_per_class():
    out = SmallCNN()(torch.zeros(4, 1, 28, 28))
    assert out.shape == (4, 10)


def test_the_model_is_small_enough_to_commit():
    assert sum(p.numel() for p in SmallCNN().parameters()) < 1_000_000


def test_to_batch_adds_the_batch_and_channel_axes():
    assert to_batch(np.zeros((28, 28), dtype=np.float32)).shape == (1, 1, 28, 28)
    assert to_batch(np.zeros((5, 28, 28), dtype=np.float32)).shape == (5, 1, 28, 28)


def test_to_batch_rejects_the_wrong_image_size():
    with pytest.raises(ValueError):
        to_batch(np.zeros((3, 32, 32), dtype=np.float32))


def test_predict_batch_returns_probabilities_that_sum_to_one():
    labels, probs = predict_batch(SmallCNN(), np.zeros((3, 28, 28), dtype=np.float32))
    assert labels.shape == (3,)
    assert probs.shape == (3, 10)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_predict_batch_agrees_across_batch_boundaries():
    model = SmallCNN().eval()
    images = np.random.default_rng(0).random((9, 28, 28)).astype(np.float32)
    one_shot, _ = predict_batch(model, images, batch_size=64)
    chunked, _ = predict_batch(model, images, batch_size=2)
    assert np.array_equal(one_shot, chunked)


def test_saved_weights_round_trip(tmp_path):
    path = tmp_path / "weights.pt"
    original = SmallCNN().eval()
    save_model(original, path)
    restored = load_model(path)
    images = np.random.default_rng(1).random((4, 28, 28)).astype(np.float32)
    assert np.array_equal(predict_batch(original, images)[0], predict_batch(restored, images)[0])


def test_load_model_says_what_to_do_when_weights_are_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="train"):
        load_model(tmp_path / "absent.pt")


def test_a_loaded_model_is_in_eval_mode(tmp_path):
    path = tmp_path / "weights.pt"
    save_model(SmallCNN(), path)
    assert load_model(path).training is False


def test_find_images_picks_up_supported_suffixes_and_ignores_the_rest(tmp_path):
    for name in ("b.png", "a.JPG", "notes.txt", "c.jpeg"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in find_images(tmp_path)] == ["a.JPG", "b.png", "c.jpeg"]


def test_find_images_rejects_a_path_that_is_not_a_directory(tmp_path):
    target = tmp_path / "file.png"
    target.write_bytes(b"")
    with pytest.raises(NotADirectoryError):
        find_images(target)


def test_predict_folder_returns_nothing_for_an_empty_folder(tmp_path):
    assert predict_folder(tmp_path) == []


def test_predict_folder_works_on_an_arbitrary_folder(tmp_path):
    weights = tmp_path / "weights.pt"
    save_model(SmallCNN(), weights)
    folder = tmp_path / "images"
    folder.mkdir()
    canvas = np.zeros((50, 50), dtype=np.uint8)
    canvas[10:40, 20:30] = 255
    Image.fromarray(canvas, mode="L").save(folder / "one.png")
    rows = predict_folder(folder, mode="mnist_style", weights=weights)
    assert len(rows) == 1
    path, digit, confidence = rows[0]
    assert path.name == "one.png"
    assert 0 <= digit <= 9
    assert 0.0 <= confidence <= 1.0


def test_format_predictions_reports_the_mode_it_used():
    from pathlib import Path

    text = format_predictions([(Path("a.png"), 4, 0.9)], "naive")
    assert "naive" in text
    assert "a.png" in text


def test_format_report_labels_the_distorted_set_as_synthetic():
    results = {
        "n": 10,
        "seed": 0,
        "clean": {"naive": 0.9, "mnist_style": 0.9},
        "distorted": {"naive": 0.2, "mnist_style": 0.8},
        "distorted_upright": {"naive": 0.3, "mnist_style": 0.8},
        "distorted_inverted": {"naive": 0.1, "mnist_style": 0.8},
    }
    text = format_report(results)
    assert "synthetic" in text.lower()
    assert "60.00" in text


def test_the_cli_exposes_train_evaluate_and_predict():
    parser = build_parser()
    for command in ("train", "evaluate", "predict"):
        argv = [command, "."] if command == "predict" else [command]
        assert parser.parse_args(argv).command == command


def test_predict_defaults_to_mnist_style():
    assert build_parser().parse_args(["predict", "/some/folder"]).mode == "mnist_style"


def test_predict_accepts_the_naive_flag():
    args = build_parser().parse_args(["predict", "/some/folder", "--preprocess", "naive"])
    assert args.mode == "naive"


def test_the_cli_rejects_an_unknown_preprocessing_mode():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["predict", "/f", "--preprocess", "blur"])


def test_the_cli_requires_a_folder_for_predict():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["predict"])
