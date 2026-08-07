"""Command line entry point: train, evaluate, predict."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .preprocess import PREPROCESSORS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="numrec",
        description="Train a digit classifier and measure the MNIST preprocessing gap.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_cmd = sub.add_parser("train", help="train the CNN on MNIST")
    train_cmd.add_argument("--epochs", type=int, default=3)
    train_cmd.add_argument("--batch-size", type=int, default=128)
    train_cmd.add_argument("--lr", type=float, default=1e-3)
    train_cmd.add_argument("--seed", type=int, default=0)
    train_cmd.add_argument("--out", type=Path, default=None, help="where to write weights")

    eval_cmd = sub.add_parser(
        "evaluate",
        help="compare naive and mnist_style preprocessing on the synthetic distorted set",
    )
    eval_cmd.add_argument(
        "--limit", type=int, default=10000, help="images per set, max 10000"
    )
    eval_cmd.add_argument("--seed", type=int, default=0, help="distortion seed")
    eval_cmd.add_argument("--weights", type=Path, default=None)

    predict_cmd = sub.add_parser("predict", help="classify every image in a folder")
    predict_cmd.add_argument("folder", type=Path, help="folder of PNG or JPG files")
    predict_cmd.add_argument(
        "--preprocess",
        dest="mode",
        choices=sorted(PREPROCESSORS),
        default="mnist_style",
    )
    predict_cmd.add_argument("--weights", type=Path, default=None)

    examples_cmd = sub.add_parser(
        "make-examples", help="write sample distorted digits into a folder"
    )
    examples_cmd.add_argument("folder", type=Path)
    examples_cmd.add_argument("--count", type=int, default=3)
    examples_cmd.add_argument("--seed", type=int, default=7)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "train":
        from .train import train

        train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            out=args.out,
        )
        return 0

    if args.command == "evaluate":
        from .evaluate import evaluate, format_report

        print(format_report(evaluate(limit=args.limit, seed=args.seed, weights=args.weights)))
        return 0

    if args.command == "predict":
        from .predict import format_predictions, predict_folder

        rows = predict_folder(args.folder, mode=args.mode, weights=args.weights)
        print(format_predictions(rows, args.mode))
        return 0

    from .examples import write_examples

    for path, digit in write_examples(args.folder, count=args.count, seed=args.seed):
        print(f"wrote {path}  (true label {digit})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
