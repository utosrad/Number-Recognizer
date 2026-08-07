"""Train the classifier on MNIST.

No augmentation. The model is deliberately a plain MNIST model, because the
point of this repo is what preprocessing does at inference time, not how far
augmentation can paper over it.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .model import SmallCNN, data_root, save_model, weights_path


def loaders(batch_size: int, root: Path):
    """MNIST train and test loaders. Pixels arrive as floats in [0, 1]."""
    tf = transforms.ToTensor()
    train = datasets.MNIST(root=str(root), train=True, download=True, transform=tf)
    test = datasets.MNIST(root=str(root), train=False, download=True, transform=tf)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(test, batch_size=1000, shuffle=False, num_workers=0),
    )


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        correct += int((model(x).argmax(dim=1) == y).sum())
        total += int(y.numel())
    return correct / total


def train(
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
    out: Path | None = None,
    root: Path | None = None,
) -> dict:
    """Train, evaluate on the clean MNIST test split, save weights.

    Returns a dict with the final test accuracy and wall-clock training time.
    """
    torch.manual_seed(seed)
    out = Path(out) if out is not None else weights_path()
    root = Path(root) if root is not None else data_root()

    train_loader, test_loader = loaders(batch_size, root)
    model = SmallCNN()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for step, (x, y) in enumerate(train_loader, start=1):
            optimiser.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimiser.step()
            running += float(loss)
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss {running / step:.4f}")
        acc = accuracy(model, test_loader)
        print(f"epoch {epoch} done: mnist test accuracy {acc:.4f}")

    elapsed = time.time() - started
    test_accuracy = accuracy(model, test_loader)
    save_model(model, out)
    params = sum(p.numel() for p in model.parameters())
    print(f"saved {params} parameters to {out}")
    print(f"mnist test accuracy: {test_accuracy:.4f}  ({elapsed:.1f}s of training)")
    return {
        "test_accuracy": test_accuracy,
        "seconds": elapsed,
        "parameters": params,
        "weights": str(out),
    }
