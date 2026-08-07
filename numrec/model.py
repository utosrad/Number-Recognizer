"""The classifier and the weight file it lives in."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "mnist_cnn.pt"
DEFAULT_DATA_ROOT = REPO_ROOT / "data"


def weights_path() -> Path:
    """Where the trained weights live. Override with NUMREC_WEIGHTS."""
    return Path(os.environ.get("NUMREC_WEIGHTS", DEFAULT_WEIGHTS))


def data_root() -> Path:
    """Where torchvision caches MNIST. Override with NUMREC_DATA."""
    return Path(os.environ.get("NUMREC_DATA", DEFAULT_DATA_ROOT))


class SmallCNN(nn.Module):
    """Two conv blocks and a classifier head. Inputs are 1x28x28 in [0, 1]."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def save_model(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: Path | None = None) -> SmallCNN:
    """Load trained weights into a fresh model, in eval mode."""
    path = Path(path) if path is not None else weights_path()
    if not path.exists():
        raise FileNotFoundError(
            f"no weights at {path}. Run `python -m numrec train` first."
        )
    model = SmallCNN()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def to_batch(images) -> torch.Tensor:
    """Stack 28x28 float arrays into an (N, 1, 28, 28) tensor."""
    arr = np.asarray(images, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[1:] != (28, 28):
        raise ValueError(f"expected (N, 28, 28) input, got {arr.shape}")
    return torch.from_numpy(arr).unsqueeze(1)


@torch.no_grad()
def predict_batch(model: nn.Module, images, batch_size: int = 512):
    """Return (labels, probabilities) for a sequence of 28x28 float arrays."""
    model.eval()
    labels = []
    probs = []
    tensor = to_batch(images)
    for start in range(0, tensor.shape[0], batch_size):
        chunk = tensor[start : start + batch_size]
        logits = model(chunk)
        p = torch.softmax(logits, dim=1)
        labels.append(p.argmax(dim=1))
        probs.append(p)
    return torch.cat(labels).numpy(), torch.cat(probs).numpy()
