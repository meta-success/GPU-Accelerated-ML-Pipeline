"""PyTorch CIFAR-10 CNN with GPU training, checkpointing, and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import NUM_CLASSES, PipelineConfig
from src.data import iterate_batches, to_nchw
from src.logging_utils import get_logger

logger = get_logger("pytorch")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    F = None
    Adam = None
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:

    class ResidualBlock(nn.Module):
        """Basic residual block: Conv-BN-ReLU-Conv-BN + skip, then ReLU."""

        def __init__(self, channels: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return F.relu(out + residual)

    class CIFARCNN(nn.Module):
        """Compact ResNet-style CNN for 32x32 CIFAR images."""

        def __init__(self, num_classes: int = NUM_CLASSES) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.layer1 = nn.Sequential(ResidualBlock(32), nn.MaxPool2d(2))  # 16x16
            self.down1 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.layer2 = ResidualBlock(64)  # 8x8
            self.down2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
            self.layer3 = ResidualBlock(128)  # 4x4
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.stem(x)
            x = self.layer1(x)
            x = self.down1(x)
            x = self.layer2(x)
            x = self.down2(x)
            x = self.layer3(x)
            return self.head(x)

else:  # pragma: no cover

    class ResidualBlock:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PyTorch is not installed")

    class CIFARCNN:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PyTorch is not installed")


@dataclass
class PyTorchTrainResult:
    model: object
    history: list[dict]
    checkpoint_path: Path
    device: str
    accuracy: float


def resolve_torch_device(preferred: str = "auto") -> "torch.device":
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed")
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; using CPU")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: "nn.Module") -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model: "nn.Module", path: Path, extra: Optional[dict] = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), "extra": extra or {}}
    torch.save(payload, path)
    logger.info("Saved PyTorch checkpoint -> %s", path)
    return path


def load_checkpoint(path: Path, device: Optional["torch.device"] = None) -> CIFARCNN:
    map_location = device or torch.device("cpu")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model = CIFARCNN()
    model.load_state_dict(payload["state_dict"])
    model.to(map_location)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model: "nn.Module", images: np.ndarray, labels: np.ndarray, device, batch_size: int = 128) -> float:
    model.eval()
    correct = 0
    total = 0
    x = to_nchw(images)
    for xb, yb in iterate_batches(x, labels, batch_size, shuffle=False):
        inputs = torch.from_numpy(xb).to(device)
        targets = torch.from_numpy(yb.astype(np.int64)).to(device)
        logits = model(inputs)
        pred = logits.argmax(dim=1)
        correct += int((pred == targets).sum().item())
        total += int(targets.shape[0])
    return correct / max(total, 1)


def train_pytorch(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    config: PipelineConfig,
) -> PyTorchTrainResult:
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed")

    device = resolve_torch_device(config.device)
    model = CIFARCNN().to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    logger.info(
        "PyTorch training on %s | params=%s | train=%s | epochs=%s",
        device,
        f"{count_parameters(model):,}",
        x_train.shape[0],
        config.epochs,
    )

    x_tr = to_nchw(x_train)
    history: list[dict] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for xb, yb in iterate_batches(x_tr, y_train, config.batch_size, shuffle=True, seed=config.seed + epoch):
            if np.random.default_rng(config.seed + epoch + seen).random() < 0.5:
                xb = np.ascontiguousarray(xb[:, :, :, ::-1])
            inputs = torch.from_numpy(xb).to(device)
            targets = torch.from_numpy(yb.astype(np.int64)).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * inputs.size(0)
            seen += inputs.size(0)

        train_loss = running_loss / max(seen, 1)
        acc = evaluate(model, x_test, y_test, device, config.batch_size)
        history.append({"epoch": epoch, "loss": train_loss, "val_accuracy": acc})
        logger.info("PyTorch epoch %s/%s | loss=%.4f | val_acc=%.3f", epoch, config.epochs, train_loss, acc)

    ckpt = save_checkpoint(
        model,
        config.checkpoint_dir / "pytorch_cifar_cnn.pt",
        extra={"history": history, "device": str(device)},
    )
    return PyTorchTrainResult(
        model=model,
        history=history,
        checkpoint_path=ckpt,
        device=str(device),
        accuracy=history[-1]["val_accuracy"] if history else 0.0,
    )


@torch.no_grad()
def infer_pytorch(model: "nn.Module", images: np.ndarray, device, batch_size: int = 128) -> np.ndarray:
    model.eval()
    outputs = []
    x = to_nchw(images)
    for xb, _ in iterate_batches(x, np.zeros((x.shape[0],), dtype=np.int32), batch_size, shuffle=False):
        inputs = torch.from_numpy(xb).to(device)
        logits = model(inputs)
        outputs.append(logits.softmax(dim=1).cpu().numpy())
    return np.concatenate(outputs, axis=0)
