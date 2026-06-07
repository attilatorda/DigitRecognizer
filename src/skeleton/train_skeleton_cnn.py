import argparse
import os
import sys

import numpy as np
import torch
from skimage.transform import probabilistic_hough_line
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.common.data_io import load_mnist_idx
from src.common.utils import ensure_dir, set_seed
from src.local_cnn.model import SimpleCNN
from src.skeleton.skeleton_cache import load_or_build
from src.skeleton.skeletonize import skeletonize_batch  # noqa: F401 — re-exported for callers


def hough_line_map_uint8(
    skeleton_img_2d: np.ndarray,
    threshold: int = 8,
    line_length: int = 5,
    line_gap: int = 2,
) -> np.ndarray:
    """Generate a uint8 line map from a skeletonized image using probabilistic Hough."""
    binary = skeleton_img_2d > 0
    lines = probabilistic_hough_line(
        binary,
        threshold=threshold,
        line_length=line_length,
        line_gap=line_gap,
    )
    line_map = np.zeros_like(skeleton_img_2d, dtype=np.uint8)
    for (x0, y0), (x1, y1) in lines:
        num = max(abs(x1 - x0), abs(y1 - y0)) + 1
        xs = np.linspace(x0, x1, num=num, dtype=int)
        ys = np.linspace(y0, y1, num=num, dtype=int)
        line_map[ys, xs] = 255
    return line_map


def build_input_tensor(
    images: np.ndarray,
    channel_mode: str,
    hough_threshold: int,
    hough_line_length: int,
    hough_line_gap: int,
    raw_images: np.ndarray | None = None,
) -> torch.Tensor:
    if channel_mode == "skeleton":
        x_np = images[:, None, :, :].astype(np.float32) / 255.0
    elif channel_mode == "skeleton_hough":
        hough_maps = np.stack(
            [
                hough_line_map_uint8(img, threshold=hough_threshold,
                                     line_length=hough_line_length, line_gap=hough_line_gap)
                for img in images
            ],
            axis=0,
        )
        x_np = np.stack([images, hough_maps], axis=1).astype(np.float32) / 255.0
    elif channel_mode == "raw_thin":
        # Ch0 = raw pixels, Ch1 = Guo-Hall (thin) skeleton
        if raw_images is None:
            raise ValueError("raw_images must be provided for channel_mode='raw_thin'")
        x_np = np.stack([raw_images, images], axis=1).astype(np.float32) / 255.0
    else:
        raise ValueError(f"Unsupported channel_mode: {channel_mode}")
    return torch.tensor(x_np, dtype=torch.float32)


def to_loader(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 128,
    shuffle: bool = False,
    channel_mode: str = "skeleton",
    hough_threshold: int = 8,
    hough_line_length: int = 5,
    hough_line_gap: int = 2,
    raw_images: np.ndarray | None = None,
) -> DataLoader:
    x = build_input_tensor(images, channel_mode, hough_threshold, hough_line_length, hough_line_gap,
                           raw_images=raw_images)
    y = torch.tensor(labels, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Return overall accuracy."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(1, total)


def evaluate_detailed(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> tuple[float, dict[int, float]]:
    """Return (overall_accuracy, {class_id: per_class_accuracy})."""
    model.eval()
    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            all_pred.append(pred.cpu().numpy())
            all_true.append(y.cpu().numpy())
    preds = np.concatenate(all_pred)
    trues = np.concatenate(all_true)
    overall = float((preds == trues).mean())
    per_class = {}
    for c in range(num_classes):
        mask = trues == c
        per_class[c] = float((preds[mask] == c).mean()) if mask.sum() > 0 else 0.0
    return overall, per_class


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_images, train_labels = load_mnist_idx(args.mnist_path, "train")
    test_images, test_labels = load_mnist_idx(args.mnist_path, "t10k")

    train_skel, _ = load_or_build(
        train_images, args.skeleton_method, "train",
        args.cache_dir, args.skeletonize_progress_every,
    )
    test_skel, _ = load_or_build(
        test_images, args.skeleton_method, "test",
        args.cache_dir, args.skeletonize_progress_every,
    )

    raw_tr = train_images if args.channel_mode == "raw_thin" else None
    raw_te = test_images if args.channel_mode == "raw_thin" else None

    train_loader = to_loader(
        train_skel, train_labels, args.batch_size, shuffle=True,
        channel_mode=args.channel_mode,
        hough_threshold=args.hough_threshold,
        hough_line_length=args.hough_line_length,
        hough_line_gap=args.hough_line_gap,
        raw_images=raw_tr,
    )
    test_loader = to_loader(
        test_skel, test_labels, args.batch_size, shuffle=False,
        channel_mode=args.channel_mode,
        hough_threshold=args.hough_threshold,
        hough_line_length=args.hough_line_length,
        hough_line_gap=args.hough_line_gap,
        raw_images=raw_te,
    )

    in_channels = 1 if args.channel_mode == "skeleton" else 2
    model = SimpleCNN(num_classes=10, in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    ensure_dir(args.out_dir)
    best_acc = 0.0
    best_path = os.path.join(args.out_dir, "best_skeleton_cnn.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        acc = evaluate(model, test_loader, device)
        print(f"[skeleton_cnn] epoch={epoch} test_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_path)

    _, per_class = evaluate_detailed(model, test_loader, device)
    per_class_str = "  ".join(f"{c}:{v*100:.1f}%" for c, v in sorted(per_class.items()))
    print(f"[skeleton_cnn] best_test_acc={best_acc:.4f} saved={best_path}")
    print(f"[skeleton_cnn] per_class_acc  {per_class_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-path", default="mnist_data")
    parser.add_argument("--cache-dir", default="data/processed/mnist_skeleton")
    parser.add_argument("--out-dir", default="experiments/checkpoints/skeleton")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--channel-mode",
        choices=["skeleton", "skeleton_hough", "raw_thin"],
        default="skeleton",
    )
    parser.add_argument("--hough-threshold", type=int, default=8)
    parser.add_argument("--hough-line-length", type=int, default=5)
    parser.add_argument("--hough-line-gap", type=int, default=2)
    parser.add_argument(
        "--skeleton-method",
        choices=["zhang", "lee", "thin", "medial_axis"],
        default="zhang",
    )
    parser.add_argument("--skeletonize-progress-every", type=int, default=500)
    main(parser.parse_args())
