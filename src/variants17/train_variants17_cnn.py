import argparse
import json
import os
import sys

import numpy as np
import torch
from skimage.morphology import binary_dilation, binary_erosion
from skimage.transform import AffineTransform, rotate, warp
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.common.data_io import load_mnist_idx
from src.common.utils import ensure_dir, set_seed
from src.local_cnn.model import SimpleCNN
from src.variants17.label_schema import CLASS17_TO_DIGIT10


def augment_templates(images: np.ndarray, labels: np.ndarray, repeats: int = 256, noise_std: float = 0.04):
    x = images.astype(np.float32) / 255.0
    X, Y = [], []
    rng = np.random.default_rng(42)
    for img, y in zip(x, labels):
        for _ in range(repeats):
            jitter = rng.normal(0.0, noise_std, size=img.shape).astype(np.float32)
            sample = np.clip(img + jitter, 0.0, 1.0)
            X.append(sample)
            Y.append(y)
    return np.stack(X, axis=0), np.array(Y, dtype=np.int64)


def _apply_controlled_transform(img01: np.ndarray, params: dict, rng: np.random.Generator) -> np.ndarray:
    out = img01

    out = rotate(
        out,
        angle=float(params["rotation_deg"]),
        resize=False,
        mode="edge",
        preserve_range=True,
    )

    tf = AffineTransform(scale=(float(params["stretch_x"]), float(params["stretch_y"])))
    out = warp(out, tf.inverse, preserve_range=True, mode="edge", output_shape=out.shape)

    fg = out < 0.7
    if params["thickness_op"] == "dilate":
        fg = binary_dilation(fg)
    elif params["thickness_op"] == "erode":
        fg = binary_erosion(fg)
    out = np.where(fg, np.minimum(out, 0.15), out)

    noise = rng.normal(0.0, float(params["noise_std"]), size=out.shape).astype(np.float32)
    out = np.clip(out + noise, 0.0, 1.0)
    return out.astype(np.float32)


def build_transformed_eval_set(
    templates_u8: np.ndarray,
    per_class: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    base = templates_u8.astype(np.float32) / 255.0

    images = []
    labels17 = []
    logs = []

    for class_id, template in enumerate(base):
        for sample_idx in range(per_class):
            params = {
                "class_id": int(class_id),
                "sample_idx": int(sample_idx),
                "rotation_deg": float(rng.uniform(-8.0, 8.0)),
                "stretch_x": float(rng.uniform(0.92, 1.08)),
                "stretch_y": float(rng.uniform(0.92, 1.08)),
                "thickness_op": str(rng.choice(["none", "dilate", "erode"], p=[0.4, 0.3, 0.3])),
                "noise_std": float(rng.uniform(0.01, 0.03)),
            }
            transformed = _apply_controlled_transform(template, params, rng)
            images.append(transformed)
            labels17.append(class_id)
            logs.append(params)

    x = np.stack(images, axis=0)
    y17 = np.array(labels17, dtype=np.int64)
    return x, y17, logs


def to_loader(images, labels, batch_size=128, shuffle=False):
    x = torch.tensor(images, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(labels, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def eval_on_mnist(model, test_images, test_labels, device):
    model.eval()
    x = torch.tensor(test_images, dtype=torch.float32).unsqueeze(1).to(device) / 255.0
    with torch.no_grad():
        pred17 = model(x).argmax(dim=1).cpu().numpy()
    pred10 = np.array([CLASS17_TO_DIGIT10[int(c)] for c in pred17], dtype=np.int64)
    acc = (pred10 == test_labels).mean()
    return float(acc)


def eval_17class(model, images01, labels17, device):
    model.eval()
    x = torch.tensor(images01, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        pred17 = model(x).argmax(dim=1).cpu().numpy()
    acc = (pred17 == labels17).mean()
    return float(acc)


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_templates = np.load(os.path.join(args.data_dir, "train_images.npy"))
    train_labels = np.load(os.path.join(args.data_dir, "train_labels17.npy"))
    aug_x, aug_y = augment_templates(train_templates, train_labels, repeats=args.repeats, noise_std=args.noise_std)

    transformed_x, transformed_y17, transform_logs = build_transformed_eval_set(
        train_templates,
        per_class=args.eval_transforms_per_class,
        seed=args.seed,
    )

    _, test_labels = load_mnist_idx(args.mnist_path, "t10k")
    test_images, _ = load_mnist_idx(args.mnist_path, "t10k")

    train_loader = to_loader(aug_x, aug_y, batch_size=args.batch_size, shuffle=True)

    model = SimpleCNN(num_classes=17).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    ensure_dir(args.out_dir)
    eval_dir = os.path.join(args.out_dir, "eval_transformed")
    ensure_dir(eval_dir)
    np.save(os.path.join(eval_dir, "images.npy"), transformed_x)
    np.save(os.path.join(eval_dir, "labels17.npy"), transformed_y17)
    with open(os.path.join(eval_dir, "transformations.json"), "w", encoding="utf-8") as f:
        json.dump(transform_logs, f, indent=2)

    best_acc = 0.0
    best_path = os.path.join(args.out_dir, "best_variants17_cnn.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        acc = eval_on_mnist(model, test_images, test_labels, device)
        acc_transformed = eval_17class(model, transformed_x, transformed_y17, device)
        print(
            f"[variants17_cnn] epoch={epoch} "
            f"mnist_test_acc={acc * 100:.2f}% transformed17_acc={acc_transformed * 100:.2f}%"
        )
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_path)

    print(f"[variants17_cnn] best_mnist_test_acc={best_acc * 100:.2f}% saved={best_path}")
    print(
        "[variants17_cnn] transformed_eval_saved="
        f"{eval_dir} count={len(transformed_y17)} "
        f"({args.eval_transforms_per_class} per class)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-path", default="mnist_data")
    parser.add_argument("--data-dir", default="data/processed/mnist17_variants")
    parser.add_argument("--out-dir", default="experiments/checkpoints/variants17")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--repeats", type=int, default=256)
    parser.add_argument("--noise-std", type=float, default=0.04)
    parser.add_argument("--eval-transforms-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
