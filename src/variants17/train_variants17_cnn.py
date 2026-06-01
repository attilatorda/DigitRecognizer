import argparse
import json
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.common.data_io import load_mnist_idx
from src.common.utils import ensure_dir, set_seed
from src.local_cnn.model import SimpleCNN
from src.variants17.augment import apply_augmentation, augment_dataset, sample_augmentation_params
from src.variants17.label_schema import CLASS17_TO_DIGIT10


def build_transformed_eval_set(
    templates_u8: np.ndarray,
    per_class: int,
    seed: int,
    elastic_prob: float = 0.70,
    stroke_prob: float = 0.80,
):
    rng = np.random.default_rng(seed)
    base = templates_u8.astype(np.float32) / 255.0
    images, labels17, logs = [], [], []
    for class_id, template in enumerate(base):
        for sample_idx in range(per_class):
            params = sample_augmentation_params(rng, elastic_prob=elastic_prob, stroke_prob=stroke_prob)
            params["class_id"] = int(class_id)
            params["sample_idx"] = int(sample_idx)
            images.append(apply_augmentation(template, params, rng))
            labels17.append(class_id)
            logs.append(params)
    return np.stack(images, axis=0), np.array(labels17, dtype=np.int64), logs


def to_loader(images, labels, batch_size=128, shuffle=False):
    x = torch.tensor(images, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(labels, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def eval_on_mnist(model, images_u8, labels, device, batch_size=512):
    """Evaluate model on raw uint8 MNIST images. Returns accuracy."""
    model.eval()
    correct = total = 0
    dataset = TensorDataset(
        torch.tensor(images_u8, dtype=torch.float32).unsqueeze(1) / 255.0,
        torch.tensor(labels, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred17 = model(x).argmax(dim=1).cpu().numpy()
            pred10 = np.array([CLASS17_TO_DIGIT10[int(c)] for c in pred17], dtype=np.int64)
            correct += (pred10 == y.cpu().numpy()).sum()
            total += len(y)
    return correct / max(1, total)


def eval_17class(model, images01, labels17, device):
    model.eval()
    x = torch.tensor(images01, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        pred17 = model(x).argmax(dim=1).cpu().numpy()
    return float((pred17 == labels17).mean())


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --no-augmentation zeroes structural augmentation; keeps tiny noise for variety
    elastic_prob = 0.0 if args.no_augmentation else args.elastic_prob
    stroke_prob = 0.0 if args.no_augmentation else args.stroke_prob

    train_templates = np.load(os.path.join(args.data_dir, "train_images.npy"))
    train_labels = np.load(os.path.join(args.data_dir, "train_labels17.npy"))
    aug_x, aug_y = augment_dataset(
        train_templates, train_labels,
        repeats=args.repeats, seed=args.seed,
        elastic_prob=elastic_prob, stroke_prob=stroke_prob,
    )

    transformed_x, transformed_y17, transform_logs = build_transformed_eval_set(
        train_templates, per_class=args.eval_transforms_per_class,
        seed=args.seed, elastic_prob=elastic_prob, stroke_prob=stroke_prob,
    )

    test_images, test_labels = load_mnist_idx(args.mnist_path, "t10k")
    train_images_mnist = train_labels_mnist = None
    if args.eval_mnist_train:
        train_images_mnist, train_labels_mnist = load_mnist_idx(args.mnist_path, "train")

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

    if args.eval_mnist_train and train_images_mnist is not None and train_labels_mnist is not None:
        model.load_state_dict(torch.load(best_path, map_location=device))
        acc_train = eval_on_mnist(model, train_images_mnist, train_labels_mnist, device)
        print(f"[variants17_cnn] mnist_train_acc={acc_train * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-path", default="mnist_data")
    parser.add_argument("--data-dir", default="data/processed/mnist17_variants")
    parser.add_argument("--out-dir", default="experiments/checkpoints/variants17")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--repeats", type=int, default=256)
    parser.add_argument("--eval-transforms-per-class", type=int, default=20)
    parser.add_argument("--elastic-prob", type=float, default=0.70)
    parser.add_argument("--stroke-prob", type=float, default=0.80)
    parser.add_argument("--no-augmentation", action="store_true",
                        help="Disable elastic and stroke-width augmentation (noise-only baseline)")
    parser.add_argument("--eval-mnist-train", action="store_true",
                        help="Also evaluate on the 60K MNIST training split at end")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
