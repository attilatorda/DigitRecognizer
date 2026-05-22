import argparse
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.common.utils import set_seed, ensure_dir
from src.local_cnn.model import SimpleCNN


def to_loader(images, labels, batch_size=128, shuffle=False):
    x = torch.tensor(images, dtype=torch.float32).unsqueeze(1) / 255.0
    y = torch.tensor(labels, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(1, total)


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_images = np.load(os.path.join(args.data_dir, "train_images.npy"))
    train_labels = np.load(os.path.join(args.data_dir, "train_labels14.npy"))
    test_images = np.load(os.path.join(args.data_dir, "t10k_images.npy"))
    test_labels = np.load(os.path.join(args.data_dir, "t10k_labels14.npy"))

    train_loader = to_loader(train_images, train_labels, args.batch_size, shuffle=True)
    test_loader = to_loader(test_images, test_labels, args.batch_size, shuffle=False)

    model = SimpleCNN(num_classes=14).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    ensure_dir(args.out_dir)
    best_acc = 0.0
    best_path = os.path.join(args.out_dir, "best_variants14_cnn.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        acc = evaluate(model, test_loader, device)
        print(f"[variants14_cnn] epoch={epoch} test_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_path)

    print(f"[variants14_cnn] best_test_acc={best_acc:.4f} saved={best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/mnist14_variants")
    parser.add_argument("--out-dir", default="experiments/checkpoints/variants14")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
