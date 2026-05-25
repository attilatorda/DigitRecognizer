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

from src.common.data_io import load_mnist_idx
from src.common.utils import set_seed
from src.local_cnn.model import SimpleCNN
from src.variants17.label_schema import CLASS17_TO_DIGIT10


class EmbeddingCNN(nn.Module):
    def __init__(self, emb_dim: int = 64):
        super().__init__()
        base = SimpleCNN(num_classes=17)
        self.features = base.features
        self.embed_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, emb_dim),
        )

    def forward(self, x):
        z = self.features(x)
        z = self.embed_head(z)
        return nn.functional.normalize(z, dim=1)


def augment_templates(images: np.ndarray, labels: np.ndarray, repeats: int, noise_std: float, seed: int):
    rng = np.random.default_rng(seed)
    x = images.astype(np.float32) / 255.0
    X, Y = [], []
    for img, y in zip(x, labels):
        for _ in range(repeats):
            jitter = rng.normal(0.0, noise_std, size=img.shape).astype(np.float32)
            sample = np.clip(img + jitter, 0.0, 1.0)
            X.append(sample)
            Y.append(y)
    return np.stack(X, axis=0), np.array(Y, dtype=np.int64)


def make_loader(images, labels, batch_size=128, shuffle=True):
    x = torch.tensor(images, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(labels, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def compute_prototypes(model: nn.Module, support_x01: np.ndarray, support_y17: np.ndarray, device):
    model.eval()
    x = torch.tensor(support_x01, dtype=torch.float32).unsqueeze(1).to(device)
    y = torch.tensor(support_y17, dtype=torch.long).to(device)
    with torch.no_grad():
        z = model(x)
    protos = []
    for c in range(17):
        protos.append(z[y == c].mean(dim=0))
    return torch.stack(protos, dim=0)


def classify_with_prototypes(model, images01: np.ndarray, prototypes: torch.Tensor, device):
    model.eval()
    x = torch.tensor(images01, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        z = model(x)
        d = torch.cdist(z, prototypes)
        pred = d.argmin(dim=1)
    return pred.cpu().numpy()


def eval_mnist_proj(model, test_images_u8, test_labels10, prototypes, device):
    pred17 = classify_with_prototypes(model, test_images_u8.astype(np.float32) / 255.0, prototypes, device)
    pred10 = np.array([CLASS17_TO_DIGIT10[int(c)] for c in pred17], dtype=np.int64)
    return float((pred10 == test_labels10).mean())


def eval_transformed17(model, transformed_x01, transformed_y17, prototypes, device):
    pred17 = classify_with_prototypes(model, transformed_x01, prototypes, device)
    return float((pred17 == transformed_y17).mean())


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_templates = np.load(os.path.join(args.data_dir, "train_images.npy"))
    train_labels17 = np.load(os.path.join(args.data_dir, "train_labels17.npy"))
    aug_x, aug_y = augment_templates(train_templates, train_labels17, args.repeats, args.noise_std, args.seed)
    train_loader = make_loader(aug_x, aug_y, batch_size=args.batch_size, shuffle=True)

    transformed_x = np.load(os.path.join(args.transformed_dir, "images.npy"))
    transformed_y = np.load(os.path.join(args.transformed_dir, "labels17.npy"))
    mnist_images, mnist_labels = load_mnist_idx(args.mnist_path, "t10k")

    model = EmbeddingCNN(emb_dim=args.emb_dim).to(device)
    clf = nn.Linear(args.emb_dim, 17).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(clf.parameters()), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        clf.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            emb = model(xb)
            logits = clf(emb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()

        support_x01 = train_templates.astype(np.float32) / 255.0
        prototypes = compute_prototypes(model, support_x01, train_labels17, device)
        mn = eval_mnist_proj(model, mnist_images, mnist_labels, prototypes, device)
        tr = eval_transformed17(model, transformed_x, transformed_y, prototypes, device)
        print(f"[variants17_proto] epoch={epoch} mnist_test_acc={mn*100:.2f}% transformed17_acc={tr*100:.2f}%")

    print(f"[variants17_proto] final_mnist_test_acc={mn*100:.2f}% final_transformed17_acc={tr*100:.2f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mnist-path", default="mnist_data")
    p.add_argument("--data-dir", default="data/processed/mnist17_variants")
    p.add_argument("--transformed-dir", default="experiments/checkpoints/variants17/eval_transformed")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--repeats", type=int, default=256)
    p.add_argument("--noise-std", type=float, default=0.04)
    p.add_argument("--emb-dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
