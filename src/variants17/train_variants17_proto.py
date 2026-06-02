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
from src.common.utils import ensure_dir, set_seed
from src.local_cnn.model import SimpleCNN
from src.variants17.augment import augment_dataset
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


def make_loader(images, labels, batch_size=128, shuffle=True):
    x = torch.tensor(images, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(labels, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def compute_prototypes(model, support_x01, support_y17, device):
    model.eval()
    x = torch.tensor(support_x01, dtype=torch.float32).unsqueeze(1).to(device)
    y = torch.tensor(support_y17, dtype=torch.long).to(device)
    with torch.no_grad():
        z = model(x)
    return torch.stack([z[y == c].mean(dim=0) for c in range(17)], dim=0)


def classify_with_prototypes(model, images01, prototypes, device):
    model.eval()
    x = torch.tensor(images01, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        z = model(x)
        pred = torch.cdist(z, prototypes).argmin(dim=1)
    return pred.cpu().numpy()


def eval_mnist_proj(model, images_u8, labels10, prototypes, device, batch_size=512):
    """Evaluate on raw uint8 MNIST images. Returns accuracy."""
    correct = total = 0
    dataset = TensorDataset(
        torch.tensor(images_u8, dtype=torch.float32).unsqueeze(1) / 255.0,
        torch.tensor(labels10, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            z = model(x)
            pred17 = torch.cdist(z, prototypes).argmin(dim=1).cpu().numpy()
            pred10 = np.array([CLASS17_TO_DIGIT10[int(c)] for c in pred17])
            correct += (pred10 == y.numpy()).sum()
            total += len(y)
    return correct / max(1, total)


def get_mnist_predictions(model, images_u8, prototypes, device, batch_size=512):
    """Return (pred10, pred17) arrays for all MNIST images (for confusion matrix analysis)."""
    pred10_all, pred17_all = [], []
    loader = DataLoader(
        TensorDataset(torch.tensor(images_u8, dtype=torch.float32).unsqueeze(1) / 255.0),
        batch_size=batch_size, shuffle=False,
    )
    model.eval()
    with torch.no_grad():
        for (x,) in loader:
            z = model(x.to(device))
            p17 = torch.cdist(z, prototypes).argmin(dim=1).cpu().numpy()
            p10 = np.array([CLASS17_TO_DIGIT10[int(c)] for c in p17])
            pred17_all.append(p17)
            pred10_all.append(p10)
    return np.concatenate(pred10_all), np.concatenate(pred17_all)


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    elastic_prob = 0.0 if args.no_augmentation else 0.70
    stroke_prob = 0.0 if args.no_augmentation else 0.80

    train_templates = np.load(os.path.join(args.data_dir, "train_images.npy"))
    train_labels17 = np.load(os.path.join(args.data_dir, "train_labels17.npy"))
    aug_x, aug_y = augment_dataset(
        train_templates, train_labels17,
        repeats=args.repeats, seed=args.seed,
        elastic_prob=elastic_prob, stroke_prob=stroke_prob,
    )
    train_loader = make_loader(aug_x, aug_y, batch_size=args.batch_size, shuffle=True)

    test_images, test_labels = load_mnist_idx(args.mnist_path, "t10k")
    train_images_mnist = train_labels_mnist = None
    if args.eval_mnist_train:
        train_images_mnist, train_labels_mnist = load_mnist_idx(args.mnist_path, "train")

    model = EmbeddingCNN(emb_dim=args.emb_dim).to(device)
    clf = nn.Linear(args.emb_dim, 17).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(clf.parameters()), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    ensure_dir(args.out_dir)
    best_acc = 0.0
    best_path = os.path.join(args.out_dir, "best_variants17_proto.pt")
    mn = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        clf.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = ce(clf(model(xb)), yb)
            loss.backward()
            opt.step()

        support_x01 = train_templates.astype(np.float32) / 255.0
        prototypes = compute_prototypes(model, support_x01, train_labels17, device)
        mn = eval_mnist_proj(model, test_images, test_labels, prototypes, device)
        print(f"[variants17_proto] epoch={epoch} mnist_test_acc={mn*100:.2f}%")

        if mn > best_acc:
            best_acc = mn
            torch.save({"model": model.state_dict(), "clf": clf.state_dict()}, best_path)

    print(f"[variants17_proto] best_mnist_test_acc={best_acc*100:.2f}% saved={best_path}")

    if args.eval_mnist_train and train_images_mnist is not None and train_labels_mnist is not None:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        support_x01 = train_templates.astype(np.float32) / 255.0
        prototypes = compute_prototypes(model, support_x01, train_labels17, device)
        acc_train = eval_mnist_proj(model, train_images_mnist, train_labels_mnist, prototypes, device)
        print(f"[variants17_proto] mnist_train_acc={acc_train*100:.2f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mnist-path", default="mnist_data")
    p.add_argument("--data-dir", default="data/processed/mnist17_variants")
    p.add_argument("--out-dir", default="experiments/checkpoints/variants17")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--repeats", type=int, default=256)
    p.add_argument("--emb-dim", type=int, default=64)
    p.add_argument("--no-augmentation", action="store_true",
                   help="Disable elastic and stroke-width augmentation (noise-only baseline)")
    p.add_argument("--eval-mnist-train", action="store_true",
                   help="Also evaluate on the 60K MNIST training split at end")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
