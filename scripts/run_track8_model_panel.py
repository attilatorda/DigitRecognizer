"""
Track 8 (#1 + #2): read meaning from the DIFFERENCES between competing networks.

A panel of three classifiers with increasing spatial structure is trained on MNIST, then we
ask what their differences *encode*:
  - MLP        : flattened pixels, no spatial prior
  - SimpleCNN  : 2 conv blocks + pooling (translation-tolerant)
  - RecordM3   : the An et al. record-holder single model (10 conv layers, no pooling) -- "the
                 best digit net, from another project" (github ansh941/MnistSimpleCNN)

We measure, over several seeds:
  (a) clean test accuracy;
  (b) ROBUSTNESS to translation and rotation -- the quantitative reading of "model A > model B":
      the MLP collapses under a few-pixel shift while the CNNs do not (translation invariance is
      what the convolutional prior buys); the deeper record model degrades least;
  (c) how much of an unsupervised STYLE variant (crossed-7) survives in each representation
      (linear probe) -- the stronger/deeper net preserves more structure;
  (d) pairwise DISAGREEMENT and the shift-sensitivity of each model's unique errors.

Usage:  python scripts/run_track8_model_panel.py [--smoke]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms.functional import affine

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.common.data_io import load_mnist_idx
from src.common.utils import set_seed
from src.local_cnn.model import SimpleCNN
from src.track9.record_models import RecordModel
from scripts.probe_variant_recovery import variant_labels


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU())
        self.head = nn.Linear(128, 10)

    def forward(self, x):
        return self.head(self.net(x))

    def embed(self, x):
        return self.net(x)


def _tensor(images_u8):
    return torch.tensor(images_u8.astype(np.float32) / 255.0).unsqueeze(1)


def make_model(kind):
    return {"mlp": MLP, "simplecnn": lambda: SimpleCNN(10, 1),
            "recordM3": lambda: RecordModel(3)}[kind]()


def train(model, X, y, device, epochs, lr=1e-3):
    model.to(device).train()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, torch.tensor(y, dtype=torch.long)),
        batch_size=128, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
    return model


@torch.no_grad()
def predict(model, X, device, bs=1000):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(model(X[i:i + bs].to(device)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def _embed_one(model, xb):
    """Penultimate features, dispatched by model type."""
    if isinstance(model, MLP):
        return model.embed(xb)
    if isinstance(model, SimpleCNN):
        return model.classifier[:3](model.features(xb))     # Flatten->Linear->ReLU = 128-d
    return model.features(xb).flatten(1)                     # RecordModel conv features


@torch.no_grad()
def embed(model, X, device, bs=1000):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(_embed_one(model, X[i:i + bs].to(device)).cpu().numpy())
    return np.concatenate(out).reshape(len(X), -1)


def perturbed(X, translate=(0, 0), angle=0.0):
    return affine(X, angle=angle, translate=list(translate), scale=1.0, shear=[0.0, 0.0])


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[panel] device={device}", flush=True)
    raw_tr, y_tr = load_mnist_idx(os.path.join(ROOT, "mnist_data"), "train")
    raw_te, y_te = load_mnist_idx(os.path.join(ROOT, "mnist_data"), "t10k")
    if args.smoke:
        raw_tr, y_tr = raw_tr[:6000], y_tr[:6000]
        raw_te, y_te = raw_te[:2000], y_te[:2000]
    Xtr, Xte = _tensor(raw_tr), _tensor(raw_te)

    kinds = ["mlp", "simplecnn", "recordM3"]
    epochs = {"mlp": 15, "simplecnn": 8, "recordM3": 25}
    if args.smoke:
        epochs = {k: 2 for k in kinds}
    seeds = 1 if args.smoke else args.seeds
    shifts = [0, 1, 2, 3, 4]
    angles = [0, 5, 10, 15, 20]

    # accumulators
    acc = {k: [] for k in kinds}
    rob_t = {k: [[] for _ in shifts] for k in kinds}
    rob_r = {k: [[] for _ in angles] for k in kinds}
    probe = {k: [] for k in kinds}
    preds_last = {}

    # digit-7 variant labels for the representation probe (computed once)
    m7 = y_te == 7
    v7 = variant_labels(raw_te[m7], 7)

    for s in range(seeds):
        for k in kinds:
            set_seed(s)
            model = train(make_model(k), Xtr, y_tr, device, epochs[k])
            p = predict(model, Xte, device)
            acc[k].append(float((p == y_te).mean()))
            preds_last[k] = p
            for i, dx in enumerate(shifts):
                Xp = perturbed(Xte, translate=(dx, 0))
                rob_t[k][i].append(float((predict(model, Xp, device) == y_te).mean()))
            for i, ang in enumerate(angles):
                Xp = perturbed(Xte, angle=ang)
                rob_r[k][i].append(float((predict(model, Xp, device) == y_te).mean()))
            e7 = embed(model, Xte[m7], device)
            probe[k].append(float(cross_val_score(
                LogisticRegression(max_iter=2000), e7, v7, cv=3).mean()))
            print(f"  seed {s} {k:10s} acc={acc[k][-1]*100:.2f}  "
                  f"shift4={rob_t[k][-1][-1]*100:.1f}  rot20={rob_r[k][-1][-1]*100:.1f}  "
                  f"probe7={probe[k][-1]*100:.1f}", flush=True)

    def ms(d):
        return float(np.mean(d)), float(np.std(d))

    base7 = max(v7.mean(), 1 - v7.mean())
    summary = {"seeds": seeds, "shifts": shifts, "angles": angles,
               "variant7_base_rate": float(base7),
               "accuracy": {k: ms(acc[k]) for k in kinds},
               "robust_translate": {k: [ms(rob_t[k][i]) for i in range(len(shifts))] for k in kinds},
               "robust_rotate": {k: [ms(rob_r[k][i]) for i in range(len(angles))] for k in kinds},
               "variant7_probe": {k: ms(probe[k]) for k in kinds}}

    # disagreement (last seed): pairwise + MLP-unique-error shift sensitivity
    dis = {}
    for a in range(len(kinds)):
        for b in range(a + 1, len(kinds)):
            ka, kb = kinds[a], kinds[b]
            dis[f"{ka}_vs_{kb}"] = float((preds_last[ka] != preds_last[kb]).mean())
    summary["pairwise_disagreement"] = dis

    print("\n=== summary (mean over seeds) ===")
    for k in kinds:
        a, _ = summary["accuracy"][k]
        t4 = summary["robust_translate"][k][-1][0]
        r20 = summary["robust_rotate"][k][-1][0]
        pr = summary["variant7_probe"][k][0]
        print(f"  {k:10s} acc={a*100:5.2f}  acc@shift4={t4*100:5.1f} (drop {(a-t4)*100:4.1f})  "
              f"acc@rot20={r20*100:5.1f}  variant7-probe={pr*100:.1f} (base {base7*100:.0f})")

    # --- figures ---
    colors = {"mlp": "#c0392b", "simplecnn": "#2980b9", "recordM3": "#27ae60"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.9))
    for k in kinds:
        mt = [summary["robust_translate"][k][i][0] * 100 for i in range(len(shifts))]
        a1.plot(shifts, mt, "o-", color=colors[k], label=k)
        mr = [summary["robust_rotate"][k][i][0] * 100 for i in range(len(angles))]
        a2.plot(angles, mr, "o-", color=colors[k], label=k)
    a1.set_xlabel("horizontal shift (px)"); a1.set_ylabel("test accuracy (%)")
    a1.set_title("Translation robustness: the MLP collapses,\nthe CNNs (esp. record model) hold")
    a1.legend(fontsize=8); a1.grid(True, ls="--", alpha=0.4)
    a2.set_xlabel("rotation (deg)"); a2.set_ylabel("test accuracy (%)")
    a2.set_title("Rotation robustness")
    a2.legend(fontsize=8); a2.grid(True, ls="--", alpha=0.4)
    fig.suptitle("Track 8 #2: reading meaning from model differences", fontsize=12)
    plt.tight_layout()
    fp = os.path.join(ROOT, "experiments", "reports", "figures", "fig_track8_panel_robust.png")
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    plt.savefig(fp, dpi=180, facecolor="white"); plt.close()
    print(f"[panel] saved {fp}")

    if not args.smoke:
        out = os.path.join(ROOT, "experiments", "reports", "track8_panel_results.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[panel] saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seeds", type=int, default=3)
    main(p.parse_args())
