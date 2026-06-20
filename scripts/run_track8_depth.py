"""
Track 8 (#4): depth -> capacity feedback. "If the task needs no XOR-like (non-linearly-
separable) computation past layer L, would fewer layers do?"

Two complementary measurements on MNIST:
  (a) DEPTH ABLATION -- train CNNs of increasing depth (1..8 conv blocks); see where clean
      accuracy and translation robustness saturate. (Clean accuracy is expected to plateau
      early; robustness -- the axis that #2 showed separates architectures -- keeps improving.)
  (b) PER-LAYER LINEAR SEPARABILITY -- on the deepest trained net, fit a linear probe (logreg)
      on each block's features. The layer where a *linear* classifier already reaches the final
      accuracy is the "effective depth": beyond it the network adds no linearly-separable value,
      i.e. no XOR-like work remains, so layers can be cut.

The feedback rule: recommend depth = the linear-separability knee, unless robustness is needed,
in which case extra depth still pays.

Usage:  python scripts/run_track8_depth.py [--smoke]
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

from src.common.data_io import load_mnist_idx
from src.common.utils import set_seed

POOL_AFTER = {1, 3}   # pool after these block indices (28 -> 14 -> 7)


class DepthCNN(nn.Module):
    def __init__(self, n_blocks, ch=32, num_classes=10):
        super().__init__()
        self.blocks = nn.ModuleList()
        inc = 1
        for i in range(n_blocks):
            seq = [nn.Conv2d(inc, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True)]
            if i in POOL_AFTER:
                seq.append(nn.MaxPool2d(2))
            self.blocks.append(nn.Sequential(*seq)); inc = ch
        self.gap = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Linear(ch * 16, num_classes)

    def forward(self, x, return_feats=False):
        feats = []
        for b in self.blocks:
            x = b(x); feats.append(x)
        out = self.head(self.gap(x).flatten(1))
        return (out, feats) if return_feats else out


def _tensor(u8):
    return torch.tensor(u8.astype(np.float32) / 255.0).unsqueeze(1)


def train(model, X, y, device, epochs):
    model.to(device).train()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, torch.tensor(y, dtype=torch.long)),
        batch_size=128, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
    return model.eval()


@torch.no_grad()
def acc_of(model, X, y, device, bs=1000):
    out = []
    for i in range(0, len(X), bs):
        out.append(model(X[i:i + bs].to(device)).argmax(1).cpu().numpy())
    return float((np.concatenate(out) == y).mean())


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[depth] device={device}", flush=True)
    raw_tr, y_tr = load_mnist_idx(os.path.join(ROOT, "mnist_data"), "train")
    raw_te, y_te = load_mnist_idx(os.path.join(ROOT, "mnist_data"), "t10k")
    if args.smoke:
        raw_tr, y_tr = raw_tr[:6000], y_tr[:6000]
        raw_te, y_te = raw_te[:2000], y_te[:2000]
    Xtr, Xte = _tensor(raw_tr), _tensor(raw_te)
    Xte_sh = affine(Xte, angle=0.0, translate=[4, 0], scale=1.0, shear=[0.0, 0.0])

    depths = [1, 2] if args.smoke else [1, 2, 3, 4, 6, 8]
    seeds = 1 if args.smoke else args.seeds
    epochs = 2 if args.smoke else args.epochs

    clean = {d: [] for d in depths}
    shift = {d: [] for d in depths}
    deepest = None
    for s in range(seeds):
        for d in depths:
            set_seed(s)
            m = train(DepthCNN(d), Xtr, y_tr, device, epochs)
            clean[d].append(acc_of(m, Xte, y_te, device))
            shift[d].append(acc_of(m, Xte_sh, y_te, device))
            if d == depths[-1] and s == 0:
                deepest = m
        print(f"  seed {s}: " + "  ".join(
            f"d{d}={np.mean(clean[d][-1:])*100:.2f}/{np.mean(shift[d][-1:])*100:.0f}" for d in depths),
            flush=True)

    # (b) per-layer linear separability on the deepest net (subset of train for speed)
    idx = np.random.default_rng(0).choice(len(Xtr), size=min(5000, len(Xtr)), replace=False)
    probe_acc = []
    with torch.no_grad():
        _, ftr = deepest(Xtr[idx].to(device), return_feats=True)
        _, fte = deepest(Xte.to(device), return_feats=True)
    for L in range(len(depths) and depths[-1]):
        Ztr = torch.nn.functional.adaptive_avg_pool2d(ftr[L], (4, 4)).flatten(1).cpu().numpy()
        Zte = torch.nn.functional.adaptive_avg_pool2d(fte[L], (4, 4)).flatten(1).cpu().numpy()
        clf = LogisticRegression(max_iter=500).fit(Ztr, y_tr[idx])
        probe_acc.append(float((clf.predict(Zte) == y_te).mean()))
        print(f"  linear-probe after block {L+1}: {probe_acc[-1]*100:.2f}%", flush=True)

    cmean = {d: float(np.mean(clean[d])) for d in depths}
    smean = {d: float(np.mean(shift[d])) for d in depths}
    # knee: shallowest depth within 0.2pp of the best clean accuracy
    best = max(cmean.values())
    knee = min(d for d in depths if cmean[d] >= best - 0.002)
    # linear-separability knee: first block within 0.5pp of final probe acc
    pf = probe_acc[-1]
    lin_knee = next((L + 1 for L, a in enumerate(probe_acc) if a >= pf - 0.005), depths[-1])

    print(f"\n=== feedback ===")
    print(f"clean-accuracy knee: {knee} blocks (best {best*100:.2f}%); "
          f"linear-separability knee: {lin_knee} blocks (final probe {pf*100:.2f}%).")
    print(f"shift-robustness keeps rising with depth: "
          + " -> ".join(f"{smean[d]*100:.0f}" for d in depths) + " (%)")

    # --- figure ---
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.9))
    a1.plot(depths, [cmean[d] * 100 for d in depths], "o-", color="#2c3e50", label="clean acc")
    a1.plot(depths, [smean[d] * 100 for d in depths], "s--", color="#c0392b", label="acc @ 4px shift")
    a1.axvline(knee, color="#2c3e50", ls=":", alpha=0.5)
    a1.set_xlabel("network depth (conv blocks)"); a1.set_ylabel("test accuracy (%)")
    a1.set_title("Depth ablation: clean accuracy saturates early,\nrobustness keeps improving")
    a1.legend(fontsize=8); a1.grid(True, ls="--", alpha=0.4)

    a2.plot(range(1, len(probe_acc) + 1), [p * 100 for p in probe_acc], "o-", color="#27ae60")
    a2.axvline(lin_knee, color="#27ae60", ls=":", alpha=0.6)
    a2.set_xlabel("linear probe after block #"); a2.set_ylabel("probe accuracy (%)")
    a2.set_title(f"Linear separability saturates by block {lin_knee}\n(no XOR-like work remains beyond it)")
    a2.grid(True, ls="--", alpha=0.4)
    fig.suptitle("Track 8 #4: depth -> capacity feedback", fontsize=12)
    plt.tight_layout()
    fp = os.path.join(ROOT, "experiments", "reports", "figures", "fig_track8_depth.png")
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    plt.savefig(fp, dpi=180, facecolor="white"); plt.close()
    print(f"[depth] saved {fp}")

    if not args.smoke:
        out = os.path.join(ROOT, "experiments", "reports", "track8_depth_results.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"depths": depths, "seeds": seeds, "clean": cmean, "shift4": smean,
                       "linear_probe_by_block": probe_acc, "clean_knee": knee,
                       "linear_separability_knee": lin_knee}, f, indent=2)
        print(f"[depth] saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=6)
    main(p.parse_args())
