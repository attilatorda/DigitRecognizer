"""MNIST CNN record holder (light re-implementation) — Track 9 opponent.

Faithful re-implementation of An et al. 2020, "An Ensemble of Simple Convolutional
Neural Network Models for MNIST Digit Recognition" (arXiv:2008.10400,
github.com/ansh941/MnistSimpleCNN). Published result: a majority vote of the three
models below reaches 99.87% on MNIST (60k train); a two-level ensemble reaches 99.91%.

Each model is a stack of valid (no-padding) conv layers with BatchNorm + ReLU, no
pooling, followed by a single fully-connected layer and a BatchNorm1d, then log_softmax:

  M3: kernel 3, 10 conv layers, channels 32,48,...,176  -> 8x8 feature map
  M5: kernel 5,  5 conv layers, channels 32,64,...,160   -> 8x8 feature map
  M7: kernel 7,  4 conv layers, channels 48,96,144,192   -> 4x4 feature map

This is the LIGHT repro used as the Track 9 opponent: the architecture is faithful, but
we train one model per kernel for a reduced epoch budget (not the paper's 150-epoch,
multi-seed homogeneous sub-ensembles). Accuracy therefore lands below the published
99.87%; we report what we measure and cite the published record.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Per-kernel channel progressions, read from the reference implementation.
_CHANNELS = {
    3: [32, 48, 64, 80, 96, 112, 128, 144, 160, 176],
    5: [32, 64, 96, 128, 160],
    7: [48, 96, 144, 192],
}


class RecordModel(nn.Module):
    """One An et al. "simple" CNN for a given kernel size (3, 5, or 7)."""

    def __init__(self, kernel_size: int, num_classes: int = 10):
        super().__init__()
        if kernel_size not in _CHANNELS:
            raise ValueError(f"kernel_size must be one of {sorted(_CHANNELS)}")
        chans = _CHANNELS[kernel_size]
        layers = []
        in_c = 1
        for out_c in chans:
            layers += [
                nn.Conv2d(in_c, out_c, kernel_size, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ]
            in_c = out_c
        self.features = nn.Sequential(*layers)
        # Infer the flattened feature dimension with a dry run on a 28x28 input.
        with torch.no_grad():
            flat = self.features(torch.zeros(1, 1, 28, 28)).flatten(1).shape[1]
        self.fc = nn.Linear(flat, num_classes, bias=False)
        self.bn_out = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        x = (x - 0.5) * 2.0  # match the reference normalization to [-1, 1]
        x = self.features(x)
        x = x.flatten(1)
        x = self.bn_out(self.fc(x))
        return torch.log_softmax(x, dim=1)


# ---------------------------------------------------------------------------
# Training / inference
# ---------------------------------------------------------------------------

def _augment(xb: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Light rotation + translation augmentation (the paper's augmentation family).

    Random rotation in +/-15 deg and integer-ish shift up to +/-2 px, applied per batch
    with a single affine grid_sample. Operates on (B,1,28,28) tensors in [0,1].
    """
    b = xb.shape[0]
    device = xb.device
    deg = (torch.rand(b, generator=rng, device=device) * 2 - 1) * (15.0 * np.pi / 180.0)
    cos, sin = torch.cos(deg), torch.sin(deg)
    # max shift = 2px on a 28px image -> normalized coord 2/14 ~= 0.143
    tx = (torch.rand(b, generator=rng, device=device) * 2 - 1) * (2.0 / 14.0)
    ty = (torch.rand(b, generator=rng, device=device) * 2 - 1) * (2.0 / 14.0)
    theta = torch.zeros(b, 2, 3, device=device)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, 1, 2] = ty
    grid = torch.nn.functional.affine_grid(theta, xb.shape, align_corners=False)
    return torch.nn.functional.grid_sample(xb, grid, align_corners=False, padding_mode="zeros")


def train_record_model(model, images_u8, labels, device, epochs=40, batch_size=128,
                       lr=1e-3, augment=True, seed=0):
    """Train one RecordModel on (N,28,28) uint8 images. fp32 (no AMP)."""
    rng = torch.Generator(device=device).manual_seed(seed)
    X = torch.tensor(images_u8.astype(np.float32) / 255.0).unsqueeze(1)
    y = torch.tensor(labels, dtype=torch.long)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.NLLLoss()  # model outputs log-probabilities
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if augment:
                xb = _augment(xb, rng)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
    return model


@torch.no_grad()
def _predict_probs(model, images_u8, device, batch_size=512):
    model.eval()
    X = torch.tensor(images_u8.astype(np.float32) / 255.0).unsqueeze(1)
    out = []
    for i in range(0, len(X), batch_size):
        logp = model(X[i:i + batch_size].to(device))
        out.append(torch.exp(logp).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


class RecordHolderEnsemble:
    """An et al. heterogeneous M3/M5/M7 majority-vote ensemble (light repro).

    fit(images_u8, labels) trains one model per kernel on the SAME data (matched-budget,
    so it competes fairly with the other Track 9 contestants). predict() does a hard
    majority vote, breaking 3-way ties by summed class probability.
    """

    def __init__(self, device, epochs=40, augment=True, kernels=(3, 5, 7)):
        self.device, self.epochs, self.augment = device, epochs, augment
        self.kernels = kernels
        self.models = {}

    def fit(self, images_u8, labels, seed=0):
        for k in self.kernels:
            m = RecordModel(k)
            train_record_model(m, images_u8, labels, self.device, epochs=self.epochs,
                               augment=self.augment, seed=seed + k)
            self.models[k] = m
        return self

    def member_probs(self, images_u8):
        return {k: _predict_probs(m, images_u8, self.device) for k, m in self.models.items()}

    def predict(self, images_u8):
        probs = self.member_probs(images_u8)
        preds = np.stack([p.argmax(1) for p in probs.values()], axis=0)  # (K, N)
        summed = np.sum(list(probs.values()), axis=0)                    # (N, 10)
        n = preds.shape[1]
        out = np.empty(n, dtype=np.int64)
        for i in range(n):
            vals, counts = np.unique(preds[:, i], return_counts=True)
            top = counts.max()
            if top >= 2:
                # majority (or plurality) winner; if multiple tie, fall back to summed prob
                cands = vals[counts == top]
                out[i] = cands[summed[i, cands].argmax()] if len(cands) > 1 else cands[0]
            else:
                out[i] = summed[i].argmax()
        return out

    def score(self, images_u8, labels):
        return float((self.predict(images_u8) == labels).mean())

    def member_scores(self, images_u8, labels):
        return {f"M{k}": float((p.argmax(1) == labels).mean())
                for k, p in self.member_probs(images_u8).items()}
