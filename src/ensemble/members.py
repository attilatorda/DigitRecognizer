"""Track 7 supervised members — the strongest pieces of the prior tracks.

Each member exposes fit(images_u8, labels) and predict_proba(images_u8) -> (N,10):
  - CNNMember            : SimpleCNN on raw pixels (Track 2)
  - FusionCNNMember      : SimpleCNN on raw + Guo-Hall("thin") skeleton, 2 channels (Track 3)
  - StructuralRFMember   : RandomForest on 93-dim rich skeleton features (Track 6)

CNNMember also exposes embed() (128-dim penultimate features) for the exemplar selector.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.local_cnn.model import SimpleCNN
from src.skeleton.skeletonize import skeletonize_batch
from src.structural.rich_features import extract_rich_vector, RICH_DIM


# ---------------------------------------------------------------------------
# Small CNN trainer (the local_cnn loop lives in main(); we replicate ~6 lines)
# ---------------------------------------------------------------------------

def _to_tensor(images_u8, channels_raw, channels_thin=None):
    """Build a (N, C, 28, 28) float tensor. channels_* are (N,28,28) uint8 arrays."""
    chans = [channels_raw.astype(np.float32) / 255.0]
    if channels_thin is not None:
        chans.append(channels_thin.astype(np.float32) / 255.0)
    x = np.stack(chans, axis=1)
    return torch.tensor(x, dtype=torch.float32)


def train_cnn(model, X, y, device, epochs=6, batch_size=128, lr=1e-3):
    model.to(device).train()
    loader = DataLoader(TensorDataset(X, torch.tensor(y, dtype=torch.long)),
                        batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
    return model


@torch.no_grad()
def _forward_probs(model, X, device, batch_size=512):
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        logits = model(X[i:i+batch_size].to(device))
        out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Members
# ---------------------------------------------------------------------------

class CNNMember:
    """SimpleCNN on raw pixels. Also provides 128-dim embeddings."""

    def __init__(self, device, epochs=6):
        self.device, self.epochs = device, epochs
        self.model = SimpleCNN(num_classes=10, in_channels=1)

    def fit(self, images_u8, labels):
        X = _to_tensor(images_u8, images_u8)
        train_cnn(self.model, X, labels, self.device, self.epochs)
        return self

    def predict_proba(self, images_u8):
        return _forward_probs(self.model, _to_tensor(images_u8, images_u8), self.device)

    @torch.no_grad()
    def embed(self, images_u8, batch_size=512):
        """128-dim penultimate features (classifier[:3] = Flatten->Linear->ReLU)."""
        self.model.to(self.device).eval()
        X = _to_tensor(images_u8, images_u8)
        head = self.model.classifier[:3]
        out = []
        for i in range(0, len(X), batch_size):
            f = self.model.features(X[i:i+batch_size].to(self.device))
            out.append(head(f).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)


def thin_batch(images_u8):
    """Guo-Hall ('thin') skeletons for a batch — Track 3's best skeletonizer."""
    return skeletonize_batch(images_u8, method="thin", progress_every=10**9)


def rich_features(thin_u8):
    """(N,28,28) thin skeletons -> (N,93) rich structural features."""
    X = np.zeros((len(thin_u8), RICH_DIM), dtype=np.float32)
    for i, s in enumerate(thin_u8):
        X[i] = extract_rich_vector(s)
    return X


class FusionCNNMember:
    """SimpleCNN on a 2-channel raw + Guo-Hall('thin') skeleton stack (Track 3 best).

    fit/predict accept an optional precomputed `thin_u8` to avoid re-skeletonising.
    """

    def __init__(self, device, epochs=6):
        self.device, self.epochs = device, epochs
        self.model = SimpleCNN(num_classes=10, in_channels=2)

    def fit(self, images_u8, labels, thin_u8=None):
        thin = thin_batch(images_u8) if thin_u8 is None else thin_u8
        train_cnn(self.model, _to_tensor(images_u8, images_u8, thin), labels, self.device, self.epochs)
        return self

    def predict_proba(self, images_u8, thin_u8=None):
        thin = thin_batch(images_u8) if thin_u8 is None else thin_u8
        return _forward_probs(self.model, _to_tensor(images_u8, images_u8, thin), self.device)


class StructuralRFMember:
    """RandomForest on 93-dim rich features from 'thin' skeletons (Track 6).

    fit/predict accept optional precomputed `feats` (N,93) to skip featurisation.
    """

    def __init__(self, n_estimators=300, random_state=0):
        self.clf = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=random_state),
        )

    def fit(self, images_u8, labels, feats=None):
        X = rich_features(thin_batch(images_u8)) if feats is None else feats
        self.clf.fit(X, labels)
        return self

    def predict_proba(self, images_u8, feats=None):
        X = rich_features(thin_batch(images_u8)) if feats is None else feats
        return self.clf.predict_proba(X).astype(np.float32)
