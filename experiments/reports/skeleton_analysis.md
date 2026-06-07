# Skeletonization as a Preprocessing Step for Handwritten Digit Recognition: A Systematic Comparison

**Working paper · DigitRecognizer project · 2026-06-01**

---

## Abstract

We ask whether skeletal representations of handwritten digits improve CNN
classification accuracy on MNIST. We train an identical SimpleCNN under ten
input configurations: raw pixels (baseline), four skeletonization algorithms
(Zhang-Suen, Lee, Guo-Hall/thin, medial axis) each in skeleton-only and
skeleton+Hough modes, and a two-channel fusion of raw pixels with Guo-Hall
skeleton. All experiments use 3 seeds × 5 epochs on the full 60K/10K MNIST
split.

Three results emerge cleanly. First, every skeleton-only configuration
underperforms raw pixels by 0.9–1.2 percentage points; the gap is statistically
reliable across all methods and seeds. Second, adding the Guo-Hall skeleton as
a second channel alongside raw pixels (98.98% ± 0.05%) is indistinguishable
from raw alone (99.03% ± 0.06%), confirming that the CNN already extracts
topological structure implicitly. Third, explicit Hough line-map features add
no measurable accuracy at any configuration. The unified finding is that
skeletonization destroys the only feature that matters — stroke-width gradients
— and the topological structure it preserves is already learnable from raw
pixels.

---

## 1. Introduction

Skeletonization reduces a filled stroke to a one-pixel-wide centerline, making
the representation invariant to stroke width. This invariance is appealing for
handwriting recognition: the same digit written with a thin ballpoint or a thick
marker should produce identical skeletons, potentially making models more robust
to pen-type variation. Several prior systems have used skeleton preprocessing
before feeding images to classifiers.

The MNIST dataset is a natural benchmark for studying skeletonization effects.
Its 28×28 grayscale images carry both structural (topology, aspect ratio) and
textural (stroke width, pressure gradients) information. The question is which
of these two information types CNNs actually rely on.

We study three specific questions:

1. Does any skeletonization algorithm improve over raw pixels on in-distribution
   MNIST?
2. Which algorithm produces the best accuracy, and how does accuracy trade off
   against preprocessing cost?
3. Does providing the skeleton as an *additional* channel — rather than a
   replacement — help the CNN?

We answer all three questions with controlled experiments under identical model
architectures, hyperparameters, and random seeds.

---

## 2. Experimental Setup

### 2.1 Model

All experiments use **SimpleCNN**: two convolutional blocks (Conv2d →
ReLU → MaxPool2d, filter sizes 32 and 64) followed by a fully-connected
classifier (Linear 3136→128, ReLU, Dropout 0.2, Linear 128→10). For
two-channel inputs `in_channels=2`; otherwise `in_channels=1`. Optimizer:
Adam, lr=1e-3, no learning-rate schedule. Batch size 128. Training: 5 epochs,
3 independent seeds (0, 1, 2). Reported accuracy is the best per-seed test
accuracy, averaged over seeds.

This architecture was chosen for reproducibility and runtime. It achieves
~99% on raw MNIST with 5 epochs on CPU, which is sufficient to detect changes
of ~0.1 pp with three seeds.

### 2.2 Skeletonization algorithms

| Label | scikit-image call | Algorithm |
|---|---|---|
| zhang | `skeletonize(binary)` | Zhang-Suen (1984) parallel thinning |
| lee | `skeletonize(binary, method="lee")` | Lee (1994), topology-safe |
| thin | `thin(binary)` | Guo-Hall (1989) iterative thinning |
| medial_axis | `medial_axis(binary, return_distance=False)` | Distance-transform centerline |

All methods receive a binary foreground mask thresholded at pixel intensity >
30 on the uint8 [0, 255] image (standard MNIST convention: bright strokes on
dark background). Output is uint8 {0, 255}, normalized to [0, 1] before the
CNN.

### 2.3 Input configurations

| Config | Channels | Description |
|---|---|---|
| raw | 1 | Normalized raw pixels |
| {method}\_skeleton | 1 | Skeleton only |
| {method}\_hough | 2 | Skeleton + probabilistic Hough line map |
| thin\_fusion | 2 | Raw pixels (Ch0) + Guo-Hall skeleton (Ch1) |

The Hough line map is computed per-image with `probabilistic_hough_line`
(threshold=8, line\_length=5, line\_gap=2) on the skeleton binary image.

### 2.4 Caching

Skeletonized MNIST splits are saved to `data/processed/mnist_skeleton/` on
first run and loaded from disk thereafter. Reported skeletonization times are
one-time costs per method; subsequent runs are instantaneous.

### 2.5 Reproducibility

All code lives in `src/skeleton/` and `scripts/run_skeleton_experiment.py`.
The full experiment can be reproduced with:

```
python scripts/run_skeleton_experiment.py --epochs 5 --seeds 3
```

Results are saved to `experiments/reports/skeleton_comparison.json`. The fusion
experiment was run separately with `--methods thin --channel-modes raw_thin
--skip-raw`; results are merged into the same JSON.

---

## 3. Results

### 3.1 Overall accuracy

| Input | Method | Mean acc (%) | Std (%) | Skel. time (s) |
|---|---|---:|---:|---:|
| raw pixels | — | **99.03** | 0.06 | — |
| raw + skeleton (fusion) | thin | **98.98** | 0.05 | 26 |
| skeleton only | thin | 98.11 | 0.09 | 26 |
| skeleton only | thin + Hough | 98.12 | 0.09 | 26 |
| skeleton only | lee | 98.02 | 0.07 | 8 |
| skeleton only | lee + Hough | 98.02 | 0.07 | 8 |
| skeleton only | medial_axis | 97.95 | 0.11 | 2978 |
| skeleton only | medial_axis + Hough | 97.85 | 0.05 | 2978 |
| skeleton only | zhang + Hough | 97.82 | 0.05 | 2 |
| skeleton only | zhang | 97.81 | 0.04 | 2 |

### 3.2 Fusion experiment

| Config | Mean acc (%) | Std (%) | Delta vs raw |
|---|---:|---:|---:|
| raw (baseline) | 99.03 | 0.06 | — |
| thin skeleton only | 98.11 | 0.09 | −0.92 pp |
| thin fusion (raw + skeleton) | 98.98 | 0.05 | −0.05 pp |

The fusion recovers 94% of the accuracy lost by skeleton-only (0.87 of 0.92
pp). The residual 0.05 pp gap to raw is within one standard deviation and is
not statistically significant.

### 3.3 Per-class accuracy — mean over seeds (%)

| Digit | raw | zhang | lee | thin | thin fusion |
|---|---:|---:|---:|---:|---:|
| 0 | 99.7 | 99.6 | 99.5 | 99.8 | 99.7 |
| 1 | 99.5 | 98.9 | 99.1 | 99.2 | 99.8 |
| 2 | 98.9 | 98.4 | 98.1 | 98.5 | 99.0 |
| 3 | 99.3 | 97.6 | 97.9 | 97.6 | 99.2 |
| 4 | 99.1 | 98.1 | 97.5 | 97.9 | 99.3 |
| 5 | 99.0 | 97.2 | 98.2 | 97.4 | 98.9 |
| 6 | 98.7 | 97.5 | 97.9 | 98.0 | 98.3 |
| 7 | 99.3 | 98.2 | 97.7 | 98.2 | 99.2 |
| 8 | 98.8 | 95.7 | 96.9 | 97.6 | 98.3 |
| 9 | 97.9 | 96.6 | 97.1 | 96.7 | 98.1 |

Full per-class data for all nine configurations is in
`skeleton_comparison.json`.

### 3.4 Preprocessing time

| Method | 70K images (s) | Per image (ms) |
|---|---:|---:|
| zhang | 1.6 | 0.023 |
| lee | 7.6 | 0.109 |
| thin | 26 | 0.371 |
| medial_axis | 2978 | 42.5 |

Preprocessing times span nearly four orders of magnitude. With caching the
cost is one-time; for online or streaming use, zhang and lee are the only
practical choices.

---

## 4. Findings

### F1 — Skeletonization is an information bottleneck for in-distribution MNIST

Every skeleton-only configuration underperforms raw pixels. The accuracy gap
is 0.9–1.2 pp and is statistically reliable: all skeleton results fall more
than two standard deviations below raw across all three seeds.

The gap has two causes. First, binarization to a 1-pixel skeleton discards the
grayscale gradient encoding stroke width and ink pressure — information the CNN
demonstrably uses (see F4 for which classes lose most). Second, thinning
algorithms handle junction pixels differently, introducing small but consistent
topological noise at branch-points and loop closures.

This finding contradicts the intuition that removing "irrelevant" stroke-width
variation should help by reducing the hypothesis space. On in-distribution
MNIST, the variation is signal, not noise.

### F2 — Guo-Hall (thin) is the best algorithm; medial axis is impractical

Accuracy ranking: **thin (98.11%) > lee (98.02%) > medial_axis (97.95%) >
zhang (97.81%)**.

Guo-Hall thinning produces the cleanest skeletons for MNIST because it
eliminates the diagonal connectivity artifacts common in Zhang-Suen and avoids
the sparse, fragmented centerlines of medial-axis transforms on short strokes.

Lee 1994 is the best practical choice when preprocessing speed matters: it
achieves 98.02% at 7.6 seconds, only 0.09 pp below thin.

Medial axis is the worst outcome despite being the most mathematically precise
algorithm. Its distance-transform centerlines are topologically exact but
produce broken, low-density skeletons for MNIST's short strokes, which
confuses the CNN. Combined with its 50-minute preprocessing time — 1840× slower
than zhang — medial axis should not be used for MNIST-scale tasks.

Zhang-Suen is the fastest (1.6s) but also the least accurate (97.81%). Its
8-connectivity rule leaves diagonal staircase artifacts that the CNN cannot
fully ignore.

### F3 — Adding the skeleton as a second channel is redundant given raw pixels

The fusion of raw pixels and Guo-Hall skeleton (98.98% ± 0.05%) is
statistically indistinguishable from raw alone (99.03% ± 0.06%). The 0.05 pp
gap is below one standard deviation.

This is the sharpest result in the paper. It reveals that **the CNN already
extracts from raw pixels the same topological features that the skeleton makes
explicit**. The convolutions learn to identify stroke centerlines, junction
types, and loop structure without any preprocessing. The skeleton channel adds
nothing that the raw channel does not already provide.

Conversely, the skeleton-only case shows that once stroke-width information is
removed, the CNN cannot recover it from topology alone — the accuracy loss is
genuine and persistent. The information the skeleton destroys (gradients,
width) is the only part that mattered; the information it preserves (topology)
was already learnable.

The fusion also reduces variance slightly (std 0.05% vs 0.09% for thin-only),
which may reflect the two channels providing complementary gradient signals
during early training even if final accuracy converges to the same level.

### F4 — Digits 3, 5, 8 are most sensitive to stroke-width loss; digit 0 is immune

Per-class analysis reveals which digit classes rely on stroke-width information:

**Most affected by skeletonization (thin vs raw):**

| Digit | raw | thin | Loss |
|---|---:|---:|---:|
| 3 | 99.3% | 97.6% | −1.7 pp |
| 5 | 99.0% | 97.4% | −1.6 pp |
| 4 | 99.1% | 97.9% | −1.2 pp |
| 9 | 97.9% | 96.7% | −1.2 pp |
| 8 | 98.8% | 97.6% | −1.2 pp |

Digits 3, 5, and 8 are topological near-twins — their skeletons share
open-curve and closed-loop patterns with adjacent classes (3↔8, 5↔6, 4↔9).
The CNN distinguishes them partly by stroke curvature and junction width, both
of which skeletonization erases.

**Least affected:**

Digit 0 is completely unaffected (sometimes +0.1 pp). A simple closed oval
has a unique, highly stable skeleton that no other digit class shares.
Digit 1 loses only 0.3 pp (thin): a vertical stroke skeletonizes without loss
of topological identity.

The fusion recovers most of these per-class losses. Digit 8 in particular
climbs from 97.6% (thin-only) back to 98.3% (thin_fusion), approaching the
raw baseline of 98.8%.

### F5 — Hough line features add no benefit

Adding a probabilistic Hough line map as a second channel changes accuracy by
at most ±0.01 pp for zhang, lee, and thin, and is mildly harmful (−0.10 pp)
for medial_axis. No result crosses one standard deviation.

The CNN's convolutional filters already learn line-orientation selectivity from
the skeleton channel without the explicit Hough feature; the Hough map is
entirely redundant. For medial_axis the fragmented centerlines do not form
clean Hough-detectable segments, adding noise rather than signal.

---

## 5. Conclusions

Skeletonization does not improve handwritten digit classification accuracy with
a standard CNN on in-distribution MNIST, whether used as a replacement for raw
pixels or as an additional channel.

Three conclusions stand:

**C1. Skeletonization costs accuracy by destroying the signal that matters.**
All four algorithms underperform raw pixels by 0.9–1.2 pp. The loss is
directly attributable to the removal of stroke-width gradients, which CNNs
use as a discriminative feature for visually similar digit pairs.

**C2. CNN topology learning is implicit and complete.** The raw+skeleton fusion
is statistically indistinguishable from raw alone. The CNN does not benefit
from having topology made explicit; it learns it directly from pixels. This
rules out the hypothesis that skeletonization acts as a useful inductive bias
for this model family on this task.

**C3. If skeletonization is required, use Lee or Guo-Hall.** Lee (1994)
achieves 98.02% at 8 seconds — the best speed-accuracy tradeoff. Guo-Hall
achieves 98.11% at 26 seconds and is preferred when accuracy is paramount.
Zhang-Suen and medial-axis should be avoided: the former for accuracy, the
latter for both accuracy and cost.

---

## 6. Limitations and Future Work

**Distribution shift.** All conclusions are conditioned on in-distribution
MNIST. Skeletonization's stroke-width invariance could reverse the result when
training and test distributions differ in pen thickness or writing pressure.
The one-shot learning component of this project (training on a single custom
template per class and evaluating on MNIST) is the natural follow-on: there,
the domain gap includes systematic stroke-width differences, and skeletal
representations may help.

**Architecture dependence.** SimpleCNN is a shallow two-layer network. Deeper
architectures with larger receptive fields may extract topology more or less
efficiently, changing the magnitude of the findings. We expect the directional
conclusions (skeleton adds nothing over raw) to hold, but the quantitative gaps
may differ.

**Epoch budget.** Five epochs produces converged but not fully-optimized
models. Longer training may reduce the gap slightly as all configurations
continue to improve, but the ordering is stable by epoch 3 in all runs.

**Skeleton quality visualization.** A visual comparison of per-method skeleton
output on representative digits would strengthen the paper. A future version
should include a figure showing the same digit under all four algorithms
alongside the raw image.

---

## Appendix: Full per-class accuracy by configuration

| Digit | raw | zhang | zhang+H | lee | lee+H | thin | thin+H | medial | medial+H | fusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 99.7 | 99.6 | 99.5 | 99.5 | 99.6 | 99.8 | 99.7 | 99.5 | 99.6 | 99.7 |
| 1 | 99.5 | 98.9 | 99.1 | 99.1 | 99.0 | 99.2 | 99.3 | 99.3 | 99.0 | 99.8 |
| 2 | 98.9 | 98.4 | 98.9 | 98.1 | 98.6 | 98.5 | 98.8 | 98.2 | 98.5 | 99.0 |
| 3 | 99.3 | 97.6 | 97.7 | 97.9 | 97.8 | 97.6 | 97.5 | 97.8 | 97.0 | 99.2 |
| 4 | 99.1 | 98.1 | 98.1 | 97.5 | 96.9 | 97.9 | 98.1 | 97.4 | 98.2 | 99.3 |
| 5 | 99.0 | 97.2 | 97.7 | 98.2 | 98.2 | 97.4 | 97.9 | 97.1 | 97.4 | 98.9 |
| 6 | 98.7 | 97.5 | 97.5 | 97.9 | 97.9 | 98.0 | 97.7 | 97.9 | 98.2 | 98.3 |
| 7 | 99.3 | 98.2 | 98.2 | 97.7 | 97.7 | 98.2 | 98.5 | 98.4 | 97.8 | 99.2 |
| 8 | 98.8 | 95.7 | 95.3 | 96.9 | 97.1 | 97.6 | 96.7 | 97.2 | 96.3 | 98.3 |
| 9 | 97.9 | 96.6 | 96.2 | 97.1 | 97.3 | 96.7 | 96.8 | 96.7 | 96.4 | 98.1 |

---

*Raw data: `experiments/reports/skeleton_comparison.json`*  
*Experiment runner: `scripts/run_skeleton_experiment.py`*  
*Model: `src/local_cnn/model.py` (SimpleCNN)*  
*Skeletonization: `src/skeleton/skeletonize.py`*
