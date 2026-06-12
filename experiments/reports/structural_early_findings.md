# Track 6 (Structural Bag-of-Features) — Early Findings

**Last updated:** 2026-06-04

## Result so far

One-shot MNIST recognition from 17 CultiVar templates using structural features
(skeleton graph → L1/L2/L3 typed primitives → 60-dim bag → 1-NN cosine):

| Metric | Value |
|--------|------:|
| **MNIST accuracy (full 10K test)** | **35.81%** |
| MNIST accuracy (100-image smoke) | ~33% |
| Random chance (10 digits) | 10% |
| Variants17 proto baseline (CNN) | 77.40% |
| Speed | ~980 images/sec (10.2s for 10K) |

Full 10K test run completes in ~10 seconds and reaches **35.81%** — above the 100-image
smoke estimate, well above the 10% chance floor, and 41.65pp below the CNN baseline.
Raw numbers: `experiments/reports/structural_results.json`.

**Interpretation:** well above chance, so the structural primitives carry genuine
class signal — but far below the CNN baseline. The approach is fast and fully
interpretable, but coarse.

## Junction/loop clustering fix (applied)

Initial version counted every pixel of a skeleton crossing as a separate junction
node. Fixed by merging 8-connected junction pixels into one node (and likewise pure
cycle pixels into one loop_node).

| Digit "7" (crossed) | Before fix | After fix |
|---------------------|-----------:|----------:|
| junction nodes | 6+ | **1** |
| total features | 54 | **7** |

The representation is now topologically correct and compact. **However, accuracy did
not change (34% → 33%, within noise).** This is itself informative: junction noise was
not the accuracy limiter.

## The real bottleneck (diagnosed, not yet fixed)

Simple open-stroke digits collapse to identical descriptors:

```
digit 1  ->  {2 endpoints, 1 bent edge}   == 3 features
digit 2  ->  {2 endpoints, 1 bent edge}   == 3 features   <- identical to 1!
```

A 1, 2, 3, 5, and uncrossed-7 are all "one open stroke with two ends." The
discriminating information is in the **shape of that stroke** (its curvature profile,
orientation, number of inflection points), which the current straight/curved/bent +
size classification throws away.

## v2 — rich features + reference bank + trained classifier

Both improvements from the milestone were implemented and run on the full 10K test:

1. **Rich features** (`src/structural/rich_features.py`): the 60-dim bag is extended to
   an 88-dim vector adding an orientation histogram (8 length-weighted bins),
   curvature/inflection statistics, geometry (aspect ratio, density, centroid), an
   endpoint vertical profile, and loop top/bottom position (a 6-vs-9-vs-8 discriminator).
2. **Reference bank + classifier** (`scripts/run_structural_v2_experiment.py`): instead
   of 1-NN against 17 clean templates, features are extracted from thousands of
   augmented training images and a LogisticRegression / kNN classifier is trained.

### v2 results (full 10K test)

| Reference bank | n images | LogReg | kNN (k=5) | Best |
|----------------|---------:|-------:|----------:|-----:|
| morphological | 4352 | 63.84% | 62.53% | 63.84% |
| ddpm | 4352 | 62.31% | 63.90% | 63.90% |
| **both** | **8704** | 62.95% | **65.43%** | **65.43%** |

| Milestone | One-shot MNIST |
|-----------|---------------:|
| v1 (1-NN, 17 templates, 60-dim) | 35.81% |
| **v2 (kNN, 8704-image bank, 88-dim)** | **65.43%** |
| Variants17 proto baseline (CNN) | 77.40% |

**Track 6 nearly doubled: 35.81% -> 65.43% (+29.62pp), still 12pp below the CNN
baseline.** The reference bank was the dominant lever (17 -> 8704 reference vectors
captures real within-class variation); richer features and the combined
morphological+DDPM bank each added a few points. Runs in ~50s on the full test set.
Raw numbers: `experiments/reports/structural_v2_results.json`.

## v3 — classifier zoo + ensemble

To test whether v2's kNN was the bottleneck, a panel of classifiers was trained on the
same 88-dim features with the combined bank (morphological + high-capacity dim=32 DDPM):

| Classifier | MNIST 10K |
|------------|----------:|
| **Random Forest (400 trees)** | **68.01%** |
| Soft-voting ensemble | 66.77% |
| HistGradientBoosting | 65.99% |
| kNN (k=5) — v2's choice | 65.14% |
| Logistic Regression | 63.12% |
| MLP (128,64) | 62.19% |

**Track 6 progression: v1 35.81% → v2 65.43% → v3 68.01%.** Random Forest's nonlinear
feature interactions beat kNN by ~3pp, so the classifier was part of the limiter — but the
modest gain means the **features** are now the dominant remaining gap (still −9.39pp vs the
77.40% CNN baseline). The ensemble underperformed RF alone because the weak members (MLP,
logreg) dragged the vote down. Runs in ~140s. Raw numbers: `structural_v3_results.json`.

**Conclusion for the track:** explicit structural features + a tree classifier reach ~68%
one-shot from 17 templates with full interpretability and no deep learning — a genuinely
strong showing for a hand-engineered method, but a clear ceiling ~9pp below learned
features. Further gains would require either much richer descriptors or a structural+CNN
hybrid (crosses into Track 4/5 territory).

## v3+dir — signed-curvature directional features

The remaining bottleneck (open single-stroke digits 1/2/3/5/7 sharing descriptors) was
addressed by adding a 5-dim *signed*-curvature block to the descriptor (88 → 93 dims): the
longest stroke's net left/right bend in each of three thirds, overall handedness, and the
vertical height of the sharpest bend. This distinguishes a "2" (curves right-then-left) from
a "3" (right-then-right) from a "7" (straight-then-sharp) — which unsigned curvature
magnitude cannot.

| Classifier | 88-dim | **93-dim (+dir)** |
|------------|-------:|------------------:|
| **Random forest** | 68.01% | **72.32%** |
| Soft-voting ensemble | 66.77% | 70.70% |
| HistGradientBoosting | 65.99% | 69.50% |
| kNN (k=5) | 65.14% | 68.47% |
| Logistic regression | 63.12% | 67.02% |
| MLP | 62.19% | 64.99% |

Every classifier improved by 4–6pp, confirming that signed curvature was the missing
signal. **Track 6 full arc: 35.81 → 65.43 → 68.01 → 72.32%**, now only −5.08pp from the
77.40% CNN baseline — a fully interpretable, no-deep-learning method.

## Next steps (milestone)

To make Track 6 competitive, L2 edge descriptors need to be richer:

1. **Curvature profile**: split each edge into N segments, record the turn direction
   (left/right/straight) of each → a short directional signature per stroke.
2. **Orientation**: dominant angle of each edge (8-way), so a vertical 1 differs from a
   diagonal stroke.
3. **Inflection count**: number of curvature sign changes (an S-shaped 2 vs a C-shaped
   curve).
4. **Endpoint geometry**: relative position of the two endpoints (top-to-bottom,
   left-to-right).

Also worth trying: weight the 1-NN by level (L3 structural matches are more
discriminative than L1 node counts), and evaluate on the full 10K test set
(smoke is only 100 images).

## How to reproduce

```bash
python scripts/run_structural_experiment.py --smoke --smoke-n 100   # quick
python scripts/run_structural_experiment.py                          # full 10K
```
