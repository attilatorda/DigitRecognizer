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
| Variants17 proto baseline (CNN) | 77.46% |
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
| Variants17 proto baseline (CNN) | 77.46% |

**Track 6 nearly doubled: 35.81% -> 65.43% (+29.62pp), still 12pp below the CNN
baseline.** The reference bank was the dominant lever (17 -> 8704 reference vectors
captures real within-class variation); richer features and the combined
morphological+DDPM bank each added a few points. Runs in ~50s on the full test set.
Raw numbers: `experiments/reports/structural_v2_results.json`.

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
