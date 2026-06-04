# Track 6 (Structural Bag-of-Features) — Early Findings

**Last updated:** 2026-06-04

## Result so far

One-shot MNIST recognition from 17 CultiVar templates using structural features
(skeleton graph → L1/L2/L3 typed primitives → 60-dim bag → 1-NN cosine):

| Metric | Value |
|--------|------:|
| MNIST accuracy (100-image smoke) | **~33%** |
| Random chance (10 digits) | 10% |
| Variants17 proto baseline (CNN) | 77.46% |
| Speed | ~500 images/sec (0.1s for 100) |

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
