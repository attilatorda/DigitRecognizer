# Skeleton CNN vs Local CNN (MNIST) — Experimental Analysis

Date: 2026-05-22

## Setup

- Dataset: MNIST from `mnist_data/`
- Model backbone: same `SimpleCNN` architecture
- Epochs: 5
- Batch size: 128
- LR: 0.001
- Runs:
  - Skeletonized input pipeline: `src.novel_skeleton.train_skeleton_cnn`
  - Regular grayscale pipeline: `src.local_cnn.train_local_cnn`

## Results

### Skeleton CNN

- Epoch 1: 0.9666
- Epoch 2: 0.9722
- Epoch 3: 0.9778
- Epoch 4: 0.9771
- Epoch 5: 0.9770
- **Best test accuracy: 0.9778**

### Local CNN (regular)

- Epoch 1: 0.9800
- Epoch 2: 0.9864
- Epoch 3: 0.9892
- Epoch 4: 0.9898
- Epoch 5: 0.9911
- **Best test accuracy: 0.9911**

## Comparison

- Absolute accuracy gap: **0.0133** (1.33 percentage points)
- Relative gap vs local baseline: **~1.34%**

Interpretation:

1. Skeletonization preserves enough shape information for strong performance (~97.8%).
2. Regular grayscale still performs better in this short run, likely because stroke thickness and grayscale cues contain additional discriminative information removed by thinning.
3. Skeleton model appears to plateau earlier (around epoch 3), while local CNN continues improving through epoch 5.

## Recommendations (next iteration)

1. **Tune skeleton thresholding** (currently fixed binarization at `>30`).
2. Add **morphological cleanup** before thinning (remove speckles, close gaps).
3. Try **2-channel input**: `[original, skeleton]` to retain intensity + topology.
4. Train longer (e.g., 15–25 epochs with scheduler) for both models.
5. Add per-class confusion matrix to analyze where skeletonization helps/hurts most.

## Hough-Augmented Skeleton Follow-up (2026-05-22)

### Goal

Evaluate whether adding a Hough line-map channel to skeleton input improves test accuracy.

### Implementation summary

- Added optional 2-channel mode in `src.novel_skeleton.train_skeleton_cnn`:
  - `channel_mode=skeleton` (baseline, 1 channel)
  - `channel_mode=skeleton_hough` (2 channels: `[skeleton, hough_line_map]`)
- Added tunable Hough parameters:
  - `hough_threshold`
  - `hough_line_length`
  - `hough_line_gap`
- Updated `SimpleCNN` to accept configurable `in_channels`.

### Completed experiment results (5 epochs, seed=42)

1) **Baseline skeleton** (`channel_mode=skeleton`)
- Best test accuracy: **0.9778**

2) **Skeleton + Hough** (`threshold=6, line_length=4, line_gap=1`)
- Epochs: 0.9665, 0.9722, 0.9768, 0.9774, 0.9795
- Best test accuracy: **0.9795**

3) **Skeleton + Hough** (`threshold=8, line_length=5, line_gap=2`)
- Epochs: 0.9656, 0.9733, 0.9762, 0.9769, 0.9793
- Best test accuracy: **0.9793**

### Analysis

- Hough augmentation improved over skeleton baseline in both completed settings:
  - +0.0017 (0.17pp) for `(6,4,1)`
  - +0.0015 (0.15pp) for `(8,5,2)`
- Best observed Hough run: **0.9795**, which is a modest but consistent gain vs **0.9778** baseline.
- Local grayscale CNN (`0.9911`) remains stronger overall, but the Hough channel narrows the gap slightly while preserving a topology-focused pipeline.

### Conclusion

Hough Transform is beneficial in this setup as an auxiliary channel. The improvement is incremental (not dramatic), but positive and easy to keep behind a configuration flag.

## Artifacts

- Logs:
  - `experiments/logs/skeleton_run.log`
  - `experiments/logs/local_run.log`
- Checkpoints:
  - `experiments/checkpoints/skeleton/best_skeleton_cnn.pt`
  - `experiments/checkpoints/local_cnn/best_local_cnn.pt`
