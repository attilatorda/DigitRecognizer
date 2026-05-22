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

## Artifacts

- Logs:
  - `experiments/logs/skeleton_run.log`
  - `experiments/logs/local_run.log`
- Checkpoints:
  - `experiments/checkpoints/skeleton/best_skeleton_cnn.pt`
  - `experiments/checkpoints/local_cnn/best_local_cnn.pt`
