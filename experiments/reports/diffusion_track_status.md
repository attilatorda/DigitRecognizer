# Track 5 (Diffusion) — Status & How to Continue

**Last updated:** 2026-06-04

## Where it stands

**Phase 1 — COMPLETE.**
Class-conditional DDPM trained on MNIST images for the four single-variant CultiVar
classes (3, 5, 6, 8 → class IDs 10, 12, 13, 15).

- Config: `dim=16`, `timesteps=250`, 50 epochs, 20,000 images (5000/class)
- Runtime: 4.5 hours on CPU
- Final loss: **0.0427** (converged — flat after ~epoch 25)
- Checkpoint: `experiments/checkpoints/diffusion/phase1_final.pt`
  (also `phase1_epoch0025.pt`, `phase1_epoch0050.pt`)

## How to continue — Phase 2 (fine-tune on all 17 classes)

Phase 2 adapts the Phase-1 model to the full 17-class CultiVar taxonomy using the
morphological augmented set as training data.

```bash
python scripts/train_diffusion.py --phase 2 \
    --resume experiments/checkpoints/diffusion/phase1_final.pt \
    --epochs 100 --batch-size 128 --dim 16 --timesteps 250 --save-every 25
```

- Input data (already present): `experiments/checkpoints/variants17/augmented_train/images.npy`
  (4352 images) + `labels17.npy`
- **IMPORTANT:** `--dim 16 --timesteps 250` MUST match Phase 1, or the checkpoint
  won't load (architecture mismatch).
- Estimated runtime: ~2 hours (4352 images × 100 epochs ≈ same batch count as Phase 1's
  20000 × 50 ÷ ... → roughly 2h on CPU). Reduce `--epochs` to 50 to halve it.

## Then: generate + evaluate

```bash
# Generate 256 images per class (4352 total) using DDIM
python scripts/generate_diffusion_aug.py \
    --checkpoint experiments/checkpoints/diffusion/phase2_final.pt \
    --n-per-class 256 --dim 16 --timesteps 250

# Run one-shot classifier on generated data, compare to 77.46% baseline
python scripts/run_diffusion_experiment.py
```

## Notes / caveats

- `dim=16` is a speed compromise; the loss floor (0.043) is higher than a `dim=32`
  model would reach (~0.030). Generated digits will be recognisable but blurry.
- Only 4 of 17 class embeddings saw real data in Phase 1; the other 13 are learned
  entirely in Phase 2 from the (synthetic) augmented set. Expect the 4 MNIST-trained
  classes (3,5,6,8) to generate cleaner samples than the 13 variant classes.
- For a publication-quality result, rerun the whole pipeline with `dim=32`,
  `timesteps=1000`, on GPU.
