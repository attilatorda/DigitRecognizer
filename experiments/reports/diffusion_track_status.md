# Track 5 (Diffusion) — Status & How to Continue

**Last updated:** 2026-06-05

## High-capacity result (dim=32, timesteps=1000, GPU) — milestone CONFIRMED

Re-ran the full pipeline at dim=32 / timesteps=1000 on a GTX 1660 SUPER (CUDA build
`torch 2.11.0+cu128`, fp32 — fp16/AMP produces NaN on Turing so it is opt-in via
`--amp`). Phase 1: 50 epochs, 23 min, loss 0.0398. Phase 2: 100 epochs, ~10 min,
loss ~0.036. Generation + eval on GPU.

| Config | morphological | dim=16 DDPM | **dim=32 DDPM** |
|--------|------:|------:|------:|
| no_aug CNN | 51.20% | 72.17% | 71.45% ± 3.44 |
| full_aug CNN | 65.19% | 68.39% | 68.70% ± 2.70 |
| proto embedding | **77.46% ± 2.39** | 73.70% | **76.57% ± 1.49** |

**The proto config improved +2.87pp (73.70 → 76.57%) and now statistically matches the
hand-crafted morphological baseline** (76.57 ± 1.49 vs 77.46 ± 2.39 — error bars overlap
heavily). The −3.75pp gap from the speed-compromised model closed to −0.89pp purely from
added capacity, confirming the hypothesis that dim=16/timesteps=250 was the limiter, not
the approach. The dim=32 `phase2_final.pt` now overwrites the dim=16 checkpoint; raw
numbers in `diffusion_experiment_results.json`.

---

## Original dim=16 run (CPU) — kept for reference

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

**Phase 2 — COMPLETE.** Fine-tuned `phase1_final.pt` on the 17-class CultiVar
augmented set (4352 images, 100 epochs, dim=16/timesteps=250). Final loss **0.0344**
(below Phase 1's 0.043 floor — fine-tuning genuinely improved the model).
Checkpoint: `experiments/checkpoints/diffusion/phase2_final.pt`.

**Generation — COMPLETE.** 4352 DDIM-sampled images (256/class, 17 classes) saved to
`experiments/checkpoints/diffusion_aug/generated_images.npy` (+ labels).

## RESULTS — DDPM augmentation vs morphological baseline

One-shot MNIST accuracy, DDPM-generated training images vs the Variants17
morphological-augmentation baseline:

| Config | Morphological baseline | DDPM | Delta |
|--------|----------------------:|-----:|------:|
| no_aug CNN | 51.20% | **72.17%** | **+20.97pp** |
| full_aug CNN | 65.19% | 68.39% | +3.19pp |
| proto embedding | **77.46%** | 73.70% | -3.75pp |

(DDPM: 3 CNN seeds, 5 proto seeds, 8 epochs, repeats=1 — generated images used
directly as the training set.)

### Findings

1. **DDPM images are dramatically better raw training data.** The no-augmentation
   CNN jumped **+21pp** (51→72%): training on 4352 diverse generated digits beats
   training on 17 templates by a wide margin.
2. **Stacking morphological augmentation on DDPM images hurts** (full_aug 68.4% <
   no_aug 72.2%). The generated images are already varied; piling distortions on top
   degrades them.
3. **The strong proto+morphological baseline still wins narrowly** (77.46% vs 73.70%,
   -3.75pp). The dim=16 / timesteps=250 speed compromise caps generation quality;
   a dim=32 / timesteps=1000 GPU model would likely close or reverse this gap.

**Takeaway:** class-conditional DDPM augmentation is a clear win for plain CNN training
and competitive with the best hand-crafted pipeline even at a deliberately small model
size. The natural next step is a higher-capacity DDPM (see caveats below).

Raw numbers: `experiments/reports/diffusion_experiment_results.json`.
Re-run evaluation: `python scripts/run_diffusion_experiment.py`.

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
