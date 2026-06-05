# CultiVar-17 — paper summary (superseded)

> **This markdown is no longer the working draft.** The canonical, up-to-date paper is the
> LaTeX source and its compiled PDF:
>
> - **`experiments/reports/cultivar17_paper.tex`** (IEEEtran, source of truth)
> - **`experiments/reports/cultivar17_paper.pdf`** (compiled, 5 pages)
> - References: `experiments/reports/cultivar17_refs.bib`
>
> This file is kept only as a short human-readable overview and is **not maintained in
> lockstep** with the LaTeX. If anything here disagrees with the PDF, the PDF wins.

---

## What the paper is about

**Title:** *CultiVar-17: Three Data-Centric Strategies for One-Shot Digit Recognition from a
Culturally-Motivated Template Set.*

From exactly **17 hand-drawn digit templates** (one per culturally-motivated style class —
crossed vs. uncrossed 0/7, serif vs. plain 1, etc.), the paper studies how far three
complementary *data-centric* strategies can push one-shot transfer to the MNIST test set.
The thesis: **the augmentation _process_, not model capacity, is the dominant lever.**

## Unified results (MNIST one-shot transfer from 17 templates)

| Strategy | Best model | MNIST test (%) |
|----------|-----------|---------------:|
| Hand-crafted augmentation | Proto embedding | **77.46 ± 2.39** |
| Learned augmentation (class-conditional DDPM) | Proto embedding | 76.57 ± 1.49 |
| Structural bag-of-features | Random forest | 68.01 |
| Nearest-template | L2 | 38.60 |
| Supervised reference | CNN, 60k labels | 99.11 |
| Chance | — | 10.00 |

**Headline findings**
1. **Hand-crafted augmentation** + a prototype-embedding model reaches 77.46% — 78% of
   supervised accuracy from 17 examples. Elastic and stroke-width augmentation are
   independently and additively useful.
2. **Learned augmentation**: a class-conditional DDPM that synthesises training data lifts a
   plain CNN by **+20.97pp** and, at full capacity (dim=32), the proto config (76.57%)
   *statistically matches* the hand-crafted baseline. Stacking morphological augmentation on
   top of DDPM images hurts (already varied).
3. **Structural recognition**: an interpretable 88-dim skeleton-graph descriptor + random
   forest reaches 68.0% with no deep learning — within ~9pp of the neural pipelines.

## Detailed per-track write-ups

- Diffusion (Track 5): `experiments/reports/diffusion_track_status.md`
- Structural (Track 6): `experiments/reports/structural_early_findings.md`
- Raw numbers: `oneshot_results.json`, `diffusion_experiment_results.json`,
  `structural_v3_results.json`

## Reproduce

```bash
python scripts/run_oneshot_experiment.py --ablation --proto-seeds 5   # hand-crafted
python scripts/train_diffusion.py --phase 1 ...                       # learned (DDPM)
python scripts/run_diffusion_experiment.py
python scripts/run_structural_v3_experiment.py                        # structural
python scripts/make_figures.py                                        # figures
```

**Target venue:** ICFHR 2026 (fallback: NeurIPS Data-Centric AI workshop).
