# Publication Plan: CultiVar-17 One-Shot Paper

**Goal:** Make the one-shot digit recognition paper publishable with minimum additional effort.

---

## Target venue

**Primary:** ICFHR 2026 (International Conference on Frontiers in Handwriting Recognition)
- 8-page IEEE two-column format
- Directly relevant: handwriting, cultural variants, digit recognition
- MNIST still accepted at this venue

**Fallback:** Data-Centric AI workshop at NeurIPS 2026
- 4-6 pages, lower bar, values reproducibility and dataset contributions

---

## Core reframing

The paper claims to be about *one-shot learning accuracy on MNIST*.
It should claim to be about: **a culturally-motivated dataset and pipeline that shows how far
17 hand-drawn templates can go.**

Old angle: "we achieve 77.94% on MNIST"
New angle: "we introduce CultiVar-17 and show a complete reproducible pipeline
reaching 78% of supervised accuracy from 17 examples"

This reframing requires zero new experiments.

---

## What was already done

- [x] 17-template dataset with cultural variant rationale
- [x] Augmentation pipeline (elastic, stroke-width, affine, noise)
- [x] 4-config comparison (nearest, no_aug_cnn, full_aug_cnn, proto)
- [x] Test=train consistency check (confirms genuine generalisation)
- [x] Reproducible runner (one command)
- [x] Working paper draft (oneshot_analysis.md)

---

## Phase 1 — Code additions (est. ~half day)

### 1a. Figure generation script (`scripts/make_figures.py`)

Three figures needed:

- **Figure 1** — CultiVar-17 template grid: 17 templates in a 3×6 labeled grid,
  upscaled 8× from 28×28, with class labels below each. Rendered with PIL.

- **Figure 2** — Augmentation gallery: one template × 6 augmentation modes side by side
  (original, elastic mild, elastic strong, disk dilate, stroke blur, full combined),
  plus real MNIST reference column. Reuses augment.py internals. PIL grid.

- **Figure 3** — Results bar chart: 5 horizontal bars (nearest-template, no_aug,
  full_aug, proto, supervised reference line). matplotlib, clean academic style.

Output: `experiments/reports/figures/fig1_templates.png`,
`fig2_augmentation.png`, `fig3_results.png`

### 1b. Ablation extension (`scripts/run_oneshot_experiment.py`)

Add `--ablation` flag that runs 2 additional CNN configs:
- `elastic_only`: elastic_prob=0.70, stroke_prob=0.0
- `stroke_only`: elastic_prob=0.0, stroke_prob=0.80

Combined with the existing no_aug and full_aug corners this gives a 2×2 table.
Runtime: ~3 min (2 new configs × 3 seeds × 8 epochs).

Also: add `--proto-seeds` argument (default 5) to run proto over more seeds.

### 1c. Run

```
python scripts/run_oneshot_experiment.py --ablation --proto-seeds 5
```

---

## Phase 2 — Paper rewrite (est. ~1 day)

Target: `experiments/reports/oneshot_analysis.md` → full 8-page paper draft.

### Sections to write/rewrite

| Section | Status | Action |
|---|---|---|
| Title | needs update | CultiVar-17: One-Shot Digit Recognition... |
| Abstract | needs reframe | dataset + pipeline + results, ~150 words |
| 1. Introduction | needs reframe | dataset contribution angle, 3 research questions |
| 2. Related Work | MISSING | ~500 words, ~8 citations |
| 3. Dataset: CultiVar-17 | exists as setup section | expand cultural rationale |
| 4. Method | exists | add figure references |
| 5. Experiments | exists | add ablation table |
| 6. Results | exists | fill ablation, add figures, add benchmark table |
| 7. Conclusions | needs update | dataset contribution language |
| 8. Limitations & Future Work | exists | keep, minor edits |
| References | MISSING | add all citations |

### References to include

- Koch et al. (2015) — Siamese Networks for One-Shot Image Recognition
- Vinyals et al. (2016) — Matching Networks for One Shot Learning
- Snell et al. (2017) — Prototypical Networks for Few-shot Learning
- Finn et al. (2017) — MAML
- Simard et al. (2003) — Best practices for CNNs (elastic distortion)
- LeCun et al. (1998) — Gradient-based learning (LeNet, MNIST baselines)
- Lake et al. (2015) — Human-level concept learning (Omniglot)

---

## Phase 3 — Polish (est. ~half day)

- [ ] Read full draft aloud; fix clunky sentences
- [ ] Verify every number in the paper matches oneshot_results.json
- [ ] Verify every figure is referenced in the text
- [ ] Add IEEE formatting note (paper needs LaTeX conversion before submission)
- [ ] Check abstract is exactly the key claim + result
- [ ] Add acknowledgements placeholder

---

## Files involved

| File | Role |
|---|---|
| `experiments/reports/oneshot_analysis.md` | The paper |
| `experiments/reports/oneshot_results.json` | Raw data (source of truth) |
| `experiments/reports/figures/` | Generated figures |
| `scripts/make_figures.py` | Figure generation |
| `scripts/run_oneshot_experiment.py` | Extended with ablation |
| `src/variants17/benchmarks.py` | Reference table |

---

## What we explicitly will NOT do

- Move to Omniglot or miniImageNet
- Implement MAML or Matching Networks
- Add templates from other writers
- Run statistical significance tests (mentioned in limitations)
- Convert to LaTeX (that's a formatting step after draft is complete)
