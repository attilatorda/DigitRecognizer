# Project status — tracks and publications

Single source of truth for where every track stands and the two papers that consolidate them.
Last updated after Track 9c (corruption robustness) was strengthened and Track 8 was closed.

## Tracks

| # | Track | Status | Outcome |
|---|-------|--------|---------|
| 1 | Baseline | done | reference supervised pipeline (~99%) |
| 2 | Local CNN | done | accuracy ceiling, 99.11% (clean MNIST) |
| 3 | Skeleton CNN | done | raw+skeleton fusion 98.98%; skeleton-only 97.8–98.1% |
| 4 | Variants17 one-shot | done → **Paper 1** | 77.4% MNIST from 17 templates (proto) |
| 5 | Diffusion augmentation | done → **Paper 1** | learned aug matches/exceeds hand-crafted (79.1%) |
| 6 | Structural bag-of-features | done → **Paper 1** | 72.3% from skeleton graph, no deep learning |
| 7 | Combined system + exemplar selection | done | stacking + data-efficiency curve; prototypical-vs-random study |
| 8 | CNN introspection (decode/reconstruct/logic) | **reopened** | F1/F2: variant decodable (probe 83.8%, ARI 0.83), hard relabeling hurts. F3: activation-max rebuilds all 10 digits (100% self-classify). F4: variant axis is generative (probe score steers −8.2→+25.0). F5: XOR/AND/OR exactly extracted from minimal nets (XOR=(¬a∧b)∨(a∧¬b); linear can't fit XOR). Auxiliary head still future work. |
| 9 | Data efficiency + robustness | done → **Paper 2** | 9: record holder unbeatable head-to-head; 9b: held-out-split artifact; 9c: corruption-agnostic diversity helps on unseen corruptions |

## Paper 1 — CultiVar-17 one-shot (Tracks 4/5/6)

- **Source:** `cultivar17_paper.tex` (Elsevier `elsarticle`, 7 pp, compiles, 0 undefined refs).
- **Target:** Pattern Recognition Letters (Elsevier).
- **Complete:** manuscript, all declarations (CRediT, competing interest, data availability,
  generative-AI), `cultivar17_highlights.txt`, `cultivar17_cover_letter.md`.
- **Remaining (human-only):** insert the real public repository URL into the data-availability
  statement; add suggested reviewers; confirm the generative-AI declaration matches your usage;
  submit via Editorial Manager.

## Paper 2 — Track 9c corruption robustness (Track 9)

- **Source:** `track9c_paper_neurips.tex` (**NeurIPS workshop format**, 6 pp, compiles, 0
  undefined refs) — the venue-formatted submission version. `track9c_paper.tex` (IEEEtran, 4 pp)
  kept as the arXiv/preprint variant. Both use `track9c_refs.bib`; figures fig9–13.
- **Claim:** in the unknown-corruption setting, corruption-agnostic structural/generative
  diversity generalises to unseen corruptions as well as or better than targeted augmentation
  (blur/stroke/occlusion); reproduces on KMNIST; the gain grows with severity.
- **Venue plan:** post to **arXiv now**; submit to a **NeurIPS 2026 / ICLR 2027 workshop**
  (robustness, distribution shift, or data-centric ML) when CFPs open. NeurIPS style already applied.
- **Optional strengtheners (not blocking):** physical/real corruptions; a noise-robust agnostic
  member; more datasets.

## Paper 3 — MNIST data-efficiency benchmark (Tracks 9/9b)

- **Source:** `track9_benchmark_paper.tex` (**NeurIPS workshop format**, 3 pp, compiles, 0
  undefined refs); `track9_benchmark_refs.bib`; figures fig7/fig8.
- **Claim (honest benchmark + pitfall):** the MNIST record holder is the data-efficiency winner
  at every budget (64% @ n=20 → 99.4% @ n=5000); a structural+generative ensemble converges to
  within ~2pp by n≥1000 but never overtakes; and the apparent "augment the record holder" gain
  is largely a **held-out-split artifact** (low-data stacking must use full-budget baselines).
- **Venue plan:** NeurIPS workshop / Datasets & Benchmarks-style data-centric venue; arXiv.

## Venue formats (all three target their venue's LaTeX template)
- P1 → Elsevier `elsarticle` (Pattern Recognition Letters). P2, P3 → NeurIPS style
  (`neurips_2024.sty`, numbered natbib citations). All compile clean with 0 undefined refs.

## What is intentionally NOT being done
- No third paper from Tracks 1–3/7 (supporting findings, not standalone-paper-worthy).
- Track 8 auxiliary sub-class head left as documented future work.
- No move to Omniglot/miniImageNet or new model families.
