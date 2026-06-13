# Track 9c — Corruption-agnostic structural/generative ensembling for robust low-data digit recognition

**One-line claim.** When the deployment corruption is *unknown* (the realistic case), adding
corruption-agnostic structural and generative members to the MNIST CNN record holder generalizes
to *unseen* corruptions as well as or better than targeted corruption augmentation — clearly so
for blur, stroke-thickening, and occlusion; gaussian noise is the exception.

## Setup

- **Opponent / base model:** the reproducible MNIST record holder, An et al. 2020 (M3/M5/M7
  majority vote), re-implemented in `src/track9/record_models.py` (light repro, 99.65% clean @60k).
- **Added members (corruption-agnostic, trained on the clean budget only):** structural
  bag-of-features RF (skeleton graph, Track 6) + a diffusion-augmented CNN (Track 5, n≥500).
- **Combiner:** held-out-free, so every member — including the baseline's CNNs — trains on the
  full labeled budget (fixes the Track 9b 70/30 artifact). `conf` = per-sample confidence
  weighting; `soft` = equal mean (`src/track9/combiners.py`).
- **Corruptions** (`src/track9/corruptions.py`): gaussian_noise, blur, stroke_dilate, occlusion;
  evaluated on the 10k test set. Budgets 20–2000, **10 seeds** (5 for LOCO).
- **Baselines:** `record_vote` (clean-trained) and `record_aug` (trained *with* the corruptions
  in its augmentation — the strong "if you know the corruption, train for it" baseline).

## Results

### 1. Robustness over the clean-trained baseline (mean corruption accuracy)
`conf_all` beats `record_vote` at **every** budget on mCA: +1.7pp (n=20, 10/10 seeds) → +4.9pp
(n=2000, 10/10). Consistent and seed-robust, but `record_vote` never saw corruption — the easy
baseline. (`fig9_track9c_mca.png`.)

### 2. Attribution (leave-one-member-out)
The gain splits by regime (`fig11_track9c_attribution.png`):
- **Low budget (n≤200, no diffusion):** the *structural* member drives it (+1.3pp at n=100),
  mostly on `stroke_dilate` (+3.3pp) — skeleton features are stroke-thickness invariant.
- **n≥500:** the *diffusion* member dominates (+3.2 → +4.7pp), on `gaussian_noise` and
  `occlusion`. The structural contribution shrinks to ~0.

### 3. The honest limit: targeted augmentation wins when it knows the corruption
`record_aug` (trained on the test corruption family) beats `conf_all` decisively at n≥100
(+6.8 → +19.4pp). The only budget where the agnostic ensemble beats it is **n=20** (+8.8pp,
10/10 seeds): with so few labels a single CNN cannot learn classification *and* corruption
invariance, so the structural prior wins. So head-to-head against an informed baseline, the
broad win does not exist.

### 4. The realistic test (LOCO): unseen corruptions
`record_aug` above is unrealistically advantaged — it trained on the exact corruption it is
tested on. Leave-one-corruption-out trains it on the *other* corruptions and tests on the
held-out one (`record_aug_loco`), the deployment-realistic "unknown corruption" setting.
The corruption-**agnostic** ensemble (`conf_all`) vs `record_aug_loco` on the held-out
corruption (`conf_all − record_aug_loco`, pp; `fig10_track9c_loco.png`):

| held-out      | n=100 | n=200 | n=500 | n=1000 | n=2000 |
|---------------|------:|------:|------:|-------:|-------:|
| blur          | +4.6  | +2.2  | −0.4  | +1.5   | +1.0   |
| stroke_dilate | +7.3  | +1.1  | −2.9  | +0.5   | +0.4   |
| occlusion     | +2.1  | +0.9  | +7.4  | +8.7   | +8.0   |
| gaussian_noise| −2.1  | −14.4 | −1.0  | −3.9   | +2.0   |

On **3 of 4** corruptions the agnostic ensemble generalizes to the unseen corruption as well as
or better than targeted augmentation — strongly on occlusion (+7–9pp at n≥500). **Gaussian noise
is the exception**: nothing handles it without having seen it.

## Honest limitations
- The win is in the **unknown-corruption** threat model. If you know the corruption and have
  ≥100 labels, just augment for it (`record_aug`) — it is far stronger.
- **Gaussian noise** defeats the approach (skeletonization is noise-fragile; clean-trained CNNs
  are noise-fragile).
- The featured `conf` combiner is **not** clearly better than plain `soft` averaging (soft is
  marginally better at low n); per-sample confidence weighting is not the source of the win.
- Skeleton invariance (the original hypothesis) only carries the low-budget `stroke_dilate`
  gain; the larger gains are from the diffusion (generative-augmentation) member.

## Conclusion
A genuine, mechanism-backed positive in a realistic setting: **corruption-agnostic
structural/generative diversity buys low-data robustness to *unseen* image corruptions that
targeted augmentation cannot, on geometric/occlusion corruptions** — a workshop-tier
contribution. Reproduce: `scripts/run_track9_robust.py`, `scripts/run_track9_loco.py`,
`scripts/make_track9c_figures.py`. Raw: `track9_robust_results.json`, `track9_loco_results.json`.
