# Skeleton vs Local CNN — Evidence Matrix Audit

Date: 2026-05-23
Auditor: Cline
Scope: Validate claims in `experiments/reports/skeleton_vs_local_analysis.md` against available logs, configs, and source code.

## Verdict Legend

- **Verified**: claim is directly supported by available artifacts.
- **Partially Verified**: claim is broadly correct but has caveats.
- **Not Verifiable (Current Artifacts)**: implementation exists or claim is plausible, but numeric evidence is not present in inspected logs/artifacts.

## Evidence Matrix

| # | Claim (from analysis report) | Evidence source(s) | Verdict | Notes |
|---|---|---|---|---|
| 1 | Skeleton run epoch accuracies: 0.9666, 0.9722, 0.9778, 0.9771, 0.9770 | `experiments/logs/skeleton_run.log` lines 1–5 | **Verified** | Exact numeric match. |
| 2 | Skeleton best test accuracy: 0.9778 | `experiments/logs/skeleton_run.log` line 6 | **Verified** | Exact numeric match. |
| 3 | Local run epoch accuracies: 0.9800, 0.9864, 0.9892, 0.9898, 0.9911 | `experiments/logs/local_run.log` lines 1–5 | **Verified** | Exact numeric match. |
| 4 | Local best test accuracy: 0.9911 | `experiments/logs/local_run.log` line 6 | **Verified** | Exact numeric match. |
| 5 | Absolute gap = 0.0133 (1.33 pp) | Derived from claims #2 and #4 | **Verified** | `0.9911 - 0.9778 = 0.0133`. |
| 6 | Relative gap vs local baseline ≈ 1.34% | Derived from claims #2 and #4 | **Verified** | `0.0133 / 0.9911 ≈ 0.0134` (1.34%). |
| 7 | Same backbone architecture used | `src/local_cnn/train_local_cnn.py` line 15, line 47; `src/novel_skeleton/train_skeleton_cnn.py` line 17, line 137 | **Verified** | Both use `SimpleCNN`; skeleton run sets `in_channels` by mode. |
| 8 | Same core training hyperparameters (epochs=5, batch=128, lr=0.001, seed=42) | `experiments/configs/local_cnn.yaml` lines 3–6; `experiments/configs/skeleton_cnn.yaml` lines 3–6 | **Verified** | Config defaults align across compared runs. |
| 9 | Skeletonization uses fixed threshold `>30` | `src/novel_skeleton/skeletonize.py` line 10 | **Verified** | Code explicitly thresholds with `img_2d > 30`. |
| 10 | Hough-augmented mode implemented with tunable params and 2-channel option | `src/novel_skeleton/train_skeleton_cnn.py` lines 25–38, 49–67, 171–179 | **Verified** | `channel_mode` + Hough params exist and are wired through input pipeline. |
| 11 | Hough numeric results for two parameter settings (single-seed section) | Claimed in `skeleton_vs_local_analysis.md` lines 72–95 | **Not Verifiable (Current Artifacts)** | No dedicated Hough run logs/checkpoints were inspected in `experiments/logs/` beyond baseline skeleton/local logs. |
| 12 | Multi-seed table (seeds 42/123/999) and means | Claimed in `skeleton_vs_local_analysis.md` lines 97–120 | **Not Verifiable (Current Artifacts)** | No per-seed logs/artifacts for these runs were present in inspected evidence set. |

## Audit Conclusion

1. **Core Skeleton vs Local comparison is strongly supported** by raw logs and code/config alignment.
2. **Computed gap metrics are correct** based on logged best accuracies.
3. **Hough and multi-seed numeric claims remain unproven from currently available log artifacts**, even though the implementation capability is present in code.

## Confidence Assessment

- **High confidence**: claims #1–10.
- **Low confidence pending more evidence**: claims #11–12.

## What evidence is needed to fully close open claims

To fully verify the Hough and multi-seed sections, add/locate logs for:

- `channel_mode=skeleton_hough --hough-threshold 6 --hough-line-length 4 --hough-line-gap 1`
- `channel_mode=skeleton_hough --hough-threshold 8 --hough-line-length 5 --hough-line-gap 2`
- each with seeds `42`, `123`, and `999` (plus baseline skeleton seeds if means are compared directly).
