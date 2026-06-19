"""Figure for the skeletonization-negative paper (numbers from skeleton_comparison.md,
scripts/run_skeleton_experiment.py, 3 seeds)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG = os.path.join(ROOT, "experiments", "reports", "figures")
os.makedirs(FIG, exist_ok=True)

# overall accuracy (mean%, std) -- skeleton_comparison.md
configs = [("raw", 99.03, 0.06), ("Guo-Hall", 98.11, 0.09), ("Lee", 98.02, 0.07),
           ("medial-axis", 97.95, 0.11), ("Zhang-Suen", 97.81, 0.04),
           ("raw+skel\n(fusion)", 98.98, 0.05)]
colors = ["#2c3e50", "#c0392b", "#c0392b", "#c0392b", "#c0392b", "#e67e22"]

# per-class accuracy: raw vs Guo-Hall ("thin") skeleton
raw_pc = [99.7, 99.5, 98.9, 99.3, 99.1, 99.0, 98.7, 99.3, 98.8, 97.9]
skel_pc = [99.8, 99.2, 98.5, 97.6, 97.9, 97.4, 98.0, 98.2, 97.6, 96.7]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))

names = [c[0] for c in configs]
means = [c[1] for c in configs]
errs = [c[2] for c in configs]
a1.bar(range(len(configs)), means, yerr=errs, color=colors, capsize=3)
a1.set_xticks(range(len(configs)))
a1.set_xticklabels(names, fontsize=8)
a1.set_ylim(97.0, 99.4)
a1.axhline(99.03, color="#2c3e50", ls="--", lw=0.8, alpha=0.6)
a1.set_ylabel("MNIST test accuracy (%)")
a1.set_title("Every skeletonizer (red) trails raw pixels;\nfusion (orange) does not recover it")
a1.grid(True, axis="y", ls="--", alpha=0.3)

x = np.arange(10)
a2.bar(x - 0.2, raw_pc, 0.4, label="raw", color="#2c3e50")
a2.bar(x + 0.2, skel_pc, 0.4, label="Guo-Hall skeleton", color="#c0392b")
a2.set_xticks(x); a2.set_xticklabels(range(10))
a2.set_ylim(95, 100); a2.set_xlabel("digit")
a2.set_ylabel("per-class accuracy (%)")
a2.set_title("Loss concentrates on stroke-width-sensitive\ndigits (3, 5, 9, 8, 4)")
a2.legend(fontsize=8, loc="lower left")
a2.grid(True, axis="y", ls="--", alpha=0.3)

plt.tight_layout()
out = os.path.join(FIG, "fig_skeleton_negative.png")
plt.savefig(out, dpi=180, facecolor="white")
print("saved", out)
