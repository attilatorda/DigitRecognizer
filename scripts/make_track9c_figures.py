"""Figures for Track 9c (corruption robustness + leave-one-corruption-out)."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REP = os.path.join(ROOT, "experiments", "reports")
FIG = os.path.join(REP, "figures")
os.makedirs(FIG, exist_ok=True)

rob = json.load(open(os.path.join(REP, "track9_robust_results.json")))
loco = json.load(open(os.path.join(REP, "track9_loco_results.json")))
B = rob["budgets"]


def m(line, n):
    v = rob["mca"][line][str(n)]
    return v["mean"] * 100 if v else np.nan


def s(line, n):
    v = rob["mca"][line][str(n)]
    return v["std"] * 100 if v else np.nan


def acc(line, c, n, src=rob):
    v = src["accuracy"][line][c][str(n)] if line in src["accuracy"] else None
    return v["mean"] * 100 if v else np.nan


# ---- fig9: mean corruption accuracy vs budget ----
fig, ax = plt.subplots(figsize=(7.0, 4.4))
for line, c, mk, ls, lab in [
    ("record_vote", "#c0392b", "s", "-", "Record holder (clean-trained)"),
    ("record_aug", "#8e44ad", "D", "--", "Record holder + corruption aug (saw corruptions)"),
    ("conf_all", "#27ae60", "^", "-", "+ structural + diffusion (agnostic)"),
    ("soft_all", "#7f8c8d", "o", ":", "+ structural + diffusion (soft vote)"),
]:
    ys = [m(line, n) for n in B]
    es = [s(line, n) for n in B]
    ax.errorbar(B, ys, yerr=es, marker=mk, color=c, ls=ls, capsize=2, label=lab, ms=5)
ax.set_xscale("log")
ax.set_xlabel("Labeled MNIST images (budget n)")
ax.set_ylabel("Mean corruption accuracy (mCA, %)")
ax.set_title("Track 9c: low-data corruption robustness")
ax.legend(fontsize=7.5, loc="lower right")
ax.grid(True, ls="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig9_track9c_mca.png"), dpi=180, facecolor="white")
plt.close()

# ---- fig10: LOCO unseen-corruption (the headline positive) ----
held = loco["held_out"]
LB = loco["budgets"]
fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), sharex=True)
for ax, h in zip(axes.ravel(), held):
    ax.plot(LB, [acc("record_vote", h, n) for n in LB], "s:", color="#c0392b", ms=4, label="clean-trained")
    ax.plot(LB, [acc("record_aug", h, n) for n in LB], "D--", color="#8e44ad", ms=4,
            label="aug on THIS corruption (unfair)")
    ax.plot(LB, [loco["record_aug_loco"][h][str(n)]["mean"] * 100 for n in LB], "v-", color="#e67e22",
            ms=4, label="aug on OTHER corruptions (unseen h)")
    ax.plot(LB, [acc("conf_all", h, n) for n in LB], "^-", color="#27ae60", ms=5,
            label="agnostic ensemble (ours)")
    ax.set_xscale("log")
    ax.set_title(h, fontsize=10)
    ax.grid(True, ls="--", alpha=0.4)
axes[1, 0].set_xlabel("budget n"); axes[1, 1].set_xlabel("budget n")
axes[0, 0].set_ylabel("test accuracy (%)"); axes[1, 0].set_ylabel("test accuracy (%)")
axes[0, 0].legend(fontsize=7, loc="lower right")
fig.suptitle("Track 9c LOCO: generalization to an UNSEEN corruption (agnostic vs targeted augmentation)",
             fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig10_track9c_loco.png"), dpi=180, facecolor="white")
plt.close()

# ---- fig11: per-corruption gain + member ablation ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
corrs = [c for c in rob["corruptions"] if c != "clean"]
x = np.arange(len(corrs))
for i, n in enumerate([100, 500, 2000]):
    deltas = [acc("conf_all", c, n) - acc("record_vote", c, n) for c in corrs]
    a1.bar(x + (i - 1) * 0.26, deltas, width=0.26, label=f"n={n}")
a1.axhline(0, color="k", lw=0.8)
a1.set_xticks(x); a1.set_xticklabels(corrs, rotation=20, ha="right", fontsize=8)
a1.set_ylabel("conf_all − record_vote (pp)")
a1.set_title("Per-corruption gain over clean-trained baseline")
a1.legend(fontsize=8); a1.grid(True, axis="y", ls="--", alpha=0.4)

struct = [m("conf_all", n) - m("conf_drop_struct", n) for n in B]
diff = [m("conf_all", n) - m("conf_drop_diff", n) for n in B]
a2.plot(B, struct, "^-", color="#2980b9", label="structural member")
a2.plot(B, diff, "o-", color="#c0392b", label="diffusion member")
a2.axhline(0, color="k", lw=0.8)
a2.set_xscale("log"); a2.set_xlabel("budget n")
a2.set_ylabel("mCA contribution (pp, leave-one-out)")
a2.set_title("Which member drives the robustness gain")
a2.legend(fontsize=8); a2.grid(True, ls="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig11_track9c_attribution.png"), dpi=180, facecolor="white")
plt.close()

print("saved fig9_track9c_mca.png, fig10_track9c_loco.png, fig11_track9c_attribution.png")
