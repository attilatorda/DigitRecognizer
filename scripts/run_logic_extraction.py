"""
Track 8.3 -- can we extract logic (e.g. XOR) from a neural network?

We take the cleanest, fully-demonstrable case: minimal MLPs trained on 2-input Boolean
functions. For each gate we (a) train a 2-2-1 MLP to fit the truth table, (b) EXTRACT the
recovered logic by enumerating the four inputs, (c) for XOR read out the mechanism from the
weights (each hidden unit is a linear threshold = a Boolean primitive; the output composes
them), and (d) show the textbook control: a 0-hidden linear unit solves the linearly separable
gates (AND/OR) but cannot fit XOR.

Usage:  python scripts/run_logic_extraction.py
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
GATES = {
    "AND":  [0, 0, 0, 1], "OR":   [0, 1, 1, 1], "XOR":  [0, 1, 1, 0],
    "XNOR": [1, 0, 0, 1], "NAND": [1, 1, 1, 0], "NOR":  [1, 0, 0, 0],
}
LINEARLY_SEPARABLE = {"AND", "OR", "NAND", "NOR"}  # XOR/XNOR are not


def _fit(net, X, y, steps=4000, lr=0.08):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        opt.zero_grad()
        crit(net(X), y).backward()
        opt.step()
    with torch.no_grad():
        pred = (torch.sigmoid(net(X)) > 0.5).int().squeeze(1).tolist()
        loss = crit(net(X), y).item()
    return pred, loss


def _fit_until(make_net, X, y, target, tries=12):
    """Retry seeds until the net reproduces the truth table (or give best)."""
    best = None
    for s in range(tries):
        torch.manual_seed(s)
        net = make_net()
        pred, loss = _fit(net, X, y)
        if pred == target:
            return net, pred, loss, s
        if best is None or loss < best[2]:
            best = (net, pred, loss, s)
    return best


def _bool_pattern(vals):
    """Map 4 real activations to a Boolean truth-table tuple via sign>0."""
    return tuple(int(v > 0) for v in vals)


_PRIM = {(0, 0, 0, 1): "a^b (AND)", (0, 1, 1, 1): "a|b (OR)", (1, 1, 1, 0): "NAND",
         (1, 0, 0, 0): "~a^~b (NOR)", (0, 1, 1, 0): "XOR", (1, 0, 0, 1): "XNOR",
         (0, 0, 0, 0): "FALSE", (1, 1, 1, 1): "TRUE",
         (1, 0, 1, 0): "~b", (0, 1, 0, 1): "b", (1, 1, 0, 0): "~a", (0, 0, 1, 1): "a",
         (0, 1, 0, 0): "~a^b", (0, 0, 1, 0): "a^~b"}


def main():
    X = torch.tensor(INPUTS)
    results = {}
    print("=== Track 8.3: logic extraction from minimal MLPs ===\n")
    xor_net = None
    for gate, truth in GATES.items():
        y = torch.tensor(truth, dtype=torch.float32).unsqueeze(1)
        net, pred, loss, seed = _fit_until(
            lambda: nn.Sequential(nn.Linear(2, 2), nn.Tanh(), nn.Linear(2, 1)), X, y, truth)
        # linear control (0 hidden units)
        lin, lin_pred, _ = (lambda r: (r[0], r[1], r[2]))(_fit_until(
            lambda: nn.Linear(2, 1), X, y, truth)[:3])
        lin_acc = float(np.mean(np.array(lin_pred) == np.array(truth)))
        results[gate] = {"target": truth, "recovered_mlp": pred,
                         "mlp_exact": pred == truth, "linear_acc": lin_acc,
                         "linearly_separable": gate in LINEARLY_SEPARABLE}
        print(f"{gate:4s} target={truth}  MLP recovered={pred}  exact={pred==truth}  "
              f"| linear(0-hidden) acc={lin_acc*100:3.0f}%")
        if gate == "XOR":
            xor_net = net

    # --- XOR mechanism read-out ---
    print("\n=== XOR mechanism (from the 2 hidden units) ===")
    W1 = xor_net[0].weight.detach().numpy(); b1 = xor_net[0].bias.detach().numpy()
    W2 = xor_net[2].weight.detach().numpy()[0]; b2 = float(xor_net[2].bias.detach().numpy()[0])
    with torch.no_grad():
        pre = (X @ torch.tensor(W1.T) + torch.tensor(b1)).numpy()   # (4,2) pre-activation
    prims = []
    for j in range(2):
        patt = _bool_pattern(pre[:, j])
        name = _PRIM.get(patt, str(patt))
        prims.append(name)
        print(f"  hidden unit {j}: w={W1[j].round(2).tolist()} b={b1[j]:+.2f}  "
              f"-> threshold pattern {patt} = {name}")
    print(f"  output: w={W2.round(2).tolist()} b={b2:+.2f} over (h0={prims[0]}, h1={prims[1]})")
    print(f"  => XOR is composed from the primitives [{prims[0]}, {prims[1]}] "
          f"(two lines carve the four corners).")
    results["XOR"]["mechanism"] = {"hidden_primitives": prims,
                                   "W1": W1.tolist(), "b1": b1.tolist(),
                                   "W2": W2.tolist(), "b2": b2}

    # --- figure: XOR input space + the two hidden-unit decision lines ---
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for (a, b), t in zip(INPUTS, GATES["XOR"]):
        ax.scatter(a, b, s=260, c=("#27ae60" if t else "#c0392b"),
                   edgecolors="k", zorder=3, marker=("o" if t else "s"))
        ax.annotate(f"{int(a)}{int(b)}→{t}", (a, b), textcoords="offset points",
                    xytext=(8, 8), fontsize=9)
    xs = np.linspace(-0.4, 1.4, 100)
    for j, name in enumerate(prims):
        if abs(W1[j, 1]) > 1e-3:
            ax.plot(xs, -(W1[j, 0] * xs + b1[j]) / W1[j, 1], "--",
                    label=f"hidden {j} ({name})")
    ax.set_xlim(-0.4, 1.4); ax.set_ylim(-0.4, 1.4)
    ax.set_xlabel("input a"); ax.set_ylabel("input b")
    ax.set_title("XOR extracted: two hidden units = two lines\n(green o = 1, red ■ = 0)")
    ax.legend(loc="upper right", fontsize=8); ax.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    fig_path = os.path.join(ROOT, "experiments", "reports", "figures", "fig_track8_logic.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=180, facecolor="white"); plt.close()
    print(f"\n[logic] saved {fig_path}")

    out = os.path.join(ROOT, "experiments", "reports", "track8_logic_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[logic] saved {out}")

    n_exact = sum(r["mlp_exact"] for r in results.values())
    xor_lin = results["XOR"]["linear_acc"]
    print(f"\nSummary: {n_exact}/6 gates exactly recovered by the 2-2-1 MLP; "
          f"linear net on XOR = {xor_lin*100:.0f}% (cannot exceed 75%).")


if __name__ == "__main__":
    main()
