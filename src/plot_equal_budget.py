# -*- coding: utf-8 -*-
"""
Figure 10: Gap to the regional maximum of direct K-J pressure versus the number of
direct K-J calls, for surrogate-assisted search (RF, MLP) and direct search.
Reads equal_budget_summary.csv written by PBX_RHO_optimizer.py (Section 9).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["font.size"] = 12

here = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(here, "equal_budget_summary.csv"))

arms = [
    ("RF", "RF surrogate + search", "#2a78d6", "o", "-"),
    ("MLP", "MLP surrogate + search", "#eb6834", "s", "-"),
    ("Direct K-J", "Direct K–J search (cap = budget)", "#1baf7a", "^", "--"),
]
panels = [("DE (polish=False)", "(a) Differential Evolution (no polish)"),
          ("Basin-Hopping (L-BFGS-B)", "(b) Basin-Hopping (L-BFGS-B)")]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
for ax, (solver, title) in zip(axes, panels):
    for arm, lab, col, mk, ls in arms:
        s = d[(d.arm == arm) & (d.solver == solver)].sort_values("budget")
        ax.errorbar(s.KJ_calls, s.gap_mean, yerr=s.gap_sd, color=col, marker=mk, linestyle=ls,
                    linewidth=2, markersize=7, capsize=3, label=lab,
                    markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xscale("log")
    ax.set_title(title, fontsize=12)
    ax.grid(color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("Gap to regional maximum K–J pressure (GPa)")
axes[0].set_ylim(bottom=-0.05)
axes[1].legend(loc="upper right", frameon=False, fontsize=10)
fig.supxlabel("Direct K–J calls per run (surrogate: training draws + 1 recalculation; direct: search evaluations)",
              fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(here, "Figure10_Equal_Budget.png"), dpi=600, bbox_inches="tight")
print("Saved Figure10_Equal_Budget.png")
