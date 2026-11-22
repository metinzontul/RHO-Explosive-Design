# -*- coding: utf-8 -*-
"""
Figure 9: Direct K-J pressure of the candidate returned by each of the 30 runs,
for the RF, MLP and direct K-J search arms (common outcome scale).

Reads run_level_candidates.csv written by PBX_RHO_optimizer.py.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["font.size"] = 12

here = os.path.dirname(os.path.abspath(__file__))
runs = pd.read_csv(os.path.join(here, "run_level_candidates.csv"))

solvers = [
    ("DE (polish=False)", "DE\n(no polish)"),
    ("Dual Annealing (L-BFGS-B)", "Dual Annealing\n(L-BFGS-B)"),
    ("Dual Annealing (No Local)", "Dual Annealing\n(no local)"),
    ("Basin-Hopping (L-BFGS-B)", "Basin-Hopping\n(L-BFGS-B)"),
    ("Basin-Hopping (Nelder-Mead)", "Basin-Hopping\n(Nelder–Mead)"),
]
arms = [
    ("RF", "RF-predicted P (search score)", "#2a78d6", "o"),
    ("MLP", "MLP-predicted P (search score)", "#eb6834", "s"),
    ("Direct K-J", "Direct K–J P (search score)", "#1baf7a", "^"),
]
p_ref = (runs.KJ_P_GPa + runs.gap_to_ref_GPa).iloc[0]

fig, ax = plt.subplots(figsize=(11, 5.2))
rng = np.random.default_rng(0)
width = 0.26
for j, (arm, lab, col, mk) in enumerate(arms):
    for i, (s, _) in enumerate(solvers):
        y = runs[(runs.arm == arm) & (runs.solver == s)].KJ_P_GPa.values
        x0 = i + (j - 1) * width
        ax.boxplot(
            y, positions=[x0], widths=width * 0.8, showfliers=False, patch_artist=True,
            boxprops=dict(facecolor="white", edgecolor="#555555", linewidth=1),
            medianprops=dict(color="#222222", linewidth=1.5),
            whiskerprops=dict(color="#555555"), capprops=dict(color="#555555"),
        )
        ax.scatter(
            x0 + rng.uniform(-width * 0.3, width * 0.3, len(y)), y, s=22, color=col,
            marker=mk, edgecolor="white", linewidth=0.6, zorder=3,
            label=lab if i == 0 else None,
        )
ax.axhline(p_ref, color="#444444", linestyle="--", linewidth=1,
           label=f"Max direct K–J P in search region ({p_ref:.2f} GPa)")
ax.set_xticks(range(len(solvers)))
ax.set_xticklabels([b for _, b in solvers])
ax.set_ylabel("Direct K–J pressure at returned candidate (GPa)")
ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(title="Search objective", loc="lower right", frameon=False, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(here, "Figure9_Common_Outcome_KJ.png"), dpi=600, bbox_inches="tight")
print("Saved Figure9_Common_Outcome_KJ.png")
