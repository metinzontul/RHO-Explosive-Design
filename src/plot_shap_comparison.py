# -*- coding: utf-8 -*-
"""
Figure 11: SHAP attributions of the fitted surrogates versus the attributions of the known
K-J pressure model. (a) Mean |SHAP| per component; (b) per-composition RF attributions
against K-J attributions for the default TreeSHAP and the simplex-consistent explanation.
Reads shap_comparison_values.csv written by PBX_RHO_optimizer.py (Section 10).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["font.size"] = 12

here = os.path.dirname(os.path.abspath(__file__))
v = pd.read_csv(os.path.join(here, "shap_comparison_values.csv"))
FEATS = ["RDX", "TNT", "HTPB"]
EXPL = [
    ("K-J model (reference)", "K–J model (reference)", "#444444"),
    ("RF, TreeSHAP default (Fig. 7)", "RF, default TreeSHAP", "#86b6ef"),
    ("RF, simplex-consistent", "RF, simplex-consistent", "#2a78d6"),
    ("MLP, simplex-consistent", "MLP, simplex-consistent", "#eb6834"),
]
FEAT_COL = {"RDX": "#2a78d6", "TNT": "#eb6834", "HTPB": "#1baf7a"}
FEAT_MK = {"RDX": "o", "TNT": "s", "HTPB": "^"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})

# (a) mean |SHAP|
w = 0.2
x = np.arange(len(FEATS))
for k, (key, lab, col) in enumerate(EXPL):
    m = [v[(v.explanation == key) & (v.feature == f)].shap_GPa.abs().mean() for f in FEATS]
    ax1.bar(x + (k - 1.5) * w, m, width=w * 0.92, color=col, label=lab, edgecolor="white", linewidth=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(FEATS)
ax1.set_ylabel("Mean |SHAP| for pressure (GPa)")
ax1.set_title("(a) Attribution magnitude by component", fontsize=12)
ax1.legend(frameon=False, fontsize=10, loc="upper left")
ax1.grid(axis="y", color="#e5e5e5", linewidth=0.8)
ax1.set_axisbelow(True)

# (b) RF vs K-J per composition
ref = v[v.explanation == "K-J model (reference)"].set_index(["sample", "feature"]).shap_GPa
for key, filled in [("RF, TreeSHAP default (Fig. 7)", False), ("RF, simplex-consistent", True)]:
    d = v[v.explanation == key].set_index(["sample", "feature"]).shap_GPa
    for f in FEATS:
        idx = [(i, f) for i in sorted(v["sample"].unique())]
        xs, ys = ref.loc[idx].values, d.loc[idx].values
        if filled:
            ax2.scatter(xs, ys, s=16, color=FEAT_COL[f], marker=FEAT_MK[f], edgecolor="white", linewidth=0.4,
                        label=f"{f}, simplex-consistent", zorder=3)
        else:
            ax2.scatter(xs, ys, s=16, facecolor="none", edgecolor=FEAT_COL[f], marker=FEAT_MK[f], linewidth=0.7,
                        alpha=0.6, label=f"{f}, default TreeSHAP", zorder=2)
lim = [min(ax2.get_xlim()[0], ax2.get_ylim()[0]), max(ax2.get_xlim()[1], ax2.get_ylim()[1])]
ax2.plot(lim, lim, color="#888888", linestyle="--", linewidth=1, zorder=1)
ax2.set_xlim(lim)
ax2.set_ylim(lim)
ax2.set_xlabel("SHAP of the K–J model (GPa)")
ax2.set_ylabel("SHAP of the RF surrogate (GPa)")
ax2.set_title("(b) RF attributions vs K–J attributions", fontsize=12)
ax2.legend(frameon=False, fontsize=9, loc="upper left", ncol=1)
ax2.grid(color="#e5e5e5", linewidth=0.8)
ax2.set_axisbelow(True)
for ax in (ax1, ax2):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(here, "Figure11_SHAP_vs_KJ.png"), dpi=600, bbox_inches="tight")
print("Saved Figure11_SHAP_vs_KJ.png")
