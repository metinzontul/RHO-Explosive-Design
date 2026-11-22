# -*- coding: utf-8 -*-
"""
Cross-Validation Boxplots for the RF Surrogate (Figure 3)
Evaluates 10 repeats (seeds 10..100) x 5 folds with closed C-H-N-O balance.

Author: Prof. Dr. Metin Zontul
Date: November 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
import warnings

warnings.filterwarnings("ignore")


def calculate_KJ(w_rdx, w_tnt, w_htpb):
    """
    Closed-balance Kistiakowsky-Wilson (K-W) & Kamlet-Jacobs (K-J) thermochemical
    calculation on a 1 g formulation basis with explicit C-H-N-O atom conservation:
      - Water:           n_H2O  = min(H_g / 2, O_g)
      - Free hydrogen:   n_H2   = (H_g - 2 * n_H2O) / 2
      - Carbon monoxide: n_CO0  = min(C_g, O_rem1)
      - Carbon dioxide:  n_CO2  = min(n_CO0, O_rem2)
      - Net CO:          n_CO   = n_CO0 - n_CO2   (prevents carbon double counting)
      - Solid carbon:    n_Cs   = C_g - n_CO - n_CO2 (excluded from gas yield N)
      - Nitrogen:        n_N2   = N_g / 2
    """
    total = w_rdx + w_tnt + w_htpb
    w_rdx, w_tnt, w_htpb = w_rdx / total, w_tnt / total, w_htpb / total

    rho_rdx, rho_tnt, rho_htpb = 1.80, 1.65, 0.92
    rho_mix = 1.0 / (w_rdx / rho_rdx + w_tnt / rho_tnt + w_htpb / rho_htpb)

    hf_rdx, hf_tnt, hf_htpb = 66.18, -70.44, -350.0
    hf_mix = (w_rdx * hf_rdx) + (w_tnt * hf_tnt) + (w_htpb * hf_htpb)

    C_g = w_rdx * (3.0 / 222.12) + w_tnt * (7.0 / 227.13) + w_htpb * (7.3 / 99.0)
    H_g = w_rdx * (6.0 / 222.12) + w_tnt * (5.0 / 227.13) + w_htpb * (10.9 / 99.0)
    N_g = w_rdx * (6.0 / 222.12) + w_tnt * (3.0 / 227.13) + w_htpb * 0.0
    O_g = w_rdx * (6.0 / 222.12) + w_tnt * (6.0 / 227.13) + w_htpb * (0.05 / 99.0)

    H2O_moles = min(H_g / 2.0, O_g)
    H2_moles = max(0.0, (H_g - 2.0 * H2O_moles) / 2.0)
    O_rem = O_g - H2O_moles

    CO_initial = min(C_g, O_rem)
    O_rem2 = O_rem - CO_initial
    CO2_moles = min(CO_initial, O_rem2)
    CO_moles = CO_initial - CO2_moles
    Cs_moles = max(0.0, C_g - CO_moles - CO2_moles)
    N2_moles = N_g / 2.0

    assert abs((CO_moles + CO2_moles + Cs_moles) - C_g) < 1e-12, "C balance error"
    assert abs((2.0 * H2O_moles + 2.0 * H2_moles) - H_g) < 1e-12, "H balance error"
    assert abs((2.0 * N2_moles) - N_g) < 1e-12, "N balance error"
    assert abs((H2O_moles + CO_moles + 2.0 * CO2_moles) - O_g) < 1e-12, "O balance error"

    N_gas = H2O_moles + H2_moles + CO_moles + CO2_moles + N2_moles
    gas_mass = (
        (H2O_moles * 18.015)
        + (H2_moles * 2.016)
        + (CO_moles * 28.010)
        + (CO2_moles * 44.010)
        + (N2_moles * 28.013)
    )
    M_gas = gas_mass / N_gas if N_gas > 0 else 0.0

    hf_H2O, hf_CO, hf_CO2 = -57790.0, -26420.0, -94050.0
    hf_products_g = (H2O_moles * hf_H2O) + (CO_moles * hf_CO) + (CO2_moles * hf_CO2)
    Q = -(hf_products_g - hf_mix)
    if Q < 0.0:
        Q = 0.0

    P = 1.558 * (rho_mix ** 2) * N_gas * np.sqrt(M_gas * Q)
    D = 1.01 * np.sqrt(N_gas * np.sqrt(M_gas * Q)) * (1.0 + 1.30 * rho_mix) * 1000.0

    return P, D


N_SAMPLES = 1500
np.random.seed(42)
data = []

while len(data) < N_SAMPLES:
    w1, w2, w3 = np.random.dirichlet((1, 1, 1))
    p, d = calculate_KJ(w1, w2, w3)
    if p > 0.1:
        data.append([w1, w2, w3, p, d])

df = pd.DataFrame(data, columns=["RDX", "TNT", "HTPB", "P", "D"])
X_all = df[["RDX", "TNT", "HTPB"]].values
y_p_all = df["P"].values
y_d_all = df["D"].values

seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
all_cv_p, all_cv_d = [], []

for seed in seeds:
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    rf_cv = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=seed)
    all_cv_p.extend(cross_val_score(rf_cv, X_all, y_p_all, cv=cv, scoring="r2"))
    all_cv_d.extend(cross_val_score(rf_cv, X_all, y_d_all, cv=cv, scoring="r2"))

print(f"CV Pressure R2: {np.mean(all_cv_p):.4f} ± {np.std(all_cv_p):.4f}")
print(f"CV Velocity R2: {np.mean(all_cv_d):.4f} ± {np.std(all_cv_d):.4f}")

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 12

fig, ax = plt.subplots(figsize=(8, 6))
data_box = [all_cv_p, all_cv_d]
box = ax.boxplot(
    data_box,
    patch_artist=True,
    widths=0.4,
    showmeans=True,
    meanprops={"marker": "D", "markerfacecolor": "red", "markeredgecolor": "black", "markersize": 8},
)

colors = ["royalblue", "forestgreen"]
for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

for median in box["medians"]:
    median.set(color="black", linewidth=2)

ax.set_xticklabels(["Detonation Pressure", "Detonation Velocity"], fontsize=12, fontweight="bold")
ax.set_ylabel("R² Score (10 Repeats × 5 Folds)", fontsize=12, fontweight="bold")
ax.set_title("Cross-Validation Stability of the RF Surrogate", fontsize=14, fontweight="bold")
ax.set_ylim(min(min(all_cv_p), min(all_cv_d)) - 0.005, 1.000)
ax.grid(True, linestyle=":", alpha=0.7)

ax.text(1, np.mean(all_cv_p) - 0.002, f"Mean: {np.mean(all_cv_p):.4f}", ha="center", fontsize=11, fontweight="bold")
ax.text(2, np.mean(all_cv_d) - 0.002, f"Mean: {np.mean(all_cv_d):.4f}", ha="center", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("Figure3_CV_Boxplots.png", dpi=600, bbox_inches="tight")
plt.show()
