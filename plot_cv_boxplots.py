# -*- coding: utf-8 -*-
"""
Cross-Validation Boxplots for RHO Surrogate Engine
Updated to reflect the verified Full Simplex Kamlet-Jacobs formulation.

Author: Prof. Dr. Metin Zontul
Date: October 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

# --- 1. Verified Kamlet-Jacobs Physical Baseline ---
def calculate_KJ(w_rdx, w_tnt, w_htpb):
    total = w_rdx + w_tnt + w_htpb
    w_rdx, w_tnt, w_htpb = w_rdx/total, w_tnt/total, w_htpb/total

    rho_rdx, rho_tnt, rho_htpb = 1.80, 1.65, 0.92
    rho_mix = 1.0 / (w_rdx/rho_rdx + w_tnt/rho_tnt + w_htpb/rho_htpb)

    hf_rdx, hf_tnt, hf_htpb = 66.18, -70.44, -350.0
    hf_mix = (w_rdx * hf_rdx) + (w_tnt * hf_tnt) + (w_htpb * hf_htpb)

    C_g = w_rdx*(3/222.12) + w_tnt*(7/227.13) + w_htpb*(7.3/99.0)
    H_g = w_rdx*(6/222.12) + w_tnt*(5/227.13) + w_htpb*(10.9/99.0)
    N_g = w_rdx*(6/222.12) + w_tnt*(3/227.13) + w_htpb*(0.0)
    O_g = w_rdx*(6/222.12) + w_tnt*(6/227.13) + w_htpb*(0.05/99.0)

    H2O_moles = min(H_g / 2, O_g)
    O_rem = O_g - H2O_moles
    CO_moles = min(C_g, O_rem)
    O_rem = O_rem - CO_moles
    CO2_moles = min(CO_moles, O_rem)
    N2_moles = N_g / 2

    N_gas = H2O_moles + CO_moles + CO2_moles + N2_moles
    gas_mass = (H2O_moles*18.015) + (CO_moles*28.01) + (CO2_moles*44.01) + (N2_moles*28.013)
    M_gas = gas_mass / N_gas if N_gas > 0 else 0

    hf_H2O, hf_CO, hf_CO2 = -57790, -26420, -94050 
    hf_products_g = (H2O_moles * hf_H2O) + (CO_moles * hf_CO) + (CO2_moles * hf_CO2)
    Q = -(hf_products_g - hf_mix)
    if Q < 0: Q = 0

    P = 1.558 * (rho_mix**2) * N_gas * np.sqrt(M_gas * Q)
    D = 1.01 * np.sqrt(N_gas * np.sqrt(M_gas * Q)) * (1 + 1.30 * rho_mix) * 1000
    return P, D

# --- 2. Data Generation (Full Simplex) ---
N_SAMPLES = 1500
np.random.seed(42)
data = []

while len(data) < N_SAMPLES:
    w1, w2, w3 = np.random.dirichlet((1, 1, 1))
    p, d = calculate_KJ(w1, w2, w3)
    if p > 0.1:
        data.append([w1, w2, w3, p, d])

df = pd.DataFrame(data, columns=['RDX', 'TNT', 'HTPB', 'P', 'D'])
X_all = df[['RDX', 'TNT', 'HTPB']].values
y_p_all = df['P'].values
y_d_all = df['D'].values

# --- 3. 50-Iteration Multi-Seed CV ---
seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
all_cv_p, all_cv_d = [], []

for seed in seeds:
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    rf_cv = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=seed)
    all_cv_p.extend(cross_val_score(rf_cv, X_all, y_p_all, cv=cv, scoring='r2'))
    all_cv_d.extend(cross_val_score(rf_cv, X_all, y_d_all, cv=cv, scoring='r2'))

# --- 4. Plotting (Q1 Journal Standards) ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

fig, ax = plt.subplots(figsize=(8, 6))

data_box = [all_cv_p, all_cv_d]
box = ax.boxplot(data_box, patch_artist=True, widths=0.4, showmeans=True, 
                 meanprops={"marker":"D", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":8})

colors = ['royalblue', 'forestgreen']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

for median in box['medians']:
    median.set(color='black', linewidth=2)

ax.set_xticklabels(['Detonation Pressure', 'Detonation Velocity'], fontsize=12, fontweight='bold')
ax.set_ylabel('R² Score (50 Iterations)', fontsize=12, fontweight='bold')
ax.set_title('Cross-Validation Stability of RHO Surrogate Engine', fontsize=14, fontweight='bold')
# Adjusting ylim slightly to accommodate new realistic full-simplex scores
ax.set_ylim(min(min(all_cv_p), min(all_cv_d)) - 0.005, 1.000)
ax.grid(True, linestyle=':', alpha=0.7)

ax.text(1, np.mean(all_cv_p) - 0.002, f'Mean: {np.mean(all_cv_p):.4f}', horizontalalignment='center', fontsize=11, fontweight='bold')
ax.text(2, np.mean(all_cv_d) - 0.002, f'Mean: {np.mean(all_cv_d):.4f}', horizontalalignment='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('Figure2_Boxplots.png', dpi=600, bbox_inches='tight')
plt.show()