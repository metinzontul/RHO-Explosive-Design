# -*- coding: utf-8 -*-
"""
Integrated Parity Plot Generator for RHO Surrogate Model (Figure 2a & Figure 2b)
Verifies consistency against Logs.txt and outputs separate 600 DPI publication plots:
- Figure2a_Detonation_Pressure_Parity.png
- Figure2b_Detonation_Velocity_Parity.png

Author: Prof. Dr. Metin Zontul
Date: November 2026
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")


# --- 1. Verified Kamlet-Jacobs Physical Baseline ---
def calculate_KJ(w_rdx, w_tnt, w_htpb):
    total = w_rdx + w_tnt + w_htpb
    w_rdx, w_tnt, w_htpb = w_rdx / total, w_tnt / total, w_htpb / total

    rho_rdx, rho_tnt, rho_htpb = 1.80, 1.65, 0.92
    rho_mix = 1.0 / (w_rdx / rho_rdx + w_tnt / rho_tnt + w_htpb / rho_htpb)

    hf_rdx, hf_tnt, hf_htpb = 66.18, -70.44, -350.0
    hf_mix = (w_rdx * hf_rdx) + (w_tnt * hf_tnt) + (w_htpb * hf_htpb)

    C_g = w_rdx * (3 / 222.12) + w_tnt * (7 / 227.13) + w_htpb * (7.3 / 99.0)
    H_g = w_rdx * (6 / 222.12) + w_tnt * (5 / 227.13) + w_htpb * (10.9 / 99.0)
    N_g = w_rdx * (6 / 222.12) + w_tnt * (3 / 227.13) + w_htpb * (0.0)
    O_g = w_rdx * (6 / 222.12) + w_tnt * (6 / 227.13) + w_htpb * (0.05 / 99.0)

    H2O_moles = min(H_g / 2, O_g)
    O_rem = O_g - H2O_moles
    CO_moles = min(C_g, O_rem)
    O_rem = O_rem - CO_moles
    CO2_moles = min(CO_moles, O_rem)
    N2_moles = N_g / 2

    N_gas = H2O_moles + CO_moles + CO2_moles + N2_moles
    gas_mass = (
        (H2O_moles * 18.015)
        + (CO_moles * 28.01)
        + (CO2_moles * 44.01)
        + (N2_moles * 28.013)
    )
    M_gas = gas_mass / N_gas if N_gas > 0 else 0

    hf_H2O, hf_CO, hf_CO2 = -57790, -26420, -94050
    hf_products_g = (H2O_moles * hf_H2O) + (CO_moles * hf_CO) + (CO2_moles * hf_CO2)
    Q = -(hf_products_g - hf_mix)
    if Q < 0:
        Q = 0

    P = 1.558 * (rho_mix ** 2) * N_gas * np.sqrt(M_gas * Q)
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

df = pd.DataFrame(data, columns=["RDX", "TNT", "HTPB", "P", "D"])
X_all = df[["RDX", "TNT", "HTPB"]].values
y_p_all = df["P"].values
y_d_all = df["D"].values

# --- 3. Train/Test Split & Model Training (Consistent max_depth=15) ---
X_tr, X_te, yp_tr, yp_te, yd_tr, yd_te = train_test_split(
    X_all, y_p_all, y_d_all, test_size=0.2, random_state=42
)

rf_p = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
rf_p.fit(X_tr, yp_tr)
test_pred_p = rf_p.predict(X_te)
r2_p = r2_score(yp_te, test_pred_p)
mape_p = mean_absolute_percentage_error(yp_te, test_pred_p) * 100
mae_p = mean_absolute_error(yp_te, test_pred_p)

rf_d = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
rf_d.fit(X_tr, yd_tr)
test_pred_d = rf_d.predict(X_te)
r2_d = r2_score(yd_te, test_pred_d)
mape_d = mean_absolute_percentage_error(yd_te, test_pred_d) * 100
mae_d = mean_absolute_error(yd_te, test_pred_d)

# Verify consistency with Logs.txt if present
if os.path.exists("Logs.txt"):
    with open("Logs.txt", "r", encoding="utf-8", errors="ignore") as f:
        log_txt = f.read()
    m = re.search(
        r"Random Forest \(RHO\)\s*\|\s*Global -> R2:\s*([\d.]+),\s*MAE:\s*([\d.]+)",
        log_txt,
    )
    if m:
        assert np.isclose(r2_p, float(m.group(1)), atol=1e-4), "R2 mismatch with Logs.txt!"
        print(f"Verified consistency with Logs.txt: R2={r2_p:.4f}, MAE={mae_p:.3f} GPa")

# --- 4. Plotting (Separate 600 DPI Publication Parity Plots) ---
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 12

# ==========================================
# FIGURE 2a: DETONATION PRESSURE PARITY PLOT
# ==========================================
fig1, ax1 = plt.subplots(figsize=(7.5, 6.2))
ax1.scatter(
    yp_te, test_pred_p, color="royalblue", alpha=0.65, edgecolors="k", s=42, label="Test Data"
)
p_min, p_max = min(yp_te), max(yp_te)
ax1.plot([p_min, p_max], [p_min, p_max], "r--", lw=2, label="Ideal Fit (y = x)")

ax1.set_title("Detonation Pressure Parity Plot", fontsize=14, fontweight="bold", pad=12)
ax1.set_xlabel("Actual Pressure (GPa)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Predicted Pressure (GPa)", fontsize=12, fontweight="bold")
ax1.text(
    0.05,
    0.83,
    f"R² = {r2_p:.4f}\nMAE = {mae_p:.2f} GPa\nMAPE = {mape_p:.2f}%",
    transform=ax1.transAxes,
    fontsize=11.5,
    bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"),
)
ax1.grid(True, linestyle=":", alpha=0.7)
ax1.legend(loc="lower right", framealpha=0.95)

plt.tight_layout()
plt.savefig("Figure2a_Detonation_Pressure_Parity.png", dpi=600, bbox_inches="tight")
plt.show()

# ==========================================
# FIGURE 2b: DETONATION VELOCITY PARITY PLOT
# ==========================================
fig2, ax2 = plt.subplots(figsize=(7.5, 6.2))
ax2.scatter(
    yd_te, test_pred_d, color="forestgreen", alpha=0.65, edgecolors="k", s=42, label="Test Data"
)
d_min, d_max = min(yd_te), max(yd_te)
ax2.plot([d_min, d_max], [d_min, d_max], "r--", lw=2, label="Ideal Fit (y = x)")

ax2.set_title("Detonation Velocity Parity Plot", fontsize=14, fontweight="bold", pad=12)
ax2.set_xlabel("Actual Velocity (m/s)", fontsize=12, fontweight="bold")
ax2.set_ylabel("Predicted Velocity (m/s)", fontsize=12, fontweight="bold")
ax2.text(
    0.05,
    0.83,
    f"R² = {r2_d:.4f}\nMAE = {mae_d:.2f} m/s\nMAPE = {mape_d:.2f}%",
    transform=ax2.transAxes,
    fontsize=11.5,
    bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"),
)
ax2.grid(True, linestyle=":", alpha=0.7)
ax2.legend(loc="lower right", framealpha=0.95)

plt.tight_layout()
plt.savefig("Figure2b_Detonation_Velocity_Parity.png", dpi=600, bbox_inches="tight")
plt.show()