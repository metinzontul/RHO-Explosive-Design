# -*- coding: utf-8 -*-
"""
Parity plots of the RF pressure and velocity surrogates (Figures 2a and 2b).
Uses the closed-balance K-J model and checks the errors against Logs.txt.

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

script_dir = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(script_dir, "Logs.txt")
if not os.path.exists(log_path):
    log_path = "Logs.txt"

if os.path.exists(log_path):
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        log_txt = f.read()
    m = re.search(
        r"Random Forest \(RHO\)\s*\|\s*Global -> R2:\s*([\d.]+),\s*MAE:\s*([\d.]+)",
        log_txt,
    )
    if m:
        assert np.isclose(r2_p, float(m.group(1)), atol=1e-4), "R2 mismatch with Logs.txt!"
        print(f"Verified consistency with Logs.txt: R2={r2_p:.4f}, MAE={mae_p:.3f} GPa")

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 12

fig1, ax1 = plt.subplots(figsize=(7.5, 6.2))
ax1.scatter(yp_te, test_pred_p, color="royalblue", alpha=0.65, edgecolors="k", s=42, label="Test Data")
p_min, p_max = min(yp_te), max(yp_te)
ax1.plot([p_min, p_max], [p_min, p_max], "r--", lw=2, label="Ideal Fit (y = x)")
ax1.set_title("Detonation Pressure Parity Plot", fontsize=14, fontweight="bold", pad=12)
ax1.set_xlabel("K–J Target Pressure (GPa)", fontsize=12.5, fontweight="bold")
ax1.set_ylabel("RF Predicted Pressure (GPa)", fontsize=12.5, fontweight="bold")
ax1.text(
    0.05, 0.83,
    f"R² = {r2_p:.4f}\nMAE = {mae_p:.2f} GPa\nMAPE = {mape_p:.2f}%",
    transform=ax1.transAxes, fontsize=11.5,
    bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"),
)
ax1.grid(True, linestyle=":", alpha=0.7)
ax1.legend(loc="lower right", framealpha=0.95)
plt.tight_layout()
plt.savefig("Figure2a_Detonation_Pressure_Parity.png", dpi=600, bbox_inches="tight")
plt.show()

fig2, ax2 = plt.subplots(figsize=(7.5, 6.2))
ax2.scatter(yd_te, test_pred_d, color="forestgreen", alpha=0.65, edgecolors="k", s=42, label="Test Data")
d_min, d_max = min(yd_te), max(yd_te)
ax2.plot([d_min, d_max], [d_min, d_max], "r--", lw=2, label="Ideal Fit (y = x)")
ax2.set_title("Detonation Velocity Parity Plot", fontsize=14, fontweight="bold", pad=12)
ax2.set_xlabel("K–J Target Velocity (m/s)", fontsize=12.5, fontweight="bold")
ax2.set_ylabel("RF Predicted Velocity (m/s)", fontsize=12.5, fontweight="bold")
ax2.text(
    0.05, 0.83,
    f"R² = {r2_d:.4f}\nMAE = {mae_d:.2f} m/s\nMAPE = {mape_d:.2f}%",
    transform=ax2.transAxes, fontsize=11.5,
    bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"),
)
ax2.grid(True, linestyle=":", alpha=0.7)
ax2.legend(loc="lower right", framealpha=0.95)
plt.tight_layout()
plt.savefig("Figure2b_Detonation_Velocity_Parity.png", dpi=600, bbox_inches="tight")
plt.show()
