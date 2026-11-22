# -*- coding: utf-8 -*-
"""
Direct K-J pressure and velocity of the five DE candidates on the maximum RF plateau
(Candidates A-E) and two reference compositions; candidates are read from Logs.txt.
Saves Figure 5 (Pressure) and Figure 6 (Velocity) at 600 DPI.

Author: Prof. Dr. Metin Zontul
Date: November 2026
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 13


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


def parse_top5_from_logs(log_path="Logs.txt"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, log_path)
    if not os.path.exists(full_path):
        full_path = log_path
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"'{log_path}' not found. Run PBX_RHO_optimizer.py > Logs.txt first.")
    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    pattern = re.compile(
        r"Rank (\d+):\s*RDX=([\d.]+)%,\s*TNT=([\d.]+)%,\s*HTPB=([\d.]+)%\s*\|\s*"
        r"RF_P=([\d.]+)\s*GPa,\s*Direct_KJ_P=([\d.]+)\s*GPa\s*\(\|Diff\|=[\d.]+\s*GPa\)\s*\|\s*"
        r"RF_D=([\d.]+)\s*m/s,\s*Direct_KJ_D=([\d.]+)\s*m/s"
    )
    matches = pattern.findall(content)
    if not matches:
        raise ValueError(f"Could not parse Top-5 candidate lines from {log_path}.")

    return [
        {
            "rank": int(m[0]),
            "rdx": float(m[1]),
            "tnt": float(m[2]),
            "htpb": float(m[3]),
            "rf_p": float(m[4]),
            "kj_p": float(m[5]),
            "rf_d": float(m[6]),
            "kj_d": float(m[7]),
        }
        for m in matches
    ]


top5 = parse_top5_from_logs("Logs.txt")
print(f"Successfully loaded {len(top5)} candidates from Logs.txt.")

P_pure, D_pure = calculate_KJ(1.0, 0.0, 0.0)
P_inert, D_inert = calculate_KJ(0.55, 0.20, 0.25)

labels = ["Pure RDX\n(100 / 0 / 0%)"]
pressures = [P_pure]
velocities = [D_pure]
colors = ["#B22222"]

# The five candidates share one bit-identical RF prediction (same RF plateau), so they are
# labelled A-E in run (seed) order rather than ranked (see Logs.txt, Section 7).
for k, item in enumerate(top5):
    labels.append(
        f"Candidate {chr(65 + k)} (DE)\n({item['rdx']:.2f} / {item['tnt']:.2f} / {item['htpb']:.2f}%)"
    )
    pressures.append(item["kj_p"])
    velocities.append(item["kj_d"])
    colors.append("#F0E442")

labels.append("High-HTPB Bound\n(55 / 20 / 25%)")
pressures.append(P_inert)
velocities.append(D_inert)
colors.append("#2B5C8F")

rf_p_ref = top5[0]["rf_p"]
rf_d_ref = top5[0]["rf_d"]
y_pos = np.arange(len(labels))
text_bbox = dict(facecolor="white", alpha=0.92, edgecolor="lightgray", boxstyle="round,pad=0.2")

fig1, ax1 = plt.subplots(figsize=(9.0, 6.8))
bars1 = ax1.barh(y_pos, pressures, height=0.60, color=colors, edgecolor="black", linewidth=1.3)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(labels, fontsize=12.5, fontweight="bold")
ax1.invert_yaxis()
ax1.set_xlabel("Direct K–J Detonation Pressure (GPa)", fontsize=13.5, fontweight="bold")
ax1.set_title("Theoretical Detonation Pressure Comparison", fontsize=15.5, fontweight="bold", pad=14)
ax1.tick_params(axis="x", labelsize=12)
ax1.set_xlim(0, 44)
ax1.grid(axis="x", linestyle="--", alpha=0.6)
ax1.axvline(
    rf_p_ref, color="darkgreen", linestyle=":", linewidth=2.3, zorder=4,
    label=f"RF prediction, all five candidates ({rf_p_ref:.2f} GPa)",
)
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), fontsize=12, framealpha=0.95, edgecolor="gray")
for bar in bars1:
    w = bar.get_width()
    ax1.text(
        w + 0.7, bar.get_y() + bar.get_height() / 2, f"{w:.2f} GPa",
        va="center", ha="left", fontsize=12.5, fontweight="bold", color="black",
        bbox=text_bbox, zorder=5,
    )
plt.tight_layout()
plt.savefig("Figure5_Detonation_Pressure.png", dpi=600, bbox_inches="tight")
plt.show()

fig2, ax2 = plt.subplots(figsize=(9.0, 6.8))
bars2 = ax2.barh(y_pos, velocities, height=0.60, color=colors, edgecolor="black", linewidth=1.3)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(labels, fontsize=12.5, fontweight="bold")
ax2.invert_yaxis()
ax2.set_xlabel("Direct K–J Detonation Velocity (m/s)", fontsize=13.5, fontweight="bold")
ax2.set_title("Theoretical Detonation Velocity Comparison", fontsize=15.5, fontweight="bold", pad=14)
ax2.tick_params(axis="x", labelsize=12)
ax2.set_xlim(0, 11800)
ax2.grid(axis="x", linestyle="--", alpha=0.6)
ax2.axvline(
    rf_d_ref, color="darkgreen", linestyle=":", linewidth=2.3, zorder=4,
    label=f"RF prediction at candidate A ({rf_d_ref:.2f} m/s)",
)
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), fontsize=12, framealpha=0.95, edgecolor="gray")
for bar in bars2:
    w = bar.get_width()
    ax2.text(
        w + 180, bar.get_y() + bar.get_height() / 2, f"{w:.2f} m/s",
        va="center", ha="left", fontsize=12.5, fontweight="bold", color="black",
        bbox=text_bbox, zorder=5,
    )
plt.tight_layout()
plt.savefig("Figure6_Detonation_Velocity.png", dpi=600, bbox_inches="tight")
plt.show()
