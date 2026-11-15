# -*- coding: utf-8 -*-
"""
Thermochemical Evaluation Charts for Top-5 RF-Selected Candidates & Reference Boundaries
Parses all Top-5 candidates, RF predictions, and direct K-J values directly from Logs.txt.
Saves Detonation Pressure and Detonation Velocity into two separate 600 DPI figures:
- Figure5a_Detonation_Pressure.png
- Figure5b_Detonation_Velocity.png

Author: Prof. Dr. Metin Zontul
Date: November 2026
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 12


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


def parse_top5_from_logs(log_path="Logs.txt"):
    """Parses Top-5 distinct DE candidates from Logs.txt."""
    if not os.path.exists(log_path):
        raise FileNotFoundError(
            f"'{log_path}' not found. Run PBX_RHO_optimizer.py and save output to {log_path} first."
        )
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    pattern = re.compile(
        r"Rank (\d+):\s*RDX=([\d.]+)%,\s*TNT=([\d.]+)%,\s*HTPB=([\d.]+)%\s*\|\s*"
        r"RF_P=([\d.]+)\s*GPa,\s*Direct_KJ_P=([\d.]+)\s*GPa\s*\(\|Diff\|=[\d.]+\s*GPa\)\s*\|\s*"
        r"RF_D=([\d.]+)\s*m/s,\s*Direct_KJ_D=([\d.]+)\s*m/s"
    )
    matches = pattern.findall(content)
    if not matches:
        raise ValueError(f"Could not parse Top-5 candidate lines from {log_path}.")

    parsed = []
    for m in matches:
        parsed.append(
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
        )
    return parsed


top5 = parse_top5_from_logs("Logs.txt")
print(f"Successfully loaded {len(top5)} candidates from Logs.txt.")

# Calculate reference baselines dynamically
P_pure, D_pure = calculate_KJ(1.0, 0.0, 0.0)
P_inert, D_inert = calculate_KJ(0.55, 0.20, 0.25)

labels = ["Pure RDX\n(100 / 0 / 0%)"]
pressures = [P_pure]
velocities = [D_pure]
colors = ["#B22222"]

for item in top5:
    labels.append(
        f"Rank {item['rank']} (DE)\n({item['rdx']:.2f} / {item['tnt']:.2f} / {item['htpb']:.2f}%)"
    )
    pressures.append(item["kj_p"])
    velocities.append(item["kj_d"])
    colors.append("#E69F00" if item["rank"] == 1 else "#F0E442")

labels.append("High-HTPB Bound\n(55 / 20 / 25%)")
pressures.append(P_inert)
velocities.append(D_inert)
colors.append("#2B5C8F")

rf_p_ref = top5[0]["rf_p"]
rf_d_ref = top5[0]["rf_d"]
y_pos = np.arange(len(labels))

text_bbox = dict(facecolor="white", alpha=0.88, edgecolor="none", pad=1.5)

# ==========================================
# FIGURE 5a: DETONATION PRESSURE
# ==========================================
fig1, ax1 = plt.subplots(figsize=(8.5, 6.2))
bars1 = ax1.barh(
    y_pos, pressures, height=0.58, color=colors, edgecolor="black", linewidth=1.2
)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(labels, fontsize=11, fontweight="bold")
ax1.invert_yaxis()
ax1.set_xlabel("Direct K-J Detonation Pressure (GPa)", fontsize=12, fontweight="bold")
ax1.set_title("Theoretical Detonation Pressure Comparison", fontsize=14, fontweight="bold", pad=12)
ax1.set_xlim(0, 42)
ax1.grid(axis="x", linestyle="--", alpha=0.6)
ax1.axvline(
    rf_p_ref,
    color="darkgreen",
    linestyle=":",
    linewidth=2.0,
    zorder=4,
    label=f"RF Surrogate Prediction ({rf_p_ref:.2f} GPa)",
)
ax1.legend(loc="lower right", fontsize=10.5, framealpha=0.95, edgecolor="gray")

for bar in bars1:
    w = bar.get_width()
    ax1.text(
        w + 0.6,
        bar.get_y() + bar.get_height() / 2,
        f"{w:.2f} GPa",
        va="center",
        ha="left",
        fontsize=10.5,
        fontweight="bold",
        color="black",
        bbox=text_bbox,
        zorder=5,
    )

plt.tight_layout()
plt.savefig("Figure5a_Detonation_Pressure.png", dpi=600, bbox_inches="tight")
plt.show()

# ==========================================
# FIGURE 5b: DETONATION VELOCITY
# ==========================================
fig2, ax2 = plt.subplots(figsize=(8.5, 6.2))
bars2 = ax2.barh(
    y_pos, velocities, height=0.58, color=colors, edgecolor="black", linewidth=1.2
)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(labels, fontsize=11, fontweight="bold")
ax2.invert_yaxis()
ax2.set_xlabel("Direct K-J Detonation Velocity (m/s)", fontsize=12, fontweight="bold")
ax2.set_title("Theoretical Detonation Velocity Comparison", fontsize=14, fontweight="bold", pad=12)
ax2.set_xlim(0, 11200)
ax2.grid(axis="x", linestyle="--", alpha=0.6)
ax2.axvline(
    rf_d_ref,
    color="darkgreen",
    linestyle=":",
    linewidth=2.0,
    zorder=4,
    label=f"RF Surrogate Prediction ({rf_d_ref:.2f} m/s)",
)
ax2.legend(loc="lower right", fontsize=10.5, framealpha=0.95, edgecolor="gray")

for bar in bars2:
    w = bar.get_width()
    ax2.text(
        w + 150,
        bar.get_y() + bar.get_height() / 2,
        f"{w:.2f} m/s",
        va="center",
        ha="left",
        fontsize=10.5,
        fontweight="bold",
        color="black",
        bbox=text_bbox,
        zorder=5,
    )

plt.tight_layout()
plt.savefig("Figure5b_Detonation_Velocity.png", dpi=600, bbox_inches="tight")
plt.show()