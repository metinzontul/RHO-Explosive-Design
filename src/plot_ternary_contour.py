# -*- coding: utf-8 -*-
"""
Ternary pressure plot for the RF-selected formulation (Figure 4).
Direct K-J pressure surface (closed C-H-N-O balance) with the constrained search
region, the DE candidate read from Logs.txt and an inset of the TNT-HTPB corner.

Authors: Prof. Dr. Metin Zontul & Assoc. Prof. Dr. Seda Karateke
Date: November 2026
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import ternary

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["font.size"] = 13

scale = 100


def parse_rank1_from_logs(log_path="Logs.txt"):
    """Parse the Rank 1 DE candidate coordinates from Logs.txt."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, log_path)
    if not os.path.exists(full_path):
        full_path = log_path
    if not os.path.exists(full_path):
        return None
    with open(full_path, "r", encoding="utf-8", errors="ignore") as file:
        content = file.read()

    match = re.search(
        r"Rank 1:\s*RDX=([\d.]+)%,\s*TNT=([\d.]+)%,\s*HTPB=([\d.]+)%",
        content,
    )
    if not match:
        raise ValueError(f"Could not find 'Rank 1' candidate coordinates in {log_path}.")
    return tuple(float(match.group(i)) for i in (1, 2, 3))


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


def calculate_pressure(p):
    w_rdx = max(p[0] / scale, 1e-6)
    w_tnt = max(p[1] / scale, 1e-6)
    w_htpb = max(p[2] / scale, 1e-6)
    P, _ = calculate_KJ(w_rdx, w_tnt, w_htpb)
    return P


selected_point = parse_rank1_from_logs("Logs.txt")

if selected_point is None:
    # Fallback coordinates from the closed-balance execution log
    selected_point = (74.55, 10.44, 15.01)
    print("Logs.txt not found; using embedded closed-balance Rank 1 coordinates.")
else:
    print("Rank 1 coordinates read from Logs.txt.")

print(
    f"Rank 1 candidate: "
    f"RDX={selected_point[0]}%, TNT={selected_point[1]}%, HTPB={selected_point[2]}%"
)

if not np.isclose(sum(selected_point), 100.0, atol=0.02):
    raise ValueError("Logged mass fractions do not sum to 100% (within rounding).")

assert 10.0 <= selected_point[1] <= 20.0, "TNT out of bounds!"
assert 15.0 <= selected_point[2] <= 25.0, "HTPB out of bounds!"

cmap = LinearSegmentedColormap.from_list(
    "pastel_pressure",
    ["#637FA9", "#97BED2", "#BFDCCE", "#E6DFB8", "#ECC3A7", "#D58F96"],
    N=256,
)
ink = "#26364B"

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor("#FAFBFC")
ax.set_facecolor("#FAFBFC")

tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale)
tax.heatmapf(
    calculate_pressure,
    boundary=True,
    style="triangular",
    cmap=cmap,
    colorbar=True,
    cb_kwargs={"label": "Direct K–J Detonation Pressure (GPa)"},
)
tax.boundary(linewidth=2.0, axes_colors={"l": ink, "r": ink, "b": ink})
tax.gridlines(multiple=10, color=ink, linestyle=":", linewidth=0.5, alpha=0.48)

p1 = (75.0, 10.0, 15.0)
p2 = (65.0, 20.0, 15.0)
p3 = (55.0, 20.0, 25.0)
p4 = (65.0, 10.0, 25.0)

tax.plot([p1, p2, p3, p4, p1], linewidth=3.8, color=ink, linestyle="-", zorder=8)
tax.plot([p1, p2, p3, p4, p1], linewidth=2.2, color="#FFFFFF", linestyle="--", zorder=9)

candidate_xy = ternary.helpers.project_point(selected_point)
ax.scatter(
    candidate_xy[0],
    candidate_xy[1],
    marker="o",
    color="#A73F59",
    edgecolor="white",
    linewidth=1.2,
    s=56,
    zorder=11,
)

# High-contrast legend handles so dashed boundary is clearly visible on light background
legend_handles = [
    Line2D([0], [0], color=ink, lw=2.2, ls="--", label="Constrained Search Region"),
    Line2D(
        [0], [0], marker="o", color="none", markerfacecolor="#A73F59",
        markeredgecolor="white", markeredgewidth=1.2, markersize=8.5,
        label="RF-selected candidate (DE)",
    ),
]
ax.legend(
    handles=legend_handles,
    loc="upper right",
    framealpha=0.96,
    fontsize=11.5,
    facecolor="#F4F7F8",
    edgecolor="#8FA4B4",
)

# Local view of the reported TNT and HTPB coordinates (adaptive limits)
tnt_val, htpb_val = selected_point[1], selected_point[2]
x_min = 9.96
x_max = max(10.60, round(tnt_val + 0.16, 2))
y_min = 14.995
y_max = max(15.08, round(htpb_val + 0.07, 2))

zoom_ax = ax.inset_axes([0.025, 0.705, 0.265, 0.195], zorder=20)
zoom_ax.set_facecolor("#FFFFFF")
zoom_ax.add_patch(
    Rectangle((10.0, 15.0), x_max - 10.0, y_max - 15.0, facecolor="#DDEDE6", edgecolor="none", zorder=0)
)
zoom_ax.axvline(10.0, color=ink, ls="--", lw=1.0, alpha=0.75)
zoom_ax.axhline(15.0, color=ink, ls="--", lw=1.0, alpha=0.75)
zoom_ax.scatter(tnt_val, htpb_val, s=55, facecolor="#A73F59", edgecolor="white", linewidth=1.0, zorder=4)
zoom_ax.annotate(
    f"({tnt_val:.2f}, {htpb_val:.2f})",
    (tnt_val, htpb_val),
    xytext=(0, 8),
    textcoords="offset points",
    ha="center",
    va="bottom",
    fontsize=8,
    color=ink,
)
zoom_ax.set_xlim(x_min, x_max)
zoom_ax.set_ylim(y_min, y_max)
zoom_ax.tick_params(labelsize=8, colors=ink, length=2)
zoom_ax.set_xlabel("TNT (%)", fontsize=9, color=ink, labelpad=1)
zoom_ax.set_ylabel("HTPB (%)", fontsize=9, color=ink, labelpad=1)
zoom_ax.set_title("Candidate near lower HTPB bound", fontsize=9, color=ink, pad=4)
for spine in zoom_ax.spines.values():
    spine.set_color("#8FA4B4")

tax.left_axis_label("HTPB Fraction (%)", fontsize=14, offset=0.14)
tax.right_axis_label("TNT Fraction (%)", fontsize=14, offset=0.14)
tax.bottom_axis_label("RDX Fraction (%)", fontsize=14, offset=0.08)
tax.ticks(axis="lbr", linewidth=1, multiple=10, tick_formats="%d")

plt.title("Direct K–J Pressure Surface & RF-Selected Candidate", fontsize=15, fontweight="bold", pad=30)
tax.clear_matplotlib_ticks()
tax._redraw_labels()
plt.tight_layout()
plt.savefig("Figure4_Ternary_Plot.png", dpi=600, bbox_inches="tight")
plt.show()
