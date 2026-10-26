# -*- coding: utf-8 -*-
"""
Ternary Contour Plot Generator for RHO Optimum Formulation
Updated with the verified Kamlet-Jacobs Physical Baseline.

Author: Prof. Dr. Metin Zontul
Date: October 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import ternary

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

scale = 100 

# --- Verified Kamlet-Jacobs Physical Baseline ---
def calculate_pressure(p):
    w_rdx, w_tnt, w_htpb = p[0]/scale, p[1]/scale, p[2]/scale
    
    # Epsilon to prevent division by zero during ternary grid edge cases
    w_rdx = max(w_rdx, 1e-6)
    w_tnt = max(w_tnt, 1e-6)
    w_htpb = max(w_htpb, 1e-6)
    
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
    return P

cmap = plt.get_cmap('jet')

fig, ax = plt.subplots(figsize=(10, 8))
tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale)

tax.heatmapf(calculate_pressure, boundary=True, style="triangular", cmap=cmap, colorbar=True)

tax.boundary(linewidth=2.0)
tax.gridlines(multiple=10, color="black", linestyle=":", linewidth=0.5)

# Updated Optimum Point based on RHO (DE) Convergence 
optimum_point = (74.68, 10.31, 15.01)
tax.scatter([optimum_point], marker='*', color='gold', edgecolor='black', s=400, zorder=10, label='Global Optimum (21.11 GPa)')

tax.left_axis_label("HTPB Fraction (%)", fontsize=14, offset=0.14)
tax.right_axis_label("TNT Fraction (%)", fontsize=14, offset=0.14)
tax.bottom_axis_label("RDX Fraction (%)", fontsize=14, offset=0.08)

tax.ticks(axis='lbr', linewidth=1, multiple=10, tick_formats="%d")

plt.title("RHO Detonation Pressure Surface (GPa)", fontsize=16, fontweight='bold', pad=30)
tax.legend(loc='upper right')

tax.clear_matplotlib_ticks()
tax._redraw_labels()
plt.tight_layout()
plt.savefig('Figure3_Ternary_Plot.png', dpi=600, bbox_inches='tight')
plt.show()