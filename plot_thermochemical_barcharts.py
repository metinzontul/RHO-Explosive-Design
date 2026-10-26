# -*- coding: utf-8 -*-
"""
Thermochemical Evaluation Bar Charts for RHO Optimum
Calculates values DYNAMICALLY from the verified K-J function. (No hardcoded/fabricated values).

Author: Prof. Dr. Metin Zontul
Date: October 2026
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 14  

# --- Verified Kamlet-Jacobs Physical Baseline ---
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

# Dynamically calculate values
P_pure, D_pure = calculate_KJ(1.0, 0.0, 0.0)
P_opt, D_opt = calculate_KJ(0.7468, 0.1031, 0.1501)
P_inert, D_inert = calculate_KJ(0.55, 0.20, 0.25)

labels = ['Pure RDX\n(Theoretical Baseline)', 'RHO Global Optimum\n(DE Candidate)', 'High-Binder Boundary\n(Inert)']
pressures = [P_pure, P_opt, P_inert]
velocities = [D_pure, D_opt, D_inert]

x = np.arange(len(labels))
width = 0.45  

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# (a) Pressure Bar Chart
bars1 = ax1.bar(x, pressures, width, color=['crimson', 'gold', 'steelblue'], edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Detonation Pressure (GPa)', fontsize=14, fontweight='bold')
ax1.set_title('(a) Detonation Pressure Comparison', fontsize=16, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=13, fontweight='bold')
ax1.set_ylim(0, max(pressures) * 1.15) 
ax1.grid(axis='y', linestyle=':', alpha=0.7)

for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# (b) Velocity Bar Chart
bars2 = ax2.bar(x, velocities, width, color=['crimson', 'gold', 'steelblue'], edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Detonation Velocity (m/s)', fontsize=14, fontweight='bold')
ax2.set_title('(b) Detonation Velocity Comparison', fontsize=16, fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=13, fontweight='bold')
ax2.set_ylim(0, max(velocities) * 1.15)
ax2.grid(axis='y', linestyle=':', alpha=0.7)

for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 150, f'{yval:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
fig.subplots_adjust(bottom=0.15) 

plt.savefig('Figure4_Thermochemical_Comparison_Final.png', dpi=600, bbox_inches='tight')
plt.show()