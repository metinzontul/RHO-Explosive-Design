"""
RHO Framework: Random-forest Heuristic Optimizer for PBX Thermochemical Design

Features Included (Revised per Claude's Critique):
- Unified K-J Thermochemical Equations (Accurate Baseline for Pure RDX ~35.25 GPa)
- Full Simplex Dirichlet Sampling (No artificial windowing during data generation)
- 2 Degrees of Freedom Optimization (TNT, HTPB) to strictly enforce RDX = 1 - TNT - HTPB
- MAE and RMSE physical metrics included
- Fair Global-vs-Global Benchmark: Differential Evolution vs Dual Annealing vs Basin-Hopping
- Multi-seed Robustness Analysis (30 independent runs) & Wilcoxon Testing

Author: Prof. Dr. Metin Zontul
Date: October 2026
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from scipy.optimize import differential_evolution, dual_annealing, basinhopping
from scipy.stats import wilcoxon
import shap

warnings.filterwarnings('ignore')

# --- CONFIGURATION & CONSTANTS ---
N_SAMPLES = 1500
N_SEEDS = 30  
# Bounds for [TNT, HTPB]. RDX will be derived as 1.0 - TNT - HTPB
BOUNDS = [(0.10, 0.20), (0.15, 0.25)]  

# ==========================================
# 1. PHYSICAL BASELINE: K-J CALCULATIONS
# ==========================================
def calculate_KJ(w_rdx, w_tnt, w_htpb):
    """
    Standardized Kamlet-Jacobs calculation applied universally across all files.
    """
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

# ==========================================
# 2. DATASET GENERATION (FULL SIMPLEX)
# ==========================================
print("--- 1. Generating Thermochemical Dataset (Full Simplex) ---")
np.random.seed(42)
data = []

# Claude's Fix: We sample from the entire Dirichlet distribution without artificial windowing
while len(data) < N_SAMPLES:
    w1, w2, w3 = np.random.dirichlet((1, 1, 1)) 
    P, D = calculate_KJ(w1, w2, w3)
    if P > 0.1:  
        data.append([w1, w2, w3, P, D])

df = pd.DataFrame(data, columns=['RDX', 'TNT', 'HTPB', 'P_GPa', 'D_ms'])

# ==========================================
# 3. BASELINE BENCHMARKS & SURROGATE TRAINING
# ==========================================
print("\n--- 2. Surrogate Model Benchmark Tests ---")
X = df[['RDX', 'TNT', 'HTPB']]
y_P = df['P_GPa']
y_D = df['D_ms']

X_train, X_test, y_P_train, y_P_test = train_test_split(X, y_P, test_size=0.2, random_state=42)
_, _, y_D_train, y_D_test = train_test_split(X, y_D, test_size=0.2, random_state=42)

models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "MLP Neural Network": MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42),
    "Random Forest (RHO)": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
}

rf_P, rf_D = None, None 

for name, model in models.items():
    model.fit(X_train, y_P_train)
    preds_p = model.predict(X_test)
    r2_p = r2_score(y_P_test, preds_p)
    mape_p = mean_absolute_percentage_error(y_P_test, preds_p) * 100
    mae_p = mean_absolute_error(y_P_test, preds_p)
    rmse_p = np.sqrt(mean_squared_error(y_P_test, preds_p))
    
    print(f"{name} -> P_Test R2: {r2_p:.4f} | MAPE: {mape_p:.2f}% | MAE: {mae_p:.2f} GPa | RMSE: {rmse_p:.2f} GPa")
    if name == "Random Forest (RHO)":
        rf_P = model
        
rf_D = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42).fit(X_train, y_D_train)

# ==========================================
# 4. FAIR GLOBAL-VS-GLOBAL OPTIMIZATION BENCHMARK
# ==========================================
print("\n--- 3. Multi-Seed Global Benchmark (Topological Stagnation Test - 30 Runs) ---")

def objective_rf(x):
    # 2 Degrees of Freedom enforcement. Prevents bounds violation.
    tnt, htpb = x[0], x[1]
    rdx = 1.0 - (tnt + htpb)
    if rdx < 0 or rdx > 1: return 1e6 
    return -rf_P.predict(np.array([[rdx, tnt, htpb]]))[0]

de_results, da_results, bh_results = [], [], []

for i in range(N_SEEDS):
    res_de = differential_evolution(objective_rf, BOUNDS, strategy='best1bin', popsize=30, seed=i)
    de_results.append(-res_de.fun)
    
    res_da = dual_annealing(objective_rf, bounds=BOUNDS, maxiter=30, seed=i)
    da_results.append(-res_da.fun)

    np.random.seed(i)
    x0 = [np.random.uniform(0.10, 0.20), np.random.uniform(0.15, 0.25)]
    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": BOUNDS}
    res_bh = basinhopping(objective_rf, x0, minimizer_kwargs=minimizer_kwargs, niter=10, seed=i)
    bh_results.append(-res_bh.fun)

print(f"RHO (DE) Mean Pressure:       {np.mean(de_results):.4f} GPa ± {np.std(de_results):.4f}")
print(f"Dual Annealing Mean Pressure: {np.mean(da_results):.4f} GPa ± {np.std(da_results):.4f}")
print(f"Basin-Hopping Mean Pressure:  {np.mean(bh_results):.4f} GPa ± {np.std(bh_results):.4f}")

# Wilcoxon Test
stat_da, p_val_da = wilcoxon(de_results, da_results)
stat_bh, p_val_bh = wilcoxon(de_results, bh_results)
print(f"Wilcoxon Test (DE vs Dual Annealing) p-value: {p_val_da:.4e}")
print(f"Wilcoxon Test (DE vs Basin-Hopping) p-value: {p_val_bh:.4e}")

# ==========================================
# 5. FINAL OPTIMAL COMPOSITIONS & XAI
# ==========================================
print("\n--- 4. Final Optimal Compositions (Best Discovered Formulation per Algorithm) ---")

best_idx_de = np.argmax(de_results)
best_res_de = differential_evolution(objective_rf, BOUNDS, strategy='best1bin', popsize=30, seed=best_idx_de)
opt_tnt_de, opt_htpb_de = best_res_de.x
print(f"RHO (DE) Optimal        -> RDX: { (1.0 - (opt_tnt_de + opt_htpb_de))*100:.2f}% | TNT: {opt_tnt_de*100:.2f}% | HTPB: {opt_htpb_de*100:.2f}% | P: {-best_res_de.fun:.2f} GPa")

print("\n--- Generating SHAP Plots ---")
# Explicitly linking SHAP background to Test Set to avoid simplex multicollinearity confusion
explainer_P = shap.TreeExplainer(rf_P)
shap_values_P = explainer_P.shap_values(X_test)
plt.figure(figsize=(8, 5))
shap.summary_plot(shap_values_P, X_test, show=False, plot_type="dot")
plt.title("SHAP Summary for Detonation Pressure (P)", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("SHAP_Pressure_Robust.png", dpi=600, bbox_inches='tight')
plt.close()

explainer_D = shap.TreeExplainer(rf_D)
shap_values_D = explainer_D.shap_values(X_test)
plt.figure(figsize=(8, 5))
shap.summary_plot(shap_values_D, X_test, show=False, plot_type="dot")
plt.title("SHAP Summary for Detonation Velocity (D)", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("SHAP_Velocity_Robust.png", dpi=600, bbox_inches='tight')
plt.close()
print("SHAP plots successfully saved.")