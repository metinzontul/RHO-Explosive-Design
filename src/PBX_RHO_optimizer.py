# -*- coding: utf-8 -*-
"""
RHO Framework: Random-forest Heuristic Optimizer for PBX Thermochemical Design
Revised Implementation for MDPI Processes (Manuscript ID: processes-4701050)

Key Revisions Addressing Reviewer Comments:
1. Strict Consistency (Reviewer 4): Explicit `polish=False` in Differential Evolution
   and documented `max_depth=15` in Random Forest.
2. Matched Evaluation Budget & Runtime Tracking (Reviewer 3 & 4): Enforces an exact
   maximum objective evaluation budget (MAX_NFEV = 1000) across DE, Dual Annealing,
   and Basin-Hopping, recording mean nfev, wall-clock runtime, and Wilcoxon p-values.
3. Ablation on Local Search (Reviewer 3 & 4): Evaluates Dual Annealing with
   `no_local_search=True` and Basin-Hopping with derivative-free `Nelder-Mead`
   to isolate the effect of gradient-based L-BFGS-B polishing on RF plateaus.
4. Surrogate Topology Comparison: RF vs. MLP (Reviewer 3): Runs the 30-seed
   benchmark on the smooth MLP surrogate to contrast continuous vs. piecewise-constant
   optimization landscapes.
5. Local Constrained-Region Validation & Multi-Candidate K-J Verification (Reviewer 3):
   Evaluates surrogate error on 300 independent samples strictly inside the constrained
   domain and verifies the Top-5 DE candidates against direct K-J equations.

Author: Prof. Dr. Metin Zontul
Date: November 2026
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error,
)
from scipy.optimize import differential_evolution, dual_annealing, basinhopping
from scipy.stats import wilcoxon
import shap

warnings.filterwarnings("ignore")

# --- CONFIGURATION & CONSTANTS ---
N_SAMPLES = 1500
N_LOCAL_SAMPLES = 300
N_SEEDS = 30
MAX_NFEV = 1000  # Matched evaluation budget across all global optimizers
BOUNDS = [(0.10, 0.20), (0.15, 0.25)]  # Bounds for [TNT, HTPB]; RDX = 1 - TNT - HTPB


# ==========================================
# 1. PHYSICAL BASELINE: K-J CALCULATIONS
# ==========================================
def calculate_KJ(w_rdx, w_tnt, w_htpb):
    """
    Standardized Kistiakowsky-Wilson (K-W) & Kamlet-Jacobs (K-J) calculation
    on a 1 g formulation basis.
    """
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


# ==========================================
# 2. DATASET GENERATION (FULL SIMPLEX & LOCAL REGION)
# ==========================================
print("--- 1. Generating Thermochemical Datasets ---")
np.random.seed(42)
data = []

while len(data) < N_SAMPLES:
    w1, w2, w3 = np.random.dirichlet((1, 1, 1))
    P, D = calculate_KJ(w1, w2, w3)
    if P > 0.1:
        data.append([w1, w2, w3, P, D])

df = pd.DataFrame(data, columns=["RDX", "TNT", "HTPB", "P_GPa", "D_ms"])

# Independent local validation set strictly inside the constrained optimization region
np.random.seed(123)
local_data = []
for _ in range(N_LOCAL_SAMPLES):
    w_tnt = np.random.uniform(BOUNDS[0][0], BOUNDS[0][1])
    w_htpb = np.random.uniform(BOUNDS[1][0], BOUNDS[1][1])
    w_rdx = 1.0 - (w_tnt + w_htpb)
    P, D = calculate_KJ(w_rdx, w_tnt, w_htpb)
    local_data.append([w_rdx, w_tnt, w_htpb, P, D])

df_local = pd.DataFrame(local_data, columns=["RDX", "TNT", "HTPB", "P_GPa", "D_ms"])
print(f"Full-simplex dataset: {len(df)} samples | Local constrained validation set: {len(df_local)} samples")


# ==========================================
# 3. SURROGATE BENCHMARK (GLOBAL & LOCAL ACCURACY)
# ==========================================
print("\n--- 2. Surrogate Model Benchmark Tests (Global Held-Out vs. Local Constrained) ---")
X = df[["RDX", "TNT", "HTPB"]]
y_P = df["P_GPa"]
y_D = df["D_ms"]

X_train, X_test, y_P_train, y_P_test = train_test_split(
    X, y_P, test_size=0.2, random_state=42
)
_, _, y_D_train, y_D_test = train_test_split(X, y_D, test_size=0.2, random_state=42)

X_local = df_local[["RDX", "TNT", "HTPB"]]
y_P_local = df_local["P_GPa"]
y_D_local = df_local["D_ms"]

models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "MLP Neural Network": MLPRegressor(
        hidden_layer_sizes=(50, 50), max_iter=500, random_state=42
    ),
    "Random Forest (RHO)": RandomForestRegressor(
        n_estimators=100, max_depth=15, random_state=42
    ),
}

rf_P, mlp_P = None, None

for name, model in models.items():
    model.fit(X_train, y_P_train)
    # Global Test Set Evaluation
    preds_p = model.predict(X_test)
    r2_p = r2_score(y_P_test, preds_p)
    mape_p = mean_absolute_percentage_error(y_P_test, preds_p) * 100
    mae_p = mean_absolute_error(y_P_test, preds_p)
    rmse_p = np.sqrt(mean_squared_error(y_P_test, preds_p))

    # Local Constrained Domain Evaluation
    preds_loc = model.predict(X_local)
    r2_loc = r2_score(y_P_local, preds_loc)
    mape_loc = mean_absolute_percentage_error(y_P_local, preds_loc) * 100
    mae_loc = mean_absolute_error(y_P_local, preds_loc)
    rmse_loc = np.sqrt(mean_squared_error(y_P_local, preds_loc))

    print(
        f"{name:20s} | Global -> R2: {r2_p:.4f}, MAE: {mae_p:.3f} GPa, RMSE: {rmse_p:.3f} GPa, MAPE: {mape_p:.2f}%"
    )
    print(
        f"{'':20s} | Local  -> R2: {r2_loc:.4f}, MAE: {mae_loc:.3f} GPa, RMSE: {rmse_loc:.3f} GPa, MAPE: {mape_loc:.2f}%"
    )

    if name == "Random Forest (RHO)":
        rf_P = model
    elif name == "MLP Neural Network":
        mlp_P = model

rf_D = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42).fit(
    X_train, y_D_train
)
preds_d_loc = rf_D.predict(X_local)
print(
    f"{'RF Velocity (Local)':20s} | Local  -> R2: {r2_score(y_D_local, preds_d_loc):.4f}, "
    f"MAE: {mean_absolute_error(y_D_local, preds_d_loc):.2f} m/s, "
    f"RMSE: {np.sqrt(mean_squared_error(y_D_local, preds_d_loc)):.2f} m/s"
)


# ==========================================
# 4. BUDGET-MATCHED OPTIMIZATION BENCHMARK (RF vs. MLP)
# ==========================================
class BudgetedObjective:
    """
    Wraps a surrogate model to strictly enforce a maximum evaluation budget (max_nfev),
    track actual function evaluations, and record the best feasible coordinate found.
    """

    def __init__(self, surrogate, max_nfev=MAX_NFEV):
        self.surrogate = surrogate
        self.max_nfev = max_nfev
        self.nfev = 0
        self.best_val = -np.inf
        self.best_x = None

    def __call__(self, x):
        tnt, htpb = float(x[0]), float(x[1])
        # Enforce bounds strictly in case a local solver steps slightly outside
        tnt = np.clip(tnt, BOUNDS[0][0], BOUNDS[0][1])
        htpb = np.clip(htpb, BOUNDS[1][0], BOUNDS[1][1])
        rdx = 1.0 - (tnt + htpb)
        if rdx < 0 or rdx > 1:
            return 1e6

        if self.nfev >= self.max_nfev:
            # Return best observed value once budget is exhausted so solver terminates cleanly
            return -self.best_val if self.best_val > -np.inf else 1e6

        self.nfev += 1
        pred = float(self.surrogate.predict(np.array([[rdx, tnt, htpb]]))[0])
        if pred > self.best_val:
            self.best_val = pred
            self.best_x = (rdx, tnt, htpb)
        return -pred


def run_benchmark_on_surrogate(surrogate_model, surrogate_label):
    print(f"\n--- 3. Budget-Matched Benchmark on {surrogate_label} Surrogate (30 Runs, Max NFEV={MAX_NFEV}) ---")

    methods = [
        "DE (polish=False)",
        "Dual Annealing (L-BFGS-B)",
        "Dual Annealing (No Local)",
        "Basin-Hopping (L-BFGS-B)",
        "Basin-Hopping (Nelder-Mead)",
    ]
    results = {m: {"vals": [], "nfevs": [], "times": [], "xs": []} for m in methods}

    for i in range(N_SEEDS):
        # 1. RHO: Differential Evolution with explicit polish=False
        # popsize=10 with 2 variables -> 20 individuals/gen -> maxiter=49 gives ~1000 evals
        t0 = time.perf_counter()
        obj_de = BudgetedObjective(surrogate_model, MAX_NFEV)
        differential_evolution(
            obj_de,
            BOUNDS,
            strategy="best1bin",
            popsize=10,
            maxiter=49,
            polish=False,
            seed=i,
        )
        dt = time.perf_counter() - t0
        results["DE (polish=False)"]["vals"].append(obj_de.best_val)
        results["DE (polish=False)"]["nfevs"].append(obj_de.nfev)
        results["DE (polish=False)"]["times"].append(dt)
        results["DE (polish=False)"]["xs"].append(obj_de.best_x)

        # 2. Dual Annealing (Default L-BFGS-B local search)
        t0 = time.perf_counter()
        obj_da = BudgetedObjective(surrogate_model, MAX_NFEV)
        dual_annealing(
            obj_da, bounds=BOUNDS, maxfun=MAX_NFEV, no_local_search=False, seed=i
        )
        dt = time.perf_counter() - t0
        results["Dual Annealing (L-BFGS-B)"]["vals"].append(obj_da.best_val)
        results["Dual Annealing (L-BFGS-B)"]["nfevs"].append(obj_da.nfev)
        results["Dual Annealing (L-BFGS-B)"]["times"].append(dt)
        results["Dual Annealing (L-BFGS-B)"]["xs"].append(obj_da.best_x)

        # 3. Dual Annealing (Without local gradient search)
        t0 = time.perf_counter()
        obj_da_nl = BudgetedObjective(surrogate_model, MAX_NFEV)
        dual_annealing(
            obj_da_nl, bounds=BOUNDS, maxfun=MAX_NFEV, no_local_search=True, seed=i
        )
        dt = time.perf_counter() - t0
        results["Dual Annealing (No Local)"]["vals"].append(obj_da_nl.best_val)
        results["Dual Annealing (No Local)"]["nfevs"].append(obj_da_nl.nfev)
        results["Dual Annealing (No Local)"]["times"].append(dt)
        results["Dual Annealing (No Local)"]["xs"].append(obj_da_nl.best_x)

        # 4. Basin-Hopping (L-BFGS-B local polisher)
        t0 = time.perf_counter()
        obj_bh = BudgetedObjective(surrogate_model, MAX_NFEV)
        rng = np.random.RandomState(i)
        x0 = [rng.uniform(0.10, 0.20), rng.uniform(0.15, 0.25)]
        basinhopping(
            obj_bh,
            x0,
            minimizer_kwargs={"method": "L-BFGS-B", "bounds": BOUNDS},
            niter=40,
            seed=i,
        )
        dt = time.perf_counter() - t0
        results["Basin-Hopping (L-BFGS-B)"]["vals"].append(obj_bh.best_val)
        results["Basin-Hopping (L-BFGS-B)"]["nfevs"].append(obj_bh.nfev)
        results["Basin-Hopping (L-BFGS-B)"]["times"].append(dt)
        results["Basin-Hopping (L-BFGS-B)"]["xs"].append(obj_bh.best_x)

        # 5. Basin-Hopping (Derivative-free Nelder-Mead local polisher)
        t0 = time.perf_counter()
        obj_bh_nm = BudgetedObjective(surrogate_model, MAX_NFEV)
        basinhopping(
            obj_bh_nm,
            x0,
            minimizer_kwargs={"method": "Nelder-Mead", "bounds": BOUNDS},
            niter=25,
            seed=i,
        )
        dt = time.perf_counter() - t0
        results["Basin-Hopping (Nelder-Mead)"]["vals"].append(obj_bh_nm.best_val)
        results["Basin-Hopping (Nelder-Mead)"]["nfevs"].append(obj_bh_nm.nfev)
        results["Basin-Hopping (Nelder-Mead)"]["times"].append(dt)
        results["Basin-Hopping (Nelder-Mead)"]["xs"].append(obj_bh_nm.best_x)

    de_vals = results["DE (polish=False)"]["vals"]
    for m in methods:
        vals = results[m]["vals"]
        nfevs = results[m]["nfevs"]
        times = results[m]["times"]
        best_idx = int(np.argmax(vals))
        bx = results[m]["xs"][best_idx]
        if m == "DE (polish=False)":
            p_str = "Reference"
        else:
            if np.allclose(de_vals, vals):
                p_str = "p = 1.0000 (Tie)"
            else:
                _, p_val = wilcoxon(de_vals, vals)
                p_str = f"p = {p_val:.4e}"
        print(
            f"{m:28s} | Mean P: {np.mean(vals):.4f} ± {np.std(vals):.4f} GPa | "
            f"Best: ({bx[0]*100:.2f}/{bx[1]*100:.2f}/{bx[2]*100:.2f}%) [{vals[best_idx]:.2f} GPa] | "
            f"Mean NFEV: {np.mean(nfevs):.1f} | Time: {np.mean(times)*1000:.1f} ms | {p_str}"
        )

    return results


rf_bench = run_benchmark_on_surrogate(rf_P, "Random Forest (Piecewise-Constant)")
mlp_bench = run_benchmark_on_surrogate(mlp_P, "MLP Neural Network (Smooth Continuous)")


# ==========================================
# 5. MULTI-CANDIDATE DIRECT K-J VERIFICATION (TOP-5 CANDIDATES)
# ==========================================
print("\n--- 4. Direct K-J Verification of Top-5 Distinct DE Candidates on RF Surrogate ---")
de_runs = list(
    zip(
        rf_bench["DE (polish=False)"]["vals"],
        rf_bench["DE (polish=False)"]["xs"],
    )
)
# Sort descending by predicted pressure
de_runs.sort(key=lambda item: item[0], reverse=True)

distinct_candidates = []
for pred_p, coords in de_runs:
    # Check if coordinate is distinct (Euclidean distance > 1e-4)
    if all(np.linalg.norm(np.array(coords) - np.array(c[1])) > 1e-4 for c in distinct_candidates):
        distinct_candidates.append((pred_p, coords))
    if len(distinct_candidates) == 5:
        break

for rank, (pred_p, (w_rdx, w_tnt, w_htpb)) in enumerate(distinct_candidates, start=1):
    kj_p, kj_d = calculate_KJ(w_rdx, w_tnt, w_htpb)
    rf_d_pred = float(rf_D.predict(np.array([[w_rdx, w_tnt, w_htpb]]))[0])
    err_p = abs(pred_p - kj_p)
    print(
        f"Rank {rank}: RDX={w_rdx*100:.2f}%, TNT={w_tnt*100:.2f}%, HTPB={w_htpb*100:.2f}% | "
        f"RF_P={pred_p:.2f} GPa, Direct_KJ_P={kj_p:.2f} GPa (|Diff|={err_p:.2f} GPa) | "
        f"RF_D={rf_d_pred:.2f} m/s, Direct_KJ_D={kj_d:.2f} m/s"
    )


# ==========================================
# 6. EXPLAINABLE AI (TREESHAP)
# ==========================================
print("\n--- 5. Generating TreeSHAP Plots ---")
explainer_P = shap.TreeExplainer(rf_P)
shap_values_P = explainer_P.shap_values(X_test)
plt.figure(figsize=(8, 5))
shap.summary_plot(shap_values_P, X_test, show=False, plot_type="dot")
plt.title("TreeSHAP Summary for Detonation Pressure (P)", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("SHAP_Pressure_Robust.png", dpi=600, bbox_inches="tight")
plt.close()

explainer_D = shap.TreeExplainer(rf_D)
shap_values_D = explainer_D.shap_values(X_test)
plt.figure(figsize=(8, 5))
shap.summary_plot(shap_values_D, X_test, show=False, plot_type="dot")
plt.title("TreeSHAP Summary for Detonation Velocity (D)", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("SHAP_Velocity_Robust.png", dpi=600, bbox_inches="tight")
plt.close()
print("SHAP plots successfully saved.")