# -*- coding: utf-8 -*-
"""
RHO Framework: Random-forest Heuristic Optimizer for PBX Thermochemical Design
Known-optimum benchmark of surrogate-assisted search on RDX/TNT/HTPB compositions.

Sections (each writes a block of Logs.txt):
  0   Software versions and Differential Evolution (DE) configuration.
  1   Kamlet-Jacobs (K-J) targets with exact C-H-N-O conservation; 1,500 accepted
      Dirichlet compositions and a 300-composition constrained-region set.
  2   Ridge, MLP and RF surrogates: errors on the global held-out and constrained sets.
  3   DE (polish=False), Dual Annealing and Basin-Hopping on the RF and MLP pressure
      surfaces; 30 runs per configuration, 1,000-evaluation cap.
  4   Direct K-J recalculation of five distinct DE candidates on the maximum RF plateau.
  5   TreeSHAP summary plots.
  6   Direct K-J search arm (same solvers, seeds, bounds, wrapper and cap) and direct
      K-J pressure of the candidate returned by every run of every arm.
  7   RF-score order versus direct K-J order of the 30 RF-DE candidates.
  8   Search-only and end-to-end computational cost.
  9   Equal budgets of direct K-J calls: surrogate-assisted versus direct search.
  10  SHAP attributions of the surrogates versus those of the K-J model.

Outputs: Logs.txt, run_level_candidates.csv, common_outcome_summary.csv,
rf_de_ranking_30runs.csv, cost_summary.csv, equal_budget_runs.csv,
equal_budget_summary.csv, shap_comparison_summary.csv, shap_comparison_values.csv
and the TreeSHAP figures.

Author: Prof. Dr. Metin Zontul
Date: November 2026
"""

import os
import sys
import time
import platform
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
import scipy
import sklearn
from scipy.optimize import differential_evolution, dual_annealing, basinhopping
from scipy.stats import spearmanr, kendalltau
import shap

warnings.filterwarnings("ignore")

N_SAMPLES = 1500
N_LOCAL_SAMPLES = 300
N_SEEDS = 30
MAX_NFEV = 1000
BOUNDS = [(0.10, 0.20), (0.15, 0.25)]
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(OUT_DIR, "Logs.txt")


class _Tee:
    """Write everything printed to the console AND to Logs.txt (UTF-8)."""

    def __init__(self, console, log_file):
        self.console = console
        self.log_file = log_file

    def write(self, text):
        self.console.write(text)
        self.log_file.write(text)
        self.log_file.flush()

    def flush(self):
        self.console.flush()
        self.log_file.flush()


# Unwrap a tee left over from an interrupted run in the same Spyder kernel
while isinstance(sys.stdout, _Tee) or type(sys.stdout).__name__ == "_Tee":
    try:
        sys.stdout.log_file.close()
    except Exception:
        pass
    sys.stdout = sys.stdout.console
_CONSOLE = sys.stdout
_LOG_FILE = open(LOG_PATH, "w", encoding="utf-8")
sys.stdout = _Tee(_CONSOLE, _LOG_FILE)

print("--- 0. Software Environment ---")
print(
    f"Python {platform.python_version()} | NumPy {np.__version__} | SciPy {scipy.__version__} | "
    f"scikit-learn {sklearn.__version__} | SHAP {shap.__version__} | {platform.platform()}"
)
print("DE configuration: strategy=best1bin, popsize=10, maxiter=49, polish=False (explicit), seeds 0-29\n")


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


print("--- 1. Generating Thermochemical Datasets (Closed C-H-N-O Balance) ---")
t_data0 = time.perf_counter()
np.random.seed(42)
data = []
raw_draws = 0

while len(data) < N_SAMPLES:
    raw_draws += 1
    w1, w2, w3 = np.random.dirichlet((1, 1, 1))
    P, D = calculate_KJ(w1, w2, w3)
    if P > 0.1:
        data.append([w1, w2, w3, P, D])

df = pd.DataFrame(data, columns=["RDX", "TNT", "HTPB", "P_GPa", "D_ms"])
T_DATA_GEN = time.perf_counter() - t_data0
KJ_CALLS_DATA = raw_draws
print(f"Dirichlet draws: {raw_draws} total | {len(df)} accepted (P > 0.1 GPa) | {raw_draws - len(df)} rejected")

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

P_pure_ref, D_pure_ref = calculate_KJ(1.0, 0.0, 0.0)
P_bound_ref, D_bound_ref = calculate_KJ(0.55, 0.20, 0.25)
print(f"Reference Pure RDX (100/0/0%) -> P: {P_pure_ref:.2f} GPa, D: {D_pure_ref:.2f} m/s")
print(f"Reference High-HTPB Bound (55/20/25%) -> P: {P_bound_ref:.2f} GPa, D: {D_bound_ref:.2f} m/s")

print("\n--- 2. Surrogate Model Benchmark Tests (Global Held-Out vs. Local Constrained) ---")
X = df[["RDX", "TNT", "HTPB"]]
y_P = df["P_GPa"]
y_D = df["D_ms"]

X_train, X_test, y_P_train, y_P_test = train_test_split(X, y_P, test_size=0.2, random_state=42)
_, _, y_D_train, y_D_test = train_test_split(X, y_D, test_size=0.2, random_state=42)

X_local = df_local[["RDX", "TNT", "HTPB"]]
y_P_local = df_local["P_GPa"]
y_D_local = df_local["D_ms"]

models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "MLP Neural Network": MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42),
    "Random Forest (RHO)": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
}

rf_P, mlp_P = None, None
fit_times = {}

for name, model in models.items():
    t_fit0 = time.perf_counter()
    model.fit(X_train, y_P_train)
    fit_times[name] = time.perf_counter() - t_fit0
    preds_p = model.predict(X_test)
    r2_p = r2_score(y_P_test, preds_p)
    mape_p = mean_absolute_percentage_error(y_P_test, preds_p) * 100
    mae_p = mean_absolute_error(y_P_test, preds_p)
    rmse_p = np.sqrt(mean_squared_error(y_P_test, preds_p))

    preds_loc = model.predict(X_local)
    r2_loc = r2_score(y_P_local, preds_loc)
    mape_loc = mean_absolute_percentage_error(y_P_local, preds_loc) * 100
    mae_loc = mean_absolute_error(y_P_local, preds_loc)
    rmse_loc = np.sqrt(mean_squared_error(y_P_local, preds_loc))

    print(f"{name:20s} | Global -> R2: {r2_p:.4f}, MAE: {mae_p:.3f} GPa, RMSE: {rmse_p:.3f} GPa, MAPE: {mape_p:.2f}%")
    print(f"{'':20s} | Local  -> R2: {r2_loc:.4f}, MAE: {mae_loc:.3f} GPa, RMSE: {rmse_loc:.3f} GPa, MAPE: {mape_loc:.2f}%")

    if name == "Random Forest (RHO)":
        rf_P = model
    elif name == "MLP Neural Network":
        mlp_P = model

t_fit0 = time.perf_counter()
rf_D = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42).fit(X_train, y_D_train)
fit_times["RF Velocity"] = time.perf_counter() - t_fit0
preds_d_glob = rf_D.predict(X_test)
preds_d_loc = rf_D.predict(X_local)
print(
    f"{'RF Velocity (Global)':20s} | Global -> R2: {r2_score(y_D_test, preds_d_glob):.4f}, "
    f"MAE: {mean_absolute_error(y_D_test, preds_d_glob):.2f} m/s, "
    f"RMSE: {np.sqrt(mean_squared_error(y_D_test, preds_d_glob)):.2f} m/s, "
    f"MAPE: {mean_absolute_percentage_error(y_D_test, preds_d_glob)*100:.2f}%"
)
print(
    f"{'RF Velocity (Local)':20s} | Local  -> R2: {r2_score(y_D_local, preds_d_loc):.4f}, "
    f"MAE: {mean_absolute_error(y_D_local, preds_d_loc):.2f} m/s, "
    f"RMSE: {np.sqrt(mean_squared_error(y_D_local, preds_d_loc)):.2f} m/s, "
    f"MAPE: {mean_absolute_percentage_error(y_D_local, preds_d_loc)*100:.2f}%"
)


class DirectKJPressure:
    """Direct K-J pressure exposed through the same predict() interface as the surrogates."""

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([calculate_KJ(r[0], r[1], r[2])[0] for r in X])


class BudgetedObjective:
    def __init__(self, surrogate, max_nfev=MAX_NFEV):
        self.surrogate = surrogate
        self.max_nfev = max_nfev
        self.nfev = 0
        self.best_val = -np.inf
        self.best_x = None

    def __call__(self, x):
        tnt, htpb = float(x[0]), float(x[1])
        tnt = np.clip(tnt, BOUNDS[0][0], BOUNDS[0][1])
        htpb = np.clip(htpb, BOUNDS[1][0], BOUNDS[1][1])
        rdx = 1.0 - (tnt + htpb)
        if rdx < 0 or rdx > 1:
            return 1e6

        if self.nfev >= self.max_nfev:
            return -self.best_val if self.best_val > -np.inf else 1e6

        self.nfev += 1
        pred = float(self.surrogate.predict(np.array([[rdx, tnt, htpb]]))[0])
        if pred > self.best_val:
            self.best_val = pred
            self.best_x = (rdx, tnt, htpb)
        return -pred


def run_benchmark_on_surrogate(surrogate_model, surrogate_label, section="3", kind="Surrogate"):
    print(f"\n--- {section}. Evaluation-Cap Benchmark on {surrogate_label} {kind} (30 Runs, Max NFEV={MAX_NFEV}) ---")

    methods = [
        "DE (polish=False)",
        "Dual Annealing (L-BFGS-B)",
        "Dual Annealing (No Local)",
        "Basin-Hopping (L-BFGS-B)",
        "Basin-Hopping (Nelder-Mead)",
    ]
    results = {m: {"vals": [], "nfevs": [], "times": [], "xs": []} for m in methods}
    results["_label"] = surrogate_label

    for i in range(N_SEEDS):
        t0 = time.perf_counter()
        obj_de = BudgetedObjective(surrogate_model, MAX_NFEV)
        differential_evolution(obj_de, BOUNDS, strategy="best1bin", popsize=10, maxiter=49, polish=False, seed=i)
        dt = time.perf_counter() - t0
        results["DE (polish=False)"]["vals"].append(obj_de.best_val)
        results["DE (polish=False)"]["nfevs"].append(obj_de.nfev)
        results["DE (polish=False)"]["times"].append(dt)
        results["DE (polish=False)"]["xs"].append(obj_de.best_x)

        t0 = time.perf_counter()
        obj_da = BudgetedObjective(surrogate_model, MAX_NFEV)
        dual_annealing(obj_da, bounds=BOUNDS, maxfun=MAX_NFEV, no_local_search=False, seed=i)
        dt = time.perf_counter() - t0
        results["Dual Annealing (L-BFGS-B)"]["vals"].append(obj_da.best_val)
        results["Dual Annealing (L-BFGS-B)"]["nfevs"].append(obj_da.nfev)
        results["Dual Annealing (L-BFGS-B)"]["times"].append(dt)
        results["Dual Annealing (L-BFGS-B)"]["xs"].append(obj_da.best_x)

        t0 = time.perf_counter()
        obj_da_nl = BudgetedObjective(surrogate_model, MAX_NFEV)
        dual_annealing(obj_da_nl, bounds=BOUNDS, maxfun=MAX_NFEV, no_local_search=True, seed=i)
        dt = time.perf_counter() - t0
        results["Dual Annealing (No Local)"]["vals"].append(obj_da_nl.best_val)
        results["Dual Annealing (No Local)"]["nfevs"].append(obj_da_nl.nfev)
        results["Dual Annealing (No Local)"]["times"].append(dt)
        results["Dual Annealing (No Local)"]["xs"].append(obj_da_nl.best_x)

        t0 = time.perf_counter()
        obj_bh = BudgetedObjective(surrogate_model, MAX_NFEV)
        rng = np.random.RandomState(i)
        x0 = [rng.uniform(0.10, 0.20), rng.uniform(0.15, 0.25)]
        basinhopping(obj_bh, x0, minimizer_kwargs={"method": "L-BFGS-B", "bounds": BOUNDS}, niter=40, seed=i)
        dt = time.perf_counter() - t0
        results["Basin-Hopping (L-BFGS-B)"]["vals"].append(obj_bh.best_val)
        results["Basin-Hopping (L-BFGS-B)"]["nfevs"].append(obj_bh.nfev)
        results["Basin-Hopping (L-BFGS-B)"]["times"].append(dt)
        results["Basin-Hopping (L-BFGS-B)"]["xs"].append(obj_bh.best_x)

        t0 = time.perf_counter()
        obj_bh_nm = BudgetedObjective(surrogate_model, MAX_NFEV)
        basinhopping(obj_bh_nm, x0, minimizer_kwargs={"method": "Nelder-Mead", "bounds": BOUNDS}, niter=25, seed=i)
        dt = time.perf_counter() - t0
        results["Basin-Hopping (Nelder-Mead)"]["vals"].append(obj_bh_nm.best_val)
        results["Basin-Hopping (Nelder-Mead)"]["nfevs"].append(obj_bh_nm.nfev)
        results["Basin-Hopping (Nelder-Mead)"]["times"].append(dt)
        results["Basin-Hopping (Nelder-Mead)"]["xs"].append(obj_bh_nm.best_x)

    for m in methods:
        vals = results[m]["vals"]
        nfevs = results[m]["nfevs"]
        times = results[m]["times"]
        best_idx = int(np.argmax(vals))
        bx = results[m]["xs"][best_idx]
        print(
            f"{m:28s} | Mean P: {np.mean(vals):.4f} ± {np.std(vals):.4f} GPa | "
            f"Best: ({bx[0]*100:.2f}/{bx[1]*100:.2f}/{bx[2]*100:.2f}%) [{vals[best_idx]:.2f} GPa] | "
            f"Mean NFEV: {np.mean(nfevs):.1f} | Time: {np.mean(times)*1000:.1f} ms"
        )

    return results


rf_bench = run_benchmark_on_surrogate(rf_P, "Random Forest (Piecewise-Constant)")
mlp_bench = run_benchmark_on_surrogate(mlp_P, "MLP Neural Network (Smooth Continuous)")

print("\n--- 4. Direct K-J Verification of Top-5 Distinct DE Candidates on RF Surrogate ---")
de_runs = list(zip(rf_bench["DE (polish=False)"]["vals"], rf_bench["DE (polish=False)"]["xs"]))
de_runs.sort(key=lambda item: item[0], reverse=True)

distinct_candidates = []
for pred_p, coords in de_runs:
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

print("\n--- 5. Generating TreeSHAP Plots ---")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
explainer_P = shap.TreeExplainer(rf_P)
shap_values_P = explainer_P.shap_values(X_test)
plt.figure(figsize=(8, 5))
shap.summary_plot(shap_values_P, X_test, show=False, plot_type="dot")
plt.title("TreeSHAP Summary for Detonation Pressure (P)", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("Figure7_SHAP_Pressure_Robust.png", dpi=600, bbox_inches="tight")
plt.close()

explainer_D = shap.TreeExplainer(rf_D)
shap_values_D = explainer_D.shap_values(X_test)
plt.figure(figsize=(8, 5))
shap.summary_plot(shap_values_D, X_test, show=False, plot_type="dot")
plt.title("TreeSHAP Summary for Detonation Velocity (D)", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("Figure8_SHAP_Velocity_Robust.png", dpi=600, bbox_inches="tight")
plt.close()
print("SHAP plots successfully saved.")


# =====================================================================================
# 6. Common-outcome comparison: RF arm, MLP arm and direct K-J arm
#    Outcome for every arm = direct K-J pressure of the candidate returned by each run.
# =====================================================================================
kj_bench = run_benchmark_on_surrogate(
    DirectKJPressure(), "Direct K-J (No Surrogate)", section="6a", kind="Objective"
)

# Reference: maximum direct K-J pressure in the feasible box on a dense grid (501 x 501)
g_tnt = np.linspace(BOUNDS[0][0], BOUNDS[0][1], 501)
g_htpb = np.linspace(BOUNDS[1][0], BOUNDS[1][1], 501)
grid_best = (-np.inf, None)
for a_t in g_tnt:
    for a_h in g_htpb:
        pk = calculate_KJ(1.0 - a_t - a_h, a_t, a_h)[0]
        if pk > grid_best[0]:
            grid_best = (pk, (1.0 - a_t - a_h, a_t, a_h))
P_REF, X_REF = grid_best
print(
    f"\nReference (501x501 grid, direct K-J): max P = {P_REF:.4f} GPa at "
    f"RDX/TNT/HTPB = {X_REF[0]*100:.2f}/{X_REF[1]*100:.2f}/{X_REF[2]*100:.2f}%"
)

SOLVERS = [
    "DE (polish=False)",
    "Dual Annealing (L-BFGS-B)",
    "Dual Annealing (No Local)",
    "Basin-Hopping (L-BFGS-B)",
    "Basin-Hopping (Nelder-Mead)",
]
ARMS = [("RF", rf_bench), ("MLP", mlp_bench), ("Direct K-J", kj_bench)]

rows = []
for arm, bench in ARMS:
    for m in SOLVERS:
        r = bench[m]
        for seed in range(N_SEEDS):
            rdx, tnt, htpb = r["xs"][seed]
            t_rc = time.perf_counter()
            kj_p, kj_d = calculate_KJ(rdx, tnt, htpb)
            t_rc = time.perf_counter() - t_rc
            is_sur = arm != "Direct K-J"
            rows.append(
                {
                    "arm": arm,
                    "solver": m,
                    "seed": seed,
                    "RDX": rdx,
                    "TNT": tnt,
                    "HTPB": htpb,
                    "search_score_GPa": r["vals"][seed],
                    "KJ_P_GPa": kj_p,
                    "KJ_D_ms": kj_d,
                    "score_minus_KJ_GPa": (r["vals"][seed] - kj_p) if is_sur else 0.0,
                    "gap_to_ref_GPa": P_REF - kj_p,
                    "nfev": r["nfevs"][seed],
                    "search_time_ms": r["times"][seed] * 1000.0,
                    "direct_KJ_calls_search": 0 if is_sur else r["nfevs"][seed],
                    "direct_KJ_calls_recalc": 1 if is_sur else 0,
                    "recalc_time_ms": t_rc * 1000.0 if is_sur else 0.0,
                }
            )
runs = pd.DataFrame(rows)
runs.to_csv(os.path.join(OUT_DIR, "run_level_candidates.csv"), index=False, float_format="%.12g")

print("\n--- 6b. Direct K-J Pressure of the Candidate Returned by Each Run (30 runs per arm x solver) ---")
print("Search score = objective used during search (RF, MLP or direct K-J). Outcome = direct K-J P at returned candidate.")
print("SD uses ddof=0, as in Table 3.")
summ_rows = []
for arm, _ in ARMS:
    for m in SOLVERS:
        d = runs[(runs.arm == arm) & (runs.solver == m)]
        rec = {
            "arm": arm,
            "solver": m,
            "score_mean": d.search_score_GPa.mean(),
            "KJ_mean": d.KJ_P_GPa.mean(),
            "KJ_sd": d.KJ_P_GPa.std(ddof=0),
            "KJ_min": d.KJ_P_GPa.min(),
            "KJ_max": d.KJ_P_GPa.max(),
            "gap_mean": d.gap_to_ref_GPa.mean(),
            "abs_diff_mean": d.score_minus_KJ_GPa.abs().mean(),
            "nfev_mean": d.nfev.mean(),
            "time_ms_mean": d.search_time_ms.mean(),
            "kj_calls_per_run": (d.direct_KJ_calls_search + d.direct_KJ_calls_recalc).mean(),
        }
        summ_rows.append(rec)
        diff_str = f"|score-KJ| {rec['abs_diff_mean']:.4f}" if arm != "Direct K-J" else "|score-KJ| 0 (n/a)"
        print(
            f"{arm:10s} | {m:28s} | score {rec['score_mean']:.4f} | KJ_P {rec['KJ_mean']:.4f} ± {rec['KJ_sd']:.4f} "
            f"[{rec['KJ_min']:.4f}, {rec['KJ_max']:.4f}] GPa | gap {rec['gap_mean']:.4f} | {diff_str} | "
            f"NFEV {rec['nfev_mean']:.1f} | {rec['time_ms_mean']:.1f} ms | KJ calls/run {rec['kj_calls_per_run']:.1f}"
        )
summ = pd.DataFrame(summ_rows)
summ.to_csv(os.path.join(OUT_DIR, "common_outcome_summary.csv"), index=False, float_format="%.6f")

# =====================================================================================
# 7. Ranking: RF-score order vs direct K-J order (all 30 RF-DE runs)
# =====================================================================================
print("\n--- 7. RF-Score Ranking vs Direct K-J Ranking (RF surface, DE, all 30 runs) ---")
de_rf = runs[(runs.arm == "RF") & (runs.solver == "DE (polish=False)")].copy()
de_rf["rank_RF"] = de_rf.search_score_GPa.rank(ascending=False, method="min").astype(int)
de_rf["rank_KJ"] = de_rf.KJ_P_GPa.rank(ascending=False, method="min").astype(int)
rho_s, p_s = spearmanr(de_rf.search_score_GPa, de_rf.KJ_P_GPa)
tau_k, p_k = kendalltau(de_rf.search_score_GPa, de_rf.KJ_P_GPa)
n_unique_scores = de_rf.search_score_GPa.round(10).nunique()
print(f"Distinct RF scores among 30 runs: {n_unique_scores} (RF plateaus produce ties)")
n_tied_top = int((de_rf.search_score_GPa == de_rf.search_score_GPa.max()).sum())
tied = de_rf[de_rf.search_score_GPa == de_rf.search_score_GPa.max()]
print(
    f"Runs sharing the maximum RF score ({de_rf.search_score_GPa.max():.10f} GPa, bit-identical): {n_tied_top}/30 | "
    f"their direct K-J P: {tied.KJ_P_GPa.min():.4f}-{tied.KJ_P_GPa.max():.4f} GPa (mean {tied.KJ_P_GPa.mean():.4f})"
)
print("Note: the five distinct candidates of Section 4 all belong to this tied set; their order there is the")
print("      stable-sort (seed) order, not a difference in RF-predicted pressure.")
print(f"Spearman rho (RF score vs K-J P) = {rho_s:.3f} | Kendall tau-b = {tau_k:.3f}")
top_rf = de_rf.loc[de_rf.search_score_GPa.idxmax()]
top_kj = de_rf.loc[de_rf.KJ_P_GPa.idxmax()]
print(
    f"Run with highest RF score: seed {int(top_rf.seed)}, RF {top_rf.search_score_GPa:.4f}, K-J {top_rf.KJ_P_GPa:.4f} GPa "
    f"(K-J rank {int(top_rf.rank_KJ)}/30)"
)
print(
    f"Run with highest K-J P:    seed {int(top_kj.seed)}, RF {top_kj.search_score_GPa:.4f}, K-J {top_kj.KJ_P_GPa:.4f} GPa "
    f"(RF rank {int(top_kj.rank_RF)}/30)"
)
print(
    f"RF score - K-J P over 30 runs: mean {de_rf.score_minus_KJ_GPa.mean():.4f}, "
    f"min {de_rf.score_minus_KJ_GPa.min():.4f}, max {de_rf.score_minus_KJ_GPa.max():.4f} GPa "
    f"(positive = RF over-predicts at the selected point)"
)
for arm in ("RF", "MLP"):
    d = runs[runs.arm == arm]
    print(
        f"{arm} arm, all solvers (n={len(d)}): mean (score - K-J) = {d.score_minus_KJ_GPa.mean():.4f} GPa; "
        f"fraction over-predicted = {(d.score_minus_KJ_GPa > 0).mean():.3f}"
    )
de_rf.sort_values("rank_RF").to_csv(
    os.path.join(OUT_DIR, "rf_de_ranking_30runs.csv"), index=False, float_format="%.12g"
)

# =====================================================================================
# 8. Computational cost: search-only and end-to-end (shared data cost counted once)
# =====================================================================================
print("\n--- 8. Computational Cost (wall-clock, this machine) ---")
x_probe = np.array([[0.745, 0.104, 0.151]])
n_probe = 2000
t0 = time.perf_counter()
for _ in range(n_probe):
    calculate_KJ(*x_probe[0])
t_kj_call = (time.perf_counter() - t0) / n_probe
t0 = time.perf_counter()
for _ in range(200):
    rf_P.predict(x_probe)
t_rf_call = (time.perf_counter() - t0) / 200
t0 = time.perf_counter()
for _ in range(n_probe):
    mlp_P.predict(x_probe)
t_mlp_call = (time.perf_counter() - t0) / n_probe
print(
    f"Per-call time: direct K-J {t_kj_call*1e6:.1f} us | RF predict {t_rf_call*1e6:.1f} us | "
    f"MLP predict {t_mlp_call*1e6:.1f} us"
)
print(
    f"Shared setup (counted once for RF and MLP): data generation {T_DATA_GEN*1000:.1f} ms "
    f"({KJ_CALLS_DATA} direct K-J calls, {KJ_CALLS_DATA - N_SAMPLES} rejected draws)"
)
print(
    f"Training: RF pressure {fit_times['Random Forest (RHO)']*1000:.1f} ms | "
    f"MLP pressure {fit_times['MLP Neural Network']*1000:.1f} ms | "
    f"RF velocity {fit_times['RF Velocity']*1000:.1f} ms (reporting only)"
)
print("Evaluation-only sets (300 local + 300 held-out) are excluded from search cost.")
cost_rows = []
for m in SOLVERS:
    for arm, _ in ARMS:
        d = runs[(runs.arm == arm) & (runs.solver == m)]
        search_tot = d.search_time_ms.sum()
        recalc_tot = d.recalc_time_ms.sum()
        if arm == "RF":
            setup = T_DATA_GEN * 1000 + fit_times["Random Forest (RHO)"] * 1000
            kj_calls = KJ_CALLS_DATA + int(d.direct_KJ_calls_recalc.sum())
        elif arm == "MLP":
            setup = T_DATA_GEN * 1000 + fit_times["MLP Neural Network"] * 1000
            kj_calls = KJ_CALLS_DATA + int(d.direct_KJ_calls_recalc.sum())
        else:
            setup = 0.0
            kj_calls = int(d.direct_KJ_calls_search.sum())
        e2e = setup + search_tot + recalc_tot
        cost_rows.append(
            {
                "solver": m,
                "arm": arm,
                "search_ms_per_run": d.search_time_ms.mean(),
                "nfev_per_run": d.nfev.mean(),
                "setup_ms": setup,
                "search_ms_30runs": search_tot,
                "recalc_ms_30runs": recalc_tot,
                "end_to_end_ms_30runs": e2e,
                "direct_KJ_calls_total": kj_calls,
                "marginal_ms_per_new_search": d.search_time_ms.mean() + d.recalc_time_ms.mean(),
            }
        )
        print(
            f"{m:28s} | {arm:10s} | search {d.search_time_ms.mean():8.1f} ms/run (NFEV {d.nfev.mean():6.1f}) | "
            f"setup {setup:8.1f} ms | 30-run end-to-end {e2e:9.1f} ms | direct K-J calls {kj_calls:6d}"
        )
both_setup = T_DATA_GEN * 1000 + (fit_times["Random Forest (RHO)"] + fit_times["MLP Neural Network"]) * 1000
print(f"RF + MLP arms together: shared data generation counted once -> combined setup {both_setup:.1f} ms")
pd.DataFrame(cost_rows).to_csv(os.path.join(OUT_DIR, "cost_summary.csv"), index=False, float_format="%.3f")
print("Saved: run_level_candidates.csv, common_outcome_summary.csv, rf_de_ranking_30runs.csv, cost_summary.csv")

# =====================================================================================
# 9. Equal thermochemical-call budget: surrogate-assisted vs direct search
#    Budget B = number of direct K-J calls. Surrogate arm: B Dirichlet draws (rejected
#    draws included in B) -> fit RF / MLP -> search with the 1,000-evaluation cap ->
#    1 direct K-J recalculation. Direct arm: the same solver with its evaluation cap set
#    to B. Outcome for both: direct K-J pressure of the returned candidate.
# =====================================================================================
print("\n--- 9. Equal Direct K-J Call Budget: Surrogate-Assisted vs Direct Search ---")
BUDGETS = [25, 50, 100, 200, 400, 800, 1545]
N_DATA_REPS = 5          # independent training sets per budget (seeds 1000-1004)
N_SEARCH_SEEDS = 10      # search seeds per training set (surrogate arm)
LC_SOLVERS = ["DE (polish=False)", "Basin-Hopping (L-BFGS-B)"]


def run_solver(name, objective, seed):
    if name == "DE (polish=False)":
        differential_evolution(objective, BOUNDS, strategy="best1bin", popsize=10, maxiter=49,
                               polish=False, seed=seed)
    else:
        rng_l = np.random.RandomState(seed)
        x0_l = [rng_l.uniform(0.10, 0.20), rng_l.uniform(0.15, 0.25)]
        basinhopping(objective, x0_l, minimizer_kwargs={"method": "L-BFGS-B", "bounds": BOUNDS},
                     niter=40, seed=seed)
    return objective


def draw_training_set(budget, seed):
    rng_d = np.random.default_rng(seed)
    X_l, y_l = [], []
    for _ in range(budget):
        w = rng_d.dirichlet((1, 1, 1))
        p_kj, _ = calculate_KJ(*w)
        if p_kj > 0.1:
            X_l.append(w)
            y_l.append(p_kj)
    return np.array(X_l), np.array(y_l)


lc_rows = []
t_lc0 = time.perf_counter()
for B in BUDGETS:
    # Surrogate arm
    for rep in range(N_DATA_REPS):
        Xb, yb = draw_training_set(B, 1000 + rep)
        sur = {
            "RF": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42).fit(Xb, yb),
            "MLP": MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42).fit(Xb, yb),
        }
        for arm, model in sur.items():
            for m in LC_SOLVERS:
                for sd in range(N_SEARCH_SEEDS):
                    ob = run_solver(m, BudgetedObjective(model, MAX_NFEV), sd)
                    kj_p = calculate_KJ(*ob.best_x)[0]
                    lc_rows.append({"budget": B, "arm": arm, "solver": m, "data_rep": rep, "seed": sd,
                                    "n_train": len(yb), "KJ_calls": B + 1, "KJ_P_GPa": kj_p,
                                    "gap_GPa": P_REF - kj_p})
    # Direct arm: evaluation cap = B direct K-J calls
    for m in LC_SOLVERS:
        for sd in range(N_SEEDS):
            ob = run_solver(m, BudgetedObjective(DirectKJPressure(), B), sd)
            lc_rows.append({"budget": B, "arm": "Direct K-J", "solver": m, "data_rep": -1, "seed": sd,
                            "n_train": 0, "KJ_calls": ob.nfev, "KJ_P_GPa": ob.best_val,
                            "gap_GPa": P_REF - ob.best_val})
lc = pd.DataFrame(lc_rows)
lc.to_csv(os.path.join(OUT_DIR, "equal_budget_runs.csv"), index=False, float_format="%.12g")
print(f"Surrogate arm: {N_DATA_REPS} training sets x {N_SEARCH_SEEDS} search seeds per budget; "
      f"direct arm: {N_SEEDS} seeds per budget. Elapsed {time.perf_counter() - t_lc0:.1f} s")
print("Gap = max direct K-J P in region (%.4f GPa) - direct K-J P at returned candidate; mean +/- SD (ddof=0)" % P_REF)
lc_sum = (
    lc.groupby(["budget", "arm", "solver"])
    .agg(n=("gap_GPa", "size"), n_train=("n_train", "mean"), KJ_calls=("KJ_calls", "mean"),
         KJ_mean=("KJ_P_GPa", "mean"), gap_mean=("gap_GPa", "mean"),
         gap_sd=("gap_GPa", lambda v: np.std(v, ddof=0)), gap_max=("gap_GPa", "max"),
         frac_within_001=("gap_GPa", lambda v: float(np.mean(np.asarray(v) <= 0.01))))
    .reset_index()
)
lc_sum.to_csv(os.path.join(OUT_DIR, "equal_budget_summary.csv"), index=False, float_format="%.6f")
for _, r in lc_sum.iterrows():
    print(
        f"B={int(r.budget):5d} | {r.arm:10s} | {r.solver:26s} | n={int(r.n):2d} | train {r.n_train:7.1f} | "
        f"K-J calls {r.KJ_calls:7.1f} | K-J P {r.KJ_mean:.4f} | gap {r.gap_mean:.4f} +/- {r.gap_sd:.4f} "
        f"(max {r.gap_max:.4f}) | within 0.01 GPa: {r.frac_within_001:.2f}"
    )

# =====================================================================================
# 10. Faithfulness of SHAP attributions against the known K-J model (pressure)
#     The K-J function renormalizes its inputs to the simplex. Default TreeSHAP evaluates
#     the RF on coalitions whose fractions do not sum to one (outside the training
#     domain). A simplex-consistent explanation maps every coalition back onto the
#     simplex (x / sum(x)) before evaluation, i.e. the same convention as K-J itself.
#     All Kernel explanations use the same 100-composition background (seed 0); with
#     three features KernelSHAP enumerates all coalitions, so it is exact.
# =====================================================================================
print("\n--- 10. SHAP Attributions vs Attributions of the Known K-J Model (Pressure, 300 Held-Out Compositions) ---")
from scipy.stats import pearsonr

FEATS = ["RDX", "TNT", "HTPB"]
bg_shap = shap.sample(X_train, 100, random_state=0)
X_shap = X_test.values


def to_simplex(f):
    return lambda Z: f(np.asarray(Z, float) / np.asarray(Z, float).sum(axis=1, keepdims=True))


def kj_pressure_vec(Z):
    return np.array([calculate_KJ(*r)[0] for r in np.asarray(Z, float)])


shap_sets = {}
t0 = time.perf_counter()
shap_sets["K-J model (reference)"] = np.array(
    shap.KernelExplainer(kj_pressure_vec, bg_shap).shap_values(X_shap, silent=True))
t_kj_shap = time.perf_counter() - t0
t0 = time.perf_counter()
shap_sets["RF, TreeSHAP default (Fig. 7)"] = np.array(shap.TreeExplainer(rf_P).shap_values(X_test))
t_rf_tree = time.perf_counter() - t0
t0 = time.perf_counter()
shap_sets["RF, simplex-consistent"] = np.array(
    shap.KernelExplainer(to_simplex(rf_P.predict), bg_shap).shap_values(X_shap, silent=True))
t_rf_simplex = time.perf_counter() - t0
t0 = time.perf_counter()
shap_sets["MLP, simplex-consistent"] = np.array(
    shap.KernelExplainer(to_simplex(mlp_P.predict), bg_shap).shap_values(X_shap, silent=True))
t_mlp_simplex = time.perf_counter() - t0
times_shap = {"K-J model (reference)": t_kj_shap, "RF, TreeSHAP default (Fig. 7)": t_rf_tree,
              "RF, simplex-consistent": t_rf_simplex, "MLP, simplex-consistent": t_mlp_simplex}

ref = shap_sets["K-J model (reference)"]
shap_rows = []
for name, sv in shap_sets.items():
    row = {"explanation": name, "time_s": times_shap[name]}
    for j, f in enumerate(FEATS):
        row[f"meanabs_{f}"] = float(np.abs(sv[:, j]).mean())
        if name != "K-J model (reference)":
            row[f"rmse_{f}"] = float(np.sqrt(np.mean((sv[:, j] - ref[:, j]) ** 2)))
            row[f"r_{f}"] = float(pearsonr(sv[:, j], ref[:, j])[0])
    shap_rows.append(row)
    msg = " | ".join(f"{f}: {row[f'meanabs_{f}']:.3f}" for f in FEATS)
    if name != "K-J model (reference)":
        msg += " || RMSE vs K-J " + ", ".join(f"{row[f'rmse_{f}']:.3f}" for f in FEATS)
        msg += " || r vs K-J " + ", ".join(f"{row[f'r_{f}']:+.3f}" for f in FEATS)
    print(f"{name:30s} | mean|SHAP| (GPa) {msg} | {row['time_s']:.1f} s")
pd.DataFrame(shap_rows).to_csv(os.path.join(OUT_DIR, "shap_comparison_summary.csv"), index=False,
                               float_format="%.6f")
long_rows = []
for name, sv in shap_sets.items():
    for i in range(sv.shape[0]):
        for j, f in enumerate(FEATS):
            long_rows.append({"explanation": name, "sample": i, "feature": f, "value": X_shap[i, j],
                              "shap_GPa": sv[i, j]})
pd.DataFrame(long_rows).to_csv(os.path.join(OUT_DIR, "shap_comparison_values.csv"), index=False,
                               float_format="%.10g")
print("Background: 100 training compositions (shap.sample, random_state=0). Saved: shap_comparison_summary.csv, "
      "shap_comparison_values.csv")

# Restore the console and close Logs.txt
sys.stdout = _CONSOLE
_LOG_FILE.close()
print(f"Log written to {LOG_PATH}")
