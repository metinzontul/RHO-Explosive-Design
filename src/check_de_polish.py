"""Effect of the polishing step on the Differential Evolution results (Table 3).

Re-creates the data and fitted RF/MLP pressure surrogates exactly as PBX_RHO_optimizer.py
does (Sections 0-2), then repeats the 30 DE runs with polish=False and with polish=True.
The evaluation counter (BudgetedObjective) wraps the objective, so evaluations made by a
polishing step are counted in NFEV and can change the best recorded value.

Expected output (the study uses polish=False):
  RF  polish=False: 21.0625 +/- 0.0268 GPa, NFEV 198.0 | polish=True: 21.0625 +/- 0.0268, NFEV 201.0
  MLP polish=False: 20.8451 +/- 0.0402 GPa, NFEV 182.0 | polish=True: 20.9045 +/- 0.0000, NFEV 188.0

Run from the src folder:  python check_de_polish.py
"""
import contextlib
import io
import os

import numpy as np
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "PBX_RHO_optimizer.py"), encoding="utf-8").read()
src = src[: src.index("rf_bench = run_benchmark_on_surrogate(rf_P")]  # stop before Section 3 runs

TMP = os.path.join(HERE, "_check_tmp")  # the setup writes its log here, not over src/Logs.txt
os.makedirs(TMP, exist_ok=True)
ns = {"__name__": "pbx_setup", "__file__": os.path.join(TMP, "PBX_RHO_optimizer.py")}
cwd = os.getcwd()
os.chdir(TMP)
try:
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, "PBX_RHO_optimizer.py", "exec"), ns)
finally:
    import sys
    sys.stdout = sys.__stdout__
    os.chdir(cwd)

for label, model in (("RF", ns["rf_P"]), ("MLP", ns["mlp_P"])):
    for polish in (False, True):
        vals, nfevs = [], []
        for seed in range(30):
            obj = ns["BudgetedObjective"](model, ns["MAX_NFEV"])
            differential_evolution(obj, ns["BOUNDS"], strategy="best1bin", popsize=10,
                                   maxiter=49, polish=polish, seed=seed)
            vals.append(obj.best_val)
            nfevs.append(obj.nfev)
        print(f"{label:3s} polish={str(polish):5s}: mean P {np.mean(vals):.4f} +/- {np.std(vals):.4f} GPa"
              f" | mean NFEV {np.mean(nfevs):.1f}")
