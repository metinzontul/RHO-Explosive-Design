# Random-forest Heuristic Optimizer (RHO): a known-optimum benchmark for RDX/TNT/HTPB formulation search

[![Journal](https://img.shields.io/badge/Journal-MDPI%20Processes-blue.svg)](https://www.mdpi.com/journal/processes)
[![Python](https://img.shields.io/badge/Python-3.10-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

Code, execution log and figure scripts accompanying the article:

> **Surrogate-Assisted Search for RDX/TNT/HTPB Formulations Using Thermochemical Simulations: A Known-Optimum Benchmark of Surrogate Topology, Thermochemical-Call Efficiency and Explanation Fidelity**
> Yılmaz Mert, Metin Zontul, Seda Karateke\*, Mehmet Akif Cifci — submitted to *Processes* (MDPI), 2026 (processes-4701050)

Version 1.1.0 (22 November 2026) accompanies the article. It replaces v1.0.0, whose thermochemical
product allocation did not conserve all elements; all data, models and results were regenerated.

All numbers in the manuscript are produced by `PBX_RHO_optimizer.py` and recorded in `Logs.txt`.
The log, CSV files and figures in this repository come from one execution on Windows with Python 3.10.20,
NumPy 2.0.1, SciPy 1.15.2, scikit-learn 1.7.2 and SHAP 0.48.0 (printed in Section 0 of `Logs.txt`).
Runtimes depend on the machine; every other value is deterministic for these library versions.

## What the main script does

| Log section | Content | Manuscript |
|---|---|---|
| 0 | Python / NumPy / SciPy / scikit-learn / SHAP versions; DE configuration | §2.3 |
| 1 | 1,545 Dirichlet draws → 1,500 accepted K–J compositions; 300-composition constrained-region set | §2.1–2.2 |
| 2 | Ridge, MLP, RF pressure (and RF velocity) errors, global held-out and constrained region | Table 1 |
| 3 | DE (`polish=False`, explicit in the call), Dual Annealing, Basin-Hopping on RF and MLP surfaces, 30 runs, cap 1,000 evaluations | Table 3 |
| 4 | Direct K–J recalculation at five distinct DE candidates sharing the maximum RF prediction (A–E) | Table 4 |
| 5 | TreeSHAP plots | Figs. 7–8 |
| 6 | Direct K–J arm (same solvers, seeds, bounds, wrapper, cap) and direct K–J recalculation of the candidate returned by **every** run of every arm | Table 5, Fig. 9 |
| 7 | RF-score order vs direct K–J order for all 30 RF-DE runs (ties included) | Table 7 |
| 8 | Search-only and end-to-end computational cost (shared data generation counted once) | Table 6 |
| 9 | Equal direct K–J call budget (25–1,545 calls): surrogate-assisted vs direct search | Table 8, Fig. 10 |
| 10 | SHAP attributions of RF and MLP vs SHAP of the known K–J model (default TreeSHAP and simplex-consistent) | Table 9, Fig. 11 |

Output files: `run_level_candidates.csv` (450 rows: 3 arms × 5 solver configurations × 30 runs),
`common_outcome_summary.csv`, `rf_de_ranking_30runs.csv`, `cost_summary.csv`,
`equal_budget_runs.csv`, `equal_budget_summary.csv`, `shap_comparison_summary.csv`, `shap_comparison_values.csv`.

## Main results (from `Logs.txt`)

* Pressure MAE, global / constrained region: RF 0.117 / 0.092 GPa; MLP 0.073 / 0.065 GPa.
* Mean RF-predicted pressure (30 runs): DE 21.0625 ± 0.0268 GPa (198.0 evaluations); Basin-Hopping L-BFGS-B 21.0774 ± 0.0008 GPa (123.0).
* Common outcome (direct K–J pressure at each returned candidate; region maximum 21.0368 GPa at 75/10/15):
  RF-DE 20.9660 ± 0.0505 GPa; RF Basin-Hopping L-BFGS-B 21.0331 ± 0.0103 GPa;
  MLP and direct K–J with Dual Annealing L-BFGS-B or Basin-Hopping: 21.0368 GPa in all 30 runs.
* RF over-predicts pressure at all 150 RF-selected candidates (mean +0.074 GPa); MLP under-predicts at all 150 (mean −0.135 GPa).
* 14 of 30 RF-DE runs share a bit-identical RF score (21.0791564426 GPa); their direct K–J pressures span 20.98–21.02 GPa.
* At equal budgets of direct K–J calls, MLP-assisted DE returned candidates 0.07–0.08 GPa below the regional maximum with 26–101 calls, whereas direct DE was 0.60, 0.35 and 0.15 GPa below it with 25, 50 and 100 calls; RF-assisted DE was worse than direct DE at every budget. Direct Basin-Hopping (L-BFGS-B) reached the maximum with 25 calls.
* Default TreeSHAP attributes almost no pressure to TNT (mean |SHAP| 0.07 GPa vs 1.11 GPa for the K–J model). With a simplex-consistent explanation, RF attributions reproduce the K–J attributions (r ≥ 0.999, RMSE ≤ 0.09 GPa).
* One DE search took 7.3 ms on direct K–J pressure, 20.5 ms on the MLP and 635.4 ms on the RF; the direct K–J arm had the lowest search-only and end-to-end cost for every solver configuration. For this analytic objective, surrogate-assisted search offers no wall-clock benefit; its benefit is measured in thermochemical calls (Section 9).

## Figure files (manuscript numbering)

| File | Figure |
|---|---|
| `Figure1_Workflow.pdf` | 1, workflow |
| `Figure2a_Detonation_Pressure_Parity.png`, `Figure2b_Detonation_Velocity_Parity.png` | 2a, 2b |
| `Figure3_CV_Boxplots.png` | 3 |
| `Figure4_Ternary_Plot.png` | 4 |
| `Figure5_Detonation_Pressure.png` | 5, direct K–J pressure of candidates A–E |
| `Figure6_Detonation_Velocity.png` | 6, direct K–J velocity of candidates A–E |
| `Figure7_SHAP_Pressure_Robust.png`, `Figure8_SHAP_Velocity_Robust.png` | 7, 8 (default TreeSHAP) |
| `Figure9_Common_Outcome_KJ.png` | 9 |
| `Figure10_Equal_Budget.png` | 10 |
| `Figure11_SHAP_vs_KJ.png` | 11 |

## Reproduce

```bash
pip install -r requirements.txt
cd src
python PBX_RHO_optimizer.py          # writes Logs.txt itself (console + file)
python plot_parity_RF_surrogate.py
python plot_cv_boxplots.py
python plot_ternary_contour.py
python plot_thermochemical_barcharts.py
python plot_common_outcome.py
python plot_equal_budget.py
python plot_shap_comparison.py
python make_latex_tables.py         # writes latex_tables.tex (Tables 5–9)
python check_de_polish.py           # optional: DE with polish=False vs polish=True (see below)
```

The plot scripts read `Logs.txt` and the CSV files from the `src` folder and write the PNG files there;
the copies in `Figures/` use the manuscript numbering.

## Differential Evolution configuration

Every `differential_evolution` call sets `polish=False` explicitly (Section 3 benchmark, direct K–J arm and
equal-budget comparison), and Section 0 of `Logs.txt` records this setting. `check_de_polish.py` repeats the
30 DE runs with the same models and seeds with and without polishing. Because the evaluation counter wraps the
objective, polishing evaluations would appear in NFEV:

| Surface | polish=False (Table 3) | polish=True |
|---|---|---|
| RF | 21.0625 ± 0.0268 GPa, 198.0 evaluations | 21.0625 ± 0.0268 GPa, 201.0 evaluations |
| MLP | 20.8451 ± 0.0402 GPa, 182.0 evaluations | 20.9045 ± 0.0000 GPa, 188.0 evaluations |

`run_level_candidates.csv` lists the candidate, objective value, evaluation count and run time of every run;
its means and standard deviations reproduce Table 3.

## Scope

All targets are theoretical Kamlet–Jacobs outputs; no experimental data are used. Direct K–J recalculation
uses the same model as the training targets and is not an independent physical validation.
