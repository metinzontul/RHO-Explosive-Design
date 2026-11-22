# -*- coding: utf-8 -*-
"""
Builds LaTeX Tables 5-9 of the article from the CSV files written by
PBX_RHO_optimizer.py. Output: latex_tables.tex
"""

import os
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
summ = pd.read_csv(os.path.join(here, "common_outcome_summary.csv"))
cost = pd.read_csv(os.path.join(here, "cost_summary.csv"))
rank = pd.read_csv(os.path.join(here, "rf_de_ranking_30runs.csv"))

SOLV = {
    "DE (polish=False)": ("DE", "none"),
    "Dual Annealing (L-BFGS-B)": ("DA", "L-BFGS-B"),
    "Dual Annealing (No Local)": ("DA", "none"),
    "Basin-Hopping (L-BFGS-B)": ("BH", "L-BFGS-B"),
    "Basin-Hopping (Nelder-Mead)": ("BH", "N--M"),
}
ABBR = ("DE, Differential Evolution (polish=False); DA, Dual Annealing; BH, Basin-Hopping; "
        "local search in parentheses (N--M, Nelder--Mead).")
ARM = {"RF": "RF", "MLP": "MLP", "Direct K-J": "Direct K--J"}

out = [r"\newcolumntype{C}{>{\centering\arraybackslash}X}  % as in the MDPI template; harmless if already defined", ""]
P_REF = summ.KJ_mean.iloc[0] + summ.gap_mean.iloc[0]
out.append(r"""\begin{table}[H]
\caption{Direct K--J pressure of the candidate returned by each of the 30 runs. Search score: mean value of the
objective used during search (RF, MLP or direct K--J pressure). K--J columns: direct K--J pressure at the returned
composition (common outcome); SD with $\mathrm{ddof}=0$, as in Table 3. Gap: mean distance from the maximum direct
K--J pressure in the search region (%.4f GPa, $501\times501$ grid). $|\Delta P|$: mean absolute difference between
search score and direct K--J pressure. All pressures in GPa. ABBR\label{tab5}}
\begin{adjustwidth}{-\extralength}{0cm}
\small
\begin{tabularx}{\fulllength}{llCCCCCCCC}
\toprule
\textbf{Objective} & \textbf{Solver} & \textbf{Search score} & \textbf{K--J mean} & \textbf{K--J SD} & \textbf{K--J min} & \textbf{K--J max} & \textbf{Gap} & \textbf{$|\Delta P|$} & \textbf{NFEV}\\
\midrule""".replace("%.4f", f"{P_REF:.4f}").replace("ABBR", ABBR))
for arm in ["RF", "MLP", "Direct K-J"]:
    first = True
    for _, r in summ[summ.arm == arm].iterrows():
        sv, loc = SOLV[r.solver]
        dp = f"{r.abs_diff_mean:.4f}" if arm != "Direct K-J" else "---"
        out.append(
            f"{ARM[arm] if first else ''} & {sv} ({loc}) & {r.score_mean:.4f} & {r.KJ_mean:.4f} & {r.KJ_sd:.4f} & "
            f"{r.KJ_min:.4f} & {r.KJ_max:.4f} & {r.gap_mean:.4f} & {dp} & {r.nfev_mean:.1f}\\\\"
        )
        first = False
    if arm != "Direct K-J":
        out.append(r"\midrule")
out.append(r"""\bottomrule
\end{tabularx}
\end{adjustwidth}
\end{table}
""")

out.append(r"""\begin{table}[H]
\caption{Computational cost on a common machine. Search-only cost is the mean wall-clock time and realized
evaluations per run. End-to-end cost for 30 runs adds data generation (1,545 direct K--J calls, shared by the RF
and MLP arms and counted once per arm here), model training and recalculation of the 30 returned candidates.
Validation sets used only for error assessment are excluded. Times in ms on one machine. ABBR\label{tab6}}
\begin{adjustwidth}{-\extralength}{0cm}
\small
\begin{tabularx}{\fulllength}{llCCCCC}
\toprule
\textbf{Solver} & \textbf{Objective} & \textbf{Search per run} & \textbf{NFEV per run} & \textbf{Setup} & \textbf{End-to-end, 30 runs} & \textbf{Direct K--J calls}\\
\midrule""".replace("ABBR", ABBR))
for s_key in SOLV:
    for j, arm in enumerate(["RF", "MLP", "Direct K-J"]):
        r = cost[(cost.solver == s_key) & (cost.arm == arm)].iloc[0]
        sv, loc = SOLV[s_key]
        out.append(
            f"{sv + ' (' + loc + ')' if j == 0 else ''} & {ARM[arm]} & {r.search_ms_per_run:.1f} & {r.nfev_per_run:.1f} & {r.setup_ms:.1f} & "
            f"{r.end_to_end_ms_30runs:.1f} & {int(r.direct_KJ_calls_total)}\\\\"
        )
    out.append(r"\midrule")
out[-1] = r"\bottomrule"
out.append(r"""\end{tabularx}
\end{adjustwidth}
\end{table}
""")

top = rank[rank.rank_RF == 1].sort_values("seed")
out.append(r"""\begin{table}[H]
\caption{RF-score order and direct K--J order of the candidates returned by the 30 DE runs on the RF surface
that share the maximum RF prediction (NTIED of 30 runs; identical unrounded RF value). Ranks are among all 30 runs;
tied RF scores share the minimum rank. All 30 runs are listed in rf\_de\_ranking\_30runs.csv.\label{tab7}}
\begin{tabularx}{\textwidth}{CCCCCCCC}
\toprule
\textbf{Seed} & \textbf{RDX \%} & \textbf{TNT \%} & \textbf{HTPB \%} & \textbf{RF $P$ (GPa)} & \textbf{K--J $P$ (GPa)} & \textbf{RF rank} & \textbf{K--J rank}\\
\midrule""".replace("NTIED", str(len(top))))
for _, r in top.iterrows():
    out.append(
        f"{int(r.seed)} & {r.RDX*100:.2f} & {r.TNT*100:.2f} & {r.HTPB*100:.2f} & {r.search_score_GPa:.4f} & "
        f"{r.KJ_P_GPa:.4f} & {int(r.rank_RF)} & {int(r.rank_KJ)}\\\\"
    )
out.append(r"""\bottomrule
\end{tabularx}
\end{table}
""")

# Table 8: equal direct K-J call budget (Section 9 of the log)
eb_path = os.path.join(here, "equal_budget_summary.csv")
if os.path.exists(eb_path):
    eb = pd.read_csv(eb_path)
    out.append(r"""\begin{table}[H]
\caption{Gap between the regional maximum of direct K--J pressure (%.4f GPa) and the direct K--J pressure of the
returned candidate, at equal budgets of direct K--J calls. Surrogate arms: $B$ Dirichlet draws (rejected draws
included) used to train RF or MLP, search with the 1,000-evaluation cap and one recalculation ($B+1$ calls; five
training sets $\times$ ten search seeds). Direct arm: the same solver with its evaluation cap set to $B$ (30 seeds);
DE and BH terminated before the cap for $B \geq 200$ (mean 170 and 218 calls). Mean (SD) in GPa. DE, Differential
Evolution (polish=False); BH, Basin-Hopping with L-BFGS-B.\label{tab8}}
\begin{adjustwidth}{-\extralength}{0cm}
\small
\begin{tabularx}{\fulllength}{cCCCCCC}
\toprule
 & \multicolumn{3}{c}{\textbf{DE}} & \multicolumn{3}{c}{\textbf{BH (L-BFGS-B)}}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
\textbf{Budget $B$} & \textbf{RF} & \textbf{MLP} & \textbf{Direct K--J} & \textbf{RF} & \textbf{MLP} & \textbf{Direct K--J}\\
\midrule""".replace("%.4f", f"{P_REF:.4f}"))
    for B in sorted(eb.budget.unique()):
        cells = []
        for solver in ["DE (polish=False)", "Basin-Hopping (L-BFGS-B)"]:
            for arm in ["RF", "MLP", "Direct K-J"]:
                r = eb[(eb.budget == B) & (eb.solver == solver) & (eb.arm == arm)].iloc[0]
                cells.append(f"{r.gap_mean:.3f} ({r.gap_sd:.3f})")
        out.append(f"{int(B)} & " + " & ".join(cells) + r"\\")
    out.append(r"""\bottomrule
\end{tabularx}
\end{adjustwidth}
\end{table}
""")

# Table 9: SHAP attributions vs attributions of the known K-J model (Section 10 of the log)
sh_path = os.path.join(here, "shap_comparison_summary.csv")
if os.path.exists(sh_path):
    sh = pd.read_csv(sh_path)
    out.append(r"""\begin{table}[H]
\caption{SHAP attributions for pressure on the 300 held-out compositions compared with the attributions of the
K--J model that generated the targets. Default TreeSHAP evaluates the RF on coalitions whose mass fractions do not
sum to one; the simplex-consistent explanation rescales every coalition to the simplex before evaluation, as the
K--J implementation does. Kernel explanations use the same 100-composition background and enumerate all coalitions.
Mean $|\mathrm{SHAP}|$ and RMSE in GPa; $r$, Pearson correlation with the K--J attributions.\label{tab9}}
\begin{adjustwidth}{-\extralength}{0cm}
\small
\begin{tabularx}{\fulllength}{lCCCCCCCCC}
\toprule
 & \multicolumn{3}{c}{\textbf{Mean $|\mathrm{SHAP}|$}} & \multicolumn{3}{c}{\textbf{RMSE vs K--J}} & \multicolumn{3}{c}{\textbf{$r$ vs K--J}}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
\textbf{Explanation} & \textbf{RDX} & \textbf{TNT} & \textbf{HTPB} & \textbf{RDX} & \textbf{TNT} & \textbf{HTPB} & \textbf{RDX} & \textbf{TNT} & \textbf{HTPB}\\
\midrule""")
    names = {"K-J model (reference)": "K--J model (reference)",
             "RF, TreeSHAP default (Fig. 7)": "RF, default TreeSHAP",
             "RF, simplex-consistent": "RF, simplex-consistent",
             "MLP, simplex-consistent": "MLP, simplex-consistent"}
    for _, r in sh.iterrows():
        cells = [f"{r[f'meanabs_{f}']:.3f}" for f in ["RDX", "TNT", "HTPB"]]
        if r.explanation == "K-J model (reference)":
            cells += ["---"] * 6
        else:
            cells += [f"{r[f'rmse_{f}']:.3f}" for f in ["RDX", "TNT", "HTPB"]]
            cells += [f"{r[f'r_{f}']:+.3f}".replace("-", "$-$") for f in ["RDX", "TNT", "HTPB"]]
        out.append(names[r.explanation] + " & " + " & ".join(cells) + r"\\")
    out.append(r"""\bottomrule
\end{tabularx}
\end{adjustwidth}
\end{table}
""")

with open(os.path.join(here, "latex_tables.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("Saved latex_tables.tex")
