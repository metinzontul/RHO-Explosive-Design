# RHO Framework: Surrogate-Assisted Design of Polymer Bonded Explosives

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the official Python implementation and data generation scripts for the **Random-forest Heuristic Optimizer (RHO)** framework. The RHO framework is an explainable digital surrogate workflow designed to address the topological optimization conflicts inherent in the machine-learning-assisted design of ternary RDX/TNT/HTPB polymer-bonded explosives (PBXs).

## Overview
Tree-based machine learning surrogates (like Random Forest) inherently construct non-differentiable, piecewise-constant response surfaces (flat plateaus and sharp step-edges). Conventional global optimizers (such as Dual Annealing and Basin-Hopping) rely on local gradient polishers (e.g., L-BFGS-B) which evaluate to zero on these plateaus, causing the algorithms to become "blind" and stagnate. 

This repository empirically demonstrates this **topological mismatch** using Kamlet-Jacobs (K-J) thermochemical equations as a high-speed analytical sandbox. By employing a strictly gradient-free **Differential Evolution (DE)** algorithm operating on 2 degrees of freedom, the RHO framework successfully resists plateau-induced stagnation to isolate the true theoretical optimal formulation.

## Repository Contents

The codebase is organized in the root directory and consists of the main optimization pipeline and visualization scripts:

*   **`PBX_RHO_optimizer.py`**: The core script. Generates the full-simplex Kamlet-Jacobs dataset, trains the Random Forest surrogate (with Ridge and MLP benchmarks), runs the 30-seed global optimization benchmark (DE vs. DA vs. BH), executes Wilcoxon statistical tests, and generates SHAP (Explainable AI) beeswarm plots.
*   **`plot_parity_RF_surrogate.py`**: Evaluates the surrogate model on the held-out test set and generates journal-quality parity plots (Figure 1).
*   **`plot_cv_boxplots.py`**: Runs a 10-seed, 5-fold cross-validation (50 iterations) to prove the statistical stability of the surrogate and plots the resulting boxplots (Figure 2).
*   **`plot_ternary_contour.py`**: Generates a high-resolution ternary heatmap of the detonation pressure response surface and marks the best discovered RHO candidate (Figure 3).
*   **`plot_thermochemical_barcharts.py`**: Dynamically calculates and plots the theoretical K-J performance of the optimal RHO candidate against the pure RDX and high-binder boundaries (Figure 4).

## Installation

It is recommended to use a virtual environment. Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Usage

You can run the scripts independently. To execute the main optimization benchmark and generate the SHAP plots, run:

```bash
python PBX_RHO_optimizer.py
```
*Note: Generating the SHAP plots might take a few moments depending on your CPU.*

To generate the specific figures for the paper, simply run the respective plotting scripts:
```bash
python plot_parity_RF_surrogate.py
python plot_cv_boxplots.py
python plot_ternary_contour.py
python plot_thermochemical_barcharts.py
```

## Key Findings
- **Surrogate Fidelity**: The Random Forest digital surrogate achieved a Mean Absolute Error (MAE) of 0.13 GPa (MAPE = 1.77%) for theoretical detonation pressure on the full-simplex dataset.
- **Topological Stagnation**: Dual Annealing and Basin-Hopping suffered from premature boundary stagnation due to zero-gradient plateaus.
- **RHO Superiority**: The gradient-free DE engine consistently navigated to the maximum plateau region (21.10 GPa ± 0.01), demonstrating statistical superiority ($p < 0.005$).
- **Optimal Candidate**: The framework isolated the optimal constrained formulation at **74.68% RDX, 10.31% TNT, and 15.01% HTPB**.

## Dependencies
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `matplotlib`
- `shap`
- `python-ternary`

## License
This project is licensed under the MIT License - see the LICENSE file for details.