# gm2_analysis

GM2 gangliosidosis (Tay-Sachs / Sandhoff) mechanistic-model sensitivity
analysis supporting the PLOS Computational Biology submission. This
directory contains the model itself, every driver script, and every
output file behind the paper's core Tables 2, 3, 5, 6 and the
convergence / saturation / Shapley figures, plus the follow-up
acoustic-mechanism robustness check for the FUS (focused ultrasound)
arm.

This is a separate subproject from `task_outputs/` (percentile-CI,
verification, and external-benchmark outputs already published here).
The two subprojects share no code or outputs; keep them in their own
top-level directories.

## Contents

- `gm2_model_v2.py` — the 17-state GM2 mechanistic model. This is the
  exact, unmodified file used to produce every result in the paper
  (SHA-256: `6693427fa909a0a95093cf6797890c6c4004912fd2a54b7c9e9966e77fa8a4a6`).

- `drivers/` — every driver script that ran a Sobol, factorial/Shapley,
  convergence, or saturation analysis against `gm2_model_v2.py`
  (`task1_full14.py`, `task6_cross_disease.py`, `factorial_arms_shapley.py`,
  `task5_paired_contrasts.py`, `task2_convergence_grid.py`,
  `task2c_convergence_grid_second_order_true.py`,
  `task3_saturation_sweep.py`), plus the modules they import
  (`sobol_driver_v2.py`, `sobol_driver_lsd.py`, `lsd_disease_params.py`,
  `gm2_solver.py`, `parallel_eval.py`) and the two acoustic-robustness
  drivers (`derive_fus_gain_mapping.py`, `run_acoustic_fus_analysis.py`).

- `sobol_outputs/` — the 14-parameter full-model Sobol results
  (`gm2_full14_*.json`, three prior-width scenarios), the 10-disease
  cross-disease Sobol results (`sobol_lsd_*.json`, three variants each),
  the convergence grids, and the saturation sweep.

- `factorial_shapley_outputs/` — the 8-arm factorial design and exact
  Shapley decomposition (`factorial_arms_shapley*.json`; the unsuffixed
  file is the day-365 primary endpoint, `_day45`/`_day90`/`_day180` are
  the earlier timepoints), `shapley_with_ci.json`, and `paired_contrasts.json`.

- `provenance/` — parameter provenance, equal-width coefficient-of-variation
  check, FUS protocol parameters, and the software-version manifest.

- `acoustic_robustness_check/` — the 2026-09-15 follow-up rerun verifying
  the paper's FUS conclusion is robust to the choice of acoustic
  MI-to-BBB-opening mapping. `gm2_model_v2_acoustic_fus.py` is a variant
  of the core model with a derived `fus_entry_gain_scale` substituted in;
  `fus_gain_mapping.json` / `fus_gain_mapping_derivation.md` document how
  the two bounding values (k=1.0 conservative, k=2.0 permissive) were
  derived; the `*_acoustic_k1*` / `*_acoustic_k2*` JSON files are the
  factorial/Shapley/paired-contrast results rerun at each bound.

- `figure_scripts/` — the scripts that produced the manuscript figures
  (`final/`), earlier versions kept for provenance (`superseded/`), and
  the manuscript figure files for comparison (`reference_outputs/`). See
  `figure_scripts/README.md` for the figure-to-script map and provenance.
