# gm2_analysis

GM2 gangliosidosis (Tay-Sachs / Sandhoff) mechanistic-model sensitivity
analysis supporting the PLOS Computational Biology submission. This
directory contains the model itself, the driver scripts, and the
output files behind the paper's core Tables 2, 3, 5, 6 and the
convergence / saturation / Shapley figures, plus the follow-up
acoustic-mechanism robustness check for the FUS (focused ultrasound)
arm and a dt = 0.05 d time-step refinement.

`../task_outputs/` (percentile CIs, verification tests, external
benchmark, seed sensitivity) is kept in its own top-level directory, but
it is not independent of this one: its scripts import `gm2_model_v2.py`
and `drivers/sobol_driver_v2.py` by module name (so these must be on the
Python path), `task9b_matched_benchmark.py` imports the model from
`../gm2_analysis/`, and `task8_width_sensitivity_correlation.py` reads
`sobol_outputs/gm2_full14_as_documented.json`.

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
  `gm2_solver.py`, `parallel_eval.py`), the two acoustic-robustness
  drivers (`derive_fus_gain_mapping.py`, `run_acoustic_fus_analysis.py`),
  and `trajectory_driver.py`, which wrote the ten
  `figure_inputs/trajectories/<disease>_trajectory.csv` files (baseline
  parameters, all 17 states, dt = 0.2, model seed 42, run to each
  disease's `t_ref_days`; it writes to `data/trajectories/` relative to
  the working directory, as on TACC). Not archived here: the
  saturation-elasticity driver `task3b_saturation_sweep_extended.py` and
  `enzyme_saturation_sweep.py` (on TACC `$WORK` only; see
  `../task_outputs/seed_sensitivity/seed42_dependency_audit.md`).

- `sobol_outputs/` — the 14-parameter full-model Sobol results
  (`gm2_full14_*.json`, three prior-width scenarios), the 10-disease
  cross-disease Sobol results (`sobol_lsd_*.json`, three variants each),
  the convergence grids, and the saturation sweep. The full-14 files, both
  convergence grids and all 30 `sobol_lsd_*.json` files were computed on a
  single fixed noise path (model seed 42); see "Seed 42 and the
  seed-averaged results" below.

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

- `figure_inputs/` — the per-disease `Y_raw` CSVs and model-trajectory
  CSVs read by two of the figure scripts. See `figure_inputs/README.md`.

- `dt_refinement_dt0.05/` — the factorial/Shapley, paired-contrast,
  acoustic k1/k2 and full-14 Sobol analyses rerun at dt = 0.05 d, plus two
  dt diagnostics. See `dt_refinement_dt0.05/README.md` and "Time step"
  below.

## Seed 42 and the seed-averaged results

The Sobol drivers evaluate every sample row with the same model noise
seed (42). That applies to `sobol_outputs/gm2_full14_*.json` and the 3
GM2 full-14 records in `../task_outputs/percentile_cis.json`, to the two
convergence grids (`sobol_outputs/convergence_grid.json`,
`sobol_outputs/convergence_grid_second_order_true.json`), and to the 30
cross-disease Sobol files `sobol_outputs/sobol_lsd_*.json` (10 diseases x
3 prior scenarios; their 30 records in `percentile_cis.json` re-analyse
the same seed-42 Y). Each of these results is conditional on one noise
path.

`../task_outputs/seed_sensitivity/` (task10/task11) repeats the full-14
analysis with 15 distinct fixed seeds, per-row independent seeds, and a
10-seed average. From its README: `gm2_synth` is rank 1 in 14/15 fixed
seeds, but ranks 3-5 reshuffle, and seed 42 gives the lowest `gm2_synth`
S_T of the 15 seeds (0.24 vs median 0.36); under the 10-seed average
`gm2_synth` is about 0.44. The prior-width-artifact direction holds in
every seed. **The manuscript's Table 4 and Fig 2A now use the
10-seed-averaged results in
`../task_outputs/seed_sensitivity/seed_averaged_percentile_cis.json`.**
The seed-42 files here are unchanged.

The original full-14 sample (`drivers/task1_full14.py`) used SALib's
default `seed=None` and its X was not saved, so it is not reproducible
(the convergence-grid and cross-disease drivers also sample with
`seed=None`). The reproducible reference is the sample from
`../task_outputs/task7_percentile_cis.py` (sample seed 20260816); the seed
sensitivity runs use that X, and their seed-42 control reproduces
`percentile_cis.json` exactly.

The factorial/Shapley outputs, paired contrasts and saturation sweep are
multi-seed ensembles (200 and 50 seeds), not single-path results.

## Matched-control external benchmark

`../task_outputs/task9b_matched_benchmark.py` recomputes the task9
AAV-only / SRT-only brain-substrate reductions against a paired untreated
arm (same 10 seeds, same day, dt = 0.2), because the published values are
relative to each arm's own initial value and the untreated arm declines on
its own. Output: `../task_outputs/external_benchmark_matched.json`; the
numbers are tabulated in the root `README.md`.

## Time step

All production runs use a Milstein step of dt = 0.2 d.
`dt_refinement_dt0.05/` reruns them at dt = 0.05 d. Numbers below are
copied from the output of `dt_refinement_dt0.05/summarize_dt_refinement.py`
(full tables in `dt_refinement_dt0.05/README.md`).

- Factorial/Shapley, day 365, 200 matched seeds (base model): Shapley
  benefit of AAV 216.662 [199.660, 235.666] at dt = 0.2 vs 203.366
  [189.230, 218.593] at dt = 0.05; SRT 11.703 [11.012, 12.372] vs 9.120
  [8.609, 9.638]; FUS -0.013 [-0.018, -0.007] vs -0.007 [-0.011, -0.004].
  Untreated burden 250.76 vs 233.56; total benefit 228.35 vs 212.48.
- Paired FUS contrasts, day 365: all 12 (4 contrasts x base, k1, k2) have
  the same sign and are significant at both step sizes; magnitudes are
  smaller at dt = 0.05 (e.g. base AAV vs AAV+FUS -0.03183 vs -0.01722).
- dt-convergence ensemble (seeds 0-199, dt = 0.2 to 0.0125): day-365
  untreated mean 250.76 +/- 9.55 (dt = 0.2), 233.56 +/- 7.50 (0.05),
  244.89 +/- 8.26 (0.0125). The day-45 paired AAV minus AAV+FUS difference
  is 5.07 +/- 0.07 at dt = 0.2 and 2.87, 2.60, 2.64, 2.62 at dt = 0.1 to
  0.0125.
- Full-14 Sobol with model seed 42: indices change substantially (e.g.
  as_documented `t4_dose_total` S_T 0.079 vs 0.440; Y_mean 3.830 vs
  50.114). This is not a clean step-size comparison: seed 42 is a
  different Brownian path at each dt (one normal draw per step), and the
  seed-sensitivity check shows these indices depend strongly on the
  noise path.
