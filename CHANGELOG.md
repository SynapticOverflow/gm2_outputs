# Changelog

## v1.1.0

No existing output file was modified or removed. `gm2_analysis/gm2_model_v2.py`
is unchanged (SHA-256 `6693427fa909a0a95093cf6797890c6c4004912fd2a54b7c9e9966e77fa8a4a6`).

### Added

- `task_outputs/seed_sensitivity/`: model-noise seed-sensitivity check of the
  14-parameter Sobol analysis (task10) and its manuscript packaging (task11):
  `README.md`, `SUMMARY_N256.md`, `SUMMARY_smoke_N8.md`,
  `manuscript_numbers.json`, `seed_averaged_percentile_cis.json` (the
  10-seed-averaged indices now used for Table 4 / Fig 3A),
  `seed_sensitivity_summary_N256.json`, `seed_sensitivity_summary_smoke_N8.json`,
  `fixed_seed_table.csv`, `fixed_seed_param_summary.csv`,
  `seed42_dependency_audit.md`, the scripts `task10_seed_sensitivity.py`,
  `task10_seed_sensitivity.slurm`, `task10_summary.py`,
  `task11_manuscript_package.py`, and the `N256/`, `smoke_N8/` and `logs/`
  folders (Sobol samples X, per-run JSON results, SLURM logs).
- `task_outputs/seed_sensitivity_rawY.zip`: the raw model outputs (`Y_*.npy`,
  including run C's per-seed matrices) for `N256/` and `smoke_N8/`. Unzip at
  the repository root; it extracts into `task_outputs/seed_sensitivity/`.
- `task_outputs/task9b_matched_benchmark.py`, `task_outputs/task9b_matched.slurm`,
  `task_outputs/external_benchmark_matched.json`: matched-control
  recomputation of the task9 external-benchmark reductions.
- `gm2_analysis/drivers/trajectory_driver.py`: the script that generated
  `gm2_analysis/figure_inputs/trajectories/*_trajectory.csv`.
- `gm2_analysis/dt_refinement_dt0.05/`: the factorial/Shapley, paired-contrast,
  acoustic k1/k2 and full-14 Sobol runs repeated at dt = 0.05 d (20 JSON files)
  and two dt diagnostics (`diagnostic_dt_convergence_ensemble.json`,
  `diagnostic_sobol_fixed_matrix_dt0.2_vs_dt0.05.json`), copied unmodified
  from TACC; a `README.md`; and `summarize_dt_refinement.py`, which prints the
  dt = 0.2 vs 0.05 comparison tables in that README (reads JSONs only).
- `CHANGELOG.md` (this file).

### Changed (documentation only)

- `README.md`: describes the seed-sensitivity check, the matched-control
  benchmark, the trajectory driver and the dt refinement; states which outputs
  rest on a single fixed noise path (seed 42) and that Table 4 / Fig 3A now use
  `task_outputs/seed_sensitivity/seed_averaged_percentile_cis.json`; notes the
  task1 sample used `seed=None` and that the reproducible reference sample is
  task7's (seed 20260816); "all simulations use dt = 0.2" now reads "all
  production simulations".
- `gm2_analysis/README.md`: the same, plus the dt = 0.05 refinement summary;
  corrects the statement that `task_outputs/` shares no code or outputs with
  `gm2_analysis/`; lists `trajectory_driver.py` and the two drivers that are
  not archived (`task3b_saturation_sweep_extended.py`,
  `enzyme_saturation_sweep.py`).
- `gm2_analysis/figure_scripts/README.md`: the "no file records any other dt"
  statement is scoped to the figure inputs; adds known limitation 5 (archived
  Fig 3 script and reference PNG still use the seed-42 files); notes that
  `fig2_robustness_combined.png` is manuscript Fig 3 (panels A and B) and that
  the file name predates the final figure numbering.
- `gm2_analysis/acoustic_robustness_check/inputs/README.md`: file count
  corrected ("three files" -> five files in the folder).
