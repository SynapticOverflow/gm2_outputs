# gm2_outputs

Archive supporting the manuscript "A prior-robust sensitivity and
factorial-necessity analysis of tri-modal viral-ultrasound gene delivery
for lysosomal neurodegeneration: uncertainty quantification and
cross-disease generalizability" (PLOS Computational Biology submission).

- `gm2_analysis/`: the simulator (`gm2_model_v2.py`), Sobol / factorial /
  Shapley drivers, all raw outputs behind the paper's tables and figures,
  parameter provenance, figure scripts and their inputs (including
  `gm2_analysis/drivers/trajectory_driver.py`, which wrote the trajectory
  CSVs), the acoustic-FUS robustness check, and the dt = 0.05 d
  refinement (`gm2_analysis/dt_refinement_dt0.05/`). See its README.
- `task_outputs/`: model-verification tests, external-benchmark
  comparison (and its matched-control recomputation), bootstrap percentile
  intervals, the width-sensitivity correlation, and the model-noise seed
  sensitivity check (`task_outputs/seed_sensitivity/`), with the scripts
  that produced them.

See `CHANGELOG.md` for what changed between archived versions.

## Single noise path (seed 42) and the seed-averaged replacement

The model is a stochastic differential equation. These outputs were each
computed on one fixed noise path (model seed 42 for every Sobol row):

- `gm2_analysis/sobol_outputs/gm2_full14_*.json` (3 files) and the 3 GM2
  full-14 records in `task_outputs/percentile_cis.json`;
- the two convergence grids,
  `gm2_analysis/sobol_outputs/convergence_grid.json` and
  `gm2_analysis/sobol_outputs/convergence_grid_second_order_true.json`;
- the 30 cross-disease Sobol files,
  `gm2_analysis/sobol_outputs/sobol_lsd_*.json` (and their 30 records in
  `task_outputs/percentile_cis.json`).

`task_outputs/seed_sensitivity/` shows that the rankings below rank 1 and
the index magnitudes depend on that seed. The manuscript's Table 4 and
Fig 3A now use the 10-seed-averaged results in
`task_outputs/seed_sensitivity/seed_averaged_percentile_cis.json`. The
seed-42 files above are kept unchanged for provenance. (Manuscript Fig 3
is the file `gm2_analysis/figure_scripts/reference_outputs/fig2_robustness_combined.png`;
the file name predates the final figure numbering.)

The raw model outputs that `task_outputs/seed_sensitivity/README.md`
lists under `N256/` and `smoke_N8/` (`Y_*.npy`, including run C's
`*_perseed.npy` matrices) are packed separately in
`task_outputs/seed_sensitivity_rawY.zip`. Run `unzip
task_outputs/seed_sensitivity_rawY.zip` from the repository root to put
them in place; `task11_manuscript_package.py` needs them.

The original full-14 Sobol sample (`gm2_analysis/drivers/task1_full14.py`)
was drawn with SALib's default `seed=None` and its X was not saved, so it
cannot be reproduced. The reproducible reference sample is the one from
`task_outputs/task7_percentile_cis.py` (sample seed 20260816), which is the
X used by `task_outputs/seed_sensitivity/`.

The factorial/Shapley results, paired contrasts, saturation sweep and
external benchmark average over many noise paths (multi-seed ensembles)
and are not single-path results. See
`task_outputs/seed_sensitivity/seed42_dependency_audit.md` for the
per-driver audit (it also flags
`task_outputs/width_sensitivity_correlation.json`, which inherits seed 42
from `gm2_full14_as_documented.json`).

## Matched-control external benchmark

`task_outputs/external_benchmark.json` (task9) reports the AAV-only and
SRT-only brain-substrate reductions relative to each arm's own initial
value (890 nmol/g). The untreated arm in this model also declines, so
`task_outputs/task9b_matched_benchmark.py` (SLURM:
`task_outputs/task9b_matched.slurm`; output
`task_outputs/external_benchmark_matched.json`) recomputes them against a
paired untreated arm (same 10 seeds, same day, dt = 0.2, configuration
otherwise unchanged from task9). It first reproduces the published values
exactly (abs diff 0.0). Mean percent reduction, own-baseline vs matched
control [95% bootstrap CI of the matched value]:

| arm | day | vs own initial value | vs matched untreated |
|---|---|---|---|
| AAV-only | 84 | 99.62 | 99.18 [98.95, 99.35] |
| AAV-only | 168 | 99.51 | 98.26 [97.57, 98.83] |
| SRT-only | 84 | 50.74 | 2.27 [1.94, 2.73] |
| SRT-only | 168 | 63.39 | 5.50 [4.82, 6.20] |

Of the two validity controls recorded in the output, the untreated arm
integrated to 200 d instead of 730 d is identical at days 84/168; the
untreated arm with FUS on is not bit-identical to FUS off (max abs diff
5.8e-7 at day 84, 5.2e-6 at day 168 nmol/g), although the script's
docstring expected it to be.

## Integrity and time step

The SHA-256 of `gm2_analysis/gm2_model_v2.py` begins
`6693427fa909a0a9`; every reported result was produced with that file.
All production simulations use a Milstein step of dt = 0.2 d. A dt = 0.05 d
refinement of the factorial/Shapley, paired-contrast, acoustic and full-14
Sobol runs is in `gm2_analysis/dt_refinement_dt0.05/` (see the summary in
`gm2_analysis/README.md`); it does not replace any production result.

## Earlier versions

An earlier exploratory LIFU subproject (`lifu_simulation/`) is
present in the earlier archived versions of this repository (GitHub
releases v1.0.0 and v1.0.1, and the Zenodo record
10.5281/zenodo.22926574). It was removed from the main branch because
no result in the paper depends on it; it remains in those archived
versions and in git history.
