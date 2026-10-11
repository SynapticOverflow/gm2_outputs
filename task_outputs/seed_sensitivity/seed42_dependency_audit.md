# Read-only audit: other uses of a single fixed model-noise seed

Done 2026-10-09 by reading code only; nothing was rerun. Line numbers refer to
`gm2_analysis/drivers/` in this repo, which is byte-identical to the `$WORK`
copies (checked with `cmp`), except where a `$WORK` path is given.
`task7_percentile_cis.py` lives in `task_outputs/` (also identical to `$WORK`).

"Shared by every row" means every model evaluation in that analysis uses the same
noise path (common random numbers with one path). That is the design task10 showed
gives seed-dependent S_T.

| driver | file:line | model-eval seed | every row shares it? | output(s) | single fixed noise path? |
|---|---|---|---|---|---|
| task1_full14.py (production full-14 Sobol) | task1_full14.py:40, :51 | 42 | yes | `sobol_outputs/gm2_full14_*.json` | **yes**. Tables 4/5, Fig 3; replaced by `seed_averaged_percentile_cis.json` |
| task2_convergence_grid.py | task2_convergence_grid.py:35, :47 (bootstrap also seeded with 42 at :60) | 42 | yes, at every N in the grid | `sobol_outputs/convergence_grid.json` | **yes**. Also, X is sampled with SALib's default `seed=None` (:56), so it can't be reproduced |
| task2c_convergence_grid_second_order_true.py | task2c_…py:42, :55 (bootstrap :68) | 42 | yes | `sobol_outputs/convergence_grid_second_order_true.json` | **yes**. X uses `seed=None` (:64) |
| task3_saturation_sweep.py | task3_saturation_sweep.py:51 (`N_REALIZATIONS=50`), :54 (`SEED_BASE=0`), :88 | 0–49 | no. 50 realizations per k; the same 50 seeds at every k (CRN), averaged | `sobol_outputs/saturation_sweep_gm2_model_v2.json` | no (50-path ensemble) |
| task3b_saturation_sweep_extended.py (the saturation-elasticity driver; `$WORK` only, not in repo) | $WORK/task3b_…py:56, :59, :73, :170 | 0–49 | no. Same 50-seed CRN scheme as task3 | `$WORK/data/saturation_sweep_extended{,_elasticity}.json` | no (50-path ensemble) |
| enzyme_saturation_sweep.py (`$WORK` only, not in repo) | $WORK/enzyme_saturation_sweep.py:29 → $WORK/ml/robustness_audit.py:322 (`extreme_value_sweep(..., seed=42)` default) | 42 | yes. All 9 points × 2 params | `$WORK/data/enzyme_saturation_sweep.json` | **yes**. Manuscript use unknown (not archived in the repo) |
| sobol_driver_lsd.py (cross-disease Sobol, as_documented leg) | sobol_driver_lsd.py:64, :79 (default `seed=42`), :106 (called without seed) | 42 | yes | `sobol_outputs/sobol_lsd_<disease>.json` (10 diseases) | **yes**. X uses `seed=None` (:84) |
| task6_cross_disease.py (equal_width / tier1_promotion legs) | task6_cross_disease.py:43, :94 | 42 | yes | `sobol_outputs/sobol_lsd_<disease>_{equal_width,tier1_promotion}.json` (20 files) | **yes**. X uses `seed=None` (:103) |
| task7_percentile_cis.py | task_outputs/task7_percentile_cis.py:78, :149 (full14 rerun); LSD part re-analyses the archived seed-42 `Y_raw` (:105–120) | 42 | yes | `task_outputs/percentile_cis.json` (all 33 records) | **yes**, all 33 records. The 3 full14 records are superseded by `seed_averaged_percentile_cis.json`; the 30 LSD records aren't |

Other drivers found while auditing (not on the requested list):

| driver | seed scheme | single fixed path? |
|---|---|---|
| task8_width_sensitivity_correlation.py | reads `gm2_full14_as_documented.json` (task_outputs/task8_…py:48) | **yes, inherited** from task1 seed 42 |
| factorial_arms_shapley.py / task5_paired_contrasts.py | seeds 0..N_SEEDS−1 (200), CRN across arms (factorial_arms_shapley.py:41, :80) | no (ensemble) |
| task9_external_benchmark.py / task9b | 10 seeds per arm (task9:81) | no (ensemble) |

## Results that rest on a single fixed noise path (seed 42)

1. Full-14 Sobol S_T/S1/S2 and the prior-width comparison: Tables 4/5, Fig 3. These now
   switch to seed-averaged values.
2. Sobol convergence grids (`convergence_grid.json`, `convergence_grid_second_order_true.json`):
   the convergence-in-N figure(s). Convergence was assessed on a single noise path, so it shows
   Monte-Carlo convergence of the Sobol estimator *for that path*, not convergence of the
   seed-averaged indices.
3. Cross-disease Sobol, all 10 diseases × 3 prior scenarios (`sobol_lsd_*.json`, 30 files), and
   their percentile CIs in `percentile_cis.json`.
4. Width-sensitivity correlation (task8), which is derived from `gm2_full14_as_documented.json`.
5. Enzyme (vmax/km) extreme-value saturation sweep, if the manuscript uses it.

The k_T4_entry saturation sweep and elasticity (task3/task3b), the factorial/Shapley results,
and the external benchmark use multi-seed ensembles, so they are not single-path results.

Mapping of these output files to specific manuscript table or figure numbers (other than
Tables 4/5 and Fig 3) isn't recorded on this filesystem; `gm2_analysis/README.md` only says
the archive backs "Tables 2, 3, 5, 6 and the convergence / saturation / Shapley figures".
