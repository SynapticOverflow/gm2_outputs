# dt_refinement_dt0.05

Time-step refinement check. **Production runs use a Milstein step of dt = 0.2 d**
(every file in `../sobol_outputs/`, `../factorial_shapley_outputs/` and
`../acoustic_robustness_check/`). The files here rerun those analyses at
**dt = 0.05 d** with everything else unchanged. Nothing here replaces a
production result.

Lonestar6, SLURM jobs 3492405 (factorial/Shapley/paired contrasts), 3492406
(acoustic k1/k2 reruns) and 3492408 (full-14 Sobol and the two diagnostics),
2026-10-06/07. Every file's `run_meta` records `gm2_model_v2.py` sha256
`6693427fa909a0a95093cf6797890c6c4004912fd2a54b7c9e9966e77fa8a4a6` before and
after the job. The JSON files were copied from TACC unmodified.

## What was run

| files | dt | seeds | arms / scenarios | dt=0.2 counterpart |
|---|---|---|---|---|
| `factorial_arms_shapley_dt0.05{,_day45,_day90,_day180}.json`, `shapley_with_ci_dt0.05.json`, `paired_contrasts_dt0.05.json` | 0.05 | SDE seeds 0-199, matched across all 8 arms; bootstrap B=1000, `default_rng(42)` | 8-arm factorial (SRT x AAV x FUS on/off), Tay-Sachs, day 45/90/180/365 | `../factorial_shapley_outputs/` |
| `*_acoustic_k1*_dt0.05.json`, `*_acoustic_k2*_dt0.05.json` | 0.05 | as above | same 8 arms with `fus_entry_gain_scale` = 1.5 (k1) / 1.665961 (k2) | `../acoustic_robustness_check/` |
| `gm2_full14_{as_documented,equal_width,tier1_promotion}_dt0.05.json` | 0.05 | model seed 42 for every row (task1 design), N=256 | 14-parameter Sobol, three prior-width scenarios | `../sobol_outputs/gm2_full14_*.json` |
| `diagnostic_sobol_fixed_matrix_dt0.2_vs_dt0.05.json` | 0.2 and 0.05 | model seed 42 per row; Sobol sample seed 42, same matrix at both dt | 14-parameter Sobol, three scenarios | (self-contained) |
| `diagnostic_dt_convergence_ensemble.json` | 0.2, 0.1, 0.05, 0.025, 0.0125 | seeds 0-199 | arms none, SRT, AAV, AAV+FUS, SRT+AAV, tri; day 45 and 365 | (self-contained) |

The two `diagnostic_*` files are marked in their `run_meta` as "EXTRA
diagnostic, not a requested task".

The driver scripts and SLURM files for these jobs were not in the staging copy
from TACC, so they are not archived here. `summarize_dt_refinement.py` is not
a driver: it only reads the JSONs and prints the tables below (no model
evaluations). The tables below are its unedited output.

## Reading the comparison

- **Ensemble results (factorial/Shapley, paired contrasts, convergence
  ensemble)** average 200 noise paths, so dt=0.2 vs dt=0.05 differences
  there reflect the step size plus Monte-Carlo error.
- **Single-seed Sobol results (`gm2_full14_*_dt0.05.json`, the fixed-matrix
  diagnostic)** are *not* a clean step-size comparison. `simulate_milstein`
  draws one `standard_normal` vector per step from `default_rng(seed)`, so
  "seed 42" is a different Brownian path at dt=0.05 than at dt=0.2. The
  differences below mix the step-size effect with the noise-path effect that
  `../../task_outputs/seed_sensitivity/` shows is large for these indices.

## Shapley benefit, day 365 (point [95% CI]), 200 matched seeds

| config | modality | dt=0.2 | dt=0.05 |
|---|---|---|---|
| base | AAV | 216.662 [199.660, 235.666] | 203.366 [189.230, 218.593] |
| base | SRT | 11.703 [11.012, 12.372] | 9.120 [8.609, 9.638] |
| base | FUS | -0.013 [-0.018, -0.007] | -0.007 [-0.011, -0.004] |
| acoustic k1 | AAV | 216.662 [199.660, 235.666] | 203.366 [189.230, 218.593] |
| acoustic k1 | SRT | 11.703 [11.012, 12.372] | 9.120 [8.609, 9.638] |
| acoustic k1 | FUS | -0.013 [-0.018, -0.007] | -0.007 [-0.011, -0.004] |
| acoustic k2 | AAV | 216.660 [199.658, 235.665] | 203.365 [189.229, 218.592] |
| acoustic k2 | SRT | 11.703 [11.012, 12.372] | 9.120 [8.609, 9.638] |
| acoustic k2 | FUS | -0.014 [-0.021, -0.008] | -0.008 [-0.013, -0.005] |

| config | untreated burden, dt=0.2 | dt=0.05 | total benefit, dt=0.2 | dt=0.05 |
|---|---|---|---|---|
| base | 250.76 | 233.56 | 228.35 | 212.48 |
| acoustic k1 | 250.76 | 233.56 | 228.35 | 212.48 |
| acoustic k2 | 250.76 | 233.56 | 228.35 | 212.48 |

## Paired FUS contrasts, day 365 (mean diff [95% CI]; significant?)

| config | contrast | dt=0.2 | dt=0.05 |
|---|---|---|---|
| base | untreated_vs_FUS_alone | 7.33e-07 [6.442e-07, 8.278e-07]; True | 3.824e-07 [3.313e-07, 4.363e-07]; True |
| base | AAV_vs_AAV_FUS | -0.03183 [-0.04538, -0.01915]; True | -0.01722 [-0.02476, -0.01031]; True |
| base | SRT_vs_SRT_FUS | 0.01283 [0.01038, 0.01565]; True | 0.004011 [0.003406, 0.004697]; True |
| base | SRT_AAV_vs_tri_modal | -0.02838 [-0.04073, -0.01725]; True | -0.0157 [-0.02241, -0.009463]; True |
| acoustic k1 | untreated_vs_FUS_alone | 7.33e-07 [6.425e-07, 8.334e-07]; True | 3.824e-07 [3.307e-07, 4.339e-07]; True |
| acoustic k1 | AAV_vs_AAV_FUS | -0.03183 [-0.04677, -0.01965]; True | -0.01722 [-0.0245, -0.01092]; True |
| acoustic k1 | SRT_vs_SRT_FUS | 0.01283 [0.01044, 0.01586]; True | 0.004011 [0.003396, 0.004682]; True |
| acoustic k1 | SRT_AAV_vs_tri_modal | -0.02838 [-0.0406, -0.01753]; True | -0.0157 [-0.02271, -0.009818]; True |
| acoustic k2 | untreated_vs_FUS_alone | 8.147e-07 [7.143e-07, 9.264e-07]; True | 4.247e-07 [3.673e-07, 4.82e-07]; True |
| acoustic k2 | AAV_vs_AAV_FUS | -0.03545 [-0.05206, -0.02192]; True | -0.01914 [-0.02722, -0.01214]; True |
| acoustic k2 | SRT_vs_SRT_FUS | 0.01284 [0.01044, 0.01586]; True | 0.004012 [0.003396, 0.004682]; True |
| acoustic k2 | SRT_AAV_vs_tri_modal | -0.03161 [-0.0452, -0.01955]; True | -0.01744 [-0.02523, -0.01091]; True |

## Full-14 Sobol S_T, seed 42 (task1 production at dt=0.2 vs task1 rerun at dt=0.05)

### as_documented  (Y_mean: dt=0.2 3.830, dt=0.05 50.114)

| parameter | S_T dt=0.2 | S_T dt=0.05 |
|---|---|---|
| k_T4_entry | 0.044 | 0.060 |
| t4_dose_total | 0.079 | 0.440 |
| gm2_synth | 0.228 | 0.249 |
| k_T4_payload | 0.099 | 0.038 |
| km_brain | 0.197 | 0.044 |
| vmax_brain | 0.218 | 0.102 |
| k_T4_decay | 0.227 | 0.069 |
| IC50 | 0.003 | 0.002 |

### equal_width  (Y_mean: dt=0.2 3.635, dt=0.05 51.339)

| parameter | S_T dt=0.2 | S_T dt=0.05 |
|---|---|---|
| k_T4_entry | 0.001 | 0.018 |
| t4_dose_total | 0.002 | 0.159 |
| gm2_synth | 0.255 | 0.408 |
| k_T4_payload | 0.096 | 0.060 |
| km_brain | 0.217 | 0.076 |
| vmax_brain | 0.255 | 0.175 |
| k_T4_decay | 0.260 | 0.122 |
| IC50 | 0.003 | 0.003 |

### tier1_promotion  (Y_mean: dt=0.2 3.760, dt=0.05 50.809)

| parameter | S_T dt=0.2 | S_T dt=0.05 |
|---|---|---|
| k_T4_entry | 0.003 | 0.009 |
| t4_dose_total | 0.039 | 0.542 |
| gm2_synth | 0.241 | 0.257 |
| k_T4_payload | 0.096 | 0.039 |
| km_brain | 0.195 | 0.044 |
| vmax_brain | 0.237 | 0.103 |
| k_T4_decay | 0.244 | 0.069 |
| IC50 | 0.003 | 0.002 |

## Diagnostic: same Sobol matrix (sample seed 42), model seed 42, dt=0.2 vs 0.05

run_meta.seeds: SDE seed 42 per row; sobol_sample seed=42 (fixed matrix shared by both dt)

| scenario | Y_mean dt=0.2 | Y_mean dt=0.05 | top-3 S_T dt=0.2 | top-3 S_T dt=0.05 |
|---|---|---|---|---|
| as_documented | 3.83 | 50.13 | gm2_synth 0.227, k_T4_decay 0.219, vmax_brain 0.213 | t4_dose_total 0.476, gm2_synth 0.257, vmax_brain 0.100 |
| equal_width | 3.64 | 51.31 | gm2_synth 0.249, k_T4_decay 0.237, vmax_brain 0.232 | gm2_synth 0.410, vmax_brain 0.172, t4_dose_total 0.156 |
| tier1_promotion | 3.76 | 50.81 | gm2_synth 0.234, k_T4_decay 0.228, vmax_brain 0.222 | t4_dose_total 0.501, gm2_synth 0.264, vmax_brain 0.102 |

## Diagnostic: dt-convergence ensemble, seeds 0-199 (mean gm2_brain +/- SEM)

| day | arm | dt=0.2 | dt=0.1 | dt=0.05 | dt=0.025 | dt=0.0125 |
|---|---|---|---|---|---|---|
| day45 | none | 661.04 +/- 18.60 | 651.17 +/- 17.32 | 667.19 +/- 20.05 | 659.46 +/- 18.55 | 647.63 +/- 18.33 |
| day45 | SRT | 655.76 +/- 18.54 | 646.48 +/- 17.27 | 662.82 +/- 20.00 | 655.23 +/- 18.51 | 643.54 +/- 18.28 |
| day45 | AAV | 551.50 +/- 18.07 | 538.02 +/- 16.74 | 554.30 +/- 19.35 | 544.21 +/- 18.08 | 532.60 +/- 17.74 |
| day45 | AAV+FUS | 546.43 +/- 18.04 | 535.14 +/- 16.71 | 551.70 +/- 19.33 | 541.57 +/- 18.07 | 529.97 +/- 17.72 |
| day45 | SRT+AAV | 546.34 +/- 18.01 | 533.43 +/- 16.69 | 550.03 +/- 19.29 | 540.08 +/- 18.04 | 528.61 +/- 17.69 |
| day45 | tri | 541.22 +/- 17.98 | 530.54 +/- 16.66 | 547.41 +/- 19.27 | 537.42 +/- 18.02 | 525.97 +/- 17.67 |
| day45 | paired_AAV_minus_AAVFUS | 5.07 +/- 0.07 | 2.87 +/- 0.04 | 2.60 +/- 0.04 | 2.64 +/- 0.03 | 2.62 +/- 0.04 |
| day365 | none | 250.76 +/- 9.55 | 240.96 +/- 8.27 | 233.56 +/- 7.50 | 244.43 +/- 8.92 | 244.89 +/- 8.26 |
| day365 | SRT | 230.12 +/- 8.92 | 223.02 +/- 7.70 | 217.39 +/- 7.04 | 228.18 +/- 8.37 | 228.87 +/- 7.77 |
| day365 | AAV | 25.14 +/- 1.30 | 24.44 +/- 1.35 | 23.13 +/- 1.23 | 22.31 +/- 1.05 | 24.35 +/- 1.11 |
| day365 | AAV+FUS | 25.17 +/- 1.30 | 24.46 +/- 1.35 | 23.15 +/- 1.23 | 22.33 +/- 1.05 | 24.36 +/- 1.11 |
| day365 | SRT+AAV | 22.38 +/- 1.15 | 22.06 +/- 1.23 | 21.06 +/- 1.12 | 20.38 +/- 0.94 | 22.23 +/- 1.01 |
| day365 | tri | 22.41 +/- 1.16 | 22.08 +/- 1.23 | 21.08 +/- 1.12 | 20.40 +/- 0.95 | 22.25 +/- 1.01 |
| day365 | paired_AAV_minus_AAVFUS | -0.03 +/- 0.01 | -0.02 +/- 0.00 | -0.02 +/- 0.00 | -0.02 +/- 0.00 | -0.02 +/- 0.00 |

