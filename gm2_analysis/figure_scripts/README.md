# Figure-generation scripts: GM2 PLOS-CB manuscript

## Provenance

These scripts were written and run in a Claude (Anthropic) chat session
during manuscript development, not on TACC. A search of Lonestar6 for
figure code therefore (correctly) finds none.

They were recovered from the saved conversation transcripts by extracting
the logged file-creation calls. The latest logged version of each script
is included unmodified, with one exception documented below. Paths inside
the scripts refer to the chat environment (`/home/claude/...`,
`/mnt/user-data/uploads/...`) and have NOT been rewritten; see the path
map below.

## Final figures -> scripts (the 7 figures in the manuscript)

| Manuscript figure file | Script (in `final/`) |
|---|---|
| fig0_claim_audit_reversal.png | build_remaining_figures.py |
| fig_convergence_saturation_elasticity.png | build_convergence_saturation_elasticity.py |
| fig2_robustness_combined.png | build_fig2_v3.py |
| fig_real_trajectories.png | build_trajectory_figure_v2.py (see edit below) |
| fig_fus_shapley_time.png | build_factorial_shapley_v3.py |
| fig3_factorial_necessity.png | build_remaining_figures.py (final rebuild, with paired-contrast panel B; build_factorial_shapley_v3.py also writes this filename but was run earlier and overwritten) |
| fig_raw_output_distributions.png | build_real_distribution_figure.py |

`build_remaining_figures.py` also writes `fig_elasticity.png`, which is no
longer used (superseded by panel C of the merged convergence/saturation/
elasticity figure).

## The one post-creation edit

`build_trajectory_figure_v2.py`: as logged at creation, the y-axis label
was `"Brain GM2 (nmol/g)"`. A later logged command
(`sed -i 's/"Brain GM2 (nmol\/g)"/"Substrate burden (nmol\/g)"/'`) changed
it before the final run. The included file has this edit applied; line 44
reads `ax.set_ylabel("Substrate burden (nmol/g)", fontsize=7)`, matching
the output logged in the transcript. Transcripts were searched for other
in-place edits (sed, perl, shell redirection, str_replace) to the final
scripts; none were found.

## Input data -> location in the data repository

| Path used inside scripts | Where the file is in the repository |
|---|---|
| .../gm2_lsd_tasks1-6_results/gm2_full14_*.json | gm2_analysis/sobol_outputs/ |
| .../saturation_sweep_gm2_model_v2.json, convergence_grid.json | gm2_analysis/sobol_outputs/ |
| .../factorial_arms_shapley*.json, shapley_with_ci.json | gm2_analysis/factorial_shapley_outputs/ |
| /mnt/user-data/uploads/factorial_arms_shapley.json | gm2_analysis/factorial_shapley_outputs/ |
| /mnt/user-data/uploads/<disease>_Y_raw.csv (x10) | **NOT IN THE REPOSITORY** |
| /mnt/user-data/uploads/<disease>_trajectory.csv (x10) | **NOT IN THE REPOSITORY** |

## Input CSVs for two figures

`fig_raw_output_distributions.png` needs the ten `*_Y_raw.csv` files and
`fig_real_trajectories.png` needs the ten `*_trajectory.csv` files. They
were located on Lonestar6 at
`/work/11502/kartheek_nekkanti/vista/gm2_sobol/data/raw_csv/` and
`.../data/trajectories/` and should be archived in
`gm2_analysis/figure_inputs/` (subfolders `Y_raw/`, `trajectories/`).
Facts about these files:
- `Y_raw`: columns `index, Y_raw`; row count equals k = 16N for every
  disease (e.g., 6,144 for Tay-Sachs, 16,384 for Pompe). Per the data
  owner's audit, the values equal the `Y_raw` arrays in
  `sobol_lsd_<disease>.json` (base configuration); no script on disk wrote
  the CSVs, so they were extracted from the JSONs by hand.
- trajectories: 19 columns (17 model states, `time_days`, `disease`); a
  uniform step of **dt = 0.2 d** in every file; each ends at twice the
  disease's Sobol evaluation window (for four diseases the window is a
  half-day value, e.g., 287.5 d, shown rounded in the manuscript table).

## Verification performed

All six final scripts were re-run on the archived data, and **all seven
manuscript figures were reproduced byte-for-byte** (matching SHA-256, zero
differing pixels against `reference_outputs/`). Inputs: the 20 CSVs above
and the 42 JSON files in `gm2_analysis/sobol_outputs/` and
`gm2_analysis/factorial_shapley_outputs/`. Only path strings were changed
in scratch copies (including the hard-coded `/mnt/user-data/uploads`
strings inside `build_factorial_shapley_v3.py` and
`build_remaining_figures.py`); plotting logic was untouched.
`build_factorial_shapley_v3.py` must run before `build_remaining_figures.py`,
because both write `fig3_factorial_necessity.png` and the latter produces
the final version.
Environment: Python 3.12.3, matplotlib 3.10.8, numpy 2.4.4, pandas 3.0.2.

Further checks performed on the data:
- `Y_raw` CSVs vs. `sobol_lsd_<disease>.json`: all ten are exactly equal
  to the JSON `Y_raw` arrays when parsed exactly
  (`pandas.read_csv(..., float_precision="round_trip")`). pandas' default
  fast parser differs from the JSON by about 1e-16 relative, which is a
  parsing artifact, not a data difference.
- The hand-typed Panel B constants in `build_fig2_v3.py` (dominant-parameter
  and entry-rate S_T for ten diseases) equal the values in the
  `sobol_lsd_*.json` files rounded to three decimals.
- The `sed` label edit described above is confirmed by the byte-identical
  trajectory figure.

## Time step

All 20 trajectory CSVs have a uniform step of dt = 0.2 d. The run
metadata records dt = 0.2 in: the three `gm2_full14_*.json`, both
convergence grids, the saturation sweep, the four factorial/Shapley JSONs,
and the 20 equal-width and Tier-1-promoted cross-disease JSONs. No file
records any other value. The ten as-documented `sobol_lsd_<disease>.json`
files record no dt; 0.2 d is the default of the driver that produced them
(per the data owner's audit of the code).

## Known limitations

1. **Reproduction environment differs from the original.** The re-run used the library versions listed above; the original session's versions were not recorded.
2. **Hard-coded values in fig3 panel B.** `build_remaining_figures.py`
   types in the paired-contrast means, CI bounds, and the
   practical-equivalence margin (2.5076) as constants rather than reading
   them from a file. They correspond to the day-365 values in manuscript
   Table 6 (the original paired_contrasts.json), not to the later
   acoustic-rerun files.
3. **build_convergence_saturation_elasticity.py** was written after the
   transcripts end. Relative to the run that produced the PNG, only the
   data/output path constants and a closing print message were changed;
   the plotting logic is the same. It was run and visually checked when
   written, and not re-run since.
4. **Fig 3A vs. Table 4.** `fig2_robustness_combined.png` plots the production-run indices read from gm2_full14_*.json, while manuscript Table 4 reports the verification-run values with bootstrap CIs. The largest difference is 0.021 (total AAV dose, as-documented: 0.079 vs. 0.058); BBB-entry rate is 0.044 vs. 0.039. The figure draws point estimates only (no error bars); the manuscript caption states this and points to Tables 4 and 5 for intervals. Rebuilding the figure from `task_outputs/percentile_cis.json` would make figure and tables agree.

## superseded/

Earlier versions kept for provenance only; none produced a current figure:
make_figures_aug5.py, make_figures_v2.py, make_figures_jul30_ondisk_version.py,
build_convergence_saturation_v2.py (separate convergence + saturation
figure, replaced by the merged 3-panel figure),
build_trajectory_figure_v1.py, build_factorial_and_shapley_time.py.

## reference_outputs/

Copies of the seven figure files as they appear in the manuscript, for
comparison against any re-run.
