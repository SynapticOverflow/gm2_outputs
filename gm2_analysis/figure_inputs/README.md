# figure_inputs

Input CSVs for two manuscript figures (see `../figure_scripts/README.md`).

## `Y_raw/` (10 files, `<disease>_Y_raw.csv`)

- Columns: `index, Y_raw`; one row per Saltelli evaluation (k = 16N per
  disease).
- Values equal the `Y_raw` arrays in
  `../sobol_outputs/sobol_lsd_<disease>.json`. They were extracted from
  those JSONs; no script wrote them.
- Feeds `fig_raw_output_distributions.png`.

## `trajectories/` (10 files, `<disease>_trajectory.csv`)

- 19 columns: the 17 model states, `time_days`, and `disease`.
- Uniform time step dt = 0.2 d.
- Each file ends at 2x the disease's Sobol evaluation window.
- Feeds `fig_real_trajectories.png`.
