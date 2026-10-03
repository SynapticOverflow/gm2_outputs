# inputs

These three files are the documented source of the infant
mechanical-index bound (`k` in [1.0, 2.0], `MI0_infant = MI0_adult / k`)
used in the acoustic-FUS robustness check. They are included unchanged.

- `literature_calibration_data.json`: literature sources and the
  `infant_bbb_dose_response_scaling_2026-09-12` derivation of the bound.
- `eta_results.json`: acoustic-model output, including the reference
  operating point (0.5 MHz, 0.6 MPa, duty cycle 0.01) read by
  `../../drivers/derive_fus_gain_mapping.py`.
- `LIFU_MODEL_V2_ADDENDUM.md`: notes on the acoustic model revision that
  produced the two files above.

`lifu_acoustic_model_v2.py` (the code that produced `eta_results.json`,
and which `derive_fus_gain_mapping.py` imports) and
`lifu_acoustic_model_v1.py` (which v2 imports) are included unchanged.
`../fus_gain_mapping.json` contains every derived number the paper uses.

## Re-running the derivation

`../../drivers/derive_fus_gain_mapping.py` has a hard-coded `LIFU_DIR`
path (a TACC directory, `/home1/11502/kartheek_nekkanti/lifu_gm2_250k`)
that must be pointed at this `inputs/` folder to re-run it. It needs numpy
and scipy, and it writes `fus_gain_mapping.json` next to the script.

Reproduction test (2026-10-03): a copy of the script with only `LIFU_DIR`
changed was run in a scratch folder alongside copies of the four files
here (Python 3.13.5, NumPy 2.1.3, SciPy 1.15.3). Its
`fus_gain_mapping.json` was compared with `../fus_gain_mapping.json`
across all 68 numeric fields: the largest absolute difference was
9.7e-10 (`saturation_ceiling_on_ratio`), and no non-numeric field
differed. `fus_entry_gain_scale` came out at 1.500000 (k=1.0) and
1.665961 (k=2.0), matching the archived values to six decimals.
Differences near 1e-9 are expected from the tolerance of the scipy
`curve_fit` fit of the MI sigmoid.
