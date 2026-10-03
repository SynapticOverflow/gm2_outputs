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

`lifu_acoustic_model_v2.py`, the code that produced `eta_results.json`
(and which `derive_fus_gain_mapping.py` imports), is NOT included here.
`../fus_gain_mapping.json` contains every derived number the paper uses.
