"""
task8_width_sensitivity_correlation.py

Task 3: Spearman rank correlation between each parameter's assigned
prior-width (relative half-width / coefficient-of-variation-like scale
used to build the as_documented Sobol sampling bounds) and its
total-order Sobol index ST, for the GM2 three-way analysis's 9 nonzero
parameters under the as_documented scenario.

Source of ST: data/gm2_full14_as_documented.json (N=256, 14 params,
written by task1_full14.py / sobol_driver_v2.run_scenario()).

"9 nonzero parameters": of the 14 parameters in that file, 5 have
ST == 0.0 exactly (inf_threshold, k_inf, k_res, rho_g, rho_i --
mechanically disconnected from gm2_brain, the model's Sobol output, at
this evaluation timepoint/config). The remaining 9 all have ST > 0
(smallest is fus_entry_gain_scale at ~3e-6, still nonzero) -- these are
the 9 used here, matching the task's "9 nonzero parameters" count
exactly.

Source of widths: sobol_driver_v2.py's make_bounds('as_documented')
construction -- TIER2_AS_DOCUMENTED params (k_T4_entry, t4_dose_total,
inf_threshold, rho_g, rho_i) get half-width 0.60, all others get 0.25.
This IS the prior width assignment as-documented in the driver (not
re-derived or guessed): bounds = base_val * (1 +/- hw), so hw is
directly the relative half-width / coefficient-of-variation-like scale
assigned to each parameter.

Reuses sobol_driver_v2.PARAM_NAMES / TIER2_AS_DOCUMENTED /
TIER1_HALF_WIDTH / TIER2_HALF_WIDTH unmodified; does not rerun the
model or the Sobol analysis.

Output: data/width_sensitivity_correlation.json
"""
import json
import os
import sys
from datetime import datetime, timezone

from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sobol_driver_v2 as sdv2

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")

with open(os.path.join(DATA_DIR, "gm2_full14_as_documented.json")) as f:
    full14 = json.load(f)

names_all = full14["names"]
ST_all = full14["ST"]
ST_conf_all = full14["ST_conf"]

nonzero_names, nonzero_ST, nonzero_ST_conf, nonzero_width = [], [], [], []
zero_names = []
for name, st, conf in zip(names_all, ST_all, ST_conf_all):
    if st == 0.0:
        zero_names.append(name)
        continue
    hw = sdv2.TIER2_HALF_WIDTH if name in sdv2.TIER2_AS_DOCUMENTED else sdv2.TIER1_HALF_WIDTH
    nonzero_names.append(name)
    nonzero_ST.append(st)
    nonzero_ST_conf.append(conf)
    nonzero_width.append(hw)

assert len(nonzero_names) == 9, f"expected 9 nonzero parameters, got {len(nonzero_names)}: {nonzero_names}"

rho, pval = spearmanr(nonzero_width, nonzero_ST)

out = {
    "task": "width_sensitivity_correlation",
    "source_file": "gm2_full14_as_documented.json",
    "scenario": "as_documented",
    "n_params_total": len(names_all),
    "n_params_nonzero_ST": len(nonzero_names),
    "params_excluded_ST_exactly_zero": zero_names,
    "parameter_names": nonzero_names,
    "widths": nonzero_width,
    "width_definition": "as_documented half-width hw used in sobol_driver_v2.make_bounds: bounds=[base*(1-hw), base*(1+hw)]; TIER2_AS_DOCUMENTED params (k_T4_entry, t4_dose_total, inf_threshold, rho_g, rho_i) use hw=0.60, all other params use hw=0.25",
    "ST_values": nonzero_ST,
    "ST_conf_values": nonzero_ST_conf,
    "spearman_r": float(rho),
    "spearman_p": float(pval),
    "interpretation": (
        "Strong positive correlation would mean the as_documented ranking tracks "
        "assigned prior width rather than biology (the paper's central claim under "
        "test); a weak/non-significant correlation is evidence against that."
    ),
    "run_meta": {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    },
}

outpath = os.path.join(DATA_DIR, "width_sensitivity_correlation.json")
with open(outpath, "w") as f:
    json.dump(out, f, indent=2)

print(json.dumps({k: v for k, v in out.items() if k not in ("run_meta",)}, indent=2))
print(f"\nWrote {outpath}")
