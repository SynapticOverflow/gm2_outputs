"""
task7_percentile_cis.py

Task 1 (follow-up request): recompute Sobol S_T uncertainty as 95%
*percentile* bootstrap intervals (B=1000 resamples) instead of the
mean +/- Z*SD symmetric interval that SALib's `*_conf` fields report by
default. The symmetric interval can imply values outside the valid
[0,1] range for indices near the boundary -- e.g. Niemann-Pick C's
gm2_synth in data/sobol_lsd_niemann_pick_c.json is ST=0.9586 +/-
ST_conf=0.1293, whose symmetric 95% interval [0.829, 1.088] exceeds 1.
Percentile intervals, built directly from the empirical bootstrap
resample distribution, cannot do this.

Two data sources, handled differently:

1. data/sobol_lsd_*.json (10 diseases x 3 scenarios: as_documented has
   no filename suffix, equal_width and tier1_promotion do). These were
   written by sobol_driver_lsd.py, which DOES save raw model outputs
   ('Y_raw' key, confirmed present in every file checked). SALib's
   bootstrap resampling operates on Y alone (it re-indexes the A/B/AB
   sub-arrays derived from Y; the resampling has nothing to do with how
   Y was generated), so we can recover the exact-methodology bootstrap
   resample distribution for ST by re-calling
   SALib.analyze.sobol.analyze(..., keep_resamples=True,
   num_resamples=1000) directly on the archived Y_raw. NO model
   re-evaluation is needed or performed for these 30 entries.

2. data/gm2_full14_*.json (3 scenarios: as_documented / equal_width /
   tier1_promotion). Checked and confirmed: these were written by
   sobol_driver_v2.run_scenario() via task1_full14.py, which only saves
   Y_mean/Y_std, not Y_raw -- there is no archived Y to reanalyze. Per
   the task instructions ("only rerun if they weren't saved"), this
   triggers a fresh Sobol run at the SAME configuration as the original
   task1_full14.py run (N=256, 14 params, calc_second_order=True,
   model-eval seed=42, same gm2_model_v2.run_single + parallel_eval.py
   multiprocessing path) -- the only addition is passing
   keep_resamples=True/num_resamples=1000 to the analyze step so the
   bootstrap resample distribution is available for percentile CIs.
   This is a genuinely new Sobol sample (a new X), not a reconstruction
   of the original run's exact Y -- logged explicitly below and in
   run_meta so it isn't mistaken for a recovered archive.

Does not modify gm2_model_v2.py or any existing data/*.json file.
Output: data/percentile_cis.json, a list of per (disease, scenario)
records: {disease, scenario, names, ST_mean, ST_ci_lo_2.5,
ST_ci_hi_97.5, ST_bootstrap_resamples (full B=1000 draws per param),
n_evals, source ("archived_Y_raw" | "rerun"), run_meta}.
"""
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
from SALib.analyze.sobol import analyze as sobol_analyze
from SALib.sample.sobol import sample as sobol_sample

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sobol_driver_v2 as sdv2
from gm2_model_v2 import run_single
from parallel_eval import evaluate_parallel, n_workers_from_env

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")

MODEL_FILE = os.path.join(ROOT, "gm2_model_v2.py")
with open(MODEL_FILE, "rb") as f:
    MODEL_SHA256 = hashlib.sha256(f.read()).hexdigest()[:16]

B = 1000                 # number of bootstrap resamples requested
BOOT_SEED = 8162026      # fixed seed for the bootstrap resampling itself (reproducible CIs)
CONF_LEVEL = 0.95
RERUN_N = 256             # match task1_full14.py's original N
RERUN_DT = 0.2
RERUN_SEED = 42           # match task1_full14.py's original model-eval seed
RERUN_SAMPLE_SEED = 20260816  # seed for the fresh Sobol X sample (new archive didn't exist to reuse)

DISEASES = ['tay-sachs', 'sandhoff', 'pompe', 'krabbe_infantile', 'gm1_infantile',
            'mps1_hurler', 'mld_late_infantile', 'cln2', 'niemann_pick_c', 'fabry']
LSD_SCENARIO_SUFFIX = {"as_documented": "", "equal_width": "_equal_width", "tier1_promotion": "_tier1_promotion"}
FULL14_SCENARIOS = ["as_documented", "equal_width", "tier1_promotion"]


def percentile_ci_from_Y(names, Y, calc_second_order, seed):
    problem = {"num_vars": len(names), "names": list(names)}
    Y = np.asarray(Y, dtype=float)
    Si = sobol_analyze(problem, Y, calc_second_order=calc_second_order,
                        num_resamples=B, conf_level=CONF_LEVEL,
                        print_to_console=False, keep_resamples=True, seed=seed)
    ST_conf_all = np.asarray(Si["ST_conf_all"])  # (B, D) bootstrap resample distribution of ST
    lo = np.percentile(ST_conf_all, 2.5, axis=0)
    hi = np.percentile(ST_conf_all, 97.5, axis=0)
    return {
        "ST_mean": Si["ST"].tolist(),
        "ST_ci_lo_2.5": lo.tolist(),
        "ST_ci_hi_97.5": hi.tolist(),
        "ST_bootstrap_resamples": ST_conf_all.tolist(),
        "ST_conf_symmetric_original_method": Si["ST_conf"].tolist(),
    }


def process_lsd_archives():
    records = []
    for disease in DISEASES:
        for scenario, suffix in LSD_SCENARIO_SUFFIX.items():
            path = os.path.join(DATA_DIR, f"sobol_lsd_{disease}{suffix}.json")
            if not os.path.exists(path):
                print(f"[MISSING] {path}", flush=True)
                continue
            with open(path) as f:
                d = json.load(f)
            if "Y_raw" not in d:
                print(f"[NO Y_raw, SKIPPED -- should not happen for sobol_lsd_*] {path}", flush=True)
                continue
            names = d["names"]
            Y = d["Y_raw"]
            ci = percentile_ci_from_Y(names, Y, calc_second_order=False, seed=BOOT_SEED)
            # sanity check against archived ST (should match to numerical precision --
            # both computed by the same SALib estimator on the same Y)
            max_abs_diff = float(np.max(np.abs(np.array(ci["ST_mean"]) - np.array(d["ST"]))))
            rec = {
                "disease": disease,
                "scenario": scenario,
                "source": "archived_Y_raw",
                "names": names,
                "n_evals": d["n_evals"],
                "tmax_days": d.get("tmax_days"),
                **ci,
                "sanity_check_max_abs_diff_vs_archived_ST": max_abs_diff,
                "run_meta": {
                    "bootstrap_B": B, "bootstrap_seed": BOOT_SEED, "conf_level": CONF_LEVEL,
                    "source_file": os.path.basename(path),
                    "model_sha256_16": MODEL_SHA256,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                },
            }
            records.append(rec)
            print(f"[lsd] {disease:20s} {scenario:16s} n_evals={d['n_evals']:6d} "
                  f"max|dST|={max_abs_diff:.2e}", flush=True)
    return records


def _eval_row_full14(row):
    overrides = dict(zip(sdv2.PARAM_NAMES, row))
    dose_total = overrides.pop("t4_dose_total")
    return run_single(overrides, dt=RERUN_DT, tmax_days=365.0, seed=RERUN_SEED, t4_dose_total=dose_total)


def process_full14_rerun():
    records = []
    for scenario in FULL14_SCENARIOS:
        problem = {
            "num_vars": len(sdv2.PARAM_NAMES),
            "names": sdv2.PARAM_NAMES,
            "bounds": sdv2.make_bounds(scenario),
        }
        X = sobol_sample(problem, N=RERUN_N, calc_second_order=True, seed=RERUN_SAMPLE_SEED)
        print(f"[full14 rerun] scenario={scenario} N={RERUN_N} -> {X.shape[0]} evals", flush=True)
        t0 = time.time()
        Y = evaluate_parallel(_eval_row_full14, list(X), label=f"  {scenario} ")
        elapsed = time.time() - t0

        ci = percentile_ci_from_Y(sdv2.PARAM_NAMES, Y, calc_second_order=True, seed=BOOT_SEED)

        orig_path = os.path.join(DATA_DIR, f"gm2_full14_{scenario}.json")
        orig = json.load(open(orig_path)) if os.path.exists(orig_path) else None
        max_abs_diff_vs_archive = (
            float(np.max(np.abs(np.array(ci["ST_mean"]) - np.array(orig["ST"]))))
            if orig is not None else None
        )

        rec = {
            "disease": "gm2_tay_sachs_full14",
            "scenario": scenario,
            "source": "rerun",
            "names": sdv2.PARAM_NAMES,
            "n_evals": int(X.shape[0]),
            "Y_mean": float(np.mean(Y)), "Y_std": float(np.std(Y)),
            **ci,
            "note": (
                "Archived gm2_full14_*.json did not save Y_raw, so no exact bootstrap "
                "resamples existed to reuse -- this is a FRESH Sobol rerun at the same "
                "N/params/seed configuration as the original task1_full14.py run, not a "
                "recovery of the original run's exact resample distribution. "
                "ST_mean here may differ slightly from the archived ST value due to a new "
                "Sobol sample (X); max_abs_diff_vs_archived_ST_if_available quantifies that."
            ),
            "max_abs_diff_vs_archived_ST_if_available": max_abs_diff_vs_archive,
            "run_meta": {
                "N": RERUN_N, "D": len(sdv2.PARAM_NAMES), "dt": RERUN_DT,
                "model_eval_seed": RERUN_SEED, "sobol_sample_seed": RERUN_SAMPLE_SEED,
                "bootstrap_B": B, "bootstrap_seed": BOOT_SEED, "conf_level": CONF_LEVEL,
                "calc_second_order": True, "elapsed_s": elapsed,
                "n_workers": n_workers_from_env(),
                "model_sha256_16": MODEL_SHA256,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
        }
        records.append(rec)
        print(f"[full14] {scenario:16s} n_evals={X.shape[0]:6d} elapsed={elapsed:.0f}s "
              f"max|dST vs archive|={max_abs_diff_vs_archive}", flush=True)
    return records


if __name__ == "__main__":
    t_all = time.time()
    lsd_records = process_lsd_archives()
    full14_records = process_full14_rerun()
    all_records = full14_records + lsd_records

    outpath = os.path.join(DATA_DIR, "percentile_cis.json")
    with open(outpath, "w") as f:
        json.dump(all_records, f, indent=2)

    print(f"\nWrote {len(all_records)} records to {outpath} ({time.time()-t_all:.0f}s total)")
    print("\n=== Summary: parameters whose original symmetric CI exceeded [0,1] ===")
    for rec in all_records:
        for name, mean, lo, hi in zip(rec["names"], rec["ST_mean"], rec["ST_ci_lo_2.5"], rec["ST_ci_hi_97.5"]):
            sym_conf = rec.get("ST_conf_symmetric_original_method")
            if sym_conf is not None:
                idx = rec["names"].index(name)
                sym_hi = mean + sym_conf[idx]
                sym_lo = mean - sym_conf[idx]
                if sym_hi > 1.0 or sym_lo < 0.0:
                    print(f"  {rec['disease']:20s} {rec['scenario']:16s} {name:20s} "
                          f"ST={mean:.4f} symmetric=[{sym_lo:.4f},{sym_hi:.4f}] "
                          f"percentile=[{lo:.4f},{hi:.4f}]")
    print("\nDone.")
