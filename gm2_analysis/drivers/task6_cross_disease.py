"""
task6_cross_disease.py

Task 6: cross-disease Sobol analysis, equal_width + tier1_promotion
scenarios, all 10 diseases, matching each disease's existing N and t_max
(the as_documented scenario already exists per-disease in
data/sobol_lsd_<disease>.json -- this fills in the other two legs of the
three-way comparison sobol_driver_v2.py already does for GM2 alone).

Reuses sobol_driver_lsd.py's PARAM_NAMES, TIER2 set, TIER1_HW/TIER2_HW,
get_disease_baseline(), and eval_day_for() UNCHANGED. Its make_bounds()
only implements the as_documented tier logic (no scenario parameter) --
extended here with the exact same 3-scenario branching sobol_driver_v2.py
already uses (as_documented / equal_width / tier1_promotion), just
applied to a per-disease baseline dict instead of the fixed tay-sachs one.
Per-disease N is read back from the existing data/sobol_lsd_<disease>.json
n_evals field (n_evals / 16, since calc_second_order=False and D=14) so
"matching each disease's existing N" is exact, not eyeballed.

evaluate_batch is parallelized (see parallel_eval.py) -- serial cost
for all 10 diseases x 2 scenarios is ~30 CPU-hours (Fabry's 22.5-year
natural-history timescale at dt=0.2 dominates), not workable as a plain
for-loop in one job.
"""
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from SALib.sample.sobol import sample as sobol_sample
from SALib.analyze.sobol import analyze as sobol_analyze

import sobol_driver_lsd as sdl
from gm2_model_v2 import make_base_params, run_single
from parallel_eval import evaluate_parallel, n_workers_from_env

DT = 0.2
SEED = 42
SCENARIOS = ["equal_width", "tier1_promotion"]
_HERE = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(_HERE, "gm2_model_v2.py")
with open(MODEL_FILE, "rb") as f:
    MODEL_SHA256 = hashlib.sha256(f.read()).hexdigest()[:16]

_ACCEPTED_KEYS = set(make_base_params("tay-sachs").keys())


def make_bounds_scenario(baseline: dict, scenario: str):
    """Same 3-scenario branching as sobol_driver_v2.make_bounds(), applied
    to a per-disease baseline dict instead of the fixed tay-sachs one."""
    bounds = []
    for name in sdl.PARAM_NAMES:
        base_val = baseline["t4_dose_total"] if name == "t4_dose_total" else baseline[name]
        if scenario == "as_documented":
            hw = sdl.TIER2_HW if name in sdl.TIER2 else sdl.TIER1_HW
        elif scenario == "equal_width":
            hw = sdl.TIER1_HW
        elif scenario == "tier1_promotion":
            if name == "k_T4_entry":
                hw = sdl.TIER1_HW
            elif name in sdl.TIER2:
                hw = sdl.TIER2_HW
            else:
                hw = sdl.TIER1_HW
        else:
            raise ValueError(scenario)
        bounds.append([base_val * (1.0 - hw), base_val * (1.0 + hw)])
    return bounds


def existing_N_and_tmax(disease):
    path = os.path.join(_HERE, f"data/sobol_lsd_{disease}.json")
    d = json.load(open(path))
    n_evals = d["n_evals"]
    N = n_evals // (len(sdl.PARAM_NAMES) + 2)
    assert N * (len(sdl.PARAM_NAMES) + 2) == n_evals, (disease, n_evals, N)
    return N, float(d["tmax_days"])


def _eval_row(row_disease):
    row, disease = row_disease
    overrides = dict(zip(sdl.PARAM_NAMES, row))
    dose_total = overrides.pop("t4_dose_total")
    base = sdl.get_disease_baseline(disease)
    base.update(overrides)
    overrides_full = {k: v for k, v in base.items() if k in _ACCEPTED_KEYS}
    tmax = sdl.eval_day_for(disease)
    return run_single(overrides_full, dt=DT, tmax_days=tmax, seed=SEED, t4_dose_total=dose_total,
                       t4_admin_day=min(30.0, 0.1 * tmax))


def run_disease_scenario(disease, scenario, N, tmax):
    baseline = sdl.get_disease_baseline(disease)
    baseline["t4_dose_total"] = 1.0
    problem = {"num_vars": len(sdl.PARAM_NAMES), "names": sdl.PARAM_NAMES,
               "bounds": make_bounds_scenario(baseline, scenario)}
    X = sobol_sample(problem, N=N, calc_second_order=False)
    print(f"[{disease}/{scenario}] N={N}, tmax={tmax:.0f}d -> {X.shape[0]} evals", flush=True)
    t0 = time.time()
    rows = [(row, disease) for row in X]
    Y = evaluate_parallel(_eval_row, rows, label=f"  [{disease}/{scenario}] ")
    elapsed = time.time() - t0
    Si = sobol_analyze(problem, Y, calc_second_order=False, print_to_console=False)
    print(f"[{disease}/{scenario}] done in {elapsed:.0f}s, Y_mean={Y.mean():.2f} Y_std={Y.std():.2f}", flush=True)
    return {
        "disease": disease, "scenario": scenario, "tmax_days": tmax, "n_evals": int(X.shape[0]),
        "names": sdl.PARAM_NAMES, "S1": Si["S1"].tolist(), "S1_conf": Si["S1_conf"].tolist(),
        "ST": Si["ST"].tolist(), "ST_conf": Si["ST_conf"].tolist(),
        "Y_mean": float(Y.mean()), "Y_std": float(Y.std()), "Y_raw": Y.tolist(),
        "elapsed_s": elapsed,
    }


if __name__ == "__main__":
    diseases = sys.argv[1].split(",") if len(sys.argv) > 1 else sdl.DISEASES
    scenarios = sys.argv[2].split(",") if len(sys.argv) > 2 else SCENARIOS

    run_meta = {
        "task": "task6_cross_disease", "diseases": diseases, "scenarios": scenarios,
        "dt": DT, "seed": SEED, "n_params": len(sdl.PARAM_NAMES), "param_names": sdl.PARAM_NAMES,
        "n_workers": n_workers_from_env(), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_file": "gm2_model_v2.py", "model_sha256_16": MODEL_SHA256,
    }
    print(json.dumps(run_meta, indent=2))

    t_all = time.time()
    summary = []
    for disease in diseases:
        N, tmax = existing_N_and_tmax(disease)
        for scenario in scenarios:
            result = run_disease_scenario(disease, scenario, N, tmax)
            result["run_meta"] = run_meta
            outpath = os.path.join(_HERE, f"data/sobol_lsd_{disease}_{scenario}.json")
            with open(outpath, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Written to {outpath}")
            ranked = sorted(zip(result["names"], result["ST"]), key=lambda t: -t[1])[:3]
            summary.append((disease, scenario, N, tmax, result["n_evals"], result["elapsed_s"], ranked))

    print(f"\n=== Task 6 summary (total {time.time()-t_all:.0f}s) ===")
    print(f"{'disease':20s} {'scenario':16s} {'N':>6s} {'tmax':>9s} {'n_evals':>8s} {'time_s':>7s}  top-3 ST")
    for disease, scenario, N, tmax, n_evals, elapsed, ranked in summary:
        top3 = ", ".join(f"{n}={v:.3f}" for n, v in ranked)
        print(f"{disease:20s} {scenario:16s} {N:6d} {tmax:9.0f} {n_evals:8d} {elapsed:7.0f}  {top3}")
    print("\nDone.")
