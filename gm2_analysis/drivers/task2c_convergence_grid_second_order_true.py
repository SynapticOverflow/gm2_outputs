"""
task2c_convergence_grid_second_order_true.py

Follow-up to task2_convergence_grid.py: identical convergence protocol
(same parameter k_T4_entry, same N grid, same as_documented 14-parameter
priors via sobol_driver_v2.PARAM_NAMES/make_bounds, same B=1000 bootstrap
resamples via SALib's keep_resamples=True machinery, same seed/dt), but
with calc_second_order=True throughout (sample AND analyze), matching the
actual production GM2 three-way scenario runs (gm2_full14_*.json, task1_full14.py)
which all used calc_second_order=True (k=30N evals) rather than the
calc_second_order=False (k=16N) scheme the original convergence_grid.json
used. This validates convergence under the sampling scheme that was
actually used for the reported results, not a cheaper stand-in that only
happens to share N=256.

Only diff vs. task2_convergence_grid.py: calc_second_order=True passed to
both sobol_sample and sobol_analyze; n_evals per N is therefore 30*N
instead of 16*N; output path/task name changed accordingly. Everything
else (PARAM_NAMES, make_bounds('as_documented'), DT, SEED, NUM_RESAMPLES,
evaluate_parallel, bootstrap SE computation, top5 ranking) reused
unmodified from sobol_driver_v2 / task2_convergence_grid's own logic.
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

import sobol_driver_v2 as sdv2
from gm2_model_v2 import run_single
from parallel_eval import evaluate_parallel, n_workers_from_env

N_GRID = [64, 128, 256, 512, 1024]
DT = 0.2
SEED = 42
NUM_RESAMPLES = 1000
CALC_SECOND_ORDER = True
K_ENTRY_IDX = sdv2.PARAM_NAMES.index("k_T4_entry")

MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gm2_model_v2.py")
with open(MODEL_FILE, "rb") as f:
    MODEL_SHA256 = hashlib.sha256(f.read()).hexdigest()[:16]


def _eval_row(row):
    overrides = dict(zip(sdv2.PARAM_NAMES, row))
    dose_total = overrides.pop("t4_dose_total")
    return run_single(overrides, dt=DT, tmax_days=365.0, seed=SEED, t4_dose_total=dose_total)


def run_at_N(N):
    problem = {
        "num_vars": len(sdv2.PARAM_NAMES),
        "names": sdv2.PARAM_NAMES,
        "bounds": sdv2.make_bounds("as_documented"),
    }
    X = sobol_sample(problem, N=N, calc_second_order=CALC_SECOND_ORDER)
    print(f"[N={N}] -> {X.shape[0]} model evaluations", flush=True)
    Y = evaluate_parallel(_eval_row, list(X), label=f"  N={N} ")
    Si = sobol_analyze(problem, Y, calc_second_order=CALC_SECOND_ORDER, num_resamples=NUM_RESAMPLES,
                        keep_resamples=True, print_to_console=False, seed=SEED)

    st_all = np.asarray(Si["ST"])
    st_point = float(st_all[K_ENTRY_IDX])
    st_resamples = np.asarray(Si["ST_conf_all"])[:, K_ENTRY_IDX]
    boot_se_abs = float(st_resamples.std(ddof=1))
    boot_se_rel = None if st_point == 0.0 else float(boot_se_abs / abs(st_point))

    ranked = sorted(zip(sdv2.PARAM_NAMES, st_all.tolist()), key=lambda t: -t[1])
    top5 = [[name, float(val)] for name, val in ranked[:5]]

    return {
        "N": N, "n_evals": int(X.shape[0]),
        "k_entry_ST": st_point,
        "k_entry_ST_bootstrap_se_abs": boot_se_abs,
        "k_entry_ST_bootstrap_se_rel": boot_se_rel,
        "top5_params": top5,
        "Y_mean": float(Y.mean()), "Y_std": float(Y.std()),
    }


if __name__ == "__main__":
    run_meta = {
        "task": "task2c_convergence_grid_second_order_true", "N_grid": N_GRID, "dt": DT, "seed": SEED,
        "num_resamples": NUM_RESAMPLES, "calc_second_order": CALC_SECOND_ORDER,
        "n_params": len(sdv2.PARAM_NAMES), "param_names": sdv2.PARAM_NAMES,
        "prior_scenario": "as_documented", "n_workers": n_workers_from_env(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_file": "gm2_model_v2.py", "model_sha256_16": MODEL_SHA256,
        "note": ("Identical protocol to task2_convergence_grid.py / data/convergence_grid.json "
                 "(same param, N grid, B=1000 bootstrap resamples, seed, dt, as_documented priors) "
                 "except calc_second_order=True throughout (sample+analyze), matching the actual "
                 "production gm2_full14_*.json / task1_full14.py sampling scheme (k=30N) rather than "
                 "the calc_second_order=False k=16N scheme the original convergence_grid.json used."),
    }
    print(json.dumps(run_meta, indent=2))

    t_all = time.time()
    per_N = []
    for N in N_GRID:
        t0 = time.time()
        row = run_at_N(N)
        row["elapsed_s"] = time.time() - t0
        per_N.append(row)

    out = {
        "run_meta": run_meta,
        "N": [r["N"] for r in per_N],
        "n_evals": [r["n_evals"] for r in per_N],
        "k_entry_ST": [r["k_entry_ST"] for r in per_N],
        "top5_params": [r["top5_params"] for r in per_N],
        "bootstrap_se": [r["k_entry_ST_bootstrap_se_abs"] for r in per_N],
        "bootstrap_se_relative": [r["k_entry_ST_bootstrap_se_rel"] for r in per_N],
        "Y_mean": [r["Y_mean"] for r in per_N],
        "Y_std": [r["Y_std"] for r in per_N],
        "elapsed_s": [r["elapsed_s"] for r in per_N],
        "total_elapsed_s": time.time() - t_all,
    }
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/convergence_grid_second_order_true.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== Task 2c summary (total {out['total_elapsed_s']:.0f}s) ===")
    print(f"{'N':>6s} {'n_evals':>8s} {'ST(k_entry)':>12s} {'SE_abs':>10s} {'SE_rel':>10s}  top-5")
    for r in per_N:
        rel = r["k_entry_ST_bootstrap_se_rel"]
        rel_str = f"{rel:10.3f}" if rel is not None else "      None"
        top5_str = ", ".join(f"{n}={v:.3f}" for n, v in r["top5_params"])
        print(f"{r['N']:6d} {r['n_evals']:8d} {r['k_entry_ST']:12.4f} "
              f"{r['k_entry_ST_bootstrap_se_abs']:10.4f} {rel_str}  {top5_str}")
    print(f"\nWritten to {outpath}")
    print("Done.")
