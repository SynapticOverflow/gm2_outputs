"""
task1_full14.py

Task 1 (robustness audit follow-up): re-run the GM2 three-way prior-width
comparison (as_documented / equal_width / tier1_promotion) over the FULL
14-parameter list, at N=256.

Reuses sobol_driver_v2.py's PARAM_NAMES, make_bounds(), and run_scenario()
UNCHANGED (same problem construction, same SALib sobol sample/analyze
calls, same output schema) -- monkeypatches only its serial evaluate_batch
for a multiprocessing-parallel one calling the exact same per-row
gm2_model_v2.run_single() logic, since a serial N=256/14-param/
calc_second_order=True run is ~23,040 model evals (~91 CPU-minutes serial;
tractable in parallel on a shared node, not worth reimplementing anything
for).

NOTE on "previously run over only 7 parameters": the on-disk
data/sobol_v2_*.json files (presumably what that referred to) actually
already contain all 14 parameters (see 'names' field), just at N=64
(n_evals=1920 = 64*(2*14+2)), not 7 parameters at any N. No 7-parameter
version of this analysis was found anywhere in this project directory.
Reported as a factual correction, not silently assumed -- see the run
summary this script prints.
"""
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sobol_driver_v2 as sdv2
from gm2_model_v2 import run_single
from parallel_eval import evaluate_parallel, n_workers_from_env

N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
DT = 0.2
SEED = 42
SCENARIOS = ["as_documented", "equal_width", "tier1_promotion"]

MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gm2_model_v2.py")
with open(MODEL_FILE, "rb") as f:
    MODEL_SHA256 = hashlib.sha256(f.read()).hexdigest()[:16]


def _eval_row(row):
    overrides = dict(zip(sdv2.PARAM_NAMES, row))
    dose_total = overrides.pop("t4_dose_total")
    return run_single(overrides, dt=DT, tmax_days=365.0, seed=SEED, t4_dose_total=dose_total)


def parallel_evaluate_batch(X, seed=SEED, dt=DT):
    assert seed == SEED and dt == DT, "monkeypatched evaluate_batch is wired for fixed SEED/DT"
    return evaluate_parallel(_eval_row, list(X), label="  ")


if __name__ == "__main__":
    sdv2.evaluate_batch = parallel_evaluate_batch  # reuse run_scenario() unmodified otherwise

    run_meta = {
        "task": "task1_full14",
        "N": N, "dt": DT, "seed": SEED,
        "n_params": len(sdv2.PARAM_NAMES),
        "param_names": sdv2.PARAM_NAMES,
        "calc_second_order": True,
        "n_workers": n_workers_from_env(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_file": "gm2_model_v2.py",
        "model_sha256_16": MODEL_SHA256,
        "note_on_prior_scope": (
            "On-disk data/sobol_v2_*.json (the presumed prior run) already contains "
            "all 14 parameters at N=64 (n_evals=1920=64*30), not a 7-parameter run at "
            "any N -- no 7-parameter version of this analysis was found anywhere in "
            "this project directory (grep across all .py/.json/.log files, no git "
            "history exists). This run is N=64 -> N=256 at the existing 14-parameter "
            "list, not 7 -> 14 params."
        ),
    }
    print(json.dumps(run_meta, indent=2))

    summary_rows = []
    t_all = time.time()
    for scenario in SCENARIOS:
        print(f"\n=== Running scenario: {scenario} (N={N}, 14 params) ===", flush=True)
        t0 = time.time()
        result = sdv2.run_scenario(scenario, N=N, dt=DT, seed=SEED)
        result["run_meta"] = dict(run_meta, scenario=scenario, elapsed_s=time.time() - t0)
        outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"data/gm2_full14_{scenario}.json")
        with open(outpath, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Written to {outpath} ({time.time()-t0:.0f}s)")

        ranked = sorted(zip(result["names"], result["ST"], result["ST_conf"]), key=lambda t: -t[1])
        summary_rows.append((scenario, result["n_evals"], result["Y_mean"], result["Y_std"], ranked[:5]))

    print(f"\n=== Task 1 summary (total {time.time()-t_all:.0f}s) ===")
    print(f"{'scenario':18s} {'n_evals':>8s} {'Y_mean':>10s} {'Y_std':>10s}  top-5 ST(param)")
    for scenario, n_evals, y_mean, y_std, top5 in summary_rows:
        top5_str = ", ".join(f"{n}={s:.3f}" for n, s, _ in top5)
        print(f"{scenario:18s} {n_evals:8d} {y_mean:10.3f} {y_std:10.3f}  {top5_str}")
    print("\nDone.")
