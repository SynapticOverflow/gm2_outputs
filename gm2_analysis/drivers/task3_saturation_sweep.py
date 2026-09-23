"""
task3_saturation_sweep.py

Task 3: continuous log-scale saturation sweep for k_T4_entry over
logspace(-18, -6, 25), N=50 stochastic realizations per point.

UNITS-REGIME NOTE (read before trusting one file over the other):
The task's own framing -- "Currently we only have 3 discrete measured
points (baseline ~2e-8, 1e-12, 1e-16)" -- names a baseline (~2e-8) that
is gm2_solver.py's RAW-UNITS k_T4_entry value, not gm2_model_v2.py's
renormalized baseline (0.06, "BUG 2" fix -- see gm2_model_v2.py's header).
The requested sweep range logspace(-18,-6) brackets 2e-8 naturally; it
does NOT bracket gm2_model_v2's 0.06 baseline (1e-6 is still 5 orders of
magnitude below it). Rather than silently pick one interpretation for a
choice this consequential, this script runs BOTH, clearly labeled:

  data/saturation_sweep.json                    -- PRIMARY. Against
      gm2_solver.py (raw-units, unmodified -- same file/functions used
      for the task-4 diagnostic), matching the "~2e-8 baseline" framing
      literally. This is also a real, rigorous version of the informal
      check gm2_model_v2.py's own header claims ("saturation held from
      k_T4_entry=1e-2 down to 1e-14; it only broke below ~1e-15") --
      worth actually verifying at N=50/point instead of taking on faith.
  data/saturation_sweep_gm2_model_v2.json        -- SECONDARY. Same exact
      k-value grid, against the current/fixed gm2_model_v2 (dt=0.2,
      t4_dose_total=1.0, other params at make_base_params('tay-sachs')
      defaults) for direct contrast -- this is expected to show ~zero
      effect at every point (confirmed by a pre-run smoke test), since
      the whole swept range sits below gm2_model_v2's calibrated scale.
      That "flat/no effect" result IS the answer for this file, not a bug.

Only saturation_sweep.json matches the literal filename the task
specified; it is the one to treat as "the" answer for task 3. The second
file is provided so the units ambiguity is falsifiable/inspectable rather
than silently resolved one way.
"""
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from parallel_eval import evaluate_parallel, n_workers_from_env

K_VALUES = np.logspace(-18, -6, 25).tolist()
N_REALIZATIONS = 50
DT = 0.2
TMAX_DAYS = 365.0
SEED_BASE = 0

_HERE = os.path.dirname(os.path.abspath(__file__))


def _sha256_16(fname):
    with open(os.path.join(_HERE, fname), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _eval_row_solver(row):
    """gm2_solver.py, raw-units, unmodified -- mirrors the task-4 diagnostic
    (dose_mg=0, no SRT; AAV admin at its own default day/particle count;
    no FUS; only k_T4_entry swept)."""
    from gm2_solver import make_base_params, make_x0, simulate_milstein_full
    k_val, seed = row
    base = make_base_params("tay-sachs")
    base["k_T4_entry"] = k_val
    x0 = make_x0(base)
    df = simulate_milstein_full(
        x0, TMAX_DAYS, DT, dose_mg=0.0, disease="tay-sachs",
        t4_admin_day=29.9, t4_admin_particles=5e12, fus_events=None,
        seed=seed, params_override={"k_T4_entry": k_val},
    )
    return float(df["gm2_brain"].iloc[-1])


def _eval_row_v2(row):
    from gm2_model_v2 import run_single
    k_val, seed = row
    return run_single({"k_T4_entry": k_val}, dt=DT, tmax_days=TMAX_DAYS, seed=seed, t4_dose_total=1.0)


def run_sweep(eval_fn, label, extra_meta):
    rows = [(k, SEED_BASE + s) for k in K_VALUES for s in range(N_REALIZATIONS)]
    run_meta = {
        "task": "task3_saturation_sweep", "variant": label,
        "k_values_logspace": [-18, -6, 25], "n_points": len(K_VALUES),
        "n_realizations": N_REALIZATIONS, "dt": DT, "tmax_days": TMAX_DAYS,
        "seed_scheme": f"seed = {SEED_BASE} + realization_index (0..{N_REALIZATIONS-1}), "
                        "same across k-points (common random numbers)",
        "n_workers": n_workers_from_env(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **extra_meta,
    }
    print(json.dumps(run_meta, indent=2))
    t0 = time.time()
    Y = evaluate_parallel(eval_fn, rows, label=f"  [{label}] ")
    Y = Y.reshape(len(K_VALUES), N_REALIZATIONS)
    mean_burden = Y.mean(axis=1)
    std_burden = Y.std(axis=1, ddof=1)
    elapsed = time.time() - t0
    out = {
        "run_meta": dict(run_meta, elapsed_s=elapsed),
        "k_values": K_VALUES,
        "mean_burden": mean_burden.tolist(),
        "std_burden": std_burden.tolist(),
        "n_realizations": N_REALIZATIONS,
    }
    print(f"\n=== [{label}] summary ({elapsed:.0f}s, {len(rows)} evals) ===")
    print(f"{'k_T4_entry':>14s} {'mean_burden':>12s} {'std_burden':>11s}")
    for k, m, s in zip(K_VALUES, mean_burden, std_burden):
        print(f"{k:14.3e} {m:12.4f} {s:11.4f}")
    return out


if __name__ == "__main__":
    out_solver = run_sweep(
        _eval_row_solver, "gm2_solver_raw_units",
        {"model_file": "gm2_solver.py", "model_sha256_16": _sha256_16("gm2_solver.py"),
         "other_params": "gm2_solver.make_base_params('tay-sachs') defaults; dose_mg=0 (no SRT); "
                          "AAV admin t4_admin_day=29.9, t4_admin_particles=5e12; fus_events=None"},
    )
    with open(os.path.join(_HERE, "data/saturation_sweep.json"), "w") as f:
        json.dump(out_solver, f, indent=2)
    print(f"\nWritten to data/saturation_sweep.json (PRIMARY)")

    out_v2 = run_sweep(
        _eval_row_v2, "gm2_model_v2_normalized",
        {"model_file": "gm2_model_v2.py", "model_sha256_16": _sha256_16("gm2_model_v2.py"),
         "other_params": "gm2_model_v2.make_base_params('tay-sachs') defaults, t4_dose_total=1.0"},
    )
    with open(os.path.join(_HERE, "data/saturation_sweep_gm2_model_v2.json"), "w") as f:
        json.dump(out_v2, f, indent=2)
    print(f"\nWritten to data/saturation_sweep_gm2_model_v2.json (SECONDARY, for contrast)")
    print("Done.")
