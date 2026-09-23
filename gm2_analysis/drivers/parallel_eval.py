"""
parallel_eval.py

Shared multiprocessing wrapper for the embarrassingly-parallel per-row
model-evaluation loop that sobol_driver.py / sobol_driver_v2.py /
sobol_driver_lsd.py / factorial_arms_shapley.py all implement as a serial
Python for-loop over gm2_model_v2.run_single() calls. Each row/seed is an
independent SDE realization (own RNG seed), so this changes nothing about
the model, the Sobol sampling, or the Shapley game -- it only distributes
the same, unmodified run_single() calls across worker processes instead
of running them one at a time. Needed because several of the requested
analyses (full 14-param N=256 x3 scenarios, 5-point convergence grid,
10-disease x2-scenario cross-disease run) are serially 30-100+ CPU-hours
at this model's ~0.2-0.3s/eval cost (worse for long-tmax diseases like
Fabry at t_ref=8212.5 days), which is not a reasonable interactive or
even single-job turnaround.

Usage: pass a worker function `f(row) -> float` (must be a module-level
function so it's picklable) and the array of rows; get back Y in the same
order.
"""
import os
import time
import multiprocessing as mp

import numpy as np


def n_workers_from_env(default=64):
    return int(os.environ.get("SLURM_CPUS_ON_NODE", os.environ.get("SLURM_CPUS_PER_TASK", default)))


def evaluate_parallel(worker_fn, rows, n_workers=None, chunksize=4, progress_every=500, label=""):
    """rows: sequence of arbitrary picklable args, one per model eval.
    worker_fn(row) -> float. Returns np.ndarray of results in input order."""
    if n_workers is None:
        n_workers = n_workers_from_env()
    n_workers = max(1, min(n_workers, len(rows)))
    t0 = time.time()
    Y = np.empty(len(rows))
    with mp.Pool(n_workers) as pool:
        n_done = 0
        for i, y in enumerate(pool.imap(worker_fn, rows, chunksize=chunksize)):
            Y[i] = y
            n_done += 1
            if progress_every and n_done % progress_every == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                eta = (len(rows) - n_done) / rate if rate > 0 else float("nan")
                print(f"  {label}{n_done}/{len(rows)} evals, {elapsed:.0f}s elapsed, "
                      f"{n_workers} workers, ETA {eta:.0f}s", flush=True)
    elapsed = time.time() - t0
    print(f"  {label}done: {len(rows)} evals in {elapsed:.0f}s with {n_workers} workers "
          f"({elapsed/len(rows)*n_workers:.3f}s/eval effective serial cost)", flush=True)
    return Y
