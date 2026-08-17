"""
generate_timing_data.py

Generates the training dataset for the LIFU-timing ML model (see
train_timing_model.py). The parameter sweep in run_sweep.py only varied
acoustic parameters (f, P, DC) against a *fixed* sonication schedule
(sessions at day 0/3/7/14/30). It therefore contains no information about
how session *timing* affects outcome -- that is a completely separate
question this script answers by sampling the timing space directly.

Timing parameterisation
------------------------
"When to administer LIFU" is reduced to a 3-parameter policy (see
sde_lifu.build_schedule):
    t_first_hr   : time of the first session relative to the AAV-T4 bolus
    n_sessions   : number of sessions (1-5)
    spacing_hr   : spacing between consecutive sessions
so the full input space for one training example is:
    [depth_mm, f_mhz, P_mpa, DC, t_first_hr, n_sessions, spacing_hr]
and the target is the day-365 GM2 percent reduction vs. natural history.

Sampling: Latin Hypercube (scipy.stats.qmc) over the 6 continuous
dimensions, with n_sessions sampled as an integer 1-5 from the same LHS
draw (7th dimension, floored). LHS gives much more even coverage of the
7-D input space than uniform random sampling for a fixed sample budget.

Each sampled point is evaluated with N_REALISATIONS SDE realisations
(averaged) rather than the 300 used in run_sweep.py, since we need many
more *points* rather than tight per-point noise estimates -- this is a
standard tradeoff for building a regression training set from an
expensive simulator.
"""

import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.stats import qmc

from lifu_acoustic import compute_eta
from sde_lifu import simulate_trajectory, build_schedule, T_TOTAL

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(OUT_DIR, "timing_training_data.csv")

N_SAMPLES = 3000
N_REALISATIONS = 10
N_BASELINE_REALISATIONS = 50

# Input parameter bounds (order matters -- matches the LHS draw below)
BOUNDS = dict(
    depth_mm=(20.0, 60.0),      # slightly wider than the 8 named regions (25-55mm)
    f_mhz=(0.4, 1.1),
    P_mpa=(0.15, 0.50),
    DC=(0.003, 0.025),
    t_first_hr=(0.0, 96.0),     # first session up to 4 days after AAV dosing
    spacing_hr=(6.0, 96.0),     # 6 hr (back-to-back-ish) to 4 days between sessions
    n_sessions=(1.0, 5.999),    # floored to int in {1,2,3,4,5}
)
PARAM_ORDER = ["depth_mm", "f_mhz", "P_mpa", "DC", "t_first_hr", "spacing_hr", "n_sessions"]


def sample_inputs(n_samples, seed=0):
    """Latin Hypercube sample of the 7-D input space, scaled to BOUNDS."""
    sampler = qmc.LatinHypercube(d=len(PARAM_ORDER), seed=seed)
    unit = sampler.random(n=n_samples)
    lo = np.array([BOUNDS[p][0] for p in PARAM_ORDER])
    hi = np.array([BOUNDS[p][1] for p in PARAM_ORDER])
    scaled = qmc.scale(unit, lo, hi)
    rows = []
    for row in scaled:
        d = dict(zip(PARAM_ORDER, row))
        d["n_sessions"] = int(np.floor(d["n_sessions"]))
        rows.append(d)
    return rows


def _worker(args):
    depth_mm, f_mhz, P_mpa, DC, t_first_hr, spacing_hr, n_sessions, seed = args
    sched = build_schedule(t_first_hr, n_sessions, spacing_hr)
    sched = sched[sched < T_TOTAL]  # safety; bounds already keep this well inside T
    traj = simulate_trajectory(
        "lifu_tri", depth_mm=depth_mm, f_mhz=f_mhz, P_mpa=P_mpa, DC=DC,
        seed=seed, sonication_times=sched,
    )
    return traj[-1]


def _baseline_worker(seed):
    traj = simulate_trajectory("natural", depth_mm=30.0, seed=seed)
    return traj[-1]


def main():
    n_workers = cpu_count()
    print(f"Using {n_workers} workers. {N_SAMPLES} samples x {N_REALISATIONS} "
          f"realisations = {N_SAMPLES * N_REALISATIONS} trajectories, plus "
          f"{N_BASELINE_REALISATIONS} baseline realisations.", flush=True)

    with Pool(processes=n_workers) as pool:
        t0 = time.time()
        baseline_vals = pool.map(_baseline_worker, range(N_BASELINE_REALISATIONS))
        natural_mean = float(np.mean(baseline_vals))
        print(f"Natural-history baseline (depth-independent): "
              f"mean day-365 GM2 = {natural_mean:.2f} nmol/g "
              f"({time.time()-t0:.1f}s)", flush=True)

        samples = sample_inputs(N_SAMPLES, seed=42)

        tasks = []
        for i, s in enumerate(samples):
            for r in range(N_REALISATIONS):
                tasks.append((s["depth_mm"], s["f_mhz"], s["P_mpa"], s["DC"],
                              s["t_first_hr"], s["spacing_hr"], s["n_sessions"],
                              i * N_REALISATIONS + r))

        t0 = time.time()
        results = pool.map(_worker, tasks, chunksize=8)
        print(f"Simulated {len(tasks)} trajectories in {time.time()-t0:.1f}s", flush=True)

    results = np.array(results, dtype=np.float64).reshape(N_SAMPLES, N_REALISATIONS)
    gm2_mean = results.mean(axis=1)
    gm2_std = results.std(axis=1)
    reduction_pct = (natural_mean - gm2_mean) / natural_mean * 100.0

    with open(OUT_CSV, "w") as fh:
        fh.write("depth_mm,f_mhz,P_mpa,DC,eta,t_first_hr,n_sessions,spacing_hr,"
                  "mean_gm2_day365,std_gm2_day365,reduction_pct\n")
        for s, m, sd, red in zip(samples, gm2_mean, gm2_std, reduction_pct):
            eta = compute_eta(s["f_mhz"], s["P_mpa"], s["DC"], s["depth_mm"])
            fh.write(f"{s['depth_mm']:.4f},{s['f_mhz']:.4f},{s['P_mpa']:.4f},"
                      f"{s['DC']:.5f},{eta:.5f},{s['t_first_hr']:.3f},"
                      f"{s['n_sessions']},{s['spacing_hr']:.3f},"
                      f"{m:.4f},{sd:.4f},{red:.4f}\n")

    print(f"Wrote {OUT_CSV} ({N_SAMPLES} rows)")


if __name__ == "__main__":
    main()
