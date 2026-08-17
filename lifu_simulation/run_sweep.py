"""
run_sweep.py

Full LIFU parameter sweep + control arms for the tri-modal GM2 gangliosidosis
model. Designed to run on TACC Vista as a single SLURM batch job (see
submit.sh), parallelised across all available cores with multiprocessing.Pool.

Sweep:
    frequency f  in {0.5, 0.65, 1.0} MHz
    pressure  P  in {0.20, 0.35, 0.45} MPa
    duty cycle DC in {0.5%, 1%, 2%}
    -> 27 combinations x 8 regions x 300 realisations = 64,800 trajectories

Controls (no LIFU):
    natural history, mono (AAV only), bi (AAV+SRT), tri_no_lifu (AAV+SRT+generic FUS)
    -> 4 arms x 8 regions x 300 realisations = 9,600 trajectories

Each worker task simulates exactly one (combination-or-arm, region, seed)
tuple and returns the GM2-burden trajectory x[5] (17,521 timesteps).
Results are grouped and written one compressed .npz file per LIFU
combination / per control arm, to keep peak memory bounded to a single
combination's worth of trajectories (~170 MB) at a time.

Output directory: ./results/
    results_{f}MHz_{P}MPa_{DC}pct.npz   (27 files)
    results_control_{arm}.npz            (4 files)
"""

import itertools
import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np

from lifu_acoustic import REGION_NAMES, compute_eta_per_region
from sde_lifu import simulate_trajectory, N_POINTS

N_REALISATIONS = 300
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

FREQUENCIES_MHZ = [0.5, 0.65, 1.0]
PRESSURES_MPA = [0.20, 0.35, 0.45]
DUTY_CYCLES = [0.005, 0.01, 0.02]

CONTROL_ARMS = ["natural", "mono", "bi", "tri_no_lifu"]

LIFU_COMBOS = list(itertools.product(FREQUENCIES_MHZ, PRESSURES_MPA, DUTY_CYCLES))
assert len(LIFU_COMBOS) == 27


def _worker_lifu(args):
    f_mhz, P_mpa, DC, region_idx, seed = args
    region = REGION_NAMES[region_idx]
    traj = simulate_trajectory("lifu_tri", region, f_mhz=f_mhz, P_mpa=P_mpa,
                                DC=DC, seed=seed)
    return traj


def _worker_control(args):
    arm, region_idx, seed = args
    region = REGION_NAMES[region_idx]
    traj = simulate_trajectory(arm, region, seed=seed)
    return traj


def _combo_filename(f_mhz, P_mpa, DC):
    return f"results_{f_mhz:.2f}MHz_{P_mpa:.2f}MPa_{DC*100:.1f}pct.npz"


def _run_lifu_combo(pool, f_mhz, P_mpa, DC, chunksize):
    n_regions = len(REGION_NAMES)
    tasks = [
        (f_mhz, P_mpa, DC, region_idx, seed)
        for region_idx in range(n_regions)
        for seed in range(N_REALISATIONS)
    ]
    t0 = time.time()
    results = pool.map(_worker_lifu, tasks, chunksize=chunksize)
    elapsed = time.time() - t0

    # results is ordered region-major, seed-minor -> reshape to (region, seed, time)
    arr = np.array(results, dtype=np.float32).reshape(n_regions, N_REALISATIONS, N_POINTS)
    trajectories = np.transpose(arr, (1, 0, 2))  # (realisation, region, time)
    gm2_day365 = trajectories[:, :, -1]

    eta_per_region = compute_eta_per_region(f_mhz, P_mpa, DC, REGION_NAMES)
    params = dict(f_mhz=f_mhz, P_mpa=P_mpa, DC=DC, region_names=REGION_NAMES)

    out_path = os.path.join(OUTPUT_DIR, _combo_filename(f_mhz, P_mpa, DC))
    np.savez_compressed(
        out_path,
        trajectories=trajectories,
        gm2_day365=gm2_day365,
        eta_per_region=eta_per_region,
        params=np.array(params, dtype=object),
    )
    print(f"[lifu]  f={f_mhz:.2f}MHz P={P_mpa:.2f}MPa DC={DC*100:.1f}%  "
          f"-> {out_path}  ({elapsed:.1f}s, {len(tasks)} trajectories)", flush=True)


def _run_control_arm(pool, arm, chunksize):
    n_regions = len(REGION_NAMES)
    tasks = [
        (arm, region_idx, seed)
        for region_idx in range(n_regions)
        for seed in range(N_REALISATIONS)
    ]
    t0 = time.time()
    results = pool.map(_worker_control, tasks, chunksize=chunksize)
    elapsed = time.time() - t0

    arr = np.array(results, dtype=np.float32).reshape(n_regions, N_REALISATIONS, N_POINTS)
    trajectories = np.transpose(arr, (1, 0, 2))
    gm2_day365 = trajectories[:, :, -1]

    params = dict(arm=arm, region_names=REGION_NAMES)
    out_path = os.path.join(OUTPUT_DIR, f"results_control_{arm}.npz")
    np.savez_compressed(
        out_path,
        trajectories=trajectories,
        gm2_day365=gm2_day365,
        eta_per_region=np.zeros(n_regions),
        params=np.array(params, dtype=object),
    )
    print(f"[ctrl]  arm={arm:12s} -> {out_path}  ({elapsed:.1f}s, {len(tasks)} trajectories)",
          flush=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n_workers = cpu_count()
    print(f"Using {n_workers} worker processes.", flush=True)
    print(f"Sweep: {len(LIFU_COMBOS)} LIFU combos x {len(REGION_NAMES)} regions x "
          f"{N_REALISATIONS} realisations = "
          f"{len(LIFU_COMBOS) * len(REGION_NAMES) * N_REALISATIONS} trajectories", flush=True)
    print(f"Controls: {len(CONTROL_ARMS)} arms x {len(REGION_NAMES)} regions x "
          f"{N_REALISATIONS} realisations = "
          f"{len(CONTROL_ARMS) * len(REGION_NAMES) * N_REALISATIONS} trajectories", flush=True)

    chunksize = 4
    t_start = time.time()
    with Pool(processes=n_workers) as pool:
        for f_mhz, P_mpa, DC in LIFU_COMBOS:
            _run_lifu_combo(pool, f_mhz, P_mpa, DC, chunksize)
        for arm in CONTROL_ARMS:
            _run_control_arm(pool, arm, chunksize)

    print(f"Sweep complete in {time.time() - t_start:.1f}s. Results in {OUTPUT_DIR}",
          flush=True)


if __name__ == "__main__":
    main()
