# sobol_driver_v2.py
#
# Multi-parameter Sobol analysis (first-order S1, total-order ST, AND
# second-order S2 interaction indices) against the rebuilt 17-state model
# in gm2_model_v2.py.
#
# calc_second_order=True is the direct answer to "is this multiparameter,
# and could enzyme+delivery jointly be the bottleneck rather than just
# one parameter": S1/ST alone describe each parameter's own (marginal +
# total) contribution to output variance, but the interaction terms S2_ij
# are what would show up if two parameters only matter TOGETHER (e.g. a
# high entry rate only helps when enzyme capacity is also high, or vice
# versa) even when their individual S1 looks small.
#
# 14 parameters (the 13 from sobol_driver.py, plus t4_dose_total -- the
# AAV dose itself, since the original saturation bug showed dose and
# entry rate are mechanically coupled through the same product term, so
# testing entry rate's sensitivity without also varying dose was
# incomplete).

import json
import time
import numpy as np
from SALib.sample.sobol import sample as sobol_sample
from SALib.analyze.sobol import analyze as sobol_analyze

from gm2_model_v2 import make_base_params, run_single

BASE = make_base_params('tay-sachs')

PARAM_NAMES = [
    'k_T4_entry', 't4_dose_total', 'fus_entry_gain_scale', 'gm2_synth',
    'k_T4_payload', 'km_brain', 'vmax_brain', 'k_T4_decay', 'IC50',
    'inf_threshold', 'k_inf', 'k_res', 'rho_g', 'rho_i',
]

# t4_dose_total isn't in BASE (it's a run_single argument, not a model
# param) -- give it an explicit baseline here.
BASELINES = dict(BASE)
BASELINES['t4_dose_total'] = 1.0

TIER2_AS_DOCUMENTED = {'k_T4_entry', 't4_dose_total', 'inf_threshold', 'rho_g', 'rho_i'}
TIER1_HALF_WIDTH = 0.25
TIER2_HALF_WIDTH = 0.60


def make_bounds(scenario: str):
    bounds = []
    for name in PARAM_NAMES:
        base_val = BASELINES[name]
        if scenario == 'as_documented':
            hw = TIER2_HALF_WIDTH if name in TIER2_AS_DOCUMENTED else TIER1_HALF_WIDTH
        elif scenario == 'equal_width':
            hw = TIER1_HALF_WIDTH
        elif scenario == 'tier1_promotion':
            if name == 'k_T4_entry':
                hw = TIER1_HALF_WIDTH
            elif name in TIER2_AS_DOCUMENTED:
                hw = TIER2_HALF_WIDTH
            else:
                hw = TIER1_HALF_WIDTH
        else:
            raise ValueError(scenario)
        bounds.append([base_val * (1.0 - hw), base_val * (1.0 + hw)])
    return bounds


def evaluate_batch(X: np.ndarray, seed: int = 42, dt: float = 0.2) -> np.ndarray:
    Y = np.empty(X.shape[0])
    t0 = time.time()
    for i, row in enumerate(X):
        overrides = dict(zip(PARAM_NAMES, row))
        dose_total = overrides.pop('t4_dose_total')
        Y[i] = run_single(overrides, dt=dt, tmax_days=365.0, seed=seed, t4_dose_total=dose_total)
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (X.shape[0] - (i + 1)) / rate
            print(f"  {i+1}/{X.shape[0]} evals, {elapsed:.0f}s elapsed, ETA {eta:.0f}s", flush=True)
    return Y


def run_scenario(scenario: str, N: int, dt: float = 0.2, seed: int = 42):
    problem = {
        'num_vars': len(PARAM_NAMES),
        'names': PARAM_NAMES,
        'bounds': make_bounds(scenario),
    }
    X = sobol_sample(problem, N=N, calc_second_order=True)
    print(f"[{scenario}] N={N} -> {X.shape[0]} model evaluations", flush=True)
    Y = evaluate_batch(X, seed=seed, dt=dt)
    Si = sobol_analyze(problem, Y, calc_second_order=True, print_to_console=False)

    n = len(PARAM_NAMES)
    S2 = Si['S2']
    S2_conf = Si['S2_conf']
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if not np.isnan(S2[i, j]):
                pairs.append({
                    'pair': [PARAM_NAMES[i], PARAM_NAMES[j]],
                    'S2': float(S2[i, j]),
                    'S2_conf': float(S2_conf[i, j]),
                })
    pairs.sort(key=lambda p: -abs(p['S2']))

    return {
        'scenario': scenario,
        'names': PARAM_NAMES,
        'S1': Si['S1'].tolist(),
        'S1_conf': Si['S1_conf'].tolist(),
        'ST': Si['ST'].tolist(),
        'ST_conf': Si['ST_conf'].tolist(),
        'S2_pairs': pairs,
        'n_evals': int(X.shape[0]),
        'Y_mean': float(Y.mean()),
        'Y_std': float(Y.std()),
    }


if __name__ == "__main__":
    import sys
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    dt = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
    scenarios = sys.argv[3].split(',') if len(sys.argv) > 3 else ['as_documented', 'equal_width', 'tier1_promotion']

    for scenario in scenarios:
        print(f"\n=== Running scenario: {scenario} ===", flush=True)
        result = run_scenario(scenario, N=N, dt=dt)
        outpath = f'data/sobol_v2_{scenario}.json'
        with open(outpath, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Written to {outpath}")

    print("\nDone.")
