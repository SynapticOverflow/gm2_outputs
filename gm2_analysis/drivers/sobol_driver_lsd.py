# sobol_driver_lsd.py
#
# Cross-disease Sobol analysis: for each of the 9 LSDs, evaluate at a
# disease-appropriate timepoint (fixing the bug where all 9 were
# evaluated at a fixed day 365 regardless of their real natural-history
# timescale -- for slow diseases like Fabry that trivially saturates
# clearance near 100% and makes the sensitivity analysis meaningless).
#
# Fix: eval_day = 0.5 * t_ref_days, matching the existing GM2/Tay-Sachs
# anchor (365 days = 0.5 * 730-day reference survival). This keeps every
# disease's evaluation at the SAME relative point in its own natural
# history, instead of the same absolute day.
#
# Single scenario (as_documented prior widths), first+total order only
# (no second-order interactions, to keep 9-disease compute tractable on
# the login node without Slurm). This directly answers: does the
# enzyme-kinetics-dominates-BBB-entry pattern found for GM2 generalize
# across the other 8 LSDs, once each is evaluated on its own timescale?

import json
import time
import numpy as np
from SALib.sample.sobol import sample as sobol_sample
from SALib.analyze.sobol import analyze as sobol_analyze

from gm2_model_v2 import make_base_params, run_single
from lsd_disease_params import make_lsd_params, DISEASE_FACTS

PARAM_NAMES = [
    'k_T4_entry', 't4_dose_total', 'fus_entry_gain_scale', 'gm2_synth',
    'k_T4_payload', 'km_brain', 'vmax_brain', 'k_T4_decay', 'IC50',
    'inf_threshold', 'k_inf', 'k_res', 'rho_g', 'rho_i',
]
TIER2 = {'k_T4_entry', 't4_dose_total', 'inf_threshold', 'rho_g', 'rho_i'}
TIER1_HW, TIER2_HW = 0.25, 0.60

DISEASES = ['tay-sachs', 'sandhoff', 'pompe', 'krabbe_infantile', 'gm1_infantile',
            'mps1_hurler', 'mld_late_infantile', 'cln2', 'niemann_pick_c', 'fabry']


def get_disease_baseline(disease: str) -> dict:
    if disease in ('tay-sachs', 'sandhoff'):
        return make_base_params(disease)
    return make_lsd_params(disease, make_base_params)


def make_bounds(baseline: dict):
    bounds = []
    for name in PARAM_NAMES:
        base_val = baseline['t4_dose_total'] if name == 't4_dose_total' else baseline[name]
        hw = TIER2_HW if name in TIER2 else TIER1_HW
        bounds.append([base_val * (1.0 - hw), base_val * (1.0 + hw)])
    return bounds


def eval_day_for(disease: str) -> float:
    if disease in ('tay-sachs', 'sandhoff'):
        t_ref = DISEASE_FACTS[disease]['t_ref_days']
    else:
        t_ref = DISEASE_FACTS[disease]['t_ref_days']
    return max(60.0, 0.5 * t_ref)  # same relative anchor as GM2 (365 = 0.5*730)


def evaluate_batch(X: np.ndarray, disease: str, tmax_days: float, seed: int = 42, dt: float = 0.2) -> np.ndarray:
    Y = np.empty(X.shape[0])
    for i, row in enumerate(X):
        overrides = dict(zip(PARAM_NAMES, row))
        dose_total = overrides.pop('t4_dose_total')
        # baseline_gm2_brain/liver etc. must come from the disease's own
        # scaled baseline, not the tay-sachs default inside run_single
        base = get_disease_baseline(disease)
        base.update(overrides)
        overrides_full = {k: v for k, v in base.items() if k in make_base_params('tay-sachs')}
        Y[i] = run_single(overrides_full, dt=dt, tmax_days=tmax_days, seed=seed, t4_dose_total=dose_total,
                           t4_admin_day=min(30.0, 0.1 * tmax_days))
    return Y


def run_disease(disease: str, N: int, dt: float = 0.2, seed: int = 42):
    baseline = get_disease_baseline(disease)
    baseline['t4_dose_total'] = 1.0
    tmax = eval_day_for(disease)
    problem = {'num_vars': len(PARAM_NAMES), 'names': PARAM_NAMES, 'bounds': make_bounds(baseline)}
    X = sobol_sample(problem, N=N, calc_second_order=False)
    t0 = time.time()
    Y = evaluate_batch(X, disease, tmax, seed=seed, dt=dt)
    elapsed = time.time() - t0
    Si = sobol_analyze(problem, Y, calc_second_order=False, print_to_console=False)
    print(f"[{disease}] tmax={tmax:.0f}d, {X.shape[0]} evals, {elapsed:.0f}s, Y_mean={Y.mean():.2f} Y_std={Y.std():.2f}", flush=True)
    return {
        'disease': disease, 'tmax_days': tmax, 'n_evals': int(X.shape[0]),
        'names': PARAM_NAMES, 'S1': Si['S1'].tolist(), 'S1_conf': Si['S1_conf'].tolist(),
        'ST': Si['ST'].tolist(), 'ST_conf': Si['ST_conf'].tolist(),
        'Y_mean': float(Y.mean()), 'Y_std': float(Y.std()),
        'Y_raw': Y.tolist(),
    }


if __name__ == "__main__":
    import sys
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    dt = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
    diseases = sys.argv[3].split(',') if len(sys.argv) > 3 else DISEASES

    for disease in diseases:
        result = run_disease(disease, N=N, dt=dt)
        with open(f'data/sobol_lsd_{disease}.json', 'w') as f:
            json.dump(result, f, indent=2)
    print("Done.")
