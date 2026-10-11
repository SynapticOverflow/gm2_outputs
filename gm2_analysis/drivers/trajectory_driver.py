# trajectory_driver.py
#
# Full raw state trajectories (all 17 states + time) at baseline/nominal
# parameters for each of the 10 LSDs, over that disease's own natural-history
# timescale (DISEASE_FACTS[disease]['t_ref_days']). Companion to
# sobol_driver_lsd.py, which only keeps the scalar gm2_brain endpoint at
# 0.5*t_ref for sensitivity analysis and discards the rest of the trajectory.

import pandas as pd

from gm2_model_v2 import make_base_params, make_x0, simulate_milstein, DEFAULT_FUS_EVENTS, STATE_NAMES
from lsd_disease_params import make_lsd_params, DISEASE_FACTS

DISEASES = ['tay-sachs', 'sandhoff', 'pompe', 'krabbe_infantile', 'gm1_infantile',
            'mps1_hurler', 'mld_late_infantile', 'cln2', 'niemann_pick_c', 'fabry']


def get_disease_baseline(disease: str) -> dict:
    if disease in ('tay-sachs', 'sandhoff'):
        return make_base_params(disease)
    return make_lsd_params(disease, make_base_params)


def run_trajectory(disease: str, dt: float = 0.2, seed: int = 42, t4_dose_total: float = 1.0) -> pd.DataFrame:
    base = get_disease_baseline(disease)
    tmax = DISEASE_FACTS[disease]['t_ref_days']
    t4_admin_day = min(30.0, 0.1 * tmax)
    x0 = make_x0(base)
    df = simulate_milstein(
        x0, tmax, dt, base, t4_admin_day=t4_admin_day, t4_dose_total=t4_dose_total,
        fus_events=DEFAULT_FUS_EVENTS, seed=seed,
    )
    df['disease'] = disease
    return df


if __name__ == "__main__":
    for disease in DISEASES:
        df = run_trajectory(disease)
        out = f'data/trajectories/{disease}_trajectory.csv'
        df.to_csv(out, index=False)
        print(f"[{disease}] tmax={DISEASE_FACTS[disease]['t_ref_days']:.0f}d, "
              f"{len(df)} timesteps, gm2_brain end={df['gm2_brain'].iloc[-1]:.2f}", flush=True)
    print("Done.")
