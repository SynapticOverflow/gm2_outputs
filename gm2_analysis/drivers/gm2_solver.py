# gm2_solver.py
# Adapted from SynapticOverflow/comp-fus-aav9-therapy "Simulation Code"
# (gm2_sde_simulator_bayley_iii.py), commit history through 2026-04-10.
# Only change from upstream: dropped the unused tensorflow/keras/sklearn
# imports (the file imports them but never calls into them) so this runs
# on a plain numpy/pandas environment without a GPU/TF install.

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')


def disease_params(disease: str = 'tay-sachs') -> Dict[str, float]:
    d = disease.lower()
    if d in ('sandhoff', 'sandhof', 'sandhoff disease'):
        return {
            'name': 'Sandhoff',
            'baseline_gm2_brain': 1450.0,
            'baseline_gm2_liver': 280.0,
            'residual_hexa': 0.018,
            'residual_hexb': 0.0,
            'gm2_synth': 2.8,
            'gm2_km_brain': 85.0,
            'gm2_km_liver': 65.0,
            'vmax_brain': 12.5,
            'vmax_liver': 18.0,
            'inf_threshold': 800.0,
            'bayley_motor_baseline': 55.0,
            'bayley_cognitive_baseline': 60.0,
            'seizure_baseline': 8.5,
            'respiratory_baseline': 65.0,
            'qol_baseline': 40.0,
        }
    else:
        return {
            'name': 'Tay-Sachs',
            'baseline_gm2_brain': 890.0,
            'baseline_gm2_liver': 156.0,
            'residual_hexa': 0.0,
            'residual_hexb': 0.85,
            'gm2_synth': 2.3,
            'gm2_km_brain': 75.0,
            'gm2_km_liver': 55.0,
            'vmax_brain': 12.5,
            'vmax_liver': 18.0,
            'inf_threshold': 650.0,
            'bayley_motor_baseline': 65.0,
            'bayley_cognitive_baseline': 68.0,
            'seizure_baseline': 3.2,
            'respiratory_baseline': 75.0,
            'qol_baseline': 55.0,
        }


def hill(B: float, IC50: float, Emax: float, n: float) -> float:
    num = Emax * (B ** n)
    den = (IC50 ** n) + (B ** n) + 1e-12
    return num / den


def mm_clearance(S: float, E_eff: float, vmax: float, km: float, ki: Optional[float] = None) -> float:
    if ki is None:
        return (vmax * E_eff * S) / (km + S + 1e-12)
    else:
        return (vmax * E_eff * S) / (km + S + (S ** 2 / ki) + 1e-12)


def dose_pulse_smooth(t_days: float, dose_mg: float, admin_times_per_day: int = 2) -> float:
    frac = t_days - np.floor(t_days)
    pulses = 0.0
    for p in np.linspace(0, 1, admin_times_per_day, endpoint=False):
        pulses += dose_mg * np.exp(-((frac - p) / 0.03) ** 2)
    return pulses


@dataclass
class FUSEvent:
    day: float
    duration_hours: float = 1.0
    gain: float = 1.5
    smooth_tau_hours: float = 0.5

    def multiplier(self, t_day: float) -> float:
        tau = max(1e-6, self.smooth_tau_hours / 24.0)
        width = max(1e-6, self.duration_hours / 24.0)
        return 1.0 + self.gain * np.exp(-0.5 * ((t_day - self.day) / (width / 2.355 + tau)) ** 2)


def fus_multiplier(t_day: float, events: Optional[List[FUSEvent]]) -> float:
    if not events:
        return 1.0
    m = 1.0
    for ev in events:
        m = max(m, ev.multiplier(t_day))
    return m


def make_base_params(disease: str) -> Dict[str, Any]:
    dp = disease_params(disease)
    base = {
        'Vd': 2.1, 'ka': 1.2, 'F': 0.85, 'k_el': 0.035,
        'k_p2b': 0.12, 'k_b2p': 0.015, 'k_brain_elim': 0.15,
        'IC50': 50.0, 'Emax': 0.6, 'hill_n': 1.2,
        'protein_binding': 0.15, 'Kp_uu': 0.12,
        'T4_clear': 0.002, 'k_T4_entry': 2e-8,
        'k_T4_payload': 1 / 25.0, 'k_T4_decay': 1 / 240.0,
        'expr_cap': 8.0,
        'gm2_synth': dp['gm2_synth'], 'vmax_brain': dp['vmax_brain'],
        'km_brain': dp['gm2_km_brain'], 'vmax_liver': dp['vmax_liver'],
        'km_liver': dp['gm2_km_liver'], 'inf_threshold': dp['inf_threshold'],
        'k_inf': 0.2, 'k_res': 0.05,
        'bayley_motor_baseline': dp['bayley_motor_baseline'],
        'bayley_cognitive_baseline': dp['bayley_cognitive_baseline'],
        'seiz_baseline': dp['seizure_baseline'],
        'resp_baseline': dp['respiratory_baseline'],
        'qol_baseline': dp['qol_baseline'],
        'rho_g': 1e-3, 'rho_i': 1e-2,
        'sigma_Agut': 0.05, 'sigma_p': 0.12, 'sigma_b': 0.10,
        'sigma_t4': 0.08, 'sigma_entry': 0.05, 'sigma_e': 0.04,
        'sigma_g': 0.06, 'sigma_i': 0.05, 'sigma_cl': 0.03,
        'fus_uptake_gain_scale': 1.0, 'fus_entry_gain_scale': 1.5,
    }
    base.update(dp)
    return base


def drift_full(x: np.ndarray, t: float, dose_mg: float, params: Dict[str, Any],
               t4_admin_day: Optional[float] = None, t4_admin_particles: float = 0.0,
               fus_events: Optional[List[FUSEvent]] = None,
               dt_for_admin_window: float = 1e-6) -> np.ndarray:
    A_gut, P, B, T4_sys, T4_ent, E_expr, G_B, G_L, I, D, Bayley_motor, Bayley_cog, SEIZ, RESP, QOL, Ab = x

    m_fus = fus_multiplier(t, fus_events)

    if t4_admin_day is not None:
        # windowed (nearest-time-step) injection: the original exact
        # floating-point equality check (`abs(t - t4_admin_day) < 1e-6`)
        # only fires when dt evenly divides t4_admin_day, so most dt
        # choices silently skip AAV administration entirely. Using a
        # half-step window makes the bolus fire exactly once regardless
        # of dt, on whichever grid point is nearest to t4_admin_day.
        if abs(t - t4_admin_day) <= 0.5 * dt_for_admin_window:
            T4_sys += t4_admin_particles

    A_add = dose_pulse_smooth(t, dose_mg)
    dA = -params['ka'] * A_gut + A_add

    dP = params['F'] * params['ka'] * A_gut - (params['k_el'] + params['k_p2b']) * P + params['k_b2p'] * B

    k_p2b_eff = params['k_p2b'] * (1.0 + params['fus_uptake_gain_scale'] * (m_fus - 1.0))
    dB = k_p2b_eff * P - params['k_brain_elim'] * B

    B_free = B * (1.0 - params['protein_binding'])

    k_entry_eff = params['k_T4_entry'] * (1.0 + params['fus_entry_gain_scale'] * (m_fus - 1.0))
    dT4_sys = -params['T4_clear'] * T4_sys - k_entry_eff * T4_sys

    dT4_ent = k_entry_eff * T4_sys * (1.0 - T4_ent) - 0.001 * T4_ent

    dE = params['k_T4_payload'] * T4_ent * (params['expr_cap'] - E_expr) - params['k_T4_decay'] * E_expr

    inhib = hill(B_free, params['IC50'], params['Emax'], params['hill_n'])
    synth = params['gm2_synth']
    brain_clear = mm_clearance(G_B, params['residual_hexa'] + E_expr, params['vmax_brain'], params['km_brain'])
    liver_clear = mm_clearance(G_L, params['residual_hexa'] + E_expr, params['vmax_liver'], params['km_liver'])
    dG_B = synth * (1 - inhib) - brain_clear - 0.01 * G_B
    dG_L = 0.7 * synth * (1 - inhib) - liver_clear - 0.05 * G_L

    trigger = max(0.0, (G_B - params['inf_threshold']) / params['inf_threshold'])
    anti_inf = 0.15 * inhib + 0.05 * (params['residual_hexa'] + E_expr)
    dI = params['k_inf'] * trigger - params['k_res'] * I - anti_inf

    dD = params['rho_g'] * max(0.0, G_B - params['inf_threshold']) + params['rho_i'] * I - 0.0001 * D

    gm2_motor_factor = np.clip(1 - (G_B / 2500.0), 0.01, 2.0)
    infl_motor_factor = np.clip(1 - 0.8 * I, 0.01, 1.5)
    enzyme_motor_factor = np.clip(1 + 0.4 * (params['residual_hexa'] + E_expr), 0.5, 2.5)

    target_motor = 100.0 * gm2_motor_factor * infl_motor_factor * enzyme_motor_factor
    target_motor = np.clip(target_motor, 46.0, 154.0)

    k_motor = 0.02
    dBayley_motor = k_motor * (target_motor - Bayley_motor)

    damage_cog_factor = np.exp(-0.015 * D)
    gm2_cog_factor = np.clip(1 - (G_B / 2000.0), 0.01, 1.5)
    enzyme_cog_factor = np.clip(1 + 0.3 * (params['residual_hexa'] + E_expr), 0.5, 2.0)

    target_cog = 100.0 * damage_cog_factor * gm2_cog_factor * enzyme_cog_factor
    target_cog = np.clip(target_cog, 55.0, 145.0)

    k_cog = 0.012
    dBayley_cog = k_cog * (target_cog - Bayley_cog)

    targ_SEIZ = params['seiz_baseline'] / (1 + 0.6 * (params['residual_hexa'] + E_expr))
    targ_RESP = params['resp_baseline'] * gm2_motor_factor * enzyme_motor_factor
    targ_QOL = params['qol_baseline'] * gm2_motor_factor * infl_motor_factor

    dSEIZ = 0.03 * (targ_SEIZ - SEIZ)
    dRESP = 0.025 * (targ_RESP - RESP)
    dQOL = 0.02 * (targ_QOL - QOL)

    dAb = 0.015 * (T4_sys > 0) - 0.01 * Ab

    return np.array([dA, dP, dB, dT4_sys, dT4_ent, dE, dG_B, dG_L, dI, dD,
                     dBayley_motor, dBayley_cog, dSEIZ, dRESP, dQOL, dAb], dtype=float)


def diffusion_full(x: np.ndarray, t: float, params: Dict[str, Any]) -> np.ndarray:
    A_gut, P, B, T4_sys, T4_ent, E_expr, G_B, G_L, I, D, Bayley_motor, Bayley_cog, SEIZ, RESP, QOL, Ab = x
    b = np.zeros_like(x)
    b[0] = params['sigma_Agut'] * max(1e-8, A_gut)
    b[1] = params['sigma_p'] * max(1e-8, P)
    b[2] = params['sigma_b'] * max(1e-8, B)
    b[3] = params['sigma_t4'] * max(1e-8, T4_sys)
    b[4] = params['sigma_entry'] * max(1e-8, T4_ent)
    b[5] = params['sigma_e'] * max(1e-8, E_expr)
    b[6] = params['sigma_g'] * max(1e-8, G_B)
    b[7] = params['sigma_g'] * max(1e-8, G_L)
    b[8] = params['sigma_i'] * max(1e-8, I)
    b[9] = 0.005 * max(1e-8, D)
    b[10] = params['sigma_cl'] * max(1e-8, Bayley_motor)
    b[11] = params['sigma_cl'] * max(1e-8, Bayley_cog)
    b[12] = params['sigma_cl'] * max(1e-8, SEIZ)
    b[13] = params['sigma_cl'] * max(1e-8, RESP)
    b[14] = params['sigma_cl'] * max(1e-8, QOL)
    b[15] = 0.02 * max(1e-8, Ab)
    return b


def simulate_milstein_full(x0: np.ndarray, tmax_days: float, dt: float, dose_mg: float,
                            disease: str = 'tay-sachs',
                            t4_admin_day: Optional[float] = None, t4_admin_particles: float = 0.0,
                            fus_events: Optional[List[FUSEvent]] = None,
                            seed: Optional[int] = None, corr: Optional[np.ndarray] = None,
                            params_override: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_params = make_base_params(disease)
    if params_override:
        base_params.update(params_override)
    params = base_params

    n_steps = int(np.ceil(tmax_days / dt)) + 1
    times = np.linspace(0, tmax_days, n_steps)
    dim = len(x0)
    X = np.zeros((n_steps, dim))
    X[0, :] = x0.copy()

    if corr is None:
        corr = np.eye(dim)
    L = np.linalg.cholesky(corr + 1e-12 * np.eye(dim))
    sqrt_dt = np.sqrt(dt)

    for i in range(1, n_steps):
        t = times[i - 1]
        x = X[i - 1, :].copy()
        f = drift_full(x, t, dose_mg, params, t4_admin_day=t4_admin_day,
                       t4_admin_particles=t4_admin_particles, fus_events=fus_events,
                       dt_for_admin_window=dt)
        b = diffusion_full(x, t, params)
        z = rng.standard_normal(dim)
        dW = (L @ z) * sqrt_dt
        x_new = np.zeros_like(x)
        for j in range(dim):
            drift_term = f[j] * dt
            diff_term = b[j] * dW[j]
            eps = 1e-9
            local_sigma = b[j] / max(abs(x[j]), eps)
            milstein_corr = 0.5 * local_sigma * b[j] * ((dW[j] ** 2) - dt)
            x_new[j] = x[j] + drift_term + diff_term + milstein_corr

        x_new[0] = max(x_new[0], 0.0)
        x_new[1] = max(x_new[1], 0.0)
        x_new[2] = max(x_new[2], 0.0)
        x_new[3] = max(x_new[3], 0.0)
        x_new[4] = np.clip(x_new[4], 0.0, 1.0)
        x_new[5] = np.clip(x_new[5], 0.0, params['expr_cap'])
        x_new[6] = max(x_new[6], 0.0)
        x_new[7] = max(x_new[7], 0.0)
        x_new[8] = max(x_new[8], 0.0)
        x_new[9] = max(x_new[9], 0.0)
        x_new[10] = np.clip(x_new[10], 46.0, 154.0)
        x_new[11] = np.clip(x_new[11], 55.0, 145.0)
        x_new[12] = max(x_new[12], 0.0)
        x_new[13] = np.clip(x_new[13], 0.0, 150.0)
        x_new[14] = np.clip(x_new[14], 0.0, 100.0)
        x_new[15] = max(x_new[15], 0.0)

        X[i, :] = x_new

    cols = ["A_gut", "plasma", "brain_drug", "T4_sys", "T4_entry", "E_expr", "gm2_brain", "gm2_liver",
            "inflammation", "damage", "bayley_motor_composite", "bayley_cognitive_composite",
            "seizures_week", "respiratory", "quality_of_life", "anti_phage_ab"]
    df = pd.DataFrame(X, columns=cols)
    df["time_days"] = times
    return df


def make_x0(params: Dict[str, Any]) -> np.ndarray:
    return np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.02,
        params['baseline_gm2_brain'], params['baseline_gm2_liver'],
        0.3, 0.0,
        params['bayley_motor_baseline'], params['bayley_cognitive_baseline'],
        params['seizure_baseline'] if 'seizure_baseline' in params else params['seiz_baseline'],
        params['respiratory_baseline'] if 'respiratory_baseline' in params else params['resp_baseline'],
        params['qol_baseline'], 0.0
    ], dtype=float)


DEFAULT_FUS_EVENTS = [
    FUSEvent(day=30.0, duration_hours=1.0, gain=1.8),
    FUSEvent(day=60.0, duration_hours=1.0, gain=1.6),
    FUSEvent(day=90.0, duration_hours=1.0, gain=1.5),
]


def run_single(params_override: Dict[str, Any], disease: str = 'tay-sachs',
               tmax_days: float = 365.0, dt: float = 0.2, dose_mg: float = 50.0,
               t4_admin_day: float = 29.9, t4_admin_particles: float = 5e12,
               seed: int = 42) -> float:
    """Run one trajectory under the tri-modal protocol and return terminal brain GM2 (nmol/g)."""
    base = make_base_params(disease)
    base.update(params_override)
    x0 = make_x0(base)
    df = simulate_milstein_full(
        x0, tmax_days, dt, dose_mg, disease=disease,
        t4_admin_day=t4_admin_day, t4_admin_particles=t4_admin_particles,
        fus_events=DEFAULT_FUS_EVENTS, seed=seed,
        params_override=params_override,
    )
    return float(df["gm2_brain"].iloc[-1])
