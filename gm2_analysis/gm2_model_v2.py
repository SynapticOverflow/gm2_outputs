# gm2_model_v2.py
#
# Ground-up rebuild of the GM2 tri-modal therapy model, in response to two
# confirmed defects in the original repo code (SynapticOverflow/comp-fus-
# aav9-therapy, "Simulation Code"):
#
#   BUG 1 (root cause, not just patched): AAV-T4 administration was coded
#   as an instantaneous delta injection gated by exact floating-point time
#   equality (`abs(t - t4_admin_day) < 1e-6`). Whether that ever fires
#   depends on whether dt evenly divides the admin day -- for most dt
#   choices it silently never fires, so "tri-modal" runs were silently
#   SRT-only. Patching the equality check to a dt-sized window (done in
#   gm2_solver.py) treats the symptom. Here the fix is architectural:
#   admin is a smooth finite-width pulse (like the SRT dosing already
#   was), so its integral is dt-independent by construction -- no grid
#   alignment condition to accidentally miss.
#
#   BUG 2: the AAV dose was carried in raw particle counts (~5e12) while
#   the entry-rate constant was ~2e-8/day. Their product is a ~1e5/day
#   effective rate -- BBB entry saturates in a fraction of a day for ANY
#   entry-rate value across 10+ orders of magnitude (verified by direct
#   sweep: saturation held from k_T4_entry=1e-2 down to 1e-14; it only
#   broke below ~1e-15, six-plus orders of magnitude below baseline).
#   That makes entry rate structurally incapable of being "the" sensitive
#   parameter regardless of any reasonable prior width, independent of
#   whether entry is biologically a bottleneck or not. Here, both the AAV
#   compartment and the dose are carried in NORMALIZED units (dose = 1.0
#   means "100% of the administered vector," entry rate constants are
#   per-day rates of comparable order to the model's other rate
#   constants), so BBB entry has an actual day-to-week timescale and can
#   show up as sensitive or not on its own mechanistic merits.
#
# ALSO ADDED: a 17th explicit state completing the state count the MICCAI
# draft claims (16 in the recovered code -- no CTL/cellular-immunity
# state existed anywhere in either accessible repo, confirmed by
# exhaustive grep across all files/branches). The added state (A_T,
# cellular/CTL anti-capsid response) and its coupling into the model
# (CTL-mediated killing of transduced, expressing cells; humoral+cellular
# suppression of ongoing effective BBB entry) are THIS REBUILD'S OWN
# construction, written to match the paper's stated immunogenicity
# mechanism as closely as the available description allows -- they are
# NOT recovered original equations, because no original 17-state
# implementation was found to recover. Treat this file as a from-scratch,
# clearly-labeled reconstruction, not a bug-fixed copy of Kartheek's
# original code.
#
# States (17):
#   0  A_gut        SRT (SP2) drug, gut compartment
#   1  P            SRT drug, plasma
#   2  B            SRT drug, brain (site of synthesis inhibition)
#   3  T4_sys       AAV-T4 vector, systemic (normalized: 1.0 = full dose)
#   4  T4_ent       AAV-T4 vector, crossed into brain parenchyma (fraction)
#   5  E_expr       HEXA/HEXB enzyme expression level (brain)
#   6  G_B          GM2 substrate burden, brain (nmol/g)
#   7  G_L          GM2 substrate burden, liver (nmol/g)
#   8  I            Neuroinflammation index
#   9  D            Cumulative neuronal damage
#   10 Bayley_motor Bayley-III motor composite
#   11 Bayley_cog   Bayley-III cognitive composite
#   12 SEIZ         Seizures/week
#   13 RESP         Respiratory function score
#   14 QOL          Quality-of-life score
#   15 Ab           Humoral (antibody) anti-capsid response
#   16 A_T          Cellular (CTL) anti-capsid response  [NEW]

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

N_STATES = 17

STATE_NAMES = [
    "A_gut", "plasma", "brain_drug", "T4_sys", "T4_entry", "E_expr",
    "gm2_brain", "gm2_liver", "inflammation", "damage",
    "bayley_motor_composite", "bayley_cognitive_composite",
    "seizures_week", "respiratory", "quality_of_life",
    "anti_capsid_humoral", "anti_capsid_cellular",
]


def disease_params(disease: str = 'tay-sachs') -> Dict[str, float]:
    d = disease.lower()
    if d in ('sandhoff', 'sandhof', 'sandhoff disease'):
        return {
            'name': 'Sandhoff',
            'baseline_gm2_brain': 1450.0, 'baseline_gm2_liver': 280.0,
            'residual_hexa': 0.018, 'residual_hexb': 0.0,
            'gm2_synth': 2.8, 'gm2_km_brain': 85.0, 'gm2_km_liver': 65.0,
            'vmax_brain': 12.5, 'vmax_liver': 18.0, 'inf_threshold': 800.0,
            'bayley_motor_baseline': 55.0, 'bayley_cognitive_baseline': 60.0,
            'seizure_baseline': 8.5, 'respiratory_baseline': 65.0, 'qol_baseline': 40.0,
        }
    else:
        return {
            'name': 'Tay-Sachs',
            'baseline_gm2_brain': 890.0, 'baseline_gm2_liver': 156.0,
            'residual_hexa': 0.0, 'residual_hexb': 0.85,
            'gm2_synth': 2.3, 'gm2_km_brain': 75.0, 'gm2_km_liver': 55.0,
            'vmax_brain': 12.5, 'vmax_liver': 18.0, 'inf_threshold': 650.0,
            'bayley_motor_baseline': 65.0, 'bayley_cognitive_baseline': 68.0,
            'seizure_baseline': 3.2, 'respiratory_baseline': 75.0, 'qol_baseline': 55.0,
        }


def hill(B: float, IC50: float, Emax: float, n: float) -> float:
    B = max(B, 0.0)
    num = Emax * (B ** n)
    den = (IC50 ** n) + (B ** n) + 1e-12
    return num / den


def mm_clearance(S: float, E_eff: float, vmax: float, km: float) -> float:
    S = max(S, 0.0)
    E_eff = max(E_eff, 0.0)
    return (vmax * E_eff * S) / (km + S + 1e-12)


def gaussian_pulse_rate(t_days: float, center_day: float, total_dose: float,
                         width_days: float) -> float:
    """
    A finite-width Gaussian delivery pulse whose time-integral equals
    total_dose, regardless of the integration dt used to sample it
    (as long as dt is a reasonably small fraction of width_days).
    Replaces an instantaneous delta injection with a smooth rate term
    added directly into a state's drift -- eliminates any grid-alignment
    dependence by construction.
    """
    norm = 1.0 / (width_days * np.sqrt(2.0 * np.pi))
    return total_dose * norm * np.exp(-0.5 * ((t_days - center_day) / width_days) ** 2)


def daily_dose_rate(t_days: float, dose_per_day: float, n_per_day: int = 2,
                     pulse_width_day: float = 0.08) -> float:
    """
    Smooth analogue of twice-daily oral dosing. Pulse width is wide
    enough (~2 hours) that dt <= 0.2 days resolves it without severe
    discretization error, unlike the original's 0.03-day (43-minute)
    pulses.
    """
    frac = t_days - np.floor(t_days)
    total = 0.0
    for p in np.linspace(0, 1, n_per_day, endpoint=False):
        total += gaussian_pulse_rate(frac, p, dose_per_day / n_per_day, pulse_width_day)
    return total


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
        # SRT (SP2) PK
        'ka': 1.2, 'F': 0.85, 'k_el': 0.035,
        'k_p2b': 0.12, 'k_b2p': 0.015, 'k_brain_elim': 0.15,
        'IC50': 50.0, 'Emax': 0.6, 'hill_n': 1.2, 'protein_binding': 0.15,

        # AAV-T4 delivery + expression -- NORMALIZED units (dose=1.0 full dose)
        'T4_clear': 0.05,          # systemic vector clearance, /day
        'k_T4_entry': 0.06,        # BBB entry rate, /day (was 2e-8 in raw-particle units)
        'k_T4_payload': 0.08,      # expression buildup rate, /day
        'k_T4_decay': 1 / 240.0,   # enzyme decay, /day
        'expr_cap': 8.0,

        # GM2 metabolism
        'gm2_synth': dp['gm2_synth'], 'vmax_brain': dp['vmax_brain'],
        'km_brain': dp['gm2_km_brain'], 'vmax_liver': dp['vmax_liver'],
        'km_liver': dp['gm2_km_liver'], 'inf_threshold': dp['inf_threshold'],
        'k_inf': 0.2, 'k_res': 0.05,

        # Clinical baselines
        'bayley_motor_baseline': dp['bayley_motor_baseline'],
        'bayley_cognitive_baseline': dp['bayley_cognitive_baseline'],
        'seiz_baseline': dp['seizure_baseline'],
        'resp_baseline': dp['respiratory_baseline'],
        'qol_baseline': dp['qol_baseline'],
        'rho_g': 1e-3, 'rho_i': 1e-2,

        # Noise scales
        'sigma_Agut': 0.05, 'sigma_p': 0.12, 'sigma_b': 0.10,
        'sigma_t4': 0.08, 'sigma_entry': 0.05, 'sigma_e': 0.04,
        'sigma_g': 0.06, 'sigma_i': 0.05, 'sigma_cl': 0.03,
        'sigma_ab': 0.05,

        # FUS gains
        'fus_uptake_gain_scale': 1.0, 'fus_entry_gain_scale': 1.5,

        # Immunogenicity (NEW: both arms actually feed back on entry/expression,
        # unlike the recovered code where Ab was a decorative output only)
        'k_ab_prime': 0.02,      # humoral priming rate from antigen exposure
        'k_ab_decay': 0.01,
        'c_hum_suppress': 0.5,   # max fractional suppression of entry from Ab
        'k_ctl_prime': 0.008,    # cellular priming rate (slower onset than humoral)
        'k_ctl_decay': 0.004,    # slower decay (longer-lived than humoral)
        'c_cell_suppress': 0.6,  # max fractional suppression of entry from CTL
        'k_ctl_clear_expr': 0.05,  # CTL-mediated clearance of expressing cells
    }
    base.update(dp)
    return base


def drift_full(x: np.ndarray, t: float, dose_mg_per_day: float, params: Dict[str, Any],
               t4_admin_day: float, t4_dose_total: float, t4_pulse_width_days: float,
               fus_events: Optional[List[FUSEvent]] = None) -> np.ndarray:
    (A_gut, P, B, T4_sys, T4_ent, E_expr, G_B, G_L, I, D,
     Bayley_motor, Bayley_cog, SEIZ, RESP, QOL, Ab, A_T) = x

    m_fus = fus_multiplier(t, fus_events)

    # --- SRT (SP2) PK: smooth twice-daily dosing, gut -> plasma -> brain ---
    A_add = daily_dose_rate(t, dose_mg_per_day)
    dA = -params['ka'] * A_gut + A_add
    dP = params['F'] * params['ka'] * A_gut - (params['k_el'] + params['k_p2b']) * P + params['k_b2p'] * B
    k_p2b_eff = params['k_p2b'] * (1.0 + params['fus_uptake_gain_scale'] * (m_fus - 1.0))
    dB = k_p2b_eff * P - params['k_brain_elim'] * B
    B_free = B * (1.0 - params['protein_binding'])

    # --- AAV-T4: smooth finite-width admin pulse (dt-independent by construction) ---
    admin_rate = gaussian_pulse_rate(t, t4_admin_day, t4_dose_total, t4_pulse_width_days)

    # Immune suppression of ongoing entry (both arms; NEW vs. recovered code,
    # where Ab existed only as an output with no feedback into the dynamics)
    immune_suppression = max(0.0, 1.0 - params['c_hum_suppress'] * min(Ab, 1.0)
                              - params['c_cell_suppress'] * min(A_T, 1.0))
    k_entry_eff = (params['k_T4_entry']
                   * (1.0 + params['fus_entry_gain_scale'] * (m_fus - 1.0))
                   * immune_suppression)

    dT4_sys = admin_rate - params['T4_clear'] * T4_sys - k_entry_eff * T4_sys
    dT4_ent = k_entry_eff * T4_sys * (1.0 - T4_ent) - 0.02 * T4_ent

    # CTL-mediated killing of transduced/expressing cells reduces net expression
    dE = (params['k_T4_payload'] * T4_ent * (params['expr_cap'] - E_expr)
          - params['k_T4_decay'] * E_expr
          - params['k_ctl_clear_expr'] * min(A_T, 1.0) * E_expr)

    # --- GM2 metabolism ---
    inhib = hill(B_free, params['IC50'], params['Emax'], params['hill_n'])
    synth = params['gm2_synth']
    brain_clear = mm_clearance(G_B, params['residual_hexa'] + E_expr, params['vmax_brain'], params['km_brain'])
    liver_clear = mm_clearance(G_L, params['residual_hexa'] + E_expr, params['vmax_liver'], params['km_liver'])
    dG_B = synth * (1 - inhib) - brain_clear - 0.01 * G_B
    dG_L = 0.7 * synth * (1 - inhib) - liver_clear - 0.05 * G_L

    # --- Neuroinflammation / damage ---
    trigger = max(0.0, (G_B - params['inf_threshold']) / params['inf_threshold'])
    anti_inf = 0.15 * inhib + 0.05 * (params['residual_hexa'] + E_expr)
    dI = params['k_inf'] * trigger - params['k_res'] * I - anti_inf
    dD = params['rho_g'] * max(0.0, G_B - params['inf_threshold']) + params['rho_i'] * I - 0.0001 * D

    # --- Clinical composites ---
    gm2_motor_factor = np.clip(1 - (G_B / 2500.0), 0.01, 2.0)
    infl_motor_factor = np.clip(1 - 0.8 * I, 0.01, 1.5)
    enzyme_motor_factor = np.clip(1 + 0.4 * (params['residual_hexa'] + E_expr), 0.5, 2.5)
    target_motor = np.clip(100.0 * gm2_motor_factor * infl_motor_factor * enzyme_motor_factor, 46.0, 154.0)
    dBayley_motor = 0.02 * (target_motor - Bayley_motor)

    damage_cog_factor = np.exp(-0.015 * D)
    gm2_cog_factor = np.clip(1 - (G_B / 2000.0), 0.01, 1.5)
    enzyme_cog_factor = np.clip(1 + 0.3 * (params['residual_hexa'] + E_expr), 0.5, 2.0)
    target_cog = np.clip(100.0 * damage_cog_factor * gm2_cog_factor * enzyme_cog_factor, 55.0, 145.0)
    dBayley_cog = 0.012 * (target_cog - Bayley_cog)

    targ_SEIZ = params['seiz_baseline'] / (1 + 0.6 * (params['residual_hexa'] + E_expr))
    targ_RESP = params['resp_baseline'] * gm2_motor_factor * enzyme_motor_factor
    targ_QOL = params['qol_baseline'] * gm2_motor_factor * infl_motor_factor
    dSEIZ = 0.03 * (targ_SEIZ - SEIZ)
    dRESP = 0.025 * (targ_RESP - RESP)
    dQOL = 0.02 * (targ_QOL - QOL)

    # --- Immunogenicity: humoral (fast onset, faster decay) + cellular (slow, persistent) ---
    antigen = T4_ent + 0.3 * T4_sys  # capsid antigen exposure signal
    dAb = params['k_ab_prime'] * antigen * (1.0 - Ab) - params['k_ab_decay'] * Ab
    dA_T = params['k_ctl_prime'] * antigen * (1.0 - A_T) - params['k_ctl_decay'] * A_T

    return np.array([dA, dP, dB, dT4_sys, dT4_ent, dE, dG_B, dG_L, dI, dD,
                      dBayley_motor, dBayley_cog, dSEIZ, dRESP, dQOL, dAb, dA_T], dtype=float)


def diffusion_full(x: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    (A_gut, P, B, T4_sys, T4_ent, E_expr, G_B, G_L, I, D,
     Bayley_motor, Bayley_cog, SEIZ, RESP, QOL, Ab, A_T) = x
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
    b[15] = params['sigma_ab'] * max(1e-8, Ab)
    b[16] = params['sigma_ab'] * max(1e-8, A_T)
    return b


BOUNDS = {
    0: (0.0, None), 1: (0.0, None), 2: (0.0, None), 3: (0.0, None),
    4: (0.0, 1.0), 5: (0.0, 'expr_cap'), 6: (0.0, None), 7: (0.0, None),
    8: (0.0, None), 9: (0.0, None), 10: (46.0, 154.0), 11: (55.0, 145.0),
    12: (0.0, None), 13: (0.0, 150.0), 14: (0.0, 100.0), 15: (0.0, 1.0), 16: (0.0, 1.0),
}


def simulate_milstein(x0: np.ndarray, tmax_days: float, dt: float, params: Dict[str, Any],
                       dose_mg_per_day: float = 3.0,
                       t4_admin_day: float = 30.0, t4_dose_total: float = 1.0,
                       t4_pulse_width_days: float = 1.0,
                       fus_events: Optional[List[FUSEvent]] = None,
                       seed: Optional[int] = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_steps = int(np.ceil(tmax_days / dt)) + 1
    times = np.linspace(0, tmax_days, n_steps)
    dim = len(x0)
    X = np.zeros((n_steps, dim))
    X[0, :] = x0.copy()
    sqrt_dt = np.sqrt(dt)

    for i in range(1, n_steps):
        t = times[i - 1]
        x = X[i - 1, :].copy()
        f = drift_full(x, t, dose_mg_per_day, params,
                       t4_admin_day=t4_admin_day, t4_dose_total=t4_dose_total,
                       t4_pulse_width_days=t4_pulse_width_days, fus_events=fus_events)
        b = diffusion_full(x, params)
        dW = rng.standard_normal(dim) * sqrt_dt
        x_new = np.empty_like(x)
        for j in range(dim):
            eps = 1e-9
            local_sigma = b[j] / max(abs(x[j]), eps)
            milstein_corr = 0.5 * local_sigma * b[j] * ((dW[j] ** 2) - dt)
            x_new[j] = x[j] + f[j] * dt + b[j] * dW[j] + milstein_corr

            lo, hi = BOUNDS[j]
            hi_val = params[hi] if isinstance(hi, str) else hi
            if lo is not None:
                x_new[j] = max(x_new[j], lo)
            if hi_val is not None:
                x_new[j] = min(x_new[j], hi_val)

        X[i, :] = x_new

    df = pd.DataFrame(X, columns=STATE_NAMES)
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
        params['qol_baseline'], 0.0, 0.0,
    ], dtype=float)


DEFAULT_FUS_EVENTS = [
    FUSEvent(day=30.0, duration_hours=1.0, gain=1.8),
    FUSEvent(day=60.0, duration_hours=1.0, gain=1.6),
    FUSEvent(day=90.0, duration_hours=1.0, gain=1.5),
]


def run_single(params_override: Dict[str, Any], disease: str = 'tay-sachs',
               tmax_days: float = 365.0, dt: float = 0.2,
               dose_mg_per_day: float = 3.0, t4_admin_day: float = 30.0,
               t4_dose_total: float = 1.0, t4_pulse_width_days: float = 1.0,
               seed: int = 42) -> float:
    base = make_base_params(disease)
    base.update(params_override)
    x0 = make_x0(base)
    df = simulate_milstein(
        x0, tmax_days, dt, base, dose_mg_per_day=dose_mg_per_day,
        t4_admin_day=t4_admin_day, t4_dose_total=t4_dose_total,
        t4_pulse_width_days=t4_pulse_width_days,
        fus_events=DEFAULT_FUS_EVENTS, seed=seed,
    )
    return float(df["gm2_brain"].iloc[-1])
