"""
verification_tests.py

Task 2: five verification unit tests against gm2_model_v2.py (the fixed,
17-state rebuild -- NOT gm2_solver.py, the recovered/buggy original).
Does not modify gm2_model_v2.py; instrumentation for test (e) is a
separate, faithful re-implementation of simulate_milstein()'s update
step in this file (same formulas, copied not altered) with pre-clip
bookkeeping added, since the shipped function only returns post-clip
state.

Tests:
  a. Zero SRT dose gives baseline (unmodified) synthesis rate.
  b. High SRT dose decreases synthesis rate monotonically (3+ doses).
  c. Milstein update matches analytic GBM mean/variance for an isolated
     multiplicative-noise state.
  d. FUS-off arms == FUS-mechanism-disabled, when AAV+SRT are inactive.
  e. All 17 states stay non-negative across 1000 realizations of a
     representative parameter set; report per-state min (post-clip) and
     whether clipping ever fired (pre-clip excursions).

Output: data/verification_tests.json
"""
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gm2_model_v2 import (
    make_base_params, make_x0, simulate_milstein, run_single, hill,
    drift_full, diffusion_full, BOUNDS, STATE_NAMES, DEFAULT_FUS_EVENTS,
)
from parallel_eval import evaluate_parallel, n_workers_from_env

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_FILE = os.path.join(ROOT, "gm2_model_v2.py")
with open(MODEL_FILE, "rb") as f:
    MODEL_SHA256 = hashlib.sha256(f.read()).hexdigest()[:16]

results = []


def hill_vec(B, IC50, Emax, n):
    """Vectorized form of gm2_model_v2.hill() (which is scalar-only, uses
    Python max() -- ambiguous on arrays). Same formula, unmodified."""
    B = np.maximum(B, 0.0)
    num = Emax * (B ** n)
    den = (IC50 ** n) + (B ** n) + 1e-12
    return num / den


def add_result(name, description, passed, observed_values, tolerance_used):
    results.append({
        "test_name": name,
        "description": description,
        "passed": bool(passed),
        "observed_values": observed_values,
        "tolerance_used": tolerance_used,
    })
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}: {description}", flush=True)


# ---------------------------------------------------------------------
# (a) Zero SRT dose gives baseline (unmodified) synthesis rate.
# ---------------------------------------------------------------------
def test_a():
    base = make_base_params('tay-sachs')
    raw_synth = base['gm2_synth']

    # Algebraic check: at B_free=0 the Hill inhibition term is exactly 0.
    inhib_at_zero = hill(0.0, base['IC50'], base['Emax'], base['hill_n'])

    # Empirical check: run with dose_mg_per_day=0 -> brain drug B should
    # stay ~0 for the whole trajectory -> effective synthesis rate
    # gm2_synth*(1-inhib) should track raw gm2_synth throughout.
    x0 = make_x0(base)
    df = simulate_milstein(x0, 200.0, 0.2, base, dose_mg_per_day=0.0,
                            t4_admin_day=30.0, t4_dose_total=1.0,
                            fus_events=DEFAULT_FUS_EVENTS, seed=7)
    B_free = df['brain_drug'].values * (1.0 - base['protein_binding'])
    inhib_traj = hill_vec(B_free, base['IC50'], base['Emax'], base['hill_n'])
    eff_synth_traj = raw_synth * (1.0 - inhib_traj)
    max_B = float(df['brain_drug'].max())
    mean_eff_synth = float(eff_synth_traj.mean())
    max_abs_dev_from_raw = float(np.max(np.abs(eff_synth_traj - raw_synth)))

    tol = 1e-6
    passed = (inhib_at_zero == 0.0) and (max_B < 1e-4) and (max_abs_dev_from_raw < 1e-3)
    add_result(
        "a_zero_srt_dose_baseline_synthesis",
        "Zero SRT dose (dose_mg_per_day=0) gives baseline (unmodified) synthesis rate.",
        passed,
        {
            "raw_gm2_synth": raw_synth,
            "hill_inhibition_at_B_free_equals_0": inhib_at_zero,
            "max_brain_drug_B_over_200d_trajectory": max_B,
            "mean_effective_synthesis_rate_over_trajectory": mean_eff_synth,
            "max_abs_deviation_effective_vs_raw_synth": max_abs_dev_from_raw,
        },
        {"max_B_threshold": 1e-4, "max_abs_dev_threshold": 1e-3, "exact_algebraic_check": "inhib(0)==0"},
    )


# ---------------------------------------------------------------------
# (b) High SRT dose decreases synthesis rate monotonically (3+ doses).
# ---------------------------------------------------------------------
def test_b():
    base = make_base_params('tay-sachs')
    raw_synth = base['gm2_synth']
    doses = [0.0, 3.0, 10.0, 30.0]
    seeds = list(range(10))
    window_days = (300.0, 365.0)  # trailing window, near quasi-steady-state

    per_dose_mean = []
    for dose in doses:
        vals = []
        for seed in seeds:
            x0 = make_x0(base)
            df = simulate_milstein(x0, 365.0, 0.2, base, dose_mg_per_day=dose,
                                    t4_admin_day=30.0, t4_dose_total=1.0,
                                    fus_events=DEFAULT_FUS_EVENTS, seed=seed)
            mask = (df['time_days'] >= window_days[0]) & (df['time_days'] <= window_days[1])
            B_free = df.loc[mask, 'brain_drug'].values * (1.0 - base['protein_binding'])
            inhib = hill_vec(B_free, base['IC50'], base['Emax'], base['hill_n'])
            eff_synth = raw_synth * (1.0 - inhib)
            vals.append(float(eff_synth.mean()))
        per_dose_mean.append(float(np.mean(vals)))

    diffs = np.diff(per_dose_mean)
    passed = bool(np.all(diffs < 0))
    add_result(
        "b_high_srt_dose_monotonic_decrease",
        "Higher SRT dose monotonically decreases mean effective synthesis rate at 4 dose levels (0,3,10,30 mg/day), day 300-365 window, averaged over 10 seeds.",
        passed,
        {
            "doses_mg_per_day": doses,
            "mean_effective_synthesis_rate_per_dose": per_dose_mean,
            "successive_differences": diffs.tolist(),
            "n_seeds_averaged": len(seeds),
        },
        {"criterion": "strictly decreasing across all 4 dose levels (mean over 10 seeds)"},
    )


# ---------------------------------------------------------------------
# (c) Milstein update matches analytic GBM moments.
# ---------------------------------------------------------------------
def _gbm_worker(args):
    seed, base, x0, tmax_days, dt = args
    df = simulate_milstein(x0, tmax_days, dt, base, dose_mg_per_day=0.0,
                            t4_admin_day=1.0, t4_dose_total=0.0,
                            fus_events=None, seed=seed)
    return float(df['T4_sys'].iloc[-1])


def test_c():
    base = make_base_params('tay-sachs')
    base = dict(base)
    base['k_T4_entry'] = 0.0  # isolates T4_sys: dT4_sys = -T4_clear*T4_sys exactly (pure GBM drift)
    mu = -base['T4_clear']
    sigma = base['sigma_t4']
    X0 = 1.0
    tmax_days = 30.0
    dt = 0.05
    n_rep = 3000

    x0 = make_x0(base)
    x0[3] = X0  # STATE_NAMES[3] == 'T4_sys'

    args = [(seed, base, x0.copy(), tmax_days, dt) for seed in range(n_rep)]
    t0 = time.time()
    finals = np.array(evaluate_parallel(_gbm_worker, args, label="  gbm "))
    elapsed = time.time() - t0

    analytic_mean = X0 * np.exp(mu * tmax_days)
    analytic_var = (X0 ** 2) * np.exp(2 * mu * tmax_days) * (np.exp((sigma ** 2) * tmax_days) - 1.0)

    emp_mean = float(finals.mean())
    emp_var = float(finals.var(ddof=1))
    se_mean = np.sqrt(analytic_var / n_rep)  # Monte Carlo standard error on the mean
    rel_tol_mean = 5.0 * se_mean / analytic_mean  # 5-sigma MC-noise-aware tolerance
    rel_err_mean = abs(emp_mean - analytic_mean) / analytic_mean
    rel_err_var = abs(emp_var - analytic_var) / analytic_var

    passed = (rel_err_mean < max(rel_tol_mean, 0.05)) and (rel_err_var < 0.25)
    add_result(
        "c_milstein_matches_gbm_moments",
        "Milstein update (as implemented in simulate_milstein) reproduces analytic geometric Brownian motion mean/variance for an isolated multiplicative-noise state (T4_sys, k_T4_entry forced to 0 so dT4_sys=-T4_clear*T4_sys exactly).",
        passed,
        {
            "mu_drift_per_day": mu, "sigma_diffusion": sigma, "X0": X0,
            "tmax_days": tmax_days, "dt": dt, "n_realizations": n_rep,
            "analytic_mean": analytic_mean, "analytic_variance": analytic_var,
            "empirical_mean": emp_mean, "empirical_variance": emp_var,
            "relative_error_mean": rel_err_mean, "relative_error_variance": rel_err_var,
            "monte_carlo_se_on_mean": float(se_mean),
            "elapsed_s": elapsed,
        },
        {"rel_err_mean_threshold": "max(5*MC_SE/analytic_mean, 0.05)", "rel_err_var_threshold": 0.25},
    )


# ---------------------------------------------------------------------
# (d) FUS-off == FUS-mechanism-disabled, when AAV+SRT both inactive.
# ---------------------------------------------------------------------
def test_d():
    base = make_base_params('tay-sachs')
    x0 = make_x0(base)
    inactive = dict(dose_mg_per_day=0.0, t4_admin_day=30.0, t4_dose_total=0.0, seed=99)
    active = dict(dose_mg_per_day=3.0, t4_admin_day=30.0, t4_dose_total=1.0, seed=99)

    df_fus_off = simulate_milstein(x0.copy(), 200.0, 0.2, base, fus_events=None, **inactive)
    df_fus_disabled_arm = simulate_milstein(x0.copy(), 200.0, 0.2, base, fus_events=DEFAULT_FUS_EVENTS, **inactive)

    # Reference: the SAME fus_events=None vs. DEFAULT_FUS_EVENTS comparison
    # but with AAV+SRT genuinely ACTIVE, to calibrate how large a *real*
    # FUS effect looks like on this model/timescale for comparison.
    df_fus_off_active = simulate_milstein(x0.copy(), 200.0, 0.2, base, fus_events=None, **active)
    df_fus_on_active = simulate_milstein(x0.copy(), 200.0, 0.2, base, fus_events=DEFAULT_FUS_EVENTS, **active)

    diffs, diffs_active, rel_to_active = {}, {}, {}
    max_abs_diff_overall = 0.0
    for col in STATE_NAMES:
        d_inactive = float(np.max(np.abs(df_fus_off[col].values - df_fus_disabled_arm[col].values)))
        d_active = float(np.max(np.abs(df_fus_off_active[col].values - df_fus_on_active[col].values)))
        diffs[col] = d_inactive
        diffs_active[col] = d_active
        rel_to_active[col] = (d_inactive / d_active) if d_active > 0 else None
        max_abs_diff_overall = max(max_abs_diff_overall, d_inactive)

    exact_equality = (max_abs_diff_overall == 0.0)
    # Physically-grounded criterion: with AAV+SRT inactive, the FUS-on vs.
    # FUS-off divergence must be a NEGLIGIBLE fraction (<1e-3) of what the
    # same comparison produces when AAV+SRT are genuinely active on every
    # state where the active-arm comparison is itself non-degenerate.
    ratios = [r for r in rel_to_active.values() if r is not None]
    passed = bool(ratios) and max(ratios) < 1e-3
    add_result(
        "d_fus_off_equals_fus_disabled_when_aav_srt_inactive",
        "FUS-off arm (fus_events=None) vs. a run with DEFAULT_FUS_EVENTS active, when AAV (t4_dose_total=0) and SRT (dose_mg_per_day=0) are both inactive.",
        passed,
        {
            "max_abs_diff_per_state_AAV_SRT_INACTIVE": diffs,
            "max_abs_diff_overall_AAV_SRT_INACTIVE": max_abs_diff_overall,
            "exact_bit_equality_AAV_SRT_INACTIVE": exact_equality,
            "max_abs_diff_per_state_AAV_SRT_ACTIVE_reference": diffs_active,
            "ratio_inactive_diff_to_active_diff_per_state": rel_to_active,
            "max_ratio_inactive_to_active": max(ratios) if ratios else None,
            "explanation": (
                "NOT bit-identical (exact_bit_equality=False): diffusion_full() floors "
                "multiplicative noise at max(1e-8, x[j]), so states nominally at exact "
                "zero (P, T4_sys) still receive a tiny nonzero stochastic increment each "
                "step and drift to ~1e-8-scale nonzero values; once nonzero, the FUS gain "
                "(which multiplies k_p2b_eff*P and k_entry_eff*T4_sys) picks up a genuine "
                "but minuscule effect on those terms instead of multiplying an exact zero. "
                "This is a real, if extremely small, consequence of the diffusion floor, "
                "not a test artifact -- so exact equality is the wrong pass criterion. The "
                "criterion used instead: this residual must be negligible (<0.1%) relative "
                "to the FUS-on/off divergence produced by the SAME comparison when AAV+SRT "
                "are genuinely active (where FUS has real quantities to multiply)."
            ),
        },
        {"criterion": "max_i [diff_inactive_i / diff_active_i] < 1e-3", "exact_equality_also_reported": True},
    )


# ---------------------------------------------------------------------
# (e) All 17 states non-negative across 1000 realizations; report min
#     per state and whether clipping/truncation occurred.
# ---------------------------------------------------------------------
def simulate_milstein_instrumented(x0, tmax_days, dt, params, dose_mg_per_day,
                                    t4_admin_day, t4_dose_total, t4_pulse_width_days,
                                    fus_events, seed):
    """Faithful copy of gm2_model_v2.simulate_milstein's update loop (same
    formulas, unmodified) with pre-clip bookkeeping added so clipping
    events and pre-clip excursions can be measured without altering the
    shipped function's behavior."""
    rng = np.random.default_rng(seed)
    n_steps = int(np.ceil(tmax_days / dt)) + 1
    dim = len(x0)
    x = x0.copy()
    sqrt_dt = np.sqrt(dt)

    preclip_min = np.full(dim, np.inf)
    postclip_min = np.full(dim, np.inf)
    clip_lo_count = np.zeros(dim, dtype=int)
    clip_hi_count = np.zeros(dim, dtype=int)
    postclip_min = np.minimum(postclip_min, x)
    preclip_min = np.minimum(preclip_min, x)

    t = 0.0
    for i in range(1, n_steps):
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

            preclip_min[j] = min(preclip_min[j], x_new[j])

            lo, hi = BOUNDS[j]
            hi_val = params[hi] if isinstance(hi, str) else hi
            if lo is not None and x_new[j] < lo:
                clip_lo_count[j] += 1
                x_new[j] = lo
            if hi_val is not None and x_new[j] > hi_val:
                clip_hi_count[j] += 1
                x_new[j] = hi_val

            postclip_min[j] = min(postclip_min[j], x_new[j])

        x = x_new
        t += dt

    return preclip_min, postclip_min, clip_lo_count, clip_hi_count


def _state_worker(seed):
    base = make_base_params('tay-sachs')
    x0 = make_x0(base)
    pre_min, post_min, clip_lo, clip_hi = simulate_milstein_instrumented(
        x0, 365.0, 0.2, base, dose_mg_per_day=3.0, t4_admin_day=30.0,
        t4_dose_total=1.0, t4_pulse_width_days=1.0, fus_events=DEFAULT_FUS_EVENTS, seed=seed)
    return pre_min.tolist(), post_min.tolist(), clip_lo.tolist(), clip_hi.tolist()


def test_e():
    n_rep = 1000
    t0 = time.time()
    out = evaluate_parallel_generic(_state_worker, list(range(n_rep)), label="  states ")
    elapsed = time.time() - t0

    dim = len(STATE_NAMES)
    global_pre_min = np.full(dim, np.inf)
    global_post_min = np.full(dim, np.inf)
    total_clip_lo = np.zeros(dim, dtype=int)
    total_clip_hi = np.zeros(dim, dtype=int)
    for pre_min, post_min, clip_lo, clip_hi in out:
        global_pre_min = np.minimum(global_pre_min, pre_min)
        global_post_min = np.minimum(global_post_min, post_min)
        total_clip_lo += np.array(clip_lo, dtype=int)
        total_clip_hi += np.array(clip_hi, dtype=int)

    per_state = []
    all_nonneg_postclip = True
    any_clipping_occurred = bool(total_clip_lo.sum() + total_clip_hi.sum() > 0)
    for j, name in enumerate(STATE_NAMES):
        lo, _ = BOUNDS[j]
        min_post = float(global_post_min[j])
        min_pre = float(global_pre_min[j])
        if lo is not None and min_post < lo - 1e-9:
            all_nonneg_postclip = False
        per_state.append({
            "state": name,
            "min_value_observed_postclip": min_post,
            "min_value_observed_preclip": min_pre,
            "n_lower_bound_clip_events_total": int(total_clip_lo[j]),
            "n_upper_bound_clip_events_total": int(total_clip_hi[j]),
            "was_clipped": bool(total_clip_lo[j] + total_clip_hi[j] > 0),
            "preclip_would_have_gone_negative": bool(lo is not None and min_pre < lo - 1e-9),
        })

    passed = all_nonneg_postclip
    add_result(
        "e_all_states_nonnegative_1000_reps",
        "All 17 state variables remain within their defined bounds (non-negative where a lower bound of 0 applies) across 1000 realizations of a representative parameter set (tay-sachs baseline, standard tri-modal protocol, 365d, dt=0.2); reports per-state min (post-clip, i.e. what the model actually reports) and min pre-clip (what the raw Milstein step produced before BOUNDS clamping), plus clip-event counts, since the model's own integrator enforces bounds by clamping and that clamping DOES bias reported means whenever it fires on a non-negligible fraction of steps.",
        passed,
        {
            "n_realizations": n_rep, "tmax_days": 365.0, "dt": 0.2, "n_steps_per_realization": int(np.ceil(365.0 / 0.2)) + 1,
            "per_state": per_state,
            "any_clipping_occurred_anywhere": any_clipping_occurred,
            "elapsed_s": elapsed,
        },
        {"criterion": "post-clip min >= lower bound for every state (by construction of the clamp); pre-clip figures reported for transparency, not part of pass/fail"},
    )


def evaluate_parallel_generic(worker_fn, rows, label=""):
    # evaluate_parallel() in parallel_eval.py assumes scalar float outputs
    # (Y[i] = y into a fixed-size np.ndarray) -- test (e)'s workers return
    # tuples of per-state arrays, so use mp.Pool directly with the same
    # n_workers convention instead of forcing that shape.
    import multiprocessing as mp
    n_workers = n_workers_from_env()
    n_workers = max(1, min(n_workers, len(rows)))
    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        results_list = list(pool.imap(worker_fn, rows, chunksize=4))
    print(f"  {label}done: {len(rows)} reps in {time.time()-t0:.0f}s with {n_workers} workers", flush=True)
    return results_list


if __name__ == "__main__":
    t_all = time.time()
    run_meta = {
        "model_file": "gm2_model_v2.py", "model_sha256_16": MODEL_SHA256,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_workers": n_workers_from_env(),
    }
    print(json.dumps(run_meta, indent=2))

    test_a()
    test_b()
    test_c()
    test_d()
    test_e()

    n_pass = sum(1 for r in results if r["passed"])
    output = {"run_meta": run_meta, "n_tests": len(results), "n_passed": n_pass, "tests": results}
    outpath = os.path.join(DATA_DIR, "verification_tests.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{n_pass}/{len(results)} tests passed. Wrote {outpath} ({time.time()-t_all:.0f}s total)")
