"""
sde_lifu.py

16-dimensional Ito SDE system for tri-modal (AAV-T4 + SP2/SRT + LIFU) gene
therapy in infantile GM2 gangliosidosis, integrated with the Milstein scheme
(strong order 1.0):

    x_{n+1} = x_n + f(x_n) dt + g(x_n) dW_n
              + 0.5 * g(x_n) * g'(x_n) * (dW_n^2 - dt)

Diagonal-noise assumption: each state's diffusion term g_i depends only on
x_i itself (g_i(x) = sigma_i * x_i, i.e. multiplicative/geometric noise),
which is the standard choice for strictly non-negative biological state
variables and makes the Milstein correction closed-form
(g_i' = sigma_i everywhere).

State vector x[0..15] (per brain region, per realisation):

    x[0]  AAV-T4 plasma concentration          (nmol/L)
    x[1]  AAV-T4 CNS concentration              (nmol/g)
    x[2]  k_entry_effective                     (hr^-1)   [deterministic, see below]
    x[3]  SP2/SRT editing efficiency             (fraction, 0-1)
    x[4]  Hex-A enzyme activity                  (nmol/mg/hr)
    x[5]  GM2 burden                             (nmol/g)  <-- PRIMARY OUTPUT
    x[6]  GM2 synthesis rate                     (nmol/g/hr)
    x[7]  Lysosomal storage load                 (a.u.)
    x[8]  Neuroinflammation proxy                (a.u.)
    x[9]  Microglial activation                  (a.u.)
    x[10] Astrocyte response                     (a.u.)
    x[11] FUS/LIFU permeability window           (binary)  [deterministic]
    x[12] MB cavitation state                    (a.u.)
    x[13] Tight-junction re-closure timer         (0-1)     [deterministic]
    x[14] AAV-T4 transduction efficiency          (fraction, 0-1, durable/non-decaying:
                                                    AAV integrates into post-mitotic
                                                    neurons, so once cells are
                                                    transduced they keep expressing
                                                    even after the vector bolus clears)
    x[15] Cumulative Hex-A restoration            (nmol/mg, cumulative)

x[2], x[11], x[12], x[13] are schedule/gating bookkeeping variables driven
directly by the LIFU sonication schedule and the acoustic model (this
matches the brief: "k_entry_effective ... updated at each LIFU sonication
event") and are assigned/relaxed deterministically rather than integrated.
x[3] and x[14] are saturating, durable cell-population fractions driven by
a deterministic dosing schedule and are integrated with plain Euler
(drift only -- see NOISY_DIMS below for why they are excluded from the
stochastic update). The remaining 10 dimensions are integrated with the
Milstein scheme above.

Parameter provenance
---------------------
Every parameter below is tagged inline with where it actually comes from.
Three tags are used, and the distinction matters:

    # OSMON_FIT      Anchored to a number actually read off a figure/table
                      in Osmon et al., "Treatment of GM2 Gangliosidosis in
                      Adult Sandhoff Mice Using an Intravenous
                      Self-Complementary Hexosaminidase Vector," Current
                      Gene Therapy 22(3), 2021 (PMID 34530708) -- verified
                      by reading the actual PDF. NOTE: this paper reports
                      Hex-A activity as a MUGS/MUGal fluorescent-substrate
                      ratio (Fig. 5A) and GM2 storage as a GM2/GD1a ratio
                      (Fig. 5B) -- it does NOT report absolute nmol/mg/hr
                      or nmol/g values, and has no Table with an enzyme
                      "%WT" figure. Anything tagged OSMON_FIT is therefore
                      a *relative-severity or fold-change* anchor (e.g.
                      "untreated Hex-A activity is ~0.5% of the
                      heterozygote-carrier reference level," "AAV therapy
                      reduced GM2/GD1a by ~60% in the low-dose 16-week
                      cohort"), mapped into this model's nmol/mg/hr and
                      nmol/g unit convention, NOT a literal transcription
                      of an absolute value from the paper. It is also an
                      adult-mouse, ~16-week-endpoint study being used to
                      calibrate an infantile-human, 365-day model -- a
                      cross-species/cross-timescale extrapolation, not a
                      direct fit.
    # LIT_CONSISTENT  Not fitted to a specific reported number, but checked
                      against a real paper's qualitative/order-of-magnitude
                      finding and found consistent (see the parameter's own
                      comment for what was checked and against what).
    # ASSUMED         No literature check performed. Chosen only so the
                      16-D state vector (which needs compartments the brief
                      didn't give numbers for -- plasma/CNS PK, SRT editing
                      kinetics, inflammatory cascade, cavitation-state
                      relaxation, tight-junction closure) produces sane,
                      non-degenerate dynamics (verified by simulation, not
                      by citation).

An earlier version of this file's docstring said k_entry_baseline,
k_entry_boost, V_max, K_m, g_synth, k_deg, and k_clear_enzyme came from "a
validated Osmon et al. fit," per the original task brief. That claim has
NOT been independently verified -- the real Osmon paper contains no
Michaelis-Menten kinetic parameters for Hex-A/GM2 in these units at all, so
V_max/K_m/g_synth cannot actually have come from it. They are left
unchanged from the brief (per explicit instruction) but should be
understood as unverified, not confirmed-Osmon, values.
"""

import numpy as np

from lifu_acoustic import REGION_DEPTHS_MM, REGION_NAMES, compute_eta

# ---------------------------------------------------------------------------
# Primary parameters (given in task brief as "validated Osmon fit" -- that
# provenance claim is UNVERIFIED, see module docstring. Left unchanged from
# the brief since no independent replacement value was found or requested.)
# ---------------------------------------------------------------------------
K_ENTRY_BASELINE = 0.18      # hr^-1               # ASSUMED (per brief, unverified)
K_ENTRY_BOOST = 4.5          # dimensionless        # ASSUMED (per brief, unverified)
V_MAX = 2.3                  # nmol/mg/hr           # ASSUMED (per brief, unverified)
K_M = 180.0                  # nmol/g               # ASSUMED (per brief, unverified)
G_SYNTH = 0.42                # nmol/g/hr           # ASSUMED (per brief, unverified)
K_DEG = 0.003                 # hr^-1               # ASSUMED (per brief, unverified)
K_CLEAR_ENZYME = 0.08         # hr^-1               # ASSUMED (per brief, unverified)

SIGMA_GM2 = 0.04               # x[5] noise level
SIGMA_HEXA = 0.02              # x[4] noise level
SIGMA_OTHER = 0.01             # all other noisy dims

DT = 0.5                       # hr
T_TOTAL = 8760.0               # hr (365 days)
N_STEPS = int(round(T_TOTAL / DT))          # 17520
N_POINTS = N_STEPS + 1                       # 17521, incl. t=0

# LIFU sonication schedule (hours): day 0, 3, 7, 14, 30, each opening the
# BBB for a 6-hour window.
SONICATION_TIMES_HR = np.array([0.0, 72.0, 168.0, 336.0, 720.0])
SONICATION_WINDOW_HR = 6.0

# Fixed generic-FUS boost used only by the pre-LIFU "tri_no_lifu" baseline
# arm (the original, unoptimized tri-modal model this study is refining).
# Not from any paper -- "generic FUS" is our own modeling device, no cited
# study has an equivalent arm to anchor this to.
GENERIC_FUS_BOOST = 1.5        # k_entry multiplier, constant for whole run  # ASSUMED

# ---------------------------------------------------------------------------
# Auxiliary parameters. See module docstring for what OSMON_FIT /
# LIT_CONSISTENT / ASSUMED mean.
# ---------------------------------------------------------------------------
AUX_PARAMS = dict(
    dose_T4=50.0,             # nmol/L, single bolus AAV-T4 dose at t=0            # ASSUMED
    # AAV9 in mice clears from blood over roughly a 6-48 hr window (<1% of
    # starting genome copies remaining by 48 hr) per Zincarelli et al.,
    # "Analysis of AAV serotypes 1-9 mediated gene expression and tropism
    # in mice after systemic injection," Mol Ther 16(6), 2008 -- confirmed
    # by search, but that paper reports a clearance *window*, not a single
    # half-life number, so no precise rate can be fitted. A 0.15/hr rate
    # (~4.6 hr half-life) leaves <0.1% by 48 hr, consistent with (on the
    # fast side of) that window. A ~2 hr half-life would clear >99.9% by
    # ~20 hr, which reads as faster than "still clearing out to 48 hr."
    k_plasma_clear=0.15,      # hr^-1, systemic AAV-T4 clearance               # LIT_CONSISTENT (Zincarelli 2008, order-of-magnitude only)
    k_cns_clear=0.01,         # hr^-1, CNS-compartment AAV-T4 clearance           # ASSUMED
    k_edit=0.004,             # hr^-1, SP2/SRT editing-efficiency ramp rate       # ASSUMED
    k_srt_reduce=0.0005,      # hr^-1, substrate-reduction term coefficient       # ASSUMED
    k_meanrev=0.01,           # hr^-1, GM2 synthesis-rate mean reversion to g_synth  # ASSUMED
    k_store=0.01,             # hr^-1, lysosomal storage build-up from GM2 burden # ASSUMED
    k_store_clear=0.004,      # hr^-1, storage load clearance                     # ASSUMED
    k_inflame=0.02,           # hr^-1, neuroinflammation driven by storage load   # ASSUMED
    k_inflame_clear=0.03,     # hr^-1                                             # ASSUMED
    k_microglia=0.02,         # hr^-1, microglial activation driven by inflammation  # ASSUMED
    k_microglia_clear=0.02,   # hr^-1                                             # ASSUMED
    k_astro=0.015,            # hr^-1, astrocyte response driven by inflammation  # ASSUMED
    k_astro_clear=0.015,      # hr^-1                                             # ASSUMED
    k_cav_build=0.5,          # hr^-1, MB cavitation-state relaxation towards eta # ASSUMED
    k_cav_clear=0.3,          # hr^-1, cavitation-state decay once window closes  # ASSUMED
    # k_transduce / k_transduce_rate jointly set the AAV-therapy response
    # magnitude. Retuned (see recalibration notes below) so that the
    # model's "mono" (AAV-only) arm produces a day-365 GM2 reduction in the
    # same ballpark as the ~60% GM2/GD1a reduction Osmon Fig. 5B reports
    # for the low-dose adult cohort at 16 weeks. This maps a 16-week
    # adult-mouse readout onto a 365-day infantile-human model timeline --
    # a magnitude-matching heuristic, not a literal timepoint fit.
    k_transduce=2.3,          # nmol/mg/hr, max Hex-A production rate at full (x14=1) transduction  # OSMON_FIT (magnitude only, see above)
    k_transduce_rate=4.0e-4,  # (nmol/g)^-1 hr^-1, transduction-efficiency build-up per unit CNS vector  # OSMON_FIT (magnitude only, see above)
    k_restore=0.05,           # hr^-1, cumulative Hex-A restoration accumulator   # ASSUMED
    tau_closure=1.0,          # hr, tight-junction re-closure time constant       # ASSUMED
    # Osmon Fig. 5A: PBS/untreated Sandhoff Hex-A activity (MUGS/MUGal) is
    # ~0.02, vs. ~3.2-5.5 for heterozygote carriers (the closest thing to a
    # "normal" reference this study measured -- true WT was not tested) --
    # a ~0.5% residual-activity ratio. This model has no explicit WT/normal
    # reference state, so we anchor x4_baseline to ~0.5% of the model's own
    # maximum achievable Hex-A activity (k_transduce/k_clear_enzyme = 27.5
    # at full transduction): 0.5% x 27.5 = ~0.14. NOT a transcription of an
    # absolute Osmon value (the paper reports no absolute nmol/mg/hr
    # number) -- the *ratio* is Osmon-anchored, the absolute scale is ours.
    x4_baseline=0.14,         # nmol/mg/hr, residual endogenous Hex-A activity    # OSMON_FIT (ratio only, see above)
    # x5_baseline: Osmon reports GM2 storage as a GM2/GD1a ratio (Fig. 5B),
    # not an absolute nmol/g concentration, so there is no real anchor to
    # convert from. Left at its previous self-consistent value. (A
    # previously-proposed "890 nmol/g" value was rejected for a second
    # reason beyond lacking a source: it is inconsistent with this model's
    # own natural-history dynamics, whose emergent untreated steady state
    # is g_synth/k_deg = 140 nmol/g -- starting above the model's own
    # eventual plateau would make the natural-history arm's GM2 burden
    # *decrease* over time, the opposite of progressive disease.)
    x5_baseline=20.0,         # nmol/g, GM2 burden at t=0                        # ASSUMED
)

# Indices integrated stochastically (Milstein). Excluded (deterministic,
# assigned/updated directly each step):
#   x2, x11, x13  -- schedule/gating bookkeeping (entry rate, window flag, TJ timer)
#   x12           -- MB cavitation state (relaxes toward a schedule-driven target)
#   x3, x14       -- SP2/SRT editing fraction and AAV transduction fraction. These
#                    are saturating, essentially irreversible cell-population
#                    fractions driven by a deterministic dosing schedule. Giving
#                    them ongoing multiplicative (GBM-style) noise for the full
#                    8760 hr run makes them an unbounded random walk once their
#                    drift term hits zero (dosing exhausted), which can erode an
#                    otherwise-durable gene-therapy effect over a long horizon --
#                    not a real biological process. They are therefore integrated
#                    with plain Euler (drift only, no diffusion), consistent with
#                    x2/x11/x13.
NOISY_DIMS = [0, 1, 4, 5, 6, 7, 8, 9, 10, 15]
SIGMA = {
    0: SIGMA_OTHER, 1: SIGMA_OTHER, 4: SIGMA_HEXA,
    5: SIGMA_GM2, 6: SIGMA_OTHER, 7: SIGMA_OTHER, 8: SIGMA_OTHER,
    9: SIGMA_OTHER, 10: SIGMA_OTHER, 15: SIGMA_OTHER,
}

ARM_TYPES = ("natural", "mono", "bi", "tri_no_lifu", "lifu_tri")


def _in_sonication_window(t, sonication_times, window_hr=SONICATION_WINDOW_HR):
    """True if time t (hr) falls within `window_hr` of any scheduled sonication."""
    delta = t - sonication_times
    return bool(np.any((delta >= 0.0) & (delta < window_hr)))


def build_schedule(t_first_hr, n_sessions, spacing_hr):
    """Build a sonication-time array from a 3-parameter timing policy:
    first session at t_first_hr, then n_sessions-1 further sessions spaced
    spacing_hr apart. This is the parameterisation used by the LIFU-timing
    ML model (see generate_timing_data.py / train_timing_model.py) -- it
    reduces "when to administer LIFU" to 3 learnable/optimisable knobs
    instead of an arbitrary list of times.
    """
    n_sessions = int(round(n_sessions))
    return np.array([t_first_hr + k * spacing_hr for k in range(n_sessions)], dtype=float)


def _arm_config(arm):
    """Return (has_aav_dose, has_srt, has_lifu, has_generic_fus) flags."""
    if arm == "natural":
        return False, False, False, False
    if arm == "mono":
        return True, False, False, False
    if arm == "bi":
        return True, True, False, False
    if arm == "tri_no_lifu":
        return True, True, False, True
    if arm == "lifu_tri":
        return True, True, True, False
    raise ValueError(f"Unknown arm '{arm}'")


def _drift(x, has_srt, aux):
    """Drift vector f(x) for all non-gating dims (both the Milstein-integrated
    ones and the deterministic-Euler ones, x3/x14). Returns a length-16
    array; entries for the pure gating/bookkeeping dims (2, 11, 12, 13) are
    unused (computed separately in the step loop)."""
    f = np.zeros(16)
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = x[0:11]
    x12, x14, x15 = x[12], x[14], x[15]

    f[0] = -aux["k_plasma_clear"] * x0
    f[1] = x2 * x0 - aux["k_cns_clear"] * x1
    f[3] = aux["k_edit"] * (1.0 - x3) if has_srt else 0.0
    f[4] = aux["k_transduce"] * x14 - K_CLEAR_ENZYME * x4
    f[5] = x6 - V_MAX * x4 / (K_M + x5) - K_DEG * x5 - aux["k_srt_reduce"] * x3 * x5
    f[6] = aux["k_meanrev"] * (G_SYNTH - x6)
    f[7] = aux["k_store"] * x5 - aux["k_store_clear"] * x7
    f[8] = aux["k_inflame"] * x7 - aux["k_inflame_clear"] * x8
    f[9] = aux["k_microglia"] * x8 - aux["k_microglia_clear"] * x9
    f[10] = aux["k_astro"] * x8 - aux["k_astro_clear"] * x10
    # AAV transduction of post-mitotic neurons is durable (no decay term):
    # x14 only increases, while CNS vector x1 is present, saturating at 1.
    f[14] = aux["k_transduce_rate"] * (1.0 - x14) * x1
    f[15] = aux["k_restore"] * x4
    # f[12] handled separately (relaxation target depends on window/eta, see step loop)
    return f


def simulate_trajectory(arm, region=None, f_mhz=None, P_mpa=None, DC=None, seed=None,
                         dt=DT, T=T_TOTAL, aux=None, depth_mm=None,
                         sonication_times=None, window_hr=SONICATION_WINDOW_HR):
    """Simulate one realisation of the 16-D SDE for a given therapy arm and
    brain region, returning the GM2-burden trajectory x[5] as an
    (N_POINTS,) float32 array.

    Parameters
    ----------
    arm : one of ARM_TYPES
    region : region name (key of lifu_acoustic.REGION_DEPTHS_MM). Ignored if
        `depth_mm` is given directly (used by the timing-sweep scripts, which
        need continuous depth values not tied to a named region).
    f_mhz, P_mpa, DC : LIFU parameters, required iff arm == "lifu_tri"
    seed : RNG seed (int), for reproducibility across the sweep
    depth_mm : override for the fontanelle-to-target depth (mm); takes
        precedence over `region` when given.
    sonication_times : override for the LIFU session start times (hr array).
        Defaults to the brief's fixed schedule (SONICATION_TIMES_HR) if not
        given. Use build_schedule(t_first, n_sessions, spacing) to construct
        one from a 3-parameter timing policy.
    window_hr : BBB-open duration per session (hr). Defaults to 6 (per brief).
    """
    if aux is None:
        aux = AUX_PARAMS
    if depth_mm is None:
        depth_mm = REGION_DEPTHS_MM[region]
    if sonication_times is None:
        sonication_times = SONICATION_TIMES_HR

    has_dose, has_srt, has_lifu, has_generic_fus = _arm_config(arm)

    eta = 0.0
    if has_lifu:
        if f_mhz is None or P_mpa is None or DC is None:
            raise ValueError("f_mhz, P_mpa, DC required for arm='lifu_tri'")
        eta = compute_eta(f_mhz, P_mpa, DC, depth_mm)

    n_steps = int(round(T / dt))
    n_points = n_steps + 1

    rng = np.random.default_rng(seed)
    # Pre-draw all Gaussian increments for the 13 noisy dims at once
    # (vectorised RNG call is much faster than per-step draws).
    dW = rng.standard_normal((n_steps, len(NOISY_DIMS))) * np.sqrt(dt)

    x = np.zeros(16)
    x[0] = aux["dose_T4"] if has_dose else 0.0
    x[4] = aux["x4_baseline"]
    x[5] = aux["x5_baseline"]
    x[6] = G_SYNTH
    x[2] = K_ENTRY_BASELINE * (1.0 + GENERIC_FUS_BOOST) if has_generic_fus else K_ENTRY_BASELINE

    gm2_traj = np.empty(n_points, dtype=np.float32)
    gm2_traj[0] = x[5]

    for step in range(n_steps):
        t = step * dt

        # --- deterministic gating dims (updated first, used by drift this step) ---
        in_window = has_lifu and _in_sonication_window(t, sonication_times, window_hr)
        x[11] = 1.0 if in_window else 0.0
        x[13] = x[13] + dt * (x[11] - x[13]) / aux["tau_closure"]
        if has_lifu:
            x[2] = K_ENTRY_BASELINE * (1.0 + eta * K_ENTRY_BOOST) if in_window else K_ENTRY_BASELINE
        # (x[2] for tri_no_lifu / generic-FUS arms is constant, set at init;
        #  for natural/mono/bi it stays at K_ENTRY_BASELINE.)

        # --- MB cavitation state (deterministic relaxation ODE, Euler) ---
        cav_target = eta if in_window else 0.0
        rate = aux["k_cav_build"] if in_window else aux["k_cav_clear"]
        x[12] = x[12] + dt * rate * (cav_target - x[12])

        # --- drift for every dim (shared by the deterministic and Milstein updates) ---
        fvec = _drift(x, has_srt, aux)

        # --- deterministic (Euler, no noise) saturating fractions ---
        x[3] = x[3] + fvec[3] * dt
        x[14] = x[14] + fvec[14] * dt

        # --- stochastically-integrated dims (Milstein) ---
        for j, i in enumerate(NOISY_DIMS):
            sigma_i = SIGMA[i]
            gi = sigma_i * x[i]
            dWi = dW[step, j]
            milstein_corr = 0.5 * sigma_i * gi * (dWi * dWi - dt)
            x[i] = x[i] + fvec[i] * dt + gi * dWi + milstein_corr

        # keep all state variables physically valid (non-negative; fractions in [0,1])
        np.clip(x, 0.0, None, out=x)
        x[3] = min(x[3], 1.0)
        x[14] = min(x[14], 1.0)

        gm2_traj[step + 1] = x[5]

    return gm2_traj


if __name__ == "__main__":
    # Smoke test: one realisation of each arm in Frontal Cortex.
    for arm in ARM_TYPES:
        kwargs = dict(f_mhz=0.65, P_mpa=0.35, DC=0.01) if arm == "lifu_tri" else {}
        traj = simulate_trajectory(arm, "Frontal Cortex", seed=0, **kwargs)
        print(f"{arm:12s}  GM2(t=0)={traj[0]:8.2f}  GM2(day365)={traj[-1]:8.2f} nmol/g")
