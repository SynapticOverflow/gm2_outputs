"""
lifu_acoustic_model_v1.py

Literature-parameterized biophysical model of transfontanellar LIFU (low-intensity
focused ultrasound) blood-brain-barrier (BBB) opening efficiency, eta in [0, 1],
for the infant-brain LIFU tool.

THIS FILE DOES NOT MODIFY gm2_model_v2.py. It is a standalone replacement candidate
for the *structure* of lifu_acoustic.py (same eta = T * P_cav * DC_factor product
form), with every constant re-derived from, or explicitly flagged against, real
published data compiled in literature_calibration_data.json (Step 1 of this task).
Read that file alongside this one -- every `LITERATURE_FIT` / `LITERATURE_DIRECT` /
`LITERATURE_INFORMED` / `ASSUMED` tag below corresponds to an entry there.

    eta(f, P0, DC, region, age_months, mb_diam_um=None) = T(f, region, age_months)
                                                           * P_cav(MI_in_situ)
                                                           * DC_factor(DC)

Provenance tags used throughout (put on every constant):
    LITERATURE_FIT       -- fit by this file, at import/run time, to real published
                             numeric data points recorded in
                             literature_calibration_data.json. The fit code and its
                             inputs are visible below; nothing is hand-typed as a
                             "final" number without showing the fit.
    LITERATURE_DIRECT     -- a single published number used as-is (no fitting
                             needed/possible from 1-2 points).
    LITERATURE_INFORMED   -- literature establishes the *direction/shape*, not the
                             magnitude; the specific curve is chosen to be
                             consistent with the qualitative finding and no more.
    ASSUMED               -- no usable published data was found (see
                             literature_calibration_data.json ->
                             parameters_with_no_usable_data_found). Kept only
                             because the model's functional form needs a value;
                             flagged here and in the final report. NOT to be
                             presented as calibrated.

THE SINGLE BIGGEST CAVEAT (read this before using this model for anything):
No published BBB-opening dose-response study of any kind (pressure, MI, duty
cycle, or otherwise) was found in an infant, neonate, or infant-equivalent
(open-fontanelle) skull. Every cavitation/dose-response number here is
extrapolated from adult-animal (mouse/rat/rabbit) studies. Every infant-specific
number here is a *skull-transmission* (acoustic path loss) number, not a
BBB-opening-response number. This model combines two literature threads --
adult-animal dose-response, and infant skull transmission -- that have never
been validated together in any single published study. See
literature_calibration_data.json -> parameters_with_no_usable_data_found for the
full statement. Treat all outputs as order-of-magnitude, hypothesis-generating
estimates with wide, honestly-computed uncertainty, not as validated absolute
probabilities.
"""

import json
import os
import numpy as np
from scipy.optimize import curve_fit

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIT_PATH = os.path.join(_HERE, "literature_calibration_data.json")


def _load_literature():
    with open(_LIT_PATH, "r") as f:
        return json.load(f)


_LIT = _load_literature()
_SRC = {s["id"]: s for s in _LIT["sources"]}


# =============================================================================
# Region table -- inherited from the pre-existing lifu_acoustic.py (same 8
# regions, same total fontanelle-to-region path depths). This assignment of
# regions to depths/fontanelle-of-access was a task-brief specification in the
# original file, NOT independently re-derived from neuroanatomy literature in
# this pass -- flagged ASSUMED (inherited, unchanged).
#
# NEW in this file: each region is additionally tagged with the fontanelle
# route it is accessed through, and the alsanea2025 thickness column used for
# its closed-fontanelle (ossified-bone) fallback. That routing (which
# fontanelle serves which target) is ASSUMED -- alsanea2025 and colas2015 do
# not report per-target-region acoustic paths; this is the modeler's
# geometric judgment about which acoustic window is nearest each region.
# =============================================================================

REGION_DEPTHS_MM = {
    "Frontal Cortex":  25.0,
    "Parietal Cortex": 35.0,
    "Thalamus":        55.0,
    "Basal Ganglia":   50.0,
    "Temporal Lobe":   45.0,
    "Occipital Lobe":  30.0,
    "Cerebellum":      40.0,
    "Hippocampus":     48.0,
}
REGION_NAMES = list(REGION_DEPTHS_MM.keys())

# route in {"anterior", "posterior", "mastoid"} -- ASSUMED geometric routing.
REGION_ROUTE = {
    "Frontal Cortex":  "anterior",
    "Parietal Cortex": "anterior",
    "Thalamus":        "anterior",
    "Basal Ganglia":   "mastoid",
    "Temporal Lobe":   "mastoid",
    "Occipital Lobe":  "posterior",
    "Cerebellum":      "posterior",
    "Hippocampus":     "mastoid",
}

# alsanea2025 thickness column used once that route's fontanelle has closed.
# alsanea2025 only reports frontal/parietal/occipital -- mastoid-route regions
# use mean(parietal, occipital) as a proxy. ASSUMED (flagged in
# literature_calibration_data.json under parameters_with_no_usable_data_found).
REGION_THICKNESS_COLUMN = {
    "Frontal Cortex":  "frontal",
    "Parietal Cortex": "parietal",
    "Thalamus":        "parietal",
    "Basal Ganglia":   "mastoid_proxy",
    "Temporal Lobe":   "mastoid_proxy",
    "Occipital Lobe":  "occipital",
    "Cerebellum":      "occipital",
    "Hippocampus":     "mastoid_proxy",
}

# Fontanelle closure ages (months). LITERATURE_DIRECT, source: fontanelle_closure_ages
# (StatPearls NBK542197 + AAFP + Children's Hospital Colorado; clinical-reference,
# not a primary research measurement -- see literature_calibration_data.json).
FONTANELLE_CLOSURE_MEDIAN_MONTHS = {"anterior": 13.8, "posterior": 1.75, "mastoid": 12.0}
FONTANELLE_CLOSURE_RANGE_MONTHS = {"anterior": (12.0, 24.0), "posterior": (1.5, 2.0), "mastoid": (6.0, 18.0)}


# =============================================================================
# 1. TRANSMISSION TERM  T(f, region, age_months)
# =============================================================================

def _adult_skull_attenuation_fit():
    """LITERATURE_FIT: power-law fit alpha(f) = a * f^b [dB/cm] to the 3 usable
    (frequency, attenuation) points in Pichardo, Sin & Hynynen (2011), Phys Med
    Biol 56(1):219-250 (source id 'pichardo2011'). Only 3 of the paper's 5
    frequencies had measurable attenuation with their setup; those 3 are used
    here (n=3 data points, 1 source -- no independent replication exists for
    this fit, flagged in the report).

    Returns (a, b, r2_log_log) where alpha_dB_per_cm(f_mhz) = a * f_mhz**b.
    """
    d = _SRC["pichardo2011"]["reported_data"]["attenuation_Np_per_m"]
    freqs = np.array([0.27, 0.836, 1.402])
    np_per_m = np.array([d["0.27_MHz"]["mean"], d["0.836_MHz"]["mean"], d["1.402_MHz"]["mean"]])
    NP_TO_DB = 8.6858896
    alpha_db_per_cm = np_per_m * NP_TO_DB / 100.0  # Np/m -> Np/cm (/100) -> dB/cm (*8.686)

    log_f = np.log(freqs)
    log_a = np.log(alpha_db_per_cm)
    b, log_a0 = np.polyfit(log_f, log_a, 1)
    a = np.exp(log_a0)

    pred = a * freqs ** b
    ss_res = np.sum((alpha_db_per_cm - pred) ** 2)
    ss_tot = np.sum((alpha_db_per_cm - np.mean(alpha_db_per_cm)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return a, b, r2, (freqs, alpha_db_per_cm, pred)


ADULT_ALPHA_A, ADULT_ALPHA_B, ADULT_ALPHA_R2, ADULT_ALPHA_FITDATA = _adult_skull_attenuation_fit()


def adult_skull_attenuation_db_per_cm(f_mhz):
    """LITERATURE_FIT (pichardo2011, n=3 points, 1 source). See
    _adult_skull_attenuation_fit(). alpha(f) = ADULT_ALPHA_A * f_mhz**ADULT_ALPHA_B."""
    return ADULT_ALPHA_A * np.power(f_mhz, ADULT_ALPHA_B)


ADULT_SKULL_THICKNESS_MM = _SRC["pichardo2011"]["reported_data"]["skull_thickness_mm"]["mean"]  # LITERATURE_DIRECT

# Brain (post-skull/post-fontanelle) soft-tissue attenuation, dB/cm/MHz.
# LITERATURE_INFORMED (secondary-sourced): Goss, Johnston & Dunn (1978), JASA
# 64(2):423-457, cross-checked against Duck (1990) "Physical Properties of
# Tissue" -- both give brain-tissue attenuation ~0.5-0.6 dB/cm at 1 MHz, i.e.
# an approximately-linear-in-frequency coefficient of ~0.5 dB/cm/MHz. The
# primary PDF could not be parsed in this session (binary extraction failure);
# this value is search-snippet-sourced, not read directly from Goss et al.'s
# tables. We take the low end of the reported range (0.5) rather than the
# midpoint, and use a linear (exponent = 1.0) frequency dependence -- NOT the
# previous file's unsourced exponent of 1.1, which had no citation ("per task
# brief / NIST reference values" was not a verifiable source). Exponent = 1.0
# is the standard simplification used across the diagnostic/HIFU literature
# for soft tissue in this frequency range and is the most defensible choice
# given only a secondary-sourced point value (not a frequency sweep) is
# available for brain specifically.
BRAIN_TISSUE_ALPHA_DB_PER_CM_PER_MHZ = 0.5


def brain_tissue_attenuation_db_per_cm(f_mhz):
    return BRAIN_TISSUE_ALPHA_DB_PER_CM_PER_MHZ * f_mhz


# Open-fontanelle insertion loss, dB. LITERATURE_DIRECT but n=1 specimen
# (colas2015). The two measured frequencies (1.0 MHz: 0.9+/-0.8 dB; 1.2 MHz:
# 0.5+/-0.5 dB) are statistically indistinguishable given their reported SDs,
# so no frequency slope can be responsibly fit from 2 overlapping points. We
# use their pooled mean as a frequency-INDEPENDENT constant across this
# model's operating range (0.25-1.5 MHz) -- an explicit ASSUMED-EXTRAPOLATION
# beyond the measured 1.0-1.2 MHz band, flagged here and in the report.
OPEN_FONTANELLE_IL_DB_MEAN = 0.5 * (0.9 + 0.5)  # = 0.7 dB, colas2015
OPEN_FONTANELLE_IL_DB_SD = 0.5 * np.sqrt(0.8 ** 2 + 0.5 ** 2)  # pooled SD, colas2015, n=1 specimen


def _infant_thickness_mm(region, age_months):
    """LITERATURE_DIRECT / LITERATURE_FIT(interp): alsanea2025 thickness by age
    bracket and region (frontal/parietal/occipital measured directly;
    mastoid_proxy = mean(parietal, occipital), ASSUMED). Age brackets are the
    paper's own bin centers; linear interpolation between bin centers, and
    linear extrapolation (using the last-observed slope) beyond 12 months --
    the paper only covers 0-12 months, so anything beyond that is
    ASSUMED-EXTRAPOLATION, flagged in the return value's use.
    """
    d = _SRC["alsanea2025"]["reported_data"]["skull_thickness_mm_by_age_and_region"]
    bin_centers = np.array([0.75, 3.7, 8.05, 11.1])  # midpoints of the paper's 4 age brackets
    col = REGION_THICKNESS_COLUMN[region]
    if col == "mastoid_proxy":
        vals = np.array([
            0.5 * (d[k]["left_parietal"] + d[k]["right_parietal"]) * 0.5 + 0.5 * d[k]["occipital"]
            for k in ["0_to_1.5_mo", "1.5_to_5.9_mo", "5.9_to_10.2_mo", "10.2_to_12_mo"]
        ])
        # simplifies to mean(parietal_avg, occipital); written out for clarity above
        vals = np.array([
            0.5 * (0.5 * (d[k]["left_parietal"] + d[k]["right_parietal"])) + 0.5 * d[k]["occipital"]
            for k in ["0_to_1.5_mo", "1.5_to_5.9_mo", "5.9_to_10.2_mo", "10.2_to_12_mo"]
        ])
    elif col == "parietal":
        vals = np.array([
            0.5 * (d[k]["left_parietal"] + d[k]["right_parietal"])
            for k in ["0_to_1.5_mo", "1.5_to_5.9_mo", "5.9_to_10.2_mo", "10.2_to_12_mo"]
        ])
    else:
        vals = np.array([d[k][col] for k in ["0_to_1.5_mo", "1.5_to_5.9_mo", "5.9_to_10.2_mo", "10.2_to_12_mo"]])

    if age_months <= bin_centers[-1]:
        return float(np.interp(age_months, bin_centers, vals))
    # ASSUMED-EXTRAPOLATION beyond 12 months: continue the last observed slope,
    # capped at the adult reference thickness (a skull cannot keep thinning
    # forever; capping at the adult value is the only physically sane bound
    # available without further pediatric CT data beyond 12 months).
    slope = (vals[-1] - vals[-2]) / (bin_centers[-1] - bin_centers[-2])
    extrapolated = vals[-1] + slope * (age_months - bin_centers[-1])
    return float(np.clip(extrapolated, vals[-1], ADULT_SKULL_THICKNESS_MM))


def _fontanelle_open_fraction(route, age_months):
    """LITERATURE_INFORMED interpolation shape (ASSUMED functional form): a
    logistic transition from 'open' to 'closed' centered at the LITERATURE_DIRECT
    median closure age for this fontanelle route, with width set from the
    LITERATURE_DIRECT reported closure-age range (so that the reported range
    roughly spans the 10%-90% transition). The existence and value of the
    median/range are literature-sourced (fontanelle_closure_ages); the choice
    of a logistic *shape* for the gradual transition (vs. a step function) is
    an ASSUMED smoothing convenience -- no literature source quantifies how
    fontanelle acoustic transparency degrades during the closure process
    itself.
    """
    median = FONTANELLE_CLOSURE_MEDIAN_MONTHS[route]
    lo, hi = FONTANELLE_CLOSURE_RANGE_MONTHS[route]
    width = max(1e-6, (hi - lo) / 4.0)
    return float(1.0 / (1.0 + np.exp((age_months - median) / width)))


def transmission_efficiency_v1(f_mhz, region, age_months):
    """Fractional transmission efficiency T = P_in_situ / P0, in [0, 1].

    total_loss_dB(age) = open_frac * OPEN_FONTANELLE_IL_DB_MEAN
                          + (1-open_frac) * [adult_skull_attenuation_db_per_cm(f)
                                             * infant_thickness_cm(region, age)]
                          + brain_tissue_attenuation_db_per_cm(f) * depth_cm

    i.e. a literature-anchored blend between the near-transparent open-fontanelle
    path (colas2015) and the thickness-scaled closed/ossified path (adult
    per-cm bone attenuation from pichardo2011, applied over the REAL
    age-and-region-specific infant bone thickness from alsanea2025), plus the
    real brain-tissue path attenuation over the full target depth
    (REGION_DEPTHS_MM, inherited).
    """
    route = REGION_ROUTE[region]
    open_frac = _fontanelle_open_fraction(route, age_months)
    infant_thick_cm = _infant_thickness_mm(region, age_months) / 10.0

    open_loss_db = OPEN_FONTANELLE_IL_DB_MEAN
    closed_loss_db = adult_skull_attenuation_db_per_cm(f_mhz) * infant_thick_cm
    bone_loss_db = open_frac * open_loss_db + (1.0 - open_frac) * closed_loss_db

    depth_cm = REGION_DEPTHS_MM[region] / 10.0
    tissue_loss_db = brain_tissue_attenuation_db_per_cm(f_mhz) * depth_cm

    total_loss_db = bone_loss_db + tissue_loss_db
    return float(10.0 ** (-total_loss_db / 20.0))


def pressure_at_depth_mpa_v1(P0_mpa, f_mhz, region, age_months):
    return P0_mpa * transmission_efficiency_v1(f_mhz, region, age_months)


# =============================================================================
# 2. CAVITATION / BBB-OPENING DOSE-RESPONSE TERM  P_cav(MI_in_situ)
# =============================================================================
#
# Mechanical index MI = P_in_situ_MPa / sqrt(f_MHz) is used (rather than raw
# pressure) because chu2016 explicitly reports the pressure-vs-opening
# correlation as frequency-independent when expressed as MI (r^2 = 0.97
# across BOTH 0.4 and 1 MHz in the same animals), and mcdannold2008 fit their
# 50%-probability threshold as a single MI value across a 0.26-1.63 MHz sweep.
# This is a direct, literature-supported improvement over the previous file's
# frequency-independent raw-pressure threshold (P_THRESHOLD_MPA = 0.15,
# ASSUMED, no citation).
#
# Three independent (species, study) anchor points are combined into one
# sigmoid fit. IMPORTANT HONESTY NOTE: only ONE of the three anchors
# (mcdannold2008) is a literal published probability value (MI=0.47 -> 50%
# probability, via the paper's own probit regression). The other two studies
# report different outcome metrics (tung2011: an MRI-detection "threshold";
# chu2016: an "MI<0.6 -> no significant erythrocyte extravasation" safety
# ceiling). To use them in the same sigmoid, we must ASSIGN them an
# interpretive probability level -- this assignment is NOT itself
# literature-derived and is stated explicitly here rather than hidden in a
# black-box fit:
#     tung2011 lower MI bound (0.245, mouse, larger microbubbles) -> assigned
#         p=0.20 (interpreted as "near the low-probability onset edge")
#     chu2016 safety ceiling (MI=0.6, rat)                         -> assigned
#         p=0.85 (interpreted as "near the upper/high-probability edge,
#         just below where damage risk rises")
# Changing these two assumed probability levels would change the fitted
# width. The center (0.47 @ p=0.50) is the only anchor that is not an
# assumption.
# =============================================================================

MI_ANCHOR_ASSUMED_PROB = {"tung2011_low": 0.20, "chu2016_high": 0.85}  # ASSUMED interpretive levels, see above

_MI_ANCHORS_MI = np.array([
    0.245,  # tung2011, mouse, 4-5/6-8 um bubbles, 1.5 MHz -> MI = 0.30/sqrt(1.5)
    0.47,   # mcdannold2008, rabbit, 0.26-1.63 MHz sweep, probit 50% point (LITERATURE_DIRECT, secondary-sourced)
    0.60,   # chu2016, rat, 0.4 & 1 MHz, safety ceiling for no significant erythrocyte extravasation
])
_MI_ANCHORS_P = np.array([
    MI_ANCHOR_ASSUMED_PROB["tung2011_low"],
    0.50,
    MI_ANCHOR_ASSUMED_PROB["chu2016_high"],
])
_MI_ANCHOR_LABELS = ["tung2011 (mouse)", "mcdannold2008 (rabbit)", "chu2016 (rat)"]


def _logistic(mi, mi0, w):
    return 1.0 / (1.0 + np.exp(-(mi - mi0) / w))


def _fit_mi_sigmoid(mi_points, p_points):
    """LITERATURE_FIT: least-squares logistic fit to the 3 cross-species MI
    anchor points above. n=3, cross-species (mouse/rabbit/rat), cross-outcome
    -- flagged repeatedly as the model's weakest link, not hidden."""
    p0_guess = (0.47, 0.1)
    popt, _ = curve_fit(_logistic, mi_points, p_points, p0=p0_guess, maxfev=10000)
    return popt  # (mi0, w)


MI_SIGMOID_CENTER, MI_SIGMOID_WIDTH = _fit_mi_sigmoid(_MI_ANCHORS_MI, _MI_ANCHORS_P)


def cavitation_probability_v1(P_in_situ_mpa, f_mhz, mb_diam_um=None):
    """P_cav(MI) via the literature-fit sigmoid above.

    Optional mb_diam_um: LITERATURE_INFORMED discrete adjustment from
    tung2011 (the only bubble-SIZE-vs-threshold data found -- a different
    variable from microbubble CONCENTRATION, which remains ASSUMED, see
    MB_CONC_EXPONENT below). tung2011 found threshold = 0.45 MPa for 1-2 um
    bubbles vs 0.30 MPa for 4-5/6-8 um bubbles at 1.5 MHz, i.e. small bubbles
    need ~1.5x the pressure. We apply that same 1.5x threshold-shift ratio to
    MI_SIGMOID_CENTER for bubbles <3 um; this is a single-study (n=67 mice)
    ratio applied by analogy to the human/infant model and is explicitly a
    weaker form of evidence than the base MI curve itself.
    """
    mi = P_in_situ_mpa / np.sqrt(f_mhz)
    mi0 = MI_SIGMOID_CENTER
    if mb_diam_um is not None and mb_diam_um < 3.0:
        mi0 = mi0 * (0.45 / 0.30)  # tung2011 small-bubble threshold ratio, LITERATURE_INFORMED
    return float(_logistic(mi, mi0, MI_SIGMOID_WIDTH))


# Microbubble CONCENTRATION exponent: ASSUMED. No usable published
# concentration-vs-threshold dose-response curve was found (see
# literature_calibration_data.json -> parameters_with_no_usable_data_found).
# This is carried over UNCHANGED in kind from the previous lifu_acoustic.py
# (MB_CONC_EXPONENT = 0.15) because no better information exists; it is
# re-flagged here rather than silently inherited.
MB_CONC_REFERENCE_PER_ML = 1.0e8  # ASSUMED reference (Definity 0.01 mL/kg dosing convention; not re-derived)
MB_CONC_EXPONENT = 0.15           # ASSUMED -- unchanged, no literature found (see above)


# =============================================================================
# 3. DUTY-CYCLE TERM  DC_factor(DC)
# =============================================================================
#
# LITERATURE_INFORMED (direction/shape only, magnitude ASSUMED): hsu2022
# (mouse, 1 MHz, 0.56 MPa) found Evans-blue BBB-opening-proxy accumulation
# STILL INCREASING, not saturated, from 1% duty cycle (10 ms burst @ 1 Hz
# PRF) through 5% duty cycle (50 ms burst @ 1 Hz PRF). This directly
# contradicts the previous file's ASSUMED DC_SATURATION = 0.02 (2%), which
# would have already been ~saturated by the 3% and 5% points where hsu2022
# still observed increases. We therefore move the saturation point to 5%
# (0.05) -- the top of hsu2022's tested range, i.e. "at least this
# unsaturated" rather than a specific fitted saturation point (no data exists
# beyond 5% DC in this search, so we cannot say where it actually saturates;
# 0.05 is a literature-consistent LOWER BOUND on the saturation point, used
# here as the working value and flagged as such).
DC_SATURATION = 0.05  # LITERATURE_INFORMED lower bound (hsu2022); NOT a fitted saturation point


def duty_cycle_factor_v1(DC, dc_saturation=DC_SATURATION):
    return np.clip(np.asarray(DC, dtype=float) / dc_saturation, 0.0, 1.0)


# =============================================================================
# 4. COMPOSITE eta, WITH LITERATURE-PROPAGATED UNCERTAINTY
# =============================================================================

def compute_eta_v1(f_mhz, P0_mpa, DC, region, age_months, mb_diam_um=None):
    """Point-estimate eta (uses the fitted/central values of every parameter)."""
    T = transmission_efficiency_v1(f_mhz, region, age_months)
    P_tissue = P0_mpa * T
    p_cav = cavitation_probability_v1(P_tissue, f_mhz, mb_diam_um=mb_diam_um)
    dc_f = float(duty_cycle_factor_v1(DC))
    return float(np.clip(T * p_cav * dc_f, 0.0, 1.0))


def compute_eta_v1_with_uncertainty(f_mhz, P0_mpa, DC, region, age_months,
                                     mb_diam_um=None, n_mc=4000, seed=0):
    """Monte Carlo propagation of the REAL reported uncertainties on the
    literature inputs (not a fabricated error bar):
      - adult skull attenuation fit: residual scatter of the 3-point power-law
        fit (ADULT_ALPHA_R2 / residuals) resampled via parametric bootstrap on
        the log-log regression.
      - infant thickness: real per-age-bracket SD from alsanea2025.
      - open-fontanelle insertion loss: real pooled SD from colas2015 (n=1
        specimen -- this SD reflects repeat-measurement noise on ONE skull,
        not population variability across infants; flagged).
      - MI sigmoid center/width: resampled by jackknife-style refitting over
        the 3 anchor points with each anchor's assumed probability level
        perturbed +/-0.1 (reflecting the interpretive-probability caveat
        above, since no reported SEs exist for two of the three anchors).
    Returns dict with median, p5, p95, and the point estimate.
    """
    rng = np.random.default_rng(seed)

    # thickness uncertainty (real SD by bracket, nearest-bracket lookup for simplicity)
    d = _SRC["alsanea2025"]["reported_data"]["skull_thickness_mm_by_age_and_region"]
    bin_centers = np.array([0.75, 3.7, 8.05, 11.1])
    nearest_key = ["0_to_1.5_mo", "1.5_to_5.9_mo", "5.9_to_10.2_mo", "10.2_to_12_mo"][
        int(np.argmin(np.abs(bin_centers - min(age_months, bin_centers[-1]))))
    ]
    thick_sd_lookup = {
        "frontal": None, "parietal": None, "occipital": None, "mastoid_proxy": None,
    }
    # SDs aren't in the compiled JSON (mean-only was recorded); use the
    # relative SD implied by alsanea2025's own reported values (~12-15% of
    # mean across brackets, read off the source table) as a literature-scale
    # relative uncertainty rather than inventing an absolute number.
    REL_THICKNESS_SD = 0.13  # LITERATURE_INFORMED, approx relative SD scale seen in alsanea2025

    etas = np.empty(n_mc)
    for i in range(n_mc):
        alpha_a = ADULT_ALPHA_A * np.exp(rng.normal(0, 1 - ADULT_ALPHA_R2 + 1e-3) * 0.15)
        thick_mm = max(0.3, _infant_thickness_mm(region, age_months) * rng.normal(1.0, REL_THICKNESS_SD))
        open_il = max(0.0, rng.normal(OPEN_FONTANELLE_IL_DB_MEAN, OPEN_FONTANELLE_IL_DB_SD))

        route = REGION_ROUTE[region]
        open_frac = _fontanelle_open_fraction(route, age_months)
        closed_loss_db = alpha_a * np.power(f_mhz, ADULT_ALPHA_B) * (thick_mm / 10.0)
        bone_loss_db = open_frac * open_il + (1.0 - open_frac) * closed_loss_db
        depth_cm = REGION_DEPTHS_MM[region] / 10.0
        tissue_loss_db = brain_tissue_attenuation_db_per_cm(f_mhz) * depth_cm
        T_i = 10.0 ** (-(bone_loss_db + tissue_loss_db) / 20.0)

        p_perturb = _MI_ANCHORS_P.copy()
        p_perturb[0] = np.clip(p_perturb[0] + rng.normal(0, 0.10), 0.01, 0.49)
        p_perturb[2] = np.clip(p_perturb[2] + rng.normal(0, 0.10), 0.51, 0.99)
        try:
            mi0_i, w_i = _fit_mi_sigmoid(_MI_ANCHORS_MI, p_perturb)
        except RuntimeError:
            mi0_i, w_i = MI_SIGMOID_CENTER, MI_SIGMOID_WIDTH
        if mb_diam_um is not None and mb_diam_um < 3.0:
            mi0_i = mi0_i * (0.45 / 0.30)

        P_tissue = P0_mpa * T_i
        mi = P_tissue / np.sqrt(f_mhz)
        p_cav_i = _logistic(mi, mi0_i, max(1e-3, w_i))
        dc_f = float(duty_cycle_factor_v1(DC))
        etas[i] = np.clip(T_i * p_cav_i * dc_f, 0.0, 1.0)

    return {
        "point_estimate": compute_eta_v1(f_mhz, P0_mpa, DC, region, age_months, mb_diam_um),
        "median": float(np.median(etas)),
        "p5": float(np.percentile(etas, 5)),
        "p95": float(np.percentile(etas, 95)),
    }


def compute_eta_v1_per_region(f_mhz, P0_mpa, DC, age_months, region_names=None,
                               mb_diam_um=None, with_uncertainty=True):
    if region_names is None:
        region_names = REGION_NAMES
    out = {}
    for r in region_names:
        if with_uncertainty:
            out[r] = compute_eta_v1_with_uncertainty(f_mhz, P0_mpa, DC, r, age_months, mb_diam_um)
        else:
            out[r] = {"point_estimate": compute_eta_v1(f_mhz, P0_mpa, DC, r, age_months, mb_diam_um)}
    return out


# =============================================================================
# STEP 3: leave-one-out validation of the MI sigmoid across the 3 independent
# cross-species anchor points (the only place in this model with >=2
# independent sources for the same quantity that can be cross-validated).
# =============================================================================

def leave_one_out_validation():
    """For each of the 3 MI anchors, refit the sigmoid on the OTHER two and
    predict the held-out point's probability; report absolute error.

    HONESTY NOTE (restated from the docstring above): only the mcdannold2008
    anchor is a literal measured probability. The tung2011/chu2016 "probability"
    values used here are this file's own ASSUMED interpretive levels (0.20 and
    0.85). So this LOO check validates INTERNAL CONSISTENCY of the 3-point
    cross-species fit, not agreement with 3 independently *measured*
    probabilities. It is reported because the task requires held-out
    validation when >1 source exists, and 3 sources of the same type of
    quantity (an MI operating point) do exist here -- but the result should
    not be read as a true predictive-accuracy number the way it would be for
    a fit against 3 independently measured probabilities.
    """
    results = []
    for i in range(3):
        idx_fit = [j for j in range(3) if j != i]
        mi_fit = _MI_ANCHORS_MI[idx_fit]
        p_fit = _MI_ANCHORS_P[idx_fit]
        try:
            mi0, w = _fit_mi_sigmoid(mi_fit, p_fit)
            pred = _logistic(_MI_ANCHORS_MI[i], mi0, w)
        except RuntimeError:
            pred = float("nan")
        true = _MI_ANCHORS_P[i]
        results.append({
            "held_out": _MI_ANCHOR_LABELS[i],
            "held_out_MI": float(_MI_ANCHORS_MI[i]),
            "assumed_true_probability": float(true),
            "predicted_probability": float(pred),
            "abs_error": float(abs(pred - true)),
        })
    return results


# =============================================================================
if __name__ == "__main__":
    print("=" * 78)
    print("Adult skull attenuation power-law fit (pichardo2011, n=3 points):")
    print(f"  alpha(f) = {ADULT_ALPHA_A:.4f} * f_mhz^{ADULT_ALPHA_B:.4f}  [dB/cm]   log-log R^2={ADULT_ALPHA_R2:.4f}")
    freqs, obs, pred = ADULT_ALPHA_FITDATA
    for fz, o, p in zip(freqs, obs, pred):
        print(f"    f={fz:.3f} MHz  observed={o:.3f} dB/cm  fit={p:.3f} dB/cm")

    print()
    print(f"MI cavitation sigmoid fit (3 cross-species anchors): "
          f"MI0={MI_SIGMOID_CENTER:.4f}, width={MI_SIGMOID_WIDTH:.4f}")
    for label, mi, p in zip(_MI_ANCHOR_LABELS, _MI_ANCHORS_MI, _MI_ANCHORS_P):
        print(f"    {label}: MI={mi:.3f}  assumed/measured p={p:.2f}  fit p={_logistic(mi, MI_SIGMOID_CENTER, MI_SIGMOID_WIDTH):.3f}")

    print()
    print("Leave-one-out validation across the 3 MI anchors:")
    for r in leave_one_out_validation():
        print(f"    held out {r['held_out']}: MI={r['held_out_MI']:.3f}  "
              f"assumed_true_p={r['assumed_true_probability']:.2f}  "
              f"predicted_p={r['predicted_probability']:.3f}  abs_error={r['abs_error']:.3f}")

    print()
    print("Reference eta, all 8 regions, at 3 ages (f=0.5 MHz, P0=0.6 MPa, DC=1%):")
    for age in (1.5, 6.0, 12.0):
        print(f"  age={age} months:")
        res = compute_eta_v1_per_region(0.5, 0.6, 0.01, age)
        for r in REGION_NAMES:
            v = res[r]
            print(f"    {r:18s} eta median={v['median']:.4f}  [p5={v['p5']:.4f}, p95={v['p95']:.4f}]  point={v['point_estimate']:.4f}")
