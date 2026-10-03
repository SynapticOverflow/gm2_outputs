"""
lifu_acoustic_model_v2.py

Extends lifu_acoustic_model_v1.py with the 2026-09-12 task items 1-3:

  Item 1 (primary source): mcdannold2008's MI=0.47 anchor was secondary-sourced
      in v1. The primary text (PMC2442477) has now been read directly:
      MI_50% = 0.46 (95% CI 0.42-0.50), not 0.47. This file re-fits the MI
      sigmoid on the corrected anchor. See literature_calibration_data.json ->
      sources -> mcdannold2008 (updated verification field).

  Item 2 (infant/neonatal BBB-opening dose-response, broadened search): still
      no direct infant/neonatal FUS-BBB dose-response study was found (see
      literature_calibration_data.json -> infant_bbb_dose_response_scaling_2026-09-12
      for the full derivation). Per the task's own escalation order, Option A
      (age-scaling from general, non-FUS BBB-permeability-maturation
      literature) was reached and used, combined with Option C (the model now
      reports a BOUNDED RANGE, not a point estimate, for the infant
      biological-scaling factor) because the source evidence itself is
      bidirectional (tracer-size dependent): a real citation
      (hanafy_dietrich2026) gives a 2.0x neonate:adult ratio for a
      small-molecule tracer and a 1.0x (no difference) ratio for a
      macromolecule (BSA) closer in size to this project's actual FUS-relevant
      cargo (AAV capsids). Both numbers are used as the two ends of a bound on
      the MI sigmoid center for the model's target 0-12 month age range,
      NOT averaged or collapsed into one point value.

  Item 3 (existing CT data): re-checked alsanea2025's primary text and the
      filesystem for any not-yet-used infant CT data (density/Hounsfield/
      fontanelle-area). None was found beyond the geometric thickness table
      already used in v1's transmission term -- see
      literature_calibration_data.json -> sources -> alsanea2025 (unchanged)
      and this task's chat report. No additional CT-derived input was
      available to add.

THIS FILE DOES NOT MODIFY gm2_model_v2.py, lifu_acoustic.py, or
lifu_acoustic_model_v1.py (v1 is kept, unmodified, as the 2026-09-11
historical record -- same pattern as lifu_demo_disclaimer_patch.html ->
_patch_v2.html). Everything in v1 not explicitly discussed above (transmission
term T, duty-cycle term, region table, Monte Carlo transmission-uncertainty
machinery) is reused unchanged by importing v1.
"""

import numpy as np
from scipy.optimize import curve_fit

import lifu_acoustic_model_v1 as v1

# Re-export the pieces of v1 that are unchanged in this pass, so callers of v2
# don't need to also import v1 directly.
REGION_NAMES = v1.REGION_NAMES
REGION_DEPTHS_MM = v1.REGION_DEPTHS_MM
REGION_ROUTE = v1.REGION_ROUTE
transmission_efficiency_v2 = v1.transmission_efficiency_v1
duty_cycle_factor_v2 = v1.duty_cycle_factor_v1
_logistic = v1._logistic
_SRC = v1._SRC


# =============================================================================
# ITEM 1: corrected mcdannold2008 anchor, primary-sourced 2026-09-12.
# =============================================================================
_MI_ANCHORS_MI = np.array([0.245, 0.46, 0.60])  # tung2011, mcdannold2008 (CORRECTED 0.47->0.46), chu2016
_MI_ANCHORS_P = v1._MI_ANCHORS_P  # unchanged: assumed interpretive probability levels (0.20, 0.50, 0.85)
_MI_ANCHOR_LABELS = v1._MI_ANCHOR_LABELS
MCDANNOLD2008_MI_95CI = (0.42, 0.50)  # PMC2442477, primary text, item 1

MI_SIGMOID_CENTER, MI_SIGMOID_WIDTH = v1._fit_mi_sigmoid(_MI_ANCHORS_MI, _MI_ANCHORS_P)


# =============================================================================
# ITEM 2: bounded infant biological-scaling factor on the MI sigmoid center.
#
# See literature_calibration_data.json -> infant_bbb_dose_response_scaling_2026-09-12
# for the full derivation and honesty caveats. Summary:
#   - hanafy_dietrich2026 (mouse, general/passive BBB tracer-permeability
#     maturation, NOT FUS): FM1-43 (small molecule) neonatal:adult(P25) ratio
#     = 0.18/0.09 = 2.0 (exact table values, SD given). BSA (66 kDa
#     macromolecule, closer in size to this project's actual AAV cargo)
#     neonatal:adult ratio = 1.0 (no measurable difference).
#   - ASSUMED (stated, not hidden): MI0_infant = MI0_adult / k, i.e. a
#     baseline-permeability ratio maps linearly and inversely onto the
#     mechanical dose needed for FUS-cavitation-forced opening. This
#     proportionality is a physically-motivated analogy, not a validated
#     pharmacodynamic law -- no source found here establishes this functional
#     form for cavitation-induced (vs. passive) BBB opening.
#   - k=1.0 (macromolecule/AAV-relevant, LITERATURE_INFORMED-by-analogy) is the
#     CONSERVATIVE bound: infant no more permissive than the adult curve.
#   - k=2.0 (small-molecule, LITERATURE_INFORMED-by-analogy) is the
#     PERMISSIVE bound: infant opens at half the adult MI.
#   - No literature evidence supports k<1.0 (infant LESS permeable than
#     adult); that direction is not included in the bound because nothing
#     found here supports it -- this is not silently defaulted, it is the
#     stated absence of evidence in that direction.
#   - Applied uniformly across the model's whole 0-12 month target range
#     (a coarse binary infant-vs-adult application), NOT as a continuous
#     function of age within infancy: hanafy_dietrich2026 is a MOUSE
#     postnatal-day timeline (E18-P25) with no citation found in this search
#     that maps mouse postnatal days onto human infant months for BBB
#     maturation specifically. Inventing an interpolated human-month curve
#     from that mouse timeline would be exactly the kind of unsupported
#     precision this task explicitly prohibits.
# =============================================================================
INFANT_MI_K_CONSERVATIVE = 1.0  # BSA / macromolecule-consistent bound -> MI0_infant = MI0_adult (upper bound, least permissive)
INFANT_MI_K_PERMISSIVE = 2.0    # FM1-43 / small-molecule-consistent bound -> MI0_infant = MI0_adult/2 (lower bound, most permissive)
INFANT_SCALING_APPLIES_UP_TO_AGE_MONTHS = 12.0  # this model's whole target range; see docstring


def infant_mi_sigmoid_center_bounds(age_months):
    """Returns (mi0_conservative, mi0_permissive) -- the two ends of the
    Option-A/C bounded range on the MI sigmoid center, for age_months within
    the model's 0-12 month infant/open-fontanelle target range. Outside that
    range (age_months > 12), both bounds collapse to the adult value: the
    hanafy_dietrich2026 analogy is specifically about the neonatal/infant
    period, and this model's own alsanea2025-based transmission term already
    treats >12 months as adult-converging, so applying an infant-specific
    biological bound there would be inconsistent with the rest of the model.
    """
    if age_months <= INFANT_SCALING_APPLIES_UP_TO_AGE_MONTHS:
        mi0_conservative = MI_SIGMOID_CENTER / INFANT_MI_K_CONSERVATIVE
        mi0_permissive = MI_SIGMOID_CENTER / INFANT_MI_K_PERMISSIVE
    else:
        mi0_conservative = MI_SIGMOID_CENTER
        mi0_permissive = MI_SIGMOID_CENTER
    return float(mi0_conservative), float(mi0_permissive)


def cavitation_probability_v2_bounds(P_in_situ_mpa, f_mhz, age_months, mb_diam_um=None):
    """Returns (p_cav_conservative, p_cav_permissive): the cavitation
    probability at the two ends of the item-2 infant biological-scaling
    bound. p_cav_permissive >= p_cav_conservative always (permissive bound =
    lower MI0 = higher probability at the same applied MI)."""
    mi = P_in_situ_mpa / np.sqrt(f_mhz)
    mi0_cons, mi0_perm = infant_mi_sigmoid_center_bounds(age_months)
    if mb_diam_um is not None and mb_diam_um < 3.0:
        shift = 0.45 / 0.30  # tung2011 small-bubble shift, unchanged from v1
        mi0_cons *= shift
        mi0_perm *= shift
    p_cons = _logistic(mi, mi0_cons, MI_SIGMOID_WIDTH)
    p_perm = _logistic(mi, mi0_perm, MI_SIGMOID_WIDTH)
    return float(p_cons), float(p_perm)


# =============================================================================
# ITEM 3: existing CT data (alsanea2025) re-checked; no additional usable
# density/Hounsfield/fontanelle-area data was found beyond the geometric
# thickness table v1 already uses. No code change results from item 3; the
# transmission term is reused unchanged from v1 (see re-export above).
# =============================================================================


# =============================================================================
# Composite eta with the item-2 bounded range, ON TOP OF v1's existing
# literature-uncertainty Monte Carlo (transmission-term fit residuals, infant
# thickness SD, open-fontanelle IL SD, MI-anchor interpretive-probability
# perturbation). This widens v1's reported uncertainty -- it does not replace
# or narrow it. v1 is equivalent to silently fixing k=1.0 (no infant-specific
# bound at all); v2 makes that choice explicit and reports the alternative.
# =============================================================================

def compute_eta_v2_point_bounds(f_mhz, P0_mpa, DC, region, age_months, mb_diam_um=None):
    """Point-estimate (no Monte Carlo) eta at the two ends of the item-2 bound."""
    T = transmission_efficiency_v2(f_mhz, region, age_months)
    P_tissue = P0_mpa * T
    p_cons, p_perm = cavitation_probability_v2_bounds(P_tissue, f_mhz, age_months, mb_diam_um)
    dc_f = float(duty_cycle_factor_v2(DC))
    eta_cons = float(np.clip(T * p_cons * dc_f, 0.0, 1.0))
    eta_perm = float(np.clip(T * p_perm * dc_f, 0.0, 1.0))
    return {"conservative_k1.0": eta_cons, "permissive_k2.0": eta_perm}


def compute_eta_v2_with_uncertainty(f_mhz, P0_mpa, DC, region, age_months,
                                     mb_diam_um=None, n_mc=4000, seed=0):
    """Monte Carlo propagation combining:
      (a) v1's existing literature-uncertainty axes (transmission-term fit
          residuals, infant thickness SD, open-fontanelle IL SD, MI-anchor
          interpretive-probability perturbation on the ADULT anchors), and
      (b) the NEW item-2 axis: for age_months <= 12, k is drawn uniformly
          from [INFANT_MI_K_CONSERVATIVE, INFANT_MI_K_PERMISSIVE] = [1.0, 2.0]
          each draw (a uniform prior over the bounded range is the least
          additional assumption beyond the range itself -- no source
          establishes a preferred interior distribution, so none is
          invented).

    Returns a dict with the same median/p5/p95 structure as v1, PLUS the two
    explicit bound point-estimates (conservative_k1.0, permissive_k2.0) so the
    two ends of the item-2 range are visible and labeled, not just folded
    anonymously into a single percentile band per the task's instruction to
    report this "as this full range, clearly labeled 'plausible bound, not
    measured'".
    """
    rng = np.random.default_rng(seed)

    d = _SRC["alsanea2025"]["reported_data"]["skull_thickness_mm_by_age_and_region"]
    REL_THICKNESS_SD = 0.13  # unchanged from v1

    infant_regime = age_months <= INFANT_SCALING_APPLIES_UP_TO_AGE_MONTHS

    etas = np.empty(n_mc)
    for i in range(n_mc):
        alpha_a = v1.ADULT_ALPHA_A * np.exp(rng.normal(0, 1 - v1.ADULT_ALPHA_R2 + 1e-3) * 0.15)
        thick_mm = max(0.3, v1._infant_thickness_mm(region, age_months) * rng.normal(1.0, REL_THICKNESS_SD))
        open_il = max(0.0, rng.normal(v1.OPEN_FONTANELLE_IL_DB_MEAN, v1.OPEN_FONTANELLE_IL_DB_SD))

        route = v1.REGION_ROUTE[region]
        open_frac = v1._fontanelle_open_fraction(route, age_months)
        closed_loss_db = alpha_a * np.power(f_mhz, v1.ADULT_ALPHA_B) * (thick_mm / 10.0)
        bone_loss_db = open_frac * open_il + (1.0 - open_frac) * closed_loss_db
        depth_cm = v1.REGION_DEPTHS_MM[region] / 10.0
        tissue_loss_db = v1.brain_tissue_attenuation_db_per_cm(f_mhz) * depth_cm
        T_i = 10.0 ** (-(bone_loss_db + tissue_loss_db) / 20.0)

        p_perturb = _MI_ANCHORS_P.copy()
        p_perturb[0] = np.clip(p_perturb[0] + rng.normal(0, 0.10), 0.01, 0.49)
        p_perturb[2] = np.clip(p_perturb[2] + rng.normal(0, 0.10), 0.51, 0.99)
        try:
            mi0_i, w_i = v1._fit_mi_sigmoid(_MI_ANCHORS_MI, p_perturb)
        except RuntimeError:
            mi0_i, w_i = MI_SIGMOID_CENTER, MI_SIGMOID_WIDTH

        # NEW (item 2): draw k uniformly from the bounded range for the
        # infant regime; k=1.0 (no shift) outside it.
        k_i = rng.uniform(INFANT_MI_K_CONSERVATIVE, INFANT_MI_K_PERMISSIVE) if infant_regime else 1.0
        mi0_i = mi0_i / k_i

        if mb_diam_um is not None and mb_diam_um < 3.0:
            mi0_i = mi0_i * (0.45 / 0.30)

        P_tissue = P0_mpa * T_i
        mi = P_tissue / np.sqrt(f_mhz)
        p_cav_i = _logistic(mi, mi0_i, max(1e-3, w_i))
        dc_f = float(duty_cycle_factor_v2(DC))
        etas[i] = np.clip(T_i * p_cav_i * dc_f, 0.0, 1.0)

    bounds = compute_eta_v2_point_bounds(f_mhz, P0_mpa, DC, region, age_months, mb_diam_um)
    return {
        "median": float(np.median(etas)),
        "p5": float(np.percentile(etas, 5)),
        "p95": float(np.percentile(etas, 95)),
        "conservative_k1.0_point": bounds["conservative_k1.0"],
        "permissive_k2.0_point": bounds["permissive_k2.0"],
        "note": ("p5/p95 combine v1's literature-uncertainty axes with a uniform draw over the "
                 "item-2 infant biological-scaling bound (k in [1.0, 2.0], age<=12mo only). "
                 "conservative_k1.0_point and permissive_k2.0_point are the two labeled ends of "
                 "that bound alone (no other uncertainty), shown separately per the task's "
                 "instruction to report a labeled plausible range, not a single collapsed number.")
    }


def compute_eta_v2_per_region(f_mhz, P0_mpa, DC, age_months, region_names=None,
                               mb_diam_um=None):
    if region_names is None:
        region_names = REGION_NAMES
    return {r: compute_eta_v2_with_uncertainty(f_mhz, P0_mpa, DC, r, age_months, mb_diam_um)
            for r in region_names}


def leave_one_out_validation_v2():
    """Same LOO structure as v1, re-run on the item-1-corrected anchor set."""
    results = []
    for i in range(3):
        idx_fit = [j for j in range(3) if j != i]
        mi_fit = _MI_ANCHORS_MI[idx_fit]
        p_fit = _MI_ANCHORS_P[idx_fit]
        try:
            mi0, w = v1._fit_mi_sigmoid(mi_fit, p_fit)
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


if __name__ == "__main__":
    print("=" * 78)
    print("ITEM 1: MI sigmoid re-fit on corrected mcdannold2008 anchor (0.47 -> 0.46):")
    print(f"  MI0={MI_SIGMOID_CENTER:.4f} (was {v1.MI_SIGMOID_CENTER:.4f} in v1), "
          f"width={MI_SIGMOID_WIDTH:.4f} (was {v1.MI_SIGMOID_WIDTH:.4f} in v1)")
    print(f"  mcdannold2008 primary-source 95% CI: {MCDANNOLD2008_MI_95CI}")

    print()
    print("Leave-one-out validation (corrected anchors):")
    for r in leave_one_out_validation_v2():
        print(f"    held out {r['held_out']}: MI={r['held_out_MI']:.3f}  "
              f"assumed_true_p={r['assumed_true_probability']:.2f}  "
              f"predicted_p={r['predicted_probability']:.3f}  abs_error={r['abs_error']:.3f}")

    print()
    print("ITEM 2: infant MI-sigmoid-center bounds (age <= 12mo):")
    mi0_c, mi0_p = infant_mi_sigmoid_center_bounds(6.0)
    print(f"  at age=6mo: MI0_conservative(k=1.0)={mi0_c:.4f}  MI0_permissive(k=2.0)={mi0_p:.4f}")
    mi0_c, mi0_p = infant_mi_sigmoid_center_bounds(18.0)
    print(f"  at age=18mo (outside infant regime): MI0_conservative={mi0_c:.4f}  MI0_permissive={mi0_p:.4f}")

    print()
    print("Reference eta with item-1+2 corrections, all 8 regions, at 3 ages "
          "(f=0.5 MHz, P0=0.6 MPa, DC=1%):")
    for age in (1.5, 6.0, 12.0):
        print(f"  age={age} months:")
        res = compute_eta_v2_per_region(0.5, 0.6, 0.01, age)
        for r in REGION_NAMES:
            v = res[r]
            print(f"    {r:18s} median={v['median']:.4f} [p5={v['p5']:.4f}, p95={v['p95']:.4f}]  "
                  f"conservative={v['conservative_k1.0_point']:.4f}  permissive={v['permissive_k2.0_point']:.4f}")
