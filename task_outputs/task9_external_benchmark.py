"""
task9_external_benchmark.py

Task 4: compare gm2_model_v2.py trajectories against published
natural-history / treatment-response literature. Runs the model itself
(untreated / AAV-only / SRT-only tay-sachs arms) and pairs each against
literature gathered via live web search (citations + extracted numbers
recorded below; searched 2026-08-16). Does not modify gm2_model_v2.py.

Literature sources (see run_meta.sources_checked for search provenance):

(a) Bley AE et al., "Natural History of Infantile GM2 Gangliosidosis",
    Pediatrics 2011;128(5):e1233 (PMID pending / AAP publications site,
    NTSAD survey N=237 + literature review, 97 infantile cases timed).
    Quantitative: mean age at first symptom 5.0mo (SD 3.3); 55% of
    infantile cases achieved sitting-without-support (mean age gained
    6.8mo, SD 1.5); of those, almost all LOST it within 12 months (mean
    age lost 13.1mo, SD 6.8) -- i.e. a gain-then-irreversible-loss
    pattern, part of a monotonically progressive/fatal disease course.
    This is already cited in lsd_disease_params.py as the source of the
    t_ref_days=730 (~2y untreated survival) anchor for tay-sachs/sandhoff
    -- i.e. it WAS used for calibration, but only of the survival
    timescale anchor, not of the functional-decline shape/timing.

(b) Eichler F et al., "Dual-vector rAAVrh8 gene therapy for GM2
    gangliosidosis: a phase 1/2 trial", Nature Medicine 2025;31(9):
    2927-2935 (PMC12443631). N=9 (7 TSD, 2 SD; 6 infantile + 3
    juvenile). Combined bilateral intrathalamic + intracisternal +
    intrathecal rAAVrh8 delivery, dose escalation 1.42e14 to 3.56e14 vg
    total. Quantitative: CSF C20:0-GM2 ganglioside decreased from
    baseline to nadir (12-24 weeks) by 9.1% (low-dose patient) up to
    52.5% (mid-dose) / 49.5% (high-dose) -- dose-dependent, partial.
    CSF HexA activity peaked at 13% of the normal mean at 12 weeks in
    the best responder. (Cachon-Gonzalez et al.'s original rAAV2/1
    Sandhoff MOUSE work -- intracranial HEXA+HEXB co-injection giving
    survival to 2y -- was also searched; only a categorical
    survival-benefit finding was recoverable via search, no quantitative
    brain-substrate-reduction percentage, so the more recent human trial
    is used as the primary quantitative AAV comparator per the task's
    own guidance to prefer "more recent GM2 AAV gene therapy trial
    data".)

(c) SRT literature, two sources (neither gives a clean single number,
    reported as found rather than forced into one):
    (i) Maegawa/Bembi et al.-style case report, "Substrate reduction
        therapy with miglustat in chronic GM2 gangliosidosis type
        Sandhoff: results of a 3-year follow-up", J Inherit Metab Dis
        2010, PMID 20821051. N=1 chronic (juvenile/SMA-phenotype)
        Sandhoff patient, miglustat 100mg t.i.d. x3y. Abstract reports
        only "minor effects on neurological progression" -- no
        substrate-level quantification given.
    (ii) Denny CA et al./Seyfried lab, PMC3126858 (Sandhoff MICE, a
        glucosylceramide-synthase inhibitor, iminosugar class incl.
        Genz-529468/NB-DNJ). Quantitative and directly relevant to
        interpreting the model's SRT mechanism: brain GM2 INCREASED to
        120-150% of untreated levels (days 56/84/112), brain GL1 rose
        >10-20-fold, while LIVER GM2 fell 40-60% at day 112; median
        survival still extended 34-41% despite the CNS substrate
        increase (proposed non-substrate-reduction mechanism of
        benefit, e.g. anti-inflammatory).

Output: data/external_benchmark.json
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gm2_model_v2 import make_base_params, make_x0, simulate_milstein, DEFAULT_FUS_EVENTS

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_FILE = os.path.join(ROOT, "gm2_model_v2.py")
with open(MODEL_FILE, "rb") as f:
    MODEL_SHA256 = hashlib.sha256(f.read()).hexdigest()[:16]

N_SEEDS = 10  # replicate every arm across seeds -- a single Milstein realization
              # of this SDE turned out (see below) to carry substantial run-to-run
              # noise for some arms, so no comparison here relies on seed=1 alone.
SEEDS = list(range(N_SEEDS))
DT = 0.2


def run_arm_seed(dose_mg_per_day, t4_dose_total, tmax_days, seed, fus_on=True):
    base = make_base_params('tay-sachs')
    x0 = make_x0(base)
    df = simulate_milstein(x0, tmax_days, DT, base, dose_mg_per_day=dose_mg_per_day,
                            t4_admin_day=30.0, t4_dose_total=t4_dose_total,
                            fus_events=DEFAULT_FUS_EVENTS if fus_on else None, seed=seed)
    return base, df


def sample_at(df, day, col):
    idx = (df['time_days'] - day).abs().idxmin()
    return float(df.loc[idx, col]), float(df.loc[idx, 'time_days'])


def pct_change(val, base_val):
    return 100.0 * (base_val - val) / base_val


def run_arm_replicated(dose_mg_per_day, t4_dose_total, tmax_days, days, fus_on=True):
    """Run N_SEEDS independent Milstein realizations of one arm; return the
    last realization's full dataframe (for trajectory plots/series) plus
    per-day mean/sd/min/max of gm2_brain and bayley_motor_composite across
    seeds, and the base params (identical across seeds)."""
    base = None
    df_last = None
    per_day = {day: {"gm2_brain": [], "bayley_motor_composite": []} for day in days}
    for seed in SEEDS:
        base, df = run_arm_seed(dose_mg_per_day, t4_dose_total, tmax_days, seed, fus_on=fus_on)
        df_last = df
        for day in days:
            for col in ("gm2_brain", "bayley_motor_composite"):
                val, _ = sample_at(df, day, col)
                per_day[day][col].append(val)
    stats = {}
    for day, cols in per_day.items():
        stats[day] = {}
        for col, vals in cols.items():
            arr = np.array(vals)
            stats[day][col] = {
                "mean": float(arr.mean()), "sd": float(arr.std(ddof=1)),
                "min": float(arr.min()), "max": float(arr.max()), "n_seeds": len(arr),
            }
    return base, df_last, stats


# --- (a) Untreated arm vs Bley et al. 2011 ---
UNTREATED_DAYS = [0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 207, 300, 398, 500, 600, 730]
base_u, df_untreated, stats_u = run_arm_replicated(0.0, 0.0, 730.0, UNTREATED_DAYS, fus_on=False)
untreated_series = [
    {"day": day, "gm2_brain_mean": stats_u[day]["gm2_brain"]["mean"], "gm2_brain_sd": stats_u[day]["gm2_brain"]["sd"],
     "bayley_motor_composite_mean": stats_u[day]["bayley_motor_composite"]["mean"],
     "bayley_motor_composite_sd": stats_u[day]["bayley_motor_composite"]["sd"]}
    for day in UNTREATED_DAYS
]

motor_at_207 = stats_u[207]["bayley_motor_composite"]  # Bley: mean age sitting gained, 6.8mo
motor_at_398 = stats_u[398]["bayley_motor_composite"]  # Bley: mean age sitting lost, 13.1mo
motor_min_0_50_val = min(stats_u[d]["bayley_motor_composite"]["mean"] for d in [0, 5, 10, 20, 30, 50])

comparison_a = {
    "label": "Untreated infantile GM2 (tay-sachs baseline) vs. Bley et al. 2011 natural history",
    "citation": "Bley AE, Giannikopoulos OA, Hayden D, Kubilus K, Tifft CJ, Eichler FS. Natural History of Infantile GM2 Gangliosidosis. Pediatrics. 2011;128(5):e1233-e1241.",
    "literature_quantitative": {
        "mean_age_first_symptom_months": 5.0, "sd_months": 3.3,
        "pct_achieving_unsupported_sitting": 55,
        "mean_age_sitting_gained_months": 6.8, "sd_months_gained": 1.5,
        "mean_age_sitting_lost_months": 13.1, "sd_months_lost": 6.8,
        "pattern": "functional gain followed by near-universal, irreversible loss -- monotonically progressive, fatal course; NOT self-limiting or improving",
    },
    "used_for_calibration_originally": (
        "PARTIALLY: t_ref_days=730 (~2y untreated survival) in lsd_disease_params.py cites this "
        "paper as its source and IS used as the disease timescale anchor for tay-sachs/sandhoff. "
        "The functional-decline SHAPE/TIMING (Bayley composite dynamics, target_motor formula) was "
        "NOT fit to this paper's sitting-gained/sitting-lost timeline -- it is a constructed formula."
    ),
    "model_output": {
        "n_seeds": N_SEEDS,
        "trajectory_days_gm2_brain_bayley_motor_mean_sd_across_seeds": untreated_series,
        "bayley_motor_at_day207_matched_to_sitting_gained_age": motor_at_207,
        "bayley_motor_at_day398_matched_to_sitting_lost_age": motor_at_398,
        "bayley_motor_mean_min_over_first_50_days": motor_min_0_50_val,
        "bayley_motor_baseline": base_u['bayley_motor_baseline'],
    },
    "agreement_assessment": (
        f"QUALITATIVE ONLY (Bley's data are population survey ages/percentages, not a kinetic "
        f"trajectory in units this model outputs -- no RMSE is meaningful here). Checked across "
        f"{N_SEEDS} independent seeds (not a single realization) because this is a stochastic SDE: "
        f"the pattern below is robust to seed choice, not an artifact of one draw. PARTIAL agreement: "
        f"the model DOES show an early decline in bayley_motor_composite over the first ~20-50 days "
        f"(mean down to {motor_min_0_50_val:.1f} from baseline {base_u['bayley_motor_baseline']:.1f}), "
        f"matching the qualitative direction of early regression. However, the model's untreated arm "
        f"then RECOVERS -- bayley_motor_composite mean rises back above baseline by day ~200-400 "
        f"(day207 mean={motor_at_207['mean']:.1f}+/-{motor_at_207['sd']:.1f} [min {motor_at_207['min']:.1f}, "
        f"max {motor_at_207['max']:.1f}], day398 mean={motor_at_398['mean']:.1f}+/-{motor_at_398['sd']:.1f} "
        f"[min {motor_at_398['min']:.1f}, max {motor_at_398['max']:.1f}]) and every one of the 10 seeds is "
        f"above baseline by day 398 -- the recovery is NOT a single-seed fluke. This DISAGREES with the "
        "literature's monotonic, irreversible progressive-loss pattern (sitting gained then lost by "
        "13.1mo/~398d, never recovered). Root cause (see model inspection, not a benchmark artifact): "
        "the dG_B equation's flat -0.01*G_B decay term dominates gm2_brain dynamics at the initial "
        "condition (890 nmol/g) regardless of enzyme activity, pulling gm2_brain down toward a much "
        "lower quasi-steady-state within ~100-150 days even with ZERO treatment and ZERO residual "
        "enzyme; since bayley_motor's target is a decreasing function of gm2_brain, this drives the "
        "spontaneous 'recovery.' This is a genuine, reportable model limitation for the untreated arm, "
        "not a benchmark-comparison error or seed-selection artifact."
    ),
}


# --- (b) AAV-only arm vs Eichler et al. 2025 (Nature Medicine) ---
base_a, df_aav, stats_a = run_arm_replicated(0.0, 1.0, 200.0, [84, 168], fus_on=True)
aav_84, aav_168 = stats_a[84]["gm2_brain"], stats_a[168]["gm2_brain"]
baseline_gm2_brain = base_a['baseline_gm2_brain']
aav_84_pct = pct_change(aav_84["mean"], baseline_gm2_brain)
aav_84_pct_sd = aav_84["sd"] / baseline_gm2_brain * 100.0
aav_168_pct = pct_change(aav_168["mean"], baseline_gm2_brain)
aav_168_pct_sd = aav_168["sd"] / baseline_gm2_brain * 100.0

comparison_b = {
    "label": "AAV-only arm vs. Eichler et al. 2025 dual-vector rAAVrh8 phase 1/2 trial",
    "citation": "Eichler F, Cataltepe OI, Daci R, et al. Dual-vector rAAVrh8 gene therapy for GM2 gangliosidosis: a phase 1/2 trial. Nat Med. 2025;31(9):2927-2935.",
    "citation_note": (
        "Cachon-Gonzalez MB et al.'s original rAAV2/1 HEXA+HEXB Sandhoff-mouse work (survival to 2y) "
        "was also searched but only a categorical survival finding was recoverable (no quantitative "
        "brain-substrate % reduction in the sources found), so per the task's own preference for "
        "'more recent GM2 AAV gene therapy trial data', the 2025 human trial (which DOES report "
        "quantitative substrate numbers) is used as the primary comparator here."
    ),
    "literature_quantitative": {
        "n_patients": 9, "tsd": 7, "sd": 2,
        "csf_C20_GM2_pct_reduction_from_baseline_to_nadir": {"low_dose": 9.1, "mid_dose": 52.5, "high_dose": 49.5},
        "nadir_timing_weeks": "12-24",
        "csf_hexA_activity_peak_pct_of_normal_mean": 13,
        "pattern": "dose-dependent, PARTIAL substrate reduction; far from complete clearance even at highest tested dose",
    },
    "used_for_calibration_originally": "NO. gm2_model_v2.py's AAV compartment (normalized-dose rebuild, see file docstring BUG2) was constructed to fix a units/saturation defect, not fit to any external substrate-reduction magnitude from this or any other AAV trial/mouse study.",
    "model_output": {
        "n_seeds": N_SEEDS,
        "baseline_gm2_brain": baseline_gm2_brain,
        "gm2_brain_at_day84_across_seeds": aav_84, "pct_reduction_at_day84_mean": aav_84_pct, "pct_reduction_at_day84_sd": aav_84_pct_sd,
        "gm2_brain_at_day168_across_seeds": aav_168, "pct_reduction_at_day168_mean": aav_168_pct, "pct_reduction_at_day168_sd": aav_168_pct_sd,
    },
    "agreement_assessment": (
        f"Direction agrees (both show AAV reduces GM2 burden). Magnitude does NOT: across {N_SEEDS} seeds "
        f"the model's AAV-only arm is TIGHTLY reproducible (day84 {aav_84_pct:.1f}%+/-{aav_84_pct_sd:.2f}%, "
        f"day168 {aav_168_pct:.1f}%+/-{aav_168_pct_sd:.2f}%, i.e. seed noise is negligible here -- this is "
        "essentially complete clearance every run) versus the trial's dose-dependent 9-53% CSF GM2 "
        "reduction over the same window, with CSF enzyme activity reaching only 13% of normal even in "
        "the best responder. The model's brain compartment (a single well-mixed nmol/g burden) and the "
        "trial's CSF C20:0-GM2 species concentration are not the same quantity (whole-brain-tissue "
        "burden vs. one CSF lipid species), so no RMSE is computed -- but the normalized "
        "(%-reduction-from-baseline) comparison is legitimate and, being low-noise on the model side, "
        "the gap to the trial's 9-53% is a real, well-resolved disagreement, not sampling noise: the "
        "model substantially OVER-predicts AAV monotherapy efficacy relative to the actual clinical trial."
    ),
}


# --- (c) SRT-only arm vs SRT literature ---
base_s, df_srt, stats_s = run_arm_replicated(3.0, 0.0, 200.0, [84, 168], fus_on=True)
srt_84, srt_168 = stats_s[84]["gm2_brain"], stats_s[168]["gm2_brain"]
srt_84_pct = pct_change(srt_84["mean"], baseline_gm2_brain)
srt_84_pct_sd = srt_84["sd"] / baseline_gm2_brain * 100.0
srt_168_pct = pct_change(srt_168["mean"], baseline_gm2_brain)
srt_168_pct_sd = srt_168["sd"] / baseline_gm2_brain * 100.0
# worst-case (max gm2_brain across seeds = smallest reduction) still checked
# explicitly below, since this arm turned out to have high seed-to-seed
# variance and a single point estimate would be misleading either way.
srt_84_pct_min = pct_change(srt_84["max"], baseline_gm2_brain)   # largest remaining gm2_brain -> smallest % reduction
srt_168_pct_min = pct_change(srt_168["max"], baseline_gm2_brain)

comparison_c = {
    "label": "SRT-only arm vs. published SRT literature for GM2 gangliosidosis / LSDs",
    "citations": [
        "Case report: Substrate reduction therapy with miglustat in chronic GM2 gangliosidosis type Sandhoff: results of a 3-year follow-up. J Inherit Metab Dis. 2010. PMID 20821051.",
        "Denny CA, Kasperzyk JL, Gorham KN, Bagel JH, Seyfried TN et al. (glucosylceramide synthase inhibitor study in Sandhoff mice), PMC3126858.",
    ],
    "literature_quantitative": {
        "miglustat_case_report_N": 1,
        "miglustat_case_report_finding": "minor effects on neurological progression over 3y (no substrate-level numbers given in abstract)",
        "sandhoff_mouse_GCS_inhibitor_brain_GM2_pct_of_untreated_days_56_84_112": "120-150% (INCREASE, not decrease)",
        "sandhoff_mouse_GCS_inhibitor_brain_GL1_fold_increase": ">10 to >20-fold",
        "sandhoff_mouse_GCS_inhibitor_liver_GM2_pct_reduction_day112": "40-60%",
        "sandhoff_mouse_GCS_inhibitor_median_survival_pct_extension": "34-41%",
        "pattern": "GCS-inhibitor-class SRT does NOT reliably reduce brain GM2 in Sandhoff mice (can INCREASE it); peripheral (liver) reduction is real; survival benefit likely via a non-substrate-reduction mechanism",
    },
    "used_for_calibration_originally": "NO. gm2_model_v2.py's SRT mechanism (Hill-function inhibition of brain synthesis rate, gated on brain drug level B_free) is a constructed mechanism, not fit to miglustat or GCS-inhibitor pharmacodynamic data.",
    "model_output": {
        "n_seeds": N_SEEDS,
        "baseline_gm2_brain": baseline_gm2_brain,
        "gm2_brain_at_day84_across_seeds": srt_84, "pct_reduction_at_day84_mean": srt_84_pct, "pct_reduction_at_day84_sd": srt_84_pct_sd,
        "gm2_brain_at_day168_across_seeds": srt_168, "pct_reduction_at_day168_mean": srt_168_pct, "pct_reduction_at_day168_sd": srt_168_pct_sd,
        "note_on_variance": (
            "Unlike the AAV-only arm, this arm's %-reduction is HIGH-VARIANCE across seeds "
            f"(day84 sd={srt_84_pct_sd:.1f}pp, day168 sd={srt_168_pct_sd:.1f}pp, over {N_SEEDS} seeds) -- "
            "a single realization (as originally reported) would have been a misleading point estimate. "
            "Reporting mean+/-sd and the worst-case (max gm2_brain across seeds = smallest reduction) below "
            "instead of one seed's number."
        ),
        "pct_reduction_at_day84_worst_case_min": srt_84_pct_min,
        "pct_reduction_at_day168_worst_case_min": srt_168_pct_min,
    },
    "agreement_assessment": (
        f"DISAGREEMENT in mechanism/magnitude, and this is the most substantive of the three findings -- "
        f"but note this arm required checking across {N_SEEDS} seeds rather than one, because a single "
        f"realization here has real run-to-run variance (day168 %-reduction ranged "
        f"{srt_168_pct_min:.0f}-{pct_change(srt_168['min'], baseline_gm2_brain):.0f}% across 10 seeds, "
        f"sd={srt_168_pct_sd:.1f}pp): the model's SRT-only arm still produces a substantial mean "
        f"{srt_84_pct:.0f}% (day84) to {srt_168_pct:.0f}% (day168) brain GM2 reduction, and even its "
        f"WORST-CASE seed ({srt_168_pct_min:.0f}% at day168) is still a clear net reduction, so the "
        "conclusion is not an artifact of averaging over noise. This implies SRT directly and "
        "substantially clears brain substrate in the model. The best available quantitative brain-tissue "
        "data (Sandhoff MICE, GCS-inhibitor class) show the OPPOSITE in the brain compartment specifically "
        "-- GM2 INCREASED to 120-150% of untreated -- with real substrate reduction confined to the liver "
        "(40-60%) and the drug's survival benefit attributed to a non-substrate-reduction mechanism. The "
        "single human GM2/Sandhoff case report (miglustat, same drug class) is consistent with this being "
        "a real limitation of GCS-inhibitor SRT in the brain ('minor effects' only, no substrate data). "
        "The model's SRT submodel (direct Hill-inhibition of brain synthesis) is a simplification that "
        "does not reproduce this known brain/liver asymmetry or the paradoxical brain-substrate increase "
        "reported for this drug class -- worth flagging explicitly rather than treating the model's SRT "
        "mechanism as validated against real GCS-inhibitor pharmacology. Separately: because this arm is "
        "high-variance, any SINGLE-realization number quoted for it (including in other analyses in this "
        "project that may only use one seed) should be treated with caution."
    ),
}


out = {
    "task": "external_benchmark_comparison",
    "comparisons": {
        "a_untreated_vs_bley_2011": comparison_a,
        "b_aav_only_vs_eichler_2025": comparison_b,
        "c_srt_only_vs_srt_literature": comparison_c,
    },
    "run_meta": {
        "model_file": "gm2_model_v2.py", "model_sha256_16": MODEL_SHA256,
        "n_seeds": N_SEEDS, "seeds": SEEDS, "dt": DT,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "revision_note": (
            "Revised from an initial single-seed (seed=1) version after checking sample-size "
            "adequacy: the AAV-only arm turned out low-variance across seeds (single seed was "
            "fine), but the SRT-only arm turned out high-variance (single seed was NOT "
            "representative) -- all three arms now use N_SEEDS=10 replication."
        ),
        "sources_checked_but_not_usable": [
            "Cachon-Gonzalez et al. original Sandhoff-mouse AAV2/1 paper: only a categorical "
            "survival-to-2y finding was recoverable via search snippets; no quantitative brain "
            "substrate/enzyme percentage was found, so it is cited for context only, not used "
            "as the quantitative AAV comparator.",
        ],
        "note_on_rmse": (
            "No RMSE is reported for any comparison: (a) Bley 2011 reports population survey ages/"
            "percentages, not a trajectory in the model's units; (b)/(c) the model's brain GM2 burden "
            "(nmol/g, whole-tissue) and the trials'/studies' CSF ganglioside species concentrations or "
            "categorical survival outcomes are not the same physical quantity. Percent-change-from-"
            "baseline is reported for (b)/(c) as a legitimate normalized comparison; RMSE was not "
            "forced where units/quantities were incommensurable, per the task instructions."
        ),
    },
}

outpath = os.path.join(DATA_DIR, "external_benchmark.json")
with open(outpath, "w") as f:
    json.dump(out, f, indent=2)

for key, comp in out["comparisons"].items():
    print(f"\n=== {key} ===")
    print(comp["agreement_assessment"])

print(f"\nWrote {outpath}")
