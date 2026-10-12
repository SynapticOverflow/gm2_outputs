"""
task10_summary.py

Summary analysis of the task10 seed-sensitivity runs (task10_seed_sensitivity.py).
Reads only the per-run JSON written by that driver (plus, for the seed-42
reproducibility control, task_outputs/percentile_cis.json and
gm2_analysis/sobol_outputs/gm2_full14_*.json). Writes seed_sensitivity_summary.json
and SUMMARY.md next to the run directory, and prints the tables.

Usage: python3 task10_summary.py [RUNDIR]   (default ./N256)
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SCENARIOS = ["as_documented", "equal_width", "tier1_promotion"]
REF = "A_seed42"
# The four AAV delivery/dose parameters (vector entry, dose, expression buildup,
# expressed-enzyme decay). Table 4 itself is not on this filesystem; this is the
# k_T4_* / t4_* family in sobol_driver_v2.PARAM_NAMES. Change here to re-slice.
DELIVERY4 = ["k_T4_entry", "t4_dose_total", "k_T4_payload", "k_T4_decay"]
FUS = "fus_entry_gain_scale"
# parameters whose prior width differs between scenarios (the ones the
# prior-width-artifact claim is about)
WIDTH_VARIED = ["k_T4_entry", "t4_dose_total"]
# S_T at or below this is treated as numerically zero for ranking / ordering
# purposes (seed-42 fus_entry_gain_scale is ~1.9e-6; next-smallest non-null is
# IC50 ~3e-3).
NEGLIGIBLE = 1e-4
# Scenario contrasts in which the parameter's OWN prior width differs
# (sobol_driver_v2.make_bounds): (wider, narrower). Contrasts between scenarios
# where its width is identical are not part of the prior-width claim.
WIDTH_CONTRASTS = {
    "k_T4_entry": [("as_documented", "equal_width"), ("as_documented", "tier1_promotion")],
    "t4_dose_total": [("as_documented", "equal_width"), ("tier1_promotion", "equal_width")],
}


def load(rundir):
    runs = {}
    for p in sorted(glob.glob(os.path.join(rundir, "*_*.json"))):
        d = json.load(open(p))
        runs.setdefault(d["run_id"], {})[d["scenario"]] = d
    return runs


def run_order(run_ids):
    key = lambda r: (r[0], int(r.split("seed")[1]) if r.startswith("A_") else 0)
    a = sorted([r for r in run_ids if r.startswith("A_")], key=key)
    return [REF] + [r for r in a if r != REF] + sorted(r for r in run_ids if not r.startswith("A_"))


def ranks(st):
    return rankdata(-np.asarray(st), method="average")


def fmt(x, nd=4):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    rundir = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, "N256")
    runs = load(rundir)
    order = run_order(runs.keys())
    names = runs[REF][SCENARIOS[0]]["names"]
    idx = {n: i for i, n in enumerate(names)}
    failed = [(r, s, runs[r][s]["status"]) for r in order for s in SCENARIOS
              if runs[r].get(s, {}).get("status") != "ok"]
    out = {"rundir": rundir, "reference_run": REF, "delivery4": DELIVERY4, "fus_param": FUS,
           "runs": order, "failed_or_missing": failed}
    md = [f"# Seed-sensitivity summary ({os.path.basename(rundir)})", "",
          f"Reference = `{REF}`. Delivery/dose set = {', '.join(DELIVERY4)}; FUS = `{FUS}`.", ""]
    if failed:
        md += ["**FAILED / MISSING runs:** " + ", ".join(f"{r}/{s} ({st})" for r, s, st in failed), ""]
    ok = lambda r, s: runs[r].get(s, {}).get("status") == "ok"
    ST = {(r, s): np.array(runs[r][s]["ST"]) for r in order for s in SCENARIOS if ok(r, s)}

    # ---- 0. reproducibility control ------------------------------------
    ctrl = {}
    pc_path = os.path.join(REPO, "task_outputs", "percentile_cis.json")
    pc = {r["scenario"]: r for r in json.load(open(pc_path)) if r.get("source") == "rerun"}
    rows = []
    for s in SCENARIOS:
        me = runs[REF][s]
        t7 = pc[s]
        prod = json.load(open(os.path.join(REPO, "gm2_analysis", "sobol_outputs", f"gm2_full14_{s}.json")))
        d_t7 = float(np.max(np.abs(np.array(me["ST"]) - np.array(t7["ST_mean"]))))
        d_t7_ci = float(max(np.max(np.abs(np.array(me["ST_ci_lo_2.5"]) - np.array(t7["ST_ci_lo_2.5"]))),
                            np.max(np.abs(np.array(me["ST_ci_hi_97.5"]) - np.array(t7["ST_ci_hi_97.5"])))))
        pst = np.array(prod["ST"])
        lo, hi = np.array(me["ST_ci_lo_2.5"]), np.array(me["ST_ci_hi_97.5"])
        inside = [(n, bool(lo[i] - 1e-12 <= pst[i] <= hi[i] + 1e-12)) for i, n in enumerate(names)]
        d_prod = np.abs(np.array(me["ST"]) - pst)
        ctrl[s] = {"max_abs_dST_vs_task7_same_X": d_t7, "max_abs_dCI_vs_task7_same_X": d_t7_ci,
                   "Ymean_task7": t7["Y_mean"], "Ymean_this": me["Y_summary"]["mean"],
                   "max_abs_dST_vs_production_full14_different_X": float(d_prod.max()),
                   "production_ST_inside_this_95pct_CI": dict(inside),
                   "n_outside": sum(not v for _, v in inside)}
        rows.append([s, d_t7, d_t7_ci, float(d_prod.max()), f"{14 - ctrl[s]['n_outside']}/14"])
    out["reproducibility_control"] = ctrl
    md += ["## 0. Seed-42 reproducibility control", "",
           "vs task7 rerun (`percentile_cis.json`, same X, same bootstrap seed) and vs the original "
           "production `gm2_full14_*.json` (different, unrecoverable X; SALib default B=100).", "",
           md_table(["scenario", "max|ΔS_T| vs task7", "max|ΔCI| vs task7",
                     "max|ΔS_T| vs production", "production S_T inside our 95% CI"], rows), ""]

    # ---- 1. delivery + FUS S_T per seed -----------------------------------
    sec1 = {}
    md += ["## 1. S_T of delivery/dose params and FUS gain, per run", ""]
    for s in SCENARIOS:
        rows = []
        sec1[s] = {}
        for p in DELIVERY4 + [FUS]:
            vals = {r: float(ST[(r, s)][idx[p]]) for r in order if (r, s) in ST}
            a_vals = [v for r, v in vals.items() if r.startswith("A_")]
            sec1[s][p] = {"per_run": vals, "A_mean": float(np.mean(a_vals)),
                          "A_min": float(np.min(a_vals)), "A_max": float(np.max(a_vals)),
                          "A_range": float(np.ptp(a_vals)),
                          "ref_ci": [runs[REF][s]["ST_ci_lo_2.5"][idx[p]], runs[REF][s]["ST_ci_hi_97.5"][idx[p]]]}
            rows.append([p] + [vals.get(r) for r in order] +
                        [sec1[s][p]["A_mean"], f"{sec1[s][p]['A_min']:.4f}–{sec1[s][p]['A_max']:.4f}",
                         f"[{sec1[s][p]['ref_ci'][0]:.4f}, {sec1[s][p]['ref_ci'][1]:.4f}]"])
        md += [f"**{s}**", "", md_table(["param"] + order + ["mean (A)", "range (A)", "seed-42 95% CI"], rows), ""]
    out["delivery_fus_ST"] = sec1

    # ---- 2/3/4/6 per-run stats vs reference ---------------------------
    ref_nonzero = {s: [i for i in range(len(names)) if ST[(REF, s)][i] > NEGLIGIBLE] for s in SCENARIOS}
    per_run = {}
    md += ["## 2–4, 6. Per-run comparison to seed 42", ""]
    for s in SCENARIOS:
        rows = []
        r_ref = ranks(ST[(REF, s)])
        nz = ref_nonzero[s]
        var_ref = runs[REF][s]["Y_summary"]["var"]
        for r in order:
            if (r, s) not in ST:
                continue
            st = ST[(r, s)]
            rk = ranks(st)
            rho = spearmanr(ST[(REF, s)], st).statistic
            tau = kendalltau(ST[(REF, s)], st).statistic
            rho_nz = spearmanr(ST[(REF, s)][nz], st[nz]).statistic
            tau_nz = kendalltau(ST[(REF, s)][nz], st[nz]).statistic
            shifts = [(names[i], float(r_ref[i]), float(rk[i])) for i in range(len(names))
                      if abs(rk[i] - r_ref[i]) >= 2]
            d = np.abs(st - ST[(REF, s)])
            ys = runs[r][s]["Y_summary"]
            rec = {"spearman_all14": float(rho), "kendall_all14": float(tau),
                   "spearman_nonzero": float(rho_nz), "kendall_nonzero": float(tau_nz),
                   "nonzero_params_in_ref": [names[i] for i in nz],
                   "rank_shifts_ge2": shifts, "ranks": dict(zip(names, map(float, rk))),
                   "max_abs_dST": float(d.max()), "argmax_abs_dST": names[int(d.argmax())],
                   "mean_abs_dST": float(d.mean()),
                   "sum_ST_delivery4": float(sum(st[idx[p]] for p in DELIVERY4)),
                   "sum_ST": float(st.sum()), "sum_S1": runs[r][s]["sum_S1"],
                   "var_Y": ys["var"], "var_Y_ratio_vs_ref": ys["var"] / var_ref,
                   "Y_summary": ys,
                   "top4": [names[i] for i in np.argsort(-st)[:4]]}
            per_run.setdefault(r, {})[s] = rec
            rows.append([r, rec["spearman_all14"], rec["kendall_all14"], rec["spearman_nonzero"],
                         rec["kendall_nonzero"],
                         ", ".join(f"{n} {a:g}→{b:g}" for n, a, b in shifts) or "none",
                         rec["max_abs_dST"], rec["mean_abs_dST"], rec["sum_ST_delivery4"],
                         rec["sum_ST"], ys["var"], ys["mean"], ys["sd"], ys["min"], ys["max"],
                         ys["frac_at_floor_eq_0"]])
        md += [f"**{s}** (non-zero-S_T subset in seed 42: {len(nz)} params)", "",
               md_table(["run", "Spearman (14)", "Kendall (14)", "Spearman (nz)", "Kendall (nz)",
                         "rank shifts ≥2 (ref→run)", "max|ΔS_T|", "mean|ΔS_T|", "ΣS_T delivery4",
                         "ΣS_T", "Var(Y)", "mean Y", "SD Y", "min Y", "max Y", "frac Y=0"], rows), ""]
    out["per_run_vs_ref"] = per_run

    # pairwise Spearman among all runs
    pair = {}
    md += ["### Pairwise Spearman ρ (all 14 S_T) between runs", ""]
    for s in SCENARIOS:
        rs = [r for r in order if (r, s) in ST]
        M = [[float(spearmanr(ST[(a, s)], ST[(b, s)]).statistic) for b in rs] for a in rs]
        pair[s] = {"runs": rs, "spearman": M}
        md += [f"**{s}**", "", md_table(["", *rs], [[a, *row] for a, row in zip(rs, M)]), ""]
    out["pairwise_spearman"] = pair

    # Var(Y): B vs A
    vy = {}
    for s in SCENARIOS:
        a = [per_run[r][s]["var_Y"] for r in order if r.startswith("A_") and s in per_run.get(r, {})]
        b = [per_run[r][s]["var_Y"] for r in order if r.startswith("B_") and s in per_run.get(r, {})]
        c = [per_run[r][s]["var_Y"] for r in order if r.startswith("C_") and s in per_run.get(r, {})]
        vy[s] = {"A_mean": float(np.mean(a)), "A_min": float(np.min(a)), "A_max": float(np.max(a)),
                 "B": b[0] if b else None, "C": c[0] if c else None,
                 "B_over_A_mean": (b[0] / np.mean(a)) if b else None}
    out["varY_B_vs_A"] = vy
    md += ["### Var(Y): per-row-seed run B vs fixed-seed runs A", "",
           md_table(["scenario", "A mean", "A min", "A max", "B", "B / A mean", "C"],
                    [[s, v["A_mean"], v["A_min"], v["A_max"], v["B"], v["B_over_A_mean"], v["C"]]
                     for s, v in vy.items()]), ""]

    # ---- 5. prior-width artifact: scenario ordering per param -------------
    sec5 = {}
    rows = []
    for p in names:
        ref_vals = [ST[(REF, s)][idx[p]] for s in SCENARIOS]
        if max(ref_vals) <= NEGLIGIBLE:
            sec5[p] = {"ref_order": None, "note": f"S_T <= {NEGLIGIBLE} in all three scenarios for seed 42"}
            continue
        ref_ord = [SCENARIOS[i] for i in np.argsort(-np.array(ref_vals))]
        holds = {}
        for r in order:
            if all((r, s) in ST for s in SCENARIOS):
                v = [ST[(r, s)][idx[p]] for s in SCENARIOS]
                holds[r] = [SCENARIOS[i] for i in np.argsort(-np.array(v))] == ref_ord
        ratio = {r: float(ST[(r, "equal_width")][idx[p]] / ST[(r, "as_documented")][idx[p]])
                 for r in holds if ST[(r, "as_documented")][idx[p]] > 0}
        sec5[p] = {"ref_order": ref_ord, "holds": holds, "all_hold": all(holds.values()),
                   "ST_equal_over_as_documented": ratio, "width_varied_param": p in WIDTH_VARIED}
        rows.append([p + (" *" if p in WIDTH_VARIED else ""), " > ".join(ref_ord)] +
                    ["yes" if holds.get(r) else "NO" for r in order] +
                    ["yes" if sec5[p]["all_hold"] else "NO",
                     f"{min(ratio.values()):.3f}–{max(ratio.values()):.3f}" if ratio else "—"])
    out["scenario_ordering"] = sec5
    md += ["## 5. Prior-width artifact: does seed 42's ordering of the three scenarios hold?", "",
           "`*` = parameter whose prior width changes between scenarios. Last column: "
           "S_T(equal_width)/S_T(as_documented) range over runs.", "",
           md_table(["param", "seed-42 order"] + order + ["all runs", "eq/as_doc ratio"], rows), ""]

    # ---- 7. full S_T tables ----------------------------------------------
    md += ["## 7. Full S_T (all 14 params) per run", ""]
    for s in SCENARIOS:
        rs = [r for r in order if (r, s) in ST]
        md += [f"**{s}**", "", md_table(["param", *rs], [[n, *[float(ST[(r, s)][i]) for r in rs]]
                                                       for i, n in enumerate(names)]), ""]

    # ---- 8. supplementary fixed-seed replicates from C's per-seed columns ---
    # Run C saved Y for each of its 10 seeds separately; each column is itself a
    # complete fixed-seed (CRN) Sobol run on the same X. Re-analysing them costs
    # no model evaluations and gives the distinct fixed-seed replicates (15: seed 7 is in both A and C, so its C
    # column duplicates A_seed7 and is skipped).
    from SALib.analyze.sobol import analyze as sobol_analyze
    c_id = next((r for r in order if r.startswith("C_")), None)
    supp = {}
    fixed_ST = {s: {r: ST[(r, s)] for r in order if r.startswith("A_") and (r, s) in ST} for s in SCENARIOS}
    if c_id:
        seeds_c = runs[c_id][SCENARIOS[0]]["run_meta"]["seeds"]
        for s in SCENARIOS:
            Ymat = np.load(os.path.join(rundir, f"Y_{c_id}_{s}_perseed.npy"))
            for k, sd in enumerate(seeds_c):
                a_twin = f"A_seed{sd}"
                if a_twin in runs:
                    # same seed as a run-A seed: same noise path, not an extra replicate
                    # (seed 7 is in both lists). Verify bit-identity, then skip it.
                    assert np.array_equal(Ymat[:, k], np.load(os.path.join(rundir, f"Y_{a_twin}_{s}.npy"))), (s, sd)
                    supp.setdefault("skipped_duplicate_C_columns", {})[s] = supp.get(
                        "skipped_duplicate_C_columns", {}).get(s, []) + [sd]
                    continue
                Si = sobol_analyze({"num_vars": len(names), "names": names}, Ymat[:, k],
                                   calc_second_order=True, num_resamples=1000, seed=8162026)
                fixed_ST[s][f"Cseed{sd}"] = np.asarray(Si["ST"])
            # two-way (row x seed) decomposition of the CRN design
            gm = Ymat.mean()
            row_eff = Ymat.mean(axis=1) - gm
            seed_eff = Ymat.mean(axis=0) - gm
            tot = Ymat.var()
            supp[s] = {"var_total_rows_x_seeds": float(tot),
                       "frac_parameter_main": float(np.var(row_eff) / tot),
                       "frac_seed_main": float(np.var(seed_eff) / tot),
                       "frac_interaction": float(1 - (np.var(row_eff) + np.var(seed_eff)) / tot),
                       "per_seed_mean_Y": Ymat.mean(axis=0).tolist(),
                       "per_seed_var_Y": Ymat.var(axis=0).tolist()}
    pct = {}
    rows = []
    for s in SCENARIOS:
        R = list(fixed_ST[s])
        M = np.array([fixed_ST[s][r] for r in R])           # (n_seeds, 14)
        pct[s] = {}
        for i, n in enumerate(names):
            if M[:, i].max() <= NEGLIGIBLE:
                continue
            v42 = fixed_ST[s][REF][i]
            pct[s][n] = {"seed42": float(v42), "median": float(np.median(M[:, i])),
                         "min": float(M[:, i].min()), "max": float(M[:, i].max()),
                         "sd": float(M[:, i].std(ddof=1)),
                         "seed42_rank_among_seeds_1=lowest": int((M[:, i] < v42).sum() + 1),
                         "n_seeds": len(R)}
            rows.append([s, n, float(v42), pct[s][n]["median"], pct[s][n]["min"], pct[s][n]["max"],
                         pct[s][n]["sd"], f"{pct[s][n]['seed42_rank_among_seeds_1=lowest']}/{len(R)}"])
        rank1 = {r: names[int(np.argmax(fixed_ST[s][r]))] for r in R}
        supp.setdefault(s, {})["rank1_param_per_fixed_seed"] = rank1
        supp[s]["top4_per_fixed_seed"] = {r: [names[i] for i in np.argsort(-fixed_ST[s][r])[:4]] for r in R}
        supp[s]["sum_ST_delivery4_per_fixed_seed"] = {
            r: float(sum(fixed_ST[s][r][idx[p]] for p in DELIVERY4)) for r in R}
    out["fixed_seed_replicates_distinct"] = {"seed42_position": pct, "extra": supp,
                                      "ST": {s: {r: v.tolist() for r, v in fixed_ST[s].items()} for s in SCENARIOS}}
    md += [f"## 8. Where seed 42 sits among {len(fixed_ST[SCENARIOS[0]])} distinct fixed-seed replicates",
           "(6 A runs + the per-seed columns of run C, re-analysed, excluding C's seed-7 column, which is bit-identical to A_seed7; no new model evaluations)", "",
           md_table(["scenario", "param", "S_T seed 42", "median", "min", "max", "SD", "seed-42 rank (1=lowest)"], rows), ""]
    if c_id:
        md += ["### Variance decomposition of Y in the CRN design (C per-seed matrix, 7680 rows × 10 seeds)", "",
               md_table(["scenario", "Var total", "parameter main", "seed main", "row×seed interaction"],
                        [[s, supp[s]["var_total_rows_x_seeds"], supp[s]["frac_parameter_main"],
                          supp[s]["frac_seed_main"], supp[s]["frac_interaction"]] for s in SCENARIOS]), ""]
    rows = [[s, ", ".join(f"{p}: {sum(1 for r in supp[s]['rank1_param_per_fixed_seed'].values() if r == p)}"
                          for p in sorted(set(supp[s]["rank1_param_per_fixed_seed"].values()))),
             min(supp[s]["sum_ST_delivery4_per_fixed_seed"].values()),
             float(np.median(list(supp[s]["sum_ST_delivery4_per_fixed_seed"].values()))),
             max(supp[s]["sum_ST_delivery4_per_fixed_seed"].values()),
             supp[s]["sum_ST_delivery4_per_fixed_seed"][REF]] for s in SCENARIOS]
    md += ["### Rank-1 parameter and Σ S_T(delivery4) across the distinct fixed seeds", "",
           md_table(["scenario", "rank-1 param (count)", "ΣS_T del4 min", "median", "max", "seed 42"], rows), ""]

    # ---- 9. width contrasts that actually change the parameter's width ----
    wc = {}
    rows = []
    for p, contrasts in WIDTH_CONTRASTS.items():
        for wide, narrow in contrasts:
            ratios = {r: float(fixed_ST[narrow][r][idx[p]] / fixed_ST[wide][r][idx[p]])
                      for r in fixed_ST[wide] if fixed_ST[wide][r][idx[p]] > 0}
            for r in [c_id] if c_id else []:
                ratios[r] = float(ST[(r, narrow)][idx[p]] / ST[(r, wide)][idx[p]])
            wc[f"{p}: {narrow}/{wide}"] = ratios
            vals = [v for r, v in ratios.items() if r != c_id]
            rows.append([p, f"{narrow} / {wide}", ratios[REF], float(np.median(vals)), min(vals), max(vals),
                         ratios.get(c_id), "yes" if all(v < 1 for v in ratios.values()) else "NO"])
    out["width_contrasts"] = wc
    md += ["## 9. Prior-width contrasts (only scenario pairs where the parameter's own width differs)", "",
           "Ratio = S_T(narrower prior) / S_T(wider prior). Fixed-seed columns: distinct fixed-seed replicates (15).", "",
           md_table(["param", "contrast", "seed 42", "median (15)", "min (15)", "max (15)", c_id or "C",
                     "narrower < wider in all"], rows), ""]

    # ---- headline verdicts ---------------------------------------------
    fixed_like = [r for r in order if not r.startswith("B_")]
    def every(fn, rs):
        return all(fn(r, s) for r in rs for s in SCENARIOS if (r, s) in ST)
    ref_top4 = {s: set(per_run[REF][s]["top4"]) for s in SCENARIOS}
    verdict = {
        "rank1_param_same_as_seed42_all_distinct_fixed_seeds_and_C": all(
            v == names[int(np.argmax(ST[(REF, s)]))] for s in SCENARIOS
            for v in list(supp[s]["rank1_param_per_fixed_seed"].values()) +
            ([names[int(np.argmax(ST[(c_id, s)]))]] if c_id else [])),
        "top4_set_same_as_seed42_all_A_and_C": every(lambda r, s: set(per_run[r][s]["top4"]) == ref_top4[s], fixed_like),
        "fus_ST_below_0.01_all_A_and_C": every(lambda r, s: ST[(r, s)][idx[FUS]] < 0.01, fixed_like),
        "fus_ST_below_0.01_B": every(lambda r, s: ST[(r, s)][idx[FUS]] < 0.01, [r for r in order if r.startswith("B_")]),
        "strict_3scenario_ordering_holds_all_runs_width_varied_params": all(sec5[p]["all_hold"] for p in WIDTH_VARIED),
        "width_narrowing_lowers_ST_all_distinct_fixed_seeds_and_C": all(
            v < 1 for ratios in wc.values() for v in ratios.values()),
        "max_narrow_over_wide_ratio_k_T4_entry_eq_vs_asdoc": max(wc["k_T4_entry: equal_width/as_documented"].values()),
        "max_narrow_over_wide_ratio_t4_dose_eq_vs_asdoc": max(wc["t4_dose_total: equal_width/as_documented"].values()),
        "no_rank_shift_ge2_among_nonnegligible_params_A_runs": all(
            all(n not in per_run[r][s]["nonzero_params_in_ref"] for n, _, _ in per_run[r][s]["rank_shifts_ge2"])
            for r in order if r.startswith("A_") for s in SCENARIOS if s in per_run.get(r, {})),
        "min_spearman_nonnegligible_A_and_C": float(min(per_run[r][s]["spearman_nonzero"]
                                                        for r in fixed_like for s in per_run[r])),
        "min_spearman_nonnegligible_B": float(min(per_run[r][s]["spearman_nonzero"]
                                                  for r in order if r.startswith("B_") for s in per_run[r])),
    }
    out["headline_checks"] = verdict
    md += ["## Headline checks", "", md_table(["check", "result"], [[k, v] for k, v in verdict.items()]), ""]

    with open(os.path.join(HERE, f"seed_sensitivity_summary_{os.path.basename(rundir)}.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(HERE, f"SUMMARY_{os.path.basename(rundir)}.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
