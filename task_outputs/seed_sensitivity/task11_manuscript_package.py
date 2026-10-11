"""
task11_manuscript_package.py

Makes the task10 seed-averaged Sobol results table-ready for the manuscript.
NO model evaluations: every number comes from Y arrays saved by
task10_seed_sensitivity.py in N256/ (plus the shared X sample there).

Writes (next to this script):
  seed_averaged_percentile_cis.json  run-C S_T + 95% percentile CIs, same record
                                     schema as task_outputs/percentile_cis.json
                                     (extra keys: S1/S2), 3 records
  fixed_seed_table.csv               one row per (scenario, fixed seed), 16 seeds
  fixed_seed_param_summary.csv       per (scenario, param) across the distinct fixed seeds
                                     (15: seed 7 appears in both run A and run C)
  manuscript_numbers.json            items (a)-(g) + half-split convergence (step 4)
                                     + sanity checks

Settings are identical to task7 / task10: SALib sobol.analyze,
calc_second_order=True, num_resamples=1000, conf_level=0.95, seed=8162026;
percentile CI = 2.5/97.5 percentiles of SALib's *_conf_all resamples.
"""
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version

import numpy as np
from SALib.analyze.sobol import analyze as sobol_analyze
from SALib.sample.sobol import sample as sobol_sample
from scipy.stats import kendalltau, rankdata, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RUNDIR = os.path.join(HERE, "N256")
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MODEL_DIR = os.path.join(REPO, "gm2_analysis")
sys.path.insert(0, os.path.join(MODEL_DIR, "drivers"))
sys.path.insert(0, MODEL_DIR)

with open(os.path.join(MODEL_DIR, "gm2_model_v2.py"), "rb") as f:
    MODEL_SHA256 = hashlib.sha256(f.read()).hexdigest()
assert MODEL_SHA256.startswith("6693427fa909a0a9"), MODEL_SHA256

import sobol_driver_v2 as sdv2  # noqa: E402  (PARAM_NAMES / make_bounds only)

NAMES = sdv2.PARAM_NAMES
D = len(NAMES)
IDX = {n: i for i, n in enumerate(NAMES)}
SCENARIOS = ["as_documented", "equal_width", "tier1_promotion"]
N = 256
SOBOL_SAMPLE_SEED = 20260816
BOOT_B, BOOT_SEED, CONF = 1000, 8162026, 0.95
NEGLIGIBLE = 1e-4                       # same threshold as task10_summary.py
DELIVERY4 = ["k_T4_entry", "t4_dose_total", "k_T4_payload", "k_T4_decay"]
FUS = "fus_entry_gain_scale"
A_SEEDS = [42, 7, 101, 2024, 31337, 90210]
C_ID = "C_mean10_seeds0-9"
B_ID = "B_perrow1000"


def analyze(Y):
    Si = sobol_analyze({"num_vars": D, "names": NAMES}, np.asarray(Y, float),
                       calc_second_order=True, num_resamples=BOOT_B, conf_level=CONF,
                       print_to_console=False, keep_resamples=True, seed=BOOT_SEED)
    st_all = np.asarray(Si["ST_conf_all"])
    s1_all = np.asarray(Si["S1_conf_all"])
    return {
        "ST": np.asarray(Si["ST"]), "ST_lo": np.percentile(st_all, 2.5, axis=0),
        "ST_hi": np.percentile(st_all, 97.5, axis=0), "ST_all": st_all,
        "ST_conf": np.asarray(Si["ST_conf"]),
        "S1": np.asarray(Si["S1"]), "S1_lo": np.percentile(s1_all, 2.5, axis=0),
        "S1_hi": np.percentile(s1_all, 97.5, axis=0), "S1_conf": np.asarray(Si["S1_conf"]),
        "S2": np.asarray(Si["S2"]), "S2_conf": np.asarray(Si["S2_conf"]),
    }


def desc_rank(v):
    return rankdata(-np.asarray(v), method="average")


def corr(a, b, mask=None):
    a, b = np.asarray(a), np.asarray(b)
    if mask is not None:
        a, b = a[mask], b[mask]
    return {"spearman": float(spearmanr(a, b).statistic), "kendall_tau_b": float(kendalltau(a, b).statistic),
            "n_params": int(len(a))}


def jload(p):
    with open(p) as f:
        return json.load(f)


def main():
    checks = {"unexpected": []}
    def flag(msg):
        checks["unexpected"].append(msg)
        print("!!! " + msg, flush=True)

    # ---------- load saved arrays, verify X ----------------------------------
    X, Yc, Ymat, Yfixed, metaC, Yb = {}, {}, {}, {}, {}, {}
    for s in SCENARIOS:
        X[s] = np.load(os.path.join(RUNDIR, f"X_{s}.npy"))
        Xre = sobol_sample({"num_vars": D, "names": NAMES, "bounds": sdv2.make_bounds(s)},
                           N=N, calc_second_order=True, seed=SOBOL_SAMPLE_SEED)
        assert np.array_equal(X[s], Xre), f"saved X_{s} != regenerated task7 sample"
        jc = jload(os.path.join(RUNDIR, f"{C_ID}_{s}.json"))
        metaC[s] = jc
        assert jc["run_meta"]["X_sha256"] == hashlib.sha256(np.ascontiguousarray(X[s]).tobytes()).hexdigest()
        Yc[s] = np.load(os.path.join(RUNDIR, f"Y_{C_ID}_{s}.npy"))
        Ymat[s] = np.load(os.path.join(RUNDIR, f"Y_{C_ID}_{s}_perseed.npy"))
        Yb[s] = np.load(os.path.join(RUNDIR, f"Y_{B_ID}_{s}.npy"))
        d = float(np.max(np.abs(Ymat[s].mean(axis=1) - Yc[s])))
        checks.setdefault("C_mean_of_perseed_equals_saved_C_Y_maxabs", {})[s] = d
        if d > 1e-9:
            flag(f"{s}: saved run-C Y differs from mean of its per-seed columns by {d}")
        seedsC = jc["run_meta"]["seeds"]
        Yfixed[s] = {}
        for sd in A_SEEDS:
            Yfixed[s][("A", sd)] = np.load(os.path.join(RUNDIR, f"Y_A_seed{sd}_{s}.npy"))
        for k, sd in enumerate(seedsC):
            Yfixed[s][("Ccol", sd)] = Ymat[s][:, k]
        for key, y in list(Yfixed[s].items()) + [(("C", "mean"), Yc[s]), (("B", "perrow"), Yb[s])]:
            if not np.all(np.isfinite(y)):
                flag(f"{s} {key}: {int(np.sum(~np.isfinite(y)))} non-finite Y")
    seeds_C = metaC[SCENARIOS[0]]["run_meta"]["seeds"]

    # duplicate seed values / identical Y columns among the 16 fixed-seed runs
    all_seed_vals = A_SEEDS + list(seeds_C)
    dup_vals = sorted({v for v in all_seed_vals if all_seed_vals.count(v) > 1})
    checks["duplicate_seed_values_among_16"] = dup_vals
    if dup_vals:
        flag(f"duplicate seed values among the 16 fixed-seed runs: {dup_vals}")
    identical = []
    for s in SCENARIOS:
        keys = list(Yfixed[s])
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if np.array_equal(Yfixed[s][keys[i]], Yfixed[s][keys[j]]):
                    identical.append([s, list(keys[i]), list(keys[j])])
    checks["identical_Y_pairs_among_16"] = identical
    if identical:
        flag(f"identical Y arrays between distinct fixed-seed runs: {identical}")

    # Distinct fixed-seed set: a run-C column whose seed is also an A seed is the
    # same noise path (verified bit-identical above when it happens); keep the A copy.
    for s in SCENARIOS:
        for sd in dup_vals:
            if ("Ccol", sd) in Yfixed[s] and ("A", sd) in Yfixed[s]:
                assert np.array_equal(Yfixed[s][("Ccol", sd)], Yfixed[s][("A", sd)]), (s, sd)
    DISTINCT = [k for k in Yfixed[SCENARIOS[0]] if not (k[0] == "Ccol" and k[1] in A_SEEDS)]
    checks["n_fixed_seed_rows"] = len(Yfixed[SCENARIOS[0]])
    checks["n_distinct_fixed_seeds"] = len(DISTINCT)
    checks["distinct_fixed_seeds"] = [k[1] for k in DISTINCT]

    # ---------- analyses ----------------------------------------------------
    resC = {s: analyze(Yc[s]) for s in SCENARIOS}
    resF = {s: {k: analyze(y) for k, y in Yfixed[s].items()} for s in SCENARIOS}
    resB = {s: analyze(Yb[s]) for s in SCENARIOS}

    # cross-check against task10's own JSON outputs (same Y, same settings)
    xc = {}
    for s in SCENARIOS:
        xc[s] = {"C": float(np.max(np.abs(resC[s]["ST"] - np.array(metaC[s]["ST"]))))}
        for sd in A_SEEDS:
            j = jload(os.path.join(RUNDIR, f"A_seed{sd}_{s}.json"))
            xc[s][f"A_seed{sd}"] = float(np.max(np.abs(resF[s][("A", sd)]["ST"] - np.array(j["ST"]))))
    checks["max_abs_dST_vs_task10_json"] = xc
    if max(v for d in xc.values() for v in d.values()) > 1e-12:
        flag(f"recomputed S_T differs from task10 JSON: {xc}")
    pc = {r["scenario"]: r for r in jload(os.path.join(REPO, "task_outputs", "percentile_cis.json"))
          if r.get("source") == "rerun"}
    d42 = {s: float(np.max(np.abs(resF[s][("A", 42)]["ST"] - np.array(pc[s]["ST_mean"])))) for s in SCENARIOS}
    checks["seed42_max_abs_dST_vs_percentile_cis_json"] = d42
    if max(d42.values()) > 1e-12:
        flag(f"seed-42 recompute differs from percentile_cis.json: {d42}")

    # CI-excludes-point-estimate and sum checks, over every analysis made here
    excl = []
    def ci_check(label, s, r):
        for i, n in enumerate(NAMES):
            if not (r["ST_lo"][i] - 1e-12 <= r["ST"][i] <= r["ST_hi"][i] + 1e-12):
                excl.append({"run": label, "scenario": s, "param": n, "ST": float(r["ST"][i]),
                             "ci": [float(r["ST_lo"][i]), float(r["ST_hi"][i])]})
    for s in SCENARIOS:
        ci_check(C_ID, s, resC[s])
        for k, r in resF[s].items():
            ci_check(f"{k[0]}_seed{k[1]}", s, r)
    checks["ST_point_outside_own_percentile_CI"] = excl
    if excl:
        flag(f"{len(excl)} (run, scenario, param) S_T point estimates fall outside their own 95% percentile CI "
             f"(see checks.ST_point_outside_own_percentile_CI)")

    # ---------- step 1: seed_averaged_percentile_cis.json ---------------------
    records = []
    for s in SCENARIOS:
        r = resC[s]
        prod = jload(os.path.join(MODEL_DIR, "sobol_outputs", f"gm2_full14_{s}.json"))
        rm = metaC[s]["run_meta"]
        s2 = [[None if np.isnan(v) else float(v) for v in row] for row in r["S2"]]
        s2c = [[None if np.isnan(v) else float(v) for v in row] for row in r["S2_conf"]]
        records.append({
            "disease": "gm2_tay_sachs_full14",
            "scenario": s,
            "source": "rerun",
            "names": NAMES,
            "n_evals": int(len(Yc[s])),
            "Y_mean": float(np.mean(Yc[s])), "Y_std": float(np.std(Yc[s])),
            "ST_mean": r["ST"].tolist(),
            "ST_ci_lo_2.5": r["ST_lo"].tolist(),
            "ST_ci_hi_97.5": r["ST_hi"].tolist(),
            "ST_bootstrap_resamples": r["ST_all"].tolist(),
            "ST_conf_symmetric_original_method": r["ST_conf"].tolist(),
            "note": ("SEED-AVERAGED Sobol (task10 run C): each of the n_evals rows is the mean of "
                     f"run_single over model seeds {seeds_C[0]}-{seeds_C[-1]} (same 10 seeds for every "
                     "row, common random numbers), on the same Sobol X as the task7 rerun "
                     f"(sample seed {SOBOL_SAMPLE_SEED}). Recomputed by task11_manuscript_package.py "
                     "from the saved Y array only; no model evaluations. Same record schema as "
                     "percentile_cis.json; extra keys S1*, S2* added."),
            "max_abs_diff_vs_archived_ST_if_available": float(np.max(np.abs(r["ST"] - np.array(prod["ST"])))),
            "run_meta": {
                "N": N, "D": D, "dt": rm["dt"], "model_eval_seed": list(seeds_C),
                "sobol_sample_seed": SOBOL_SAMPLE_SEED, "bootstrap_B": BOOT_B,
                "bootstrap_seed": BOOT_SEED, "conf_level": CONF, "calc_second_order": True,
                "elapsed_s": rm["elapsed_s"], "n_workers": rm["n_workers"],
                "model_sha256_16": MODEL_SHA256[:16],
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "seed_scheme": "seed_averaged_crn (row value = mean over model_eval_seed list)",
                "n_model_evals": int(Ymat[s].size), "tmax_days": rm["tmax_days"],
                "source_Y_file": f"N256/Y_{C_ID}_{s}.npy", "source_slurm_job": rm["slurm_job_id"],
                "salib_version": pkg_version("SALib"),
            },
            "S1": r["S1"].tolist(), "S1_ci_lo_2.5": r["S1_lo"].tolist(), "S1_ci_hi_97.5": r["S1_hi"].tolist(),
            "S1_conf_symmetric_original_method": r["S1_conf"].tolist(),
            "S2_matrix": s2, "S2_conf_symmetric_matrix": s2c,
            "S2_note": "upper triangle (i<j) populated; S2 has no resamples in SALib 1.5.2, so only the symmetric conf is available",
        })
    with open(os.path.join(HERE, "seed_averaged_percentile_cis.json"), "w") as f:
        json.dump(records, f, indent=2)
    ref_keys = set(pc[SCENARIOS[0]].keys())
    checks["schema_missing_keys_vs_percentile_cis_rerun"] = sorted(ref_keys - set(records[0].keys()))
    checks["schema_type_changes"] = sorted(k for k in ref_keys if type(records[0][k]) is not type(pc[SCENARIOS[0]][k]))
    checks["schema_run_meta_type_changes"] = sorted(
        k for k in pc[SCENARIOS[0]]["run_meta"] if type(records[0]["run_meta"].get(k)) is not type(pc[SCENARIOS[0]]["run_meta"][k]))

    # ---------- step 2: fixed-seed CSVs --------------------------------------
    def label(k):
        return ("task10_runA_seed42_control" if k == ("A", 42) else
                "task10_runA_fixed_seed" if k[0] == "A" else "task10_runC_perseed_column")
    rows = []
    for s in SCENARIOS:
        for k, r in resF[s].items():
            rows.append({"scenario": s, "run_label": f"A_seed{k[1]}" if k[0] == "A" else f"C_col_seed{k[1]}",
                         "seed": k[1], "source": label(k),
                         "duplicate_seed_value": k[1] in dup_vals,
                         "in_distinct_set": k in DISTINCT,
                         "Y_mean": float(np.mean(Yfixed[s][k])), "Y_var": float(np.var(Yfixed[s][k])),
                         "sum_ST": float(r["ST"].sum()),
                         **{f"ST_{n}": float(r["ST"][i]) for i, n in enumerate(NAMES)}})
    with open(os.path.join(HERE, "fixed_seed_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summ = []
    sixteen = {}
    for s in SCENARIOS:
        M = np.array([resF[s][k]["ST"] for k in DISTINCT])
        sixteen[s] = M
        r42 = resF[s][("A", 42)]["ST"]
        for i, n in enumerate(NAMES):
            v = M[:, i]
            q25, q75 = np.percentile(v, [25, 75])
            summ.append({"scenario": s, "param": n, "n_seeds": len(v),
                         "ST_seed42": float(r42[i]), "ST_median": float(np.median(v)),
                         "ST_min": float(v.min()), "ST_max": float(v.max()),
                         "ST_q25": float(q25), "ST_q75": float(q75), "ST_IQR": float(q75 - q25),
                         "seed42_rank_1_is_lowest": int((v < r42[i]).sum() + 1),
                         "negligible_all_seeds": bool(v.max() <= NEGLIGIBLE),
                         "ST_seed_averaged_C": float(resC[s]["ST"][i])})
    with open(os.path.join(HERE, "fixed_seed_param_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)

    # ---------- step 3: manuscript_numbers.json ------------------------------
    mn = {"generated_by": "task11_manuscript_package.py", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
          "model_sha256": MODEL_SHA256, "negligible_threshold": NEGLIGIBLE, "delivery4": DELIVERY4}
    # (a)
    a = {}
    for s in SCENARIOS:
        r = resC[s]
        order = np.argsort(-r["ST"])
        a[s] = [{"rank": int(k + 1), "param": NAMES[i], "ST": float(r["ST"][i]),
                 "ci_lo_2.5": float(r["ST_lo"][i]), "ci_hi_97.5": float(r["ST_hi"][i]),
                 "negligible": bool(r["ST"][i] <= NEGLIGIBLE)} for k, i in enumerate(order)]
    mn["a_seed_averaged_ST_ranking"] = a
    # (b)
    b = {}
    for s in SCENARIOS:
        st42, stc = resF[s][("A", 42)]["ST"], resC[s]["ST"]
        mask = (st42 > NEGLIGIBLE) | (stc > NEGLIGIBLE)
        r42, rc = desc_rank(st42), desc_rank(stc)
        b[s] = {"all14": corr(st42, stc), "non_negligible": corr(st42, stc, mask),
                "non_negligible_params": [n for n, m in zip(NAMES, mask) if m],
                "rank_seed42": dict(zip(NAMES, map(float, r42))),
                "rank_seed_averaged": dict(zip(NAMES, map(float, rc))),
                "rank_shift_ge2": [n for i, n in enumerate(NAMES) if abs(r42[i] - rc[i]) >= 2]}
    mn["b_seed42_vs_seed_averaged_rank_correlation"] = b
    # (c)
    c = {}
    for s in SCENARIOS:
        c[s] = {}
        for p in DELIVERY4:
            i = IDX[p]
            c[s][p] = {"seed42": float(resF[s][("A", 42)]["ST"][i]),
                       "seed42_ci": [float(resF[s][("A", 42)]["ST_lo"][i]), float(resF[s][("A", 42)]["ST_hi"][i])],
                       "seed_averaged": float(resC[s]["ST"][i]),
                       "seed_averaged_ci": [float(resC[s]["ST_lo"][i]), float(resC[s]["ST_hi"][i])],
                       "median_distinct_seeds": float(np.median(sixteen[s][:, i]))}
        c[s]["sum_delivery4"] = {
            "seed42": float(sum(resF[s][("A", 42)]["ST"][IDX[p]] for p in DELIVERY4)),
            "seed_averaged": float(sum(resC[s]["ST"][IDX[p]] for p in DELIVERY4)),
            "median_distinct_seeds": float(np.median(sixteen[s][:, [IDX[p] for p in DELIVERY4]].sum(axis=1)))}
    mn["c_delivery4_ST"] = c
    # (d)
    dd = {}
    key_params = [n for n in NAMES if max(resC[s]["ST"][IDX[n]] for s in SCENARIOS) > NEGLIGIBLE]
    for p in key_params + [FUS]:
        i = IDX[p]
        dd[p] = {}
        for num in ["equal_width", "tier1_promotion"]:
            per_seed = sixteen[num][:, i] / sixteen["as_documented"][:, i]
            dd[p][f"{num}_over_as_documented"] = {
                "seed42": float(resF[num][("A", 42)]["ST"][i] / resF["as_documented"][("A", 42)]["ST"][i]),
                "seed_averaged": float(resC[num]["ST"][i] / resC["as_documented"]["ST"][i]),
                "median_distinct_seeds": float(np.median(per_seed)),
                "min_distinct_seeds": float(per_seed.min()), "max_distinct_seeds": float(per_seed.max())}
        if p == FUS:
            dd[p]["note"] = "S_T is numerically ~0 (<= NEGLIGIBLE) in every scenario; ratio is meaningless"
    mn["d_prior_width_ratios"] = {"key_params_rule": f"seed-averaged S_T > {NEGLIGIBLE} in any scenario",
                                  "width_varied_params": ["k_T4_entry", "t4_dose_total"], "ratios": dd}
    # (e)
    fus_fixed = max(float(r["ST"][IDX[FUS]]) for s in SCENARIOS for r in resF[s].values())
    fus_C = max(float(resC[s]["ST"][IDX[FUS]]) for s in SCENARIOS)
    fus_C_hi = max(float(resC[s]["ST_hi"][IDX[FUS]]) for s in SCENARIOS)
    fus_B = max(float(resB[s]["ST"][IDX[FUS]]) for s in SCENARIOS)
    mn["e_max_FUS_ST"] = {"max_over_all_fixed_seed_runs_x_3_scenarios": fus_fixed,
                          "max_over_seed_averaged_3_scenarios": fus_C,
                          "max_seed_averaged_ci_hi": fus_C_hi,
                          "max_over_all_fixed_and_seed_averaged": max(fus_fixed, fus_C),
                          "per_row_seed_run_B_max_(noise_artifact,_not_parameter_effect)": fus_B}
    # (f)
    mn["f_seed_averaged_sums"] = {s: {"sum_ST": float(resC[s]["ST"].sum()), "sum_S1": float(resC[s]["S1"].sum()),
                                      "Var_Y": float(np.var(Yc[s])), "Y_mean": float(np.mean(Yc[s]))}
                                  for s in SCENARIOS}
    # (g)
    mn["g_run_metadata"] = {
        "N": N, "D": D, "rows_per_scenario": int(len(Yc[SCENARIOS[0]])), "dt": metaC[SCENARIOS[0]]["run_meta"]["dt"],
        "tmax_days": metaC[SCENARIOS[0]]["run_meta"]["tmax_days"], "sobol_sample_seed": SOBOL_SAMPLE_SEED,
        "calc_second_order": True, "bootstrap_B": BOOT_B, "bootstrap_seed": BOOT_SEED, "conf_level": CONF,
        "runs": {
            "seed_averaged_C": {"seeds": list(seeds_C), "model_evals_per_row": len(seeds_C),
                                "rows": int(len(Yc[SCENARIOS[0]])), "model_evals_per_scenario": int(Ymat[SCENARIOS[0]].size),
                                "slurm_job": metaC[SCENARIOS[0]]["run_meta"]["slurm_job_id"]},
            "fixed_seed_A": {"seeds": A_SEEDS, "rows": int(len(Yc[SCENARIOS[0]])), "model_evals_per_row": 1},
            "fixed_seed_from_C_columns": {"seeds": list(seeds_C), "rows": int(len(Yc[SCENARIOS[0]])),
                                          "model_evals_per_row": 1, "note": "per-seed columns of run C, re-analysed"},
            "per_row_seed_B": {"seeds": "row i -> 1000+i", "rows": int(len(Yc[SCENARIOS[0]]))}},
        "salib_version": pkg_version("SALib"), "numpy_version": np.__version__}

    # ---------- step 4: half-split convergence ------------------------------
    hs = {}
    for s in SCENARIOS:
        h1 = analyze(Ymat[s][:, :5].mean(axis=1))
        h2 = analyze(Ymat[s][:, 5:].mean(axis=1))
        t1 = [NAMES[i] for i in np.argsort(-h1["ST"])[:5]]
        t2 = [NAMES[i] for i in np.argsort(-h2["ST"])[:5]]
        tc = [NAMES[i] for i in np.argsort(-resC[s]["ST"])[:5]]
        mask = (h1["ST"] > NEGLIGIBLE) | (h2["ST"] > NEGLIGIBLE)
        top5_idx = sorted({IDX[n] for n in t1 + t2})
        hs[s] = {"seeds_half1": list(seeds_C[:5]), "seeds_half2": list(seeds_C[5:]),
                 "top5_half1": t1, "top5_half2": t2, "top5_full10": tc,
                 "top5_same_order": t1 == t2, "top5_same_set": set(t1) == set(t2),
                 "spearman_kendall_all14": corr(h1["ST"], h2["ST"]),
                 "spearman_kendall_non_negligible": corr(h1["ST"], h2["ST"], mask),
                 "spearman_kendall_union_of_top5": corr(h1["ST"][top5_idx], h2["ST"][top5_idx]),
                 "ST_half1": dict(zip(NAMES, map(float, h1["ST"]))),
                 "ST_half2": dict(zip(NAMES, map(float, h2["ST"]))),
                 "max_abs_dST_half1_vs_half2": float(np.max(np.abs(h1["ST"] - h2["ST"]))),
                 "argmax_abs_dST_half1_vs_half2": NAMES[int(np.argmax(np.abs(h1["ST"] - h2["ST"])))],
                 # The bootstrap CI resamples Sobol rows only; it does not include
                 # uncertainty from the finite set of noise seeds. Record where a
                 # 5-seed half falls outside the 10-seed CI.
                 "half_ST_outside_full10_bootstrap_CI": [
                     n for i, n in enumerate(NAMES) if resC[s]["ST"][i] > NEGLIGIBLE and not (
                         resC[s]["ST_lo"][i] <= h1["ST"][i] <= resC[s]["ST_hi"][i] and
                         resC[s]["ST_lo"][i] <= h2["ST"][i] <= resC[s]["ST_hi"][i])]}
        if hs[s]["half_ST_outside_full10_bootstrap_CI"]:
            flag(f"{s}: 5-seed half-average S_T falls outside the 10-seed bootstrap CI for "
                 f"{hs[s]['half_ST_outside_full10_bootstrap_CI']} -- CIs omit noise-seed uncertainty")
        if not hs[s]["top5_same_set"]:
            flag(f"{s}: top-5 set differs between seed halves {hs[s]['top5_half1']} vs {hs[s]['top5_half2']}")
    mn["step4_half_split_convergence"] = hs

    # sum-near-1 check
    for s in SCENARIOS:
        if abs(mn["f_seed_averaged_sums"][s]["sum_ST"] - 1) > 0.15:
            flag(f"{s}: seed-averaged sum S_T = {mn['f_seed_averaged_sums'][s]['sum_ST']:.3f} (not near 1)")
    mn["checks"] = checks
    with open(os.path.join(HERE, "manuscript_numbers.json"), "w") as f:
        json.dump(mn, f, indent=2)

    # ---------- print -------------------------------------------------------
    print(json.dumps({k: v for k, v in mn.items()}, indent=1, default=str))


if __name__ == "__main__":
    main()
