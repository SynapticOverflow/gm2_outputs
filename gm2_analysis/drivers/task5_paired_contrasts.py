"""
task5_paired_contrasts.py

Task 5: paired matched-seed contrasts + Shapley values with bootstrap CIs.

Uses the EXISTING factorial raw per-seed data (data/factorial_arms_shapley*.json)
as-is -- task 4's investigation found no bug/config difference in the
current gm2_model_v2/factorial_arms_shapley.py pipeline that needed fixing
(the AAV~AAV+FUS near-equality there is the corrected model's real
behavior, not an artifact; see data/task4_aav_fus_discrepancy.json), so
"the corrected, per task 4, factorial raw per-seed data" is this existing
data, unmodified and not re-run.

Reuses factorial_arms_shapley.exact_shapley() UNCHANGED for the Shapley
game; only adds a bootstrap wrapper around it (resampling the 200 matched
seed-pairs jointly across all 8 arms, so the common-random-numbers pairing
structure is preserved in every resample) and a paired-difference /
percentile-bootstrap-CI helper for the 4 named contrasts.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factorial_arms_shapley import exact_shapley

B = 1000
SEED = 42
_HERE = os.path.dirname(os.path.abspath(__file__))

CONTRASTS = [
    ("untreated_vs_FUS_alone", (0, 0, 0), (0, 0, 1)),
    ("AAV_vs_AAV_FUS", (0, 1, 0), (0, 1, 1)),
    ("SRT_vs_SRT_FUS", (1, 0, 0), (1, 0, 1)),
    ("SRT_AAV_vs_tri_modal", (1, 1, 0), (1, 1, 1)),
]

DAY_FILES = {
    45.0: "data/factorial_arms_shapley_day45.json",
    90.0: "data/factorial_arms_shapley_day90.json",
    180.0: "data/factorial_arms_shapley_day180.json",
    365.0: "data/factorial_arms_shapley.json",
}


def load_arms(path):
    d = json.load(open(os.path.join(_HERE, path)))
    arms = {(a["SRT"], a["AAV"], a["FUS"]): np.asarray(a["raw_burden"], dtype=float) for a in d["arms"]}
    return d, arms


def paired_contrast(arms, key_a, key_b, rng):
    diff = arms[key_a] - arms[key_b]  # burden(A) - burden(B); positive = B (extra intervention) reduced burden
    n = len(diff)
    mean_diff = float(diff.mean())
    boot_means = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = diff[idx].mean()
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    return {
        "mean_paired_diff": mean_diff,
        "ci95_lo": float(ci_lo), "ci95_hi": float(ci_hi),
        "n_pairs": n, "n_bootstrap": B,
        "significant": bool(ci_lo > 0 or ci_hi < 0),
    }


def shapley_with_ci(arms, no_treatment_key, rng):
    players = ("SRT", "AAV", "FUS")
    key_to_S = {}
    for srt, aav, fus in arms:
        S = frozenset(x for x, on in zip(players, (srt, aav, fus)) if on)
        key_to_S[(srt, aav, fus)] = S

    def point_and_boot():
        no_tx_mean = arms[no_treatment_key].mean()
        v = {key_to_S[k]: no_tx_mean - arr.mean() for k, arr in arms.items()}
        return exact_shapley(v, players=players)

    point = point_and_boot()
    n = len(arms[no_treatment_key])
    boot = {p: np.empty(B) for p in players}
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        resampled = {k: arr[idx] for k, arr in arms.items()}
        no_tx_mean = resampled[no_treatment_key].mean()
        v = {key_to_S[k]: no_tx_mean - arr.mean() for k, arr in resampled.items()}
        sh = exact_shapley(v, players=players)
        for p in players:
            boot[p][b] = sh[p]

    out = {}
    for p in players:
        lo, hi = np.percentile(boot[p], [2.5, 97.5])
        out[p] = {"point_estimate": float(point[p]), "ci95_lo": float(lo), "ci95_hi": float(hi),
                   "bootstrap_se": float(boot[p].std(ddof=1))}
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)

    print("=== Task 5a: paired matched-seed contrasts ===")
    contrasts_out = {"run_meta": {"B": B, "seed": SEED, "method": "percentile bootstrap on 200 matched seed-pairs"}}
    for day, path in DAY_FILES.items():
        if not os.path.exists(os.path.join(_HERE, path)):
            print(f"  [day {day}] SKIPPED -- {path} not found")
            continue
        d, arms = load_arms(path)
        day_out = {}
        print(f"\n  --- day {day:.0f} (n_seeds={d['n_seeds']}) ---")
        for name, key_a, key_b in CONTRASTS:
            res = paired_contrast(arms, key_a, key_b, rng)
            day_out[name] = res
            sig = "*" if res["significant"] else " "
            print(f"    {name:24s} mean_diff={res['mean_paired_diff']:9.4f}  "
                  f"95% CI=[{res['ci95_lo']:9.4f}, {res['ci95_hi']:9.4f}] {sig}")
        contrasts_out[f"day{int(day)}"] = day_out

    with open(os.path.join(_HERE, "data/paired_contrasts.json"), "w") as f:
        json.dump(contrasts_out, f, indent=2)
    print("\nWritten to data/paired_contrasts.json")

    print("\n=== Task 5b: Shapley values with bootstrap 95% CIs ===")
    shapley_out = {"run_meta": {"B": B, "seed": SEED, "method": "joint percentile bootstrap on 200 matched seed-pairs, all 8 arms resampled together"}}
    for day, path in DAY_FILES.items():
        if not os.path.exists(os.path.join(_HERE, path)):
            continue
        d, arms = load_arms(path)
        sh = shapley_with_ci(arms, (0, 0, 0), rng)
        shapley_out[f"day{int(day)}"] = sh
        print(f"\n  --- day {day:.0f} ---")
        for p, v in sh.items():
            print(f"    {p:5s} shapley={v['point_estimate']:9.4f}  "
                  f"95% CI=[{v['ci95_lo']:9.4f}, {v['ci95_hi']:9.4f}]  SE={v['bootstrap_se']:.4f}")

    with open(os.path.join(_HERE, "data/shapley_with_ci.json"), "w") as f:
        json.dump(shapley_out, f, indent=2)
    print("\nWritten to data/shapley_with_ci.json")
    print("Done.")
