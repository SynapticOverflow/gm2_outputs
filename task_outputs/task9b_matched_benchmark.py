"""
task9b_matched_benchmark.py

Matched-control recomputation of the AAV-only / SRT-only brain-substrate
reductions reported in task_outputs/external_benchmark.json.

The published percentages are each arm's change relative to its OWN initial
value (baseline_gm2_brain = 890). But the untreated arm in this model is not
flat -- gm2_brain falls on its own (the dG_B equation's flat -0.01*G_B decay
term, documented in task9's own comparison (a) assessment), so a
change-from-own-baseline number attributes spontaneous decline to treatment.
This script computes the matched (paired, same-seed, same-day) contrast
against a concurrently simulated untreated arm, which was never done.

Configuration is taken from task9_external_benchmark.py unchanged: same model
file, Tay-Sachs baseline, DT = 0.2, SEEDS = range(10), same arm parameters,
same nearest-index day sampling. gm2_model_v2.py is imported read-only and is
not modified.

Arm definitions (exactly as in task9):
  untreated : dose_mg_per_day=0.0, t4_dose_total=0.0, tmax=730, fus_on=False
  AAV-only  : dose_mg_per_day=0.0, t4_dose_total=1.0, tmax=200, fus_on=True
  SRT-only  : dose_mg_per_day=3.0, t4_dose_total=0.0, tmax=200, fus_on=True

Two controls on the comparison's validity are run and recorded rather than
assumed:
  * untreated_fus_on   -- the untreated arm with fus_on=True, i.e. differing
    from the treated arms ONLY in treatment. In this model m_fus enters only
    k_p2b_eff (drug plasma->brain) and k_entry_eff (vector entry), both of
    which are multiplied by states that are identically zero with no drug and
    no vector, so this should be bit-identical to fus_on=False. Checked, not
    asserted.
  * untreated_tmax200  -- the untreated arm integrated to 200 d instead of
    730 d. simulate_milstein builds its grid as linspace(0, tmax, ceil(tmax/dt)+1),
    which is exactly 0.2 d for both 200 and 730, and the RNG is drawn one
    standard_normal(dim) per step, so the two should agree exactly at days
    84/168. Checked, not asserted.

Output: task_outputs/external_benchmark_matched.json
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

# gm2_model_v2.py lives in gm2_analysis/; import it read-only.
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(HERE, "..", "gm2_analysis"))
sys.path.insert(0, MODEL_DIR)
from gm2_model_v2 import make_base_params, make_x0, simulate_milstein, DEFAULT_FUS_EVENTS

MODEL_FILE = os.path.join(MODEL_DIR, "gm2_model_v2.py")
with open(MODEL_FILE, "rb") as f:
    _blob = f.read()
MODEL_SHA256 = hashlib.sha256(_blob).hexdigest()

# --- task9 configuration, reused verbatim ---
N_SEEDS = 10
SEEDS = list(range(N_SEEDS))
DT = 0.2
DISEASE = "tay-sachs"
DAYS = [84, 168]

N_BOOT = 10000
BOOT_SEED = 20261005  # fixed so the CI is reproducible


def run_arm_seed(dose_mg_per_day, t4_dose_total, tmax_days, seed, fus_on):
    """Identical to task9_external_benchmark.run_arm_seed, with fus_on explicit."""
    base = make_base_params(DISEASE)
    x0 = make_x0(base)
    df = simulate_milstein(x0, tmax_days, DT, base, dose_mg_per_day=dose_mg_per_day,
                           t4_admin_day=30.0, t4_dose_total=t4_dose_total,
                           fus_events=DEFAULT_FUS_EVENTS if fus_on else None, seed=seed)
    return base, df


def sample_at(df, day, col):
    """Nearest-index sampling, identical to task9. No interpolation."""
    idx = (df["time_days"] - day).abs().idxmin()
    return float(df.loc[idx, col]), float(df.loc[idx, "time_days"])


ARMS = {
    "untreated":        dict(dose_mg_per_day=0.0, t4_dose_total=0.0, tmax_days=730.0, fus_on=False),
    "aav_only":         dict(dose_mg_per_day=0.0, t4_dose_total=1.0, tmax_days=200.0, fus_on=True),
    "srt_only":         dict(dose_mg_per_day=3.0, t4_dose_total=0.0, tmax_days=200.0, fus_on=True),
    # validity controls
    "untreated_fus_on": dict(dose_mg_per_day=0.0, t4_dose_total=0.0, tmax_days=730.0, fus_on=True),
    "untreated_tmax200": dict(dose_mg_per_day=0.0, t4_dose_total=0.0, tmax_days=200.0, fus_on=False),
}

# per-arm, per-day, per-seed gm2_brain -- the raw material for everything below
raw = {arm: {day: {} for day in DAYS} for arm in ARMS}
sampled_days = {arm: {} for arm in ARMS}
baseline_gm2_brain = None
errors = []

for arm, cfg in ARMS.items():
    for seed in SEEDS:
        try:
            base, df = run_arm_seed(cfg["dose_mg_per_day"], cfg["t4_dose_total"],
                                    cfg["tmax_days"], seed, cfg["fus_on"])
        except Exception as exc:  # report, never silently fill
            errors.append({"arm": arm, "seed": seed, "error": f"{type(exc).__name__}: {exc}"})
            continue
        baseline_gm2_brain = float(base["baseline_gm2_brain"])
        for day in DAYS:
            val, actual = sample_at(df, day, "gm2_brain")
            raw[arm][day][seed] = val
            sampled_days[arm][day] = actual
    print(f"[{arm}] done ({len(raw[arm][DAYS[0]])}/{N_SEEDS} seeds)", flush=True)

if errors:
    print("ERRORS:", json.dumps(errors, indent=2), flush=True)


def boot_ci(vals, n_boot=N_BOOT, seed=BOOT_SEED):
    """Percentile bootstrap 95% CI of the mean. Returns (lo, hi)."""
    arr = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def describe(vals, label):
    arr = np.asarray(vals, dtype=float)
    lo, hi = boot_ci(arr)
    return {
        "label": label,
        "n_seeds": int(arr.size),
        "mean": float(arr.mean()),
        "sd_across_seeds": float(arr.std(ddof=1)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "bootstrap95_ci_of_mean": [lo, hi],
        "per_seed": {str(s): float(v) for s, v in zip(SEEDS, arr)},
    }


# --- validity controls ---
controls = {}
for ctrl in ("untreated_fus_on", "untreated_tmax200"):
    diffs = {}
    for day in DAYS:
        d = [abs(raw[ctrl][day][s] - raw["untreated"][day][s]) for s in SEEDS
             if s in raw[ctrl][day] and s in raw["untreated"][day]]
        diffs[f"day{day}_max_abs_diff_vs_untreated"] = float(max(d)) if d else None
    diffs["identical_to_untreated"] = all(
        v == 0.0 for v in diffs.values() if isinstance(v, float))
    controls[ctrl] = diffs

# --- untreated trajectory at the comparison days, stored explicitly ---
untreated_at_days = {
    f"day{day}": describe([raw["untreated"][day][s] for s in SEEDS],
                          f"untreated gm2_brain (nmol/g) at day {day}")
    for day in DAYS
}

# --- the two comparisons, for each treated arm and each day ---
results = {}
for arm in ("aav_only", "srt_only"):
    results[arm] = {}
    for day in DAYS:
        treated = np.array([raw[arm][day][s] for s in SEEDS], dtype=float)
        untr = np.array([raw["untreated"][day][s] for s in SEEDS], dtype=float)

        # (a) percent change vs the arm's OWN initial value (reproduces published numbers).
        #     task9 computed this from the MEAN treated value, not per-seed, so reproduce
        #     that exactly AND give the per-seed version alongside.
        pct_vs_own_per_seed = 100.0 * (baseline_gm2_brain - treated) / baseline_gm2_brain
        pct_vs_own_from_mean = 100.0 * (baseline_gm2_brain - treated.mean()) / baseline_gm2_brain

        # (b) percent change vs the UNTREATED arm, same day, same seed (paired).
        pct_vs_matched_per_seed = 100.0 * (untr - treated) / untr

        results[arm][f"day{day}"] = {
            "gm2_brain_treated": describe(treated, f"{arm} gm2_brain at day {day}"),
            "gm2_brain_untreated_matched": describe(untr, f"untreated gm2_brain at day {day}"),
            "a_pct_change_vs_own_initial_value": {
                "definition": "100 * (890 - treated) / 890",
                "reproduces_published_value_computed_from_arm_mean": pct_vs_own_from_mean,
                "per_seed_stats": describe(pct_vs_own_per_seed, "pct change vs own initial value"),
                "sd_as_task9_computed_it_sd_of_values_over_890_pct": float(
                    treated.std(ddof=1) / baseline_gm2_brain * 100.0),
            },
            "b_pct_change_vs_matched_untreated_same_day_same_seed": {
                "definition": "100 * (untreated[seed,day] - treated[seed,day]) / untreated[seed,day]",
                "paired": True,
                "per_seed_stats": describe(pct_vs_matched_per_seed,
                                           "paired pct change vs matched untreated control"),
            },
        }

PUBLISHED = {  # from task_outputs/external_benchmark.json, for the match check
    "aav_only": {"day84": 99.61894568394915, "day168": 99.50601974127412},
    "srt_only": {"day84": 50.73560876829471, "day168": 63.39169585775233},
}
reproduction_check = {}
for arm, days in PUBLISHED.items():
    reproduction_check[arm] = {}
    for dk, pub in days.items():
        got = results[arm][dk]["a_pct_change_vs_own_initial_value"][
            "reproduces_published_value_computed_from_arm_mean"]
        reproduction_check[arm][dk] = {
            "published": pub, "recomputed": got, "abs_diff": abs(got - pub),
            "matches_to_2dp": round(got, 2) == round(pub, 2),
        }

out = {
    "task": "external_benchmark_matched_control",
    "purpose": (
        "Recompute the AAV-only and SRT-only brain-substrate reductions against a "
        "paired, concurrently simulated untreated arm (same seeds, same days), because "
        "the published percentages are relative to each arm's own initial value (890) "
        "while the untreated arm in this model declines on its own."
    ),
    "reproduction_check_of_published_numbers": reproduction_check,
    "untreated_arm_gm2_brain_at_comparison_days": untreated_at_days,
    "comparisons": results,
    "validity_controls": controls,
    "errors": errors,
    "run_meta": {
        "model_file": os.path.relpath(MODEL_FILE, os.path.dirname(HERE)),
        "model_sha256": MODEL_SHA256,
        "model_sha256_16": MODEL_SHA256[:16],
        "config_source": "task9_external_benchmark.py (reused unchanged)",
        "disease": DISEASE,
        "n_seeds": N_SEEDS, "seeds": SEEDS, "dt": DT, "days": DAYS,
        "arms": ARMS,
        "day_sampling": "nearest grid index to the target day; no interpolation",
        "actual_sampled_time_days": {a: {str(k): v for k, v in d.items()}
                                     for a, d in sampled_days.items()},
        "baseline_gm2_brain": baseline_gm2_brain,
        "bootstrap": {"n_resamples": N_BOOT, "method": "percentile", "rng_seed": BOOT_SEED},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
    },
}

def _jsonable(o):
    """numpy scalars/arrays -> plain python. Raises on anything genuinely
    unserializable rather than coercing it to a string."""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


outpath = os.path.join(HERE, "external_benchmark_matched.json")
with open(outpath, "w") as f:
    json.dump(out, f, indent=2, default=_jsonable)

# --- console summary ---
print("\n=== reproduction of published (a) numbers ===")
for arm, days in reproduction_check.items():
    for dk, r in days.items():
        print(f"{arm:10s} {dk:6s} published={r['published']:.5f}  recomputed={r['recomputed']:.5f}  "
              f"match_2dp={r['matches_to_2dp']}")

print("\n=== untreated arm gm2_brain (nmol/g) ===")
for dk, s in untreated_at_days.items():
    print(f"{dk:6s} mean={s['mean']:8.2f}  sd={s['sd_across_seeds']:7.2f}  "
          f"[min {s['min']:.2f}, max {s['max']:.2f}]")

print("\n=== (a) vs own initial value 890  |  (b) vs matched untreated (paired) ===")
hdr = f"{'arm':10s} {'day':5s} {'(a) %':>9s} {'(b) mean%':>10s} {'(b) SD':>8s} {'(b) 95% CI':>22s}"
print(hdr)
for arm in ("aav_only", "srt_only"):
    for day in DAYS:
        r = results[arm][f"day{day}"]
        a = r["a_pct_change_vs_own_initial_value"][
            "reproduces_published_value_computed_from_arm_mean"]
        b = r["b_pct_change_vs_matched_untreated_same_day_same_seed"]["per_seed_stats"]
        lo, hi = b["bootstrap95_ci_of_mean"]
        print(f"{arm:10s} {day:<5d} {a:9.2f} {b['mean']:10.2f} {b['sd_across_seeds']:8.2f} "
              f"{f'[{lo:.2f}, {hi:.2f}]':>22s}")

print("\n=== validity controls ===")
print(json.dumps(controls, indent=2))
print(f"\nmodel_sha256 = {MODEL_SHA256}")
print(f"Wrote {outpath}")
