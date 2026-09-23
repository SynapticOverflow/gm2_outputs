"""
run_acoustic_fus_analysis.py

Step 2 of the 2026-09-15 acoustic-FUS-mapping task: rerun the full 8-arm
factorial design (n=200 matched seeds/arm) and the temporal Shapley
decomposition at days 45/90/180/365, under gm2_model_v2_acoustic_fus.py's
two derived fus_entry_gain_scale configurations
(see fus_gain_mapping_derivation.md / fus_gain_mapping.json, Step 1):
    k=1.0 (conservative) -> fus_entry_gain_scale = 1.500000
    k=2.0 (permissive)   -> fus_entry_gain_scale = 1.665961

Identical protocol to factorial_arms_shapley.py / task5_paired_contrasts.py:
same 8 arms (SRT, AAV, FUS binary factorial), same N_SEEDS=200, same seeds
(range(200), reused across arms via common random numbers exactly as the
original does), same DEFAULT_FUS_EVENTS, same tay-sachs disease params,
same dt=0.2, same day cutoffs, same bootstrap protocol (B=1000, seed=42).
Only the model module (gm2_model_v2_acoustic_fus instead of gm2_model_v2)
and the fus_entry_gain_scale value differ between this and the paper's runs,
and between the two configs run here.

Reuses (imports, does not modify) exact_shapley from factorial_arms_shapley.py
and paired_contrast / shapley_with_ci / CONTRASTS from task5_paired_contrasts.py.

Does NOT modify gm2_model_v2.py. Verifies its sha256 before and after.

Intended to run under SLURM (see submit_acoustic_fus.slurm), not interactively
on a login node.
"""
import hashlib
import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from factorial_arms_shapley import exact_shapley  # noqa: E402  (reused, unmodified)
from task5_paired_contrasts import paired_contrast, shapley_with_ci, CONTRASTS  # noqa: E402

DISEASE = 'tay-sachs'
DT = 0.2
N_SEEDS = 200
DAYS = [45.0, 90.0, 180.0, 365.0]

# From fus_gain_mapping_derivation.md Step 1 (§7):
GAIN = {
    "k1": 1.500000,   # conservative, k=1.0 -- reproduces the existing calibration exactly (control)
    "k2": 1.665961,   # permissive,   k=2.0
}

GM2_MODEL_V2_SHA256 = "6693427fa909a0a95093cf6797890c6c4004912fd2a54b7c9e9966e77fa8a4a6"


def check_untouched():
    h = hashlib.sha256(open(os.path.join(HERE, "gm2_model_v2.py"), "rb").read()).hexdigest()
    if h != GM2_MODEL_V2_SHA256:
        raise RuntimeError(f"gm2_model_v2.py CHANGED! expected {GM2_MODEL_V2_SHA256}, got {h}")
    return h


def run_arm(model, srt, aav, fus, seed, tmax):
    base = model.make_base_params(DISEASE)
    x0 = model.make_x0(base)
    df = model.simulate_milstein(
        x0, tmax, DT, base,
        dose_mg_per_day=3.0 if srt else 0.0,
        t4_admin_day=30.0,
        t4_dose_total=1.0 if aav else 0.0,
        t4_pulse_width_days=1.0,
        fus_events=model.DEFAULT_FUS_EVENTS if fus else None,
        seed=seed,
    )
    return float(df["gm2_brain"].iloc[-1])


def run_factorial(model, tmax):
    arms = list(itertools.product([0, 1], repeat=3))
    t0 = time.time()
    arm_table = {}
    for srt, aav, fus in arms:
        vals = [run_arm(model, bool(srt), bool(aav), bool(fus), seed, tmax) for seed in range(N_SEEDS)]
        mean = sum(vals) / len(vals)
        var = sum((y - mean) ** 2 for y in vals) / (len(vals) - 1)
        arm_table[(srt, aav, fus)] = {'mean': mean, 'std': var ** 0.5, 'n_seeds': N_SEEDS, 'raw': vals}
    elapsed = time.time() - t0

    no_treatment = arm_table[(0, 0, 0)]['mean']
    v = {}
    for srt, aav, fus in arms:
        S = frozenset(x for x, on in zip(("SRT", "AAV", "FUS"), (srt, aav, fus)) if on)
        v[S] = no_treatment - arm_table[(srt, aav, fus)]['mean']
    shapley = exact_shapley(v)
    total_benefit = v[frozenset({"SRT", "AAV", "FUS"})]

    out = {
        'disease': DISEASE, 'tmax_days': tmax, 'dt': DT, 'n_seeds': N_SEEDS,
        'arms': [
            {'SRT': srt, 'AAV': aav, 'FUS': fus, 'mean_burden': arm_table[(srt, aav, fus)]['mean'],
             'std_burden': arm_table[(srt, aav, fus)]['std'],
             'raw_burden': arm_table[(srt, aav, fus)]['raw']}
            for srt, aav, fus in arms
        ],
        'no_treatment_burden': no_treatment,
        'characteristic_function_v': {','.join(sorted(s)) if s else 'none': val for s, val in v.items()},
        'shapley_benefit': shapley,
        'total_benefit': total_benefit,
        'elapsed_s': elapsed,
    }
    return out


def main():
    h_before = check_untouched()
    print(f"gm2_model_v2.py sha256 (pre-run):  {h_before}")

    import gm2_model_v2_acoustic_fus as model  # noqa: E402 (import after path setup)

    for cfg, gain in GAIN.items():
        print(f"\n{'=' * 78}\nCONFIG {cfg}: fus_entry_gain_scale = {gain:.6f}\n{'=' * 78}")
        os.environ["GM2_ACOUSTIC_FUS_ENTRY_GAIN_SCALE"] = repr(gain)
        # sanity: confirm the env var actually takes effect before burning compute on it
        check_gain = model.make_base_params(DISEASE)['fus_entry_gain_scale']
        assert check_gain == gain, f"env override failed: got {check_gain}, expected {gain}"

        day_files = {}
        for tmax in DAYS:
            t_start = time.time()
            res = run_factorial(model, tmax)
            suffix = "" if tmax == 365.0 else f"_day{int(tmax)}"
            fname = f"data/factorial_arms_shapley_acoustic_{cfg}{suffix}.json"
            with open(os.path.join(HERE, fname), "w") as f:
                json.dump(res, f, indent=2)
            day_files[tmax] = fname
            print(f"  day {tmax:5.0f}: no_tx={res['no_treatment_burden']:10.4f}  "
                  f"total_benefit={res['total_benefit']:10.4f}  "
                  f"elapsed={res['elapsed_s']:.1f}s  -> {fname}", flush=True)

        # --- paired contrasts + shapley-with-CI, identical protocol to task5 ---
        rng = np.random.default_rng(42)
        contrasts_out = {"run_meta": {"B": 1000, "seed": 42,
                                       "method": "percentile bootstrap on 200 matched seed-pairs",
                                       "config": cfg, "fus_entry_gain_scale": gain}}
        shapley_out = {"run_meta": {"B": 1000, "seed": 42,
                                     "method": ("joint percentile bootstrap on 200 matched seed-pairs, "
                                                "all 8 arms resampled together"),
                                     "config": cfg, "fus_entry_gain_scale": gain}}
        for tmax, fname in day_files.items():
            d = json.load(open(os.path.join(HERE, fname)))
            arms = {(a["SRT"], a["AAV"], a["FUS"]): np.asarray(a["raw_burden"], dtype=float) for a in d["arms"]}
            day_out = {}
            for name, key_a, key_b in CONTRASTS:
                day_out[name] = paired_contrast(arms, key_a, key_b, rng)
            contrasts_out[f"day{int(tmax)}"] = day_out
            shapley_out[f"day{int(tmax)}"] = shapley_with_ci(arms, (0, 0, 0), rng)

        pc_fname = f"data/paired_contrasts_acoustic_{cfg}.json"
        sh_fname = f"data/shapley_with_ci_acoustic_{cfg}.json"
        with open(os.path.join(HERE, pc_fname), "w") as f:
            json.dump(contrasts_out, f, indent=2)
        with open(os.path.join(HERE, sh_fname), "w") as f:
            json.dump(shapley_out, f, indent=2)
        print(f"  Written {pc_fname}, {sh_fname}")

    h_after = check_untouched()
    print(f"\ngm2_model_v2.py sha256 (post-run): {h_after}")
    if h_after != h_before:
        raise RuntimeError("gm2_model_v2.py hash changed during the run!")
    print("CONFIRMED: gm2_model_v2.py unmodified before and after.")
    print("\nDone.")


if __name__ == "__main__":
    main()
