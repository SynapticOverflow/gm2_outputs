"""
task10_seed_sensitivity.py

Seed-sensitivity check of the 14-parameter GM2 Sobol analysis.

The production full-14 Sobol (task1_full14.py -> sobol_driver_v2.run_scenario)
evaluates every row with model seed 42 (common random numbers: one shared
noise path). This driver re-runs the SAME configuration with other noise
seeds so the only thing that changes between runs is the model-evaluation
noise:

  A  fixed seed, one seed for all rows: 42 (reproducibility control), 7, 101,
     2024, 31337, 90210
  B  per-row seed: row i (0-based index into the scenario's 7680 rows) uses
     seed 1000+i (independent noise per evaluation)
  C  seed-averaged: each row = mean of run_single over seeds 0..9 (the same
     10 seeds for every row, i.e. CRN-averaged). The 10 per-seed Y columns
     are saved too.

Held fixed exactly as production / task7_percentile_cis.py (the run whose
seed-42 S_T + percentile CIs are in task_outputs/percentile_cis.json):
  N=256, D=14, calc_second_order=True (7680 rows/scenario), dt=0.2,
  tmax=365, sobol_driver_v2.PARAM_NAMES / make_bounds(), three scenarios,
  Sobol X from SALib sobol.sample(seed=20260816) -- the same X for every run,
  SALib analyze with num_resamples=1000, bootstrap seed 8162026, 95%
  percentile CI from ST_conf_all / S1_conf_all.

NOTE: the original task1_full14.py sample was drawn with SALib's default
seed=None (scrambled Sobol), and its Y was never saved, so that exact X is
unrecoverable. The task7 rerun (fixed sample seed 20260816) is the only
reproducible seed-42 reference, so it is the X used here.

gm2_model_v2.py is imported read-only; its sha256 is checked against the
expected prefix before anything runs and recorded in every output.

Usage:
  python3 task10_seed_sensitivity.py RUNSET [N] [OUTDIR]
    RUNSET in {A, B, C, all}; N default 256; OUTDIR default ./N{N}
"""
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version

import numpy as np
from SALib.analyze.sobol import analyze as sobol_analyze
from SALib.sample.sobol import sample as sobol_sample

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "gm2_analysis"))
DRIVER_DIR = os.path.join(MODEL_DIR, "drivers")
sys.path.insert(0, DRIVER_DIR)
sys.path.insert(0, MODEL_DIR)

EXPECTED_SHA_PREFIX = "6693427fa909a0a9"
MODEL_FILE = os.path.join(MODEL_DIR, "gm2_model_v2.py")
with open(MODEL_FILE, "rb") as f:
    MODEL_SHA256 = hashlib.sha256(f.read()).hexdigest()
if not MODEL_SHA256.startswith(EXPECTED_SHA_PREFIX):
    sys.exit(f"ABORT: gm2_model_v2.py sha256 {MODEL_SHA256} != expected prefix {EXPECTED_SHA_PREFIX}")

import sobol_driver_v2 as sdv2  # noqa: E402
from gm2_model_v2 import run_single  # noqa: E402
from parallel_eval import evaluate_parallel, n_workers_from_env  # noqa: E402

DT = 0.2
TMAX = 365.0
SCENARIOS = ["as_documented", "equal_width", "tier1_promotion"]
SOBOL_SAMPLE_SEED = 20260816   # = task7_percentile_cis.RERUN_SAMPLE_SEED
BOOT_B = 1000                  # = task7_percentile_cis.B
BOOT_SEED = 8162026            # = task7_percentile_cis.BOOT_SEED
CONF_LEVEL = 0.95
FIXED_SEEDS = [42, 7, 101, 2024, 31337, 90210]
PER_ROW_SEED_OFFSET = 1000
AVG_SEEDS = list(range(10))


def _eval(args):
    row, seed = args
    overrides = dict(zip(sdv2.PARAM_NAMES, row))
    dose_total = overrides.pop("t4_dose_total")
    return run_single(overrides, dt=DT, tmax_days=TMAX, seed=int(seed), t4_dose_total=dose_total)


def problem_for(scenario):
    return {"num_vars": len(sdv2.PARAM_NAMES), "names": sdv2.PARAM_NAMES,
            "bounds": sdv2.make_bounds(scenario)}


def y_summary(Y):
    return {
        "mean": float(np.mean(Y)), "sd": float(np.std(Y)), "var": float(np.var(Y)),
        "min": float(np.min(Y)), "max": float(np.max(Y)),
        "frac_at_floor_eq_0": float(np.mean(Y <= 0.0)),
        "frac_below_1e-6": float(np.mean(Y < 1e-6)),
        "n_nonfinite": int(np.sum(~np.isfinite(Y))),
    }


def analyze(problem, Y):
    Si = sobol_analyze(problem, Y, calc_second_order=True, num_resamples=BOOT_B,
                       conf_level=CONF_LEVEL, print_to_console=False,
                       keep_resamples=True, seed=BOOT_SEED)
    n = problem["num_vars"]
    names = problem["names"]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if not np.isnan(Si["S2"][i, j]):
                pairs.append({"pair": [names[i], names[j]], "S2": float(Si["S2"][i, j]),
                              "S2_conf": float(Si["S2_conf"][i, j])})
    pairs.sort(key=lambda p: -abs(p["S2"]))
    ST_all = np.asarray(Si["ST_conf_all"])
    S1_all = np.asarray(Si["S1_conf_all"])
    return {
        "ST": Si["ST"].tolist(),
        "ST_ci_lo_2.5": np.percentile(ST_all, 2.5, axis=0).tolist(),
        "ST_ci_hi_97.5": np.percentile(ST_all, 97.5, axis=0).tolist(),
        "ST_conf_symmetric": Si["ST_conf"].tolist(),
        "S1": Si["S1"].tolist(),
        "S1_ci_lo_2.5": np.percentile(S1_all, 2.5, axis=0).tolist(),
        "S1_ci_hi_97.5": np.percentile(S1_all, 97.5, axis=0).tolist(),
        "S1_conf_symmetric": Si["S1_conf"].tolist(),
        "S2_pairs": pairs,
        "sum_ST": float(np.sum(Si["ST"])),
        "sum_S1": float(np.sum(Si["S1"])),
    }


def run_specs(runset):
    specs = []
    if runset in ("A", "all"):
        for s in FIXED_SEEDS:
            specs.append({"run_id": f"A_seed{s}", "scheme": "fixed", "seeds": [s]})
    if runset in ("B", "all"):
        specs.append({"run_id": f"B_perrow{PER_ROW_SEED_OFFSET}", "scheme": "per_row",
                      "seeds": f"row i -> {PER_ROW_SEED_OFFSET}+i"})
    if runset in ("C", "all"):
        specs.append({"run_id": f"C_mean{len(AVG_SEEDS)}_seeds{AVG_SEEDS[0]}-{AVG_SEEDS[-1]}",
                      "scheme": "seed_averaged_crn", "seeds": AVG_SEEDS})
    return specs


def evaluate(spec, X, label):
    n = X.shape[0]
    if spec["scheme"] == "fixed":
        return evaluate_parallel(_eval, [(r, spec["seeds"][0]) for r in X], label=label), None
    if spec["scheme"] == "per_row":
        return evaluate_parallel(_eval, [(r, PER_ROW_SEED_OFFSET + i) for i, r in enumerate(X)],
                                 label=label), None
    if spec["scheme"] == "seed_averaged_crn":
        k = len(AVG_SEEDS)
        flat = evaluate_parallel(_eval, [(r, s) for r in X for s in AVG_SEEDS], label=label)
        Ymat = flat.reshape(n, k)
        return Ymat.mean(axis=1), Ymat
    raise ValueError(spec["scheme"])


def main():
    runset = sys.argv[1] if len(sys.argv) > 1 else "all"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    outdir = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else os.path.join(HERE, f"N{N}")
    os.makedirs(outdir, exist_ok=True)
    n_workers = n_workers_from_env()
    print(f"model sha256 = {MODEL_SHA256}  runset={runset} N={N} outdir={outdir} "
          f"workers={n_workers} SALib={pkg_version('SALib')}", flush=True)

    t_all = time.time()
    for scenario in SCENARIOS:
        problem = problem_for(scenario)
        X = sobol_sample(problem, N=N, calc_second_order=True, seed=SOBOL_SAMPLE_SEED)
        X_sha = hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()
        xpath = os.path.join(outdir, f"X_{scenario}.npy")
        if os.path.exists(xpath):
            if not np.array_equal(np.load(xpath), X):
                sys.exit(f"ABORT: regenerated X differs from saved {xpath}")
        else:
            np.save(xpath, X)

        for spec in run_specs(runset):
            run_id = spec["run_id"]
            label = f"{run_id} {scenario} "
            jpath = os.path.join(outdir, f"{run_id}_{scenario}.json")
            if os.path.exists(jpath):
                sys.exit(f"ABORT: refusing to overwrite existing {jpath}")
            print(f"\n=== {run_id} / {scenario}: {X.shape[0]} rows ===", flush=True)
            t0 = time.time()
            Y, Ymat = evaluate(spec, X, label)
            elapsed = time.time() - t0
            np.save(os.path.join(outdir, f"Y_{run_id}_{scenario}.npy"), Y)
            if Ymat is not None:
                np.save(os.path.join(outdir, f"Y_{run_id}_{scenario}_perseed.npy"), Ymat)

            ys = y_summary(Y)
            rec = {
                "run_id": run_id, "scenario": scenario, "names": sdv2.PARAM_NAMES,
                "n_evals_rows": int(X.shape[0]),
                "Y_summary": ys,
                "run_meta": {
                    "seed_scheme": spec["scheme"], "seeds": spec["seeds"],
                    "N": N, "D": len(sdv2.PARAM_NAMES), "dt": DT, "tmax_days": TMAX,
                    "calc_second_order": True,
                    "sobol_sample_seed": SOBOL_SAMPLE_SEED, "X_sha256": X_sha,
                    "bootstrap_B": BOOT_B, "bootstrap_seed": BOOT_SEED,
                    "conf_level": CONF_LEVEL, "ci_method": "percentile (2.5, 97.5) of SALib *_conf_all",
                    "model_file": "gm2_analysis/gm2_model_v2.py", "model_sha256": MODEL_SHA256,
                    "salib_version": pkg_version("SALib"), "numpy_version": np.__version__,
                    "python_version": sys.version.split()[0],
                    "n_workers": n_workers, "elapsed_s": elapsed,
                    "n_model_evals": int(X.shape[0] * (len(AVG_SEEDS) if Ymat is not None else 1)),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "host": os.uname().nodename,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                },
            }
            if ys["n_nonfinite"] > 0:
                bad = np.where(~np.isfinite(Y))[0]
                rec["status"] = "FAILED_NONFINITE"
                rec["nonfinite_row_indices"] = bad[:1000].tolist()
                print(f"!!! {run_id}/{scenario}: {len(bad)} non-finite Y values "
                      f"(first rows {bad[:10].tolist()}); Sobol analysis NOT run", flush=True)
            else:
                rec["status"] = "ok"
                rec.update(analyze(problem, Y))
            with open(jpath, "w") as f:
                json.dump(rec, f, indent=2)
            if rec["status"] == "ok":
                top = sorted(zip(sdv2.PARAM_NAMES, rec["ST"]), key=lambda t: -t[1])[:5]
                print(f"  {run_id}/{scenario}: {elapsed:.0f}s  Var(Y)={ys['var']:.4f} "
                      f"sumST={rec['sum_ST']:.3f}  top5: " +
                      ", ".join(f"{n}={s:.3f}" for n, s in top), flush=True)
    print(f"\nAll done in {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
