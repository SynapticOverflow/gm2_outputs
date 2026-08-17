"""
recommend_timing.py

Uses the trained surrogate model (train_timing_model.py) to answer the
actual clinical question: for a given target brain region, what LIFU
protocol (frequency, pressure, duty cycle, and -- the new part -- WHEN and
how many sessions to administer) maximises predicted day-365 GM2 percent
reduction, and how good is that prediction expected to be?

Because the surrogate model is ~instant to evaluate (no 8760-hour SDE
integration), the search here is a large random search over the full
7-D input space per region (200,000 candidates), which is more than
sufficient to closely approximate the true optimum of a smooth-ish
gradient-boosted-tree response surface, and is far simpler/more robust
than gradient-based optimisation of a non-differentiable tree ensemble.

Outputs:
    lifu_timing_recommendations.csv  -- one row per named brain region:
        recommended f, P, DC, t_first_hr, n_sessions, spacing_hr,
        resulting eta, and the model's predicted GM2 reduction (%).
    Also prints a plain-text protocol summary per region.
"""

import os

import joblib
import numpy as np

from lifu_acoustic import REGION_DEPTHS_MM, compute_eta
from generate_timing_data import BOUNDS

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(OUT_DIR, "lifu_timing_model.joblib")
OUT_CSV = os.path.join(OUT_DIR, "lifu_timing_recommendations.csv")

N_CANDIDATES = 200_000


def sample_candidates(depth_mm, n_candidates, seed):
    """Random search candidates over (f, P, DC, t_first, n_sessions,
    spacing) at a fixed depth, matching generate_timing_data.py's bounds."""
    rng = np.random.default_rng(seed)
    f_mhz = rng.uniform(*BOUNDS["f_mhz"], size=n_candidates)
    P_mpa = rng.uniform(*BOUNDS["P_mpa"], size=n_candidates)
    DC = rng.uniform(*BOUNDS["DC"], size=n_candidates)
    t_first_hr = rng.uniform(*BOUNDS["t_first_hr"], size=n_candidates)
    spacing_hr = rng.uniform(*BOUNDS["spacing_hr"], size=n_candidates)
    n_sessions = rng.integers(1, 6, size=n_candidates)  # 1..5 inclusive
    depth = np.full(n_candidates, depth_mm)
    eta = compute_eta(f_mhz, P_mpa, DC, depth)
    return dict(depth_mm=depth, f_mhz=f_mhz, P_mpa=P_mpa, DC=DC, eta=eta,
                t_first_hr=t_first_hr, n_sessions=n_sessions, spacing_hr=spacing_hr)


def recommend_for_region(model, features, depth_mm, seed):
    cand = sample_candidates(depth_mm, N_CANDIDATES, seed)
    X = np.column_stack([cand[f] for f in features])
    pred = model.predict(X)
    best_idx = int(np.argmax(pred))
    best = {f: cand[f][best_idx] for f in features}
    best["predicted_reduction_pct"] = float(pred[best_idx])
    # also report the top-3 for a sense of the plateau/robustness around the optimum
    top3_idx = np.argsort(pred)[-3:][::-1]
    top3 = [
        dict(t_first_hr=float(cand["t_first_hr"][i]),
             n_sessions=int(cand["n_sessions"][i]),
             spacing_hr=float(cand["spacing_hr"][i]),
             predicted_reduction_pct=float(pred[i]))
        for i in top3_idx
    ]
    return best, top3


def main():
    bundle = joblib.load(MODEL_PATH)
    model, features = bundle["model"], bundle["features"]

    rows = []
    print("=== Recommended LIFU protocol per brain region ===\n")
    for region, depth_mm in REGION_DEPTHS_MM.items():
        best, top3 = recommend_for_region(model, features, depth_mm, seed=hash(region) % (2**32))
        rows.append(dict(region=region, **best))

        print(f"{region} (depth {depth_mm:.0f} mm)")
        print(f"  Protocol : f={best['f_mhz']:.2f} MHz  P={best['P_mpa']:.2f} MPa  "
              f"DC={best['DC']*100:.2f}%  (eta={best['eta']:.3f})")
        print(f"  Timing   : first session at t={best['t_first_hr']:.1f} hr post-dose, "
              f"{int(round(best['n_sessions']))} session(s), "
              f"{best['spacing_hr']:.1f} hr apart")
        print(f"  Predicted GM2 reduction (day 365): {best['predicted_reduction_pct']:.1f}%")
        print(f"  Top-3 timing variants (protocol fixed at above f/P/DC):")
        for t in top3:
            print(f"    t_first={t['t_first_hr']:5.1f}hr  n={t['n_sessions']}  "
                  f"spacing={t['spacing_hr']:5.1f}hr  -> {t['predicted_reduction_pct']:.1f}%")
        print()

    with open(OUT_CSV, "w") as fh:
        cols = ["region", "depth_mm", "f_mhz", "P_mpa", "DC", "eta",
                "t_first_hr", "n_sessions", "spacing_hr", "predicted_reduction_pct"]
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
