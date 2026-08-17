"""
analyze_results.py

Post-processes the .npz files produced by run_sweep.py into summary tables:

    lifu_results_summary.csv
        One row per (LIFU combination, region): eta, mean/std GM2 at day
        365, and mean/std percent reduction vs. natural history.

    lifu_results_full.pkl
        Python dict with the full per-realisation terminal GM2 values
        (gm2_day365) for every LIFU combination, every control arm, and
        every region, plus the summary table and sensitivity results.
        NOTE: this does *not* include the full time-series trajectories
        (65k+ x 17,521 timesteps would be tens of GB) -- only the day-365
        terminal values needed for the reduction/sensitivity analysis. Load
        the individual results_*.npz files directly if full trajectories
        are needed (e.g. for the poster's Fig. 1 time-course plot).

Also prints:
    - best LIFU combination per region (highest mean % reduction)
    - single best combination overall (highest mean % reduction across
      all 8 regions)
    - a simple range-based main-effect sensitivity for f, P, DC (this is a
      3x3x3 full-factorial grid, not a Sobol sample, so we report main-
      effect ranges rather than variance-based Sobol indices)
    - top 5 combinations ranked by mean reduction across all regions
"""

import glob
import os
import pickle

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SUMMARY_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lifu_results_summary.csv")
FULL_PKL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lifu_results_full.pkl")


def _load_npz(path):
    data = np.load(path, allow_pickle=True)
    params = data["params"].item()
    return data["gm2_day365"], data["eta_per_region"], params


def load_all_results():
    """Load every combo/control .npz. Returns (lifu_entries, control_entries)."""
    lifu_entries = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "results_*MHz_*MPa_*pct.npz"))):
        gm2_day365, eta_per_region, params = _load_npz(path)
        lifu_entries.append(dict(path=path, gm2_day365=gm2_day365,
                                  eta_per_region=eta_per_region, **params))

    control_entries = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "results_control_*.npz"))):
        gm2_day365, _eta, params = _load_npz(path)
        control_entries[params["arm"]] = gm2_day365

    return lifu_entries, control_entries


def build_summary_table(lifu_entries, natural_gm2_day365, region_names):
    """Return list of row-dicts, one per (combo, region)."""
    natural_mean = natural_gm2_day365.mean(axis=0)  # (n_regions,)
    rows = []
    for entry in lifu_entries:
        gm2 = entry["gm2_day365"]  # (300, n_regions)
        for r_idx, region in enumerate(region_names):
            region_vals = gm2[:, r_idx]
            mean_gm2 = region_vals.mean()
            std_gm2 = region_vals.std()
            baseline = natural_mean[r_idx]
            reduction_pct = (baseline - region_vals) / baseline * 100.0
            rows.append(dict(
                f_MHz=entry["f_mhz"],
                P_MPa=entry["P_mpa"],
                DC_pct=entry["DC"] * 100.0,
                region=region,
                eta=float(entry["eta_per_region"][r_idx]),
                mean_gm2_day365=mean_gm2,
                std_gm2_day365=std_gm2,
                mean_reduction_pct=reduction_pct.mean(),
                std_reduction_pct=reduction_pct.std(),
            ))
    return rows


def write_summary_csv(rows, path):
    import csv
    fieldnames = ["f_MHz", "P_MPa", "DC_pct", "region", "eta",
                  "mean_gm2_day365", "std_gm2_day365",
                  "mean_reduction_pct", "std_reduction_pct"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def best_combo_per_region(rows, region_names):
    best = {}
    for region in region_names:
        region_rows = [r for r in rows if r["region"] == region]
        best[region] = max(region_rows, key=lambda r: r["mean_reduction_pct"])
    return best


def overall_best_combo(rows):
    """Group by (f, P, DC), average mean_reduction_pct across regions,
    return the top combination."""
    combos = {}
    for r in rows:
        key = (r["f_MHz"], r["P_MPa"], r["DC_pct"])
        combos.setdefault(key, []).append(r["mean_reduction_pct"])
    combo_means = {k: float(np.mean(v)) for k, v in combos.items()}
    best_key = max(combo_means, key=combo_means.get)
    return best_key, combo_means[best_key], combo_means


def top_n_combos(combo_means, n=5):
    return sorted(combo_means.items(), key=lambda kv: kv[1], reverse=True)[:n]


def main_effect_sensitivity(combo_means):
    """Simple range-based main-effect sensitivity for f, P, DC over the
    3x3x3 full-factorial grid: group combo means by each parameter's level,
    report the spread (max - min) of the per-level averages."""
    by_f, by_P, by_DC = {}, {}, {}
    for (f_MHz, P_MPa, DC_pct), mean_red in combo_means.items():
        by_f.setdefault(f_MHz, []).append(mean_red)
        by_P.setdefault(P_MPa, []).append(mean_red)
        by_DC.setdefault(DC_pct, []).append(mean_red)

    def spread(d):
        level_means = {k: float(np.mean(v)) for k, v in d.items()}
        rng = max(level_means.values()) - min(level_means.values())
        return level_means, rng

    f_levels, f_range = spread(by_f)
    P_levels, P_range = spread(by_P)
    DC_levels, DC_range = spread(by_DC)

    return dict(
        frequency=dict(level_means=f_levels, range_pct=f_range),
        pressure=dict(level_means=P_levels, range_pct=P_range),
        duty_cycle=dict(level_means=DC_levels, range_pct=DC_range),
    )


def main():
    lifu_entries, control_entries = load_all_results()
    if not lifu_entries or "natural" not in control_entries:
        raise RuntimeError(
            f"No results found in {RESULTS_DIR}. Run run_sweep.py first."
        )

    region_names = lifu_entries[0]["region_names"]
    natural_gm2_day365 = control_entries["natural"]

    rows = build_summary_table(lifu_entries, natural_gm2_day365, region_names)
    write_summary_csv(rows, SUMMARY_CSV)
    print(f"Wrote {SUMMARY_CSV} ({len(rows)} rows)")

    best_per_region = best_combo_per_region(rows, region_names)
    best_key, best_mean, combo_means = overall_best_combo(rows)
    sensitivity = main_effect_sensitivity(combo_means)
    top5 = top_n_combos(combo_means, n=5)

    # Also fold in control-arm reduction (mono/bi/tri_no_lifu vs natural)
    # for reference in the full pickle.
    control_reduction = {}
    natural_mean = natural_gm2_day365.mean(axis=0)
    for arm, gm2 in control_entries.items():
        if arm == "natural":
            continue
        red = (natural_mean - gm2.mean(axis=0)) / natural_mean * 100.0
        control_reduction[arm] = {region_names[i]: float(red[i]) for i in range(len(region_names))}

    full_data = dict(
        summary_rows=rows,
        best_combo_per_region={k: v for k, v in best_per_region.items()},
        overall_best_combo=dict(f_MHz=best_key[0], P_MPa=best_key[1],
                                 DC_pct=best_key[2], mean_reduction_pct=best_mean),
        combo_means=combo_means,
        sensitivity=sensitivity,
        top5_combos=top5,
        control_gm2_day365={arm: gm2 for arm, gm2 in control_entries.items()},
        control_reduction_pct_vs_natural=control_reduction,
        region_names=region_names,
    )
    with open(FULL_PKL, "wb") as fh:
        pickle.dump(full_data, fh)
    print(f"Wrote {FULL_PKL}")

    # ---------------- console report ----------------
    print("\n=== Best LIFU combination per region ===")
    for region, row in best_per_region.items():
        print(f"  {region:16s}  f={row['f_MHz']:.2f}MHz  P={row['P_MPa']:.2f}MPa  "
              f"DC={row['DC_pct']:.1f}%  reduction={row['mean_reduction_pct']:.1f}% "
              f"(eta={row['eta']:.3f})")

    print(f"\n=== Overall best combination (mean across all {len(region_names)} regions) ===")
    print(f"  f={best_key[0]:.2f}MHz  P={best_key[1]:.2f}MPa  DC={best_key[2]:.1f}%  "
          f"-> {best_mean:.1f}% mean GM2 reduction")

    print("\n=== Main-effect sensitivity (range of mean reduction across parameter levels) ===")
    for pname, d in sensitivity.items():
        print(f"  {pname:12s} range = {d['range_pct']:.1f} pct points  levels: {d['level_means']}")

    print("\n=== Top 5 parameter combinations (mean reduction across all regions) ===")
    for rank, ((f_MHz, P_MPa, DC_pct), mean_red) in enumerate(top5, start=1):
        print(f"  {rank}. f={f_MHz:.2f}MHz  P={P_MPa:.2f}MPa  DC={DC_pct:.1f}%  "
              f"-> {mean_red:.1f}% mean reduction")

    print("\n=== Control arms (no LIFU) vs. natural history ===")
    for arm, red_by_region in control_reduction.items():
        mean_red = float(np.mean(list(red_by_region.values())))
        print(f"  {arm:12s}  mean reduction = {mean_red:.1f}%")


if __name__ == "__main__":
    main()
