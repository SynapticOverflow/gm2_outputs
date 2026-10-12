"""
summarize_dt_refinement.py

Prints the dt=0.2 vs dt=0.05 comparison tables quoted in this directory's
README.md. Reads only: the dt=0.05 JSONs in this directory and their dt=0.2
production counterparts elsewhere in gm2_analysis/. Runs no model evaluations
and writes no files.

Usage (from the repository root):
    python gm2_analysis/dt_refinement_dt0.05/summarize_dt_refinement.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GA = os.path.dirname(HERE)
FS = os.path.join(GA, "factorial_shapley_outputs")
AC = os.path.join(GA, "acoustic_robustness_check")
SO = os.path.join(GA, "sobol_outputs")


def load(*p):
    with open(os.path.join(*p)) as f:
        return json.load(f)


def ci(d, k="point_estimate"):
    return f"{d[k]:.3f} [{d['ci95_lo']:.3f}, {d['ci95_hi']:.3f}]"


def shapley_table():
    print("## Shapley benefit, day 365 (point [95% CI]), 200 matched seeds\n")
    print("| config | modality | dt=0.2 | dt=0.05 |")
    print("|---|---|---|---|")
    pairs = [
        ("base", (FS, "shapley_with_ci.json"), (HERE, "shapley_with_ci_dt0.05.json")),
        ("acoustic k1", (AC, "shapley_with_ci_acoustic_k1.json"), (HERE, "shapley_with_ci_acoustic_k1_dt0.05.json")),
        ("acoustic k2", (AC, "shapley_with_ci_acoustic_k2.json"), (HERE, "shapley_with_ci_acoustic_k2_dt0.05.json")),
    ]
    for name, a, b in pairs:
        A, B = load(*a)["day365"], load(*b)["day365"]
        for m in ("AAV", "SRT", "FUS"):
            print(f"| {name} | {m} | {ci(A[m])} | {ci(B[m])} |")
    print()
    print("| config | untreated burden, dt=0.2 | dt=0.05 | total benefit, dt=0.2 | dt=0.05 |")
    print("|---|---|---|---|---|")
    pairs = [
        ("base", (FS, "factorial_arms_shapley.json"), (HERE, "factorial_arms_shapley_dt0.05.json")),
        ("acoustic k1", (AC, "factorial_arms_shapley_acoustic_k1.json"), (HERE, "factorial_arms_shapley_acoustic_k1_dt0.05.json")),
        ("acoustic k2", (AC, "factorial_arms_shapley_acoustic_k2.json"), (HERE, "factorial_arms_shapley_acoustic_k2_dt0.05.json")),
    ]
    for name, a, b in pairs:
        A, B = load(*a), load(*b)
        print(f"| {name} | {A['no_treatment_burden']:.2f} | {B['no_treatment_burden']:.2f} "
              f"| {A['total_benefit']:.2f} | {B['total_benefit']:.2f} |")
    print()


def paired_table():
    print("## Paired FUS contrasts, day 365 (mean diff [95% CI]; significant?)\n")
    print("| config | contrast | dt=0.2 | dt=0.05 |")
    print("|---|---|---|---|")
    pairs = [
        ("base", (FS, "paired_contrasts.json"), (HERE, "paired_contrasts_dt0.05.json")),
        ("acoustic k1", (AC, "paired_contrasts_acoustic_k1.json"), (HERE, "paired_contrasts_acoustic_k1_dt0.05.json")),
        ("acoustic k2", (AC, "paired_contrasts_acoustic_k2.json"), (HERE, "paired_contrasts_acoustic_k2_dt0.05.json")),
    ]
    for name, a, b in pairs:
        A, B = load(*a)["day365"], load(*b)["day365"]
        for c in A:
            fa = f"{A[c]['mean_paired_diff']:.4g} [{A[c]['ci95_lo']:.4g}, {A[c]['ci95_hi']:.4g}]; {A[c]['significant']}"
            fb = f"{B[c]['mean_paired_diff']:.4g} [{B[c]['ci95_lo']:.4g}, {B[c]['ci95_hi']:.4g}]; {B[c]['significant']}"
            print(f"| {name} | {c} | {fa} | {fb} |")
    print()


def full14_table():
    print("## Full-14 Sobol S_T, seed 42 (task1 production at dt=0.2 vs task1 rerun at dt=0.05)\n")
    for sc in ("as_documented", "equal_width", "tier1_promotion"):
        A = load(SO, f"gm2_full14_{sc}.json")
        B = load(HERE, f"gm2_full14_{sc}_dt0.05.json")
        assert A["names"] == B["names"]
        print(f"### {sc}  (Y_mean: dt=0.2 {A['Y_mean']:.3f}, dt=0.05 {B['Y_mean']:.3f})\n")
        print("| parameter | S_T dt=0.2 | S_T dt=0.05 |")
        print("|---|---|---|")
        for n, x, y in zip(A["names"], A["ST"], B["ST"]):
            if max(x, y) > 1e-4:
                print(f"| {n} | {x:.3f} | {y:.3f} |")
        print()
    D = load(HERE, "diagnostic_sobol_fixed_matrix_dt0.2_vs_dt0.05.json")
    print("## Diagnostic: same Sobol matrix (sample seed 42), model seed 42, dt=0.2 vs 0.05\n")
    print(f"run_meta.seeds: {D['run_meta']['seeds']}\n")
    print("| scenario | Y_mean dt=0.2 | Y_mean dt=0.05 | top-3 S_T dt=0.2 | top-3 S_T dt=0.05 |")
    print("|---|---|---|---|---|")
    for sc in ("as_documented", "equal_width", "tier1_promotion"):
        r = []
        for dt in ("0.2", "0.05"):
            d = D[sc][dt]
            top = sorted(zip(d["names"], d["ST"]), key=lambda t: -t[1])[:3]
            r.append((d["Y_mean"], ", ".join(f"{n} {v:.3f}" for n, v in top)))
        print(f"| {sc} | {r[0][0]:.2f} | {r[1][0]:.2f} | {r[0][1]} | {r[1][1]} |")
    print()


def ensemble_table():
    D = load(HERE, "diagnostic_dt_convergence_ensemble.json")
    dts = [str(x) for x in D["run_meta"]["dts"]]
    print(f"## Diagnostic: dt-convergence ensemble, seeds {D['run_meta']['seeds']} (mean gm2_brain +/- SEM)\n")
    print("| day | arm | " + " | ".join(f"dt={d}" for d in dts) + " |")
    print("|---|---|" + "---|" * len(dts))
    for day, arms in D["results"].items():
        for arm, v in arms.items():
            print(f"| {day} | {arm} | " + " | ".join(f"{v[d]['mean']:.2f} +/- {v[d]['sem']:.2f}" for d in dts) + " |")
    print()


if __name__ == "__main__":
    shapley_table()
    paired_table()
    full14_table()
    ensemble_table()
