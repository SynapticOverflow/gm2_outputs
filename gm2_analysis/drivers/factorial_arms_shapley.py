# factorial_arms_shapley.py
#
# Completes the 8-arm (2^3) SRT x AAV x FUS factorial design (the 6/8-arm
# table was missing FUS-alone and SRT+FUS) and replaces the Bliss-
# independence synergy index with an exact intervention-level Shapley
# decomposition over the three binary interventions.
#
# QOI matches the rest of the robustness audit: terminal (day-365) brain
# GM2 burden, tay-sachs headline protocol, gm2_model_v2's corrected
# solver (both the SRT-only silent-failure bug and the entry-rate units
# bug from the original repo are fixed here, so this table supersedes
# rather than reproduces the original Table 1 -- see gm2_model_v2.py's
# header for what was fixed).
#
# Each arm is run at n_seeds independent stochastic replicates, using the
# SAME seed across arms at a given replicate index (common random
# numbers) so that arm-to-arm differences aren't swamped by the ~85%
# process-noise variance fraction found elsewhere in this audit
# (data/matched_comparison.log).
#
# Characteristic function for the Shapley game: v(S) = E[burden(no
# treatment)] - E[burden(interventions in S on, rest off)], i.e. the
# expected burden REDUCTION from the untreated baseline. v({}) = 0 by
# construction; v({SRT,AAV,FUS}) = the full tri-modal benefit. Shapley
# values then partition that total benefit exactly among the three
# interventions, crediting synergy through the coalitions rather than
# assuming independence (what Bliss does).

import itertools
import json
import math
import time

from gm2_model_v2 import make_base_params, make_x0, simulate_milstein, DEFAULT_FUS_EVENTS

import sys

DISEASE = 'tay-sachs'
TMAX = float(sys.argv[1]) if len(sys.argv) > 1 else 365.0
DT = 0.2
N_SEEDS = 200
OUT_SUFFIX = f"_day{int(TMAX)}" if TMAX != 365.0 else ""


def run_arm(srt: bool, aav: bool, fus: bool, seed: int) -> float:
    base = make_base_params(DISEASE)
    x0 = make_x0(base)
    df = simulate_milstein(
        x0, TMAX, DT, base,
        dose_mg_per_day=3.0 if srt else 0.0,
        t4_admin_day=30.0,
        t4_dose_total=1.0 if aav else 0.0,
        t4_pulse_width_days=1.0,
        fus_events=DEFAULT_FUS_EVENTS if fus else None,
        seed=seed,
    )
    return float(df["gm2_brain"].iloc[-1])


def exact_shapley(v: dict, players=("SRT", "AAV", "FUS")) -> dict:
    """v: dict mapping frozenset(subset of players) -> characteristic value."""
    n = len(players)
    shapley = {p: 0.0 for p in players}
    for p in players:
        others = [q for q in players if q != p]
        for r in range(len(others) + 1):
            for subset in itertools.combinations(others, r):
                S = frozenset(subset)
                weight = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                marginal = v[S | {p}] - v[S]
                shapley[p] += weight * marginal
    return shapley


if __name__ == "__main__":
    t0 = time.time()
    arms = list(itertools.product([0, 1], repeat=3))  # (SRT, AAV, FUS)
    arm_table = {}
    for srt, aav, fus in arms:
        vals = [run_arm(bool(srt), bool(aav), bool(fus), seed) for seed in range(N_SEEDS)]
        mean = sum(vals) / len(vals)
        var = sum((y - mean) ** 2 for y in vals) / (len(vals) - 1)
        arm_table[(srt, aav, fus)] = {'mean': mean, 'std': var ** 0.5, 'n_seeds': N_SEEDS, 'raw': vals}
    elapsed = time.time() - t0

    no_treatment = arm_table[(0, 0, 0)]['mean']
    v = {}
    for srt, aav, fus in arms:
        S = frozenset(x for x, on in zip(("SRT", "AAV", "FUS"), (srt, aav, fus)) if on)
        v[S] = no_treatment - arm_table[(srt, aav, fus)]['mean']  # benefit vs. untreated

    shapley = exact_shapley(v)
    total_benefit = v[frozenset({"SRT", "AAV", "FUS"})]

    print(f"8-arm factorial ({DISEASE}, day-{TMAX:.0f} brain GM2 burden, n_seeds={N_SEEDS}), {elapsed:.0f}s")
    print(f"{'SRT':>4} {'AAV':>4} {'FUS':>4}   {'mean_burden':>12} {'std':>10}")
    for srt, aav, fus in arms:
        a = arm_table[(srt, aav, fus)]
        print(f"{srt:4d} {aav:4d} {fus:4d}   {a['mean']:12.4f} {a['std']:10.4f}")
    print(f"\nNo-treatment baseline burden: {no_treatment:.4f}")
    print(f"Full tri-modal benefit (burden reduction): {total_benefit:.4f}")
    print("\nShapley decomposition of tri-modal benefit over {SRT, AAV, FUS}:")
    for p, val in shapley.items():
        print(f"  {p:5s}  shapley={val:10.4f}  ({100 * val / total_benefit:5.1f}% of total benefit)")
    print(f"  sum of Shapley values = {sum(shapley.values()):.4f} (should equal total benefit = {total_benefit:.4f})")

    out = {
        'disease': DISEASE, 'tmax_days': TMAX, 'dt': DT, 'n_seeds': N_SEEDS,
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
    with open(f'data/factorial_arms_shapley{OUT_SUFFIX}.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\nDone.")
