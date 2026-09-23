"""
derive_fus_gain_mapping.py  --  Step 1 of the 2026-09-15 task.

Derives a mapping from lifu_acoustic_model_v2.py's MI->BBB-opening-probability
sigmoid onto gm2_model_v2.py's scalar `fus_entry_gain_scale`, for the two ends
of the infant biological-scaling bound (k=1.0 conservative, k=2.0 permissive)
established in literature_calibration_data.json ->
infant_bbb_dose_response_scaling_2026-09-12.

Reads only. Does NOT modify gm2_model_v2.py or any acoustic-model file.
Writes fus_gain_mapping.json; the prose derivation it backs is
fus_gain_mapping_derivation.md.
"""
import json
import os
import sys

import numpy as np

LIFU_DIR = "/home1/11502/kartheek_nekkanti/lifu_gm2_250k"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LIFU_DIR)

import lifu_acoustic_model_v2 as m2  # noqa: E402

# --- reference operating point: taken verbatim from eta_results.json ---------
ETA = json.load(open(os.path.join(LIFU_DIR, "eta_results.json")))
ROP = ETA["reference_operating_point"]
F_MHZ = ROP["frequency_MHz"]
P0_MPA = ROP["source_pressure_P0_MPa"]
DC = ROP["duty_cycle"]

# gm2_model_v2.py's calibrated scalar (read, never written)
CALIBRATED_GAIN = 1.5

PRIMARY_AGE_MONTHS = 6.0          # midpoint of the acoustic model's 0-12mo window
SENSITIVITY_AGES = [1.5, 6.0, 12.0]


def opening_probabilities(age_months):
    """Per-region p_open at the two ends of the k-bound, plus the unweighted
    whole-brain mean (gm2_model_v2 has ONE lumped brain compartment and the
    acoustic model supplies no regional volume weights)."""
    rows = []
    for r in m2.REGION_NAMES:
        T = float(m2.transmission_efficiency_v2(F_MHZ, r, age_months))
        P_tissue = P0_MPA * T
        mi = P_tissue / np.sqrt(F_MHZ)
        p1, p2 = m2.cavitation_probability_v2_bounds(P_tissue, F_MHZ, age_months)
        rows.append({"region": r, "T": T, "MI_in_situ": float(mi),
                     "p_open_k1.0": p1, "p_open_k2.0": p2, "ratio": p2 / p1})
    p1_mean = float(np.mean([x["p_open_k1.0"] for x in rows]))
    p2_mean = float(np.mean([x["p_open_k2.0"] for x in rows]))
    return rows, p1_mean, p2_mean, p2_mean / p1_mean


def main():
    mi0_c, mi0_p = m2.infant_mi_sigmoid_center_bounds(PRIMARY_AGE_MONTHS)
    rows, p1, p2, ratio = opening_probabilities(PRIMARY_AGE_MONTHS)

    gain_k1 = CALIBRATED_GAIN * 1.0
    gain_k2 = CALIBRATED_GAIN * ratio

    sens = {}
    for a in SENSITIVITY_AGES:
        _, a1, a2, ar = opening_probabilities(a)
        sens[str(a)] = {"p_open_k1.0_mean": a1, "p_open_k2.0_mean": a2,
                        "ratio": ar, "gain_k2.0": CALIBRATED_GAIN * ar}

    # the rejected alternative rule, computed so the .md can show it explicitly
    alt_gain_k2 = 1.0 + (CALIBRATED_GAIN - 1.0) * ratio

    out = {
        "_generated_by": "derive_fus_gain_mapping.py (2026-09-15, Step 1)",
        "gm2_model_v2_sha256_16": "6693427fa909a0a9",
        "reference_operating_point": {"frequency_MHz": F_MHZ,
                                      "source_pressure_P0_MPa": P0_MPA,
                                      "duty_cycle": DC,
                                      "source": "eta_results.json -> reference_operating_point"},
        "mi_sigmoid": {"MI0_adult_fit": float(m2.MI_SIGMOID_CENTER),
                       "width": float(m2.MI_SIGMOID_WIDTH),
                       "MI0_k1.0": mi0_c, "MI0_k2.0": mi0_p},
        "primary_age_months": PRIMARY_AGE_MONTHS,
        "per_region": rows,
        "p_open_k1.0_brainmean": p1,
        "p_open_k2.0_brainmean": p2,
        "opening_probability_ratio": ratio,
        "saturation_ceiling_on_ratio": 1.0 / p1,
        "calibrated_fus_entry_gain_scale": CALIBRATED_GAIN,
        "fus_entry_gain_scale_k1.0": gain_k1,
        "fus_entry_gain_scale_k2.0": gain_k2,
        "rejected_alternative_rule_gain_k2.0": alt_gain_k2,
        "age_sensitivity": sens,
    }
    with open(os.path.join(HERE, "fus_gain_mapping.json"), "w") as f:
        json.dump(out, f, indent=2)

    # ---- console report -----------------------------------------------------
    print("=" * 78)
    print("STEP 1 DERIVATION: acoustic opening probability -> fus_entry_gain_scale")
    print("=" * 78)
    print(f"Reference operating point (eta_results.json): f={F_MHZ} MHz, "
          f"P0={P0_MPA} MPa, DC={DC}")
    print(f"MI sigmoid (lifu_acoustic_model_v2): MI0_adult={m2.MI_SIGMOID_CENTER:.6f}, "
          f"width={m2.MI_SIGMOID_WIDTH:.6f}")
    print(f"Infant bound on sigmoid center at age {PRIMARY_AGE_MONTHS} mo: "
          f"MI0(k=1.0)={mi0_c:.6f}, MI0(k=2.0)={mi0_p:.6f}")
    print()
    print(f"{'region':<18} {'T':>7} {'MI_insitu':>10} {'p_k1.0':>9} {'p_k2.0':>9} {'ratio':>7}")
    for r in rows:
        print(f"{r['region']:<18} {r['T']:7.4f} {r['MI_in_situ']:10.4f} "
              f"{r['p_open_k1.0']:9.6f} {r['p_open_k2.0']:9.6f} {r['ratio']:7.4f}")
    print(f"{'BRAIN MEAN':<18} {'':>7} {'':>10} {p1:9.6f} {p2:9.6f} {ratio:7.4f}")
    print()
    print(f"Opening-probability ratio R = p_open(k=2.0)/p_open(k=1.0) = {ratio:.6f}")
    print(f"  (hard ceiling imposed by p_open <= 1: R <= 1/{p1:.6f} = {1.0/p1:.4f})")
    print()
    print("RULE:  fus_entry_gain_scale(k) = 1.5 * [ p_open(k) / p_open(k=1.0) ]")
    print()
    print(f"  fus_entry_gain_scale (k=1.0, conservative) = {gain_k1:.6f}")
    print(f"  fus_entry_gain_scale (k=2.0, permissive)   = {gain_k2:.6f}")
    print()
    print("Age sensitivity of the permissive value (mapping rule unchanged):")
    for a, s in sens.items():
        print(f"  age={a:>5} mo:  R={s['ratio']:.4f}  ->  gain(k=2.0)={s['gain_k2.0']:.6f}")
    print()
    print("Written: fus_gain_mapping.json  (prose derivation: fus_gain_mapping_derivation.md)")


if __name__ == "__main__":
    main()
