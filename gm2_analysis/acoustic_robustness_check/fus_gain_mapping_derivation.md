# Mapping the LIFU acoustic model's BBB-opening probability onto `fus_entry_gain_scale`

**Date:** 2026-09-15
**Status of this document:** Step 1 deliverable. It defines the rule; it does not
itself run the GM2 model.
**Files read (none written):** `/home1/11502/kartheek_nekkanti/lifu_gm2_250k/lifu_acoustic_model_v2.py`,
`.../lifu_acoustic_model_v1.py`, `.../eta_results.json`,
`.../literature_calibration_data.json`,
`/work/11502/kartheek_nekkanti/vista/gm2_sobol/gm2_model_v2.py` (read-only, `sha256_16 = 6693427fa909a0a9`).
**Reproducer:** `derive_fus_gain_mapping.py` → `fus_gain_mapping.json`. Every number
below is printed by that script; nothing here is hand-typed independently of it.

This closes the gap that `LIFU_MODEL_V2_ADDENDUM.md` item 6 states explicitly: no
mapping between the acoustic model's `eta` and `gm2_model_v2.py`'s
`fus_entry_gain_scale` existed. It closes it *as an explicitly-assumption-laden
bridge*, not as a calibration — §6 below states what would have to be measured to
make it one.

---

## 1. The two quantities, and why they are not interchangeable

**Acoustic side.** `lifu_acoustic_model_v2.py` produces

```
eta(f, P0, DC, region, age) = T(f, region, age) * P_cav(MI_in_situ) * DC_factor(DC)
```

a dimensionless efficiency in [0, 1]. Its middle factor, `P_cav`, is the piece
this mapping uses: a **probability that a sonication opens the BBB**, from a
logistic fit to three cross-species MI anchors,

```
P_cav(MI) = 1 / (1 + exp(-(MI - MI0) / w)),    MI = P_in_situ / sqrt(f)
```

with (item-1-corrected fit) `MI0_adult = 0.436075`, `w = 0.117185`.

**GM2 side.** `gm2_model_v2.py` has no acoustic submodel at all. FUS enters as a
smoothed Gaussian envelope `m_FUS(t)` (peaking at `1 + FUSEvent.gain` per session:
1.8, 1.6, 1.5 on days 30, 60, 90) multiplying one rate constant:

```
k_entry_eff = k_T4_entry * (1 + fus_entry_gain_scale * (m_FUS(t) - 1)) * immune_suppression
```

`fus_entry_gain_scale` (calibrated value **1.5**) is therefore a **dimensionless
coupling coefficient on a rate multiplier**, with units of "fractional increase in
AAV BBB-entry rate per unit of FUS envelope". It is not a probability and has no
upper bound of 1.

Substituting `eta` (or `P_cav`) for `fus_entry_gain_scale` directly would be a
category error: it would assert that a 0.88 opening probability means a 0.88×
coupling coefficient, silently *reducing* the calibrated FUS effect by 41% for
reasons that have nothing to do with infant physiology. That is not done here.

## 2. What the mapping is allowed to touch

The only thing the infant bound `k` changes in the acoustic model is the **sigmoid
center**: `MI0_infant = MI0_adult / k`
(`literature_calibration_data.json → infant_bbb_dose_response_scaling_2026-09-12`,
derivation step 4; `k ∈ [1.0, 2.0]`, giving `MI0_infant ∈ [0.23, 0.46]`).

`k` does **not** touch the transmission term `T`, the duty-cycle term, the
operating point, or the acoustic protocol. So for a fixed protocol and region:

```
eta(k=2.0) / eta(k=1.0)  ==  P_cav(k=2.0) / P_cav(k=1.0)
```

exactly — `T` and `DC_factor` cancel. The entire infant-vs-adult signal available
from this model, at a fixed protocol, is the **opening-probability ratio**. That
ratio is the only quantity this mapping transfers. Everything else about the FUS
mechanism stays at its calibrated value.

## 3. Operating point and aggregation (both are choices; both are stated)

- **Operating point:** `f = 0.5 MHz`, `P0 = 0.6 MPa`, `DC = 0.01`, taken verbatim
  from `eta_results.json → reference_operating_point` (itself the adult-rat point
  of Wei/Chu/Hsu et al. 2013). Not re-derived here.
- **Age:** `gm2_model_v2.py` has **no patient-age state or parameter** — this was
  checked; it is a lumped 17-state model with Bayley-III infant composites and no
  age anchor. An age must therefore be *chosen* to evaluate the acoustic model.
  **6.0 months** is used: the midpoint of the acoustic model's own 0–12 month
  applicability window, and one of the three ages `eta_results.json` tabulates.
  §5 shows the whole 1.5–12 month range, which moves the answer by <5%.
- **Region:** `gm2_model_v2.py` has **one lumped brain compartment**; the acoustic
  model has eight regions with no volume weights supplied anywhere in the project.
  The whole-brain value is therefore the **unweighted mean of the eight regional
  `P_cav` values**. Unweighted, because inventing volume weights would be
  unsupported precision; the spread across regions is small (§4) so this choice is
  not load-bearing.

## 4. The computed opening-probability shift (age 6.0 months)

`MI0(k=1.0) = 0.436075`, `MI0(k=2.0) = 0.218037`.

| region | T | MI in situ | P_cav (k=1.0) | P_cav (k=2.0) | ratio |
|---|---|---|---|---|---|
| Frontal Cortex | 0.8503 | 0.7215 | 0.919490 | 0.986561 | 1.0729 |
| Parietal Cortex | 0.8290 | 0.7035 | 0.907359 | 0.984364 | 1.0849 |
| Thalamus | 0.7827 | 0.6641 | 0.875009 | 0.978260 | 1.1180 |
| Basal Ganglia | 0.7899 | 0.6702 | 0.880617 | 0.979345 | 1.1121 |
| Temporal Lobe | 0.8013 | 0.6800 | 0.889061 | 0.980957 | 1.1034 |
| Occipital Lobe | 0.7645 | 0.6487 | 0.859890 | 0.975277 | 1.1342 |
| Cerebellum | 0.7428 | 0.6303 | 0.839878 | 0.971194 | 1.1564 |
| Hippocampus | 0.7945 | 0.6741 | 0.884045 | 0.980002 | 1.1085 |
| **brain mean** | | | **0.881919** | **0.979495** | **1.110641** |

```
R  ==  P_cav(k=2.0) / P_cav(k=1.0)  =  0.979495 / 0.881919  =  1.110641
```

**The single most important fact in this derivation:** the reference operating
point sits *far above* the sigmoid center (in-situ MI ≈ 0.63–0.72 vs `MI0_adult` =
0.436), so `P_cav` is already ~0.88 saturated in the adult/conservative case. A 2×
reduction in the MI threshold — a large change in the threshold — can therefore
only buy a **+11.1%** relative increase in opening probability. The arithmetic
ceiling, from `P_cav ≤ 1`, is `R ≤ 1/0.881919 = 1.1339`; the realized 1.1106 is
already 82% of the way to it. **A factor-of-2 shift in the underlying biological
threshold is compressed into a ~11% effect by sigmoid saturation at this protocol.**
Any weak result downstream should be read against that fact before being read as a
statement about infant physiology.

## 5. Age sensitivity of R (rule unchanged, only the evaluation age varies)

| age (months) | P_cav mean (k=1.0) | P_cav mean (k=2.0) | R | resulting gain (k=2.0) |
|---|---|---|---|---|
| 1.5 | 0.899195 | 0.982824 | 1.0930 | 1.639506 |
| **6.0 (used)** | **0.881919** | **0.979495** | **1.110641** | **1.665961** |
| 12.0 | 0.853543 | 0.973874 | 1.1410 | 1.711468 |

Younger = thinner skull = higher transmission = deeper into sigmoid saturation =
*smaller* infant-vs-adult ratio. The full 0–12 month span moves the derived
permissive gain by 1.640 → 1.711, i.e. ±2% about the 6-month value. The age choice
is not what decides this analysis.

## 6. The rule

> **`fus_entry_gain_scale(k) = 1.5 × [ P_cav(k) / P_cav(k = 1.0) ]`**
>
> where 1.5 is `gm2_model_v2.py`'s existing calibrated value, `P_cav` is
> `lifu_acoustic_model_v2.cavitation_probability_v2_bounds` evaluated at the
> `eta_results.json` reference operating point, age 6.0 months, averaged unweighted
> over the eight modeled regions.

Three things this rule asserts, each of which is an assumption, not a result:

1. **The FUS-attributable increment in AAV BBB-entry rate is proportional to the
   probability that the sonication opens the barrier.** This is the actual physical
   content. It says: if a protocol opens the BBB 11% more often, it delivers 11%
   more FUS-attributable vector flux. It is a linear, no-threshold, no-saturation
   assumption about how opening *probability* converts into transported *mass*, and
   no source in this project establishes that functional form. It is the same class
   of stated-but-unvalidated proportionality as derivation step 4 in
   `literature_calibration_data.json` — this mapping inherits that weakness and adds
   one of its own.
2. **`k = 1.0` is what the existing calibration already means.** `fus_entry_gain_scale
   = 1.5` was fit with no infant-specific acoustic adjustment of any kind, which is
   exactly the `k = 1.0` (infant-indistinguishable-from-adult) branch. So the
   conservative configuration must reproduce the calibrated value identically —
   and it does, `1.5 × 1.0 = 1.5`, *by construction*. This is a consistency
   requirement on the mapping, not a coincidence, and it makes the k=1.0 arm a
   built-in control: any difference between k=1.0 results and the published results
   would indicate a pipeline error, not a finding.
3. **The ratio, not the level, is transferred.** `P_cav`'s absolute level (0.88) is
   the weakest number in the acoustic model — it rests on a 3-point cross-species
   sigmoid whose leave-one-out error is ~0.14–0.16 in probability space. Its
   *ratio* between two values of `k` is far more robust, because the sigmoid's fitted
   center and width, the transmission term, the duty-cycle term and the operating
   point are all common to numerator and denominator and cancel or nearly cancel.
   Transferring the level would import the acoustic model's worst-calibrated
   quantity into the GM2 model; transferring the ratio does not.

### Why not "scale the gain's departure from 1.0"

The task suggested, as one candidate, scaling `fus_entry_gain_scale`'s departure
from 1.0. That is rejected, on the model's own algebra: in
`k_entry_eff = k_T4_entry * (1 + fus_entry_gain_scale * (m_FUS - 1)) * ...`, the
value of `fus_entry_gain_scale` at which FUS has **no effect** is **0**, not 1 —
setting it to 0 gives `k_entry_eff = k_T4_entry` exactly. 1.0 is not a neutral
point; it is just "the FUS envelope passes through at unit coupling". So the
FUS-attributable entry increment is proportional to `fus_entry_gain_scale` itself,
and scaling it from a zero baseline is the transformation that makes "11% more
opening → 11% more FUS effect" true. (`fus_uptake_gain_scale`, the structurally
identical SRT-uptake coefficient, has calibrated value exactly 1.0, which is
probably where the 1.0-as-baseline intuition comes from; it is still 0 that is
neutral there too.)

For the record, the rejected rule would have given
`1.0 + 0.5 × 1.110641 = 1.555320` instead of 1.665961 — a *smaller* permissive
departure, i.e. this choice is not the one that flatters the infant hypothesis.

### What would make this a calibration instead of a bridge

A single measurement of FUS-mediated AAV brain entry (not tracer permeability) at
two or more MI values in an open-fontanelle infant-equivalent skull. That would
replace assumption 1 with a measured dose-response and replace the `k ∈ [1.0, 2.0]`
bound with a number. Until then this mapping propagates a literature-analogy bound
through a stated-but-unvalidated proportionality, twice.

## 7. Result

| configuration | k | MI0 (infant) | P_cav (brain mean) | R | **`fus_entry_gain_scale`** |
|---|---|---|---|---|---|
| conservative (adult-equivalent, BSA/macromolecule branch) | 1.0 | 0.436075 | 0.881919 | 1.000000 | **1.500000** |
| permissive (FM1-43/small-molecule branch) | 2.0 | 0.218037 | 0.979495 | 1.110641 | **1.665961** |

These two values are carried into Step 2 as two separate model configurations.
They are **not** averaged, and no interior value is used: the range is the result,
per `literature_calibration_data.json`'s own reasoning that collapsing a
bidirectional, tracer-size-dependent evidence base to one number hides real
disagreement in the sources.
