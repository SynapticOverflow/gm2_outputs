# LIFU acoustic efficiency model v2 — addendum (items 1-6, 2026-09-12)

**Date:** 2026-09-12
**Supersedes (for the items below only):** Sections 1-2, 4 and the mcdannold2008 row of
`LIFU_MODEL_V1_REPORT.md` (2026-09-11), which is kept unmodified as the historical record.
**Files produced this pass:** `lifu_acoustic_model_v2.py`, `generate_eta_results_v2.py`,
updated `literature_calibration_data.json` (13 sources, up from 10; new top-level
`infant_bbb_dose_response_scaling_2026-09-12` derivation section), regenerated `eta_results.json`
(old version preserved at `eta_results_v1_archived_2026-09-11.json`), this report.
**Files NOT touched:** `gm2_model_v2.py`, `lifu_acoustic.py`, `lifu_acoustic_model_v1.py`
(kept as the 2026-09-11 historical record, same pattern as the disclaimer-patch versioning).

---

## Item 1 — primary source for the central calibration anchor

**Resolved.** `mcdannold2008`'s MI=0.47-at-50%-probability figure was flagged in v1 as
secondary-sourced (primary PDF paywalled). The primary text (PMC2442477, free full text) was
fetched and read directly this pass. It gives:

> MI at 50% probability of BBB disruption = **0.46** (95% CI: 0.42–0.50)

not 0.47. The frequency range for the pooled regression is also wider than previously recorded
(0.26–2.04 MHz, not capped at 1.63 MHz, because a new 2.04 MHz rabbit dataset — n=11, 40
sonicated locations — is pooled with historical rabbit/rat data in the primary text). The
core finding this model relies on — that MI, not raw pressure, is the frequency-independent
threshold variable — is confirmed directly from the primary text, not just corroborated by
search snippets as in v1.

**Effect on the model:** the MI sigmoid was re-fit on the corrected anchor set
`[0.245, 0.46, 0.60]` (was `[0.245, 0.47, 0.60]`). New fit: MI₀=0.4361, width=0.1172 (v1:
MI₀=0.4414, width=0.1182) — a small (~1.2%) shift, well within the anchor's own reported 95%
CI, but now resting on a primary-sourced number with a real confidence interval rather than a
secondary-sourced point value. Leave-one-out validation was re-run on the corrected anchors;
errors are essentially unchanged (0.123–0.138 vs. v1's 0.144–0.160), as expected for a small
shift in one of three anchors.

---

## Item 2 — infant/neonatal/juvenile BBB-opening dose-response, broadened search

**Broadened search performed; still no direct hit — escalated to Option A, delivered as a
bounded range (Option A+C combined), per the task's instructions.**

### What the broadened search found

| Search target | Result |
|---|---|
| Piglet/neonatal-pig FUS-BBB-opening dose-response | One study located (closed-loop cavitation-feedback BBBO in ~3 pigs, ~4 weeks old) — **primary text paywalled/inaccessible this session.** No usable number extracted; not cited with numbers. |
| Juvenile rodent FUS-BBB-opening dose-response | None located with accessible primary text beyond the existing adult-animal anchors (tung2011 mouse, chu2016 rat, mcdannold2008 rabbit — all adult animals). |
| Human pediatric FUS-BBB-opening trial | **Found and obtained in full text this pass** (previously 403/inaccessible in the 2026-09-11 search): Wu et al. 2025, *Sci Transl Med* 17:eadq6645 — neuronavigation-guided FUS-BBBO in 3 children (trial-eligible ages 4–21y) with diffuse midline glioma. See below — **does not fill the gap**, but is now documented with real numbers. |
| Age-dependent general BBB-permeability-maturation literature (Option A target) | **Found, primary-sourced:** Hanafy & Dietrich (2026), *Fluids and Barriers of the CNS* 23:98 — mouse, E18–P25, quantitative tracer-permeability-by-age data. Used as the basis for the bounded scaling below. |
| Allometric/tissue-property data for a first-principles derivation (Option B target) | **Not found beyond alsanea2025** (already fully used for geometric thickness in the transmission term — see item 3). No independently-measured infant tissue stiffness, water content, or myelination dataset usable for a cavitation-threshold derivation was located. Option B was therefore not reached. |

### Why the human pediatric trial (Wu et al. 2025) does not fill the gap, despite being real, quantitative, human, pediatric FUS-BBB data

This is worth stating precisely because it is the closest real-world match found, and it would
be dishonest to either ignore it or overstate what it shows:

- Trial-eligible ages were 4–21 years; DMG epidemiology means the 3 treated patients were very
  likely school-age or older — **ossified skull**, not the 0–12-month open-fontanelle regime
  this model targets.
- The device (UltraNav, 0.25 MHz) targeted an **in-situ peak-negative pressure of 0.2 MPa**
  (MI = 0.2/√0.25 = **0.40**), individually derated per patient from a pre-treatment CT-based
  skull attenuation simulation (k-Wave) — methodologically similar in spirit to this model's own
  CT-based transmission term, though for a different (ossified) skull regime.
- Critically, the paper states this pressure was **the lowest value permitted under the trial's
  investigational-device-exemption (IND) safety constraints**, not a value derived from measuring
  a pediatric-specific biological threshold. A partial (not-all, not-none) confirmed-opening rate
  at this MI is therefore consistent with, but does not validate or calibrate, the adult MI curve
  for this age range — the operating point was chosen for regulatory reasons, so its outcome
  can't be attributed to biology one way or the other.
- **This is documented in `literature_calibration_data.json` (`wu2025_pediatric_fus`) as context,
  not used as a calibration anchor.**

### The bounded infant biological-scaling correction actually implemented

Per the task's A→B→C order, Option A was reached: **Hanafy & Dietrich (2026)** measured mouse
BBB tracer permeability across development (E18–P25) using in-situ microperfusion + two-photon
microscopy. Two real, table-exact numbers matter here:

- **FM1-43** (small-molecule probe): P1 = 0.18 ± 0.02 vs. P25 = 0.09 ± 0.01 → **2.0× higher**
  neonatal permeability.
- **BSA** (66 kDa macromolecule — the tracer size class closest to this project's actual
  FUS-relevant cargo, AAV capsids, ~20–25 nm, far larger than the small-molecule dyes): **no
  measurable difference** between P2 and P25 → **1.0×** (no effect).

This is a **general (non-FUS) BBB-biology finding**, not a measurement of FUS-cavitation
threshold. Applying it required one explicit, stated (not hidden) assumption:

> **ASSUMPTION:** MI₀_infant = MI₀_adult / k — the mechanical index needed to reach a given
> BBB-opening probability scales inversely and linearly with the baseline (passive) paracellular
> permeability ratio. This is a physically-motivated analogy (a barrier that already leaks more
> at baseline should need less additional mechanical force to force open further), **not a
> validated pharmacodynamic law** — no source found in this search establishes this functional
> form for cavitation-induced (as opposed to passive) BBB opening.

Because the source evidence itself is **bidirectional by tracer size** (2× for small molecules,
1× for the macromolecule that best matches this project's actual gene-therapy cargo), collapsing
it to a single point value would hide a real disagreement in the underlying evidence. Per the
task's own permission to combine methods, this was delivered as **Option C's bounded range, with
both bounds now literature-derived rather than merely qualitative**:

| Bound | k | MI₀ (using corrected adult anchor 0.46) | Basis |
|---|---|---|---|
| Conservative (least permissive) | 1.0 | 0.46 (= adult, unchanged) | BSA / macromolecule, AAV-size-relevant |
| Permissive (most permissive) | 2.0 | 0.23 | FM1-43 / small-molecule |

No literature evidence was found supporting k<1.0 (infant *less* permeable/needing *higher* MI
than adult) — that direction is genuinely absent from the search, not silently assumed away, so
the range is one-sided by evidence, not by choice.

**Applied uniformly across the model's 0–12 month target range** (a coarse infant/adult binary),
not as a continuous function of age within infancy: Hanafy & Dietrich's timeline is mouse
postnatal days (E18–P25), and no citation was found in this search mapping mouse postnatal-day
BBB maturation onto human infant months. Inventing an interpolated human-age curve from a mouse
timeline would itself be an unsupported precision claim.

**Combined uncertainty is now wider, not narrower, than v1's:** `lifu_acoustic_model_v2.py`'s
Monte Carlo draws `k` uniformly from `[1.0, 2.0]` (age ≤ 12mo) as an additional axis on top of
v1's existing transmission/cavitation-fit uncertainty. v1 was equivalent to silently fixing
`k=1.0` with no infant-specific axis at all. Every `eta_by_age_months` entry in the regenerated
`eta_results.json` now reports `eta_conservative_k1.0` and `eta_permissive_k2.0` as two explicitly
labeled bounds, in addition to the combined `eta_median`/`eta_p5`/`eta_p95`. Numerically, the gap
between the two bounds is operating-point-dependent — near the sigmoid's inflection (sub-threshold
pressures) the permissive:conservative ratio reaches ~4×; at pressures well above threshold
(where both bounds saturate near probability 1) the gap narrows to a few percent. This is
physically sensible sigmoid behavior, not a bug — verified by sweeping P0 from 0.15 to 0.6 MPa
at the reference frequency/age (see chat transcript / re-run `lifu_acoustic_model_v2.py`).

---

## Item 3 — using existing CT data

**Re-checked; nothing additional found; no code change resulted.**

- Re-fetched and re-read `alsanea2025`'s primary text specifically looking for bone
  density/Hounsfield-unit data, fontanelle-area/size measurements, or any acoustic/mechanical
  property alongside the geometric thickness table. **None is reported** — the paper states
  explicitly that it is "strictly a geometric/morphometric study of skull thickness
  distributions" and cites (without presenting) other authors' work on material properties.
- Searched the filesystem (`$HOME` and `$WORK`) for any infant CT/DICOM/head-model data not
  already reflected in `literature_calibration_data.json`. **None found** — no `.dcm` files, no
  head-model files, no CT data of any kind exist on this filesystem outside the published
  `alsanea2025` table already in use.
- **Conclusion:** the existing CT data (alsanea2025's 266-infant thickness-by-age-and-region
  table) was already being used to its full extent in v1's transmission term. There was no
  additional existing CT data — on this filesystem or in that paper's own reported results — left
  to incorporate. `transmission_efficiency_v1`/`v2` is unchanged.

---

## Items 4-6 — propagate uncertainty, regenerate output, revised integration assessment

**Item 4 (propagate uncertainty):** Done — see item 2 above and `lifu_acoustic_model_v2.py`'s
`compute_eta_v2_with_uncertainty`. The new infant biological-scaling axis is combined with, not
substituted for, v1's existing literature-uncertainty Monte Carlo (transmission-term fit
residuals, infant-thickness SD, open-fontanelle insertion-loss SD, MI-anchor
interpretive-probability perturbation).

**Item 5 (regenerate output):** Done — `eta_results.json` regenerated via
`generate_eta_results_v2.py`. The 2026-09-11 output is preserved unmodified at
`eta_results_v1_archived_2026-09-11.json`. **Note for whoever next touches the (off-filesystem)
demo tool:** the JSON schema changed — the old `eta_point_estimate` field is replaced by two
explicitly labeled fields, `eta_conservative_k1.0` and `eta_permissive_k2.0`, plus an
`infant_biological_scaling` block explaining them. `lifu_demo_disclaimer_patch_v2.html`'s banner
text and amber/red tagging logic are still accurate in spirit (per-region eta is
literature-calibrated-but-unvalidated-in-infants; other panels remain uncalibrated) but its
example field names were written against v1's schema. A `_v3` patch documenting the field-name
change is a small, mechanical follow-up not done in this pass since the demo's own source code
is still not present on this filesystem to test against (per `PHASE1_acoustic_model_investigation.md`).

**Item 6 (revised paper-integration assessment):** **Still not solid enough for the paper's FUS
mechanism claim — the reasons have shifted, but the bottom line has not.** Specifically:

1. Item 1 removed one real weakness (secondary-sourced central anchor) — the anchor is now
   primary-sourced with a reported 95% CI. This is a genuine, if small, improvement.
2. Item 2 did **not** close the underlying gap (no direct infant FUS-BBB dose-response data
   exists) — it replaced a *silent* assumption (v1 implicitly used k=1.0, infant biology = adult
   biology, without saying so) with an *explicit, bounded, literature-anchored* one (k ∈
   [1.0, 2.0]). This is more honest, but it necessarily makes the model's stated uncertainty
   **wider**, which is the opposite of "more ready to integrate" in the narrow sense of producing
   a tighter number for a paper claim. A 2× spread in the effective in-situ dose threshold, on
   top of the pre-existing cross-species LOO error of ~0.12–0.14 in probability space, is a
   large combined uncertainty for a mechanism claim.
3. Item 3 found no new data, so the transmission term's limitations (single adult ex vivo
   source, single n=1 neonatal specimen, no infant-specific attenuation *coefficient* — only
   total insertion loss at 2 close frequencies) are unchanged from v1.
4. The wiring gap Section 6 of `LIFU_MODEL_V1_REPORT.md` identified — that `gm2_model_v2.py`'s
   scalar `fus_entry_gain_scale` and this model's dimensionless per-region `eta` have no
   published or derived mapping between them — is untouched by any of items 1-3 and remains
   unsolved.

**What changed in the recommendation's texture, not its conclusion:** v1's report said the model
was "not solid enough… the infant-specificity is transmission-only, not biological." That is
still true in the sense that no *direct* infant biological measurement exists. What v2 adds is a
*documented, bounded, literature-anchored* treatment of that gap instead of an *undocumented,
point-value* one — which is the right thing to do methodologically, and is what would be defended
in a methods section, but it does not manufacture calibration data that isn't there. **The
model remains appropriate as a documented, honestly-uncertain exploration tool, and is now
somewhat more defensible as one** (every number, including the width of the uncertainty itself,
traces to a real source or a stated, non-hidden assumption) **but integrating it into the paper's
FUS mechanism claim would still overstate what has actually been calibrated.**
