# Seed-sensitivity check of the 14-parameter GM2 Sobol analysis (task10)

Lonestar6, allocation IRI23021, 2026-10-09. `gm2_model_v2.py` sha256
`6693427fa909a0a95093cf6797890c6c4004912fd2a54b7c9e9966e77fa8a4a6`, checked by
the driver at import and by the SLURM script before and after every job; unchanged.

**Question.** Production full-14 Sobol (`task1_full14.py` → `sobol_driver_v2.py`)
evaluates every row with model seed 42 (common random numbers). Do total-order
indices, rankings and the prior-width-artifact conclusion depend on that seed?

**Short answer.** Partly. The prior-width-artifact *direction* is robust; the
*rankings below rank 1 and the magnitudes* are not, and seed 42 is an extreme
seed for the top parameters. Details and numbers: `SUMMARY_N256.md`.

## Design

Everything held fixed except the model-evaluation noise seed:
N=256, D=14, `calc_second_order=True` (7680 rows/scenario), dt=0.2, tmax=365,
`sobol_driver_v2.PARAM_NAMES` / `make_bounds()`, three scenarios, same X for
every run, SALib 1.5.2 `analyze(num_resamples=1000, seed=8162026)`, 95%
percentile CI from `ST_conf_all`/`S1_conf_all`.

| run | seed scheme |
|---|---|
| A | one seed for all rows: 42 (control), 7, 101, 2024, 31337, 90210 |
| B | row *i* uses seed 1000+*i* (independent noise per evaluation) |
| C | each row = mean of seeds 0–9 (same 10 seeds for all rows); per-seed Y saved |

**Which X.** The original `task1_full14.py` sample was drawn with SALib's default
`seed=None` and its Y was never saved, so that X cannot be recovered. The X here is
the one from `task7_percentile_cis.py` (sample seed 20260816), which is what
`task_outputs/percentile_cis.json` reports. The seed-42 control reproduces that file
**exactly** (max |ΔS_T| = 0, max |ΔCI| = 0, all scenarios). Against the original
production `gm2_full14_*.json` (different X, SALib default B=100 normal CI):
max |ΔS_T| = 0.021 / 0.017 / 0.012, and 13–14/14 production S_T fall inside our
95% CI. The only miss is FUS: production 0.0 vs ours 1.9e-6 with a CI of width ~0.

**Supplementary replicates.** C's 10 per-seed Y columns are each a complete
fixed-seed Sobol run on the same X; `task10_summary.py` re-analyses them (no new
model evaluations). **Correction (task11):** seed 7 is in both run A's list and run C's
seeds 0–9, and the two Y arrays are bit-identical. That gives **15 distinct** fixed-seed
replicates, not the 16 first reported; the summary now skips C's seed-7 column, and the
numbers below are the corrected 15-seed values. No conclusion changed.

**Choices to be aware of**
- "Four delivery/dose params" = `k_T4_entry`, `t4_dose_total`, `k_T4_payload`,
  `k_T4_decay` (the AAV `k_T4_*`/`t4_*` family). Table 4 isn't on this filesystem;
  change `DELIVERY4` in `task10_summary.py` if it names a different set.
- S_T ≤ 1e-4 is treated as numerically zero for ranking and ordering
  (`NEGLIGIBLE`). This was set after a first pass at 1e-6 spuriously ranked FUS
  (S_T = 1.9e-6) and "ordered" its three ~zero values. No other threshold changed.
- §5 is the literal check that was requested (seed 42's full three-scenario order).
  §9 additionally restricts to scenario pairs where the parameter's *own* prior width
  differs, because the other pairs compare identical priors.

## Results (N=256)

1. **Rank 1 is robust, the rest is not.** `gm2_synth` is rank 1 in 14/15 fixed seeds
   and in C (`vmax_brain` in the other one), with `vmax_brain` almost always 2nd. Ranks
   3–5 (`k_T4_decay`, `km_brain`, `k_T4_payload`, `t4_dose_total`) reshuffle across seeds.
   The top-4 set differs from seed 42's in several seeds. Spearman ρ vs seed 42 over the
   non-negligible params is 0.83–1.00 (A and C).
2. **Seed 42 is extreme for the top parameters.** Among the 15 distinct seeds it gives the
   *lowest* `gm2_synth` S_T (1/15 in all three scenarios: 0.24 vs median 0.36), the
   *highest* `k_T4_decay` (15/15: 0.22 vs median 0.14), and the 13–14/15th-highest
   `km_brain`. Seed 42 makes the top four look like a near-tie at ~0.2. Under the
   10-seed average (C), `gm2_synth` is ≈0.44, about 2× `vmax_brain` (0.22), and three
   params tie at ≈0.10–0.11. Its mean Y (3.8) is also the lowest of all 15 seeds
   (other seeds ≈9–83, depending on scenario).
3. **ΣS_T(delivery4):** seed 42 = 0.42 / 0.34 / 0.38 (as_doc / equal / tier1). The 15-seed
   range is 0.24–0.47 / 0.19–0.34 / 0.24–0.46, and C gives 0.29 / 0.23 / 0.28. Seed 42 is
   at the maximum for equal_width.
4. **FUS (`fus_entry_gain_scale`):** S_T ≤ 5.4e-6 in every A run, ≤ 2.3e-5 across all 15
   fixed seeds, and 1.9e-6 seed-averaged (corrected in task11; first reported as ≤ 2e-6).
   The "FUS is negligible" conclusion is unchanged.
5. **Prior-width artifact.** In every one of the 15 distinct fixed seeds and in C, narrowing
   a parameter's own prior lowers its S_T. `k_T4_entry`: S_T(equal)/S_T(as_doc) is
   at most 0.24, median 0.08, seed 42 0.04. `t4_dose_total`: at most 0.28, median
   0.22, seed 42 0.03 (15-seed values). **The direction holds, but seed 42 exaggerates the size:** for
   dose the typical collapse is to ~22% of the as-documented value, not ~3%. The
   literal three-way ordering from seed 42 does *not* hold in every seed (§5). The
   flips are between scenarios where that parameter's width is identical (e.g.
   tier1 vs equal for `k_T4_entry`), so they are seed noise, not counter-evidence.
6. **Noise dominates Var(Y).** In the CRN design (C, 7680 rows × 10 seeds), ~80% of
   Var(Y) is the seed main effect, only ~12% is parameter main effect, and ~5–8% is
   row×seed interaction. Var(Y) in B (independent noise per row) is ~10× the mean
   fixed-seed Var(Y) (and ~200× seed 42's). As a result, B's S_T are all 0.76–1.23
   (ΣS_T ≈ 13), including the six parameters that are exactly 0 under fixed seeds:
   per-row noise enters every Saltelli difference, so B's indices measure noise and
   not parameter importance. The production Sobol indices are therefore conditional
   on a single noise path. They decompose the ~12–17% of output variance that the
   parameters explain within that path.

## Files

- `task10_seed_sensitivity.py`: driver (`RUNSET` ∈ A/B/C/all, `N`, `OUTDIR`).
- `task10_seed_sensitivity.slurm`: job script with hash checks.
- `task10_summary.py`: summary analysis → `seed_sensitivity_summary_N256.json`, `SUMMARY_N256.md`.
- `N256/`: production outputs.
  - `X_{scenario}.npy`: the shared Sobol sample.
  - `{run}_{scenario}.json`: S_T/S1 with percentile CIs, S2 pairs, Y summary, run_meta.
  - `Y_{run}_{scenario}.npy`: raw Y.
  - `Y_C_..._perseed.npy`: run C's 7680×10 per-seed matrix.
- `smoke_N8/` + `SUMMARY_smoke_N8.md`/`.json`: N=8 smoke test of all run types (pipeline check only).
- `logs/`: SLURM stdout/stderr.

## Jobs and cost

| job | run | node | wall | CPU-time (128 cores allocated) |
|---|---|---|---|---|
| 3499618 | smoke, N=8, development | c309-006 | 0:55 | 1:57:20 |
| 3499626 | A (6 seeds) | c312-193 | 5:25 | 11:33:20 |
| 3499627 | B | c312-201 | 0:58 | 2:03:44 |
| 3499628 | C (10-seed mean) | c312-201 | 7:58 | 16:59:44 |

Total: 32.6 allocated CPU-hours (30.6 for N=256), 0.25 node-hours. The N=256 jobs ran
concurrently, 13:27:41–13:37:10 CDT (9 m 29 s span). All exited 0 with empty stderr.
No NaN or non-finite Y in any run, and no Y at the numerical floor (gm2_brain clipped at 0).

## task11: manuscript packaging (no model runs)

`task11_manuscript_package.py` recomputes everything from the saved Y arrays in `N256/`
and the shared X sample. Each recompute matches the task10 JSON and `percentile_cis.json`
exactly (|ΔS_T| = 0).

- `seed_averaged_percentile_cis.json`: run-C (10-seed CRN mean) S_T with 95% percentile CIs
  (B=1000, bootstrap seed 8162026). It uses the record schema of `percentile_cis.json` (same
  keys and types) plus S1/S2 keys; `source` stays `"rerun"`. **One type change:**
  `run_meta.model_eval_seed` is a list `[0..9]` rather than an int.
- `fixed_seed_table.csv`: 16 rows per scenario (6 A seeds + 10 C columns). Columns
  `duplicate_seed_value` / `in_distinct_set` mark seed 7's duplicate C column.
- `fixed_seed_param_summary.csv`: per-parameter seed-42 value, median, min, max, IQR and
  seed-42 rank across the 15 distinct seeds, plus the seed-averaged value.
- `manuscript_numbers.json`: items (a)–(g), the step-4 half-split check, and all sanity checks.
- `seed42_dependency_audit.md`: read-only audit of other single-seed analyses.

**Caveats for the manuscript (from `manuscript_numbers.json`)**
- The seed-averaged bootstrap CIs resample Sobol rows only, so they **don't include
  noise-seed uncertainty**. Averages over 5 seeds (seeds 0–4 vs 5–9) fall outside the 10-seed
  CI for 3–5 non-negligible params per scenario. The largest gap is `t4_dose_total`
  (0.029 vs 0.143 as_documented).
- The top five aren't stable between seed halves in as_documented and tier1_promotion:
  `t4_dose_total` replaces `k_T4_payload` in seeds 5–9. Ranks 1–2 (`gm2_synth`, `vmax_brain`)
  are stable. Ranks 3–5 sit within about 0.01 of each other in the 10-seed average and
  should be reported as unresolved, or more seeds should be run.
