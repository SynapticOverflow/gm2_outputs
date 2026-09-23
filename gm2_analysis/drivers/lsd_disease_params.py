# lsd_disease_params.py
#
# Cross-disease parameterization for the 9 lysosomal storage disorders
# (LSDs) named in the MICCAI draft's cross-disease generalization claim:
# GM2 (Tay-Sachs/Sandhoff), Fabry, GM1, Pompe, MPS I, MLD, CLN2, Krabbe,
# Niemann-Pick C.
#
# WHAT IS REAL vs. CONSTRUCTED HERE (read before using these numbers for
# any claim):
#
#   REAL, citation-backed: each disease's causative enzyme, and its
#   approximate untreated natural-history timescale (age of onset,
#   median/typical survival or life expectancy without treatment), taken
#   from the published natural-history literature (sources listed per
#   disease below, gathered via live web search).
#
#   CONSTRUCTED, not literature-derived: none of these papers report
#   compartmental-SDE rate constants (a "km_brain", "gm2_synth", or
#   "vmax_brain" in the sense this toy model uses them) -- that
#   granularity of kinetic parameter simply isn't published for most of
#   these diseases. The per-disease synthesis/clearance parameters below
#   are derived from a single heuristic: scale the GM2/Tay-Sachs
#   baseline's synthesis-vs-clearance balance by the ratio of reference
#   timescales (T_ref_tay_sachs / T_disease), where T_disease is the
#   real, citation-backed untreated survival timescale. This reproduces
#   the right ORDER of disease severity/urgency, not validated
#   disease-specific kinetics. Treat any per-disease absolute output
#   number from this model as illustrative, not a clinical or
#   publication-ready claim.

# Real anchors (citations inline). Timescales in days.
DISEASE_FACTS = {
    'tay-sachs': {
        'enzyme': 'beta-hexosaminidase A (HEXA)',
        't_ref_days': 730,  # ~2y untreated survival, existing model anchor
        'source': 'Bley et al. 2011, Pediatrics (existing model anchor)',
    },
    'sandhoff': {
        'enzyme': 'beta-hexosaminidase A+B (HEXA+HEXB)',
        't_ref_days': 730,
        'source': 'Same natural-history class as Tay-Sachs (existing model anchor)',
    },
    'pompe': {
        'enzyme': 'acid alpha-glucosidase (GAA)',
        't_ref_days': 300,  # ~25.7% survival at 12mo, ~12.3% at 18mo untreated -> median well under 1y
        'source': 'Kishnani et al., retrospective infantile-onset Pompe natural history (ScienceDirect); PMC8518093',
    },
    'krabbe_infantile': {
        'enzyme': 'galactocerebrosidase (GALC)',
        't_ref_days': 730,  # median survival ~2 years, early-onset (0-5mo) infantile form
        'source': 'PMC6378723 (early progression of Krabbe disease, onset 0-5mo); Genet Med 2019 (PMID 41436-019-0480-7)',
    },
    'gm1_infantile': {
        'enzyme': 'beta-galactosidase (GLB1)',
        't_ref_days': 575,  # mean age of death 18.9 months
        'source': 'PubMed 37381921, natural history of GM1 gangliosidosis, French cohort',
    },
    'mps1_hurler': {
        'enzyme': 'alpha-L-iduronidase (IDUA)',
        't_ref_days': 3285,  # mortality within first decade untreated; ~9y midpoint
        'source': 'PMC7463646 (MPS I natural history and molecular pathology review)',
    },
    'mld_late_infantile': {
        'enzyme': 'arylsulfatase A (ARSA)',
        't_ref_days': 1825,  # onset ~1.5y + rapid decline; ~5y total illustrative course
        'source': 'PMC6489348 (natural history of MLD, caregiver interviews)',
    },
    'cln2': {
        'enzyme': 'tripeptidyl peptidase 1 (TPP1)',
        't_ref_days': 3300,  # onset ~2-4y, blind/non-ambulatory 6-8y, childhood death ~9y
        'source': 'PMC7127909 (Changing Times for CLN2 Disease: enzyme replacement therapy era)',
    },
    'niemann_pick_c': {
        'enzyme': 'NPC1 cholesterol transporter (95% of cases)',
        't_ref_days': 4380,  # classic childhood-onset life expectancy ~12y
        'source': 'PMC4678528 (UK NPC natural history cohort, 5-year update)',
    },
    'fabry': {
        'enzyme': 'alpha-galactosidase A (GLA)',
        't_ref_days': 16425,  # classic phenotype, death from renal/cardiac/cerebrovascular disease typically ~4th-5th decade
        'source': 'NCBI Bookshelf NBK435996 (Fabry Disease, StatPearls); PMC7918333',
        'note': 'Substrate storage requires enzyme activity below ~10-15% of normal (a real reported '
                'threshold) -- unlike the other 8 (near-total deficiency, infantile), classic Fabry is '
                'adult-onset and vastly slower; included for contrast, not as a like-for-like comparison.',
    },
}


def make_lsd_params(disease: str, base_params_fn, reference_disease: str = 'tay-sachs'):
    """
    Scale a reference disease's make_base_params() output to a target LSD
    using the timescale-ratio heuristic described above. base_params_fn
    is gm2_model_v2.make_base_params (kept as a parameter to avoid a
    circular import).
    """
    ref = base_params_fn(reference_disease)
    facts = DISEASE_FACTS[disease]
    t_ref = DISEASE_FACTS[reference_disease]['t_ref_days']
    t_dis = facts['t_ref_days']
    scale = t_ref / t_dis  # >1 for more aggressive disease, <1 for slower (e.g. Fabry)

    out = dict(ref)
    out['name'] = disease
    out['gm2_synth'] = ref['gm2_synth'] * scale
    out['vmax_brain'] = ref['vmax_brain'] / max(scale, 1e-6)
    out['vmax_liver'] = ref['vmax_liver'] / max(scale, 1e-6)
    out['inf_threshold'] = ref['inf_threshold'] / max(scale, 1e-6) ** 0.5  # milder scaling; threshold is a burden level, not a rate
    out['baseline_gm2_brain'] = ref['baseline_gm2_brain']  # same starting point; severity plays out via kinetics
    out['baseline_gm2_liver'] = ref['baseline_gm2_liver']
    out['_disease_facts'] = facts
    out['_timescale_ratio_vs_tay_sachs'] = scale
    return out
