# gm2_outputs

Archive supporting the manuscript "A prior-robust sensitivity and
factorial-necessity analysis of tri-modal viral-ultrasound gene delivery
for lysosomal neurodegeneration: uncertainty quantification and
cross-disease generalizability" (PLOS Computational Biology submission).

- `gm2_analysis/`: the simulator (`gm2_model_v2.py`), Sobol / factorial /
  Shapley drivers, all raw outputs behind the paper's tables and figures,
  parameter provenance, figure scripts and their inputs, and the
  acoustic-FUS robustness check. See its README.
- `task_outputs/`: model-verification tests, external-benchmark
  comparison, bootstrap percentile intervals, and the width-sensitivity
  correlation, with the scripts that produced them.

Integrity: the SHA-256 of `gm2_analysis/gm2_model_v2.py` begins
`6693427fa909a0a9`; every reported result was produced with that file.
All simulations use a Milstein step of 0.2 d.

Note: an earlier exploratory LIFU subproject (`lifu_simulation/`) shipped
in releases v1.0.0-v1.0.1. It was removed from the main branch because no
result in the paper depends on it; it remains in those earlier releases
and in git history.
