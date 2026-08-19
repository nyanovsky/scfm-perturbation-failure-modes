"""scfm_pfm — Nadig Perturb-seq effect matrices and single-cell FM response contracts.

Run 1 of the feature-swap thread. This package owns two artifacts that every
later run reads through a *file* contract (never a shared Python process):

* ``effects/<cell_line>/`` — per-perturbation effect matrices, QC columns,
  effect-size strata and replicate-split noise ceilings.
* ``gene_universe/`` + ``models/<model>/`` — the harmonized gene universe and
  the per-model response contract (input normalization, output head,
  differentiability, symmetry) with a demonstration response vector and
  Jacobian column.

Later runs add a per-model feature extractor implementing
``scfm_pfm.models.base.ResponseModel`` against the same files; they do not
modify this run's code.
"""

__version__ = "1.0.0"
