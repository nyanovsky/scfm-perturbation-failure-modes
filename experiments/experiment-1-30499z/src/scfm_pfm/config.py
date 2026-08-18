"""Every analysis constant this run commits to, in one auditable place.

These values are the deliverable, not an implementation detail: published
processing of GSE264667 disagrees on surviving perturbation counts by ~3x,
so the thresholds and the counts they produce are reported explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SEED = 42

#: Non-targeting control label in ``obs['gene']``.
CONTROL_LABEL = "non-targeting"


@dataclass(frozen=True)
class CellFilters:
    """Per-cell quality criteria, applied and reported in this order."""

    #: cell must carry exactly one target assignment (obs.gene / obs.sgID_AB present)
    require_single_target: bool = True
    #: total UMI over the measured gene matrix
    min_counts: int = 2000
    #: number of genes with >= 1 count
    min_genes_detected: int = 500
    #: mitochondrial fraction (obs.mitopercent, already a fraction in [0, 1])
    max_mito_fraction: float = 0.15


@dataclass(frozen=True)
class PerturbationFilters:
    #: retained perturbations need at least this many QC-passing cells
    min_cells: int = 25
    #: ... spread over at least this many sequencing batches (gem_group)
    min_batches: int = 2


@dataclass(frozen=True)
class EffectConfig:
    #: counts-per-10k normalization target
    norm_target: float = 1e4
    #: pseudocount on the CP10K scale for log2 fold change
    lfc_eps: float = 0.01
    #: a gene enters magnitude / ceiling / cross-check statistics only if its
    #: pooled control mean reaches this CP10K level
    expressed_min_cp10k: float = 0.5
    #: FDR threshold for the differential-expression gene count
    de_fdr: float = 0.05
    #: k for the top-k differentially expressed gene overlap metric
    topk: int = 50
    #: on-target knockdown below this is "not detected" -- the perturbation is
    #: flagged so that a later run does not score a model on a knockdown that
    #: did not happen
    knockdown_detect_min: float = 0.30
    #: number of effect-size strata (quantile bins of the L2 log2FC magnitude)
    n_strata: int = 4
    strata_labels: tuple[str, ...] = field(
        default_factory=lambda: ("S1_lowest", "S2", "S3", "S4_highest")
    )


CELL_FILTERS = CellFilters()
PERTURBATION_FILTERS = PerturbationFilters()
EFFECTS = EffectConfig()

#: gate for the external cross-check (see verify.py)
#: effect-size stratum labels, lowest to highest (mirrors EFFECTS.strata_labels)
STRATUM_NAMES = EFFECTS.strata_labels

CROSSCHECK_MIN_SPEARMAN = 0.70

CEILING_METRICS = ("pearson_lfc", "cosine_delta", "topk_overlap")
CEILING_SPLITS = ("split_half_cells", "split_by_batch")
