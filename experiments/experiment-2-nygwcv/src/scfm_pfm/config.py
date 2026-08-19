"""Every constant run 2 commits to, in one auditable place.

The grids here are the deliverable's definition, not tuning knobs: a feature
block is only interpretable next to the value grid, baseline grid, expression
floor and step that produced it, so each one is recorded in the block's manifest
exactly as written here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SEED = 42

CELL_LINES = ("hepg2", "jurkat")
MODELS = ("scgpt", "aido_cell", "scprint", "transcriptformer")

#: run 1's control label and cell filters, reused verbatim by the demo-cell export
CONTROL_LABEL = "non-targeting"


@dataclass(frozen=True)
class CellFilters:
    require_single_target: bool = True
    min_counts: int = 2000
    min_genes_detected: int = 500
    max_mito_fraction: float = 0.15


CELL_FILTERS = CellFilters()

# --------------------------------------------------------------------------- #
# step 1 — the expression floor
# --------------------------------------------------------------------------- #
#: run 1's floor, and the higher floors the sensitivity sweep re-derives the
#: replicate-split ceilings at. Every higher floor is a subset of the genes the
#: persisted split-half vectors already carry, so this needs no raw data.
FLOOR_GRID_CP10K = (0.5, 1.0, 2.0, 5.0, 10.0)
RUN1_FLOOR_CP10K = 0.5

#: "the ceiling has stopped rising materially" — the read-out floor is the
#: LOWEST floor whose top-stratum median Pearson ceiling is within this much of
#: the best ceiling any higher floor reaches, in both split forms.
FLOOR_PLATEAU_TOLERANCE = 0.01
#: the stratum whose ceiling the rule reads
FLOOR_RULE_STRATUM = "S4_highest"
#: strata whose perturbations are in scope as perturbed genes
PERTURBED_STRATA = ("S3", "S4_highest")

# --------------------------------------------------------------------------- #
# step 3/4 — anchors, value grids, feature forms
# --------------------------------------------------------------------------- #
ANCHORS = ("uniform_virtual", "mean_control", "individual_cells")
FORMS = ("zero_out", "vvp", "gdt", "gene_embedding")
NORMALIZATIONS = ("raw", "standardized", "recentred", "log2fc")

#: Every anchor and every perturbation value is expressed in CP10K, the one unit
#: all four models can be given a defensible map into: counts-scale models take
#: the value as the count of a 10,000-count virtual cell, and scGPT takes the
#: matching quantile bin (see features/anchors.py).
VALUE_GRID_CP10K = (0.0, 1.0, 4.0, 16.0, 64.0)
#: gradient-trajectory base values: the same grid, so one forward pass at each
#: value serves both the interval feature and the centre of its derivative
GDT_BASE_GRID_CP10K = VALUE_GRID_CP10K

#: uniform-virtual anchor: every supplied gene sits at the same share of a
#: 10,000-count cell, i.e. 1e4 / (number of supplied genes) CP10K
UNIFORM_ANCHOR_TOTAL_CP10K = 1.0e4

#: number of real control cells averaged for the individual-cell anchor
N_INDIVIDUAL_CELLS = 4

# --------------------------------------------------------------------------- #
# step 5 — finite-difference step selection
# --------------------------------------------------------------------------- #
#: candidate steps in each model's OWN input units (CP10K counts for the three
#: counts-scale models, bin index for scGPT), swept on a fixed subset
STEP_GRID = {
    "scgpt": (0.02, 0.05, 0.1, 0.25, 0.5, 1.0),
    "aido_cell": (0.02, 0.05, 0.1, 0.25, 0.5, 1.0),
    "scprint": (0.02, 0.05, 0.1, 0.25, 0.5, 1.0),
    "transcriptformer": (0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
}
#: adjacent-step agreement a step must reach to be called stable in magnitude
STEP_STABLE_PEARSON = 0.999
STEP_STABLE_RELATIVE_L2 = 0.10
#: perturbed genes the step sweep runs on
N_STEP_SWEEP_GENES = 50

# --------------------------------------------------------------------------- #
# step 3 — masking regime
# --------------------------------------------------------------------------- #
#: fallback mask rate when a checkpoint does not record its own
DEFAULT_MASK_RATE = 0.15
#: how many times every read-out gene must be read masked
MASK_MIN_COVERAGE = 3

# --------------------------------------------------------------------------- #
# step 6 — normalization
# --------------------------------------------------------------------------- #
#: pseudocount for the predicted log fold-change bridge, on the model's own
#: output scale after it has been mapped to a linear expression scale
LOG2FC_EPS = 1e-6

# --------------------------------------------------------------------------- #
# step 8 — wiring checks
# --------------------------------------------------------------------------- #
N_WIRING_GENES = 40
#: fraction of sampled perturbed genes whose own predicted output must rise when
#: their input is raised
SELF_RESPONSE_MIN_PASS = 0.95

# --------------------------------------------------------------------------- #
# scale of the sweep (set from measured throughput; see README)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Scope:
    """How many perturbed genes each anchor runs on.

    ``n_perturbed`` is filled from the measured per-forward cost of the slowest
    model rather than guessed; the value in force for a block is recorded in its
    manifest.
    """

    n_perturbed: int = 250
    n_perturbed_cells_anchor: int = 60
    n_perturbed_masked: int = 60
    strata: tuple[str, ...] = field(default_factory=lambda: PERTURBED_STRATA)


SCOPE = Scope()
