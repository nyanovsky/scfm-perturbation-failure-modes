"""Every constant run 3 commits to, in one auditable place.

This file is the pre-registration. The scoring rule, the split rule, the
translator and the arm definitions are fixed here BEFORE any arm is scored, so
that the choice among the eight numbers of the correlation grid cannot be made
after seeing which one flatters a conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass

SEED = 42

CELL_LINES = ("hepg2", "jurkat")
MODELS = ("scgpt", "aido_cell", "scprint", "transcriptformer")

#: run 1 / run 2 artifact roots (read through presigned links; see fetch.py)
RUN1 = "exp_01m09d588jez38j6trqk30499z"
RUN2 = "exp_01m0c7ecnnf3fvxskxj5nygwcv"
TENANT = "fundacion-instituto-leloir-553557"

#: block prefix per (model, cell line) — from run 2's feature_index/blocks.json
BLOCK_PREFIX = {
    ("scgpt", "hepg2"): "features_full_scope/scgpt/hepg2",
    ("scgpt", "jurkat"): "features_full_scope/scgpt/jurkat",
    ("scprint", "hepg2"): "features_full_scope/scprint/hepg2",
    ("scprint", "jurkat"): "features_full_scope/scprint/jurkat",
    ("transcriptformer", "hepg2"): "features_full_scope/transcriptformer/hepg2",
    ("transcriptformer", "jurkat"): "features_full_scope/transcriptformer/jurkat",
    ("aido_cell", "hepg2"): "features/aido_cell/hepg2",
    ("aido_cell", "jurkat"): "features/aido_cell/jurkat",
}
#: blocks whose perturbed-gene rows are a strict subset of the in-scope set, so
#: the row keys must be read from the block instead of recomputed
SUBSET_BLOCKS = (("aido_cell", "hepg2"), ("aido_cell", "jurkat"))

# --------------------------------------------------------------------------- #
# what is measured
# --------------------------------------------------------------------------- #
#: run 1's primary effect estimator (batch-matched control), log2 fold change
EFFECT_KEY = "lfc_within_batch"
#: run 2's in-scope strata; a perturbed gene must also be a read-out gene
PERTURBED_STRATA = ("S3", "S4_highest")

#: the anchor every primary arm is read at. mean_control asks the model about
#: the same cell line the measured effect comes from, which is the query a
#: cell-line-specific co-expression baseline has to be compared against.
PRIMARY_ANCHOR = "mean_control"
#: anchors the sensitivity step reads, HepG2 only
SENSITIVITY_ANCHORS = ("uniform_virtual", "individual_cells", "mean_control_masked")
SENSITIVITY_CELL_LINE = "hepg2"
#: the feature form every model is read in (run 2: raw scales differ 1,900-fold)
FEATURE_FORM = "standardized"
VALUE_GRID_CP10K = (0.0, 1.0, 4.0, 16.0, 64.0)

#: forms a no-op query can touch. A no-op is a value substitution that changed
#: nothing because the gene already carried that value, so it makes the value
#: response an exact zero; a gradient at the same base value is a derivative
#: rather than a substitution and is unaffected.
NO_OP_AFFECTED_FORMS = ("vvp", "zero_out")

#: TranscriptFormer's gradient magnitudes were measured unstable in run 2, so
#: its GDT features enter as signed rank percentile within a query only
RANK_ONLY_GDT_MODELS = ("transcriptformer",)

# --------------------------------------------------------------------------- #
# co-expression side
# --------------------------------------------------------------------------- #
#: control cells are split in two halves; the co-expression half is disjoint
#: from the half reported as the leakage check
COEXPR_HALF = 1
COEXPR_CHECK_HALF = 0
#: bins for the mutual-information estimate over control cells
MI_BINS = 8
#: ridge shrinkage on the gene-gene correlation matrix before inversion, for the
#: partial correlation (2,721 genes from ~2,458 cells is not full rank)
PARTIAL_CORR_SHRINKAGE = 0.10
#: "strong" co-expression partner threshold for the node-degree feature
DEGREE_ABS_CORR = 0.30

# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
N_FOLDS = 5
SPLIT_TYPES = ("grouped", "random")
#: perturbed genes whose measured effect vectors correlate above this join into
#: one group, and whole groups are held out together (average linkage on
#: 1 - Pearson over the read-out set)
GROUP_CORR_THRESHOLD = 0.50

#: the two perturbation scopes every arm is run on. "full" is each block's own
#: coverage; "common" is the perturbed genes every model covers (AIDO.Cell's
#: 250), which is the only scope where a four-way model ranking is matched.
SCOPES = ("full", "common")

# --------------------------------------------------------------------------- #
# the translator — identical across every arm
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MLPConfig:
    """UGRN's projector: two hidden layers 128 / 64, ReLU, scalar output.

    UGRN trained with Adam at 1e-3, batch 128, 50 epochs on a balanced pair set
    of tens of thousands of pairs. Here one training fold holds ~1.1 M pairs, so
    the batch is scaled up and the epoch count down to keep the number of
    optimizer steps in the same range; the shape, optimizer and learning rate
    are UGRN's. Whatever is set here applies to every arm without exception.
    """

    hidden: tuple[int, ...] = (128, 64)
    lr: float = 1e-3
    batch_size: int = 4096
    epochs: int = 30
    weight_decay: float = 0.0
    output: str = "linear"
    loss: str = "weighted_mse"


MLP = MLPConfig()
#: ridge run beside the MLP as a capacity control; alpha by GCV on the training
#: fold from this fixed grid
RIDGE_ALPHAS = tuple(float(10.0**k) for k in range(-2, 8))

# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
#: the pre-registered primary cell of the correlation grid
PRIMARY_METRIC = "residual__weighted__disattenuated"
GRID_TARGETS = ("residual", "centered", "raw")
GRID_WEIGHTINGS = ("weighted", "unweighted")
GRID_CORRECTIONS = ("disattenuated", "attenuated")

#: how the destination line's mean response profile and reliability weights are
#: estimated in the cross-cell-line direction. ``dest_reference`` holds out half
#: the destination perturbations to estimate them and scores the other half;
#: ``source_only`` takes both from the training line and touches no destination
#: outcome; ``dest_all`` is the transductive convention, kept for comparison.
TRANSFER_MODES = ("dest_reference", "source_only", "dest_all")
TRANSFER_SCOPES = ("full", "common")

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_CI = 0.95
#: per-gene reliability below this contributes no weight (clip, not threshold)
RELIABILITY_FLOOR = 0.0
#: paired-difference correlations assumed when projecting the minimum
#: resolvable difference before any arm has been scored
POWER_PAIRED_RHO = (0.5, 0.8)
POWER_ALPHA = 0.05
POWER_TARGET = 0.80


@dataclass(frozen=True)
class ArmSpec:
    """One feature family. ``blocks`` names what is concatenated."""

    name: str
    family: str
    blocks: tuple[str, ...]
    model: str | None = None
    anchor: str | None = None
    note: str = ""
    #: a second model's gene table, concatenated onto the first through the
    #: ``emb2`` block. Only the two-table arm uses it.
    second_model: str | None = None
    #: per-gene width of the ``randgene`` null, so it can be matched to the
    #: embedding arm it is the null for (feature dim is twice this)
    rand_dim: int = 256
    #: the two genes' features go through separate copies of the translator and
    #: are summed, so the arm cannot represent anything pairwise. This is what
    #: makes the node-only control a control: it answers "how far do the two
    #: genes' separate properties get you", not "how far does a free function of
    #: both gene's properties get you".
    additive: bool = False


#: feature blocks an arm can concatenate:
#:   resp   — zero_out (1) + vvp (5) + gdt (5) for this model/anchor
#:   emb    — this model's gene embedding for i and for j
#:   emb2   — a second model's gene embedding for i and for j
#:   coexp  — Pearson, Spearman, MI, partial correlation over control cells
#:   node   — per-gene control-cell properties of i and of j, no pair term
#:   randpair / randgene — matched-dimension random projections
FEATURE_BLOCKS = ("resp", "emb", "emb2", "coexp", "node", "randpair", "randgene")

#: the one two-table arm: scGPT's learned gene table concatenated with the raw
#: ESM2-3B protein table. scGPT's table scores as well as the protein tables
#: while carrying no measurable protein-space structure, so either the two are
#: redundant routes to one quantity or they carry different information. The
#: arm goes through the identical translator, folds, scoring and nulls, and its
#: width-matched random projection is what separates a real gain from the extra
#: dimensions.
CONCAT_TABLES = ("scgpt", "esm2_3b")


def primary_arms(models: tuple[str, ...] = MODELS) -> list[ArmSpec]:
    """The arms every (cell line, split, scope) runs, at the primary anchor."""
    arms: list[ArmSpec] = [
        ArmSpec("coexp", "statistical", ("coexp",), note="co-expression pair statistics only"),
        ArmSpec("node", "node_only", ("node",), note="per-gene properties, no pair term"),
        ArmSpec("coexp_node", "statistical", ("coexp", "node")),
        ArmSpec("null_randpair", "null", ("randpair",)),
        ArmSpec("null_randgene", "null", ("randgene",)),
    ]
    for m in models:
        arms += [
            ArmSpec(f"{m}_resp", "scfm", ("resp",), model=m, anchor=PRIMARY_ANCHOR),
            ArmSpec(f"{m}_emb", "scfm", ("emb",), model=m, anchor=PRIMARY_ANCHOR),
            ArmSpec(f"{m}_full", "scfm", ("resp", "emb"), model=m, anchor=PRIMARY_ANCHOR),
            ArmSpec(
                f"{m}_hybrid",
                "hybrid",
                ("coexp", "node", "resp", "emb"),
                model=m,
                anchor=PRIMARY_ANCHOR,
            ),
        ]
    return arms


def sensitivity_arms(models: tuple[str, ...] = MODELS) -> list[ArmSpec]:
    """Anchor and form sensitivity: the same response feature, other queries."""
    arms: list[ArmSpec] = []
    for m in models:
        for anchor in SENSITIVITY_ANCHORS:
            if anchor == "mean_control_masked" and m == "transcriptformer":
                continue
            arms.append(
                ArmSpec(f"{m}_resp@{anchor}", "scfm_anchor", ("resp",), model=m, anchor=anchor)
            )
    return arms


#: shuffled-perturbation-label null: the arm's own features, targets permuted
#: across perturbed genes. Run on these arms only (it is a property of the
#: pipeline, not of the feature). The runner takes extra names on the command
#: line for an arm added after this list was fixed, so that adding a control to
#: a later job cannot rewrite what the main sweep ran.
SHUFFLE_NULL_ARMS = ("coexp", "scgpt_full")

#: models whose gene table comes from a protein language model with no
#: single-cell model anywhere in its path. They enter the sweep exactly like a
#: model embedding arm and are labelled ``protein`` so the report can keep them
#: apart from the single-cell arms.
PROTEIN_MODELS = ("esm2_650m", "esm2_3b")

#: how many independent permutations to draw for each shuffled-label arm. One
#: draw is a single sample from the null and says nothing about its spread.
N_SHUFFLE_DRAWS = 5


@dataclass(frozen=True)
class Discrimination:
    """Perturbation discrimination, the second scoring axis.

    Computed with Arc Institute's own ``cell-eval`` implementation at a pinned
    version rather than a reimplementation of the definition; the resolved
    version is recorded in the manifest. The metric is defined on effect
    profiles, so the predicted residual is put back on the effect scale by
    adding the training mean effect before scoring.
    """

    package: str = "cell-eval"
    version: str = "0.5.19"
    reconstitution: str = "prediction + training-fold mean effect"


DISCRIMINATION = Discrimination()
