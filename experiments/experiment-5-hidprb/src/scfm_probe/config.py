"""Every constant run 5 commits to, written before any number was read.

The grids and thresholds here are the diagnostic's definition, not tuning knobs.
Each one is echoed verbatim into the block manifest, so a block carries the rule
it was produced under and two blocks made under different rules cannot be mixed
silently.

Three of them are load-bearing enough to say why in one place rather than four:

``VALUE_GRID_CP10K`` is run 2's grid, unchanged. The probe has no reason of its
own to prefer a value, and reusing the grid means a propagation number here sits
on the same x-axis as the response features run 3 scored.

``LAYER_STRIDE`` exists because a last-layer read alone cannot separate the two
hypotheses this run is between. Information that rises through the stack and
collapses at the head is *discarded*; flat-and-low at every depth is *absent*.
Only a depth profile distinguishes them, and layers are free --- one forward
pass, several hooks.

``N_ANCHOR_CELLS`` is 24 rather than run 2's 4 because the trained probe needs a
held-out axis. Run 2 averages its four cells' responses together, which is the
right call for a response feature and destroys exactly the variation a probe
trains on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: one seed, derived per purpose at each use, never set globally
SEED = 42

# --------------------------------------------------------------------------- #
# the intervention
# --------------------------------------------------------------------------- #

#: run 2's value grid, in CP10K, unchanged: the probe reads the same doses the
#: scored response features were read at
VALUE_GRID_CP10K = (0.0, 1.0, 4.0, 16.0, 64.0)

#: the dose the headline statistics are read at. Zero-out is the CRISPRi analog
#: and the only value whose meaning does not depend on the gene's own scale.
PRIMARY_VALUE_CP10K = 0.0

#: anchors this run queries. `uniform_virtual` is deliberately absent: it is off
#: the training distribution by construction and confounds the intervention with
#: the removal of the read-out gene's own value (run 2's report, section on the
#: uniform anchor). Add it only if `mean_control` comes back ambiguous.
ANCHORS = ("mean_control", "individual_cells")

#: the anchor every headline number is read at --- the same one run 3 scores
#: every primary arm at, so a null here speaks to the published result
PRIMARY_ANCHOR = "mean_control"

#: real non-targeting control cells for the individual-cell anchor, kept
#: unaveraged. HepG2 has 4,916 available, so this is a design choice and not a
#: supply limit.
N_ANCHOR_CELLS = 24

#: how many of those cells are held out for the trained probe. The split is over
#: cells, never over perturbations: the question is whether a signature learned
#: in one cell survives into a cell the probe never saw.
N_HELDOUT_CELLS = 8

# --------------------------------------------------------------------------- #
# the layer sweep
# --------------------------------------------------------------------------- #

#: layer `l` is the OUTPUT of transformer block `l`, 1-based. `l = 0` is the
#: post-embedding input to block 1 and is a control, not a sweep point: an edit
#: at gene i cannot have reached gene j's token yet, so the delta there must be
#: exactly zero and any other value means the hooks are mis-attributed.
EMBEDDING_LAYER = 0

#: every third block, plus the last one whatever the stride lands on
LAYER_STRIDE = 3

#: transformer blocks per checkpoint, read from the checkpoints themselves:
#: scGPT_human args.json `nlayers`, scPRINT medium-v1.5 hyper_parameters
#: `nlayers`, TranscriptFormer tf_sapiens config `model.model_config.num_layers`
N_LAYERS = {"scgpt": 12, "scprint": 8, "transcriptformer": 12}

#: hidden width per checkpoint, from the same three places (`embsize`,
#: `d_model`, `model_dim`). Recorded so a block's array shape can be checked
#: against the checkpoint rather than against itself.
D_MODEL = {"scgpt": 512, "scprint": 256, "transcriptformer": 2048}


def sweep_layers(model: str) -> tuple[int, ...]:
    """The blocks probed for one model: every `LAYER_STRIDE`th, and the last.

    The last layer is forced in regardless of where the stride lands, because it
    is the one the output head actually reads --- the layer whose contents the
    "present but not decodable" hypothesis is a claim about.
    """
    n = N_LAYERS[model]
    return tuple(sorted(set(range(1, n + 1, LAYER_STRIDE)) | {n}))


# --------------------------------------------------------------------------- #
# pooling and the retrieval score
# --------------------------------------------------------------------------- #

#: The perturbed gene's own read-out position is excluded from every headline
#: statistic. Decoding "which gene was perturbed" from the token whose input you
#: just changed is trivial and says nothing about propagation; it is kept only
#: as the positive control that proves the intervention landed at all.
EXCLUDE_SELF_POSITION = True

#: read-out positions are split into two disjoint halves, so a perturbation's
#: signature can be matched against itself across positions it was not built
#: from. This is what gives `mean_control` --- which supplies exactly one delta
#: per perturbation --- a held-out axis at all.
N_POSITION_HALVES = 2

#: seed for that split, derived from SEED so it is reproducible but independent
#: of the anchor-cell draw
HALF_SPLIT_SEED = SEED + 101

#: ranks reported for every retrieval statistic. Top-1 is the headline; top-10
#: is reported beside it because with 524 classes a top-1 miss and a rank-2 hit
#: are not the same result.
TOPK = (1, 10)

#: fresh shuffled-label draws per (model, cell line, layer). An arm counts only
#: if it clears its own null in its own cell --- never a band copied from
#: elsewhere in the thread.
N_NULL_DRAWS = 5

#: resamples for every interval. Bootstrap is over response groups, never over
#: perturbations: paralogs and complex members are not independent draws.
N_BOOTSTRAP = 2000

#: Worker processes `analysis.py` scores depth-units on. 22 of the box's 24 cores,
#: leaving two for the OS and the parquet writer. This is an execution knob and
#: NOT part of the pre-registration: every seed in `analysis.py` is derived from
#: the metric cell's own name, so the worker count cannot reach a number, only the
#: wall clock. `--jobs` overrides it.
N_ANALYSIS_WORKERS = 22

#: the linkage cut run 3 clusters measured effect vectors at, reused verbatim so
#: "same response group" means the same thing in both runs
GROUP_LINKAGE_THRESHOLD = 0.50

# --------------------------------------------------------------------------- #
# scope
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Scope:
    """How much of the in-scope perturbation set a probe block runs on.

    ``0`` means "every gene in scope", which is how full coverage is requested
    and what every production block uses: 524 perturbations in HepG2 and 705 in
    Jurkat. The smoke path sets a small number instead. Whatever is in force is
    recorded in the block manifest beside the size of the in-scope set, so two
    blocks at different coverage cannot be compared by accident.
    """

    n_perturbed: int = 0
    strata: tuple[str, ...] = field(default_factory=lambda: ("S3", "S4_highest"))


SCOPE = Scope()

#: the smoke path: enough genes to exercise every reassembly branch, few enough
#: to run in minutes on one 3090
SMOKE_SCOPE = Scope(n_perturbed=8)


# --------------------------------------------------------------------------- #
# tie handling, and the seeds derived for it
# --------------------------------------------------------------------------- #

#: seed for the random tie-break inside the retrieval ranking. Ties are not a
#: corner case here: a layer whose off-target delta is exactly zero makes EVERY
#: cosine distance equal, and the rule chosen for that case decides what a dead
#: layer reports. Breaking ties at random puts a fully degenerate block at
#: chance, which is the honest reading; breaking them by index would report 1.0
#: or 0.0 depending on which way the array happened to be built.
RETRIEVAL_TIE_SEED = SEED + 202

#: two cosine distances closer than this count as tied. Exact float equality
#: would let 1e-17 of accumulation order decide a rank, which is the difference
#: between "at chance" and "perfect" on a degenerate block.
RETRIEVAL_TIE_ATOL = 1e-12

#: which control cells the individual-cell anchor draws, independent of the
#: half-split and of the perturbation subsample so that changing one does not
#: move the others
ANCHOR_CELL_SEED = SEED + 303


# --------------------------------------------------------------------------- #
# the analysis pass: nulls, intervals, and the frozen shape of its output
# --------------------------------------------------------------------------- #

#: base seed for the shuffled-label draws. Every draw derives its own seed from
#: this one AND from the cell it is drawn in --- model, line, anchor, layer,
#: value, statistic, draw index --- so no two cells share a draw and no null is
#: ever a band copied from a neighbouring cell.
NULL_SEED = SEED + 404

#: base seed for the group bootstrap, separate from the null seed so that
#: changing the number of null draws does not move a single interval
BOOTSTRAP_SEED = SEED + 505

#: base seed for the train/test split over anchor cells in the held-out-cell
#: probe. Separate from ANCHOR_CELL_SEED, which decides WHICH cells were drawn:
#: which cells exist and which of them are held out are two decisions and must
#: not move together.
HELDOUT_CELL_SEED = SEED + 606

#: coverage of every interval reported, two-sided
BOOTSTRAP_CI = 0.95

#: a CKA denominator at or below this is treated as undefined rather than
#: divided by. It is reached exactly when one of the two representations is
#: constant across perturbations --- layer 0's identically-zero delta, for
#: instance --- where a "similarity" would be an artefact of float noise.
CKA_MIN_DENOM = 1e-12

#: the statistic vocabulary of results/probe/summary.parquet, frozen before the
#: figure module was written against it. `propagation_off_target_norm` exists
#: because TranscriptFormer has no denominator for the ratio (it right-shifts
#: counts, so a gene's own count is never an input at its own position) while
#: the numerator is perfectly well defined --- and the numerator is what its
#: depth profile has to be read from.
SUMMARY_STATISTICS = (
    "propagation_ratio",
    #: `propagation_ratio` divides a Frobenius norm over n_readout-1 positions by a
    #: norm at ONE position, so it grows as sqrt(n_readout): identical per-position
    #: propagation reads 9.3% higher in Jurkat (3247 read-out genes) than in HepG2
    #: (2721). The headline figure puts the two cell lines on shared axes, so the
    #: per-position form is emitted beside it and is the one to compare ACROSS
    #: LINES. The Frobenius form is kept because it is what the plan pre-registered
    #: and what the within-line depth profile was read from.
    "propagation_ratio_per_position",
    "propagation_off_target_norm",
    "retrieval_top1",
    "retrieval_top10",
    "retrieval_mrr",
    "cka_static_table",
    "norm_only_top1",
    "group_collapsed_top1",
)

#: which held-out axis a row's score was read across. `split_half` is the
#: primary (halves of the read-out positions), `heldout_cell` the trained
#: secondary, `self_position` the controls read at the perturbed gene's own
#: position.
SUMMARY_SPLITS = ("split_half", "heldout_cell", "self_position")

#: the column order of summary.parquet, frozen: a second module builds figures
#: against it and a reordered or renamed column is a silent break.
SUMMARY_COLUMNS = (
    "model",
    "cell_line",
    "anchor",
    "layer",
    "value_cp10k",
    "statistic",
    "split",
    "score",
    "lo",
    "hi",
    "null_max",
    "n_perturbations",
    "n_undefined",
    "detail",
)

# --------------------------------------------------------------------------- #
# checkpoint-layout bounds
# --------------------------------------------------------------------------- #

#: leading auxiliary (assay/species) tokens a model may prepend to the gene
#: sequence. TranscriptFormer prepends exactly one and recovers the genes with
#: `transformer_output[:, num_aux_tokens:, :]`; the bound exists so a layout
#: change raises instead of silently slicing real genes off the front.
MAX_AUX_TOKENS = 4

#: the lowest input value any adapter is asked for, in the model's own native
#: units. None of the four models takes a negative input, so a perturbed value is
#: clamped here rather than passed through --- an out-of-range input would be
#: answered, not refused, and the answer would look like a measurement.
NATIVE_FLOOR = 0.0

#: input rows run 2 spends per (perturbed gene, anchor): one untouched anchor
#: plus a (v-step, v, v+step) triple at each of five grid values, for its
#: finite-difference derivatives. Recorded so a block states the ratio it saved
#: rather than a reader having to recompute it.
RUN2_INPUT_ROWS_PER_GENE = 16

#: Statistics whose verdict is read against a shuffled-label null. These are the
#: DECISION statistics: each answers "is the true perturbation closer than the
#: others", so permuting the labels is a meaningful null and a score without its
#: band is not a result.
#:
#: The S1 magnitudes are deliberately absent. `propagation_ratio` and
#: `propagation_off_target_norm` are descriptive quantities --- how far the edit
#: travelled, in the model's own units --- and permuting perturbation labels does
#: not change a norm, so a shuffled-label draw there would be the statistic itself
#: wearing a null's name. They are reported with their group bootstrap and no
#: band, and nothing is decided from them alone.
NULL_BEARING_STATISTICS = (
    "retrieval_top1",
    "retrieval_top10",
    "retrieval_mrr",
    "cka_static_table",
    "norm_only_top1",
    "group_collapsed_top1",
)
