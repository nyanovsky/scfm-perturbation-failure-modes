"""Every constant run 6 commits to, written before any arm is scored.

This file is the pre-registration. Run 6's whole claim rests on scoring a new
*input* through run 3's unchanged rule, so the only things run 6 may choose are
what the input is and how it is reduced --- and those choices are all here,
echoed verbatim into every block manifest and into ``qc.json``. Nothing in
``basis/``, ``extract/``, ``install.py`` or ``nulls.py`` carries a magic number.

Three groups of constant, and why each is fixed rather than swept:

**The intervention** is run 2's, unchanged: the same value grid, the same
per-model finite-difference step, the same anchor. A hidden-state feature read
at a different dose grid would not sit on the same x-axis as the head features
runs 2--4 published, and the whole point of run 6 is that the comparison is
possible at all.

**The reduction** (``K_STORE``, ``K_PRIMARY``, ``K_LADDER``, ``N_BASIS_GENES``)
is the one genuinely new degree of freedom. It is fixed here, with the ladder
declared as a column prefix of the stored block so that reading a second ``k``
costs nothing and cannot be a decision made after seeing a score.

**The scoring cells.** ``PRIMARY_METRIC`` is run 3's pre-registered cell,
unchanged and unchosen. ``ORDERING_METRIC`` is run 3's post-hoc secondary, kept
in the same subordinate role (``docs/INDEX.md`` s5 --- these two roles reversed
at cp-23, and text written before then has them backwards). ``EXTRA_METRICS``
adds weighted cosine and the L1 perturbation-discrimination score as further
cells; they are declared here, before any arm is scored, which is what keeps
them pre-registered secondaries rather than a metric chosen for its answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: one seed, derived per purpose at each use, never set globally
SEED = 42

CELL_LINES = ("hepg2", "jurkat")

# --------------------------------------------------------------------------- #
# the models
# --------------------------------------------------------------------------- #
#: the three frozen models run 5 already probes, plus run 4's two fine-tuned
#: scGPT checkpoints. AIDO.Cell is deliberately absent: it is not on local disk,
#: it scored at zero in run 3, and it costs ~40x scGPT per forward pass.
FROZEN_MODELS = ("scgpt", "scprint", "transcriptformer")
FT_MODELS = ("scgpt_ft", "scgpt_ft_frozen")
MODELS = FROZEN_MODELS + FT_MODELS

#: which run-4 arm each fine-tuned key reads. ``own_trainable`` is the standard
#: published recipe; ``own_frozen`` holds scGPT's own gene table fixed and is
#: the control that splits any gain between the readout and the table.
FT_ARM_OF = {"scgpt_ft": "own_trainable", "scgpt_ft_frozen": "own_frozen"}

#: run 4 ran its sixteen arms twice. ``_pearson`` is the reported execution
#: (highest validation Pearson, as scGPT's tutorial selects); unsuffixed is the
#: earlier masked-MSE execution. Leaving this unset silently reads the wrong
#: tree, so run 6 records it verbatim and a named check fails if it is empty.
FT_RUN_SUFFIX = "_pearson"

#: transformer blocks per checkpoint, and hidden width, from run 5's config.
#: The two fine-tuned keys are scGPT's architecture with a substituted gene
#: table, so they carry scGPT's depth and width.
N_LAYERS = {
    "scgpt": 12, "scprint": 8, "transcriptformer": 12,
    "scgpt_ft": 12, "scgpt_ft_frozen": 12,
}
D_MODEL = {
    "scgpt": 512, "scprint": 256, "transcriptformer": 2048,
    "scgpt_ft": 512, "scgpt_ft_frozen": 512,
}

#: the conda environment each model's extraction job runs in. scPRINT's is the
#: one built on this machine on 2026-08-29; see run 5's ``envs/README.md`` for
#: the import trap (``import scprint`` drags in lamindb; run 2 imports
#: ``scprint.model.model`` alone).
MODEL_ENV = {
    "scgpt": "scgpt",
    "scprint": "scprint_env",
    "transcriptformer": "transcriptformer_env",
    "scgpt_ft": "scgpt",
    "scgpt_ft_frozen": "scgpt",
}

# --------------------------------------------------------------------------- #
# the intervention --- run 2's, unchanged
# --------------------------------------------------------------------------- #
#: run 2's value grid in CP10K. Read at the same doses the scored response
#: features were read at, so a hidden-state number sits on the published x-axis.
VALUE_GRID_CP10K = (0.0, 1.0, 4.0, 16.0, 64.0)

#: the anchor every scored arm in the paper was read at. ``individual_cells``
#: would be ~16x the cost and would not speak to the published numbers.
ANCHOR = "mean_control"

#: the central-difference step for ``gdt``, in each model's OWN input units
#: (quantile bin index for scGPT, CP10K counts for the others). These are the
#: steps run 2's step sweep chose and its shipped ``mean_control`` blocks were
#: built at, read out of ``results/report/bundles_full_scope/*.json``. Reusing
#: them is what makes run 6's ``gdt`` the same derivative estimator as run 2's,
#: differing only in what is differentiated.
GDT_STEP = {
    "scgpt": 0.02,
    "scprint": 0.02,
    "transcriptformer": 1.0,
    "scgpt_ft": 0.02,
    "scgpt_ft_frozen": 0.02,
}

#: run 2 used forward-mode autodiff for scPRINT's ``gdt`` and a central
#: difference for the rest. Run 6 uses a central difference for EVERY model.
#: Recorded divergence: the hidden-state JVP is available in principle for
#: scPRINT alone, and using it there only would make one model's ``gdt`` a
#: different estimator from the others' --- precisely the confound a cross-model
#: comparison cannot carry.
GDT_ESTIMATOR = "central_finite_difference"

#: input rows per (gene, anchor): 1 baseline + 3 per grid value. Run 2's number,
#: not run 5's 6 --- run 5 computes no derivative, run 6 does.
INPUT_ROWS_PER_GENE = 1 + 3 * len(VALUE_GRID_CP10K)

#: native input values are floored here before being written into a query row,
#: matching run 2's ``features/extract.py:NATIVE_FLOOR``
NATIVE_FLOOR = 0.0

# --------------------------------------------------------------------------- #
# the layer sweep
# --------------------------------------------------------------------------- #
#: layer ``l`` is the OUTPUT of transformer block ``l``, 1-based. Layer 0 is the
#: post-embedding input to block 1 and is control **C2**, not a sweep point: an
#: edit at gene i cannot have reached gene j's token before the first block, so
#: the off-target delta there must be exactly 0.0 and anything else means the
#: hooks are mis-attributed.
EMBEDDING_LAYER = 0

#: run 5's stride: every third block, plus the last whatever the stride lands on
LAYER_STRIDE = 3


def sweep_layers(model: str) -> tuple[int, ...]:
    """The blocks one model's extraction captures, run 5's rule reused verbatim.

    The last layer is forced in regardless of where the stride lands: it is the
    layer the decoder head actually reads, and therefore the one the "present
    but not decodable" hypothesis is a claim about.
    """
    n = N_LAYERS[model]
    return tuple(sorted(set(range(1, n + 1, LAYER_STRIDE)) | {n}))


def capture_layers(model: str) -> tuple[int, ...]:
    """Every layer a forward pass hooks: the control layer plus the sweep.

    Layer 0 leads the list because it is the cheapest possible check on hook
    attribution, and because a block whose layer 0 is not exactly zero
    off-target is not a null result --- it is unwired.
    """
    return (EMBEDDING_LAYER,) + sweep_layers(model)


def primary_layer(model: str) -> int:
    """The layer the pre-registered primary cell is read at: the last block."""
    return N_LAYERS[model]


# --------------------------------------------------------------------------- #
# the slots --- what a hidden response feature IS
# --------------------------------------------------------------------------- #
#: Run 2's ``resp`` block is 11 scalars per pair: zero_out (1) + vvp (5) +
#: gdt (5). Run 6 keeps vvp and gdt and DROPS zero_out, because with no
#: normalisation applied the hidden zero_out slot would be bit-identical to the
#: vvp dose-0 slot. Recorded here rather than silently omitted, and asserted by
#: ``tests/test_slots.py``.
SLOT_FORMS = ("vvp", "gdt")
N_SLOTS = len(SLOT_FORMS) * len(VALUE_GRID_CP10K)
DROPPED_RUN2_SLOT = "zero_out"
DROPPED_RUN2_SLOT_REASON = (
    "with no per-query normalisation applied, the hidden-state zero_out slot is "
    "bit-identical to the vvp slot at dose 0.0 CP10K; keeping both would put one "
    "column in the feature vector twice"
)

#: the axis-1 order of the installed block: C-order over (dose, pc) within each
#: of ``vvp`` and ``gdt``, i.e. ``index = dose_i * k + pc``. Run 3's reader
#: gathers ``e[form][m, :, cols]`` whole, so this ordering is a run-6 contract
#: that only run 6 can assert --- ``tests/test_slots.py`` does.
SLOT_AXIS_ORDER = "(dose, pc) C-order within each of vvp and gdt"

# --------------------------------------------------------------------------- #
# the reduction: PCA on the model dimension only, slots kept separate
# --------------------------------------------------------------------------- #
#: principal components stored on disk. The ladder below is a COLUMN PREFIX of
#: this block, because the components are eigenvalue-ordered --- so every rung
#: below K_STORE is free and none of them is a choice made after seeing a score.
K_STORE = 64
#: The rung the primary cell is read at.
#:
#: ⚠️ **RE-REGISTERED 2026-08-31, from 16 to 64, and the change is deliberate,
#: dated and made by the user rather than drifted into.** It was taken AFTER the
#: captured-energy diagnostic was read on the eight extracted block sets and
#: BEFORE any arm was scored --- so it cannot have been chosen for its answer,
#: because no answer existed yet, but it was chosen with knowledge of a
#: diagnostic and that is weaker than the original pre-registration.
#:
#: What was known when it was taken: at k = 16, 19 of 38 (model, line, layer)
#: cells reached ``BASIS_COVERAGE_TARGET``; at k = 64, 20 of 38. The worst cell
#: improves from 0.7665 to 0.8590 and the spread across cells halves, 0.2330 to
#: 0.1408. So the move buys ONE additional cell over the target, a better worst
#: case and a tighter spread --- not a closed concern, and the arm-to-arm
#: asymmetry that matters (``scgpt_ft``/HepG2 0.981-0.990 against its own control
#: ``scgpt_ft_frozen``/Jurkat 0.859-0.923) survives it.
#:
#: Costs nothing in compute: k = 64 is ``K_STORE``, already on disk for every
#: extracted block, and every rung of ``K_LADDER`` is reported regardless of
#: which one is primary. What it costs is interpretive, and the write-up must
#: carry it: the k = 16 rung stays reported and must be quoted beside the k = 64
#: verdict, so a reader can see that the primary rung was moved and what the
#: other rung said. Arm dim at this rung is 10 x 64 = 640, against 160 at k = 16.
K_PRIMARY = 64
#: every rung reported. k = 1 gives dim 10, within one of run 3's ``resp`` dim
#: 11, so the ladder contains a genuinely width-matched comparison against
#: ``null_randpair``.
K_LADDER = (1, 4, 16, 64)

#: ONE basis per (model, cell line, layer), shared by all ten slots. A per-slot
#: basis would put each dose in its own coordinate frame, and "how the response
#: changes with dose" would stop being a comparison of like coordinates.
BASIS_SHARED_ACROSS_SLOTS = True
#: fitted on RAW delta-h pooled over slots, so high-dose slots dominate the
#: leading PCs. Deliberate: RMS-normalising per slot before the fit would remove
#: exactly the dose-magnitude axis this run exists to preserve.
BASIS_SLOT_RESCALED = False
#: uncentred (second-moment) PCA is primary. A centred variant is a sensitivity
#: arm and is exactly run 5's missing generic-per-gene-response control
#: (``docs/METRICS.md`` s12.7).
BASIS_CENTRED = False
#: no whitening: run 3's ``Standardizer`` z-scores every column in-fold anyway,
#: so whitening would be absorbed and would only obscure the eigenvalue ordering
BASIS_WHITENED = False

#: perturbations the basis is fitted on. Drawn from the SCREENED perturbations
#: whose target is itself a read-out gene but whose effect stratum excludes them
#: from run 3's scoring set --- real CRISPRi knockdowns with measured knockdown
#: efficiency, in the same experimental regime as the scored set, and excluded
#: from every fold on either split because the stratum filter removes them.
N_BASIS_GENES = 64
BASIS_SEED = SEED + 601
#: the strata run 3 scores. A basis gene must be in NEITHER of them.
SCORED_STRATA = ("S3", "S4_highest")
#: the strata the basis pool is drawn from
BASIS_STRATA = ("S1_lowest", "S2")
#: the draw is stratified on both axes, so the basis pool spans the same
#: expression and effect-size range as the scored set rather than one tail of it
BASIS_STRATIFY_ON = ("pooled_control_cp10k", "effect_rms_lfc")
BASIS_N_EXPRESSION_BINS = 4
BASIS_N_EFFECT_BINS = 4

#: the captured-energy diagnostic of README s2: a second d x d second-moment
#: matrix accumulated over the SCORED perturbations in the same streaming pass,
#: eigendecomposed to ``W_all``, used to report what fraction of the scored
#: perturbations' delta-h energy the basis captures. **Diagnostic only, never
#: used to score.** At or above this the concern is closed; below it the arm is
#: known to be downward-biased and by how much, and that goes in the limitations
#: BEFORE any score is read.
BASIS_COVERAGE_TARGET = 0.95

# --------------------------------------------------------------------------- #
# arm keys --- and the character that must never appear in one
# --------------------------------------------------------------------------- #
#: Run 3's bundle reader splits a feature file name on the FIRST ``__`` to
#: separate the model key from the anchor, so a key containing ``__`` makes the
#: block unreadable and the response arm **vanish from the scored set with no
#: error at all**. This cost run 4 a whole round of re-scoring (cp-13). Run 6's
#: keys are built and parsed only through ``model_stem`` / ``run_of_stem``, and
#: both refuse a key that would break the reader.
KEY_SEPARATOR = "-"
FORBIDDEN_IN_KEY = "__"

#: the feature variants a key can name.
#:   hid       the real projected delta-h
#:   hidrp     the same delta-h through a frozen RANDOM orthonormal basis
#:   hidperm   the real projected features with the perturbed-gene index
#:             permuted independently per read-out gene
#:   hidrand   iid Gaussian in k per slot, per-slot scale matched
#:   norm      stage 0: run 5's per-pair ||delta_h||, dim 5, no k
#:   normperm  stage 0's machinery null
#:   normrand  stage 0's width-matched frozen random null
VARIANTS = ("hid", "hidrp", "hidperm", "hidrand")
STAGE0_VARIANTS = ("norm", "normperm", "normrand")
#: which variants carry real delta-h rather than a null
REAL_VARIANTS = ("hid", "hidrp", "norm")

#: variants whose block is DERIVED from the real block at stage time rather than
#: written by the extractor. A permutation of a stored array and a seeded
#: Gaussian are both reproducible from a seed, so storing them would triple the
#: extraction budget for no information.
DERIVED_VARIANTS = ("hidperm", "hidrand", "normperm", "normrand")

#: per-variant null seeds, distinct so that changing one draw cannot move
#: another. Every actual draw derives a seed from these AND from the key it is
#: drawn for, so two keys never share a permutation.
NULL_SEEDS = {
    "hidperm": SEED + 701, "hidrand": SEED + 702, "hidrp": SEED + 703,
    "normperm": SEED + 711, "normrand": SEED + 712,
}


def model_stem(model: str, variant: str, layer: int, k: int | None = None) -> str:
    """The bundle model key for one (model, variant, layer, k).

    Hyphen-separated on purpose: run 3 splits the feature file name on ``__``,
    so a key that contained one would make the block unreadable and the arm
    disappear without an error. Encoding and decoding both go through this pair
    of functions rather than through an f-string at the call site.
    """
    if variant not in VARIANTS + STAGE0_VARIANTS:
        raise ValueError(
            f"unknown variant {variant!r}; expected one of {VARIANTS + STAGE0_VARIANTS}"
        )
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")
    parts = [model.replace("_", "-"), variant, f"l{int(layer)}"]
    if k is not None:
        parts.append(f"k{int(k)}")
    key = KEY_SEPARATOR.join(parts)
    if FORBIDDEN_IN_KEY in key:
        raise RuntimeError(
            f"the model key {key!r} contains {FORBIDDEN_IN_KEY!r}; run 3's bundle reader "
            "splits the feature file name on it, so the block would be unreadable and the "
            "arm would vanish from the scored set with no error"
        )
    return key


def run_of_stem(key: str) -> dict:
    """Recover ``{model, variant, layer, k}`` from a bundle model key."""
    parts = key.split(KEY_SEPARATOR)
    if len(parts) < 3:
        raise ValueError(f"{key!r} is not a run-6 model key")
    k: int | None = None
    if parts[-1].startswith("k") and parts[-1][1:].isdigit():
        k = int(parts[-1][1:])
        parts = parts[:-1]
    if not (parts[-1].startswith("l") and parts[-1][1:].isdigit()):
        raise ValueError(f"{key!r} carries no layer field")
    layer = int(parts[-1][1:])
    variant = parts[-2]
    model = KEY_SEPARATOR.join(parts[:-2]).replace("-", "_")
    if model not in MODELS:
        raise ValueError(f"{key!r} names no run-6 model (parsed {model!r})")
    if variant not in VARIANTS + STAGE0_VARIANTS:
        raise ValueError(f"{key!r} names no run-6 variant (parsed {variant!r})")
    return {"model": model, "variant": variant, "layer": layer, "k": k}


def arm_name(key: str) -> str:
    """The run-3 arm a run-6 block is scored as.

    ``resolve_arms`` emits ``{m}_resp`` for every model key present in the
    bundle, so a run-6 block installed as a pseudo-model is read by run 3's
    UNMODIFIED response arm. ``{m}_emb`` / ``_full`` / ``_hybrid`` would
    ``KeyError`` on ``bundle.emb[m]`` and are never requested.
    """
    return f"{key}_resp"


# --------------------------------------------------------------------------- #
# the sweep run 3 is asked for
# --------------------------------------------------------------------------- #
#: ``full`` only. Installing a block redefines ``scope_rows("common")`` for
#: EVERY arm in the sweep, and ``common`` exists for AIDO.Cell comparability
#: which run 6 has no arm for.
SCOPES = ("full",)
SPLITS = ("grouped", "random")
TRANSLATORS = ("mlp", "ridge")

#: in-job reference arms, per run 3 s7. Deliberately NOT ``<model>_resp``: run 6
#: emits no decoder-head reading at all (README s5b), and the head comparison is
#: post-hoc against runs 2-4 as published. Reference rows stay OUT of the merged
#: run table; a separate check compares them to run 3's published rows so that
#: no scored row is ever overwritten.
REFERENCE_ARMS = ("coexp", "coexp_node", "null_randpair")
#: the static gene tables read as references beside them, one ``_emb`` arm each
REFERENCE_EMB_MODELS = ("scgpt", "scprint", "transcriptformer")

# --------------------------------------------------------------------------- #
# scoring cells
# --------------------------------------------------------------------------- #
#: run 3's pre-registered primary cell, unchanged and unchosen. The verdict is
#: read here. Residual is taken against the TRAINING folds' mean effect, so a
#: held-out perturbation's own outcome never enters its score.
PRIMARY_METRIC = "residual__weighted__disattenuated"
#: run 3's post-hoc secondary. Arms are ORDERED by it because it asks the harder
#: question --- separating one held-out perturbation from another --- but the
#: verdict does not rest on it, and neither cell confirms the other.
#: WARNING: these two roles reversed at cp-23; text predating that has them
#: backwards (``docs/INDEX.md`` s5).
ORDERING_METRIC = "centered__weighted__disattenuated"

#: the layer / k / scope / split / translator the primary cell is read at
PRIMARY_CELL = {
    "layer": "last block",
    "k": K_PRIMARY,
    "scope": "full",
    "split": "grouped",
    "translator": "mlp",
    "anchor": ANCHOR,
    "metric": PRIMARY_METRIC,
}

#: The two scoring families run 6 adds beside run 3's weighted Pearson, declared
#: here BEFORE any arm is scored so that they are pre-registered secondaries and
#: not a metric picked for its answer. Both are computed from the SAVED
#: per-perturbation predictions, so they add no fit and cannot change one.
#:
#:   cosine  row-wise WEIGHTED cosine similarity between the predicted and the
#:           measured profile, with the same per-gene reliability weights and the
#:           same zeroed self column the Pearson cell uses. It is the Pearson
#:           cell without the per-row mean removal, so it keeps the part of the
#:           response a mean-centred correlation discards.
#:   pds     L1 perturbation discrimination: ``1 - own_rank / N``, where the
#:           prediction for perturbation p is ranked against every measured
#:           profile in the scored pool by Manhattan distance. 1.0 is perfect,
#:           ~0.5 is chance. Same definition as run 3's
#:           ``discrimination.own_discrimination``, at the denominator the
#:           design asks for; run 3's own column is reported beside it unchanged.
EXTRA_METRICS = ("cosine", "pds")
#: the distance the discrimination ranking uses
PDS_DISTANCE = "l1"
#: ``1 - own_rank / N`` with N the number of perturbations in the scored pool.
#: Run 3's ``own_discrimination`` divides by n-1; the two differ by 0.2% at
#: n = 524 and both are reported, because silently substituting one for the
#: other is how a number stops being traceable to a definition.
PDS_DENOMINATOR = "n"
#: the perturbed gene's own read-out column is dropped from the distance for
#: BOTH members of every pair, matching the zero weight it carries in every
#: other cell of the grid. Leaving it in would let the discrimination score ride
#: on the knockdown column, which is the one column no arm is asked to predict.
PDS_EXCLUDE_SELF_COLUMNS = True
#: the targets and weightings the extra metrics are computed on, matching the
#: grid's own so that a cosine cell and a Pearson cell name the same quantity
EXTRA_METRIC_TARGETS = ("residual", "centered", "raw")
EXTRA_METRIC_WEIGHTINGS = ("weighted", "unweighted")

#: Non-finite handling for the discrimination score, recorded here because it is
#: a scientific rule and not an implementation detail. TranscriptFormer never
#: reads a gene's count at its own position, so its ``(p, p)`` hidden delta is
#: `nan` and every cell of the grid excludes that column. The distance sets those
#: entries to 0 BEFORE the pairwise pass and the rank-1 self-column corrections
#: then remove them exactly, so an excluded column contributes exactly zero.
#: Sanitising WITHOUT the exclusion would score a fabricated value, so that
#: combination raises instead.
PDS_SANITISED_NOTE = (
    "non-finite entries were set to 0.0 before the pairwise distance and the "
    "self-column corrections were computed on the same sanitised arrays, so the "
    "excluded columns contribute exactly zero; the only non-finite entries "
    "expected are TranscriptFormer's undefined self positions, and the count of "
    "any outside a row's own column travels beside every score"
)

#: The pool a discrimination rank is taken against, reported BOTH ways because
#: they answer different questions and neither subsumes the other.
#:   pooled    every scored perturbation at once --- the whole-screen reading,
#:             and the pool run 3's own `discrimination_ours_residual_*` column
#:             uses, so the two are comparable
#:   per_fold  each held-out perturbation against its own fold's held-out rows
#:             only --- the reading whose pool matches the fold structure the
#:             ceiling is aggregated over
#: A restricted reading (the fine-tuned arms on folds 0 and 1) RECOMPUTES the
#: pooled variant on the restricted rows: restricting a pool changes every rank
#: in it, so a subset of the full-pool numbers would report a 524-way ranking as
#: if it were a ~210-way one.
PDS_POOLS = ("pooled", "per_fold")

#: Disattenuation. Run 3 divides a correlation by sqrt(ceiling) under the
#: classical measurement model; run 6 applies that to the cosine cells, whose
#: ceiling is the same split-half quantity computed with the same estimator, and
#: does NOT apply it to the discrimination score, for which no such correction
#: is defined. The discrimination ceiling --- one measured half's profiles
#: ranked against the other's --- is computed and REPORTED beside every score,
#: because reading a number against 1.0 rather than against what the measurement
#: reproduces is the error this project's claims discipline exists to prevent.
DISATTENUATE = {"pearson": True, "cosine": True, "pds": False}

# --------------------------------------------------------------------------- #
# the fine-tuned arms and their fold restriction
# --------------------------------------------------------------------------- #
#: run 4's ``MODEL_HELDOUT_FOLDS``. The fine-tuned checkpoints TRAINED on folds
#: 2, 3 and 4, so run 3's unmodified five-fold sweep would make three fifths of
#: their perturbations a training number. A named check asserts this equals run
#: 4's own value and that run 6's grouped fold vector equals run 4's element-wise
#: --- if it differs, the restriction does not mean what run 6 thinks and the arm
#: is unscoreable.
MODEL_HELDOUT_FOLDS = (0, 1)
#: the frozen counterpart must be re-read on the SAME two folds, or a two-fold
#: score is being compared against five-fold scores
FT_FROZEN_COUNTERPART = "scgpt"

#: the perturbation flag is SET ON THE PERTURBED GENE IN BOTH PASSES, so the
#: only thing that changes between them is the dose. This is neither of run 4's
#: two query families and it inherits neither of their null calibrations.
FT_FLAG_IN_BOTH_PASSES = True
FT_FLAG_NOTE = (
    "perturbation_flags: set on the perturbed gene in BOTH the baseline and the "
    "perturbed pass; only the input value moves. This contradicts the inherited "
    "layout note on run 4's `configure`, and it is neither of run 4's two query "
    "families: run 4's value substitution has the flag off in both passes, and "
    "run 4's flag family has it off in the baseline and on in the perturbed pass "
    "with no dose axis at all."
)
#: half precision would destroy the feature: a response is a difference between
#: two passes agreeing to ~1e-4, which is below the representable gap in fp16.
#: That applies with more force to a delta-h than to a scalar.
FT_AMP = False
#: run 4 deviates from the published recipe here, and the deviation is a
#: PRECONDITION for the flag-on query to mean anything: a flag set on a gene the
#: sampled sequence dropped is a flag on nothing. Asserted, never assumed.
FT_FORCE_PERTURBED_GENE_IN_SEQUENCE = True

# --------------------------------------------------------------------------- #
# nulls
# --------------------------------------------------------------------------- #
#: run 3's own shuffled-label draws, permuted WITHIN the training perturbations
#: only. An arm counts only if it clears the LARGEST draw in ITS OWN cell,
#: against fresh draws --- never a number copied from run 3 or from run 4.
N_SHUFFLE_DRAWS = 5

#: what ``hidrand`` does and does not control for, recorded because the
#: temptation to draw it in d and project is real and would be wasted work:
#: if z ~ N(0, s^2 I_d) then W'z ~ N(0, s^2 W'W) = N(0, s^2 I_k) for an
#: orthonormal W, independent of W. So this null controls for WIDTH only, and
#: the substantive machinery controls are ``hidperm`` and ``hidrp``.
HIDRAND_NOTE = (
    "iid Gaussian in k per slot with the per-slot scale matched to the real "
    "block. Projecting a d-dimensional Gaussian through an orthonormal basis "
    "gives exactly this distribution, so drawing in d would burn 15-94 GB of RNG "
    "to reach it. This null controls for width only."
)

# --------------------------------------------------------------------------- #
# staging
# --------------------------------------------------------------------------- #
#: ``LineBundle`` puts every array on the scoring device and this machine has
#: 24 GB of VRAM, so a bundle is staged with as many blocks as fit under this
#: budget and never with all of them. Purely an execution knob: which blocks
#: share a bundle cannot reach a score, because ``scope_rows("full")`` is every
#: row and no arm reads another arm's block.
STAGE_BYTES_BUDGET = 8 * 1024**3

#: The main pass's ``(n_pert, n_values, n_readout, k)`` accumulator is allocated
#: **on disk and memory-mapped**, never in RAM. Measured off the 8-gene smoke
#: (2026-08-31): 2 variants x 2 forms x 5 sweep layers is **34.0 GB for HepG2 and
#: 54.6 GB for Jurkat**, against this machine's 70 GB total --- so the in-RAM form
#: would have died on the larger line *after every forward pass had already run*,
#: which is the most expensive possible moment to fail.
#:
#: There is nothing to stream out early instead: the gene loop is the OUTER loop,
#: because one forward pass captures every layer at once, so no layer's block is
#: complete until the last gene is done. The alternatives were re-running the
#: forwards once per layer group --- for TranscriptFormer/Jurkat that is +3.4 h of
#: GPU per extra group --- or storing fewer components, and ``K_STORE`` is
#: pre-registered. Mapping the accumulator is what keeps the pass at one forward
#: pass per gene. ``/biodata`` sustains ~480 MB/s on mmap, so the 34-55 GB of
#: writes cost ~1-2 min against a 0.3-3.4 h pass.
#:
#: **An execution knob and nothing else: the bytes written are the same bytes.**
STORE_MEMMAPPED = True

# --------------------------------------------------------------------------- #
# scope
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Scope:
    """How much of the in-scope perturbation set an extraction block runs on.

    ``0`` means every gene in scope --- 524 in HepG2, 705 in Jurkat --- which is
    what every production block uses. The smoke path sets a small number. What
    is in force is recorded in the block manifest beside the size of the
    in-scope set, so two blocks at different coverage cannot be compared by
    accident.
    """

    n_perturbed: int = 0
    strata: tuple[str, ...] = field(default_factory=lambda: SCORED_STRATA)


SCOPE = Scope()
#: the smoke path: enough genes to exercise every reassembly branch, few enough
#: to run in minutes on one 3090
SMOKE_SCOPE = Scope(n_perturbed=8)
SMOKE_N_BASIS_GENES = 8
