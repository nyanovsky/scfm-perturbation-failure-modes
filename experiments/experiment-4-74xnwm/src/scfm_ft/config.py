"""Run 4's pre-registration: what is trained, on what, and what is scored.

Everything run 3 fixed stays fixed and is read from run 3's own ``config.py``
rather than restated here. What is new in run 4 is the fine-tuning recipe, the
five training arms, and the rule that decides which perturbations a fine-tuned
model is allowed to be scored on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SEED = 42
CELL_LINES = ("hepg2", "jurkat")

# --------------------------------------------------------------------------- #
# base model
# --------------------------------------------------------------------------- #
#: the same checkpoint runs 1-3 queried frozen, at the same revision
SCGPT_REPO = "wanglab/scGPT-human"
SCGPT_REVISION = "a24c237737a40f3720f75abb555489e9fe753be6"
SCGPT_FILES = ("vocab.json", "args.json", "best_model.pt")

# --------------------------------------------------------------------------- #
# which perturbations the model may see
# --------------------------------------------------------------------------- #
#: Run 3 scores five grouped folds. A fine-tuned model may only be scored on
#: perturbations it never saw, so two of those folds are withheld from
#: fine-tuning and are the only folds run 4 reads a number from. Two rather than
#: one because the whole comparison is a paired difference between arms
#: bootstrapped over response groups, and one fold's worth of groups resolves
#: too little; three training folds still leave more perturbations than the
#: perturbation benchmarks this recipe was published on.
MODEL_HELDOUT_FOLDS = (0, 1)
#: of the remaining folds, this fraction of whole response groups is held out
#: again as the fine-tuning validation set (model selection, early stopping).
#: Never drawn from MODEL_HELDOUT_FOLDS: selecting on the scored folds would
#: make the held-out number a training number.
VAL_GROUP_FRACTION = 0.10

#: cells per perturbation entering fine-tuning, drawn without replacement.
#: GEARS pairs one control cell with each perturbed cell; this caps how many
#: cells a heavily sampled perturbation contributes so the loss is not dominated
#: by a handful of targets.
MAX_CELLS_PER_PERT = 100
#: a perturbation with fewer cells than this contributes no training examples
MIN_CELLS_PER_PERT = 10

# --------------------------------------------------------------------------- #
# the published recipe (scGPT tutorials/Tutorial_Perturbation.ipynb)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Recipe:
    """scGPT's own perturbation fine-tuning settings, copied from the tutorial.

    Only ``max_seq_len``'s consequence differs from the published runs: the
    read-out set here is larger than 1,536, so the recipe's random gene subset
    per batch is active. scGPT has no positional encoding over genes, and
    inference still uses the whole set, exactly as ``pred_perturb`` does.
    """

    lr: float = 1e-4
    batch_size: int = 64
    eval_batch_size: int = 64
    epochs: int = 15
    schedule_interval: int = 1
    schedule_gamma: float = 0.9
    early_stop: int = 10
    grad_clip: float = 1.0
    amp: bool = True
    dropout: float = 0.0
    pad_token: str = "<pad>"
    pad_value: int = 0
    pert_pad_id: int = 0
    include_zero_gene: str = "all"
    max_seq_len: int = 1536
    #: the pretrained tensors the recipe transfers; the expression decoder is
    #: initialised fresh, as in the tutorial
    load_param_prefixes: tuple[str, ...] = ("encoder", "value_encoder", "transformer_encoder")
    loss: str = "masked_mse_loss(all positions)"
    optimizer: str = "Adam"
    #: the tutorial's training loop keeps the epoch with the highest validation
    #: Pearson correlation between the predicted and the measured mean expression
    #: profile of each held-out perturbation, and applies its patience to that
    #: score. The masked MSE on the same cells is recorded every epoch and its
    #: selected epoch kept as the obvious alternative rule; the agreement
    #: between predicted and measured response is a diagnostic only.
    val_metric: str = (
        'validation Pearson over held-out perturbations (the tutorial\'s '
        'val_metrics["pearson"]); masked MSE recorded and selected on in parallel'
    )


RECIPE = Recipe()

# --------------------------------------------------------------------------- #
# the five training arms
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Arm:
    """One fine-tuning run: which gene table, and whether it updates."""

    name: str
    table: str  # "scgpt" | "esm2_3b" | "random"
    trainable_table: bool
    #: a table wider than the model needs a learned projection into d_model
    project: bool
    note: str = ""

    @property
    def model_key(self) -> str:
        """The name this arm's features and gene table carry into the bundle."""
        return f"scgpt_ft_{self.name}"


ARMS: tuple[Arm, ...] = (
    Arm("own_trainable", "scgpt", True, False, "the standard published recipe"),
    Arm("own_frozen", "scgpt", False, False, "gain attributable to the readout alone"),
    Arm("esm2_trainable", "esm2_3b", True, True, "protein provenance as the starting table"),
    Arm("esm2_frozen", "esm2_3b", False, True, "the same, with the table held fixed"),
    Arm(
        "rand_frozen",
        "random",
        False,
        True,
        "capacity-matched null for the two ESM2 arms: the same width and the same "
        "learned projection, over vectors that carry no information about the gene",
    ),
)
ARMS_BY_NAME = {a.name: a for a in ARMS}


#: A run name is ``<arm>`` or ``<arm>__<tag>``. Run 3's bundle reader splits a
#: feature file name on the FIRST ``__`` to separate the model key from the
#: anchor, so a model key containing ``__`` makes the block unreadable and the
#: response arms silently vanish from the scored set. Model keys therefore carry
#: the tag behind a character run 3 never splits on, and the run name is
#: recovered from the key by the inverse.
def model_stem(run_name: str) -> str:
    """The run name as it may appear inside a bundle model key."""
    if "-" in run_name:
        raise ValueError(f"run name must not contain '-': {run_name}")
    return run_name.replace("__", "-")


def run_of_stem(stem: str) -> str:
    """Inverse of :func:`model_stem`."""
    return stem.replace("-", "__")

#: the wide table the substitution arms start from, and where it comes from
ESM2_TABLE = "esm2_3b"
ESM2_TABLE_REF = (
    "artifact://fundacion-instituto-leloir-553557/experiments/"
    "exp_01m0emwktvfe892x6ebqyga5ws/protein/esm2_3b.npz"
)
RUN3_BUNDLE_REF = (
    "artifact://fundacion-instituto-leloir-553557/experiments/"
    "exp_01m0emwktvfe892x6ebqyga5ws/bundle/"
)

# --------------------------------------------------------------------------- #
# what is read off each checkpoint
# --------------------------------------------------------------------------- #
#: the model's own perturbation query: set the perturbation flag on gene i and
#: read every gene's predicted expression, minus the same forward pass with no
#: flag set. This is the query the fine-tuning objective is defined on.
ANCHOR_PERT_FLAG = "pert_flag"
#: run 2's query family, unchanged: substitute a value for gene i's input
#: expression and read the difference in every gene's predicted output. This is
#: the query the frozen model was read with in runs 2 and 3, so it is the one
#: that makes "before fine-tuning" and "after fine-tuning" the same measurement.
ANCHOR_VALUE = "mean_control"

#: features are also extracted from the same architecture before any gradient
#: step, so the run carries its own before/after contrast in addition to run 3's
STAGES = ("init", "final")

#: the form every feature enters the translator in, matching run 3's
#: FEATURE_FORM: a z-score across read-out genes within one query, which is what
#: makes models with different output scales comparable at all
FEATURE_FORM_KEY = "standardized"

#: how many control cells the anchor averages over. Run 3's primary anchor is
#: the mean control profile, one query per perturbed gene.
N_ANCHOR_PROFILES = 1

# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
#: run 3's arms re-run inside the same job on the same restricted folds, so the
#: fine-tuned arms are compared against numbers produced by the same code path
REFERENCE_ARMS = ("coexp", "node", "coexp_node", "scgpt_resp", "scgpt_emb", "scgpt_full", "mean_only")
#: shuffled-label draws for the fine-tuned response arm and co-expression
SHUFFLE_ARMS = ("coexp",)
TRANSLATORS = ("mlp", "ridge")
SPLIT = "grouped"
SCOPE = "full"


@dataclass
class ArmOutputs:
    """What one training job leaves behind, per arm and cell line."""

    weights: str = "model_best.pt / model_final.pt"
    config: str = "train_config.json"
    gene_table: str = "gene_table_init.npz / gene_table_final.npz"
    history: str = "loss_history.json"
    provenance: str = "provenance.json"
    features: str = "features/<model_key>__<anchor>.npz"
    direct: str = "direct_prediction.npz"
    extras: list[str] = field(default_factory=list)
