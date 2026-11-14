"""Hidden states from run 4's fine-tuned scGPT, with the perturbation flag ON.

Run 4 found the *decoder head* of a fine-tuned scGPT still inside run 3's band.
Reading the same checkpoints' hidden states asks the one question that separates
"fine-tuning installed nothing" from "fine-tuning installed something the
readout cannot expose". scGPT is the cheapest model in the thread, so two more
checkpoints cost about half an hour of GPU.

Why this file exists at all, rather than a fourth entry in run 5's ``BUILDERS``:
run 5's three adapters take ``(cache, universe, vocab, device)`` and build a
frozen published checkpoint; run 4's ``ScGPTPertQuery`` takes an ALREADY BUILT
model, because a fine-tuned arm is defined by a substituted gene table that only
run 4 knows how to assemble. The bridge is here, and run 5's file is untouched.

The estimator, and why the flag is on in BOTH passes
----------------------------------------------------
::

    dh[p, g] at dose v  =  h( flags[p]=1, x[p]=v      )[g]
                         - h( flags[p]=1, x[p]=anchor )[g]

The flag is how this model is *told* a perturbation is happening; it was
fine-tuned with the flag set, so a flag-off reading is the off-distribution one.
Holding it fixed across both passes buys two things that a flag-off baseline
would destroy:

* **the estimator stays structurally identical to the frozen models'.** Exactly
  one thing changes between the two passes --- the value. The flag is fixed
  input context, like the anchor profile. So ``vvp`` and ``gdt`` mean the same
  operation here as everywhere else, and the dose sweep and the flag are
  orthogonal rather than confounded;
* **control C2 survives.** Because the flag is identical in both passes, the
  layer-0 off-target ``dh`` is still exactly ``0.0``. Flag onset does reach
  other tokens, so a flag-off baseline would have broken C2 outright and would
  additionally have conflated flag onset with dose.

This is neither of run 4's two query families (its value substitution has the
flag off in both passes; its flag family has it off in the baseline, on in the
perturbed pass, and no dose axis at all), so it inherits NEITHER of their null
calibrations. It takes fresh shuffled-label draws in its own cell.

⚠️ Run 4's inherited ``configure`` layout note reads *"perturbation_flags: all
zero for this query family: only the input value moves"*. For run 6 that is
false, and ``layout_notes`` below overrides it verbatim so the block cannot
misdescribe itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from .. import config as C
from .. import paths, sibling


def _run5_hidden():
    """Run 5's hidden-state machinery, imported rather than copied.

    ``HiddenQuery`` carries the hook plan, the stack resolution and the capture
    checks; reproducing them here would mean two implementations of "which
    module is layer 7" that could disagree without either being wrong on its own
    terms.
    """
    return sibling.run5("models.hidden")


def _run4_query():
    return sibling.run4("scgpt_pert.query")


def _run4_model():
    return sibling.run4("scgpt_pert.model")


def make_hidden_class():
    """Build ``ScGPTPertHiddenQuery`` once, after the siblings are importable.

    The class cannot be declared at module scope: its bases are run 5's
    ``HiddenQuery`` and run 4's ``ScGPTPertQuery``, and both live behind the
    sibling registration, which drags in torch and each run's model stack. Every
    other module in run 6 must stay importable on a machine with neither.
    """
    hidden = _run5_hidden()
    HiddenQuery = hidden.HiddenQuery
    ScGPTPertQuery = _run4_query().ScGPTPertQuery

    class ScGPTPertHiddenQuery(HiddenQuery, ScGPTPertQuery):  # type: ignore[misc]
        """Run 4's fine-tuned scGPT behind run 5's hidden-state interface.

        Mixin first, exactly as run 5's three adapters do it: ``HiddenQuery``
        adds methods and overrides nothing, so an instance is still the run 4
        adapter run 4's own extraction drove --- which is what makes "the head
        said this" and "the hidden state said that" two readings of one object.
        """

        #: scGPT reads a gene's own value at its own position, so the perturbed
        #: gene's own hidden state IS a function of its own input and C1 is a
        #: real positive control here (unlike TranscriptFormer's).
        hidden_self_position_defined = True

        #: pinned from the built model rather than guessed: ``build_arm_model``
        #: constructs ``TransformerGenerator`` with ``use_fast_transformer=False``,
        #: whose trunk is a ``torch.nn.TransformerEncoder`` at this path with
        #: exactly ``N_LAYERS`` blocks. Verified on all four checkpoints by
        #: ``scratch/probe_ft_checkpoint.py`` before any extraction was run.
        _stack_candidates = ("transformer_encoder.layers",)

        def __init__(self, model, n_readout: int, device: str = "cuda", amp: bool = C.FT_AMP):
            ScGPTPertQuery.__init__(self, model, n_readout, device=device, amp=amp)
            self._hidden_plan = None
            self._reset()

        # ---- run 5's per-model constants ---------------------------------- #
        def _model_key(self) -> str:
            """``"scgpt"``, because the ARCHITECTURE is scGPT's.

            ``HiddenQuery._plan`` and ``_take`` look this key up in run 5's
            ``N_LAYERS`` and ``D_MODEL`` to decide how many blocks to expect and
            how wide a captured state must be. A fine-tuned arm is scGPT's
            twelve 512-wide blocks with a substituted gene table, so run 5's
            constants apply verbatim and reporting a new key here would only
            mean adding two entries that had to equal scGPT's anyway. The run-6
            identity travels in ``hidden_notes`` instead, where it cannot be
            mistaken for an architectural fact.
            """
            return "scgpt"

        def _final_capture(self):
            """Layer ``L`` is read at ``transformer_encoder``, not at its last block.

            That module's output is by construction the tensor
            ``TransformerGenerator`` hands to its expression decoder, so hooking
            it makes the last layer the state the head actually reads --- rather
            than a state that merely usually equals it. Run 5's scGPT adapter
            makes the identical choice for the identical reason.
            """
            return self.model.transformer_encoder, "transformer_encoder"

        # ---- the flag ------------------------------------------------------ #
        def _flags(self, batch_rows: int, perturbed_col: int) -> torch.Tensor:
            """The perturbation flag, set at exactly one column, for every row.

            One tensor is built and reused for both passes of a query, which is
            what makes "the flag is identical in both passes" a property of the
            code rather than a claim about it.
            """
            f = torch.zeros(batch_rows, self.n_readout, dtype=torch.long, device=self.device)
            f[:, int(perturbed_col)] = 1
            return f

        def _hidden_pass(
            self, values: np.ndarray, perturbed_col: int, plan, layers: Sequence[int]
        ) -> dict[int, np.ndarray]:
            """One supplied set through run 4's own forward path, hooks attached.

            ``self._forward`` is called rather than ``self.model(...)`` directly,
            so the captured state comes from byte-identical code to the one that
            produced run 4's published responses.
            """
            width = _run5_hidden().C.D_MODEL[self._model_key()]
            acc: dict[int, list[np.ndarray]] = {int(l): [] for l in layers}
            with _run5_hidden()._capture(plan, layers) as store:
                for _, chunk in self._batches(values):
                    store.clear()
                    t = torch.as_tensor(chunk, device=self.device, dtype=torch.float32)
                    f = self._flags(int(chunk.shape[0]), perturbed_col)
                    with torch.no_grad():
                        with torch.autocast("cuda", enabled=self.amp):
                            self._forward(t, f)
                    got = self._take(store, layers, int(self.n_readout), width)
                    for l in layers:
                        acc[int(l)].append(got[int(l)])
            return {int(l): np.concatenate(acc[int(l)], axis=0) for l in layers}

        def hidden_batch(
            self,
            x_native: np.ndarray,
            layers: Sequence[int],
            perturbed_col: int | None = None,
        ) -> dict[int, np.ndarray] | None:
            """``{layer: (B, n_readout, d)}`` in read-out order.

            ``perturbed_col`` is REQUIRED here, unlike in run 5's three
            adapters. For them it selects a sequence composition and may be
            omitted; here it is the column the flag is set at, and a flag set on
            nothing is not this estimator --- it is run 4's value-substitution
            family under a different name.
            """
            if perturbed_col is None:
                raise RuntimeError(
                    "hidden_batch was called with no perturbed_col on the fine-tuned "
                    "adapter. The perturbation flag has to be set somewhere: with no "
                    "column this is run 4's value-substitution query, which run 4 has "
                    "already published, and not run 6's flag-constant estimator"
                )
            col = int(perturbed_col)
            # The published recipe samples 1536 genes and can drop the perturbed
            # one; run 4 deviates and forces it in, because a flag set on a gene
            # the sequence dropped is a flag on nothing. This adapter supplies
            # the WHOLE read-out set in one sequence, so the guarantee is
            # structural rather than sampled --- asserted, never assumed.
            if not (0 <= col < self.n_readout):
                raise RuntimeError(
                    f"perturbed column {col} is outside the {self.n_readout} read-out "
                    "positions, so the flag would be set on a gene this query never sees"
                )
            plan = self._plan()
            if plan is None:
                return None
            self._reset()
            wanted = self._check_layers(layers, plan)
            x = np.atleast_2d(np.asarray(x_native, dtype=np.float32))
            if x.shape[1] != self.n_readout:
                raise RuntimeError(
                    f"hidden_batch received {x.shape[1]} input columns but this model was "
                    f"configured for {self.n_readout} read-out genes"
                )
            out = self._hidden_pass(x, col, plan, wanted)
            self._hidden_notes.update(
                {
                    "run6_model_key": getattr(self, "run6_key", "scgpt_ft"),
                    "architecture": "scGPT TransformerGenerator, 12 blocks, d_model 512",
                    "supplied_mode": "readout",
                    "n_passes": 1,
                    "composition_depends_on_perturbed_col": False,
                    "perturbation_flags": C.FT_FLAG_NOTE,
                    "perturbed_gene_in_sequence": (
                        "structural: the whole read-out set is supplied in one sequence, so "
                        "the perturbed gene is always present and run 4's "
                        "force_perturbed_gene_in_sequence deviation is satisfied by "
                        "construction rather than by sampling"
                    ),
                    "amp": bool(self.amp),
                    "layers": list(wanted),
                    "hidden_source": {l: self.hidden_source(l) for l in wanted},
                }
            )
            return out

        def layout_notes(self) -> dict:
            """The block manifest's layout section, correcting the inherited note.

            Run 4's ``configure`` describes its value-substitution family and
            says the flags are all zero. For run 6 that is false; a block that
            carried the inherited sentence would misdescribe the very thing that
            distinguishes it.
            """
            return {
                "input_scale": "log1p(CP10K), the scale the perturbation recipe trains on",
                "perturbation_flags": C.FT_FLAG_NOTE,
                "output_head": "not read: run 6 emits hidden states only",
                "amp": bool(self.amp),
                "amp_reason": (
                    "a response is a difference between two passes agreeing to ~1e-4, "
                    "which in half precision is below the representable gap; that applies "
                    "with more force to a delta-h than to a scalar"
                ),
            }

    return ScGPTPertHiddenQuery


# --------------------------------------------------------------------------- #
# building one
# --------------------------------------------------------------------------- #
def train_config(cell_line: str, key: str) -> dict:
    """Run 4's recorded training configuration, where it was shipped.

    ``own_trainable`` carries one; ``own_frozen`` does not (only its
    ``model_best.pt`` was fetched). An absent file is recorded as an absence
    rather than filled in from ``config.py``, because the point of reading it is
    to check run 6's assumptions against run 4's record, and a default would
    check them against themselves.
    """
    arm = C.FT_ARM_OF[key]
    p = paths.ft_train_config(cell_line, arm)
    if not p.exists():
        return {
            "available": False,
            "path": str(p),
            "reason": (
                f"run 4 shipped no train_config.json for {arm}; only model_best.pt was "
                "fetched. The fold vector check for this arm therefore reads run 4's "
                "config.py rather than the run's own record"
            ),
        }
    out = json.loads(p.read_text())
    out["available"] = True
    out["path"] = str(p)
    return out


def build(key: str, cell_line: str, device: str = "cuda"):
    """One fine-tuned adapter, checkpoint loaded and verified.

    The rebuild is run 4's own (``build_arm_model`` then ``load_state_dict``),
    not a re-derivation: the architecture a checkpoint loads into is decided by
    the recipe constants in run 4's ``config.RECIPE``, and a second copy of them
    here would be a second thing that can drift.

    ⚠️ ``strict=False`` with a raise on any mismatch, rather than
    ``strict=True``: the error torch raises names one tensor, and the two
    failures worth telling apart --- a checkpoint from a different arm, and a
    checkpoint that is not a checkpoint at all --- are distinguished by the
    COUNTS. The 2026-08-25 fetch wrote a 134-byte HTML redirect body under a
    ``.pt`` name and stood for six days; an existence check passed it and a size
    threshold would have too.
    """
    if key not in C.FT_MODELS:
        raise KeyError(f"{key!r} is not a run-6 fine-tuned model key; expected {C.FT_MODELS}")
    arm_name = C.FT_ARM_OF[key]
    r4 = sibling.run4("config")
    modelmod = _run4_model()

    ckpt = paths.ft_checkpoint(cell_line, arm_name)
    if not ckpt.exists():
        raise RuntimeError(f"no fine-tuned checkpoint at {ckpt}")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise RuntimeError(
            f"{ckpt} loaded as {type(state).__name__} with no tensors; a checkpoint that "
            "is not a state dict is a failed download wearing a .pt name"
        )

    readout = pd.read_parquet(paths.readout_table(cell_line))
    positions = readout["scgpt_position"].to_numpy(dtype=np.int64)
    n_readout = int(readout.shape[0])

    arm = r4.ARMS_BY_NAME[arm_name]
    model, report = modelmod.build_arm_model(
        arm, paths.checkpoint_cache() / "scgpt", None, positions, device
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{ckpt} does not match the rebuilt {arm_name} architecture: "
            f"{len(missing)} missing and {len(unexpected)} unexpected tensors "
            f"(first missing {list(missing)[:3]}, first unexpected {list(unexpected)[:3]}). "
            "Either the checkpoint is a different arm's or run 4's build recipe has moved"
        )
    cls = make_hidden_class()
    q = cls(model, n_readout, device=device, amp=C.FT_AMP)
    q.run6_key = key
    q.substitutions = list(report.get("substitutions", []))
    q.build_report = {
        k: v for k, v in report.items() if k != "substitutions"
    } | {
        "run4_arm": arm_name,
        "run4_execution_suffix": C.FT_RUN_SUFFIX,
        "checkpoint": str(ckpt),
        "n_tensors_loaded": len(state),
        "train_config": train_config(cell_line, key),
    }
    return q


def checkpoint_audit(cell_line: str, key: str) -> dict:
    """Is this file a readable archive with a plausible tensor count?

    Not "does it exist" and not "is it bigger than N bytes": the 2026-08-25
    failure would have passed both, and a larger error body would defeat the
    size threshold too. The only check that separates a checkpoint from an HTTP
    error page is loading it and counting tensors.
    """
    arm = C.FT_ARM_OF[key]
    p = paths.ft_checkpoint(cell_line, arm)
    rec = {"cell_line": cell_line, "key": key, "arm": arm, "path": str(p),
           "exists": p.exists()}
    if not p.exists():
        rec["ok"] = False
        rec["detail"] = f"absent: {p}"
        return rec
    rec["bytes"] = int(p.stat().st_size)
    try:
        state = torch.load(p, map_location="cpu", weights_only=True)
        rec["n_tensors"] = int(len(state))
        rec["ok"] = bool(isinstance(state, dict) and len(state) > 0)
        rec["detail"] = f"{rec['n_tensors']} tensors, {rec['bytes']} bytes"
    except Exception as exc:  # the failure is itself the datum
        rec["ok"] = False
        rec["detail"] = f"{type(exc).__name__}: {exc}"[:300]
    return rec
