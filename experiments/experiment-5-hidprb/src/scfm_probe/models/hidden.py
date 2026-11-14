"""Per-gene hidden states out of run 2's three bidirectional adapters.

Why this module exists
----------------------
Run 3's verdict is about a *scalar* per (perturbed gene, read-out gene): every
one of these checkpoints ends in a shared per-token head that projects a
``d``-dimensional gene state down to one number. That projection is lossy in one
direction only, so ``delta_h`` cannot carry less than the scalar effect it
produces. Run 5's question --- did the intervention propagate at all, or is it
the readout that loses it --- is therefore a question about ``h``, and nothing in
the tree reads ``h``: ``grep register_forward_hook`` over runs 1-4 returns a
single unrelated attention patch. This module is that missing accessor and
nothing else. It computes no statistic, applies no normalisation and persists
nothing; pooling, the half split and the retrieval score live in ``probe/``.

Why subclassing, and not a patch to run 2
-----------------------------------------
Run 2's blocks, manifests and reports are cited by the paper and must stay
byte-identical, so editing ``scfm_pfm.models.queries`` is out of the question ---
even an addition, because the module is imported by run 3 and run 4 and a new
method is a new surface those runs would inherit unreviewed. Subclassing gets
the loading path, the attention substitutions, the CP10K map, the recorded
contract, the tiling and the sequence permutation for free and unmodified, which
also means a hidden state captured here comes from *the same forward pass* that
produced the responses run 3 scored --- not from a re-implementation that could
drift. The rejected alternative was a standalone hook harness that re-derived
tokenisation and tiling: it would have been independent of run 2 in exactly the
way that makes two numbers incomparable.

Layer convention
----------------
Layer ``l`` is the output of transformer block ``l``, 1-based. Layer ``0`` is the
post-embedding state that *enters* block 1; it is control C2, where an edit at
gene *i* cannot have reached gene *j* yet, so ``delta_h`` there must be exactly
zero. For scGPT layer ``L`` is hooked on ``transformer_encoder`` itself rather
than on its last block, because that tensor is by construction the one
``_encode`` returns and the decoder consumes, which makes the capture
independent of whether the encoder carries a final normalisation. scPRINT and
TranscriptFormer are hooked at their last block, so if their trunk normalises
before the head, layer ``L`` is the pre-normalisation state. Which module every
layer came from is not asserted here --- it is reported by :meth:`hidden_source`
and belongs in the block manifest, the analog of run 2's
``gene_embedding_source``.

The three traps
---------------
Raw hidden states are **not** in read-out order, and in two of the three models a
read-out gene appears in several passes with only one of them authoritative.
Three specific ways that corrupts ``delta_h`` silently, and what is done here:

1. **scGPT tile composition.** ``forward_batch`` appends the perturbed gene to
   any tile that does not already contain it, so the *sequence* differs between
   two queries by one token. If the baseline pass and the perturbed pass were run
   with different compositions, ``delta_h`` would mix a value change with a
   change in what else was in the context --- a difference that has nothing to do
   with propagation. Composition is therefore made a pure function of
   ``perturbed_col`` alone (:meth:`ScGPTHiddenQuery.tile_composition`), never of
   the values, and is published as a token signature
   (:meth:`ScGPTHiddenQuery.tile_token_signature`) so the caller can assert that
   its two passes agree. :func:`check_token_composition` raises when they do not.
2. **TranscriptFormer ordering.** The perturbed gene is forced to position 0 of
   every pass, so the ordering --- and hence the whole conditioning prefix ---
   is a function of ``perturbed_col``; baseline and perturbed must be called with
   the same one. Because nothing precedes position 0 in a causal model, the
   perturbed gene's own state cannot depend on any count, so
   ``self_response_defined`` is ``False`` here and the self position is filled
   with ``nan`` rather than the structural zero that would read as a measured
   "no propagation". C1 is unavailable for this model, and that is a recorded
   absence rather than a pass.
3. **Order.** scPRINT returns one sequence in vocabulary order; scGPT and
   TranscriptFormer return tiles and chunks. Reassembly goes through the
   adapters' own maps (``tile_owned``, ``readout_to_seq``, ``_query_chunks``) and
   asserts that every read-out position was written exactly once
   (:func:`reassemble_tiles`, :func:`reassemble_sequence`). Both are pure
   functions of arrays and index maps --- no torch, no model --- so the trap is
   testable without a GPU, which is the only reason it is testable at all.

Absence versus corruption
-------------------------
Two failure kinds, deliberately handled differently. A model that *cannot* supply
hidden states --- the block stack is not discoverable in this build --- returns
``None`` with a verbatim sentence in ``self._hidden_error``, mirroring the
``exact_jvp`` / ``_jvp_error`` contract: a measured absence, never a silent one,
and never zeros. A *broken invariant* --- a hook that did not fire, a read-out
position written twice, a composition mismatch, a width that disagrees with the
checkpoint --- raises ``RuntimeError``, because that is not a property of the
model, it is a number that would be wrong without saying so.

Memory: every capture is moved to CPU float32 numpy inside the hook, so no GPU
tensor survives the batch loop. The peak is the assembled block itself,
``B * n_readout * d``; for TranscriptFormer (``d = 2048``) that is ~213 MB per
layer at ``B = 8``, so a caller sweeping five layers should lower
``adapter.batch_size`` rather than expect the default to be free.

Invocation: imported, not run. ``build("scgpt", cache, universe, vocab, device)``
mirrors ``scfm_pfm.models.queries.build``; AIDO.Cell is deliberately absent (it
scored at zero and costs ~40x scGPT per forward pass, so it is not local).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .. import config as C
from .. import sibling

_r2 = sibling.run2("models.queries")


# --------------------------------------------------------------------------- #
# Reassembly: pure functions of (raw arrays, index maps). No torch, no model.
# --------------------------------------------------------------------------- #


from .reassembly import (  # noqa: F401  (re-exported: callers import from either)
    check_token_composition,
    composition_signature,
    reassemble_sequence,
    reassemble_tiles,
)


@dataclass(frozen=True)
class HiddenPlan:
    """Which modules the requested layers are read from, resolved once per model."""

    blocks: torch.nn.ModuleList
    blocks_name: str
    n_layers: int
    #: hooked instead of ``blocks[-1]`` for layer ``L`` where the trunk's own
    #: output --- not its last block's --- is the tensor the head consumes
    final_module: torch.nn.Module | None = None
    final_name: str | None = None


def _as_numpy(value: object, where: str) -> np.ndarray:
    """One captured activation -> CPU float32 numpy, immediately.

    Moved off the device inside the hook rather than at the end of the batch
    loop: a captured GPU tensor kept per layer per pass is what turns a five-layer
    sweep into an out-of-memory error on a 24 GB card.
    """
    if isinstance(value, (tuple, list)):
        if not value:
            raise RuntimeError(f"{where}: the module returned an empty tuple, not an activation")
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(
            f"{where}: captured a {type(value).__name__}, not a tensor; the hook is on a "
            "module whose output is not a per-token hidden state"
        )
    if getattr(value, "is_nested", False):
        # torch's TransformerEncoder may run its layers on a nested tensor when a
        # padding mask is supplied. With no padding present the padded form is the
        # same numbers; converting keeps the capture independent of that fast path.
        value = value.to_padded_tensor(0.0)
    if value.dim() != 3:
        raise RuntimeError(
            f"{where}: captured shape {tuple(value.shape)}; a per-gene hidden state is "
            "(batch, position, width) and anything else would be reassembled onto the "
            "wrong genes"
        )
    return value.detach().float().cpu().numpy()


@contextlib.contextmanager
def _capture(plan: HiddenPlan, layers: Sequence[int]) -> Iterator[dict[int, np.ndarray]]:
    """Register the hooks for ``layers``, and remove every one of them, always.

    A leaked hook does not raise: it keeps firing on the next call, overwrites a
    store that call believes it owns, and costs a device-to-host copy per forward
    pass forever after. Removal is in a ``finally`` for that reason, not for
    tidiness.
    """
    store: dict[int, np.ndarray] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def out_hook(layer: int, where: str):
        def fn(module, args, output):  # noqa: ANN001
            store[layer] = _as_numpy(output, where)

        return fn

    def in_hook(layer: int, where: str):
        def fn(module, args):  # noqa: ANN001
            store[layer] = _as_numpy(args[0] if args else None, where)

        return fn

    try:
        for layer in layers:
            if layer == 0:
                handles.append(
                    plan.blocks[0].register_forward_pre_hook(
                        in_hook(0, f"{plan.blocks_name}.0 input")
                    )
                )
            elif layer == plan.n_layers and plan.final_module is not None:
                handles.append(
                    plan.final_module.register_forward_hook(
                        out_hook(layer, f"{plan.final_name} output")
                    )
                )
            else:
                handles.append(
                    plan.blocks[layer - 1].register_forward_hook(
                        out_hook(layer, f"{plan.blocks_name}.{layer - 1} output")
                    )
                )
        yield store
    finally:
        for handle in handles:
            handle.remove()


def _resolve_stack(
    model: torch.nn.Module, expected: int, candidates: Sequence[str]
) -> tuple[torch.nn.ModuleList, str] | tuple[None, str]:
    """Find the transformer block stack, by name first and by shape as a fallback.

    Named paths are tried first because they are deterministic; the shape
    fallback exists because two of these three packages are installed from source
    with substituted attention and their internal attribute names are not part of
    any published contract. Ambiguity is refused rather than guessed: picking one
    of several equally-long ``ModuleList``s would attribute a layer to the wrong
    part of the model and nothing downstream could tell.
    """
    named = dict(model.named_modules())
    for path in candidates:
        found = named.get(path)
        if isinstance(found, torch.nn.ModuleList) and len(found) == expected:
            return found, path
    hits = [
        (name, mod)
        for name, mod in named.items()
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == expected and name
    ]
    if len(hits) == 1:
        return hits[0][1], hits[0][0]
    if not hits:
        return None, (
            f"no torch.nn.ModuleList of {expected} blocks was found on "
            f"{type(model).__name__}; the tried paths were {list(candidates)} and this "
            "build exposes no stack to hook, so its hidden states are unavailable"
        )
    return None, (
        f"{len(hits)} torch.nn.ModuleList candidates of {expected} blocks were found on "
        f"{type(model).__name__} ({[n for n, _ in hits][:6]}); hooking one of them would "
        "attribute a layer to a module this code cannot confirm is the transformer stack, "
        "so no hidden state is reported"
    )


# --------------------------------------------------------------------------- #
# The shared surface
# --------------------------------------------------------------------------- #
class HiddenQuery:
    """``hidden_batch`` and its bookkeeping, shared by the three subclasses.

    Mixed in ahead of run 2's adapter in the MRO. It adds methods and overrides
    nothing, so an instance is still exactly the run 2 adapter the extraction
    driver knows.
    """

    #: is the perturbed gene's own hidden state a function of its own input?
    #: False for a causal model that puts it at position 0, where the answer is
    #: structurally zero and must not be read as a measurement.
    hidden_self_position_defined: bool = True
    #: candidate dotted paths for the block stack, most specific first
    _stack_candidates: tuple[str, ...] = ()
    #: verbatim reason the last call could supply no hidden state, or None. The
    #: `exact_jvp` / `_jvp_error` contract: a measured absence, never a silent one.
    _hidden_error: str | None = None

    # ---- plan ------------------------------------------------------------ #
    def _model_key(self) -> str:
        return str(getattr(self, "name", ""))

    def _final_capture(self) -> tuple[torch.nn.Module | None, str | None]:
        """Module hooked for layer ``L`` instead of the last block, if any."""
        return None, None

    def _plan(self) -> HiddenPlan | None:
        """Resolve the block stack once, and record why if it cannot be resolved."""
        cached = getattr(self, "_hidden_plan", None)
        if cached is not None:
            return cached
        model = getattr(self, "model", None)
        if model is None:
            raise RuntimeError(
                f"{type(self).__name__}.hidden_batch was called before load(); there is no "
                "module to hook and a hidden state cannot be invented"
            )
        expected = C.N_LAYERS[self._model_key()]
        blocks, detail = _resolve_stack(model, expected, self._stack_candidates)
        if blocks is None:
            self._hidden_error = detail
            print(f"[hidden] {self._model_key()}: {detail}", flush=True)
            return None
        final_module, final_name = self._final_capture()
        plan = HiddenPlan(
            blocks=blocks,
            blocks_name=detail,
            n_layers=len(blocks),
            final_module=final_module,
            final_name=final_name,
        )
        self._hidden_plan = plan
        print(
            f"[hidden] {self._model_key()}: {plan.n_layers} blocks at "
            f"{type(model).__name__}.{plan.blocks_name}"
            + (f"; layer {plan.n_layers} read at .{final_name}" if final_name else ""),
            flush=True,
        )
        return plan

    # ---- provenance ------------------------------------------------------ #
    def hidden_source(self, layer: int) -> str:
        """Qualified module path layer ``layer`` was read from, for the manifest.

        The analog of run 2's ``gene_embedding_source``: the block records where
        each number came from instead of asserting where it should have come
        from, which is the only way a mis-attributed hook is visible after the
        fact.
        """
        plan = self._plan()
        if plan is None:
            return f"unavailable: {self._hidden_error}"
        root = type(self.model).__name__
        layer = int(layer)
        if layer == 0:
            return (
                f"{root}.{plan.blocks_name}.0 (forward pre-hook, module input): the "
                "post-embedding state that enters transformer block 1"
            )
        if layer == plan.n_layers and plan.final_module is not None:
            return (
                f"{root}.{plan.final_name} (module output): the trunk output that "
                f"transformer block {layer} feeds, and the tensor the output head reads"
            )
        return (
            f"{root}.{plan.blocks_name}.{layer - 1} (module output): the output of "
            f"transformer block {layer}"
        )

    def hidden_layer_definition(self) -> str:
        """Prose for the block manifest's ``layer_definition`` field."""
        return (
            "layer l is the output of transformer block l, 1-based; layer 0 is the "
            "post-embedding state entering block 1, captured by a forward pre-hook, and "
            "is control C2 rather than a sweep point. Every layer's module path is "
            "recorded separately under hidden_source."
        )

    def hidden_notes(self) -> dict:
        """What the last call measured about its own wiring, for the manifest."""
        return dict(getattr(self, "_hidden_notes", {}))

    # ---- shared helpers -------------------------------------------------- #
    def _check_layers(self, layers: Sequence[int], plan: HiddenPlan) -> list[int]:
        out: list[int] = []
        for raw in layers:
            layer = int(raw)
            if layer < 0 or layer > plan.n_layers:
                raise RuntimeError(
                    f"layer {layer} was requested of a {plan.n_layers}-block "
                    f"{self._model_key()}; layers run 0 (post-embedding) to "
                    f"{plan.n_layers} (the last block) and anything else would be read "
                    "off a module that is not a transformer block"
                )
            if layer not in out:
                out.append(layer)
        if not out:
            raise RuntimeError("hidden_batch was asked for no layers at all")
        return out

    def _take(self, store: dict[int, np.ndarray], layers: Sequence[int], rows: int, width: int):
        """Pull one forward pass's captures out of the store, checking each fired."""
        got = {}
        for layer in layers:
            arr = store.get(layer)
            if arr is None:
                raise RuntimeError(
                    f"the hook for layer {layer} never fired during a "
                    f"{self._model_key()} forward pass; the module it is attached to is "
                    "not on the path this query takes, so that layer's block would be "
                    "silently missing"
                )
            if arr.shape[1] != rows:
                raise RuntimeError(
                    f"layer {layer} captured {arr.shape[1]} positions for a pass of "
                    f"{rows} genes; the states cannot be attributed to genes"
                )
            if arr.shape[2] != width:
                raise RuntimeError(
                    f"layer {layer} captured width {arr.shape[2]}, but the "
                    f"{self._model_key()} checkpoint records d_model={width}; the hook "
                    "is on the wrong module"
                )
            got[layer] = arr
        return got

    def _reset(self) -> None:
        self._hidden_error = None
        self._hidden_notes = {
            "model": self._model_key(),
            "self_position_defined": bool(self.hidden_self_position_defined),
        }

    def hidden_batch(
        self,
        x_native: np.ndarray,
        layers: Sequence[int],
        perturbed_col: int | None = None,
    ) -> dict[int, np.ndarray] | None:
        """``{layer: (B, n_readout, d)}`` in read-out order, or ``None`` with a reason.

        ``x_native`` is ``(B, n_readout)`` on this model's own input scale, exactly
        what ``forward_batch`` takes, and the rows come back in the read-out gene
        order of ``readout_genes.parquet`` --- the same order ``forward_batch``
        returns --- so a hidden state and the response it produces are joined on
        the same axis. There is no ``mask`` argument: the probe reads the visible
        regime only, which is the regime every anchor in run 3 was scored at.
        """
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# scGPT
# --------------------------------------------------------------------------- #
class ScGPTHiddenQuery(HiddenQuery, _r2.ScGPTQuery):
    """scGPT hidden states, with the tile composition pinned to the perturbed gene.

    scGPT is the model the paper's response arm fails on outright, so it is the
    one this probe exists for. It is also the one with the composition trap: the
    production layout is ``tiled``, and a tile that does not already hold the
    perturbed gene gets it appended.
    """

    _stack_candidates = ("transformer_encoder.layers",)

    def _final_capture(self) -> tuple[torch.nn.Module | None, str | None]:
        """Layer ``L`` is read at ``transformer_encoder``, not at its last block.

        That module's output is by construction the ``out`` in
        ``ScGPTQuery._run`` --- the tensor ``decoder`` turns into the recorded
        response --- so hooking it makes the last layer the state the head reads
        whether or not the encoder carries a final normalisation, instead of a
        state that merely usually equals it.
        """
        return self.model.transformer_encoder, "transformer_encoder"

    # ---- trap 1: composition is a function of the perturbed gene alone ---- #
    def tile_composition(self, perturbed_col: int | None) -> list[np.ndarray]:
        """Read-out column indices entering each tile, for this perturbed gene.

        Reproduces ``ScGPTQuery.forward_batch`` exactly, including the append of
        the perturbed gene to any tile that lacks it, and depends on *nothing
        else*: not on the values, not on the anchor, not on call order. That is
        what makes the baseline pass and the perturbed pass comparable --- two
        calls with the same ``perturbed_col`` are guaranteed to have run the same
        sequences, and :meth:`tile_token_signature` lets the caller check it
        rather than trust it.
        """
        if self.tiles is None:
            return [np.arange(self.supplied_tokens.size, dtype=np.int64)]
        out: list[np.ndarray] = []
        for tile in self.tiles:
            cols = tile
            if perturbed_col is not None and int(perturbed_col) not in set(tile.tolist()):
                cols = np.concatenate([tile, [int(perturbed_col)]])
            out.append(np.asarray(cols, dtype=np.int64))
        return out

    def tile_token_signature(self, perturbed_col: int | None) -> tuple[tuple[int, ...], ...]:
        """The scGPT vocabulary token ids entering each tile, as a comparable tuple."""
        if self.tiles is None:
            return composition_signature([np.asarray(self.supplied_tokens, dtype=np.int64)])
        return composition_signature(
            [self.readout_tokens[cols] for cols in self.tile_composition(perturbed_col)]
        )

    def assert_same_composition(
        self, previous: tuple[tuple[int, ...], ...], perturbed_col: int | None
    ) -> None:
        """Raise unless this query's tiles match a signature taken earlier."""
        current = self.tile_token_signature(perturbed_col)
        if current != previous:
            check_token_composition(
                [np.asarray(t, dtype=np.int64) for t in current],
                [np.asarray(t, dtype=np.int64) for t in previous],
                f"scGPT tile composition at perturbed_col={perturbed_col}",
            )

    # ---- capture --------------------------------------------------------- #
    def _hidden_pass(
        self, tokens: np.ndarray, values: np.ndarray, plan: HiddenPlan, layers: Sequence[int]
    ) -> dict[int, np.ndarray]:
        """One supplied set through the production forward path, hooks attached.

        ``self._run`` is called rather than ``self.model._encode`` alone, so the
        captured state comes from byte-identical code to the one that produced
        run 2's responses; the decoder call it also makes is a small MLP and is
        discarded.
        """
        width = C.D_MODEL[self._model_key()]
        tok = torch.as_tensor(tokens, device=self.device, dtype=torch.long)
        acc: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
        with _capture(plan, layers) as store:
            for _, chunk in self._batches(values):
                store.clear()
                t = tok.unsqueeze(0).expand(chunk.shape[0], -1)
                v = torch.as_tensor(chunk, device=self.device, dtype=torch.float32)
                with torch.no_grad():
                    self._run(t, v)
                got = self._take(store, layers, int(tokens.size), width)
                for layer in layers:
                    acc[layer].append(got[layer])
        return {layer: np.concatenate(acc[layer], axis=0) for layer in layers}

    def hidden_batch(
        self,
        x_native: np.ndarray,
        layers: Sequence[int],
        perturbed_col: int | None = None,
    ) -> dict[int, np.ndarray] | None:
        plan = self._plan()
        if plan is None:
            return None
        self._reset()
        wanted = self._check_layers(layers, plan)
        x = np.atleast_2d(np.asarray(x_native, dtype=np.float32))
        if x.shape[1] != self.n_readout:
            raise RuntimeError(
                f"hidden_batch received {x.shape[1]} input columns but scGPT was "
                f"configured for {self.n_readout} read-out genes"
            )

        if self.tiles is None:
            if self.supplied_mode == "universe":
                values = np.repeat(
                    self.universe_base_native[None, :], x.shape[0], axis=0
                ).astype(np.float32)
                values[:, self.readout_slots] = x
            else:
                values = x.copy()
            parts = self._hidden_pass(self.supplied_tokens, values, plan, wanted)
            out = {
                layer: np.ascontiguousarray(parts[layer][:, self.readout_slots, :])
                for layer in wanted
            }
        else:
            cols_list = self.tile_composition(perturbed_col)
            per_layer: dict[int, list[np.ndarray]] = {layer: [] for layer in wanted}
            for cols in cols_list:
                got = self._hidden_pass(
                    self.readout_tokens[cols], x[:, cols], plan, wanted
                )
                for layer in wanted:
                    per_layer[layer].append(got[layer])
            out = {
                layer: reassemble_tiles(
                    per_layer[layer], cols_list, self.tile_owned, self.n_readout
                )
                for layer in wanted
            }

        signature = self.tile_token_signature(perturbed_col)
        self.last_hidden_composition = signature
        self._hidden_notes.update(
            {
                "supplied_mode": self.supplied_mode,
                "n_passes": len(signature),
                "composition_depends_on_perturbed_col": self.tiles is not None,
                "composition_note": (
                    "the perturbed gene is appended to any tile that does not already "
                    "hold it, so two queries with different perturbed genes are not the "
                    "same sequence; composition is a function of perturbed_col alone and "
                    "is published as last_hidden_composition for the caller to assert on"
                ),
                "layers": list(wanted),
                "hidden_source": {layer: self.hidden_source(layer) for layer in wanted},
            }
        )
        return out


# --------------------------------------------------------------------------- #
# scPRINT
# --------------------------------------------------------------------------- #
class ScPrintHiddenQuery(HiddenQuery, _r2.ScPrintQuery):
    """scPRINT hidden states: one sequence, one permutation, no composition trap.

    Every read-out gene fits in a single sequence, so nothing about the input
    depends on which gene was perturbed --- the composition trap does not arise
    here, and that is recorded rather than left implicit. What does arise is the
    order: the sequence is sorted by vocabulary position and carries scPRINT's
    cell-embedding tokens ahead of the genes, so the capture is sliced to its
    trailing ``n_readout`` positions and then unscattered with the adapter's own
    ``readout_to_seq``.
    """

    _stack_candidates = (
        "transformer.blocks",
        "transformer.layers",
        "transformer.encoder.layers",
        "transformer.transformer.blocks",
    )

    def hidden_batch(
        self,
        x_native: np.ndarray,
        layers: Sequence[int],
        perturbed_col: int | None = None,
    ) -> dict[int, np.ndarray] | None:
        plan = self._plan()
        if plan is None:
            return None
        self._reset()
        wanted = self._check_layers(layers, plan)
        width = C.D_MODEL[self._model_key()]
        x = np.atleast_2d(np.asarray(x_native, dtype=np.float32))
        if x.shape[1] != self.n_readout:
            raise RuntimeError(
                f"hidden_batch received {x.shape[1]} input columns but scPRINT was "
                f"configured for {self.n_readout} read-out genes"
            )
        seq = x[:, self.seq_order]

        acc: dict[int, list[np.ndarray]] = {layer: [] for layer in wanted}
        prefix = None
        with _capture(plan, wanted) as store:
            for _, chunk in self._batches(seq):
                store.clear()
                expr = torch.as_tensor(chunk, device=self.device, dtype=torch.float32)
                with torch.no_grad():
                    self._forward_many(expr, None)
                for layer in wanted:
                    arr = store.get(layer)
                    if arr is None:
                        raise RuntimeError(
                            f"the hook for scPRINT layer {layer} never fired; that layer "
                            "would be silently absent from the block"
                        )
                    # The cell-embedding tokens scPRINT prepends are not genes. Their
                    # count is read off the captured length rather than assumed, and a
                    # sequence shorter than the read-out set is a wiring error, not a
                    # short cell.
                    offset = arr.shape[1] - self.n_readout
                    if offset < 0:
                        raise RuntimeError(
                            f"scPRINT layer {layer} captured {arr.shape[1]} positions for "
                            f"{self.n_readout} read-out genes; the trunk is carrying "
                            "fewer tokens than genes were supplied, so no slice of it is "
                            "the gene block"
                        )
                    if prefix is None:
                        prefix = offset
                    elif offset != prefix:
                        raise RuntimeError(
                            f"scPRINT layer {layer} has a {offset}-token prefix where an "
                            f"earlier layer had {prefix}; the layers disagree about where "
                            "the genes start and one of the slices is wrong"
                        )
                    if arr.shape[2] != width:
                        raise RuntimeError(
                            f"scPRINT layer {layer} captured width {arr.shape[2]}, but the "
                            f"checkpoint records d_model={width}; the hook is on the wrong "
                            "module"
                        )
                    acc[layer].append(arr[:, offset:, :])

        out = {
            layer: reassemble_sequence(
                np.concatenate(acc[layer], axis=0), self.readout_to_seq, self.n_readout
            )
            for layer in wanted
        }
        self.last_hidden_composition = composition_signature(
            [np.asarray(self.seq_positions, dtype=np.int64)]
        )
        self._hidden_notes.update(
            {
                "supplied_mode": "readout",
                "n_passes": 1,
                "composition_depends_on_perturbed_col": False,
                "composition_note": (
                    "all read-out genes fit one sequence, so the baseline and perturbed "
                    "passes are the same sequence by construction and differ only in the "
                    "value at the perturbed gene"
                ),
                "cell_embedding_prefix_tokens": int(prefix if prefix is not None else -1),
                "layers": list(wanted),
                "hidden_source": {layer: self.hidden_source(layer) for layer in wanted},
            }
        )
        return out


# --------------------------------------------------------------------------- #
# TranscriptFormer
# --------------------------------------------------------------------------- #
class TranscriptFormerHiddenQuery(HiddenQuery, _r2.TranscriptFormerQuery):
    """TranscriptFormer hidden states, under the ordering the query itself imposes.

    Two properties of this model shape the whole method. It is autoregressive, so
    the ordering is part of the query and the adapter forces the perturbed gene to
    position 0 of every pass; a baseline and a perturbed pass are only comparable
    when they were built with the same ``perturbed_col``, which is why that
    argument is not optional in practice. And nothing precedes position 0, so the
    perturbed gene's own state is a function of its identity alone: its delta is
    structurally zero, not measurably zero, and it is written as ``nan`` so it
    cannot be mistaken for a propagation measurement. C1 --- decode the perturbed
    gene from its own position --- is therefore unavailable here, and that is a
    recorded absence rather than a passing control.
    """

    hidden_self_position_defined = False

    _stack_candidates = (
        "encoder.layers",
        "encoder.blocks",
        "transformer.layers",
        "transformer.blocks",
        "model.layers",
        "blocks",
        "layers",
    )

    def hidden_batch(
        self,
        x_native: np.ndarray,
        layers: Sequence[int],
        perturbed_col: int | None = None,
    ) -> dict[int, np.ndarray] | None:
        plan = self._plan()
        if plan is None:
            return None
        self._reset()
        wanted = self._check_layers(layers, plan)
        width = C.D_MODEL[self._model_key()]
        x = np.atleast_2d(np.asarray(x_native, dtype=np.float32))
        if x.shape[1] != self.n_readout:
            raise RuntimeError(
                f"hidden_batch received {x.shape[1]} input columns but TranscriptFormer "
                f"was configured for {self.n_readout} read-out genes"
            )

        chunks = self._query_chunks(perturbed_col)
        cols_list: list[np.ndarray] = []
        owned_list: list[np.ndarray] = []
        per_layer: dict[int, list[np.ndarray]] = {layer: [] for layer in wanted}

        for ci, rows in enumerate(chunks):
            n = int(rows.size)
            cols_list.append(np.asarray(rows, dtype=np.int64))
            # Ownership is run 2's rule verbatim: the shared context block is read
            # from the first pass only, every later pass owns just its own body.
            owned_list.append(
                np.arange(n, dtype=np.int64)
                if ci == 0
                else np.arange(self.context_block, n, dtype=np.int64)
            )
            acc: dict[int, list[np.ndarray]] = {layer: [] for layer in wanted}
            with _capture(plan, wanted) as store:
                for _, chunk in self._batches(x[:, rows]):
                    store.clear()
                    with torch.no_grad():
                        self._forward_rows(rows, chunk)
                    for layer in wanted:
                        arr = store.get(layer)
                        if arr is None:
                            raise RuntimeError(
                                f"the hook for TranscriptFormer layer {layer} never fired; "
                                "that layer would be silently absent from the block"
                            )
                        # The aux (assay) tokens LEAD, and this is not inferred: the
                        # checkpoint's own forward builds the trunk input as
                        # ``cat([aux_embeddings, right_shifted_gene_embeddings], dim=1)``
                        # and then recovers the genes with
                        # ``transformer_output[:, num_aux_tokens:, :]``. We take the model's
                        # own slice rather than a guess, because guessing the wrong end
                        # offsets every gene's state by one and yields a perfectly
                        # well-shaped array belonging to the wrong genes.
                        n_aux = int(arr.shape[1]) - int(self.seq_len)
                        if n_aux < 0 or n_aux > C.MAX_AUX_TOKENS:
                            raise RuntimeError(
                                f"TranscriptFormer layer {layer} captured {arr.shape[1]} "
                                f"positions against a seq_len of {self.seq_len}, a "
                                f"difference of {n_aux} that is not a plausible number of "
                                "leading auxiliary tokens; the hook is on the wrong module "
                                "or the checkpoint's layout has changed"
                            )
                        arr = arr[:, n_aux:, :]
                        if arr.shape[2] != width:
                            raise RuntimeError(
                                f"TranscriptFormer layer {layer} captured width "
                                f"{arr.shape[2]}, but the checkpoint records "
                                f"model_dim={width}; the hook is on the wrong module"
                            )
                        # positions past n are pad tokens, exactly as forward_batch
                        # slices mu to [:, :n]
                        acc[layer].append(arr[:, :n, :])
            for layer in wanted:
                per_layer[layer].append(np.concatenate(acc[layer], axis=0))

        out = {
            layer: reassemble_tiles(per_layer[layer], cols_list, owned_list, self.n_readout)
            for layer in wanted
        }

        self_note = "the perturbed gene's own position is measurable"
        if perturbed_col is not None:
            for layer in wanted:
                out[layer][:, int(perturbed_col), :] = np.nan
            self_note = (
                f"read-out position {int(perturbed_col)} is the perturbed gene, forced to "
                "sequence position 0 where a causal model conditions on nothing; its state "
                "cannot depend on any input value, so it is written as nan rather than the "
                "structural zero that would read as a measured absence of propagation"
            )

        self.last_hidden_composition = composition_signature(
            [self.readout_tokens[cols] for cols in cols_list]
        )
        self._hidden_notes.update(
            {
                "supplied_mode": "readout_in_passes",
                "n_passes": len(cols_list),
                "composition_depends_on_perturbed_col": True,
                "composition_note": (
                    "the perturbed gene is moved to position 0 of every pass, so the "
                    "ordering --- and therefore the conditioning prefix of every other "
                    "gene --- is a function of perturbed_col; a baseline pass must be run "
                    "with the same perturbed_col as the pass it is subtracted from"
                ),
                "self_position_defined": False,
                "self_position_note": self_note,
                "layers": list(wanted),
                "hidden_source": {layer: self.hidden_source(layer) for layer in wanted},
            }
        )
        return out


# --------------------------------------------------------------------------- #
# Construction, mirroring scfm_pfm.models.queries.build
# --------------------------------------------------------------------------- #
BUILDERS = {
    "scgpt": ScGPTHiddenQuery,
    "scprint": ScPrintHiddenQuery,
    "transcriptformer": TranscriptFormerHiddenQuery,
}


def build(model: str, cache, universe, vocab, device: str = "cuda"):
    """One hidden-state adapter, by run 2's model key and with run 2's signature.

    AIDO.Cell is absent by decision, not by oversight: it is not on this machine,
    it scored at zero in run 3, and it costs roughly 40x scGPT per forward pass,
    so a hidden-state sweep over it would buy nothing the other three do not
    already answer. Asking for it says so rather than raising a bare KeyError.
    """
    if model not in BUILDERS:
        raise KeyError(
            f"no hidden-state adapter for {model!r}; run 5 covers "
            f"{sorted(BUILDERS)}. AIDO.Cell is deliberately out of scope (not local, "
            "scored at zero in run 3, ~40x scGPT per forward pass)"
        )
    return BUILDERS[model](cache=cache, universe=universe, vocab=vocab, device=device)
