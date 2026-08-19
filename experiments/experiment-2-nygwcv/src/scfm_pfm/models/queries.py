"""The batched query interface run 2 needs, on top of run 1's four response adapters.

Run 1 answered "what does this model predict for one cell, and what is one
Jacobian column". Run 2 asks the same model tens of thousands of questions that
differ in a single input coordinate, so the interface it needs is different in
three ways:

* **batched** -- a value grid and its finite-difference neighbours go through in
  one forward pass rather than one at a time;
* **anchored in a common unit** -- every query is specified in CP10K (counts per
  10,000), the one unit all four models can be given a defensible map into, and
  each adapter converts CP10K to its own input scale;
* **fixed supplied set** -- the genes that enter the forward pass are pinned per
  (model, cell line) and recorded, because two of these models end in a softmax
  across exactly those genes.

The driver in :mod:`scfm_pfm.features.extract` only ever calls the methods
declared here, so it is model-agnostic and the model-specific parts stay in one
class each. Run 1's classes are subclassed, not edited: the loading path, the
attention substitutions and the recorded contract are inherited unchanged.

The CP10K map, per model
------------------------
``aido_cell``, ``scprint``, ``transcriptformer``
    the input is a raw count, and the anchor is a virtual cell of 10,000 counts,
    so a CP10K value *is* the count. TranscriptFormer additionally clips at 30,
    which is its own training-time clip and is recorded as saturation rather
    than hidden.
``scgpt``
    the input is a quantile bin index. The bin edges are built once per cell line
    from that line's mean control profile and then held fixed for every query, so
    the CP10K to bin map is one recorded monotone function rather than a
    per-query re-binning (which would make two queries incomparable).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from .aido_model import AidoCellResponse, N_GENES as AIDO_N_GENES
from .scgpt_model import ScGPTResponse, quantile_bin_edges
from .scprint_model import ScPrintResponse
from .transcriptformer_model import TranscriptFormerResponse


@dataclass
class QueryLayout:
    """What entered the forward pass, recorded per (model, cell line)."""

    model: str
    cell_line: str
    n_readout: int
    n_supplied: int
    supplied_mode: str
    #: how the supplied set is split across forward passes, when it is
    n_passes: int = 1
    context_block: int = 0
    #: model-specific notes that belong in the block manifest
    notes: dict = field(default_factory=dict)


class QueryModel:
    """Methods the extraction driver relies on. Implemented once per model."""

    #: default number of queries per forward batch; tuned per model at run time
    batch_size = 8
    #: does raising an input value raise that gene's own predicted output at all?
    #: false for an autoregressive model, whose first-position gene cannot see
    #: its own count
    self_response_defined = True

    def configure(self, readout: pd.DataFrame, anchor_cp10k: np.ndarray, cell_line: str) -> QueryLayout:
        raise NotImplementedError

    def to_native(self, cp10k: np.ndarray) -> np.ndarray:
        """Elementwise CP10K -> this model's own input scale."""
        raise NotImplementedError

    def forward_batch(
        self,
        x_native: np.ndarray,
        perturbed_col: int | None = None,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """(B, n_readout) inputs -> (B, n_readout) responses, in read-out order."""
        raise NotImplementedError

    def mask_spec(self) -> dict | None:
        """How this model marks a value as unknown, or None if it cannot."""
        return None

    def gene_embeddings(self) -> tuple[np.ndarray, str]:
        """(n_readout, d) vocabulary embedding for the read-out genes, and its source."""
        raise NotImplementedError

    def exact_jvp(self, x_native: np.ndarray, col: int) -> np.ndarray | None:
        """Forward-mode derivative of the response w.r.t. input coordinate ``col``."""
        return None

    # -- shared helpers ----------------------------------------------------
    def _batches(self, x: np.ndarray, size: int | None = None):
        size = int(size or self.batch_size)
        for start in range(0, x.shape[0], size):
            yield start, x[start : start + size]


# --------------------------------------------------------------------------- #
# scGPT
# --------------------------------------------------------------------------- #
class ScGPTQuery(ScGPTResponse, QueryModel):
    """scGPT with fixed quantile bin edges and a choice of supplied-set width.

    scGPT was trained at a maximum sequence length of 1,200 but carries no
    positional encoding over genes, so a longer input is architecturally
    admissible and run 1 used the whole 7,444-gene universe. Run 2 treats the
    width as a measured choice: ``readout`` (the expressed set), ``universe``
    (run 1's width) and ``tiled`` (passes of at most 1,200 sharing a fixed
    prefix) all produce a full response vector, and the extraction reports how
    far apart they are before the production sweep commits to one.
    """

    batch_size = 8
    TILE_LIMIT = 1200
    TILE_PREFIX = 256

    def configure(
        self,
        readout: pd.DataFrame,
        anchor_cp10k: np.ndarray,
        cell_line: str,
        supplied_mode: str = "readout",
    ) -> QueryLayout:
        self.cell_line = cell_line
        self.supplied_mode = supplied_mode
        self.readout_tokens = readout["scgpt_position"].to_numpy(dtype=np.int64)
        self.n_readout = int(self.readout_tokens.size)

        # bin edges: fixed once per cell line from the anchor profile, so the
        # CP10K -> bin map is the same function for every query
        log_anchor = np.log1p(np.asarray(anchor_cp10k, dtype=np.float64))
        self.bin_edges = quantile_bin_edges(log_anchor, self.n_bins)
        self.bin_source = (
            f"{self.n_bins}-bin quantile edges of log1p(mean control CP10K) over the "
            f"{self.n_readout} read-out genes of {cell_line}, held fixed for every query"
        )

        if supplied_mode == "universe":
            self.supplied_tokens = self.universe["scgpt_position"].to_numpy(dtype=np.int64)
            token_to_slot = {int(t): i for i, t in enumerate(self.supplied_tokens)}
            self.readout_slots = np.array(
                [token_to_slot[int(t)] for t in self.readout_tokens], dtype=np.int64
            )
            # universe genes outside the read-out set enter at their anchor value,
            # which for a gene the read-out set excludes is its own control level
            self.universe_anchor_native = None
            self.tiles = None
            n_supplied, n_passes = int(self.supplied_tokens.size), 1
        elif supplied_mode == "readout":
            self.supplied_tokens = self.readout_tokens
            self.readout_slots = np.arange(self.n_readout, dtype=np.int64)
            self.tiles = None
            n_supplied, n_passes = self.n_readout, 1
        elif supplied_mode == "tiled":
            order = np.argsort(-np.asarray(anchor_cp10k, dtype=np.float64), kind="stable")
            prefix = order[: self.TILE_PREFIX]
            body = order[self.TILE_PREFIX :]
            per = self.TILE_LIMIT - self.TILE_PREFIX
            self.tiles = [
                np.concatenate([prefix, body[s : s + per]]) for s in range(0, body.size, per)
            ] or [prefix]
            self.supplied_tokens = self.readout_tokens
            self.readout_slots = np.arange(self.n_readout, dtype=np.int64)
            n_supplied, n_passes = self.n_readout, len(self.tiles)
        else:
            raise ValueError(f"unknown supplied_mode {supplied_mode!r}")

        return QueryLayout(
            model="scgpt",
            cell_line=cell_line,
            n_readout=self.n_readout,
            n_supplied=n_supplied,
            supplied_mode=supplied_mode,
            n_passes=n_passes,
            context_block=self.TILE_PREFIX if supplied_mode == "tiled" else 0,
            notes={
                "training_max_seq_len": int(self.args.get("max_seq_len", 1200)),
                "positional_encoding_over_genes": False,
                "bin_edges_source": self.bin_source,
                "n_bins": int(self.n_bins),
                "tile_limit": self.TILE_LIMIT if supplied_mode == "tiled" else None,
                "tile_prefix_genes": self.TILE_PREFIX if supplied_mode == "tiled" else None,
                "tile_note": (
                    "every pass opens with the same highest-expression prefix and the "
                    "perturbed gene is placed at slot 0 of that prefix, so the context "
                    "differs between queries by exactly one gene"
                    if supplied_mode == "tiled"
                    else None
                ),
            },
        )

    def to_native(self, cp10k: np.ndarray) -> np.ndarray:
        """CP10K -> smooth (differentiable) quantile bin index on the fixed edges."""
        v = np.log1p(np.asarray(cp10k, dtype=np.float64))
        edges = self.bin_edges
        return np.interp(v, edges, np.arange(1, edges.size + 1), left=0.0).astype(np.float32)

    def _run(self, tokens: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        pad = torch.zeros(values.shape, dtype=torch.bool, device=values.device)
        out = self.model._encode(tokens, values, pad)
        return self.model.decoder(out)["pred"]

    def forward_batch(
        self, x_native: np.ndarray, perturbed_col: int | None = None, mask: np.ndarray | None = None
    ) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x_native, dtype=np.float32))
        b = x.shape[0]
        out = np.empty((b, self.n_readout), dtype=np.float32)
        mask_value = float(self.mask_spec()["value"])  # type: ignore[index]

        if self.tiles is None:
            tokens = torch.as_tensor(self.supplied_tokens, device=self.device)
            if self.supplied_mode == "universe":
                values_full = np.repeat(self.universe_base_native[None, :], b, axis=0)
                values_full[:, self.readout_slots] = x
            else:
                values_full = x.copy()
            if mask is not None:
                values_full[:, self.readout_slots] = np.where(
                    mask, mask_value, values_full[:, self.readout_slots]
                )
            for start, chunk in self._batches(values_full):
                t = tokens.unsqueeze(0).expand(chunk.shape[0], -1)
                v = torch.as_tensor(chunk, device=self.device)
                with torch.no_grad():
                    pred = self._run(t, v).float().cpu().numpy()
                out[start : start + chunk.shape[0]] = pred[:, self.readout_slots]
            return out

        # tiled: each pass carries the shared prefix plus one body chunk
        filled = np.zeros(self.n_readout, dtype=bool)
        for tile in self.tiles:
            cols = tile if perturbed_col is None or perturbed_col in set(tile.tolist()) else np.concatenate(
                [[perturbed_col], tile[1:]]
            )
            tok = torch.as_tensor(self.readout_tokens[cols], device=self.device)
            vals = x[:, cols].copy()
            if mask is not None:
                vals = np.where(mask[:, cols], mask_value, vals)
            for start, chunk in self._batches(vals):
                t = tok.unsqueeze(0).expand(chunk.shape[0], -1)
                v = torch.as_tensor(chunk, device=self.device)
                with torch.no_grad():
                    pred = self._run(t, v).float().cpu().numpy()
                # a gene appearing in several tiles keeps its first pass's value
                fresh = ~filled[cols]
                out[start : start + chunk.shape[0]][:, cols[fresh]] = pred[:, fresh]
            filled[cols] = True
        if not filled.all():
            raise RuntimeError(f"{int((~filled).sum())} read-out genes never entered a tile")
        return out

    def mask_spec(self) -> dict:
        return {
            "kind": "value",
            "value": float(self.args.get("mask_value", -1)),
            "rate": float(self.args.get("mask_ratio", 0.4) or 0.4),
            "source": "scGPT args.json (mask_value / mask_ratio) with the published defaults",
        }

    def gene_embeddings(self) -> tuple[np.ndarray, str]:
        emb = self.model.encoder.embedding.weight.detach()
        return (
            emb[torch.as_tensor(self.readout_tokens, device=emb.device)].float().cpu().numpy(),
            "TransformerModel.encoder.embedding (gene token embedding table)",
        )

    def exact_jvp(self, x_native: np.ndarray, col: int) -> np.ndarray | None:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        if self.tiles is not None:
            return None  # a tiled pass is not one differentiable function of the input
        tokens = torch.as_tensor(self.supplied_tokens, device=self.device).unsqueeze(0)
        base = torch.as_tensor(np.atleast_2d(x_native), device=self.device, dtype=torch.float32)
        slots = torch.as_tensor(self.readout_slots, device=self.device)

        def fn(v: torch.Tensor) -> torch.Tensor:
            return self._run(tokens, v)[0, slots]

        tangent = torch.zeros_like(base)
        tangent[0, int(col)] = 1.0
        try:
            with sdpa_kernel(SDPBackend.MATH):
                _, jvp = torch.func.jvp(fn, (base,), (tangent,))
            return jvp.detach().float().cpu().numpy()
        except Exception as exc:  # noqa: BLE001
            self._jvp_error = f"{type(exc).__name__}: {exc}"
            return None


# --------------------------------------------------------------------------- #
# AIDO.Cell / GB.Cell-100M
# --------------------------------------------------------------------------- #
class AidoQuery(AidoCellResponse, QueryModel):
    """AIDO.Cell: the supplied set is the whole transcriptome by construction.

    Position *i* of the input is gene *i* of the checkpoint's fixed 19,264-gene
    order, so there is no gene selection to make: the read-out genes carry their
    anchor value and every other position stays at zero, exactly as an unmeasured
    gene does.
    """

    batch_size = 4

    def configure(self, readout: pd.DataFrame, anchor_cp10k: np.ndarray, cell_line: str) -> QueryLayout:
        self.cell_line = cell_line
        self.readout_positions = readout["aido_cell_position"].to_numpy(dtype=np.int64)
        self.n_readout = int(self.readout_positions.size)
        return QueryLayout(
            model="aido_cell",
            cell_line=cell_line,
            n_readout=self.n_readout,
            n_supplied=AIDO_N_GENES,
            supplied_mode="fixed_transcriptome",
            notes={
                "gene_order": "OS_scRNA_gene_index.19264.tsv, a model constraint",
                "unsupplied_positions": "left at zero, as an unmeasured gene is",
                "sequence_length": AIDO_N_GENES + 2,
            },
        )

    def to_native(self, cp10k: np.ndarray) -> np.ndarray:
        return np.asarray(cp10k, dtype=np.float32)  # count of a 10,000-count cell

    def _scatter(self, x: np.ndarray) -> np.ndarray:
        full = np.zeros((x.shape[0], AIDO_N_GENES), dtype=np.float32)
        full[:, self.readout_positions] = x
        return full

    def forward_batch(
        self, x_native: np.ndarray, perturbed_col: int | None = None, mask: np.ndarray | None = None
    ) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x_native, dtype=np.float32))
        if mask is not None:
            spec = self.mask_spec()
            if spec is None:
                raise RuntimeError("aido_cell has no recorded mask convention")
            x = np.where(mask, float(spec["value"]), x)
        full = self._scatter(x)
        out = np.empty((x.shape[0], self.n_readout), dtype=np.float32)
        for start, chunk in self._batches(full):
            counts = torch.as_tensor(chunk, device=self.device, dtype=torch.float32)
            with torch.no_grad():
                logits = self.model(input_ids=self._transform(counts)).logits[:, :-2, 0]
            out[start : start + chunk.shape[0]] = (
                logits[:, torch.as_tensor(self.readout_positions, device=self.device)]
                .float()
                .cpu()
                .numpy()
            )
        return out

    def mask_spec(self) -> dict | None:
        return getattr(self, "_mask_spec", None)

    def gene_embeddings(self) -> tuple[np.ndarray, str]:
        emb = self.model.bert.embeddings.position_embeddings.weight.detach()
        idx = torch.as_tensor(self.readout_positions, device=emb.device)
        return (
            emb[idx].float().cpu().numpy(),
            "CellFoundation position embedding: gene identity IS the position in the "
            "fixed 19,264-gene order, so this table is the model's gene embedding",
        )


# --------------------------------------------------------------------------- #
# scPRINT
# --------------------------------------------------------------------------- #
class ScPrintQuery(ScPrintResponse, QueryModel):
    """scPRINT over a fixed supplied gene set, with its exact forward-mode derivative."""

    batch_size = 8

    def configure(self, readout: pd.DataFrame, anchor_cp10k: np.ndarray, cell_line: str) -> QueryLayout:
        self.cell_line = cell_line
        pos = readout["scprint_position"].to_numpy(dtype=np.int64)
        order = np.argsort(pos, kind="stable")  # the model reads a sorted gene list
        self.seq_positions = pos[order]
        self.readout_to_seq = np.empty(order.size, dtype=np.int64)
        self.readout_to_seq[order] = np.arange(order.size)
        self.n_readout = int(pos.size)
        return QueryLayout(
            model="scprint",
            cell_line=cell_line,
            n_readout=self.n_readout,
            n_supplied=self.n_readout,
            supplied_mode="readout",
            notes={
                "softmax_over": "the supplied gene positions, so a response is a share "
                "of the cell's counts and depends on which genes were supplied together",
                "vocabulary_size": len(getattr(self, "genes", [])),
            },
        )

    def to_native(self, cp10k: np.ndarray) -> np.ndarray:
        return np.asarray(cp10k, dtype=np.float32)

    def _forward_many(self, expr: torch.Tensor) -> torch.Tensor:
        gp = torch.as_tensor(self.seq_positions, device=self.device).unsqueeze(0).expand(
            expr.shape[0], -1
        )
        depth = expr.sum(dim=1)
        out = self.model.forward(
            gene_pos=gp,
            expression=expr,
            req_depth=depth,
            get_gene_emb=False,
            do_mvc=False,
            do_class=False,
        )
        return out["mean"] if isinstance(out, dict) else out

    def forward_batch(
        self, x_native: np.ndarray, perturbed_col: int | None = None, mask: np.ndarray | None = None
    ) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x_native, dtype=np.float32))
        if mask is not None:
            spec = self.mask_spec()
            if spec is None:
                raise RuntimeError("scprint has no recorded mask convention")
            x = np.where(mask, float(spec["value"]), x)
        seq = x[:, self.readout_to_seq.argsort()] if False else x[:, np.argsort(self.readout_to_seq)]
        out = np.empty((x.shape[0], self.n_readout), dtype=np.float32)
        for start, chunk in self._batches(seq):
            expr = torch.as_tensor(chunk, device=self.device, dtype=torch.float32)
            with torch.no_grad():
                mean = self._forward_many(expr).float().cpu().numpy()
            out[start : start + chunk.shape[0]] = mean[:, self.readout_to_seq]
        return out

    def mask_spec(self) -> dict | None:
        return getattr(self, "_mask_spec", None)

    def gene_embeddings(self) -> tuple[np.ndarray, str]:
        emb = self.model.gene_encoder.embeddings.weight.detach()
        idx = torch.as_tensor(self.seq_positions, device=emb.device)
        table = emb[idx].float().cpu().numpy()
        return table[self.readout_to_seq], "scPrint.gene_encoder.embeddings (frozen ESM-derived)"

    def exact_jvp(self, x_native: np.ndarray, col: int) -> np.ndarray | None:
        seq_order = np.argsort(self.readout_to_seq)
        base = torch.as_tensor(
            np.atleast_2d(np.asarray(x_native, dtype=np.float32))[:, seq_order],
            device=self.device,
            dtype=torch.float32,
        )
        out_idx = torch.as_tensor(self.readout_to_seq, device=self.device)

        def fn(e: torch.Tensor) -> torch.Tensor:
            return self._forward_many(e)[0, out_idx]

        tangent = torch.zeros_like(base)
        tangent[0, int(self.readout_to_seq[int(col)])] = 1.0
        try:
            _, jvp = torch.func.jvp(fn, (base,), (tangent,))
            return jvp.detach().float().cpu().numpy()
        except Exception as exc:  # noqa: BLE001
            self._jvp_error = f"{type(exc).__name__}: {exc}"
            return None


# --------------------------------------------------------------------------- #
# TranscriptFormer
# --------------------------------------------------------------------------- #
class TranscriptFormerQuery(TranscriptFormerResponse, QueryModel):
    """TranscriptFormer: autoregressive, so the ordering is part of the query.

    The perturbed gene is placed at position 0 of every pass, which makes every
    read-out gene downstream of it and therefore measurable. The cost is that the
    context differs between queries by that one gene; the fraction of pairs that
    would be structurally unmeasurable under a single fixed ordering is reported
    instead of being hidden.
    """

    batch_size = 8
    self_response_defined = False
    #: the model's own training-time count clip (data_config.clip_counts)
    COUNT_CLIP = 30.0

    def configure(self, readout: pd.DataFrame, anchor_cp10k: np.ndarray, cell_line: str) -> QueryLayout:
        self.cell_line = cell_line
        self.readout_tokens = readout["transcriptformer_position"].to_numpy(dtype=np.int64)
        self.n_readout = int(self.readout_tokens.size)
        anchor = np.asarray(anchor_cp10k, dtype=np.float64)
        gene_ids = readout["gene_id"].to_numpy().astype(str)
        self.anchor_order = np.lexsort((gene_ids, -anchor))  # descending anchor, id tie-break
        self.anchor_rank = np.empty(self.n_readout, dtype=np.int64)
        self.anchor_rank[self.anchor_order] = np.arange(self.n_readout)
        self.pad_id = int(self.gene_vocab_dict[self.model.data_config.gene_pad_token])
        from .transcriptformer_model import ASSAY_LABEL

        self.assay_token = int(self.aux_vocab["assay"][ASSAY_LABEL])
        chunks = self._query_chunks(perturbed_col=int(self.anchor_order[0]))
        return QueryLayout(
            model="transcriptformer",
            cell_line=cell_line,
            n_readout=self.n_readout,
            n_supplied=self.n_readout,
            supplied_mode="readout_in_passes",
            n_passes=len(chunks),
            context_block=self.context_block,
            notes={
                "ordering": "descending anchor expression, gene id as tie-break, with the "
                "perturbed gene moved to position 0",
                "seq_len": int(self.seq_len),
                "softmax_over": "the tokens of the same pass; the response is mu divided "
                "by that pass's total count so passes are comparable",
                "count_clip": self.COUNT_CLIP,
                "unmeasurable_fraction_fixed_ordering": self.unmeasurable_fraction(),
            },
        )

    def unmeasurable_fraction(self) -> float:
        """Share of ordered pairs a single fixed ordering could not measure.

        Under one fixed ordering, gene *j* responds to gene *i* only if *i* comes
        first, so exactly the pairs with rank(i) > rank(j) are structurally zero:
        half of all ordered pairs, minus the diagonal.
        """
        n = self.n_readout
        return float((n * (n - 1) / 2) / (n * (n - 1))) if n > 1 else float("nan")

    def to_native(self, cp10k: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(cp10k, dtype=np.float32), 0.0, self.COUNT_CLIP)

    def _query_chunks(self, perturbed_col: int | None) -> list[np.ndarray]:
        order = list(self.anchor_order)
        if perturbed_col is not None:
            order = [int(perturbed_col)] + [r for r in order if r != int(perturbed_col)]
        context = order[: self.context_block]
        body = order[self.context_block :]
        per = self.seq_len - len(context)
        return [
            np.array(context + body[s : s + per], dtype=np.int64) for s in range(0, len(body), per)
        ] or [np.array(context, dtype=np.int64)]

    def _forward_rows(self, rows: np.ndarray, values: np.ndarray) -> np.ndarray:
        """(B, len(rows)) counts -> (B, len(rows)) shares for one pass."""
        from transcriptformer.data.dataclasses import BatchData

        b, n = values.shape
        tok = torch.full((b, self.seq_len), self.pad_id, dtype=torch.long, device=self.device)
        tok[:, :n] = torch.as_tensor(
            self.readout_tokens[rows], dtype=torch.long, device=self.device
        ).unsqueeze(0).expand(b, -1)
        val = torch.zeros((b, self.seq_len), dtype=torch.float32, device=self.device)
        val[:, :n] = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        aux = torch.full((b, 1), self.assay_token, dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model(BatchData(gene_counts=val, gene_token_indices=tok, aux_token_indices=aux))
        mu = out["mu"].reshape(b, -1)[:, :n]
        total = val.sum(dim=1, keepdim=True)
        if bool((total <= 0).any()):
            raise RuntimeError("a TranscriptFormer pass carries no counts to divide")
        return (mu / total).float().cpu().numpy()

    def forward_batch(
        self, x_native: np.ndarray, perturbed_col: int | None = None, mask: np.ndarray | None = None
    ) -> np.ndarray:
        if mask is not None:
            raise RuntimeError(
                "TranscriptFormer is autoregressive: a gene never sees its own value, so "
                "it is natively in the masked regime and has no masked variant"
            )
        x = np.atleast_2d(np.asarray(x_native, dtype=np.float32))
        out = np.zeros((x.shape[0], self.n_readout), dtype=np.float32)
        filled = np.zeros(self.n_readout, dtype=bool)
        for ci, rows in enumerate(self._query_chunks(perturbed_col)):
            take = rows if ci == 0 else rows[self.context_block :]
            for start, chunk in self._batches(x[:, rows]):
                share = self._forward_rows(rows, chunk)
                sl = slice(start, start + chunk.shape[0])
                out[sl, take] = share[:, : len(rows)][:, (slice(None) if ci == 0 else slice(self.context_block, None))]
            filled[take] = True
        if not filled.all():
            raise RuntimeError(f"{int((~filled).sum())} read-out genes never entered a pass")
        return out

    def gene_embeddings(self) -> tuple[np.ndarray, str]:
        emb = self.model.gene_embedding.weight if hasattr(self.model, "gene_embedding") else None
        if emb is None:
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Embedding) and module.num_embeddings >= len(
                    self.gene_vocab_dict
                ):
                    emb = module.weight
                    source = f"Transcriptformer.{name} (ESM2-derived gene embedding table)"
                    break
        else:
            source = "Transcriptformer.gene_embedding"
        if emb is None:
            raise RuntimeError("no gene embedding table found on the TranscriptFormer model")
        idx = torch.as_tensor(self.readout_tokens, device=emb.device)
        return emb.detach()[idx].float().cpu().numpy(), source


BUILDERS = {
    "scgpt": ScGPTQuery,
    "aido_cell": AidoQuery,
    "scprint": ScPrintQuery,
    "transcriptformer": TranscriptFormerQuery,
}


def build(model: str, cache, universe: pd.DataFrame, vocab: pd.DataFrame, device: str = "cuda"):
    return BUILDERS[model](cache=cache, universe=universe, vocab=vocab, device=device)


def partition_mask(n: int, rate: float, cycles: int, seed: int) -> list[np.ndarray]:
    """Disjoint mask blocks: every gene is masked exactly once per cycle.

    Masking a random subset independently per pass would need ~50 passes before
    the least lucky gene had been masked once. A seeded partition into
    ``ceil(1/rate)`` blocks keeps the per-pass masked fraction at the model's own
    pretraining rate and gives every gene exact, auditable coverage.
    """
    rng = np.random.default_rng(seed)
    blocks: list[np.ndarray] = []
    n_blocks = max(1, int(math.ceil(1.0 / max(rate, 1e-6))))
    for _ in range(cycles):
        order = rng.permutation(n)
        blocks.extend(np.array_split(order, n_blocks))
    return blocks
