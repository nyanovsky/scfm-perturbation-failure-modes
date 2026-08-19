"""The batched query interface run 2 needs, on top of run 1's four response adapters.

Run 1 answered "what does this model predict for one cell, and what is one
Jacobian column". Run 2 asks each model tens of thousands of questions that
differ in a single input coordinate, so the interface it needs differs in three
ways:

* **batched** -- a value grid and its finite-difference neighbours go through in
  one forward pass rather than one at a time;
* **anchored in a common unit** -- every query is specified in CP10K (counts per
  10,000), the one unit all four models can be given a defensible map into, and
  each adapter converts CP10K to its own input scale;
* **fixed supplied set** -- the genes that enter the forward pass are pinned per
  (model, cell line) and recorded, because two of these models end in a softmax
  across exactly those genes.

The driver in :mod:`scfm_pfm.features.extract` only calls the methods declared on
:class:`QueryModel`, so it is model-agnostic. Run 1's classes are subclassed, not
edited: their loading path, attention substitutions and recorded contract are
inherited unchanged.

The CP10K map, per model
------------------------
``aido_cell``, ``scprint``
    the input is a raw count and the query is a virtual cell of 10,000 counts,
    so a CP10K value *is* the count.
``transcriptformer``
    the same, then clipped at 30, which is the model's own training-time clip
    (``data_config.clip_counts``). Values above 30 saturate, and that is recorded
    rather than hidden.
``scgpt``
    the input is a quantile bin index. The bin edges are built once per cell line
    from that line's mean control profile and then held fixed for every query, so
    the CP10K-to-bin map is one recorded monotone function instead of a per-query
    re-binning, which would make two queries incomparable.

Masking regimes, read from each checkpoint rather than assumed
--------------------------------------------------------------
``scgpt``    input value ``-1`` (``args.json: mask_value``); pretraining rates
             ``[0.25, 0.5, 0.75]``, and the mildest is used.
``aido_cell`` transformed input value ``-1`` (``config.json: mask_token_id``),
             applied after the read-depth transform, which is where the model's
             ``GeneEmbedding`` looks for it.
``scprint``  a boolean ``mask`` argument to ``forward``; the value encoder zeroes
             the expression embedding at those positions.
``transcriptformer``  none: it is autoregressive, so a gene never sees its own
             value and is natively in the masked regime.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from ..forms import partition_masks as _partition_masks
from .aido_model import AidoCellResponse, N_GENES as AIDO_N_GENES
from .scgpt_model import ScGPTResponse, quantile_bin_edges
from .scprint_model import ScPrintResponse
from .transcriptformer_model import ASSAY_LABEL, TranscriptFormerResponse

DEFAULT_MASK_RATE = 0.15


@dataclass
class QueryLayout:
    """What entered the forward pass, recorded per (model, cell line)."""

    model: str
    cell_line: str
    n_readout: int
    n_supplied: int
    supplied_mode: str
    n_passes: int = 1
    context_block: int = 0
    notes: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "cell_line": self.cell_line,
            "n_readout_genes": self.n_readout,
            "n_supplied_genes": self.n_supplied,
            "supplied_mode": self.supplied_mode,
            "forward_passes_per_query": self.n_passes,
            "context_block_genes": self.context_block,
            **self.notes,
        }


class QueryModel:
    """Methods the extraction driver relies on. One implementation per model."""

    batch_size = 8
    #: can raising a gene's own input change its own predicted output at all?
    self_response_defined = True
    #: does the response depend on where the perturbed gene sits in the input?
    ordering_dependent = False
    #: how to read the model's output as a linear expression level, or None
    linear_scale: str | None = None

    def configure(self, readout: pd.DataFrame, anchor_cp10k: np.ndarray, cell_line: str) -> QueryLayout:
        raise NotImplementedError

    def to_native(self, cp10k) -> np.ndarray:
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
        return None

    def gene_embeddings(self) -> tuple[np.ndarray, str]:
        raise NotImplementedError

    def exact_jvp(self, x_native: np.ndarray, col: int) -> np.ndarray | None:
        """The exact forward-mode derivative column, or None with a recorded reason.

        An adapter that cannot provide one sets ``self._jvp_error`` to say why, so
        the block manifest records a measured absence rather than a default.
        """
        self._jvp_error = (
            f"{type(self).__name__} does not implement a forward-mode column; the base "
            "interface returns none"
        )
        return None

    def forward_mode_probe(self, x_native: np.ndarray, col: int) -> tuple[np.ndarray | None, str | None]:
        """Try the forward-mode route once and report what happened.

        Separate from :meth:`exact_jvp` because a model may support forward-mode
        differentiation while a block's features were produced by finite
        differences: the probe measures the route without changing the features.
        """
        try:
            col_out = self.exact_jvp(np.asarray(x_native), int(col))
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"
        if col_out is None:
            return None, str(getattr(self, "_jvp_error", "unavailable, no reason recorded"))
        return col_out, None

    def _batches(self, x: np.ndarray, size: int | None = None):
        size = int(size or self.batch_size)
        for start in range(0, x.shape[0], size):
            yield start, x[start : start + size]


# --------------------------------------------------------------------------- #
# scGPT
# --------------------------------------------------------------------------- #
class ScGPTQuery(ScGPTResponse, QueryModel):
    """scGPT with fixed quantile bin edges and a choice of supplied-set width.

    scGPT was trained at a maximum sequence length of 1,200 but has no positional
    encoding over genes, so a longer input is architecturally admissible and run
    1 supplied the whole 7,444-gene universe. Run 2 treats the width as a
    measured choice: ``readout``, ``universe`` and ``tiled`` (passes of at most
    1,200 sharing a fixed prefix) each produce a full response vector, and the
    extraction reports how far apart they are before the sweep commits to one.
    """

    batch_size = 8
    linear_scale = None  # a per-cell quantile bin index is not an expression level
    TILE_LIMIT = 1200
    TILE_PREFIX = 256

    def configure(
        self,
        readout: pd.DataFrame,
        anchor_cp10k: np.ndarray,
        cell_line: str,
        supplied_mode: str = "readout",
        universe_anchor_cp10k: np.ndarray | None = None,
    ) -> QueryLayout:
        self.cell_line = cell_line
        self.supplied_mode = supplied_mode
        self.readout_tokens = readout["scgpt_position"].to_numpy(dtype=np.int64)
        self.n_readout = int(self.readout_tokens.size)

        log_anchor = np.log1p(np.asarray(anchor_cp10k, dtype=np.float64))
        self.bin_edges = quantile_bin_edges(log_anchor, self.n_bins)
        self.bin_source = (
            f"{self.n_bins}-bin quantile edges of log1p(anchor CP10K) over the "
            f"{self.n_readout} read-out genes of {cell_line}, fixed for every query"
        )
        self.tiles: list[np.ndarray] | None = None
        self.tile_owned: list[np.ndarray] | None = None
        self.universe_base_native = None

        if supplied_mode == "universe":
            self.supplied_tokens = self.universe["scgpt_position"].to_numpy(dtype=np.int64)
            slot_of = {int(t): i for i, t in enumerate(self.supplied_tokens)}
            self.readout_slots = np.array(
                [slot_of[int(t)] for t in self.readout_tokens], dtype=np.int64
            )
            if universe_anchor_cp10k is None:
                raise ValueError("supplied_mode='universe' needs universe_anchor_cp10k")
            self.universe_base_native = self.to_native(universe_anchor_cp10k)
            n_supplied, n_passes = int(self.supplied_tokens.size), 1
        elif supplied_mode == "readout":
            self.supplied_tokens = self.readout_tokens
            self.readout_slots = np.arange(self.n_readout, dtype=np.int64)
            n_supplied, n_passes = self.n_readout, 1
        elif supplied_mode == "tiled":
            order = np.argsort(-np.asarray(anchor_cp10k, dtype=np.float64), kind="stable")
            prefix = order[: self.TILE_PREFIX]
            body = order[self.TILE_PREFIX :]
            per = self.TILE_LIMIT - self.TILE_PREFIX
            self.tiles = [
                np.concatenate([prefix, body[s : s + per]]) for s in range(0, body.size, per)
            ] or [prefix]
            self.tile_owned = [np.arange(self.tiles[0].size)] + [
                np.arange(self.TILE_PREFIX, t.size) for t in self.tiles[1:]
            ]
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
                "context_extrapolation_factor": round(n_supplied / 1200.0, 3),
                "tile_limit": self.TILE_LIMIT if supplied_mode == "tiled" else None,
                "tile_prefix_genes": self.TILE_PREFIX if supplied_mode == "tiled" else None,
                "tile_note": (
                    "every pass carries the same highest-expression prefix; when the "
                    "perturbed gene is not already in a pass it is appended to it, so "
                    "the context differs between queries by exactly one gene"
                    if supplied_mode == "tiled"
                    else None
                ),
            },
        )

    def to_native(self, cp10k) -> np.ndarray:
        v = np.log1p(np.asarray(cp10k, dtype=np.float64))
        edges = self.bin_edges
        return np.interp(v, edges, np.arange(1, edges.size + 1), left=0.0).astype(np.float32)

    def _run(self, tokens: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        pad = torch.zeros(values.shape, dtype=torch.bool, device=values.device)
        out = self.model._encode(tokens, values, pad)
        return self.model.decoder(out)["pred"]

    def _predict(self, tokens: np.ndarray, values: np.ndarray) -> np.ndarray:
        """(B, L) values over the given token list -> (B, L) predictions."""
        tok = torch.as_tensor(tokens, device=self.device, dtype=torch.long)
        out = np.empty(values.shape, dtype=np.float32)
        for start, chunk in self._batches(values):
            t = tok.unsqueeze(0).expand(chunk.shape[0], -1)
            v = torch.as_tensor(chunk, device=self.device, dtype=torch.float32)
            with torch.no_grad():
                out[start : start + chunk.shape[0]] = self._run(t, v).float().cpu().numpy()
        return out

    def forward_batch(
        self, x_native: np.ndarray, perturbed_col: int | None = None, mask: np.ndarray | None = None
    ) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x_native, dtype=np.float32))
        b = x.shape[0]
        mask_value = float(self.mask_spec()["value"])  # type: ignore[index]

        if self.tiles is None:
            if self.supplied_mode == "universe":
                values = np.repeat(self.universe_base_native[None, :], b, axis=0).astype(np.float32)
                values[:, self.readout_slots] = x
                if mask is not None:
                    values[:, self.readout_slots] = np.where(
                        mask, mask_value, values[:, self.readout_slots]
                    )
            else:
                values = x.copy()
                if mask is not None:
                    values = np.where(mask, mask_value, values).astype(np.float32)
            pred = self._predict(self.supplied_tokens, values)
            return pred[:, self.readout_slots]

        out = np.empty((b, self.n_readout), dtype=np.float32)
        for tile, owned in zip(self.tiles, self.tile_owned):  # type: ignore[arg-type]
            cols = tile
            if perturbed_col is not None and int(perturbed_col) not in set(tile.tolist()):
                cols = np.concatenate([tile, [int(perturbed_col)]])
            values = x[:, cols].copy()
            if mask is not None:
                values = np.where(mask[:, cols], mask_value, values).astype(np.float32)
            pred = self._predict(self.readout_tokens[cols], values)
            out[:, cols[owned]] = pred[:, owned]
        return out

    def mask_spec(self) -> dict:
        rates = self.args.get("mask_ratio", [0.25])
        rates = rates if isinstance(rates, (list, tuple)) else [rates]
        return {
            "kind": "input_value",
            "value": float(self.args.get("mask_value", -1)),
            "rate": float(min(rates)),
            "pretraining_rates": [float(r) for r in rates],
            "source": "scGPT args.json: mask_value and mask_ratio; the mildest "
            "pretraining rate is used",
        }

    def gene_embeddings(self) -> tuple[np.ndarray, str]:
        emb = self.model.encoder.embedding.weight.detach()
        idx = torch.as_tensor(self.readout_tokens, device=emb.device)
        return (
            emb[idx].float().cpu().numpy(),
            "TransformerModel.encoder.embedding, the gene token embedding table",
        )

    def exact_jvp(self, x_native: np.ndarray, col: int) -> np.ndarray | None:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        if self.tiles is not None or self.supplied_mode == "universe":
            # a multi-pass or padded query is not one function of the input
            self._jvp_error = (
                f"the {self.supplied_mode} layout delivers a query in "
                f"{len(self.tiles) if self.tiles is not None else 1} pass(es) over a padded "
                "supplied set, so it is not one differentiable function of the input; the "
                "untiled layout is differentiable and is cross-checked separately"
            )
            return None
        tokens = torch.as_tensor(self.supplied_tokens, device=self.device).unsqueeze(0)
        base = torch.as_tensor(
            np.atleast_2d(np.asarray(x_native, dtype=np.float32)), device=self.device
        )

        def fn(v: torch.Tensor) -> torch.Tensor:
            return self._run(tokens, v)[0]

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
    order, so there is no gene selection to make: read-out genes carry their
    anchor value and every other position stays at zero, which is what an
    unmeasured gene is.
    """

    batch_size = 4
    linear_scale = "expm1"  # the head predicts log1p CP10K

    def configure(self, readout: pd.DataFrame, anchor_cp10k: np.ndarray, cell_line: str) -> QueryLayout:
        self.cell_line = cell_line
        self.readout_positions = readout["aido_cell_position"].to_numpy(dtype=np.int64)
        self.n_readout = int(self.readout_positions.size)
        self._out_idx = torch.as_tensor(self.readout_positions, device=self.device)
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
                "hidden_size": int(self.config_dict.get("hidden_size", 0)),
                "num_hidden_layers": int(self.config_dict.get("num_hidden_layers", 0)),
            },
        )

    def to_native(self, cp10k) -> np.ndarray:
        return np.asarray(cp10k, dtype=np.float32)

    def _full_counts(self, x: np.ndarray) -> np.ndarray:
        full = np.zeros((np.atleast_2d(x).shape[0], AIDO_N_GENES), dtype=np.float32)
        full[:, self.readout_positions] = np.atleast_2d(x)
        return full

    def _response_torch(self, counts: "torch.Tensor") -> "torch.Tensor":
        """The production response as one differentiable function of the input counts."""
        inputs = self._transform(counts)
        logits = self.model(input_ids=inputs).logits[:, :-2, 0]
        return logits[:, self._out_idx]

    def exact_jvp(self, x_native: np.ndarray, col: int) -> np.ndarray | None:
        """Not the production route for this model, and recorded as such.

        Run 2's AIDO.Cell blocks are central differences at the selected step. The
        forward-mode route is measured separately by :mod:`scfm_pfm.features.cost_probe`
        through :meth:`forward_mode_probe`, so that the record in the manifest is a
        measurement and enabling the exact path later cannot silently change what a
        shipped block contains.
        """
        self._jvp_error = (
            "not attempted in the production path: this run's AIDO.Cell gradient "
            "trajectories are central differences at the selected step, and the "
            "forward-mode route is measured by features.cost_probe instead"
        )
        return None

    def forward_mode_probe(self, x_native: np.ndarray, col: int):
        """Attempt ``torch.func.jvp`` through AIDO.Cell's own forward pass."""
        counts = torch.as_tensor(
            self._full_counts(np.asarray(x_native, dtype=np.float32)),
            device=self.device, dtype=torch.float32,
        )
        position = int(self.readout_positions[int(col)])

        def fn(c: torch.Tensor) -> torch.Tensor:
            return self._response_torch(c)[0]

        tangent = torch.zeros_like(counts)
        tangent[0, position] = 1.0
        try:
            _, jvp = torch.func.jvp(fn, (counts,), (tangent,))
            return jvp.detach().float().cpu().numpy(), None
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"

    def forward_batch(
        self, x_native: np.ndarray, perturbed_col: int | None = None, mask: np.ndarray | None = None
    ) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x_native, dtype=np.float32))
        full = np.zeros((x.shape[0], AIDO_N_GENES), dtype=np.float32)
        full[:, self.readout_positions] = x
        out = np.empty((x.shape[0], self.n_readout), dtype=np.float32)
        mask_value = float(self.mask_spec()["value"])  # type: ignore[index]
        for start, chunk in self._batches(full):
            counts = torch.as_tensor(chunk, device=self.device, dtype=torch.float32)
            # the read-depth transform runs on the true counts, then the masked
            # positions are set to the model's own mask id, which is what
            # GeneEmbedding(mask_id=-1) looks for
            inputs = self._transform(counts)
            if mask is not None:
                block = mask[start : start + chunk.shape[0]]
                idx = torch.as_tensor(self.readout_positions, device=self.device)
                m = torch.zeros(inputs.shape, dtype=torch.bool, device=self.device)
                m[:, idx] = torch.as_tensor(block, device=self.device)
                inputs = torch.where(m, torch.full_like(inputs, mask_value), inputs)
            with torch.no_grad():
                logits = self.model(input_ids=inputs).logits[:, :-2, 0]
            out[start : start + chunk.shape[0]] = (
                logits[:, self._out_idx].float().cpu().numpy()
            )
        return out

    def mask_spec(self) -> dict:
        return {
            "kind": "transformed_input_value",
            "value": float(self.config_dict.get("mask_token_id", -1)),
            "rate": DEFAULT_MASK_RATE,
            "source": "config.json: mask_token_id; the checkpoint records no "
            f"pretraining mask rate, so {DEFAULT_MASK_RATE} is used and recorded as a "
            "fallback",
        }

    def gene_embeddings(self) -> tuple[np.ndarray, str]:
        found = None
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Embedding) and module.num_embeddings >= AIDO_N_GENES:
                found, source = module, name
                break
        if found is None:
            raise RuntimeError("no gene-identity embedding table found on AIDO.Cell")
        idx = torch.as_tensor(self.readout_positions, device=found.weight.device)
        return (
            found.weight.detach()[idx].float().cpu().numpy(),
            f"CellFoundation {source}: gene identity is the position in the fixed "
            "19,264-gene order, so this table is the model's gene embedding",
        )


# --------------------------------------------------------------------------- #
# scPRINT
# --------------------------------------------------------------------------- #
class ScPrintQuery(ScPrintResponse, QueryModel):
    """scPRINT over a fixed supplied gene set, with its exact forward-mode derivative."""

    batch_size = 8
    linear_scale = "identity"  # a share of the cell's counts is proportional to CP10K

    def configure(self, readout: pd.DataFrame, anchor_cp10k: np.ndarray, cell_line: str) -> QueryLayout:
        self.cell_line = cell_line
        pos = readout["scprint_position"].to_numpy(dtype=np.int64)
        self.seq_order = np.argsort(pos, kind="stable")  # read-out index -> sequence slot
        self.seq_positions = pos[self.seq_order]
        self.readout_to_seq = np.empty(self.seq_order.size, dtype=np.int64)
        self.readout_to_seq[self.seq_order] = np.arange(self.seq_order.size)
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
                "input_order": "ascending vocabulary position; the model carries no gene "
                "positional encoding, so the order is a convention, not a constraint",
            },
        )

    def to_native(self, cp10k) -> np.ndarray:
        return np.asarray(cp10k, dtype=np.float32)

    def _forward_many(self, expr: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        gp = (
            torch.as_tensor(self.seq_positions, device=self.device)
            .unsqueeze(0)
            .expand(expr.shape[0], -1)
        )
        depth = expr.sum(dim=1)
        out = self.model.forward(
            gene_pos=gp,
            expression=expr,
            mask=mask,
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
        seq = x[:, self.seq_order]
        seq_mask = mask[:, self.seq_order] if mask is not None else None
        out = np.empty((x.shape[0], self.n_readout), dtype=np.float32)
        for start, chunk in self._batches(seq):
            expr = torch.as_tensor(chunk, device=self.device, dtype=torch.float32)
            m = (
                torch.as_tensor(seq_mask[start : start + chunk.shape[0]], device=self.device)
                if seq_mask is not None
                else None
            )
            with torch.no_grad():
                mean = self._forward_many(expr, m).float().cpu().numpy()
            out[start : start + chunk.shape[0]] = mean[:, self.readout_to_seq]
        return out

    def mask_spec(self) -> dict:
        return {
            "kind": "boolean_mask_argument",
            "value": None,
            "rate": DEFAULT_MASK_RATE,
            "source": "scPrint.forward(mask=...); the value encoder zeroes the "
            "expression embedding at masked positions. The checkpoint records no "
            f"pretraining mask rate, so {DEFAULT_MASK_RATE} is used and recorded as a "
            "fallback",
        }

    def gene_embeddings(self) -> tuple[np.ndarray, str]:
        emb = self.model.gene_encoder.embeddings.weight.detach()
        idx = torch.as_tensor(self.seq_positions, device=emb.device)
        table = emb[idx].float().cpu().numpy()
        return (
            table[self.readout_to_seq],
            "scPrint.gene_encoder.embeddings, the frozen ESM-derived gene embedding",
        )

    def exact_jvp(self, x_native: np.ndarray, col: int) -> np.ndarray | None:
        base = torch.as_tensor(
            np.atleast_2d(np.asarray(x_native, dtype=np.float32))[:, self.seq_order],
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

    The perturbed gene goes to position 0 of every pass, which makes every
    read-out gene downstream of it and therefore measurable. The cost is that the
    context differs between queries by that one gene, and the fraction of pairs a
    single fixed ordering could not measure at all is reported rather than hidden.
    """

    batch_size = 8
    self_response_defined = False
    ordering_dependent = True
    linear_scale = "identity"  # a share of the cell's counts
    COUNT_CLIP = 30.0

    def configure(self, readout: pd.DataFrame, anchor_cp10k: np.ndarray, cell_line: str) -> QueryLayout:
        self.cell_line = cell_line
        self.readout_tokens = readout["transcriptformer_position"].to_numpy(dtype=np.int64)
        self.n_readout = int(self.readout_tokens.size)
        anchor = np.asarray(anchor_cp10k, dtype=np.float64)
        gene_ids = readout["gene_id"].to_numpy().astype(str)
        self.anchor_order = np.lexsort((gene_ids, -anchor))
        self.pad_id = int(self.gene_vocab_dict[self.model.data_config.gene_pad_token])
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
                "ordering": "descending anchor expression, gene id as tie-break, with "
                "the perturbed gene moved to position 0",
                "seq_len": int(self.seq_len),
                "softmax_over": "the tokens of the same pass; the response is mu divided "
                "by that pass's total count, so passes are comparable",
                "count_clip": self.COUNT_CLIP,
                "count_clip_note": "the model's own training-time clip; a query value "
                "above it saturates",
                "unmeasurable_fraction_under_fixed_ordering": self.unmeasurable_fraction(),
                "masked_variant": "not applicable: a causal mask means a gene never sees "
                "its own value, so this model is natively in the masked regime",
            },
        )

    def unmeasurable_fraction(self) -> float:
        """Share of ordered pairs one fixed ordering could not measure.

        Under a single ordering, gene *j* responds to gene *i* only when *i* comes
        first, so exactly the pairs with rank(i) > rank(j) are structurally zero.
        """
        n = self.n_readout
        return float(0.5) if n > 1 else float("nan")

    def to_native(self, cp10k) -> np.ndarray:
        return np.clip(np.asarray(cp10k, dtype=np.float32), 0.0, self.COUNT_CLIP)

    def _query_chunks(self, perturbed_col: int | None) -> list[np.ndarray]:
        order = list(int(r) for r in self.anchor_order)
        if perturbed_col is not None:
            p = int(perturbed_col)
            order = [p] + [r for r in order if r != p]
        context = order[: self.context_block]
        body = order[self.context_block :]
        per = self.seq_len - len(context)
        return [
            np.array(context + body[s : s + per], dtype=np.int64) for s in range(0, len(body), per)
        ] or [np.array(context, dtype=np.int64)]

    def _forward_rows(self, rows: np.ndarray, values: np.ndarray) -> np.ndarray:
        from transcriptformer.data.dataclasses import BatchData

        b, n = values.shape
        tok = torch.full((b, self.seq_len), self.pad_id, dtype=torch.long, device=self.device)
        tok[:, :n] = (
            torch.as_tensor(self.readout_tokens[rows], dtype=torch.long, device=self.device)
            .unsqueeze(0)
            .expand(b, -1)
        )
        val = torch.zeros((b, self.seq_len), dtype=torch.float32, device=self.device)
        val[:, :n] = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        aux = torch.full((b, 1), self.assay_token, dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model(
                BatchData(gene_counts=val, gene_token_indices=tok, aux_token_indices=aux)
            )
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
                "it has no masked variant"
            )
        x = np.atleast_2d(np.asarray(x_native, dtype=np.float32))
        out = np.zeros((x.shape[0], self.n_readout), dtype=np.float32)
        filled = np.zeros(self.n_readout, dtype=bool)
        for ci, rows in enumerate(self._query_chunks(perturbed_col)):
            keep = slice(None) if ci == 0 else slice(self.context_block, None)
            cols = rows[keep]
            for start, chunk in self._batches(x[:, rows]):
                share = self._forward_rows(rows, chunk)
                out[start : start + chunk.shape[0], cols] = share[:, keep]
            filled[cols] = True
        if not filled.all():
            raise RuntimeError(f"{int((~filled).sum())} read-out genes never entered a pass")
        return out

    def exact_jvp(self, x_native: np.ndarray, col: int) -> np.ndarray | None:
        """Not available in this layout, for the same reason as scGPT's tiled one."""
        self._jvp_error = (
            "the read-out set does not fit one sequence, so a query is several forward "
            "passes stitched together and is not one differentiable function of the input; "
            "the derivative is a central difference at the selected step"
        )
        return None

    def gene_embeddings(self) -> tuple[np.ndarray, str]:
        best, source = None, ""
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Embedding) and module.num_embeddings >= len(
                self.gene_vocab_dict
            ):
                best, source = module, name
                break
        if best is None:
            raise RuntimeError("no gene embedding table found on TranscriptFormer")
        idx = torch.as_tensor(self.readout_tokens, device=best.weight.device)
        return (
            best.weight.detach()[idx].float().cpu().numpy(),
            f"Transcriptformer.{source}, the ESM2-derived gene embedding table",
        )


BUILDERS = {
    "scgpt": ScGPTQuery,
    "aido_cell": AidoQuery,
    "scprint": ScPrintQuery,
    "transcriptformer": TranscriptFormerQuery,
}


def build(model: str, cache, universe: pd.DataFrame, vocab: pd.DataFrame, device: str = "cuda"):
    return BUILDERS[model](cache=cache, universe=universe, vocab=vocab, device=device)


#: re-exported so `queries.partition_masks` keeps working for callers that had it
#: here; the implementation is torch-free so the pod can import it
partition_masks = _partition_masks
