"""The response contract every model in this thread implements.

A *response* is one model's per-gene expression output for one input cell,
reported over the shared gene universe. A *Jacobian column* is the derivative of
that whole response vector with respect to the input expression of one source
gene -- a directional derivative, so it is one forward-mode pass when the model
supports forward-mode automatic differentiation and a central finite difference
otherwise. Which route was used is recorded, never assumed.

Later runs add a feature extractor by subclassing ``ResponseModel``; they read
the same ``gene_universe.parquet`` and effect matrices and write to the same
per-model directory, so nothing in this run has to change.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ResponseContract:
    """What a later run needs to know before trusting a model's response."""

    model: str
    checkpoint: str
    checkpoint_revision: str
    checkpoint_md5: str | None
    #: exact transformation applied to raw counts before the model sees them
    input_normalization: str
    #: which head produces the per-gene value, and what the value means
    output_head: str
    output_semantics: str
    #: does the architecture let gene j see gene i regardless of position?
    information_flow: str
    ordering_dependent: bool
    response_symmetric: bool | None
    symmetry_evidence: str
    #: differentiability of the per-gene output w.r.t. an input gene's expression
    #: as mathematics; whether *this* build can differentiate it is separate
    differentiable_wrt_input_expression: bool
    jacobian_available: bool
    jacobian_method: str
    jacobian_unavailable_reason: str | None
    #: whether ``torch.func.jvp`` actually runs through this build's forward pass,
    #: as measured in the process that wrote this contract. A mathematically smooth
    #: model can still lack a forward-mode rule for one of its kernels, so this is
    #: recorded from an attempt and never inferred from differentiability.
    forward_mode_autodiff: str = "not probed in the process that wrote this contract"
    #: finite-difference agreement when both routes are available
    jacobian_check: dict = field(default_factory=dict)
    vocabulary_kind: str = ""
    vocabulary_size: int = 0
    context_length: int = 0
    dtype: str = ""
    substitutions: list[str] = field(default_factory=list)
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


class ResponseModel(ABC):
    """One model's response interface. Subclasses own loading and the forward pass."""

    name: str = ""

    def __init__(self, cache: Path, universe: pd.DataFrame, vocab: pd.DataFrame, device: str = "cuda"):
        self.cache = cache
        self.universe = universe
        self.vocab = vocab
        self.device = device
        self.substitutions: list[str] = []

    # ---- to implement -----------------------------------------------------
    @abstractmethod
    def load(self) -> None:
        """Materialize the checkpoint on ``self.device`` in eval mode."""

    @abstractmethod
    def prepare_input(self, counts: np.ndarray, gene_ids: np.ndarray) -> dict:
        """Map one cell's raw counts (over ``gene_ids``) into model input space."""

    @abstractmethod
    def response(self, prepared: dict) -> np.ndarray:
        """Per-gene response over the shared universe, in universe row order."""

    @abstractmethod
    def contract(self) -> ResponseContract:
        """The declared response contract, filled in after ``load``."""

    def jacobian_column(self, prepared: dict, source_position: int) -> tuple[np.ndarray | None, str, str | None]:
        """(column over the universe, method, reason-if-unavailable)."""
        return None, "not_implemented", "the model adapter does not implement a Jacobian column"

    # ---- shared helpers ---------------------------------------------------
    def universe_gene_ids(self) -> np.ndarray:
        return self.universe["gene_id"].to_numpy().astype(str)

    def universe_positions(self) -> np.ndarray:
        return self.universe[f"{self.name}_position"].to_numpy().astype(np.int64)

    def position_by_gene(self) -> dict[str, int]:
        """Gene id -> this model's token position.

        A vocabulary can carry two tokens that resolve to the same Ensembl gene
        (duplicate or aliased symbols), so this mapping is ambiguous on its own.
        The gene universe already recorded one position per gene, and that choice
        is the contract every run reads, so it wins here.
        """
        tab = self.vocab[self.vocab["gene_id"].astype(str) != ""]
        out = dict(
            zip(tab["gene_id"].to_numpy().astype(str), tab["position"].to_numpy().astype(int))
        )
        universe = dict(zip(self.universe_gene_ids(), (int(p) for p in self.universe_positions())))
        self._n_positions_from_universe = sum(
            1 for g, p in universe.items() if out.get(g) != p
        )
        out.update(universe)
        return out

    def vocab_input_vector(self, counts: np.ndarray, gene_ids: np.ndarray, size: int) -> np.ndarray:
        """Scatter a cell's measured counts onto this model's own vocabulary order.

        Genes the model knows but the assay did not measure stay at zero, which
        is what every one of these models does with an absent gene.
        """
        pos_by_gene = self.position_by_gene()
        x = np.zeros(size, dtype=np.float32)
        hits = 0
        for gid, c in zip(gene_ids, counts):
            p = pos_by_gene.get(gid)
            if p is not None and 0 <= p < size:
                x[p] = c
                hits += 1
        self._input_genes_used = hits
        return x

    def measured_sequence(
        self, counts: np.ndarray, gene_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sequence of this model's vocabulary positions for the *measured* genes.

        Returns ``(positions, values, universe_row_index)``: the token positions in
        increasing order, the matching counts, and for each universe gene its row
        in that sequence. Restricting the sequence to measured genes keeps the
        input to what the assay actually observed instead of padding tens of
        thousands of unmeasured vocabulary slots with zeros.
        """
        pos_by_gene = self.position_by_gene()
        pairs = [
            (pos_by_gene[g], float(c))
            for g, c in zip(np.asarray(gene_ids).astype(str), np.asarray(counts))
            if g in pos_by_gene
        ]
        pairs.sort()
        positions = np.array([p for p, _ in pairs], dtype=np.int64)
        values = np.array([v for _, v in pairs], dtype=np.float32)
        self._input_genes_used = int(positions.size)
        row_of = {int(p): i for i, p in enumerate(positions)}
        upos = self.universe_positions()
        missing = [int(p) for p in upos if int(p) not in row_of]
        if missing:
            raise RuntimeError(
                f"{len(missing)} universe genes are absent from the {self.name} input "
                f"sequence; the universe and the measured gene list disagree"
            )
        return positions, values, np.array([row_of[int(p)] for p in upos], dtype=np.int64)


def finite_difference_column(
    fn, x: np.ndarray, index: int, step: float | None = None
) -> tuple[np.ndarray, float]:
    """Central difference of a vector-valued ``fn`` along coordinate ``index``."""
    if step is None:
        scale = float(abs(x[index])) if abs(x[index]) > 0 else 1.0
        step = max(1e-3, 1e-2 * scale)
    xp, xm = x.copy(), x.copy()
    xp[index] += step
    xm[index] -= step
    return (fn(xp) - fn(xm)) / (2.0 * step), step


def jacobian_cross_check(
    col: np.ndarray, method: str, npfn, x0: np.ndarray, index: int, alt_step: float = 1.0
) -> dict:
    """Compare a Jacobian column against an independent second estimate.

    Autodiff is checked against a central finite difference. A column that was
    itself computed by finite difference is checked against a second difference
    at a different step, so the recorded agreement is never a method compared
    with itself.
    """
    autodiff = method.startswith("forward_mode")
    step = None if autodiff else alt_step
    alt, step = finite_difference_column(npfn, x0, index, step=step)
    denom = np.linalg.norm(col) + np.linalg.norm(alt)
    return {
        "primary_method": method,
        "cross_check_method": (
            f"central_finite_difference(step={step:.4g})"
            if autodiff
            else f"second central finite difference at step={step:.4g}"
        ),
        "pearson_primary_vs_cross_check": (
            float(np.corrcoef(col, alt)[0, 1]) if np.std(col) > 0 and np.std(alt) > 0 else None
        ),
        "max_abs_difference": float(np.max(np.abs(col - alt))),
        "relative_l2_difference": (
            float(2 * np.linalg.norm(col - alt) / denom) if denom > 0 else None
        ),
        "primary_l2": float(np.linalg.norm(col)),
        "cross_check_l2": float(np.linalg.norm(alt)),
    }


def environment_manifest(model: str, image: str, extra: dict) -> dict:
    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=180
        ).stdout.splitlines()
    except Exception as exc:  # noqa: BLE001
        freeze = [f"pip freeze unavailable: {exc}"]
    return {
        "model": model,
        "image": image,
        "python": sys.version,
        "platform": platform.platform(),
        "pinned_packages": sorted(freeze),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **extra,
    }
