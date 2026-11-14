"""Score the probe blocks: S1--S3, controls C4--C7, nulls, intervals, one table.

    PYTHONPATH=src python -m scfm_probe.analysis --all
    PYTHONPATH=src python -m scfm_probe.analysis --model scgpt --cell-line hepg2

What this module is for
-----------------------
The driver (``probe/run.py``) writes pooled hidden-state deltas and their
per-position norms. It deliberately computes no score: a block is a measurement,
and the reading of a measurement is a separate decision that has to be
re-runnable without a GPU. This module is that reading, and it is the only place
in run 5 where a number that could go into a sentence is produced.

Everything it emits lands in one small table, ``results/probe/summary.parquet``,
one row per (model, cell line, anchor, layer, value, statistic) with its own
null band and its own interval. The column set is frozen --- the figure module
was written against it --- and is declared in ``config.SUMMARY_COLUMNS`` rather
than here, so a reader can see the contract without reading the producer.

Four disciplines it enforces
----------------------------
**A null is drawn in its own cell, never borrowed.** ``config.N_NULL_DRAWS``
fresh shuffled-label draws are taken separately for every (model, line, anchor,
layer, value, statistic), each with a seed derived from that cell's own name.
``null_max`` is the LARGEST of those draws, not their mean: an arm that clears
the mean of its null but not its maximum has not cleared it. Copying a band from
a neighbouring layer would be cheaper and would silently understate the noise on
a cell with fewer perturbations.

**Every interval is a cluster bootstrap over response groups.** Perturbed genes
are not independent draws --- HepG2's 524 fall into a few hundred groups, one of
them very large --- so resampling rows would give an interval narrower than the
design supports. The groups come from run 3's own clustering of the MEASURED
effect matrix (``translate/splits.py:effect_groups`` through
``groups.GroupTable``), reused rather than re-derived so that "same response
group" means the same thing in both runs. Where the effect matrix cannot be
read, the interval is ``nan`` with a reason, never a row bootstrap quietly
substituted.

**An absence is recorded, not scored.** TranscriptFormer is autoregressive and
right-shifts both gene tokens and counts, so a gene's own count is never an
input at its own position. The perturbed gene's own hidden state therefore does
not move, the block stores ``nan`` there, and the S1 ratio has no denominator
and C1 no positive control. Both come back with ``score = nan``, an
``n_undefined`` count and a ``detail`` saying why --- and the ratio's
*numerator*, the absolute off-target norm, is emitted as its own statistic
(``propagation_off_target_norm``) so that model's depth profile is still
plottable. A zero would have been a lie; a dropped row would have been a
different one.

**A statistic reports what it measured, at the coverage it measured it at.**
Block coverage is read from the block's own ``manifest.json`` and carried into
every ``detail`` string, because the sweep that fills these directories
overwrites a smoke block with a full one in place and two rows that differ only
in coverage would otherwise be indistinguishable.

What was rejected
-----------------
*Re-implementing the ranking rule.* ``probe/retrieval.py`` already owns the
cosine distance, the random tie-break and the top-k/MRR readout, and
``probe/pool.py`` owns the half split. This module calls them and adds no second
definition of a rank. The one place it constructs a distance matrix itself is
the norm-only control, where the "signature" is a scalar and a cosine distance
between one-dimensional vectors is identically degenerate; there the distance is
``|a - b|`` and the *ranking* is still ``retrieval.rank_of_own``.

*Run 3's ``fit_ridge`` for the held-out-cell probe.* It fits a weighted
gene-pair regression against an effect matrix through an ``ArmFeatures`` object
on a torch device; the held-out-cell probe is a one-vs-rest classification over
perturbation identity with no such object and no effect matrix in play, so
reusing it would mean building a fake arm rather than reusing a rule. The
primary is a nearest centroid, which the plan names as primary precisely because
it has no hyperparameter and therefore cannot be tuned into a false positive.
The ridge variant is deferred, and this sentence is the record of the deferral.

*A row bootstrap, or a null over the mean of the draws.* Both are narrower than
the design supports; see above.

*Scoring each individual-cell block on its own.* Blocks whose anchor family is
``individual_cells`` are one block per control cell, and their reason for
existing is the held-out-cell axis (S3). Emitting a split-half row for each of
24 cells would multiply the table by 24 to answer a question the mean-control
rows already answer. So the individual-cell blocks contribute ``heldout_cell``
rows and nothing else, and that is a scope decision, not an oversight.

Torch-free. Needs numpy, pandas, pyarrow and (through run 3's clustering) scipy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import paths, sibling
from .probe import retrieval as retr


# --------------------------------------------------------------------------- #
# seeds
# --------------------------------------------------------------------------- #
def derive_seed(base: int, *purpose: object) -> int:
    """A reproducible seed for one named purpose, and for no other.

    Every draw in this module --- each null, each bootstrap, each cell split ---
    names itself and gets a seed from that name. Two cells therefore never share
    a draw, which is what makes "an arm counts only if it clears its own null in
    its own cell" mechanically true rather than a convention the caller has to
    remember. Numpy is never seeded globally.
    """
    digest = hashlib.sha256("|".join(str(p) for p in purpose).encode()).digest()
    return int((int(base) + int.from_bytes(digest[:8], "big")) % (2**63 - 1))


# --------------------------------------------------------------------------- #
# a block on disk
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Block:
    """One (model, cell line, anchor) probe block, with its labels loaded.

    The label arrays are read once and asserted equal across ``pool.npz``,
    ``norms.npz``, ``self_pool.npz`` and ``half_assignment.npz`` before any
    number is computed. Row order is a contract in this tree, and a contract is
    asserted rather than assumed: a silent misalignment between the pooled
    signatures and the perturbed gene ids would report chance on a model that
    works, which is exactly this run's headline finding and therefore the one
    error that must not be possible.
    """

    path: Path
    model: str
    cell_line: str
    anchor: str
    anchor_family: str
    anchor_cell_label: str
    layers: tuple[int, ...]
    value_grid_cp10k: np.ndarray
    perturbed_gene_id: np.ndarray
    perturbed_target: np.ndarray
    readout_gene_id: np.ndarray
    self_cols: np.ndarray
    self_position_defined: bool
    coverage: str
    manifest: dict

    @property
    def n_perturbations(self) -> int:
        return int(self.perturbed_gene_id.size)

    @property
    def no_op_mask(self) -> np.ndarray:
        """``(n_perturbed, n_values)``: which queries were bit-identical to baseline.

        A no-op is a query whose perturbed value equalled the anchor's own value at
        that gene, so the two passes were the same input and the delta is exactly
        zero by construction rather than by measurement. Those rows must be dropped
        and counted, never averaged in: a zero signature is not a model that failed
        to respond, it is a question that was never asked.

        They are not hypothetical and not confined to the individual-cell anchor.
        TranscriptFormer clips counts at its training-time limit of 30 CP10K, so the
        64 CP10K query is a no-op for every gene already above the clip --- 20 of
        2620 queries in the full HepG2 block, all at the top of the grid.

        Read from the manifest by (gene id, value) rather than by recomputing, so
        this agrees with what the driver asserted was exactly zero.
        """
        mask = np.zeros((self.n_perturbations, self.value_grid_cp10k.size), dtype=bool)
        rows = (self.manifest.get("diagnostics") or {}).get("no_op_queries") or []
        gene_at = {g: i for i, g in enumerate(self.perturbed_gene_id.astype(str))}
        for r in rows:
            gi = gene_at.get(str(r["perturbed_gene_id"]))
            vi = int(np.argmin(np.abs(self.value_grid_cp10k - float(r["value_cp10k"]))))
            if gi is None:
                raise RuntimeError(
                    f"the manifest of {self.path} names a no-op at gene "
                    f"{r['perturbed_gene_id']!r}, which is not a row of this block; the "
                    "no-op record and the arrays describe different perturbation sets"
                )
            mask[gi, vi] = True
        return mask

    def arrays(self, layer: int) -> dict[str, np.ndarray]:
        """The three arrays of one layer, joined on their own label arrays.

        Loaded on demand rather than held: TranscriptFormer's ``pool`` is
        (705, 5, 2, 2048) float32 per layer and there are five layers per block.
        """
        ldir = self.path / f"layer={int(layer)}"
        out: dict[str, np.ndarray] = {}
        for name, keys in (
            ("pool.npz", ("pool", "pool_unit")),
            ("norms.npz", ("norm",)),
            ("self_pool.npz", ("self_delta",)),
        ):
            f = ldir / name
            if not f.is_file():
                raise RuntimeError(
                    f"{f} is missing, so layer {layer} of {self.path} cannot be scored. "
                    "A missing input is an error here and never a skipped row: a layer "
                    "absent from the table and a layer that scored at chance would be "
                    "indistinguishable downstream"
                )
            z = np.load(f, allow_pickle=False)
            _assert_same_labels(
                z["perturbed_gene_id"].astype(str), self.perturbed_gene_id, f, "perturbed gene"
            )
            _assert_same_grid(np.asarray(z["value_grid_cp10k"], dtype=np.float64), self, f)
            if "readout_gene_id" in z:
                _assert_same_labels(
                    z["readout_gene_id"].astype(str), self.readout_gene_id, f, "read-out gene"
                )
            for k in keys:
                out[k] = np.asarray(z[k], dtype=np.float64)
        return out


def _assert_same_labels(got: np.ndarray, want: np.ndarray, f: Path, what: str) -> None:
    if got.size != want.size or bool((got != want).any()):
        raise RuntimeError(
            f"the {what} labels in {f} do not match the block's; the arrays in this "
            "block were built from different tables and joining them on row index "
            "would put one gene's signature under another gene's name"
        )


def _assert_same_grid(got: np.ndarray, block: "Block", f: Path) -> None:
    if got.size != block.value_grid_cp10k.size or not np.array_equal(
        got, block.value_grid_cp10k
    ):
        raise RuntimeError(
            f"the value grid in {f} is {got.tolist()} but the block's is "
            f"{block.value_grid_cp10k.tolist()}; a dose column would be read under the "
            "wrong dose"
        )


def _self_position_defined(manifest: dict, path: Path) -> bool:
    """Is the perturbed gene's own hidden state a measurable quantity here?

    ``False`` for an autoregressive model that right-shifts counts: a gene's own
    count is never an input at its own position, so nothing moves there. The
    driver records that as an ABSENT check rather than a failed one, and this
    reads the record back rather than re-deriving it from the architecture --- a
    block should be readable from its own manifest.

    The record and the arrays are cross-checked against each other, because the
    two disagreeing would mean either the manifest or the block is stale, and
    silently trusting one of them would decide whether C1 is reported as a
    failure or as an absence.
    """
    stated = manifest.get("self_position_defined")
    if stated is None:
        checks = (manifest.get("verification") or {}).get("checks") or []
        c1 = [c for c in checks if str(c.get("check", "")).startswith("C1/")]
        if c1:
            stated = not all(c.get("ok") is None for c in c1)
    layers = [int(l) for l in manifest.get("layers", [])]
    observed: bool | None = None
    if layers:
        f = path / f"layer={layers[0]}" / "self_pool.npz"
        if f.is_file():
            sd = np.load(f, allow_pickle=False)["self_delta"]
            observed = not bool(np.isnan(sd).all())
    if stated is None and observed is None:
        raise RuntimeError(
            f"{path} records neither a self-position verdict in its manifest nor a "
            "self_pool.npz to read one from, so whether C1 is an absence or a failure "
            "cannot be decided; refusing to guess"
        )
    if stated is not None and observed is not None and bool(stated) != bool(observed):
        raise RuntimeError(
            f"{path}: the manifest says self_position_defined={bool(stated)} but "
            f"self_delta is {'all nan' if not observed else 'finite'}; the manifest and "
            "the arrays disagree about whether the positive control exists at all"
        )
    return bool(stated if stated is not None else observed)


def _coverage_tag(manifest: dict) -> str:
    """A short, verbatim record of how much of the scope this block covers.

    Read from the manifest rather than inferred from an array length, because
    the full-coverage sweep overwrites a smoke block in place: two rows that
    differ only in coverage must not be indistinguishable in the table.
    """
    cov = manifest.get("coverage") or {}
    mode = cov.get("mode", "unknown")
    n = cov.get("n_perturbed", manifest.get("scope", {}).get("n_perturbed", "?"))
    n_scope = cov.get("n_in_scope", "?")
    return f"coverage={mode} {n}/{n_scope}"


def load_block(path: Path) -> Block | None:
    """Read one anchor directory. ``None`` if the driver recorded an absence.

    A block whose manifest says ``hidden_states.available = False`` carried a
    verbatim reason and no arrays. It is skipped with that reason printed, not
    scored as a zero: a model that could not be read and a model whose
    intervention propagates nothing are different results and this run's whole
    point is the difference between them.
    """
    path = Path(path)
    mf = path / "manifest.json"
    if not mf.is_file():
        raise RuntimeError(f"{mf} does not exist, so {path} is not a probe block")
    manifest = json.loads(mf.read_text())
    if not (manifest.get("hidden_states") or {}).get("available", False):
        reason = (manifest.get("hidden_states") or {}).get("reason", "no reason recorded")
        print(f"[skip] {path}: hidden states unavailable --- {reason}", flush=True)
        return None
    ha = np.load(path / "half_assignment.npz", allow_pickle=False)
    pert = pd.read_parquet(path / "perturbed_genes.parquet")
    gene_id = ha["perturbed_gene_id"].astype(str)
    _assert_same_labels(
        pert["gene_id"].to_numpy().astype(str), gene_id, path / "perturbed_genes.parquet", "perturbed gene"
    )
    return Block(
        path=path,
        model=str(manifest["model"]),
        cell_line=str(manifest["cell_line"]),
        anchor=str(manifest["anchor"]),
        anchor_family=str(manifest.get("anchor_family", manifest["anchor"])),
        anchor_cell_label=str(manifest.get("anchor_cell_label", manifest["anchor"])),
        layers=tuple(int(l) for l in manifest["layers"]),
        value_grid_cp10k=np.asarray(manifest["value_grid_cp10k"], dtype=np.float64),
        perturbed_gene_id=gene_id,
        perturbed_target=pert["target"].to_numpy().astype(str),
        readout_gene_id=ha["readout_gene_id"].astype(str),
        self_cols=np.asarray(ha["perturbed_readout_col"], dtype=np.int64),
        self_position_defined=_self_position_defined(manifest, path),
        coverage=_coverage_tag(manifest),
        manifest=manifest,
    )


def discover(
    model: str | None = None, cell_line: str | None = None, anchor: str | None = None
) -> list[Path]:
    """Anchor directories under the run's data root, in a fixed order.

    Discovery rather than enumeration, because the full-coverage sweep is
    written model by model and a run of this module in the middle of it must
    score what exists rather than fail on what does not yet.
    """
    out: list[Path] = []
    for m in paths.MODELS:
        if model and m != model:
            continue
        for line in paths.CELL_LINES:
            if cell_line and line != cell_line:
                continue
            root = paths.probe_dir(m, line)
            if not root.is_dir():
                continue
            for d in sorted(root.glob("anchor=*")):
                if anchor and d.name != f"anchor={anchor}":
                    continue
                if (d / "manifest.json").is_file():
                    out.append(d)
    return out


# --------------------------------------------------------------------------- #
# response groups (C7, and the unit of every interval)
# --------------------------------------------------------------------------- #
class Groups:
    """Response-group ids per perturbation, from run 3's measured effect matrix.

    Run 3's ``groups.GroupTable`` is called rather than re-clustered: the
    grouping is average linkage on 1 - Pearson between MEASURED effect vectors
    cut at the same threshold, and re-deriving it here would let "same response
    group" drift between the run that defines it and the run that reports
    against it. The table is keyed by run 3's perturbation names (gene symbols),
    so the mapping from this block's Ensembl ids is taken from run 3's own
    ``targets.npz`` and cross-checked against the block's parquet.

    If the bundle is not on this machine the grouping is absent, and every
    interval and every C7 row says so instead of falling back to a row
    bootstrap, which would be narrower than the design supports.
    """

    def __init__(self) -> None:
        self._table = None
        self._reason: str | None = None
        self._id_to_target: dict[str, dict[str, str]] = {}

    def _load(self):
        if self._table is None and self._reason is None:
            root = paths.run3_bundle_root()
            if not root.is_dir():
                self._reason = f"run 3's bundle is not on this machine ({root})"
            else:
                self._table = sibling.run3("groups").GroupTable(
                    root, threshold=C.GROUP_LINKAGE_THRESHOLD
                )
        return self._table

    def _targets(self, cell_line: str) -> dict[str, str]:
        if cell_line not in self._id_to_target:
            f = paths.run3_targets(cell_line)
            if not f.is_file():
                raise RuntimeError(f"{f} does not exist; C7's response groups cannot be built")
            z = np.load(f, allow_pickle=False)
            self._id_to_target[cell_line] = dict(
                zip(z["pert_gene_id"].astype(str).tolist(), z["pert_target"].astype(str).tolist())
            )
        return self._id_to_target[cell_line]

    def of(self, block: Block) -> tuple[np.ndarray | None, str]:
        """``(group id per perturbation, a sentence)``, or ``(None, why not)``."""
        table = self._load()
        if table is None:
            return None, str(self._reason)
        lut = self._targets(block.cell_line)
        missing = [g for g in block.perturbed_gene_id if g not in lut]
        if missing:
            return None, (
                f"{len(missing)} of {block.n_perturbations} perturbed genes are absent from "
                f"run 3's effect matrix for {block.cell_line} (first: {missing[0]}), so no "
                "grouping covers this block"
            )
        names = [lut[g] for g in block.perturbed_gene_id]
        if list(names) != list(block.perturbed_target):
            raise RuntimeError(
                "run 3's Ensembl-to-symbol pairing disagrees with this block's "
                "perturbed_genes.parquet; the response groups would be attached to the "
                "wrong perturbations"
            )
        g = np.asarray(table.groups(block.cell_line, names), dtype=np.int64)
        n_groups = int(np.unique(g).size)
        return g, (
            f"{n_groups} response groups over {g.size} perturbations, largest "
            f"{int(np.bincount(g - g.min()).max())}, run 3 linkage cut "
            f"{C.GROUP_LINKAGE_THRESHOLD}"
        )


# --------------------------------------------------------------------------- #
# the cluster bootstrap
# --------------------------------------------------------------------------- #
def group_bootstrap(
    values: np.ndarray,
    groups: np.ndarray | None,
    reducer: str,
    seed: int,
    n_resamples: int = C.N_BOOTSTRAP,
    ci: float = C.BOOTSTRAP_CI,
) -> tuple[float, float, float]:
    """``(point estimate, lo, hi)`` resampling response GROUPS, rows carried whole.

    The unit of resampling is the group and never the row. Perturbed genes are
    not independent draws --- paralogs and members of one complex give
    near-identical responses --- so a row bootstrap would report an interval
    narrower than the design supports. The resampling scheme itself is run 3's:
    drawing ``G`` groups with replacement is a Multinomial(G, 1/G) over the
    groups, and ``metrics._group_counts`` is called so the two runs draw from
    one implementation rather than two that can drift.

    ``reducer`` is ``"mean"`` for the retrieval accuracies --- the pre-registered
    readout is the fraction of perturbations whose own signature came first, and
    the median of a 0/1 indicator is 0 or 1 --- and ``"median"`` for the
    propagation statistics, whose per-perturbation values are heavy-tailed
    ratios. Run 3's own ``bootstrap_median_grouped`` is used for the latter
    verbatim.

    With no grouping available the interval is ``nan``: a row bootstrap is not a
    conservative substitute for a group bootstrap, it is a different and smaller
    number.
    """
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    v = v[ok]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean(v)) if reducer == "mean" else float(np.median(v))
    if groups is None:
        return point, float("nan"), float("nan")
    g = np.asarray(groups)[ok]
    metrics = sibling.run3("metrics")
    if reducer == "median":
        boot = metrics.bootstrap_median_grouped(
            v, g, n_resamples=n_resamples, ci=ci, seed=seed
        )
        return point, float(boot["lo"]), float(boot["hi"])
    if reducer != "mean":
        raise RuntimeError(f"unknown bootstrap reducer {reducer!r}")
    rng = np.random.default_rng(seed)
    gidx, counts, _ = metrics._group_counts(g, n_resamples, rng)
    w = counts[:, gidx].astype(np.float64)  # (n_resamples, n_rows) row multiplicities
    total = w.sum(axis=1)
    means = np.divide(w @ v, total, out=np.full(total.shape, np.nan), where=total > 0)
    alpha = (1.0 - ci) / 2.0
    return point, float(np.nanquantile(means, alpha)), float(np.nanquantile(means, 1 - alpha))


# --------------------------------------------------------------------------- #
# linear CKA (C4), in its Gram form
# --------------------------------------------------------------------------- #
def gram(x: np.ndarray) -> np.ndarray:
    """``X X^T``. Uncentred, because the bootstrap re-centres per resample.

    The Gram form is used rather than ``||Xc^T Yc||_F`` because it makes CKA
    computable between representations of DIFFERENT width --- TranscriptFormer's
    hidden state is 2048-wide and its static gene table 2560-wide --- and
    because ``n`` (a few hundred perturbations) is far below ``d``, so the
    ``n x n`` form is also the cheap one under two thousand resamples.
    """
    x = np.asarray(x, dtype=np.float64)
    return x @ x.T


def _double_centre(k: np.ndarray) -> np.ndarray:
    """``H K H``, which equals the Gram matrix of the feature-centred rows."""
    r = k.mean(axis=1, keepdims=True)
    c = k.mean(axis=0, keepdims=True)
    return k - r - c + k.mean()


def cka_from_grams(k: np.ndarray, l: np.ndarray) -> float:
    """Linear CKA of two representations from their Gram matrices.

    Returns ``nan`` rather than a number when either representation is constant
    across perturbations. That case is reached exactly at layer 0, where the
    off-target delta is identically zero by construction: a "similarity" there
    would be the shape of float noise, and reporting it would put a control's
    artefact in the same column as a measurement.
    """
    kc, lc = _double_centre(k), _double_centre(l)
    den = float(np.sqrt(float((kc * kc).sum()) * float((lc * lc).sum())))
    if not np.isfinite(den) or den <= C.CKA_MIN_DENOM:
        return float("nan")
    return float((kc * lc).sum() / den)


def _group_members(groups: np.ndarray) -> list[np.ndarray]:
    """Row indices of each response group, computed once for many resamples."""
    uniq = np.unique(groups)
    return [np.flatnonzero(groups == g) for g in uniq]


def _resample_index(members: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Row indices of one group-bootstrap draw: whole groups, with replacement.

    ``G`` groups are drawn from ``G`` with replacement and every member of a
    drawn group is carried, so a 149-member family enters or leaves the draw as
    one unit. This is the same scheme ``metrics._group_counts`` implements for
    the mean and median statistics; it is written out here because CKA is not a
    reduction over per-row values and cannot be expressed as a weighted mean.
    """
    drawn = rng.integers(0, len(members), size=len(members))
    return np.concatenate([members[int(i)] for i in drawn])


# --------------------------------------------------------------------------- #
# the rows
# --------------------------------------------------------------------------- #
def _row(
    block_model: str,
    cell_line: str,
    anchor: str,
    layer: int,
    value: float,
    statistic: str,
    split: str,
    score: float,
    lo: float,
    hi: float,
    null_max: float,
    n_perturbations: int,
    n_undefined: int,
    detail: str,
) -> dict:
    if statistic not in C.SUMMARY_STATISTICS:
        raise RuntimeError(
            f"{statistic!r} is not in the frozen statistic vocabulary "
            f"{C.SUMMARY_STATISTICS}; the figure module reads that vocabulary and a new "
            "name would be silently dropped from every plot"
        )
    if split not in C.SUMMARY_SPLITS:
        raise RuntimeError(f"{split!r} is not one of the frozen splits {C.SUMMARY_SPLITS}")
    return {
        "model": str(block_model),
        "cell_line": str(cell_line),
        "anchor": str(anchor),
        "layer": int(layer),
        "value_cp10k": float(value),
        "statistic": str(statistic),
        "split": str(split),
        "score": float(score),
        "lo": float(lo),
        "hi": float(hi),
        "null_max": float(null_max),
        "n_perturbations": int(n_perturbations),
        "n_undefined": int(n_undefined),
        "detail": str(detail),
    }


# --------------------------------------------------------------------------- #
# S2, C6, C7: retrieval over the split-half signatures
# --------------------------------------------------------------------------- #
def _group_collapsed_hits(
    dist: np.ndarray,
    groups: np.ndarray,
    k: int,
    database_groups: np.ndarray | None = None,
) -> np.ndarray:
    """Per-perturbation C7 indicator: is any of the top ``k`` in my own group?

    Written out rather than taken from ``retrieval.retrieval_scores`` only
    because the bootstrap needs the per-row indicator and that function returns
    the mean. Its value is asserted against the function's own number by the
    caller, so the two definitions cannot drift apart unnoticed.

    ``database_groups`` is the response group of each *column* of ``dist`` and
    defaults to ``groups``, which is what every scored row uses: the query rows
    and the database columns are the same perturbations in the same order.

    It exists so that C7's null can be written the way it actually reads --- as a
    permutation of the GROUP LABELS over a fixed geometry --- rather than only as
    the database permutation the callers use. The two are equivalent (measured
    0.1088 either way at scgpt/hepg2/layer 12/dose 0, against an analytic
    collision rate of 0.0963), and `tests/test_c7_null.py` asserts that, because
    the equivalence is the only thing that makes the callers' unpermuted-labels
    call correct rather than a bug. Permuting the labels *together with* the
    database is the one combination that is wrong: it returns the query's own
    group by construction and the null becomes exactly 1.000.
    """
    # Same seeded jitter as `retrieval.retrieval_scores`, and for the same reason:
    # a stable argsort puts row 0 first in every tied row, so on a dead layer this
    # control would report row 0's response group as everyone's answer. Asserting
    # this against that function's number is only a guard if both break ties the
    # same way --- otherwise the two agree by being wrong identically.
    rng = np.random.default_rng(C.RETRIEVAL_TIE_SEED)
    keys = dist + rng.random(dist.shape) * max(
        C.RETRIEVAL_TIE_ATOL, np.finfo(np.float64).eps
    )
    order = np.argsort(keys, axis=1, kind="stable")
    kk = min(int(k), dist.shape[1])
    db = groups if database_groups is None else np.asarray(database_groups)
    if db.size != dist.shape[1]:
        raise RuntimeError(
            f"{db.size} database group labels for {dist.shape[1]} database columns; the "
            "retrieved item's response group would be read off the wrong perturbation"
        )
    return np.array(
        [bool(np.isin(db[order[i, :kk]], groups[i]).any()) for i in range(dist.shape[0])]
    )


def _null_max_retrieval(
    query: np.ndarray,
    database: np.ndarray,
    statistic: str,
    cell: tuple,
    groups: np.ndarray | None = None,
) -> float:
    """The largest of ``N_NULL_DRAWS`` fresh shuffled-label draws, in this cell.

    A draw permutes the database rows, which breaks the pairing between a
    perturbation's half-A signature and its own half-B signature while leaving
    every vector, every norm and every tie exactly as it was. That is the null
    the plan pre-registers: it asks what this block's geometry scores when
    identity is removed and nothing else is.

    The maximum, not the mean: a score inside the spread of five draws has not
    cleared its null, and reporting the mean would let it look as though it had.
    """
    best = -np.inf
    for draw in range(C.N_NULL_DRAWS):
        seed = derive_seed(C.NULL_SEED, *cell, statistic, "draw", draw)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(database.shape[0])
        dist = retr.cosine_distance(query, database[perm])
        ranks = retr.rank_of_own(dist, seed=derive_seed(C.RETRIEVAL_TIE_SEED, *cell, statistic, draw))
        if statistic == "retrieval_mrr":
            val = float(np.mean(1.0 / ranks))
        elif statistic == "group_collapsed_top1":
            if groups is None:
                return float("nan")
            val = float(_group_collapsed_hits(dist, groups, 1).mean())
        else:
            k = 10 if statistic.endswith("top10") else 1
            val = float(np.mean(ranks <= k))
        best = max(best, val)
    return float(best)


def retrieval_rows(
    block: Block,
    layer: int,
    vi: int,
    query: np.ndarray,
    database: np.ndarray,
    groups: np.ndarray | None,
    group_note: str,
    split: str,
    anchor: str,
    extra: str = "",
    labels: np.ndarray | None = None,
) -> list[dict]:
    """S2 (and C7) at one (layer, dose): top-1, top-10, MRR, group-collapsed.

    The per-perturbation outcome (``rank_i <= k``, or ``1/rank_i``) is computed
    once against the FULL database of ``n`` candidates and only then bootstrapped
    over groups. Resampling before ranking would shrink the database, and with it
    the chance level of ``1/n`` the score is read against, so the interval would
    be an interval on a different quantity than the point estimate.
    """
    value = float(block.value_grid_cp10k[vi])
    cell = (block.model, block.cell_line, anchor, layer, value, split)
    n = int(query.shape[0])
    if n < 2:
        return [
            _row(
                block.model, block.cell_line, anchor, layer, value, stat, split,
                float("nan"), float("nan"), float("nan"), float("nan"), n, n,
                f"{block.coverage}; a retrieval over {n} perturbation(s) has no chance "
                "level to be read against",
            )
            for stat in ("retrieval_top1", "retrieval_top10", "retrieval_mrr")
        ]
    # `labels` is the KEPT subset when this dose dropped no-op queries; falling
    # back to the block's full list would hand `retrieval_scores` more labels than
    # rows, and its label-equality contract exists precisely to catch that.
    lab = block.perturbed_gene_id if labels is None else labels
    scores = retr.retrieval_scores(query, database, labels=lab, groups=groups)
    ranks = np.asarray(scores["ranks"], dtype=np.float64)
    dist = retr.cosine_distance(query, database)
    n_deg = int(scores["n_degenerate"])
    base = (
        f"{block.coverage}; chance top1 {scores['chance_top1']:.4g}, median rank "
        f"{scores['median_rank']:.4g}, run-3 normalised rank "
        f"{scores['normalized_rank']:.4g}, {n_deg} of {n} pooled signatures exactly "
        f"zero (ranked by random tie-break); {group_note}{extra}"
    )
    rows: list[dict] = []
    for stat, per_row in (
        ("retrieval_top1", (ranks <= 1).astype(np.float64)),
        ("retrieval_top10", (ranks <= 10).astype(np.float64)),
        ("retrieval_mrr", 1.0 / ranks),
    ):
        point, lo, hi = group_bootstrap(
            per_row, groups, "mean", derive_seed(C.BOOTSTRAP_SEED, *cell, stat)
        )
        expected = float(scores["mrr"] if stat == "retrieval_mrr" else scores[stat.split("_")[-1]])
        if not np.isclose(point, expected, rtol=0, atol=1e-12):
            raise RuntimeError(
                f"the per-perturbation {stat} indicator averages to {point} but "
                f"retrieval_scores reported {expected}; the bootstrap is being run over "
                "a different quantity than the one being reported"
            )
        rows.append(
            _row(
                block.model, block.cell_line, anchor, layer, value, stat, split,
                point, lo, hi,
                _null_max_retrieval(query, database, stat, cell),
                n, n_deg, base,
            )
        )
    if groups is not None:
        hits = _group_collapsed_hits(dist, groups, 1).astype(np.float64)
        if not np.isclose(hits.mean(), float(scores["group_collapsed_top1"]), atol=1e-12):
            raise RuntimeError(
                "the per-row group-collapsed indicator disagrees with "
                "retrieval_scores' own group_collapsed_top1; two definitions of the same "
                "control have drifted apart"
            )
        point, lo, hi = group_bootstrap(
            hits, groups, "mean", derive_seed(C.BOOTSTRAP_SEED, *cell, "group_collapsed_top1")
        )
        rows.append(
            _row(
                block.model, block.cell_line, anchor, layer, value,
                "group_collapsed_top1", split, point, lo, hi,
                _null_max_retrieval(query, database, "group_collapsed_top1", cell, groups),
                n, n_deg,
                f"{base}. Reported BESIDE strict top-1 "
                f"({float(scores['top1']):.4g}), never instead of it: a hit here only "
                "means the retrieved gene shares a measured response group with the "
                "true one",
            )
        )
    else:
        rows.append(
            _row(
                block.model, block.cell_line, anchor, layer, value,
                "group_collapsed_top1", split, float("nan"), float("nan"), float("nan"),
                float("nan"), n, n,
                f"{block.coverage}; not computed: {group_note}",
            )
        )
    return rows


def norm_only_rows(
    block: Block, layer: int, vi: int, pool: np.ndarray, pool_unit: np.ndarray,
    groups: np.ndarray | None, anchor: str, labels: np.ndarray | None = None,
) -> list[dict]:
    """C6: is identity carried by the response's MAGNITUDE rather than its direction?

    ``||delta_h||`` tracks the perturbed gene's own expression level, so a
    retrieval that scores well on the scalar norm alone is not evidence about
    regulatory reach. The signature is therefore replaced by that one number and
    the retrieval re-run.

    The distance has to change with it. Cosine on a one-dimensional vector is
    identically 1 - sign(a)sign(b), so every positive norm would tie with every
    other and the control would report chance on any input at all --- it would
    be incapable of failing, which is the one property a control may not have.
    The distance is ``|a - b|``; the RANKING is still
    ``retrieval.rank_of_own``, so the tie rule and the tie tolerance are the
    module's, not a second copy.

    The direction-only variant (``pool_unit``, per-position unit-normalised
    before pooling) is the other half of C6 and is reported in ``detail``,
    because the statistic vocabulary of the summary table was frozen before this
    module was written and a new name would be dropped by the figure module.
    """
    value = float(block.value_grid_cp10k[vi])
    cell = (block.model, block.cell_line, anchor, layer, value, "split_half")
    n = int(pool.shape[0])
    qn = np.linalg.norm(pool[:, 0, :], axis=1)[:, None]
    dn = np.linalg.norm(pool[:, 1, :], axis=1)[:, None]
    dist = np.abs(qn - dn.T)
    ranks = retr.rank_of_own(dist, seed=derive_seed(C.RETRIEVAL_TIE_SEED, *cell, "norm_only"))
    hits = (ranks <= 1).astype(np.float64)
    point, lo, hi = group_bootstrap(
        hits, groups, "mean", derive_seed(C.BOOTSTRAP_SEED, *cell, "norm_only_top1")
    )
    best = -np.inf
    for draw in range(C.N_NULL_DRAWS):
        rng = np.random.default_rng(derive_seed(C.NULL_SEED, *cell, "norm_only_top1", draw))
        perm = rng.permutation(n)
        d = np.abs(qn - dn[perm].T)
        r = retr.rank_of_own(d, seed=derive_seed(C.RETRIEVAL_TIE_SEED, *cell, "norm_only", draw))
        best = max(best, float(np.mean(r <= 1)))
    unit = retr.retrieval_scores(
        pool_unit[:, 0, :], pool_unit[:, 1, :],
        labels=block.perturbed_gene_id if labels is None else labels,
    )
    unit_null = _null_max_retrieval(
        pool_unit[:, 0, :], pool_unit[:, 1, :], "retrieval_top1",
        (block.model, block.cell_line, anchor, layer, value, "split_half_unit"),
    )
    return [
        _row(
            block.model, block.cell_line, anchor, layer, value, "norm_only_top1",
            "split_half", point, lo, hi, float(best), n,
            int((qn == 0).sum()),
            f"{block.coverage}; the signature is the SCALAR ||delta_h|| alone, matched by "
            f"|a-b| and ranked by retrieval.rank_of_own; chance {1.0 / n:.4g}. "
            f"unit_normalized_top1={float(unit['top1']):.4g} "
            f"(null_max={unit_null:.4g}, {int(unit['n_degenerate'])} degenerate) --- the "
            "direction-only half of C6, carried here because the summary's statistic "
            "vocabulary was frozen before this module was written",
        )
    ]


# --------------------------------------------------------------------------- #
# C4: is the signature just the static gene table?
# --------------------------------------------------------------------------- #
def _static_table(block: Block) -> tuple[np.ndarray | None, str]:
    """The model's static gene table, rows aligned to this block's perturbations."""
    f = paths.gene_embedding_table(block.model, block.cell_line)
    if not f.is_file():
        return None, f"run 2's static gene table is not on this machine ({f})"
    z = np.load(f, allow_pickle=False)
    lut = {g: i for i, g in enumerate(z["gene_id"].astype(str).tolist())}
    missing = [g for g in block.perturbed_gene_id if g not in lut]
    if missing:
        return None, (
            f"{len(missing)} of {block.n_perturbations} perturbed genes have no row in "
            f"run 2's {block.model} gene table (first: {missing[0]})"
        )
    rows = np.array([lut[g] for g in block.perturbed_gene_id], dtype=np.int64)
    return np.asarray(z["embedding"], dtype=np.float64)[rows], str(f)


def cka_rows(
    block: Block, layer: int, vi: int, pool: np.ndarray, table: np.ndarray | None,
    table_note: str, groups: np.ndarray | None, anchor: str,
    keep: np.ndarray | None = None,
) -> list[dict]:
    """C4: linear CKA between the pooled signatures and the static gene table.

    The control that decides how the result is WORDED. If the hidden signature's
    geometry is the static gene table's geometry, then a positive S2 is the
    paper's existing static-table result echoed down the residual stream, not a
    new one about response.

    The signature used is the mean of the two half pools --- the full off-target
    pooled delta --- rather than one half: CKA is not a retrieval, there is
    nothing to hold out from, and using half the positions would only add noise
    to a comparison of geometries.

    A retrieval whose database is the static table is the natural companion, and
    it is computed WHERE THE TWO SPACES HAVE THE SAME WIDTH. For
    TranscriptFormer they do not (2048-wide hidden state, 2560-wide ESM2-derived
    gene table), so a cross-space cosine is undefined and the fact is recorded
    in ``detail`` rather than papered over with a projection this run never
    pre-registered. CKA itself is width-agnostic and is available for all three
    models, which is why it, and not the retrieval, is the statistic.
    """
    value = float(block.value_grid_cp10k[vi])
    cell = (block.model, block.cell_line, anchor, layer, value, "split_half")
    n = int(pool.shape[0])
    if table is not None and keep is not None:
        # `pool` has already had this dose's no-op rows dropped; the static table is
        # indexed by perturbation and has not. Comparing the two geometries requires
        # them to describe the same genes in the same order, so the same mask is
        # applied here rather than assumed to be the identity.
        if table.shape[0] != keep.size:
            raise RuntimeError(
                f"the static gene table has {table.shape[0]} rows for {keep.size} "
                "perturbations; the CKA would be comparing two different gene sets"
            )
        table = table[keep]
    if table is not None and table.shape[0] != n:
        raise RuntimeError(
            f"the static gene table has {table.shape[0]} rows against {n} pooled "
            "signatures; CKA compares them row for row, so a mismatch would silently "
            "pair one gene's hidden state with another gene's embedding"
        )
    if table is None:
        return [
            _row(
                block.model, block.cell_line, anchor, layer, value, "cka_static_table",
                "split_half", float("nan"), float("nan"), float("nan"), float("nan"),
                n, n, f"{block.coverage}; not computed: {table_note}",
            )
        ]
    sig = pool.mean(axis=1)  # (n, d): mean of the two half pools
    k, l = gram(sig), gram(table)
    point = cka_from_grams(k, l)
    lo = hi = float("nan")
    if groups is not None and np.isfinite(point):
        rng = np.random.default_rng(derive_seed(C.BOOTSTRAP_SEED, *cell, "cka_static_table"))
        members = _group_members(groups)
        draws = np.empty(C.N_BOOTSTRAP, dtype=np.float64)
        for b in range(C.N_BOOTSTRAP):
            idx = _resample_index(members, rng)
            draws[b] = cka_from_grams(k[np.ix_(idx, idx)], l[np.ix_(idx, idx)])
        alpha = (1.0 - C.BOOTSTRAP_CI) / 2.0
        lo = float(np.nanquantile(draws, alpha))
        hi = float(np.nanquantile(draws, 1 - alpha))
    best = -np.inf
    for draw in range(C.N_NULL_DRAWS):
        rng = np.random.default_rng(derive_seed(C.NULL_SEED, *cell, "cka_static_table", draw))
        perm = rng.permutation(n)
        val = cka_from_grams(k, l[np.ix_(perm, perm)])
        if np.isfinite(val):
            best = max(best, val)
    detail = (
        f"{block.coverage}; linear CKA between the mean-of-halves pooled signature "
        f"({sig.shape[1]}-wide) and run 2's static gene table ({table.shape[1]}-wide) over "
        f"{n} perturbations, Gram form. Table: {table_note}"
    )
    if sig.shape[1] == table.shape[1]:
        scores = retr.retrieval_scores(sig, table, labels=block.perturbed_gene_id)
        null = _null_max_retrieval(sig, table, "retrieval_top1", cell + ("static_table",))
        detail += (
            f". static_table_retrieval_top1={float(scores['top1']):.4g} "
            f"(null_max={null:.4g}, chance {float(scores['chance_top1']):.4g})"
        )
    else:
        detail += (
            f". static_table_retrieval_top1=nan: the hidden state is {sig.shape[1]}-wide and "
            f"the gene table {table.shape[1]}-wide, so a cross-space cosine is undefined; "
            "CKA is width-agnostic and is the statistic for that reason"
        )
    return [
        _row(
            block.model, block.cell_line, anchor, layer, value, "cka_static_table",
            "split_half", point, lo, hi, float(best) if np.isfinite(best) else float("nan"),
            n, 0 if np.isfinite(point) else n, detail,
        )
    ]


# --------------------------------------------------------------------------- #
# S1 and C1: the perturbed gene's own position
# --------------------------------------------------------------------------- #
def self_position_rows(
    block: Block, layer: int, arrays: dict[str, np.ndarray], groups: np.ndarray | None,
    group_note: str, anchor: str,
) -> list[dict]:
    """S1 (propagation) and C1 (did the intervention land) at one layer.

    S1 asks whether the edit left the perturbed gene's own token before any
    decoding question is asked. Its ratio is dimensionless and comparable across
    layers of different width; its numerator, the absolute off-target norm, is
    not --- it is in the model's own hidden-state units --- so both are emitted
    and the numerator's ``detail`` says exactly that. The numerator exists
    because TranscriptFormer has no denominator: the model right-shifts counts,
    a gene's own count is never an input at its own position, and the block
    stores ``nan`` there. That is a property of the architecture, recorded as an
    absence with ``score = nan`` and an ``n_undefined`` count, never as a zero
    and never as a failure.

    No shuffled-label null is drawn for either. Both are label-free per-
    perturbation quantities --- permuting which perturbation a norm belongs to
    changes nothing about the set of norms --- so a "null" here would be the
    statistic itself dressed up as its own control. The control S1 actually has
    is C2, layer 0, where the off-target norm must be exactly 0.0, and that is
    checked by the driver over the same array.
    """
    norm = arrays["norm"]
    cols = block.self_cols
    prop = retr.propagation_ratio(norm, cols)
    off_masked = retr.off_target_norms(norm, cols)
    finite_or_self = np.isnan(off_masked)
    expected_nan = np.zeros_like(finite_or_self)
    expected_nan[np.arange(norm.shape[0]), :, cols] = True
    if not np.array_equal(finite_or_self, expected_nan):
        raise RuntimeError(
            "the per-position norm array carries nan somewhere other than the perturbed "
            "gene's own position, so the off-target norm would silently drop read-out "
            "positions rather than sum over all of them"
        )
    # The off-target norm is recomputed here with `nansum` rather than taken from
    # `propagation_ratio`'s own `off_target_norm`, which is `total - self` and is
    # therefore nan for a model whose self position is nan --- exactly the model
    # this statistic exists for. Where that column IS finite the two must agree,
    # and that is asserted rather than assumed, so the second route can never
    # drift into being a second definition.
    off_norm = np.sqrt(np.nansum(off_masked**2, axis=2))
    ref = np.asarray(prop["off_target_norm"], dtype=np.float64)
    both = np.isfinite(ref) & np.isfinite(off_norm)
    if both.any() and not np.allclose(ref[both], off_norm[both], rtol=1e-9, atol=1e-9):
        raise RuntimeError(
            "the nan-aware off-target norm disagrees with retrieval.propagation_ratio's "
            f"own (max abs diff {float(np.abs(ref[both] - off_norm[both]).max()):.6g}); "
            "two routes to the same quantity have drifted apart"
        )
    # The same no-op drop the split-half statistics get. A no-op query's delta is
    # exactly zero at every position, so its propagation is exactly zero --- not
    # because nothing travelled, but because nothing was asked. Twenty such rows
    # at TranscriptFormer's top dose would pull the median of 524 toward zero and
    # read as the model going quiet at high input.
    no_op = block.no_op_mask
    rows: list[dict] = []
    for vi, value in enumerate(block.value_grid_cp10k.tolist()):
        cell = (block.model, block.cell_line, anchor, layer, float(value), "self_position")
        keep = ~no_op[:, vi]
        n = int(keep.sum())
        n_dropped = int((~keep).sum())
        drop_note = (
            ""
            if n_dropped == 0
            else f" {n_dropped} no-op queries dropped at this dose."
        )
        ratio = prop["ratio"][keep, vi]
        groups_v = None if groups is None else groups[keep]
        n_undef = int((~np.isfinite(ratio)).sum())
        if block.self_position_defined:
            point, lo, hi = group_bootstrap(
                ratio, groups_v, "median",
                derive_seed(C.BOOTSTRAP_SEED, *cell, "propagation_ratio")
            )
            detail = (
                f"{block.coverage}; median over perturbations of ||delta_h off-target|| / "
                f"||delta_h at the perturbed gene's own position||, {n - n_undef} of {n} "
                f"defined; {group_note}. No shuffled-label null exists for a label-free "
                "per-perturbation ratio; its control is C2 (layer 0 exactly 0.0), checked "
                "by the driver"
            )
        else:
            point = lo = hi = float("nan")
            n_undef = n
            detail = (
                f"{block.coverage}; ABSENT by construction, not failed and not zero: this "
                "model is autoregressive and right-shifts both gene tokens and counts, so "
                "a gene's own count is never an input at its own position and the ratio's "
                "denominator does not exist. The numerator is defined and is reported as "
                "propagation_off_target_norm; read this model's depth profile from that row"
            )
        rows.append(
            _row(
                block.model, block.cell_line, anchor, layer, float(value),
                "propagation_ratio", "self_position", point, lo, hi, float("nan"),
                n, n_undef, detail,
            )
        )
        # The same ratio per read-out position rather than summed over them, which
        # is the only form comparable BETWEEN cell lines: HepG2 supplies 2721
        # read-out genes and Jurkat 3247, so the Frobenius numerator alone carries a
        # sqrt(3247/2721) = 1.093 factor that has nothing to do with the models.
        n_off = max(int(norm.shape[2]) - 1, 1)
        pp = ratio / np.sqrt(n_off)
        pp_point, pp_lo, pp_hi = group_bootstrap(
            pp, groups_v, "median",
            derive_seed(C.BOOTSTRAP_SEED, *cell, "propagation_ratio_per_position"),
        )
        rows.append(
            _row(
                block.model, block.cell_line, anchor, layer, float(value),
                "propagation_ratio_per_position", "self_position", pp_point, pp_lo,
                pp_hi, float("nan"), n, n_undef,
                f"{block.coverage}; propagation_ratio divided by sqrt({n_off}), the "
                "number of off-target read-out positions. This is the form to compare "
                "across cell lines: the Frobenius numerator grows as sqrt(n_readout), "
                "so identical per-position propagation reads 9.3% higher in Jurkat "
                f"than in HepG2 on the undivided form. {detail}",
            )
        )
        onv = off_norm[keep, vi]
        point, lo, hi = group_bootstrap(
            onv, groups_v, "median",
            derive_seed(C.BOOTSTRAP_SEED, *cell, "propagation_off_target_norm"),
        )
        rows.append(
            _row(
                block.model, block.cell_line, anchor, layer, float(value),
                "propagation_off_target_norm", "self_position", point, lo, hi,
                float("nan"), n, int((~np.isfinite(onv)).sum()),
                f"{block.coverage};{drop_note} median over perturbations of the Frobenius norm of "
                f"delta_h over the {norm.shape[2] - 1} off-target read-out positions, in "
                "this model's own hidden-state units. NOT comparable across models or "
                "across layers of different width --- only its depth profile within one "
                f"model is; the scale-free form is propagation_ratio. {group_note}",
            )
        )
    # C1: can the perturbed gene be identified from its OWN position's delta?
    for vi, value in enumerate(block.value_grid_cp10k.tolist()):
        cell = (block.model, block.cell_line, anchor, layer, float(value), "self_position")
        if not block.self_position_defined:
            for stat in ("retrieval_top1", "retrieval_top10", "retrieval_mrr"):
                rows.append(
                    _row(
                        block.model, block.cell_line, anchor, layer, float(value), stat,
                        "self_position", float("nan"), float("nan"), float("nan"),
                        float("nan"), n, n,
                        f"{block.coverage}; C1 is unavailable BY CONSTRUCTION for this "
                        "model: it right-shifts tokens and counts, so a gene's own count "
                        "is never an input at its own position and nothing moves there. "
                        "A recorded absence, never a failure and never a score of 0",
                    )
                )
            continue
        sd = arrays["self_delta"][:, vi, :]
        rows.extend(
            retrieval_rows(
                block, layer, vi, sd, sd, groups, group_note, "self_position", anchor,
                extra=(
                    ". C1: query and database are the SAME array (the delta at the "
                    "perturbed gene's own position), so this is 1.0 unless a self delta is "
                    "exactly zero or is not distinct from another gene's --- which is "
                    "precisely the dead-intervention / wrong-token-id failure C1 exists to "
                    "catch, and nothing more"
                ),
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# S3: the held-out-cell probe
# --------------------------------------------------------------------------- #
def heldout_plan(blocks: list[Block], groups_for: "Groups") -> dict | None:
    """The depth-independent half of S3: the cell split, the checks, the grouping.

    S3 asks whether a signature learned in some control cells survives into
    others. Everything about *which* cells are held out is decided here, once per
    (model, cell line), and :func:`heldout_cell_layer` then scores one depth
    against that decision. Returns ``None`` when there are too few cells to split.

    Original docstring, unchanged, because the reasoning is unchanged:

    The split is over CELLS and never over perturbations. Every perturbation
    appears on both sides by design --- the classes are the perturbations, so a
    held-out class would have no centroid to be retrieved against --- and what is
    held out is the anchor cell the signature was read in. A cell appearing on
    both sides would leak the test signature into its own centroid, so that is
    checked and raised on rather than trusted.

    Nearest centroid is the primary because it has no hyperparameter and
    therefore cannot be tuned into a false positive; it is the version of this
    probe that supports the word "generalizes", which S2 does not.
    """
    if not blocks:
        return None
    ref = blocks[0]
    labels = [b.anchor_cell_label for b in blocks]
    if len(set(labels)) != len(labels):
        raise RuntimeError(
            f"two individual-cell blocks of {ref.model}/{ref.cell_line} carry the same "
            f"anchor_cell_label ({labels}); the cell split could not be over cells"
        )
    for b in blocks[1:]:
        _assert_same_labels(b.perturbed_gene_id, ref.perturbed_gene_id, b.path, "perturbed gene")
        if b.layers != ref.layers:
            raise RuntimeError(
                f"{b.path} carries layers {b.layers} but {ref.path} carries {ref.layers}; "
                "the cells were probed at different depths and cannot be stacked"
            )
    n_cells = len(blocks)
    if n_cells <= C.N_HELDOUT_CELLS:
        print(
            f"[skip] {ref.model}/{ref.cell_line}: {n_cells} individual-cell block(s), "
            f"fewer than the {C.N_HELDOUT_CELLS + 1} a held-out-cell split needs; no "
            "heldout_cell rows emitted",
            flush=True,
        )
        return None
    rng = np.random.default_rng(
        derive_seed(C.HELDOUT_CELL_SEED, ref.model, ref.cell_line, n_cells)
    )
    order = rng.permutation(n_cells)
    test_idx = np.sort(order[: C.N_HELDOUT_CELLS])
    train_idx = np.sort(order[C.N_HELDOUT_CELLS :])
    if set(test_idx.tolist()) & set(train_idx.tolist()):
        raise RuntimeError(
            "a control cell landed on both sides of the held-out-cell split; the test "
            "signature would be inside its own centroid"
        )
    groups, group_note = groups_for.of(ref)
    return {
        "labels": labels,
        "n_cells": n_cells,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "groups": groups,
        "group_note": group_note,
        "anchor": ref.anchor_family,
        "n": ref.n_perturbations,
        "layers": tuple(ref.layers),
    }


def heldout_cell_layer(blocks: list[Block], layer: int, plan: dict) -> list[dict]:
    """The ``heldout_cell`` rows of ONE depth, given :func:`heldout_plan`'s split.

    Split out from the planner above so that depths can be scored in
    parallel. The cell split, the label checks and the response grouping are all
    depth-independent, so they are computed once in the parent and passed in
    rather than recomputed per worker --- recomputing them would be wasteful and,
    worse, would put the split's RNG in more than one place.
    """
    ref = blocks[0]
    labels = plan["labels"]
    train_idx, test_idx = plan["train_idx"], plan["test_idx"]
    groups, group_note = plan["groups"], plan["group_note"]
    anchor, n, n_cells = plan["anchor"], plan["n"], plan["n_cells"]
    rows: list[dict] = []
    layer = int(layer)
    # (n_cells, n_pert, n_values, d), held in float32 --- the dtype the driver
    # wrote --- because 24 cells x 705 perturbations x 5 doses x 2048 is 1.4 GB
    # in float64 and nothing here needs more precision than the block has.
    stacked = np.stack(
        [b.arrays(layer)["pool"].mean(axis=2).astype(np.float32) for b in blocks]
    )
    for vi, value in enumerate(ref.value_grid_cp10k.tolist()):
        cell = (ref.model, ref.cell_line, anchor, layer, float(value), "heldout_cell")
        centroid = stacked[train_idx, :, vi, :].mean(axis=0)  # (n_pert, d)
        hits1: list[np.ndarray] = []
        hits10: list[np.ndarray] = []
        recips: list[np.ndarray] = []
        ghits: list[np.ndarray] = []
        unit_groups: list[np.ndarray] = []
        for c in test_idx:
            q = stacked[c, :, vi, :]
            dist = retr.cosine_distance(q, centroid)
            ranks = retr.rank_of_own(
                dist, seed=derive_seed(C.RETRIEVAL_TIE_SEED, *cell, "heldout", int(c))
            )
            hits1.append((ranks <= 1).astype(np.float64))
            hits10.append((ranks <= 10).astype(np.float64))
            recips.append(1.0 / ranks)
            if groups is not None:
                ghits.append(_group_collapsed_hits(dist, groups, 1).astype(np.float64))
                unit_groups.append(groups)
        # every evaluation unit is one (perturbation, held-out cell) pair, and its
        # group is the perturbation's: the same paralog family repeated across cells
        # is still one cluster, not eight independent draws
        gunits = np.concatenate(unit_groups) if unit_groups else None
        note = (
            f"{ref.coverage}; nearest centroid over {train_idx.size} training cells, "
            f"evaluated on {test_idx.size} held-out cells "
            f"({[labels[i] for i in test_idx]}), split over CELLS at seed "
            f"{derive_seed(C.HELDOUT_CELL_SEED, ref.model, ref.cell_line, n_cells)}; "
            f"chance top1 {1.0 / n:.4g}; {group_note}. One-vs-rest ridge deferred: "
            "run 3's fit_ridge is a weighted gene-pair regression against an effect "
            "matrix and does not apply to a class-identity probe"
        )
        for stat, vals in (
            ("retrieval_top1", np.concatenate(hits1)),
            ("retrieval_top10", np.concatenate(hits10)),
            ("retrieval_mrr", np.concatenate(recips)),
        ):
            point, lo, hi = group_bootstrap(
                vals, gunits, "mean", derive_seed(C.BOOTSTRAP_SEED, *cell, stat)
            )
            best = -np.inf
            for draw in range(C.N_NULL_DRAWS):
                r = np.random.default_rng(derive_seed(C.NULL_SEED, *cell, stat, draw))
                # ONE relabelling per draw, shared by every held-out cell. A fresh
                # permutation per cell would average several independent nulls
                # together and shrink the band, making the null easier to clear
                # than the design intends; a label is a property of the
                # perturbation, not of the cell it was read in.
                perm = r.permutation(n)
                acc = []
                for c in test_idx:
                    d = retr.cosine_distance(stacked[c, :, vi, :], centroid[perm])
                    rk = retr.rank_of_own(
                        d, seed=derive_seed(C.RETRIEVAL_TIE_SEED, *cell, stat, draw, int(c))
                    )
                    acc.append(
                        1.0 / rk if stat == "retrieval_mrr"
                        else (rk <= (10 if stat.endswith("top10") else 1)).astype(float)
                    )
                best = max(best, float(np.concatenate(acc).mean()))
            rows.append(
                _row(
                    ref.model, ref.cell_line, anchor, layer, float(value), stat,
                    "heldout_cell", point, lo, hi, float(best), n,
                    int((~np.isfinite(vals)).sum()), note,
                )
            )
        if groups is not None:
            vals = np.concatenate(ghits)
            point, lo, hi = group_bootstrap(
                vals, gunits, "mean",
                derive_seed(C.BOOTSTRAP_SEED, *cell, "group_collapsed_top1"),
            )
            # C7 gets its own shuffled-label band here, exactly as the strict
            # statistics above do. It carried `nan` until 2026-08-31, which
            # `verify`'s `every_decision_score_has_a_null` caught the first time S3
            # ran: the split-half path has always given this control a null, so the
            # held-out path lacking one was an omission and not an exemption.
            #
            # ⚠️ The group labels are deliberately NOT permuted alongside the
            # database, and this is the opposite of what it looks like. Permuting
            # the database breaks the STRICT score because `rank_of_own` takes the
            # diagonal as truth, and after a permutation the diagonal is a different
            # perturbation. It does not break C7, whose truth is the response GROUP
            # of whatever was retrieved: the geometry still returns the query's own
            # centroid, so relabelling the database consistently would return the
            # query's own group and the null would be exactly 1.000 --- vacuous.
            # Leaving the labels unpermuted makes the retrieved column index an
            # effectively uniform perturbation, which IS the null C7 needs: what
            # does group-collapsed top-1 score when group membership carries no
            # information. Measured equal to a direct group-label permutation
            # (0.1088 both ways at scgpt/hepg2/layer 12/dose 0) and to the analytic
            # collision rate sum(p^2) = 0.0963. `tests/test_c7_null.py` pins it.
            best_g = -np.inf
            for draw in range(C.N_NULL_DRAWS):
                r = np.random.default_rng(
                    derive_seed(C.NULL_SEED, *cell, "group_collapsed_top1", draw)
                )
                # ONE relabelling per draw shared by every held-out cell, for the
                # same reason as the strict nulls above.
                perm = r.permutation(n)
                acc_g = []
                for c in test_idx:
                    d = retr.cosine_distance(stacked[c, :, vi, :], centroid[perm])
                    acc_g.append(
                        _group_collapsed_hits(d, groups, 1).astype(np.float64)
                    )
                best_g = max(best_g, float(np.concatenate(acc_g).mean()))
            rows.append(
                _row(
                    ref.model, ref.cell_line, anchor, layer, float(value),
                    "group_collapsed_top1", "heldout_cell", point, lo, hi,
                    float(best_g), n, 0,
                    note + ". Reported BESIDE strict top-1, never instead of it",
                )
            )
    print(
        f"[{ref.model}/{ref.cell_line}/{anchor}] layer {layer}: {len(rows)} heldout_cell "
        f"rows over {len(blocks)} cells",
        flush=True,
    )
    return rows


# --------------------------------------------------------------------------- #
# one block, every statistic
# --------------------------------------------------------------------------- #
def score_block_layer(block: Block, layer: int, groups_for: "Groups") -> list[dict]:
    """Every split-half and self-position row of ONE depth of one block.

    One depth is the unit the work is parallelised over, and it is the right
    one: it is the level at which the arrays are loaded from disk
    (``Block.arrays`` reads one ``layer=<l>/`` directory), and no statistic
    crosses depths. Everything a depth needs is derived from the block and its
    own arrays, and every seed in this module is derived from the metric cell's
    own name (:func:`derive_seed`), so scoring depths concurrently gives bit-wise
    the same rows as scoring them in sequence.
    """
    groups, group_note = groups_for.of(block)
    table, table_note = _static_table(block)
    anchor = block.anchor
    rows: list[dict] = []
    layer = int(layer)
    arrays = block.arrays(layer)
    rows.extend(self_position_rows(block, layer, arrays, groups, group_note, anchor))
    no_op = block.no_op_mask
    for vi in range(block.value_grid_cp10k.size):
        # No-op queries are dropped here, once, before any statistic sees them.
        # Leaving them in would not merely add noise: a zero signature is a dead
        # row in the retrieval database, so it dilutes the score while `1/n`
        # still claims n live candidates, and the reported chance level would be
        # for a comparison that was never run.
        keep = ~no_op[:, vi]
        n_dropped = int((~keep).sum())
        drop_note = (
            ""
            if n_dropped == 0
            else (
                f"; {n_dropped} of {keep.size} perturbations dropped as no-ops at "
                f"this dose (the anchor already sat at the perturbed value, so the "
                f"two passes were one input) and the chance level is 1/{int(keep.sum())}"
            )
        )
        pool = arrays["pool"][keep, vi, :, :]  # (n_kept, 2, d)
        pool_unit = arrays["pool_unit"][keep, vi, :, :]
        g = None if groups is None else groups[keep]
        if int(keep.sum()) == 0:
            raise RuntimeError(
                f"every perturbation is a no-op at dose "
                f"{float(block.value_grid_cp10k[vi])} CP10K in {block.path}; that is "
                "a broken anchor rather than a measurement"
            )
        rows.extend(
            retrieval_rows(
                block, layer, vi, pool[:, 0, :], pool[:, 1, :], g, group_note,
                "split_half", anchor, extra=drop_note,
                labels=block.perturbed_gene_id[keep],
            )
        )
        rows.extend(norm_only_rows(block, layer, vi, pool, pool_unit, g, anchor,
                                   labels=block.perturbed_gene_id[keep]))
        rows.extend(cka_rows(block, layer, vi, pool, table, table_note, g, anchor,
                             keep=keep))
    print(
        f"[{block.model}/{block.cell_line}/{anchor}] layer {layer}: {len(rows)} rows so far",
        flush=True,
    )
    return rows


# --------------------------------------------------------------------------- #
# running the depths concurrently
# --------------------------------------------------------------------------- #
#: Rebuilt once per worker process rather than pickled from the parent. ``Groups``
#: lazily loads run 3's clustered effect table, which is a few MB and identical in
#: every worker; sending it through the pickle channel with every unit of work
#: would dominate the cost of the small units.
_WORKER_GROUPS: "Groups | None" = None


def _worker_init() -> None:
    """Pin each worker to one BLAS thread, then build its ``Groups`` once.

    Without the pin, ``N_ANALYSIS_WORKERS`` processes each start a thread pool
    sized for the whole machine and the box thrashes: the bootstrap is thousands
    of small numpy reductions, which get slower under oversubscription, not
    faster. One thread per process and one process per core is the right shape
    for this workload.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    global _WORKER_GROUPS
    _WORKER_GROUPS = Groups()


def _groups() -> "Groups":
    global _WORKER_GROUPS
    if _WORKER_GROUPS is None:          # serial path, or a worker that skipped init
        _WORKER_GROUPS = Groups()
    return _WORKER_GROUPS


def _unit(job: tuple) -> list[dict]:
    """Score one unit of work. Must be module-level to be picklable.

    A unit is one depth: either one depth of one ordinary block, or one depth of
    an individual-cells family against its precomputed split.
    """
    kind = job[0]
    if kind == "block":
        _, block, layer = job
        return score_block_layer(block, int(layer), _groups())
    if kind == "heldout":
        _, blocks, layer, plan = job
        return heldout_cell_layer(blocks, int(layer), plan)
    raise RuntimeError(f"unknown unit kind {kind!r}")


def run_units(jobs: list[tuple], workers: int) -> list[list[dict]]:
    """Every unit's rows, in submission order, on ``workers`` processes.

    Submission order is preserved so that the concatenated row list is identical
    to the serial one. That is belt-and-braces --- :func:`build` sorts the frame
    on a stable key before writing, so completion order could not reach the file
    anyway --- but it makes a parallel run diffable against a serial one row for
    row, which is how this was checked.
    """
    if workers <= 1 or len(jobs) <= 1:
        _worker_init()
        return [_unit(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        return list(pool.map(_unit, jobs))


# --------------------------------------------------------------------------- #
# the table
# --------------------------------------------------------------------------- #
def build(
    model: str | None = None,
    cell_line: str | None = None,
    anchor: str | None = None,
    out: Path | None = None,
    write: bool = True,
    workers: int | None = None,
) -> pd.DataFrame:
    """Score every discovered block and write ``results/probe/summary.parquet``."""
    dirs = discover(model, cell_line, anchor)
    if not dirs:
        raise RuntimeError(
            f"no probe blocks under {paths.data_root() / 'probe'} matching "
            f"model={model} cell_line={cell_line} anchor={anchor}; a missing input is an "
            "error here, not an empty table"
        )
    groups_for = Groups()
    rows: list[dict] = []
    individual: dict[tuple[str, str], list[Block]] = {}
    jobs: list[tuple] = []
    for d in dirs:
        block = load_block(d)
        if block is None:
            continue
        if block.anchor_family == "individual_cells":
            individual.setdefault((block.model, block.cell_line), []).append(block)
            continue
        jobs.extend(("block", block, int(l)) for l in block.layers)
    for (m, line), blocks in sorted(individual.items()):
        blocks = sorted(blocks, key=lambda b: b.anchor)
        plan = heldout_plan(blocks, groups_for)
        if plan is None:
            continue
        print(
            f"[{m}/{line}] {len(blocks)} individual-cell block(s); "
            "held-out-cell probe over cells",
            flush=True,
        )
        jobs.extend(("heldout", blocks, int(l), plan) for l in plan["layers"])
    n_workers = max(1, int(workers if workers else C.N_ANALYSIS_WORKERS))
    print(
        f"[analysis] {len(jobs)} depth-units on {n_workers} worker(s)",
        flush=True,
    )
    for produced in run_units(jobs, n_workers):
        rows.extend(produced)
    if not individual:
        # S3 is unmeasured, and "unmeasured" is a state the table has to carry.
        # Emitting nothing leaves a hole indistinguishable from a statistic that was
        # computed and came back empty, and it makes every downstream consumer guess:
        # the figure module, correctly, refuses to draw a panel whose rows are simply
        # missing. So the absence is written as rows --- score nan, the reason in
        # `detail` --- which is the same contract TranscriptFormer's C1 uses for an
        # absence that is architectural rather than merely unrun.
        print(
            "[heldout_cell] no individual_cells block exists on disk, so S3 is "
            "recorded as UNMEASURED rows rather than omitted",
            flush=True,
        )
        seen = {(r["model"], r["cell_line"], r["anchor"]) for r in rows}
        for m, line, anch in sorted(seen):
            for stat in ("retrieval_top1", "retrieval_top10", "retrieval_mrr"):
                rows.append(
                    _row(
                        m, line, anch, C.EMBEDDING_LAYER, C.PRIMARY_VALUE_CP10K, stat,
                        "heldout_cell", float("nan"), float("nan"), float("nan"),
                        float("nan"), 0, 0,
                        "not measured: the individual_cells anchor was not run, so no "
                        "cell was ever held out. This is an absence of data, NOT a "
                        "measurement of no effect --- nothing here bears on whether a "
                        "signature learned in one cell survives into another.",
                    )
                )
    df = pd.DataFrame(rows, columns=list(C.SUMMARY_COLUMNS))
    if list(df.columns) != list(C.SUMMARY_COLUMNS):
        raise RuntimeError(
            f"the summary table's columns are {list(df.columns)} but the frozen contract "
            f"is {list(C.SUMMARY_COLUMNS)}"
        )
    df = df.sort_values(
        ["model", "cell_line", "anchor", "split", "statistic", "layer", "value_cp10k"],
        kind="stable",
    ).reset_index(drop=True)
    if write:
        target = Path(out) if out else paths.summary_table()
        paths.ensure(target.parent)
        df.to_parquet(target, index=False)
        print(f"[summary] {len(df)} rows -> {target}", flush=True)
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="score every block on disk")
    ap.add_argument("--model", default=None, choices=list(paths.MODELS))
    ap.add_argument("--cell-line", default=None, choices=list(paths.CELL_LINES))
    ap.add_argument("--anchor", default=None, help="one anchor directory name")
    ap.add_argument("--out", default=None, help="override the summary parquet path")
    ap.add_argument("--no-write", action="store_true", help="compute and print, write nothing")
    ap.add_argument("--print-rows", type=int, default=40, help="how many rows to print")
    ap.add_argument(
        "--jobs",
        type=int,
        default=C.N_ANALYSIS_WORKERS,
        help=(
            "worker processes over depth-units (default "
            f"{C.N_ANALYSIS_WORKERS}). 1 runs in-process. Results do not depend on "
            "this: every seed is derived from the metric cell's own name and the "
            "frame is sorted on a stable key before writing"
        ),
    )
    args = ap.parse_args(argv)
    if not args.all and not (args.model or args.cell_line or args.anchor):
        ap.error("pass --all, or narrow with --model / --cell-line / --anchor")
    df = build(
        model=args.model,
        cell_line=args.cell_line,
        anchor=args.anchor,
        out=Path(args.out) if args.out else None,
        write=not args.no_write,
        workers=int(args.jobs),
    )
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(
            df.head(int(args.print_rows))[
                [
                    "model", "cell_line", "layer", "value_cp10k", "statistic", "split",
                    "score", "lo", "hi", "null_max", "n_perturbations", "n_undefined",
                ]
            ].to_string(index=False),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
