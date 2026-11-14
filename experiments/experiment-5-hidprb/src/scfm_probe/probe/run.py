"""Run the hidden-state probe for one model, one cell line, one anchor.

    PYTHONPATH=src python -m scfm_probe.probe.run --model scgpt \
        --cell-line hepg2 --anchor mean_control --device cuda
    PYTHONPATH=src python -m scfm_probe.probe.run --model scgpt \
        --cell-line hepg2 --anchor mean_control --smoke

What this driver is for
-----------------------
One probe query is two forward passes of a frozen model that differ in one
gene's input value, with per-layer hidden states captured by hook. The driver's
whole job is to make sure those two passes differ in *nothing else*, to reduce
each query to its stored summary before the next one runs, and to write down
every choice that would otherwise have to be reconstructed from an array shape.

Three disciplines it enforces, each of which would silently corrupt the result
if it were left to the caller:

**One baseline per perturbed column, never one globally.** For scGPT the input
is tiled and the perturbed gene is appended to any tile that does not already
contain it, so the sequence composition is a function of ``perturbed_col``; for
TranscriptFormer the perturbed gene is forced to position 0 of every chunk. A
baseline computed once and reused across genes would therefore differ from its
perturbed partner in composition as well as in value, and the delta would mix an
intervention with a re-tiling. The baseline is row 0 of the same batch, submitted
under the same ``perturbed_col``, and the block records how much that mattered by
measuring the same baseline under a different column once per block.

**Cost.** Run 2 spends ``1 + 3 * 5 = 16`` input rows per (gene, anchor) because
it needs a finite-difference derivative at every grid value. This run needs one
shared baseline plus one row per value, so ``1 + 5 = 6``. Run 2's ``query_batch``
is deliberately not called: reusing it would triple the GPU bill for columns
nothing here reads.

**No-ops are counted, not averaged.** If the perturbed value equals the anchor's
own value at that gene, the two passes are bit-identical and the delta is exactly
zero. That is not a small response, it is not a response at all. At
``mean_control`` it should never happen --- every read-out gene sits above run
1's 0.5 CP10K floor --- and at an individual-cell anchor it happens often. Each
one is asserted to be exactly zero, counted, listed in the manifest by
(gene, value), and left for the analysis to drop.

Layer 0 is written as a block like any other rather than skipped. It costs
nothing --- the same forward pass, one more hook --- and it is control C2: an
edit at gene *i* cannot have reached gene *j*'s token before the first block, so
the off-target delta there must be exactly 0.0. A control that is computed is a
check; a control that is only argued for is a claim.

The measured-absence contract is inherited from run 2's ``exact_jvp``: an adapter
that cannot supply hidden states returns ``None`` with a verbatim reason, and
this driver writes that reason into the manifest rather than an empty block.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from .. import paths
from .. import sibling
from . import pool as poolmod
from . import retrieval as retr



# --------------------------------------------------------------------------- #
# the layer plan
# --------------------------------------------------------------------------- #
def layer_plan(model_name: str, requested: tuple[int, ...] | None = None) -> tuple[int, ...]:
    """Which layers this block captures: the control layer plus the sweep.

    ``EMBEDDING_LAYER`` leads the list because it is the cheapest possible check
    on the hook attribution, and because a block whose layer 0 is not exactly
    zero off-target is not a null result --- it is unwired.
    """
    if requested:
        return tuple(sorted(set(int(x) for x in requested)))
    return (C.EMBEDDING_LAYER,) + C.sweep_layers(model_name)


# --------------------------------------------------------------------------- #
# one query's input rows
# --------------------------------------------------------------------------- #
def query_batch(x0: np.ndarray, col: int, values_native: np.ndarray) -> np.ndarray:
    """Row 0 is the baseline; row ``1 + vi`` sets gene ``col`` to grid value ``vi``.

    Deliberately not run 2's ``query_batch``: that one brackets every value with
    ``(v - h, v, v + h)`` for a central difference this run does not compute.
    """
    rows = [np.asarray(x0, dtype=np.float32)]
    for v in np.asarray(values_native, dtype=np.float64):
        x = np.asarray(x0, dtype=np.float32).copy()
        x[int(col)] = np.float32(max(float(v), C.NATIVE_FLOOR))
        rows.append(x)
    batch = np.stack(rows)
    differs = np.flatnonzero((batch[1:] != batch[0]).any(axis=0))
    if differs.size > 1 or (differs.size == 1 and int(differs[0]) != int(col)):
        raise RuntimeError(
            f"the perturbed rows differ from the baseline row at input columns "
            f"{differs.tolist()} rather than only at the perturbed column {col}; the "
            "delta would mix the intervention with a second input change"
        )
    return batch


# --------------------------------------------------------------------------- #
# the sweep over one anchor profile
# --------------------------------------------------------------------------- #
def probe_anchor(
    model,
    anchor_cp10k: np.ndarray,
    perturbed: pd.DataFrame,
    layers: tuple[int, ...],
    grid_cp10k: tuple[float, ...] = C.VALUE_GRID_CP10K,
    half_seed: int = C.HALF_SPLIT_SEED,
    log_every: int = 25,
    tag: str = "probe",
) -> dict:
    """Every perturbed gene at one anchor profile, reduced as it goes.

    Model-agnostic on purpose: it touches only ``to_native``, ``n_readout`` and
    ``hidden_batch``, so the whole estimator path runs against the numpy toys in
    ``tests/toy.py`` on a machine with no GPU and no checkpoint. That is what
    makes the acceptance gate --- a propagating toy that must be retrieved and an
    isolated toy that must sit at chance --- runnable at all.
    """
    cols = perturbed["readout_col"].to_numpy(dtype=np.int64)
    gene_ids = perturbed["gene_id"].to_numpy().astype(str)
    n_p, n_v, n_r = cols.size, len(grid_cp10k), int(model.n_readout)
    if cols.min() < 0 or cols.max() >= n_r:
        raise RuntimeError(
            f"a perturbed gene's read-out column ({cols.min()}..{cols.max()}) falls "
            f"outside the {n_r} read-out positions the model was configured with"
        )
    x0 = np.asarray(model.to_native(np.asarray(anchor_cp10k, dtype=np.float64)), dtype=np.float32)
    if x0.size != n_r:
        raise RuntimeError(
            f"the anchor profile has {x0.size} entries but the model reads out {n_r} "
            "genes; the anchor and the read-out table are not the pair this block needs"
        )
    values_native = np.asarray(
        model.to_native(np.asarray(grid_cp10k, dtype=np.float64)), dtype=np.float64
    )
    assignment = poolmod.half_assignment(n_r, seed=half_seed)

    #: a query whose perturbed value already equals the anchor's own value at that
    #: gene runs two identical inputs and is not a measurement. The comparison is
    #: made on the float32 values the batch actually carries, not on the float64
    #: grid, because a no-op is defined by the two input rows being bit-identical
    #: -- which is what makes the delta exactly zero rather than merely small.
    row_values = np.array(
        [np.float32(max(float(v), C.NATIVE_FLOOR)) for v in values_native], dtype=np.float32
    )
    no_op = np.zeros((n_p, n_v), dtype=bool)
    for pi, col in enumerate(cols):
        no_op[pi] = row_values == x0[int(col)]

    arrays: dict[int, dict[str, np.ndarray]] = {}
    n_pooled = np.zeros((n_p, n_v, C.N_POSITION_HALVES), dtype=np.int64)
    n_zero_positions = np.zeros((n_p, n_v), dtype=np.int64)
    started = time.time()
    baseline_probe: dict | None = None

    for pi, col in enumerate(cols):
        batch = query_batch(x0, int(col), values_native)
        hidden = model.hidden_batch(batch, layers, perturbed_col=int(col))
        if hidden is None:
            return {
                "available": False,
                "reason": str(
                    getattr(model, "_hidden_error", "the adapter returned no hidden states")
                ),
                "n_perturbed": int(n_p),
            }
        missing = [l for l in layers if int(l) not in hidden]
        if missing:
            raise RuntimeError(
                f"the adapter returned no hidden state for layer(s) {missing}; a layer "
                "silently absent would be read as a layer that did not move"
            )
        if pi == 0:
            d = int(hidden[int(layers[0])].shape[2])
            for l in layers:
                arrays[int(l)] = {
                    "pool": np.zeros((n_p, n_v, C.N_POSITION_HALVES, d), dtype=np.float32),
                    "pool_unit": np.zeros((n_p, n_v, C.N_POSITION_HALVES, d), dtype=np.float32),
                    "norm": np.zeros((n_p, n_v, n_r), dtype=np.float32),
                    "self_delta": np.zeros((n_p, n_v, d), dtype=np.float32),
                }
            baseline_probe = _baseline_pairing_probe(model, x0, cols, layers, hidden)

        for l in layers:
            h = np.asarray(hidden[int(l)], dtype=np.float64)
            if h.shape[:2] != (1 + n_v, n_r):
                raise RuntimeError(
                    f"layer {l} came back with shape {h.shape}; the driver expects "
                    f"({1 + n_v}, {n_r}, d) in read-out order, one baseline row then one "
                    "row per grid value"
                )
            if h.shape[2] != arrays[int(l)]["pool"].shape[3]:
                raise RuntimeError(
                    f"layer {l} changed width from {arrays[int(l)]['pool'].shape[3]} to "
                    f"{h.shape[2]} between perturbed genes"
                )
            base = h[0]
            for vi in range(n_v):
                delta = h[1 + vi] - base
                if no_op[pi, vi] and not poolmod.is_no_op(delta, self_col=int(col)):
                    raise RuntimeError(
                        f"gene {gene_ids[pi]} at value {grid_cp10k[vi]} CP10K is a no-op "
                        f"(the anchor already sits there) but layer {l} moved by "
                        f"{float(np.abs(delta).max()):.6g}; two bit-identical inputs "
                        "through a frozen model must give exactly zero"
                    )
                s = poolmod.pool_query(delta, assignment, int(col))
                arrays[int(l)]["pool"][pi, vi] = s.pool
                arrays[int(l)]["pool_unit"][pi, vi] = s.pool_unit
                arrays[int(l)]["norm"][pi, vi] = s.norm
                arrays[int(l)]["self_delta"][pi, vi] = s.self_delta
                if l == layers[-1]:
                    n_pooled[pi, vi] = s.n_pooled
                    n_zero_positions[pi, vi] = s.n_zero_positions
        if log_every and (pi + 1) % log_every == 0:
            rate = (time.time() - started) / (pi + 1)
            print(
                f"    [{tag}] gene {pi + 1}/{n_p} ({rate:.2f}s per gene, "
                f"{rate * (n_p - pi - 1) / 60:.1f} min left)",
                flush=True,
            )

    no_op_rows = [
        {"perturbed_gene_id": str(gene_ids[pi]), "value_cp10k": float(grid_cp10k[vi])}
        for pi, vi in zip(*np.nonzero(no_op))
    ]
    return {
        "available": True,
        # Whether the perturbed gene's own position is a measurable quantity at all.
        # False for an autoregressive model that right-shifts counts, where a gene's
        # count is never an input at its own position --- the positive control and the
        # propagation ratio's denominator are then absences, not zeros.
        "self_position_defined": bool(getattr(model, "hidden_self_position_defined", True)),
        "arrays": arrays,
        "assignment": assignment,
        "perturbed_gene_id": gene_ids,
        "readout_col": cols,
        "value_grid_cp10k": np.asarray(grid_cp10k, dtype=np.float64),
        "diagnostics": {
            "n_perturbed": int(n_p),
            "n_values": int(n_v),
            "n_readout": int(n_r),
            "d_model": int(arrays[int(layers[0])]["pool"].shape[3]),
            "input_rows_per_gene": int(1 + n_v),
            "input_rows_per_gene_run2": C.RUN2_INPUT_ROWS_PER_GENE,
            "n_no_op_queries": int(no_op.sum()),
            "no_op_queries": no_op_rows,
            "no_op_rule": (
                "a query whose perturbed native value equals the anchor's own native "
                "value at that gene; the two passes are bit-identical, the delta is "
                "exactly zero, and it is asserted to be so rather than assumed"
            ),
            "half_split": {
                "seed": int(half_seed),
                "n_half_a": int((assignment == 0).sum()),
                "n_half_b": int((assignment == 1).sum()),
                "self_position_excluded": bool(C.EXCLUDE_SELF_POSITION),
                "median_pooled_a": float(np.median(n_pooled[:, :, 0])),
                "median_pooled_b": float(np.median(n_pooled[:, :, 1])),
            },
            "n_zero_off_target_positions_median": float(np.median(n_zero_positions)),
            "baseline_pairing": baseline_probe,
            "seconds": round(time.time() - started, 1),
        },
    }


def _baseline_pairing_probe(model, x0, cols, layers, hidden_first) -> dict:
    """Does the baseline depend on which column is declared perturbed? Measure it.

    The driver takes its baseline from row 0 of the same batch, under the same
    ``perturbed_col``, because for scGPT and TranscriptFormer the sequence
    composition is a function of that column. That is an argument until it is a
    number, so the first gene's baseline is recomputed under a *different*
    declared column and the difference is recorded. A non-zero difference is not
    a failure --- it is the evidence that per-column pairing was necessary.
    """
    if cols.size < 2:
        return {"measured": False, "reason": "fewer than two perturbed genes in the block"}
    other = int(cols[1])
    alt = model.hidden_batch(np.asarray(x0, dtype=np.float32)[None, :], layers, perturbed_col=other)
    if alt is None:
        return {"measured": False, "reason": str(getattr(model, "_hidden_error", "unavailable"))}
    # The two passes declare DIFFERENT perturbed columns, so each writes its `nan`
    # self position at a different index on a model that has one. A plain `max`
    # then returns nan and the probe reports nothing for exactly the model where
    # composition depends most strongly on the declared column --- TranscriptFormer
    # forces the perturbed gene to sequence position 0, so the entire causal
    # conditioning prefix of every other gene is a function of it, a far larger
    # change than scGPT's one appended token. Both declared columns are excluded
    # and the comparison is made over the positions that are defined in both.
    out = {}
    excluded = sorted({int(cols[0]), other})
    for l in layers:
        a = np.asarray(hidden_first[int(l)], dtype=np.float64)[0]
        b = np.asarray(alt[int(l)], dtype=np.float64)[0]
        d = np.abs(np.delete(a, excluded, axis=0) - np.delete(b, excluded, axis=0))
        if not np.isfinite(d).all():
            raise RuntimeError(
                f"layer {l} carries a non-finite value at a position that is neither "
                f"declared column ({excluded}); only the perturbed gene's own position "
                "may be undefined, so this is a broken capture rather than an absence"
            )
        out[str(int(l))] = float(d.max())
    return {
        "measured": True,
        "declared_column_a": int(cols[0]),
        "declared_column_b": other,
        "excluded_positions": excluded,
        "excluded_reason": (
            "both declared columns are dropped from the comparison: a model that "
            "does not read a gene's count at its own position writes nan there, at "
            "a different index in each of the two passes"
        ),
        "max_abs_baseline_difference_by_layer": out,
        "note": (
            "the same anchor input, declared perturbed at two different columns. Zero "
            "means the layout is composition-independent; non-zero means a baseline "
            "shared across genes would have contaminated every delta in the block"
        ),
    }


# --------------------------------------------------------------------------- #
# anchors
# --------------------------------------------------------------------------- #
def anchor_profiles(
    anchor: str, readout: pd.DataFrame, cells: pd.DataFrame | None, cell_line: str,
    n_cells: int = C.N_ANCHOR_CELLS, seed: int = C.ANCHOR_CELL_SEED,
) -> dict:
    """The anchor's CP10K profiles, kept UNAVERAGED, and the record of where from.

    ``mean_control`` and ``uniform_virtual`` come straight from run 2's
    ``features/extract.py:build_anchors`` --- the same construction every scored
    arm in the paper was read at, so a null here speaks to the published result.

    ``individual_cells`` does not, and the difference is deliberate. Run 2 uses
    four demonstration cells and *averages the responses over them*, which is the
    right call for a response feature and destroys exactly the axis a trained
    probe needs. Here ``N_ANCHOR_CELLS`` fresh non-targeting control cells are
    drawn through run 3's ``ingest/cells.py:load_control_cells`` --- the same
    filter funnel run 1 applied, each cell converted to CP10K by its own
    full-transcriptome total --- and kept separate, one block per cell.
    """
    extract = sibling.run2("features.extract")
    built = extract.build_anchors(readout, cells)
    if anchor != "individual_cells":
        if anchor not in built:
            raise RuntimeError(
                f"run 2's anchor construction does not supply '{anchor}'; the probe must "
                "read at an anchor the scored arms were read at, not at a new one"
            )
        spec = dict(built[anchor])
        spec["source"] = "run 2 features/extract.py:build_anchors, unmodified"
        return spec

    cellmod = sibling.run3("ingest.cells")
    key = f"{cell_line}_matrix_index"
    if key not in readout.columns:
        raise RuntimeError(
            f"the read-out table has no '{key}' column, so its genes cannot be located in "
            "the raw screen matrix and the control cells cannot be read"
        )
    matrix_cols = readout[key].to_numpy(dtype=np.int64)
    loaded = cellmod.load_control_cells(cell_line, paths.SCREENS_DIR, matrix_cols)
    got = np.asarray(loaded["gene_ids_matrix"])[matrix_cols].astype(str)
    want = readout["gene_id"].to_numpy().astype(str)
    if not np.array_equal(got, want):
        raise RuntimeError(
            f"{int((got != want).sum())} read-out columns do not match the raw matrix gene "
            "ids; every anchor value would sit on the wrong gene"
        )
    cp10k = np.asarray(loaded["cp10k"], dtype=np.float64)
    rng = np.random.default_rng(seed)
    take = np.sort(rng.choice(cp10k.shape[0], size=min(n_cells, cp10k.shape[0]), replace=False))
    if take.size < n_cells:
        raise RuntimeError(
            f"only {take.size} control cells survive the filter in {cell_line}, fewer than "
            f"the {n_cells} the design asks for"
        )
    return {
        "profiles": [cp10k[k] for k in take],
        "labels": [f"control_cell_{int(k)}" for k in take],
        "description": (
            f"{take.size} non-targeting control cells of {cell_line}, drawn at seed {seed} "
            f"from the {cp10k.shape[0]} that survive run 1's filter funnel, each converted "
            "to CP10K by its own full-transcriptome total and kept UNAVERAGED --- one probe "
            "block per cell, because the frozen block layout has no cell axis and averaging "
            "them is exactly what run 2 did and what this anchor exists to undo"
        ),
        "source": "run 3 ingest/cells.py:load_control_cells over the local GSE264667 h5ad",
        "n_available": int(cp10k.shape[0]),
    }


# --------------------------------------------------------------------------- #
# writing a block
# --------------------------------------------------------------------------- #
def write_block(out_dir: Path, result: dict, readout_ids: np.ndarray, layers: tuple[int, ...]) -> dict:
    """The frozen on-disk layout, every array beside its own label arrays.

    Row order is a contract, so it is written down rather than implied: every
    ``npz`` carries the perturbed gene ids, the read-out gene ids where the array
    is indexed by them, and the value grid, and a reader joins on those.
    """
    out_dir = paths.ensure(out_dir)
    shapes: dict[str, dict] = {}
    for l in layers:
        arr = result["arrays"][int(l)]
        ldir = paths.ensure(out_dir / f"layer={int(l)}")
        np.savez(
            ldir / "pool.npz",
            pool=arr["pool"],
            pool_unit=arr["pool_unit"],
            perturbed_gene_id=result["perturbed_gene_id"],
            value_grid_cp10k=result["value_grid_cp10k"],
        )
        np.savez(
            ldir / "norms.npz",
            norm=arr["norm"],
            perturbed_gene_id=result["perturbed_gene_id"],
            readout_gene_id=readout_ids,
            value_grid_cp10k=result["value_grid_cp10k"],
        )
        np.savez(
            ldir / "self_pool.npz",
            self_delta=arr["self_delta"],
            perturbed_gene_id=result["perturbed_gene_id"],
            readout_gene_id=readout_ids,
            value_grid_cp10k=result["value_grid_cp10k"],
        )
        shapes[str(int(l))] = {k: list(v.shape) for k, v in arr.items()}
    np.savez(
        out_dir / "half_assignment.npz",
        half=result["assignment"],
        readout_gene_id=readout_ids,
        perturbed_gene_id=result["perturbed_gene_id"],
        perturbed_readout_col=result["readout_col"],
    )
    return shapes


# --------------------------------------------------------------------------- #
# checks over the block that was just written
# --------------------------------------------------------------------------- #
def verify(result: dict, layers: tuple[int, ...], tag: str) -> dict:
    """C1, C2 and C3 over real output, each printing the number it checked.

    These are checks and not analyses: they decide whether the block is readable
    as evidence at all. A block whose C1 fails is not a null result, it is an
    unwired model, and reporting it as a null would be the exact error the
    project's claims discipline warns about.
    """
    rows: list[dict] = []

    def add(name: str, ok: bool, detail: str) -> None:
        rows.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {tag}/{name}: {detail}", flush=True)

    def absent(name: str, detail: str) -> None:
        """A quantity this architecture cannot define, recorded rather than failed.

        A check that cannot be computed and a check that computed a bad number are
        different results, and collapsing them loses the distinction the whole run
        depends on. An absence carries its reason and is counted apart from both
        passes and failures.
        """
        rows.append({"check": name, "ok": None, "detail": detail})
        print(f"[ABSENT] {tag}/{name}: {detail}", flush=True)

    cols = result["readout_col"]
    #: TranscriptFormer is autoregressive over genes and right-shifts both tokens
    #: and counts, so a gene's own count is never an input at its own position --- a
    #: fact of the architecture, recorded by run 2 as `self_response_defined = False`
    #: and confirmed here by the layer-0 probe. C1 and the S1 ratio are both defined
    #: against that position, so for this model they are absences, not failures.
    self_defined = bool(result.get("self_position_defined", True))
    for l in layers:
        arr = result["arrays"][int(l)]
        prop = retr.propagation_ratio(arr["norm"], cols)
        self_norm = prop["self_norm"]
        off_max = prop["max_off_target_position_norm"]
        if int(l) == C.EMBEDDING_LAYER:
            add(
                f"C2/layer{int(l)}_off_target_exactly_zero",
                off_max == 0.0,
                f"max off-target position norm at the post-embedding layer is {off_max:.6g} "
                f"over {arr['norm'].shape[0]} genes x {arr['norm'].shape[1]} values; an edit "
                "cannot have reached another gene's token before the first block",
            )
        elif not self_defined:
            # The ratio's denominator does not exist for this model. The numerator
            # does, and it is the quantity the depth profile is actually read from,
            # so it is reported rather than discarded with the ratio.
            absent(
                f"S1/layer{int(l)}_propagation",
                f"the ratio is undefined: this model never reads a gene's count at "
                f"its own position, so the denominator does not exist. The off-target "
                f"norm itself is defined and is {off_max:.6g} at its maximum over "
                f"{self_norm.size} (gene, value) queries",
            )
        else:
            add(
                f"S1/layer{int(l)}_propagation",
                np.isfinite(prop["median_ratio"]),
                f"median ||delta_h off-target|| / ||delta_h self|| = "
                f"{prop['median_ratio']:.6g} over {prop['n_defined']} defined of "
                f"{self_norm.size} (gene, value) queries; max off-target position norm "
                f"{off_max:.6g}",
            )
        if not self_defined:
            absent(
                f"C1/layer{int(l)}_self_position_moved",
                "this model right-shifts tokens and counts, so a gene's own count is "
                "never an input at its own position; the positive control is "
                "unavailable by construction, not failed. Confirmed by the layer-0 "
                "probe: no position moves when a count is edited, because the count "
                "enters as an attention score modifier and not through the residual "
                "stream",
            )
        else:
            add(
                f"C1/layer{int(l)}_self_position_moved",
                float(np.median(self_norm)) > 0.0,
                f"median ||delta_h|| at the perturbed gene's own position is "
                f"{float(np.median(self_norm)):.6g}, zero for {prop['n_undefined']} of "
                f"{self_norm.size} queries",
            )
    n_no_op = result["diagnostics"]["n_no_op_queries"]
    if n_no_op == 0:
        # The assertion above compares nothing when there is nothing to compare.
        # Reporting that as PASS would put a control in the record that never ran
        # --- "it ran" is not "it is right", and neither is "it had no work to do".
        # At mean_control this is the expected state: every read-out gene sits
        # above run 1's 0.5 CP10K floor, so a zero-out query is never a no-op.
        absent(
            "C3/no_op_queries_exactly_zero",
            f"no query was a no-op in this block, so the exactly-zero comparison "
            f"never executed. This is the expected state at an anchor whose genes "
            f"all sit above the expression floor; it is not evidence that the "
            f"comparison would have passed. "
            f"{result['diagnostics']['n_perturbed'] * result['diagnostics']['n_values']} "
            "queries were checked for no-op status.",
        )
    else:
        add(
            "C3/no_op_queries_exactly_zero",
            True,
            f"{n_no_op} of "
            f"{result['diagnostics']['n_perturbed'] * result['diagnostics']['n_values']} "
            "queries were no-ops (the anchor already sat at the perturbed value); each "
            "was asserted to give a delta of exactly 0.0 at every layer, counted, and "
            "listed in the manifest for the analysis to drop",
        )
    ok = sum(1 for r in rows if r["ok"] is True)
    failed = sum(1 for r in rows if r["ok"] is False)
    missing = sum(1 for r in rows if r["ok"] is None)
    return {
        "n_passed": ok,
        "n_failed": failed,
        "n_absent": missing,
        "n_checks": len(rows),
        "checks": rows,
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run(
    model_name: str,
    cell_line: str,
    anchor: str,
    cache: Path,
    out_root: Path | None = None,
    scope: C.Scope = C.SCOPE,
    layers: tuple[int, ...] | None = None,
    device: str = "cuda",
    batch: int | None = None,
    n_cells: int = C.N_ANCHOR_CELLS,
    grid_cp10k: tuple[float, ...] = C.VALUE_GRID_CP10K,
    image: str = "unrecorded",
) -> dict:
    started = time.time()
    extract = sibling.run2("features.extract")
    scopemod = sibling.run2("features.scope")
    r2config = sibling.run2("config")
    base = sibling.run2("models.base")

    data = extract.load_inputs(model_name, cell_line)
    readout = data["readout"]
    readout_ids = readout["gene_id"].to_numpy().astype(str)
    on_disk = pd.read_parquet(paths.readout_table(cell_line))["gene_id"].to_numpy().astype(str)
    if not np.array_equal(readout_ids, on_disk):
        raise RuntimeError(
            f"the read-out table run 2's loader returned does not match "
            f"{paths.readout_table(cell_line)} row for row; the block's read-out order "
            "would not be the order its labels claim"
        )

    # run 2's Scope carries two extra counts for anchor subsets this run does not
    # use; 0 means "every gene in scope" on each, so the smoke count is threaded
    # into all three and full coverage stays full.
    r2scope = r2config.Scope(
        n_perturbed=scope.n_perturbed,
        n_perturbed_cells_anchor=scope.n_perturbed,
        n_perturbed_masked=scope.n_perturbed,
        strata=tuple(scope.strata),
    )
    perturbed = scopemod.select_perturbed(data["perturbations"], readout, r2scope, seed=C.SEED)
    coverage = scopemod.coverage_record(data["perturbations"], readout, r2scope, perturbed)
    print(
        f"[{model_name}/{cell_line}/{anchor}] {readout.shape[0]} read-out genes, "
        f"{perturbed.shape[0]} perturbed of {coverage['n_in_scope']} in scope "
        f"(coverage {coverage['mode']})",
        flush=True,
    )

    # run 5's subclass, not run 2's adapter: it is run 2's loader, tiling and
    # token mapping inherited unchanged, with `hidden_batch` added on top. Building
    # run 2's class here would silently drop the hidden-state readout this whole
    # run exists to take.
    # imported here rather than at module scope: `models/hidden.py` pulls torch and
    # run 2's model stack, and `probe_anchor` above must stay importable without
    # either so the estimator can be tested on a machine with no GPU.
    from ..models import hidden as hidden_models

    model = hidden_models.build(
        model_name, cache, data["universe"], data["vocab"], device=device
    )
    model.load()
    if batch:
        model.batch_size = int(batch)
    # configure() is called once per (model, cell line) with the MEAN-CONTROL
    # profile and reused for every anchor, exactly as run 2 does: scGPT's
    # quantile bin edges and TranscriptFormer's sequence ordering therefore
    # derive from mean control whatever the anchor is. Changing that would make
    # this probe incomparable with the blocks run 3 scored.
    control = readout["pooled_control_cp10k"].to_numpy(dtype=np.float64)
    kwargs = (
        {"supplied_mode": r2config.SCGPT_PRODUCTION_MODE} if model_name == "scgpt" else {}
    )
    layout = model.configure(readout, control, cell_line, **kwargs)
    print(f"[{model_name}/{cell_line}] layout {json.dumps(layout.as_dict(), default=str)}", flush=True)

    layers = layer_plan(model_name, layers)
    spec = anchor_profiles(anchor, readout, data["cells"], cell_line, n_cells=n_cells)
    profiles, labels = spec["profiles"], spec["labels"]
    print(
        f"[{model_name}/{cell_line}/{anchor}] {len(profiles)} anchor profile(s), "
        f"layers {list(layers)}",
        flush=True,
    )

    written = []
    for ci, profile in enumerate(profiles):
        # one block per anchor profile: the frozen block layout has no cell axis,
        # and collapsing the cells is what run 2 did and what this run undoes
        name = anchor if len(profiles) == 1 else f"{anchor}_{labels[ci]}"
        out_dir = Path(out_root) / f"anchor={name}" if out_root else paths.anchor_dir(
            model_name, cell_line, name
        )
        tag = f"{model_name}/{cell_line}/{name}"
        result = probe_anchor(
            model, profile, perturbed, layers, grid_cp10k=grid_cp10k, tag=tag
        )
        manifest: dict = {
            "model": model_name,
            "cell_line": cell_line,
            "anchor": name,
            "anchor_family": anchor,
            "anchor_cell_label": labels[ci],
            "anchor_description": spec["description"],
            "anchor_source": spec["source"],
            "layout": layout.as_dict(),
            "layout_configured_from": (
                "the pooled mean-control profile, once per (model, cell line), reused for "
                "every anchor exactly as run 2 does (features/extract.py:467,473)"
            ),
            "scope": {"n_perturbed": int(perturbed.shape[0]), "strata": list(scope.strata)},
            "coverage": coverage,
            "value_grid_cp10k": list(grid_cp10k),
            "sign_convention": (
                "perturbed minus baseline; a positive entry means raising the perturbed "
                "gene's input moved the read-out gene's hidden state in that direction. "
                "Run 2's convention, unchanged, so the two runs' deltas have the same sign"
            ),
            "layers": [int(l) for l in layers],
            "sweep_layers": [int(l) for l in C.sweep_layers(model_name)],
            "control_layers": [int(C.EMBEDDING_LAYER)],
            "layer_definition": (
                f"layer l is the OUTPUT of transformer block l, 1-based, captured by "
                f"forward hook on that block. Layer {C.EMBEDDING_LAYER} is the "
                "post-embedding input to block 1 -- the token state before any attention "
                "-- and is a control, not a sweep point: an edit at gene i cannot have "
                "reached gene j's token there, so the off-target delta must be exactly "
                "0.0 and anything else means the hooks are mis-attributed. Where each "
                "hook sat for this model is recorded per layer under hidden_source"
            ),
            "hidden_source": {
                str(int(l)): str(model.hidden_source(int(l))) for l in layers
            },
            "pooling": (
                "read-out positions are partitioned once per block into two disjoint "
                f"halves at seed {C.HALF_SPLIT_SEED}; the perturbed gene's own position "
                "is removed from both before pooling and stored separately in "
                "self_pool.npz as the positive control. pool is the MEAN delta-h over "
                "each half; pool_unit is the mean over per-position unit-normalised "
                "delta-h, so identity carried by direction can be separated from "
                "identity carried by magnitude. A position whose delta-h is exactly zero "
                "contributes the zero vector to pool_unit rather than a nan, and the "
                "count of such positions travels in the manifest"
            ),
            "excluded_self_position": bool(C.EXCLUDE_SELF_POSITION),
            "seed": C.SEED,
            "half_split_seed": int(C.HALF_SPLIT_SEED),
            "mask_spec": model.mask_spec(),
            "substitutions": list(getattr(model, "substitutions", [])),
            "contract": json.loads(model.contract().to_json()) if hasattr(model, "contract") else {},
        }
        if not result["available"]:
            manifest["hidden_states"] = {
                "available": False,
                "reason": result["reason"],
                "note": (
                    "a measured absence, recorded verbatim: no layer arrays were written, "
                    "because writing zeros here would be indistinguishable from a model "
                    "that propagates nothing"
                ),
            }
            paths.ensure(out_dir)
            (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
            print(f"[{tag}] hidden states unavailable: {result['reason']}", flush=True)
            written.append({"anchor": name, "available": False})
            continue

        shapes = write_block(out_dir, result, readout_ids, layers)
        readout.to_parquet(out_dir / "readout_genes.parquet", index=False)
        perturbed.to_parquet(out_dir / "perturbed_genes.parquet", index=False)
        manifest["hidden_states"] = {"available": True}
        manifest["array_shapes"] = shapes
        manifest["diagnostics"] = result["diagnostics"]
        manifest["inputs"] = {
            "readout_table": str(paths.readout_table(cell_line)),
            "perturbations_table": str(paths.perturbations_table(cell_line)),
        }
        manifest["verification"] = verify(result, layers, tag)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        (out_dir / "environment.json").write_text(
            json.dumps(
                base.environment_manifest(
                    model_name,
                    image,
                    {"device": device, "cell_line": cell_line, "anchor": name},
                ),
                indent=2,
                default=str,
            )
        )
        print(
            f"[{tag}] wrote {len(layers)} layers to {out_dir} in "
            f"{result['diagnostics']['seconds']}s",
            flush=True,
        )
        written.append({"anchor": name, "available": True, "path": str(out_dir)})

    print(
        f"[{model_name}/{cell_line}/{anchor}] {len(written)} block(s) in "
        f"{round(time.time() - started, 1)}s",
        flush=True,
    )
    return {"blocks": written, "layers": [int(l) for l in layers]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=list(paths.MODELS))
    ap.add_argument("--cell-line", required=True, choices=list(paths.CELL_LINES))
    ap.add_argument("--anchor", default=C.PRIMARY_ANCHOR, choices=list(C.ANCHORS))
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--values", type=float, nargs="*", default=list(C.VALUE_GRID_CP10K))
    ap.add_argument("--n-cells", type=int, default=C.N_ANCHOR_CELLS)
    ap.add_argument("--n-perturbed", type=int, default=None)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help=(
            f"the smoke scope ({C.SMOKE_SCOPE.n_perturbed} perturbed genes) and the first "
            "and last layers only: enough to exercise every reassembly branch and run "
            "C1-C3 over real output in minutes, never enough to read a number from"
        ),
    )
    ap.add_argument("--image", default="unrecorded")
    args = ap.parse_args(argv)

    scope = C.SMOKE_SCOPE if args.smoke else C.SCOPE
    if args.n_perturbed is not None:
        scope = C.Scope(n_perturbed=int(args.n_perturbed), strata=scope.strata)
    layers = tuple(args.layers) if args.layers else None
    if layers is None and args.smoke:
        layers = (C.EMBEDDING_LAYER, C.sweep_layers(args.model)[0], C.sweep_layers(args.model)[-1])
    cache = Path(args.cache) if args.cache else paths.checkpoint_cache()
    out_root = Path(args.out) if args.out else None
    fail_dir = out_root or paths.probe_dir(args.model, args.cell_line)
    try:
        run(
            args.model,
            args.cell_line,
            args.anchor,
            cache,
            out_root=out_root,
            scope=scope,
            layers=layers,
            device=args.device,
            batch=args.batch,
            n_cells=args.n_cells,
            grid_cp10k=tuple(args.values),
            image=args.image,
        )
    except Exception:
        paths.ensure(Path(fail_dir))
        (Path(fail_dir) / "failure.txt").write_text(traceback.format_exc())
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
