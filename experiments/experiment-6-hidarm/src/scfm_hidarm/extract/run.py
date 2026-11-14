"""Extract one model's projected hidden-state response block, one cell line.

    PYTHONPATH=src python -m scfm_hidarm.extract.run --model scgpt \
        --cell-line hepg2 --stage basis
    PYTHONPATH=src python -m scfm_hidarm.extract.run --model scgpt \
        --cell-line hepg2 --stage main
    PYTHONPATH=src python -m scfm_hidarm.extract.run --model scgpt \
        --cell-line hepg2 --stage main --smoke

**This driver emits hidden states only. It never computes, stores or scores a
decoder-head value.** Runs 2, 3 and 4 measured the head, those results are
published, and run 6 does not recompute, re-normalise or re-score them; the
head comparison is made afterwards, against those runs as published. A
head-derived array anywhere in run 6's output tree is a verification failure,
not a bonus.

Two stages, and why they cannot be one
--------------------------------------
``--stage basis`` runs the 64 basis genes and accumulates a ``d x d``
second-moment matrix, which becomes the PCA basis ``W``. ``--stage main`` runs
the scored perturbations, projects every slot through that ``W`` as it goes, and
writes only the projected block. The basis must exist before the main pass can
project, which is what makes them two passes rather than one; and because the
basis is fitted on perturbations that are disjoint from every scored one, the
main pass needs no fold argument and run 3's runner needs no modification.

The main pass accumulates a SECOND ``d x d`` matrix over the scored
perturbations at the same time. That one is never used to project anything: it
is the captured-energy diagnostic, the number that says how much of the scored
perturbations' delta-h the basis actually spans. It is free (33 MB at d = 2048)
and it is the only honest answer to the objection that a basis fitted on
weak-effect knockdowns may under-span the strong-effect ones. The error runs one
way --- under-spanning loses signal and cannot manufacture it --- so the failure
mode is a false null, never a false positive, but "one-directional" is a reason
to measure it rather than a reason not to.

What never touches disk
-----------------------
Raw ``delta_h``. It is 15 GB (scGPT/HepG2) to 94 GB (TranscriptFormer/Jurkat)
per layer at five doses, and 2.2x that again for ten slots. Each gene's
``(n_readout, d)`` slot is projected on the device and the raw array is dropped
before the next gene runs.

The PROJECTED accumulator does touch disk, and has to: at ``K_STORE = 64`` it is
34 GB (HepG2) to 55 GB (Jurkat) against this machine's 70 GB of RAM, so it is
memory-mapped rather than allocated (``config.STORE_MEMMAPPED``). That is an
execution decision and changes no number --- the same float32 bytes are written
either way.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from .. import paths, sibling
from ..basis import fit as basisfit
from . import slots as slotmod


# --------------------------------------------------------------------------- #
# building the adapter
# --------------------------------------------------------------------------- #
def build_model(model: str, cell_line: str, device: str, cache: Path | None = None):
    """One hidden-state adapter, configured exactly as run 2 configures it.

    The frozen three go through run 5's ``BUILDERS`` unchanged; the two
    fine-tuned keys go through run 6's own bridge, because a fine-tuned arm is
    an already-built model rather than a checkpoint path. Either way the object
    that comes back is the run 2 adapter that produced the published responses,
    with ``hidden_batch`` added on top --- which is what makes "the head said
    this" and "the hidden state said that" two readings of one model rather than
    two models.
    """
    if model in C.FT_MODELS:
        from ..models import hidden_ft

        q = hidden_ft.build(model, cell_line, device=device)
        readout = pd.read_parquet(paths.readout_table(cell_line))
        anchor = readout["pooled_control_cp10k"].to_numpy(dtype=np.float64)
        layout = q.configure(readout, anchor, cell_line)
        notes = dict(layout.as_dict().get("notes", {}))
        notes.update(q.layout_notes())
        return q, readout, {**layout.as_dict(), "notes": notes}

    extract = sibling.run2("features.extract")
    r2config = sibling.run2("config")
    hidden_models = sibling.run5("models.hidden")

    data = extract.load_inputs(model, cell_line)
    readout = data["readout"]
    on_disk = pd.read_parquet(paths.readout_table(cell_line))["gene_id"].to_numpy().astype(str)
    if not np.array_equal(readout["gene_id"].to_numpy().astype(str), on_disk):
        raise RuntimeError(
            f"the read-out table run 2's loader returned does not match "
            f"{paths.readout_table(cell_line)} row for row; the block's read-out order "
            "would not be the order its labels claim"
        )
    q = hidden_models.build(
        model, cache or paths.checkpoint_cache(), data["universe"], data["vocab"], device=device
    )
    q.load()
    # configure() is called once per (model, cell line) with the MEAN-CONTROL
    # profile, exactly as run 2 does: scGPT's quantile bin edges and
    # TranscriptFormer's sequence ordering therefore derive from mean control.
    # Changing that would make this block incomparable with the ones run 3 scored.
    control = readout["pooled_control_cp10k"].to_numpy(dtype=np.float64)
    kwargs = {"supplied_mode": r2config.SCGPT_PRODUCTION_MODE} if model == "scgpt" else {}
    layout = q.configure(readout, control, cell_line, **kwargs)
    return q, readout, layout.as_dict()


def anchor_profile(readout: pd.DataFrame) -> np.ndarray:
    """The ``mean_control`` anchor, from run 2's own construction.

    Read at the anchor every scored arm in the paper was read at, so a number
    here speaks to the published result. ``individual_cells`` would be ~16x the
    cost and would speak to nothing already measured.
    """
    extract = sibling.run2("features.extract")
    built = extract.build_anchors(readout, None)
    if C.ANCHOR not in built:
        raise RuntimeError(
            f"run 2's anchor construction does not supply {C.ANCHOR!r}; run 6 must read at "
            "an anchor the scored arms were read at, not at a new one"
        )
    return np.asarray(built[C.ANCHOR]["profiles"][0], dtype=np.float64)


def scope_perturbations(model: str, cell_line: str, readout: pd.DataFrame, scope: C.Scope):
    """The perturbations the main pass runs, in run 2's own scope order.

    Run 2's ``select_perturbed`` is the definition of "in scope" every later run
    inherits, and run 3's bundle rows are in its order. Recomputing the set here
    with a different rule would make ``pert_index`` mean something the bundle
    does not agree with.
    """
    extract = sibling.run2("features.extract")
    scopemod = sibling.run2("features.scope")
    r2config = sibling.run2("config")
    data = extract.load_inputs("scgpt" if model in C.FT_MODELS else model, cell_line)
    r2scope = r2config.Scope(
        n_perturbed=scope.n_perturbed,
        n_perturbed_cells_anchor=scope.n_perturbed,
        n_perturbed_masked=scope.n_perturbed,
        strata=tuple(scope.strata),
    )
    perturbed = scopemod.select_perturbed(data["perturbations"], readout, r2scope, seed=C.SEED)
    coverage = scopemod.coverage_record(data["perturbations"], readout, r2scope, perturbed)
    return perturbed, coverage


def assert_bundle_order(cell_line: str, gene_ids: np.ndarray, readout_ids: np.ndarray) -> dict:
    """Run 6's rows and columns must be run 3's, element for element.

    Row order is a contract, and a contract you assert rather than assume. If
    the perturbed genes came back in a different order the block would install
    cleanly, score cleanly, and put every perturbation's features on a different
    perturbation's target.
    """
    t = np.load(paths.run3_targets(cell_line), allow_pickle=False)
    want_p = t["pert_gene_id"].astype(str)
    want_r = t["readout_gene_id"].astype(str)
    got_p = np.asarray(gene_ids).astype(str)
    got_r = np.asarray(readout_ids).astype(str)
    ok_r = np.array_equal(got_r, want_r)
    ok_p = np.array_equal(got_p, want_p)
    if not ok_r:
        raise RuntimeError(
            f"{int((got_r != want_r).sum()) if got_r.shape == want_r.shape else 'a different number of'} "
            "read-out genes differ from run 3's bundle; every feature would sit on the "
            "wrong gene"
        )
    return {
        "readout_rows_match": bool(ok_r),
        "perturbed_rows_match": bool(ok_p),
        "n_perturbed": int(got_p.size),
        "n_perturbed_bundle": int(want_p.size),
        "detail": (
            f"{int(got_p.size)} perturbed genes, {'in bundle order' if ok_p else 'a subset or a different order (smoke or basis pass)'}; "
            f"{int(got_r.size)} read-out genes in bundle order"
        ),
    }


# --------------------------------------------------------------------------- #
# the basis pass
# --------------------------------------------------------------------------- #
def fit_basis(
    model: str,
    cell_line: str,
    device: str = "cuda",
    layers: tuple[int, ...] | None = None,
    n_basis: int = C.N_BASIS_GENES,
    log_every: int = 8,
) -> dict:
    """Accumulate ``sum dh dh^T`` over the basis genes and write ``W`` per layer.

    Every slot of every basis gene contributes, pooled, because the basis is
    SHARED across the ten slots (``config.BASIS_SHARED_ACROSS_SLOTS``): a
    per-slot basis would put each dose in its own coordinate frame and "how the
    response changes with dose" would stop being a comparison of like
    coordinates. It is fitted on RAW delta-h, so the high-dose slots dominate
    the leading components --- deliberate, because RMS-normalising per slot
    would remove exactly the dose-magnitude axis run 6 exists to recover.
    """
    q, readout, layout = build_model(model, cell_line, device)
    x0_cp10k = anchor_profile(readout)
    x0 = np.asarray(q.to_native(x0_cp10k), dtype=np.float32)
    values_native = np.asarray(q.to_native(np.asarray(C.VALUE_GRID_CP10K, dtype=np.float64)))
    step = float(C.GDT_STEP[model])
    layers = tuple(layers) if layers else C.capture_layers(model)
    sweep = tuple(l for l in layers if l != C.EMBEDDING_LAYER)
    d = int(C.D_MODEL[model])

    genes = basisfit.select_basis_genes(cell_line, n=n_basis)
    leak = basisfit.leakage_check(cell_line, genes)
    if leak["n_overlap"]:
        raise RuntimeError(
            f"{leak['detail']}: a basis gene that is also a scored perturbation would let "
            "a scored perturbation shape the basis it is scored through"
        )
    print(f"[basis] {model}/{cell_line}: {leak['detail']}", flush=True)

    acc = {int(l): basisfit.SecondMoment(d, centred=C.BASIS_CENTRED) for l in sweep}
    cols = genes["readout_col"].to_numpy(dtype=np.int64)
    gene_ids = genes["gene_id"].to_numpy().astype(str)
    started = time.time()
    for pi, col in enumerate(cols):
        batch = slotmod.query_rows(x0, int(col), values_native, step)
        hidden = q.hidden_batch(batch, layers, perturbed_col=int(col))
        if hidden is None:
            return {
                "available": False,
                "reason": str(getattr(q, "_hidden_error", "the adapter returned no hidden states")),
            }
        for l in sweep:
            built = slotmod.build_slots(hidden[int(l)], values_native, step)
            for form in C.SLOT_FORMS:
                for vi in range(len(C.VALUE_GRID_CP10K)):
                    x = built[form][vi]
                    # the perturbed gene's own position is dropped: it is defined
                    # for two of the three architectures and undefined (nan) for
                    # the third, and a basis fitted partly on nan is not a basis
                    acc[int(l)].update(np.delete(x, int(col), axis=0))
        if log_every and (pi + 1) % log_every == 0:
            rate = (time.time() - started) / (pi + 1)
            print(
                f"    [basis {model}/{cell_line}] gene {pi + 1}/{cols.size} "
                f"({rate:.2f}s per gene, {rate * (cols.size - pi - 1) / 60:.1f} min left)",
                flush=True,
            )

    written = {}
    for l in sweep:
        M = acc[int(l)].matrix()
        eig = basisfit.eigendecompose(M, C.K_STORE)
        out = basisfit.write_basis(
            model=model,
            cell_line=cell_line,
            layer=int(l),
            W=eig["W"],
            eigenvalues=eig["eigenvalues"],
            explained=eig["explained"],
            genes=genes,
            provenance={
                "stage": "basis",
                "anchor": C.ANCHOR,
                "layout": layout,
                "value_grid_cp10k": list(C.VALUE_GRID_CP10K),
                "gdt_step": step,
                "gdt_estimator": C.GDT_ESTIMATOR,
                "n_slots": C.N_SLOTS,
                "slots_pooled": True,
                "self_position_excluded_from_fit": True,
                "centred": C.BASIS_CENTRED,
                "whitened": C.BASIS_WHITENED,
                "slot_rescaled": C.BASIS_SLOT_RESCALED,
                "n_vectors": int(acc[int(l)].n),
                "leakage": leak,
                "seconds": round(time.time() - started, 1),
            },
        )
        written[str(int(l))] = str(out)
        print(f"[basis] {model}/{cell_line} layer {l}: wrote {out}", flush=True)
    return {"available": True, "written": written, "n_genes": int(cols.size),
            "seconds": round(time.time() - started, 1)}


# --------------------------------------------------------------------------- #
# the main pass
# --------------------------------------------------------------------------- #
def extract_line(
    model: str,
    cell_line: str,
    device: str = "cuda",
    layers: tuple[int, ...] | None = None,
    scope: C.Scope = C.SCOPE,
    out_root: Path | None = None,
    log_every: int = 25,
) -> dict:
    """Every scored perturbation, projected as it goes, written per layer.

    Two blocks come out of each layer and both are real ``delta_h``:
    ``hid`` through the fitted PCA basis, and ``hidrp`` through a frozen random
    orthonormal basis of the same width. ``hidrp`` costs one extra matmul and no
    extra forward pass, and it is what separates "the PCA basis is doing the
    work" from "delta_h carries pair-specific signal in any k-dimensional
    slice". The two remaining nulls --- ``hidperm`` and ``hidrand`` --- are
    derived from these at stage time from a seed, because storing a permutation
    of an array already on disk buys nothing and costs another 190 GB.

    Layer 0 is captured but no block is written for it: its off-target delta is
    exactly zero by construction, so the block would be a zero matrix with one
    non-zero column. It is control C2 and its assertion is what this pass does
    with it.
    """
    from ..nulls import random_orthonormal

    q, readout, layout = build_model(model, cell_line, device)
    readout_ids = readout["gene_id"].to_numpy().astype(str)
    x0_cp10k = anchor_profile(readout)
    x0 = np.asarray(q.to_native(x0_cp10k), dtype=np.float32)
    values_native = np.asarray(q.to_native(np.asarray(C.VALUE_GRID_CP10K, dtype=np.float64)))
    step = float(C.GDT_STEP[model])
    layers = tuple(layers) if layers else C.capture_layers(model)
    sweep = tuple(l for l in layers if l != C.EMBEDDING_LAYER)
    d = int(C.D_MODEL[model])
    k = int(C.K_STORE)

    perturbed, coverage = scope_perturbations(model, cell_line, readout, scope)
    cols = perturbed["readout_col"].to_numpy(dtype=np.int64)
    gene_ids = perturbed["gene_id"].to_numpy().astype(str)
    n_p, n_r, n_v = cols.size, int(q.n_readout), len(C.VALUE_GRID_CP10K)
    order = assert_bundle_order(cell_line, gene_ids, readout_ids)
    print(
        f"[{model}/{cell_line}] {n_r} read-out genes, {n_p} perturbed "
        f"(coverage {coverage['mode']}); {order['detail']}",
        flush=True,
    )

    bases: dict[int, dict[str, np.ndarray]] = {}
    for l in sweep:
        b = basisfit.load_basis(model, cell_line, int(l))
        if int(b["W"].shape[0]) != d or int(b["W"].shape[1]) != k:
            raise RuntimeError(
                f"the basis for layer {l} is {b['W'].shape}, expected ({d}, {k}); a block "
                "projected through the wrong basis is a well-shaped array of the wrong "
                "quantity"
            )
        bases[int(l)] = {
            "hid": np.asarray(b["W"], dtype=np.float32),
            "hidrp": random_orthonormal(d, k, C.NULL_SEEDS["hidrp"] + 1000 * int(l)),
        }

    # (n_pert, n_values, n_readout, k) per (variant, form, layer). float32, and
    # allocated up front so a partial run is visibly incomplete rather than
    # quietly short --- but allocated ON DISK and memory-mapped, because the whole
    # set is 34.0 GB for HepG2 and 54.6 GB for Jurkat against 70 GB of RAM. In
    # memory the larger line would have died AFTER every forward pass had run.
    # There is nothing to stream out early instead: the gene loop is the outer
    # loop, so no layer is finished until the last gene is done, and mapping the
    # accumulator is what keeps this one forward pass per gene rather than one per
    # layer group. config.STORE_MEMMAPPED carries the measurement and the
    # alternatives; paths.store_scratch_dir carries why it lives where it does.
    scratch = paths.ensure(paths.store_scratch_dir(model, cell_line))
    store = {
        (v, form, int(l)): np.lib.format.open_memmap(
            # hyphens, not the '__' run 3's bundle reader splits on: these are
            # not feature files, and a name that looked like one would be a trap
            scratch / f"{v}-{form}-l{int(l)}.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n_p, n_v, n_r, k),
        )
        for v in ("hid", "hidrp")
        for form in C.SLOT_FORMS
        for l in sweep
    }
    scored_moment = {int(l): basisfit.SecondMoment(d, centred=C.BASIS_CENTRED) for l in sweep}
    no_op = slotmod.no_op_mask(x0, cols, values_native)
    c2 = {int(l): 0.0 for l in layers if int(l) == C.EMBEDDING_LAYER}
    self_finite = np.zeros((n_p, n_v), dtype=bool)
    baseline_probe: dict | None = None
    started = time.time()

    for pi, col in enumerate(cols):
        batch = slotmod.query_rows(x0, int(col), values_native, step)
        hidden = q.hidden_batch(batch, layers, perturbed_col=int(col))
        if hidden is None:
            return {
                "available": False,
                "reason": str(getattr(q, "_hidden_error", "the adapter returned no hidden states")),
                "n_perturbed": int(n_p),
            }
        if pi == 0:
            probe = sibling.run5("probe.run")._baseline_pairing_probe
            baseline_probe = probe(q, x0, cols, layers, hidden)
        for l in layers:
            built = slotmod.build_slots(hidden[int(l)], values_native, step)
            if int(l) == C.EMBEDDING_LAYER:
                # C2: an edit at gene i cannot have reached gene j's token before
                # the first block, so the off-target delta must be EXACTLY zero
                for form in C.SLOT_FORMS:
                    off = np.delete(built[form], int(col), axis=1)
                    c2[int(l)] = max(c2[int(l)], float(np.abs(off).max()))
                continue
            for form in C.SLOT_FORMS:
                for vi in range(n_v):
                    if form == "vvp" and no_op[pi, vi]:
                        slotmod.assert_no_op_zero(
                            built[form], vi, int(col), str(gene_ids[pi]),
                            float(C.VALUE_GRID_CP10K[vi]),
                        )
                    x = built[form][vi]
                    if form == "vvp":
                        self_finite[pi, vi] |= bool(np.isfinite(x[int(col)]).all())
                    scored_moment[int(l)].update(np.delete(x, int(col), axis=0))
                    for variant, W in bases[int(l)].items():
                        store[(variant, form, int(l))][pi, vi] = slotmod.project(x, W)
        if log_every and (pi + 1) % log_every == 0:
            rate = (time.time() - started) / (pi + 1)
            print(
                f"    [{model}/{cell_line}] gene {pi + 1}/{n_p} ({rate:.2f}s per gene, "
                f"{rate * (n_p - pi - 1) / 60:.1f} min left)",
                flush=True,
            )

    # bound the dirty page set before the pack-and-save phase, which needs a
    # 1.7-2.7 GB contiguous copy per form and should not be competing with
    # tens of GB of unflushed mapped writes for the same memory
    for arr in store.values():
        arr.flush()

    diagnostics = {
        "n_perturbed": int(n_p),
        "n_readout": int(n_r),
        "n_values": int(n_v),
        "d_model": int(d),
        "k_store": int(k),
        "input_rows_per_gene": int(1 + 3 * n_v),
        "gdt_step": step,
        "gdt_estimator": C.GDT_ESTIMATOR,
        "gdt_one_sided_by_dose": [
            bool(float(v) - step < C.NATIVE_FLOOR) for v in values_native
        ],
        "value_cp10k": list(C.VALUE_GRID_CP10K),
        "value_native": [float(v) for v in values_native],
        "n_no_op_queries": int(no_op.sum()),
        "no_op_queries": [
            {"perturbed_gene_id": str(gene_ids[pi]), "value_cp10k": float(C.VALUE_GRID_CP10K[vi])}
            for pi, vi in zip(*np.nonzero(no_op))
        ],
        "self_position_defined": bool(getattr(q, "hidden_self_position_defined", True)),
        "n_self_positions_finite": int(self_finite.sum()),
        "n_self_positions_total": int(self_finite.size),
        "c2_layer0_max_off_target": c2,
        "store_memmapped": bool(C.STORE_MEMMAPPED),
        "store_scratch_dir": str(scratch),
        "baseline_pairing": baseline_probe,
        "bundle_order": order,
        "coverage": coverage,
        "seconds": round(time.time() - started, 1),
    }
    print(
        f"[{model}/{cell_line}] C2 layer-0 max off-target |dh| = "
        f"{json.dumps(c2)}; {diagnostics['n_no_op_queries']} no-op queries; "
        f"{diagnostics['seconds']}s",
        flush=True,
    )

    root = Path(out_root) if out_root else paths.blocks_dir(model, cell_line)
    paths.ensure(root)
    written = []
    for l in sweep:
        # the captured-energy diagnostic: what fraction of the SCORED
        # perturbations' delta-h energy the basis spans, against the best any
        # k-dimensional basis could. Never used to score.
        M = scored_moment[int(l)].matrix()
        W_all = basisfit.eigendecompose(M, C.K_STORE)["W"]
        coverage_rows = basisfit.captured_energy(M, bases[int(l)]["hid"], W_all, C.K_LADDER)
        for variant in ("hid", "hidrp"):
            key = C.model_stem(model, variant, int(l), k)
            f = root / f"{key}.npz"
            np.savez(
                f,
                vvp=slotmod.pack_block(store[(variant, "vvp", int(l))], k),
                gdt=slotmod.pack_block(store[(variant, "gdt", int(l))], k),
                pert_index=np.arange(n_p, dtype=np.int64),
                perturbed_gene_id=gene_ids,
                readout_gene_id=readout_ids,
                value_cp10k=np.repeat(np.asarray(C.VALUE_GRID_CP10K, dtype=np.float64), k),
                dose_index=np.repeat(np.arange(n_v, dtype=np.int64), k),
                pc_index=np.tile(np.arange(k, dtype=np.int64), n_v),
                no_op=no_op,
            )
            manifest = {
                "key": key,
                "model": model,
                "cell_line": cell_line,
                "variant": variant,
                "layer": int(l),
                "k_store": k,
                "anchor": C.ANCHOR,
                "emits": "hidden_states_only",
                "emits_note": (
                    "run 6 never computes, stores or scores a decoder-head value; runs 2-4 "
                    "settled those and the comparison is post-hoc against them as published"
                ),
                "basis": (
                    "PCA on the model dimension, fitted on 64 out-of-stratum screened "
                    "perturbations disjoint from every scored one"
                    if variant == "hid"
                    else "frozen random orthonormal, seeded; brackets the PCA basis"
                ),
                "slot_axis_order": C.SLOT_AXIS_ORDER,
                "dropped_run2_slot": C.DROPPED_RUN2_SLOT,
                "dropped_run2_slot_reason": C.DROPPED_RUN2_SLOT_REASON,
                "layout": layout,
                "diagnostics": diagnostics,
                "captured_energy": coverage_rows,
                "substitutions": list(getattr(q, "substitutions", [])),
                "build_report": getattr(q, "build_report", None),
            }
            (root / f"{key}.json").write_text(json.dumps(manifest, indent=2, default=str))
            written.append(str(f))
            print(f"[{model}/{cell_line}] wrote {f}", flush=True)
        np.savez(
            root / f"scored_moment_l{int(l)}.npz",
            M=M.astype(np.float64),
            W_all=W_all.astype(np.float32),
            n_vectors=np.asarray(scored_moment[int(l)].n),
        )

    # Every block is on disk, so the mapped accumulators are now redundant
    # duplicates of it and are 34-55 GB. Dropped only on the success path: an
    # exception anywhere above leaves them, along with the driver's failure.txt,
    # so a crashed pass can be inspected rather than only re-run.
    store.clear()
    shutil.rmtree(scratch, ignore_errors=True)
    return {"available": True, "written": written, "diagnostics": diagnostics}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=list(C.MODELS))
    ap.add_argument("--cell-line", required=True, choices=list(C.CELL_LINES))
    ap.add_argument("--stage", required=True, choices=("basis", "main"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--n-perturbed", type=int, default=None)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help=(
            f"the smoke scope ({C.SMOKE_SCOPE.n_perturbed} perturbed genes, "
            f"{C.SMOKE_N_BASIS_GENES} basis genes) and the first and last layers only: "
            "enough to exercise every reassembly branch and run C1-C3 over real output "
            "in minutes, never enough to read a number from"
        ),
    )
    args = ap.parse_args(argv)

    scope = C.SMOKE_SCOPE if args.smoke else C.SCOPE
    if args.n_perturbed is not None:
        scope = C.Scope(n_perturbed=int(args.n_perturbed), strata=scope.strata)
    layers = tuple(args.layers) if args.layers else None
    if layers is None and args.smoke:
        sw = C.sweep_layers(args.model)
        layers = (C.EMBEDDING_LAYER, sw[0], sw[-1])
    out = Path(args.out) if args.out else None
    fail_dir = out or paths.blocks_dir(args.model, args.cell_line)
    try:
        if args.stage == "basis":
            res = fit_basis(
                args.model, args.cell_line, device=args.device, layers=layers,
                n_basis=C.SMOKE_N_BASIS_GENES if args.smoke else C.N_BASIS_GENES,
            )
        else:
            res = extract_line(
                args.model, args.cell_line, device=args.device, layers=layers,
                scope=scope, out_root=out,
            )
    except Exception:
        paths.ensure(Path(fail_dir))
        (Path(fail_dir) / "failure.txt").write_text(traceback.format_exc())
        raise
    print(json.dumps({k: v for k, v in res.items() if k != "diagnostics"}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
