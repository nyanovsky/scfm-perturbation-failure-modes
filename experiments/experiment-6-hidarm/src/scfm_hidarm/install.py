"""Write run 6's hidden-state features into a bundle as a run-2-shaped block.

Run 6's whole integration with run 3 is this file. ``LineBundle`` reads a
response block from ``<bundle>/<line>/features/<model>__<anchor>.npz`` and
never validates that the value axis has length five --- ``_resp_dim`` reads
``e[form].shape[1]`` and ``batch`` gathers ``e[form][m, :, cols]`` whole. So a
``(n_pert, 10k, n_readout)`` array installed under a pseudo-model key is read by
run 3's **unmodified** ``resp`` arm, and run 6 scores a new input through a rule
that did not change. That is the point: forking ``arms.py`` to widen
``randpair`` (hardcoded to 11) would put run 6's numbers outside the ~2,300 runs
already scored, and would reach run 4, which imports the same file.

Four things this module refuses to let happen silently, each because the
failure mode is a wrong number rather than a traceback:

**A key containing ``__``.** Run 3 splits the feature file name on the first
``__`` to separate model from anchor, so such a key makes the block unreadable
and the arm *vanish from the scored set with no error*. This cost run 4 a whole
round of re-scoring (cp-13). Keys are built and parsed only through
``config.model_stem`` / ``config.run_of_stem``, and a key that does not
round-trip through the pair is rejected here rather than at score time.

**A ``pert_index`` that is not every scope row.** ``LineBundle`` fills
``row_map`` with ``-1`` for uncovered rows and ``runner.run_cell`` then
*silently skips* the model. A block covering 523 of 524 perturbations does not
fail; it disappears.

**Labels that do not match the bundle.** Row order is a contract, and one that
is asserted rather than assumed: the block's ``perturbed_gene_id`` and
``readout_gene_id`` are compared element-wise against the bundle's own
``targets.npz`` before anything is written. A block whose rows are the same
length but a different order would put gene A's response on gene B and score
perfectly well.

**A ``zero_out`` key.** Run 2's ``resp`` block carries one, and its presence
adds 1 to ``_resp_dim``. Run 6 drops it because with no per-query normalisation
applied the hidden ``zero_out`` slot is bit-identical to the ``vvp`` slot at
dose 0 (``config.DROPPED_RUN2_SLOT_REASON``, asserted by ``tests/test_slots.py``),
so keeping both would put one column in the feature vector twice.

Rejected: writing the manifest beside the block. ``LineBundle`` globs every
``*.npz`` under ``features/`` and splits its stem on ``__``, so a file with no
``__`` in its name raises ``ValueError`` at bundle construction and kills every
arm for that line. ``features/`` holds nothing but blocks; manifests go in
``block_manifests/``.

    python -m scfm_hidarm.install --source <projected.npz> --bundle <staged> \
        --line hepg2 --model scgpt --variant norm --layer 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import config as C
from . import paths

#: the block's file name is ``<key>__<anchor>.npz`` and its manifest lives in a
#: sibling directory, never in ``features/`` --- see the module docstring.
MANIFEST_DIRNAME = "block_manifests"


def block_filename(key: str, anchor: str = C.ANCHOR) -> str:
    """The name run 3's reader parses back into ``(model, anchor)``."""
    return f"{key}__{anchor}.npz"


def block_path(bundle_dir: Path, cell_line: str, key: str, anchor: str = C.ANCHOR) -> Path:
    return Path(bundle_dir) / cell_line / "features" / block_filename(key, anchor)


def manifest_path(bundle_dir: Path, cell_line: str, key: str, anchor: str = C.ANCHOR) -> Path:
    return Path(bundle_dir) / cell_line / MANIFEST_DIRNAME / f"{key}__{anchor}.json"


# --------------------------------------------------------------------------- #
# the key
# --------------------------------------------------------------------------- #
def check_key(key: str) -> dict:
    """Reject a key run 3's reader would mis-parse, and return what it decodes to.

    Both directions are checked, not just the forbidden substring: a key that
    parses to a different key than it was built from means ``run_of_stem`` and
    ``model_stem`` disagree, and every downstream join keyed on the decoded
    ``(model, variant, layer, k)`` would then describe the wrong block.
    """
    if C.FORBIDDEN_IN_KEY in key:
        raise RuntimeError(
            f"the model key {key!r} contains {C.FORBIDDEN_IN_KEY!r}; run 3's bundle reader "
            "splits the feature file name on it, so the block would be unreadable and the "
            "response arm would vanish from the scored set with no error at all"
        )
    decoded = C.run_of_stem(key)
    rebuilt = C.model_stem(decoded["model"], decoded["variant"], decoded["layer"], decoded["k"])
    if rebuilt != key:
        raise RuntimeError(
            f"the model key {key!r} does not round-trip through config.run_of_stem / "
            f"config.model_stem (it decodes to {decoded!r} and re-encodes to {rebuilt!r}); "
            "every manifest, null and merge keyed on the decoded fields would then name a "
            "different block than the one on disk"
        )
    return decoded


# --------------------------------------------------------------------------- #
# the slot layout
# --------------------------------------------------------------------------- #
def lay_out_slots(
    slots: dict[tuple[str, int], np.ndarray],
    k: int,
    n_dose: int = len(C.VALUE_GRID_CP10K),
) -> dict[str, np.ndarray]:
    """Flatten ``{(form, dose_index): (n_pert, k, n_readout)}`` onto run 3's axis 1.

    The order is ``config.SLOT_AXIS_ORDER``: C-order over ``(dose, pc)`` within
    each of ``vvp`` and ``gdt``, so column ``dose_i * k + pc``. Run 3 gathers
    ``e[form][m, :, cols]`` whole and has no idea what the columns mean, which
    makes this ordering a contract only run 6 can assert --- ``tests/test_slots.py``
    does, and every reader of a run-6 block decodes it from the ``dose_index`` /
    ``pc_index`` arrays written beside the data rather than re-deriving it.

    A form is written only if *every* dose is supplied for it. A half-filled
    form would leave whichever columns were missing at whatever ``np.empty``
    happened to hold, and would score.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    forms = sorted({f for (f, _) in slots})
    unknown = [f for f in forms if f not in C.SLOT_FORMS]
    if unknown:
        raise ValueError(f"unknown slot form(s) {unknown}; expected {C.SLOT_FORMS}")
    out: dict[str, np.ndarray] = {}
    for form in forms:
        have = sorted(d for (f, d) in slots if f == form)
        if have != list(range(n_dose)):
            raise RuntimeError(
                f"slot form {form!r} supplies dose indices {have} but the value grid has "
                f"{n_dose} doses; a form written with a dose missing would leave those "
                "columns uninitialised and they would still be scored"
            )
        first = np.asarray(slots[(form, 0)])
        if first.ndim != 3 or first.shape[1] != k:
            raise ValueError(
                f"slot ({form!r}, 0) has shape {first.shape}; expected (n_pert, {k}, n_readout)"
            )
        arr = np.empty((first.shape[0], n_dose * k, first.shape[2]), dtype=np.float32)
        for dose_i in range(n_dose):
            s = np.asarray(slots[(form, dose_i)])
            if s.shape != first.shape:
                raise RuntimeError(
                    f"slot ({form!r}, {dose_i}) has shape {s.shape} but ({form!r}, 0) has "
                    f"{first.shape}; the doses of one form must share a shape or the flattened "
                    "axis does not decode as (dose, pc)"
                )
            arr[:, dose_i * k : (dose_i + 1) * k, :] = s.astype(np.float32, copy=False)
        out[form] = arr
    return out


def column_labels(k: int, values_cp10k: tuple[float, ...] | np.ndarray) -> dict[str, np.ndarray]:
    """The ``(dose, pc)`` decode of axis 1, written to disk beside the data.

    Run 3's ``values`` key is the reader's contract and carries the dose
    repeated ``k`` times; it is not self-describing, and run 6 may not rename
    it. ``value_cp10k`` / ``dose_index`` / ``pc_index`` are the labels a later
    reader joins on instead of re-deriving ``index = dose_i * k + pc`` from this
    module's source.
    """
    values = np.asarray(values_cp10k, dtype=np.float64)
    return {
        "values": np.repeat(values, k),
        "value_cp10k": np.repeat(values, k),
        "dose_index": np.repeat(np.arange(values.size, dtype=np.int64), k),
        "pc_index": np.tile(np.arange(k, dtype=np.int64), values.size),
    }


# --------------------------------------------------------------------------- #
# the writer
# --------------------------------------------------------------------------- #
def write_block(
    bundle_dir: Path,
    cell_line: str,
    key: str,
    vvp: np.ndarray,
    gdt: np.ndarray | None,
    pert_index: np.ndarray,
    perturbed_gene_id: np.ndarray,
    readout_gene_id: np.ndarray,
    k: int,
    values_cp10k: tuple[float, ...] | np.ndarray = C.VALUE_GRID_CP10K,
    manifest_extra: dict | None = None,
    anchor: str = C.ANCHOR,
) -> dict:
    """One run-6 block, in run 3's block format, with its identity asserted.

    ``gdt=None`` is the stage-0 norm arm: run 5's ``norms.npz`` holds a
    magnitude per (pair, dose) and no derivative, so that block is ``vvp``-only
    and dim 5. Its width is therefore *not* matched to run 3's ``resp`` dim 11,
    which is why its verdict is read against the dim-5 null written by
    ``nulls.py`` and never against ``null_randpair``.

    Every check below raises rather than warns, because each of them fails by
    producing a plausible score rather than an error.
    """
    decoded = check_key(key)
    base = Path(bundle_dir) / cell_line
    values = np.asarray(values_cp10k, dtype=np.float64)
    n_dose = int(values.size)

    forms: dict[str, np.ndarray] = {"vvp": np.asarray(vvp, dtype=np.float32)}
    if gdt is not None:
        forms["gdt"] = np.asarray(gdt, dtype=np.float32)
    if len(forms) == 2 and forms["vvp"].shape != forms["gdt"].shape:
        raise RuntimeError(
            f"{key}: vvp has shape {forms['vvp'].shape} and gdt {forms['gdt'].shape}; run 3 "
            "concatenates the two along axis 1 for the same (row, column) gather, so a shape "
            "mismatch would either raise deep inside the translator or silently mis-align the "
            "two halves of the feature vector"
        )
    n_pert, n_cols, n_readout = forms["vvp"].shape
    if n_cols != n_dose * k:
        raise RuntimeError(
            f"{key}: axis 1 is {n_cols} wide but the value grid has {n_dose} doses at k={k}, "
            f"so it should be {n_dose * k}; run 3 reads the width as the feature dimension "
            "without checking it, so a wrong width scores rather than fails"
        )

    pert_index = np.asarray(pert_index, dtype=np.int64)
    perturbed_gene_id = np.asarray(perturbed_gene_id).astype(str)
    readout_gene_id = np.asarray(readout_gene_id).astype(str)

    # ---- block identity: the named check, asserted before anything is written #
    t = np.load(base / "targets.npz", allow_pickle=False)
    scope_pert = t["pert_gene_id"].astype(str)
    scope_readout = t["readout_gene_id"].astype(str)
    if not np.array_equal(pert_index, np.arange(scope_pert.size, dtype=np.int64)):
        raise RuntimeError(
            f"{key}/{cell_line}: pert_index covers {pert_index.size} of {scope_pert.size} scope "
            "rows and must be exactly arange(n_pert); LineBundle fills row_map with -1 for an "
            "uncovered row and runner.run_cell then SILENTLY SKIPS the model for that cell"
        )
    if not np.array_equal(perturbed_gene_id, scope_pert):
        raise RuntimeError(
            f"{key}/{cell_line}: perturbed_gene_id does not match targets.npz:pert_gene_id row "
            "for row; the block would put one perturbation's response on another's row and "
            "would score"
        )
    if not np.array_equal(readout_gene_id, scope_readout):
        raise RuntimeError(
            f"{key}/{cell_line}: readout_gene_id does not match targets.npz:readout_gene_id row "
            "for row; the block would put one read-out gene's response in another's column and "
            "would score"
        )
    if forms["vvp"].shape[2] != scope_readout.size:
        raise RuntimeError(
            f"{key}/{cell_line}: the block has {forms['vvp'].shape[2]} read-out columns and the "
            f"bundle has {scope_readout.size}"
        )

    payload: dict[str, np.ndarray] = {
        "pert_index": pert_index,
        "value_keep": np.ones(n_cols, dtype=bool),
        "perturbed_gene_id": perturbed_gene_id,
        "readout_gene_id": readout_gene_id,
    }
    payload.update(column_labels(k, values))
    payload.update(forms)
    if C.DROPPED_RUN2_SLOT in payload:
        raise RuntimeError(
            f"{key}: a {C.DROPPED_RUN2_SLOT!r} array reached the payload; its presence adds 1 to "
            f"run 3's _resp_dim and run 6 drops it because {C.DROPPED_RUN2_SLOT_REASON}"
        )

    (base / "features").mkdir(parents=True, exist_ok=True)
    out = block_path(bundle_dir, cell_line, key, anchor)
    np.savez(out, **payload)

    manifest = {
        "key": key,
        "arm": C.arm_name(key),
        "cell_line": cell_line,
        "anchor": anchor,
        "block": str(out),
        "forms": sorted(forms),
        "n_perturbed": int(n_pert),
        "n_readout": int(n_readout),
        "n_columns": int(n_cols),
        "k": int(k),
        "n_doses": n_dose,
        "value_grid_cp10k": [float(v) for v in values],
        "resp_dim": int(len(forms) * n_cols),
        "slot_axis_order": C.SLOT_AXIS_ORDER,
        "dropped_run2_slot": C.DROPPED_RUN2_SLOT,
        "dropped_run2_slot_reason": C.DROPPED_RUN2_SLOT_REASON,
        "npz_keys": sorted(payload),
        "config": {
            "VALUE_GRID_CP10K": list(C.VALUE_GRID_CP10K),
            "ANCHOR": C.ANCHOR,
            "SLOT_FORMS": list(C.SLOT_FORMS),
            "K_STORE": C.K_STORE,
            "K_PRIMARY": C.K_PRIMARY,
            "GDT_ESTIMATOR": C.GDT_ESTIMATOR,
        },
        **decoded,
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    mpath = manifest_path(bundle_dir, cell_line, key, anchor)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=2))
    print(
        f"[install] {cell_line}/{key}: {sorted(forms)} {forms['vvp'].shape} "
        f"resp_dim={manifest['resp_dim']} -> {out}",
        flush=True,
    )
    return manifest


def install_from_slots(
    bundle_dir: Path,
    cell_line: str,
    key: str,
    slots: dict[tuple[str, int], np.ndarray],
    pert_index: np.ndarray,
    perturbed_gene_id: np.ndarray,
    readout_gene_id: np.ndarray,
    k: int,
    values_cp10k: tuple[float, ...] | np.ndarray = C.VALUE_GRID_CP10K,
    manifest_extra: dict | None = None,
    anchor: str = C.ANCHOR,
) -> dict:
    """Lay ten ``(n_pert, k, n_readout)`` slots down in the declared order and write.

    This is the extractor's entry point: it never sees a flattened array, so it
    cannot get the ``(dose, pc)`` order wrong by writing the columns in the
    order the doses happened to be computed in.
    """
    n_dose = int(np.asarray(values_cp10k, dtype=np.float64).size)
    arrays = lay_out_slots(slots, k, n_dose)
    return write_block(
        bundle_dir,
        cell_line,
        key,
        arrays["vvp"],
        arrays.get("gdt"),
        pert_index,
        perturbed_gene_id,
        readout_gene_id,
        k,
        values_cp10k=values_cp10k,
        manifest_extra=manifest_extra,
        anchor=anchor,
    )


# --------------------------------------------------------------------------- #
# reading a stored projected block
# --------------------------------------------------------------------------- #
def unpack_block(
    arr: np.ndarray,
    dose_index: np.ndarray,
    pc_index: np.ndarray,
) -> np.ndarray:
    """Undo ``slots.pack_block``: ``(n_pert, n_dose*k, n_readout)`` -> 4-D.

    The decode is read off the ``dose_index`` / ``pc_index`` labels the
    extractor writes beside the data, and then **asserted** against the
    declared C-order ``index = dose_i * k + pc``, rather than re-derived from
    the array's shape. Shape cannot do it: a packed block at ``k = 1`` and a
    stage-0 ``norm`` block have the identical three-dimensional shape, and
    reading one as the other would put a PC index on the dose axis and still
    score. The labels separate them exactly, which is the reason arrays in this
    run carry their own labels at all.
    """
    a = np.asarray(arr)
    dose = np.asarray(dose_index, dtype=np.int64).ravel()
    pc = np.asarray(pc_index, dtype=np.int64).ravel()
    if a.ndim != 3:
        raise RuntimeError(
            f"a packed block must be (n_pert, n_dose*k, n_readout); got shape {a.shape}"
        )
    if dose.size != a.shape[1] or pc.size != a.shape[1]:
        raise RuntimeError(
            f"axis 1 is {a.shape[1]} wide but dose_index carries {dose.size} labels and "
            f"pc_index {pc.size}; the labels are the only decode of that axis and a "
            "mismatch means the block does not describe itself"
        )
    n_dose = int(dose.max()) + 1
    k = int(pc.max()) + 1
    if n_dose * k != a.shape[1]:
        raise RuntimeError(
            f"the labels span {n_dose} doses x {k} components = {n_dose * k} columns but "
            f"axis 1 is {a.shape[1]} wide; a partially-labelled axis would leave whichever "
            "columns were unlabelled at whatever they happened to hold, and they would score"
        )
    want_dose = np.repeat(np.arange(n_dose, dtype=np.int64), k)
    want_pc = np.tile(np.arange(k, dtype=np.int64), n_dose)
    if not (np.array_equal(dose, want_dose) and np.array_equal(pc, want_pc)):
        raise RuntimeError(
            f"axis 1 is not in the declared {C.SLOT_AXIS_ORDER} order (index = dose*k + pc); "
            "run 3 gathers the axis whole and has no idea what the columns mean, so a "
            "differently-ordered block would score with its dose axis permuted"
        )
    return a.reshape(a.shape[0], n_dose, k, a.shape[2])


def source_slots(
    z, k: int | None = None
) -> tuple[dict[tuple[str, int], np.ndarray], int, str]:
    """Turn a stored array into slots, and say what ``k`` and which layout.

    Three on-disk layouts are accepted, and only three:

    ``packed`` --- ``(n_pert, n_dose*k, n_readout)`` carrying ``dose_index`` and
    ``pc_index``. This is what ``extract/run.py`` actually writes (via
    ``slots.pack_block``) and, until 2026-09-01, the one layout the CLI could
    not read --- so there was no command-line path from a run-6 extraction to an
    installed block at all, and every score had to go through ``write_block``
    in Python. It failed loudly rather than silently, so nothing on disk was
    ever wrong. It is unpacked here through its own labels, never through its
    shape (``unpack_block``).

    ``unpacked`` --- ``(n_pert, n_dose, k, n_readout)``, the same content before
    packing; kept because tests and ``stage0`` build it directly.

    ``(n_pert, n_dose, n_readout)`` --- run 5's ``norms.npz:norm``, which is the
    stage-0 arm. It is a ``vvp`` block at ``k = 1`` under a different name: a
    scalar magnitude per (pair, dose) is exactly what run 3's reader expects on
    that axis, which is why stage 0 needs no GPU and no extraction code.
    """
    if "vvp" in z.files:
        names = {"vvp": "vvp"} | ({"gdt": "gdt"} if "gdt" in z.files else {})
    elif "norm" in z.files:
        names = {"vvp": "norm"}
    else:
        raise RuntimeError(
            f"the source npz carries {sorted(z.files)} and none of 'vvp' or 'norm'; run 6 "
            "installs a response block and has no other source of one"
        )
    packed = "dose_index" in z.files and "pc_index" in z.files
    layout = "packed" if packed else "unpacked_or_norm"
    slots: dict[tuple[str, int], np.ndarray] = {}
    found_k: int | None = None
    for form, name in names.items():
        arr = np.asarray(z[name])
        if packed:
            # the labels, not the shape --- a packed k=1 block and a stage-0
            # norm block are the same shape and only the labels tell them apart
            arr = unpack_block(arr, z["dose_index"], z["pc_index"])
            layout = "packed"
        elif arr.ndim == 3:
            arr = arr[:, :, None, :]
            layout = "norm_3d"
        if arr.ndim != 4:
            raise RuntimeError(
                f"{name}: shape {arr.shape}; expected (n_pert, n_dose*k, n_readout) with "
                "dose_index/pc_index labels, (n_pert, n_dose, k, n_readout), or "
                "(n_pert, n_dose, n_readout)"
            )
        this_k = int(arr.shape[2]) if k is None else int(k)
        if k is not None and k > arr.shape[2]:
            raise RuntimeError(
                f"{name}: asked for k={k} but only {arr.shape[2]} components are stored; the "
                "ladder is a column PREFIX of the stored block and cannot be extended upward"
            )
        found_k = this_k if found_k is None else found_k
        for dose_i in range(arr.shape[1]):
            slots[(form, dose_i)] = arr[:, dose_i, :this_k, :]
    assert found_k is not None
    return slots, found_k, layout


def install_source(
    source: Path,
    bundle_dir: Path,
    cell_line: str,
    model: str,
    variant: str,
    layer: int,
    k: int | None = None,
    anchor: str = C.ANCHOR,
    manifest_extra: dict | None = None,
) -> dict:
    """Install one stored block into a bundle. The CLI and ``stage1`` share this.

    Factored out of ``main`` so the wave driver and the command line take the
    identical path: a driver that reimplemented the read would be free to decode
    the packed axis differently from the CLI, and both would score.
    """
    z = np.load(Path(source), allow_pickle=False)
    slots, resolved_k, layout = source_slots(z, k)
    # The key names the k actually installed, not the flag that was typed: a
    # packed block installed without an explicit k would otherwise get a key
    # with no k in it, and every later join keyed on the decoded (model,
    # variant, layer, k) would name a block that is not this one. The stage-0
    # norm arm genuinely has no k and keeps a k-less key.
    key_k = k if k is not None else (resolved_k if layout == "packed" else None)
    key = C.model_stem(model, variant, int(layer), key_k)
    t = np.load(Path(bundle_dir) / cell_line / "targets.npz", allow_pickle=False)
    n_pert = int(t["pert_gene_id"].size)
    pert_gene_id = (
        z["perturbed_gene_id"].astype(str) if "perturbed_gene_id" in z.files
        else t["pert_gene_id"].astype(str)
    )
    readout_gene_id = (
        z["readout_gene_id"].astype(str) if "readout_gene_id" in z.files
        else t["readout_gene_id"].astype(str)
    )
    values = (
        z["value_grid_cp10k"] if "value_grid_cp10k" in z.files
        else np.asarray(C.VALUE_GRID_CP10K)
    )
    extra = {"source": str(source), "source_layout": layout}
    if manifest_extra:
        extra.update(manifest_extra)
    return install_from_slots(
        bundle_dir, cell_line, key, slots,
        np.arange(n_pert, dtype=np.int64), pert_gene_id, readout_gene_id, resolved_k,
        values_cp10k=values, manifest_extra=extra, anchor=anchor,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, type=Path,
                   help="the extractor's packed block (n_pert, n_dose*k, n_readout) with its "
                        "dose_index/pc_index labels, the same content unpacked as "
                        "(n_pert, n_dose, k, n_readout), or run 5's norms.npz for the "
                        "stage-0 arm")
    p.add_argument("--bundle", type=Path, default=None,
                   help="staged bundle root (default: run 6's staging root)")
    p.add_argument("--line", required=True, choices=list(C.CELL_LINES))
    p.add_argument("--model", required=True, choices=list(C.MODELS))
    p.add_argument("--variant", required=True,
                   choices=list(C.VARIANTS + C.STAGE0_VARIANTS))
    p.add_argument("--layer", required=True, type=int)
    p.add_argument("--k", type=int, default=None,
                   help="components to install; a prefix of the stored block. Omit for the "
                        "stage-0 norm arm, which has no k.")
    p.add_argument("--anchor", default=C.ANCHOR)
    a = p.parse_args(argv)

    bundle = a.bundle if a.bundle is not None else paths.stage_root()
    install_source(a.source, bundle, a.line, a.model, a.variant, a.layer,
                   k=a.k, anchor=a.anchor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
