"""Stage 1: score the hidden-state vector arm, one (line, model, layer) at a time.

Stage 0 scored run 5's per-pair ``‖Δh‖`` magnitudes. This is the run's actual
arm: the 10 projected slots of ``Δh`` --- ``vvp`` at each of five doses and
``gdt`` at each --- installed as a synthetic run-2-shaped block of a pseudo-model
and read by run 3's **unmodified** ``resp`` arm. Nothing in run 3 is edited, and
nothing in the scoring rule is re-derived; the only thing that changed is the
input.

**The unit of work is one (cell line, model, layer), not one wave.** Three
things force it, and each is a number rather than a preference:

- a ``k = 64`` block is 3.5 GB (HepG2) / 5.6 GB (Jurkat), ``LineBundle`` moves
  every array it holds onto the scoring device, and this machine has 24 GB of
  VRAM. Four variants of one layer do not fit in one bundle on the larger line,
  so a unit is split further by ``stage.plan_bundles`` under
  ``config.STAGE_BYTES_BUDGET``;
- run 3's ``score_bundle`` persists **nothing** until every requested split has
  finished, so a one-shot wave that dies at hour six leaves no table at all.
  That is exactly how the 2026-08-31 stage-0 pilot lost its per-arm scores. A
  per-unit output directory means a failure costs one unit;
- the installed blocks are ~430 GB across the full grid. Each unit's staged
  blocks are deleted once its table is on disk, unless ``--keep-blocks``.

Which blocks share a bundle **cannot reach a score** --- ``scope_rows("full")``
is every row and no arm reads another arm's block --- so the grouping is an
execution knob and is recorded as one.

**The reference arms are scored once per cell line**, in their own unit, and
never merged into the run table: they are re-runs of rows run 3 already
published, and their job is to say whether this machine reproduces them.

    python -m scfm_hidarm.stage1 --cell-line hepg2 --models scgpt --pilot
    python -m scfm_hidarm.stage1 --cell-line hepg2      # every model, every layer
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from . import config as C
from . import install
from . import nulls
from . import paths
from . import score
from . import stage

#: the four variants of one (model, layer). ``hid`` is the arm; ``hidperm`` is
#: the machinery null and the band the depth profile is read against at EVERY
#: depth; ``hidrand`` controls for width only (README s7 --- projecting an
#: isotropic Gaussian through an orthonormal W gives N(0, s^2 I_k) whatever W
#: is); ``hidrp`` is real dh through a frozen random orthonormal basis and is
#: the arm that says whether the PCA basis is doing the work. ``hidrp`` is
#: load-bearing rather than a nicety here, because the fitted basis under-spans
#: unevenly across arms (README s2).
UNIT_VARIANTS = ("hid", "hidrp", "hidperm", "hidrand")
#: the two that are stored by the extraction pass; the other two are derived at
#: stage time from the installed ``hid`` block (``config.DERIVED_VARIANTS``)
STORED_VARIANTS = ("hid", "hidrp")


def unit_name(cell_line: str, model: str, layer: int, k: int) -> str:
    """The output directory's name. Hyphens only --- never ``__``."""
    return f"{cell_line}-{model.replace('_', '-')}-l{int(layer)}-k{int(k)}"


def stored_block(model: str, cell_line: str, variant: str, layer: int) -> Path:
    """Where the extraction wave left one block. ``K_STORE``, always."""
    key = C.model_stem(model, variant, int(layer), C.K_STORE)
    return paths.blocks_dir(model, cell_line) / f"{key}.npz"


def install_unit(
    cell_line: str,
    model: str,
    layer: int,
    k: int,
    bundle_dir: Path,
    variants: tuple[str, ...] = UNIT_VARIANTS,
) -> list[dict]:
    """Install one (model, layer)'s blocks and derive its nulls, in one bundle.

    The stored blocks are read at ``K_STORE`` and installed at ``k``, which is a
    **column prefix** of what is stored and is only ever taken by slicing the
    stored block. Re-projecting to reach a lower rung differs by 1--3 float32
    ulps (README s1) and is not available here by construction: this module
    never sees a basis.

    The derived nulls are derived from the **installed** ``hid`` block rather
    than from the stored one, so they inherit the same prefix, the same labels
    and the same identity assertions rather than a second decode of them.
    """
    records: list[dict] = []
    for variant in variants:
        if variant not in STORED_VARIANTS:
            continue
        src = stored_block(model, cell_line, variant, layer)
        if not src.exists():
            raise RuntimeError(
                f"no extracted block at {src}; stage 1 installs what the extraction wave "
                "wrote and cannot compute it. A missing file fails, it never skips"
            )
        records.append(
            install.install_source(
                src, bundle_dir, cell_line, model, variant, int(layer), k=k,
                manifest_extra={"stage": 1, "k_store": C.K_STORE,
                                "k_is_a_column_prefix_of_k_store": True},
            )
        )

    real_key = C.model_stem(model, "hid", int(layer), k)
    real = install.block_path(bundle_dir, cell_line, real_key)
    for variant, writer in (("hidperm", nulls.hidperm), ("hidrand", nulls.hidrand)):
        if variant not in variants:
            continue
        nk = C.model_stem(model, variant, int(layer), k)
        records.append(
            writer(real, install.block_path(bundle_dir, cell_line, nk), key=nk)
        )
    return records


def unit_arms(model: str, layer: int, k: int,
              variants: tuple[str, ...] = UNIT_VARIANTS) -> tuple[str, ...]:
    return tuple(C.arm_name(C.model_stem(model, v, int(layer), k)) for v in variants)


def shuffle_bases(model: str, layer: int, k: int) -> tuple[str, ...]:
    """Fresh shuffled-label draws, and only at the pre-registered primary layer.

    Five extra fits per arm per depth would triple the wave for a band that
    ``hidperm`` --- a stronger machinery null, drawn at EVERY depth --- already
    brackets. Which nulls a depth was read against travels with the row, so a
    depth without its own draws may not be quoted as though it had them.

    They are the arm's **own** draws in its **own** cell. Run 4's flag-family
    band and the frozen models' band are both refused for the fine-tuned arm
    (README s6): that family was the flag-off, no-dose estimator and was itself
    flagged as not a clean null.
    """
    if int(layer) != C.primary_layer(model):
        return ()
    return (C.arm_name(C.model_stem(model, "hid", int(layer), k)),)


def reference_arms() -> tuple[str, ...]:
    """``coexp``, ``coexp_node``, ``null_randpair`` and one ``_emb`` per model.

    Deliberately **no** ``<model>_resp``: run 6 emits no decoder-head reading at
    all, and the head comparison is post-hoc against runs 2--4 as published
    (README s5b).
    """
    return C.REFERENCE_ARMS + tuple(f"{m}_emb" for m in C.REFERENCE_EMB_MODELS)


def _score_groups(
    cell_line: str,
    blocks: list[Path],
    arms_of: dict[str, tuple[str, ...]],
    shuffles: tuple[str, ...],
    work: Path,
    out: Path,
    device: str,
    splits: tuple[str, ...],
    translators: tuple[str, ...],
) -> list[dict]:
    """Stage under the VRAM budget and score each group with only its own arms.

    ``arms_of`` maps a block file name to the arms that block supplies, so a
    group is scored for exactly the arms it holds. Asking a bundle for an arm
    whose block is in a different group would not fail --- run 3 skips an
    unresolved model silently --- which is why ``score_bundle`` raises on a
    requested arm that did not appear in its table.
    """
    summaries: list[dict] = []
    groups = stage.plan_bundles(blocks) if blocks else [[]]
    for i, group in enumerate(groups):
        wdir = paths.ensure(Path(work) / f"b{i}")
        odir = paths.ensure(Path(out) / f"b{i}")
        bundle = stage.stage(cell_line, [Path(g) for g in group], wdir, scopes=C.SCOPES)
        arms = tuple(a for g in group for a in arms_of[Path(g).name])
        if not group:
            arms = tuple(arms_of[""])
        sh = tuple(s for s in shuffles if s in arms)
        t0 = time.time()
        summaries.append(
            score.score_bundle(
                bundle, cell_line, arms, sh, odir,
                device=device, splits=splits, translators=translators,
            )
            | {"bundle_index": i, "n_blocks": len(group), "seconds": round(time.time() - t0, 1)}
        )
        print(
            f"[stage1/{cell_line}] bundle {i}: {len(group)} block(s), {len(arms)} arm(s), "
            f"{summaries[-1]['seconds']}s -> {odir}",
            flush=True,
        )
    return summaries


def run_unit(
    cell_line: str,
    model: str,
    layer: int,
    k: int = C.K_PRIMARY,
    variants: tuple[str, ...] = UNIT_VARIANTS,
    device: str = "cuda",
    splits: tuple[str, ...] = ("grouped",),
    translators: tuple[str, ...] = C.TRANSLATORS,
    keep_blocks: bool = False,
) -> dict:
    """Install, stage, score and record one (cell line, model, layer)."""
    name = unit_name(cell_line, model, layer, k)
    work = paths.ensure(paths.stage_root() / "stage1" / name)
    out = paths.ensure(paths.sweep_dir("stage1") / name)
    factory = paths.ensure(work / "blocks")
    stage.stage(cell_line, [], factory, scopes=C.SCOPES)

    started = time.time()
    installed = install_unit(cell_line, model, int(layer), k, factory, variants)
    blocks = [Path(r["block"]) for r in installed]
    arms_of = {
        Path(r["block"]).name: (C.arm_name(r["key"]),) for r in installed
    }
    summaries = _score_groups(
        cell_line, blocks, arms_of, shuffle_bases(model, int(layer), k),
        work, out, device, splits, translators,
    )

    record = {
        "stage": 1,
        "unit": name,
        "cell_line": cell_line,
        "model": model,
        "layer": int(layer),
        "k": int(k),
        "k_store": C.K_STORE,
        "variants": list(variants),
        "arms": sorted({a for v in arms_of.values() for a in v}),
        "shuffle_bases": list(shuffle_bases(model, int(layer), k)),
        "splits": list(splits),
        "translators": list(translators),
        "out": str(out),
        "bundles": summaries,
        "seconds": round(time.time() - started, 1),
        "captured_energy": captured_energy_row(model, cell_line, int(layer), k),
        "emits": "hidden_states_only",
    }
    (Path(out) / "unit.json").write_text(json.dumps(record, indent=2, default=str))
    if not keep_blocks:
        # ~4 x 3.5-5.6 GB per unit, and the stored blocks they were installed
        # from are untouched, so this is recoverable by re-running the unit
        shutil.rmtree(work, ignore_errors=True)
        record["staged_blocks_removed"] = True
    print(f"[stage1/{name}] done in {record['seconds']}s -> {out}", flush=True)
    return record


def captured_energy_row(model: str, cell_line: str, layer: int, k: int) -> dict | None:
    """The basis's captured energy at this ``k``, carried beside every score.

    The fitted basis under-spans the scored perturbations' subspace, and **not
    uniformly across arms** --- ``scgpt_ft``/HepG2 spans 0.981--0.990 while its
    own control ``scgpt_ft_frozen``/Jurkat spans 0.859--0.923, and those two
    arms exist to be subtracted (README s2). The bias is downward only, so it
    can never inflate a positive; but a depth profile drawn without this number
    is a profile of two mixed quantities. It therefore travels in the unit
    record rather than being looked up later, when it would be optional.
    """
    key = C.model_stem(model, "hid", int(layer), C.K_STORE)
    f = paths.blocks_dir(model, cell_line) / f"{key}.json"
    if not f.exists():
        return None
    rows = json.loads(f.read_text()).get("captured_energy") or []
    hit = [r for r in rows if int(r.get("k", -1)) == int(k)]
    return hit[0] if hit else None


def run_references(
    cell_line: str,
    device: str = "cuda",
    splits: tuple[str, ...] = ("grouped",),
    translators: tuple[str, ...] = C.TRANSLATORS,
) -> dict:
    """Run 3's reference arms, in this job, once per cell line.

    They are **not** merged into the run table --- they are re-runs of rows run
    3 already published, and a merge would overwrite a published row with a
    re-run of it. Their job is to say whether this machine reproduces run 3,
    which is what makes run 6's absolute scores readable at all: the best run 3
    arm reaches 0.2719 (HepG2) / 0.2371 (Jurkat) on the noise-ceiling scale, and
    a bare run-6 number against 1.0 means nothing.

    ``coexp`` gets fresh shuffled-label draws here because it is the arm every
    depth profile is read beside.
    """
    name = f"{cell_line}-references"
    work = paths.ensure(paths.stage_root() / "stage1" / name)
    out = paths.ensure(paths.sweep_dir("stage1") / name)
    started = time.time()
    summaries = _score_groups(
        cell_line, [], {"": reference_arms()}, ("coexp",),
        work, out, device, splits, translators,
    )
    record = {
        "stage": 1,
        "unit": name,
        "cell_line": cell_line,
        "arms": list(reference_arms()),
        "shuffle_bases": ["coexp"],
        "splits": list(splits),
        "translators": list(translators),
        "out": str(out),
        "bundles": summaries,
        "seconds": round(time.time() - started, 1),
        "merged_into_run_table": False,
        "reference_note": (
            "re-runs of rows run 3 published; compared against them by a separate check "
            "and never merged, so no scored row is overwritten"
        ),
    }
    (Path(out) / "unit.json").write_text(json.dumps(record, indent=2, default=str))
    shutil.rmtree(work, ignore_errors=True)
    print(f"[stage1/{name}] done in {record['seconds']}s -> {out}", flush=True)
    return record


def plan(cell_line: str, models: tuple[str, ...], layers: tuple[int, ...] | None,
         pilot: bool) -> list[tuple[str, int]]:
    """The (model, layer) units this invocation would run, in order."""
    units: list[tuple[str, int]] = []
    for model in models:
        if layers:
            ls = tuple(int(x) for x in layers)
        elif pilot:
            ls = (C.primary_layer(model),)
        else:
            ls = C.sweep_layers(model)
        units += [(model, int(l)) for l in ls]
    return units


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell-line", required=True, choices=list(C.CELL_LINES))
    ap.add_argument("--models", default="scgpt,scgpt_ft,scgpt_ft_frozen,scprint",
                    help="comma-separated; TranscriptFormer is deferred and its hidperm "
                         "null is unscoreable as written (README s7)")
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--pilot", action="store_true",
                    help="the pre-registered primary layer only")
    ap.add_argument("--k", type=int, default=C.K_PRIMARY)
    ap.add_argument("--variants", default=",".join(UNIT_VARIANTS))
    ap.add_argument("--splits", default="grouped")
    ap.add_argument("--translators", default=",".join(C.TRANSLATORS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--keep-blocks", action="store_true",
                    help="do not delete a unit's staged blocks after it is scored")
    ap.add_argument("--references", action="store_true",
                    help="score run 3's reference arms for this line and stop")
    ap.add_argument("--dry-run", action="store_true", help="print the units and stop")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a unit that already wrote its unit.json. A unit.json is "
                         "written last, after every bundle's table is on disk, so its "
                         "presence means the unit is complete and its absence means the "
                         "unit is not --- which is what makes a wave resumable after the "
                         "kind of failure that stops a process mid-line")
    a = ap.parse_args(argv)

    models = tuple(m.strip() for m in a.models.split(",") if m.strip())
    unknown = [m for m in models if m not in C.MODELS]
    if unknown:
        raise SystemExit(f"unknown model(s) {unknown}; expected from {list(C.MODELS)}")
    variants = tuple(v.strip() for v in a.variants.split(",") if v.strip())
    splits = tuple(s.strip() for s in a.splits.split(",") if s.strip())
    translators = tuple(t.strip() for t in a.translators.split(",") if t.strip())

    if a.references:
        # Idempotent, and deliberately without needing --skip-existing: the
        # reference arms are re-runs of run 3's published rows and a second
        # execution produces the identical numbers, so redoing a complete one
        # buys nothing and costs 43 minutes --- transcriptformer_emb under ridge
        # is 19 of those on its own. unit.json is written last, so its presence
        # means the unit finished.
        done = paths.sweep_dir("stage1") / f"{a.cell_line}-references" / "unit.json"
        if done.exists():
            print(f"[stage1/{a.cell_line}] references already complete at {done}, skipped",
                  flush=True)
            return 0
        run_references(a.cell_line, device=a.device, splits=splits, translators=translators)
        return 0

    units = plan(a.cell_line, models, a.layers, a.pilot)
    print(f"[stage1/{a.cell_line}] {len(units)} unit(s) at k={a.k}: "
          f"{[f'{m}/l{l}' for m, l in units]}", flush=True)
    if a.dry_run:
        return 0
    failed: list[str] = []
    for model, layer in units:
        name = unit_name(a.cell_line, model, layer, a.k)
        if a.skip_existing and (paths.sweep_dir("stage1") / name / "unit.json").exists():
            print(f"[stage1/{a.cell_line}] {name}: already complete, skipped", flush=True)
            continue
        try:
            run_unit(a.cell_line, model, layer, k=a.k, variants=variants, device=a.device,
                     splits=splits, translators=translators, keep_blocks=a.keep_blocks)
        except Exception as exc:  # noqa: BLE001 --- the failure IS the record here
            # One unit's failure must not take the rest of the line with it. Each
            # unit is independent --- its own blocks, its own bundle, its own
            # table --- and a wave that aborts on the first bad one throws away
            # every unit after it for a reason that was local to one. The
            # failure is written where a reader will find it and re-raised as a
            # non-zero exit at the end, never swallowed.
            failed.append(name)
            d = paths.ensure(paths.sweep_dir("stage1") / name)
            (d / "failure.txt").write_text(f"{type(exc).__name__}: {exc}\n")
            print(f"[stage1/{a.cell_line}] {name} FAILED: {type(exc).__name__}: {exc}",
                  flush=True)
    if failed:
        print(f"[stage1/{a.cell_line}] {len(failed)} of {len(units)} unit(s) failed: "
              f"{failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
