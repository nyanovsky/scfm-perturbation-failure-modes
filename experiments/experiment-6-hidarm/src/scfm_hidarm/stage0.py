"""Stage 0: the per-pair hidden-MAGNITUDE arm, off run 5's existing norms.

    PYTHONPATH=src python -m scfm_hidarm.stage0 --cell-line hepg2 --pilot
    PYTHONPATH=src python -m scfm_hidarm.stage0 --cell-line hepg2

No GPU and no fetch. Run 5 already stored ``norm`` at
``(n_pert, n_values, n_readout)`` --- the norm of ``delta_h`` for every (perturbed
gene, dose, read-out gene) triple, per layer, for three models and both cell
lines --- and that is exactly the on-disk ``vvp`` layout run 3's bundle reader
expects. So a per-pair hidden-magnitude arm of dim 5 is installable today,
before a single line of extraction code has to work.

Why this runs first, rather than after the vector arm
-----------------------------------------------------
* It is the **per-pair analogue of run 5's C6 control**. C6 (``norm_only_top1``,
  0.016-0.702 against a directional 1.000) said that at the pooled level the
  perturbed gene's identity is carried by the DIRECTION of ``delta_h`` and not by
  its magnitude. Stage 0 asks the same question at pair resolution, in run 3's
  own metric. A magnitude arm that clears its band would be a finding on its own.
* It **de-risks about nine GPU hours**. If magnitude alone clears the band, the
  vector arm is a refinement rather than the whole bet.
* It exercises install -> stage -> score -> analyse on REAL data end to end,
  which is the only way to find out that a synthetic block is readable by run 3
  before there is anything expensive to lose.

What it is not
--------------
Dim 5 against run 3's ``resp`` dim 11 is **not** width-matched, so this arm is
read against its own dim-5 nulls (``normrand``, ``normperm``) and never against
``null_randpair``. ``norm`` also includes the perturbed gene's own read-out
position --- ``nan`` for TranscriptFormer --- and covers the five ``vvp`` doses
only, with no ``gdt``. All three are recorded in the block manifest rather than
smoothed over.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import config as C
from . import install, nulls, paths, score, stage


#: the arms scored beside every stage-0 block: run 3's own statistical
#: references and the three models' static gene tables, re-run in this job so
#: that a run-6 number is compared against a reference scored in the same job on
#: the same rows. These rows stay OUT of the merged run table; a separate check
#: compares them to run 3's published rows, so no scored row is overwritten.
def reference_arms() -> tuple[str, ...]:
    return tuple(C.REFERENCE_ARMS) + tuple(f"{m}_emb" for m in C.REFERENCE_EMB_MODELS)


def layers_for(model: str, layers: tuple[int, ...] | None, pilot: bool) -> tuple[int, ...]:
    """Which layers stage 0 installs for one model.

    ``--pilot`` is the last block only: it is the layer the decoder head reads,
    it is the pre-registered primary cell's layer, and it returns a real scored
    number in one wave instead of fourteen.
    """
    if layers:
        return tuple(int(x) for x in layers)
    if pilot:
        return (C.primary_layer(model),)
    return C.sweep_layers(model)


def install_line(
    cell_line: str,
    bundle_dir: Path,
    models: tuple[str, ...],
    layers: tuple[int, ...] | None = None,
    pilot: bool = False,
) -> dict:
    """Write run 5's ``norm`` arrays into a staged bundle as run-2-shaped blocks.

    The join is on ``perturbed_gene_id`` and ``readout_gene_id``, never on row
    index. Run 5's blocks happen to be in run 3's bundle order today; ``install``
    asserts it rather than relying on it, because a block that installs cleanly
    in the wrong order scores cleanly and puts every perturbation's features on
    a different perturbation's target.
    """
    installed: list[dict] = []
    for model in models:
        for layer in layers_for(model, layers, pilot):
            src = paths.run5_norms(model, cell_line, int(layer), C.ANCHOR)
            if not src.exists():
                raise RuntimeError(
                    f"run 5 wrote no norms for {model}/{cell_line}/layer={layer} at {src}; "
                    "stage 0 reads run 5's stored blocks and cannot compute them"
                )
            z = np.load(src, allow_pickle=False)
            key = C.model_stem(model, "norm", int(layer))
            grid = np.asarray(z["value_grid_cp10k"], dtype=np.float64)
            if not np.allclose(grid, np.asarray(C.VALUE_GRID_CP10K, dtype=np.float64)):
                raise RuntimeError(
                    f"{src} was written at doses {grid.tolist()}, not run 6's "
                    f"{list(C.VALUE_GRID_CP10K)}; the dose axis would not mean the same thing"
                )
            norm = np.asarray(z["norm"], dtype=np.float32)
            rec = install.write_block(
                bundle_dir,
                cell_line,
                key,
                vvp=norm,
                gdt=None,
                pert_index=np.arange(norm.shape[0], dtype=np.int64),
                perturbed_gene_id=z["perturbed_gene_id"].astype(str),
                readout_gene_id=z["readout_gene_id"].astype(str),
                k=1,
                manifest_extra={
                    "stage": 0,
                    "source": str(src),
                    "source_run": "experiment-5-hidprb",
                    "quantity": (
                        "||delta_h[p, v, g]||_2 --- the per-pair magnitude of the "
                        "hidden-state response, with no direction and no projection"
                    ),
                    "emits": "hidden_states_only",
                    "forms": ["vvp"],
                    "gdt_absent_reason": (
                        "run 5 stored no derivative: its 6-row query batch has one row per "
                        "dose and no (v-h, v, v+h) bracket, so a central difference cannot "
                        "be formed from it. The vector arm's blocks carry gdt"
                    ),
                    "self_position_included": True,
                    "self_position_note": (
                        "run 5's norm includes the perturbed gene's own read-out position; "
                        "it is nan for TranscriptFormer, which never reads a gene's count at "
                        "its own position. Every scoring cell excludes that column by "
                        "construction, so the value is carried rather than invented"
                    ),
                    "width_note": (
                        "dim 5 against run 3's resp dim 11: NOT width-matched, so this arm "
                        "is read against its own dim-5 nulls and never against null_randpair"
                    ),
                    "c6_relation": (
                        "the per-pair analogue of run 5's C6 pooled magnitude-only control "
                        "(norm_only_top1 0.016-0.702 against a directional 1.000)"
                    ),
                },
            )
            installed.append(rec)
            real = install.block_path(bundle_dir, cell_line, key)
            for variant, writer in (("normperm", nulls.hidperm), ("normrand", nulls.hidrand)):
                nk = C.model_stem(model, variant, int(layer))
                installed.append(
                    writer(real, install.block_path(bundle_dir, cell_line, nk), key=nk)
                )
    return {"cell_line": cell_line, "n_blocks": len(installed), "blocks": installed}


def arm_lists(cell_line: str, models: tuple[str, ...], layers, pilot: bool) -> dict:
    """The arms scored, and which of them get fresh shuffled-label draws.

    Shuffled draws are taken for the REAL arm at the pre-registered primary
    layer and for ``coexp``, in every cell those arms are scored in. They are
    not taken at every depth: five extra fits per arm per layer would triple the
    wave for a band that ``normperm`` --- a stronger machinery null, and one that
    IS drawn at every depth --- already brackets. Which nulls a depth was read
    against travels with the row, so a depth without its own shuffled draws
    cannot be quoted as though it had them.
    """
    real, null, shuffle = [], [], ["coexp"]
    for model in models:
        for layer in layers_for(model, layers, pilot):
            real.append(C.arm_name(C.model_stem(model, "norm", int(layer))))
            null += [
                C.arm_name(C.model_stem(model, v, int(layer)))
                for v in ("normperm", "normrand")
            ]
            if int(layer) == C.primary_layer(model):
                shuffle.append(C.arm_name(C.model_stem(model, "norm", int(layer))))
    return {
        "real": tuple(real),
        "null": tuple(null),
        "reference": reference_arms(),
        "all": tuple(real) + tuple(null) + reference_arms(),
        "shuffle_bases": tuple(shuffle),
    }


def run(
    cell_line: str,
    models: tuple[str, ...] = C.FROZEN_MODELS,
    layers: tuple[int, ...] | None = None,
    pilot: bool = False,
    work: Path | None = None,
    out: Path | None = None,
    device: str = "cuda",
    splits: tuple[str, ...] = C.SPLITS,
    translators: tuple[str, ...] = C.TRANSLATORS,
) -> dict:
    """Install, stage, score and record one cell line's stage-0 wave."""
    name = "stage0_pilot" if pilot else "stage0"
    work = Path(work) if work else paths.stage_root() / name
    out = Path(out) if out else paths.sweep_dir(name)
    bundle = stage.stage(cell_line, [], work, scopes=C.SCOPES)
    inst = install_line(cell_line, bundle, models, layers, pilot)
    arms = arm_lists(cell_line, models, layers, pilot)
    print(
        f"[stage0/{cell_line}] {inst['n_blocks']} blocks installed; "
        f"{len(arms['all'])} arms, {len(arms['shuffle_bases'])} shuffled bases",
        flush=True,
    )
    summary = score.score_bundle(
        bundle, cell_line, arms["all"], arms["shuffle_bases"], out,
        device=device, splits=splits, translators=translators,
    )
    record = {
        "stage": 0,
        "wave": name,
        "cell_line": cell_line,
        "models": list(models),
        "bundle": str(bundle),
        "out": str(out),
        "install": {k: v for k, v in inst.items() if k != "blocks"},
        "arms": {k: list(v) for k, v in arms.items()},
        "scoring": {k: v for k, v in summary.items() if k != "bundle_inventory"},
        "reference_rows_excluded_from_merge": True,
        "reference_note": (
            "the reference arms are re-run in this job and compared against run 3's "
            "published rows by a separate check; they are never merged into the run table, "
            "so no scored row is overwritten"
        ),
    }
    paths.ensure(Path(out))
    (Path(out) / f"stage0_{cell_line}.json").write_text(json.dumps(record, indent=2, default=str))
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell-line", required=True, choices=list(C.CELL_LINES))
    ap.add_argument("--models", default=",".join(C.FROZEN_MODELS))
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument(
        "--pilot",
        action="store_true",
        help="the last block only: the layer the head reads and the primary cell's layer",
    )
    ap.add_argument("--work", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--splits", default=",".join(C.SPLITS))
    ap.add_argument("--translators", default=",".join(C.TRANSLATORS))
    args = ap.parse_args(argv)
    rec = run(
        args.cell_line,
        tuple(m.strip() for m in args.models.split(",") if m.strip()),
        tuple(args.layers) if args.layers else None,
        args.pilot,
        Path(args.work) if args.work else None,
        Path(args.out) if args.out else None,
        args.device,
        tuple(s.strip() for s in args.splits.split(",") if s.strip()),
        tuple(t.strip() for t in args.translators.split(",") if t.strip()),
    )
    print(json.dumps({k: v for k, v in rec.items() if k != "arms"}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
