"""Re-read stage 1's fine-tuned arms --- and frozen scGPT --- on folds 0 and 1.

Run 3's sweep scores every arm on all five folds, and stage 1 runs it unchanged.
For the three frozen models that is the whole story. For run 4's two fine-tuned
checkpoints it is **not**: they trained on folds 2, 3 and 4, so three fifths of
a five-fold row is a training number. ``README.md`` s6 is explicit that this is
the one place run 6 cannot reuse run 3's rule end to end.

So the five-fold rows for ``scgpt_ft`` and ``scgpt_ft_frozen`` **must not be
quoted**, and this module produces the rows that may be. It is a post-pass, not
a re-fit: run 6 keeps the held-out prediction for every arm and every cell
(``score.force_prediction_saving``), so the restriction is a re-*reading* of
predictions that already exist, through run 4's own ``restrict_scores`` and
``restricted_ceilings``, imported rather than reinvented.

**Frozen scGPT is restricted too, at every layer.** A two-fold fine-tuned score
beside a five-fold frozen score is not a comparison, and the whole point of the
fine-tuned arm is the difference between them. That costs nothing here because
the predictions are already on disk.

Two things this cannot fix, and the wording has to carry both:

- on folds 0 and 1 **co-expression itself is not separated from the band**
  (``docs/INDEX.md`` s9), so a fine-tuned result read here is a **bound at the
  achieved resolution**, exactly as run 4's headline is;
- fine-tuned against frozen is "the fine-tuned model in its native regime"
  against "the frozen model in the only regime it has" --- the flag is on in one
  and absent from the other by construction --- so it is never worded as
  isolating what fine-tuning put in the weights.

    python -m scfm_hidarm.restrict_stage1 --cell-line hepg2
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from . import config as C
from . import paths, restrict, stage
from .sibling import run3

#: the models whose stage-1 rows are only readable after restriction, plus the
#: frozen counterpart they are compared against. ``scprint`` is not here: it
#: never trained on this screen, so all five folds are held out for it.
RESTRICTED_MODELS = C.FT_MODELS + ("scgpt",)


def unit_dirs(sweep: str, cell_line: str, models: tuple[str, ...]) -> list[dict]:
    """Every (unit, bundle) output directory belonging to one of ``models``."""
    out: list[dict] = []
    for f in sorted(paths.sweep_dir(sweep).glob("*/unit.json")):
        rec = json.loads(f.read_text())
        if rec.get("cell_line") != cell_line or rec.get("model") not in models:
            continue
        for b in sorted(f.parent.glob("b*")):
            if (b / f"per_pert_{cell_line}.parquet").exists():
                out.append({"unit": rec["unit"], "model": rec["model"],
                            "layer": int(rec["layer"]), "k": int(rec["k"]), "dir": b})
    return out


def run(cell_line: str, sweep: str = "stage1", device: str = "cuda",
        models: tuple[str, ...] = RESTRICTED_MODELS) -> dict:
    """Restrict every unit of ``models`` for one cell line.

    The bundle used here holds **no run-6 blocks at all**. Nothing the
    restriction needs comes from a feature block --- it needs the scope rows,
    the folds, the reliability weights, the replicate halves and the ceilings,
    all of which live in run 3's ``targets.npz`` --- and staging without blocks
    keeps the check honest: the fold vector is derived from the measurement
    alone, with no run-6 artefact anywhere in the path.
    """
    units = unit_dirs(sweep, cell_line, models)
    if not units:
        raise RuntimeError(
            f"no stage-1 units for {cell_line} among {list(models)} under "
            f"{paths.sweep_dir(sweep)}; the fine-tuned arms cannot be quoted at all until "
            "they are read on the folds they did not train on, so an empty result is a "
            "failure and not 'nothing to do'"
        )
    work = paths.ensure(paths.stage_root() / "restrict" / cell_line)
    stage.stage(cell_line, [], work, scopes=C.SCOPES)
    LineBundle = run3("translate.data").LineBundle
    bundle = LineBundle(cell_line, root=work, device=device)

    done: list[dict] = []
    for u in units:
        d = Path(u["dir"])
        per_pert = pd.read_parquet(d / f"per_pert_{cell_line}.parquet")
        summary = restrict.restrict_run(
            bundle, per_pert, cell_line, (), d, d / "restricted",
        )
        done.append(u | {"dir": str(d), "restriction": summary})
        print(
            f"[restrict/{cell_line}] {u['unit']} {d.name}: "
            f"{summary['n_scored']} of {summary['n_perturbations_total']} perturbations, "
            f"{summary['n_groups_scored']} of {summary['n_groups_total']} response groups "
            f"on folds {summary['eval_folds']}",
            flush=True,
        )

    out = paths.ensure(paths.sweep_dir(sweep) / "restricted")
    frames = [
        pd.read_parquet(p).assign(unit=u["unit"], model=u["model"], layer=u["layer"],
                                  k=u["k"])
        for u in done
        for p in [Path(u["dir"]) / "restricted" / f"restricted_{cell_line}.parquet"]
        if p.exists()
    ]
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    merged.to_parquet(out / f"restricted_{cell_line}.parquet", index=False)

    record = {
        "cell_line": cell_line,
        "sweep": sweep,
        "models": list(models),
        "eval_folds": list(C.MODEL_HELDOUT_FOLDS),
        "n_units": len(done),
        "n_rows": int(len(merged)),
        "units": [{k: v for k, v in u.items() if k != "restriction"} for u in done],
        "fold_vector_check": done[0]["restriction"]["fold_vector_check"] if done else None,
        "n_scored": done[0]["restriction"]["n_scored"] if done else None,
        "n_groups_scored": done[0]["restriction"]["n_groups_scored"] if done else None,
        "bound_note": done[0]["restriction"]["bound_note"] if done else None,
        "contrast_note": done[0]["restriction"]["contrast_note"] if done else None,
        "five_fold_rows_unquotable": sorted(C.FT_MODELS),
        "five_fold_note": (
            "the five-fold stage-1 rows for the fine-tuned models are training numbers for "
            "three folds in five and may not be quoted; only the rows in this table may"
        ),
    }
    (out / f"restriction_{cell_line}.json").write_text(json.dumps(record, indent=2, default=str))
    shutil.rmtree(work, ignore_errors=True)
    print(f"[restrict/{cell_line}] {len(done)} unit(s), {len(merged)} rows -> {out}", flush=True)
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell-line", required=True, choices=list(C.CELL_LINES))
    ap.add_argument("--sweep", default="stage1")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    print(json.dumps(run(a.cell_line, a.sweep, a.device), indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
