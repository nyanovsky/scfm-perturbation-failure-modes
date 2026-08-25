"""Run 3's pre-registered shuffled-label band, recomputed from its bundle alone.

The band this experiment reads every fine-tuned arm against is run 3's own null:
the co-expression arm re-scored five times against shuffled perturbation labels,
restricted to the folds a fine-tuned model is entitled to be read on. Nothing in
it involves a fine-tuned checkpoint, so it is computed here -- once, in the merge
job -- rather than as a side effect of whichever training job happened to carry
the flag. The output table has the same schema as every other restricted table,
so the merge reads it exactly as it reads an arm's own.

    python3 -m scfm_ft.coexp_band --cell-line hepg2 --out _merge/scores
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import paths
from . import score
from .sibling import run3


def band_rows(
    bundle_dir: Path,
    cell_line: str,
    work: Path,
    device: str = "cuda",
    eval_folds=None,
) -> pd.DataFrame:
    """The five shuffled draws of the co-expression arm, read on the eval folds."""
    C3 = run3("config")
    runner = run3("translate.runner")
    LineBundle = run3("translate.data").LineBundle

    bundle = LineBundle(cell_line, root=Path(bundle_dir), device=device)
    rows, fold_info, ref = score._fold_context(bundle, device)
    fold, groups = fold_info["fold"], fold_info["groups"]
    eval_folds = np.asarray(C.MODEL_HELDOUT_FOLDS if eval_folds is None else eval_folds)
    ceilings = score.restricted_ceilings(bundle, ref, rows, eval_folds)

    arms = tuple(f"coexp+shuffled_labels_d{d}" for d in range(C3.N_SHUFFLE_DRAWS))
    # run 3 builds the shuffled null itself for the arms on its own list; asking
    # for one of those again would score the same arm twice
    extra = () if "coexp" in tuple(C3.SHUFFLE_NULL_ARMS) else ("coexp",)

    def always(_key) -> bool:
        return True

    _records, tables = runner.run_cell(
        bundle, C.SCOPE, C.SPLIT, always, Path(work), 0, 1,
        translators=("mlp",), with_cell_eval=False, shuffle_null=True,
        only_arms=arms, save_predictions=False, extra_shuffle_arms=extra,
    )
    per_pert = pd.concat(tables, ignore_index=True)
    per_pert["fold"] = per_pert["fold"].astype(np.int64)
    counts = per_pert.groupby(["arm", "translator"], sort=True).size()
    bad = counts[counts != int(rows.size)]
    if len(bad):
        raise RuntimeError(f"an arm was scored more than once: {bad.to_dict()}")
    missing = sorted(set(arms) - set(per_pert["arm"].unique()))
    if missing:
        raise RuntimeError(f"{cell_line}: shuffled draws not scored: {missing}")

    out = []
    for (arm, translator), grp in per_pert.groupby(["arm", "translator"], sort=True):
        out += [
            {**r, "arm": arm, "translator": translator}
            for r in score.restrict_scores(
                grp, fold, groups, eval_folds, ceilings, cell_line, f"{arm}/{translator}"
            )
        ]
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell-line", required=True, choices=list(C.CELL_LINES))
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--out", required=True, help="the merge job's score-table directory")
    ap.add_argument("--work", default="_band")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)

    line = a.cell_line
    bundle_root = Path(a.bundle) if a.bundle else paths.bundle_root()
    work = Path(a.work)
    bdir = score.stage_bundle(bundle_root, work / "bundle", line)
    frame = band_rows(bdir, line, work / "runner", device=a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"restricted_{line}__coexp_shuffle.parquet"
    frame.to_parquet(dest)
    keep = frame[(frame.target == "residual") & (frame.weighting == "weighted")]
    print(
        f"{line}: {len(frame)} rows -> {dest.name};  residual/weighted band "
        f"{keep.disattenuated.min():.6f} .. {keep.disattenuated.max():.6f} "
        f"over {len(keep)} draws",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
