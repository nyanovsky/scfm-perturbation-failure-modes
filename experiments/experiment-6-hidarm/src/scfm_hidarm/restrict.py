"""Read the fine-tuned arms on the folds they never trained on --- run 4's rule.

Run 4's ``MODEL_HELDOUT_FOLDS = (0, 1)``. The fine-tuned checkpoints trained on
folds 2, 3 and 4, so run 3's unmodified five-fold sweep would make three fifths
of their perturbations a training number. This is the one place run 6 cannot
reuse run 3's rule end to end, and the fix is run 4's, **imported rather than
reinvented**: run 3's runner scores per perturbation, then ``restrict_scores``
keeps the evaluation folds and ``restricted_ceilings`` recomputes the ceiling on
that subset --- the same correlations, the same weights, read on the subset the
model is entitled to be read on.

Three things this module refuses to let slide.

**The frozen counterpart is restricted too.** A two-fold fine-tuned score set
beside a five-fold frozen score is not a comparison. Run 6 keeps predictions for
every arm, so re-reading frozen scGPT on the same two folds is free.

**The discrimination score is RECOMPUTED, never subset.** A weighted correlation
is per-perturbation and restricting it is a row filter. A discrimination score
is a ranking against a pool, so restricting the pool changes every rank in it.
Subsetting the full-pool numbers would report a 524-way ranking as if it were a
~210-way one, which is a different and easier problem.

**The bound is worded as a bound.** On folds 0 and 1 co-expression itself is not
separated from the band (``docs/INDEX.md`` s9), so whatever the fine-tuned arm
shows here is a bound at the achieved resolution --- exactly as run 4's headline
is. That is run 4's limitation carried over verbatim, not a new caveat.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import metrics_extra as MX
from . import paths, score
from .sibling import run3, run4


def eval_folds() -> np.ndarray:
    """Run 4's evaluation folds, read from run 4 and checked against run 6's copy.

    ``config.MODEL_HELDOUT_FOLDS`` exists so that run 6's pre-registration is
    readable in one file, but the authority is run 4's own constant. If the two
    disagree the restriction does not mean what run 6 thinks and the arm is
    unscoreable, so this raises rather than picking one.
    """
    r4 = run4("config")
    theirs = tuple(int(f) for f in r4.MODEL_HELDOUT_FOLDS)
    ours = tuple(int(f) for f in C.MODEL_HELDOUT_FOLDS)
    if theirs != ours:
        raise RuntimeError(
            f"run 4 holds out folds {theirs} but run 6's config says {ours}; the fine-tuned "
            "arms would be read on folds the checkpoints trained on"
        )
    return np.asarray(ours)


def fold_vector_check(bundle, cell_line: str) -> dict:
    """Do run 6's grouped folds equal run 4's, element for element?

    ``ingest.py:fold_assignment`` recomputes run 3's grouped folds, and
    ``make_folds`` is deterministic given the same effect matrix and scope rows,
    so run 6's fold vector *should* equal run 4's. "Should" is not a check. If
    it differs, ``MODEL_HELDOUT_FOLDS`` names different perturbations in the two
    runs and the fine-tuned arm cannot be scored at all.
    """
    ctx = score.fold_context(bundle, "full", "grouped")
    ours = np.asarray(ctx["fold"], dtype=np.int64)
    # run 4's own function, over run 3's UNSTAGED bundle. Deliberately not the
    # staged copy: the staged tree carries run 6's installed blocks, and the
    # point of the check is that the two runs agree on folds derived from the
    # measurement alone, with no run-6 artefact anywhere in the path.
    theirs_info = run4("ingest").fold_assignment(cell_line, paths.run3_bundle_root())
    theirs = np.asarray(theirs_info["fold"], dtype=np.int64)
    same_genes = np.array_equal(
        np.asarray(bundle.pert_gene_id)[ctx["rows"]].astype(str),
        theirs_info["pert_gene_id"].astype(str),
    )
    n = int(min(ours.size, theirs.size))
    agree = int((ours[:n] == theirs[:n]).sum())
    ok = bool(same_genes and ours.size == theirs.size and agree == ours.size)
    return {
        "ok": ok,
        "n": int(ours.size),
        "n_agree": agree,
        "same_perturbation_order": bool(same_genes),
        "detail": (
            f"{agree} of {ours.size} agree between run 6's grouped folds and run 4's"
            + ("" if same_genes else "; the two runs' perturbation orders DIFFER, so the "
               "element-wise comparison above is not meaningful")
        ),
    }


def restrict_run(
    bundle,
    per_pert: pd.DataFrame,
    cell_line: str,
    arms: tuple[str, ...],
    pred_root: Path,
    out_dir: Path,
    folds=None,
    split: str = "grouped",
) -> dict:
    """Every requested arm, read on the evaluation folds, both metric families.

    The Pearson cells go through run 4's ``restrict_scores`` unchanged. The
    cosine and discrimination cells are recomputed here from the saved
    predictions on the restricted pool, for the reason in this module's
    docstring.
    """
    r4score = run4("score")
    folds = eval_folds() if folds is None else np.asarray(folds)
    ctx = score.fold_context(bundle, "full", split)
    rows, fold, groups, ref = ctx["rows"], ctx["fold"], ctx["groups"], ctx["ref"]
    ceilings = r4score.restricted_ceilings(bundle, ref, rows, folds)
    self_cols = bundle.self_col_np[rows]
    keep = np.isin(np.asarray(fold), folds)

    pearson_rows: list[dict] = []
    for (arm, translator), grp in per_pert[per_pert["split"] == split].groupby(
        ["arm", "translator"], sort=True
    ):
        if arms and arm not in arms:
            continue
        pearson_rows += [
            {**r, "arm": arm, "translator": translator, "split": split,
             "metric_family": "pearson"}
            for r in r4score.restrict_scores(
                grp, fold, groups, folds, ceilings, cell_line, f"{arm}/{translator}"
            )
        ]

    extra_rows: list[dict] = []
    # arm-independent and pool-size dependent, so: computed ONCE, on this
    # restricted subset, and handed to every arm. Recomputing them per
    # prediction file repeats a pairwise L1 over the whole pool for an identical
    # number, and `score_extra` refuses a subset mismatch rather than dividing
    # by the wrong denominator, so the single call is also the checked one.
    keep_rows = np.flatnonzero(keep)
    extra_ceil = MX.extra_ceilings(
        bundle.half0[rows], bundle.half1[rows], ref, keep_rows
    )
    pdir = Path(pred_root) / "predictions"
    for f in sorted(pdir.glob(f"{cell_line}__full__{split}__*.npz")):
        _, scope, spl, arm, translator = f.stem.split("__", 4)
        if arms and arm not in arms:
            continue
        z = np.load(f, allow_pickle=False)
        summary, _ = MX.score_extra(
            pred_resid=z["prediction_residual"],
            ref=ref,
            fold=np.asarray(fold),
            n_folds=run3("config").N_FOLDS,
            self_cols=self_cols,
            # the reading is restricted; the CONSTRUCTION is not. The three
            # targets and the weight matrix still come from the training-fold
            # means the prediction was actually made against, and only then is
            # the score read on the evaluation folds --- which is run 4's rule.
            rows_subset=keep_rows,
            groups=np.asarray(groups),
            ceilings=extra_ceil,
        )
        extra_rows.append(
            {"cell_line": cell_line, "scope": scope, "split": spl, "arm": arm,
             "translator": translator, "metric_family": "extra",
             "eval_folds": [int(x) for x in folds], **summary}
        )

    out_dir = paths.ensure(Path(out_dir))
    pd.DataFrame(pearson_rows).to_parquet(out_dir / f"restricted_{cell_line}.parquet", index=False)
    pd.DataFrame(extra_rows).to_parquet(
        out_dir / f"restricted_extra_{cell_line}.parquet", index=False
    )
    summary = {
        "cell_line": cell_line,
        "eval_folds": [int(f) for f in folds],
        "split": split,
        "n_perturbations_total": int(rows.size),
        "n_scored": int(keep.sum()),
        "n_groups_total": int(np.unique(groups).size),
        "n_groups_scored": int(np.unique(np.asarray(groups)[keep]).size),
        "ceilings_restricted": ceilings,
        "ceilings_all_folds": ref["ceilings"],
        "fold_vector_check": fold_vector_check(bundle, cell_line),
        "pds_recomputed_on_restricted_pool": True,
        "bound_note": (
            "on folds 0 and 1 co-expression itself is not separated from the band, so any "
            "fine-tuned result read here is a bound at the achieved resolution, exactly as "
            "run 4's headline is"
        ),
        "contrast_note": (
            "fine-tuned against frozen scGPT is the fine-tuned model in its native regime "
            "against the frozen model in the only regime it has, never a weights contrast"
        ),
    }
    (out_dir / f"restriction_{cell_line}.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--cell-line", required=True, choices=list(C.CELL_LINES))
    ap.add_argument("--per-pert", required=True, help="per_pert_<line>.parquet from score.py")
    ap.add_argument("--predictions", required=True, help="the sweep dir holding predictions/")
    ap.add_argument("--arms", default="", help="comma-separated; empty = every arm present")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    LineBundle = run3("translate.data").LineBundle
    bundle = LineBundle(args.cell_line, root=Path(args.bundle), device=args.device)
    per_pert = pd.read_parquet(args.per_pert)
    res = restrict_run(
        bundle, per_pert, args.cell_line,
        tuple(a for a in args.arms.split(",") if a),
        Path(args.predictions), Path(args.out),
    )
    print(json.dumps(res, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
