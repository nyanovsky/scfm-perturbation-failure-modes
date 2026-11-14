"""Score run 6's installed blocks with run 3's runner, unmodified.

Everything scientific is run 3's: the folds, the leave-one-fold-out mean, the
reliability weights, the translator, the feature standardization, the arm
definitions, the correlation grid and the bootstrap. Run 6 supplies four things
and nothing else.

1. **The bundle a run gets** --- run 3's own bundle for the cell line, plus run
   6's projected hidden blocks written in run 3's block format, so
   ``resolve_arms`` picks them up as it picks up any model.
2. **The arm list**, because run 6 has arms run 3 never had.
3. **A prediction-saving policy**, described below.
4. **The restriction to the evaluation folds** for the fine-tuned arms and their
   frozen counterpart, which is run 4's and is imported rather than reinvented.

Why the prediction-saving policy is here and not in run 3
---------------------------------------------------------
Run 3's ``_keep_predictions`` keeps held-out predictions for a short list of
arms in exactly one cell of the sweep (``mlp``, ``full``, ``grouped``), because
one prediction matrix is ~15 MB and run 3 needed them only for its figures. Run
6 needs them for EVERY scored arm and EVERY cell, because two of its three
scoring families --- weighted cosine and the L1 discrimination score --- are
computed from the predictions after the runner has finished rather than inside
it. A discrimination score in particular cannot be recovered from a
per-perturbation correlation: it is a ranking over the whole scored pool.

So run 6 replaces ``runner._keep_predictions`` **in its own process**. Run 3's
file is not edited, its own jobs are unaffected, and the replacement is
asserted, printed and recorded in the sweep's ``qc.json`` rather than done
quietly. The alternative --- run 6 reimplementing ``run_cell``'s fold loop so it
could keep its own predictions --- would put a second copy of run 3's scoring
loop in the tree, and a copy that drifted by one line would make run 6's numbers
incomparable with the ~2,300 already scored while looking identical.

⚠️ This module reads no decoder-head value and installs no head-derived block.
``<model>_resp`` is deliberately absent from every arm list here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import metrics_extra as MX
from . import paths
from .sibling import run3, run4


# --------------------------------------------------------------------------- #
# the prediction-saving policy
# --------------------------------------------------------------------------- #
#: run 3's own policy, restated so that replacing it is a comparison against a
#: known thing rather than an overwrite of whatever happens to be there
RUN3_KEEP_PREDICTIONS_CELL = {"translator": "mlp", "scope": "full", "split": "grouped"}


def force_prediction_saving(verbose: bool = True) -> dict:
    """Make run 3's runner keep the held-out predictions of every arm it scores.

    Applied to the imported module object in run 6's process only. Run 3's
    source is untouched and its own sweeps keep their own policy.

    The original is exercised on four known inputs before it is replaced, so
    that a change to run 3's policy shows up here as a mismatch rather than
    being silently overwritten by run 6's.
    """
    runner = run3("translate.runner")
    original = runner._keep_predictions  # noqa: SLF001
    probes = {
        ("coexp", "mlp", "full", "grouped"): True,
        ("scgpt_resp", "mlp", "full", "grouped"): True,
        ("coexp", "ridge", "full", "grouped"): False,
        ("coexp", "mlp", "full", "random"): False,
    }
    observed = {k: bool(original(*k)) for k in probes}
    matches = observed == probes
    if not matches:
        print(
            "[score] WARNING: run 3's _keep_predictions no longer behaves as recorded; "
            f"expected {probes}, observed {observed}. Run 6's policy is applied anyway, "
            "but the note in this module's docstring is now stale",
            flush=True,
        )

    def keep_everything(arm: str, translator: str, scope: str, split: str) -> bool:
        """Run 6's policy: every scored arm, every cell of the sweep.

        Storage is the only cost and it is small beside the extraction budget;
        the alternative is that the cosine and discrimination cells exist for
        one translator and one split, which would make them unreadable against
        the ridge capacity control and against the random split.
        """
        return True

    runner._keep_predictions = keep_everything  # noqa: SLF001
    record = {
        "policy": "run 6 keeps held-out predictions for every arm and every cell",
        "reason": (
            "the weighted-cosine and L1-discrimination cells are computed from saved "
            "predictions after the runner finishes; a discrimination score is a ranking "
            "over the whole scored pool and cannot be recovered from per-perturbation "
            "correlations"
        ),
        "run3_source_edited": False,
        "run3_original_policy_cell": RUN3_KEEP_PREDICTIONS_CELL,
        "run3_original_policy_matched_record": bool(matches),
        "run3_original_policy_observed": {"/".join(k): v for k, v in observed.items()},
    }
    if verbose:
        print(f"[score] {json.dumps(record)}", flush=True)
    return record


# --------------------------------------------------------------------------- #
# folds
# --------------------------------------------------------------------------- #
def fold_context(bundle, scope: str = "full", split: str = "grouped") -> dict:
    """The scope rows, folds, groups and fold reference every scorer shares.

    Recomputed with run 3's own three calls rather than read off a run --- which
    is exactly what makes them run 3's folds. ``make_folds`` groups from the
    MEASURED effect matrix of the scope rows and is independent of the feature,
    so a run-6 arm inherits run 3's exact folds provided the scope rows are
    unchanged. That "provided" is asserted by ``verify.py``, not assumed.
    """
    import torch

    C3 = run3("config")
    runner = run3("translate.runner")
    splitsmod = run3("translate.splits")
    rows = bundle.scope_rows(scope, C3.PRIMARY_ANCHOR)
    effect_rows = bundle.effect[torch.as_tensor(rows, device=bundle.effect.device)].cpu().numpy()
    info = splitsmod.make_folds(effect_rows, split, C3.N_FOLDS, C3.SEED, C3.GROUP_CORR_THRESHOLD)
    ref = runner.fold_reference(bundle, rows, info["fold"], C3.N_FOLDS)
    return {"rows": rows, "info": info, "fold": info["fold"], "groups": info["groups"], "ref": ref}


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def score_bundle(
    bundle_dir: Path,
    cell_line: str,
    arms: tuple[str, ...],
    shuffle_bases: tuple[str, ...],
    out_dir: Path,
    device: str = "cuda",
    splits: tuple[str, ...] = C.SPLITS,
    translators: tuple[str, ...] = C.TRANSLATORS,
) -> dict:
    """Run every requested arm through run 3's ``run_cell``, both splits.

    ``--scopes full`` only, and that is not a default: installing a feature
    block redefines ``scope_rows("common")`` for every arm sharing the bundle,
    and ``common`` exists for AIDO.Cell comparability that run 6 has no arm for.
    Asking for it here is refused rather than silently honoured.
    """
    C3 = run3("config")
    runner = run3("translate.runner")
    LineBundle = run3("translate.data").LineBundle

    policy = force_prediction_saving()
    bundle = LineBundle(cell_line, root=Path(bundle_dir), device=device)
    out_dir = paths.ensure(Path(out_dir))
    try:
        return _score_bundle_inner(
            bundle, cell_line, arms, shuffle_bases, out_dir, policy,
            splits, translators, runner, C3,
        )
    finally:
        # ``LineBundle`` moves EVERY array it holds onto the scoring device, and
        # a k=64 response block is 3.6 GB (HepG2) / 5.9 GB (Jurkat). One process
        # scores dozens of bundles in sequence, so a bundle that is merely
        # dropped --- left for the garbage collector, its memory left in torch's
        # caching allocator --- accumulates until the next construction cannot
        # fit. That is not hypothetical: the 2026-09-01 wave died with
        # ``CUDA out of memory ... 22.45 GiB in use`` on its second unit, after
        # the first had already been scored and written. The failure is at
        # bundle CONSTRUCTION, i.e. after the previous unit's numbers are safely
        # on disk, so nothing computed was ever wrong --- but the wave stops.
        release_bundle(bundle, device)


def release_bundle(bundle, device: str) -> None:
    """Drop a bundle's device tensors and hand the memory back to the allocator.

    ``del`` alone is not enough: torch's caching allocator keeps freed blocks
    for reuse, which is right within one bundle and wrong across bundles of
    different shapes, where the cached blocks fragment instead of being reused.
    ``empty_cache`` is the documented way to return them.
    """
    import gc

    import torch

    for attr in list(vars(bundle)):
        try:
            delattr(bundle, attr)
        except (AttributeError, TypeError):
            pass
    del bundle
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _score_bundle_inner(
    bundle,
    cell_line: str,
    arms: tuple[str, ...],
    shuffle_bases: tuple[str, ...],
    out_dir: Path,
    policy: dict,
    splits,
    translators,
    runner,
    C3,
) -> dict:
    """The body of ``score_bundle``, split out only so the bundle can be freed."""

    def always(_key) -> bool:
        return True

    # run 3 already draws a shuffled null for the arms in its own list; asking
    # for those again would score the same arm twice and give the aggregation
    # two rows per perturbation
    extra_shuffle = tuple(b for b in shuffle_bases if b not in tuple(C3.SHUFFLE_NULL_ARMS))
    shuffled = tuple(
        f"{b}+shuffled_labels_d{d}" for b in shuffle_bases for d in range(C3.N_SHUFFLE_DRAWS)
    )

    records: list[dict] = []
    tables: list[pd.DataFrame] = []
    for split in splits:
        r, t = runner.run_cell(
            bundle, "full", split, always, out_dir, 0, 1,
            translators=tuple(translators), with_cell_eval=False, shuffle_null=False,
            only_arms=tuple(arms), save_predictions=True,
        )
        records += r
        tables += t
        if shuffled:
            r2, t2 = runner.run_cell(
                bundle, "full", split, always, out_dir, 0, 1,
                translators=("mlp",), with_cell_eval=False, shuffle_null=True,
                only_arms=shuffled, save_predictions=True,
                extra_shuffle_arms=extra_shuffle,
            )
            records += r2
            tables += t2

    if not records:
        raise RuntimeError(
            f"{cell_line}: no arm was scored. The most likely cause is that a feature "
            "block was not picked up by the bundle reader (a key containing '__', or a "
            "row_map with a -1, both of which make run 3 skip the model silently) --- "
            f"the bundle resolved models {sorted({m for m, _ in bundle.features})}"
        )
    per_pert = pd.concat(tables, ignore_index=True)
    per_pert["fold"] = per_pert["fold"].astype(np.int64)

    # an arm whose block the reader did not resolve is DROPPED silently, which
    # would leave the run's headline arm missing from the table rather than wrong
    scored = set(per_pert["arm"].unique())
    absent = sorted(set(arms) - scored)
    if absent:
        raise RuntimeError(
            f"{cell_line}: the arms {absent} were requested but not scored; the bundle "
            f"resolved models {sorted({m for m, _ in bundle.features})}. A key containing "
            "'__' or a row_map with a -1 makes run 3 skip a model with no error at all"
        )

    extra = extra_cells(bundle, out_dir, cell_line, splits, out_dir)
    pd.DataFrame(records).to_parquet(out_dir / f"runs_{cell_line}.parquet", index=False)
    per_pert.to_parquet(out_dir / f"per_pert_{cell_line}.parquet", index=False)
    summary = {
        "cell_line": cell_line,
        "n_arms": int(len(scored)),
        "n_runs": len(records),
        "prediction_policy": policy,
        "bundle_inventory": runner.bundle_inventory(bundle),
        "extra_metrics": extra,
        "emits": "hidden_states_only",
    }
    (out_dir / f"scoring_{cell_line}.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def extra_cells(bundle, pred_root: Path, cell_line: str, splits, out_dir: Path) -> dict:
    """The weighted-cosine and discrimination cells, from the saved predictions.

    Read back rather than computed inside the runner, so that the runner stays
    run 3's file unmodified and so that the same code path serves both the
    whole-screen reading and the fold-restricted one --- a discrimination score
    has to be RECOMPUTED on a restricted pool rather than subset from the full
    one, because the ranking's reference set changes with the pool.
    """
    pdir = Path(pred_root) / "predictions"
    if not pdir.is_dir():
        raise RuntimeError(
            f"no predictions under {pdir}; the cosine and discrimination cells are "
            "computed from them, so their absence is a failure and not an empty result"
        )
    contexts = {split: fold_context(bundle, "full", split) for split in splits}
    # The ceilings are arm-INDEPENDENT and every one of them is expensive: the
    # discrimination ceiling is a pairwise L1 over the whole scored pool. So
    # they are computed once per split and handed to every arm, which is what
    # `metrics_extra.extra_ceilings`'s own docstring says they exist for.
    # Computing them inside score_extra instead --- which is what this function
    # did until 2026-09-01 --- repeats that pairwise pass once per prediction
    # file, i.e. eight or nine times per bundle for an identical number.
    ceilings = {
        split: MX.extra_ceilings(
            bundle.half0[ctx["rows"]], bundle.half1[ctx["rows"]], ctx["ref"]
        )
        for split, ctx in contexts.items()
    }
    rows: list[dict] = []
    per_pert: list[pd.DataFrame] = []
    for f in sorted(pdir.glob(f"{cell_line}__full__*.npz")):
        _, scope, split, arm, translator = f.stem.split("__", 4)
        ctx = contexts.get(split)
        if ctx is None:
            continue
        z = np.load(f, allow_pickle=False)
        summary, table = MX.score_extra(
            pred_resid=z["prediction_residual"],
            ref=ctx["ref"],
            fold=ctx["fold"],
            n_folds=run3("config").N_FOLDS,
            self_cols=bundle.self_col_np[ctx["rows"]],
            # every interval is a GROUP bootstrap, from the start rather than
            # re-widened afterwards: perturbations are not independent draws
            # (HepG2's 524 fall into 226 response groups, one of them 149 rows)
            # and a row-level interval is too narrow for this design
            groups=np.asarray(ctx["groups"]),
            ceilings=ceilings[split],
        )
        rows.append(
            {"cell_line": cell_line, "scope": scope, "split": split, "arm": arm,
             "translator": translator, **summary}
        )
        per_pert.append(
            table.assign(cell_line=cell_line, scope=scope, split=split, arm=arm,
                         translator=translator,
                         perturbation=bundle.pert_target[ctx["rows"]])
        )
    if not rows:
        raise RuntimeError(f"no prediction files matched {cell_line} under {pdir}")
    pd.DataFrame(rows).to_parquet(Path(out_dir) / f"extra_runs_{cell_line}.parquet", index=False)
    pd.concat(per_pert, ignore_index=True).to_parquet(
        Path(out_dir) / f"extra_per_pert_{cell_line}.parquet", index=False
    )
    return {"n_rows": len(rows), "n_prediction_files": len(rows),
            "metrics": list(C.EXTRA_METRICS)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True, help="a staged bundle root from stage.py")
    ap.add_argument("--cell-line", required=True, choices=list(C.CELL_LINES))
    ap.add_argument("--arms", required=True, help="comma-separated arm names")
    ap.add_argument("--shuffle-arms", default="", help="arms to draw the shuffled-label null for")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default=",".join(C.SPLITS))
    ap.add_argument("--translators", default=",".join(C.TRANSLATORS))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    res = score_bundle(
        Path(args.bundle),
        args.cell_line,
        tuple(a for a in args.arms.split(",") if a),
        tuple(a for a in args.shuffle_arms.split(",") if a),
        Path(args.out),
        device=args.device,
        splits=tuple(s.strip() for s in args.splits.split(",") if s.strip()),
        translators=tuple(t.strip() for t in args.translators.split(",") if t.strip()),
    )
    print(json.dumps({k: v for k, v in res.items() if k != "bundle_inventory"}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
