"""Reduce the fold-restricted rows to a small tracked table the paper may read.

``restrict_stage1`` writes one parquet per cell line beside the sweep, under
``$SCFM_PFM6_DATA_ROOT``. Those files are heavy and untracked, and the rule this
project writes papers under is that a number which cannot be recovered from a
tracked file does not go in a paper. So the fine-tuned reading --- the only one
of run 6's results that exists solely in those parquets --- has to come through
here before it can be cited.

This module reduces, it never recomputes. Every score, interval, ceiling and
count is copied from the row ``restrict.restrict_run`` already wrote; nothing is
re-derived, so the tracked table cannot disagree with the parquet it came from.

Two shapes are deliberate.

**Both metric cells travel side by side, in the same row.** The verdict is run
3's pre-registered ``residual__weighted__disattenuated``; the fold-centred cell
is the post-hoc secondary arms are only *ordered* by. Writing them as two
columns of one row rather than two rows of one column makes it impossible to
quote one while believing it was the other.

**The interval columns are not optional.** These are the only rows in run 6 that
carry a group bootstrap, and they are also the rows whose point estimates are
least readable on their own: the restriction leaves 54 of 226 response groups in
HepG2 and the intervals overlap throughout. A table of these point estimates
without their intervals would read as a demonstrated null, which is precisely
what this reading is not.

    python -m scfm_hidarm.restricted_summary --sweep stage1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import analysis
from . import config as C
from . import paths

#: the two (target, weighting) cells the record quotes, in the order the run's
#: claims discipline reads them: the pre-registered verdict first.
CELLS = (
    ("residual", "weighted", C.PRIMARY_METRIC),
    ("centered", "weighted", C.ORDERING_METRIC),
)

#: identity of one restricted reading. Everything else in the parquet is either
#: one of the two metric cells or a copy of these.
KEY_COLUMNS = (
    "cell_line", "arm", "model", "variant", "layer", "k", "split", "translator",
    "is_shuffle_draw", "shuffle_base",
)

#: copied through unchanged, once per row, and asserted equal across the cells
#: --- a count that differed between two cells of the same reading would mean
#: the two were computed on different rows. The **ceiling is not here**: it is a
#: per-target quantity (0.7632 residual against 0.6079 fold-centred in HepG2),
#: and the two are the pair `docs/METRICS.md` s12 records as both being called
#: "the residual ceiling". Each travels with the cell it divides.
SHARED_COLUMNS = ("n_scored", "n_groups")


def restricted_dir(sweep: str) -> Path:
    """Where ``restrict_stage1`` merged its per-unit output."""
    return paths.sweep_dir(sweep) / "restricted"


def read_restricted(sweep: str, cell_line: str) -> pd.DataFrame:
    """One cell line's restricted rows, with run 6's key decoded onto them.

    The parquet carries run 3's ``model`` column, which for a run-6 block is the
    whole pseudo-model key. ``analysis.decode_arm`` is the same decoder the depth
    table uses, so the two tracked tables join on identical keys.
    """
    path = restricted_dir(sweep) / f"restricted_{cell_line}.parquet"
    if not path.is_file():
        raise RuntimeError(
            f"no restricted parquet at {path}; run `python -m scfm_hidarm.restrict_stage1 "
            f"--cell-line {cell_line}` first. A missing file fails here rather than writing "
            "an empty tracked table, which would read as 'the fine-tuned arms scored nothing'"
        )
    df = pd.read_parquet(path)
    decoded = pd.DataFrame([analysis.decode_arm(a) for a in df["arm"]], index=df.index)
    for col in ("model", "variant", "layer", "k"):
        df[col] = decoded[col] if col in decoded else pd.NA
    df["is_shuffle_draw"] = df["arm"].str.contains(analysis.SHUFFLE_MARK, regex=False)
    df["shuffle_base"] = df["arm"].str.split(analysis.SHUFFLE_MARK, regex=False).str[0]
    return df


def reduce_cell_line(sweep: str, cell_line: str) -> pd.DataFrame:
    """One row per restricted reading, both metric cells and both intervals on it."""
    df = read_restricted(sweep, cell_line)
    if set(df["metric_family"]) != {"pearson"}:
        raise RuntimeError(
            f"{cell_line}'s restricted parquet carries metric families "
            f"{sorted(set(df['metric_family']))}; only the pearson family is disattenuated "
            "under run 3's rule and only it may be reduced here"
        )

    frames = []
    for target, weighting, metric in CELLS:
        sel = df[(df["target"] == target) & (df["weighting"] == weighting)].copy()
        if sel.empty:
            raise RuntimeError(
                f"{cell_line} has no ({target}, {weighting}) rows; the pre-registered cell "
                "cannot be assembled and the reduction would silently drop it"
            )
        keep = sel[list(KEY_COLUMNS) + list(SHARED_COLUMNS) +
                   ["disattenuated", "disattenuated_lo", "disattenuated_hi", "ceiling"]]
        frames.append(
            keep.rename(columns={
                "disattenuated": metric,
                "disattenuated_lo": f"{metric}_lo",
                "disattenuated_hi": f"{metric}_hi",
                "ceiling": f"{metric}_ceiling",
            })
        )

    out = frames[0]
    for extra in frames[1:]:
        metric_cols = [c for c in extra.columns if c not in KEY_COLUMNS + SHARED_COLUMNS]
        out = out.merge(
            extra[list(KEY_COLUMNS) + list(SHARED_COLUMNS) + metric_cols],
            on=list(KEY_COLUMNS), how="outer", suffixes=("", "_other"), validate="one_to_one",
        )
        for col in SHARED_COLUMNS:
            other = out.pop(f"{col}_other")
            if not out[col].equals(other):
                raise RuntimeError(
                    f"{cell_line}: '{col}' differs between the two metric cells of the same "
                    "reading, so they were not computed on the same restricted rows"
                )
    return out.sort_values(["model", "layer", "variant", "translator", "arm"])


def build(sweep: str = "stage1", out_dir: Path | None = None) -> dict:
    """Write ``<sweep>_restricted.csv`` and its provenance record, both tracked."""
    tables, meta = [], {}
    for cell_line in C.CELL_LINES:
        tables.append(reduce_cell_line(sweep, cell_line))
        record = restricted_dir(sweep) / f"restriction_{cell_line}.json"
        loaded = json.loads(record.read_text())
        # only the fields that qualify a quoted number; the unit list is a path
        # dump and belongs beside the sweep, not in the repo.
        meta[cell_line] = {
            k: loaded.get(k)
            for k in ("eval_folds", "n_units", "n_rows", "fold_vector_check",
                      "n_scored", "n_groups_scored", "bound_note", "contrast_note")
        }
    merged = pd.concat(tables, ignore_index=True)

    out = paths.ensure(Path(out_dir) if out_dir is not None else paths.summary_dir())
    merged.to_csv(out / f"{sweep}_restricted.csv", index=False)
    headline = {
        "sweep": sweep,
        "source": "sweep/<sweep>/restricted/restricted_<cell_line>.parquet",
        "reduced_not_recomputed": True,
        "primary_metric": C.PRIMARY_METRIC,
        "ordering_metric": C.ORDERING_METRIC,
        "eval_folds": list(C.MODEL_HELDOUT_FOLDS),
        "frozen_counterpart": C.FT_FROZEN_COUNTERPART,
        "n_rows": int(len(merged)),
        "intervals": "group bootstrap over response groups, copied from the restricted parquet",
        "cell_lines": meta,
    }
    (out / f"{sweep}_restricted_meta.json").write_text(
        json.dumps(headline, indent=2, default=str)
    )
    print(
        f"[restricted-summary] {len(merged)} rows -> {out / (sweep + '_restricted.csv')}",
        flush=True,
    )
    return headline


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default="stage1")
    ap.add_argument("--out", type=Path, default=None,
                    help="where the tracked table goes (default: results/summary)")
    a = ap.parse_args(argv)
    print(json.dumps(build(a.sweep, a.out), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
