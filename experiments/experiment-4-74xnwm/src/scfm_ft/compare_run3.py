"""Check run 4's whole-screen re-reading of run 3's arms against run 3 itself.

    PYTHONPATH=src apy -m scfm_ft.compare_run3

Run 4 re-runs three of run 3's arms -- co-expression, the frozen response
features and the frozen gene table -- over the whole screen, so the fine-tuned
arm has something to be read beside. Those re-runs use run 3's code through the
sibling alias, on run 3's grouped folds, against run 3's effect matrices, so
they should land on run 3's published numbers.

They are not expected to land on them exactly. The translator is an MLP fitted
per held-out fold, and refitting it is not deterministic across processes, so a
weak arm moves by more than a strong one. What the check asks is therefore
whether each re-reading sits inside run 3's bootstrap interval and run 3's
sits inside the re-reading's, which is the agreement two fits of the same
quantity should show.

The output is one small table, written next to the summary rather than into it:
the summary directory is what the merge job produced, and this comparison is
computed pod-side against a sibling experiment's results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import config as C
from . import paths
from .sibling import run3_dir

#: the cell of run 3's grid run 4 reads: whole screen, grouped folds, the MLP
#: translator, and the effect matrices run 1 built from every control cell
RUN3_FILTER = dict(scope="full", split="grouped", translator="mlp",
                   target_source="run1_all_controls")

#: run 3's column names for the verdict cell
R3 = "residual__weighted__disattenuated"

#: the arms run 4 re-reads, with the label the report uses
ARMS = (
    ("scgpt_resp", "response features, before fine-tuning"),
    ("coexp", "co-expression"),
    ("scgpt_emb", "gene table, before fine-tuning"),
    ("node", "node covariates"),
)


def build(out_dir: Path | None = None, summary: Path | None = None) -> Path:
    out_dir = Path(out_dir) if out_dir else paths.results_dir()
    summary = Path(summary) if summary else paths.results_dir() / "summary"
    run3 = pd.read_parquet(run3_dir() / "results" / "summary" / "runs.parquet")
    for k, v in RUN3_FILTER.items():
        run3 = run3[run3[k].astype(str) == v]

    rows = []
    for line in C.CELL_LINES:
        cf = pd.read_parquet(summary / f"crossfit_{line}__own_trainable.parquet")
        cf = cf[(cf.target == "residual") & (cf.weighting == "weighted")]
        for arm, label in ARMS:
            a = run3[(run3.cell_line == line) & (run3.arm == arm)]
            b = cf[(cf.cell_line == line) & (cf.arm == arm)]
            if a.empty or b.empty:
                continue
            a, b = a.iloc[0], b.iloc[0]
            r3, r3_lo, r3_hi = (float(a[R3]), float(a[f"{R3}_lo"]), float(a[f"{R3}_hi"]))
            r4 = float(b["disattenuated"])
            r4_lo, r4_hi = float(b["disattenuated_lo"]), float(b["disattenuated_hi"])
            rows.append({
                "cell_line": line,
                "arm": arm,
                "label": label,
                "run3": r3,
                "run3_lo": r3_lo,
                "run3_hi": r3_hi,
                "run3_n": int(a["residual__weighted__n_scored"]),
                "run4": r4,
                "run4_lo": r4_lo,
                "run4_hi": r4_hi,
                "run4_n": int(b["n_scored"]),
                "ceiling_run3": float(a["residual__weighted__ceiling"]),
                "ceiling_run4": float(b["ceiling"]),
                "difference": r4 - r3,
                # mutual containment: each point estimate inside the other's
                # interval. One-sided containment can happen when one interval
                # is much wider, so both directions are recorded.
                "run4_in_run3_interval": bool(r3_lo <= r4 <= r3_hi),
                "run3_in_run4_interval": bool(r4_lo <= r3 <= r4_hi),
            })
    frame = pd.DataFrame(rows)
    dest = Path(out_dir) / "run3_comparison.parquet"
    frame.to_parquet(dest)
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default=None)
    a = ap.parse_args(argv)
    dest = build(
        Path(a.out) if a.out else None, Path(a.summary) if a.summary else None
    )
    frame = pd.read_parquet(dest)
    print(frame.to_string(index=False))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
