"""Emit the normalised-rank reading of the retrieval probe as a tracked table.

``retrieval_scores`` has always computed ``normalized_rank`` and has never emitted
it: it reaches ``summary.parquet`` only as free text inside the ``detail`` column of
the split-half rows, and the held-out-cell path does not even do that. So a figure or
a paper that wants the rank reading has no tracked file to read, which is why this
module exists.

**It recomputes, and then checks the recomputation.** The ranks are rebuilt from the
blocks through this run's own ``Block`` loader, its own cell split
(``analysis.heldout_plan``) and its own seeded tie-breaking, so a rank here is the
rank that produced the published top-1. That is verified rather than asserted: for
every split-half cell the recomputed value is compared against the number this run
printed into its own ``detail`` string, and the largest disagreement is printed and
raised on.

**Two definitions travel together, because they are not the same number.**

``normalized_rank``   ``1 - (rank - 1)/(n - 1)``  -- run 3's definition, and the one
                      ``retrieval_scores`` already reports
``rank_fraction``     ``(n - rank + 1)/n``        -- rank 1 -> 1, rank n -> 1/n

They are affinely related and differ by at most ``1/n`` (0.0019 in HepG2). Both are
written so that whichever a reader is quoting, they can see the other.

⚠️ **Chance on either definition is about 0.5, not 1/n.** A row at 0.5 is a dead
layer, not a weak one. Block 0 sits there for the same reason it sits at 0.002 on
top-1: the intervention has not reached the other genes' tokens at all, so every
perturbation has the same signature and the rank is a tie-break.

    python -m scfm_probe.rank_summary
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis as A
from . import config as C
from . import paths
from .probe import retrieval as retr

#: the models the rank reading is emitted for. TranscriptFormer's input intervention
#: modifies attention scores rather than entering the residual stream, so its ranks
#: are not on the same axis as these two and it is left out rather than tabulated
#: beside them.
MODELS = ("scgpt", "scprint")
LINES = ("hepg2", "jurkat")


def rank_fraction(ranks: np.ndarray, n: int) -> float:
    """rank 1 -> 1, rank 2 -> (n-1)/n, rank n -> 1/n. Chance is (n+1)/2n."""
    return float(np.mean((n - ranks + 1.0) / n))


def normalized_rank(ranks: np.ndarray, n: int) -> float:
    """Run 3's definition, the one ``retrieval_scores`` already reports."""
    return float(np.mean(1.0 - (ranks - 1.0) / (n - 1.0)))


def chance(n: int) -> float:
    """``(n+1)/2n``: the mean of ``rank_fraction`` over a uniformly random rank.

    Carried in every row because it is the number these scores must be read against
    and it is NOT ``1/n``. Reading a rank score against the top-1 chance level would
    make a dead layer look like a strong one by a factor of 250.
    """
    return (n + 1.0) / (2.0 * n)


def split_half_rows(block, layer: int) -> list[dict]:
    """Halves of the read-out positions, exactly as ``analysis`` pairs them."""
    arrays = block.arrays(layer)
    no_op = block.no_op_mask
    out = []
    for vi, value in enumerate(block.value_grid_cp10k.tolist()):
        keep = ~no_op[:, vi]
        pool = arrays["pool"][keep, vi, :, :]
        n = int(pool.shape[0])
        if n < 2:
            continue
        dist = retr.cosine_distance(pool[:, 0, :], pool[:, 1, :])
        ranks = retr.rank_of_own(dist).astype(np.float64)
        out.append({
            "model": block.model, "cell_line": block.cell_line, "split": "split_half",
            "layer": layer, "value_cp10k": float(value), "n": n,
            "rank_fraction": rank_fraction(ranks, n),
            "normalized_rank": normalized_rank(ranks, n),
            "chance": chance(n),
            "top1": float(np.mean(ranks <= 1)), "median_rank": float(np.median(ranks)),
        })
    return out


def heldout_rows(blocks, layer: int, plan: dict) -> list[dict]:
    """Nearest centroid over the training cells, scored on the held-out cells.

    The signature, the centroid and the tie-break seed are taken from
    ``analysis.heldout_cell_layer`` unchanged; only the statistic differs.
    """
    ref = blocks[0]
    train_idx, test_idx = plan["train_idx"], plan["test_idx"]
    stacked = np.stack(
        [b.arrays(layer)["pool"].mean(axis=2).astype(np.float32) for b in blocks]
    )
    out = []
    for vi, value in enumerate(ref.value_grid_cp10k.tolist()):
        cell = (ref.model, ref.cell_line, plan["anchor"], layer, float(value),
                "heldout_cell")
        centroid = stacked[train_idx, :, vi, :].mean(axis=0)
        n = int(centroid.shape[0])
        per_cell = []
        for c in test_idx:
            dist = retr.cosine_distance(stacked[c, :, vi, :], centroid)
            per_cell.append(retr.rank_of_own(
                dist,
                seed=A.derive_seed(C.RETRIEVAL_TIE_SEED, *cell, "heldout", int(c)),
            ).astype(np.float64))
        ranks = np.concatenate(per_cell)
        out.append({
            "model": ref.model, "cell_line": ref.cell_line, "split": "heldout_cell",
            "layer": layer, "value_cp10k": float(value), "n": n,
            "rank_fraction": rank_fraction(ranks, n),
            "normalized_rank": normalized_rank(ranks, n),
            "chance": chance(n),
            "top1": float(np.mean(ranks <= 1)), "median_rank": float(np.median(ranks)),
        })
    return out


def build(out_dir: Path | None = None) -> dict:
    """Recompute every rank cell, check it against the published rows, write it."""
    groups_for = A.Groups()
    rows: list[dict] = []

    for model in MODELS:
        for line in LINES:
            for d in A.discover(model=model, cell_line=line, anchor="mean_control"):
                block = A.load_block(d)
                if block is None:
                    continue
                for layer in block.layers:
                    rows.extend(split_half_rows(block, int(layer)))
                print(f"[split_half] {model}/{line}: {len(rows)} rows", flush=True)

            cells = [A.load_block(p) for p in A.discover(model=model, cell_line=line)]
            cells = [b for b in cells
                     if b is not None and b.anchor_family == "individual_cells"]
            plan = A.heldout_plan(cells, groups_for) if cells else None
            if plan is None:
                print(f"[heldout] {model}/{line}: no split possible", flush=True)
                continue
            for layer in plan["layers"]:
                rows.extend(heldout_rows(cells, int(layer), plan))
            print(f"[heldout] {model}/{line}: {len(rows)} rows", flush=True)

    # Cross-check against what this run printed for itself. The split-half rows carry
    # the run-3 value inside `detail`; if the two disagree the rebuild is not reading
    # the ranks that produced the published top-1 and nothing here may be quoted.
    pub = pd.read_parquet(paths.summary_table())
    pub = pub[(pub.statistic == "retrieval_top1") & (pub.split == "split_half")]
    worst, n_checked = 0.0, 0
    for r in rows:
        if r["split"] != "split_half":
            continue
        hit = pub[(pub.model == r["model"]) & (pub.cell_line == r["cell_line"])
                  & (pub.layer == r["layer"]) & (pub.value_cp10k == r["value_cp10k"])]
        if len(hit) != 1:
            continue
        m = re.search(r"normalised rank ([0-9.eE+-]+)", hit.iloc[0]["detail"])
        if not m:
            continue
        worst = max(worst, abs(float(m.group(1)) - r["normalized_rank"]))
        worst = max(worst, abs(float(hit.iloc[0]["score"]) - r["top1"]))
        n_checked += 1
    print(f"[check] recomputed against published, {n_checked} split-half cells, "
          f"max abs diff {worst:.3g}", flush=True)
    if n_checked == 0 or worst > 5e-4:
        raise RuntimeError(
            f"the rebuilt ranks disagree with this run's own by {worst:.3g} over "
            f"{n_checked} split-half cells; they are not the ranks that produced the "
            "published top-1 and must not be emitted as though they were"
        )

    out_dir = Path(out_dir) if out_dir is not None else paths.summary_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "rank_scores.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[rank-summary] {len(rows)} rows -> {out}", flush=True)
    return {
        "n_rows": len(rows),
        "models": list(MODELS),
        "cell_lines": list(LINES),
        "split_half_cells_checked": n_checked,
        "max_abs_diff_against_published": worst,
        "chance_note": "chance on either definition is (n+1)/2n, about 0.5, not 1/n",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None,
                    help="where the tracked table goes (default: results/probe)")
    a = ap.parse_args(argv)
    import json
    print(json.dumps(build(a.out), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
