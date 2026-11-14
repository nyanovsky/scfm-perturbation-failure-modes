"""Generate the claims manifest, and re-read it to check the prose still holds.

Invocation::

    PYTHONPATH=src python -m scfm_probe.claims --write     # regenerate from results
    PYTHONPATH=src python -m scfm_probe.claims --check     # re-read and compare

Run 3's `claims_manifest.json` is the pattern: every number a report states carries
the file and the row it came from, and `verify_claims.py` re-reads them. This
module does the same job from the other end --- it *generates* the manifest by
querying `summary.parquet`, rather than asking an author to transcribe numbers
into JSON by hand.

That direction matters. A hand-written manifest records what someone believed the
number was when they wrote the sentence; a generated one cannot drift from the
table, because there is no transcription step to drift in. What a human still has
to supply is the *claim* --- the English sentence the number is doing work in ---
and a claim whose query returns no row is a failure here, not a blank.

The selectors are deliberately verbose. `model=scgpt, cell_line=hepg2, layer=12,
value_cp10k=0.0, statistic=retrieval_top1, split=split_half` is longer than a
row index and survives a re-run that changes the row order, which a row index
does not.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import config as C
from . import paths

#: (claim, selector) pairs. The selector is a dict of column -> value against
#: `summary.parquet`; it must match exactly one row, and a claim whose selector
#: matches none or several is an error rather than a gap.
CLAIMS: tuple[tuple[str, dict], ...] = (
    (
        "split-half retrieval of the perturbed gene from other genes' last-layer "
        "hidden states, scGPT, HepG2, zero-out dose",
        {"model": "scgpt", "cell_line": "hepg2", "layer": 12,
         "value_cp10k": 0.0, "statistic": "retrieval_top1", "split": "split_half"},
    ),
    (
        "the same at the first block, which is what makes it a depth statement "
        "rather than a last-layer one",
        {"model": "scgpt", "cell_line": "hepg2", "layer": 1,
         "value_cp10k": 0.0, "statistic": "retrieval_top1", "split": "split_half"},
    ),
    (
        "the post-embedding control: no edit can have reached another gene's token "
        "before the first block, so this must sit at its own null",
        {"model": "scgpt", "cell_line": "hepg2", "layer": C.EMBEDDING_LAYER,
         "value_cp10k": 0.0, "statistic": "retrieval_top1", "split": "split_half"},
    ),
    (
        "scPRINT, HepG2, last block: the second model's version of the same number",
        {"model": "scprint", "cell_line": "hepg2", "layer": 8,
         "value_cp10k": 0.0, "statistic": "retrieval_top1", "split": "split_half"},
    ),
    (
        "the magnitude control: identity decoded from the scalar norm alone, which "
        "is what would show the result is response size rather than direction",
        {"model": "scgpt", "cell_line": "hepg2", "layer": 12,
         "value_cp10k": 0.0, "statistic": "norm_only_top1", "split": "split_half"},
    ),
    (
        "the static-table control: how much of the hidden signature is the gene "
        "table echoed down the residual stream",
        {"model": "scgpt", "cell_line": "hepg2", "layer": 12,
         "value_cp10k": 0.0, "statistic": "cka_static_table", "split": "split_half"},
    ),
    (
        "the response-group control, reported beside the strict score and never "
        "instead of it",
        {"model": "scgpt", "cell_line": "hepg2", "layer": 12,
         "value_cp10k": 0.0, "statistic": "group_collapsed_top1", "split": "split_half"},
    ),
    (
        "propagation at the last block: how much of the edit left the perturbed "
        "gene's own token",
        {"model": "scgpt", "cell_line": "hepg2", "layer": 12,
         "value_cp10k": 0.0, "statistic": "propagation_ratio", "split": "self_position"},
    ),
    (
        "TranscriptFormer's substitute for that ratio, which has no denominator "
        "because the model never reads a gene's count at its own position",
        {"model": "transcriptformer", "cell_line": "hepg2", "layer": 12,
         "value_cp10k": 0.0, "statistic": "propagation_off_target_norm",
         "split": "self_position"},
    ),
)


def _select(df: pd.DataFrame, sel: dict) -> pd.Series:
    """The one row a selector names, or a loud failure saying what it matched."""
    m = np.ones(len(df), dtype=bool)
    for k, v in sel.items():
        col = df[k].to_numpy()
        m &= np.isclose(col, v) if isinstance(v, float) else (col == v)
    hit = df[m]
    if len(hit) != 1:
        raise RuntimeError(
            f"the selector {sel} matched {len(hit)} rows of {paths.summary_table()}; "
            "a claim must name exactly one row, or the number in the prose is not "
            "traceable to the table it came from"
        )
    return hit.iloc[0]


def build() -> dict:
    df = pd.read_parquet(paths.summary_table())
    out = []
    for claim, sel in CLAIMS:
        r = _select(df, sel)
        out.append({
            "claim": claim,
            "value": None if pd.isna(r.score) else float(r.score),
            "null_max": None if pd.isna(r.null_max) else float(r.null_max),
            "interval": [
                None if pd.isna(r.lo) else float(r.lo),
                None if pd.isna(r.hi) else float(r.hi),
            ],
            "n_perturbations": int(r.n_perturbations),
            "n_undefined": int(r.n_undefined),
            "detail": str(r.detail),
            "source": str(paths.summary_table().relative_to(paths.repo_dir())),
            "key": ", ".join(f"{k}={v}" for k, v in sel.items()),
        })
    return {
        "written_by": "src/scfm_probe/claims.py",
        "what": (
            "every number run 5's prose states, with the row of summary.parquet it "
            "came from. A value of null is a measured absence, not a missing claim: "
            "the detail string says why it could not be computed."
        ),
        "claims": out,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    path = paths.repo_dir() / "claims_manifest.json"

    fresh = build()
    if args.check:
        if not path.exists():
            raise RuntimeError(f"{path} does not exist; nothing to check the prose against")
        old = json.loads(path.read_text())
        drift = [
            (a["claim"], a["value"], b["value"])
            for a, b in zip(old["claims"], fresh["claims"])
            if a["value"] != b["value"]
        ]
        for claim, was, now in drift:
            print(f"[DRIFT] {claim}: manifest {was} vs table {now}", flush=True)
        print(f"[claims] {len(fresh['claims'])} claims, {len(drift)} drifted", flush=True)
        return 1 if drift else 0

    path.write_text(json.dumps(fresh, indent=2))
    for c in fresh["claims"]:
        band = "" if c["null_max"] is None else f" (null {c['null_max']:.4g})"
        print(f"[claims] {c['value']}{band}  <- {c['claim']}", flush=True)
    print(f"[claims] wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
