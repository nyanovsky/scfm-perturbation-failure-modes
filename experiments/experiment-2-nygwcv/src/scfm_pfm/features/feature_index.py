"""One small table that names the canonical block for each (model, cell line).

Coverage now differs by model: scGPT, scPRINT and TranscriptFormer cover every
in-scope perturbation, AIDO.Cell covers 250 of them. Two blocks at different
coverage share the smaller one's perturbed genes but not its row order, so a
comparison run that globs the feature tree can silently mix them. This index is
the thing to read instead: one row per (model, cell line), naming the block to
use, what it covers, and what it supersedes.

    PYTHONPATH=src python -m scfm_pfm.features.feature_index \\
        --bundles results/report/bundles --out results/feature_index
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .. import config as C
from .. import paths

FULL_TREE = "features_full_scope"
SUBSET_TREE = "features"


def build(bundles_dir: Path) -> dict:
    rows = []
    for path in sorted(Path(bundles_dir).glob("report_bundle*.json")):
        b = json.loads(path.read_text())
        m = b["manifest"]
        cov = m.get("coverage") or {}
        model, cl = b["model"], b["cell_line"]
        full = cov.get("mode") == "full"
        tree = FULL_TREE if full else SUBSET_TREE
        anchors = sorted(m.get("anchors", {}))
        rows.append(
            {
                "model": model,
                "cell_line": cl,
                "block": f"{tree}/{model}/{cl}",
                "coverage_mode": cov.get("mode", "unrecorded"),
                "n_perturbed": int(b["n_perturbed"]),
                "n_in_scope": cov.get("n_in_scope"),
                "fraction_of_scope": cov.get("fraction_of_scope"),
                "n_readout": int(b["n_readout"]),
                "anchors": ",".join(anchors),
                "has_masked": bool(m.get("masked", {}).get("ran")),
                "n_perturbed_masked": m.get("masked", {}).get("n_perturbed"),
                "supersedes": f"{SUBSET_TREE}/{model}/{cl}" if full else None,
                "hidden_readout": (
                    f"{SUBSET_TREE}/{model}/{cl}/hidden_readout"
                    if m.get("masked", {}).get("ran")
                    else None
                ),
                "gdt_source": sorted(
                    {meta.get("gdt_source", "") for meta in m.get("anchors", {}).values()}
                )[0]
                if m.get("anchors")
                else None,
            }
        )
    coverages = sorted({(r["coverage_mode"], r["n_perturbed"]) for r in rows})
    return {
        "written_by": "scfm_pfm.features.feature_index",
        "what": "the canonical response-feature block per (model, cell line) for this thread",
        "how_to_read": (
            "paths are relative to this experiment's artifact root. Matrix rows follow the "
            "block's perturbed_genes.parquet and columns follow readout_genes.parquet; join two "
            "blocks on perturbed_gene_id, never on row index, because blocks at different "
            "coverage do not share a row order."
        ),
        "coverage_mix": [{"mode": m, "n_perturbed": n} for m, n in coverages],
        "coverage_warning": (
            "this tree mixes coverages on purpose: the three cheap models cover every in-scope "
            "perturbation and AIDO.Cell covers a subset of them, because its measured cost per "
            "forward pass makes full coverage tens of GPU hours. Do not compare a per-block "
            "aggregate across two different coverages; restrict to the shared perturbed genes "
            "first."
        ),
        "strata": list(C.PERTURBED_STRATA),
        "blocks": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundles", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    repo = paths.repo_dir()
    bundles = Path(args.bundles) if args.bundles else repo / "results" / "report" / "bundles"
    out = paths.ensure(Path(args.out) if args.out else repo / "results" / "feature_index")
    index = build(bundles)
    (out / "blocks.json").write_text(json.dumps(index, indent=2, default=str))
    pd.DataFrame(index["blocks"]).to_parquet(out / "blocks.parquet", index=False)
    print(f"[index] {len(index['blocks'])} blocks -> {out / 'blocks.json'}", flush=True)
    for r in index["blocks"]:
        print(
            f"  {r['model']:17s} {r['cell_line']:7s} {r['block']:45s} "
            f"{r['coverage_mode']:8s} {r['n_perturbed']} of {r['n_in_scope']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
