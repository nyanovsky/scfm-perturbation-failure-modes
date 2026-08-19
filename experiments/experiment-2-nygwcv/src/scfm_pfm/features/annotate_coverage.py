"""Add the coverage record to a block that was extracted before it existed.

The two AIDO.Cell blocks stay at 250 perturbed genes, so they are not re-extracted;
but a block whose manifest does not say what fraction of the in-scope set it
covers is exactly the hazard the coverage record exists to remove. This runs as a
CPU job that mounts the extraction job's volume, recomputes the record from the
same tables and the same selector, and writes the annotated manifest --- keeping
the original beside it --- plus a fresh verification bundle so the report reads
coverage from every block.

Nothing about the features changes. The check that this is true is here: the
selector is re-run and its output compared with the perturbed table the block
shipped, row for row.

    python -m scfm_pfm.features.annotate_coverage --block-dir auto \\
        --out "$SILICO_EXPERIMENT_ARTIFACTS_DIR" --bundles <dir>
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from .. import config as C
from .. import paths
from .. import verify_features as vf
from .scope import coverage_record, select_perturbed


def find_block(mount: Path) -> Path:
    hits = sorted(mount.glob("**/features/*/*/manifest.json"))
    if len(hits) != 1:
        raise SystemExit(f"expected one feature block in {mount}, found {len(hits)}: {hits[:4]}")
    return hits[0].parent


def annotate(block_dir: Path, out_root: Path, bundles_dir: Path, manifest_only: bool = False) -> dict:
    manifest = json.loads((block_dir / "manifest.json").read_text())
    model, cell_line = manifest["model"], manifest["cell_line"]
    scope_rec = manifest.get("scope", {})
    scope = C.Scope(
        n_perturbed=int(scope_rec.get("n_perturbed", C.SCOPE.n_perturbed)),
        n_perturbed_cells_anchor=int(
            scope_rec.get("n_perturbed_cells_anchor", C.SCOPE.n_perturbed_cells_anchor)
        ),
        n_perturbed_masked=int(scope_rec.get("n_perturbed_masked", C.SCOPE.n_perturbed_masked)),
    )
    readout = pd.read_parquet(
        paths.handoff_root() / "readout" / f"readout_genes_{cell_line}.parquet"
    )
    perts = pd.read_parquet(paths.effects_dir(cell_line) / "perturbations.parquet")
    shipped = pd.read_parquet(block_dir / "perturbed_genes.parquet")
    rebuilt = select_perturbed(perts, readout, scope)
    identical = bool(shipped.equals(rebuilt))
    print(
        f"[annotate] {model}/{cell_line}: selector reproduces the shipped perturbed table: "
        f"{identical} ({shipped.shape} vs {rebuilt.shape})",
        flush=True,
    )
    if not identical:
        raise SystemExit(
            "the selector no longer reproduces this block's perturbed genes; the coverage record "
            "would not describe the block that shipped"
        )
    cov = coverage_record(perts, readout, scope, shipped)
    cov["annotated_after_extraction"] = (
        "added by features.annotate_coverage: the block was extracted before the coverage record "
        "existed. The selector was re-run and reproduces this block's perturbed table exactly, so "
        "the record describes the features as shipped; no array changed."
    )
    manifest["coverage"] = cov
    print(f"[annotate] coverage {json.dumps(cov)[:400]}", flush=True)

    dest = paths.ensure(Path(out_root) / "features" / model / cell_line)
    (dest / "manifest_run2_original.json").write_text(
        json.dumps(json.loads((block_dir / "manifest.json").read_text()), indent=2)
    )
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[annotate] wrote {dest / 'manifest.json'}", flush=True)

    if manifest_only:
        # a superseded block: its arrays are not mounted here and no bundle reads it, so the
        # record is written from the manifest and the perturbed table alone. The selector check
        # above is the one that matters --- it is what ties the record to the shipped rows.
        print(f"[annotate] manifest only, no bundle written for {model}/{cell_line}", flush=True)
        return {"model": model, "cell_line": cell_line, "manifest": manifest}

    # a shadow block: the annotated manifest over the mounted arrays, so the bundle the
    # report reads carries coverage without copying gigabytes
    shadow = paths.ensure(Path("/tmp") / f"block_{model}_{cell_line}")
    (shadow / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    for name in ("readout_genes.parquet", "perturbed_genes.parquet", "step_sweep.parquet",
                 "step_selection.json", "gene_embeddings.npz", "environment.json"):
        src = block_dir / name
        if src.exists() and not (shadow / name).exists():
            os.symlink(src, shadow / name)
    for adir in sorted(block_dir.glob("anchor=*")):
        link = shadow / adir.name
        if not link.exists():
            os.symlink(adir, link)
    ch = vf.Checks()
    bundle = vf.verify_block(shadow, ch)
    bundles_dir = paths.ensure(Path(bundles_dir))
    name = f"report_bundle_{model}_{cell_line}.json"
    (bundles_dir / name).write_text(json.dumps(bundle, indent=2, default=str))
    (bundles_dir / f"verification_{model}_{cell_line}.json").write_text(
        json.dumps({"checks": ch.rows}, indent=2)
    )
    ok, total = ch.summary()
    print(f"[annotate] {model}/{cell_line}: {ok}/{total} checks passed; wrote {name}", flush=True)
    return bundle


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--block-dir", default="auto")
    ap.add_argument("--out", default=None)
    ap.add_argument("--bundles", default=None)
    ap.add_argument(
        "--manifest-only",
        action="store_true",
        help="patch the manifest without re-verifying the arrays: for a superseded block whose "
        "npz files are not mounted here",
    )
    args = ap.parse_args(argv)

    if args.block_dir == "auto":
        mount = Path(os.environ.get("SILICO_REUSED_VOLUME_DIR", ""))
        block = find_block(mount)
        print(f"[annotate] block {block}", flush=True)
    else:
        block = Path(args.block_dir)
    out_root = Path(args.out) if args.out else paths.data_root()
    bundles = Path(args.bundles) if args.bundles else paths.report_dir() / "bundles_annotated"
    annotate(block, out_root, bundles, manifest_only=args.manifest_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
