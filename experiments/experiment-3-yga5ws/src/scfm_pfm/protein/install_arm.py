"""Put a gene-indexed table into a bundle as one more embedding arm.

``LineBundle`` reads every ``emb__<model>.npz`` in a cell line's directory and
offers it to the arm resolver, so an arm is added by writing one file in the
read-out order that line's targets define. A gene with no protein gets a zero
row, which the report has to state alongside the score: the arm is scored on
every read-out column, and the columns it knows nothing about are the ones that
did not map.

    python -m scfm_pfm.protein.install_arm --tables results/protein --bundle _root/bundle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict[str, np.ndarray]:
    """Read every array, ``allow_pickle=False`` first.

    ``pandas`` hands back ``dtype=object`` for a string column and ``np.savez``
    stores that as a pickled array, which the safe loader refuses -- lazily, when
    the key is read rather than when the file is opened, so the fallback has to
    wrap the read. The tables are written with a unicode dtype now; this keeps a
    table written before that readable instead of recomputing it on a GPU.
    """
    for allow in (False, True):
        z = np.load(path, allow_pickle=allow)
        try:
            return {k: z[k] for k in z.files}
        except ValueError:
            if allow:
                raise
            print(f"[install] {path.name}: object-dtype column, reading with allow_pickle",
                  flush=True)
    raise RuntimeError(f"unreadable table {path}")  # pragma: no cover


def install(table_path: Path, bundle: Path, lines: list[str], name: str,
            variant: str = "embedding") -> dict:
    z = _load(table_path)
    emb = z[variant].astype(np.float32)
    gene_id = z["gene_id"].astype(str)
    mapped = z["mapped"].astype(bool) if "mapped" in z else np.ones(len(gene_id), bool)
    pos = {g: i for i, g in enumerate(gene_id)}
    report = {"table": str(table_path), "variant": variant, "arm": name,
              "dim": int(emb.shape[1]), "lines": {}}
    for line in lines:
        t = np.load(bundle / line / "targets.npz", allow_pickle=False)
        readout = t["readout_gene_id"].astype(str)
        idx = np.array([pos.get(g, -1) for g in readout])
        out = np.zeros((readout.size, emb.shape[1]), dtype=np.float32)
        present = idx >= 0
        out[present] = emb[idx[present]]
        covered = present & np.array([mapped[i] if i >= 0 else False for i in idx])
        dest = bundle / line / f"emb__{name}.npz"
        np.savez_compressed(dest, embedding=out)
        report["lines"][line] = {
            "n_readout": int(readout.size),
            "in_table": int(present.sum()),
            "with_protein": int(covered.sum()),
            "zero_rows": int((np.linalg.norm(out, axis=1) == 0).sum()),
            "fraction_with_protein": float(covered.mean()),
            "path": str(dest),
        }
        print(f"[install] {line}: {name} {out.shape}, "
              f"{covered.sum()}/{readout.size} genes with a protein", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables", default="results/protein")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--lines", default="hepg2,jurkat")
    ap.add_argument("--names", nargs="*", default=None,
                    help="table stems to install; default every esm2_*.npz present")
    ap.add_argument("--variant", default="embedding")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    tables = Path(args.tables)
    stems = args.names or sorted(p.stem for p in tables.glob("esm2_*.npz"))
    lines = [x.strip() for x in args.lines.split(",") if x.strip()]
    reports = [
        install(tables / f"{stem}.npz", Path(args.bundle), lines, stem, args.variant)
        for stem in stems
    ]
    payload = {"written_by": "scfm_pfm.protein.install_arm", "installed": reports}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
