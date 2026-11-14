"""How close is each model's gene table to the raw protein table?

The provenance audit says where a table came from. This says how much of it
survives: two tables over the same genes are compared with linear CKA, which is
invariant to rotation and to overall scale and therefore does not care that the
tables have different widths, and with a rank correlation between their
gene-gene cosine similarities, which is the same question asked without any
linear-algebra assumption. A random table of matched width is included so the
numbers have a floor, and every pair of tables is reported so the single-cell
models can be read against each other as well as against ESM2.

    python -m scfm_pfm.protein.identity --tables results/protein --bundle _root/bundle
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA between two gene-by-feature tables over the same genes."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    xty = x.T @ y
    num = float((xty * xty).sum())
    xtx = x.T @ x
    yty = y.T @ y
    den = float(np.sqrt((xtx * xtx).sum()) * np.sqrt((yty * yty).sum()))
    return num / den if den > 0 else float("nan")


def _cosine_offdiag(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64)[idx]
    n = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / np.maximum(n, 1e-30)
    s = v @ v.T
    iu = np.triu_indices(s.shape[0], k=1)
    return s[iu]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / den) if den > 0 else float("nan")


def rsa(x: np.ndarray, y: np.ndarray, idx: np.ndarray) -> float:
    return _spearman(_cosine_offdiag(x, idx), _cosine_offdiag(y, idx))


def compare_line(line_dir: Path, n_rsa_genes: int = 1500, seed: int = 42) -> dict:
    tables: dict[str, np.ndarray] = {}
    for p in sorted(line_dir.glob("emb__*.npz")):
        tables[p.stem.split("__", 1)[1]] = np.load(p, allow_pickle=False)["embedding"].astype(
            np.float64
        )
    if not tables:
        return {"error": f"no emb__*.npz in {line_dir}"}
    n = next(iter(tables.values())).shape[0]
    nonzero = np.ones(n, dtype=bool)
    for arr in tables.values():
        nonzero &= np.linalg.norm(arr, axis=1) > 0
    rng = np.random.default_rng(seed)
    for width in sorted({int(a.shape[1]) for a in tables.values()}):
        tables[f"random_d{width}"] = rng.standard_normal((n, width))
    keep = np.flatnonzero(nonzero)
    pick = keep if keep.size <= n_rsa_genes else rng.choice(keep, n_rsa_genes, replace=False)
    pick = np.sort(pick)

    pairs = {}
    for a, b in itertools.combinations(sorted(tables), 2):
        pairs[f"{a}|{b}"] = {
            "cka": round(linear_cka(tables[a][keep], tables[b][keep]), 6),
            "rsa_spearman": round(rsa(tables[a], tables[b], pick), 6),
        }
    return {
        "n_genes_total": int(n),
        "n_genes_compared": int(keep.size),
        "n_genes_rsa": int(pick.size),
        "widths": {k: int(v.shape[1]) for k, v in sorted(tables.items())},
        "pairs": pairs,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables", default="results/protein")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--lines", default="hepg2,jurkat")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    bundle = Path(args.bundle)
    record = {"written_by": "scfm_pfm.protein.identity", "lines": {}}
    for line in (x.strip() for x in args.lines.split(",") if x.strip()):
        record["lines"][line] = compare_line(bundle / line)

    # does the shipped table look like the truncated recipe or the full-length one?
    tdir = Path(args.tables)
    variants = {}
    for p in sorted(tdir.glob("esm2_*.npz")):
        z = np.load(p, allow_pickle=False)
        if "embedding_truncated" not in z.files:
            continue
        mapped = z["mapped"].astype(bool) if "mapped" in z.files else slice(None)
        full = z["embedding"][mapped].astype(np.float64)
        trunc = z["embedding_truncated"][mapped].astype(np.float64)
        variants[p.stem] = {
            "cka_full_vs_truncated": round(linear_cka(full, trunc), 6),
            "rows_differing": int((np.abs(full - trunc).max(axis=1) > 1e-6).sum()),
        }
    record["esm2_variants"] = variants

    payload = json.dumps(record, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(payload)
    print(payload, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
