"""Exercise one model's query interface end to end, small and loud.

Loads the checkpoint, configures the read-out set, runs a batch at the mean
control anchor, checks the mask path, the exact derivative and the embedding
table, and times a realistic batch so the production sweep can be sized from a
measured per-forward cost rather than a guess.

    PYTHONPATH=src python3 scratch/smoke_query.py --model scgpt --cache /ckpt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scfm_pfm import config as C
from scfm_pfm import paths
from scfm_pfm.models import queries as Q


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache", default="/ckpt")
    ap.add_argument("--cell-line", default="hepg2")
    ap.add_argument("--supplied-mode", default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--n-time", type=int, default=16)
    args = ap.parse_args()

    universe = pd.read_parquet(paths.universe_dir() / "gene_universe.parquet")
    vocab = pd.read_parquet(paths.universe_dir() / f"vocabulary_{args.model}.parquet")
    readout = pd.read_parquet(
        paths.handoff_root() / "readout" / f"readout_genes_{args.cell_line}.parquet"
    )
    anchor = readout["pooled_control_cp10k"].to_numpy(dtype=np.float64)
    print(f"[{args.model}] read-out genes {readout.shape[0]}, anchor median "
          f"{np.median(anchor):.4f} CP10K", flush=True)

    t0 = time.time()
    model = Q.build(args.model, Path(args.cache), universe, vocab, device="cuda")
    model.load()
    load_s = time.time() - t0
    if args.batch:
        model.batch_size = args.batch
    kwargs = {"supplied_mode": args.supplied_mode} if args.supplied_mode else {}
    layout = model.configure(readout, anchor, args.cell_line, **kwargs)
    print(f"[{args.model}] loaded in {load_s:.1f}s; layout "
          f"{json.dumps(layout.as_dict(), default=str)[:600]}", flush=True)

    x0 = model.to_native(anchor)
    print(f"[{args.model}] native anchor: min {x0.min():.4f} median "
          f"{np.median(x0):.4f} max {x0.max():.4f}", flush=True)

    # one query: baseline, zero-out and two virtual values on one perturbed gene
    col = int(np.argsort(-anchor)[10])
    grid = np.array(C.VALUE_GRID_CP10K, dtype=np.float64)
    batch = np.repeat(x0[None, :], grid.size + 1, axis=0)
    for i, v in enumerate(grid):
        batch[i + 1, col] = float(model.to_native(np.array([v]))[0])
    t0 = time.time()
    resp = model.forward_batch(batch, perturbed_col=col)
    fwd_s = time.time() - t0
    finite = bool(np.isfinite(resp).all())
    base = resp[0]
    diff = resp[1:] - base[None, :]
    print(
        f"[{args.model}] responses {resp.shape} finite={finite} in {fwd_s:.2f}s "
        f"({fwd_s / batch.shape[0]:.3f}s per query)",
        flush=True,
    )
    print(
        f"[{args.model}] baseline response: min {base.min():.5g} median "
        f"{np.median(base):.5g} max {base.max():.5g}",
        flush=True,
    )
    for i, v in enumerate(grid):
        d = diff[i]
        print(
            f"    v={v:>6g} CP10K: self {d[col]:+.5g}  |d| median {np.median(np.abs(d)):.3g} "
            f"max {np.abs(d).max():.3g}  cross-gene mean {d.mean():+.3g}",
            flush=True,
        )

    # timing at a realistic batch shape
    t0 = time.time()
    model.forward_batch(np.repeat(x0[None, :], args.n_time, axis=0), perturbed_col=col)
    per = (time.time() - t0) / args.n_time
    print(f"[{args.model}] {args.n_time}-row batch: {per:.3f}s per query", flush=True)

    # mask path
    spec = model.mask_spec()
    if spec is None:
        print(f"[{args.model}] no masked variant (by architecture)", flush=True)
    else:
        blocks = Q.partition_masks(model.n_readout, float(spec["rate"]), 1, C.SEED)
        m = np.zeros((2, model.n_readout), dtype=bool)
        m[0, blocks[0]] = True
        m[1, blocks[1]] = True
        t0 = time.time()
        masked = model.forward_batch(np.repeat(x0[None, :], 2, axis=0), perturbed_col=col, mask=m)
        print(
            f"[{args.model}] mask {spec['kind']} rate {spec['rate']} -> {len(blocks)} blocks; "
            f"response finite={bool(np.isfinite(masked).all())} in {time.time() - t0:.2f}s; "
            f"masked-vs-unmasked median |delta| "
            f"{np.median(np.abs(masked[0][blocks[0]] - base[blocks[0]])):.4g}",
            flush=True,
        )

    # exact derivative
    t0 = time.time()
    jvp = model.exact_jvp(x0, col)
    if jvp is None:
        print(f"[{args.model}] exact forward-mode derivative unavailable: "
              f"{getattr(model, '_jvp_error', 'not implemented for this layout')}", flush=True)
    else:
        h = 0.1
        xp, xm = x0.copy(), x0.copy()
        xp[col] += h
        xm[col] -= h
        fd = (model.forward_batch(xp[None, :], perturbed_col=col)[0]
              - model.forward_batch(xm[None, :], perturbed_col=col)[0]) / (2 * h)
        ok = np.isfinite(jvp) & np.isfinite(fd)
        r = float(np.corrcoef(jvp[ok], fd[ok])[0, 1])
        print(
            f"[{args.model}] exact jvp in {time.time() - t0:.2f}s; vs central difference "
            f"at h={h}: pearson {r:.6f}, relative L2 "
            f"{2 * np.linalg.norm(jvp - fd) / (np.linalg.norm(jvp) + np.linalg.norm(fd)):.4g}",
            flush=True,
        )

    emb, source = model.gene_embeddings()
    print(f"[{args.model}] gene embeddings {emb.shape} from {source}", flush=True)
    print(f"[{args.model}] SMOKE OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
