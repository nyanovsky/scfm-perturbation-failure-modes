"""One vector per gene, straight from a public ESM2 checkpoint.

No single-cell model is anywhere in this path: the gene's canonical human
protein sequence goes into ESM2 and the mean of the final-layer residue
representations comes out. That is the same object the two ESM-derived
single-cell models start from, so putting it through the identical translator
says whether their advantage is theirs or their initialisation's.

Sequences longer than ESM2's 1,022-residue context are cut into consecutive
chunks and the mean is taken over every residue of every chunk, so no part of a
long protein is silently dropped. The mean over the first chunk alone is written
alongside it, because a table built by plain truncation is what some published
gene-embedding files contain and the comparison should be able to tell.

    python -m scfm_pfm.protein.esm_embed --genes results/protein --out results/protein
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import sources as S


def embed_sequences(
    repo: str,
    sequences: list[str],
    device: str = "cuda",
    max_residues: int = S.ESM2_MAX_RESIDUES,
    batch_tokens: int = 16384,
    dtype: torch.dtype = torch.float32,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Mean-pooled final-layer embeddings: full length, and first chunk only."""
    from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(repo)
    try:
        model = AutoModel.from_pretrained(repo, add_pooling_layer=False, dtype=dtype)
    except TypeError:  # the pooler is unused either way; only its presence differs
        model = AutoModel.from_pretrained(repo, dtype=dtype)
    model = model.eval().to(device)
    dim = int(model.config.hidden_size)

    # one work item per chunk, longest first so padding waste is small
    items: list[tuple[int, int, str]] = []
    for i, seq in enumerate(sequences):
        if not seq:
            continue
        pieces = [seq[k: k + max_residues] for k in range(0, len(seq), max_residues)]
        items += [(i, j, p) for j, p in enumerate(pieces)]
    items.sort(key=lambda x: -len(x[2]))

    total = np.zeros((len(sequences), dim), dtype=np.float64)
    counts = np.zeros(len(sequences), dtype=np.float64)
    first = np.zeros((len(sequences), dim), dtype=np.float64)
    have_first = np.zeros(len(sequences), dtype=bool)

    started = time.time()
    done = 0
    next_report = 0
    batch: list[tuple[int, int, str]] = []
    special = {tok.cls_token_id, tok.eos_token_id, tok.pad_token_id}

    def flush(batch):
        if not batch:
            return
        enc = tok([b[2] for b in batch], return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc).last_hidden_state
        mask = enc["attention_mask"].bool()
        for sid in special:
            if sid is not None:
                mask &= enc["input_ids"] != sid
        m = mask.unsqueeze(-1).to(out.dtype)
        summed = (out * m).sum(dim=1).double().cpu().numpy()
        n_res = mask.sum(dim=1).double().cpu().numpy()
        for row, (gene, piece, _) in enumerate(batch):
            total[gene] += summed[row]
            counts[gene] += n_res[row]
            if piece == 0:
                first[gene] = summed[row] / max(n_res[row], 1.0)
                have_first[gene] = True

    for item in items:
        batch.append(item)
        width = max(len(b[2]) for b in batch) + 2
        if width * len(batch) >= batch_tokens:
            flush(batch)
            done += len(batch)
            batch = []
            if done >= next_report:
                print(f"  [{repo}] {done}/{len(items)} chunks "
                      f"({time.time() - started:.0f}s)", flush=True)
                next_report = done + 500
    flush(batch)
    done += len(batch)

    mean = np.zeros_like(total)
    ok = counts > 0
    mean[ok] = total[ok] / counts[ok][:, None]
    meta = {
        "repo": repo,
        "hidden_size": dim,
        "n_sequences": int(ok.sum()),
        "n_chunks": len(items),
        "max_residues_per_chunk": max_residues,
        "dtype": str(dtype).replace("torch.", ""),
        "seconds": round(time.time() - started, 1),
        "residues_embedded": float(counts.sum()),
    }
    return mean.astype(np.float32), np.where(have_first[:, None], first, mean).astype(np.float32), meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--genes", default="results/protein", help="dir with gene_to_protein.parquet")
    ap.add_argument("--out", default="results/protein")
    ap.add_argument("--models", nargs="*", default=list(S.ESM2))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-tokens", type=int, default=16384)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(Path(args.genes) / "gene_to_protein.parquet")
    if args.limit:
        table = table.head(args.limit)
    seqs = table["sequence"].fillna("").astype(str).tolist()
    mapped = np.array([bool(s) for s in seqs])
    print(f"[esm] {len(seqs)} genes, {int(mapped.sum())} with a protein sequence, "
          f"device={args.device}", flush=True)

    report = {"written_by": "scfm_pfm.protein.esm_embed", "n_genes": len(seqs),
              "n_mapped": int(mapped.sum()), "models": {}}
    for name in args.models:
        spec = S.ESM2[name]
        emb, trunc, meta = embed_sequences(
            spec["repo"], seqs, device=args.device, batch_tokens=args.batch_tokens
        )
        if emb.shape[1] != spec["dim"]:
            raise RuntimeError(f"{name}: width {emb.shape[1]} != declared {spec['dim']}")
        emb[~mapped] = 0.0
        trunc[~mapped] = 0.0
        np.savez_compressed(
            out / f"{name}.npz",
            embedding=emb,
            embedding_truncated=trunc,
            mapped=mapped,
            # a unicode dtype, not pandas' object dtype: np.savez pickles an
            # object array and the safe loader then refuses to read it back
            gene_id=np.asarray(table["gene_id"].astype(str).tolist(), dtype=np.str_),
            gene_name=np.asarray(table["gene_name"].astype(str).tolist(), dtype=np.str_),
        )
        norms = np.linalg.norm(emb[mapped], axis=1)
        meta.update({
            "row_norm_mean": float(norms.mean()),
            "row_norm_sd": float(norms.std()),
            "duplicate_rows": int(int(mapped.sum()) - len(np.unique(emb[mapped], axis=0))),
            "max_abs_full_vs_truncated": float(np.abs(emb - trunc).max()),
            "genes_where_truncation_matters": int(
                (np.abs(emb - trunc).max(axis=1) > 1e-6).sum()
            ),
        })
        report["models"][name] = meta
        print(f"[esm] {name}: {emb.shape} in {meta['seconds']}s", flush=True)
    (out / "esm_embed_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
