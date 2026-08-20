"""Reproduce the source paper's scGPT embedding baseline on its own benchmark.

The paper this run supplies a missing control for evaluates gene-pair features
from single-cell foundation models on BEELINE network inference: a projector is
fitted on one (network, dataset) cell and scored on the others by AUPRC over a
balanced pair set. Its scGPT panel reports an embedding baseline (Baseline-Emb)
trained on STRING/hESC.

This module re-runs that one arm, so that a disagreement about response features
on the perturbation task rests on having reproduced the setting first:

* the same processed benchmark files (GENELink's TFs+500 split, the version the
  paper's own code reads),
* the same scGPT human checkpoint, gene-token embedding table only,
* the paper's projector hyperparameters (two hidden layers 128 and 64, ReLU,
  sigmoid output, binary cross-entropy, Adam at 1e-3, batch 128, 50 epochs),
* the paper's 1:1 negative-to-positive sampling.

What is *not* reproduced: the virtual-value and gradient-trajectory features,
which would need a fresh extraction against the BEELINE datasets, and the mouse
datasets, whose symbols are not in the human checkpoint's vocabulary. The
comparison is therefore the embedding baseline on the three human hHEP cells.

Run inside a job (it downloads ~210 MB):

    uv run python -m scfm_pfm.beeline --out "$SILICO_EXPERIMENT_ARTIFACTS_DIR/beeline"
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

#: GENELink's processed BEELINE tree, the TFs+500 split the paper's code reads.
DATA_BASE = (
    "https://raw.githubusercontent.com/zpliulab/GeneLink_Plus/master/Benchmark%20Dataset"
)
#: scGPT whole-human checkpoint, the release the paper uses, mirrored on the hub.
SCGPT_REPO = "https://huggingface.co/wanglab/scGPT-human/resolve/main"

NETWORK_DIR = {
    "STRING": "STRING Dataset",
    "Non-Specific": "Non-Specific Dataset",
    "Specific": "Specific Dataset",
}
#: the projector is fitted here and scored everywhere else, as in the paper
TRAIN_CELL = ("STRING", "hESC")
EVAL_CELLS = [("STRING", "hHEP"), ("Non-Specific", "hHEP"), ("Specific", "hHEP")]
#: the paper's reported scGPT Baseline-Emb AUPRC for those cells (Table 1)
PAPER_BASELINE_EMB = {
    ("STRING", "hHEP"): 0.732,
    ("Non-Specific", "hHEP"): 0.586,
    ("Specific", "hHEP"): 0.556,
}
PAPER_OTHER_ARMS = {
    "Origin-Pert": {("STRING", "hHEP"): 0.496, ("Non-Specific", "hHEP"): 0.516,
                    ("Specific", "hHEP"): 0.518},
    "Origin-Attn": {("STRING", "hHEP"): 0.507, ("Non-Specific", "hHEP"): 0.512,
                    ("Specific", "hHEP"): 0.560},
    "GDT": {("STRING", "hHEP"): 0.906, ("Non-Specific", "hHEP"): 0.716,
            ("Specific", "hHEP"): 0.529},
}

SEED = 42
HIDDEN = (128, 64)
LR = 1e-3
BATCH = 128
EPOCHS = 50


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def _get(url: str, dest: Path, attempts: int = 4) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    last = ""
    for k in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "scfm-pfm3/1.0"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                dest.write_bytes(resp.read())
            return dest
        except Exception as exc:  # noqa: BLE001 - retried, then raised
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"{url}: {last}")


def cell_files(net: str, dataset: str, work: Path, config: str = "TFs+500") -> dict:
    """Label / TF / Target tables for one (network, dataset) benchmark cell."""
    out = {}
    for name in ("Label.csv", "TF.csv", "Target.csv"):
        rel = f"{NETWORK_DIR[net]}/{dataset}/{config}/{name}"
        url = f"{DATA_BASE}/{urllib.parse.quote(rel)}"
        dest = work / "beeline" / NETWORK_DIR[net] / dataset / config / name
        out[name] = pd.read_csv(_get(url, dest), index_col=0)
    return out


def pair_set(tables: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Positives from the ground-truth network, negatives sampled 1:1.

    Indices in Label.csv address the shared gene index that TF.csv and
    Target.csv both carry, so both columns map through the same table.
    """
    tf, target = tables["TF.csv"], tables["Target.csv"]
    name = dict(zip(target["index"].astype(int), target["Gene"].astype(str)))
    name.update(dict(zip(tf["index"].astype(int), tf["TF"].astype(str))))
    pos = tables["Label.csv"][["TF", "Target"]].astype(int)
    pos = pos[pos.TF != pos.Target]
    taken = set(map(tuple, pos.to_numpy()))
    tf_idx = tf["index"].astype(int).to_numpy()
    tgt_idx = target["index"].astype(int).to_numpy()
    neg: list[tuple[int, int]] = []
    guard = 0
    while len(neg) < pos.shape[0] and guard < 200 * pos.shape[0]:
        a = int(rng.choice(tf_idx))
        b = int(rng.choice(tgt_idx))
        guard += 1
        if a == b or (a, b) in taken:
            continue
        taken.add((a, b))
        neg.append((a, b))
    rows = [(name.get(a, ""), name.get(b, ""), 1) for a, b in pos.to_numpy()]
    rows += [(name.get(a, ""), name.get(b, ""), 0) for a, b in neg]
    return pd.DataFrame(rows, columns=["source", "target", "label"])


def scgpt_gene_embeddings(work: Path) -> tuple[dict, np.ndarray]:
    """The gene-token embedding table from the scGPT human checkpoint."""
    import torch

    raw = json.loads(_get(f"{SCGPT_REPO}/vocab.json", work / "scgpt" / "vocab.json").read_text())
    if isinstance(raw, dict) and all(isinstance(v, int) for v in list(raw.values())[:10]):
        vocab = {str(k): int(v) for k, v in raw.items()}
    elif isinstance(raw, dict) and "stoi" in raw:
        vocab = {str(k): int(v) for k, v in raw["stoi"].items()}
    else:
        raise RuntimeError(f"unrecognised vocab layout: {list(raw)[:10]}")
    ckpt = _get(f"{SCGPT_REPO}/best_model.pt", work / "scgpt" / "best_model.pt")
    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - older checkpoints are not weights-only
        print(f"[scgpt] weights_only load failed ({type(exc).__name__}), retrying", flush=True)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        state = state.state_dict()
    hits = {
        k: v for k, v in state.items()
        if hasattr(v, "shape") and v.ndim == 2 and int(v.shape[0]) == len(vocab)
    }
    if not hits:
        raise RuntimeError(
            f"no {len(vocab)}-row embedding in the checkpoint; keys: {list(state)[:20]}"
        )
    key = sorted(hits, key=lambda k: (0 if "encoder.embedding" in k else 1, k))[0]
    emb = hits[key].detach().float().numpy()
    print(f"[scgpt] embedding {key} {emb.shape}, vocab {len(vocab)}", flush=True)
    return vocab, emb


# --------------------------------------------------------------------------- #
# the projector, as the paper specifies it
# --------------------------------------------------------------------------- #
def features(
    pairs: pd.DataFrame, vocab: dict, emb: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict, pd.DataFrame]:
    keep = [s in vocab and t in vocab for s, t in zip(pairs.source, pairs.target)]
    kept = pairs[keep]
    a = emb[[vocab[s] for s in kept.source]]
    b = emb[[vocab[t] for t in kept.target]]
    x = np.concatenate([a, b], axis=1).astype(np.float32)
    y = kept.label.to_numpy().astype(np.float32)
    cover = {
        "n_pairs_in": int(pairs.shape[0]),
        "n_pairs_kept": int(kept.shape[0]),
        "fraction_kept": float(kept.shape[0] / max(pairs.shape[0], 1)),
        "positive_fraction": float(y.mean()) if y.size else float("nan"),
    }
    return x, y, cover, kept.reset_index(drop=True)


def train_projector(x: np.ndarray, y: np.ndarray, seed: int = SEED) -> dict:
    """UGRN's projector: 128/64 ReLU, sigmoid, BCE, Adam 1e-3, batch 128, 50 epochs."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    mu, sd = x.mean(0, keepdims=True), x.std(0, keepdims=True) + 1e-8
    net = nn.Sequential(
        nn.Linear(x.shape[1], HIDDEN[0]), nn.ReLU(),
        nn.Linear(HIDDEN[0], HIDDEN[1]), nn.ReLU(),
        nn.Linear(HIDDEN[1], 1),
    )
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    xt = torch.as_tensor((x - mu) / sd)
    yt = torch.as_tensor(y).unsqueeze(1)
    n = xt.shape[0]
    g = torch.Generator().manual_seed(seed)
    first = last = float("nan")
    for epoch in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        total = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i : i + BATCH]
            opt.zero_grad()
            loss = loss_fn(net(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += float(loss) * idx.numel()
        last = total / n
        first = last if epoch == 0 else first
    return {"net": net, "mu": mu, "sd": sd, "loss_first": first, "loss_last": last}


def predict(model: dict, x: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        z = model["net"](torch.as_tensor((x - model["mu"]) / model["sd"]))
    return 1.0 / (1.0 + np.exp(-z.numpy().ravel()))


# --------------------------------------------------------------------------- #
# metrics, written here so the job needs no extra dependency
# --------------------------------------------------------------------------- #
def auprc(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision, the same summary sklearn reports."""
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, y.size + 1)
    total = y.sum()
    return float((precision * y).sum() / total) if total else float("nan")


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=float)
    sorted_s = s[order]
    i = 0
    r = np.arange(1, s.size + 1, dtype=float)
    while i < s.size:  # average ranks within ties
        j = i
        while j + 1 < s.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = r[i : j + 1].mean()
        i = j + 1
    pos, neg = y.sum(), (1 - y).sum()
    if not pos or not neg:
        return float("nan")
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="./_beeline_work")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(args.work)
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    vocab, emb = scgpt_gene_embeddings(work)
    rng_emb = np.random.default_rng(SEED)
    shuffled = emb[rng_emb.permutation(emb.shape[0])]

    train_tables = cell_files(*TRAIN_CELL, work)
    train_pairs = pair_set(train_tables, rng)
    xtr, ytr, cover_tr, kept_tr = features(train_pairs, vocab, emb)
    print(f"[train] {TRAIN_CELL} {xtr.shape} kept {cover_tr['fraction_kept']:.3f}", flush=True)
    model = train_projector(xtr, ytr)
    xtr_s, ytr_s, _, _ = features(train_pairs, vocab, shuffled)
    model_shuf = train_projector(xtr_s, ytr_s)
    train_genes = set(kept_tr.source) | set(kept_tr.target)

    rows = []
    for cell in EVAL_CELLS:
        tables = cell_files(*cell, work)
        pairs = pair_set(tables, np.random.default_rng(SEED))
        x, y, cover, kept = features(pairs, vocab, emb)
        s = predict(model, x)
        xs, ys, _, _ = features(pairs, vocab, shuffled)
        ss = predict(model_shuf, xs)
        # The paper's claim is that the projector generalizes to genes it never
        # saw. Both genes of these pairs are outside the training pair set, so
        # nothing about them can have been memorised by identity.
        unseen = np.array(
            [(a not in train_genes) and (b not in train_genes)
             for a, b in zip(kept.source, kept.target)]
        )
        paper = PAPER_BASELINE_EMB[cell]
        rows.append({
            "network": cell[0],
            "dataset": cell[1],
            "auprc": auprc(y, s),
            "auroc": auroc(y, s),
            "auprc_shuffled_embedding": auprc(ys, ss),
            "paper_baseline_emb_auprc": paper,
            "difference_from_paper": auprc(y, s) - paper,
            "n_pairs_unseen_genes": int(unseen.sum()),
            "fraction_pairs_unseen_genes": float(unseen.mean()),
            "positive_fraction_unseen": float(y[unseen].mean()) if unseen.any() else float("nan"),
            "auprc_unseen_genes": auprc(y[unseen], s[unseen]) if unseen.sum() > 20 else float("nan"),
            "auprc_unseen_genes_shuffled_embedding": (
                auprc(ys[unseen], ss[unseen]) if unseen.sum() > 20 else float("nan")
            ),
            **cover,
        })
        print(
            f"[eval] {cell}: AUPRC {rows[-1]['auprc']:.4f} vs paper {paper}; "
            f"unseen-gene pairs {rows[-1]['n_pairs_unseen_genes']} "
            f"AUPRC {rows[-1]['auprc_unseen_genes']:.4f}",
            flush=True,
        )

    got = np.array([r["auprc"] for r in rows])
    ref = np.array([r["paper_baseline_emb_auprc"] for r in rows])
    payload = {
        "setting": (
            "UGRN leave-one-dataset-out: the projector is fitted on the STRING/hESC pair set "
            "and scored on held-out (network, dataset) cells, balanced 1:1, AUPRC"
        ),
        "arm_reproduced": "Baseline-Emb (scGPT gene embedding, concatenated over the pair)",
        "arms_not_reproduced": [
            "VVP and GDT: would need a fresh feature extraction against the BEELINE datasets",
            "the mouse datasets: their symbols are outside the human checkpoint's vocabulary",
        ],
            "train_cell": {
            "network": TRAIN_CELL[0], "dataset": TRAIN_CELL[1],
            "n_genes": len(train_genes), **cover_tr,
        },
        "projector": {
            "hidden": list(HIDDEN), "lr": LR, "batch_size": BATCH, "epochs": EPOCHS,
            "loss": "binary cross-entropy", "optimizer": "Adam", "seed": SEED,
            "train_loss_first_epoch": model["loss_first"], "train_loss_last_epoch": model["loss_last"],
        },
        "sources": {"benchmark": DATA_BASE, "checkpoint": SCGPT_REPO},
        "rows": rows,
        "summary": {
            "mean_auprc": float(got.mean()),
            "paper_mean_auprc": float(ref.mean()),
            "mean_difference": float((got - ref).mean()),
            "max_abs_difference": float(np.abs(got - ref).max()),
            "n_cells": len(rows),
            "shuffled_embedding_mean_auprc": float(
                np.mean([r["auprc_shuffled_embedding"] for r in rows])
            ),
            "mean_auprc_unseen_genes": float(
                np.nanmean([r["auprc_unseen_genes"] for r in rows])
            ),
            "shuffled_embedding_mean_auprc_unseen_genes": float(
                np.nanmean([r["auprc_unseen_genes_shuffled_embedding"] for r in rows])
            ),
            "mean_fraction_pairs_unseen_genes": float(
                np.mean([r["fraction_pairs_unseen_genes"] for r in rows])
            ),
        },
        "paper_other_arms": {
            k: {f"{n}/{d}": v for (n, d), v in vals.items()}
            for k, vals in PAPER_OTHER_ARMS.items()
        },
        "seconds": round(time.time() - t0, 1),
    }
    (out / "beeline_reproduction.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2))
    print(f"-> {out / 'beeline_reproduction.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
