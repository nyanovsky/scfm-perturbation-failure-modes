"""Stage 1, second pass: compare each table against the file it came from.

The first pass established what the tables are and, for scPRINT, that the
optimizer carried Adam moments for its gene embedding. Two comparisons need the
initialisation itself rather than the checkpoint's structure:

* TranscriptFormer ships a 2,560-wide gene table inside ``model_weights.pt`` and
  the ESM2 mapping file it was built from beside it in ``vocabs/``. If the two
  are the same numbers, pretraining did not move the table.
* scPRINT initialised its table from a precomputed embedding file. If that file
  is still published, the distance between it and the checkpoint says how far
  400,000 training steps actually moved a table that was allowed to move.

It reads the checkpoints from the first pass's volume rather than downloading
them again.

    python -m scfm_pfm.protein.audit_tables --cache <mount>/_ckpt --out results/provenance
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..data.sources import download
from . import ckptread
from .audit import _compare, _fetch_text, _grep, _matrix_summary, load_matrix_file
from . import sources as S


def find_cache(start: Path, name: str = "_ckpt", max_depth: int = 8) -> Path | None:
    frontier = [(start, 0)]
    while frontier:
        here, depth = frontier.pop(0)
        cand = here / name
        if cand.is_dir():
            return cand
        if depth >= max_depth:
            continue
        try:
            frontier += [(c, depth + 1) for c in sorted(here.iterdir())
                         if c.is_dir() and not c.name.startswith(".")]
        except (PermissionError, OSError):
            continue
    return None


def h5_datasets(path: Path) -> list[dict]:
    """Every dataset in an HDF5 file, at any depth."""
    import h5py  # noqa: PLC0415

    found: list[dict] = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            found.append({"name": name, "shape": list(obj.shape), "dtype": str(obj.dtype)})

    with h5py.File(path, "r") as h:
        h.visititems(visit)
    return found


def h5_read(path: Path, name: str) -> np.ndarray:
    import h5py  # noqa: PLC0415

    with h5py.File(path, "r") as h:
        return np.asarray(h[name][:])


def transcriptformer_shipped_table(h5: Path) -> tuple[np.ndarray, list[str]]:
    """The embedding matrix TranscriptFormer builds at load time, rebuilt here.

    ``tokenizer/vocab.py::construct_gene_embeddings`` reads the mapping file into
    a dict, drops duplicate gene names keeping the first, and concatenates seven
    special-token rows drawn from ``RandomState(42).rand`` in front of the genes.
    Reproducing that exactly makes the comparison against the checkpoint a test
    of every row rather than of an alignment guess.
    """
    import h5py  # noqa: PLC0415

    with h5py.File(h5, "r") as f:
        keys = [k.decode("utf-8") if isinstance(k, bytes) else str(k) for k in f["keys"][:]]
        seen: set[str] = set()
        names, rows = [], []
        for k in keys:
            if k in seen:
                continue
            seen.add(k)
            names.append(k)
            rows.append(np.asarray(f[f"arrays/{k}"][:]))
    genes = np.asarray(rows, dtype=np.float64)
    specials = np.random.RandomState(S.TF_SPECIAL_TOKEN_SEED).rand(7, genes.shape[1])
    return np.concatenate([specials, genes], axis=0), names


def transcriptformer_tables(cache: Path) -> dict:
    root = next((p.parent for p in (cache / "transcriptformer").rglob("config.json")), None)
    if root is None:
        return {"error": f"no config.json under {cache / 'transcriptformer'}"}
    h5 = next((p for p in root.rglob("*.h5")), None)
    sd, reader = ckptread.torch_state_dict(root / "model_weights.pt")
    key = "gene_embeddings.embedding.weight"
    weights = sd.get(key)
    weights = weights.detach().float().numpy() if hasattr(weights, "detach") else None
    out: dict = {"root": str(root), "reader": reader, "weights_key": key,
                 "weights_shape": None if weights is None else list(weights.shape),
                 "h5_file": None if h5 is None else str(h5.name)}
    if h5 is None or weights is None:
        return out
    datasets = h5_datasets(h5)
    out["h5_n_datasets"] = len(datasets)
    out["h5_datasets_head"] = datasets[:5]
    shipped, names = transcriptformer_shipped_table(h5)
    out["h5_n_genes"] = len(names)
    out["h5_names_head"] = names[:5]
    out["rebuilt_shape"] = list(shipped.shape)
    out["rebuilt_summary"] = _matrix_summary(shipped)
    out["weights_summary"] = _matrix_summary(weights)
    out["weights_vs_shipped"] = _compare(weights, shipped)
    # the seven special rows are ours to reproduce, so report the genes alone too
    if shipped.shape == weights.shape:
        out["weights_vs_shipped_genes_only"] = _compare(weights[7:], shipped[7:])
        out["special_token_rows"] = _compare(weights[:7], shipped[:7])
    return out


def scprint_initialisation(cache: Path, out_dir: Path) -> dict:
    ckpt = cache / "scprint" / S.SCPRINT["files"]["weights"]
    obj, reader = ckptread.torch_state_dict(ckpt)
    sd = obj.get("state_dict", {}) if isinstance(obj, dict) else {}
    hp = obj.get("hyper_parameters", {}) if isinstance(obj, dict) else {}
    genes = [str(g) for g in (hp.get("genes") or [])]
    key = "gene_encoder.embedding.weight"
    table = sd.get(key)
    table = table.detach().float().numpy() if hasattr(table, "detach") else None
    record: dict = {"reader": reader, "key": key,
                    "shape": None if table is None else list(table.shape),
                    "n_genes_in_hparams": len(genes),
                    "precpt_gene_emb": hp.get("precpt_gene_emb"),
                    "freeze_embeddings": hp.get("freeze_embeddings")}

    listings: dict[str, list] = {}
    for sub in ("", "embeddings"):
        url = S.SCPRINT["tree_api"] + (f"/{sub}" if sub else "")
        try:
            listings[sub or "root"] = json.loads(_fetch_text(url))
        except Exception as exc:  # noqa: BLE001
            listings[sub or "root"] = [{"error": f"{type(exc).__name__}: {exc}"[:200]}]
    record["repo_listing"] = {
        k: [{"path": e.get("path"), "size": e.get("size")} for e in v if isinstance(e, dict)]
        for k, v in listings.items()
    }
    try:
        readme = _fetch_text(f"{S.HF_BASE}/{S.SCPRINT['repo']}/raw/main/README.md")
        record["readme_mentions"] = _grep(readme, ("esm", "embedding", "protein"), limit=20)
    except Exception as exc:  # noqa: BLE001
        record["readme_mentions"] = [f"{type(exc).__name__}: {exc}"[:200]]

    # only a file that could carry a named gene table; the release's npz is a
    # sparse attention bias and its .mmap carries no gene names at all
    candidates = [
        e for v in listings.values() for e in v
        if isinstance(e, dict) and isinstance(e.get("path"), str)
        and e["path"].endswith((".parquet", ".csv"))
        and 0 < (e.get("size") or 0) < 3_000_000_000
    ]
    record["candidates"] = [{"path": c["path"], "size": c["size"]} for c in candidates]
    if candidates and table is not None:
        pick = min(candidates, key=lambda e: e["size"])
        record["compared_against"] = pick["path"]
        try:
            local = out_dir / Path(pick["path"]).name
            download(f"{S.HF_BASE}/{S.SCPRINT['repo']}/resolve/main/{pick['path']}", local)
            mat, names, fmt = load_matrix_file(local)
            record["format"] = fmt
            if mat is not None:
                mat = np.asarray(mat, dtype=np.float64)
                record["init_shape"] = list(mat.shape)
                if names is not None and genes:
                    pos = {str(g): i for i, g in enumerate(names)}
                    idx = [pos.get(g, -1) for g in genes]
                    keep = [j for j, i in enumerate(idx) if i >= 0]
                    record["genes_matched"] = len(keep)
                    if keep and mat.shape[1] == table.shape[1]:
                        record["vs_checkpoint"] = _compare(
                            table[keep], mat[[idx[j] for j in keep]]
                        )
                elif mat.shape == table.shape:
                    record["vs_checkpoint"] = _compare(table, mat)
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return record


def transcriptformer_source() -> dict:
    """Find the model source in the repo through the API, then read it."""
    api = "https://api.github.com/repos/czi-ai/transcriptformer/git/trees/main?recursive=1"
    try:
        tree = json.loads(_fetch_text(api))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}
    paths = [t["path"] for t in tree.get("tree", []) if str(t.get("path", "")).endswith(".py")]
    want = [p for p in paths if p.endswith(("model/model.py", "tokenizer/vocab.py",
                                            "model/layers.py"))]
    out: dict = {"python_files": len(paths), "read": {}}
    for p in want[:4]:
        url = f"https://raw.githubusercontent.com/czi-ai/transcriptformer/main/{p}"
        try:
            text = _fetch_text(url)
            out["read"][p] = {
                "bytes": len(text),
                "embedding_lines": _grep(text, ("emb_matrix", "requires_grad", "freeze",
                                                "nn.embedding", "from_pretrained"), limit=24),
                "esm_lines": _grep(text, ("esm",), limit=10),
            }
        except Exception as exc:  # noqa: BLE001
            out["read"][p] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=None, help="the first pass's _ckpt directory")
    ap.add_argument("--mount", default=None, help="search here for it")
    ap.add_argument("--out", default="results/provenance")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache) if args.cache else None
    if cache is None or not cache.is_dir():
        cache = find_cache(Path(args.mount)) if args.mount else None
    if cache is None:
        raise SystemExit("no checkpoint cache found; pass --cache or --mount")
    print(f"[tables] cache={cache}", flush=True)

    record = {
        "written_by": "scfm_pfm.protein.audit_tables",
        "cache": str(cache),
        "transcriptformer": transcriptformer_tables(cache),
        "scprint": scprint_initialisation(cache, out),
        "transcriptformer_source": transcriptformer_source(),
    }
    (out / "audit_tables.json").write_text(json.dumps(record, indent=2, default=str))
    print(json.dumps(record, indent=2, default=str)[:8000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
