"""Stage 1: where each model's gene-identity table came from.

Every arm that beat co-expression fed the translator a model's *static* gene
embedding -- one fixed vector per gene, read before any cell is shown. Two of
the four models initialise that table from a protein language model, so the
question is whether pretraining ever changed it.

The evidence is taken from the released checkpoint, not from a docstring:

``scgpt``             a trained ``nn.Embedding`` over the gene vocabulary, no
                      protein input anywhere in the archive.
``aido_cell``         the same shape of evidence over its fixed gene order.
``scprint``           the gene table's shape does not appear anywhere in the
                      optimizer state the checkpoint carries, so no optimizer
                      step was ever taken for it.
``transcriptformer``  whether the ESM2 table lives in the weights at all, and if
                      it does, whether it still equals the copy shipped beside
                      them in ``vocabs/``.

Upstream source is fetched as corroboration and recorded, but a source string is
not what decides a verdict here.

    python -m scfm_pfm.protein.audit --cache _ckpt --out results/provenance
"""

from __future__ import annotations

import argparse
import json
import struct
import tarfile
import time
import traceback
import urllib.request
from pathlib import Path

import numpy as np

from ..data.sources import download
from . import ckptread
from . import sources as S

GENE_TABLE_HINTS = ("gene", "emb", "token", "vocab")


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _matrix_summary(arr: np.ndarray) -> dict:
    a = np.asarray(arr, dtype=np.float64)
    norms = np.linalg.norm(a, axis=1)
    return {
        "shape": list(a.shape),
        "l2_norm": float(np.linalg.norm(a)),
        "row_norm_mean": float(norms.mean()),
        "row_norm_sd": float(norms.std()),
        "zero_rows": int((norms == 0).sum()),
        "finite": bool(np.isfinite(a).all()),
    }


def _compare(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return {"comparable": False, "reason": f"shapes {a.shape} vs {b.shape}"}
    diff = np.abs(a - b)
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cos = np.where(den > 0, num / np.maximum(den, 1e-30), np.nan)
    return {
        "comparable": True,
        "bitwise_identical": bool(np.array_equal(np.asarray(a), np.asarray(b))),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "median_row_cosine": float(np.nanmedian(cos)),
        "min_row_cosine": float(np.nanmin(cos)),
    }


def _fetch_text(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "scfm-pfm3/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return fh.read().decode("utf-8", "replace")


def _grep(text: str, needles: tuple[str, ...], context: int = 0, limit: int = 12) -> list[str]:
    lines = text.splitlines()
    hits = []
    for i, line in enumerate(lines):
        low = line.lower()
        if any(n in low for n in needles):
            block = lines[max(0, i - context): i + context + 1]
            hits.append(" / ".join(x.strip() for x in block)[:300])
        if len(hits) >= limit:
            break
    return hits


def _safetensors_header(path: Path) -> dict:
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(n).decode("utf-8"))


def _safetensors_tensor(path: Path, name: str) -> np.ndarray:
    header = _safetensors_header(path)
    meta = header[name]
    start, end = meta["data_offsets"]
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        fh.seek(8 + n + start)
        raw = fh.read(end - start)
    dtype = {"F32": np.float32, "F16": np.float16, "BF16": np.float32, "F64": np.float64}[
        meta["dtype"]
    ]
    if meta["dtype"] == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        arr = u16.view(np.float32)
    else:
        arr = np.frombuffer(raw, dtype=dtype)
    return arr.reshape(meta["shape"])


def _tensor_shapes(obj) -> dict[str, list[int]]:
    """``{key: shape}`` for a state dict of either real tensors or descriptors."""
    out: dict[str, list[int]] = {}
    for k, v in (obj or {}).items():
        if isinstance(v, dict) and v.get("__tensor__"):
            out[str(k)] = list(v["shape"])
        elif hasattr(v, "shape"):
            out[str(k)] = [int(s) for s in v.shape]
    return out


def _pick_gene_table(shapes: dict[str, list[int]], n_rows: int | None) -> list[str]:
    """Keys whose 2-D shape could be a per-gene table."""
    hits = []
    for k, s in shapes.items():
        if len(s) != 2:
            continue
        looks = any(h in k.lower() for h in GENE_TABLE_HINTS)
        if n_rows is not None and s[0] in (n_rows, n_rows + 1, n_rows + 2, n_rows + 4, n_rows + 8):
            hits.append(k)
        elif looks and s[0] > 1000:
            hits.append(k)
    return sorted(set(hits))


def load_matrix_file(path: Path) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """A (names, matrix) pair out of whatever format a table ships in."""
    suffix = path.suffix.lower()
    try:
        if suffix in (".npz",):
            z = np.load(path, allow_pickle=True)
            keys = list(z.files)
            mat = next((z[k] for k in keys if z[k].ndim == 2), None)
            names = next((z[k] for k in keys if z[k].ndim == 1 and z[k].dtype.kind in "USO"), None)
            return mat, names, f"npz{keys}"
        if suffix == ".npy":
            return np.load(path, allow_pickle=True), None, "npy"
        if suffix in (".parquet", ".pq"):
            import pandas as pd  # noqa: PLC0415

            df = pd.read_parquet(path)
            num = df.select_dtypes("number")
            names = df.index.to_numpy().astype(str) if df.index.dtype == object else None
            return num.to_numpy(), names, f"parquet{list(df.columns)[:4]}"
        if suffix in (".csv", ".tsv", ".txt"):
            import pandas as pd  # noqa: PLC0415

            df = pd.read_csv(path, sep="\t" if suffix != ".csv" else ",", index_col=0)
            return df.select_dtypes("number").to_numpy(), df.index.to_numpy().astype(str), "csv"
        if suffix in (".pkl", ".pickle"):
            import pickle  # noqa: PLC0415

            obj = pickle.loads(path.read_bytes())  # noqa: S301 - a CZI release file
            if isinstance(obj, dict):
                names = np.array(list(obj), dtype=object)
                vals = list(obj.values())
                if vals and hasattr(vals[0], "__len__"):
                    return np.asarray(vals, dtype=np.float32), names, "pickle-dict"
            if hasattr(obj, "shape"):
                return np.asarray(obj), None, "pickle-array"
            return None, None, f"pickle-{type(obj).__name__}"
        if suffix in (".h5ad", ".h5", ".hdf5"):
            import h5py  # noqa: PLC0415

            with h5py.File(path, "r") as h:
                mat = None
                if "X" in h and hasattr(h["X"], "shape"):
                    mat = np.asarray(h["X"][:])
                names = None
                if "obs" in h:
                    idx = h["obs"].attrs.get("_index", "index")
                    idx = idx.decode() if isinstance(idx, bytes) else str(idx)
                    if idx in h["obs"]:
                        names = np.array(
                            [x.decode() if isinstance(x, bytes) else str(x) for x in h["obs"][idx][:]]
                        )
                if mat is None:
                    for key in h:
                        if hasattr(h[key], "shape") and len(h[key].shape) == 2:
                            mat = np.asarray(h[key][:])
                            break
                return mat, names, "h5"
    except Exception as exc:  # noqa: BLE001 - the format probe is best effort
        return None, None, f"error:{type(exc).__name__}:{exc}"[:200]
    return None, None, f"unhandled{suffix}"


# --------------------------------------------------------------------------- #
# per-model audits
# --------------------------------------------------------------------------- #
def audit_scgpt(cache: Path) -> dict:
    d = cache / "scgpt"
    files = {k: download(S.hf_url(S.SCGPT, k), d / S.SCGPT["files"][k]) for k in S.SCGPT["files"]}
    vocab = json.loads((d / "vocab.json").read_text())
    args = json.loads((d / "args.json").read_text())
    sd, reader = ckptread.torch_state_dict(d / "best_model.pt")
    shapes = _tensor_shapes(sd)
    candidates = _pick_gene_table(shapes, len(vocab))
    key = next((k for k in candidates if "encoder.embedding" in k), candidates[0] if candidates else "")
    table = None
    if key and hasattr(sd.get(key), "numpy"):
        table = sd[key].detach().float().numpy()
    esm_mentions = [k for k in shapes if "esm" in k.lower() or "prot" in k.lower()]
    return {
        "checkpoint": {"repo": S.SCGPT["repo"], "revision": S.SCGPT["revision"],
                       "files": {k: v.get("size_bytes") for k, v in files.items()}},
        "reader": reader,
        "n_tensors": len(shapes),
        "vocab_size": len(vocab),
        "model_width": args.get("embsize"),
        "gene_table_key": key,
        "gene_table_shape": shapes.get(key),
        "gene_table_candidates": candidates,
        "gene_table_summary": _matrix_summary(table) if table is not None else None,
        "keys_mentioning_protein": esm_mentions,
        "all_keys_head": sorted(shapes)[:24],
    }


def audit_aido(cache: Path) -> dict:
    d = cache / "aido_cell"
    files = {
        k: download(S.hf_url(S.AIDO_CELL, k), d / S.AIDO_CELL["files"][k])
        for k in S.AIDO_CELL["files"]
    }
    config = json.loads((d / "config.json").read_text())
    header = _safetensors_header(d / "model.safetensors")
    shapes = {k: v["shape"] for k, v in header.items() if k != "__metadata__"}
    n_genes = int(config.get("num_genes") or config.get("n_genes") or 19264)
    candidates = _pick_gene_table(shapes, n_genes)
    key = candidates[0] if candidates else ""
    table = _safetensors_tensor(d / "model.safetensors", key) if key else None
    # the modelling source, for what the table is and how it is initialised
    tgz = d / "modelgenerator_sdist.tar.gz"
    src_hits: dict[str, list[str]] = {}
    try:
        download(S.AIDO_CELL["gene_order_sdist"], tgz)
        with tarfile.open(tgz, "r:gz") as tf:
            for member in S.AIDO_CELL["modeling_members"]:
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                text = fh.read().decode("utf-8", "replace")
                src_hits[Path(member).name] = _grep(text, ("embedding", "esm", "protein"))
    except Exception as exc:  # noqa: BLE001
        src_hits["error"] = [f"{type(exc).__name__}: {exc}"[:200]]
    return {
        "checkpoint": {"repo": S.AIDO_CELL["repo"], "revision": S.AIDO_CELL["revision"],
                       "files": {k: v.get("size_bytes") for k, v in files.items()}},
        "n_tensors": len(shapes),
        "config_head": {k: v for k, v in list(config.items())[:20] if not isinstance(v, dict)},
        "n_genes": n_genes,
        "gene_table_key": key,
        "gene_table_shape": shapes.get(key),
        "gene_table_candidates": candidates,
        "gene_table_summary": _matrix_summary(table) if table is not None else None,
        "keys_mentioning_protein": [k for k in shapes if "esm" in k.lower() or "prot" in k.lower()],
        "all_keys_head": sorted(shapes)[:24],
        "source_mentions": src_hits,
    }


def audit_scprint(cache: Path) -> dict:
    d = cache / "scprint"
    meta = download(S.hf_url(S.SCPRINT, "weights"), d / S.SCPRINT["files"]["weights"])
    obj, reader = ckptread.torch_state_dict(d / S.SCPRINT["files"]["weights"])
    top = sorted(obj) if isinstance(obj, dict) else []
    sd = obj.get("state_dict", {}) if isinstance(obj, dict) else {}
    shapes = _tensor_shapes(sd)
    hp = obj.get("hyper_parameters", {}) if isinstance(obj, dict) else {}
    genes = hp.get("genes") if isinstance(hp, dict) else None
    n_genes = len(genes) if isinstance(genes, (list, tuple)) else None
    candidates = [k for k in shapes if "gene_encoder" in k and len(shapes[k]) == 2]
    key = candidates[0] if candidates else ""
    shape = shapes.get(key)

    # --- the decisive check: does the optimizer carry a state for that tensor?
    opt_states = obj.get("optimizer_states", []) if isinstance(obj, dict) else []
    opt_shapes: list[list[int]] = []
    n_opt_params = 0
    for o in opt_states:
        if not isinstance(o, dict):
            continue
        for group in o.get("param_groups", []) or []:
            if isinstance(group, dict):
                n_opt_params += len(group.get("params", []) or [])
        for _, st in (o.get("state", {}) or {}).items():
            if not isinstance(st, dict):
                continue
            for _, t in st.items():
                if isinstance(t, dict) and t.get("__tensor__"):
                    opt_shapes.append(list(t["shape"]))
                elif hasattr(t, "shape") and getattr(t, "ndim", 0) > 0:
                    opt_shapes.append([int(s) for s in t.shape])
    tracked = [s for s in opt_shapes if shape is not None and s == shape]
    same_shape_params = [k for k, s in shapes.items() if s == shape]

    # --- what the initialisation was, and whether it still ships
    tree: list[dict] = []
    try:
        tree = json.loads(_fetch_text(S.SCPRINT["tree_api"]))
    except Exception as exc:  # noqa: BLE001
        tree = [{"error": f"{type(exc).__name__}: {exc}"[:200]}]
    hp_small = ckptread.shorten(hp) if isinstance(hp, dict) else str(hp)[:400]

    # --- if the initialisation still ships, compare the checkpoint against it
    init = {"attempted": False}
    table = sd.get(key) if key else None
    table = table.detach().float().numpy() if hasattr(table, "detach") else None
    cands = [
        e for e in tree
        if isinstance(e, dict) and isinstance(e.get("path"), str)
        and any(t in e["path"].lower() for t in ("emb", "main.parquet"))
        and 0 < (e.get("size") or 0) < 3_000_000_000
    ]
    if cands and table is not None:
        pick = min(cands, key=lambda e: e["size"])
        init = {"attempted": True, "file": pick["path"], "size": pick["size"]}
        try:
            url = f"{S.HF_BASE}/{S.SCPRINT['repo']}/resolve/main/{pick['path']}"
            local = d / Path(pick["path"]).name
            download(url, local)
            mat, names, fmt = load_matrix_file(local)
            init["format"] = fmt
            if mat is not None:
                init["init_shape"] = list(np.asarray(mat).shape)
                rows = np.asarray(mat)
                if names is not None and isinstance(genes, (list, tuple)):
                    pos = {str(g): i for i, g in enumerate(names)}
                    idx = [pos.get(str(g), -1) for g in genes]
                    init["genes_matched"] = int(sum(i >= 0 for i in idx))
                    keep = [j for j, i in enumerate(idx) if i >= 0]
                    if keep:
                        rows = rows[[idx[j] for j in keep]]
                        init.update(_compare(table[keep], rows))
                elif rows.shape == table.shape:
                    init.update(_compare(table, rows))
        except Exception as exc:  # noqa: BLE001 - the numeric comparison is a bonus
            init["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return {
        "checkpoint": {"repo": S.SCPRINT["repo"], "revision": S.SCPRINT["revision"],
                       "size_bytes": meta.get("size_bytes")},
        "reader": reader,
        "top_level_keys": top,
        "epoch": obj.get("epoch") if isinstance(obj, dict) else None,
        "global_step": obj.get("global_step") if isinstance(obj, dict) else None,
        "lightning_version": obj.get("pytorch-lightning_version") if isinstance(obj, dict) else None,
        "n_tensors": len(shapes),
        "n_genes_in_hparams": n_genes,
        "gene_table_key": key,
        "gene_table_shape": shape,
        "gene_table_candidates": candidates,
        "hyper_parameters": hp_small,
        "precpt_gene_emb": (hp.get("precpt_gene_emb") if isinstance(hp, dict) else None),
        "optimizer": {
            "n_optimizer_states": len(opt_states),
            "n_params_in_param_groups": n_opt_params,
            "n_state_tensors": len(opt_shapes),
            "n_state_tensors_with_gene_table_shape": len(tracked),
            "n_state_dict_tensors": len(shapes),
            "distinct_state_shapes_head": [list(x) for x in sorted({tuple(s) for s in opt_shapes})[:12]],
            "state_dict_tensors_with_that_shape": same_shape_params,
        },
        "hf_repo_files": [
            {"path": e.get("path"), "size": e.get("size")} for e in tree if isinstance(e, dict)
        ][:40],
        "gene_table_summary": _matrix_summary(table) if table is not None else None,
        "vs_initialisation": init,
    }


def audit_transcriptformer(cache: Path) -> dict:
    d = cache / "transcriptformer"
    tgz = d / "tf_sapiens.tar.gz"
    meta = download(S.TRANSCRIPTFORMER["url"], tgz)
    extract = d / "tf_sapiens"
    if not extract.exists():
        extract.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tgz, "r:gz") as tf:
            tf.extractall(extract)  # noqa: S202 - the CZI release tarball
    root = next((p.parent for p in extract.rglob("config.json")), extract)
    members = sorted(
        (str(p.relative_to(extract)), p.stat().st_size) for p in extract.rglob("*") if p.is_file()
    )
    config = json.loads((root / "config.json").read_text())
    data_cfg = config.get("model", config).get("data_config", {})
    esm_names = data_cfg.get("esm2_mappings", [])
    weights = next(root.glob("model_weights.pt"), None)
    sd, reader = ckptread.torch_state_dict(weights) if weights else ({}, "missing")
    shapes = _tensor_shapes(sd)

    vocab_dir = root / "vocabs"
    vocab_files = sorted(
        (str(p.relative_to(vocab_dir)), p.stat().st_size) for p in vocab_dir.rglob("*") if p.is_file()
    ) if vocab_dir.exists() else []

    shipped, shipped_names, fmt = (None, None, "not-found")
    esm_path = None
    for name in esm_names or []:
        cand = vocab_dir / name
        if cand.exists():
            esm_path = cand
            break
    if esm_path is None and vocab_dir.exists():
        esm_path = next((p for p in vocab_dir.rglob("*") if "esm" in p.name.lower()), None)
    if esm_path is not None:
        shipped, shipped_names, fmt = load_matrix_file(esm_path)

    dim = int(shipped.shape[1]) if shipped is not None and shipped.ndim == 2 else None
    in_weights = [k for k, s in shapes.items() if len(s) == 2 and (dim is None or s[1] == dim)]
    comparison = {"attempted": False}
    if shipped is not None and in_weights:
        key = in_weights[0]
        arr = sd[key]
        arr = arr.detach().float().numpy() if hasattr(arr, "detach") else None
        if arr is not None:
            n = min(arr.shape[0], shipped.shape[0])
            comparison = {"attempted": True, "key": key,
                          "rows_compared": int(n),
                          "note": "state-dict copy against the file shipped in vocabs/",
                          **_compare(arr[-n:], np.asarray(shipped)[-n:])}
    return {
        "checkpoint": {"url": S.TRANSCRIPTFORMER["url"], "size_bytes": meta.get("size_bytes")},
        "reader": reader,
        "root": str(root.relative_to(extract)) if root != extract else ".",
        "archive_files": members[:40],
        "n_tensors": len(shapes),
        "esm2_mappings_declared": esm_names,
        "esm2_file_used": str(esm_path.name) if esm_path is not None else None,
        "esm2_file_format": fmt,
        "esm2_file_summary": _matrix_summary(shipped) if shipped is not None and shipped.ndim == 2 else None,
        "esm2_file_names_head": [str(x) for x in (shipped_names[:5] if shipped_names is not None else [])],
        "vocab_files": vocab_files[:20],
        "gene_table_in_weights": in_weights,
        "gene_table_shape": shapes.get(in_weights[0]) if in_weights else None,
        "weights_vs_shipped": comparison,
        "keys_mentioning_gene_emb": [k for k in shapes if "emb" in k.lower()][:20],
        "all_keys_head": sorted(shapes)[:24],
        "model_config": {k: v for k, v in config.get("model", {}).get("model_config", {}).items()
                         if not isinstance(v, (dict, list))},
    }


AUDITS = {
    "scgpt": audit_scgpt,
    "aido_cell": audit_aido,
    "scprint": audit_scprint,
    "transcriptformer": audit_transcriptformer,
}


def upstream_sources() -> dict:
    out = {}
    for name, url in S.SOURCE_SNIPPETS.items():
        try:
            text = _fetch_text(url)
            out[name] = {
                "url": url,
                "bytes": len(text),
                "embedding_lines": _grep(text, ("embedding(", "geneencoder", "emb_matrix",
                                                "requires_grad", "freeze"), limit=20),
                "esm_lines": _grep(text, ("esm",), limit=12),
            }
        except Exception as exc:  # noqa: BLE001 - corroboration only
            out[name] = {"url": url, "error": f"{type(exc).__name__}: {exc}"[:200]}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="_ckpt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--models", nargs="*", default=list(AUDITS))
    args = ap.parse_args(argv)

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else Path("results/provenance")
    out.mkdir(parents=True, exist_ok=True)

    record: dict = {"written_by": "scfm_pfm.protein.audit", "models": {}}
    for model in args.models:
        t0 = time.time()
        print(f"[audit] {model}", flush=True)
        try:
            record["models"][model] = AUDITS[model](cache)
        except Exception:  # noqa: BLE001 - one model failing must not lose the others
            record["models"][model] = {"error": traceback.format_exc()[-3000:]}
            print(f"[audit] {model} FAILED\n{traceback.format_exc()[-2000:]}", flush=True)
        record["models"][model]["seconds"] = round(time.time() - t0, 1)
        print(f"[audit] {model} done in {time.time() - t0:.0f}s", flush=True)

    record["upstream_sources"] = upstream_sources()
    (out / "audit.json").write_text(json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({m: {k: v for k, v in r.items()
                          if k in ("gene_table_key", "gene_table_shape", "reader", "seconds")}
                      for m, r in record["models"].items()}, indent=2), flush=True)
    print(f"[audit] wrote {out / 'audit.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
