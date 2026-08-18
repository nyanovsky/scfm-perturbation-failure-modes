"""Run one model's response interface on one real control cell.

Each model runs in its own pinned environment, so this is the entrypoint each of
the four jobs calls with ``--model``. Every job writes the same five files into
its own directory, which is the whole point: the four stacks never have to meet
in one Python process, they only have to agree on the files.

    models/<model>/response_vector.parquet   per-universe-gene response
    models/<model>/jacobian_column.parquet   d response / d source gene count
    models/<model>/contract.json             the declared response contract
    models/<model>/environment.json          pinned packages, image, device
    models/<model>/demo.json                 what was run, timings, checks
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths
from .base import environment_manifest

REGISTRY = {
    "scgpt": ("scgpt_model", "ScGPTResponse"),
    "aido_cell": ("aido_model", "AidoCellResponse"),
    "scprint": ("scprint_model", "ScPrintResponse"),
    "transcriptformer": ("transcriptformer_model", "TranscriptFormerResponse"),
}


def build(model: str, cache: Path, universe: pd.DataFrame, vocab: pd.DataFrame, device: str):
    import importlib

    mod_name, cls_name = REGISTRY[model]
    mod = importlib.import_module(f".{mod_name}", package=__package__)
    return getattr(mod, cls_name)(cache=cache, universe=universe, vocab=vocab, device=device)


def run(
    model: str,
    cache: Path,
    universe_dir: Path,
    demo_dir: Path,
    out_dir: Path,
    cell_line: str = "hepg2",
    cell_index: int = 0,
    device: str = "cuda",
    image: str = "unrecorded",
) -> dict:
    started = time.time()
    universe = pd.read_parquet(universe_dir / "gene_universe.parquet")
    vocab = pd.read_parquet(universe_dir / f"vocabulary_{model}.parquet")
    npz = np.load(demo_dir / f"demo_cells_{cell_line}.npz", allow_pickle=False)
    counts = npz["counts"][cell_index].astype(np.float32)
    gene_ids = npz["gene_ids"].astype(str)
    gene_names = npz["gene_names"].astype(str)
    barcode = str(npz["barcodes"][cell_index])

    uni_ids = universe["gene_id"].to_numpy().astype(str)
    count_by_gene = dict(zip(gene_ids, counts))
    uni_counts = np.array([count_by_gene.get(g, 0.0) for g in uni_ids], dtype=np.float32)
    source_row = int(np.lexsort((uni_ids, -uni_counts))[0])
    source_gene = str(uni_ids[source_row])
    source_position = int(universe[f"{model}_position"].to_numpy()[source_row])

    print(f"[{model}] loading checkpoint from {cache}", flush=True)
    m = build(model, cache, universe, vocab, device)
    t_load = time.time()
    m.load()
    load_seconds = time.time() - t_load

    prepared = m.prepare_input(counts, gene_ids)
    t_fwd = time.time()
    response = np.asarray(m.response(prepared), dtype=np.float64)
    forward_seconds = time.time() - t_fwd
    if response.shape != (universe.shape[0],):
        raise RuntimeError(
            f"response shape {response.shape} does not match the universe "
            f"({universe.shape[0]} genes)"
        )
    n_bad = int((~np.isfinite(response)).sum())
    if n_bad:
        raise RuntimeError(f"{n_bad} non-finite values in the {model} response vector")

    t_jac = time.time()
    column, method, reason = m.jacobian_column(prepared, source_position)
    jacobian_seconds = time.time() - t_jac
    if column is not None:
        column = np.asarray(column, dtype=np.float64)
        if column.shape != (universe.shape[0],):
            raise RuntimeError(f"jacobian column shape {column.shape} != universe")
        if not np.isfinite(column).all():
            raise RuntimeError("non-finite values in the jacobian column")

    contract = m.contract()
    out_dir = paths.ensure(out_dir)
    frame = pd.DataFrame(
        {
            "gene_id": uni_ids,
            "gene_name": universe["gene_name"].to_numpy().astype(str),
            "input_count": uni_counts,
            "response": response,
            "model_position": universe[f"{model}_position"].to_numpy().astype(np.int64),
        }
    )
    frame.to_parquet(out_dir / "response_vector.parquet", index=False)
    if column is not None:
        pd.DataFrame(
            {
                "gene_id": uni_ids,
                "gene_name": universe["gene_name"].to_numpy().astype(str),
                "d_response_d_source_count": column,
            }
        ).to_parquet(out_dir / "jacobian_column.parquet", index=False)
    (out_dir / "contract.json").write_text(contract.to_json())

    probe = None
    if hasattr(m, "ordering_probe"):
        target_row = int(np.lexsort((uni_ids, -uni_counts))[1])
        try:
            probe = m.ordering_probe(prepared, source_row, target_row)
        except Exception as exc:  # noqa: BLE001
            probe = {"error": f"{type(exc).__name__}: {exc}"}

    nz = response[uni_counts > 0]
    meta = {
        "model": model,
        "cell_line": cell_line,
        "cell_index": cell_index,
        "barcode": barcode,
        "device": device,
        "image": image,
        "universe_genes": int(universe.shape[0]),
        "input": {
            "measured_genes": int(gene_ids.size),
            "total_counts": float(counts.sum()),
            "genes_detected": int((counts > 0).sum()),
            "universe_genes_with_counts": int((uni_counts > 0).sum()),
            "model_input_genes_used": int(getattr(m, "_input_genes_used", -1)),
            "prepare_notes": {
                k: v
                for k, v in prepared.items()
                if isinstance(v, (int, float, str, bool))
            },
        },
        "source_gene": {
            "gene_id": source_gene,
            "gene_name": str(
                gene_names[np.flatnonzero(gene_ids == source_gene)[0]]
                if (gene_ids == source_gene).any()
                else universe["gene_name"].to_numpy()[source_row]
            ),
            "input_count": float(uni_counts[source_row]),
            "model_position": source_position,
            "selection": "highest raw count in the demo cell among universe genes",
        },
        "response_summary": {
            "min": float(response.min()),
            "median": float(np.median(response)),
            "max": float(response.max()),
            "l2": float(np.linalg.norm(response)),
            "median_expressed_genes": float(np.median(nz)) if nz.size else None,
            "spearman_response_vs_input": float(
                pd.Series(response).corr(pd.Series(uni_counts), method="spearman")
            ),
        },
        "jacobian": {
            "available": column is not None,
            "method": method,
            "unavailable_reason": reason,
            "column_l2": float(np.linalg.norm(column)) if column is not None else None,
            "max_abs": float(np.abs(column).max()) if column is not None else None,
            "self_entry": float(column[source_row]) if column is not None else None,
            "n_nonzero": int((column != 0).sum()) if column is not None else None,
            "check": getattr(m, "jacobian_check", {}),
        },
        "ordering_probe": probe,
        "substitutions": list(contract.substitutions),
        "timings_seconds": {
            "load": round(load_seconds, 2),
            "forward": round(forward_seconds, 3),
            "jacobian": round(jacobian_seconds, 2),
            "total": round(time.time() - started, 2),
        },
    }
    for key in ("load_report", "token_alignment", "output_check"):
        if hasattr(m, key):
            meta[key] = getattr(m, key)
    (out_dir / "demo.json").write_text(json.dumps(meta, indent=2, default=str))
    (out_dir / "environment.json").write_text(
        json.dumps(
            environment_manifest(
                model,
                image,
                {
                    "device": device,
                    "checkpoint": contract.checkpoint,
                    "checkpoint_revision": contract.checkpoint_revision,
                    "substitutions": list(contract.substitutions),
                },
            ),
            indent=2,
            default=str,
        )
    )
    print(json.dumps(meta, indent=2, default=str)[:6000], flush=True)
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--cache", required=True)
    ap.add_argument("--universe-dir", default=None)
    ap.add_argument("--demo-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cell-line", default="hepg2")
    ap.add_argument("--cell-index", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--image", default="unrecorded")
    args = ap.parse_args(argv)
    universe_dir = Path(args.universe_dir) if args.universe_dir else paths.universe_input_dir()
    demo_dir = Path(args.demo_dir) if args.demo_dir else paths.input_root() / "demo_cells"
    out = Path(args.out) if args.out else paths.model_dir(args.model)
    try:
        run(
            args.model,
            Path(args.cache),
            universe_dir,
            demo_dir,
            out,
            cell_line=args.cell_line,
            cell_index=args.cell_index,
            device=args.device,
            image=args.image,
        )
    except Exception:
        paths.ensure(out)
        (out / "failure.txt").write_text(traceback.format_exc())
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
