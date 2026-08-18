"""Harmonize gene identifiers across one GEO matrix and four model vocabularies.

Gene identifier handling is a correctness risk here, not a formatting one: the
Nadig matrices are indexed by unversioned Ensembl gene IDs, scGPT and AIDO.Cell
speak HGNC symbols, and scPRINT and TranscriptFormer speak Ensembl. Symbols
carry aliases and retired names, so every mapping is resolved through an
explicit route, audited in **both** directions, and unmapped tokens are counted
rather than silently dropped.

Outputs (``gene_universe/``):

* ``gene_universe.parquet`` — one row per gene in the shared universe with its
  index/token id in each of the four vocabularies and in each cell line's matrix
* ``identifier_map.parquet`` — every Nadig gene with its symbol, resolution
  route and round-trip status
* ``model_vocabularies.json`` — provenance and size of each vocabulary
* ``coverage.json`` — per-model coverage and unmapped counts
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths
from ..data import sources as data_sources
from ..models import sources as S
from . import vocabularies as V


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def nadig_gene_tables() -> dict[str, pd.DataFrame]:
    """Read only the ``var`` table of each h5ad, over HTTP range reads."""
    import fsspec
    import h5py

    out = {}
    for cell_line in paths.CELL_LINES:
        url = data_sources.h5ad_url(cell_line)
        fs = fsspec.filesystem("http", block_size=4 * 1024 * 1024)
        with fs.open(url, "rb") as fh, h5py.File(fh, "r") as h:
            ids = np.array([x.decode() for x in h["var/gene_id"][:]])
            cats = np.array([x.decode() for x in h["var/__categories/gene_name"][:]])
            names = cats[h["var/gene_name"][:]]
        out[cell_line] = pd.DataFrame(
            {"gene_id": ids, "gene_name": names, "matrix_index": np.arange(ids.size)}
        )
        print(f"[universe] {cell_line}: {ids.size} measured genes", flush=True)
    return out


def hgnc_table(cache: Path) -> pd.DataFrame | None:
    """HGNC complete set: approved symbol, aliases, previous names, Ensembl id."""
    path = cache / "hgnc" / "hgnc_complete_set.txt"
    try:
        data_sources.download(S.HGNC_COMPLETE_SET, path)
        df = pd.read_csv(path, sep="\t", low_memory=False)
    except Exception as exc:  # noqa: BLE001 - the audit degrades, it does not fail
        print(f"[universe] HGNC table unavailable ({exc}); alias audit degraded", flush=True)
        return None
    keep = [c for c in ("symbol", "alias_symbol", "prev_symbol", "ensembl_gene_id") if c in df]
    return df[keep]


# --------------------------------------------------------------------------- #
# mapping
# --------------------------------------------------------------------------- #
def strip_version(x: str) -> str:
    return x.split(".", 1)[0]


class IdentifierResolver:
    """Symbol/Ensembl resolution with an explicit, recorded route per token."""

    def __init__(self, nadig: pd.DataFrame, hgnc: pd.DataFrame | None):
        self.ensg_to_symbol: dict[str, str] = dict(zip(nadig.gene_id, nadig.gene_name))
        self.symbol_to_ensg: dict[str, str] = {}
        for ensg, sym in zip(nadig.gene_id, nadig.gene_name):
            self.symbol_to_ensg.setdefault(sym, ensg)
        self.hgnc_symbol_to_ensg: dict[str, str] = {}
        self.hgnc_alias_to_ensg: dict[str, str] = {}
        self.hgnc_ensg_to_symbol: dict[str, str] = {}
        if hgnc is not None:
            for row in hgnc.itertuples(index=False):
                ensg = getattr(row, "ensembl_gene_id", None)
                sym = getattr(row, "symbol", None)
                if not isinstance(ensg, str) or not isinstance(sym, str):
                    continue
                self.hgnc_symbol_to_ensg.setdefault(sym, ensg)
                self.hgnc_ensg_to_symbol.setdefault(ensg, sym)
                for field in ("alias_symbol", "prev_symbol"):
                    val = getattr(row, field, None)
                    if isinstance(val, str):
                        for alias in val.split("|"):
                            alias = alias.strip()
                            if alias:
                                self.hgnc_alias_to_ensg.setdefault(alias, ensg)

    def to_ensembl(self, token: str, kind: str) -> tuple[str | None, str]:
        if kind == "ensembl":
            base = strip_version(token)
            if base in self.ensg_to_symbol:
                return base, "ensembl_direct"
            if base in self.hgnc_ensg_to_symbol:
                return base, "ensembl_hgnc_only"
            return base, "ensembl_unknown_to_matrix"
        if token in self.symbol_to_ensg:
            return self.symbol_to_ensg[token], "symbol_matrix"
        if token in self.hgnc_symbol_to_ensg:
            return self.hgnc_symbol_to_ensg[token], "symbol_hgnc_approved"
        if token in self.hgnc_alias_to_ensg:
            return self.hgnc_alias_to_ensg[token], "symbol_hgnc_alias_or_prev"
        return None, "unmapped"


def round_trip_audit(nadig: pd.DataFrame, res: IdentifierResolver) -> pd.DataFrame:
    """Both directions: ENSG -> symbol -> ENSG and symbol -> ENSG -> symbol."""
    rows = []
    for ensg, sym in zip(nadig.gene_id, nadig.gene_name):
        back_ensg = res.symbol_to_ensg.get(sym)
        fwd_sym = res.ensg_to_symbol.get(ensg)
        rows.append(
            {
                "gene_id": ensg,
                "gene_name": sym,
                "ensg_to_symbol_to_ensg_ok": back_ensg == ensg,
                "symbol_to_ensg_to_symbol_ok": (
                    fwd_sym is not None and res.ensg_to_symbol.get(back_ensg or "") == sym
                ),
                "symbol_is_hgnc_approved": sym in res.hgnc_symbol_to_ensg,
                "hgnc_symbol_for_ensg": res.hgnc_ensg_to_symbol.get(ensg),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def build(cache: Path, out_dir: Path) -> dict:
    started = time.time()
    tables = nadig_gene_tables()
    nadig_union = (
        pd.concat(
            [t.assign(cell_line=cl) for cl, t in tables.items()], ignore_index=True
        )
        .drop_duplicates("gene_id")[["gene_id", "gene_name"]]
        .reset_index(drop=True)
    )
    hgnc = hgnc_table(cache)
    res = IdentifierResolver(nadig_union, hgnc)
    vocabs = V.load_all(cache)

    # per-model: token -> ensembl
    model_maps: dict[str, dict[str, int]] = {}
    coverage: dict[str, dict] = {}
    token_tables: dict[str, pd.DataFrame] = {}
    for name, vocab in vocabs.items():
        ensg_to_pos: dict[str, int] = {}
        routes: dict[str, int] = {}
        resolved, route_col = [], []
        for pos, token in enumerate(vocab.tokens):
            ensg, route = res.to_ensembl(token, vocab.kind)
            routes[route] = routes.get(route, 0) + 1
            resolved.append(ensg or "")
            route_col.append(route)
            if ensg is not None:
                ensg_to_pos.setdefault(ensg, pos)
        model_maps[name] = ensg_to_pos
        token_tables[name] = pd.DataFrame(
            {
                "position": np.arange(len(vocab.tokens)),
                "token": vocab.tokens,
                "gene_id": resolved,
                "resolution_route": route_col,
            }
        )
        hep = set(tables["hepg2"].gene_id) & set(ensg_to_pos)
        jur = set(tables["jurkat"].gene_id) & set(ensg_to_pos)
        coverage[name] = {
            "vocabulary_size": len(vocab.tokens),
            "identifier_kind": vocab.kind,
            "order_is_a_model_constraint": vocab.ordered,
            "resolution_routes": routes,
            "n_unmapped_tokens": routes.get("unmapped", 0)
            + routes.get("ensembl_unknown_to_matrix", 0),
            "n_resolved_to_distinct_ensembl": len(ensg_to_pos),
            "covers_hepg2_measured": len(hep),
            "covers_jurkat_measured": len(jur),
            "hepg2_measured_total": int(tables["hepg2"].shape[0]),
            "jurkat_measured_total": int(tables["jurkat"].shape[0]),
            "provenance": vocab.provenance,
            "special_tokens": vocab.special_tokens,
        }

    shared_measured = set(tables["hepg2"].gene_id) & set(tables["jurkat"].gene_id)
    universe = sorted(
        shared_measured.intersection(*[set(m) for m in model_maps.values()])
    )
    hep_idx = dict(zip(tables["hepg2"].gene_id, tables["hepg2"].matrix_index))
    jur_idx = dict(zip(tables["jurkat"].gene_id, tables["jurkat"].matrix_index))

    rows = []
    for ensg in universe:
        row = {
            "gene_id": ensg,
            "gene_name": res.ensg_to_symbol.get(ensg, ""),
            "hgnc_symbol": res.hgnc_ensg_to_symbol.get(ensg),
            "hepg2_matrix_index": hep_idx[ensg],
            "jurkat_matrix_index": jur_idx[ensg],
        }
        for name in vocabs:
            row[f"{name}_position"] = model_maps[name][ensg]
        rows.append(row)
    uni = pd.DataFrame(rows)

    audit = round_trip_audit(nadig_union, res)
    imap = nadig_union.merge(audit, on=["gene_id", "gene_name"], how="left")
    imap["in_universe"] = imap.gene_id.isin(set(universe))
    for name in vocabs:
        imap[f"in_{name}"] = imap.gene_id.isin(set(model_maps[name]))

    out_dir = paths.ensure(out_dir)
    uni.to_parquet(out_dir / "gene_universe.parquet", index=False)
    imap.to_parquet(out_dir / "identifier_map.parquet", index=False)
    (out_dir / "model_vocabularies.json").write_text(
        json.dumps({k: v.as_record() for k, v in vocabs.items()}, indent=2, default=str)
    )
    for name, table in token_tables.items():
        table.to_parquet(out_dir / f"vocabulary_{name}.parquet", index=False)

    summary = {
        "universe_size": int(uni.shape[0]),
        "shared_measured_genes": len(shared_measured),
        "coverage": coverage,
        "identifier_audit": {
            "n_nadig_genes": int(imap.shape[0]),
            "ensg_to_symbol_to_ensg_failures": int((~imap.ensg_to_symbol_to_ensg_ok).sum()),
            "symbol_to_ensg_to_symbol_failures": int((~imap.symbol_to_ensg_to_symbol_ok).sum()),
            "symbols_not_hgnc_approved": int((~imap.symbol_is_hgnc_approved).sum()),
            "hgnc_table_available": hgnc is not None,
            "round_trip_failure_examples": imap.loc[
                ~imap.ensg_to_symbol_to_ensg_ok, ["gene_id", "gene_name"]
            ].head(10).to_dict("records"),
        },
        "aido_fixed_order_size": len(vocabs["aido_cell"].tokens),
        "runtime_seconds": round(time.time() - started, 1),
    }
    (out_dir / "coverage.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ("universe_size", "shared_measured_genes")}, indent=1))
    for name, cov in coverage.items():
        print(
            f"[universe] {name:16s} vocab={cov['vocabulary_size']:6d} "
            f"kind={cov['identifier_kind']:8s} hepg2={cov['covers_hepg2_measured']:5d} "
            f"jurkat={cov['covers_jurkat_measured']:5d} unmapped={cov['n_unmapped_tokens']:6d}",
            flush=True,
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cache",
        default=None,
        help="where to keep downloaded checkpoints (defaults to the raw dir)",
    )
    args = ap.parse_args(argv)
    cache = Path(args.cache) if args.cache else paths.raw_dir() / "checkpoints"
    build(paths.ensure(cache), paths.universe_dir())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
