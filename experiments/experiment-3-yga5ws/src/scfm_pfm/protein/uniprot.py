"""Gene -> canonical human protein sequence, by one recorded route.

An arm computed on a biased subset of genes is not comparable to the arms
computed on all of them, so the mapping is written down rather than assumed:
every read-out gene gets a route label and every failure is counted.

The route, in order of preference:

1. ``ensembl``  the gene's Ensembl gene id appears in a reviewed (Swiss-Prot)
                human entry's Ensembl cross-references. This is an identifier
                match and needs no name matching at all.
2. ``symbol``   no cross-reference match, but the gene's symbol is some reviewed
                entry's primary gene name.
3. ``synonym``  no cross-reference and no primary-name match, but the symbol is
                listed as a synonym.
4. ``unmapped`` none of the above: the gene has no reviewed human protein. Most
                of these are genes that do not code for one.

When several reviewed entries match one gene the tie is broken in a fixed order
-- primary gene name equal to the gene's symbol, then higher UniProt annotation
score, then longer sequence, then lower accession -- and how often each rule
decided is recorded.

The sequence taken is the entry's canonical isoform, which is what UniProt
returns as the entry sequence.

    python -m scfm_pfm.protein.uniprot --out results/protein
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import sources as S

QUERY = "(organism_id:9606) AND (reviewed:true)"
FIELDS = (
    "accession,id,gene_primary,gene_synonym,xref_ensembl,sequence,length,"
    "annotation_score,protein_existence"
)
PAGE = 500


def _get(url: str, timeout: int = 120) -> tuple[dict, dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "scfm-pfm3/1.0",
                                               "Accept": "application/json"})
    last = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as fh:
                return json.loads(fh.read().decode("utf-8")), dict(fh.headers)
        except Exception as exc:  # noqa: BLE001 - a public API, retried
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"UniProt request failed after 5 attempts: {last}")


def fetch_reviewed_human(max_pages: int = 200, page: int = PAGE) -> tuple[list[dict], dict]:
    """Every reviewed human entry, paged through the search endpoint."""
    url = (
        f"{S.UNIPROT_SEARCH}?query={urllib.parse.quote(QUERY)}"
        f"&fields={FIELDS}&format=json&size={page}"
    )
    entries: list[dict] = []
    meta: dict = {"query": QUERY, "fields": FIELDS, "pages": 0}
    for _ in range(max_pages):
        payload, headers = _get(url)
        entries += payload.get("results", [])
        meta["pages"] += 1
        meta.setdefault("release", headers.get("X-UniProt-Release"))
        meta.setdefault("release_date", headers.get("X-UniProt-Release-Date"))
        meta.setdefault("total", headers.get("X-Total-Results"))
        link = headers.get("Link", "")
        nxt = re.search(r"<([^>]+)>;\s*rel=\"next\"", link)
        if not nxt:
            break
        url = nxt.group(1)
    meta["n_entries"] = len(entries)
    return entries, meta


def _index(entries: list[dict]) -> dict:
    """Three lookups over the reviewed set: Ensembl gene id, symbol, synonym."""
    by_ensg: dict[str, list[int]] = {}
    by_symbol: dict[str, list[int]] = {}
    by_synonym: dict[str, list[int]] = {}
    for i, e in enumerate(entries):
        for g in e.get("genes", []) or []:
            name = (g.get("geneName") or {}).get("value")
            if name:
                by_symbol.setdefault(str(name), []).append(i)
            for syn in g.get("synonyms", []) or []:
                val = syn.get("value")
                if val:
                    by_synonym.setdefault(str(val), []).append(i)
        for xref in e.get("uniProtKBCrossReferences", []) or []:
            if xref.get("database") != "Ensembl":
                continue
            for prop in xref.get("properties", []) or []:
                if prop.get("key") in ("GeneId", "geneId"):
                    ensg = str(prop.get("value", "")).split(".")[0]
                    if ensg.startswith("ENSG"):
                        by_ensg.setdefault(ensg, []).append(i)
    return {"ensg": by_ensg, "symbol": by_symbol, "synonym": by_synonym}


def _pick(entries: list[dict], idx: list[int], symbol: str) -> tuple[int, str]:
    """The tie-break, and which rule decided it."""
    uniq = sorted(set(idx))
    if len(uniq) == 1:
        return uniq[0], "single"
    named = [i for i in uniq if any(
        ((g.get("geneName") or {}).get("value") == symbol) for g in entries[i].get("genes", []) or []
    )]
    if len(named) == 1:
        return named[0], "primary_name"
    pool = named or uniq
    best = sorted(
        pool,
        key=lambda i: (
            -float(entries[i].get("annotationScore") or 0.0),
            -int((entries[i].get("sequence") or {}).get("length") or 0),
            str(entries[i].get("primaryAccession") or ""),
        ),
    )
    score_a = float(entries[best[0]].get("annotationScore") or 0.0)
    score_b = float(entries[best[1]].get("annotationScore") or 0.0)
    if score_a != score_b:
        return best[0], "annotation_score"
    len_a = int((entries[best[0]].get("sequence") or {}).get("length") or 0)
    len_b = int((entries[best[1]].get("sequence") or {}).get("length") or 0)
    return best[0], "length" if len_a != len_b else "accession"


def map_genes(genes: pd.DataFrame, entries: list[dict]) -> pd.DataFrame:
    """One row per gene: the accession, the sequence, and how it was found."""
    lookup = _index(entries)
    rows = []
    for gene_id, symbol in zip(genes["gene_id"].astype(str), genes["gene_name"].astype(str)):
        route, hits = "unmapped", []
        for name, table, keyval in (
            ("ensembl", lookup["ensg"], gene_id),
            ("symbol", lookup["symbol"], symbol),
            ("synonym", lookup["synonym"], symbol),
        ):
            hits = table.get(keyval, [])
            if hits:
                route = name
                break
        if route == "unmapped":
            rows.append({"gene_id": gene_id, "gene_name": symbol, "route": route,
                         "accession": "", "sequence": "", "length": 0,
                         "n_candidates": 0, "tie_break": ""})
            continue
        i, rule = _pick(entries, hits, symbol)
        e = entries[i]
        seq = (e.get("sequence") or {}).get("value", "")
        rows.append({
            "gene_id": gene_id,
            "gene_name": symbol,
            "route": route,
            "accession": str(e.get("primaryAccession") or ""),
            "sequence": seq,
            "length": len(seq),
            "n_candidates": len(set(hits)),
            "tie_break": rule,
        })
    return pd.DataFrame(rows)


def readout_union(run2_handoff: Path) -> pd.DataFrame:
    """Every read-out gene of either cell line, once."""
    frames = [
        pd.read_parquet(run2_handoff / "readout" / f"readout_genes_{line}.parquet")[
            ["gene_id", "gene_name"]
        ]
        for line in ("hepg2", "jurkat")
    ]
    out = pd.concat(frames, ignore_index=True).drop_duplicates("gene_id")
    return out.sort_values("gene_id").reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    from .. import paths  # noqa: PLC0415 - keeps the module importable without the tree

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/protein")
    ap.add_argument("--limit", type=int, default=0, help="map only the first N genes (smoke)")
    ap.add_argument("--max-pages", type=int, default=200)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    genes = readout_union(paths.run2_handoff())
    if args.limit:
        genes = genes.head(args.limit)
    t0 = time.time()
    entries, meta = fetch_reviewed_human(max_pages=args.max_pages)
    print(f"[uniprot] {len(entries)} reviewed human entries in {time.time() - t0:.0f}s "
          f"(release {meta.get('release')})", flush=True)
    table = map_genes(genes, entries)
    table.to_parquet(out / "gene_to_protein.parquet", index=False)

    counts = table["route"].value_counts().to_dict()
    report = {
        "written_by": "scfm_pfm.protein.uniprot",
        "uniprot": meta,
        "n_genes": int(table.shape[0]),
        "routes": {k: int(v) for k, v in counts.items()},
        "mapped_fraction": float((table["route"] != "unmapped").mean()),
        "tie_breaks": {k: int(v) for k, v in table["tie_break"].value_counts().to_dict().items()},
        "multi_candidate_genes": int((table["n_candidates"] > 1).sum()),
        "length": {
            "median": float(np.median(table.loc[table["length"] > 0, "length"])),
            "max": int(table["length"].max()),
            "over_esm2_context": int((table["length"] > S.ESM2_MAX_RESIDUES).sum()),
        },
        "unmapped_genes": sorted(table.loc[table["route"] == "unmapped", "gene_name"].tolist()),
        "duplicate_accessions": int(
            table.loc[table["accession"] != "", "accession"].duplicated().sum()
        ),
    }
    (out / "gene_to_protein_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "unmapped_genes"}, indent=2),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
