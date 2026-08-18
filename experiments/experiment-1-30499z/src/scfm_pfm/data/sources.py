"""Raw input locations and a streaming downloader that records size + md5."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

GEO_SERIES = "GSE264667"
GEO_SUPPL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/"
)
H5AD_FILES = {
    "hepg2": "GSE264667_hepg2_raw_singlecell_01.h5ad",
    "jurkat": "GSE264667_jurkat_raw_singlecell_01.h5ad",
}
FILELIST_URL = GEO_SUPPL + "filelist.txt"

REFERENCE_REPO = "tahoebio/replogle-nadig-de-rhaister"
REFERENCE_FILES = {
    "pdex": "pdex/all_pdex.parquet",
    "cell_eval": "cell_eval/all_delta.parquet",
    "feature_names": "definition/feature_names.json",
}


def h5ad_url(cell_line: str) -> str:
    return GEO_SUPPL + H5AD_FILES[cell_line]


def geo_md5_table() -> dict[str, str]:
    """GEO's own md5 table for the series supplementary files (best effort)."""
    try:
        with urllib.request.urlopen(FILELIST_URL, timeout=120) as fh:
            text = fh.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - the checksum comparison is a nicety
        return {}
    table: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and len(parts[1]) == 32:
            table[parts[0].strip()] = parts[1].strip()
        elif len(parts) >= 3 and len(parts[2]) == 32:
            table[parts[0].strip()] = parts[2].strip()
    return table


def download(url: str, dest: Path, chunk: int = 32 << 20) -> dict:
    """Download ``url`` to ``dest`` unless already complete; return provenance."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    meta_path = dest.with_suffix(dest.suffix + ".meta.json")
    if dest.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("size_bytes") == dest.stat().st_size:
            meta["reused_existing"] = True
            return meta

    req = urllib.request.Request(url, headers={"User-Agent": "scfm-pfm/1.0"})
    digest = hashlib.md5()  # noqa: S324 - matching GEO's published checksums
    start = time.time()
    total = 0
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=300) as resp, open(tmp, "wb") as out:
        declared = int(resp.headers.get("Content-Length", 0) or 0)
        while True:
            block = resp.read(chunk)
            if not block:
                break
            out.write(block)
            digest.update(block)
            total += len(block)
            if declared:
                pct = 100.0 * total / declared
                print(
                    f"    {dest.name}: {total / 1e9:.2f}/{declared / 1e9:.2f} GB "
                    f"({pct:.1f}%) {total / 1e6 / max(time.time() - start, 1e-6):.0f} MB/s",
                    flush=True,
                )
    os.replace(tmp, dest)
    meta = {
        "url": url,
        "path": str(dest),
        "size_bytes": total,
        "md5": digest.hexdigest(),
        "seconds": round(time.time() - start, 1),
        "reused_existing": False,
    }
    geo = geo_md5_table().get(dest.name)
    if geo:
        meta["geo_published_md5"] = geo
        meta["md5_matches_geo"] = geo == meta["md5"]
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta
