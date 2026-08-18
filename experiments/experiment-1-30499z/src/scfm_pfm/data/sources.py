"""Raw input locations and a streaming downloader that records size + md5."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
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
        complete = meta.get("size_verified") or meta.get("md5_matches_geo")
        if meta.get("size_bytes") == dest.stat().st_size and complete:
            meta["reused_existing"] = True
            return meta

    tmp = dest.with_suffix(dest.suffix + ".part")
    start = time.time()
    total = 0
    declared = 0
    attempts = 0
    # A truncated read looks exactly like a clean EOF, so a short stream would
    # otherwise be recorded as a complete file. Resume from the byte offset and
    # only accept the file once its length matches the declared Content-Length.
    while attempts < 4:
        attempts += 1
        headers = {"User-Agent": "scfm-pfm/1.0"}
        if total:
            headers["Range"] = f"bytes={total}-"
        req = urllib.request.Request(url, headers=headers)
        mode = "ab" if total else "wb"
        try:
            with urllib.request.urlopen(req, timeout=300) as resp, open(tmp, mode) as out:
                got = int(resp.headers.get("Content-Length", 0) or 0)
                if not declared:
                    declared = got
                elif resp.status == 206:
                    declared = total + got
                while True:
                    block = resp.read(chunk)
                    if not block:
                        break
                    out.write(block)
                    total += len(block)
                    if declared:
                        print(
                            f"    {dest.name}: {total / 1e9:.2f}/{declared / 1e9:.2f} GB "
                            f"({100.0 * total / declared:.1f}%) "
                            f"{total / 1e6 / max(time.time() - start, 1e-6):.0f} MB/s",
                            flush=True,
                        )
        except (OSError, urllib.error.URLError) as exc:  # noqa: PERF203
            print(f"    {dest.name}: transfer error after {total} bytes: {exc}", flush=True)
            total = tmp.stat().st_size if tmp.exists() else 0
            continue
        total = tmp.stat().st_size
        if not declared or total >= declared:
            break
        print(f"    {dest.name}: short by {declared - total} bytes, resuming", flush=True)
    if declared and total != declared:
        raise RuntimeError(
            f"{dest.name}: got {total} bytes, server declared {declared} after {attempts} attempts"
        )
    digest = hashlib.md5()  # noqa: S324 - matching GEO's published checksums
    with open(tmp, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    os.replace(tmp, dest)
    meta = {
        "url": url,
        "path": str(dest),
        "size_bytes": total,
        "declared_size_bytes": declared,
        "size_verified": bool(declared) and total == declared,
        "transfer_attempts": attempts,
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
