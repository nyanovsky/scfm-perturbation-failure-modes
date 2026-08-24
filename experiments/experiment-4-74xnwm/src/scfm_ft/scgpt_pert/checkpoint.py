"""Fetch the published scGPT-human release, at the revision runs 1-3 used."""

from __future__ import annotations

import json
from pathlib import Path

from .. import config as C
from ..sibling import run3

HF_BASE = "https://huggingface.co"


def file_url(name: str) -> str:
    return f"{HF_BASE}/{C.SCGPT_REPO}/resolve/{C.SCGPT_REVISION}/{name}"


def fetch(cache: Path) -> dict:
    """Download vocab.json, args.json and best_model.pt into ``cache``."""
    download = run3("data.sources").download
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    meta = {}
    for name in C.SCGPT_FILES:
        dest = cache / name
        info = download(file_url(name), dest)
        meta[name] = {
            "size_bytes": info["size_bytes"],
            "md5": info["md5"],
            "reused_existing": info.get("reused_existing", False),
        }
    meta["repo"] = C.SCGPT_REPO
    meta["revision"] = C.SCGPT_REVISION
    (cache / "download.json").write_text(json.dumps(meta, indent=2))
    return meta
