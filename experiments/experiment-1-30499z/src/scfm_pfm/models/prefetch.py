"""Fetch the remaining model weights into the job volume.

The vocabulary job already pulls scPRINT's Lightning checkpoint and
TranscriptFormer's weights tarball. This adds scGPT's and AIDO.Cell's weights so
that a single retained volume carries all four checkpoints, and each per-model
GPU job mounts it read-only instead of re-downloading ~2.5 GB.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import paths
from ..data.sources import download
from . import sources as S


def prefetch(cache: Path) -> dict:
    meta = {}
    for key in ("weights", "args"):
        path = cache / "scgpt" / S.SCGPT["files"][key]
        meta[f"scgpt/{key}"] = download(S.hf_url(S.SCGPT, key), path)
    for key in ("config", "weights"):
        path = cache / "aido_cell" / S.AIDO_CELL["files"][key]
        meta[f"aido_cell/{key}"] = download(S.hf_url(S.AIDO_CELL, key), path)
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args(argv)
    cache = Path(args.cache) if args.cache else paths.raw_dir() / "checkpoints"
    meta = prefetch(paths.ensure(cache))
    print(json.dumps({k: {"size_bytes": v["size_bytes"], "md5": v["md5"]} for k, v in meta.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
