"""Put one model's checkpoint on local disk, in the layout its adapter expects.

Each per-model GPU job fetches its own checkpoint: on this fabric a finished
job's outputs are moved out of its volume, so there is no shared cache to mount.
The layout is exactly the one the vocabulary build already produced and the
adapters already read, so this module reuses those download/extract paths rather
than restating them.

    python -m scfm_pfm.models.fetch --model scgpt --cache _ckpt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import paths
from ..data.sources import download
from . import sources as S


def ensure(model: str, cache: Path) -> dict:
    """Download everything ``model``'s adapter loads from ``cache``."""
    from ..genes import vocabularies as V

    cache = paths.ensure(cache)
    meta: dict[str, dict] = {}
    if model == "scgpt":
        V.scgpt_vocabulary(cache)  # vocab.json, ordered by the model's own ids
        for key in ("args", "weights"):
            meta[key] = download(
                S.hf_url(S.SCGPT, key), cache / "scgpt" / S.SCGPT["files"][key]
            )
    elif model == "aido_cell":
        V.aido_vocabulary(cache)  # fixed gene order + the two HF modeling files
        for key in ("config", "weights"):
            meta[key] = download(
                S.hf_url(S.AIDO_CELL, key), cache / "aido_cell" / S.AIDO_CELL["files"][key]
            )
    elif model == "scprint":
        V.scprint_vocabulary(cache)  # medium-v1.5.ckpt (gene list lives inside it)
    elif model == "transcriptformer":
        V.transcriptformer_vocabulary(cache)  # tf_sapiens.tar.gz, extracted
    else:
        raise SystemExit(f"unknown model {model!r}; expected one of {S.MODELS}")

    present = sorted(
        (str(p.relative_to(cache)), p.stat().st_size)
        for p in cache.rglob("*")
        if p.is_file() and p.stat().st_size > 1024
    )
    print(
        json.dumps(
            {
                "model": model,
                "cache": str(cache),
                "downloaded": {
                    k: {"size_bytes": v.get("size_bytes"), "md5": v.get("md5")}
                    for k, v in meta.items()
                },
                "files_on_disk": present[:40],
                "n_files_on_disk": len(present),
            },
            indent=2,
        ),
        flush=True,
    )
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=S.MODELS)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args(argv)
    cache = Path(args.cache) if args.cache else paths.raw_dir() / "checkpoints"
    ensure(args.model, cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
