"""Download all four checkpoints onto one job volume, in the layout the adapters read.

On this fabric a finished job's volume can be mounted read-only by a later job,
so the four checkpoints are fetched exactly once here and every GPU job reads
them from that mount instead of pulling ~3.5 GB again. Nothing in this module
imports torch, numpy or a model package: it is downloads, a tar extraction and
checksums, so it runs in a bare Python image.

    python3 -m scfm_pfm.models.prefetch --cache _ckpt
"""

from __future__ import annotations

import argparse
import json
import tarfile
import time
from pathlib import Path

from .. import paths
from ..data.sources import download
from . import sources as S


def fetch_scgpt(cache: Path) -> dict:
    out = {}
    for key in ("vocab", "args", "weights"):
        out[key] = download(S.hf_url(S.SCGPT, key), cache / "scgpt" / S.SCGPT["files"][key])
    return out


def fetch_aido(cache: Path) -> dict:
    out = {}
    for key in ("config", "weights"):
        out[key] = download(
            S.hf_url(S.AIDO_CELL, key), cache / "aido_cell" / S.AIDO_CELL["files"][key]
        )
    tgz = cache / "aido_cell" / "modelgenerator_sdist.tar.gz"
    out["modeling_sdist"] = download(S.AIDO_CELL["gene_order_sdist"], tgz)
    src = cache / "aido_cell" / "cellfoundation"
    src.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz, "r:gz") as tf:
        for name in S.AIDO_CELL["modeling_members"]:
            member = tf.extractfile(name)
            if member is None:
                raise RuntimeError(f"{name} missing from the ModelGenerator sdist")
            (src / Path(name).name).write_bytes(member.read())
        order = tf.extractfile(S.AIDO_CELL["gene_order_member"])
        if order is not None:
            (cache / "aido_cell" / "gene_order.tsv").write_bytes(order.read())
    return out


def fetch_scprint(cache: Path) -> dict:
    return {
        "weights": download(
            S.hf_url(S.SCPRINT, "weights"), cache / "scprint" / S.SCPRINT["files"]["weights"]
        )
    }


def fetch_transcriptformer(cache: Path) -> dict:
    tgz = cache / "transcriptformer" / "tf_sapiens.tar.gz"
    meta = download(S.TRANSCRIPTFORMER["url"], tgz)
    extract_dir = cache / "transcriptformer" / "tf_sapiens"
    if not (extract_dir / "tf_sapiens" / "config.json").exists() and not (
        extract_dir / "config.json"
    ).exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tgz, "r:gz") as tf:
            tf.extractall(extract_dir)  # noqa: S202 - trusted CZI release tarball
    return {"weights": meta}


FETCHERS = {
    "scgpt": fetch_scgpt,
    "aido_cell": fetch_aido,
    "scprint": fetch_scprint,
    "transcriptformer": fetch_transcriptformer,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=None, help="directory the checkpoints land in")
    ap.add_argument("--models", nargs="*", default=list(paths.MODELS))
    ap.add_argument("--manifest-out", default=None)
    args = ap.parse_args(argv)

    cache = paths.ensure(Path(args.cache) if args.cache else paths.repo_dir() / paths.CKPT_DIRNAME)
    started = time.time()
    manifest: dict[str, object] = {"cache": str(cache), "models": {}}
    for model in args.models:
        t0 = time.time()
        print(f"[prefetch] {model}", flush=True)
        meta = FETCHERS[model](cache)
        manifest["models"][model] = {  # type: ignore[index]
            key: {
                "url": val.get("url"),
                "size_bytes": val.get("size_bytes"),
                "md5": val.get("md5"),
                "size_verified": val.get("size_verified"),
            }
            for key, val in meta.items()
        }
        print(f"[prefetch] {model} done in {time.time() - t0:.0f}s", flush=True)

    files = sorted(
        (str(p.relative_to(cache)), p.stat().st_size)
        for p in cache.rglob("*")
        if p.is_file() and p.stat().st_size > 4096
    )
    manifest["files_on_disk"] = files
    manifest["total_bytes"] = sum(size for _, size in files)
    manifest["seconds"] = round(time.time() - started, 1)

    out = Path(args.manifest_out) if args.manifest_out else paths.data_root() / "checkpoints"
    paths.ensure(out)
    (out / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "files_on_disk"}, indent=2), flush=True)
    print(f"[prefetch] {len(files)} files, {manifest['total_bytes'] / 1e9:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
