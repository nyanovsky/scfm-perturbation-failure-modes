"""Pull run 1 / run 2 artifacts into a job from a presigned-link manifest.

A fresh on-demand job has no credential for the artifact store and no mount of a
sibling experiment's outputs, so the one supported way to read run 1's effect
matrices and run 2's feature blocks inside a job is a short-lived signed URL
minted on the pod. ``results/handoff/urls/artifact_urls.json`` maps each
``artifact://`` ref to such a URL; this module downloads them, verifies the
byte count against the manifest, and lays them out under a local mirror whose
paths mirror the refs so downstream code addresses artifacts by ref, not URL.

The links expire in an hour. They are never committed: the manifest is written
by the pod immediately before the job is submitted, travels in the untracked
part of the worktree snapshot, and is worthless afterwards.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def ref_to_relpath(ref: str) -> str:
    """``artifact://tenant/experiments/<eid>/a/b.npz`` -> ``<eid>/a/b.npz``."""
    body = ref.split("://", 1)[1]
    parts = body.split("/")
    if "experiments" in parts:
        i = parts.index("experiments")
        return "/".join(parts[i + 1 :])
    return "/".join(parts[1:])


def _download_one(url: str, dest: Path, attempts: int = 4) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    total = 0
    declared = 0
    start = time.time()
    last_error = ""
    for attempt in range(attempts):
        headers = {"User-Agent": "scfm-pfm3/1.0"}
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
                    block = resp.read(16 << 20)
                    if not block:
                        break
                    out.write(block)
                    total += len(block)
        except (OSError, urllib.error.URLError) as exc:  # noqa: PERF203
            last_error = f"{type(exc).__name__}: {exc}"
            total = tmp.stat().st_size if tmp.exists() else 0
            time.sleep(2 * (attempt + 1))
            continue
        total = tmp.stat().st_size
        if not declared or total >= declared:
            break
    if declared and total != declared:
        raise RuntimeError(f"{dest.name}: got {total} of {declared} bytes ({last_error})")
    tmp.replace(dest)
    return {
        "path": str(dest),
        "size_bytes": total,
        "seconds": round(time.time() - start, 1),
    }


def download_manifest(manifest_path: Path, mirror: Path, workers: int = 8) -> dict:
    """Download every ref in the manifest into ``mirror``; return a receipt."""
    manifest = json.loads(Path(manifest_path).read_text())
    urls: dict[str, str] = manifest["urls"] if "urls" in manifest else manifest
    mirror.mkdir(parents=True, exist_ok=True)
    jobs = []
    for ref, url in urls.items():
        dest = mirror / ref_to_relpath(ref)
        jobs.append((ref, url, dest))

    receipts: dict[str, dict] = {}
    errors: dict[str, str] = {}

    def run(item):
        ref, url, dest = item
        if dest.exists() and dest.stat().st_size > 0:
            return ref, {"path": str(dest), "size_bytes": dest.stat().st_size, "reused": True}
        try:
            return ref, _download_one(url, dest)
        except Exception as exc:  # noqa: BLE001 - reported per ref, not raised
            return ref, {"error": f"{type(exc).__name__}: {exc}"}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for ref, rec in pool.map(run, jobs):
            if "error" in rec:
                errors[ref] = rec["error"]
            else:
                receipts[ref] = rec
            print(
                f"  fetched {ref_to_relpath(ref)} "
                f"{rec.get('size_bytes', 0) / 1e6:.1f} MB {rec.get('error', '')}",
                flush=True,
            )
    if errors:
        raise RuntimeError(f"{len(errors)} artifact downloads failed: {json.dumps(errors)[:2000]}")
    total = sum(r["size_bytes"] for r in receipts.values())
    return {
        "n_objects": len(receipts),
        "total_bytes": int(total),
        "mirror": str(mirror),
        "refs": sorted(receipts),
    }


def sha256_head(path: Path, n_bytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(n_bytes))
    return h.hexdigest()
