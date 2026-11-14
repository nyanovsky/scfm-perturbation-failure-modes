"""Build a writable bundle: run 3's read-only one, plus run 6's own blocks.

Run 3's bundle is an input to run 6 and stays one. ``install.py`` writes into a
``features/`` directory, and a bundle is shared by every arm scored against it,
so writing into run 3's own tree would change what runs 3 and 4 read --- their
numbers are published and cited, and a later run that edits their inputs makes
the citation false. Run 6 copies instead, and every copy is disposable.

**The copy is a symlink where it can be.** ``targets.npz``, ``coexpr.npz`` and
the gene tables are hundreds of megabytes, read-only during scoring, and staged
once per (model, layer, k) --- of which there are dozens. Copying them would
cost tens of gigabytes to no end. A filesystem that refuses the symlink falls
back to a copy; a *missing* source raises, because a bundle silently short one
``emb__*.npz`` drops a reference arm from the sweep with no error, and the
reference arms are how run 6's absolute scores are read at all.

⚠️ **Which blocks share a bundle is an execution knob and cannot reach a
score.** ``scope_rows("full")`` is every row of the cell line, and no arm reads
another arm's block, so a bundle holding one block and a bundle holding six
produce identical numbers for every arm in both. This is only true of the
``full`` scope: ``scope_rows("common")`` intersects the ``pert_index`` of every
model present, so installing a block there redefines the scored row set for
**every other arm in the bundle**. Run 6 therefore scores ``--scopes full``
only, ``common`` exists for AIDO.Cell comparability and run 6 has no AIDO arm,
and ``stage`` refuses to build a bundle for a caller that asked for ``common``
rather than letting the grouping quietly become a scientific choice.

The grouping exists because ``LineBundle`` puts every array on the scoring
device and this machine has 24 GB of VRAM, not because it means anything.

    python -m scfm_hidarm.stage --line hepg2 --blocks a.npz b.npz --work <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from . import config as C
from . import paths

#: what a run-6 scoring job needs out of run 3's bundle, beyond its own blocks.
#: ``targets.npz`` defines the scope rows, the folds and the block identity
#: check; ``coexpr.npz`` carries the ``coexp`` / ``coexp_node`` references and
#: ``reference.json`` the noise ceilings every score is read against.
BUNDLE_FILES = ("targets.npz", "coexpr.npz", "reference.json")
#: every gene table present, not a fixed list: run 6's reference arms include
#: ``<model>_emb`` for each of scGPT, scPRINT and TranscriptFormer, and the ESM2
#: control tables are in the same bundle. ``resolve_arms`` reads whatever is
#: there, so staging a subset would silently drop reference arms.
BUNDLE_GLOBS = ("emb__*.npz",)
#: staged by copy rather than symlink: small enough that the link saves nothing,
#: and a JSON that turns out to be a dangling link fails at ``json.loads`` with
#: no clue where it came from
COPY_ALWAYS = ("reference.json",)


def _place(src: Path, dst: Path) -> str:
    """Symlink ``src`` at ``dst``, falling back to a copy. Missing source raises."""
    if not src.exists():
        raise RuntimeError(
            f"bundle file missing: {src}; a bundle staged without it drops the arms that read "
            "it from the sweep with no error, and run 6's scores are only readable against "
            "its reference arms"
        )
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if src.name in COPY_ALWAYS:
        shutil.copy2(src, dst)
        return "copy"
    try:
        os.symlink(os.path.realpath(src), dst)
        return "symlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def stage(
    cell_line: str,
    block_paths: list[Path] | tuple[Path, ...],
    work_root: Path,
    scopes: tuple[str, ...] = C.SCOPES,
    bundle_root: Path | None = None,
    anchor: str = C.ANCHOR,
) -> Path:
    """One writable bundle for one cell line, holding exactly ``block_paths``.

    ``scopes`` is taken rather than assumed so that the refusal below is made
    against what the caller will actually pass run 3's runner, not against what
    this module hopes it will.
    """
    if "common" in scopes:
        raise RuntimeError(
            "run 6 scores the 'full' scope only. scope_rows('common') intersects the "
            "pert_index of every model present in the bundle, so installing a run-6 block "
            "would redefine the scored row set for every other arm staged beside it --- "
            "including the reference arms whose published values run 6 compares against. "
            "'common' exists for AIDO.Cell comparability and run 6 has no AIDO arm."
        )
    src = Path(bundle_root if bundle_root is not None else paths.run3_bundle_root()) / cell_line
    dst = Path(work_root) / cell_line
    (dst / "features").mkdir(parents=True, exist_ok=True)

    placed: dict[str, str] = {}
    for name in BUNDLE_FILES:
        placed[name] = _place(src / name, dst / name)
    n_emb = 0
    for pattern in BUNDLE_GLOBS:
        for p in sorted(src.glob(pattern)):
            placed[p.name] = _place(p, dst / p.name)
            n_emb += 1
    if n_emb == 0:
        raise RuntimeError(
            f"{src}: no {'/'.join(BUNDLE_GLOBS)} gene tables found; every <model>_emb reference "
            "arm would be missing from the sweep and run 6 would have no static-table reference"
        )

    for b in block_paths:
        b = Path(b)
        if "__" not in b.stem:
            raise RuntimeError(
                f"{b.name}: a features/*.npz name with no '__' raises ValueError inside "
                "LineBundle's constructor and kills EVERY arm for this cell line"
            )
        placed[b.name] = _place(b, dst / "features" / b.name)

    manifest = {
        "cell_line": cell_line,
        "anchor": anchor,
        "scopes": list(scopes),
        "source_bundle": str(src),
        "blocks": [Path(b).name for b in block_paths],
        "placed": placed,
        "note": (
            "which blocks share a bundle is an execution knob under the VRAM budget and "
            "cannot reach a score: scope_rows('full') is every row and no arm reads another "
            "arm's block"
        ),
    }
    (Path(work_root) / f"stage_{cell_line}.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"[stage] {cell_line}: {len(block_paths)} block(s), {n_emb} gene table(s) -> {dst}",
        flush=True,
    )
    return Path(work_root)


def plan_bundles(
    blocks: list[Path] | tuple[Path, ...],
    budget_bytes: int = C.STAGE_BYTES_BUDGET,
) -> list[list[str]]:
    """Group blocks into bundles that fit on the scoring device.

    Greedy over the given order, which is deterministic and therefore
    re-derivable from a manifest; an optimal packing would be neither, and would
    buy nothing, because **this is an execution knob and cannot reach a score**.
    ``scope_rows("full")`` is every row of the cell line and no arm reads another
    arm's block, so a bundle of one and a bundle of six give every arm in both
    the identical number. The only thing at stake is whether ``LineBundle``,
    which moves every array to the scoring device at construction, fits in the
    24 GB this machine has.

    On-disk size is the estimate: ``np.savez`` writes uncompressed, and the
    device tensor is a float32 copy of the same array, so the file is within a
    few per cent of what the block will occupy. A block larger than the whole
    budget is staged alone rather than refused --- refusing it would make a
    legitimately large ``k = 64`` block unscoreable for a reason that is purely
    about this machine.
    """
    if budget_bytes <= 0:
        raise ValueError(f"budget_bytes must be positive, got {budget_bytes}")
    out: list[list[str]] = []
    cur: list[str] = []
    cur_bytes = 0
    for b in blocks:
        p = Path(b)
        size = p.stat().st_size
        if cur and cur_bytes + size > budget_bytes:
            out.append(cur)
            cur, cur_bytes = [], 0
        cur.append(str(p))
        cur_bytes += size
    if cur:
        out.append(cur)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--line", required=True, choices=list(C.CELL_LINES))
    p.add_argument("--blocks", nargs="*", type=Path, default=[],
                   help="run-6 block npz files to place in the staged features/ directory")
    p.add_argument("--work", type=Path, default=None,
                   help="staging root (default: run 6's staging root)")
    p.add_argument("--scopes", default=",".join(C.SCOPES),
                   help="the scopes this bundle will be scored on; 'common' is refused")
    p.add_argument("--plan", action="store_true",
                   help="print the greedy bundle grouping under the VRAM budget and stop")
    a = p.parse_args(argv)

    if a.plan:
        for i, group in enumerate(plan_bundles(a.blocks)):
            print(f"[stage] bundle {i}: {len(group)} block(s)", flush=True)
            for g in group:
                print(f"    {g}", flush=True)
        return 0
    work = a.work if a.work is not None else paths.stage_root() / a.line
    paths.ensure(Path(work))
    stage(a.line, a.blocks, work, scopes=tuple(s for s in a.scopes.split(",") if s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
