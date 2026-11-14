"""The acceptance gate: every check that ran, and the number it checked.

Invocation::

    PYTHONPATH=src python -m scfm_probe.verify --all

This module computes nothing. It re-reads what the driver and the analysis wrote
and asserts that the record is internally consistent, because the two ways this
run could be wrong without looking wrong are both bookkeeping failures rather
than arithmetic ones: a block quietly at smoke coverage sitting in the production
namespace, and a control reported as passing when it never executed.

Three states, not two. A check that could not be computed is **ABSENT**, never a
failure and never a zero --- TranscriptFormer's C1 is unavailable because the
architecture never reads a gene's count at its own position, and collapsing that
into either bucket loses the distinction the whole run turns on. `n_absent` is
reported beside `n_passed` and `n_failed`, and the exit status counts only
failures.

A missing file fails. It never skips: an absent block and a block that was never
run are the same thing on disk, and only one of them is a result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import paths


class Checks:
    """Named checks whose detail prints the number checked, never a bare PASS."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, section: str, name: str, ok: bool | None, detail: str) -> bool:
        self.rows.append({"section": section, "check": name, "ok": ok, "detail": detail})
        tag = "ABSENT" if ok is None else ("PASS" if ok else "FAIL")
        print(f"[{tag}] {section}/{name}: {detail}", flush=True)
        return bool(ok)

    def summary(self) -> tuple[int, int, int]:
        return (
            sum(1 for r in self.rows if r["ok"] is True),
            sum(1 for r in self.rows if r["ok"] is False),
            sum(1 for r in self.rows if r["ok"] is None),
        )


def verify_block(block: Path, ch: Checks) -> dict:
    """Re-read one block's manifest and assert its arrays match what it claims."""
    sec = "/".join(block.parts[-3:])
    mpath = block / "manifest.json"
    if not ch.add(sec, "manifest_exists", mpath.exists(), str(mpath)):
        return {}
    m = json.loads(mpath.read_text())

    cov = m.get("coverage", {})
    ch.add(
        sec,
        "coverage_is_full",
        cov.get("mode") == "full",
        f"{cov.get('n_perturbed')} of {cov.get('n_in_scope')} in-scope perturbations "
        f"(mode {cov.get('mode')!r}); a subset block is a smoke block and no number "
        "may be read from it",
    )

    layers = [int(x) for x in m.get("layers", [])]
    want = (C.EMBEDDING_LAYER,) + C.sweep_layers(m["model"])
    ch.add(
        sec,
        "layers_match_config",
        sorted(layers) == sorted(want),
        f"block carries {sorted(layers)}; config.sweep_layers({m['model']!r}) plus the "
        f"embedding control gives {sorted(want)}",
    )

    n_p = int(cov.get("n_perturbed", 0))
    n_v = len(m.get("value_grid_cp10k", []))
    d = C.D_MODEL[m["model"]]
    for l in layers:
        ld = block / f"layer={l}"
        if not ch.add(sec, f"layer{l}/exists", ld.is_dir(), str(ld.name)):
            continue
        pool = np.load(ld / "pool.npz")["pool"]
        ch.add(
            sec,
            f"layer{l}/pool_shape",
            pool.shape == (n_p, n_v, C.N_POSITION_HALVES, d),
            f"{pool.shape} against the manifest's ({n_p}, {n_v}, "
            f"{C.N_POSITION_HALVES}, {d})",
        )
        ch.add(
            sec,
            f"layer{l}/pool_finite",
            bool(np.isfinite(pool).all()),
            f"{int((~np.isfinite(pool)).sum())} non-finite entries of {pool.size}",
        )

    # C2 is the alignment proof for models whose value enters the residual stream:
    # if reassembly were off by one, a neighbouring position would move at layer 0.
    zero = block / f"layer={C.EMBEDDING_LAYER}" / "pool.npz"
    if zero.exists():
        p0 = np.load(zero)["pool"]
        ch.add(
            sec,
            "C2/layer0_pooled_delta_exactly_zero",
            float(np.abs(p0).max()) == 0.0,
            f"max |pooled off-target delta-h| at the post-embedding layer is "
            f"{float(np.abs(p0).max()):.6g} over {n_p} genes x {n_v} values",
        )

    for c in m.get("verification", {}).get("checks", []):
        ch.add(sec, "driver/" + c["check"], c["ok"], c["detail"])
    return m


def verify_summary(ch: Checks) -> None:
    """Assert the summary table matches the frozen contract, and that it is honest."""
    path = paths.summary_table()
    if not ch.add("summary", "exists", path.exists(), str(path)):
        return
    df = pd.read_parquet(path)
    ch.add(
        "summary",
        "columns_match_contract",
        list(df.columns) == list(C.SUMMARY_COLUMNS),
        f"{list(df.columns)} against config.SUMMARY_COLUMNS",
    )
    bad = sorted(set(df.statistic) - set(C.SUMMARY_STATISTICS))
    ch.add("summary", "statistics_in_vocabulary", not bad, f"unknown statistics: {bad or 'none'}")
    bad_s = sorted(set(df.split) - set(C.SUMMARY_SPLITS))
    ch.add("summary", "splits_in_vocabulary", not bad_s, f"unknown splits: {bad_s or 'none'}")

    # Every DECISION statistic must be readable against a null in its own cell: a
    # score with no band is a number without the thing that makes it a result. The
    # S1 magnitudes are exempt by name rather than by omission --- permuting
    # perturbation labels does not change a norm, so a shuffled-label draw there
    # would be the statistic itself wearing a null's name. Exempting them silently
    # would be indistinguishable from forgetting to compute their nulls.
    scored = df[df.score.notna()]
    decide = scored[scored.statistic.isin(C.NULL_BEARING_STATISTICS)]
    ch.add(
        "summary",
        "every_decision_score_has_a_null",
        bool(decide.null_max.notna().all()),
        f"{int(decide.null_max.isna().sum())} of {len(decide)} scored rows in "
        f"{list(C.NULL_BEARING_STATISTICS)} carry no null_max",
    )
    exempt = scored[~scored.statistic.isin(C.NULL_BEARING_STATISTICS)]
    ch.add(
        "summary",
        "exempt_statistics_are_the_declared_ones",
        set(exempt.statistic) <= {
            "propagation_ratio",
            "propagation_ratio_per_position",
            "propagation_off_target_norm",
        },
        f"{len(exempt)} scored rows carry no null_max, in "
        f"{sorted(set(exempt.statistic))}; these are magnitudes, not decisions, and "
        "no verdict is read from them alone",
    )
    absent = df[df.score.isna()]
    ch.add(
        "summary",
        "every_absence_states_its_reason",
        bool((absent.detail.fillna("").str.len() > 0).all()),
        f"{int((absent.detail.fillna('').str.len() == 0).sum())} of {len(absent)} "
        "rows with no score carry no detail saying why",
    )
    for model in sorted(df.model.unique()):
        sub = df[(df.model == model) & (df.statistic == "retrieval_top1")
                 & (df.split == "split_half") & (df.layer == C.EMBEDDING_LAYER)]
        if len(sub):
            # The invariant is DEGENERACY, not score-versus-null. Every layer-0
            # signature is exactly zero, so its top-1 is entirely tie-break noise and
            # so is a five-draw null of the same block: comparing them compares two
            # samples of one noise process, and a 3-of-524 hit against a 1-of-524
            # null max reads as an exceedance when it is Poisson scatter. What must
            # hold is that nothing was decodable there at all.
            n = int(sub.n_perturbations.max())
            undef = int(sub.n_undefined.max())
            worst = float(sub.score.max())
            ch.add(
                "summary",
                f"{model}/layer0_signatures_all_degenerate",
                undef == n,
                f"{undef} of {n} layer-0 split-half signatures are degenerate; the "
                f"post-embedding state cannot have moved at another gene's token, so "
                f"anything decodable there is a leak. Top-1 reads {worst:.6g} "
                f"({round(worst * n)} of {n}) against a chance of 1/{n} = "
                f"{1.0 / n:.6g}, which is tie-break scatter on an all-zero block.",
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="every block plus the summary table")
    ap.add_argument("--block", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    ch = Checks()
    blocks = (
        [Path(args.block)]
        if args.block
        else sorted((paths.data_root() / "probe").glob("*/*/anchor=*"))
    )
    if not blocks:
        raise RuntimeError(
            f"no probe blocks under {paths.data_root() / 'probe'}; a rollup over nothing "
            "would report a clean bill of health for a run that never executed"
        )
    for b in blocks:
        verify_block(b, ch)
    if args.all or not args.block:
        verify_summary(ch)

    ok, failed, absent = ch.summary()
    out = paths.ensure(Path(args.out) if args.out else paths.verification_dir())
    lines = [
        f"[{'ABSENT' if r['ok'] is None else ('PASS' if r['ok'] else 'FAIL')}] "
        f"{r['section']}/{r['check']}: {r['detail']}"
        for r in ch.rows
    ]
    tail = f"\n\n{ok} passed, {failed} failed, {absent} absent of {len(ch.rows)} checks\n"
    (out / "verify_all.txt").write_text("\n".join(lines) + tail)
    (out / "verify_all.json").write_text(
        json.dumps({"checks": ch.rows, "n_passed": ok, "n_failed": failed,
                    "n_absent": absent}, indent=2, default=str)
    )
    print(tail.strip(), flush=True)
    print(f"[verify] wrote {out}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
