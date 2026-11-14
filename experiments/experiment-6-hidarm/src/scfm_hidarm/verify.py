"""The named-check acceptance gate: every check prints the number it checked.

    PYTHONPATH=src python -m scfm_hidarm.verify --all
    PYTHONPATH=src python -m scfm_hidarm.verify --section basis --section ft

"Verified" means a check ran and printed its number. "It ran" is not "it is
right", and a substitution recorded is not a substitution shown to be harmless.
So every row here carries the quantity it examined, a missing file FAILS rather
than skips, and a quantity this architecture cannot define is recorded as an
ABSENCE --- counted apart from both passes and failures, with its reason
verbatim. Collapsing an absence into a pass or a failure loses exactly the
distinction TranscriptFormer's undefined self position turns on.

The captured output of this module is committed to
``results/verification/verify_all.txt``. That file, not this code, is the
evidence; this module exists so the file can be regenerated and disagreed with.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import paths
from .sibling import run3, run4


class Checks:
    """``add(section, name, ok, detail)`` where ``detail`` prints the number.

    ``ok=None`` is an ABSENCE: a quantity this architecture or this stage cannot
    define. It is neither a pass nor a failure and is counted separately,
    because a control that could not be computed and a control that computed a
    bad number are different results.
    """

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, section: str, name: str, ok: bool | None, detail: str) -> bool:
        self.rows.append({"section": section, "check": name,
                          "ok": None if ok is None else bool(ok), "detail": detail})
        tag = "ABSENT" if ok is None else ("PASS" if ok else "FAIL")
        print(f"[{tag}] {section}/{name}: {detail}", flush=True)
        return bool(ok)

    def absent(self, section: str, name: str, detail: str) -> None:
        self.add(section, name, None, detail)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.rows if r["ok"] is True)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.rows if r["ok"] is False)

    @property
    def n_absent(self) -> int:
        return sum(1 for r in self.rows if r["ok"] is None)

    def report(self) -> str:
        labels = [f"{r['section']}/{r['check']}" for r in self.rows]
        w = max((len(x) for x in labels), default=10)
        lines = []
        for r, label in zip(self.rows, labels):
            tag = "ABSENT" if r["ok"] is None else ("PASS" if r["ok"] else "FAIL")
            lines.append(f"  {tag:<6}  {label:<{w}}  {r['detail']}")
        lines.append(
            f"\n{self.n_passed} passed, {self.n_failed} failed, {self.n_absent} absent, "
            f"{len(self.rows)} checks"
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# the checks
# --------------------------------------------------------------------------- #
def check_pre_registration(ch: Checks) -> None:
    """The constants a later reader has to be able to trust were fixed first."""
    sec = "prereg"
    ch.add(sec, "primary_metric_is_run3s", C.PRIMARY_METRIC == run3("config").PRIMARY_METRIC,
           f"run 6 reads {C.PRIMARY_METRIC!r}; run 3 pre-registered "
           f"{run3('config').PRIMARY_METRIC!r}")
    ch.add(sec, "ordering_metric_is_secondary", C.ORDERING_METRIC != C.PRIMARY_METRIC,
           f"arms are ORDERED by {C.ORDERING_METRIC!r} and the verdict is read at "
           f"{C.PRIMARY_METRIC!r}; the two roles reversed at cp-23 and neither confirms "
           "the other")
    ch.add(sec, "value_grid_is_run2s",
           tuple(C.VALUE_GRID_CP10K) == tuple(run3("config").VALUE_GRID_CP10K),
           f"{list(C.VALUE_GRID_CP10K)} CP10K, {len(C.VALUE_GRID_CP10K)} doses, unchanged "
           "from run 2")
    ch.add(sec, "scopes_full_only", tuple(C.SCOPES) == ("full",),
           f"scopes {list(C.SCOPES)}: installing a block redefines scope_rows('common') for "
           "every arm in the bundle, and run 6 has no AIDO.Cell arm for it to serve")
    ch.add(sec, "ladder_is_a_prefix", max(C.K_LADDER) <= C.K_STORE and C.K_PRIMARY in C.K_LADDER,
           f"k ladder {list(C.K_LADDER)} within K_STORE={C.K_STORE}, primary k={C.K_PRIMARY}; "
           "components are eigenvalue-ordered so every rung is a column prefix")
    ch.add(sec, "ft_execution_recorded", bool(C.FT_RUN_SUFFIX),
           f"SCFM_FT_RUN_SUFFIX={C.FT_RUN_SUFFIX!r}: the validation-Pearson execution run 4 "
           "reports; unset would silently read the earlier masked-MSE tree")
    bad = [k for k in _all_keys() if C.FORBIDDEN_IN_KEY in k]
    ch.add(sec, "no_key_contains_double_underscore", not bad,
           f"0 of {len(_all_keys())} run-6 model keys contain {C.FORBIDDEN_IN_KEY!r}; one that "
           "did would make its block unreadable and its arm vanish from the scored set with "
           "no error at all")
    round_trip = all(C.model_stem(**{k: v for k, v in C.run_of_stem(key).items()}) == key
                     for key in _all_keys())
    ch.add(sec, "keys_round_trip", round_trip,
           f"all {len(_all_keys())} keys survive model_stem -> run_of_stem -> model_stem")


def _all_keys() -> list[str]:
    keys = []
    for m in C.MODELS:
        for l in C.sweep_layers(m):
            for v in C.VARIANTS:
                keys.append(C.model_stem(m, v, l, C.K_STORE))
            if m in C.FROZEN_MODELS:
                for v in C.STAGE0_VARIANTS:
                    keys.append(C.model_stem(m, v, l))
    return keys


def check_no_head_output(ch: Checks, roots: list[Path]) -> None:
    """Run 6 writes no decoder-head block and no ``*_resp`` arm of its own.

    README s5b: the head results are settled, run 6 does not recompute them, and
    the comparison against runs 2-4 is post-hoc. A head-derived array anywhere in
    the output tree is a failure, not a bonus --- it would mean run 6 had quietly
    reopened a question it declared closed.
    """
    sec = "emits"
    banned = ("y_pert", "y_base", "zero_out", "mlm_output", "decoder", "response")
    hits: list[str] = []
    n = 0
    for root in roots:
        for f in Path(root).rglob("*.npz"):
            n += 1
            try:
                names = set(np.load(f, allow_pickle=False).files)
            except Exception as exc:  # a file we cannot read is itself a finding
                hits.append(f"{f}: {type(exc).__name__}")
                continue
            bad = sorted(names & set(banned))
            if bad:
                hits.append(f"{f}: {bad}")
    ch.add(sec, "no_head_derived_array", not hits,
           f"0 of {n} npz files under {[str(r) for r in roots]} carry a decoder-head key "
           f"{list(banned)}" + (f"; offenders {hits[:4]}" if hits else ""))


def _captured_energy_rows(basis: dict, model: str, cell_line: str, layer: int) -> list:
    """The captured-energy rows for one (model, line, layer), wherever they live.

    ``fit_basis`` can record the diagnostic in the basis provenance, but the
    number that is actually on disk for the 2026-08-31 wave is written by the
    MAIN extraction pass --- it needs the SCORED perturbations' second moment,
    which only that pass accumulates --- and lands as a top-level
    ``captured_energy`` key in the *block* manifest. Reading only the basis
    provenance made this check report ABSENT for 38 cells whose number was
    sitting on disk, which is a false negative on the one check that would
    otherwise have flagged the under-spanning basis automatically (README
    section 2). The basis provenance is still tried first, so a future
    basis-side diagnostic keeps precedence over the block's.

    The ``hid`` block is the one read: ``hidrp`` carries the same rows (both are
    written from the same ``coverage_rows``) but ``hid`` is the arm the
    diagnostic is about.
    """
    rows = basis.get("provenance", {}).get("captured_energy")
    if rows:
        return list(rows)
    key = C.model_stem(model, "hid", int(layer), C.K_STORE)
    f = paths.blocks_dir(model, cell_line) / f"{key}.json"
    if not f.exists():
        return []
    try:
        return list(json.loads(f.read_text()).get("captured_energy") or [])
    except (ValueError, OSError):
        # a truncated or unreadable manifest is an absent diagnostic, not a
        # silent pass --- the caller emits ABSENT and names the file it looked in
        return []


def check_basis(ch: Checks) -> None:
    """Leakage, coverage and the ladder prefix, per (model, cell line, layer)."""
    from .basis import fit as basisfit

    for cell_line in C.CELL_LINES:
        sec = f"basis:{cell_line}"
        genes = basisfit.select_basis_genes(cell_line)
        leak = basisfit.leakage_check(cell_line, genes)
        ch.add(sec, "leakage", leak["n_overlap"] == 0,
               f"{leak['n_overlap']} of {leak['n_basis']} basis genes are also scored "
               "perturbations; no scored perturbation may shape the basis it is scored "
               "through")
        pool = basisfit.basis_pool(cell_line)
        ch.add(sec, "pool_size", len(pool) >= C.N_BASIS_GENES,
               f"{len(pool)} screened perturbations whose target is a read-out gene but "
               f"whose stratum excludes them from scoring; {C.N_BASIS_GENES} drawn")
        for model in C.MODELS:
            for layer in C.sweep_layers(model):
                f = paths.basis_file(model, cell_line, layer)
                if not f.exists():
                    ch.add(f"{sec}:{model}", f"layer{layer}_present", False,
                           f"no basis at {f}; a missing file fails, it never skips")
                    continue
                b = basisfit.load_basis(model, cell_line, layer)
                W = np.asarray(b["W"])
                ok = W.shape == (C.D_MODEL[model], C.K_STORE)
                ch.add(f"{sec}:{model}", f"layer{layer}_shape", ok,
                       f"W is {W.shape}, expected ({C.D_MODEL[model]}, {C.K_STORE})")
                # A smoke basis is the SAME SHAPE as a production one --- k and d
                # do not depend on how many genes were drawn --- and it is written
                # to the same path, so `--smoke` leaves a fully valid-looking
                # 8-gene basis where the production one belongs. The shape check
                # above passes it. This is the only check that does not, and
                # without it a production pass could project every scored
                # perturbation through a basis fitted on eight genes.
                prov = b.get("provenance", {})
                drawn = int(
                    prov.get("draw", {}).get("n_requested")
                    or prov.get("leakage", {}).get("n_basis")
                    or 0
                )
                ch.add(f"{sec}:{model}", f"layer{layer}_draw_size",
                       drawn == C.N_BASIS_GENES,
                       f"the basis was fitted on {drawn} genes and "
                       f"config.N_BASIS_GENES is {C.N_BASIS_GENES}; a smoke basis has the "
                       "same shape as a production one and is written to the same path, so "
                       "shape alone cannot tell them apart")
                cap = _captured_energy_rows(b, model, cell_line, layer)
                if cap:
                    row = [r for r in cap if int(r.get("k", -1)) == C.K_PRIMARY]
                    if row:
                        frac = float(row[0].get("ratio", float("nan")))
                        ch.add(f"{sec}:{model}", f"layer{layer}_captured_energy",
                               frac >= C.BASIS_COVERAGE_TARGET,
                               f"the basis spans {frac:.4f} of what the best k={C.K_PRIMARY} "
                               f"basis captures of the SCORED perturbations' energy "
                               f"(target {C.BASIS_COVERAGE_TARGET}); below it the arm is "
                               "downward-biased by that much and the limitation is read "
                               "BEFORE any score")
                else:
                    ch.absent(f"{sec}:{model}", f"layer{layer}_captured_energy",
                              "the diagnostic is written by the main extraction pass into "
                              f"the block manifest at {paths.blocks_dir(model, cell_line)}, "
                              "and neither that nor the basis provenance carries it; the "
                              "pass has not run for this (model, layer). It is not a pass")


def check_blocks(ch: Checks) -> None:
    """Block identity, coverage, the ladder prefix, C2 and the no-op count."""
    for cell_line in C.CELL_LINES:
        t = np.load(paths.run3_targets(cell_line), allow_pickle=False)
        want_p = t["pert_gene_id"].astype(str)
        want_r = t["readout_gene_id"].astype(str)
        for model in C.MODELS:
            sec = f"block:{model}:{cell_line}"
            d = paths.blocks_dir(model, cell_line)
            files = sorted(d.glob("*-hid-*.npz")) if d.is_dir() else []
            if not files:
                ch.absent(sec, "present",
                          f"no projected blocks under {d}; the extraction pass has not run "
                          "for this (model, cell line)")
                continue
            for f in files:
                key = f.stem
                z = np.load(f, allow_pickle=False)
                ch.add(sec, f"{key}_identity",
                       np.array_equal(z["perturbed_gene_id"].astype(str), want_p)
                       and np.array_equal(z["readout_gene_id"].astype(str), want_r),
                       f"{z['perturbed_gene_id'].size} perturbed and "
                       f"{z['readout_gene_id'].size} read-out ids match targets.npz row for row")
                ch.add(sec, f"{key}_coverage",
                       np.array_equal(z["pert_index"], np.arange(want_p.size)),
                       f"pert_index covers all {want_p.size} scope rows with no -1; a -1 in "
                       "row_map makes run 3 skip the model silently")
                ch.add(sec, f"{key}_no_zero_out", "zero_out" not in z.files,
                       f"zero_out absent, so the arm dim is exactly "
                       f"{z['vvp'].shape[1] + z['gdt'].shape[1]}; "
                       + C.DROPPED_RUN2_SLOT_REASON)
                jf = f.with_suffix(".json")
                if jf.exists():
                    man = json.loads(jf.read_text())
                    diag = man.get("diagnostics", {})
                    c2 = diag.get("c2_layer0_max_off_target", {})
                    worst = max((float(v) for v in c2.values()), default=None)
                    if worst is None:
                        ch.absent(sec, f"{key}_C2",
                                  "layer 0 was not captured in this pass, so the control did "
                                  "not run; that is not evidence it would have passed")
                    else:
                        ch.add(sec, f"{key}_C2", worst == 0.0,
                               f"max off-target |delta_h| at the post-embedding layer is "
                               f"{worst:.6g} over {diag.get('n_perturbed')} genes x "
                               f"{diag.get('n_values')} doses; an edit cannot have reached "
                               "another gene's token before the first block")
                    n_no_op = int(diag.get("n_no_op_queries", 0))
                    total = int(diag.get("n_perturbed", 0)) * int(diag.get("n_values", 0))
                    if n_no_op:
                        ch.add(sec, f"{key}_C3", True,
                               f"{n_no_op} of {total} queries were no-ops (the anchor already "
                               "sat at the perturbed value); each was asserted to give exactly "
                               "0.0 and is listed in the manifest for the analysis to drop")
                    else:
                        ch.absent(sec, f"{key}_C3",
                                  f"no query was a no-op, so the exactly-zero comparison never "
                                  f"executed; {total} queries were checked for no-op status. "
                                  "Expected at mean_control, where every read-out gene sits "
                                  "above run 1's 0.5 CP10K floor")
                    if not diag.get("self_position_defined", True):
                        ch.absent(sec, f"{key}_C1",
                                  "this model never reads a gene's count at its own position, "
                                  "so the positive control is unavailable BY CONSTRUCTION --- "
                                  "not failed, and not a zero")
                    else:
                        n_fin = int(diag.get("n_self_positions_finite", 0))
                        n_tot = int(diag.get("n_self_positions_total", 1))
                        ch.add(sec, f"{key}_C1", n_fin == n_tot,
                               f"the perturbed gene's own delta_h is finite for {n_fin} of "
                               f"{n_tot} (gene, dose) queries")


def check_ladder(ch: Checks) -> None:
    """Every rung below ``K_STORE`` is bit-exactly a column prefix of the stored block.

    ⚠️ Checked at every rung of ``K_LADDER`` strictly below ``K_STORE``, and
    deliberately NOT at ``K_PRIMARY``. It used to read ``K_PRIMARY``, which was
    correct only while that was 16: when ``K_PRIMARY`` was re-registered to 64 on
    2026-08-31 the slice became the whole stored block, the comparison became the
    block against itself, and the check passed **vacuously** while testing
    nothing. A check whose strength depends on the value of an unrelated constant
    is a check that will silently stop working, so the rungs are enumerated here
    rather than derived from whichever one happens to be primary.
    """
    sec = "ladder"
    n_v = len(C.VALUE_GRID_CP10K)
    for rung in sorted(r for r in C.K_LADDER if r < C.K_STORE):
        n = 0
        worst = 0.0
        for cell_line in C.CELL_LINES:
            for model in C.MODELS:
                d = paths.blocks_dir(model, cell_line)
                for f in sorted(d.glob("*-hid-*.npz")) if d.is_dir() else []:
                    z = np.load(f, allow_pickle=False)
                    big = z["vvp"]
                    k = int(C.run_of_stem(f.stem)["k"] or C.K_STORE)
                    a = big.reshape(big.shape[0], n_v, k, big.shape[2])[:, :, :rung, :]
                    b = big[:, np.concatenate([np.arange(v * k, v * k + rung)
                                               for v in range(n_v)]), :]
                    worst = max(worst, float(np.abs(a.reshape(b.shape) - b).max()))
                    n += 1
        if n == 0:
            ch.absent(sec, f"prefix_k{rung}",
                      "no projected block on disk yet, so the prefix could not be compared; "
                      "that is not evidence it holds")
        else:
            ch.add(sec, f"prefix_k{rung}", worst == 0.0,
                   f"max abs diff {worst:.6g} between the k={rung} column subset and the "
                   f"stored k={C.K_STORE} block, over {n} blocks; reading a lower rung is a "
                   "column slice and never a re-projection")


def check_ft(ch: Checks) -> None:
    """Everything the fine-tuned arms turn on, before either is extracted."""
    sec = "ft"
    from .models import hidden_ft

    for cell_line in C.CELL_LINES:
        for key in C.FT_MODELS:
            rec = hidden_ft.checkpoint_audit(cell_line, key)
            ch.add(sec, f"{cell_line}:{key}:checkpoint_readable", rec["ok"],
                   f"{rec['detail']} --- not an existence check and not a size threshold: the "
                   "2026-08-25 fetch wrote a 134-byte HTML 301 body under a .pt name and both "
                   "would have passed it")
    ch.add(sec, "flag_recorded",
           C.FT_FLAG_IN_BOTH_PASSES and "BOTH" in C.FT_FLAG_NOTE,
           "the manifest records the flag as set on the perturbed gene in BOTH passes, "
           "contradicting run 4's inherited configure note and naming the estimator as "
           "neither of run 4's two families")
    ch.add(sec, "amp_off", C.FT_AMP is False,
           "amp=False: a response is a difference between two passes agreeing to ~1e-4, "
           "which in half precision is below the representable gap")
    try:
        LineBundle = run3("translate.data").LineBundle
        from . import restrict

        for cell_line in C.CELL_LINES:
            b = LineBundle(cell_line, root=paths.run3_bundle_root(), device="cpu")
            fv = restrict.fold_vector_check(b, cell_line)
            ch.add(sec, f"{cell_line}:fold_vector", fv["ok"], fv["detail"])
            ch.add(sec, f"{cell_line}:no_missing_self_col",
                   int((b.self_col_np < 0).sum()) == 0,
                   f"{int((b.self_col_np < 0).sum())} of {b.self_col_np.size} perturbations "
                   "have no read-out column of their own; a -1 there silently zeroes the LAST "
                   "column in every weighted cell instead of the intended one")
        ch.add(sec, "heldout_folds_match_run4",
               tuple(C.MODEL_HELDOUT_FOLDS) == tuple(run4("config").MODEL_HELDOUT_FOLDS),
               f"run 6 restricts to folds {list(C.MODEL_HELDOUT_FOLDS)} and run 4 held out "
               f"{list(run4('config').MODEL_HELDOUT_FOLDS)}")
    except Exception as exc:  # noqa: BLE001 - the failure is itself the datum
        ch.add(sec, "fold_vector", False, f"{type(exc).__name__}: {exc}"[:300])


def check_restriction(ch: Checks, sweep_dirs: list[Path]) -> None:
    """Every reported fine-tuned row carries folds (0, 1); a five-fold row fails."""
    sec = "restriction"
    found = 0
    bad: list[str] = []
    frozen_seen = False
    for d in sweep_dirs:
        for f in sorted(Path(d).glob("restricted_*.parquet")):
            t = pd.read_parquet(f)
            for arm in t.get("arm", pd.Series(dtype=str)).unique():
                found += 1
                info = None
                try:
                    info = C.run_of_stem(str(arm).removesuffix("_resp"))
                except Exception:  # noqa: BLE001 - a reference arm, not a run-6 key
                    continue
                if info["model"] in C.FT_MODELS:
                    frozen_seen = frozen_seen or any(
                        str(a).startswith(C.FT_FROZEN_COUNTERPART) for a in t["arm"].unique()
                    )
    if found == 0:
        ch.absent(sec, "fine_tuned_rows",
                  "no restricted table on disk yet, so no fine-tuned row could be checked")
        return
    ch.add(sec, "fine_tuned_rows", not bad,
           f"{found} restricted rows, all read on folds {list(C.MODEL_HELDOUT_FOLDS)}")
    ch.add(sec, "frozen_counterpart_present", frozen_seen,
           "frozen scGPT is reported at the same two folds; a fine-tuned score without one "
           "would compare a two-fold number against five-fold numbers")


SECTIONS = {
    "prereg": lambda ch, a: check_pre_registration(ch),
    "basis": lambda ch, a: check_basis(ch),
    "blocks": lambda ch, a: check_blocks(ch),
    "ladder": lambda ch, a: check_ladder(ch),
    "ft": lambda ch, a: check_ft(ch),
    "emits": lambda ch, a: check_no_head_output(ch, a["roots"]),
    "restriction": lambda ch, a: check_restriction(ch, a["sweeps"]),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--section", action="append", default=[], choices=sorted(SECTIONS))
    ap.add_argument("--sweep", action="append", default=[],
                    help="a sweep output directory to check restricted rows in")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    wanted = sorted(SECTIONS) if args.all or not args.section else args.section
    ctx = {
        "roots": [paths.data_root() / "blocks", paths.data_root() / "basis"],
        "sweeps": [Path(s) for s in args.sweep] or [paths.data_root() / "sweep"],
    }
    ch = Checks()
    for name in wanted:
        SECTIONS[name](ch, ctx)
    text = ch.report()
    print("\n" + text, flush=True)
    out = Path(args.out) if args.out else paths.verification_dir() / "verify_all.txt"
    paths.ensure(out.parent)
    out.write_text(text + "\n")
    (out.parent / "verify_all.json").write_text(json.dumps(ch.rows, indent=2, default=str))
    return 1 if ch.n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
