"""Is the masking effect information removal, or an off-distribution mask token?

Run 2 measured that hiding the read-out gene's own input value changes the
response at the same anchor, and that the direction is model-specific: down to a
factor of 0.097 for scPRINT, up to 1.77 for AIDO.Cell. Two accounts fit the
shrinking direction and the run could not separate them:

* **information removal** --- a prediction that cannot see the gene's own value
  falls back on what the rest of the cell implies, and no single other gene moves
  that fallback much;
* **mask-token artefact** --- the mask token is a value the model is not being
  asked to read in the way pretraining used it, so the whole response is measured
  outside the input distribution and means little.

They are separated by a third condition at the same anchor. Instead of the mask
token, the read-out gene's input is set to an uninformative but in-distribution
value: the median pooled-control level across read-out genes, the same scalar for
every hidden gene. That removes the gene-specific information exactly as masking
does, while keeping every input value inside the range the model was trained on.
If the shrinkage survives the substitution it is information removal; if it only
appears with the mask token it is an artefact of querying the model off its input
distribution, and must not be read as evidence about reconstruction.

Three conditions, one code path, one batching, so the comparison is not confounded
by the pipeline:

``visible``             the ordinary query: the read-out gene sees its own value.
``masked``              each model's own mask convention, read from its checkpoint.
``median_substituted``  the read-out gene's input replaced by the median level.

One correction to run 2's masked estimator
------------------------------------------
The masked read cycles disjoint blocks and reads each gene only where it is
hidden. When the perturbed gene happens to fall in the same block as the gene
being read, the perturbation is overwritten by the mask before the forward pass,
so that pass is identical to its own baseline and its contribution to the
response difference is exactly zero. Run 2's masked block averaged those zeros
in, which shrinks every masked magnitude by the fraction of reads they take:
1/blocks per cycle, i.e. 14% at a 0.15 mask rate and 25% at 0.25. This module
skips those passes instead --- cheaper and unbiased --- and, because the discarded
contribution is exactly zero, also reproduces run 2's diluted estimator exactly
as ``corrected * coverage / cycles``, so the size of that bias is measured rather
than argued.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from .. import paths
from ..forms import normalize_forms, partition_masks
from .scope import coverage_record, select_perturbed

# torch enters only through a model adapter, so the estimator below is importable
# and testable on a machine without it


def substitute_level(readout: pd.DataFrame) -> dict:
    """The uninformative in-distribution value, and where it sits."""
    level = readout["pooled_control_cp10k"].to_numpy(dtype=np.float64)
    med = float(np.median(level))
    return {
        "statistic": C.HIDDEN_SUBSTITUTE_STATISTIC,
        "value_cp10k": med,
        "over": "the pooled non-targeting control level of the read-out genes of this cell line",
        "quantile_of_readout_levels": float((level <= med).mean()),
        "n_readout": int(level.size),
        "why": (
            "one scalar for every hidden gene, so it carries no gene-specific information, "
            "and a level half the read-out set already sits below, so no input value leaves "
            "the distribution the model was trained to read"
        ),
    }


def _hidden_inputs(
    x0: np.ndarray, blocks: list[np.ndarray], condition: str,
    substitute_native: float | None, perturb_col: int | None, perturb_value: float | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """One row per mask block: the same query with that block's genes hidden.

    ``masked`` returns a boolean mask and leaves the values alone, because each
    adapter applies its own mask convention inside the forward pass; the
    substitution instead writes the value and passes no mask.
    """
    rows = np.repeat(x0[None, :], len(blocks), axis=0)
    if perturb_col is not None:
        rows[:, int(perturb_col)] = float(perturb_value)
    mask = None
    if condition == "masked":
        mask = np.zeros(rows.shape, dtype=bool)
        for bi, block in enumerate(blocks):
            mask[bi, block] = True
    elif condition == "median_substituted":
        for bi, block in enumerate(blocks):
            rows[bi, block] = float(substitute_native)
    return rows.astype(np.float32), mask


def run_condition(
    model,
    x0: np.ndarray,
    cols: np.ndarray,
    values_native: np.ndarray,
    condition: str,
    blocks: list[np.ndarray],
    cycles: int,
    substitute_native: float | None,
) -> dict:
    """Response differences for one condition, over one gene subset and value grid."""
    n_p, n_v, n_r = cols.size, values_native.size, model.n_readout
    delta = np.zeros((n_p, n_v, n_r), dtype=np.float64)
    y_pert = np.zeros_like(delta)
    y_base = np.zeros_like(delta)
    coverage = np.zeros((n_p, n_r), dtype=np.int32)
    started = time.time()

    if condition == "visible":
        base = model.forward_batch(x0[None, :], perturbed_col=None)[0].astype(np.float64)
        for pi, col in enumerate(cols):
            rows = np.repeat(x0[None, :], n_v, axis=0)
            rows[np.arange(n_v), int(col)] = values_native
            resp = model.forward_batch(rows.astype(np.float32), perturbed_col=int(col)).astype(
                np.float64
            )
            delta[pi] = resp - base[None, :]
            y_pert[pi] = resp
            y_base[pi] = base[None, :]
            coverage[pi] = cycles
            if (pi + 1) % 20 == 0:
                print(f"    {condition} gene {pi + 1}/{n_p}", flush=True)
        return {
            "delta": delta, "y_pert": y_pert, "y_base": y_base, "coverage": coverage,
            "seconds": round(time.time() - started, 1), "n_forward_passes": 1 + n_p * n_v,
            "note": "the read-out gene's own value is visible; no block cycling is needed "
            "because the response does not depend on the mask partition",
        }

    # ---- block-cycled conditions
    rows, mask = _hidden_inputs(x0, blocks, condition, substitute_native, None, None)
    base_resp = model.forward_batch(rows, perturbed_col=None, mask=mask).astype(np.float64)
    n_forward = len(blocks)
    for pi, col in enumerate(cols):
        # a block that also hides the perturbed gene overwrites the perturbation before
        # the forward pass, so that pass carries no response at all: skip it
        keep = [bi for bi, block in enumerate(blocks) if int(col) not in set(block.tolist())]
        sub_blocks = [blocks[bi] for bi in keep]
        for vi, v in enumerate(values_native):
            rows, mask = _hidden_inputs(
                x0, sub_blocks, condition, substitute_native, int(col), float(v)
            )
            resp = model.forward_batch(rows, perturbed_col=int(col), mask=mask).astype(np.float64)
            n_forward += rows.shape[0]
            for k, bi in enumerate(keep):
                block = blocks[bi]
                delta[pi, vi, block] += resp[k, block] - base_resp[bi, block]
                y_pert[pi, vi, block] += resp[k, block]
                y_base[pi, vi, block] += base_resp[bi, block]
                if vi == 0:
                    coverage[pi, block] += 1
        cov = np.maximum(coverage[pi], 1)
        delta[pi] /= cov[None, :]
        y_pert[pi] /= cov[None, :]
        y_base[pi] /= cov[None, :]
        if (pi + 1) % 10 == 0:
            print(
                f"    {condition} gene {pi + 1}/{n_p} "
                f"({(time.time() - started) / (pi + 1):.2f}s per gene)",
                flush=True,
            )
    return {
        "delta": delta, "y_pert": y_pert, "y_base": y_base, "coverage": coverage,
        "seconds": round(time.time() - started, 1), "n_forward_passes": int(n_forward),
        "note": "read only where the gene is hidden, averaged over the cycles in which the "
        "perturbed gene was not hidden in the same block",
    }


def run(
    model_name: str,
    cell_line: str,
    cache: Path,
    out_dir: Path,
    scope: C.Scope,
    device: str = "cuda",
    grid_cp10k: tuple[float, ...] = C.HIDDEN_VALUE_GRID_CP10K,
    conditions: tuple[str, ...] = C.HIDDEN_CONDITIONS,
    image: str = "unrecorded",
    batch: int | None = None,
) -> dict:
    from ..models import queries as Q
    from ..models.base import environment_manifest
    from .extract import load_inputs

    started = time.time()
    data = load_inputs(model_name, cell_line)
    readout = data["readout"]
    perturbed = select_perturbed(data["perturbations"], readout, scope)
    subset = perturbed[perturbed.in_masked]
    cols = subset["readout_col"].to_numpy(dtype=np.int64)

    model = Q.build(model_name, cache, data["universe"], data["vocab"], device=device)
    model.load()
    if batch:
        model.batch_size = int(batch)
    control = readout["pooled_control_cp10k"].to_numpy(dtype=np.float64)
    kwargs = {"supplied_mode": C.SCGPT_PRODUCTION_MODE} if model_name == "scgpt" else {}
    layout = model.configure(readout, control, cell_line, **kwargs)
    spec = model.mask_spec()
    if spec is None:
        raise SystemExit(
            f"{model_name} has no masking regime: it is autoregressive, so a gene never sees "
            "its own value and there is nothing to hide"
        )

    sub = substitute_level(readout)
    substitute_native = float(np.asarray(model.to_native(np.array([sub["value_cp10k"]])))[0])
    x0 = np.asarray(model.to_native(control), dtype=np.float32)
    blocks = partition_masks(model.n_readout, float(spec["rate"]), C.MASK_MIN_COVERAGE, C.SEED)
    n_blocks_cycle = int(np.ceil(1.0 / float(spec["rate"])))
    values_native = np.asarray(model.to_native(np.asarray(grid_cp10k, dtype=np.float64)))

    # how far the substitution moves the read depth the model normalizes by; masking
    # leaves it untouched, substitution does not, and that difference is recorded
    per_block_total = [
        float(np.where(np.isin(np.arange(control.size), b), sub["value_cp10k"], control).sum())
        for b in blocks[:n_blocks_cycle]
    ]
    total_shift = {
        "anchor_total_cp10k": float(control.sum()),
        "substituted_total_cp10k_median": float(np.median(per_block_total)),
        "relative_shift_median": float(
            np.median([t / max(control.sum(), 1e-9) - 1.0 for t in per_block_total])
        ),
        "note": "the mask convention leaves the input's total unchanged, the substitution "
        "does not; both the baseline and the perturbed pass of a condition carry the same "
        "total, so the response difference is taken at one fixed background",
    }

    print(
        f"[{model_name}/{cell_line}] hidden-readout comparison: {cols.size} perturbed genes, "
        f"{len(grid_cp10k)} values, {len(blocks)} mask blocks "
        f"({C.MASK_MIN_COVERAGE} cycles of {n_blocks_cycle}), substitute "
        f"{sub['value_cp10k']:.4g} CP10K -> {substitute_native:.4g} native",
        flush=True,
    )

    out_dir = paths.ensure(out_dir)
    results, summary = {}, {}
    for condition in conditions:
        print(f"[{model_name}/{cell_line}] condition {condition}", flush=True)
        res = run_condition(
            model, x0, cols, values_native, condition, blocks, C.MASK_MIN_COVERAGE,
            substitute_native,
        )
        results[condition] = res
        zero_i = list(grid_cp10k).index(0.0)
        forms_vvp = normalize_forms(res["delta"], res["y_pert"], res["y_base"], model)
        forms_zero = normalize_forms(
            res["delta"][:, zero_i], res["y_pert"][:, zero_i], res["y_base"][:, zero_i], model
        )
        cdir = paths.ensure(out_dir / f"condition={condition}")
        common = {
            "perturbed_gene_id": subset.gene_id.to_numpy().astype(str),
            "readout_gene_id": readout["gene_id"].to_numpy().astype(str),
            "value_grid_cp10k": np.asarray(grid_cp10k, dtype=np.float64),
            "read_coverage": res["coverage"],
        }
        np.savez(cdir / "vvp.npz", **forms_vvp, **common)
        np.savez(cdir / "zero_out.npz", **forms_zero, **common)

        zero = res["delta"][:, zero_i]
        cov = res["coverage"]
        # run 2's masked block averaged the skipped passes in as zeros; that estimator is
        # exactly this one scaled by coverage/cycles, so it is reproduced rather than guessed
        diluted = zero * (cov / float(C.MASK_MIN_COVERAGE))
        covered = cov > 0
        summary[condition] = {
            "abs_median": float(np.median(np.abs(zero))),
            "abs_median_covered_only": float(np.median(np.abs(zero[covered]))),
            "abs_p99": float(np.percentile(np.abs(zero), 99)),
            "abs_median_run2_diluted_estimator": float(np.median(np.abs(diluted))),
            "cross_gene_sd_median": float(np.median(zero.std(axis=1))),
            "n_response_vectors": int(zero.shape[0]),
            "read_coverage": {
                "min": int(cov.min()), "median": float(np.median(cov)), "max": int(cov.max()),
                "cycles": C.MASK_MIN_COVERAGE,
                "mean_over_entries": float(cov.mean()),
                "uncovered_entries": int((cov == 0).sum()),
                "uncovered_fraction": float((cov == 0).mean()),
            },
            "dilution_factor_vs_run2_estimator": float(cov.mean() / C.MASK_MIN_COVERAGE),
            "seconds": res["seconds"],
            "n_forward_passes": res["n_forward_passes"],
            "note": res["note"],
        }
        print(f"    {condition}: {json.dumps(summary[condition])}", flush=True)

    ratios = {}
    base_name = "visible"
    for condition in conditions:
        if condition == base_name:
            continue
        num, den = summary[condition]["abs_median"], summary[base_name]["abs_median"]
        ratios[f"{condition}_over_{base_name}"] = float(num / den) if den else None
    if "masked" in summary and "median_substituted" in summary:
        ratios["median_substituted_over_masked"] = float(
            summary["median_substituted"]["abs_median"] / summary["masked"]["abs_median"]
        ) if summary["masked"]["abs_median"] else None

    # per-gene agreement between the three conditions: a ratio of medians can hide a
    # change of pattern, so the correlation of the response vectors is recorded too
    agreement = {}
    zero_i = list(grid_cp10k).index(0.0)
    for a, b in (("visible", "masked"), ("visible", "median_substituted"),
                 ("masked", "median_substituted")):
        if a not in results or b not in results:
            continue
        za, zb = results[a]["delta"][:, zero_i], results[b]["delta"][:, zero_i]
        ok = (results[a]["coverage"] > 0) & (results[b]["coverage"] > 0)
        per = [
            float(np.corrcoef(za[i][ok[i]], zb[i][ok[i]])[0, 1])
            for i in range(za.shape[0])
            if ok[i].sum() > 2 and za[i][ok[i]].std() > 0 and zb[i][ok[i]].std() > 0
        ]
        agreement[f"{a}_vs_{b}"] = {
            "median_pearson_per_gene": float(np.median(per)) if per else None,
            "n_genes": len(per),
        }

    manifest = {
        "model": model_name,
        "cell_line": cell_line,
        "kind": "hidden_readout_comparison",
        "question": (
            "does the masking effect survive replacing the mask token with an uninformative "
            "in-distribution value, which would make it information removal rather than an "
            "off-distribution query"
        ),
        "anchor": "mean_control",
        "layout": layout.as_dict(),
        "conditions": {
            "visible": "the read-out gene sees its own input value",
            "masked": f"the read-out gene's input replaced by the model's own mask convention "
                      f"({spec['kind']}, value {spec.get('value')})",
            "median_substituted": "the read-out gene's input replaced by one uninformative "
                                  "in-distribution level, the same scalar for every hidden gene",
        },
        "substitute_level": sub,
        "substitute_native": substitute_native,
        "mask_spec": spec,
        "read_depth": total_shift,
        "value_grid_cp10k": list(grid_cp10k),
        "value_grid_note": (
            "the zero-out query the comparison is about, plus the top of the production grid "
            "so the reading is not specific to zero-out"
        ),
        "n_perturbed": int(cols.size),
        "perturbed_subset": "the in_masked subset of run 2's 250-gene block, reproduced from "
                            f"the same seed ({C.SEED}) and scope, so the numbers sit beside the "
                            "masked block already extracted",
        "coverage": coverage_record(data["perturbations"], readout, scope, perturbed),
        "mask_blocks": {"per_cycle": n_blocks_cycle, "cycles": C.MASK_MIN_COVERAGE,
                        "total": len(blocks)},
        "estimator_correction": (
            "a mask block that also hides the perturbed gene overwrites the perturbation, so "
            "that pass contributes exactly zero; those passes are skipped here and averaged in "
            "by run 2's masked block, whose estimator is reproduced as "
            "abs_median_run2_diluted_estimator"
        ),
        "summary": summary,
        "ratios": ratios,
        "agreement": agreement,
        "seed": C.SEED,
        "seconds": round(time.time() - started, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (out_dir / "environment.json").write_text(
        json.dumps(
            environment_manifest(model_name, image, {"device": device, "cell_line": cell_line,
                                                     "kind": "hidden_readout"}),
            indent=2, default=str,
        )
    )
    # a pod-readable copy: the report is assembled where the arrays are not
    side = paths.ensure(paths.report_dir() / "hidden_readout")
    (side / f"{model_name}_{cell_line}.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    checks = verify(manifest, results, grid_cp10k)
    (out_dir / "verification.json").write_text(json.dumps(checks, indent=2, default=str))
    (side / f"verification_{model_name}_{cell_line}.json").write_text(
        json.dumps(checks, indent=2, default=str)
    )
    ok = sum(1 for r in checks["checks"] if r["ok"])
    print(f"[{model_name}/{cell_line}] {ok}/{len(checks['checks'])} checks passed", flush=True)
    print(f"[{model_name}/{cell_line}] done in {manifest['seconds']}s", flush=True)
    return manifest


def verify(manifest: dict, results: dict, grid_cp10k: tuple[float, ...]) -> dict:
    """Checks this comparison has to pass to be readable as evidence."""
    rows = []

    def add(name: str, ok: bool, detail: str) -> None:
        rows.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] hidden_readout/{name}: {detail}", flush=True)

    n_p = manifest["n_perturbed"]
    for condition, res in results.items():
        d = res["delta"]
        add(
            f"{condition}/shape_and_finite",
            d.shape[0] == n_p and d.shape[1] == len(grid_cp10k) and bool(np.isfinite(d).all()),
            f"delta {d.shape} over {n_p} genes and {len(grid_cp10k)} values, finite="
            f"{bool(np.isfinite(d).all())}",
        )
        if condition != "visible":
            cov = res["coverage"]
            add(
                f"{condition}/read_coverage",
                float((cov == 0).mean()) < 0.05,
                f"{int((cov == 0).sum())} of {cov.size} entries "
                f"({100 * float((cov == 0).mean()):.3f}%) had no cycle in which the read-out "
                f"gene was hidden and the perturbed gene was not; mean coverage "
                f"{cov.mean():.3f} of {C.MASK_MIN_COVERAGE} cycles",
            )
    s = manifest["summary"]
    if "masked" in s and "median_substituted" in s:
        add(
            "conditions_differ_from_visible",
            all(s[c]["abs_median"] > 0 for c in s),
            "; ".join(f"{c} median |zero-out| {v['abs_median']:.4g}" for c, v in s.items()),
        )
        add(
            "substitution_is_in_distribution",
            0.0 < manifest["substitute_level"]["value_cp10k"] < 1e4,
            f"substitute {manifest['substitute_level']['value_cp10k']:.4g} CP10K sits at "
            f"quantile {manifest['substitute_level']['quantile_of_readout_levels']:.3f} of the "
            "read-out genes' own control levels",
        )
    add(
        "one_code_path",
        len(results) == len(C.HIDDEN_CONDITIONS),
        f"{len(results)} conditions computed in one job with one batching and one anchor: "
        + ", ".join(results),
    )
    ok = sum(1 for r in rows if r["ok"])
    return {"model": manifest["model"], "cell_line": manifest["cell_line"],
            "n_passed": ok, "n_checks": len(rows), "checks": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=list(paths.MODELS))
    ap.add_argument("--cell-line", required=True, choices=list(paths.CELL_LINES))
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--n-masked", type=int, default=None)
    ap.add_argument("--values", type=float, nargs="*", default=list(C.HIDDEN_VALUE_GRID_CP10K))
    ap.add_argument("--conditions", nargs="*", default=list(C.HIDDEN_CONDITIONS))
    ap.add_argument("--image", default="unrecorded")
    args = ap.parse_args(argv)

    scope = C.Scope(
        n_perturbed=C.SCOPE.n_perturbed,
        n_perturbed_cells_anchor=C.SCOPE.n_perturbed_cells_anchor,
        n_perturbed_masked=(
            C.SCOPE.n_perturbed_masked if args.n_masked is None else args.n_masked
        ),
    )
    cache = Path(args.cache) if args.cache else paths.checkpoint_cache()
    out = Path(args.out) if args.out else paths.hidden_readout_dir(args.model, args.cell_line)
    try:
        run(
            args.model,
            args.cell_line,
            cache,
            out,
            scope,
            device=args.device,
            grid_cp10k=tuple(args.values),
            conditions=tuple(args.conditions),
            image=args.image,
            batch=args.batch,
        )
    except Exception:
        paths.ensure(out)
        (out / "failure.txt").write_text(traceback.format_exc())
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
