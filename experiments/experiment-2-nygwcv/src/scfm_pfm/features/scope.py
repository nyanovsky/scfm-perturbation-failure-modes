"""Which perturbed genes a block covers, and the record that says so.

Split out of :mod:`scfm_pfm.features.extract` for two reasons. It is the one
part of the extraction the pod also needs --- the report reads the perturbed
table to interpret the individual-cell anchor, and the perturbed table is a
function of (cell line, scope) alone, not of the model --- and importing the
extraction driver pod-side would pull in torch.

Coverage is now recorded explicitly in every block's manifest. Run 2's first
wave ran 250 perturbed genes out of the 991 (HepG2) and 1,100 (Jurkat) that the
in-scope strata actually offer, chosen from the slowest model's measured cost.
A block at 250 and a block at full coverage are not interchangeable, so the
selected count, the count available, and the fraction are written down rather
than left to be inferred from an array shape.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .. import config as C


def _take(available: int, requested: int) -> int:
    """``requested <= 0`` means every gene in scope; otherwise a capped count."""
    return available if requested <= 0 else min(requested, available)


def in_scope_perturbed(perts: pd.DataFrame, readout: pd.DataFrame, scope: C.Scope) -> pd.DataFrame:
    """Every perturbation the scope admits, before any subsampling.

    A perturbed gene has to be a column of the supplied set to be perturbable at
    all, so the scope is the intersection of the effect-size strata the plan keeps
    with the read-out set this cell line ended up with.
    """
    by_symbol: dict[str, int] = {}
    for col in ("gene_name", "hgnc_symbol"):
        if col in readout.columns:
            for i, sym in enumerate(readout[col].to_numpy().astype(str)):
                by_symbol.setdefault(sym, i)
    rows = []
    for rec in perts.itertuples():
        if str(rec.effect_stratum) not in scope.strata:
            continue
        col = by_symbol.get(str(rec.target))
        if col is None:
            continue
        rows.append(
            {
                "target": str(rec.target),
                "readout_col": int(col),
                "gene_id": str(readout["gene_id"].to_numpy()[col]),
                "effect_stratum": str(rec.effect_stratum),
                "n_de_genes": int(getattr(rec, "n_de_genes", 0)),
                "knockdown": float(getattr(rec, "knockdown", float("nan"))),
            }
        )
    frame = pd.DataFrame(rows).drop_duplicates("readout_col").reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("no perturbed genes survive the scope intersection")
    return frame


def select_perturbed(
    perts: pd.DataFrame, readout: pd.DataFrame, scope: C.Scope, seed: int = C.SEED
) -> pd.DataFrame:
    """The block's perturbed genes and its nested anchor subsets.

    With ``scope.n_perturbed <= 0`` every in-scope perturbation is kept, which is
    what full coverage means; with a positive count the selection is stratified,
    the same rule and seed run 2's first wave used.
    """
    frame = in_scope_perturbed(perts, readout, scope)
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    full = scope.n_perturbed <= 0
    per_stratum = 0 if full else max(1, scope.n_perturbed // max(1, len(scope.strata)))
    for stratum in scope.strata:
        idx = np.flatnonzero(frame.effect_stratum.to_numpy() == stratum)
        take = idx.size if full else min(per_stratum, idx.size)
        keep.extend(rng.choice(idx, size=take, replace=False).tolist())
    frame = frame.iloc[sorted(keep)].reset_index(drop=True)

    # nested subsets, so the small anchors are strictly inside the large one
    frame["in_main"] = True
    sub_cells = rng.choice(
        frame.shape[0], size=_take(frame.shape[0], scope.n_perturbed_cells_anchor), replace=False
    )
    sub_mask = rng.choice(
        sub_cells, size=_take(sub_cells.size, scope.n_perturbed_masked), replace=False
    )
    sub_step = rng.choice(
        frame.shape[0], size=_take(frame.shape[0], C.N_STEP_SWEEP_GENES), replace=False
    )
    frame["in_cells_anchor"] = False
    frame.loc[sorted(sub_cells), "in_cells_anchor"] = True
    frame["in_masked"] = False
    frame.loc[sorted(sub_mask), "in_masked"] = True
    frame["in_step_sweep"] = False
    frame.loc[sorted(sub_step), "in_step_sweep"] = True
    return frame


def coverage_record(
    perts: pd.DataFrame, readout: pd.DataFrame, scope: C.Scope, selected: pd.DataFrame
) -> dict:
    """What fraction of the in-scope perturbations this block actually covers.

    Written into every block manifest so a later run cannot mix a subset block
    with a full one without noticing.
    """
    available = in_scope_perturbed(perts, readout, scope)
    n_avail, n_sel = int(available.shape[0]), int(selected.shape[0])
    by_stratum = {}
    for stratum in scope.strata:
        by_stratum[stratum] = {
            "in_scope": int((available.effect_stratum == stratum).sum()),
            "selected": int((selected.effect_stratum == stratum).sum()),
        }
    mode = "full" if n_sel == n_avail else "subset"
    return {
        "mode": mode,
        "n_perturbed": n_sel,
        "n_in_scope": n_avail,
        "fraction_of_scope": round(n_sel / max(n_avail, 1), 6),
        "strata": list(scope.strata),
        "by_stratum": by_stratum,
        "n_perturbed_cells_anchor": int(selected.in_cells_anchor.sum()),
        "n_perturbed_masked": int(selected.in_masked.sum()),
        "n_step_sweep": int(selected.in_step_sweep.sum()),
        "scope_requested": asdict(scope),
        "selection_rule": (
            "every in-scope perturbation"
            if mode == "full"
            else f"stratified sample of {scope.n_perturbed} at seed {C.SEED}, "
            f"{scope.n_perturbed // max(1, len(scope.strata))} per stratum, drawn from the "
            "in-scope set; the individual-cell and masked subsets are nested inside it"
        ),
        "in_scope_definition": (
            "perturbations in strata "
            + ", ".join(scope.strata)
            + " whose target gene is also a read-out gene of this cell line, so it can be "
            "perturbed at all"
        ),
        "comparability_note": (
            "a block at partial coverage and a block at full coverage share the perturbed "
            "genes of the smaller one but not its row order; join on perturbed_gene_id, "
            "never on row index, and do not compare a per-block aggregate across two "
            "different coverages"
        ),
    }


def write_scope_tables(handoff_root, out_dir, cell_lines=C.CELL_LINES) -> list:
    """Pod-side copies of the perturbed table, one per (coverage mode, cell line).

    The report needs the perturbed genes' own anchor levels to interpret the
    individual-cell anchor. The table depends only on the cell line and the
    scope, so it is regenerated here rather than shipped back from a job.
    """
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    scopes = {"full": C.Scope(n_perturbed=0, n_perturbed_cells_anchor=0, n_perturbed_masked=0),
              f"n{C.SCOPE.n_perturbed}": C.SCOPE}
    for cl in cell_lines:
        readout = pd.read_parquet(Path(handoff_root) / "readout" / f"readout_genes_{cl}.parquet")
        perts = pd.read_parquet(Path(handoff_root) / "effects" / cl / "perturbations.parquet")
        for mode, scope in scopes.items():
            frame = select_perturbed(perts, readout, scope)
            path = out_dir / f"perturbed_genes_{mode}_{cl}.parquet"
            frame.to_parquet(path, index=False)
            written.append((mode, cl, int(frame.shape[0]), str(path)))
    return written
