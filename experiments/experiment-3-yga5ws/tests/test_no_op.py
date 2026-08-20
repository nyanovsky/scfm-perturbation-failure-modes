"""No-op queries are excluded per query, not by deleting the dose."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scfm_pfm.ingest import no_op


def _per_gene(rows: list[tuple[str, float, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "m",
                "cell_line": "l",
                "anchor": "individual_cells",
                "perturbed_gene_id": g,
                "value_cp10k": v,
                "n_cells": n,
                "n_no_op": k,
            }
            for g, v, n, k in rows
        ]
    )


def test_correction_recovers_the_average_over_the_cells_that_were_queried():
    """A no-op cell contributes an exact zero, so the stored row is diluted.

    Two of four cells were no-ops, so the stored average is half the average
    over the two cells that were really queried, and the factor has to undo
    exactly that -- while leaving every other query alone.
    """
    per_cell = np.array([[1.0, -2.0, 3.0], [3.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    stored = per_cell.mean(axis=0)
    want = per_cell[:2].mean(axis=0)

    factor = no_op.correction_factors(
        _per_gene([("g0", 0.0, 4, 2)]),
        "m",
        "l",
        "individual_cells",
        np.array(["g0", "g1"]),
        np.array([0.0, 1.0]),
    )
    assert factor.shape == (2, 2)
    np.testing.assert_allclose(stored * factor[0, 0], want)
    assert factor[0, 1] == 1.0, "a value with no no-op query is untouched"
    assert (factor[1] == 1.0).all(), "another perturbation at the same value is untouched"


def test_a_query_that_was_a_no_op_in_every_cell_has_no_measurement():
    factor = no_op.correction_factors(
        _per_gene([("g0", 0.0, 4, 4)]),
        "m",
        "l",
        "individual_cells",
        np.array(["g0"]),
        np.array([0.0, 1.0]),
    )
    assert factor[0, 0] == 0.0
    assert factor[0, 1] == 1.0


def test_the_standardized_form_is_invariant_to_the_dilution():
    """Why the correction is exact rather than approximate.

    The feature every arm reads is the per-query z-score across read-out
    genes, and the dilution is a positive scalar on that query's row, so
    correcting it cannot move the feature. What the deleted-dose rule did
    instead was remove the row entirely.
    """
    rng = np.random.default_rng(0)
    row = rng.normal(size=(1, 1, 64))
    np.testing.assert_allclose(
        no_op.standardize_rows(row), no_op.standardize_rows(row * 0.25), atol=1e-12
    )


def test_a_no_op_is_the_query_whose_response_is_identically_zero():
    qs = pd.DataFrame(
        {
            "cell_index": [0, 1, 0, 1],
            "perturbed_gene": ["g0", "g0", "g0", "g0"],
            "value_cp10k": [0.0, 0.0, 1.0, 1.0],
            "max_abs": [0.0, 0.5, 0.2, 0.3],
            "cross_gene_sd": [0.0, 0.1, 0.05, 0.07],
        }
    )
    got = no_op.per_gene_counts(qs).sort_values("value_cp10k")
    assert list(got.n_cells) == [2, 2]
    assert list(got.n_no_op) == [1, 0]
